"""Main EaseSHAP estimators.

This module keeps the public ``EaseSHAP`` and ``EaseSHAP_group`` classes in one
place, while sharing their internal workflow through small private components.
The empirical dense backend preserves the current estimator behavior; the exact
conditional backend shares the same fitting interface with solver variants.
"""

import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import linalg, special

from .base import estimatorTemplate
from .group_core import (
    _safe_comb,
    groupEstimatorTemplate,
    group_sum_coefficient,
    semivalue_coefficients,
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _distribution_cardinality(num_player, semivalue, semivalue_param):
    n = int(num_player)
    if semivalue == "shapley":
        return np.full(n, 1.0 / n, dtype=np.float64)

    if semivalue == "weighted_banzhaf":
        weights = np.ones(n, dtype=np.float64)
        for k in range(n):
            for i in range(k):
                weights[k] *= (n - 1 - i) / (i + 1) * semivalue_param * (1 - semivalue_param)
            weights[k] *= (1 - semivalue_param) ** (n - 1 - 2 * k)
        return weights

    if semivalue == "beta_shapley":
        alpha, beta = semivalue_param
        weights = np.ones(n, dtype=np.float64)
        tmp_range = np.arange(1, n, dtype=np.float64)
        weights *= np.divide(tmp_range, tmp_range + (alpha + beta - 1)).prod()
        for s in range(n):
            cur = weights[s]
            tmp_range = np.arange(1, s + 1, dtype=np.float64)
            cur *= np.divide(tmp_range + (beta - 1), tmp_range).prod()
            tmp_range = np.arange(1, n - s, dtype=np.float64)
            cur *= np.divide((alpha - 1) + tmp_range, tmp_range).prod()
            weights[s] = cur
        return weights

    raise NotImplementedError(f"Check {semivalue}")


def _is_symmetric_semivalue(semivalue, semivalue_param):
    if semivalue == "shapley":
        return True
    if semivalue == "weighted_banzhaf":
        return abs(float(semivalue_param) - 0.5) <= 1e-12
    if semivalue == "beta_shapley":
        alpha, beta = semivalue_param
        return abs(float(alpha) - float(beta)) <= 1e-12
    return False


def _boundary_subset_matrix(n):
    subsets = []
    seen = set()

    def add(mask):
        key = np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()
        if key not in seen:
            seen.add(key)
            subsets.append(mask.copy())

    empty = np.zeros(n, dtype=bool)
    add(empty)
    for player in range(n):
        subset = np.zeros(n, dtype=bool)
        subset[player] = True
        add(subset)

    full = np.ones(n, dtype=bool)
    for player in range(n):
        subset = full.copy()
        subset[player] = False
        add(subset)
    add(full)

    if not subsets:
        return np.empty((0, n), dtype=bool)
    return np.vstack(subsets)


def _normalize_boundary_policy(boundary_policy, exact_boundary_handling):
    if boundary_policy is None:
        return "fixed" if bool(exact_boundary_handling) else "none"
    if not isinstance(boundary_policy, str):
        raise ValueError(f"`boundary_policy` must be a string or None, got {boundary_policy!r}.")
    key = boundary_policy.strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "no": "none",
        "none": "none",
        "fixed": "fixed",
        "fixed_order": "fixed",
        "adaptive": "adaptive",
    }
    if key not in aliases:
        raise ValueError('`boundary_policy` must be one of {"none", "fixed", "adaptive"}.')
    policy = aliases[key]
    if policy != "none" and not bool(exact_boundary_handling):
        raise ValueError(
            "`boundary_policy` requests exact boundary handling, but "
            "`exact_boundary_handling=False` was also supplied. Remove the legacy "
            "flag or set `boundary_policy='none'`."
        )
    return policy


def _normalize_boundary_order(boundary_order):
    if isinstance(boundary_order, (bool, np.bool_)) or not isinstance(boundary_order, (int, np.integer)):
        raise ValueError(f"`boundary_order` must be an integer >= 0, got {boundary_order!r}.")
    boundary_order = int(boundary_order)
    if boundary_order < 0:
        raise ValueError(f"`boundary_order` must be >= 0, got {boundary_order!r}.")
    return boundary_order


def _normalize_boundary_sizes(n, sizes):
    n = int(n)
    seen = set()
    for size in sizes:
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
            raise ValueError(f"Boundary size must be an integer, got {size!r}.")
        s = int(size)
        if s < 0 or s > n:
            raise ValueError(f"Boundary size {s} is outside [0, {n}].")
        seen.add(s)
    return sorted(seen)


def _boundary_sizes_for_order(n, boundary_order):
    n = int(n)
    boundary_order = min(_normalize_boundary_order(boundary_order), n)
    sizes = set(range(boundary_order + 1))
    sizes.update(range(max(0, n - boundary_order), n + 1))
    return _normalize_boundary_sizes(n, sizes)


def _boundary_eval_count_for_sizes(n, sizes):
    return int(sum(math.comb(int(n), int(s)) for s in _normalize_boundary_sizes(n, sizes)))


def _adaptive_boundary_sizes(n, total_budget):
    n = int(n)
    total_budget = int(total_budget)
    if total_budget <= 0:
        return []

    all_sizes = list(range(n + 1))
    if (1 << n) <= total_budget:
        return all_sizes

    best = []
    for order in range(n + 1):
        sizes = _boundary_sizes_for_order(n, order)
        cost = _boundary_eval_count_for_sizes(n, sizes)
        if cost < total_budget:
            best = sizes
        else:
            break
    return best


def _resolve_boundary_sizes(n, total_budget, boundary_policy, boundary_order):
    if boundary_policy == "none":
        return []
    if boundary_policy == "fixed":
        return _boundary_sizes_for_order(n, boundary_order)
    if boundary_policy == "adaptive":
        return _adaptive_boundary_sizes(n, total_budget)
    raise RuntimeError(f"Unknown normalized boundary policy {boundary_policy!r}.")


def _boundary_subset_matrix_for_sizes(n, sizes):
    n = int(n)
    sizes = _normalize_boundary_sizes(n, sizes)
    if not sizes:
        return np.empty((0, n), dtype=bool)

    subsets = []
    for size in sizes:
        for combo in itertools.combinations(range(n), size):
            mask = np.zeros(n, dtype=bool)
            if combo:
                mask[list(combo)] = True
            subsets.append(mask)
    return np.vstack(subsets) if subsets else np.empty((0, n), dtype=bool)


def _evaluate_sample_batch(samples, game_func, game_args, num_player):
    """Evaluate a sampled batch without touching the heavy engine/backend graph."""
    game = game_func(**game_args)
    results = np.empty((len(samples), num_player + 2), dtype=np.float64)
    results[:, :num_player] = samples[:, :num_player]
    results[:, num_player + 1] = samples[:, num_player]
    for idx in range(len(samples)):
        results[idx, num_player] = float(game.evaluate(samples[idx, :num_player].astype(bool)))
    return results


def _evaluate_values(game_func, game_args, X):
    if len(X) == 0:
        return np.empty(0, dtype=np.float64)

    game = game_func(**game_args)
    values = np.empty(len(X), dtype=np.float64)
    for idx, subset in enumerate(X):
        values[idx] = float(game.evaluate(subset))
    return values


def _safe_solve(gram, rhs):
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _solve_with_ridge(gram, rhs, ridge):
    diag = np.diag_indices_from(gram)
    ridge = np.asarray(ridge, dtype=np.float64)
    if ridge.ndim == 0:
        gram[diag] += float(ridge)
    else:
        if ridge.shape != (gram.shape[0],):
            raise ValueError(f"Expected ridge diagonal shape {(gram.shape[0],)}, got {ridge.shape}.")
        gram[diag] += ridge
    return _safe_solve(gram, rhs)


def _ridge_as_diag(ridge, dim):
    ridge = np.asarray(ridge, dtype=np.float64)
    if ridge.ndim == 0:
        return np.full(int(dim), float(ridge), dtype=np.float64)
    if ridge.shape != (int(dim),):
        raise ValueError(f"Expected ridge diagonal shape {(int(dim),)}, got {ridge.shape}.")
    return ridge.astype(np.float64, copy=True)


def _solve_profiled_system(A_stat, B_stat, c_stat, b_stat, profile_norm, ridge):
    profile_norm = float(profile_norm)
    if not np.isfinite(profile_norm) or profile_norm <= 0.0:
        raise ValueError(f"Expected a positive finite profiling normalization, got {profile_norm!r}.")
    gram = A_stat - (B_stat.T @ B_stat) / profile_norm
    rhs = c_stat - (B_stat.T @ b_stat) / profile_norm
    return _solve_with_ridge(gram, rhs, ridge)


# ---------------------------------------------------------------------------
# Small data objects
# ---------------------------------------------------------------------------


@dataclass
class _DesignContext:
    sizes: np.ndarray
    overlaps: Optional[np.ndarray] = None
    cell_ids: Optional[np.ndarray] = None


@dataclass
class _FittedSurrogate:
    beta: np.ndarray
    phi: np.ndarray

    def predict(self, Z):
        return Z @ self.beta


class _TrajectoryAdapter:
    def __init__(self, owner, output_dim):
        self.owner = owner
        self.output_dim = int(output_dim)

    def write(self, pos, value):
        value = np.asarray(value, dtype=np.float64)
        if self.output_dim == 1 and self.owner.values_traj.ndim == 1:
            self.owner.values_traj[pos] = float(value[0])
        else:
            self.owner.values_traj[pos] = value

    def write_nan(self, pos):
        self.owner.values_traj[pos] = np.nan

    def fill_tail(self, pos, value):
        value = np.asarray(value, dtype=np.float64)
        if pos < len(self.owner.values_traj):
            if self.output_dim == 1 and self.owner.values_traj.ndim == 1:
                self.owner.values_traj[pos:] = float(value[0])
            else:
                self.owner.values_traj[pos:] = value
        elif len(self.owner.values_traj):
            if self.output_dim == 1 and self.owner.values_traj.ndim == 1:
                self.owner.values_traj[-1] = float(value[0])
            else:
                self.owner.values_traj[-1] = value

    def final_public(self, value):
        value = np.asarray(value, dtype=np.float64)
        if self.output_dim == 1 and self.owner.values_traj.ndim == 1:
            return float(value[0]), self.owner.values_traj
        return value.copy(), self.owner.values_traj


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------


class _SizeStrata:
    def __init__(self, n, sampling_mask):
        self.n = int(n)
        self.keys = list(range(self.n + 1))
        self.counts = np.array(
            [float(special.comb(self.n, s, exact=False)) for s in range(self.n + 1)],
            dtype=np.float64,
        )
        self.sampling_mask = np.asarray(sampling_mask, dtype=bool)

    def context_from_X(self, X):
        X = np.asarray(X, dtype=bool)
        return _DesignContext(sizes=X.sum(axis=1).astype(np.int64))

    def ids_from_context(self, context):
        return context.sizes


class _GroupCellStrata:
    def __init__(self, n, group, group_mask, sampling_mask):
        self.n = int(n)
        self.group = np.asarray(group, dtype=np.int64)
        self.group_mask = np.asarray(group_mask, dtype=bool)
        self.non_group = np.flatnonzero(~self.group_mask)
        self.group_size = int(len(self.group))

        keys = []
        counts = []
        index = -np.ones((self.n + 1, self.group_size + 1), dtype=np.int64)
        for s in range(self.n + 1):
            r_min = max(0, s - (self.n - self.group_size))
            r_max = min(self.group_size, s)
            for r in range(r_min, r_max + 1):
                count = _safe_comb(self.group_size, r) * _safe_comb(self.n - self.group_size, s - r)
                if count <= 0.0:
                    continue
                index[s, r] = len(keys)
                keys.append((s, r))
                counts.append(count)

        self.keys = keys
        self.counts = np.asarray(counts, dtype=np.float64)
        self.index = index
        self.sampling_mask = np.asarray(sampling_mask, dtype=bool)

    @classmethod
    def build(cls, n, group, group_mask, exact_boundary_handling, boundary_sizes=None):
        tmp = cls(n, group, group_mask, np.ones(1, dtype=bool))
        if exact_boundary_handling:
            if boundary_sizes is None:
                boundary_sizes = _boundary_sizes_for_order(n, 1)
            boundary_sizes = set(_normalize_boundary_sizes(n, boundary_sizes))
            sampling_mask = np.array([s not in boundary_sizes for s, _r in tmp.keys], dtype=bool)
        else:
            sampling_mask = np.ones(len(tmp.keys), dtype=bool)
        tmp.sampling_mask = sampling_mask
        return tmp

    def context_from_X(self, X):
        X = np.asarray(X, dtype=bool)
        sizes = X.sum(axis=1).astype(np.int64)
        overlaps = X[:, self.group].sum(axis=1).astype(np.int64)
        cell_ids = self.index[sizes, overlaps]
        return _DesignContext(sizes=sizes, overlaps=overlaps, cell_ids=cell_ids)

    def ids_from_context(self, context):
        return context.cell_ids


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


class _FeatureBuilder:
    def __init__(
        self,
        *,
        n,
        surrogate_basis,
        include_nonlinear_size_terms,
        include_group_overlap_ratio=False,
    ):
        self.n = int(n)
        self.surrogate_basis = surrogate_basis
        self.include_nonlinear_size_terms = bool(include_nonlinear_size_terms)
        self.include_group_overlap_ratio = bool(include_group_overlap_ratio)

        self.surrogate_basis_kind = None
        self.interaction_degree = None
        self._parse_surrogate_basis()

        self.interaction_blocks = []
        self.size_player_start = None
        self.log_col = None
        self.quad_col = None
        self.overlap_ratio_col = None
        self.feature_dim = None
        self._build_layout()

    @property
    def dim(self):
        return self.feature_dim

    def _parse_surrogate_basis(self):
        basis = self.surrogate_basis
        if isinstance(basis, (int, np.integer)):
            degree = int(basis)
            if degree < 0:
                raise ValueError(f"`surrogate_basis` as an integer must be >= 0, got {basis!r}.")
            self.surrogate_basis_kind = "interactions"
            self.interaction_degree = degree
            return

        if not isinstance(basis, str):
            raise ValueError(
                "`surrogate_basis` must be either an integer interaction degree "
                'or one of {"none", "size_player"}, '
                f"got {basis!r}."
            )

        key = basis.strip().lower().replace("-", "_")
        if key in {"none", "constant", "intercept"}:
            self.surrogate_basis_kind = "interactions"
            self.interaction_degree = 0
            return
        if key in {"size_player", "size_by_player", "player_size", "is", "i_s"}:
            self.surrogate_basis_kind = "size_player"
            self.interaction_degree = None
            return

        raise ValueError(
            "Unknown `surrogate_basis`. Use an integer degree, "
            '"none", or "size_player". '
            f"Got {basis!r}."
        )

    def _build_layout(self):
        n = self.n
        next_col = 1
        if self.include_nonlinear_size_terms:
            self.log_col = next_col
            next_col += 1
            self.quad_col = next_col
            next_col += 1
        if self.include_group_overlap_ratio:
            self.overlap_ratio_col = next_col
            next_col += 1

        if self.surrogate_basis_kind == "interactions":
            degree = self.interaction_degree
            if degree > n:
                raise ValueError(f"`surrogate_basis={degree}` exceeds num_player={n}.")
            self.feature_dim = next_col + sum(math.comb(n, r) for r in range(1, degree + 1))
            start = next_col
            for r in range(1, degree + 1):
                m = math.comb(n, r)
                if m == 0:
                    continue
                flat = np.fromiter(
                    itertools.chain.from_iterable(itertools.combinations(range(n), r)),
                    dtype=np.int64,
                    count=m * r,
                )
                combos = flat.reshape(m, r)
                self.interaction_blocks.append((r, combos, start))
                start += m
        elif self.surrogate_basis_kind == "size_player":
            self.size_player_start = next_col
            self.feature_dim = next_col + n * n
        else:
            raise RuntimeError(f"Unexpected surrogate basis kind {self.surrogate_basis_kind!r}.")

    def build(self, X, context):
        X = np.asarray(X, dtype=bool)
        sizes = context.sizes
        Z = np.zeros((len(X), self.feature_dim), dtype=np.float64)
        Z[:, 0] = 1.0

        if self.log_col is not None:
            Z[:, self.log_col] = np.log1p(sizes.astype(np.float64))
            Z[:, self.quad_col] = (sizes.astype(np.float64) / float(self.n)) ** 2

        if self.overlap_ratio_col is not None:
            overlaps = context.overlaps
            if overlaps is None:
                raise ValueError("`overlaps` are required for the overlap-ratio feature.")
            nonzero = sizes > 0
            Z[nonzero, self.overlap_ratio_col] = overlaps[nonzero] / sizes[nonzero]

        if self.surrogate_basis_kind == "interactions":
            for degree, combos, start in self.interaction_blocks:
                width = combos.shape[0]
                if degree == 1:
                    Z[:, start:start + width] = X[:, combos[:, 0]].astype(np.float64)
                else:
                    Z[:, start:start + width] = X[:, combos].all(axis=2).astype(np.float64)
        else:
            X_float = X.astype(np.float64)
            for size in range(1, self.n + 1):
                mask = sizes == size
                if np.any(mask):
                    start = self.size_player_start + (size - 1) * self.n
                    Z[mask, start:start + self.n] = X_float[mask]
        return Z

    def predict_from_rows(self, beta, X, context):
        beta = np.asarray(beta, dtype=np.float64)
        if beta.shape != (self.feature_dim,):
            raise ValueError(f"`beta` must have shape ({self.feature_dim},), got {beta.shape!r}.")

        X = np.asarray(X, dtype=bool)
        sizes = np.asarray(context.sizes, dtype=np.int64)
        if len(X) != len(sizes):
            raise ValueError("`X` and `context.sizes` must have the same number of rows.")

        if self.surrogate_basis_kind == "interactions" and self.interaction_degree > 2:
            return self.build(X, context) @ beta

        out = np.full(len(X), beta[0], dtype=np.float64)
        sizes_float = sizes.astype(np.float64)
        if self.log_col is not None:
            out += beta[self.log_col] * np.log1p(sizes_float)
            out += beta[self.quad_col] * (sizes_float / float(self.n)) ** 2

        if self.overlap_ratio_col is not None:
            overlaps = context.overlaps
            if overlaps is None:
                raise ValueError("`overlaps` are required for the overlap-ratio feature.")
            nonzero = sizes > 0
            overlap_ratio = np.zeros(len(X), dtype=np.float64)
            overlap_ratio[nonzero] = np.asarray(overlaps, dtype=np.float64)[nonzero] / sizes_float[nonzero]
            out += beta[self.overlap_ratio_col] * overlap_ratio

        if self.surrogate_basis_kind == "interactions":
            for degree, combos, start in self.interaction_blocks:
                width = combos.shape[0]
                block = beta[start:start + width]
                if degree == 1:
                    out += X[:, combos[:, 0]] @ block
                elif degree == 2:
                    pair_matrix = np.zeros((self.n, self.n), dtype=np.float64)
                    left = combos[:, 0]
                    right = combos[:, 1]
                    pair_matrix[left, right] = block
                    pair_matrix[right, left] = block
                    max_float_entries = 2_000_000
                    chunk_rows = max(1, max_float_entries // max(self.n, 1))
                    for pos in range(0, len(X), chunk_rows):
                        stop = min(pos + chunk_rows, len(X))
                        X_chunk = X[pos:stop].astype(np.float64)
                        out[pos:stop] += 0.5 * np.sum((X_chunk @ pair_matrix) * X_chunk, axis=1)
                else:
                    raise RuntimeError(f"Unexpected interaction degree {degree!r}.")
            return out

        if self.surrogate_basis_kind != "size_player":
            raise RuntimeError(f"Unexpected surrogate basis kind {self.surrogate_basis_kind!r}.")

        # Size-player rows activate only one size block, so prediction is a
        # size-wise player sum instead of a dense Z @ beta multiply.
        max_float_entries = 2_000_000
        chunk_rows = max(1, max_float_entries // max(self.n, 1))
        for size in range(1, self.n + 1):
            rows = np.flatnonzero(sizes == size)
            if rows.size == 0:
                continue
            start = self.size_player_start + (size - 1) * self.n
            block = beta[start:start + self.n]
            for pos in range(0, rows.size, chunk_rows):
                idx = rows[pos:pos + chunk_rows]
                out[idx] += X[idx].astype(np.float64) @ block
        return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


class _FullSemivalueTarget:
    def __init__(self, *, n, semivalue, semivalue_param, feature_builder, boundary_X, boundary_context):
        self.n = int(n)
        self.output_dim = self.n
        self.semivalue = semivalue
        self.semivalue_param = semivalue_param
        self.feature_builder = feature_builder
        self.dist_card = _distribution_cardinality(self.n, semivalue, semivalue_param)

        p = np.zeros(self.n, dtype=np.float64)
        for k in range(self.n):
            denom = float(special.comb(self.n - 1, k, exact=False))
            if denom > 0.0:
                p[k] = self.dist_card[k] / denom
        self.p = p
        self.is_symmetric = _is_symmetric_semivalue(semivalue, semivalue_param)
        self.zeta = self._build_readout()

        self.boundary_X = np.asarray(boundary_X, dtype=bool)
        self.boundary_context = boundary_context
        self.boundary_Z = (
            feature_builder.build(self.boundary_X, boundary_context)
            if len(self.boundary_X)
            else np.empty((0, feature_builder.dim), dtype=np.float64)
        )
        self.boundary_raw_gamma = self.raw_gamma(self.boundary_X, boundary_context) if len(self.boundary_X) else np.empty((0, self.n))

    def _build_readout(self):
        n = self.n
        fb = self.feature_builder
        zeta = np.zeros((n, fb.dim), dtype=np.float64)

        if fb.log_col is not None:
            s_grid = np.arange(n + 1, dtype=np.float64)
            phi_log_const = float(self.dist_card @ (np.log1p(s_grid[1:]) - np.log1p(s_grid[:-1])))
            quad_grid = (s_grid / float(n)) ** 2
            phi_quad_const = float(self.dist_card @ (quad_grid[1:] - quad_grid[:-1]))
            zeta[:, fb.log_col] = phi_log_const
            zeta[:, fb.quad_col] = phi_quad_const

        if fb.surrogate_basis_kind == "interactions":
            omega_by_degree = {}
            for r in range(1, fb.interaction_degree + 1):
                s_idx = np.arange(r - 1, n, dtype=int)
                comb_terms = special.comb(n - r, s_idx - r + 1, exact=False)
                omega_by_degree[r] = float(np.dot(comb_terms, self.p[s_idx]))

            for r, combos, start in fb.interaction_blocks:
                cols = np.arange(start, start + combos.shape[0], dtype=np.int64)
                omega = omega_by_degree[r]
                for pos in range(r):
                    zeta[combos[:, pos], cols] = omega
        else:
            for s in range(1, n + 1):
                diag = float(special.comb(n - 1, s - 1, exact=False)) * self.p[s - 1]
                alpha_cur = self.p[s] if s < n else 0.0
                off = (
                    float(special.comb(n - 2, s - 2, exact=False)) * self.p[s - 1]
                    - float(special.comb(n - 2, s - 1, exact=False)) * alpha_cur
                )
                block_start = fb.size_player_start + (s - 1) * n
                zeta[:, block_start:block_start + n] = off
                diag_idx = block_start + np.arange(n, dtype=np.int64)
                zeta[np.arange(n, dtype=np.int64), diag_idx] = diag
        return zeta

    def raw_gamma(self, X, context):
        sizes = context.sizes
        X_float = np.asarray(X, dtype=bool).astype(np.float64, copy=False)
        out_value = np.zeros(len(sizes), dtype=np.float64)
        mask_out = sizes < self.n
        if np.any(mask_out):
            out_value[mask_out] = -self.p[sizes[mask_out]]

        in_value = np.zeros(len(sizes), dtype=np.float64)
        mask_in = sizes > 0
        if np.any(mask_in):
            in_value[mask_in] = self.p[sizes[mask_in] - 1]

        return out_value[:, None] + X_float * (in_value - out_value)[:, None]

    def true_stratum_weight(self, strata):
        out = np.zeros(self.n + 1, dtype=np.float64)
        for s in range(self.n + 1):
            val = 0.0
            if s > 0:
                val += float(s) * (self.p[s - 1] ** 2)
            if s < self.n:
                val += float(self.n - s) * (self.p[s] ** 2)
            out[s] = val
        return out

    def initial_design_factor(self, strata):
        return np.sqrt(self.true_stratum_weight(strata))

    def phi_from_beta(self, beta):
        phi = self.zeta @ beta
        if len(self.boundary_X):
            phi -= self.boundary_raw_gamma.T @ (self.boundary_Z @ beta)
        return phi

    def boundary_exact(self, y_boundary):
        if len(self.boundary_raw_gamma) == 0:
            return np.zeros(self.output_dim, dtype=np.float64)
        return self.boundary_raw_gamma.T @ y_boundary


class _GroupSumTarget:
    def __init__(
        self,
        *,
        n,
        group,
        group_mask,
        semivalue,
        semivalue_param,
        feature_builder,
        strata,
        boundary_X,
        boundary_context,
        rho_support_tol,
    ):
        self.n = int(n)
        self.output_dim = 1
        self.group = np.asarray(group, dtype=np.int64)
        self.group_mask = np.asarray(group_mask, dtype=bool)
        self.group_size = int(len(self.group))
        self.alpha = semivalue_coefficients(n, semivalue, semivalue_param)
        self.feature_builder = feature_builder
        self.rho_support_tol = float(rho_support_tol)
        self.cell_rho = np.array(
            [group_sum_coefficient(s, r, self.group_size, self.alpha) for s, r in strata.keys],
            dtype=np.float64,
        )
        self.zeta = self._build_readout(strata)

        self.boundary_X = np.asarray(boundary_X, dtype=bool)
        self.boundary_context = boundary_context
        self.boundary_Z = (
            feature_builder.build(self.boundary_X, boundary_context)
            if len(self.boundary_X)
            else np.empty((0, feature_builder.dim), dtype=np.float64)
        )
        self.boundary_raw_gamma = self.raw_gamma(self.boundary_X, boundary_context) if len(self.boundary_X) else np.empty((0, 1))

    def _cell_readout(self, strata, fn):
        total = 0.0
        for idx, (s, r) in enumerate(strata.keys):
            total += strata.counts[idx] * self.cell_rho[idx] * fn(s, r)
        return float(total)

    def _interaction_readout(self, strata, a, b):
        out = 0.0
        outside_a = a - b
        for idx, (s, r) in enumerate(strata.keys):
            if s < a or r < b:
                continue
            cnt = _safe_comb(self.group_size - b, r - b) * _safe_comb(self.n - self.group_size - outside_a, s - r - outside_a)
            out += cnt * self.cell_rho[idx]
        return float(out)

    def _size_player_readout(self, strata, player, size):
        is_group = bool(self.group_mask[player])
        out = 0.0
        for idx, (s, r) in enumerate(strata.keys):
            if s != size:
                continue
            if is_group:
                cnt = _safe_comb(self.group_size - 1, r - 1) * _safe_comb(self.n - self.group_size, s - r)
            else:
                cnt = _safe_comb(self.group_size, r) * _safe_comb(self.n - self.group_size - 1, s - r - 1)
            out += cnt * self.cell_rho[idx]
        return float(out)

    def _build_readout(self, strata):
        n = self.n
        fb = self.feature_builder
        zeta = np.zeros((1, fb.dim), dtype=np.float64)
        zeta[0, 0] = self._cell_readout(strata, lambda _s, _r: 1.0)
        if fb.log_col is not None:
            zeta[0, fb.log_col] = self._cell_readout(strata, lambda s, _r: np.log1p(float(s)))
            zeta[0, fb.quad_col] = self._cell_readout(strata, lambda s, _r: (float(s) / float(n)) ** 2)
        if fb.overlap_ratio_col is not None:
            zeta[0, fb.overlap_ratio_col] = self._cell_readout(
                strata,
                lambda s, r: 0.0 if s == 0 else float(r) / float(s),
            )

        if fb.surrogate_basis_kind == "interactions":
            for degree, combos, start in fb.interaction_blocks:
                b_vals = self.group_mask[combos].sum(axis=1).astype(np.int64)
                cache = {}
                for b in np.unique(b_vals):
                    cache[int(b)] = self._interaction_readout(strata, degree, int(b))
                zeta[0, start:start + combos.shape[0]] = np.array([cache[int(b)] for b in b_vals], dtype=np.float64)
        else:
            for size in range(1, n + 1):
                start = fb.size_player_start + (size - 1) * n
                for player in range(n):
                    zeta[0, start + player] = self._size_player_readout(strata, player, size)
        return zeta

    def raw_gamma(self, X, context):
        cell_ids = context.cell_ids
        if np.any(cell_ids < 0):
            raise ValueError("Encountered invalid (size, overlap) cell.")
        return self.cell_rho[cell_ids][:, None]

    def true_stratum_weight(self, strata):
        return self.cell_rho * self.cell_rho

    def initial_design_factor(self, strata):
        abs_rho = np.abs(self.cell_rho)
        max_abs = float(abs_rho.max(initial=0.0))
        if max_abs <= 0.0:
            return abs_rho
        support = abs_rho > self.rho_support_tol * max_abs
        support &= strata.sampling_mask
        return np.where(support, abs_rho, 0.0)

    def phi_from_beta(self, beta):
        value = self.zeta @ beta
        if len(self.boundary_X):
            value -= self.boundary_raw_gamma.T @ (self.boundary_Z @ beta)
        return value

    def boundary_exact(self, y_boundary):
        if len(self.boundary_raw_gamma) == 0:
            return np.zeros(1, dtype=np.float64)
        return self.boundary_raw_gamma.T @ y_boundary


# ---------------------------------------------------------------------------
# Sampling law and storage
# ---------------------------------------------------------------------------


class _StratifiedLaw:
    def __init__(self, *, n, strata, is_paired=False):
        self.n = int(n)
        self.strata = strata
        self.is_paired = bool(is_paired)
        self.pair_stride = 2 if self.is_paired else 1
        if self.is_paired:
            assert isinstance(strata, _SizeStrata)

    def _checked_factor(self, factor, name):
        factor = np.asarray(factor, dtype=np.float64)
        expected_shape = (len(self.strata.keys),)
        if factor.shape != expected_shape:
            raise ValueError(f"Expected `{name}` shape {expected_shape}, got {factor.shape}.")
        if not np.all(np.isfinite(factor)):
            raise ValueError(f"Encountered non-finite entries in `{name}`.")
        return factor

    def normalize_density(self, factor):
        factor = self._checked_factor(factor, "factor")
        # Paired laws receive already-symmetric factors from the target/pilot moment.
        factor = np.maximum(factor, 0.0)
        factor = np.where(self.strata.sampling_mask, factor, 0.0)
        total = float(np.dot(self.strata.counts, factor))
        if total <= 0.0:
            fallback = self.strata.sampling_mask.astype(np.float64)
            total = float(np.dot(self.strata.counts, fallback))
            if total <= 0.0:
                return np.zeros_like(factor)
            factor = fallback
        return factor / total

    def mix_with_initial_mass(self, factor, q_init, mix_weight):
        factor = self._checked_factor(factor, "factor")
        q_init = self._checked_factor(q_init, "q_init")
        factor = np.maximum(factor, 0.0)
        factor = np.where(self.strata.sampling_mask, factor, 0.0)
        target_mass = self.strata.counts * factor
        target_total = float(target_mass.sum())
        if target_total <= 0.0:
            return q_init.copy()

        target_mass = target_mass / target_total
        init_mass = self.strata.counts * q_init
        init_total = float(init_mass.sum())
        if init_total > 0.0:
            init_mass = init_mass / init_total
            target_mass = (1.0 - mix_weight) * target_mass + mix_weight * init_mass

        q_density = np.zeros_like(factor)
        valid = (self.strata.counts > 0.0) & self.strata.sampling_mask
        q_density[valid] = target_mass[valid] / self.strata.counts[valid]
        return q_density

    def batch_rows(self, requested_rows, remaining_rows):
        cur = min(requested_rows, remaining_rows)
        if not self.is_paired:
            return cur
        if cur % 2 == 1:
            cur -= 1
        if cur <= 0:
            cur = min(remaining_rows, 2)
        return cur

    def assign_folds(self, fold_rng, num_rows, num_folds):
        if not self.is_paired:
            return fold_rng.integers(num_folds, size=num_rows)
        if num_rows % 2 != 0:
            raise ValueError("Complement-paired blocks must contain an even number of rows.")
        pair_folds = fold_rng.integers(num_folds, size=num_rows // 2)
        return np.repeat(pair_folds, 2)

    def _mass(self, q_density):
        mass = self.strata.counts * q_density
        total = float(mass.sum())
        if total <= 0.0:
            raise ValueError("Encountered non-positive sampling mass.")
        return mass / total

    def sample_batch(self, rng, num_rows, q_density):
        if isinstance(self.strata, _SizeStrata):
            return self._sample_size_batch(rng, num_rows, q_density)
        return self._sample_group_cell_batch(rng, num_rows, q_density)

    def _sample_size_batch(self, rng, num_rows, q_density):
        n = self.n
        mass = self._mass(q_density)
        out = np.empty((num_rows, n + 1), dtype=np.float64)

        if not self.is_paired:
            sampled_sizes = rng.choice(np.arange(n + 1), size=num_rows, p=mass)
            for row, s in enumerate(sampled_sizes):
                s = int(s)
                subset = np.zeros(n, dtype=bool)
                if s > 0:
                    subset[rng.choice(n, size=s, replace=False)] = True
                out[row, :n] = subset.astype(np.float64)
                out[row, n] = q_density[s]
            return out

        num_pairs = num_rows // 2
        sampled_sizes = rng.choice(np.arange(n + 1), size=num_pairs, p=mass)
        for pair_idx, s in enumerate(sampled_sizes):
            s = int(s)
            subset = np.zeros(n, dtype=bool)
            if s > 0:
                subset[rng.choice(n, size=s, replace=False)] = True
            comp = ~subset

            row = 2 * pair_idx
            out[row, :n] = subset.astype(np.float64)
            out[row, n] = q_density[s]
            out[row + 1, :n] = comp.astype(np.float64)
            out[row + 1, n] = q_density[int(comp.sum())]
        return out

    def _sample_group_cell_batch(self, rng, num_rows, q_density):
        strata = self.strata
        mass = self._mass(q_density)
        cell_ids = rng.choice(np.arange(len(strata.keys)), size=num_rows, p=mass)
        out = np.empty((num_rows, self.n + 1), dtype=np.float64)
        for row, cell_id in enumerate(cell_ids):
            s, r = strata.keys[int(cell_id)]
            subset = np.zeros(self.n, dtype=bool)
            if r > 0:
                subset[rng.choice(strata.group, size=r, replace=False)] = True
            outside_count = s - r
            if outside_count > 0:
                subset[rng.choice(strata.non_group, size=outside_count, replace=False)] = True
            out[row, :self.n] = subset.astype(np.float64)
            out[row, self.n] = q_density[int(cell_id)]
        return out


class _ObservationStore:
    def __init__(self, *, num_sample, n):
        self.num_sample = int(num_sample)
        self.n = int(n)
        self.num_obs = 0
        self.X = np.empty((self.num_sample, self.n), dtype=bool)
        self.y = np.empty(self.num_sample, dtype=np.float64)
        self.q = np.empty(self.num_sample, dtype=np.float64)
        self.folds = np.empty(self.num_sample, dtype=np.int64)

    def append(self, X, y, q, folds):
        start = self.num_obs
        end = start + len(y)
        self.X[start:end] = X
        self.y[start:end] = y
        self.q[start:end] = q
        self.folds[start:end] = folds
        self.num_obs = end

    def rows(self, mask=None):
        if mask is None:
            return self.X[:self.num_obs], self.y[:self.num_obs], self.q[:self.num_obs], self.folds[:self.num_obs]
        return self.X[:self.num_obs][mask], self.y[:self.num_obs][mask], self.q[:self.num_obs][mask], self.folds[:self.num_obs][mask]

    def rows_until(self, count):
        count = min(int(count), self.num_obs)
        return self.X[:count], self.y[:count], self.q[:count], self.folds[:count]

    def fold_mask(self, fold):
        return self.folds[:self.num_obs] == fold

    def estimate_memory_bytes(self):
        return int(self.num_sample * self.n + self.num_sample * 8 * 3)


# ---------------------------------------------------------------------------
# Stats backends
# ---------------------------------------------------------------------------


class _StatsBackend:
    def append(self, X, y, q, folds):
        raise NotImplementedError

    def fit_all(self, store=None):
        raise NotImplementedError

    def fit_excluding_fold(self, fold, store=None):
        raise NotImplementedError

    def fit_candidate(self, X, y, q0, q_candidate):
        """Fit the pilot-only candidate-law criterion without mutating accumulated stats."""
        raise NotImplementedError

    def fold_count(self, fold):
        raise NotImplementedError

    def estimate_memory_bytes(self):
        raise NotImplementedError


class _EmpiricalDenseStats(_StatsBackend):
    def __init__(
        self,
        *,
        target,
        strata,
        feature_builder,
        ridge_lambda,
        ridge_schedule,
        num_folds,
    ):
        self.target = target
        self.strata = strata
        self.feature_builder = feature_builder
        self.ridge_lambda = float(ridge_lambda)
        self.ridge_schedule = str(ridge_schedule)
        self.num_folds = int(num_folds)

        d = feature_builder.dim
        o = target.output_dim
        self.total_count = 0
        self.A_total = np.zeros((d, d), dtype=np.float64)
        self.c_total = np.zeros(d, dtype=np.float64)
        self.B_total = np.zeros((o, d), dtype=np.float64)
        self.b_total = np.zeros(o, dtype=np.float64)

        self.fold_counts = np.zeros(self.num_folds, dtype=np.int64)
        self.A_fold = np.zeros((self.num_folds, d, d), dtype=np.float64)
        self.c_fold = np.zeros((self.num_folds, d), dtype=np.float64)
        self.B_fold = np.zeros((self.num_folds, o, d), dtype=np.float64)
        self.b_fold = np.zeros((self.num_folds, o), dtype=np.float64)

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim, num_folds):
        d = int(feature_dim)
        o = int(output_dim)
        k = int(num_folds)
        return int(
            8 * d * d * (k + 1)
            + 8 * d * (k + 1)
            + 8 * o * d * (k + 1)
            + 8 * o * (k + 1)
            + 8 * k
        )

    def estimate_memory_bytes(self):
        return self.estimate_memory_bytes_for(
            feature_dim=self.feature_builder.dim,
            output_dim=self.target.output_dim,
            num_folds=self.num_folds,
        )

    def _effective_ridge_lambda(self, count):
        if self.ridge_schedule == "fixed":
            return self.ridge_lambda
        return float(count) * self.ridge_lambda

    def append(self, X, y, q, folds):
        context = self.strata.context_from_X(X)
        Z = self.feature_builder.build(X, context)
        gamma = self.target.raw_gamma(X, context) / q[:, None]
        ids = self.strata.ids_from_context(context)
        if np.any(ids < 0):
            raise ValueError("Encountered invalid stratum id.")
        weights = self.target.true_stratum_weight(self.strata)[ids] / (q ** 2)

        wz = weights[:, None] * Z
        self.A_total += Z.T @ wz
        self.c_total += Z.T @ (weights * y)
        self.B_total += gamma.T @ Z
        self.b_total += gamma.T @ y
        self.total_count += len(y)

        for fold in np.unique(folds):
            mask = folds == fold
            if not np.any(mask):
                continue
            Zk = Z[mask]
            yk = y[mask]
            wk = weights[mask]
            gk = gamma[mask]
            self.A_fold[fold] += Zk.T @ (wk[:, None] * Zk)
            self.c_fold[fold] += Zk.T @ (wk * yk)
            self.B_fold[fold] += gk.T @ Zk
            self.b_fold[fold] += gk.T @ yk
            self.fold_counts[fold] += int(mask.sum())

    def _fit_from_internal_stats(self, A_stat, c_stat, B_stat, b_stat, count):
        if count <= 0:
            beta = np.zeros(self.feature_builder.dim, dtype=np.float64)
            return _FittedSurrogate(beta=beta, phi=self.target.phi_from_beta(beta))

        beta = _solve_profiled_system(
            A_stat,
            B_stat,
            c_stat,
            b_stat,
            count,
            self._effective_ridge_lambda(count),
        )
        return _FittedSurrogate(beta=beta, phi=self.target.phi_from_beta(beta))

    def fit_all(self, store=None):
        return self._fit_from_internal_stats(
            self.A_total,
            self.c_total,
            self.B_total,
            self.b_total,
            self.total_count,
        )

    def fit_excluding_fold(self, fold, store=None):
        train_count = self.total_count - int(self.fold_counts[fold])
        return self._fit_from_internal_stats(
            self.A_total - self.A_fold[fold],
            self.c_total - self.c_fold[fold],
            self.B_total - self.B_fold[fold],
            self.b_total - self.b_fold[fold],
            train_count,
        )

    def fit_candidate(self, X, y, q0, q_candidate):
        """Build and solve the dense empirical system for a candidate pilot law."""
        X = np.asarray(X, dtype=bool)
        y = np.asarray(y, dtype=np.float64)
        q0 = np.asarray(q0, dtype=np.float64)
        q_candidate = np.asarray(q_candidate, dtype=np.float64)
        if y.shape != (len(X),) or q0.shape != (len(X),) or q_candidate.shape != (len(X),):
            raise ValueError("Candidate-law pilot arrays must have the same row count.")
        if len(y) == 0:
            beta = np.zeros(self.feature_builder.dim, dtype=np.float64)
            return _FittedSurrogate(beta=beta, phi=self.target.phi_from_beta(beta))
        if np.any(~np.isfinite(q0)) or np.any(q0 <= 0.0):
            raise ValueError("Pilot probabilities must be positive and finite.")
        if np.any(~np.isfinite(q_candidate)) or np.any(q_candidate <= 0.0):
            raise ValueError("Candidate probabilities must be positive and finite on all pilot rows.")

        context = self.strata.context_from_X(X)
        ids = self.strata.ids_from_context(context)
        if np.any(ids < 0):
            raise ValueError("Encountered invalid stratum id.")
        Z = self.feature_builder.build(X, context)
        true_weights = self.target.true_stratum_weight(self.strata)[ids]
        R_weights = true_weights / (q0 * q_candidate)
        gamma0 = self.target.raw_gamma(X, context) / q0[:, None]

        A_stat = Z.T @ (R_weights[:, None] * Z)
        c_stat = Z.T @ (R_weights * y)
        B_stat = gamma0.T @ Z
        b_stat = gamma0.T @ y
        profile_norm = float(np.sum(q_candidate / q0))
        beta = _solve_profiled_system(
            A_stat,
            B_stat,
            c_stat,
            b_stat,
            profile_norm,
            self._effective_ridge_lambda(len(y)),
        )
        return _FittedSurrogate(beta=beta, phi=self.target.phi_from_beta(beta))

    def fold_count(self, fold):
        return int(self.fold_counts[fold])


class _ExactStatsBase(_StatsBackend):
    """Exact design-stat backend base.

    Exact backends must not divide by q for strata removed by boundary handling.
    Columns depending only on unidentified strata must be dropped, treated as
    ridge-only, or rejected explicitly.
    """

    def _design_counts(self, excluding_fold=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Exact conditional solvers
# ---------------------------------------------------------------------------


class _ExactConditionalSolver:
    """Linear solver interface for exact conditional design statistics.

    Solvers are private companions of ``_ExactConditionalStats``. They may use
    the exact-stat combinatorial caches and helpers on ``stats``; their narrow
    responsibility is to return the fitted surrogate coefficient vector.
    """

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim):
        return 0

    @classmethod
    def needs_dense_exact_tables(cls):
        return True

    def solve(
        self,
        *,
        stats,
        R_factor,
        U_factor,
        c_stat,
        b_stat,
        count,
        ridge,
        store=None,
        excluding_fold=None,
    ):
        raise NotImplementedError


class _ExactDenseConditionalSolver(_ExactConditionalSolver):
    """Dense reference solver for the profiled exact conditional system."""

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim):
        d = int(feature_dim)
        o = int(output_dim)
        return int(8 * d * d + 8 * o * d)

    def solve(
        self,
        *,
        stats,
        R_factor,
        U_factor,
        c_stat,
        b_stat,
        count,
        ridge,
        store=None,
        excluding_fold=None,
    ):
        A_stat = stats._build_exact_R(R_factor)
        B_stat = stats._build_exact_U(U_factor)
        return _solve_profiled_system(A_stat, B_stat, c_stat, b_stat, count, ridge)


class _ExactCorrectedSolverBase(_ExactConditionalSolver):
    """Linear solver interface for empirically corrected exact systems.

    Kept as a distinct base so dense and future matrix-free corrected solvers
    can be identified through a shared type.
    """

    @classmethod
    def stores_empirical_U(cls):
        return True


class _ExactMatrixFreeCorrectedSolver(_ExactCorrectedSolverBase):
    """Preconditioned CG solver for empirically corrected exact systems.

    Memory estimates cover persistent solver workspace. Per-solve empirical
    caches are row-dependent: first-order and size-player use O(m n) cache
    beyond the observation store, while second-order also stores the flattened
    observed pair indices O(sum_t |S_t|^2).
    """

    def __init__(self):
        self.last_num_iter = None
        self.last_residual_norm = None
        self.last_relative_residual = None
        self.last_converged = None

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim):
        d = int(feature_dim)
        o = int(output_dim)
        return int(8 * (16 * d + 8 * o + 4 * o * o))

    @classmethod
    def needs_dense_exact_tables(cls):
        return False

    @classmethod
    def stores_empirical_U(cls):
        return False

    def _validate(self, stats):
        if stats._target_kind != "full" or not isinstance(stats.strata, _SizeStrata):
            raise NotImplementedError(
                "Matrix-free corrected solves currently support full size-stratified targets only."
            )
        if stats.feature_builder.overlap_ratio_col is not None:
            raise NotImplementedError("Matrix-free corrected solves do not support group overlap-ratio features.")
        if stats.solver.__class__.needs_dense_exact_tables():
            raise ValueError("Matrix-free correction requires a structured exact solver mode.")
        if abs(stats.r_correction_alpha - stats.u_correction_alpha) > 1e-12:
            raise ValueError("Matrix-free correction requires `r_correction_alpha == u_correction_alpha`.")
        if not isinstance(
            stats.solver,
            (_ExactFirstOrderInteractionSolver, _ExactSecondOrderSolver, _ExactSizePlayerSolver),
        ):
            raise NotImplementedError("Unsupported structured solver for matrix-free correction.")

    def _training_rows(self, *, store, excluding_fold):
        if store is None:
            raise ValueError("Matrix-free empirical correction requires an observation store.")
        X, _y, q, folds = store.rows()
        if excluding_fold is not None:
            mask = folds != int(excluding_fold)
            X = X[mask]
            q = q[mask]
        return X, q

    def _global_cols(self, fb):
        cols = [0]
        if fb.log_col is not None:
            cols.extend([fb.log_col, fb.quad_col])
        return np.asarray(cols, dtype=np.int64)

    def _global_values(self, fb, sizes):
        values = [np.ones(len(sizes), dtype=np.float64)]
        if fb.log_col is not None:
            sizes_float = sizes.astype(np.float64)
            values.append(np.log1p(sizes_float))
            values.append((sizes_float / float(fb.n)) ** 2)
        return np.column_stack(values)

    def _build_pair_row_cache(self, X, n):
        # Second-order rows reuse this flattened pair-index cache across all
        # PCG iterations to avoid nested Python pair loops inside matvecs.
        row_starts = np.zeros(len(X) + 1, dtype=np.int64)
        pair_chunks = []
        total = 0
        for row, subset in enumerate(X):
            players = np.flatnonzero(subset)
            if len(players) >= 2:
                left_pos, right_pos = np.triu_indices(len(players), k=1)
                left = players[left_pos]
                right = players[right_pos]
                pair_chunks.append(left * (2 * int(n) - left - 1) // 2 + (right - left - 1))
                total += len(left)
            row_starts[row + 1] = total
        pair_indices = (
            np.concatenate(pair_chunks).astype(np.int64, copy=False)
            if pair_chunks
            else np.empty(0, dtype=np.int64)
        )
        return row_starts, pair_indices

    def _build_empirical_cache(self, *, stats, X, q0, q_candidate=None):
        context = stats.strata.context_from_X(X)
        ids = stats.strata.ids_from_context(context)
        if np.any(ids < 0):
            raise ValueError("Encountered invalid stratum id.")

        fb = stats.feature_builder
        sizes = context.sizes.astype(np.int64, copy=False)
        q0 = np.asarray(q0, dtype=np.float64)
        if q_candidate is None:
            q_candidate = q0
        q_candidate = np.asarray(q_candidate, dtype=np.float64)
        if q0.shape != (len(X),) or q_candidate.shape != (len(X),):
            raise ValueError("Candidate-law probabilities must match the empirical row count.")
        if np.any(~np.isfinite(q0)) or np.any(q0 <= 0.0):
            raise ValueError("Pilot probabilities must be positive and finite.")
        if np.any(~np.isfinite(q_candidate)) or np.any(q_candidate <= 0.0):
            raise ValueError("Candidate probabilities must be positive and finite on all pilot rows.")
        cache = {
            "X": np.asarray(X, dtype=bool),
            "q0": q0,
            "q_candidate": q_candidate,
            "U_weights": 1.0 / q0,
            "sizes": sizes,
            "R_weights": stats.target.true_stratum_weight(stats.strata)[ids] / (q0 * q_candidate),
            "global_cols": self._global_cols(fb),
            "global_values": self._global_values(fb, sizes),
        }

        out_value = np.zeros(len(X), dtype=np.float64)
        has_out = sizes < fb.n
        out_value[has_out] = -stats.target.p[sizes[has_out]]

        in_value = np.zeros(len(X), dtype=np.float64)
        has_in = sizes > 0
        in_value[has_in] = stats.target.p[sizes[has_in] - 1]
        cache["gamma_out"] = out_value
        cache["gamma_delta"] = in_value - out_value

        if fb.surrogate_basis_kind == "interactions" and fb.interaction_degree >= 2:
            row_starts, pair_indices = self._build_pair_row_cache(cache["X"], fb.n)
            cache["pair_row_starts"] = row_starts
            cache["pair_indices"] = pair_indices
        return cache

    def _default_max_iter(self, stats):
        if stats.correction_max_iter is not None:
            return stats.correction_max_iter
        # Size-player uses the cheaper base inverse, not the fully profiled
        # exact solve, as preconditioner, so it gets more default correction steps.
        if isinstance(stats.solver, _ExactSizePlayerSolver):
            return 20
        return 5

    def _pair_reduce_rows(self, pair_values, row_starts, num_rows):
        out = np.zeros(num_rows, dtype=np.float64)
        lengths = np.diff(row_starts)
        nonempty = lengths > 0
        if np.any(nonempty):
            starts = row_starts[:-1][nonempty]
            out[nonempty] = np.add.reduceat(pair_values, starts)
        return out

    def _feature_matrix_apply(self, stats, cache, v):
        fb = stats.feature_builder
        X = cache["X"]
        sizes = cache["sizes"]
        out = cache["global_values"] @ v[cache["global_cols"]]

        if fb.surrogate_basis_kind == "interactions":
            singleton_start = fb.interaction_blocks[0][2]
            out = out + X @ v[singleton_start:singleton_start + fb.n]
            if fb.interaction_degree >= 2:
                pair_start = fb.interaction_blocks[1][2]
                pair_indices = cache["pair_indices"]
                if len(pair_indices):
                    pair_values = v[pair_start + pair_indices]
                    out = out + self._pair_reduce_rows(pair_values, cache["pair_row_starts"], len(X))
            return out

        if fb.surrogate_basis_kind == "size_player":
            for size in range(1, fb.n + 1):
                rows = sizes == size
                if not np.any(rows):
                    continue
                start = fb.size_player_start + (size - 1) * fb.n
                out[rows] += X[rows] @ v[start:start + fb.n]
            return out

        raise RuntimeError(f"Unexpected surrogate basis kind {fb.surrogate_basis_kind!r}.")

    def _feature_transpose_apply(self, stats, cache, weights):
        fb = stats.feature_builder
        X = cache["X"]
        sizes = cache["sizes"]
        weights = np.asarray(weights, dtype=np.float64)
        out = np.zeros(fb.dim, dtype=np.float64)
        out[cache["global_cols"]] = cache["global_values"].T @ weights

        if fb.surrogate_basis_kind == "interactions":
            singleton_start = fb.interaction_blocks[0][2]
            out[singleton_start:singleton_start + fb.n] = X.T @ weights
            if fb.interaction_degree >= 2:
                pair_indices = cache["pair_indices"]
                if len(pair_indices):
                    pair_start = fb.interaction_blocks[1][2]
                    row_lengths = np.diff(cache["pair_row_starts"])
                    pair_weights = np.repeat(weights, row_lengths)
                    np.add.at(
                        out[pair_start:pair_start + math.comb(fb.n, 2)],
                        pair_indices,
                        pair_weights,
                    )
            return out

        if fb.surrogate_basis_kind == "size_player":
            for size in range(1, fb.n + 1):
                rows = sizes == size
                if not np.any(rows):
                    continue
                start = fb.size_player_start + (size - 1) * fb.n
                out[start:start + fb.n] = X[rows].T @ weights[rows]
            return out

        raise RuntimeError(f"Unexpected surrogate basis kind {fb.surrogate_basis_kind!r}.")

    def _empirical_R_apply(self, *, stats, cache, v):
        if len(cache["X"]) == 0:
            return np.zeros(stats.feature_builder.dim, dtype=np.float64)
        row_dot = self._feature_matrix_apply(stats, cache, v)
        return self._feature_transpose_apply(stats, cache, cache["R_weights"] * row_dot)

    def _raw_gamma_transpose_apply(self, *, cache, row_values):
        row_values = np.asarray(row_values, dtype=np.float64)
        if row_values.shape != (len(cache["X"]),):
            raise ValueError("Raw-gamma row values must match the empirical row count.")
        out = np.zeros(cache["X"].shape[1], dtype=np.float64)
        if len(cache["X"]) == 0:
            return out
        common = float(np.dot(row_values, cache["gamma_out"]))
        if common != 0.0:
            out += common
        out += cache["X"].T @ (row_values * cache["gamma_delta"])
        return out

    def _empirical_U_apply(self, *, stats, cache, v):
        if len(cache["X"]) == 0:
            return np.zeros(stats.feature_builder.n, dtype=np.float64)
        row_dot = self._feature_matrix_apply(stats, cache, v)
        return self._raw_gamma_transpose_apply(
            cache=cache,
            row_values=row_dot * cache["U_weights"],
        )

    def _empirical_U_transpose_apply(self, *, stats, cache, values):
        out = np.zeros(stats.feature_builder.dim, dtype=np.float64)
        if len(cache["X"]) == 0:
            return out
        value_sum = float(np.sum(values))
        included_sum = cache["X"] @ values
        gamma_dot = (
            cache["gamma_out"] * value_sum
            + cache["gamma_delta"] * included_sum
        ) * cache["U_weights"]
        return self._feature_transpose_apply(stats, cache, gamma_dot)

    def _exact_first_order_ops(self, *, stats, R_factor, U_factor, ridge, count):
        solver = stats.solver
        singleton_start = solver._validate(stats)
        fb = stats.feature_builder
        n = fb.n
        global_cols = solver._global_cols(fb)
        player_cols = singleton_start + np.arange(n, dtype=np.int64)
        R_gg, R_gz, a_R, b_R, U_g, a_U, b_U = solver._moments(stats, R_factor, U_factor, global_cols)
        zero_b = np.zeros(stats.target.output_dim, dtype=np.float64)

        def R_apply(v):
            out = np.zeros(fb.dim, dtype=np.float64)
            v_g = v[global_cols]
            v_z = v[player_cols]
            z_sum = float(np.sum(v_z))
            out[global_cols] = R_gg @ v_g + R_gz * z_sum
            out[player_cols] = float(R_gz @ v_g) + a_R * v_z + b_R * z_sum
            return out

        def U_apply(v):
            v_g = v[global_cols]
            v_z = v[player_cols]
            return float(U_g @ v_g) + a_U * v_z + b_U * float(np.sum(v_z))

        def U_transpose_apply(values):
            value_sum = float(np.sum(values))
            out = np.zeros(fb.dim, dtype=np.float64)
            out[global_cols] = U_g * value_sum
            out[player_cols] = a_U * values + b_U * value_sum
            return out

        def precondition(residual):
            return solver.solve(
                stats=stats,
                R_factor=R_factor,
                U_factor=U_factor,
                c_stat=residual,
                b_stat=zero_b,
                count=count,
                ridge=ridge,
            )

        return R_apply, U_apply, U_transpose_apply, precondition

    def _exact_second_order_ops(self, *, stats, R_factor, U_factor, ridge, count):
        solver = stats.solver
        singleton_start, pair_start, pairs = solver._validate(stats)
        fb = stats.feature_builder
        n = fb.n
        if n < 4:
            raise NotImplementedError("Matrix-free corrected second-order solves require n >= 4.")
        num_pairs = len(pairs)
        global_cols = solver._global_cols(fb)
        singleton_cols = singleton_start + np.arange(n, dtype=np.int64)
        pair_cols = pair_start + np.arange(num_pairs, dtype=np.int64)
        R_gg, R_g1, R_g2, P, U_g, U_scalars = solver._moments(stats, R_factor, U_factor, global_cols)
        P_1, P_2, P_3, P_4 = P
        delta_1, gamma_1, delta_2, gamma_2 = U_scalars
        zero_b = np.zeros(stats.target.output_dim, dtype=np.float64)

        def R_apply(v):
            out = np.zeros(fb.dim, dtype=np.float64)
            v_g = v[global_cols]
            v_1 = v[singleton_cols]
            v_2 = v[pair_cols]
            sum_1 = float(np.sum(v_1))
            sum_2 = float(np.sum(v_2))
            M_v2 = solver._M_times_pair(v_2, pairs, n)
            MT_M_v2 = solver._MT_times_player(M_v2, pairs)
            out[global_cols] = R_gg @ v_g + R_g1 * sum_1 + R_g2 * sum_2
            out[singleton_cols] = (
                float(R_g1 @ v_g)
                + (P_1 - P_2) * v_1
                + P_2 * sum_1
                + P_3 * sum_2
                + (P_2 - P_3) * M_v2
            )
            out[pair_cols] = (
                float(R_g2 @ v_g)
                + P_3 * sum_1
                + (P_2 - P_3) * solver._MT_times_player(v_1, pairs)
                + P_4 * sum_2
                + (P_3 - P_4) * MT_M_v2
                + (P_2 - 2.0 * P_3 + P_4) * v_2
            )
            return out

        def U_apply(v):
            v_g = v[global_cols]
            v_1 = v[singleton_cols]
            v_2 = v[pair_cols]
            return (
                float(U_g @ v_g)
                + delta_1 * v_1
                + gamma_1 * float(np.sum(v_1))
                + delta_2 * solver._M_times_pair(v_2, pairs, n)
                + gamma_2 * float(np.sum(v_2))
            )

        def U_transpose_apply(values):
            value_sum = float(np.sum(values))
            out = np.zeros(fb.dim, dtype=np.float64)
            out[global_cols] = U_g * value_sum
            out[singleton_cols] = delta_1 * values + gamma_1 * value_sum
            out[pair_cols] = delta_2 * solver._MT_times_player(values, pairs) + gamma_2 * value_sum
            return out

        def precondition(residual):
            return solver.solve(
                stats=stats,
                R_factor=R_factor,
                U_factor=U_factor,
                c_stat=residual,
                b_stat=zero_b,
                count=count,
                ridge=ridge,
            )

        return R_apply, U_apply, U_transpose_apply, precondition

    def _exact_size_player_ops(self, *, stats, R_factor, U_factor, ridge):
        solver = stats.solver
        fb = stats.feature_builder
        global_cols = solver._global_cols(fb)
        R_gg, R_g_blocks, R_contrast, R_mean_rank, _U_g, _U_delta, _U_gamma = solver._moments(
            stats,
            R_factor,
            U_factor,
            global_cols,
        )
        structure = solver._build_base_structure(stats=stats, R_factor=R_factor, U_factor=U_factor, ridge=ridge)
        solver._cache_symmetric_solver(structure)
        n = structure["n"]
        block_start = structure["block_start"]

        def R_apply(v):
            out = np.zeros(fb.dim, dtype=np.float64)
            v_g = v[global_cols]
            blocks = v[block_start:block_start + n * n].reshape(n, n)
            block_sums = blocks.sum(axis=1)
            out[global_cols] = R_gg @ v_g + R_g_blocks.T @ block_sums
            block_common = R_g_blocks @ v_g + R_mean_rank * block_sums
            block_out = R_contrast[:, None] * blocks + block_common[:, None]
            out[block_start:block_start + n * n] = block_out.reshape(n * n)
            return out

        def U_apply(v):
            return solver._apply_U(structure, v)

        def U_transpose_apply(values):
            return solver._apply_U_transpose(structure, values)

        def precondition(residual):
            return solver._apply_base_inverse(structure, residual)

        return R_apply, U_apply, U_transpose_apply, precondition

    def _exact_ops(self, *, stats, R_factor, U_factor, ridge, count):
        if isinstance(stats.solver, _ExactFirstOrderInteractionSolver):
            return self._exact_first_order_ops(
                stats=stats,
                R_factor=R_factor,
                U_factor=U_factor,
                ridge=ridge,
                count=count,
            )
        if isinstance(stats.solver, _ExactSecondOrderSolver):
            return self._exact_second_order_ops(
                stats=stats,
                R_factor=R_factor,
                U_factor=U_factor,
                ridge=ridge,
                count=count,
            )
        if isinstance(stats.solver, _ExactSizePlayerSolver):
            return self._exact_size_player_ops(stats=stats, R_factor=R_factor, U_factor=U_factor, ridge=ridge)
        raise NotImplementedError("Unsupported structured solver for matrix-free correction.")

    def _active_mask(self, *, stats, R_factor, ridge, empirical_cache=None):
        d = stats.feature_builder.dim
        active = _ridge_as_diag(ridge, d) > 100.0 * np.finfo(np.float64).eps
        for stratum_idx, factor in enumerate(R_factor):
            if factor <= 100.0 * np.finfo(np.float64).eps:
                continue
            cols = np.flatnonzero(
                stats._feature_active[stratum_idx]
                & (stats._feature_scale[stratum_idx] != 0.0)
            )
            if len(cols) == 0:
                continue
            active[cols] |= stats._feature_mean_prob(stratum_idx, cols) > 100.0 * np.finfo(np.float64).eps
        if empirical_cache is not None and len(empirical_cache["X"]):
            # All supported empirical features are nonnegative, so a positive
            # column sum is equivalent to at least one observed active entry.
            empirical_support = self._feature_transpose_apply(
                stats,
                empirical_cache,
                np.ones(len(empirical_cache["X"]), dtype=np.float64),
            )
            active |= np.abs(empirical_support) > 100.0 * np.finfo(np.float64).eps
        return active

    def _record_pcg_diagnostics(self, *, num_iter, residual, rhs_norm, converged):
        residual_norm = float(np.linalg.norm(residual))
        self.last_num_iter = int(num_iter)
        self.last_residual_norm = residual_norm
        self.last_relative_residual = residual_norm / max(1.0, float(rhs_norm))
        self.last_converged = bool(converged)

    def _pcg(self, *, operator, precondition, rhs, x0, active, max_iter, tol):
        x = np.asarray(x0, dtype=np.float64).copy()
        x[~active] = 0.0
        rhs_active = rhs.copy()
        rhs_active[~active] = 0.0
        residual = rhs_active - operator(x)
        residual[~active] = 0.0

        rhs_norm = float(np.linalg.norm(rhs_active))
        residual_norm = float(np.linalg.norm(residual))
        if rhs_norm == 0.0:
            self._record_pcg_diagnostics(num_iter=0, residual=residual, rhs_norm=rhs_norm, converged=True)
            return x
        if tol > 0.0 and residual_norm <= tol * max(1.0, rhs_norm):
            self._record_pcg_diagnostics(num_iter=0, residual=residual, rhs_norm=rhs_norm, converged=True)
            return x
        if max_iter <= 0:
            self._record_pcg_diagnostics(num_iter=0, residual=residual, rhs_norm=rhs_norm, converged=False)
            return x

        z = precondition(residual)
        z[~active] = 0.0
        direction = z.copy()
        rz_old = float(residual @ z)
        if abs(rz_old) <= 100.0 * np.finfo(np.float64).eps:
            self._record_pcg_diagnostics(num_iter=0, residual=residual, rhs_norm=rhs_norm, converged=False)
            return x

        num_iter = 0
        converged = False
        for _iter in range(int(max_iter)):
            Ad = operator(direction)
            Ad[~active] = 0.0
            denom = float(direction @ Ad)
            denom_tol = 100.0 * np.finfo(np.float64).eps * max(
                1.0,
                float(np.linalg.norm(direction)) * float(np.linalg.norm(Ad)),
            )
            if denom <= denom_tol:
                break

            step = rz_old / denom
            x += step * direction
            x[~active] = 0.0
            residual -= step * Ad
            residual[~active] = 0.0
            num_iter += 1
            if tol > 0.0 and float(np.linalg.norm(residual)) <= tol * max(1.0, rhs_norm):
                converged = True
                break

            z = precondition(residual)
            z[~active] = 0.0
            rz_new = float(residual @ z)
            if abs(rz_new) <= 100.0 * np.finfo(np.float64).eps:
                break
            direction = z + (rz_new / rz_old) * direction
            direction[~active] = 0.0
            rz_old = rz_new
        self._record_pcg_diagnostics(
            num_iter=num_iter,
            residual=residual,
            rhs_norm=rhs_norm,
            converged=converged,
        )
        return x

    def solve(
        self,
        *,
        stats,
        R_factor,
        U_factor,
        c_stat,
        b_stat,
        count,
        ridge,
        store=None,
        excluding_fold=None,
        profile_norm=None,
        empirical_cache=None,
    ):
        self._validate(stats)
        alpha = stats.r_correction_alpha
        if alpha > 0.0 and empirical_cache is None:
            X, q0 = self._training_rows(store=store, excluding_fold=excluding_fold)
            empirical_cache = self._build_empirical_cache(stats=stats, X=X, q0=q0)
        if alpha > 0.0 and len(empirical_cache["X"]) == 0:
            raise ValueError("Matrix-free empirical correction requires at least one training row.")
        if profile_norm is None:
            profile_norm = count
        profile_norm_float = float(profile_norm)
        if not np.isfinite(profile_norm_float) or profile_norm_float <= 0.0:
            raise ValueError(f"Expected a positive finite profiling normalization, got {profile_norm_float!r}.")

        R_exact_apply, U_exact_apply, U_exact_T_apply, precondition = self._exact_ops(
            stats=stats,
            R_factor=R_factor,
            U_factor=U_factor,
            ridge=ridge,
            count=profile_norm_float,
        )

        def R_alpha_apply(v):
            if alpha <= 0.0:
                return R_exact_apply(v)
            empirical = self._empirical_R_apply(stats=stats, cache=empirical_cache, v=v)
            if alpha >= 1.0:
                return empirical
            return (1.0 - alpha) * R_exact_apply(v) + alpha * empirical

        def U_alpha_apply(v):
            if alpha <= 0.0:
                return U_exact_apply(v)
            empirical = self._empirical_U_apply(stats=stats, cache=empirical_cache, v=v)
            if alpha >= 1.0:
                return empirical
            return (1.0 - alpha) * U_exact_apply(v) + alpha * empirical

        def U_alpha_T_apply(values):
            if alpha <= 0.0:
                return U_exact_T_apply(values)
            empirical = self._empirical_U_transpose_apply(stats=stats, cache=empirical_cache, values=values)
            if alpha >= 1.0:
                return empirical
            return (1.0 - alpha) * U_exact_T_apply(values) + alpha * empirical

        ridge_diag = _ridge_as_diag(ridge, stats.feature_builder.dim)

        def operator(v):
            return (
                R_alpha_apply(v)
                - U_alpha_T_apply(U_alpha_apply(v)) / profile_norm_float
                + ridge_diag * v
            )

        rhs = c_stat - U_alpha_T_apply(b_stat) / profile_norm_float
        exact_beta = stats.solver.solve(
            stats=stats,
            R_factor=R_factor,
            U_factor=U_factor,
            c_stat=c_stat,
            b_stat=b_stat,
            count=profile_norm_float,
            ridge=ridge,
            store=store,
            excluding_fold=excluding_fold,
        )
        active = self._active_mask(
            stats=stats,
            R_factor=R_factor,
            ridge=ridge,
            empirical_cache=empirical_cache,
        )
        if not np.any(active):
            return np.zeros(stats.feature_builder.dim, dtype=np.float64)
        return self._pcg(
            operator=operator,
            precondition=precondition,
            rhs=rhs,
            x0=exact_beta,
            active=active,
            max_iter=self._default_max_iter(stats),
            tol=stats.correction_tol,
        )


class _ExactSizePlayerDiagonalCorrectedSolver(_ExactCorrectedSolverBase):
    """Size-player correction with diagonal-calibrated R and materialized U.

    The base matrix keeps the exact conditional size-player structure, but
    rescales it so its diagonal is blended toward the empirical R diagonal.
    The profiled U term uses the existing exact/empirical U blend and is handled
    by Woodbury against that calibrated base.
    """

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim):
        d = int(feature_dim)
        o = int(output_dim)
        # Empirical U plus H^{-1}U^T and block-factor workspaces. The block
        # factor storage is O(n^3), which is the same order as d * o here.
        return int(8 * (3 * d * o + 6 * d + 6 * o * o))

    @classmethod
    def needs_dense_exact_tables(cls):
        return False

    @classmethod
    def stores_empirical_U(cls):
        return True

    def _validate(self, stats):
        fb = stats.feature_builder
        if stats._target_kind != "full" or not isinstance(stats.strata, _SizeStrata):
            raise NotImplementedError(
                "Diagonal size-player correction currently supports full size-stratified targets only."
            )
        if fb.surrogate_basis_kind != "size_player":
            raise ValueError('Diagonal size-player correction requires `surrogate_basis="size_player"`.')
        if fb.overlap_ratio_col is not None:
            raise NotImplementedError("Diagonal size-player correction does not support group overlap-ratio features.")
        if not isinstance(stats.solver, _ExactSizePlayerSolver):
            raise ValueError("Diagonal size-player correction requires `solver_mode='size_player'` or `'size_player_streaming'`.")
        return fb.size_player_start

    def _training_rows(self, *, store, excluding_fold):
        if store is None:
            raise ValueError("Diagonal-R empirical correction requires an observation store.")
        X, _y, q, folds = store.rows()
        if excluding_fold is not None:
            mask = folds != int(excluding_fold)
            X = X[mask]
            q = q[mask]
        return X, q

    def _global_values(self, fb, sizes):
        values = [np.ones(len(sizes), dtype=np.float64)]
        if fb.log_col is not None:
            sizes_float = sizes.astype(np.float64)
            values.append(np.log1p(sizes_float))
            values.append((sizes_float / float(fb.n)) ** 2)
        return np.column_stack(values)

    def _build_empirical_R_diag(self, *, stats, store, excluding_fold=None):
        X, q = self._training_rows(store=store, excluding_fold=excluding_fold)
        fb = stats.feature_builder
        out = np.zeros(fb.dim, dtype=np.float64)
        if len(X) == 0:
            return out

        context = stats.strata.context_from_X(X)
        ids = stats.strata.ids_from_context(context)
        if np.any(ids < 0):
            raise ValueError("Encountered invalid stratum id.")
        weights = stats.target.true_stratum_weight(stats.strata)[ids] / (q ** 2)
        sizes = context.sizes.astype(np.int64, copy=False)
        global_cols = stats.solver._global_cols(fb)
        global_values = self._global_values(fb, sizes)
        out[global_cols] = (global_values * global_values).T @ weights

        for size in range(1, fb.n + 1):
            rows = sizes == size
            if not np.any(rows):
                continue
            start = fb.size_player_start + (size - 1) * fb.n
            out[start:start + fb.n] = X[rows].T @ weights[rows]
        return out

    def _factor_symmetric(self, gram):
        sym = 0.5 * (gram + gram.T)
        try:
            return "cho", linalg.cho_factor(sym, lower=True, check_finite=False)
        except (ValueError, linalg.LinAlgError):
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("error", linalg.LinAlgWarning)
            try:
                return "lu", linalg.lu_factor(gram, check_finite=False)
            except (ValueError, linalg.LinAlgError, linalg.LinAlgWarning):
                pass

        eigvals, eigvecs = np.linalg.eigh(sym)
        scale = max(1.0, float(np.max(np.abs(eigvals))))
        tol = np.finfo(np.float64).eps * max(sym.shape) * scale
        inv_eigvals = np.zeros_like(eigvals)
        good = np.abs(eigvals) > tol
        inv_eigvals[good] = 1.0 / eigvals[good]
        return "pinv_eigh", (eigvecs, inv_eigvals)

    def _solve_factor(self, factor, rhs):
        kind, payload = factor
        rhs = np.asarray(rhs, dtype=np.float64)
        was_vector = rhs.ndim == 1
        rhs_2d = rhs[:, None] if was_vector else rhs
        if kind == "cho":
            out = linalg.cho_solve(payload, rhs_2d, check_finite=False)
        elif kind == "lu":
            out = linalg.lu_solve(payload, rhs_2d, check_finite=False)
        elif kind == "pinv_eigh":
            eigvecs, inv_eigvals = payload
            out = eigvecs @ (inv_eigvals[:, None] * (eigvecs.T @ rhs_2d))
        else:
            raise RuntimeError(f"Unexpected factor kind {kind!r}.")
        return out[:, 0] if was_vector else out

    def _calibration_scale(self, exact_diag, empirical_diag, alpha):
        target_diag = exact_diag if alpha <= 0.0 else exact_diag + alpha * (empirical_diag - exact_diag)
        target_diag = np.maximum(target_diag, 0.0)
        exact_diag = np.maximum(exact_diag, 0.0)
        scale = np.ones_like(exact_diag)
        extra_diag = np.zeros_like(exact_diag)
        tol = 100.0 * np.finfo(np.float64).eps * max(
            1.0,
            float(np.max(np.abs(exact_diag), initial=0.0)),
            float(np.max(np.abs(target_diag), initial=0.0)),
        )
        good = exact_diag > tol
        scale[good] = target_diag[good] / exact_diag[good]
        extra_diag[~good] = target_diag[~good]
        return scale, extra_diag

    def _build_base_structure(self, *, stats, R_factor, U_factor, ridge, empirical_R_diag):
        solver = stats.solver
        fb = stats.feature_builder
        n = fb.n
        block_start = self._validate(stats)
        global_cols = solver._global_cols(fb)
        global_ridge, block_ridge = solver._ridge_parts(ridge, global_cols, block_start, n, fb.dim)
        R_gg, R_g_blocks, R_contrast, R_mean_rank, _U_g, _U_delta, _U_gamma = solver._moments(
            stats,
            R_factor,
            U_factor,
            global_cols,
        )

        exact_diag = np.zeros(fb.dim, dtype=np.float64)
        exact_diag[global_cols] = np.diag(R_gg)
        for size in range(1, n + 1):
            start = block_start + (size - 1) * n
            exact_diag[start:start + n] = R_contrast[size - 1] + R_mean_rank[size - 1]
        if empirical_R_diag is None:
            empirical_R_diag = exact_diag

        scale, extra_diag = self._calibration_scale(
            exact_diag,
            np.asarray(empirical_R_diag, dtype=np.float64),
            stats.r_correction_alpha,
        )
        sqrt_scale = np.sqrt(scale)
        global_scale = sqrt_scale[global_cols]

        global_matrix = global_scale[:, None] * R_gg * global_scale[None, :]
        diag = np.diag_indices_from(global_matrix)
        global_matrix[diag] += global_ridge + extra_diag[global_cols]

        l_dim = len(global_cols)
        block_factors = []
        block_crosses = []
        block_inv_crosses = []
        schur = global_matrix.copy()
        for size in range(1, n + 1):
            idx = size - 1
            start = block_start + idx * n
            cols = slice(start, start + n)
            block_scale = scale[cols]
            block_sqrt = sqrt_scale[cols]
            block_extra = extra_diag[cols]
            block_matrix = np.diag(R_contrast[idx] * block_scale + block_ridge[idx] + block_extra)
            if R_mean_rank[idx] != 0.0:
                block_matrix += R_mean_rank[idx] * np.outer(block_sqrt, block_sqrt)
            cross = block_sqrt[:, None] * (R_g_blocks[idx][None, :] * global_scale[None, :])
            factor = self._factor_symmetric(block_matrix)
            inv_cross = self._solve_factor(factor, cross)
            schur -= cross.T @ inv_cross
            block_factors.append(factor)
            block_crosses.append(cross)
            block_inv_crosses.append(inv_cross)

        return {
            "n": n,
            "dim": fb.dim,
            "block_start": block_start,
            "global_cols": global_cols,
            "global_factor": self._factor_symmetric(schur),
            "block_factors": block_factors,
            "block_crosses": block_crosses,
            "block_inv_crosses": block_inv_crosses,
            "l_dim": l_dim,
        }

    def _apply_base_inverse(self, structure, rhs):
        n = structure["n"]
        block_start = structure["block_start"]
        rhs = np.asarray(rhs, dtype=np.float64)
        was_vector = rhs.ndim == 1
        rhs_2d = rhs[:, None] if was_vector else rhs

        k = rhs_2d.shape[1]
        global_rhs = rhs_2d[structure["global_cols"]].copy()
        block_rhs = rhs_2d[block_start:block_start + n * n].reshape(n, n, k)

        block_pre = []
        for idx in range(n):
            cur = self._solve_factor(structure["block_factors"][idx], block_rhs[idx])
            block_pre.append(cur)
            global_rhs -= structure["block_crosses"][idx].T @ cur

        global_sol = self._solve_factor(structure["global_factor"], global_rhs)
        out = np.zeros_like(rhs_2d)
        out[structure["global_cols"]] = global_sol
        block_out = np.empty((n, n, k), dtype=np.float64)
        for idx in range(n):
            block_out[idx] = block_pre[idx] - structure["block_inv_crosses"][idx] @ global_sol
        out[block_start:block_start + n * n] = block_out.reshape(n * n, k)
        return out[:, 0] if was_vector else out

    def _solve_reduced(self, reduced, rhs):
        sym = 0.5 * (reduced + reduced.T)
        try:
            factor = linalg.cho_factor(sym, lower=True, check_finite=False)
            return linalg.cho_solve(factor, rhs, check_finite=False)
        except (ValueError, linalg.LinAlgError):
            warnings.warn(
                "Diagonal size-player correction encountered a non-SPD Woodbury reduced system; "
                "falling back to a generic solve. Consider stronger ridge or smaller U correction.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _safe_solve(reduced, rhs)

    def solve(
        self,
        *,
        stats,
        R_factor,
        U_factor,
        c_stat,
        b_stat,
        count,
        ridge,
        store=None,
        excluding_fold=None,
    ):
        self._validate(stats)
        empirical_R_diag = None
        if stats.r_correction_alpha > 0.0:
            empirical_R_diag = self._build_empirical_R_diag(
                stats=stats,
                store=store,
                excluding_fold=excluding_fold,
            )

        structure = self._build_base_structure(
            stats=stats,
            R_factor=R_factor,
            U_factor=U_factor,
            ridge=ridge,
            empirical_R_diag=empirical_R_diag,
        )
        count_float = float(count)
        U_stat = stats._build_corrected_U(U_factor, excluding_fold=excluding_fold)
        rhs = c_stat - (U_stat.T @ b_stat) / count_float

        base_rhs = self._apply_base_inverse(structure, rhs)
        base_U_t = self._apply_base_inverse(structure, U_stat.T)
        reduced = count_float * np.eye(stats.target.output_dim, dtype=np.float64) - U_stat @ base_U_t
        correction_rhs = U_stat @ base_rhs
        correction = self._solve_reduced(reduced, correction_rhs)
        return base_rhs + base_U_t @ correction


class _ExactDenseCorrectedSolver(_ExactCorrectedSolverBase):
    """Dense reference solver for empirically corrected exact systems.

    This is a correctness/prototyping implementation. It materializes the
    empirical R statistic from the stored rows, uses corrected R and U
    statistics, and then solves the profiled dense system.
    """

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim):
        d = int(feature_dim)
        o = int(output_dim)
        # Worst-case workspace: R_emp plus R_exact for 0 < alpha < 1.
        # The alpha == 1 case only needs R_emp, but this class does not see alpha.
        return int(16 * d * d + 8 * o * d)

    def _build_empirical_R(self, *, stats, store, excluding_fold=None):
        if store is None:
            raise ValueError("Empirical-R correction requires an observation store.")
        X, _y, q, folds = store.rows()
        if excluding_fold is not None:
            mask = folds != int(excluding_fold)
            X = X[mask]
            q = q[mask]
        if len(X) == 0:
            return np.zeros((stats.feature_builder.dim, stats.feature_builder.dim), dtype=np.float64)

        if stats.feature_builder.surrogate_basis_kind == "size_player":
            return self._build_size_player_empirical_R(stats=stats, X=X, q=q)
        return self._build_dense_empirical_R(stats=stats, X=X, q=q)

    def _empirical_weights(self, *, stats, X, q):
        context = stats.strata.context_from_X(X)
        ids = stats.strata.ids_from_context(context)
        if np.any(ids < 0):
            raise ValueError("Encountered invalid stratum id.")
        true_weights = stats.target.true_stratum_weight(stats.strata)[ids]
        return context, true_weights / (q ** 2)

    def _build_dense_empirical_R(self, *, stats, X, q):
        context, weights = self._empirical_weights(stats=stats, X=X, q=q)
        Z = stats.feature_builder.build(X, context)
        return Z.T @ (weights[:, None] * Z)

    def _build_size_player_empirical_R(self, *, stats, X, q):
        fb = stats.feature_builder
        context, weights = self._empirical_weights(stats=stats, X=X, q=q)
        sizes = context.sizes
        d = fb.dim
        R = np.zeros((d, d), dtype=np.float64)

        global_cols = [0]
        global_values = [np.ones(len(X), dtype=np.float64)]
        if fb.log_col is not None:
            global_cols.append(fb.log_col)
            global_values.append(np.log1p(sizes.astype(np.float64)))
            global_cols.append(fb.quad_col)
            global_values.append((sizes.astype(np.float64) / float(fb.n)) ** 2)
        if fb.overlap_ratio_col is not None:
            if context.overlaps is None:
                raise ValueError("`overlaps` are required for the overlap-ratio feature.")
            overlap_ratio = np.zeros(len(X), dtype=np.float64)
            nonzero = sizes > 0
            overlap_ratio[nonzero] = context.overlaps[nonzero] / sizes[nonzero]
            global_cols.append(fb.overlap_ratio_col)
            global_values.append(overlap_ratio)

        global_cols = np.asarray(global_cols, dtype=np.int64)
        G = np.column_stack(global_values)
        R[np.ix_(global_cols, global_cols)] += G.T @ (weights[:, None] * G)

        for size in range(1, fb.n + 1):
            rows = sizes == size
            if not np.any(rows):
                continue
            block_cols = fb.size_player_start + (size - 1) * fb.n + np.arange(fb.n, dtype=np.int64)
            Xs = X[rows].astype(np.float64, copy=False)
            ws = weights[rows]
            Gs = G[rows]
            weighted_Xs = ws[:, None] * Xs
            cross = Gs.T @ weighted_Xs
            R[np.ix_(global_cols, block_cols)] += cross
            R[np.ix_(block_cols, global_cols)] += cross.T
            R[np.ix_(block_cols, block_cols)] += Xs.T @ weighted_Xs
        return R

    def solve(
        self,
        *,
        stats,
        R_factor,
        U_factor,
        c_stat,
        b_stat,
        count,
        ridge,
        store=None,
        excluding_fold=None,
    ):
        A_stat = self._build_corrected_R(
            stats=stats,
            R_factor=R_factor,
            store=store,
            excluding_fold=excluding_fold,
        )
        B_stat = stats._build_corrected_U(U_factor, excluding_fold=excluding_fold)
        return _solve_profiled_system(A_stat, B_stat, c_stat, b_stat, count, ridge)

    def _build_corrected_R(self, *, stats, R_factor, store, excluding_fold):
        alpha = stats.r_correction_alpha
        if alpha <= 0.0:
            return stats._build_exact_R(R_factor)

        R_emp = self._build_empirical_R(stats=stats, store=store, excluding_fold=excluding_fold)
        if alpha >= 1.0:
            return R_emp

        R_exact = stats._build_exact_R(R_factor)
        return R_exact + alpha * (R_emp - R_exact)


class _ExactFirstOrderInteractionSolver(_ExactConditionalSolver):
    """Structured solver for the full-vector ``surrogate_basis=1`` system."""

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim):
        d = int(feature_dim)
        o = int(output_dim)
        return int(8 * (8 * d + o * o))

    @classmethod
    def needs_dense_exact_tables(cls):
        return False

    def _validate(self, stats):
        fb = stats.feature_builder
        if stats._target_kind != "full" or not isinstance(stats.strata, _SizeStrata):
            raise NotImplementedError("The structured first-order solver currently supports full size-stratified targets only.")
        if fb.surrogate_basis_kind != "interactions" or fb.interaction_degree != 1:
            raise ValueError("The structured first-order solver requires `surrogate_basis=1`.")
        if fb.overlap_ratio_col is not None:
            raise NotImplementedError("The structured first-order solver does not support group overlap-ratio features.")
        if len(fb.interaction_blocks) != 1:
            raise ValueError("Expected exactly one first-order interaction block.")

        degree, combos, start = fb.interaction_blocks[0]
        if degree != 1 or combos.shape != (fb.n, 1):
            raise ValueError("Unexpected first-order interaction layout.")
        players = combos[:, 0]
        if not np.array_equal(players, np.arange(fb.n, dtype=np.int64)):
            raise ValueError("The structured first-order solver expects singleton columns ordered by player.")
        return start

    def _global_cols(self, fb):
        cols = [0]
        if fb.log_col is not None:
            cols.extend([fb.log_col, fb.quad_col])
        return np.asarray(cols, dtype=np.int64)

    def _ridge_parts(self, ridge, global_cols, player_cols, dim):
        ridge = np.asarray(ridge, dtype=np.float64)
        if ridge.ndim == 0:
            value = float(ridge)
            return np.full(len(global_cols), value, dtype=np.float64), value

        if ridge.shape != (dim,):
            raise ValueError(f"Expected ridge diagonal shape {(dim,)}, got {ridge.shape}.")
        player_ridge = ridge[player_cols]
        if not np.allclose(player_ridge, player_ridge[0], rtol=1e-12, atol=1e-12):
            raise NotImplementedError("The structured first-order solver requires a uniform player-feature ridge.")
        return ridge[global_cols].astype(np.float64, copy=True), float(player_ridge[0])

    def _moments(self, stats, R_factor, U_factor, global_cols):
        n = stats.feature_builder.n
        sizes = stats._strata_sizes.astype(np.int64)
        sizes_float = sizes.astype(np.float64)

        p1 = sizes_float / float(n)
        if n > 1:
            p2 = sizes_float * (sizes_float - 1.0) / float(n * (n - 1))
        else:
            p2 = np.zeros_like(p1)

        alpha_prev = np.zeros_like(sizes_float)
        has_prev = sizes > 0
        alpha_prev[has_prev] = stats.target.p[sizes[has_prev] - 1]

        alpha_cur = np.zeros_like(sizes_float)
        has_cur = sizes < n
        alpha_cur[has_cur] = stats.target.p[sizes[has_cur]]

        global_values = stats._feature_scale[:, global_cols]

        R_gg = global_values.T @ (R_factor[:, None] * global_values)
        R_gz = global_values.T @ (R_factor * p1)
        a_R = float(np.dot(R_factor, p1 - p2))
        b_R = float(np.dot(R_factor, p2))

        global_gamma = alpha_prev * p1 - alpha_cur * (1.0 - p1)
        U_g = global_values.T @ (U_factor * global_gamma)

        in_1 = alpha_prev * p1
        out_1 = alpha_prev * p2 - alpha_cur * (p1 - p2)
        a_U = float(np.dot(U_factor, in_1 - out_1))
        b_U = float(np.dot(U_factor, out_1))

        return R_gg, R_gz, a_R, b_R, U_g, a_U, b_U

    def _solve_first_order(self, *, n, G_gg, G_gz, A_player, B_player, rhs_g, rhs_z):
        if n == 0:
            return _safe_solve(G_gg, rhs_g), np.zeros(0, dtype=np.float64)

        rhs_mean = float(np.mean(rhs_z))
        rhs_ctr = rhs_z - rhs_mean

        if abs(A_player) > 100.0 * np.finfo(np.float64).eps:
            beta_ctr = rhs_ctr / A_player
        else:
            beta_ctr = np.zeros_like(rhs_ctr)

        l_dim = G_gg.shape[0]
        sqrt_n = math.sqrt(float(n))
        mean_system = np.empty((l_dim + 1, l_dim + 1), dtype=np.float64)
        mean_system[:l_dim, :l_dim] = G_gg
        mean_system[:l_dim, l_dim] = sqrt_n * G_gz
        mean_system[l_dim, :l_dim] = sqrt_n * G_gz
        mean_system[l_dim, l_dim] = A_player + float(n) * B_player

        mean_rhs = np.empty(l_dim + 1, dtype=np.float64)
        mean_rhs[:l_dim] = rhs_g
        mean_rhs[l_dim] = sqrt_n * rhs_mean

        mean_solution = _safe_solve(mean_system, mean_rhs)
        beta_global = mean_solution[:l_dim]
        beta_mean = mean_solution[l_dim] / sqrt_n
        return beta_global, beta_mean + beta_ctr

    def solve(
        self,
        *,
        stats,
        R_factor,
        U_factor,
        c_stat,
        b_stat,
        count,
        ridge,
        store=None,
        excluding_fold=None,
    ):
        singleton_start = self._validate(stats)
        fb = stats.feature_builder
        n = fb.n
        global_cols = self._global_cols(fb)
        player_cols = singleton_start + np.arange(n, dtype=np.int64)
        global_ridge, player_ridge = self._ridge_parts(ridge, global_cols, player_cols, fb.dim)

        R_gg, R_gz, a_R, b_R, U_g, a_U, b_U = self._moments(stats, R_factor, U_factor, global_cols)

        count_float = float(count)
        G_gg = R_gg - (float(n) / count_float) * np.outer(U_g, U_g)
        diag = np.diag_indices_from(G_gg)
        G_gg[diag] += global_ridge

        common_u_col_sum = a_U + float(n) * b_U
        G_gz = R_gz - (U_g * common_u_col_sum) / count_float
        A_player = a_R - (a_U * a_U) / count_float + player_ridge
        B_player = b_R - (2.0 * a_U * b_U + float(n) * b_U * b_U) / count_float

        b_sum = float(np.sum(b_stat))
        rhs_g = c_stat[global_cols] - (U_g * b_sum) / count_float
        rhs_z = c_stat[player_cols] - (a_U * b_stat + b_U * b_sum) / count_float

        beta_global, beta_players = self._solve_first_order(
            n=n,
            G_gg=G_gg,
            G_gz=G_gz,
            A_player=A_player,
            B_player=B_player,
            rhs_g=rhs_g,
            rhs_z=rhs_z,
        )

        beta = np.zeros(fb.dim, dtype=np.float64)
        beta[global_cols] = beta_global
        beta[player_cols] = beta_players
        return beta


class _ExactSizePlayerSolver(_ExactConditionalSolver):
    """Structured Woodbury solver for the full-vector size-player system."""

    materializes_base_U_t = True

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim):
        d = int(feature_dim)
        o = int(output_dim)
        # Main Woodbury workspace: U^T and H^{-1}U^T, both d x o, plus the
        # reduced o x o system and structured base-inverse temporaries.
        return int(8 * (2 * d * o + 4 * o * o + 6 * d))

    @classmethod
    def needs_dense_exact_tables(cls):
        return False

    def _validate(self, stats):
        fb = stats.feature_builder
        if stats._target_kind != "full" or not isinstance(stats.strata, _SizeStrata):
            raise NotImplementedError("The structured size-player solver currently supports full size-stratified targets only.")
        if fb.surrogate_basis_kind != "size_player":
            raise ValueError('The structured size-player solver requires `surrogate_basis="size_player"`.')
        if fb.overlap_ratio_col is not None:
            raise NotImplementedError("The structured size-player solver does not support group overlap-ratio features.")
        return fb.size_player_start

    def _global_cols(self, fb):
        cols = [0]
        if fb.log_col is not None:
            cols.extend([fb.log_col, fb.quad_col])
        return np.asarray(cols, dtype=np.int64)

    def _ridge_parts(self, ridge, global_cols, block_start, n, dim):
        ridge = np.asarray(ridge, dtype=np.float64)
        if ridge.ndim == 0:
            value = float(ridge)
            return np.full(len(global_cols), value, dtype=np.float64), np.full(n, value, dtype=np.float64)

        if ridge.shape != (dim,):
            raise ValueError(f"Expected ridge diagonal shape {(dim,)}, got {ridge.shape}.")
        block_ridge = np.empty(n, dtype=np.float64)
        for size in range(1, n + 1):
            start = block_start + (size - 1) * n
            cur = ridge[start:start + n]
            if not np.allclose(cur, cur[0], rtol=1e-12, atol=1e-12):
                raise NotImplementedError("The structured size-player solver requires uniform ridge within each size block.")
            block_ridge[size - 1] = cur[0]
        return ridge[global_cols].astype(np.float64, copy=True), block_ridge

    def _moments(self, stats, R_factor, U_factor, global_cols):
        n = stats.feature_builder.n
        sizes = stats._strata_sizes.astype(np.int64)
        sizes_float = sizes.astype(np.float64)
        p1_all = sizes_float / float(n)
        p2_all = stats._subset_prob[:, 2]

        alpha_prev = np.zeros_like(sizes_float)
        has_prev = sizes > 0
        alpha_prev[has_prev] = stats.target.p[sizes[has_prev] - 1]

        alpha_cur = np.zeros_like(sizes_float)
        has_cur = sizes < n
        alpha_cur[has_cur] = stats.target.p[sizes[has_cur]]

        global_values = stats._feature_scale[:, global_cols]
        R_gg = global_values.T @ (R_factor[:, None] * global_values)
        R_g_blocks = (global_values[1:].T * (R_factor[1:] * p1_all[1:])).T

        R_contrast = R_factor[1:] * (p1_all[1:] - p2_all[1:])
        R_mean_rank = R_factor[1:] * p2_all[1:]

        global_gamma = alpha_prev * p1_all - alpha_cur * (1.0 - p1_all)
        U_g = global_values.T @ (U_factor * global_gamma)

        in_values = alpha_prev[1:] * p1_all[1:]
        out_values = alpha_prev[1:] * p2_all[1:] - alpha_cur[1:] * (p1_all[1:] - p2_all[1:])
        U_delta = U_factor[1:] * (in_values - out_values)
        U_gamma = U_factor[1:] * out_values

        return R_gg, R_g_blocks, R_contrast, R_mean_rank, U_g, U_delta, U_gamma

    def _build_base_structure(self, *, stats, R_factor, U_factor, ridge):
        fb = stats.feature_builder
        n = fb.n
        block_start = self._validate(stats)
        global_cols = self._global_cols(fb)
        global_ridge, block_ridge = self._ridge_parts(ridge, global_cols, block_start, n, fb.dim)
        R_gg, R_g_blocks, R_contrast, R_mean_rank, U_g, U_delta, U_gamma = self._moments(
            stats,
            R_factor,
            U_factor,
            global_cols,
        )

        contrast_diag = R_contrast + block_ridge
        mean_diag = contrast_diag + float(n) * R_mean_rank
        l_dim = len(global_cols)
        sqrt_n = math.sqrt(float(n))
        symmetric = np.zeros((l_dim + n, l_dim + n), dtype=np.float64)
        symmetric[:l_dim, :l_dim] = R_gg
        diag = np.diag_indices(l_dim)
        symmetric[diag] += global_ridge
        symmetric[:l_dim, l_dim:] = sqrt_n * R_g_blocks.T
        symmetric[l_dim:, :l_dim] = sqrt_n * R_g_blocks
        symmetric[l_dim:, l_dim:] = np.diag(mean_diag)

        return {
            "n": n,
            "dim": fb.dim,
            "block_start": block_start,
            "global_cols": global_cols,
            "symmetric": symmetric,
            "contrast_diag": contrast_diag,
            "U_g": U_g,
            "U_delta": U_delta,
            "U_gamma": U_gamma,
        }

    def _minimum_norm_block_divide(self, values, denom):
        # If a size block has no design mass and no ridge, match the dense
        # least-squares convention by returning the minimum-norm zero solution.
        out = np.zeros_like(values)
        good = np.abs(denom) > 100.0 * np.finfo(np.float64).eps
        if np.any(good):
            out[good] = values[good] / denom[good, None, None]
        return out

    def _cache_symmetric_solver(self, structure):
        if "symmetric_solver_kind" in structure:
            return

        gram = structure["symmetric"]
        sym = 0.5 * (gram + gram.T)
        try:
            structure["symmetric_solver_kind"] = "cho"
            structure["symmetric_solver_factor"] = linalg.cho_factor(sym, lower=True, check_finite=False)
            return
        except (ValueError, linalg.LinAlgError):
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("error", linalg.LinAlgWarning)
            try:
                structure["symmetric_solver_kind"] = "lu"
                structure["symmetric_solver_factor"] = linalg.lu_factor(gram, check_finite=False)
                return
            except (ValueError, linalg.LinAlgError, linalg.LinAlgWarning):
                pass

        eigvals, eigvecs = np.linalg.eigh(sym)
        scale = max(1.0, float(np.max(np.abs(eigvals))))
        tol = np.finfo(np.float64).eps * max(sym.shape) * scale
        inv_eigvals = np.zeros_like(eigvals)
        good = np.abs(eigvals) > tol
        inv_eigvals[good] = 1.0 / eigvals[good]
        structure["symmetric_solver_kind"] = "pinv_eigh"
        structure["symmetric_solver_factor"] = (eigvecs, inv_eigvals)

    def _solve_symmetric_cached(self, structure, rhs):
        kind = structure.get("symmetric_solver_kind")
        if kind == "cho":
            return linalg.cho_solve(structure["symmetric_solver_factor"], rhs, check_finite=False)
        if kind == "lu":
            return linalg.lu_solve(structure["symmetric_solver_factor"], rhs, check_finite=False)
        if kind == "pinv_eigh":
            eigvecs, inv_eigvals = structure["symmetric_solver_factor"]
            return eigvecs @ (inv_eigvals[:, None] * (eigvecs.T @ rhs))
        return _safe_solve(structure["symmetric"], rhs)

    def _apply_base_inverse(self, structure, rhs):
        n = structure["n"]
        l_dim = len(structure["global_cols"])
        block_start = structure["block_start"]
        rhs = np.asarray(rhs, dtype=np.float64)
        was_vector = rhs.ndim == 1
        if was_vector:
            rhs_2d = rhs[:, None]
        else:
            rhs_2d = rhs

        k = rhs_2d.shape[1]
        global_rhs = rhs_2d[structure["global_cols"]]
        block_rhs = rhs_2d[block_start:block_start + n * n].reshape(n, n, k)
        block_sums = block_rhs.sum(axis=1)
        block_means = block_sums / float(n)
        block_contrasts = block_rhs - block_means[:, None, :]

        contrast_sol = self._minimum_norm_block_divide(block_contrasts, structure["contrast_diag"])
        symmetric_rhs = np.vstack([global_rhs, block_sums / math.sqrt(float(n))])
        symmetric_sol = self._solve_symmetric_cached(structure, symmetric_rhs)

        out = np.zeros_like(rhs_2d)
        out[structure["global_cols"]] = symmetric_sol[:l_dim]
        block_out = contrast_sol + (symmetric_sol[l_dim:] / math.sqrt(float(n)))[:, None, :]
        out[block_start:block_start + n * n] = block_out.reshape(n * n, k)
        if was_vector:
            return out[:, 0]
        return out

    def _apply_U(self, structure, beta):
        n = structure["n"]
        block_start = structure["block_start"]
        beta = np.asarray(beta, dtype=np.float64)
        was_vector = beta.ndim == 1
        if was_vector:
            beta_2d = beta[:, None]
        else:
            beta_2d = beta

        global_part = structure["U_g"] @ beta_2d[structure["global_cols"]]
        out = np.ones((n, beta_2d.shape[1]), dtype=np.float64) * global_part[None, :]
        blocks = beta_2d[block_start:block_start + n * n].reshape(n, n, beta_2d.shape[1])
        block_sums = blocks.sum(axis=1)
        out += np.sum(structure["U_gamma"][:, None, None] * block_sums[:, None, :], axis=0)
        out += np.sum(structure["U_delta"][:, None, None] * blocks, axis=0)
        if was_vector:
            return out[:, 0]
        return out

    def _apply_U_transpose(self, structure, values):
        n = structure["n"]
        block_start = structure["block_start"]
        values = np.asarray(values, dtype=np.float64)
        was_vector = values.ndim == 1
        if was_vector:
            values_2d = values[:, None]
        else:
            values_2d = values

        k = values_2d.shape[1]
        value_sums = values_2d.sum(axis=0)
        out = np.zeros((structure["dim"], k), dtype=np.float64)
        out[structure["global_cols"]] = structure["U_g"][:, None] * value_sums[None, :]

        block_values = (
            structure["U_delta"][:, None, None] * values_2d[None, :, :]
            + structure["U_gamma"][:, None, None] * value_sums[None, None, :]
        )
        out[block_start:block_start + n * n] = block_values.reshape(n * n, k)
        if was_vector:
            return out[:, 0]
        return out

    def _solve_materialized(self, structure, rhs, count_float):
        base_rhs = self._apply_base_inverse(structure, rhs)
        U_t = self._apply_U_transpose(structure, np.eye(structure["n"], dtype=np.float64))
        base_U_t = self._apply_base_inverse(structure, U_t)
        reduced = count_float * np.eye(structure["n"], dtype=np.float64) - self._apply_U(structure, base_U_t)
        correction_rhs = self._apply_U(structure, base_rhs)
        correction = _safe_solve(reduced, correction_rhs)
        return base_rhs + base_U_t @ correction

    def _solve_streaming(self, structure, rhs, count_float):
        n = structure["n"]
        self._cache_symmetric_solver(structure)
        base_rhs = self._apply_base_inverse(structure, rhs)
        reduced = count_float * np.eye(n, dtype=np.float64)
        unit = np.zeros(n, dtype=np.float64)
        for target_idx in range(n):
            unit[target_idx] = 1.0
            U_t_col = self._apply_U_transpose(structure, unit)
            unit[target_idx] = 0.0
            base_col = self._apply_base_inverse(structure, U_t_col)
            reduced[:, target_idx] -= self._apply_U(structure, base_col)

        correction_rhs = self._apply_U(structure, base_rhs)
        correction = _safe_solve(reduced, correction_rhs)
        correction_vec = self._apply_U_transpose(structure, correction)
        return base_rhs + self._apply_base_inverse(structure, correction_vec)

    def solve(
        self,
        *,
        stats,
        R_factor,
        U_factor,
        c_stat,
        b_stat,
        count,
        ridge,
        store=None,
        excluding_fold=None,
    ):
        structure = self._build_base_structure(stats=stats, R_factor=R_factor, U_factor=U_factor, ridge=ridge)
        count_float = float(count)
        U_t_b = self._apply_U_transpose(structure, b_stat)
        rhs = c_stat - U_t_b / count_float

        if self.materializes_base_U_t:
            return self._solve_materialized(structure, rhs, count_float)
        return self._solve_streaming(structure, rhs, count_float)


class _ExactSizePlayerStreamingSolver(_ExactSizePlayerSolver):
    """Lower-memory size-player Woodbury solver that streams H^{-1}U^T columns."""

    materializes_base_U_t = False

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim):
        d = int(feature_dim)
        # In this full-vector solver, output_dim is the number of players n.
        target_dim = int(output_dim)
        # Streaming avoids the d x target_dim H^{-1}U^T workspace and the
        # transient n x n x target_dim U^T tensor used by the materialized path.
        return int(8 * (8 * d + 4 * target_dim * target_dim))


class _ExactSecondOrderSolver(_ExactConditionalSolver):
    """Structured solver for the full-vector ``surrogate_basis=2`` system."""

    @classmethod
    def estimate_memory_bytes_for(cls, *, feature_dim, output_dim):
        d = int(feature_dim)
        o = int(output_dim)
        return int(8 * (10 * d + 4 * o * o))

    @classmethod
    def needs_dense_exact_tables(cls):
        return False

    def _validate(self, stats):
        fb = stats.feature_builder
        if stats._target_kind != "full" or not isinstance(stats.strata, _SizeStrata):
            raise NotImplementedError("The structured second-order solver currently supports full size-stratified targets only.")
        if fb.surrogate_basis_kind != "interactions" or fb.interaction_degree != 2:
            raise ValueError("The structured second-order solver requires `surrogate_basis=2`.")
        if fb.overlap_ratio_col is not None:
            raise NotImplementedError("The structured second-order solver does not support group overlap-ratio features.")
        if len(fb.interaction_blocks) != 2:
            raise ValueError("Expected first- and second-order interaction blocks.")

        degree_1, combos_1, singleton_start = fb.interaction_blocks[0]
        if degree_1 != 1 or combos_1.shape != (fb.n, 1):
            raise ValueError("Unexpected first-order interaction layout.")
        if not np.array_equal(combos_1[:, 0], np.arange(fb.n, dtype=np.int64)):
            raise ValueError("The structured second-order solver expects singleton columns ordered by player.")

        degree_2, combos_2, pair_start = fb.interaction_blocks[1]
        expected_pairs = math.comb(fb.n, 2)
        if degree_2 != 2 or combos_2.shape != (expected_pairs, 2):
            raise ValueError("Unexpected second-order interaction layout.")
        flat_expected = np.fromiter(
            itertools.chain.from_iterable(itertools.combinations(range(fb.n), 2)),
            dtype=np.int64,
            count=2 * expected_pairs,
        ).reshape(expected_pairs, 2)
        if not np.array_equal(combos_2, flat_expected):
            raise ValueError("The structured second-order solver expects pair columns in lexicographic order.")
        return singleton_start, pair_start, combos_2

    def _global_cols(self, fb):
        cols = [0]
        if fb.log_col is not None:
            cols.extend([fb.log_col, fb.quad_col])
        return np.asarray(cols, dtype=np.int64)

    def _ridge_parts(self, ridge, global_cols, singleton_cols, pair_cols, dim):
        ridge = np.asarray(ridge, dtype=np.float64)
        if ridge.ndim == 0:
            value = float(ridge)
            return np.full(len(global_cols), value, dtype=np.float64), value, value

        if ridge.shape != (dim,):
            raise ValueError(f"Expected ridge diagonal shape {(dim,)}, got {ridge.shape}.")
        singleton_ridge = ridge[singleton_cols]
        if not np.allclose(singleton_ridge, singleton_ridge[0], rtol=1e-12, atol=1e-12):
            raise NotImplementedError("The structured second-order solver requires a uniform singleton-feature ridge.")
        pair_ridge = ridge[pair_cols]
        if len(pair_ridge) and not np.allclose(pair_ridge, pair_ridge[0], rtol=1e-12, atol=1e-12):
            raise NotImplementedError("The structured second-order solver requires a uniform pair-feature ridge.")
        pair_value = float(pair_ridge[0]) if len(pair_ridge) else 0.0
        return ridge[global_cols].astype(np.float64, copy=True), float(singleton_ridge[0]), pair_value

    def _moments(self, stats, R_factor, U_factor, global_cols):
        n = stats.feature_builder.n
        sizes = stats._strata_sizes.astype(np.int64)
        sizes_float = sizes.astype(np.float64)

        pi_1 = sizes_float / float(n)
        pi_2 = stats._subset_prob[:, 2]
        pi_3 = stats._subset_prob[:, 3]
        pi_4 = stats._subset_prob[:, 4]

        alpha_prev = np.zeros_like(sizes_float)
        has_prev = sizes > 0
        alpha_prev[has_prev] = stats.target.p[sizes[has_prev] - 1]

        alpha_cur = np.zeros_like(sizes_float)
        has_cur = sizes < n
        alpha_cur[has_cur] = stats.target.p[sizes[has_cur]]

        global_values = stats._feature_scale[:, global_cols]
        R_gg = global_values.T @ (R_factor[:, None] * global_values)
        R_g1 = global_values.T @ (R_factor * pi_1)
        R_g2 = global_values.T @ (R_factor * pi_2)

        P_1 = float(np.dot(R_factor, pi_1))
        P_2 = float(np.dot(R_factor, pi_2))
        P_3 = float(np.dot(R_factor, pi_3))
        P_4 = float(np.dot(R_factor, pi_4))

        global_gamma = alpha_prev * pi_1 - alpha_cur * (1.0 - pi_1)
        U_g = global_values.T @ (U_factor * global_gamma)

        in_1 = alpha_prev * pi_1
        out_1 = alpha_prev * pi_2 - alpha_cur * (pi_1 - pi_2)
        delta_1 = float(np.dot(U_factor, in_1 - out_1))
        gamma_1 = float(np.dot(U_factor, out_1))

        in_2 = alpha_prev * pi_2
        out_2 = alpha_prev * pi_3 - alpha_cur * (pi_2 - pi_3)
        delta_2 = float(np.dot(U_factor, in_2 - out_2))
        gamma_2 = float(np.dot(U_factor, out_2))

        return R_gg, R_g1, R_g2, (P_1, P_2, P_3, P_4), U_g, (delta_1, gamma_1, delta_2, gamma_2)

    def _M_times_pair(self, pair_values, pairs, n):
        out = np.zeros(n, dtype=np.float64)
        np.add.at(out, pairs[:, 0], pair_values)
        np.add.at(out, pairs[:, 1], pair_values)
        return out

    def _MT_times_player(self, player_values, pairs):
        return player_values[pairs[:, 0]] + player_values[pairs[:, 1]]

    def _minimum_norm_scalar_solve(self, values, denom):
        if abs(denom) > 100.0 * np.finfo(np.float64).eps:
            return values / denom
        return np.zeros_like(values)

    def _solve_structured(
        self,
        *,
        n,
        pairs,
        G_gg,
        G_g1,
        G_g2,
        A11,
        B11,
        A12,
        B12,
        A22,
        B22,
        C22,
        rhs_g,
        rhs_1,
        rhs_2,
    ):
        num_pairs = len(pairs)
        rhs_1_mean = float(np.mean(rhs_1))
        rhs_1_ctr = rhs_1 - rhs_1_mean

        rhs_2_mean = float(np.mean(rhs_2))
        rhs_2_tilde = rhs_2 - rhs_2_mean
        rhs_2_player_amp = self._M_times_pair(rhs_2_tilde, pairs, n) / float(n - 2)
        rhs_2_pure = rhs_2_tilde - self._MT_times_player(rhs_2_player_amp, pairs)

        contrast_system = np.array(
            [
                [A11, float(n - 2) * A12],
                [A12, A22 + float(n - 4) * B22],
            ],
            dtype=np.float64,
        )
        contrast_rhs = np.vstack([rhs_1_ctr, rhs_2_player_amp])
        contrast_solution = _safe_solve(contrast_system, contrast_rhs)
        beta_1_ctr = contrast_solution[0]
        beta_2_pair_contrast = self._MT_times_player(contrast_solution[1], pairs)
        beta_2_pure = self._minimum_norm_scalar_solve(rhs_2_pure, A22 - 2.0 * B22)

        l_dim = G_gg.shape[0]
        sqrt_n = math.sqrt(float(n))
        sqrt_pairs = math.sqrt(float(num_pairs))
        symmetric_system = np.empty((l_dim + 2, l_dim + 2), dtype=np.float64)
        symmetric_system[:l_dim, :l_dim] = G_gg
        symmetric_system[:l_dim, l_dim] = sqrt_n * G_g1
        symmetric_system[l_dim, :l_dim] = sqrt_n * G_g1
        symmetric_system[:l_dim, l_dim + 1] = sqrt_pairs * G_g2
        symmetric_system[l_dim + 1, :l_dim] = sqrt_pairs * G_g2
        symmetric_system[l_dim, l_dim] = A11 + float(n) * B11
        symmetric_system[l_dim, l_dim + 1] = math.sqrt(float(n * num_pairs)) * (B12 + 2.0 * A12 / float(n))
        symmetric_system[l_dim + 1, l_dim] = symmetric_system[l_dim, l_dim + 1]
        symmetric_system[l_dim + 1, l_dim + 1] = (
            A22
            + 2.0 * float(n - 2) * B22
            + float(num_pairs) * C22
        )

        symmetric_rhs = np.empty(l_dim + 2, dtype=np.float64)
        symmetric_rhs[:l_dim] = rhs_g
        symmetric_rhs[l_dim] = sqrt_n * rhs_1_mean
        symmetric_rhs[l_dim + 1] = sqrt_pairs * rhs_2_mean
        symmetric_solution = _safe_solve(symmetric_system, symmetric_rhs)

        beta_global = symmetric_solution[:l_dim]
        beta_1_mean = symmetric_solution[l_dim] / sqrt_n
        beta_2_mean = symmetric_solution[l_dim + 1] / sqrt_pairs
        beta_1 = beta_1_mean + beta_1_ctr
        beta_2 = beta_2_mean + beta_2_pair_contrast + beta_2_pure
        return beta_global, beta_1, beta_2

    def solve(
        self,
        *,
        stats,
        R_factor,
        U_factor,
        c_stat,
        b_stat,
        count,
        ridge,
        store=None,
        excluding_fold=None,
    ):
        singleton_start, pair_start, pairs = self._validate(stats)
        fb = stats.feature_builder
        n = fb.n
        if n < 4:
            return _ExactDenseConditionalSolver().solve(
                stats=stats,
                R_factor=R_factor,
                U_factor=U_factor,
                c_stat=c_stat,
                b_stat=b_stat,
                count=count,
                ridge=ridge,
                store=store,
                excluding_fold=excluding_fold,
            )

        num_pairs = len(pairs)
        global_cols = self._global_cols(fb)
        singleton_cols = singleton_start + np.arange(n, dtype=np.int64)
        pair_cols = pair_start + np.arange(num_pairs, dtype=np.int64)
        global_ridge, singleton_ridge, pair_ridge = self._ridge_parts(
            ridge,
            global_cols,
            singleton_cols,
            pair_cols,
            fb.dim,
        )

        R_gg, R_g1, R_g2, P, U_g, U_scalars = self._moments(stats, R_factor, U_factor, global_cols)
        P_1, P_2, P_3, P_4 = P
        delta_1, gamma_1, delta_2, gamma_2 = U_scalars

        count_float = float(count)
        G_gg = R_gg - (float(n) / count_float) * np.outer(U_g, U_g)
        diag = np.diag_indices_from(G_gg)
        G_gg[diag] += global_ridge

        G_g1 = R_g1 - (U_g * (delta_1 + float(n) * gamma_1)) / count_float
        G_g2 = R_g2 - (U_g * (2.0 * delta_2 + float(n) * gamma_2)) / count_float

        A11 = P_1 - P_2 - (delta_1 * delta_1) / count_float + singleton_ridge
        B11 = P_2 - (2.0 * delta_1 * gamma_1 + float(n) * gamma_1 * gamma_1) / count_float
        A12 = P_2 - P_3 - (delta_1 * delta_2) / count_float
        B12 = P_3 - (
            delta_1 * gamma_2
            + 2.0 * gamma_1 * delta_2
            + float(n) * gamma_1 * gamma_2
        ) / count_float
        A22 = P_2 - P_4 - (2.0 * delta_2 * delta_2) / count_float + pair_ridge
        B22 = P_3 - P_4 - (delta_2 * delta_2) / count_float
        C22 = P_4 - (4.0 * delta_2 * gamma_2 + float(n) * gamma_2 * gamma_2) / count_float

        b_sum = float(np.sum(b_stat))
        rhs_g = c_stat[global_cols] - (U_g * b_sum) / count_float
        rhs_1 = c_stat[singleton_cols] - (delta_1 * b_stat + gamma_1 * b_sum) / count_float
        pair_b_sum = b_stat[pairs[:, 0]] + b_stat[pairs[:, 1]]
        rhs_2 = c_stat[pair_cols] - (delta_2 * pair_b_sum + gamma_2 * b_sum) / count_float

        beta_global, beta_1, beta_2 = self._solve_structured(
            n=n,
            pairs=pairs,
            G_gg=G_gg,
            G_g1=G_g1,
            G_g2=G_g2,
            A11=A11,
            B11=B11,
            A12=A12,
            B12=B12,
            A22=A22,
            B22=B22,
            C22=C22,
            rhs_g=rhs_g,
            rhs_1=rhs_1,
            rhs_2=rhs_2,
        )

        beta = np.zeros(fb.dim, dtype=np.float64)
        beta[global_cols] = beta_global
        beta[singleton_cols] = beta_1
        beta[pair_cols] = beta_2
        return beta


def _exact_conditional_solver_class(mode="dense"):
    key = str(mode).strip().lower().replace("-", "_")
    if key in {"auto", "dense", "dense_exact"}:
        return _ExactDenseConditionalSolver
    if key in {"fo", "first_order", "first_order_interaction", "degree1"}:
        return _ExactFirstOrderInteractionSolver
    if key in {
        "size_player",
        "woodbury",
        "woodbury_cg",
        "size_player_woodbury",
        "size_player_woodbury_cg",
        "size_player_materialized",
        "size_player_woodbury_materialized",
    }:
        return _ExactSizePlayerSolver
    if key in {"size_player_streaming", "size_player_woodbury_streaming", "streaming_size_player"}:
        return _ExactSizePlayerStreamingSolver
    if key in {"second_order", "degree2"}:
        return _ExactSecondOrderSolver
    raise ValueError(f"Unknown exact conditional solver mode {mode!r}.")


def _make_exact_conditional_solver(mode="dense"):
    return _exact_conditional_solver_class(mode)()


def _exact_corrected_solver_class(mode="dense"):
    key = _normalize_correction_solver_mode(mode)
    if key == "dense":
        return _ExactDenseCorrectedSolver
    if key == "matrix_free":
        return _ExactMatrixFreeCorrectedSolver
    if key == "size_player_diagonal":
        return _ExactSizePlayerDiagonalCorrectedSolver
    raise ValueError(f"Unknown empirically corrected solver mode {mode!r}.")


def _make_exact_corrected_solver(mode="dense"):
    return _exact_corrected_solver_class(mode)()


class _ExactConditionalStats(_ExactStatsBase):
    """Conditional exact response-free design moments.

    This backend keeps the utility-dependent RHS terms empirical, but replaces
    the surrogate Gram terms with their conditional expectation given the
    observed stratum id. Its current solver is the dense reference
    implementation; empirical correction can independently blend the R block
    and U block toward their empirical counterparts. Structured Woodbury/CG
    solvers are separated as future drop-in replacements.
    """

    def __init__(
        self,
        *,
        target,
        strata,
        feature_builder,
        ridge_lambda,
        ridge_schedule,
        num_folds,
        ridge_scaling="scalar",
        solver_mode="dense",
        r_correction_alpha=0.0,
        u_correction_alpha=0.0,
        correction_solver_mode="dense",
        correction_max_iter=None,
        correction_tol=1e-8,
    ):
        self.target = target
        self.strata = strata
        self.feature_builder = feature_builder
        self.ridge_lambda = float(ridge_lambda)
        self.ridge_schedule = str(ridge_schedule)
        self.ridge_scaling = _normalize_ridge_scaling(ridge_scaling)
        self.num_folds = int(num_folds)
        self.r_correction_alpha = float(r_correction_alpha)
        if not np.isfinite(self.r_correction_alpha) or not (0.0 <= self.r_correction_alpha <= 1.0):
            raise ValueError(f"`r_correction_alpha` must lie in [0, 1], got {r_correction_alpha!r}.")
        self.u_correction_alpha = float(u_correction_alpha)
        if not np.isfinite(self.u_correction_alpha) or not (0.0 <= self.u_correction_alpha <= 1.0):
            raise ValueError(f"`u_correction_alpha` must lie in [0, 1], got {u_correction_alpha!r}.")
        if correction_max_iter is None:
            self.correction_max_iter = None
        else:
            if not isinstance(correction_max_iter, (int, np.integer)):
                raise ValueError(f"`correction_max_iter` must be an integer or None, got {correction_max_iter!r}.")
            self.correction_max_iter = int(correction_max_iter)
            if self.correction_max_iter < 0:
                raise ValueError(f"`correction_max_iter` must be >= 0, got {correction_max_iter!r}.")
        self.correction_tol = float(correction_tol)
        if not np.isfinite(self.correction_tol) or self.correction_tol < 0.0:
            raise ValueError(f"`correction_tol` must be finite and >= 0, got {correction_tol!r}.")

        if not isinstance(strata, (_SizeStrata, _GroupCellStrata)):
            raise NotImplementedError("Exact conditional stats require size or group-cell strata.")

        if target.output_dim == strata.n and isinstance(strata, _SizeStrata):
            self._target_kind = "full"
            self._raw_gamma_by_stratum = None
        elif target.output_dim == 1 and hasattr(target, "cell_rho"):
            self._target_kind = "group_scalar"
            self._raw_gamma_by_stratum = np.asarray(target.cell_rho, dtype=np.float64)
        else:
            raise NotImplementedError(
                "Exact conditional stats currently support full size-stratified "
                "targets and scalar group-cell targets."
            )
        if self.ridge_scaling == "size_trace":
            if self._target_kind != "full" or not isinstance(strata, _SizeStrata):
                raise ValueError('`surrogate_ridge_scaling="size_trace"` requires full size-stratified EaseSHAP.')
            if feature_builder.surrogate_basis_kind != "size_player":
                raise ValueError('`surrogate_ridge_scaling="size_trace"` requires `surrogate_basis="size_player"`.')

        self.solver_mode = solver_mode
        self.solver = _make_exact_conditional_solver(solver_mode)
        self.correction_solver_mode = _normalize_correction_solver_mode(correction_solver_mode)
        self.correction_solver = None
        if self._uses_correction():
            if self.correction_solver_mode == "dense" and not self.solver.__class__.needs_dense_exact_tables():
                raise ValueError(
                    "Structured exact conditional solver modes cannot be combined with empirical "
                    "R/U correction yet; use `solver_mode='dense'` or set correction alphas to 0."
                )
            if self.correction_solver_mode == "matrix_free" and self.solver.__class__.needs_dense_exact_tables():
                raise ValueError("Matrix-free empirical correction requires a structured exact solver mode.")
            self.correction_solver = _make_exact_corrected_solver(self.correction_solver_mode)
            if self.correction_solver_mode == "size_player_diagonal" and not isinstance(self.solver, _ExactSizePlayerSolver):
                raise ValueError(
                    "Diagonal size-player correction requires `solver_mode='size_player'` "
                    "or `solver_mode='size_player_streaming'`."
                )

        self.num_strata = len(strata.keys)
        self._strata_sizes = np.array([key if isinstance(key, int) else key[0] for key in strata.keys], dtype=np.int64)
        self._strata_overlaps = np.array([0 if isinstance(key, int) else key[1] for key in strata.keys], dtype=np.int64)

        d = feature_builder.dim
        o = target.output_dim
        self.total_count = 0
        self.R_factor_total = np.zeros(self.num_strata, dtype=np.float64)
        self.U_factor_total = np.zeros(self.num_strata, dtype=np.float64)
        self.c_total = np.zeros(d, dtype=np.float64)
        self.b_total = np.zeros(o, dtype=np.float64)
        store_empirical_U = (
            self.u_correction_alpha > 0.0
            and self.correction_solver is not None
            and self.correction_solver.__class__.stores_empirical_U()
        )
        self.U_emp_total = np.zeros((o, d), dtype=np.float64) if store_empirical_U else None

        self.fold_counts = np.zeros(self.num_folds, dtype=np.int64)
        self.R_factor_fold = np.zeros((self.num_folds, self.num_strata), dtype=np.float64)
        self.U_factor_fold = np.zeros((self.num_folds, self.num_strata), dtype=np.float64)
        self.c_fold = np.zeros((self.num_folds, d), dtype=np.float64)
        self.b_fold = np.zeros((self.num_folds, o), dtype=np.float64)
        self.U_emp_fold = (
            np.zeros((self.num_folds, o, d), dtype=np.float64)
            if self.U_emp_total is not None
            else None
        )

        self._req_masks, self._feature_active, self._feature_scale = self._build_feature_description()
        self._req_sizes = self._req_masks.sum(axis=1).astype(np.int64)
        self._subset_prob = self._build_subset_prob_table()
        self._req_group_counts = None
        self._union_sizes = None
        self._union_group_counts = None
        if isinstance(self.strata, _GroupCellStrata):
            self._ensure_req_group_counts()
        if self._needs_dense_exact_tables():
            self._precompute_union_tables()

    @classmethod
    def estimate_memory_bytes_for(
        cls,
        *,
        feature_dim,
        output_dim,
        num_folds,
        num_strata,
        has_group_counts=False,
        solver_mode="dense",
        r_correction_alpha=0.0,
        u_correction_alpha=0.0,
        correction_solver_mode="dense",
    ):
        d = int(feature_dim)
        o = int(output_dim)
        k = int(num_folds)
        s = int(num_strata)
        union_tables = 0
        if cls._needs_dense_exact_tables_for(
            solver_mode=solver_mode,
            r_correction_alpha=r_correction_alpha,
            u_correction_alpha=u_correction_alpha,
            correction_solver_mode=correction_solver_mode,
        ):
            union_tables = 2 * d * d
            if has_group_counts:
                union_tables += 2 * d * d
        if float(r_correction_alpha) > 0.0 or float(u_correction_alpha) > 0.0:
            corrected_solver_cls = _exact_corrected_solver_class(correction_solver_mode)
            solve_workspace = corrected_solver_cls.estimate_memory_bytes_for(
                feature_dim=d,
                output_dim=o,
            )
        else:
            corrected_solver_cls = None
            solve_workspace = _exact_conditional_solver_class(solver_mode).estimate_memory_bytes_for(
                feature_dim=d,
                output_dim=o,
            )
        empirical_u_stats = (
            8 * o * d * (k + 1)
            if float(u_correction_alpha) > 0.0
            and corrected_solver_cls is not None
            and corrected_solver_cls.stores_empirical_U()
            else 0
        )
        response_stats = 8 * (d + o + 2 * s) * (k + 1) + 8 * k
        return int(union_tables + solve_workspace + response_stats + empirical_u_stats)

    @classmethod
    def _needs_dense_exact_tables_for(
        cls,
        *,
        solver_mode,
        r_correction_alpha,
        u_correction_alpha,
        correction_solver_mode,
    ):
        if float(r_correction_alpha) > 0.0 or float(u_correction_alpha) > 0.0:
            return _exact_corrected_solver_class(correction_solver_mode).needs_dense_exact_tables()
        return _exact_conditional_solver_class(solver_mode).needs_dense_exact_tables()

    def estimate_memory_bytes(self):
        return self.estimate_memory_bytes_for(
            feature_dim=self.feature_builder.dim,
            output_dim=self.target.output_dim,
            num_folds=self.num_folds,
            num_strata=self.num_strata,
            has_group_counts=isinstance(self.strata, _GroupCellStrata),
            solver_mode=self.solver_mode,
            r_correction_alpha=self.r_correction_alpha,
            u_correction_alpha=self.u_correction_alpha,
            correction_solver_mode=self.correction_solver_mode,
        )

    def _effective_ridge_lambda(self, count):
        if self.ridge_schedule == "fixed":
            return self.ridge_lambda
        return float(count) * self.ridge_lambda

    def _ridge_penalty(self, R_factor, count):
        ridge = self._effective_ridge_lambda(count)
        if self.ridge_scaling == "scalar":
            return ridge
        if self.ridge_scaling != "size_trace":
            raise RuntimeError(f"Unexpected ridge scaling {self.ridge_scaling!r}.")

        fb = self.feature_builder
        # Global columns deliberately keep the scalar ridge schedule; only the
        # size-player blocks use the trace-scaled diagonal.
        diag = np.full(fb.dim, ridge, dtype=np.float64)
        n = fb.n
        scale = ridge / float(count)
        # _SizeStrata stores strata keys as 0..n, so R_factor[size] is valid here.
        for size in range(1, n + 1):
            block_start = fb.size_player_start + (size - 1) * n
            diag[block_start:block_start + n] = scale * R_factor[size] * float(size) / float(n)
        return diag

    def _uses_correction(self):
        return self.r_correction_alpha > 0.0 or self.u_correction_alpha > 0.0

    def _needs_dense_exact_tables(self):
        return self._active_solver().__class__.needs_dense_exact_tables()

    def _active_solver(self):
        if not self._uses_correction():
            return self.solver
        if self.correction_solver is None:
            self.correction_solver = _make_exact_corrected_solver(self.correction_solver_mode)
        return self.correction_solver

    def _empirical_U(self, excluding_fold=None):
        if self.U_emp_total is None:
            raise ValueError("Empirical-U correction requires empirical U statistics.")
        if excluding_fold is None:
            return self.U_emp_total
        return self.U_emp_total - self.U_emp_fold[int(excluding_fold)]

    def _build_corrected_U(self, U_factor, *, excluding_fold=None):
        alpha = self.u_correction_alpha
        U_exact = self._build_exact_U(U_factor)
        if alpha <= 0.0:
            return U_exact

        U_emp = self._empirical_U(excluding_fold=excluding_fold)
        if alpha >= 1.0:
            return U_emp
        return U_exact + alpha * (U_emp - U_exact)

    def _build_feature_description(self):
        fb = self.feature_builder
        d = fb.dim
        n = fb.n
        req_masks = np.zeros((d, n), dtype=bool)
        active = np.zeros((self.num_strata, d), dtype=bool)
        scale = np.zeros((self.num_strata, d), dtype=np.float64)

        def set_global(col, values):
            active[:, col] = True
            scale[:, col] = values

        set_global(0, np.ones(self.num_strata, dtype=np.float64))
        sizes_float = self._strata_sizes.astype(np.float64)
        if fb.log_col is not None:
            set_global(fb.log_col, np.log1p(sizes_float))
            set_global(fb.quad_col, (sizes_float / float(n)) ** 2)
        if fb.overlap_ratio_col is not None:
            overlap_ratio = np.zeros(self.num_strata, dtype=np.float64)
            nonzero = self._strata_sizes > 0
            overlap_ratio[nonzero] = self._strata_overlaps[nonzero] / self._strata_sizes[nonzero]
            set_global(fb.overlap_ratio_col, overlap_ratio)

        if fb.surrogate_basis_kind == "interactions":
            for degree, combos, start in fb.interaction_blocks:
                width = combos.shape[0]
                cols = np.arange(start, start + width, dtype=np.int64)
                active[:, cols] = True
                scale[:, cols] = 1.0
                for pos in range(degree):
                    req_masks[cols, combos[:, pos]] = True
        elif fb.surrogate_basis_kind == "size_player":
            for size in range(1, n + 1):
                cols = fb.size_player_start + (size - 1) * n + np.arange(n, dtype=np.int64)
                req_masks[cols, np.arange(n, dtype=np.int64)] = True
                stratum_mask = self._strata_sizes == size
                active[np.ix_(stratum_mask, cols)] = True
                scale[np.ix_(stratum_mask, cols)] = 1.0
        else:
            raise RuntimeError(f"Unexpected surrogate basis kind {fb.surrogate_basis_kind!r}.")

        return req_masks, active, scale

    def _build_subset_prob_table(self):
        n = self.feature_builder.n
        if isinstance(self.strata, _SizeStrata):
            table = np.zeros((self.num_strata, n + 2), dtype=np.float64)
            for idx, s in enumerate(self._strata_sizes):
                denom = _safe_comb(n, int(s))
                if denom <= 0.0:
                    continue
                for req_size in range(n + 2):
                    table[idx, req_size] = _safe_comb(n - req_size, int(s) - req_size) / denom
            return table

        group_size = self.strata.group_size
        outside_size = n - group_size
        table = np.zeros((self.num_strata, n + 1, group_size + 1), dtype=np.float64)
        for idx, (s, r) in enumerate(self.strata.keys):
            denom = _safe_comb(group_size, r) * _safe_comb(outside_size, s - r)
            if denom <= 0.0:
                continue
            for req_size in range(n + 1):
                for req_group in range(min(group_size, req_size) + 1):
                    req_outside = req_size - req_group
                    table[idx, req_size, req_group] = (
                        _safe_comb(group_size - req_group, r - req_group)
                        * _safe_comb(outside_size - req_outside, s - r - req_outside)
                        / denom
                    )
        return table

    def _ensure_req_group_counts(self):
        if not isinstance(self.strata, _GroupCellStrata):
            return
        if self._req_group_counts is None:
            group_req_i16 = self._req_masks[:, self.strata.group].astype(np.int16)
            self._req_group_counts = group_req_i16.sum(axis=1).astype(np.int64)

    def _precompute_union_tables(self):
        if self._union_sizes is not None:
            if not isinstance(self.strata, _GroupCellStrata) or self._union_group_counts is not None:
                return

        req_i16 = self._req_masks.astype(np.int16)
        intersections = req_i16 @ req_i16.T
        req_sizes = self._req_sizes.astype(np.int16)
        self._union_sizes = req_sizes[:, None] + req_sizes[None, :] - intersections

        if isinstance(self.strata, _GroupCellStrata):
            self._ensure_req_group_counts()
            group_req_i16 = self._req_masks[:, self.strata.group].astype(np.int16)
            group_intersections = group_req_i16 @ group_req_i16.T
            req_group_i16 = self._req_group_counts.astype(np.int16)
            self._union_group_counts = req_group_i16[:, None] + req_group_i16[None, :] - group_intersections

    def _ensure_dense_union_tables(self):
        self._precompute_union_tables()

    def _feature_mean_prob(self, stratum_idx, cols):
        req_sizes = self._req_sizes[cols]
        if isinstance(self.strata, _SizeStrata):
            return self._subset_prob[stratum_idx, req_sizes]
        self._ensure_req_group_counts()
        req_groups = self._req_group_counts[cols]
        return self._subset_prob[stratum_idx, req_sizes, req_groups]

    def _build_exact_R(self, R_factor):
        self._ensure_dense_union_tables()
        d = self.feature_builder.dim
        R = np.zeros((d, d), dtype=np.float64)
        for stratum_idx, factor in enumerate(R_factor):
            if factor == 0.0:
                continue
            active_scale = self._feature_scale[stratum_idx] * self._feature_active[stratum_idx]
            cols = np.flatnonzero(active_scale)
            if len(cols) == 0:
                continue
            row_cols = np.ix_(cols, cols)
            union_sizes = self._union_sizes[row_cols]
            if isinstance(self.strata, _SizeStrata):
                probs = self._subset_prob[stratum_idx, union_sizes]
            else:
                union_groups = self._union_group_counts[row_cols]
                probs = self._subset_prob[stratum_idx, union_sizes, union_groups]
            scale_outer = active_scale[cols][:, None] * active_scale[cols][None, :]
            R[row_cols] += factor * scale_outer * probs
        return R

    def _build_exact_U(self, U_factor):
        d = self.feature_builder.dim
        o = self.target.output_dim
        U = np.zeros((o, d), dtype=np.float64)

        if self._target_kind == "group_scalar":
            for stratum_idx, factor in enumerate(U_factor):
                raw_gamma = self._raw_gamma_by_stratum[stratum_idx]
                if factor == 0.0 or raw_gamma == 0.0:
                    continue
                active_scale = self._feature_scale[stratum_idx] * self._feature_active[stratum_idx]
                cols = np.flatnonzero(active_scale)
                if len(cols) == 0:
                    continue
                U[0, cols] += factor * raw_gamma * active_scale[cols] * self._feature_mean_prob(stratum_idx, cols)
            return U

        p = self.target.p
        req_masks_float = self._req_masks.astype(np.float64)
        for stratum_idx, factor in enumerate(U_factor):
            if factor == 0.0:
                continue
            s = int(self._strata_sizes[stratum_idx])
            alpha_prev = p[s - 1] if s > 0 else 0.0
            alpha_cur = p[s] if s < len(p) else 0.0
            active_scale = self._feature_scale[stratum_idx] * self._feature_active[stratum_idx]
            cols = np.flatnonzero(active_scale)
            if len(cols) == 0:
                continue

            req_sizes = self._req_sizes[cols]
            pi_req = self._subset_prob[stratum_idx, req_sizes]
            pi_plus = self._subset_prob[stratum_idx, req_sizes + 1]
            out_value = alpha_prev * pi_plus - alpha_cur * (pi_req - pi_plus)
            in_value = alpha_prev * pi_req
            block = out_value[None, :] + req_masks_float[cols].T * (in_value - out_value)[None, :]
            U[:, cols] += factor * active_scale[cols][None, :] * block
        return U

    def _design_counts(self, excluding_fold=None):
        if excluding_fold is None:
            return self.R_factor_total.copy(), self.U_factor_total.copy(), self.total_count
        fold = int(excluding_fold)
        return (
            self.R_factor_total - self.R_factor_fold[fold],
            self.U_factor_total - self.U_factor_fold[fold],
            self.total_count - int(self.fold_counts[fold]),
        )

    def append(self, X, y, q, folds):
        context = self.strata.context_from_X(X)
        Z = self.feature_builder.build(X, context)
        gamma = self.target.raw_gamma(X, context) / q[:, None]
        ids = self.strata.ids_from_context(context)
        if np.any(ids < 0):
            raise ValueError("Encountered invalid stratum id.")
        true_weights = self.target.true_stratum_weight(self.strata)[ids]
        R_weights = true_weights / (q ** 2)
        U_weights = 1.0 / q

        self.c_total += Z.T @ (R_weights * y)
        self.b_total += gamma.T @ y
        np.add.at(self.R_factor_total, ids, R_weights)
        np.add.at(self.U_factor_total, ids, U_weights)
        if self.U_emp_total is not None:
            self.U_emp_total += gamma.T @ Z
        self.total_count += len(y)

        for fold in np.unique(folds):
            mask = folds == fold
            if not np.any(mask):
                continue
            ids_k = ids[mask]
            Zk = Z[mask]
            yk = y[mask]
            Rk = R_weights[mask]
            gk = gamma[mask]
            self.c_fold[fold] += Zk.T @ (Rk * yk)
            self.b_fold[fold] += gk.T @ yk
            np.add.at(self.R_factor_fold[fold], ids_k, Rk)
            np.add.at(self.U_factor_fold[fold], ids_k, U_weights[mask])
            if self.U_emp_fold is not None:
                self.U_emp_fold[fold] += gk.T @ Zk
            self.fold_counts[fold] += int(mask.sum())

    def _fit_from_internal_stats(self, R_factor, U_factor, c_stat, b_stat, count, *, store=None, excluding_fold=None):
        if count <= 0:
            beta = np.zeros(self.feature_builder.dim, dtype=np.float64)
            return _FittedSurrogate(beta=beta, phi=self.target.phi_from_beta(beta))

        beta = self._active_solver().solve(
            stats=self,
            R_factor=R_factor,
            U_factor=U_factor,
            c_stat=c_stat,
            b_stat=b_stat,
            count=count,
            ridge=self._ridge_penalty(R_factor, count),
            store=store,
            excluding_fold=excluding_fold,
        )
        return _FittedSurrogate(beta=beta, phi=self.target.phi_from_beta(beta))

    def fit_all(self, store=None):
        return self._fit_from_internal_stats(
            self.R_factor_total,
            self.U_factor_total,
            self.c_total,
            self.b_total,
            self.total_count,
            store=store,
        )

    def fit_excluding_fold(self, fold, store=None):
        fold = int(fold)
        train_count = self.total_count - int(self.fold_counts[fold])
        return self._fit_from_internal_stats(
            self.R_factor_total - self.R_factor_fold[fold],
            self.U_factor_total - self.U_factor_fold[fold],
            self.c_total - self.c_fold[fold],
            self.b_total - self.b_fold[fold],
            train_count,
            store=store,
            excluding_fold=fold,
        )

    def fit_candidate(self, X, y, q0, q_candidate):
        """Apply the empirical candidate system with structured matrix-free algebra."""
        solver = self._active_solver()
        if not isinstance(solver, _ExactMatrixFreeCorrectedSolver):
            raise NotImplementedError(
                "Iterative candidate-law refitting for exact-conditional statistics requires "
                "the matrix-free empirical correction solver."
            )
        if abs(self.r_correction_alpha - 1.0) > 1e-12 or abs(self.u_correction_alpha - 1.0) > 1e-12:
            raise NotImplementedError(
                "Iterative candidate-law refitting currently requires full empirical R/U correction "
                "(`r_correction_alpha = u_correction_alpha = 1`)."
            )

        X = np.asarray(X, dtype=bool)
        y = np.asarray(y, dtype=np.float64)
        q0 = np.asarray(q0, dtype=np.float64)
        q_candidate = np.asarray(q_candidate, dtype=np.float64)
        if y.shape != (len(X),) or q0.shape != (len(X),) or q_candidate.shape != (len(X),):
            raise ValueError("Candidate-law pilot arrays must have the same row count.")
        if len(y) == 0:
            beta = np.zeros(self.feature_builder.dim, dtype=np.float64)
            return _FittedSurrogate(beta=beta, phi=self.target.phi_from_beta(beta))

        cache = solver._build_empirical_cache(
            stats=self,
            X=X,
            q0=q0,
            q_candidate=q_candidate,
        )
        context = self.strata.context_from_X(X)
        ids = self.strata.ids_from_context(context)
        if np.any(ids < 0):
            raise ValueError("Encountered invalid stratum id.")

        R_factor = np.zeros(self.num_strata, dtype=np.float64)
        U_factor = np.zeros(self.num_strata, dtype=np.float64)
        np.add.at(R_factor, ids, cache["R_weights"])
        np.add.at(U_factor, ids, cache["U_weights"])
        c_stat = solver._feature_transpose_apply(self, cache, cache["R_weights"] * y)
        b_stat = solver._raw_gamma_transpose_apply(
            cache=cache,
            row_values=cache["U_weights"] * y,
        )
        profile_norm = float(np.sum(q_candidate / q0))
        sample_count = len(y)
        beta = solver.solve(
            stats=self,
            R_factor=R_factor,
            U_factor=U_factor,
            c_stat=c_stat,
            b_stat=b_stat,
            count=sample_count,
            ridge=self._ridge_penalty(R_factor, sample_count),
            profile_norm=profile_norm,
            empirical_cache=cache,
        )
        return _FittedSurrogate(beta=beta, phi=self.target.phi_from_beta(beta))

    def fold_count(self, fold):
        return int(self.fold_counts[fold])


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class _EaseEngine:
    def __init__(
        self,
        *,
        owner,
        target,
        strata,
        feature_builder,
        law,
        store,
        backend,
        trajectory,
        boundary_X,
        game_func,
        game_args,
        num_player,
        num_sample,
        batch_size,
        nue_per_proc_run,
        interval_track,
        boundary_eval_count,
        boundary_checkpoints_missing,
        boundary_first_valid_eval,
        pilot_num_sample,
        pilot_design_updates,
        stage2_floor,
        stage2_min_count,
        num_folds,
        readout_mode,
        estimator_seed,
    ):
        self.owner = owner
        self.target = target
        self.strata = strata
        self.feature_builder = feature_builder
        self.law = law
        self.store = store
        self.backend = backend
        self.trajectory = trajectory
        self.boundary_X = np.asarray(boundary_X, dtype=bool)
        self.game_func = game_func
        self.game_args = game_args
        self.num_player = int(num_player)
        self.num_sample = int(num_sample)
        self.batch_size = int(batch_size)
        self.nue_per_proc_run = int(nue_per_proc_run)
        self.interval_track = int(interval_track)
        self.boundary_eval_count = int(boundary_eval_count)
        self.boundary_checkpoints_missing = bool(boundary_checkpoints_missing)
        self.boundary_first_valid_eval = int(boundary_first_valid_eval)
        self.pilot_num_sample = int(pilot_num_sample)
        self.pilot_design_updates = int(pilot_design_updates)
        if self.pilot_design_updates < 1:
            raise ValueError("`pilot_design_updates` must be >= 1.")
        self.stage2_floor = float(stage2_floor)
        self.stage2_min_count = int(stage2_min_count)
        self.num_folds = int(num_folds)
        self.readout_mode = readout_mode
        self.rng = np.random.Generator(np.random.PCG64(estimator_seed))
        self.fold_rng = np.random.Generator(np.random.PCG64(estimator_seed + 9173))
        self.readout_chunk_rows = 5_000

        self.boundary_values = _evaluate_values(game_func, game_args, self.boundary_X)
        self.boundary_exact = self.target.boundary_exact(self.boundary_values)
        self.current_estimate = self.boundary_exact.copy()
        self._sync_owner_current_estimate()

        self.q_init = self.law.normalize_density(self.target.initial_design_factor(self.strata))
        self.q_stage2 = self.q_init.copy()
        self.pilot_design_q_history = [self.q_init.copy()]
        self.pilot_design_beta_history = []
        self.pilot_finalized = self.pilot_num_sample == 0
        self._record_traj_up_to_current_budget()

    @property
    def num_batches_hint(self):
        return self._count_phase_batches(self.pilot_num_sample) + self._count_phase_batches(self.num_sample - self.pilot_num_sample)

    def _count_phase_batches(self, num_rows):
        if num_rows <= 0:
            return 0
        return -(-num_rows // self.batch_size)

    def _sync_owner_current_estimate(self):
        if self.target.output_dim == 1:
            self.owner._current_estimate = float(self.current_estimate[0])
        else:
            self.owner._current_estimate = self.current_estimate.copy()

    def _write_checkpoint(self, pos, value):
        checkpoint_eval = (int(pos) + 1) * self.interval_track
        if self.boundary_checkpoints_missing and checkpoint_eval < self.boundary_first_valid_eval:
            self.trajectory.write_nan(pos)
        else:
            self.trajectory.write(pos, value)

    def _record_traj_up_to_current_budget(self):
        total_eval = self.boundary_eval_count + self.store.num_obs
        while (
            self.owner.pos_traj < len(self.owner.values_traj)
            and total_eval >= (self.owner.pos_traj + 1) * self.interval_track
        ):
            self._write_checkpoint(self.owner.pos_traj, self.current_estimate)
            self.owner.pos_traj += 1

    def sampling(self):
        if self.num_sample <= 0:
            return

        pilot_remaining = self.pilot_num_sample
        while pilot_remaining > 0:
            cur = self.law.batch_rows(self.batch_size, pilot_remaining)
            yield self.law.sample_batch(self.rng, cur, self.q_init)
            pilot_remaining -= cur

        main_remaining = self.num_sample - self.pilot_num_sample
        while main_remaining > 0:
            if not self.pilot_finalized:
                self._finalize_pilot_if_needed()
            cur = self.law.batch_rows(self.batch_size, main_remaining)
            yield self.law.sample_batch(self.rng, cur, self.q_stage2)
            main_remaining -= cur

    def aggregate(self, results_collect):
        pos = 0
        while pos < len(results_collect):
            remaining = len(results_collect) - pos
            take = remaining
            if self.owner.pos_traj < len(self.owner.values_traj):
                next_track = (self.owner.pos_traj + 1) * self.interval_track
                to_track = max(next_track - (self.boundary_eval_count + self.store.num_obs), 0)
                if to_track > 0:
                    take = min(take, to_track)

            if self.law.is_paired and (take % 2 == 1):
                if take > 1:
                    take -= 1
                else:
                    take = min(remaining, 2)
            if take <= 0:
                break

            self._append_block(results_collect[pos:pos + take])
            pos += take

            if not self.pilot_finalized and self.store.num_obs >= self.pilot_num_sample:
                self._finalize_pilot_if_needed()

            while (
                self.owner.pos_traj < len(self.owner.values_traj)
                and (self.boundary_eval_count + self.store.num_obs) >= (self.owner.pos_traj + 1) * self.interval_track
            ):
                self.current_estimate = self._readout_estimate()
                self._sync_owner_current_estimate()
                self._write_checkpoint(self.owner.pos_traj, self.current_estimate)
                self.owner.pos_traj += 1

    def _append_block(self, results_collect):
        n = self.num_player
        X = results_collect[:, :n].astype(bool)
        y = results_collect[:, n].astype(np.float64)
        q = results_collect[:, n + 1].astype(np.float64)
        if np.any(q <= 0.0):
            raise ValueError("Encountered non-positive sampling probability.")
        folds = self.law.assign_folds(self.fold_rng, len(y), self.num_folds)
        self.store.append(X, y, q, folds)
        self.backend.append(X, y, q, folds)

    def _predict_surrogate(self, fit, X, context):
        return self.feature_builder.predict_from_rows(fit.beta, X, context)

    def _residual_correction_sum(self, fit, X, y, q):
        correction = np.zeros(self.target.output_dim, dtype=np.float64)
        if len(y) == 0:
            return correction

        chunk_rows = max(1, int(self.readout_chunk_rows))
        for start in range(0, len(y), chunk_rows):
            stop = min(start + chunk_rows, len(y))
            X_chunk = X[start:stop]
            y_chunk = y[start:stop]
            q_chunk = q[start:stop]
            context = self.strata.context_from_X(X_chunk)
            gamma = self.target.raw_gamma(X_chunk, context) / q_chunk[:, None]
            resid = y_chunk - self._predict_surrogate(fit, X_chunk, context)
            correction += gamma.T @ resid
        return correction

    def _size_law_from_pilot_fit(self, fit, X, y, ids):
        context = self.strata.context_from_X(X)
        resid = y - self._predict_surrogate(fit, X, context)
        second_moment = np.zeros(len(self.strata.keys), dtype=np.float64)
        counts = np.zeros(len(self.strata.keys), dtype=np.int64)
        rss = np.zeros(len(self.strata.keys), dtype=np.float64)

        if not self.law.is_paired:
            np.add.at(rss, ids, resid * resid)
            np.add.at(counts, ids, 1)
        else:
            if len(y) % 2 != 0:
                raise ValueError("Paired pilot sample must contain an even number of rows.")
            pair_diff = resid[0:len(y):2] - resid[1:len(y):2]
            pair_sq = pair_diff * pair_diff
            ids_left = ids[0:len(y):2]
            ids_right = ids[1:len(y):2]
            np.add.at(rss, ids_left, pair_sq)
            np.add.at(counts, ids_left, 1)
            np.add.at(rss, ids_right, pair_sq)
            np.add.at(counts, ids_right, 1)

        total_count = int(counts.sum())
        if total_count <= 0:
            second_moment.fill(1.0)
        else:
            global_mse = max(float(rss.sum()) / float(total_count), 1e-12)
            second_moment.fill(global_mse)
            strong = counts >= self.stage2_min_count
            if np.any(strong):
                second_moment[strong] = rss[strong] / counts[strong]
            second_moment = np.maximum(second_moment, 1e-12)
            if self.law.is_paired:
                second_moment = 0.5 * (second_moment + second_moment[::-1])

        factor = self.target.initial_design_factor(self.strata) * np.sqrt(second_moment)
        if self.stage2_floor > 0.0:
            return self.law.mix_with_initial_mass(factor, self.q_init, self.stage2_floor)
        return self.law.normalize_density(factor)

    def _finalize_pilot_if_needed(self):
        if self.pilot_finalized:
            return

        pilot_count = min(self.pilot_num_sample, self.store.num_obs)
        if pilot_count <= 0:
            self.q_stage2 = self.q_init.copy()
            self.pilot_finalized = True
            return

        X, y, q0, _folds = self.store.rows_until(pilot_count)
        context = self.strata.context_from_X(X)
        ids = self.strata.ids_from_context(context)
        q_current = self.q_init.copy()
        fit = self.backend.fit_all(store=self.store)

        for update_idx in range(self.pilot_design_updates):
            if update_idx > 0:
                q_candidate = q_current[ids]
                fit = self.backend.fit_candidate(X, y, q0, q_candidate)
            self.pilot_design_beta_history.append(fit.beta.copy())
            q_current = self._size_law_from_pilot_fit(fit, X, y, ids)
            self.pilot_design_q_history.append(q_current.copy())

        self.q_stage2 = q_current
        self.pilot_finalized = True

    def _crossfit_estimate(self):
        m = self.store.num_obs
        if m <= 0:
            return self.boundary_exact.copy()

        est_sum = np.zeros(self.target.output_dim, dtype=np.float64)
        scored = 0
        _X_all, _y_all, _q_all, folds_all = self.store.rows()
        for fold in np.unique(folds_all):
            holdout_count = self.backend.fold_count(fold)
            train_count = m - holdout_count
            if holdout_count <= 0 or train_count <= 0:
                continue

            fit = self.backend.fit_excluding_fold(fold, store=self.store)
            mask = self.store.fold_mask(fold)
            X, y, q, _folds = self.store.rows(mask)
            est_sum += float(holdout_count) * fit.phi + self._residual_correction_sum(fit, X, y, q)
            scored += holdout_count

        if scored > 0:
            return self.boundary_exact + est_sum / float(scored)

        return self._all_data_aipw_estimate()

    def _all_data_aipw_estimate(self):
        m = self.store.num_obs
        if m <= 0:
            return self.boundary_exact.copy()

        fit = self.backend.fit_all(store=self.store)
        X, y, q, _folds = self.store.rows()
        return self.boundary_exact + fit.phi + self._residual_correction_sum(fit, X, y, q) / float(m)

    def _plugin_estimate(self):
        if self.store.num_obs <= 0:
            return self.boundary_exact.copy()

        fit = self.backend.fit_all(store=self.store)
        return self.boundary_exact + fit.phi

    def _readout_estimate(self):
        if self.readout_mode == "crossfit":
            return self._crossfit_estimate()
        if self.readout_mode == "all_data_aipw":
            return self._all_data_aipw_estimate()
        if self.readout_mode == "plugin":
            return self._plugin_estimate()
        raise RuntimeError(f"Unknown readout mode {self.readout_mode!r}.")

    def finalize(self):
        if not self.pilot_finalized:
            self._finalize_pilot_if_needed()
        if self.store.num_obs > 0:
            self.current_estimate = self._readout_estimate()
        self._sync_owner_current_estimate()
        if (
            self.boundary_checkpoints_missing
            and self.owner.pos_traj >= len(self.owner.values_traj)
            and len(self.owner.values_traj) > 0
            and len(self.owner.values_traj) * self.interval_track < self.boundary_first_valid_eval
        ):
            return self.trajectory.final_public(self.current_estimate)
        self.trajectory.fill_tail(self.owner.pos_traj, self.current_estimate)
        return self.trajectory.final_public(self.current_estimate)


# ---------------------------------------------------------------------------
# Public estimators
# ---------------------------------------------------------------------------


def _normalize_stats_backend_name(name):
    if not isinstance(name, str):
        raise ValueError(
            '`surrogate_stats_backend` must be a string in {"empirical_dense", "exact_conditional"}, '
            f"got {name!r}."
        )
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "empirical": "empirical_dense",
        "dense": "empirical_dense",
        "empirical_dense": "empirical_dense",
        "exact": "exact_conditional",
        "exact_conditional": "exact_conditional",
        "exact_conditional_gram": "exact_conditional",
    }
    if key not in aliases:
        raise ValueError(
            'Unknown `surrogate_stats_backend`. Use "empirical_dense" or '
            f'"exact_conditional". Got {name!r}.'
        )
    return aliases[key]


def _normalize_ridge_scaling(name):
    if not isinstance(name, str):
        raise ValueError('`surrogate_ridge_scaling` must be a string in {"scalar", "size_trace"}, ' f"got {name!r}.")
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "scalar": "scalar",
        "global": "scalar",
        "size_trace": "size_trace",
        "block_trace": "size_trace",
    }
    if key not in aliases:
        raise ValueError('Unknown `surrogate_ridge_scaling`. Use "scalar" or "size_trace". ' f"Got {name!r}.")
    return aliases[key]


def _normalize_correction_solver_mode(name):
    if not isinstance(name, str):
        raise ValueError(
            '`surrogate_correction_solver_mode` must be a string in {"dense", "matrix_free", '
            '"size_player_diagonal"}, '
            f"got {name!r}."
        )
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "dense": "dense",
        "brute_force": "dense",
        "dense_brute_force": "dense",
        "dense_exact": "dense",
        "auto": "matrix_free",
        "matrix_free": "matrix_free",
        "cg": "matrix_free",
        "woodbury_cg": "matrix_free",
        "size_player_diagonal": "size_player_diagonal",
        "size_player_diag": "size_player_diagonal",
        "diag_size_player": "size_player_diagonal",
        "diag_r_empirical_u": "size_player_diagonal",
        "diagonal_r_empirical_u": "size_player_diagonal",
        "diagonal_r_empirical_u_woodbury": "size_player_diagonal",
        "diag_r_u_emp": "size_player_diagonal",
    }
    if key not in aliases:
        raise ValueError(
            'Unknown `surrogate_correction_solver_mode`. Use "dense", '
            '"matrix_free", or "size_player_diagonal". '
            f"Got {name!r}."
        )
    return aliases[key]


def _normalize_readout_mode(name):
    if not isinstance(name, str):
        raise ValueError(
            '`surrogate_readout_mode` must be a string in {"crossfit", "all_data_aipw", "plugin"}, '
            f"got {name!r}."
        )
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "crossfit": "crossfit",
        "cross_fit": "crossfit",
        "kfold": "crossfit",
        "k_fold": "crossfit",
        "all_data": "all_data_aipw",
        "all_data_aipw": "all_data_aipw",
        "all_data_ipw": "all_data_aipw",
        "direct_aipw": "all_data_aipw",
        "noncrossfit": "all_data_aipw",
        "non_crossfit": "all_data_aipw",
        "non_crossfit_aipw": "all_data_aipw",
        "plugin": "plugin",
        "plug_in": "plugin",
        "direct_plugin": "plugin",
    }
    if key not in aliases:
        raise ValueError(
            'Unknown `surrogate_readout_mode`. Use "crossfit", "all_data_aipw", or "plugin". '
            f"Got {name!r}."
        )
    return aliases[key]


class EaseSHAP(estimatorTemplate):
    """
    Draft 19 Appendix B full-vector EaseSHAP estimator.

    Supported working-surrogate bases:

      - ``surrogate_basis = d`` for an integer ``d >= 0``:
            intercept
          + all interaction monomials ``1{T subseteq S}`` for ``1 <= |T| <= d``
          + optional nonlinear size terms.
        In particular, ``d=1`` is the default singleton-indicator class, and
        ``d=0`` is the intercept-only class.

      - ``surrogate_basis = "size_player"``:
            intercept
          + size-by-player indicators ``h_{i,s}(S) = 1{|S| = s, i in S}``
          + optional nonlinear size terms.

    The implementation uses the pooled shared-surrogate criterion from
    Eqs. (73)-(77), replacing the paper's external auxiliary-training sample
    with K-fold cross-fitting on the collected evaluation coalitions.

    Optional exact boundary handling evaluates selected boundary coalition sizes
    once up front, charges those utility evaluations against the total budget,
    restricts random sampling to the remaining sizes, and adds the exact
    boundary contribution back to every reported estimate. The legacy
    ``exact_boundary_handling=True`` setting corresponds to
    ``boundary_policy="fixed", boundary_order=0``, so only the empty and grand
    coalitions are evaluated exactly. When ``boundary_policy`` is
    supplied, it is the source of truth; use ``boundary_policy="none"`` to
    disable boundary handling.

    Stage 1 uses the initialization law ``q_init(S)`` proportional to
    ``sqrt(A_|S|)``. The fixed pilot sample can then be reused for
    ``pilot_design_updates`` alternating surrogate/design updates before Stage
    2. Each design update uses the single-coalition pilot residual MSE rule for
    generic semivalues, or the complement-pair pilot residual-difference rule
    from Eqs. (82)-(84) for symmetric semivalues when
    ``use_complement_sampling=True``. The default
    ``pilot_design_updates=1`` recovers the original one-update procedure.

    For symmetric semivalues, the sampler emits complement pairs ``(S, S^c)``
    and stores the coalition-level law. The final estimator is recomputed at
    each tracked stage using the current cross-fitted surrogate, so previously
    collected observations are re-scored under the current refit rather than
    frozen at the time they arrived.
    """

    def __init__(
        self,
        *,
        pilot_nue=None,
        pilot_fraction=0.2,
        pilot_design_updates=1,
        use_complement_sampling=True,
        surrogate_ridge_lambda=1.0,
        surrogate_ridge_schedule="fixed",
        surrogate_ridge_scaling="scalar",
        surrogate_stats_backend="empirical_dense",
        surrogate_solver_mode="dense",
        surrogate_r_correction_alpha=0.0,
        surrogate_u_correction_alpha=0.0,
        surrogate_correction_solver_mode="dense",
        surrogate_correction_max_iter=None,
        surrogate_correction_tol=1e-8,
        surrogate_readout_mode="crossfit",
        stage2_size_floor=1e-8,
        stage2_min_size_count=2,
        num_folds=10,
        surrogate_basis=1,
        include_nonlinear_size_terms=True,
        exact_boundary_handling=True,
        boundary_policy=None,
        boundary_order=0,
        **kwargs,
    ):
        if "surrogate_r_correction_solver_mode" in kwargs:
            raise TypeError("Use `surrogate_correction_solver_mode`; the R-only solver mode was removed.")
        if pilot_nue is not None:
            if not isinstance(pilot_nue, (int, np.integer)):
                raise ValueError(f"`pilot_nue` must be an integer or None, got {pilot_nue!r}.")
            if int(pilot_nue) < 0:
                raise ValueError(f"`pilot_nue` must be >= 0, got {pilot_nue!r}.")
        pilot_fraction = float(pilot_fraction)
        if not np.isfinite(pilot_fraction) or not (0.0 <= pilot_fraction <= 1.0):
            raise ValueError("`pilot_fraction` must be finite and lie in [0, 1], " f"got {pilot_fraction!r}.")
        if isinstance(pilot_design_updates, (bool, np.bool_)) or not isinstance(
            pilot_design_updates, (int, np.integer)
        ):
            raise ValueError(
                "`pilot_design_updates` must be an integer >= 1, "
                f"got {pilot_design_updates!r}."
            )
        pilot_design_updates = int(pilot_design_updates)
        if pilot_design_updates < 1:
            raise ValueError(
                "`pilot_design_updates` must be >= 1, "
                f"got {pilot_design_updates!r}."
            )
        surrogate_ridge_lambda = float(surrogate_ridge_lambda)
        if not np.isfinite(surrogate_ridge_lambda) or surrogate_ridge_lambda < 0.0:
            raise ValueError("`surrogate_ridge_lambda` must be finite and >= 0, " f"got {surrogate_ridge_lambda!r}.")
        if not isinstance(surrogate_ridge_schedule, str):
            raise ValueError('`surrogate_ridge_schedule` must be a string in {"fixed", "times_m"}, ' f"got {surrogate_ridge_schedule!r}.")
        surrogate_ridge_schedule = surrogate_ridge_schedule.strip().lower()
        if surrogate_ridge_schedule not in {"fixed", "times_m"}:
            raise ValueError('Unknown `surrogate_ridge_schedule`. Use "fixed" or "times_m". ' f"Got {surrogate_ridge_schedule!r}.")
        surrogate_ridge_scaling = _normalize_ridge_scaling(surrogate_ridge_scaling)
        surrogate_stats_backend = _normalize_stats_backend_name(surrogate_stats_backend)
        if not isinstance(surrogate_solver_mode, str):
            raise ValueError(
                '`surrogate_solver_mode` must be a string such as "dense" or "first_order", '
                f"got {surrogate_solver_mode!r}."
            )
        surrogate_solver_mode = surrogate_solver_mode.strip().lower().replace("-", "_")
        if (
            surrogate_stats_backend != "exact_conditional"
            and surrogate_solver_mode not in {"auto", "dense", "dense_exact"}
        ):
            raise ValueError('`surrogate_solver_mode` is only used with `surrogate_stats_backend="exact_conditional"`.')
        surrogate_correction_solver_mode = _normalize_correction_solver_mode(surrogate_correction_solver_mode)
        if surrogate_correction_max_iter is None:
            correction_max_iter = None
        else:
            if not isinstance(surrogate_correction_max_iter, (int, np.integer)):
                raise ValueError(
                    "`surrogate_correction_max_iter` must be an integer or None, "
                    f"got {surrogate_correction_max_iter!r}."
                )
            correction_max_iter = int(surrogate_correction_max_iter)
            if correction_max_iter < 0:
                raise ValueError(f"`surrogate_correction_max_iter` must be >= 0, got {correction_max_iter!r}.")
        correction_tol = float(surrogate_correction_tol)
        if not np.isfinite(correction_tol) or correction_tol < 0.0:
            raise ValueError(
                "`surrogate_correction_tol` must be finite and >= 0, "
                f"got {surrogate_correction_tol!r}."
            )
        surrogate_r_correction_alpha = float(surrogate_r_correction_alpha)
        if not np.isfinite(surrogate_r_correction_alpha) or not (0.0 <= surrogate_r_correction_alpha <= 1.0):
            raise ValueError(
                "`surrogate_r_correction_alpha` must be finite and lie in [0, 1], "
                f"got {surrogate_r_correction_alpha!r}."
            )
        surrogate_u_correction_alpha = float(surrogate_u_correction_alpha)
        if not np.isfinite(surrogate_u_correction_alpha) or not (0.0 <= surrogate_u_correction_alpha <= 1.0):
            raise ValueError(
                "`surrogate_u_correction_alpha` must be finite and lie in [0, 1], "
                f"got {surrogate_u_correction_alpha!r}."
            )
        surrogate_readout_mode = _normalize_readout_mode(surrogate_readout_mode)
        if (
            surrogate_r_correction_alpha > 0.0 or surrogate_u_correction_alpha > 0.0
        ) and surrogate_stats_backend != "exact_conditional":
            raise ValueError(
                "`surrogate_r_correction_alpha > 0` or `surrogate_u_correction_alpha > 0` requires "
                '`surrogate_stats_backend="exact_conditional"`.'
            )
        if (
            surrogate_r_correction_alpha > 0.0 or surrogate_u_correction_alpha > 0.0
        ) and (
            surrogate_correction_solver_mode == "dense"
            and surrogate_solver_mode not in {"auto", "dense", "dense_exact"}
        ):
            raise ValueError(
                "Structured `surrogate_solver_mode` values cannot be combined with empirical R/U correction yet; "
                'use `surrogate_solver_mode="dense"` or set correction alphas to 0.'
            )
        if (
            surrogate_r_correction_alpha > 0.0 or surrogate_u_correction_alpha > 0.0
        ) and (
            surrogate_correction_solver_mode == "matrix_free"
            and surrogate_solver_mode in {"auto", "dense", "dense_exact"}
        ):
            raise ValueError(
                "Matrix-free empirical correction requires a structured `surrogate_solver_mode` "
                'such as "first_order", "second_order", or "size_player".'
            )
        if pilot_design_updates > 1 and surrogate_stats_backend == "exact_conditional":
            supports_candidate_refit = (
                surrogate_correction_solver_mode == "matrix_free"
                and abs(surrogate_r_correction_alpha - 1.0) <= 1e-12
                and abs(surrogate_u_correction_alpha - 1.0) <= 1e-12
            )
            if not supports_candidate_refit:
                raise ValueError(
                    "`pilot_design_updates > 1` with exact-conditional statistics requires "
                    "the matrix-free empirical correction solver with "
                    "`surrogate_r_correction_alpha = surrogate_u_correction_alpha = 1`."
                )
        stage2_size_floor = float(stage2_size_floor)
        if not np.isfinite(stage2_size_floor) or not (0.0 <= stage2_size_floor < 1.0):
            raise ValueError("`stage2_size_floor` must be finite and lie in [0, 1), " f"got {stage2_size_floor!r}.")
        if not isinstance(stage2_min_size_count, (int, np.integer)):
            raise ValueError("`stage2_min_size_count` must be an integer >= 1, " f"got {stage2_min_size_count!r}.")
        stage2_min_size_count = int(stage2_min_size_count)
        if stage2_min_size_count < 1:
            raise ValueError(f"`stage2_min_size_count` must be >= 1, got {stage2_min_size_count!r}.")
        if not isinstance(num_folds, (int, np.integer)):
            raise ValueError(f"`num_folds` must be an integer >= 2, got {num_folds!r}.")
        num_folds = int(num_folds)
        if num_folds < 2:
            raise ValueError(f"`num_folds` must be >= 2, got {num_folds!r}.")
        boundary_policy = _normalize_boundary_policy(boundary_policy, exact_boundary_handling)
        boundary_order = _normalize_boundary_order(boundary_order)

        self.pilot_nue = None if pilot_nue is None else int(pilot_nue)
        self.pilot_fraction = pilot_fraction
        self.pilot_design_updates = pilot_design_updates
        self.use_complement_sampling = bool(use_complement_sampling)
        self.surrogate_ridge_lambda = surrogate_ridge_lambda
        self.surrogate_ridge_schedule = surrogate_ridge_schedule
        self.surrogate_ridge_scaling = surrogate_ridge_scaling
        self.surrogate_stats_backend = surrogate_stats_backend
        self.surrogate_solver_mode = surrogate_solver_mode
        self.surrogate_r_correction_alpha = surrogate_r_correction_alpha
        self.surrogate_u_correction_alpha = surrogate_u_correction_alpha
        self.surrogate_correction_solver_mode = surrogate_correction_solver_mode
        self.surrogate_correction_max_iter = correction_max_iter
        self.surrogate_correction_tol = correction_tol
        self.surrogate_readout_mode = surrogate_readout_mode
        self.stage2_size_floor = stage2_size_floor
        self.stage2_min_size_count = stage2_min_size_count
        self.num_folds = num_folds
        self.surrogate_basis = surrogate_basis
        self.include_nonlinear_size_terms = bool(include_nonlinear_size_terms)
        self.boundary_policy = boundary_policy
        self.boundary_order = boundary_order
        self.exact_boundary_handling = boundary_policy != "none"
        self._requires_serial_feedback = True

        super(EaseSHAP, self).__init__(**kwargs)

        n = self.num_player
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=surrogate_basis,
            include_nonlinear_size_terms=include_nonlinear_size_terms,
        )
        if surrogate_ridge_scaling == "size_trace":
            if surrogate_stats_backend != "exact_conditional":
                raise ValueError('`surrogate_ridge_scaling="size_trace"` requires `surrogate_stats_backend="exact_conditional"`.')
            if feature_builder.surrogate_basis_kind != "size_player":
                raise ValueError('`surrogate_ridge_scaling="size_trace"` requires `surrogate_basis="size_player"`.')
        total_budget = self.nue_avg * n
        boundary_sizes = _resolve_boundary_sizes(n, total_budget, boundary_policy, boundary_order)
        sampling_mask = np.ones(n + 1, dtype=bool)
        if boundary_sizes:
            sampling_mask[np.asarray(boundary_sizes, dtype=np.int64)] = False
        boundary_X = _boundary_subset_matrix_for_sizes(n, boundary_sizes)

        boundary_eval_count = int(len(boundary_X))
        if total_budget < boundary_eval_count:
            raise ValueError(
                "Exact boundary handling requires at least "
                f"{boundary_eval_count} utility evaluations, but "
                f"`nue_avg * num_player` is only {total_budget}."
            )
        num_sample = total_budget - boundary_eval_count
        if not np.any(sampling_mask):
            num_sample = 0
        has_random_stage = np.any(sampling_mask)
        boundary_checkpoints_missing = boundary_policy == "adaptive" and boundary_eval_count > 0
        boundary_first_valid_eval = boundary_eval_count + 1 if has_random_stage else boundary_eval_count
        self.boundary_sizes = tuple(boundary_sizes)
        self.boundary_eval_count = boundary_eval_count

        strata = _SizeStrata(n, sampling_mask)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue=self.semivalue,
            semivalue_param=self.semivalue_param,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        law = _StratifiedLaw(n=n, strata=strata, is_paired=self.use_complement_sampling and target.is_symmetric)

        batch_size = max(1, self.nue_per_proc)
        if law.is_paired:
            batch_size = max(2, (batch_size // 2) * 2)
            num_sample = (num_sample // 2) * 2
        nue_per_proc_run = batch_size
        interval_track = self.nue_track_avg * n
        if self.pilot_nue is None:
            pilot_num_sample = int(round(self.pilot_fraction * num_sample))
        else:
            pilot_num_sample = self.pilot_nue * n
        pilot_num_sample = max(0, min(pilot_num_sample, num_sample))
        if law.is_paired:
            pilot_num_sample = max(0, (pilot_num_sample // 2) * 2)

        store_bytes = _ObservationStore(num_sample=num_sample, n=n).estimate_memory_bytes()
        backend_cls = _EmpiricalDenseStats if surrogate_stats_backend == "empirical_dense" else _ExactConditionalStats
        if backend_cls is _EmpiricalDenseStats:
            stats_bytes = backend_cls.estimate_memory_bytes_for(
                feature_dim=feature_builder.dim,
                output_dim=target.output_dim,
                num_folds=num_folds,
            )
        else:
            stats_bytes = backend_cls.estimate_memory_bytes_for(
                feature_dim=feature_builder.dim,
                output_dim=target.output_dim,
                num_folds=num_folds,
                num_strata=len(strata.keys),
                has_group_counts=False,
                solver_mode=surrogate_solver_mode,
                r_correction_alpha=surrogate_r_correction_alpha,
                u_correction_alpha=surrogate_u_correction_alpha,
                correction_solver_mode=surrogate_correction_solver_mode,
            )
        backend_state_bytes = store_bytes + stats_bytes
        if backend_state_bytes > 2_000_000_000:
            raise ValueError(
                "Selected surrogate basis is too large for the selected surrogate stats backend "
                f"(estimated bytes={backend_state_bytes}). Reduce `surrogate_basis`, "
                "disable nonlinear size terms, use a smaller budget, or choose a different backend."
            )

        store = _ObservationStore(num_sample=num_sample, n=n)
        backend_kwargs = dict(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=surrogate_ridge_lambda,
            ridge_schedule=surrogate_ridge_schedule,
            num_folds=num_folds,
        )
        if backend_cls is _ExactConditionalStats:
            backend_kwargs["ridge_scaling"] = surrogate_ridge_scaling
            backend_kwargs["solver_mode"] = surrogate_solver_mode
            backend_kwargs["r_correction_alpha"] = surrogate_r_correction_alpha
            backend_kwargs["u_correction_alpha"] = surrogate_u_correction_alpha
            backend_kwargs["correction_solver_mode"] = surrogate_correction_solver_mode
            backend_kwargs["correction_max_iter"] = correction_max_iter
            backend_kwargs["correction_tol"] = correction_tol
        backend = backend_cls(**backend_kwargs)
        trajectory = _TrajectoryAdapter(self, target.output_dim)
        self.interval_track = interval_track
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.nue_per_proc_run = nue_per_proc_run
        self._feature_dim = feature_builder.dim
        self._pair_sampling = law.is_paired

        self._engine = _EaseEngine(
            owner=self,
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            law=law,
            store=store,
            backend=backend,
            trajectory=trajectory,
            boundary_X=boundary_X,
            game_func=self.game_func,
            game_args=self.game_args,
            num_player=n,
            num_sample=num_sample,
            batch_size=batch_size,
            nue_per_proc_run=nue_per_proc_run,
            interval_track=interval_track,
            boundary_eval_count=boundary_eval_count,
            boundary_checkpoints_missing=boundary_checkpoints_missing,
            boundary_first_valid_eval=boundary_first_valid_eval,
            pilot_num_sample=pilot_num_sample,
            pilot_design_updates=pilot_design_updates,
            stage2_floor=stage2_size_floor,
            stage2_min_count=stage2_min_size_count,
            num_folds=num_folds,
            readout_mode=surrogate_readout_mode,
            estimator_seed=self.estimator_seed,
        )
        self._num_batches_hint = self._engine.num_batches_hint

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_engine", None)
        return state

    def sampling(self):
        return self._engine.sampling()

    def run(self, samples):
        return _evaluate_sample_batch(samples, self.game_func, self.game_args, self.num_player)

    def aggregate(self, results_collect):
        return self._engine.aggregate(results_collect)

    def finalize(self):
        return self._engine.finalize()

    def _process(self, inputs):
        raise NotImplementedError("EaseSHAP processes batches in aggregate().")

    def _estimate(self):
        return self._engine.current_estimate.copy()


class EaseSHAP_group(groupEstimatorTemplate):
    """
    Scalar group-sum EaseSHAP estimator for ``Phi_G(u)``.

    Stage 1 samples cells ``(|S|, |S cap G|)`` with probability proportional to
    ``|rho_G(S)|`` on the active design support. Stage 2 estimates cell
    residual second moments from the pilot sample and uses a density
    proportional to ``|rho_{s,r}| sqrt(M_{s,r})``. A scalar profiled ridge
    surrogate is refit by K-fold cross-fitting whenever an estimate is
    reported. Boundary handling follows the same ``boundary_policy`` precedence
    as ``EaseSHAP``.
    """

    def __init__(
        self,
        *,
        pilot_nue=None,
        pilot_fraction=0.2,
        surrogate_ridge_lambda=1.0,
        surrogate_ridge_schedule="fixed",
        surrogate_ridge_scaling="scalar",
        surrogate_stats_backend="empirical_dense",
        surrogate_r_correction_alpha=0.0,
        surrogate_u_correction_alpha=0.0,
        surrogate_correction_solver_mode="dense",
        surrogate_correction_max_iter=None,
        surrogate_correction_tol=1e-8,
        surrogate_readout_mode="crossfit",
        stage2_cell_floor=1e-8,
        stage2_min_cell_count=2,
        rho_support_tol=0.0,
        num_folds=10,
        surrogate_basis=1,
        include_nonlinear_size_terms=True,
        include_group_overlap_ratio=True,
        exact_boundary_handling=True,
        boundary_policy=None,
        boundary_order=1,
        **kwargs,
    ):
        if "surrogate_r_correction_solver_mode" in kwargs:
            raise TypeError("Use `surrogate_correction_solver_mode`; the R-only solver mode was removed.")
        boundary_policy = _normalize_boundary_policy(boundary_policy, exact_boundary_handling)
        boundary_order = _normalize_boundary_order(boundary_order)
        super(EaseSHAP_group, self).__init__(**kwargs)
        self.pilot_nue = None if pilot_nue is None else int(pilot_nue)
        self.pilot_fraction = float(pilot_fraction)
        self.surrogate_ridge_lambda = float(surrogate_ridge_lambda)
        self.surrogate_ridge_schedule = str(surrogate_ridge_schedule).strip().lower()
        self.surrogate_ridge_scaling = _normalize_ridge_scaling(surrogate_ridge_scaling)
        self.surrogate_stats_backend = _normalize_stats_backend_name(surrogate_stats_backend)
        self.surrogate_r_correction_alpha = float(surrogate_r_correction_alpha)
        self.surrogate_u_correction_alpha = float(surrogate_u_correction_alpha)
        self.surrogate_correction_solver_mode = _normalize_correction_solver_mode(surrogate_correction_solver_mode)
        self.surrogate_readout_mode = _normalize_readout_mode(surrogate_readout_mode)
        if surrogate_correction_max_iter is None:
            self.surrogate_correction_max_iter = None
        else:
            if not isinstance(surrogate_correction_max_iter, (int, np.integer)):
                raise ValueError(
                    "`surrogate_correction_max_iter` must be an integer or None, "
                    f"got {surrogate_correction_max_iter!r}."
                )
            self.surrogate_correction_max_iter = int(surrogate_correction_max_iter)
        self.surrogate_correction_tol = float(surrogate_correction_tol)
        self.stage2_cell_floor = float(stage2_cell_floor)
        self.stage2_min_cell_count = int(stage2_min_cell_count)
        self.rho_support_tol = float(rho_support_tol)
        self.num_folds = int(num_folds)
        self.surrogate_basis = surrogate_basis
        self.include_nonlinear_size_terms = bool(include_nonlinear_size_terms)
        self.include_group_overlap_ratio = bool(include_group_overlap_ratio)
        self.boundary_policy = boundary_policy
        self.boundary_order = boundary_order
        self.exact_boundary_handling = boundary_policy != "none"
        self._requires_serial_feedback = True

        if not (0.0 <= self.pilot_fraction <= 1.0):
            raise ValueError("`pilot_fraction` must lie in [0, 1].")
        if self.surrogate_ridge_schedule not in {"fixed", "times_m"}:
            raise ValueError('`surrogate_ridge_schedule` must be "fixed" or "times_m".')
        if not np.isfinite(self.surrogate_r_correction_alpha) or not (0.0 <= self.surrogate_r_correction_alpha <= 1.0):
            raise ValueError(
                "`surrogate_r_correction_alpha` must be finite and lie in [0, 1], "
                f"got {self.surrogate_r_correction_alpha!r}."
            )
        if not np.isfinite(self.surrogate_u_correction_alpha) or not (0.0 <= self.surrogate_u_correction_alpha <= 1.0):
            raise ValueError(
                "`surrogate_u_correction_alpha` must be finite and lie in [0, 1], "
                f"got {self.surrogate_u_correction_alpha!r}."
            )
        if self.surrogate_correction_max_iter is not None and self.surrogate_correction_max_iter < 0:
            raise ValueError("`surrogate_correction_max_iter` must be >= 0.")
        if not np.isfinite(self.surrogate_correction_tol) or self.surrogate_correction_tol < 0.0:
            raise ValueError("`surrogate_correction_tol` must be finite and >= 0.")
        if (
            self.surrogate_r_correction_alpha > 0.0 or self.surrogate_u_correction_alpha > 0.0
        ) and self.surrogate_stats_backend != "exact_conditional":
            raise ValueError(
                "`surrogate_r_correction_alpha > 0` or `surrogate_u_correction_alpha > 0` requires "
                '`surrogate_stats_backend="exact_conditional"`.'
            )
        if self.surrogate_ridge_scaling != "scalar":
            raise ValueError(
                '`surrogate_ridge_scaling="size_trace"` is currently only supported by full '
                'EaseSHAP with `surrogate_stats_backend="exact_conditional"` and '
                '`surrogate_basis="size_player"`.'
            )
        if self.num_folds < 2:
            raise ValueError("`num_folds` must be >= 2.")
        if self.stage2_min_cell_count < 1:
            raise ValueError("`stage2_min_cell_count` must be >= 1.")
        if self.rho_support_tol < 0.0:
            raise ValueError("`rho_support_tol` must be nonnegative.")
        if not (0.0 <= self.stage2_cell_floor < 1.0):
            raise ValueError("`stage2_cell_floor` must lie in [0, 1).")

        n = self.num_player
        boundary_sizes = _resolve_boundary_sizes(n, self.total_budget, boundary_policy, boundary_order)
        boundary_X = _boundary_subset_matrix_for_sizes(n, boundary_sizes)
        boundary_eval_count = int(len(boundary_X))
        if self.total_budget < boundary_eval_count:
            raise ValueError(
                "Exact boundary handling requires at least "
                f"{boundary_eval_count} utility evaluations, but "
                f"`nue_avg * num_player` is only {self.total_budget}."
            )

        strata = _GroupCellStrata.build(
            n,
            self.group,
            self.group_mask,
            self.exact_boundary_handling,
            boundary_sizes=boundary_sizes,
        )
        num_sample = self.total_budget - boundary_eval_count
        if not np.any(strata.sampling_mask):
            num_sample = 0
        has_random_stage = np.any(strata.sampling_mask)
        boundary_checkpoints_missing = boundary_policy == "adaptive" and boundary_eval_count > 0
        boundary_first_valid_eval = boundary_eval_count + 1 if has_random_stage else boundary_eval_count
        self.boundary_sizes = tuple(boundary_sizes)
        self.boundary_eval_count = boundary_eval_count
        batch_size = max(1, self.nue_per_proc)
        nue_per_proc_run = batch_size
        if self.pilot_nue is None:
            pilot_num_sample = int(round(self.pilot_fraction * num_sample))
        else:
            pilot_num_sample = min(num_sample, max(0, self.pilot_nue * n))
        pilot_finalized = pilot_num_sample == 0

        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=surrogate_basis,
            include_nonlinear_size_terms=include_nonlinear_size_terms,
            include_group_overlap_ratio=include_group_overlap_ratio,
        )
        boundary_context = strata.context_from_X(boundary_X)
        target = _GroupSumTarget(
            n=n,
            group=self.group,
            group_mask=self.group_mask,
            semivalue=self.semivalue,
            semivalue_param=self.semivalue_param,
            feature_builder=feature_builder,
            strata=strata,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
            rho_support_tol=rho_support_tol,
        )
        law = _StratifiedLaw(n=n, strata=strata, is_paired=False)
        store = _ObservationStore(num_sample=num_sample, n=n)
        backend_cls = _EmpiricalDenseStats if self.surrogate_stats_backend == "empirical_dense" else _ExactConditionalStats
        if backend_cls is _EmpiricalDenseStats:
            stats_bytes = backend_cls.estimate_memory_bytes_for(
                feature_dim=feature_builder.dim,
                output_dim=target.output_dim,
                num_folds=self.num_folds,
            )
        else:
            stats_bytes = backend_cls.estimate_memory_bytes_for(
                feature_dim=feature_builder.dim,
                output_dim=target.output_dim,
                num_folds=self.num_folds,
                num_strata=len(strata.keys),
                has_group_counts=True,
                r_correction_alpha=self.surrogate_r_correction_alpha,
                u_correction_alpha=self.surrogate_u_correction_alpha,
                correction_solver_mode=self.surrogate_correction_solver_mode,
            )
        backend_state_bytes = store.estimate_memory_bytes() + stats_bytes
        if backend_state_bytes > 2_000_000_000:
            raise ValueError(
                "Selected surrogate basis is too large for the selected surrogate stats backend "
                f"(estimated bytes={backend_state_bytes}). Reduce `surrogate_basis`, "
                "disable nonlinear size terms, use a smaller budget, or choose a different backend."
            )
        backend_kwargs = dict(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=self.surrogate_ridge_lambda,
            ridge_schedule=self.surrogate_ridge_schedule,
            num_folds=self.num_folds,
        )
        if backend_cls is _ExactConditionalStats:
            backend_kwargs["r_correction_alpha"] = self.surrogate_r_correction_alpha
            backend_kwargs["u_correction_alpha"] = self.surrogate_u_correction_alpha
            backend_kwargs["correction_solver_mode"] = self.surrogate_correction_solver_mode
            backend_kwargs["correction_max_iter"] = self.surrogate_correction_max_iter
            backend_kwargs["correction_tol"] = self.surrogate_correction_tol
        backend = backend_cls(**backend_kwargs)
        trajectory = _TrajectoryAdapter(self, target.output_dim)
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.nue_per_proc_run = nue_per_proc_run
        self._feature_dim = feature_builder.dim

        self._engine = _EaseEngine(
            owner=self,
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            law=law,
            store=store,
            backend=backend,
            trajectory=trajectory,
            boundary_X=boundary_X,
            game_func=self.game_func,
            game_args=self.game_args,
            num_player=n,
            num_sample=num_sample,
            batch_size=batch_size,
            nue_per_proc_run=nue_per_proc_run,
            interval_track=self.interval_track,
            boundary_eval_count=boundary_eval_count,
            boundary_checkpoints_missing=boundary_checkpoints_missing,
            boundary_first_valid_eval=boundary_first_valid_eval,
            pilot_num_sample=pilot_num_sample,
            pilot_design_updates=1,
            stage2_floor=self.stage2_cell_floor,
            stage2_min_count=self.stage2_min_cell_count,
            num_folds=self.num_folds,
            readout_mode=self.surrogate_readout_mode,
            estimator_seed=self.estimator_seed,
        )
        self._engine.pilot_finalized = pilot_finalized
        self._num_batches_hint = self._engine.num_batches_hint

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_engine", None)
        return state

    def sampling(self):
        return self._engine.sampling()

    def run(self, samples):
        return _evaluate_sample_batch(samples, self.game_func, self.game_args, self.num_player)

    def aggregate(self, results_collect):
        return self._engine.aggregate(results_collect)

    def finalize(self):
        return self._engine.finalize()
