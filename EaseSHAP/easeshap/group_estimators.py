import itertools
import math

import numpy as np
from scipy import special
from scipy.stats import hypergeom

from .estimators import runEstimator


def semivalue_coefficients(num_player, semivalue, semivalue_param):
    """Return per-subset semivalue coefficients alpha_s^(n), s=0,...,n-1."""
    n = int(num_player)
    if semivalue == "shapley":
        return np.array(
            [1.0 / (n * float(special.comb(n - 1, s, exact=False))) for s in range(n)],
            dtype=np.float64,
        )

    if semivalue == "weighted_banzhaf":
        p = float(semivalue_param)
        return np.array([p ** s * (1.0 - p) ** (n - 1 - s) for s in range(n)], dtype=np.float64)

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
            weights[s] = cur / float(special.comb(n - 1, s, exact=False))
        return weights

    raise NotImplementedError(f"Unknown semivalue {semivalue!r}.")


def _safe_comb(n, k):
    if k < 0 or k > n or n < 0:
        return 0.0
    return float(special.comb(n, k, exact=False))


def validate_group(group, num_player):
    group = np.asarray(group, dtype=np.int64)
    if group.ndim != 1:
        raise ValueError("`group` must be a one-dimensional list or array of player indices.")
    if len(group) == 0:
        raise ValueError("`group` must be nonempty.")
    if np.any(group < 0) or np.any(group >= num_player):
        raise ValueError(f"`group` entries must lie in [0, {num_player}).")
    if len(np.unique(group)) != len(group):
        raise ValueError("`group` contains duplicate player indices.")
    return np.sort(group)


def group_sum_coefficient(size, overlap, group_size, alpha):
    prev_val = alpha[size - 1] if size > 0 else 0.0
    cur_val = alpha[size] if size < len(alpha) else 0.0
    return overlap * prev_val - (group_size - overlap) * cur_val


def exact_group_sum_value(game_func, game_args, num_player, semivalue, semivalue_param, group):
    """Exact Phi_G(u)=sum_{i in G} phi_i(u), by coalition coefficient enumeration."""
    n = int(num_player)
    group = validate_group(group, n)
    group_mask = np.zeros(n, dtype=bool)
    group_mask[group] = True
    alpha = semivalue_coefficients(n, semivalue, semivalue_param)
    game = game_func(**game_args)

    total = 0.0
    for bits in itertools.product([False, True], repeat=n):
        subset = np.asarray(bits, dtype=bool)
        s = int(subset.sum())
        r = int(np.logical_and(subset, group_mask).sum())
        rho = group_sum_coefficient(s, r, len(group), alpha)
        if rho != 0.0:
            total += rho * float(game.evaluate(subset))
    return float(total)


class runGroupSumEstimator:
    """
    Orchestration wrapper for scalar group-sum semivalue estimators.

    The target is Phi_G(u)=sum_{i in G} phi_i(u). Returned trajectories are
    one-dimensional arrays indexed by the usual utility-evaluation checkpoints.
    """

    def __init__(
        self,
        *,
        estimator,
        n_process,
        semivalue,
        semivalue_param,
        group,
        game_func,
        game_args,
        num_player,
        nue_avg,
        nue_per_proc,
        nue_track_avg,
        estimator_seed=2026,
        file_prog=None,
        **kwargs_estimator,
    ):
        self.estimator = estimator
        self.n_process = n_process
        self.file_prog = file_prog
        self.semivalue = semivalue
        self.semivalue_param = semivalue_param
        self.group = group
        self.game_func = game_func
        self.game_args = game_args
        self.num_player = num_player
        self.nue_avg = nue_avg
        self.nue_per_proc = nue_per_proc
        self.nue_track_avg = nue_track_avg
        self.estimator_seed = estimator_seed
        self.kwargs_estimator = kwargs_estimator
        self.estimator_run = None

    def run(self):
        if self.estimator not in _GROUP_ALGORITHM_MAP:
            raise KeyError(
                f"Unknown group-sum estimator {self.estimator!r}. "
                f"Known estimators: {sorted(_GROUP_ALGORITHM_MAP)}"
            )
        estimator_cls = _GROUP_ALGORITHM_MAP[self.estimator]
        estimator = estimator_cls(
            semivalue=self.semivalue,
            semivalue_param=self.semivalue_param,
            group=self.group,
            game_func=self.game_func,
            game_args=self.game_args,
            num_player=self.num_player,
            nue_avg=self.nue_avg,
            nue_per_proc=self.nue_per_proc,
            nue_track_avg=self.nue_track_avg,
            estimator_seed=self.estimator_seed,
            n_process=self.n_process,
            **self.kwargs_estimator,
        )
        self.estimator_run = estimator

        if hasattr(estimator, "run_all"):
            return estimator.run_all()

        for samples in estimator.sampling():
            estimator.aggregate(estimator.run(samples))
        return estimator.finalize()


class groupEstimatorTemplate:
    def __init__(
        self,
        *,
        semivalue,
        semivalue_param,
        group,
        game_func,
        game_args,
        num_player,
        nue_avg,
        nue_per_proc,
        nue_track_avg,
        estimator_seed,
        n_process=1,
        **_kwargs,
    ):
        self.semivalue = semivalue
        self.semivalue_param = semivalue_param
        self.num_player = int(num_player)
        self.group = validate_group(group, self.num_player)
        self.group_mask = np.zeros(self.num_player, dtype=bool)
        self.group_mask[self.group] = True
        self.non_group = np.flatnonzero(~self.group_mask)
        self.game_func = game_func
        self.game_args = game_args
        self.nue_avg = int(nue_avg)
        self.nue_per_proc = int(nue_per_proc)
        self.nue_track_avg = int(nue_track_avg)
        self.estimator_seed = int(estimator_seed)
        self.n_process = int(n_process)

        self.total_budget = max(0, self.nue_avg * self.num_player)
        self.interval_track = max(1, self.nue_track_avg * self.num_player)
        num_traj = max(1, self.nue_avg // max(1, self.nue_track_avg))
        self.values_traj = np.empty(num_traj, dtype=np.float64)
        self.pos_traj = 0
        self._current_estimate = 0.0

    @property
    def group_size(self):
        return int(len(self.group))

    def _record_tracks(self, num_eval):
        while self.pos_traj < len(self.values_traj) and num_eval >= (self.pos_traj + 1) * self.interval_track:
            self.values_traj[self.pos_traj] = self._current_estimate
            self.pos_traj += 1

    def _fill_traj_tail(self):
        if self.pos_traj < len(self.values_traj):
            self.values_traj[self.pos_traj:] = self._current_estimate
        elif len(self.values_traj):
            self.values_traj[-1] = self._current_estimate


class EaseSHAP_group(groupEstimatorTemplate):
    """
    EaseSHAP scalar group-sum estimator for Phi_G(u).

    Stage 1 samples cells (|S|, |S cap G|) with probability proportional to
    |rho_G(S)|. Stage 2 estimates cell residual second moments from the pilot
    sample and uses q_{s,r} proportional to |rho_{s,r}| sqrt(M_{s,r}).
    A scalar profiled ridge surrogate is refit by K-fold cross-fitting when
    reported.
    """

    def __init__(
        self,
        *,
        pilot_nue=None,
        pilot_fraction=0.2,
        surrogate_ridge_lambda=1.0,
        surrogate_ridge_schedule="fixed",
        stage2_cell_floor=1e-8,
        stage2_min_cell_count=2,
        rho_support_tol=0.0,
        num_folds=10,
        surrogate_basis=1,
        include_nonlinear_size_terms=True,
        include_group_overlap_ratio=True,
        exact_boundary_handling=True,
        **kwargs,
    ):
        super(EaseSHAP_group, self).__init__(**kwargs)
        self.pilot_nue = None if pilot_nue is None else int(pilot_nue)
        self.pilot_fraction = float(pilot_fraction)
        self.surrogate_ridge_lambda = float(surrogate_ridge_lambda)
        self.surrogate_ridge_schedule = str(surrogate_ridge_schedule).strip().lower()
        self.stage2_cell_floor = float(stage2_cell_floor)
        self.stage2_min_cell_count = int(stage2_min_cell_count)
        self.rho_support_tol = float(rho_support_tol)
        self.num_folds = int(num_folds)
        self.surrogate_basis = surrogate_basis
        self.include_nonlinear_size_terms = bool(include_nonlinear_size_terms)
        self.include_group_overlap_ratio = bool(include_group_overlap_ratio)
        self.exact_boundary_handling = bool(exact_boundary_handling)

        if not (0.0 <= self.pilot_fraction <= 1.0):
            raise ValueError("`pilot_fraction` must lie in [0, 1].")
        if self.surrogate_ridge_schedule not in {"fixed", "times_m"}:
            raise ValueError('`surrogate_ridge_schedule` must be "fixed" or "times_m".')
        if self.num_folds < 2:
            raise ValueError("`num_folds` must be >= 2.")
        if self.stage2_min_cell_count < 1:
            raise ValueError("`stage2_min_cell_count` must be >= 1.")
        if self.rho_support_tol < 0.0:
            raise ValueError("`rho_support_tol` must be nonnegative.")
        if not (0.0 <= self.stage2_cell_floor < 1.0):
            raise ValueError("`stage2_cell_floor` must lie in [0, 1).")

        self.alpha = semivalue_coefficients(self.num_player, self.semivalue, self.semivalue_param)
        self._build_cells()
        self._build_boundary_subsets_and_sampling_mask()
        self._build_design_rho_support()
        self._parse_surrogate_basis()
        self._build_feature_map_and_readout()

        if self.total_budget < self._boundary_eval_count:
            raise ValueError(
                "Exact boundary handling requires at least "
                f"{self._boundary_eval_count} utility evaluations, but "
                f"`nue_avg * num_player` is only {self.total_budget}."
            )
        self.num_sample = self.total_budget - self._boundary_eval_count
        if not np.any(self._sampling_cell_mask):
            self.num_sample = 0
        self.batch_size = max(1, self.nue_per_proc)
        self.nue_per_proc_run = self.batch_size

        if self.pilot_nue is None:
            self._pilot_num_sample = int(round(self.pilot_fraction * self.num_sample))
        else:
            self._pilot_num_sample = min(self.num_sample, max(0, self.pilot_nue * self.num_player))
        self._pilot_finalized = self._pilot_num_sample == 0

        self._rng = np.random.Generator(np.random.PCG64(self.estimator_seed))
        self._fold_rng = np.random.Generator(np.random.PCG64(self.estimator_seed + 9173))

        self._boundary_sizes = self._boundary_X.sum(axis=1).astype(np.int64)
        self._boundary_overlaps = (
            self._boundary_X[:, self.group].sum(axis=1).astype(np.int64)
            if len(self._boundary_X)
            else np.empty(0, dtype=np.int64)
        )
        self._boundary_rho = (
            self._rho_values(self._boundary_sizes, self._boundary_overlaps)
            if len(self._boundary_X)
            else np.empty(0, dtype=np.float64)
        )
        self._boundary_Z = (
            self._build_feature_block(self._boundary_X)[0]
            if len(self._boundary_X)
            else np.empty((0, self._feature_dim), dtype=np.float64)
        )
        self._boundary_values = self._evaluate_boundary_values()
        self._boundary_exact = float(self._boundary_rho @ self._boundary_values)

        init_factor = self._cell_rho_design_abs
        self._q_init_cell = self._normalize_cell_law(init_factor)
        self._q_stage2_cell = self._q_init_cell.copy()

        m = self.num_sample
        n = self.num_player
        d = self._feature_dim
        k = self.num_folds
        self._num_obs = 0
        self._X_obs = np.empty((m, n), dtype=bool)
        self._y_obs = np.empty(m, dtype=np.float64)
        self._q_obs = np.empty(m, dtype=np.float64)
        self._size_obs = np.empty(m, dtype=np.int64)
        self._overlap_obs = np.empty(m, dtype=np.int64)
        self._cell_obs = np.empty(m, dtype=np.int64)
        self._fold_obs = np.empty(m, dtype=np.int64)

        self._total_count = 0
        self._A_total = np.zeros((d, d), dtype=np.float64)
        self._c_total = np.zeros(d, dtype=np.float64)
        self._B_total = np.zeros(d, dtype=np.float64)
        self._b_total = 0.0
        self._fold_count = np.zeros(k, dtype=np.int64)
        self._A_fold = np.zeros((k, d, d), dtype=np.float64)
        self._c_fold = np.zeros((k, d), dtype=np.float64)
        self._B_fold = np.zeros((k, d), dtype=np.float64)
        self._b_fold = np.zeros(k, dtype=np.float64)
        self._current_estimate = self._boundary_exact
        self._record_tracks(self._boundary_eval_count)

    def _build_cells(self):
        n = self.num_player
        g = self.group_size
        keys = []
        counts = []
        rhos = []
        index = -np.ones((n + 1, g + 1), dtype=np.int64)
        for s in range(n + 1):
            r_min = max(0, s - (n - g))
            r_max = min(g, s)
            for r in range(r_min, r_max + 1):
                count = _safe_comb(g, r) * _safe_comb(n - g, s - r)
                if count <= 0.0:
                    continue
                index[s, r] = len(keys)
                keys.append((s, r))
                counts.append(count)
                rhos.append(group_sum_coefficient(s, r, g, self.alpha))
        self._cell_keys = keys
        self._cell_count = np.asarray(counts, dtype=np.float64)
        self._cell_rho = np.asarray(rhos, dtype=np.float64)
        self._cell_index = index

    def _build_boundary_subset_matrix(self):
        n = self.num_player
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

    def _build_boundary_subsets_and_sampling_mask(self):
        self._sampling_cell_mask = np.ones(len(self._cell_keys), dtype=bool)
        self._boundary_X = np.empty((0, self.num_player), dtype=bool)
        if self.exact_boundary_handling:
            self._sampling_cell_mask = np.array(
                [1 < s < self.num_player - 1 for s, _r in self._cell_keys],
                dtype=bool,
            )
            self._boundary_X = self._build_boundary_subset_matrix()
        self._boundary_eval_count = int(len(self._boundary_X))

    def _evaluate_boundary_values(self):
        if len(self._boundary_X) == 0:
            return np.empty(0, dtype=np.float64)

        game = self.game_func(**self.game_args)
        values = np.empty(len(self._boundary_X), dtype=np.float64)
        for idx, subset in enumerate(self._boundary_X):
            values[idx] = float(game.evaluate(subset))
        return values

    def _build_design_rho_support(self):
        abs_rho = np.abs(self._cell_rho)
        max_abs = float(abs_rho.max(initial=0.0))
        if max_abs <= 0.0:
            self._cell_rho_design_support = np.zeros_like(abs_rho, dtype=bool)
            self._cell_rho_design_abs = abs_rho
            return

        support = abs_rho > self.rho_support_tol * max_abs
        support &= self._sampling_cell_mask
        self._cell_rho_design_support = support
        self._cell_rho_design_abs = np.where(support, abs_rho, 0.0)

    def _parse_surrogate_basis(self):
        basis = self.surrogate_basis
        if isinstance(basis, (int, np.integer)):
            if int(basis) < 0:
                raise ValueError("Integer `surrogate_basis` must be >= 0.")
            self._surrogate_basis_kind = "interactions"
            self._interaction_degree = int(basis)
            return
        key = str(basis).strip().lower().replace("-", "_")
        if key in {"none", "constant", "intercept"}:
            self._surrogate_basis_kind = "interactions"
            self._interaction_degree = 0
        elif key in {"size_player", "size_by_player", "player_size", "is", "i_s"}:
            self._surrogate_basis_kind = "size_player"
            self._interaction_degree = None
        else:
            raise ValueError('`surrogate_basis` must be an integer, "none", or "size_player".')

    def _cell_readout(self, fn):
        total = 0.0
        for idx, (s, r) in enumerate(self._cell_keys):
            total += self._cell_count[idx] * self._cell_rho[idx] * fn(s, r)
        return float(total)

    def _interaction_readout(self, a, b):
        n = self.num_player
        g = self.group_size
        out = 0.0
        outside_a = a - b
        for idx, (s, r) in enumerate(self._cell_keys):
            if s < a or r < b:
                continue
            cnt = _safe_comb(g - b, r - b) * _safe_comb(n - g - outside_a, s - r - outside_a)
            out += cnt * self._cell_rho[idx]
        return float(out)

    def _size_player_readout(self, player, size):
        n = self.num_player
        g = self.group_size
        is_group = bool(self.group_mask[player])
        out = 0.0
        for idx, (s, r) in enumerate(self._cell_keys):
            if s != size:
                continue
            if is_group:
                cnt = _safe_comb(g - 1, r - 1) * _safe_comb(n - g, s - r)
            else:
                cnt = _safe_comb(g, r) * _safe_comb(n - g - 1, s - r - 1)
            out += cnt * self._cell_rho[idx]
        return float(out)

    def _build_feature_map_and_readout(self):
        n = self.num_player
        self._interaction_blocks = []
        self._size_player_start = None
        self._log_col = None
        self._quad_col = None
        self._overlap_ratio_col = None

        next_col = 1
        if self.include_nonlinear_size_terms:
            self._log_col = next_col
            next_col += 1
            self._quad_col = next_col
            next_col += 1
        if self.include_group_overlap_ratio:
            self._overlap_ratio_col = next_col
            next_col += 1

        if self._surrogate_basis_kind == "interactions":
            degree = self._interaction_degree
            if degree > n:
                raise ValueError(f"`surrogate_basis={degree}` exceeds num_player={n}.")
            num_interactions = sum(math.comb(n, r) for r in range(1, degree + 1))
            self._feature_dim = next_col + num_interactions
        else:
            self._size_player_start = next_col
            self._feature_dim = next_col + n * n

        zeta = np.zeros(self._feature_dim, dtype=np.float64)
        zeta[0] = self._cell_readout(lambda _s, _r: 1.0)
        if self._log_col is not None:
            zeta[self._log_col] = self._cell_readout(lambda s, _r: np.log1p(float(s)))
            zeta[self._quad_col] = self._cell_readout(lambda s, _r: (float(s) / float(n)) ** 2)
        if self._overlap_ratio_col is not None:
            zeta[self._overlap_ratio_col] = self._cell_readout(
                lambda s, r: 0.0 if s == 0 else float(r) / float(s)
            )

        if self._surrogate_basis_kind == "interactions":
            start = next_col
            for degree in range(1, self._interaction_degree + 1):
                m = math.comb(n, degree)
                flat = np.fromiter(
                    itertools.chain.from_iterable(itertools.combinations(range(n), degree)),
                    dtype=np.int64,
                    count=m * degree,
                )
                combos = flat.reshape(m, degree)
                self._interaction_blocks.append((degree, combos, start))
                b_vals = self.group_mask[combos].sum(axis=1).astype(np.int64)
                cache = {}
                for b in np.unique(b_vals):
                    cache[int(b)] = self._interaction_readout(degree, int(b))
                zeta[start:start + m] = np.array([cache[int(b)] for b in b_vals], dtype=np.float64)
                start += m
        else:
            for size in range(1, n + 1):
                start = self._size_player_start + (size - 1) * n
                for player in range(n):
                    zeta[start + player] = self._size_player_readout(player, size)

        self._zeta = zeta

    def _phi_from_beta(self, beta):
        value = float(self._zeta @ beta)
        if len(self._boundary_X):
            value -= float(self._boundary_rho @ (self._boundary_Z @ beta))
        return value

    def _normalize_cell_law(self, factor):
        factor = np.maximum(np.asarray(factor, dtype=np.float64), 0.0)
        factor = np.where(self._sampling_cell_mask, factor, 0.0)
        total = float(np.dot(self._cell_count, factor))
        if total <= 0.0:
            factor = self._sampling_cell_mask.astype(np.float64)
            total = float(np.dot(self._cell_count, factor))
        if total <= 0.0:
            return np.zeros_like(factor)
        return factor / total

    def _mix_with_init_cell_mass(self, factor, mix_weight):
        factor = np.maximum(np.asarray(factor, dtype=np.float64), 0.0)
        factor = np.where(self._sampling_cell_mask, factor, 0.0)
        target_mass = self._cell_count * factor
        target_total = float(target_mass.sum())
        if target_total <= 0.0:
            return self._q_init_cell.copy()

        target_mass = target_mass / target_total
        init_mass = self._cell_count * self._q_init_cell
        init_total = float(init_mass.sum())
        if init_total > 0.0:
            init_mass = init_mass / init_total
            target_mass = (1.0 - mix_weight) * target_mass + mix_weight * init_mass

        q_cell = np.zeros_like(factor)
        valid = (self._cell_count > 0.0) & self._sampling_cell_mask
        q_cell[valid] = target_mass[valid] / self._cell_count[valid]
        return q_cell

    def _sample_batch(self, num_rows, q_cell):
        mass = self._cell_count * q_cell
        mass = mass / float(mass.sum())
        cell_ids = self._rng.choice(np.arange(len(self._cell_keys)), size=num_rows, p=mass)
        out = np.empty((num_rows, self.num_player + 1), dtype=np.float64)
        for row, cell_id in enumerate(cell_ids):
            s, r = self._cell_keys[int(cell_id)]
            subset = np.zeros(self.num_player, dtype=bool)
            if r > 0:
                subset[self._rng.choice(self.group, size=r, replace=False)] = True
            outside_count = s - r
            if outside_count > 0:
                subset[self._rng.choice(self.non_group, size=outside_count, replace=False)] = True
            out[row, :self.num_player] = subset.astype(np.float64)
            out[row, self.num_player] = q_cell[int(cell_id)]
        return out

    def sampling(self):
        remaining = self.num_sample
        pilot_remaining = min(self._pilot_num_sample, remaining)
        while pilot_remaining > 0:
            cur = min(self.batch_size, pilot_remaining)
            yield self._sample_batch(cur, self._q_init_cell)
            pilot_remaining -= cur
            remaining -= cur

        while remaining > 0:
            if not self._pilot_finalized:
                self._finalize_pilot_design()
            cur = min(self.batch_size, remaining)
            yield self._sample_batch(cur, self._q_stage2_cell)
            remaining -= cur

    def run(self, samples):
        game = self.game_func(**self.game_args)
        n = self.num_player
        results = np.empty((len(samples), n + 2), dtype=np.float64)
        results[:, :n] = samples[:, :n]
        results[:, n + 1] = samples[:, n]
        for idx in range(len(samples)):
            results[idx, n] = float(game.evaluate(samples[idx, :n].astype(bool)))
        return results

    def _build_feature_block(self, X):
        X = np.asarray(X, dtype=bool)
        sizes = X.sum(axis=1).astype(np.int64)
        overlaps = X[:, self.group].sum(axis=1).astype(np.int64)
        Z = np.zeros((len(X), self._feature_dim), dtype=np.float64)
        Z[:, 0] = 1.0

        if self._log_col is not None:
            Z[:, self._log_col] = np.log1p(sizes.astype(np.float64))
            Z[:, self._quad_col] = (sizes.astype(np.float64) / float(self.num_player)) ** 2
        if self._overlap_ratio_col is not None:
            nonzero = sizes > 0
            Z[nonzero, self._overlap_ratio_col] = overlaps[nonzero] / sizes[nonzero]

        if self._surrogate_basis_kind == "interactions":
            for degree, combos, start in self._interaction_blocks:
                width = combos.shape[0]
                if degree == 1:
                    Z[:, start:start + width] = X[:, combos[:, 0]].astype(np.float64)
                else:
                    Z[:, start:start + width] = X[:, combos].all(axis=2).astype(np.float64)
        else:
            X_float = X.astype(np.float64)
            n = self.num_player
            for size in range(1, n + 1):
                mask = sizes == size
                if np.any(mask):
                    start = self._size_player_start + (size - 1) * n
                    Z[mask, start:start + n] = X_float[mask]
        return Z, sizes, overlaps

    def _rho_values(self, sizes, overlaps):
        prev = np.zeros(len(sizes), dtype=np.float64)
        has_prev = sizes > 0
        prev[has_prev] = self.alpha[sizes[has_prev] - 1]
        cur = np.zeros(len(sizes), dtype=np.float64)
        has_cur = sizes < self.num_player
        cur[has_cur] = self.alpha[sizes[has_cur]]
        return overlaps * prev - (self.group_size - overlaps) * cur

    def _cell_ids(self, sizes, overlaps):
        return self._cell_index[sizes, overlaps]

    def _assign_folds(self, num_rows):
        return self._fold_rng.integers(self.num_folds, size=num_rows)

    def _effective_surrogate_ridge_lambda(self, count):
        if self.surrogate_ridge_schedule == "fixed":
            return self.surrogate_ridge_lambda
        return float(count) * self.surrogate_ridge_lambda

    def _fit_from_stats(self, A_stat, c_stat, B_stat, b_stat, count):
        if count <= 0:
            beta = np.zeros(self._feature_dim, dtype=np.float64)
            return beta, self._phi_from_beta(beta)

        gram = A_stat - np.outer(B_stat, B_stat) / float(count)
        rhs = c_stat - B_stat * (float(b_stat) / float(count))
        gram = gram.copy()
        gram[np.diag_indices_from(gram)] += self._effective_surrogate_ridge_lambda(count)
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        return beta, self._phi_from_beta(beta)

    def _append_block(self, results_collect):
        n = self.num_player
        X = results_collect[:, :n].astype(bool)
        y = results_collect[:, n].astype(np.float64)
        q = results_collect[:, n + 1].astype(np.float64)
        if np.any(q <= 0.0):
            raise ValueError("Encountered non-positive sampling probability.")

        Z, sizes, overlaps = self._build_feature_block(X)
        rho = self._rho_values(sizes, overlaps)
        gamma = rho / q
        weights = gamma * gamma
        folds = self._assign_folds(len(y))
        cell_ids = self._cell_ids(sizes, overlaps)
        if np.any(cell_ids < 0):
            raise ValueError("Encountered invalid (size, overlap) cell.")

        start = self._num_obs
        end = start + len(y)
        self._X_obs[start:end] = X
        self._y_obs[start:end] = y
        self._q_obs[start:end] = q
        self._size_obs[start:end] = sizes
        self._overlap_obs[start:end] = overlaps
        self._cell_obs[start:end] = cell_ids
        self._fold_obs[start:end] = folds
        self._num_obs = end

        self._A_total += Z.T @ (weights[:, None] * Z)
        self._c_total += Z.T @ (weights * y)
        self._B_total += gamma @ Z
        self._b_total += float(gamma @ y)
        self._total_count += len(y)

        for fold in np.unique(folds):
            mask = folds == fold
            Zk = Z[mask]
            yk = y[mask]
            wk = weights[mask]
            gk = gamma[mask]
            self._A_fold[fold] += Zk.T @ (wk[:, None] * Zk)
            self._c_fold[fold] += Zk.T @ (wk * yk)
            self._B_fold[fold] += gk @ Zk
            self._b_fold[fold] += float(gk @ yk)
            self._fold_count[fold] += int(mask.sum())

    def _finalize_pilot_design(self):
        if self._pilot_finalized:
            return
        pilot_count = min(self._pilot_num_sample, self._num_obs)
        if pilot_count <= 0:
            self._q_stage2_cell = self._q_init_cell.copy()
            self._pilot_finalized = True
            return

        beta, _ = self._fit_from_stats(
            self._A_total,
            self._c_total,
            self._B_total,
            self._b_total,
            self._total_count,
        )
        Z, _sizes, _overlaps = self._build_feature_block(self._X_obs[:pilot_count])
        resid = self._y_obs[:pilot_count] - Z @ beta
        rss = np.zeros(len(self._cell_keys), dtype=np.float64)
        counts = np.zeros(len(self._cell_keys), dtype=np.int64)
        np.add.at(rss, self._cell_obs[:pilot_count], resid * resid)
        np.add.at(counts, self._cell_obs[:pilot_count], 1)

        total_count = int(counts.sum())
        if total_count <= 0:
            moment = np.ones(len(self._cell_keys), dtype=np.float64)
        else:
            global_mse = max(float(rss.sum()) / float(total_count), 1e-12)
            moment = np.full(len(self._cell_keys), global_mse, dtype=np.float64)
            strong = counts >= self.stage2_min_cell_count
            moment[strong] = rss[strong] / counts[strong]
            moment = np.maximum(moment, 1e-12)

        factor = self._cell_rho_design_abs * np.sqrt(moment)
        if self.stage2_cell_floor > 0.0:
            self._q_stage2_cell = self._mix_with_init_cell_mass(factor, self.stage2_cell_floor)
        else:
            self._q_stage2_cell = self._normalize_cell_law(factor)
        self._pilot_finalized = True

    def _crossfit_estimate(self):
        m = self._num_obs
        if m <= 0:
            return self._boundary_exact

        est_sum = 0.0
        scored = 0
        for fold in np.unique(self._fold_obs[:m]):
            holdout_count = int(self._fold_count[fold])
            train_count = m - holdout_count
            if holdout_count <= 0 or train_count <= 0:
                continue

            beta, phi_h = self._fit_from_stats(
                self._A_total - self._A_fold[fold],
                self._c_total - self._c_fold[fold],
                self._B_total - self._B_fold[fold],
                self._b_total - self._b_fold[fold],
                train_count,
            )
            mask = self._fold_obs[:m] == fold
            X_hold = self._X_obs[:m][mask]
            y_hold = self._y_obs[:m][mask]
            q_hold = self._q_obs[:m][mask]
            Z_hold, sizes_hold, overlaps_hold = self._build_feature_block(X_hold)
            gamma_hold = self._rho_values(sizes_hold, overlaps_hold) / q_hold
            est_sum += float(holdout_count) * phi_h + float(gamma_hold @ (y_hold - Z_hold @ beta))
            scored += holdout_count

        if scored > 0:
            return float(self._boundary_exact + est_sum / float(scored))

        beta, phi_h = self._fit_from_stats(
            self._A_total,
            self._c_total,
            self._B_total,
            self._b_total,
            self._total_count,
        )
        X = self._X_obs[:m]
        y = self._y_obs[:m]
        q = self._q_obs[:m]
        Z, sizes, overlaps = self._build_feature_block(X)
        gamma = self._rho_values(sizes, overlaps) / q
        return float(self._boundary_exact + phi_h + (gamma @ (y - Z @ beta)) / float(m))

    def aggregate(self, results_collect):
        pos = 0
        while pos < len(results_collect):
            take = len(results_collect) - pos
            if self.pos_traj < len(self.values_traj):
                next_track = (self.pos_traj + 1) * self.interval_track
                to_track = max(next_track - (self._boundary_eval_count + self._num_obs), 0)
                if to_track > 0:
                    take = min(take, to_track)
            if take <= 0:
                break

            self._append_block(results_collect[pos:pos + take])
            pos += take

            if not self._pilot_finalized and self._num_obs >= self._pilot_num_sample:
                self._finalize_pilot_design()

            while (
                self.pos_traj < len(self.values_traj)
                and (self._boundary_eval_count + self._num_obs) >= (self.pos_traj + 1) * self.interval_track
            ):
                self._current_estimate = self._crossfit_estimate()
                self.values_traj[self.pos_traj] = self._current_estimate
                self.pos_traj += 1

    def finalize(self):
        if not self._pilot_finalized:
            self._finalize_pilot_design()
        self._current_estimate = self._crossfit_estimate()
        self._fill_traj_tail()
        return self._current_estimate, self.values_traj


class FGSVShapley(groupEstimatorTemplate):
    """Local implementation of the Shapley-only FGSV benchmark code."""

    def __init__(self, *, thres=5, **kwargs):
        super(FGSVShapley, self).__init__(**kwargs)
        if self.semivalue != "shapley":
            raise ValueError("FGSV is Shapley-only.")
        self.thres = int(thres)
        self.nue_per_proc_run = 1

    def _evaluate_indices(self, game, indices):
        subset = np.zeros(self.num_player, dtype=bool)
        if len(indices):
            subset[np.asarray(indices, dtype=np.int64)] = True
        return float(game.evaluate(subset))

    def run_all(self):
        n = self.num_player
        g = self.group_size
        rng = np.random.RandomState(self.estimator_seed)
        game = self.game_func(**self.game_args)

        u_full = self._evaluate_indices(game, np.arange(n))
        u_empty = self._evaluate_indices(game, np.array([], dtype=np.int64))
        linear_term = g / n * (u_full - u_empty)
        num_eval = 2

        Ts = np.zeros(n - 1, dtype=np.float64)
        counts = np.zeros(n - 1, dtype=np.float64)
        self._current_estimate = float(linear_term)
        self._record_tracks(num_eval)

        while num_eval < self.total_budget:
            s = int(rng.randint(1, n))
            s1_min = max(0, g + s - n)
            s1_max = min(g, s)
            Es1 = int(round(g * s / n))

            if s < self.thres:
                Ts_temp = 0.0
                for s1 in range(s1_min, s1_max + 1):
                    S1 = rng.choice(self.group, s1, replace=False)
                    S0c = rng.choice(self.non_group, s - s1, replace=False)
                    S = np.concatenate((S1, S0c))
                    Ts_temp += hypergeom.pmf(s1, n, g, s) * (s1 - s * g / n) * self._evaluate_indices(game, S)
                    num_eval += 1
                    if num_eval >= self.total_budget:
                        break
                Ts_temp *= n / float(s * (n - s))
                idx = s - 1
                Ts[idx] = (counts[idx] / (counts[idx] + 1.0)) * Ts[idx] + Ts_temp / (counts[idx] + 1.0)
                counts[idx] += 1.0
            else:
                central_ind = False
                if Es1 + 1 <= s1_max and Es1 - 1 >= s1_min:
                    central_ind = True
                    S1_temp = rng.choice(self.group, Es1 + 1, replace=False)
                    S0c_temp = rng.choice(self.non_group, s - Es1 + 1, replace=False)
                    S_upper = np.concatenate((S1_temp, S0c_temp[0:(s - Es1 - 1)]))
                    S_lower = np.concatenate((S1_temp[0:(Es1 - 1)], S0c_temp))
                elif Es1 + 1 <= s1_max:
                    S1_temp = rng.choice(self.group, Es1 + 1, replace=False)
                    S0c_temp = rng.choice(self.non_group, s - Es1, replace=False)
                    S_upper = np.concatenate((S1_temp, S0c_temp[0:(s - Es1 - 1)]))
                    S_lower = np.concatenate((S1_temp[0:Es1], S0c_temp))
                elif Es1 - 1 >= s1_min:
                    S1_temp = rng.choice(self.group, Es1, replace=False)
                    S0c_temp = rng.choice(self.non_group, s - Es1 + 1, replace=False)
                    S_upper = np.concatenate((S1_temp, S0c_temp[0:(s - Es1)]))
                    S_lower = np.concatenate((S1_temp[0:(Es1 - 1)], S0c_temp))
                else:
                    continue

                idx = s - 1
                diff = self._evaluate_indices(game, S_upper) - self._evaluate_indices(game, S_lower)
                Ts[idx] = (
                    (counts[idx] / (counts[idx] + 1.0)) * Ts[idx]
                    + g / n * (1.0 - g / n) * diff / (float(central_ind) + 1.0) / (counts[idx] + 1.0)
                )
                counts[idx] += 1.0
                num_eval += 2

            self._current_estimate = float(linear_term + np.sum(Ts))
            self._record_tracks(num_eval)

        self._fill_traj_tail()
        return self._current_estimate, self.values_traj


class IndividualSumWrapper(groupEstimatorTemplate):
    """Estimate the full semivalue vector with an existing estimator, then sum over G."""

    def __init__(self, *, base_estimator, individual_estimator_kwargs=None, **kwargs):
        super(IndividualSumWrapper, self).__init__(**kwargs)
        self.base_estimator = base_estimator
        self.individual_estimator_kwargs = dict(individual_estimator_kwargs or {})

    def run_all(self):
        estimator = runEstimator(
            estimator=self.base_estimator,
            n_process=self.n_process,
            semivalue=self.semivalue,
            semivalue_param=self.semivalue_param,
            game_func=self.game_func,
            game_args=self.game_args,
            num_player=self.num_player,
            nue_avg=self.nue_avg,
            nue_per_proc=self.nue_per_proc,
            nue_track_avg=self.nue_track_avg,
            estimator_seed=self.estimator_seed,
            **self.individual_estimator_kwargs,
        )
        values_final, values_traj = estimator.run()
        group_value = float(np.asarray(values_final)[self.group].sum())
        group_traj = np.asarray(values_traj)[:, self.group].sum(axis=1)
        self._current_estimate = group_value
        self.values_traj = group_traj.astype(np.float64, copy=False)
        return group_value, self.values_traj


class IndividualRegressionMSRUnbiased(IndividualSumWrapper):
    def __init__(self, **kwargs):
        super(IndividualRegressionMSRUnbiased, self).__init__(
            base_estimator="RegressionMSR_unbiased",
            **kwargs,
        )


class IndividualRegressionMSRUnpaired(IndividualSumWrapper):
    def __init__(self, **kwargs):
        individual_estimator_kwargs = dict(kwargs.pop("individual_estimator_kwargs", {}) or {})
        individual_estimator_kwargs["paired_sampling"] = False
        super(IndividualRegressionMSRUnpaired, self).__init__(
            base_estimator="RegressionMSR_unbiased",
            individual_estimator_kwargs=individual_estimator_kwargs,
            **kwargs,
        )


class IndividualOFAFixed(IndividualSumWrapper):
    def __init__(self, **kwargs):
        super(IndividualOFAFixed, self).__init__(
            base_estimator="OFA_fixed",
            **kwargs,
        )


_GROUP_ALGORITHM_MAP = {
    "EaseSHAP_group": EaseSHAP_group,
    "fgsv": FGSVShapley,
    "FGSV": FGSVShapley,
    "individual_regressionMSR_unbiased": IndividualRegressionMSRUnbiased,
    "individual_RegressionMSR_unbiased": IndividualRegressionMSRUnbiased,
    "individual_regressionMSR_unpaired": IndividualRegressionMSRUnpaired,
    "individual_RegressionMSR_unpaired": IndividualRegressionMSRUnpaired,
    "individual_ofa_fixed": IndividualOFAFixed,
    "individual_OFA_fixed": IndividualOFAFixed,
}
