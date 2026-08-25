"""Shared group-sum semivalue helpers and base estimator template."""

import itertools

import numpy as np
from scipy import special


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
