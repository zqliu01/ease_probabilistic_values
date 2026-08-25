import numpy as np
from scipy.stats import hypergeom

from .group_core import (
    _safe_comb,
    exact_group_sum_value,
    groupEstimatorTemplate,
    group_sum_coefficient,
    semivalue_coefficients,
    validate_group,
)
from .ease import EaseSHAP_group
from .runner import runEstimator


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


class FGSVShapley(groupEstimatorTemplate):
    """Shapley-only FGSV benchmark estimator."""

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
