"""
Tests for estimators module
"""
import pytest
import numpy as np
from easeshap import (
    exact_value,
    exact_group_sum_value,
    runGroupSumEstimator,
)
from easeshap.group_estimators import (
    EaseSHAP_group,
    group_sum_coefficient,
    semivalue_coefficients,
)
from easeshap.estimators import RegressionMSR_unbiased


class TableUtility:
    def __init__(self, num_player, values):
        self.num_player = num_player
        self.values = np.asarray(values, dtype=float)

    def evaluate(self, subset):
        subset = np.asarray(subset, dtype=bool)
        idx = 0
        for i, bit in enumerate(subset):
            if bit:
                idx |= 1 << i
        return float(self.values[idx])


class TestExactValue:
    """Test exact value computation"""
    
    def test_initialization(self):
        """Test that exact_value initializes correctly"""
        # Add your test here
        pass


class TestRegressionMSR:
    def test_use_special_surrogates_flag_controls_shapley_branch(self):
        kwargs = dict(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": 5, "values": np.zeros(2 ** 5)},
            num_player=5,
            nue_avg=4,
            nue_per_proc=5,
            nue_track_avg=2,
            estimator_seed=1,
            sampling_with_replacement=True,
            paired_sampling=False,
        )

        default = RegressionMSR_unbiased(**kwargs)
        plain = RegressionMSR_unbiased(use_special_surrogates=False, **kwargs)

        assert default._is_leverage_shap
        assert not default._is_banzhaf
        assert not plain._is_leverage_shap
        assert not plain._is_banzhaf
    
    def test_shapley_computation(self):
        """Test Shapley value computation"""
        # Add your test here
        pass


class TestGroupSumEstimators:
    def test_group_sum_coefficients_match_individual_sum(self):
        n = 5
        group = np.array([0, 2, 4])
        group_mask = np.zeros(n, dtype=bool)
        group_mask[group] = True
        alpha = semivalue_coefficients(n, "beta_shapley", (1, 4))

        for idx in range(2 ** n):
            subset = np.array([(idx >> i) & 1 for i in range(n)], dtype=bool)
            s = int(subset.sum())
            r = int(np.logical_and(subset, group_mask).sum())
            rho_group = group_sum_coefficient(s, r, len(group), alpha)

            rho_individual = 0.0
            for player in group:
                if subset[player]:
                    rho_individual += alpha[s - 1] if s > 0 else 0.0
                else:
                    rho_individual -= alpha[s] if s < n else 0.0
            assert rho_group == pytest.approx(rho_individual)

    def test_exact_group_sum_value_matches_manual_coefficients(self):
        n = 4
        group = [0, 3]
        rng = np.random.default_rng(123)
        values = rng.normal(size=2 ** n)
        args = {"num_player": n, "values": values}

        got = exact_group_sum_value(
            TableUtility,
            args,
            n,
            "weighted_banzhaf",
            0.3,
            group,
        )

        alpha = semivalue_coefficients(n, "weighted_banzhaf", 0.3)
        group_mask = np.zeros(n, dtype=bool)
        group_mask[group] = True
        expected = 0.0
        for idx in range(2 ** n):
            subset = np.array([(idx >> i) & 1 for i in range(n)], dtype=bool)
            s = int(subset.sum())
            r = int(np.logical_and(subset, group_mask).sum())
            expected += group_sum_coefficient(s, r, len(group), alpha) * values[idx]
        assert got == pytest.approx(expected)

    def test_overlap_ratio_feature_switch_changes_feature_dimension(self):
        kwargs = dict(
            semivalue="shapley",
            semivalue_param=None,
            group=[0, 1],
            game_func=TableUtility,
            game_args={"num_player": 5, "values": np.zeros(2 ** 5)},
            num_player=5,
            nue_avg=4,
            nue_per_proc=5,
            nue_track_avg=2,
            estimator_seed=1,
            surrogate_basis=0,
            include_nonlinear_size_terms=False,
            pilot_fraction=0.5,
        )
        with_ratio = EaseSHAP_group(include_group_overlap_ratio=True, **kwargs)
        without_ratio = EaseSHAP_group(include_group_overlap_ratio=False, **kwargs)
        assert with_ratio._feature_dim == without_ratio._feature_dim + 1

    def test_two_stage_group_sum_smoke(self):
        n = 5
        rng = np.random.default_rng(321)
        values = rng.normal(size=2 ** n)
        final_value, traj = runGroupSumEstimator(
            estimator="EaseSHAP_group",
            n_process=1,
            semivalue="shapley",
            semivalue_param=None,
            group=[0, 2],
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=8,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=7,
            surrogate_basis=0,
            include_nonlinear_size_terms=False,
        ).run()
        assert np.isfinite(final_value)
        assert traj.ndim == 1
        assert np.all(np.isfinite(traj))
