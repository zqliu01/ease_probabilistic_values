"""
Tests for estimator implementations
"""
import math

import pytest
import numpy as np
from easeshap import (
    exact_value,
    exact_group_sum_value,
    runEstimator,
    runGroupSumEstimator,
)
from easeshap.baselines import (
    LeverageSHAP,
    LeverageSHAP_border,
    RegressionMSR_unbiased,
    _sample_unique_indices,
)
from easeshap.ease import (
    EaseSHAP,
    EaseSHAP_group,
    _EmpiricalDenseStats,
    _ExactConditionalStats,
    _ExactCorrectedSolverBase,
    _ExactDenseConditionalSolver,
    _ExactDenseCorrectedSolver,
    _ExactFirstOrderInteractionSolver,
    _ExactMatrixFreeCorrectedSolver,
    _ExactSecondOrderSolver,
    _ExactSizePlayerDiagonalCorrectedSolver,
    _ExactSizePlayerSolver,
    _ExactSizePlayerStreamingSolver,
    _FeatureBuilder,
    _FullSemivalueTarget,
    _GroupCellStrata,
    _GroupSumTarget,
    _ObservationStore,
    _SizeStrata,
    _adaptive_boundary_sizes,
    _boundary_eval_count_for_sizes,
    _boundary_subset_matrix,
    _boundary_subset_matrix_for_sizes,
    _exact_conditional_solver_class,
    _exact_corrected_solver_class,
    _normalize_boundary_order,
    _normalize_boundary_sizes,
)
from easeshap.group_core import (
    group_sum_coefficient,
    semivalue_coefficients,
)


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


class CountSquaredUtility:
    def __init__(self, num_player):
        self.num_player = num_player

    def evaluate(self, subset):
        subset = np.asarray(subset, dtype=bool)[:self.num_player]
        count = float(subset.sum())
        return count + 0.1 * count * count


class TestExactValue:
    """Test exact value computation"""
    
    def test_initialization(self):
        """Test that exact_value initializes correctly"""
        # Add your test here
        pass


class TestRegressionMSR:
    def test_large_n_weighted_banzhaf_auto_pairing_uses_semivalue_symmetry(self):
        n = 80
        base_kwargs = dict(
            semivalue="weighted_banzhaf",
            game_func=CountSquaredUtility,
            game_args={"num_player": n},
            num_player=n,
            nue_avg=3,
            nue_per_proc=8,
            nue_track_avg=1,
            estimator_seed=1,
            sampling_with_replacement=True,
            paired_sampling=None,
        )

        low = RegressionMSR_unbiased(semivalue_param=0.25, **base_kwargs)
        high = RegressionMSR_unbiased(semivalue_param=0.75, **base_kwargs)
        half = RegressionMSR_unbiased(semivalue_param=0.5, **base_kwargs)

        assert not low._pair_sampling
        assert not high._pair_sampling
        assert half._pair_sampling

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

    def test_no_replacement_correction_uses_sampler_probabilities(self):
        n = 6
        estimator = RegressionMSR_unbiased(
            semivalue="shapley",
            semivalue_param=None,
            game_func=CountSquaredUtility,
            game_args={"num_player": n},
            num_player=n,
            nue_avg=8,
            nue_per_proc=16,
            nue_track_avg=4,
            estimator_seed=3,
            sampling_with_replacement=False,
            paired_sampling=False,
            num_folds=2,
        )
        list(estimator.sampling())
        sizes = estimator._sample_matrix.sum(axis=1).astype(int)

        correction_density = estimator._correction_density(
            sizes,
            estimator._sample_prob,
            estimator._sample_correction_density,
        )
        np.testing.assert_allclose(correction_density, estimator._sample_correction_density)

        saturated = estimator._sample_prob == 1.0
        assert saturated.any()
        replacement_density = estimator._subset_density_by_size(sizes)
        assert not np.allclose(correction_density[saturated], replacement_density[saturated])

    def test_no_replacement_fallback_keeps_per_row_correction_density(self):
        n = 6
        target = 24
        estimator = RegressionMSR_unbiased(
            semivalue="shapley",
            semivalue_param=None,
            game_func=CountSquaredUtility,
            game_args={"num_player": n},
            num_player=n,
            nue_avg=4,
            nue_per_proc=16,
            nue_track_avg=2,
            estimator_seed=5,
            sampling_with_replacement=False,
            paired_sampling=False,
            num_folds=2,
        )

        chosen_constant = None
        chosen_m_total = None
        chosen_norm = None
        binoms = np.array([
            float(math.comb(n, int(size)))
            for size in estimator._valid_sizes
        ])
        for candidate in np.geomspace(1e-4, 10.0, num=200):
            sample_dist_c = np.minimum(estimator._sample_dist * candidate, 1.0)
            m_total = int(np.sum([
                round(prob * binom)
                for prob, binom in zip(sample_dist_c, binoms)
            ]))
            if 0 < m_total < target:
                chosen_constant = float(candidate)
                chosen_m_total = m_total
                chosen_norm = float(np.sum(sample_dist_c * binoms))
                break

        assert chosen_constant is not None
        estimator._find_bernoulli_constant = lambda _: chosen_constant

        _, probs, densities = estimator._sample_without_replacement(target)

        assert len(probs) == target
        assert 0 < chosen_m_total < target
        np.testing.assert_allclose(
            densities[:chosen_m_total],
            probs[:chosen_m_total] / chosen_norm,
        )
        np.testing.assert_allclose(
            densities[chosen_m_total:],
            probs[chosen_m_total:] / estimator._subset_density_norm,
        )
        assert estimator._sample_prob_density_norm == chosen_norm
    
    def test_shapley_computation(self):
        """Test Shapley value computation"""
        # Add your test here
        pass


class TestLeverageSHAP:
    def _run_additive_estimator(self, estimator_cls=LeverageSHAP, **estimator_kwargs):
        weights = np.array([1.5, -0.25, 2.0, 0.75])
        values = np.empty(2 ** len(weights), dtype=float)
        for idx in range(len(values)):
            values[idx] = sum(weight for player, weight in enumerate(weights) if idx & (1 << player))

        estimator = estimator_cls(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": len(weights), "values": values},
            num_player=len(weights),
            nue_avg=20,
            nue_per_proc=8,
            nue_track_avg=10,
            estimator_seed=7,
            **estimator_kwargs,
        )
        for samples in estimator.sampling():
            estimator.aggregate(estimator.run(samples))
        estimates, trajectory = estimator.finalize()
        return weights, estimates, trajectory

    def test_original_leverageshap_recovers_additive_game(self):
        weights, estimates, trajectory = self._run_additive_estimator(sampling_with_replacement=False)

        np.testing.assert_allclose(estimates, weights, atol=1e-9)
        np.testing.assert_allclose(trajectory[-1], weights, atol=1e-9)

    def test_leverageshap_with_replacement_recovers_additive_game(self):
        weights, estimates, trajectory = self._run_additive_estimator(sampling_with_replacement=True)

        np.testing.assert_allclose(estimates, weights, atol=1e-9)
        np.testing.assert_allclose(trajectory[-1], weights, atol=1e-9)

    def test_leverageshap_border_recovers_additive_game(self):
        weights, estimates, trajectory = self._run_additive_estimator(
            estimator_cls=LeverageSHAP_border,
        )

        np.testing.assert_allclose(estimates, weights, atol=1e-9)
        np.testing.assert_allclose(trajectory[-1], weights, atol=1e-9)

    def test_leverageshap_border_enumerates_saturated_sizes(self):
        n = 6
        estimator = LeverageSHAP_border(
            semivalue="shapley",
            semivalue_param=None,
            game_func=CountSquaredUtility,
            game_args={"num_player": n},
            num_player=n,
            nue_avg=8,
            nue_per_proc=16,
            nue_track_avg=4,
            estimator_seed=23,
        )
        samples = np.vstack(list(estimator.sampling()))
        sizes = samples[:, :n].sum(axis=1).astype(int)

        assert np.count_nonzero(sizes == 1) == math.comb(n, 1)
        assert np.count_nonzero(sizes == n - 1) == math.comb(n, n - 1)
        assert len(samples) <= estimator.num_sample

    def test_without_replacement_subsampled_nonadditive_is_efficient(self):
        n = 10
        estimator = LeverageSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=CountSquaredUtility,
            game_args={"num_player": n},
            num_player=n,
            nue_avg=10,
            nue_per_proc=16,
            nue_track_avg=5,
            estimator_seed=11,
            sampling_with_replacement=False,
        )
        for samples in estimator.sampling():
            estimator.aggregate(estimator.run(samples))
        estimates, trajectory = estimator.finalize()

        v_empty = CountSquaredUtility(n).evaluate(np.zeros(n, dtype=bool))
        v_full = CountSquaredUtility(n).evaluate(np.ones(n, dtype=bool))
        assert np.isfinite(estimates).all()
        np.testing.assert_allclose(estimates.sum(), v_full - v_empty, atol=1e-9)
        np.testing.assert_allclose(trajectory[-1].sum(), v_full - v_empty, atol=1e-9)

    def test_without_replacement_sampling_handles_large_combination_counts(self):
        n = 40
        estimator = LeverageSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=CountSquaredUtility,
            game_args={"num_player": n},
            num_player=n,
            nue_avg=2,
            nue_per_proc=8,
            nue_track_avg=1,
            estimator_seed=13,
            sampling_with_replacement=False,
        )
        samples = next(estimator.sampling())

        assert samples.shape == (8, n + 1)
        assert np.isfinite(samples[:, -1]).all()

    def test_unique_index_sampler_handles_huge_population(self):
        rng = np.random.Generator(np.random.PCG64(19))
        population = math.comb(40, 20)
        indices = list(_sample_unique_indices(rng, population, 52))

        assert len(indices) == 52
        assert len(set(indices)) == 52
        assert min(indices) >= 0
        assert max(indices) < population

    def test_finalize_fills_tail_when_buffer_is_empty(self):
        n = 4
        estimator = LeverageSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=CountSquaredUtility,
            game_args={"num_player": n},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=1,
            estimator_seed=17,
            sampling_with_replacement=True,
        )
        samples = next(estimator.sampling())
        estimator.aggregate(estimator.run(samples))

        assert estimator.pos_buffer == 0
        assert estimator.pos_traj == 1
        estimates, trajectory = estimator.finalize()

        assert np.isfinite(estimates).all()
        expected_tail = np.repeat(estimates[None, :], len(trajectory) - 1, axis=0)
        np.testing.assert_allclose(trajectory[1:], expected_tail, atol=0.0)


class TestExactConditionalGram:
    def test_full_target_large_n_weighted_banzhaf_symmetry_is_parametric(self):
        n = 80
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=1,
            include_nonlinear_size_terms=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)

        def make_target(param):
            return _FullSemivalueTarget(
                n=n,
                semivalue="weighted_banzhaf",
                semivalue_param=param,
                feature_builder=feature_builder,
                boundary_X=boundary_X,
                boundary_context=boundary_context,
            )

        assert not make_target(0.25).is_symmetric
        assert not make_target(0.75).is_symmetric
        assert make_target(0.5).is_symmetric

    def _coalitions_matching(self, n, *, size=None, overlap=None, group=None):
        rows = []
        group = np.asarray([] if group is None else group, dtype=np.int64)
        for idx in range(2 ** n):
            subset = np.array([(idx >> i) & 1 for i in range(n)], dtype=bool)
            if size is not None and int(subset.sum()) != size:
                continue
            if overlap is not None and int(subset[group].sum()) != overlap:
                continue
            rows.append(subset)
        return np.asarray(rows, dtype=bool)

    def test_private_solver_mode_routes_dense_solver_and_memory_estimate(self):
        n = 4
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=1,
            include_nonlinear_size_terms=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="shapley",
            semivalue_param=None,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="dense",
        )

        assert isinstance(backend.solver, _ExactDenseConditionalSolver)
        assert _exact_corrected_solver_class("dense") is _ExactDenseCorrectedSolver
        assert issubclass(_ExactDenseCorrectedSolver, _ExactCorrectedSolverBase)
        assert issubclass(_ExactMatrixFreeCorrectedSolver, _ExactCorrectedSolverBase)
        assert backend.estimate_memory_bytes() == _ExactConditionalStats.estimate_memory_bytes_for(
            feature_dim=feature_builder.dim,
            output_dim=target.output_dim,
            num_folds=2,
            num_strata=len(strata.keys),
            has_group_counts=False,
            solver_mode="dense",
        )

        assert _exact_conditional_solver_class("first_order") is _ExactFirstOrderInteractionSolver
        first_order_backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="first-order",
        )
        assert isinstance(first_order_backend.solver, _ExactFirstOrderInteractionSolver)
        assert first_order_backend._union_sizes is None
        assert _ExactConditionalStats.estimate_memory_bytes_for(
            feature_dim=feature_builder.dim,
            output_dim=target.output_dim,
            num_folds=2,
            num_strata=len(strata.keys),
            has_group_counts=False,
            solver_mode="first_order",
        ) < backend.estimate_memory_bytes()
        assert _ExactFirstOrderInteractionSolver.estimate_memory_bytes_for(
            feature_dim=feature_builder.dim,
            output_dim=target.output_dim,
        ) > 0

        feature_builder_2 = _FeatureBuilder(
            n=n,
            surrogate_basis=2,
            include_nonlinear_size_terms=True,
        )
        target_2 = _FullSemivalueTarget(
            n=n,
            semivalue="shapley",
            semivalue_param=None,
            feature_builder=feature_builder_2,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        assert _exact_conditional_solver_class("second_order") is _ExactSecondOrderSolver
        second_order_backend = _ExactConditionalStats(
            target=target_2,
            strata=strata,
            feature_builder=feature_builder_2,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="second-order",
        )
        assert isinstance(second_order_backend.solver, _ExactSecondOrderSolver)
        assert second_order_backend._union_sizes is None
        assert _ExactSecondOrderSolver.estimate_memory_bytes_for(
            feature_dim=feature_builder_2.dim,
            output_dim=target_2.output_dim,
        ) > 0

        feature_builder_sp = _FeatureBuilder(
            n=n,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
        )
        target_sp = _FullSemivalueTarget(
            n=n,
            semivalue="shapley",
            semivalue_param=None,
            feature_builder=feature_builder_sp,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        assert _exact_conditional_solver_class("size_player") is _ExactSizePlayerSolver
        size_player_backend = _ExactConditionalStats(
            target=target_sp,
            strata=strata,
            feature_builder=feature_builder_sp,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="size-player",
        )
        assert isinstance(size_player_backend.solver, _ExactSizePlayerSolver)
        assert size_player_backend._union_sizes is None
        assert _ExactSizePlayerSolver.estimate_memory_bytes_for(
            feature_dim=feature_builder_sp.dim,
            output_dim=target_sp.output_dim,
        ) >= 16 * feature_builder_sp.dim * target_sp.output_dim
        assert _exact_conditional_solver_class("size_player_streaming") is _ExactSizePlayerStreamingSolver
        streaming_size_player_backend = _ExactConditionalStats(
            target=target_sp,
            strata=strata,
            feature_builder=feature_builder_sp,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="size-player-streaming",
        )
        assert isinstance(streaming_size_player_backend.solver, _ExactSizePlayerStreamingSolver)
        assert streaming_size_player_backend._union_sizes is None
        assert _ExactSizePlayerStreamingSolver.estimate_memory_bytes_for(
            feature_dim=feature_builder_sp.dim,
            output_dim=target_sp.output_dim,
        ) < _ExactSizePlayerSolver.estimate_memory_bytes_for(
            feature_dim=feature_builder_sp.dim,
            output_dim=target_sp.output_dim,
        )

        for mode in ["auto", "matrix_free", "cg", "woodbury_cg"]:
            assert _exact_corrected_solver_class(mode) is _ExactMatrixFreeCorrectedSolver
        for mode in ["size_player_diagonal", "diag_r_empirical_u", "diagonal_r_empirical_u"]:
            assert _exact_corrected_solver_class(mode) is _ExactSizePlayerDiagonalCorrectedSolver

        matrix_free_first_order_backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="first_order",
            r_correction_alpha=0.5,
            u_correction_alpha=0.5,
            correction_solver_mode="cg",
        )
        assert isinstance(matrix_free_first_order_backend.correction_solver, _ExactMatrixFreeCorrectedSolver)
        assert matrix_free_first_order_backend._union_sizes is None
        assert matrix_free_first_order_backend.U_emp_total is None
        assert (
            matrix_free_first_order_backend.correction_solver._default_max_iter(
                matrix_free_first_order_backend
            )
            == 5
        )

        matrix_free_size_player_backend = _ExactConditionalStats(
            target=target_sp,
            strata=strata,
            feature_builder=feature_builder_sp,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="size_player",
            r_correction_alpha=0.5,
            u_correction_alpha=0.5,
            correction_solver_mode="matrix_free",
            correction_max_iter=7,
        )
        assert isinstance(matrix_free_size_player_backend.correction_solver, _ExactMatrixFreeCorrectedSolver)
        assert matrix_free_size_player_backend._union_sizes is None
        assert matrix_free_size_player_backend.U_emp_total is None
        assert (
            matrix_free_size_player_backend.correction_solver._default_max_iter(
                matrix_free_size_player_backend
            )
            == 7
        )

        matrix_free_size_player_default_backend = _ExactConditionalStats(
            target=target_sp,
            strata=strata,
            feature_builder=feature_builder_sp,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="size_player",
            r_correction_alpha=0.5,
            u_correction_alpha=0.5,
            correction_solver_mode="matrix_free",
        )
        assert matrix_free_size_player_default_backend.correction_solver._default_max_iter(
            matrix_free_size_player_default_backend
        ) == 20

        diagonal_size_player_backend = _ExactConditionalStats(
            target=target_sp,
            strata=strata,
            feature_builder=feature_builder_sp,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="size_player",
            r_correction_alpha=0.5,
            u_correction_alpha=1.0,
            correction_solver_mode="diag_r_empirical_u",
        )
        assert isinstance(diagonal_size_player_backend.correction_solver, _ExactSizePlayerDiagonalCorrectedSolver)
        assert diagonal_size_player_backend._union_sizes is None
        assert diagonal_size_player_backend.U_emp_total is not None

        with pytest.raises(ValueError, match="structured exact solver"):
            _ExactConditionalStats(
                target=target,
                strata=strata,
                feature_builder=feature_builder,
                ridge_lambda=0.0,
                ridge_schedule="fixed",
                num_folds=2,
                r_correction_alpha=0.5,
                u_correction_alpha=0.5,
                correction_solver_mode="matrix_free",
            )

        with pytest.raises(ValueError, match="correction"):
            _ExactConditionalStats(
                target=target,
                strata=strata,
                feature_builder=feature_builder,
                ridge_lambda=0.0,
                ridge_schedule="fixed",
                num_folds=2,
                solver_mode="first_order",
                r_correction_alpha=0.5,
            )

        with pytest.raises(ValueError, match="correction"):
            _ExactConditionalStats(
                target=target_2,
                strata=strata,
                feature_builder=feature_builder_2,
                ridge_lambda=0.0,
                ridge_schedule="fixed",
                num_folds=2,
                solver_mode="second_order",
                u_correction_alpha=0.5,
            )

        with pytest.raises(ValueError, match="correction"):
            _ExactConditionalStats(
                target=target_sp,
                strata=strata,
                feature_builder=feature_builder_sp,
                ridge_lambda=0.0,
                ridge_schedule="fixed",
                num_folds=2,
                solver_mode="size_player",
                r_correction_alpha=0.5,
            )

        backend_without_correction = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
            r_correction_alpha=0.0,
            correction_solver_mode="matrix_free",
        )
        assert backend_without_correction.correction_solver is None

    @pytest.mark.parametrize("include_nonlinear_size_terms", [False, True])
    def test_first_order_structured_solver_matches_dense_exact_solver(self, include_nonlinear_size_terms):
        n = 5
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=1,
            include_nonlinear_size_terms=include_nonlinear_size_terms,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.4,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )

        def make_backend(mode):
            return _ExactConditionalStats(
                target=target,
                strata=strata,
                feature_builder=feature_builder,
                ridge_lambda=0.17,
                ridge_schedule="fixed",
                num_folds=3,
                solver_mode=mode,
            )

        dense = make_backend("dense")
        structured = make_backend("first_order")
        assert dense._union_sizes is not None
        assert structured._union_sizes is None

        X = self._coalitions_matching(n)
        sizes = X.sum(axis=1)
        q_by_size = np.array([0.19, 0.07, 0.05, 0.09, 0.11, 0.13], dtype=np.float64)
        q = q_by_size[sizes]
        y = np.sin(np.arange(len(X), dtype=np.float64) / 3.0) + 0.05 * sizes
        folds = np.arange(len(X)) % 3

        dense.append(X, y, q, folds)
        structured.append(X, y, q, folds)

        structured_all = structured.fit_all()
        dense_all = dense.fit_all()
        assert structured._union_sizes is None

        np.testing.assert_allclose(structured_all.beta, dense_all.beta, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            structured_all.phi,
            dense_all.phi,
            rtol=1e-10,
            atol=1e-10,
        )
        for fold in range(3):
            np.testing.assert_allclose(
                structured.fit_excluding_fold(fold).beta,
                dense.fit_excluding_fold(fold).beta,
                rtol=1e-10,
                atol=1e-10,
            )

    @pytest.mark.parametrize("n", [4, 5, 8])
    @pytest.mark.parametrize("include_nonlinear_size_terms", [False, True])
    def test_second_order_structured_solver_matches_dense_exact_solver(self, n, include_nonlinear_size_terms):
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=2,
            include_nonlinear_size_terms=include_nonlinear_size_terms,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.35,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )

        def make_backend(mode):
            return _ExactConditionalStats(
                target=target,
                strata=strata,
                feature_builder=feature_builder,
                ridge_lambda=0.23,
                ridge_schedule="fixed",
                num_folds=3,
                solver_mode=mode,
            )

        dense = make_backend("dense")
        structured = make_backend("second_order")
        assert dense._union_sizes is not None
        assert structured._union_sizes is None

        X = self._coalitions_matching(n)
        sizes = X.sum(axis=1)
        q_by_size = 0.045 + 0.015 * np.arange(n + 1, dtype=np.float64)
        q_by_size[::2] += 0.01
        q = q_by_size[sizes]
        y = np.cos(np.arange(len(X), dtype=np.float64) / 4.0) - 0.03 * sizes
        folds = (2 * np.arange(len(X)) + 1) % 3

        dense.append(X, y, q, folds)
        structured.append(X, y, q, folds)

        structured_all = structured.fit_all()
        dense_all = dense.fit_all()
        assert structured._union_sizes is None

        np.testing.assert_allclose(structured_all.beta, dense_all.beta, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(structured_all.phi, dense_all.phi, rtol=1e-10, atol=1e-10)
        for fold in range(3):
            np.testing.assert_allclose(
                structured.fit_excluding_fold(fold).beta,
                dense.fit_excluding_fold(fold).beta,
                rtol=1e-10,
                atol=1e-10,
            )

    @pytest.mark.parametrize("include_nonlinear_size_terms", [False, True])
    @pytest.mark.parametrize("ridge_scaling", ["scalar", "size_trace"])
    @pytest.mark.parametrize("boundary_like", [False, True])
    @pytest.mark.parametrize("solver_mode", ["size_player", "size_player_streaming"])
    def test_size_player_structured_solver_matches_dense_exact_solver(
        self,
        include_nonlinear_size_terms,
        ridge_scaling,
        boundary_like,
        solver_mode,
    ):
        n = 6
        sampling_mask = np.ones(n + 1, dtype=bool)
        if boundary_like:
            sampling_mask[:] = False
            sampling_mask[2:n - 1] = True
        strata = _SizeStrata(n, sampling_mask)
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=include_nonlinear_size_terms,
        )
        boundary_X = _boundary_subset_matrix(n) if boundary_like else np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.4,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )

        def make_backend(mode):
            return _ExactConditionalStats(
                target=target,
                strata=strata,
                feature_builder=feature_builder,
                ridge_lambda=0.31,
                ridge_schedule="fixed",
                ridge_scaling=ridge_scaling,
                num_folds=3,
                solver_mode=mode,
            )

        dense = make_backend("dense")
        structured = make_backend(solver_mode)
        materialized = (
            make_backend("size_player")
            if solver_mode == "size_player_streaming"
            else None
        )
        assert dense._union_sizes is not None
        assert structured._union_sizes is None

        if boundary_like:
            X = np.vstack([self._coalitions_matching(n, size=size) for size in range(2, n - 1)])
        else:
            X = self._coalitions_matching(n)
        sizes = X.sum(axis=1)
        q_by_size = 0.04 + 0.012 * np.arange(n + 1, dtype=np.float64)
        q_by_size[::2] += 0.015
        q = q_by_size[sizes]
        y = np.sin(np.arange(len(X), dtype=np.float64) / 5.0) + 0.02 * sizes
        folds = (np.arange(len(X)) + sizes) % 3

        dense.append(X, y, q, folds)
        structured.append(X, y, q, folds)
        if materialized is not None:
            materialized.append(X, y, q, folds)

        structured_all = structured.fit_all()
        dense_all = dense.fit_all()
        assert structured._union_sizes is None

        np.testing.assert_allclose(structured_all.beta, dense_all.beta, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(structured_all.phi, dense_all.phi, rtol=1e-8, atol=1e-8)
        if materialized is not None:
            materialized_all = materialized.fit_all()
            np.testing.assert_allclose(
                structured_all.beta,
                materialized_all.beta,
                rtol=1e-10,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                structured_all.phi,
                materialized_all.phi,
                rtol=1e-10,
                atol=1e-10,
            )
        for fold in range(3):
            np.testing.assert_allclose(
                structured.fit_excluding_fold(fold).beta,
                dense.fit_excluding_fold(fold).beta,
                rtol=1e-8,
                atol=1e-8,
            )
            if materialized is not None:
                np.testing.assert_allclose(
                    structured.fit_excluding_fold(fold).beta,
                    materialized.fit_excluding_fold(fold).beta,
                    rtol=1e-10,
                    atol=1e-10,
                )

    @pytest.mark.parametrize(
        "ridge_schedule,global_ridge,block_scale",
        [
            ("fixed", 0.5, 0.5 / 9.0),
            ("times_m", 0.5 * 9.0, 0.5),
        ],
    )
    def test_size_trace_ridge_diagonal_uses_size_player_block_trace_scale(
        self,
        ridge_schedule,
        global_ridge,
        block_scale,
    ):
        n = 4
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="shapley",
            semivalue_param=None,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.5,
            ridge_schedule=ridge_schedule,
            ridge_scaling="size_trace",
            num_folds=2,
        )

        R_factor = np.array([11.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float64)
        ridge = backend._ridge_penalty(R_factor, count=9)

        expected = np.full(feature_builder.dim, global_ridge, dtype=np.float64)
        for size in range(1, n + 1):
            start = feature_builder.size_player_start + (size - 1) * n
            expected[start:start + n] = block_scale * R_factor[size] * float(size) / float(n)
        np.testing.assert_allclose(ridge, expected, rtol=1e-12, atol=1e-12)

        X = self._coalitions_matching(n)
        y = np.linspace(-1.0, 1.0, len(X))
        q = np.full(len(X), 0.1, dtype=np.float64)
        folds = np.arange(len(X)) % 2
        backend.append(X, y, q, folds)
        fitted = backend.fit_all()
        assert np.all(np.isfinite(fitted.beta))
        assert np.all(np.isfinite(fitted.phi))

    @pytest.mark.parametrize("include_overlap_ratio", [False, True])
    def test_size_player_predict_from_rows_matches_dense_design(self, include_overlap_ratio):
        n = 5
        X = self._coalitions_matching(n)
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
            include_group_overlap_ratio=include_overlap_ratio,
        )
        if include_overlap_ratio:
            group = np.array([0, 2, 4], dtype=np.int64)
            group_mask = np.zeros(n, dtype=bool)
            group_mask[group] = True
            strata = _GroupCellStrata.build(
                n,
                group,
                group_mask,
                exact_boundary_handling=False,
            )
        else:
            strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        context = strata.context_from_X(X)
        beta = np.random.default_rng(123).normal(size=feature_builder.dim)

        dense = feature_builder.build(X, context) @ beta
        direct = feature_builder.predict_from_rows(beta, X, context)

        np.testing.assert_allclose(direct, dense, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("degree", [0, 1, 2, 3])
    @pytest.mark.parametrize("include_overlap_ratio", [False, True])
    def test_interaction_predict_from_rows_matches_dense_design(self, degree, include_overlap_ratio):
        n = 5
        X = self._coalitions_matching(n)
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=degree,
            include_nonlinear_size_terms=True,
            include_group_overlap_ratio=include_overlap_ratio,
        )
        if include_overlap_ratio:
            group = np.array([0, 2, 4], dtype=np.int64)
            group_mask = np.zeros(n, dtype=bool)
            group_mask[group] = True
            strata = _GroupCellStrata.build(
                n,
                group,
                group_mask,
                exact_boundary_handling=False,
            )
        else:
            strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        context = strata.context_from_X(X)
        beta = np.random.default_rng(321 + degree).normal(size=feature_builder.dim)

        dense = feature_builder.build(X, context) @ beta
        direct = feature_builder.predict_from_rows(beta, X, context)

        np.testing.assert_allclose(direct, dense, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "alpha_r,alpha_u",
        [
            (0.0, 0.5),
            (0.0, 1.0),
            (0.25, 0.5),
            (0.5, 0.25),
            (1.0, 1.0),
        ],
    )
    def test_dense_corrected_solver_matches_bruteforce_empirical_correction_system(self, alpha_r, alpha_u):
        n = 4
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.35,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.25,
            ridge_schedule="fixed",
            num_folds=2,
            r_correction_alpha=alpha_r,
            u_correction_alpha=alpha_u,
            correction_solver_mode="dense",
        )

        X = self._coalitions_matching(n)
        y = np.linspace(-1.0, 1.0, len(X))
        q_by_size = np.array([0.11, 0.07, 0.05, 0.09, 0.13], dtype=np.float64)
        q = q_by_size[X.sum(axis=1)]
        folds = np.arange(len(X)) % 2
        store = _ObservationStore(num_sample=len(X), n=n)
        store.append(X, y, q, folds)
        backend.append(X, y, q, folds)

        context = strata.context_from_X(X)
        Z = feature_builder.build(X, context)
        weights = target.true_stratum_weight(strata)[context.sizes] / (q ** 2)
        R_emp = Z.T @ (weights[:, None] * Z)
        R_exact = backend._build_exact_R(backend.R_factor_total)
        A_stat = R_emp if alpha_r == 1.0 else R_exact + alpha_r * (R_emp - R_exact)
        gamma = target.raw_gamma(X, context) / q[:, None]
        U_emp = gamma.T @ Z
        U_exact = backend._build_exact_U(backend.U_factor_total)
        B_stat = U_emp if alpha_u == 1.0 else U_exact + alpha_u * (U_emp - U_exact)
        gram = A_stat - (B_stat.T @ B_stat) / float(len(X))
        gram[np.diag_indices_from(gram)] += 0.25
        rhs = backend.c_total - (B_stat.T @ backend.b_total) / float(len(X))
        expected_beta = np.linalg.solve(gram, rhs)

        fitted = backend.fit_all(store=store if alpha_r > 0.0 else None)
        np.testing.assert_allclose(fitted.beta, expected_beta, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(backend.U_emp_total, U_emp, rtol=1e-12, atol=1e-12)
        assert isinstance(backend.correction_solver, _ExactDenseCorrectedSolver)

        if alpha_r > 0.0:
            with pytest.raises(ValueError, match="observation store"):
                backend.fit_all()

    def test_dense_corrected_solver_respects_excluded_fold(self):
        n = 4
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="shapley",
            semivalue_param=None,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.5,
            ridge_schedule="fixed",
            num_folds=2,
            r_correction_alpha=1.0,
            u_correction_alpha=1.0,
        )

        X = self._coalitions_matching(n)
        y = np.linspace(0.75, -0.25, len(X))
        q_by_size = np.array([0.13, 0.08, 0.06, 0.07, 0.12], dtype=np.float64)
        q = q_by_size[X.sum(axis=1)]
        folds = np.arange(len(X)) % 2
        store = _ObservationStore(num_sample=len(X), n=n)
        store.append(X, y, q, folds)
        backend.append(X, y, q, folds)

        train_mask = folds != 1
        X_train = X[train_mask]
        q_train = q[train_mask]
        context_train = strata.context_from_X(X_train)
        Z_train = feature_builder.build(X_train, context_train)
        weights = target.true_stratum_weight(strata)[context_train.sizes] / (q_train ** 2)
        R_emp_train = Z_train.T @ (weights[:, None] * Z_train)
        gamma_train = target.raw_gamma(X_train, context_train) / q_train[:, None]
        U_emp_train = gamma_train.T @ Z_train

        _R_factor, _U_factor, count = backend._design_counts(excluding_fold=1)
        c_stat = backend.c_total - backend.c_fold[1]
        b_stat = backend.b_total - backend.b_fold[1]
        gram = R_emp_train - (U_emp_train.T @ U_emp_train) / float(count)
        gram[np.diag_indices_from(gram)] += 0.5
        rhs = c_stat - (U_emp_train.T @ b_stat) / float(count)
        expected_beta = np.linalg.solve(gram, rhs)

        fitted = backend.fit_excluding_fold(1, store=store)
        np.testing.assert_allclose(fitted.beta, expected_beta, rtol=1e-11, atol=1e-11)

    def test_dense_corrected_solver_with_full_empirical_correction_matches_empirical_dense_backend(self):
        n = 4
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=1,
            include_nonlinear_size_terms=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.4,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        exact_backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.1,
            ridge_schedule="fixed",
            num_folds=2,
            r_correction_alpha=1.0,
            u_correction_alpha=1.0,
            correction_solver_mode="dense",
        )
        empirical_backend = _EmpiricalDenseStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.1,
            ridge_schedule="fixed",
            num_folds=2,
        )

        X = self._coalitions_matching(n)
        y = np.sin(np.arange(len(X), dtype=np.float64))
        q_by_size = np.array([0.12, 0.08, 0.05, 0.07, 0.11], dtype=np.float64)
        q = q_by_size[X.sum(axis=1)]
        folds = np.arange(len(X)) % 2
        store = _ObservationStore(num_sample=len(X), n=n)
        store.append(X, y, q, folds)
        exact_backend.append(X, y, q, folds)
        empirical_backend.append(X, y, q, folds)

        exact_fit = exact_backend.fit_all(store=store)
        empirical_fit = empirical_backend.fit_all()
        np.testing.assert_allclose(exact_fit.beta, empirical_fit.beta, rtol=1e-11, atol=1e-11)

        exact_fold_fit = exact_backend.fit_excluding_fold(1, store=store)
        empirical_fold_fit = empirical_backend.fit_excluding_fold(1)
        np.testing.assert_allclose(exact_fold_fit.beta, empirical_fold_fit.beta, rtol=1e-11, atol=1e-11)

    def test_empirical_dense_candidate_fit_matches_direct_profiled_system(self):
        n = 4
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=1,
            include_nonlinear_size_terms=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.4,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=strata.context_from_X(boundary_X),
        )
        ridge = 0.35
        backend = _EmpiricalDenseStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=ridge,
            ridge_schedule="fixed",
            num_folds=2,
        )

        X = self._coalitions_matching(n)
        sizes = X.sum(axis=1).astype(np.int64)
        y = np.sin(np.arange(len(X), dtype=np.float64) / 2.7) + 0.04 * sizes
        q0_by_size = np.array([0.13, 0.07, 0.045, 0.09, 0.16], dtype=np.float64)
        q0_by_size /= np.dot(strata.counts, q0_by_size)
        q_by_size = np.array([0.08, 0.11, 0.055, 0.06, 0.19], dtype=np.float64)
        q_by_size /= np.dot(strata.counts, q_by_size)
        q0 = q0_by_size[sizes]
        q_candidate = q_by_size[sizes]

        fitted = backend.fit_candidate(X, y, q0, q_candidate)

        context = strata.context_from_X(X)
        ids = strata.ids_from_context(context)
        Z = feature_builder.build(X, context)
        R_weights = target.true_stratum_weight(strata)[ids] / (q0 * q_candidate)
        gamma0 = target.raw_gamma(X, context) / q0[:, None]
        R = Z.T @ (R_weights[:, None] * Z)
        d = Z.T @ (R_weights * y)
        U = gamma0.T @ Z
        v = gamma0.T @ y
        profile_norm = float(np.sum(q_candidate / q0))
        assert not np.isclose(profile_norm, len(X))
        gram = R - (U.T @ U) / profile_norm
        gram[np.diag_indices_from(gram)] += ridge
        rhs = d - (U.T @ v) / profile_norm
        expected_beta = np.linalg.solve(gram, rhs)

        np.testing.assert_allclose(fitted.beta, expected_beta, rtol=1e-11, atol=1e-11)

        folds = np.arange(len(X)) % 2
        backend.append(X, y, q0, folds)
        current_fit = backend.fit_all()
        candidate_at_q0 = backend.fit_candidate(X, y, q0, q0)
        np.testing.assert_allclose(candidate_at_q0.beta, current_fit.beta, rtol=1e-11, atol=1e-11)

    def test_size_player_candidate_matrix_free_fit_matches_empirical_dense(self):
        n = 5
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.4,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=strata.context_from_X(boundary_X),
        )

        dense = _EmpiricalDenseStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.4,
            ridge_schedule="fixed",
            num_folds=3,
        )
        matrix_free = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.4,
            ridge_schedule="fixed",
            ridge_scaling="scalar",
            num_folds=3,
            solver_mode="size_player",
            r_correction_alpha=1.0,
            u_correction_alpha=1.0,
            correction_solver_mode="matrix_free",
            correction_max_iter=400,
            correction_tol=1e-12,
        )

        X = self._coalitions_matching(n)
        sizes = X.sum(axis=1).astype(np.int64)
        y = np.cos(np.arange(len(X), dtype=np.float64) / 3.1) + 0.05 * sizes
        q0_by_size = 0.04 + 0.009 * np.arange(n + 1, dtype=np.float64)
        q0_by_size[1::2] += 0.008
        q0_by_size /= np.dot(strata.counts, q0_by_size)
        q_by_size = 0.035 + 0.012 * np.arange(n + 1, dtype=np.float64)[::-1]
        q_by_size[::2] += 0.006
        q_by_size /= np.dot(strata.counts, q_by_size)
        q0 = q0_by_size[sizes]
        q_candidate = q_by_size[sizes]

        expected = dense.fit_candidate(X, y, q0, q_candidate)
        fitted = matrix_free.fit_candidate(X, y, q0, q_candidate)

        assert isinstance(matrix_free.correction_solver, _ExactMatrixFreeCorrectedSolver)
        assert matrix_free._union_sizes is None
        np.testing.assert_allclose(fitted.beta, expected.beta, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(fitted.phi, expected.phi, rtol=1e-7, atol=1e-7)

    def test_dense_corrected_size_player_fast_path_matches_dense_design(self):
        n = 5
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.35,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=3,
            r_correction_alpha=1.0,
        )

        X = self._coalitions_matching(n)
        y = np.linspace(-0.25, 1.0, len(X))
        sizes = X.sum(axis=1)
        q = 0.04 + 0.01 * sizes
        folds = np.arange(len(X)) % 3
        store = _ObservationStore(num_sample=len(X), n=n)
        store.append(X, y, q, folds)
        backend.append(X, y, q, folds)

        def dense_empirical_R(X_cur, q_cur):
            context = strata.context_from_X(X_cur)
            Z = feature_builder.build(X_cur, context)
            ids = strata.ids_from_context(context)
            weights = target.true_stratum_weight(strata)[ids] / (q_cur ** 2)
            return Z.T @ (weights[:, None] * Z)

        solver = backend.correction_solver
        got_all = solver._build_empirical_R(stats=backend, store=store)
        np.testing.assert_allclose(got_all, dense_empirical_R(X, q), rtol=1e-12, atol=1e-12)

        train_mask = folds != 1
        got_train = solver._build_empirical_R(stats=backend, store=store, excluding_fold=1)
        np.testing.assert_allclose(
            got_train,
            dense_empirical_R(X[train_mask], q[train_mask]),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "alpha_r,alpha_u,ridge_scaling",
        [
            (0.0, 1.0, "size_trace"),
            (0.65, 1.0, "scalar"),
            (0.8, 0.35, "scalar"),
        ],
    )
    def test_size_player_diagonal_corrected_solver_matches_dense_calibrated_system(
        self,
        alpha_r,
        alpha_u,
        ridge_scaling,
    ):
        n = 5
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.37,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )

        def make_backend():
            return _ExactConditionalStats(
                target=target,
                strata=strata,
                feature_builder=feature_builder,
                ridge_lambda=0.45,
                ridge_schedule="fixed",
                ridge_scaling=ridge_scaling,
                num_folds=3,
                solver_mode="size_player",
                r_correction_alpha=alpha_r,
                u_correction_alpha=alpha_u,
                correction_solver_mode="diagonal_r_empirical_u",
            )

        backend = make_backend()
        reference = make_backend()
        assert isinstance(backend.correction_solver, _ExactSizePlayerDiagonalCorrectedSolver)
        assert backend._union_sizes is None

        X = self._coalitions_matching(n)
        y = np.sin(np.arange(len(X), dtype=np.float64) / 4.0) + 0.07 * X.sum(axis=1)
        q_by_size = np.array([0.12, 0.075, 0.052, 0.083, 0.11, 0.16], dtype=np.float64)
        q = q_by_size[X.sum(axis=1)]
        folds = (np.arange(len(X)) + X.sum(axis=1)) % 3
        store = _ObservationStore(num_sample=len(X), n=n)
        store.append(X, y, q, folds)
        backend.append(X, y, q, folds)
        reference.append(X, y, q, folds)

        def empirical_R_diag(X_cur, q_cur):
            context = strata.context_from_X(X_cur)
            Z = feature_builder.build(X_cur, context)
            weights = target.true_stratum_weight(strata)[context.sizes] / (q_cur ** 2)
            return (Z * Z).T @ weights

        def calibrated_dense_solution(ref_backend, *, excluding_fold=None):
            R_factor, U_factor, count = ref_backend._design_counts(excluding_fold=excluding_fold)
            if excluding_fold is None:
                c_stat = ref_backend.c_total
                b_stat = ref_backend.b_total
                X_cur = X
                q_cur = q
            else:
                fold = int(excluding_fold)
                c_stat = ref_backend.c_total - ref_backend.c_fold[fold]
                b_stat = ref_backend.b_total - ref_backend.b_fold[fold]
                train_mask = folds != fold
                X_cur = X[train_mask]
                q_cur = q[train_mask]

            R_exact = ref_backend._build_exact_R(R_factor)
            exact_diag = np.diag(R_exact)
            emp_diag = empirical_R_diag(X_cur, q_cur)
            target_diag = np.maximum(exact_diag + alpha_r * (emp_diag - exact_diag), 0.0)
            scale = np.ones_like(exact_diag)
            extra = np.zeros_like(exact_diag)
            tol = 100.0 * np.finfo(np.float64).eps * max(
                1.0,
                float(np.max(np.abs(exact_diag), initial=0.0)),
                float(np.max(np.abs(target_diag), initial=0.0)),
            )
            good = exact_diag > tol
            scale[good] = target_diag[good] / exact_diag[good]
            extra[~good] = target_diag[~good]
            R_cal = np.sqrt(scale)[:, None] * R_exact * np.sqrt(scale)[None, :]
            R_cal[np.diag_indices_from(R_cal)] += extra

            U_stat = ref_backend._build_corrected_U(U_factor, excluding_fold=excluding_fold)
            ridge = ref_backend._ridge_penalty(R_factor, count)
            gram = R_cal - (U_stat.T @ U_stat) / float(count)
            ridge_diag = np.asarray(ridge, dtype=np.float64)
            diag = np.diag_indices_from(gram)
            if ridge_diag.ndim == 0:
                gram[diag] += float(ridge_diag)
            else:
                gram[diag] += ridge_diag
            rhs = c_stat - (U_stat.T @ b_stat) / float(count)
            return np.linalg.solve(gram, rhs)

        fit_all = backend.fit_all(store=store if alpha_r > 0.0 else None)
        assert backend._union_sizes is None
        expected_all = calibrated_dense_solution(reference)
        np.testing.assert_allclose(fit_all.beta, expected_all, rtol=1e-9, atol=1e-9)

        fit_fold = backend.fit_excluding_fold(1, store=store if alpha_r > 0.0 else None)
        expected_fold = calibrated_dense_solution(reference, excluding_fold=1)
        np.testing.assert_allclose(fit_fold.beta, expected_fold, rtol=1e-9, atol=1e-9)

        if alpha_u > 0.0:
            assert backend.U_emp_total is not None

    def test_size_player_diagonal_corrected_solver_rejects_non_size_player_basis(self):
        n = 5
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="shapley",
            semivalue_param=None,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        with pytest.raises(ValueError, match="size-player correction"):
            _ExactConditionalStats(
                target=target,
                strata=strata,
                feature_builder=feature_builder,
                ridge_lambda=0.2,
                ridge_schedule="fixed",
                num_folds=2,
                solver_mode="first_order",
                r_correction_alpha=0.5,
                u_correction_alpha=0.5,
                correction_solver_mode="diagonal_r_empirical_u",
            )

    @pytest.mark.parametrize(
        "solver_mode,surrogate_basis,include_nonlinear_size_terms,ridge_scaling",
        [
            ("first_order", 1, True, "scalar"),
            ("second_order", 2, False, "scalar"),
            ("size_player", "size_player", True, "scalar"),
            ("size_player_streaming", "size_player", True, "size_trace"),
        ],
    )
    def test_matrix_free_corrected_solver_matches_dense_corrected_solver(
        self,
        solver_mode,
        surrogate_basis,
        include_nonlinear_size_terms,
        ridge_scaling,
    ):
        n = 5
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=surrogate_basis,
            include_nonlinear_size_terms=include_nonlinear_size_terms,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.4,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )

        def make_backend(mode, correction_solver_mode, correction_max_iter=None):
            return _ExactConditionalStats(
                target=target,
                strata=strata,
                feature_builder=feature_builder,
                ridge_lambda=0.4,
                ridge_schedule="fixed",
                ridge_scaling=ridge_scaling,
                num_folds=3,
                solver_mode=mode,
                r_correction_alpha=0.6,
                u_correction_alpha=0.6,
                correction_solver_mode=correction_solver_mode,
                correction_max_iter=correction_max_iter,
                correction_tol=1e-12,
            )

        dense = make_backend("dense", "dense")
        matrix_free = make_backend(solver_mode, "matrix_free", correction_max_iter=200)
        assert isinstance(matrix_free.correction_solver, _ExactMatrixFreeCorrectedSolver)
        assert matrix_free._union_sizes is None

        X = self._coalitions_matching(n)
        y = np.sin(np.arange(len(X), dtype=np.float64) / 3.0) + 0.03 * X.sum(axis=1)
        q_by_size = 0.045 + 0.011 * np.arange(n + 1, dtype=np.float64)
        q_by_size[1::2] += 0.009
        q = q_by_size[X.sum(axis=1)]
        folds = (np.arange(len(X)) + X.sum(axis=1)) % 3
        store = _ObservationStore(num_sample=len(X), n=n)
        store.append(X, y, q, folds)
        dense.append(X, y, q, folds)
        matrix_free.append(X, y, q, folds)

        dense_all = dense.fit_all(store=store)
        matrix_free_all = matrix_free.fit_all(store=store)
        assert 0 <= matrix_free.correction_solver.last_num_iter <= 200
        assert np.isfinite(matrix_free.correction_solver.last_residual_norm)
        assert np.isfinite(matrix_free.correction_solver.last_relative_residual)
        assert isinstance(matrix_free.correction_solver.last_converged, bool)
        np.testing.assert_allclose(matrix_free_all.beta, dense_all.beta, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(matrix_free_all.phi, dense_all.phi, rtol=1e-7, atol=1e-7)

        dense_fold = dense.fit_excluding_fold(1, store=store)
        matrix_free_fold = matrix_free.fit_excluding_fold(1, store=store)
        assert 0 <= matrix_free.correction_solver.last_num_iter <= 200
        assert np.isfinite(matrix_free.correction_solver.last_residual_norm)
        assert np.isfinite(matrix_free.correction_solver.last_relative_residual)
        assert isinstance(matrix_free.correction_solver.last_converged, bool)
        np.testing.assert_allclose(matrix_free_fold.beta, dense_fold.beta, rtol=1e-7, atol=1e-7)

    def test_matrix_free_corrected_solver_rejects_mismatched_correction_alphas(self):
        n = 5
        strata = _SizeStrata(n, np.ones(n + 1, dtype=bool))
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="shapley",
            semivalue_param=None,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.2,
            ridge_schedule="fixed",
            num_folds=2,
            solver_mode="first_order",
            r_correction_alpha=0.5,
            u_correction_alpha=0.25,
            correction_solver_mode="matrix_free",
        )
        X = self._coalitions_matching(n)
        y = np.linspace(-1.0, 1.0, len(X))
        q = 0.05 + 0.01 * X.sum(axis=1)
        folds = np.arange(len(X)) % 2
        store = _ObservationStore(num_sample=len(X), n=n)
        store.append(X, y, q, folds)
        backend.append(X, y, q, folds)
        with pytest.raises(ValueError, match="r_correction_alpha == u_correction_alpha"):
            backend.fit_all(store=store)

    @pytest.mark.parametrize("surrogate_basis", [2, "size_player"])
    def test_full_exact_design_moments_match_bruteforce_conditioning(self, surrogate_basis):
        n = 4
        sampling_mask = np.ones(n + 1, dtype=bool)
        strata = _SizeStrata(n, sampling_mask)
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=surrogate_basis,
            include_nonlinear_size_terms=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="weighted_banzhaf",
            semivalue_param=0.35,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
        )

        X = np.asarray(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        )
        q_by_size = np.array([0.11, 0.07, 0.05, 0.09, 0.13], dtype=np.float64)
        q = q_by_size[X.sum(axis=1)]
        y = np.linspace(-1.0, 1.0, len(X))
        folds = np.arange(len(X)) % 2
        backend.append(X, y, q, folds)

        expected_R = np.zeros((feature_builder.dim, feature_builder.dim), dtype=np.float64)
        expected_U = np.zeros((n, feature_builder.dim), dtype=np.float64)
        true_weights = target.true_stratum_weight(strata)
        for row, q_row in zip(X, q):
            size = int(row.sum())
            cond_X = self._coalitions_matching(n, size=size)
            cond_context = strata.context_from_X(cond_X)
            Z = feature_builder.build(cond_X, cond_context)
            raw_gamma = target.raw_gamma(cond_X, cond_context)
            expected_R += (true_weights[size] / (q_row ** 2)) * (Z.T @ Z) / len(cond_X)
            expected_U += (raw_gamma.T @ Z) / (q_row * len(cond_X))

        np.testing.assert_allclose(backend._build_exact_R(backend.R_factor_total), expected_R, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(backend._build_exact_U(backend.U_factor_total), expected_U, rtol=1e-12, atol=1e-12)

    def test_group_exact_design_moments_match_bruteforce_conditioning(self):
        n = 4
        group = np.array([0, 2], dtype=np.int64)
        group_mask = np.zeros(n, dtype=bool)
        group_mask[group] = True
        strata = _GroupCellStrata.build(n, group, group_mask, exact_boundary_handling=False)
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis=2,
            include_nonlinear_size_terms=True,
            include_group_overlap_ratio=True,
        )
        boundary_X = np.empty((0, n), dtype=bool)
        boundary_context = strata.context_from_X(boundary_X)
        target = _GroupSumTarget(
            n=n,
            group=group,
            group_mask=group_mask,
            semivalue="shapley",
            semivalue_param=None,
            feature_builder=feature_builder,
            strata=strata,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
            rho_support_tol=0.0,
        )
        backend = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
        )

        X = np.asarray(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        )
        context = strata.context_from_X(X)
        q_by_cell = np.linspace(0.04, 0.16, len(strata.keys))
        q = q_by_cell[context.cell_ids]
        y = np.linspace(0.5, -0.5, len(X))
        folds = np.arange(len(X)) % 2
        backend.append(X, y, q, folds)

        expected_R = np.zeros((feature_builder.dim, feature_builder.dim), dtype=np.float64)
        expected_U = np.zeros((1, feature_builder.dim), dtype=np.float64)
        true_weights = target.true_stratum_weight(strata)
        for row, cell_id, q_row in zip(X, context.cell_ids, q):
            size, overlap = strata.keys[int(cell_id)]
            cond_X = self._coalitions_matching(n, size=size, overlap=overlap, group=group)
            cond_context = strata.context_from_X(cond_X)
            Z = feature_builder.build(cond_X, cond_context)
            raw_gamma = target.raw_gamma(cond_X, cond_context)
            expected_R += (true_weights[cell_id] / (q_row ** 2)) * (Z.T @ Z) / len(cond_X)
            expected_U += (raw_gamma.T @ Z) / (q_row * len(cond_X))

        np.testing.assert_allclose(backend._build_exact_R(backend.R_factor_total), expected_R, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(backend._build_exact_U(backend.U_factor_total), expected_U, rtol=1e-12, atol=1e-12)

    def test_boundary_handling_full_orbit_matches_empirical_dense_stats(self):
        n = 5
        sampling_mask = np.zeros(n + 1, dtype=bool)
        sampling_mask[2:n - 1] = True
        strata = _SizeStrata(n, sampling_mask)
        feature_builder = _FeatureBuilder(
            n=n,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
        )
        boundary_X = _boundary_subset_matrix(n)
        boundary_context = strata.context_from_X(boundary_X)
        target = _FullSemivalueTarget(
            n=n,
            semivalue="shapley",
            semivalue_param=None,
            feature_builder=feature_builder,
            boundary_X=boundary_X,
            boundary_context=boundary_context,
        )
        exact = _ExactConditionalStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
        )
        dense = _EmpiricalDenseStats(
            target=target,
            strata=strata,
            feature_builder=feature_builder,
            ridge_lambda=0.0,
            ridge_schedule="fixed",
            num_folds=2,
        )

        X = np.vstack([
            self._coalitions_matching(n, size=2),
            self._coalitions_matching(n, size=3),
        ])
        q_by_size = np.zeros(n + 1, dtype=np.float64)
        q_by_size[2] = 0.031
        q_by_size[3] = 0.047
        q = q_by_size[X.sum(axis=1)]
        y = np.linspace(-1.0, 1.0, len(X))
        folds = np.arange(len(X)) % 2

        exact.append(X, y, q, folds)
        dense.append(X, y, q, folds)

        np.testing.assert_allclose(exact._build_exact_R(exact.R_factor_total), dense.A_total, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(exact._build_exact_U(exact.U_factor_total), dense.B_total, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(exact.c_total, dense.c_total, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(exact.b_total, dense.b_total, rtol=1e-12, atol=1e-12)

    def test_public_full_estimator_accepts_exact_backend_alias(self):
        n = 5
        values = np.linspace(-1.0, 1.0, 2 ** n)
        final_value, traj = runEstimator(
            estimator="EaseSHAP",
            n_process=1,
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=8,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=11,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
            surrogate_stats_backend="exact_conditional",
            use_complement_sampling=False,
            num_folds=3,
        ).run()
        assert final_value.shape == (n,)
        assert traj.shape[1] == n
        assert np.all(np.isfinite(final_value))
        assert np.all(np.isfinite(traj))

    def test_boundary_subset_matrix_supports_explicit_sizes(self):
        n = 5
        X = _boundary_subset_matrix_for_sizes(n, [0, 2, 5])

        assert X.shape == (1 + 10 + 1, n)
        assert set(X.sum(axis=1).tolist()) == {0, 2, 5}
        assert _boundary_eval_count_for_sizes(n, [0, 2, 5]) == len(X)

    def test_boundary_integer_normalization_rejects_bool(self):
        with pytest.raises(ValueError):
            _normalize_boundary_order(True)
        with pytest.raises(ValueError):
            _normalize_boundary_sizes(5, [0, False, 5])

    def test_adaptive_boundary_sizes_expand_symmetric_outer_band(self):
        assert _adaptive_boundary_sizes(4, 16) == [0, 1, 2, 3, 4]
        assert _adaptive_boundary_sizes(6, 43) == [0, 1, 5, 6]
        assert _adaptive_boundary_sizes(6, 44) == [0, 1, 5, 6]
        assert _adaptive_boundary_sizes(6, 45) == [0, 1, 2, 4, 5, 6]

    def test_boundary_policy_precedence_is_explicit(self):
        n = 6
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=8,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=11,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
            use_complement_sampling=False,
            boundary_policy="none",
            num_folds=2,
        )
        assert est.boundary_policy == "none"
        assert not est.exact_boundary_handling
        assert est.boundary_sizes == ()
        assert est.boundary_eval_count == 0

        with pytest.raises(ValueError, match="exact_boundary_handling=False"):
            EaseSHAP(
                semivalue="shapley",
                semivalue_param=None,
                game_func=TableUtility,
                game_args={"num_player": n, "values": values},
                num_player=n,
                nue_avg=8,
                nue_per_proc=4,
                nue_track_avg=2,
                estimator_seed=11,
                surrogate_basis=1,
                include_nonlinear_size_terms=False,
                use_complement_sampling=False,
                exact_boundary_handling=False,
                boundary_policy="adaptive",
                num_folds=2,
            )

    def test_default_fixed_boundary_policy_uses_only_extreme_coalitions(self):
        n = 6
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=8,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=11,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
            use_complement_sampling=False,
            num_folds=2,
        )

        assert est.boundary_policy == "fixed"
        assert est.boundary_order == 0
        assert est.boundary_sizes == (0, 6)
        assert est.boundary_eval_count == 2
        assert est.pos_traj == 0

    def test_adaptive_boundary_policy_marks_pre_mc_checkpoints_nan(self):
        n = 6
        values = np.linspace(-1.0, 1.0, 2 ** n)
        final_value, traj = runEstimator(
            estimator="EaseSHAP",
            n_process=1,
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=8,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=17,
            boundary_policy="adaptive",
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
            use_complement_sampling=False,
            pilot_fraction=0.0,
            num_folds=2,
        ).run()

        assert np.all(np.isfinite(final_value))
        assert traj.shape == (4, n)
        assert np.all(np.isnan(traj[:3]))
        assert np.all(np.isfinite(traj[3]))

    def test_adaptive_boundary_policy_preserves_nan_when_no_checkpoint_enters_mc(self):
        n = 6
        values = np.linspace(-1.0, 1.0, 2 ** n)
        final_value, traj = runEstimator(
            estimator="EaseSHAP",
            n_process=1,
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=8,
            nue_per_proc=4,
            nue_track_avg=7,
            estimator_seed=19,
            boundary_policy="adaptive",
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
            use_complement_sampling=False,
            pilot_fraction=0.0,
            num_folds=2,
        ).run()

        assert np.all(np.isfinite(final_value))
        assert traj.shape == (1, n)
        assert np.all(np.isnan(traj[0]))

    def test_public_full_estimator_accepts_first_order_solver_mode(self):
        n = 5
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=23,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
            surrogate_stats_backend="exact_conditional",
            surrogate_solver_mode="first_order",
            exact_boundary_handling=False,
            num_folds=2,
        )

        assert est.surrogate_solver_mode == "first_order"
        assert isinstance(est._engine.backend.solver, _ExactFirstOrderInteractionSolver)

    def test_public_full_estimator_accepts_second_order_solver_mode(self):
        n = 5
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=31,
            surrogate_basis=2,
            include_nonlinear_size_terms=False,
            surrogate_stats_backend="exact_conditional",
            surrogate_solver_mode="second_order",
            exact_boundary_handling=False,
            num_folds=2,
        )

        assert est.surrogate_solver_mode == "second_order"
        assert isinstance(est._engine.backend.solver, _ExactSecondOrderSolver)

    def test_public_full_estimator_accepts_size_player_solver_mode(self):
        n = 5
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=37,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=False,
            surrogate_stats_backend="exact_conditional",
            surrogate_solver_mode="size_player",
            exact_boundary_handling=False,
            num_folds=2,
        )

        assert est.surrogate_solver_mode == "size_player"
        assert isinstance(est._engine.backend.solver, _ExactSizePlayerSolver)

    @pytest.mark.parametrize(
        "mode,normalized",
        [
            ("cross-fit", "crossfit"),
            ("all-data-aipw", "all_data_aipw"),
            ("plug_in", "plugin"),
        ],
    )
    def test_public_full_estimator_accepts_readout_mode_aliases(self, mode, normalized):
        n = 4
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=41,
            surrogate_readout_mode=mode,
            exact_boundary_handling=False,
            num_folds=2,
        )

        assert est.surrogate_readout_mode == normalized
        assert est._engine.readout_mode == normalized

    def test_public_full_estimator_rejects_unknown_readout_mode(self):
        n = 4
        values = np.zeros(2 ** n)
        with pytest.raises(ValueError, match="surrogate_readout_mode"):
            EaseSHAP(
                semivalue="shapley",
                semivalue_param=None,
                game_func=TableUtility,
                game_args={"num_player": n, "values": values},
                num_player=n,
                nue_avg=4,
                nue_per_proc=4,
                nue_track_avg=2,
                estimator_seed=43,
                surrogate_readout_mode="foldless_magic",
                exact_boundary_handling=False,
                num_folds=2,
            )

    def test_all_data_aipw_readout_matches_direct_formula(self):
        n = 4
        values = np.linspace(-0.5, 0.5, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=16,
            estimator_seed=47,
            surrogate_readout_mode="all_data_aipw",
            pilot_fraction=0.0,
            surrogate_ridge_lambda=0.25,
            exact_boundary_handling=False,
            num_folds=2,
        )
        X = np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            dtype=bool,
        )
        y = np.linspace(-1.2, 0.9, len(X))
        q = np.linspace(0.2, 0.7, len(X))
        est._engine._append_block(np.column_stack((X.astype(float), y, q)))

        fit = est._engine.backend.fit_all(store=est._engine.store)
        X_store, y_store, q_store, _folds = est._engine.store.rows()
        context = est._engine.strata.context_from_X(X_store)
        gamma = est._engine.target.raw_gamma(X_store, context) / q_store[:, None]
        resid = y_store - est._engine._predict_surrogate(fit, X_store, context)
        expected = est._engine.boundary_exact + fit.phi + (gamma.T @ resid) / float(len(y_store))

        np.testing.assert_allclose(est._engine._readout_estimate(), expected, rtol=1e-12, atol=1e-12)

    def test_plugin_readout_matches_fit_all_phi(self):
        n = 4
        values = np.linspace(-0.5, 0.5, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=16,
            estimator_seed=53,
            surrogate_readout_mode="plugin",
            pilot_fraction=0.0,
            surrogate_ridge_lambda=0.25,
            exact_boundary_handling=False,
            num_folds=2,
        )
        X = np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        )
        y = np.asarray([-0.3, 0.4, 1.1, -0.8], dtype=float)
        q = np.asarray([0.2, 0.35, 0.5, 0.65], dtype=float)
        est._engine._append_block(np.column_stack((X.astype(float), y, q)))

        fit = est._engine.backend.fit_all(store=est._engine.store)
        expected = est._engine.boundary_exact + fit.phi

        np.testing.assert_allclose(est._engine._readout_estimate(), expected, rtol=1e-12, atol=1e-12)

    def test_public_group_estimator_accepts_exact_backend(self):
        n = 5
        values = np.linspace(-0.5, 0.5, 2 ** n)
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
            estimator_seed=13,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
            surrogate_stats_backend="exact_conditional",
            num_folds=3,
        ).run()
        assert np.isfinite(final_value)
        assert traj.ndim == 1
        assert np.all(np.isfinite(traj))

    @pytest.mark.parametrize("ridge_schedule", ["fixed", "times_m"])
    def test_public_full_estimator_accepts_size_trace_ridge_scaling(self, ridge_schedule):
        n = 4
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=17,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
            surrogate_stats_backend="exact_conditional",
            surrogate_ridge_scaling="size_trace",
            surrogate_ridge_schedule=ridge_schedule,
            exact_boundary_handling=False,
            num_folds=2,
        )

        assert est.surrogate_ridge_scaling == "size_trace"

    def test_public_full_estimator_accepts_dense_corrected_configuration(self):
        n = 4
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=29,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
            surrogate_stats_backend="exact_conditional",
            surrogate_r_correction_alpha=0.5,
            surrogate_u_correction_alpha=0.25,
            surrogate_correction_solver_mode="Dense-Brute-Force",
            exact_boundary_handling=False,
            num_folds=2,
        )

        assert est.surrogate_r_correction_alpha == 0.5
        assert est.surrogate_u_correction_alpha == 0.25
        assert est.surrogate_correction_solver_mode == "dense"
        assert isinstance(est._engine.backend.correction_solver, _ExactDenseCorrectedSolver)

    def test_public_full_estimator_accepts_matrix_free_corrected_configuration(self):
        n = 4
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=31,
            surrogate_basis=1,
            include_nonlinear_size_terms=True,
            surrogate_stats_backend="exact_conditional",
            surrogate_solver_mode="first-order",
            surrogate_r_correction_alpha=0.5,
            surrogate_u_correction_alpha=0.5,
            surrogate_correction_solver_mode="cg",
            surrogate_correction_max_iter=6,
            surrogate_correction_tol=0.0,
            exact_boundary_handling=False,
            num_folds=2,
        )

        assert est.surrogate_correction_solver_mode == "matrix_free"
        assert est.surrogate_correction_max_iter == 6
        assert est.surrogate_correction_tol == 0.0
        assert isinstance(est._engine.backend.correction_solver, _ExactMatrixFreeCorrectedSolver)

    @pytest.mark.parametrize("pilot_design_updates", [0, -1, 1.5, True])
    def test_public_full_estimator_rejects_invalid_pilot_design_updates(self, pilot_design_updates):
        n = 4
        with pytest.raises(ValueError, match="pilot_design_updates"):
            EaseSHAP(
                semivalue="shapley",
                semivalue_param=None,
                game_func=TableUtility,
                game_args={"num_player": n, "values": np.zeros(2 ** n)},
                num_player=n,
                nue_avg=4,
                nue_per_proc=4,
                nue_track_avg=2,
                estimator_seed=41,
                pilot_design_updates=pilot_design_updates,
                exact_boundary_handling=False,
                num_folds=2,
            )

    def test_iterative_pilot_design_default_is_exactly_one_update(self):
        n = 5
        indices = np.arange(2 ** n, dtype=np.float64)
        values = np.sin(0.71 * indices) + 0.03 * indices
        base_kwargs = dict(
            semivalue="weighted_banzhaf",
            semivalue_param=0.4,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=8,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=47,
            pilot_fraction=0.5,
            use_complement_sampling=False,
            surrogate_basis=1,
            surrogate_ridge_lambda=0.3,
            exact_boundary_handling=False,
            num_folds=2,
        )

        default = EaseSHAP(**base_kwargs)
        explicit_one = EaseSHAP(pilot_design_updates=1, **base_kwargs)
        iterative = EaseSHAP(pilot_design_updates=3, **base_kwargs)

        def run_serial(estimator):
            for samples in estimator.sampling():
                estimator.aggregate(estimator.run(samples))
            return estimator.finalize()

        default_result = run_serial(default)
        explicit_result = run_serial(explicit_one)
        iterative_result = run_serial(iterative)

        np.testing.assert_allclose(default_result[0], explicit_result[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(default_result[1], explicit_result[1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(default._engine.q_stage2, explicit_one._engine.q_stage2, rtol=0.0, atol=0.0)
        assert default.pilot_design_updates == 1
        assert default._engine.pilot_design_updates == 1
        assert len(default._engine.pilot_design_beta_history) == 1
        assert len(default._engine.pilot_design_q_history) == 2

        assert iterative._engine.store.num_obs == iterative.num_sample
        assert len(iterative._engine.pilot_design_beta_history) == 3
        assert len(iterative._engine.pilot_design_q_history) == 4
        np.testing.assert_allclose(
            iterative._engine.pilot_design_q_history[1],
            default._engine.q_stage2,
            rtol=0.0,
            atol=0.0,
        )
        for q_density in iterative._engine.pilot_design_q_history:
            assert np.all(np.isfinite(q_density))
            assert np.all(q_density >= 0.0)
            np.testing.assert_allclose(
                np.dot(iterative._engine.strata.counts, q_density),
                1.0,
                rtol=1e-12,
                atol=1e-12,
            )
        assert np.all(np.isfinite(iterative_result[0]))
        assert np.all(np.isfinite(iterative_result[1]))

    def test_iterative_size_player_pilot_design_runs_with_matrix_free_empirical_fit(self):
        n = 4
        indices = np.arange(2 ** n, dtype=np.float64)
        values = np.cos(0.63 * indices) + 0.04 * indices * indices
        est = EaseSHAP(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=6,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=53,
            pilot_fraction=0.5,
            pilot_design_updates=2,
            use_complement_sampling=True,
            surrogate_basis="size_player",
            include_nonlinear_size_terms=True,
            surrogate_ridge_lambda=0.4,
            surrogate_ridge_scaling="size_trace",
            surrogate_stats_backend="exact_conditional",
            surrogate_solver_mode="size_player",
            surrogate_r_correction_alpha=1.0,
            surrogate_u_correction_alpha=1.0,
            surrogate_correction_solver_mode="matrix_free",
            surrogate_correction_max_iter=200,
            surrogate_correction_tol=1e-10,
            exact_boundary_handling=False,
            num_folds=2,
        )

        for samples in est.sampling():
            est.aggregate(est.run(samples))
        final_value, trajectory = est.finalize()

        assert isinstance(est._engine.backend.correction_solver, _ExactMatrixFreeCorrectedSolver)
        assert len(est._engine.pilot_design_beta_history) == 2
        assert len(est._engine.pilot_design_q_history) == 3
        assert np.all(np.isfinite(final_value))
        assert np.all(np.isfinite(trajectory))

    def test_iterative_exact_candidate_refit_rejects_nonempirical_configuration(self):
        n = 4
        with pytest.raises(ValueError, match="matrix-free empirical correction"):
            EaseSHAP(
                semivalue="shapley",
                semivalue_param=None,
                game_func=TableUtility,
                game_args={"num_player": n, "values": np.zeros(2 ** n)},
                num_player=n,
                nue_avg=4,
                nue_per_proc=4,
                nue_track_avg=2,
                estimator_seed=59,
                pilot_design_updates=2,
                surrogate_basis="size_player",
                surrogate_stats_backend="exact_conditional",
                surrogate_solver_mode="size_player",
                exact_boundary_handling=False,
                num_folds=2,
            )

    def test_public_group_estimator_accepts_dense_corrected_configuration(self):
        n = 4
        values = np.linspace(-1.0, 1.0, 2 ** n)
        est = EaseSHAP_group(
            semivalue="shapley",
            semivalue_param=None,
            group=[0, 2],
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=33,
            surrogate_basis=1,
            include_nonlinear_size_terms=False,
            surrogate_stats_backend="exact_conditional",
            surrogate_r_correction_alpha=0.5,
            surrogate_u_correction_alpha=0.25,
            surrogate_correction_solver_mode="Dense",
            exact_boundary_handling=False,
            num_folds=2,
        )

        assert est.surrogate_r_correction_alpha == 0.5
        assert est.surrogate_u_correction_alpha == 0.25
        assert est.surrogate_correction_solver_mode == "dense"
        assert isinstance(est._engine.backend.correction_solver, _ExactDenseCorrectedSolver)

    def test_public_full_estimator_rejects_correction_with_empirical_backend(self):
        n = 4
        values = np.zeros(2 ** n)
        with pytest.raises(ValueError, match="exact_conditional"):
            EaseSHAP(
                semivalue="shapley",
                semivalue_param=None,
                game_func=TableUtility,
                game_args={"num_player": n, "values": values},
                num_player=n,
                nue_avg=4,
                nue_per_proc=4,
                nue_track_avg=2,
                estimator_seed=31,
                surrogate_stats_backend="empirical_dense",
                surrogate_r_correction_alpha=0.5,
                surrogate_u_correction_alpha=0.5,
                exact_boundary_handling=False,
                num_folds=2,
            )

    def test_public_group_estimator_rejects_correction_with_empirical_backend(self):
        n = 4
        values = np.zeros(2 ** n)
        with pytest.raises(ValueError, match="exact_conditional"):
            EaseSHAP_group(
                semivalue="shapley",
                semivalue_param=None,
                group=[0, 2],
                game_func=TableUtility,
                game_args={"num_player": n, "values": values},
                num_player=n,
                nue_avg=4,
                nue_per_proc=4,
                nue_track_avg=2,
                estimator_seed=35,
                surrogate_stats_backend="empirical_dense",
                surrogate_r_correction_alpha=0.5,
                surrogate_u_correction_alpha=0.5,
                exact_boundary_handling=False,
                num_folds=2,
            )

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (
                {"surrogate_stats_backend": "empirical_dense", "surrogate_basis": "size_player"},
                "exact_conditional",
            ),
            (
                {"surrogate_stats_backend": "exact_conditional", "surrogate_basis": 1},
                "size_player",
            ),
        ],
    )
    def test_public_full_estimator_rejects_invalid_size_trace_ridge_scaling(self, kwargs, match):
        n = 4
        values = np.zeros(2 ** n)
        base_kwargs = dict(
            semivalue="shapley",
            semivalue_param=None,
            game_func=TableUtility,
            game_args={"num_player": n, "values": values},
            num_player=n,
            nue_avg=4,
            nue_per_proc=4,
            nue_track_avg=2,
            estimator_seed=19,
            surrogate_ridge_scaling="size_trace",
            exact_boundary_handling=False,
            num_folds=2,
        )
        base_kwargs.update(kwargs)

        with pytest.raises(ValueError, match=match):
            EaseSHAP(**base_kwargs)

    def test_group_estimator_rejects_size_trace_ridge_scaling(self):
        n = 4
        values = np.zeros(2 ** n)
        with pytest.raises(ValueError, match="full EaseSHAP"):
            EaseSHAP_group(
                semivalue="shapley",
                semivalue_param=None,
                group=[0, 2],
                game_func=TableUtility,
                game_args={"num_player": n, "values": values},
                num_player=n,
                nue_avg=4,
                nue_per_proc=4,
                nue_track_avg=2,
                estimator_seed=23,
                surrogate_ridge_scaling="size_trace",
                exact_boundary_handling=False,
                num_folds=2,
            )


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
