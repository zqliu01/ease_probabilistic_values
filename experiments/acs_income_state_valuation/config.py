"""Method configuration for the ACSIncome state-source benchmark."""

from __future__ import annotations

from typing import Any


EASESHAP_COMMON_KWARGS: dict[str, Any] = {
    "exact_boundary_handling": True,
    "boundary_policy": "fixed",
    "boundary_order": 1,
    "use_complement_sampling": True,
    "surrogate_ridge_lambda": 0.01,
    "surrogate_ridge_schedule": "times_m",
    "num_folds": 2,
    "surrogate_readout_mode": "crossfit",
}

EASESHAP_SIZE_PLAYER_RIDGE_KWARGS: dict[str, Any] = {
    "surrogate_ridge_lambda": 1.0,
    "surrogate_ridge_schedule": "fixed",
    "surrogate_ridge_scaling": "size_trace",
}

METHOD_SPECS: list[dict[str, Any]] = [
    {
        "name": "EaseSHAP_interaction_nonlinear",
        "backend": "EaseSHAP",
        "support": "all",
        "estimator_kwargs": {
            **EASESHAP_COMMON_KWARGS,
            "pilot_design_updates": 1,
            "surrogate_basis": 1,
            "include_nonlinear_size_terms": True,
        },
    },
    {
        "name": "EaseSHAP_size_player",
        "backend": "EaseSHAP",
        "support": "all",
        "estimator_kwargs": {
            **EASESHAP_COMMON_KWARGS,
            **EASESHAP_SIZE_PLAYER_RIDGE_KWARGS,
            "surrogate_basis": "size_player",
            "include_nonlinear_size_terms": False,
            "surrogate_stats_backend": "exact_conditional",
            "surrogate_solver_mode": "size_player",
            "surrogate_r_correction_alpha": 1.0,
            "surrogate_u_correction_alpha": 1.0,
            "surrogate_correction_solver_mode": "matrix_free",
            "surrogate_correction_max_iter": 10,
        },
    },
    {"name": "OFA_fixed", "backend": "OFA_fixed", "support": "all", "estimator_kwargs": {}},
    {"name": "OFA_baseline", "backend": "OFA_baseline", "support": "all", "estimator_kwargs": {}},
    {"name": "sampling_lift", "backend": "sampling_lift", "support": "all", "estimator_kwargs": {}},
    {"name": "SHAP_IQ", "backend": "SHAP_IQ", "support": "all", "estimator_kwargs": {}},
    {"name": "GELS", "backend": "GELS", "support": "all", "estimator_kwargs": {}},
    {"name": "improved_AME", "backend": "improved_AME", "support": "all", "estimator_kwargs": {}},
    {"name": "kernelSHAP", "backend": "kernelSHAP", "support": "shapley", "estimator_kwargs": {}},
    {
        "name": "LeverageSHAP",
        "backend": "LeverageSHAP",
        "support": "shapley",
        "estimator_kwargs": {"sampling_with_replacement": True},
        "label": "LeverageSHAP",
    },
    {
        "name": "LeverageSHAP_paired_border",
        "backend": "LeverageSHAP_border",
        "support": "shapley",
        "estimator_kwargs": {},
        "label": "LeverageSHAP (border trick)",
    },
    {"name": "permutation", "backend": "permutation", "support": "shapley", "estimator_kwargs": {}},
    {"name": "complement", "backend": "complement", "support": "shapley", "estimator_kwargs": {}},
    {"name": "group_testing", "backend": "group_testing", "support": "shapley", "estimator_kwargs": {}},
    {
        "name": "RegressionMSR_unbiased",
        "backend": "RegressionMSR_unbiased",
        "support": "all",
        "estimator_kwargs": {
            "sampling_with_replacement": True,
            "paired_sampling": None,
            "num_folds": 2,
        },
    },
    {
        "name": "RegressionMSR_unbiased_no_replacement",
        "backend": "RegressionMSR_unbiased",
        "support": "all",
        "estimator_kwargs": {
            "sampling_with_replacement": False,
            "paired_sampling": None,
            "num_folds": 2,
        },
        "label": "RegressionMSR (no replacement)",
    },
    {
        "name": "PolySHAP_regression_optional",
        "backend": "PolySHAP_regression",
        "support": "shapley",
        "optional": True,
        "estimator_kwargs": {"max_order": 2},
    },
]

METHOD_LABELS = {
    method["name"]: method.get("label", method["name"])
    for method in METHOD_SPECS
}
METHOD_LABELS.update(
    {
        "EaseSHAP_interaction_nonlinear": "EASE-FO",
        "EaseSHAP_size_player": "EASE-SP",
        "OFA_fixed": "OFA",
        "OFA_baseline": "OFA baseline",
        "sampling_lift": "Sampling lift",
        "SHAP_IQ": "SHAP-IQ",
        "GELS": "GELS",
        "improved_AME": "Improved AME",
        "kernelSHAP": "kernelSHAP",
        "permutation": "Permutation",
        "complement": "Complement",
        "group_testing": "Group testing",
        "RegressionMSR_unbiased": "RegressionMSR",
        "RegressionMSR_unbiased_no_replacement": "RegressionMSR (no repl.)",
        "LeverageSHAP_paired_border": "LeverageSHAP (border)",
        "PolySHAP_regression_optional": "PolySHAP-2ADD",
    }
)
