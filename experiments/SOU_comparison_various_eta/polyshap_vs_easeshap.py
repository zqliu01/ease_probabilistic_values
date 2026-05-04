"""
Structured Gaussian SOU Shapley experiment: second-order EaseSHAP vs PolySHAP.

This script reuses the same game, alpha settings, budgets, and ground-truth
definition as regmsr_unpaired_vs_easeshap.py, but changes the run-time
comparison:

- EaseSHAP:
    * no exact boundary handling,
    * no complement sampling,
    * degree <= 2 interaction surrogate,
    * no nonlinear |S| terms.
- PolySHAP_regression:
    * unpaired sampling,
    * degree <= 2 polynomial surrogate.

Ground truth is shared with the base experiment because SOU Shapley values are
computed analytically.
"""

from __future__ import annotations

import json

import regmsr_unpaired_vs_easeshap as base


BASE_RESULTS = base.SCRIPT_DIR / "results" / "regmsr_unpaired_vs_easeshap"

base.__doc__ = __doc__
base.OUT = base.SCRIPT_DIR / "results" / "polyshap_vs_easeshap"

# Ground truth is identical to the base experiment, so share the analytic SOU
# reference values across variants.
base.GROUNDTRUTH_DIR = BASE_RESULTS / "groundtruth"
base.RUNS_DIR = base.OUT / "runs"
base.PLOTS_DIR = base.OUT / "plots"
base.GAME_DIR = base.OUT / "game"
base.PLOT_OUTPUT_FILENAME = "sou_vs_polyshap.png"

base.RUN_EASESHAP_KWARGS = {
    "exact_boundary_handling": False,
    "use_complement_sampling": False,
    "surrogate_basis": 2,
    "include_nonlinear_size_terms": False,
}

base.POLYSHAP_KWARGS = {
    "max_order": 2,
}

base.ALGORITHM_SPECS = [
    {
        "name": "EaseSHAP_order2",
        "backend": "EaseSHAP",
        "estimator_kwargs": base.RUN_EASESHAP_KWARGS,
    },
    {
        "name": "PolySHAP_regression",
        "backend": "PolySHAP_regression",
        "estimator_kwargs": base.POLYSHAP_KWARGS,
    },
]


_ORIGINAL_WRITE_CONFIG = base.write_config


def write_config() -> None:
    _ORIGINAL_WRITE_CONFIG()
    config_path = base.OUT / "config.json"
    config = json.loads(config_path.read_text())
    config["shared_groundtruth_dir"] = str(base.GROUNDTRUTH_DIR)
    config_path.write_text(json.dumps(config, indent=2) + "\n")


base.write_config = write_config


if __name__ == "__main__":
    base.main()
