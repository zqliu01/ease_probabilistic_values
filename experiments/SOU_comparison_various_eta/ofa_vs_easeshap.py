"""
Structured Gaussian SOU Shapley experiment: boundary/size-player EaseSHAP vs OFA_fixed.

This script reuses the same game, alpha settings, budgets, and ground-truth
definition as regmsr_unpaired_vs_easeshap.py, but changes the run-time
comparison:

- EaseSHAP:
    * exact boundary handling,
    * no complement sampling,
    * size-player surrogate basis,
    * no nonlinear |S| terms.
- OFA_fixed:
    * no complement sampling.

Ground truth is shared with the base experiment because SOU Shapley values are
computed analytically.
"""

from __future__ import annotations

import json

import regmsr_unpaired_vs_easeshap as base


BASE_RESULTS = base.SCRIPT_DIR / "results" / "regmsr_unpaired_vs_easeshap"

base.__doc__ = __doc__
base.OUT = base.SCRIPT_DIR / "results" / "ofa_vs_easeshap"

# Ground truth is identical to the base experiment, so share the analytic SOU
# reference values across variants.
base.GROUNDTRUTH_DIR = BASE_RESULTS / "groundtruth"
base.RUNS_DIR = base.OUT / "runs"
base.PLOTS_DIR = base.OUT / "plots"
base.GAME_DIR = base.OUT / "game"
base.PLOT_OUTPUT_FILENAME = "sou_vs_ofa.png"

base.RUN_EASESHAP_KWARGS = {
    "exact_boundary_handling": True,
    "use_complement_sampling": False,
    "surrogate_basis": "size_player",
    "include_nonlinear_size_terms": False,
}

base.OFA_KWARGS = {}

base.ALGORITHM_SPECS = [
    {
        "name": "EaseSHAP_boundary_size_player",
        "backend": "EaseSHAP",
        "estimator_kwargs": base.RUN_EASESHAP_KWARGS,
    },
    {
        "name": "OFA_fixed",
        "backend": "OFA_fixed",
        "estimator_kwargs": base.OFA_KWARGS,
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
