"""
Structured Gaussian SOU full benchmark.

This benchmark uses the same n=40 gameSOUStructuredGaussian setup as
paper/experiments/SOU_comparison, then evaluates the non-CoupSamp methods from
try7/try8 together with EaseSHAP, RegressionMSR, and PolySHAP.

Runs are stored per setting, method, and seed so the benchmark can be resumed
after interruption.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import pickle
import re
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "results"
GROUNDTRUTH_DIR = OUT / "groundtruth"
RUNS_DIR = OUT / "runs"
PLOTS_DIR = OUT / "plots"
GAME_DIR = OUT / "game"

# Keep matplotlib and native numerical libraries from writing outside the repo
# or oversubscribing shared machines.
(OUT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(OUT / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT / ".cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_DIR = SCRIPT_DIR.parents[1]
PACKAGE_ROOT = PAPER_DIR / "EaseSHAP"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from easeshap.utilityFuncs import gameSOUStructuredGaussian
import easeshap.estimators as estimator_module


N = 40
GAME_SEED = 42
BASE_SEED = 2026
N_RUNS = 10
RUN_SEEDS = [BASE_SEED + i * 137 for i in range(N_RUNS)]

GAME_ALPHAS = [0.25**0.5, 0.5**0.5, 0.75**0.5]
NUM_HIGH_ORDER = N ** 2
SIGMA2 = None

RUN_NUE = 5_000
TRACK_STEP = 50
NUE_BUDGETS = np.arange(TRACK_STEP, RUN_NUE + 1, TRACK_STEP, dtype=int)

SEMIVALUE_SPECS = [
    {
        "name": "shapley",
        "title": "Shapley",
        "semivalue": "shapley",
        "semivalue_param": None,
    },
    {
        "name": "beta1_4",
        "title": "Beta(1, 4)",
        "semivalue": "beta_shapley",
        "semivalue_param": (1, 4),
    },
    {
        "name": "beta4_1",
        "title": "Beta(4, 1)",
        "semivalue": "beta_shapley",
        "semivalue_param": (4, 1),
    },
    {
        "name": "wb0p25",
        "title": "WeightedBanzhaf(0.25)",
        "semivalue": "weighted_banzhaf",
        "semivalue_param": 0.25,
    },
    {
        "name": "wb0p5",
        "title": "WeightedBanzhaf(0.5)",
        "semivalue": "weighted_banzhaf",
        "semivalue_param": 0.5,
    },
    {
        "name": "wb0p75",
        "title": "WeightedBanzhaf(0.75)",
        "semivalue": "weighted_banzhaf",
        "semivalue_param": 0.75,
    },
]

EASESHAP_COMMON_KWARGS = {
    "exact_boundary_handling": True,
    "use_complement_sampling": True,
    "surrogate_ridge_lambda": 0.01,
    "surrogate_ridge_schedule": "times_m",
}

METHOD_SPECS = [
    {
        "name": "EaseSHAP_interaction_nonlinear",
        "backend": "EaseSHAP",
        "support": "all",
        "estimator_kwargs": {
            **EASESHAP_COMMON_KWARGS,
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
            "surrogate_basis": "size_player",
            "include_nonlinear_size_terms": False,
        },
    },
    {"name": "OFA_fixed", "backend": "OFA_fixed", "support": "all", "estimator_kwargs": {}},
    {"name": "OFA_baseline", "backend": "OFA_baseline", "support": "all", "estimator_kwargs": {}},
    {"name": "sampling_lift", "backend": "sampling_lift", "support": "all", "estimator_kwargs": {}},
    {"name": "SHAP_IQ", "backend": "SHAP_IQ", "support": "all", "estimator_kwargs": {}},
    {"name": "GELS", "backend": "GELS", "support": "all", "estimator_kwargs": {}},
    {"name": "improved_AME", "backend": "improved_AME", "support": "all", "estimator_kwargs": {}},
    {"name": "kernelSHAP", "backend": "kernelSHAP", "support": "shapley", "estimator_kwargs": {}},
    {"name": "permutation", "backend": "permutation", "support": "shapley", "estimator_kwargs": {}},
    {"name": "complement", "backend": "complement", "support": "shapley", "estimator_kwargs": {}},
    {"name": "group_testing", "backend": "group_testing", "support": "shapley", "estimator_kwargs": {}},
    {"name": "WSL", "backend": "WSL", "support": "non_shapley", "estimator_kwargs": {}},
    {
        "name": "weighted_permutation",
        "backend": "weighted_permutation",
        "support": "non_shapley",
        "estimator_kwargs": {},
    },
    {"name": "OFA_optimal", "backend": "OFA_optimal", "support": "non_shapley", "estimator_kwargs": {}},
    {
        "name": "WGELS_shapley",
        "backend": "WGELS_shapley",
        "support": "non_shapley",
        "estimator_kwargs": {},
    },
    {"name": "AME", "backend": "AME", "support": "ame", "estimator_kwargs": {}},
    {
        "name": "RegressionMSR_unbiased",
        "backend": "RegressionMSR_unbiased",
        "support": "all",
        "estimator_kwargs": {
            "sampling_with_replacement": True,
            "paired_sampling": None,
        },
    },
    {
        "name": "PolySHAP_regression",
        "backend": "PolySHAP_regression",
        "support": "shapley",
        "estimator_kwargs": {
            "max_order": 2,
        },
    },
]

AUCC_SEMIVALUE_ORDER = [
    ("shapley", "Shapley"),
    ("beta4_1", "BS(4,1)"),
    ("beta1_4", "BS(1,4)"),
    ("wb0p25", "WB(0.25)"),
    ("wb0p5", "WB(0.5)"),
    ("wb0p75", "WB(0.75)"),
]

AUCC_METHOD_LABELS = {
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
    "WSL": "WSL",
    "weighted_permutation": "Weighted permutation",
    "OFA_optimal": "OFA optimal",
    "WGELS_shapley": "WGELS",
    "AME": "AME",
    "RegressionMSR_unbiased": "RegressionMSR",
    "PolySHAP_regression": "PolySHAP",
}

AUCC_DEFAULT_METHOD_ORDER = [
    "EaseSHAP_interaction_nonlinear",
    "EaseSHAP_size_player",
    "OFA_fixed",
    "sampling_lift",
    "SHAP_IQ",
    "GELS",
    "improved_AME",
    "kernelSHAP",
    "permutation",
    "complement",
    "group_testing",
    "WSL",
    "weighted_permutation",
    "WGELS_shapley",
    "AME",
    "RegressionMSR_unbiased",
    "PolySHAP_regression",
]

AUCC_HIDDEN_BY_DEFAULT = {
    "improved_AME",
    "OFA_baseline",
    "OFA_optimal",
}

AUCC_EASE_METHODS = {
    "EaseSHAP_interaction_nonlinear",
    "EaseSHAP_size_player",
}

READOUT_METHOD_BY_BACKEND = {
    "EaseSHAP": "_crossfit_estimate",
    "RegressionMSR": "_run_kfold",
    "RegressionMSR_unbiased": "_run_kfold",
    "kernelSHAP": "_estimate",
    "kernelSHAP_paired": "_estimate",
    "PolySHAP_regression": "_run_polyshap_regression",
    "PolySHAP_regression_paired": "_run_polyshap_regression",
}


def alpha_label(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace(".", "p")


def eta_value(alpha: float) -> float:
    return float(alpha) ** 2


def eta_text(alpha: float) -> str:
    return f"{eta_value(alpha):g}"


def row_eta_text(row: dict[str, str]) -> str:
    if row.get("eta"):
        return row["eta"]
    if row.get("alpha"):
        return eta_text(float(row["alpha"]))
    raise KeyError("summary row must contain either 'eta' or legacy 'alpha'")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def setting_label(alpha: float, sv_spec: dict[str, Any]) -> str:
    return f"{alpha_label(alpha)}__{sv_spec['name']}"


def build_settings() -> list[dict[str, Any]]:
    settings = []
    for alpha in GAME_ALPHAS:
        for sv_spec in SEMIVALUE_SPECS:
            setting = dict(sv_spec)
            setting["alpha"] = float(alpha)
            setting["label"] = setting_label(alpha, sv_spec)
            settings.append(setting)
    return settings


def game_args(alpha: float) -> dict[str, Any]:
    return {
        "num_player": N,
        "alpha": float(alpha),
        "num_high_order": NUM_HIGH_ORDER,
        "sigma2": SIGMA2,
        "path": str(GAME_DIR),
        "seed": GAME_SEED,
    }


def groundtruth_path(setting: dict[str, Any]) -> Path:
    return GROUNDTRUTH_DIR / f"{setting['label']}.npy"


def run_path(setting: dict[str, Any], method_name: str, run_idx: int) -> Path:
    return RUNS_DIR / setting["label"] / f"{safe_name(method_name)}_run{run_idx:02d}.pkl"


def param_text(param: Any) -> str:
    if param is None:
        return ""
    return json.dumps(param)


def is_symmetric_semivalue(semivalue: str, semivalue_param: Any) -> bool:
    if semivalue == "shapley":
        return True
    if semivalue == "weighted_banzhaf":
        return abs(float(semivalue_param) - 0.5) < 1e-12
    if semivalue == "beta_shapley":
        a, b = semivalue_param
        return abs(float(a) - float(b)) < 1e-12
    return False


def is_compatible(method: dict[str, Any], setting: dict[str, Any]) -> bool:
    support = method.get("support", "all")
    semivalue = setting["semivalue"]
    semivalue_param = setting["semivalue_param"]

    if support == "all":
        return True
    if support == "shapley":
        return semivalue == "shapley"
    if support == "non_shapley":
        return semivalue != "shapley"
    if support == "symmetric":
        return is_symmetric_semivalue(semivalue, semivalue_param)
    if support == "ame":
        if semivalue == "weighted_banzhaf":
            return True
        if semivalue == "beta_shapley":
            a, b = semivalue_param
            return float(a) > 1.0 and float(b) > 1.0
        return False
    raise ValueError(f"Unknown support code {support!r} for method {method['name']!r}.")


def compatible_methods(setting: dict[str, Any], methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [method for method in methods if is_compatible(method, setting)]


def ensure_dirs() -> None:
    for path in (GROUNDTRUTH_DIR, RUNS_DIR, PLOTS_DIR, GAME_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUT / "run.log", mode="a"),
        ],
        force=True,
    )
    return logging.getLogger("sou_full_benchmark")


def validate_method_specs(methods: list[dict[str, Any]]) -> None:
    names = set()
    for method in methods:
        name = method["name"]
        if name in names:
            raise ValueError(f"Duplicate method name: {name}")
        names.add(name)
        backend = method["backend"]
        if not hasattr(estimator_module, backend):
            raise ValueError(f"Backend {backend!r} for method {name!r} is not available.")


def instrument_readout_timing(estimator: Any, backend: str) -> dict[str, Any]:
    timing = {
        "elapsed_sec": 0.0,
        "regular_elapsed_sec": 0.0,
        "readout_elapsed_sec": 0.0,
        "final_readout_elapsed_sec": 0.0,
        "readout_call_count": 0,
        "readout_methods": [],
    }
    method_name = READOUT_METHOD_BY_BACKEND.get(backend)
    if not method_name or not hasattr(estimator, method_name):
        return timing

    original = getattr(estimator, method_name)
    timing["readout_methods"] = [method_name]

    def timed_readout(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            timing["readout_elapsed_sec"] += elapsed
            timing["final_readout_elapsed_sec"] = elapsed
            timing["readout_call_count"] += 1

    setattr(estimator, method_name, timed_readout)
    return timing


@contextlib.contextmanager
def silence_worker_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)


def run_estimator(
    *,
    backend: str,
    estimator_seed: int,
    nue_avg: int,
    nue_track_avg: int,
    setting: dict[str, Any],
    estimator_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    estimator_args = dict(
        semivalue=setting["semivalue"],
        semivalue_param=setting["semivalue_param"],
        game_func=gameSOUStructuredGaussian,
        game_args=game_args(setting["alpha"]),
        num_player=N,
        nue_avg=int(nue_avg),
        nue_per_proc=min(int(nue_avg), 20_000),
        nue_track_avg=int(nue_track_avg),
        estimator_seed=int(estimator_seed),
    )
    start = time.perf_counter()
    estimator = getattr(estimator_module, backend)(**estimator_args, **estimator_kwargs)
    timing = instrument_readout_timing(estimator, backend)

    for samples in estimator.sampling():
        estimator.aggregate(estimator.run(samples))
    values_final, values_traj = estimator.finalize()

    elapsed = time.perf_counter() - start
    timing["elapsed_sec"] = elapsed
    timing["regular_elapsed_sec"] = max(0.0, elapsed - timing["readout_elapsed_sec"])
    return (
        np.asarray(values_final, dtype=np.float64),
        np.asarray(values_traj, dtype=np.float64),
        timing,
    )


def groundtruth_worker(setting: dict[str, Any], force: bool):
    path = groundtruth_path(setting)
    if path.exists() and not force:
        return setting["label"], "cached", 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    game = gameSOUStructuredGaussian(**game_args(setting["alpha"]))
    values_exact = game.get_semivalue(
        semivalue=setting["semivalue"],
        semivalue_param=setting["semivalue_param"],
    )
    np.save(path, np.asarray(values_exact, dtype=np.float64))
    return setting["label"], "computed", time.time() - start


def run_method_worker(
    setting: dict[str, Any],
    method: dict[str, Any],
    run_idx: int,
    seed: int,
    force: bool,
    allow_failures: bool,
):
    path = run_path(setting, method["name"], run_idx)
    if path.exists() and not force:
        return setting["label"], method["name"], run_idx, "cached", 0.0

    if not groundtruth_path(setting).exists():
        raise FileNotFoundError(f"Missing ground truth: {groundtruth_path(setting)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        with silence_worker_output():
            values_final, values_traj, timing = run_estimator(
                backend=method["backend"],
                estimator_seed=seed,
                nue_avg=RUN_NUE,
                nue_track_avg=TRACK_STEP,
                setting=setting,
                estimator_kwargs=method["estimator_kwargs"],
            )
        payload = {
            "status": "ok",
            "setting": setting,
            "algorithm": method["name"],
            "backend": method["backend"],
            "estimator_kwargs": method["estimator_kwargs"],
            "run_idx": int(run_idx),
            "seed": int(seed),
            "nue_budgets": NUE_BUDGETS.copy(),
            "values_final": values_final,
            "values_traj": values_traj,
            "elapsed": timing["elapsed_sec"],
            "timing": timing,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return setting["label"], method["name"], run_idx, "computed", payload["elapsed"]
    except Exception as exc:
        if not allow_failures:
            raise
        payload = {
            "status": "failed",
            "setting": setting,
            "algorithm": method["name"],
            "backend": method["backend"],
            "estimator_kwargs": method["estimator_kwargs"],
            "run_idx": int(run_idx),
            "seed": int(seed),
            "elapsed": time.time() - start,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return setting["label"], method["name"], run_idx, "failed", payload["elapsed"]


def run_parallel(tasks, worker, n_proc: int, logger: logging.Logger, label: str) -> None:
    if not tasks:
        logger.info("%s: nothing to do.", label)
        return
    if n_proc <= 1:
        for idx, args in enumerate(tasks, 1):
            logger.info("%s [%d/%d]: %s", label, idx, len(tasks), worker(*args))
        return
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futures = {pool.submit(worker, *args): args for args in tasks}
        for idx, fut in enumerate(as_completed(futures), 1):
            logger.info("%s [%d/%d]: %s", label, idx, len(tasks), fut.result())


def compute_summary_rows(
    settings: list[dict[str, Any]],
    methods: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rows = []
    error_rows = []
    timing_rows = []
    for setting in settings:
        truth_file = groundtruth_path(setting)
        if not truth_file.exists():
            continue
        truth = np.asarray(np.load(truth_file), dtype=np.float64)
        denom_sq = float(np.dot(truth, truth))
        if denom_sq <= 0.0:
            denom_sq = 1.0
        denom_l2 = float(np.sqrt(denom_sq))

        for method in compatible_methods(setting, methods):
            for run_idx in range(N_RUNS):
                path = run_path(setting, method["name"], run_idx)
                if not path.exists():
                    continue
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                timing = payload.get("timing") or {}
                elapsed_sec = float(timing.get("elapsed_sec", payload.get("elapsed", 0.0)))
                readout_elapsed_sec = float(timing.get("readout_elapsed_sec", 0.0))
                final_readout_elapsed_sec = float(timing.get("final_readout_elapsed_sec", 0.0))
                regular_elapsed_sec = float(timing.get("regular_elapsed_sec", max(0.0, elapsed_sec - readout_elapsed_sec)))
                timing_rows.append(
                    {
                        "setting": setting["label"],
                        "eta": eta_text(setting["alpha"]),
                        "semivalue_name": setting["name"],
                        "semivalue_title": setting["title"],
                        "semivalue": setting["semivalue"],
                        "semivalue_param": param_text(setting["semivalue_param"]),
                        "run_idx": str(run_idx),
                        "seed": str(payload.get("seed", "")),
                        "algorithm": method["name"],
                        "status": str(payload.get("status", "ok")),
                        "elapsed_sec": f"{elapsed_sec:.17g}",
                        "regular_elapsed_sec": f"{regular_elapsed_sec:.17g}",
                        "readout_elapsed_sec": f"{readout_elapsed_sec:.17g}",
                        "final_readout_elapsed_sec": f"{final_readout_elapsed_sec:.17g}",
                        "readout_call_count": str(int(timing.get("readout_call_count", 0))),
                        "readout_methods": ";".join(timing.get("readout_methods", [])),
                    }
                )
                if payload.get("status") != "ok":
                    error_rows.append(
                        {
                            "setting": setting["label"],
                            "eta": eta_text(setting["alpha"]),
                            "semivalue_name": setting["name"],
                            "algorithm": method["name"],
                            "run_idx": str(run_idx),
                            "seed": str(payload.get("seed", "")),
                            "elapsed_sec": f"{elapsed_sec:.6g}",
                            "error": str(payload.get("error", "")),
                        }
                    )
                    continue

                traj = np.asarray(payload["values_traj"], dtype=np.float64)
                budgets = np.asarray(payload.get("nue_budgets", NUE_BUDGETS), dtype=int)
                num_points = min(len(traj), len(budgets))
                for idx in range(num_points):
                    err = traj[idx] - truth
                    sq_error = float(np.dot(err, err))
                    l2_error = float(np.sqrt(sq_error))
                    rows.append(
                        {
                            "setting": setting["label"],
                            "eta": eta_text(setting["alpha"]),
                            "semivalue_name": setting["name"],
                            "semivalue_title": setting["title"],
                            "semivalue": setting["semivalue"],
                            "semivalue_param": param_text(setting["semivalue_param"]),
                            "run_idx": str(run_idx),
                            "seed": str(payload["seed"]),
                            "algorithm": method["name"],
                            "nue": str(int(budgets[idx])),
                            "sq_error": f"{sq_error:.17g}",
                            "rel_sq_error": f"{sq_error / denom_sq:.17g}",
                            "l2_error": f"{l2_error:.17g}",
                            "rel_l2_error": f"{l2_error / denom_l2:.17g}",
                            "rmse": f"{float(np.sqrt(np.mean(err * err))):.17g}",
                            "elapsed_sec": f"{elapsed_sec:.6g}",
                            "regular_elapsed_sec": f"{regular_elapsed_sec:.6g}",
                            "readout_elapsed_sec": f"{readout_elapsed_sec:.6g}",
                            "final_readout_elapsed_sec": f"{final_readout_elapsed_sec:.6g}",
                        }
                    )
    return rows, error_rows, timing_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    logger: logging.Logger,
    settings: list[dict[str, Any]],
    methods: list[dict[str, Any]],
) -> list[dict[str, str]]:
    rows, error_rows, timing_rows = compute_summary_rows(settings, methods)
    summary_fields = [
        "setting",
        "eta",
        "semivalue_name",
        "semivalue_title",
        "semivalue",
        "semivalue_param",
        "run_idx",
        "seed",
        "algorithm",
        "nue",
        "sq_error",
        "rel_sq_error",
        "l2_error",
        "rel_l2_error",
        "rmse",
        "elapsed_sec",
        "regular_elapsed_sec",
        "readout_elapsed_sec",
        "final_readout_elapsed_sec",
    ]
    write_csv(OUT / "summary.csv", rows, summary_fields)
    logger.info("summary: wrote %d rows to %s", len(rows), OUT / "summary.csv")

    timing_fields = [
        "setting",
        "eta",
        "semivalue_name",
        "semivalue_title",
        "semivalue",
        "semivalue_param",
        "run_idx",
        "seed",
        "algorithm",
        "status",
        "elapsed_sec",
        "regular_elapsed_sec",
        "readout_elapsed_sec",
        "final_readout_elapsed_sec",
        "readout_call_count",
        "readout_methods",
    ]
    write_csv(OUT / "run_timing.csv", timing_rows, timing_fields)
    logger.info("summary: wrote %d timing rows to %s", len(timing_rows), OUT / "run_timing.csv")

    error_fields = ["setting", "eta", "semivalue_name", "algorithm", "run_idx", "seed", "elapsed_sec", "error"]
    write_csv(OUT / "errors.csv", error_rows, error_fields)
    if error_rows:
        logger.info("summary: wrote %d error rows to %s", len(error_rows), OUT / "errors.csv")
    return rows


def write_matrix(
    path: Path,
    row_keys: list[str],
    col_keys: list[str],
    value_fn,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting"] + col_keys)
        for row_key in row_keys:
            writer.writerow([row_key] + [value_fn(row_key, col_key) for col_key in col_keys])


def write_aucc(
    logger: logging.Logger,
    rows: list[dict[str, str]],
    settings: list[dict[str, Any]],
    methods: list[dict[str, Any]],
) -> None:
    by_run: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["setting"], row["algorithm"], row["run_idx"])
        by_run[key].append(float(row["rel_l2_error"]))

    by_setting_alg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (setting_label_cur, algorithm, _run_idx), values in by_run.items():
        if values:
            by_setting_alg[(setting_label_cur, algorithm)].append(float(np.mean(values)))

    row_keys = [setting["label"] for setting in settings]
    col_keys = [method["name"] for method in methods]

    def mean_value(row_key: str, col_key: str) -> str:
        vals = by_setting_alg.get((row_key, col_key), [])
        return "" if not vals else f"{float(np.mean(vals)):.17g}"

    def std_value(row_key: str, col_key: str) -> str:
        vals = by_setting_alg.get((row_key, col_key), [])
        return "" if not vals else f"{float(np.std(vals)):.17g}"

    def count_value(row_key: str, col_key: str) -> str:
        vals = by_setting_alg.get((row_key, col_key), [])
        return "" if not vals else str(len(vals))

    def summary_value(row_key: str, col_key: str) -> str:
        vals = by_setting_alg.get((row_key, col_key), [])
        if not vals:
            return ""
        return f"{float(np.mean(vals)):.6g}+/-{float(np.std(vals)):.3g} (n={len(vals)})"

    write_matrix(OUT / "aucc_mean.csv", row_keys, col_keys, mean_value)
    write_matrix(OUT / "aucc_std.csv", row_keys, col_keys, std_value)
    write_matrix(OUT / "aucc_count.csv", row_keys, col_keys, count_value)
    write_matrix(OUT / "aucc_summary.csv", row_keys, col_keys, summary_value)

    for sv_spec in SEMIVALUE_SPECS:
        sv_settings = [setting for setting in settings if setting["name"] == sv_spec["name"]]
        sv_row_keys = [setting["label"] for setting in sv_settings]
        if sv_row_keys:
            write_matrix(OUT / f"aucc_mean_{sv_spec['name']}.csv", sv_row_keys, col_keys, mean_value)
            write_matrix(OUT / f"aucc_std_{sv_spec['name']}.csv", sv_row_keys, col_keys, std_value)

    logger.info("aucc: wrote AUCC tables to %s", OUT)


def plot_summary(
    logger: logging.Logger,
    rows: list[dict[str, str]],
    settings: list[dict[str, Any]],
    methods: list[dict[str, Any]],
) -> None:
    if not rows:
        logger.info("plots: no summary rows available.")
        return

    color_map = plt.get_cmap("tab20")
    colors = {method["name"]: color_map(i % 20) for i, method in enumerate(methods)}

    row_lookup: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["semivalue_name"],
            row_eta_text(row),
            row["algorithm"],
            int(row["nue"]),
        )
        row_lookup[key].append(float(row["rel_l2_error"]))

    selected_setting_labels = {setting["label"] for setting in settings}
    selected_sv_names = list(dict.fromkeys(setting["name"] for setting in settings))
    selected_eta_labels = [eta_text(alpha) for alpha in GAME_ALPHAS]
    available_sv_names = {row["semivalue_name"] for row in rows}

    for sv_spec in SEMIVALUE_SPECS:
        if sv_spec["name"] not in selected_sv_names:
            continue
        if sv_spec["name"] not in available_sv_names:
            continue

        fig, axes = plt.subplots(1, len(GAME_ALPHAS), figsize=(5.2 * len(GAME_ALPHAS), 4.2), sharey=True)
        if len(GAME_ALPHAS) == 1:
            axes = [axes]

        plotted_algorithms = []
        for ax, alpha, eta_str in zip(axes, GAME_ALPHAS, selected_eta_labels):
            setting_cur = setting_label(alpha, sv_spec)
            if setting_cur not in selected_setting_labels:
                ax.set_visible(False)
                continue

            for method in methods:
                if not is_compatible(method, {**sv_spec, "alpha": alpha, "label": setting_cur}):
                    continue
                xs = []
                means = []
                stds = []
                for nue in NUE_BUDGETS:
                    vals = row_lookup.get((sv_spec["name"], eta_str, method["name"], int(nue)), [])
                    if not vals:
                        continue
                    xs.append(int(nue))
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                if not xs:
                    continue

                xs_arr = np.asarray(xs, dtype=int)
                means_arr = np.maximum(np.asarray(means, dtype=float), 1e-16)
                stds_arr = np.asarray(stds, dtype=float)
                lower = np.maximum(means_arr - stds_arr, 1e-16)
                upper = np.maximum(means_arr + stds_arr, 1e-16)
                ax.plot(xs_arr, means_arr, linewidth=1.2, label=method["name"], color=colors[method["name"]])
                ax.fill_between(xs_arr, lower, upper, alpha=0.13, color=colors[method["name"]])
                if method["name"] not in plotted_algorithms:
                    plotted_algorithms.append(method["name"])

            ax.set_title(rf"$\eta = {eta_str}$", fontsize=10)
            ax.set_xlabel("NUE per player", fontsize=9)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel("Relative L2 error", fontsize=9)
        handles = [
            plt.Line2D([0], [0], color=colors[name], linewidth=1.5, label=name)
            for name in plotted_algorithms
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(4, len(handles)),
                bbox_to_anchor=(0.5, -0.24),
                fontsize=7,
                frameon=True,
            )

        fig.suptitle(f"{sv_spec['title']}, mean +/- 1 std over {N_RUNS} runs", fontsize=12, y=1.02)
        fig.tight_layout()
        path = PLOTS_DIR / f"{sv_spec['name']}_rel_l2.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info("plots: wrote %s", path)


def select_aucc_methods(rows: list[dict[str, str]], requested_methods: list[str] | None) -> list[str]:
    available = list(dict.fromkeys(row["algorithm"] for row in rows))
    if requested_methods:
        missing = [method for method in requested_methods if method not in available]
        if missing:
            raise ValueError(f"Methods not found in summary rows: {missing}")
        return requested_methods
    visible_available = [method for method in available if method not in AUCC_HIDDEN_BY_DEFAULT]
    ordered = [method for method in AUCC_DEFAULT_METHOD_ORDER if method in visible_available]
    ordered.extend(method for method in visible_available if method not in ordered)
    return ordered


def compute_plot_aucc(
    rows: list[dict[str, str]],
    metric: str = "rel_l2_error",
    aucc_mode: str = "mean",
) -> dict[tuple[str, str, str], list[float]]:
    by_run: dict[tuple[str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if not row.get(metric):
            continue
        key = (row_eta_text(row), row["semivalue_name"], row["algorithm"], row["run_idx"])
        by_run[key].append((int(row["nue"]), float(row[metric])))

    by_setting_method: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (eta, semivalue, algorithm, _run_idx), curve in by_run.items():
        curve_sorted = sorted(curve)
        xs = np.asarray([point[0] for point in curve_sorted], dtype=float)
        ys = np.asarray([point[1] for point in curve_sorted], dtype=float)
        if len(ys) == 0:
            continue
        if aucc_mode == "trapz" and len(ys) > 1 and xs[-1] > xs[0]:
            aucc = float(np.trapz(ys, xs) / (xs[-1] - xs[0]))
        else:
            aucc = float(np.mean(ys))
        by_setting_method[(eta, semivalue, algorithm)].append(aucc)
    return by_setting_method


def style_for_aucc_methods(methods: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
    color_cycle = list(plt.get_cmap("tab20").colors)
    colors = {method: color_cycle[idx % len(color_cycle)] for idx, method in enumerate(methods)}
    if "EaseSHAP_interaction_nonlinear" in colors:
        colors["EaseSHAP_interaction_nonlinear"] = "#d62728"
    if "EaseSHAP_size_player" in colors:
        colors["EaseSHAP_size_player"] = "#1f77b4"

    markers_base = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*", "8", "H", "d", "1", "2", "3", "4"]
    markers = {method: markers_base[idx % len(markers_base)] for idx, method in enumerate(methods)}
    return colors, markers


def write_aucc_panel_figure(
    rows: list[dict[str, str]],
    requested_methods: list[str] | None,
    eta_labels: list[str],
    out_dir: Path,
    output_stem: str,
    figsize: tuple[float, float],
    dpi: int = 300,
    legend_position: str = "bottom",
    font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    show_title: bool = True,
) -> Path | None:
    if not rows:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    methods = select_aucc_methods(rows, requested_methods)
    if not methods:
        return None

    by_setting_method = compute_plot_aucc(rows)
    colors, markers = style_for_aucc_methods(methods)
    x_base = np.arange(len(AUCC_SEMIVALUE_ORDER), dtype=float)
    jitter = 0.08
    offsets = np.linspace(-jitter, jitter, len(methods)) if len(methods) > 1 else np.zeros(len(methods))

    fig, axes = plt.subplots(1, len(eta_labels), figsize=figsize, sharey=False)
    if len(eta_labels) == 1:
        axes = [axes]

    handles = []
    for method in methods:
        label = AUCC_METHOD_LABELS.get(method, method)
        linewidth = 1.6 if method in AUCC_EASE_METHODS else 1.05
        alpha_line = 0.95 if method in AUCC_EASE_METHODS else 0.68
        handle = plt.Line2D(
            [0],
            [0],
            color=colors[method],
            marker=markers[method],
            linewidth=linewidth,
            markersize=4.3,
            label=label,
            alpha=alpha_line,
        )
        handles.append(handle)

    positive_values = []
    for vals in by_setting_method.values():
        positive_values.extend([val for val in vals if val > 0])
    min_positive = min(positive_values) if positive_values else 1e-8

    for ax, eta_label in zip(axes, eta_labels):
        for method_idx, method in enumerate(methods):
            xs = []
            means = []
            stds = []
            for semivalue_idx, (semivalue, _label) in enumerate(AUCC_SEMIVALUE_ORDER):
                vals = by_setting_method.get((eta_label, semivalue, method), [])
                if not vals:
                    continue
                xs.append(x_base[semivalue_idx] + offsets[method_idx])
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))

            if not xs:
                continue

            xs_arr = np.asarray(xs, dtype=float)
            means_arr = np.asarray(means, dtype=float)
            stds_arr = np.asarray(stds, dtype=float)
            lower = np.maximum(means_arr - stds_arr, min_positive / 5.0)
            upper = np.maximum(means_arr + stds_arr, min_positive / 5.0)
            linewidth = 1.6 if method in AUCC_EASE_METHODS else 1.05
            markersize = 4.3 if method in AUCC_EASE_METHODS else 3.3
            alpha_line = 0.95 if method in AUCC_EASE_METHODS else 0.68

            if len(xs_arr) > 1:
                ax.plot(
                    xs_arr,
                    means_arr,
                    color=colors[method],
                    marker=markers[method],
                    linewidth=linewidth,
                    markersize=markersize,
                    alpha=alpha_line,
                    zorder=3 if method in AUCC_EASE_METHODS else 2,
                )
                ax.fill_between(
                    xs_arr,
                    lower,
                    upper,
                    color=colors[method],
                    alpha=0.22 if method in AUCC_EASE_METHODS else 0.14,
                    linewidth=0,
                    zorder=1,
                )
            else:
                ax.errorbar(
                    xs_arr,
                    means_arr,
                    yerr=stds_arr,
                    fmt=markers[method],
                    color=colors[method],
                    markersize=markersize,
                    alpha=alpha_line,
                    capsize=2.2,
                    linewidth=linewidth,
                    zorder=3 if method in AUCC_EASE_METHODS else 2,
                )

        if show_title:
            ax.set_title(rf"$\eta = {eta_label}$", fontsize=10.5 * font_scale, pad=7)
        ax.set_xticks(x_base)
        ax.set_xticklabels(
            [label for _name, label in AUCC_SEMIVALUE_ORDER],
            rotation=28,
            ha="right",
            fontsize=10.0 * font_scale,
        )
        ax.set_yscale("log")
        ax.tick_params(axis="y", which="major", labelsize=10.0 * font_scale)
        ax.tick_params(axis="y", which="minor", labelsize=8.0 * font_scale)
        ax.grid(True, axis="y", which="major", color="#d9d9d9", linewidth=0.65)
        ax.grid(True, axis="y", which="minor", color="#eeeeee", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.07)

    axes[0].set_ylabel("AUCC", fontsize=12.0 * font_scale)
    if legend_position == "right":
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.79, 0.5),
            ncol=1,
            frameon=False,
            fontsize=7.4 * font_scale * legend_font_scale,
            handlelength=1.9,
            columnspacing=1.1,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0), w_pad=1.0)
    elif legend_position == "bottom":
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=min(5, len(handles)),
            frameon=False,
            fontsize=7.4 * font_scale * legend_font_scale,
            handlelength=1.9,
            columnspacing=1.1,
        )
        fig.tight_layout(rect=(0.0, 0.285, 1.0, 1.0), w_pad=1.0)
    else:
        raise ValueError(f"Unknown legend_position: {legend_position}")

    path = out_dir / f"{output_stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_aucc_figure(
    logger: logging.Logger,
    rows: list[dict[str, str]],
    requested_methods: list[str] | None,
) -> None:
    if not rows:
        logger.info("plots: no summary rows available for AUCC plot.")
        return

    path = write_aucc_panel_figure(
        rows=rows,
        requested_methods=requested_methods,
        eta_labels=[eta_text(alpha) for alpha in GAME_ALPHAS],
        out_dir=PLOTS_DIR,
        output_stem="aucc_by_eta",
        figsize=(8.8, 3.65),
    )
    if path is None:
        logger.info("plots: no methods available for AUCC plot.")
        return
    logger.info("plots: wrote %s", path)


def write_config(settings: list[dict[str, Any]], methods: list[dict[str, Any]]) -> None:
    ensure_dirs()
    config = {
        "n": N,
        "game": "gameSOUStructuredGaussian",
        "game_seed": GAME_SEED,
        "base_seed": BASE_SEED,
        "run_seeds": RUN_SEEDS,
        "n_runs": N_RUNS,
        "game_alphas": GAME_ALPHAS,
        "game_etas": [float(eta_text(alpha)) for alpha in GAME_ALPHAS],
        "num_high_order": NUM_HIGH_ORDER,
        "sigma2": SIGMA2,
        "run_nue": RUN_NUE,
        "track_step": TRACK_STEP,
        "nue_budgets": NUE_BUDGETS.tolist(),
        "semivalues": SEMIVALUE_SPECS,
        "settings": settings,
        "groundtruth_backend": "analytic_sou",
        "methods": methods,
        "notes": {
            "coupsamp": "Excluded by request.",
            "easeshap_pairing": "use_complement_sampling=True; EaseSHAP pairs only symmetric semivalues.",
            "easeshap_regularization": {
                "surrogate_ridge_lambda": EASESHAP_COMMON_KWARGS["surrogate_ridge_lambda"],
                "surrogate_ridge_schedule": EASESHAP_COMMON_KWARGS["surrogate_ridge_schedule"],
            },
        },
    }
    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["all", "groundtruth", "runs", "summary", "plots"],
        default="all",
        help="Which phase to run.",
    )
    parser.add_argument("--n-proc", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Recompute existing groundtruth/run files.")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Record method failures in errors.csv instead of stopping the run phase.",
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        default=None,
        help="Optional setting labels to run, e.g. alpha_0p25__shapley.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method names to run, e.g. EaseSHAP_size_player OFA_fixed.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting in phase=all.")
    return parser.parse_args()


def select_settings(all_settings: list[dict[str, Any]], requested: list[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return all_settings
    by_label = {setting["label"]: setting for setting in all_settings}
    missing = [label for label in requested if label not in by_label]
    if missing:
        raise ValueError(f"Unknown settings {missing}. Available: {sorted(by_label)}")
    return [by_label[label] for label in requested]


def select_methods(all_methods: list[dict[str, Any]], requested: list[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return all_methods
    by_name = {method["name"]: method for method in all_methods}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown methods {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in requested]


def main() -> None:
    args = parse_args()
    ensure_dirs()
    validate_method_specs(METHOD_SPECS)

    all_settings = build_settings()
    settings = select_settings(all_settings, args.settings)
    methods = select_methods(METHOD_SPECS, args.methods)
    write_config(settings, methods)

    logger = setup_logging()
    logger.info("output: %s", OUT)
    logger.info("game: n=%d, alphas=%s, num_high_order=%d, sigma2=%s", N, GAME_ALPHAS, NUM_HIGH_ORDER, SIGMA2)
    logger.info("semivalues: %s", [spec["name"] for spec in SEMIVALUE_SPECS])
    logger.info("methods: %s", [method["name"] for method in methods])
    logger.info("budgets: run_nue=%d, track_step=%d, n_runs=%d", RUN_NUE, TRACK_STEP, N_RUNS)
    logger.info("n_proc=%d, force=%s, allow_failures=%s", args.n_proc, args.force, args.allow_failures)

    if args.phase in {"all", "groundtruth"}:
        tasks = [(setting, args.force) for setting in settings]
        # Ground truth is analytic and fast. Run it serially so semivalue tasks
        # sharing the same game alpha do not concurrently write the same game
        # cache file.
        run_parallel(tasks, groundtruth_worker, 1, logger, "groundtruth")

    if args.phase in {"all", "runs"}:
        missing_truth = [setting["label"] for setting in settings if not groundtruth_path(setting).exists()]
        if missing_truth:
            raise RuntimeError(f"Missing ground truth for settings: {missing_truth}")

        tasks = []
        for setting in settings:
            for method in compatible_methods(setting, methods):
                for run_idx, seed in enumerate(RUN_SEEDS):
                    path = run_path(setting, method["name"], run_idx)
                    if args.force or not path.exists():
                        tasks.append((setting, method, run_idx, seed, args.force, args.allow_failures))
        total_possible = sum(len(compatible_methods(setting, methods)) * N_RUNS for setting in settings)
        logger.info("runs: %d/%d already complete; submitting %d tasks.", total_possible - len(tasks), total_possible, len(tasks))
        run_parallel(tasks, run_method_worker, args.n_proc, logger, "runs")

    rows: list[dict[str, str]] = []
    if args.phase in {"all", "summary"}:
        rows = write_summary(logger, settings, methods)
        write_aucc(logger, rows, settings, methods)

    if args.phase in {"all", "plots"} and not args.no_plots:
        if not rows:
            rows, _, _ = compute_summary_rows(settings, methods)
        plot_summary(logger, rows, settings, methods)
        plot_aucc_figure(logger, rows, args.methods)


if __name__ == "__main__":
    main()
