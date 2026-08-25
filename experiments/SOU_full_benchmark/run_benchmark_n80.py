"""
Structured Gaussian SOU full benchmark.

This benchmark uses the same n=80 gameSOUStructuredGaussianBitset setup as
paper/experiments/SOU_comparison, then evaluates the non-CoupSamp methods from
try7/try8 together with EaseSHAP, RegressionMSR, and LeverageSHAP.

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
import resource
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
N = 80
DEFAULT_RESULT_TAG = f"n{N}"
RESULT_TAG = os.environ.get("SOU_RESULT_TAG", DEFAULT_RESULT_TAG).strip()
if not RESULT_TAG or re.fullmatch(r"[A-Za-z0-9._-]+", RESULT_TAG) is None:
    raise ValueError(
        "SOU_RESULT_TAG must contain only letters, digits, '.', '_', or '-'; "
        f"got {RESULT_TAG!r}."
    )
OUT = SCRIPT_DIR / "results" / RESULT_TAG
GROUNDTRUTH_DIR = OUT / "groundtruth"
RUNS_DIR = OUT / "runs"
GAME_DIR = OUT / "game"

# Keep imported third-party libraries from writing outside the run directory or
# oversubscribing shared machines.
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

import numpy as np


PAPER_DIR = SCRIPT_DIR.parents[1]
PACKAGE_ROOT = PAPER_DIR / "EaseSHAP"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from easeshap.utilityFuncs import gameSOUStructuredGaussianBitset
import easeshap.registry as estimator_module


GAME_SEED = 42
BASE_SEED = 2026
N_RUNS = 10
RUN_SEEDS = [BASE_SEED + i * 137 for i in range(N_RUNS)]

GAME_ALPHAS = [0.25**0.5, 0.5**0.5, 0.75**0.5]
NUM_HIGH_ORDER = N ** 2
SIGMA2 = None

TOTAL_NUE = 160_000
NUM_CHECKPOINTS = 50
if TOTAL_NUE % N != 0:
    raise ValueError(f"TOTAL_NUE={TOTAL_NUE} must be divisible by N={N}.")
RUN_NUE = TOTAL_NUE // N
if RUN_NUE % NUM_CHECKPOINTS != 0:
    raise ValueError(f"RUN_NUE={RUN_NUE} must be divisible by NUM_CHECKPOINTS={NUM_CHECKPOINTS}.")
TRACK_STEP = RUN_NUE // NUM_CHECKPOINTS
NUE_BUDGETS = np.arange(TRACK_STEP, RUN_NUE + 1, TRACK_STEP, dtype=int)
TOTAL_NUE_BUDGETS = NUE_BUDGETS * N

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
    "boundary_policy": "fixed",
    "boundary_order": 1,
    "use_complement_sampling": True,
    "surrogate_ridge_lambda": 0.01,
    "surrogate_ridge_schedule": "times_m",
    "num_folds": 2,
    "surrogate_readout_mode": "crossfit",
}

EASESHAP_SIZE_PLAYER_RIDGE_KWARGS = {
    "surrogate_ridge_lambda": 1.0,
    "surrogate_ridge_schedule": "fixed",
    "surrogate_ridge_scaling": "size_trace",
}

METHOD_SPECS = [
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
            "pilot_design_updates": 1,
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
    },
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
            "num_folds": 2,
        },
    },
]

READOUT_METHOD_BY_BACKEND = {
    "EaseSHAP": "_engine._readout_estimate",
    "RegressionMSR": "_run_kfold",
    "RegressionMSR_unbiased": "_run_kfold",
    "kernelSHAP": "_estimate",
    "kernelSHAP_paired": "_estimate",
    "LeverageSHAP": "_estimate",
    "LeverageSHAP_original": "_estimate",
}


def current_rss_mb() -> float:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm") as f:
                resident_pages = int(f.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
        except (OSError, IndexError, ValueError):
            pass
    return peak_rss_mb()


def peak_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def get_nested_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def set_nested_attr(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


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


def build_settings(max_high_order_size: int | None = None) -> list[dict[str, Any]]:
    settings = []
    for alpha in GAME_ALPHAS:
        for sv_spec in SEMIVALUE_SPECS:
            setting = dict(sv_spec)
            setting["alpha"] = float(alpha)
            setting["max_high_order_size"] = max_high_order_size
            setting["label"] = setting_label(alpha, sv_spec)
            settings.append(setting)
    return settings


def game_args(alpha: float, max_high_order_size: int | None = None) -> dict[str, Any]:
    return {
        "num_player": N,
        "alpha": float(alpha),
        "num_high_order": NUM_HIGH_ORDER,
        "max_high_order_size": max_high_order_size,
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
    for path in (GROUNDTRUTH_DIR, RUNS_DIR, GAME_DIR):
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
        "readout_hook_status": "not_configured",
    }
    method_path = READOUT_METHOD_BY_BACKEND.get(backend)
    if not method_path:
        return timing

    try:
        original = get_nested_attr(estimator, method_path)
    except AttributeError:
        timing["readout_hook_status"] = f"missing:{method_path}"
        return timing
    if not callable(original):
        timing["readout_hook_status"] = f"not_callable:{method_path}"
        return timing

    timing["readout_methods"] = [method_path]
    timing["readout_hook_status"] = f"wrapped:{method_path}"

    def timed_readout(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            timing["readout_elapsed_sec"] += elapsed
            timing["final_readout_elapsed_sec"] = elapsed
            timing["readout_call_count"] += 1

    set_nested_attr(estimator, method_path, timed_readout)
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
        game_func=gameSOUStructuredGaussianBitset,
        game_args=game_args(setting["alpha"], setting.get("max_high_order_size")),
        num_player=N,
        nue_avg=int(nue_avg),
        nue_per_proc=min(int(nue_avg), 20_000),
        nue_track_avg=int(nue_track_avg),
        estimator_seed=int(estimator_seed),
    )

    total_start = time.perf_counter()
    rss_mb_start = current_rss_mb()
    peak_rss_mb_start = peak_rss_mb()

    setup_start = time.perf_counter()
    estimator = getattr(estimator_module, backend)(**estimator_args, **estimator_kwargs)
    setup_sec = time.perf_counter() - setup_start
    timing = instrument_readout_timing(estimator, backend)

    sampling_sec = 0.0
    utility_eval_sec = 0.0
    aggregate_sec = 0.0

    sampling_iter = iter(estimator.sampling())
    while True:
        phase_start = time.perf_counter()
        try:
            samples = next(sampling_iter)
        except StopIteration:
            sampling_sec += time.perf_counter() - phase_start
            break
        sampling_sec += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        results = estimator.run(samples)
        utility_eval_sec += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        estimator.aggregate(results)
        aggregate_sec += time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    values_final, values_traj = estimator.finalize()
    finalize_sec = time.perf_counter() - phase_start

    elapsed = time.perf_counter() - total_start
    timing["elapsed_sec"] = elapsed
    timing["regular_elapsed_sec"] = max(0.0, elapsed - timing["readout_elapsed_sec"])
    timing["setup_sec"] = setup_sec
    timing["sampling_sec"] = sampling_sec
    timing["utility_eval_sec"] = utility_eval_sec
    timing["aggregate_sec"] = aggregate_sec
    timing["finalize_sec"] = finalize_sec
    timing["rss_mb_start"] = rss_mb_start
    timing["rss_mb_end"] = current_rss_mb()
    timing["peak_rss_mb_start"] = peak_rss_mb_start
    timing["peak_rss_mb"] = peak_rss_mb()
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
    game = gameSOUStructuredGaussianBitset(
        **game_args(setting["alpha"], setting.get("max_high_order_size"))
    )
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
            "total_nue_budgets": TOTAL_NUE_BUDGETS.copy(),
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
                setup_sec = float(timing.get("setup_sec", 0.0))
                sampling_sec = float(timing.get("sampling_sec", 0.0))
                utility_eval_sec = float(timing.get("utility_eval_sec", 0.0))
                aggregate_sec = float(timing.get("aggregate_sec", 0.0))
                finalize_sec = float(timing.get("finalize_sec", 0.0))
                rss_mb_start = float(timing.get("rss_mb_start", 0.0))
                rss_mb_end = float(timing.get("rss_mb_end", 0.0))
                peak_rss_mb_start = float(timing.get("peak_rss_mb_start", 0.0))
                peak_rss_mb = float(timing.get("peak_rss_mb", 0.0))
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
                        "setup_sec": f"{setup_sec:.17g}",
                        "sampling_sec": f"{sampling_sec:.17g}",
                        "utility_eval_sec": f"{utility_eval_sec:.17g}",
                        "aggregate_sec": f"{aggregate_sec:.17g}",
                        "finalize_sec": f"{finalize_sec:.17g}",
                        "readout_elapsed_sec": f"{readout_elapsed_sec:.17g}",
                        "final_readout_elapsed_sec": f"{final_readout_elapsed_sec:.17g}",
                        "readout_call_count": str(int(timing.get("readout_call_count", 0))),
                        "readout_methods": ";".join(timing.get("readout_methods", [])),
                        "readout_hook_status": str(timing.get("readout_hook_status", "")),
                        "rss_mb_start": f"{rss_mb_start:.17g}",
                        "rss_mb_end": f"{rss_mb_end:.17g}",
                        "peak_rss_mb_start": f"{peak_rss_mb_start:.17g}",
                        "peak_rss_mb": f"{peak_rss_mb:.17g}",
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
                total_budgets = np.asarray(payload.get("total_nue_budgets", budgets * N), dtype=int)
                num_points = min(len(traj), len(budgets), len(total_budgets))
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
                            "total_nue": str(int(total_budgets[idx])),
                            "sq_error": f"{sq_error:.17g}",
                            "rel_sq_error": f"{sq_error / denom_sq:.17g}",
                            "l2_error": f"{l2_error:.17g}",
                            "rel_l2_error": f"{l2_error / denom_l2:.17g}",
                            "rmse": f"{float(np.sqrt(np.mean(err * err))):.17g}",
                            "elapsed_sec": f"{elapsed_sec:.6g}",
                            "regular_elapsed_sec": f"{regular_elapsed_sec:.6g}",
                            "setup_sec": f"{setup_sec:.6g}",
                            "sampling_sec": f"{sampling_sec:.6g}",
                            "utility_eval_sec": f"{utility_eval_sec:.6g}",
                            "aggregate_sec": f"{aggregate_sec:.6g}",
                            "finalize_sec": f"{finalize_sec:.6g}",
                            "readout_elapsed_sec": f"{readout_elapsed_sec:.6g}",
                            "final_readout_elapsed_sec": f"{final_readout_elapsed_sec:.6g}",
                            "rss_mb_start": f"{rss_mb_start:.6g}",
                            "rss_mb_end": f"{rss_mb_end:.6g}",
                            "peak_rss_mb": f"{peak_rss_mb:.6g}",
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
        "total_nue",
        "sq_error",
        "rel_sq_error",
        "l2_error",
        "rel_l2_error",
        "rmse",
        "elapsed_sec",
        "regular_elapsed_sec",
        "setup_sec",
        "sampling_sec",
        "utility_eval_sec",
        "aggregate_sec",
        "finalize_sec",
        "readout_elapsed_sec",
        "final_readout_elapsed_sec",
        "rss_mb_start",
        "rss_mb_end",
        "peak_rss_mb",
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
        "setup_sec",
        "sampling_sec",
        "utility_eval_sec",
        "aggregate_sec",
        "finalize_sec",
        "readout_elapsed_sec",
        "final_readout_elapsed_sec",
        "readout_call_count",
        "readout_methods",
        "readout_hook_status",
        "rss_mb_start",
        "rss_mb_end",
        "peak_rss_mb_start",
        "peak_rss_mb",
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


def write_config(
    settings: list[dict[str, Any]],
    methods: list[dict[str, Any]],
    *,
    ease_fo_pilot_design_updates: int,
    ease_sp_pilot_design_updates: int,
    max_high_order_size: int | None,
) -> None:
    ensure_dirs()
    config = {
        "n": N,
        "game": "gameSOUStructuredGaussianBitset",
        "game_seed": GAME_SEED,
        "base_seed": BASE_SEED,
        "run_seeds": RUN_SEEDS,
        "n_runs": N_RUNS,
        "game_alphas": GAME_ALPHAS,
        "game_etas": [float(eta_text(alpha)) for alpha in GAME_ALPHAS],
        "num_high_order": NUM_HIGH_ORDER,
        "max_high_order_size": max_high_order_size,
        "sigma2": SIGMA2,
        "total_nue": TOTAL_NUE,
        "num_checkpoints": NUM_CHECKPOINTS,
        "run_nue": RUN_NUE,
        "track_step": TRACK_STEP,
        "nue_budgets": NUE_BUDGETS.tolist(),
        "total_nue_budgets": TOTAL_NUE_BUDGETS.tolist(),
        "semivalues": SEMIVALUE_SPECS,
        "settings": settings,
        "groundtruth_backend": "analytic_sou",
        "ease_fo_pilot_design_updates": int(ease_fo_pilot_design_updates),
        "ease_sp_pilot_design_updates": int(ease_sp_pilot_design_updates),
        "methods": methods,
        "notes": {
            "coupsamp": "Excluded by request.",
            "easeshap_pairing": "use_complement_sampling=True; EaseSHAP pairs only symmetric semivalues.",
            "easeshap_boundary_handling": {
                "exact_boundary_handling": EASESHAP_COMMON_KWARGS["exact_boundary_handling"],
                "boundary_policy": EASESHAP_COMMON_KWARGS["boundary_policy"],
                "boundary_order": EASESHAP_COMMON_KWARGS["boundary_order"],
            },
            "easeshap_common_regularization": {
                "surrogate_ridge_lambda": EASESHAP_COMMON_KWARGS["surrogate_ridge_lambda"],
                "surrogate_ridge_schedule": EASESHAP_COMMON_KWARGS["surrogate_ridge_schedule"],
                "surrogate_ridge_scaling": "scalar",
            },
            "easeshap_size_player_regularization": {
                "surrogate_ridge_lambda": EASESHAP_SIZE_PLAYER_RIDGE_KWARGS["surrogate_ridge_lambda"],
                "surrogate_ridge_schedule": EASESHAP_SIZE_PLAYER_RIDGE_KWARGS["surrogate_ridge_schedule"],
                "surrogate_ridge_scaling": EASESHAP_SIZE_PLAYER_RIDGE_KWARGS["surrogate_ridge_scaling"],
            },
            "runtime_recording": (
                "run_timing.csv reports total, setup, sampling, utility_eval, aggregate, finalize, "
                "instrumented readout time, current RSS, and process peak RSS. In parallel runs, "
                "process peak RSS is a worker high-water mark and can include earlier tasks."
            ),
        },
    }
    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["all", "groundtruth", "runs", "summary"],
        default="all",
        help="Which phase to run.",
    )
    parser.add_argument("--n-proc", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument(
        "--ease-fo-pilot-design-updates",
        type=int,
        default=1,
        help="Number of fixed-pilot design updates for EASE-FO. Default: 1.",
    )
    parser.add_argument(
        "--ease-sp-pilot-design-updates",
        type=int,
        default=1,
        help="Number of fixed-pilot design updates for EASE-SP. Default: 1.",
    )
    parser.add_argument(
        "--max-high-order-size",
        type=int,
        default=None,
        help=(
            "Inclusive maximum size of sampled high-order SOU interactions. "
            "Omit it, or use a value at least n-1, for the legacy range."
        ),
    )
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
    args = parser.parse_args()
    if args.ease_fo_pilot_design_updates < 1:
        parser.error("--ease-fo-pilot-design-updates must be at least 1.")
    if args.ease_sp_pilot_design_updates < 1:
        parser.error("--ease-sp-pilot-design-updates must be at least 1.")
    if args.max_high_order_size is not None and args.max_high_order_size < 3:
        parser.error("--max-high-order-size must be at least 3 when specified.")
    return args


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


def configure_ease_pilot_design_updates(
    methods: list[dict[str, Any]],
    *,
    ease_fo_pilot_design_updates: int,
    ease_sp_pilot_design_updates: int,
) -> list[dict[str, Any]]:
    updates_by_method = {
        "EaseSHAP_interaction_nonlinear": int(ease_fo_pilot_design_updates),
        "EaseSHAP_size_player": int(ease_sp_pilot_design_updates),
    }
    configured = []
    for method in methods:
        estimator_kwargs = dict(method["estimator_kwargs"])
        if method["name"] in updates_by_method:
            estimator_kwargs["pilot_design_updates"] = updates_by_method[method["name"]]
        configured.append({**method, "estimator_kwargs": estimator_kwargs})
    return configured


def main() -> None:
    args = parse_args()
    ensure_dirs()
    validate_method_specs(METHOD_SPECS)

    all_settings = build_settings(max_high_order_size=args.max_high_order_size)
    settings = select_settings(all_settings, args.settings)
    methods = configure_ease_pilot_design_updates(
        select_methods(METHOD_SPECS, args.methods),
        ease_fo_pilot_design_updates=args.ease_fo_pilot_design_updates,
        ease_sp_pilot_design_updates=args.ease_sp_pilot_design_updates,
    )
    write_config(
        settings,
        methods,
        ease_fo_pilot_design_updates=args.ease_fo_pilot_design_updates,
        ease_sp_pilot_design_updates=args.ease_sp_pilot_design_updates,
        max_high_order_size=args.max_high_order_size,
    )

    logger = setup_logging()
    logger.info("output: %s", OUT)
    logger.info(
        "game: n=%d, alphas=%s, num_high_order=%d, max_high_order_size=%s, sigma2=%s",
        N,
        GAME_ALPHAS,
        NUM_HIGH_ORDER,
        args.max_high_order_size,
        SIGMA2,
    )
    logger.info("semivalues: %s", [spec["name"] for spec in SEMIVALUE_SPECS])
    logger.info("methods: %s", [method["name"] for method in methods])
    logger.info(
        "EASE pilot design updates: FO=%d, SP=%d",
        args.ease_fo_pilot_design_updates,
        args.ease_sp_pilot_design_updates,
    )
    logger.info(
        "budgets: total_nue=%d, run_nue=%d, num_checkpoints=%d, track_step=%d, n_runs=%d",
        TOTAL_NUE,
        RUN_NUE,
        NUM_CHECKPOINTS,
        TRACK_STEP,
        N_RUNS,
    )
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


if __name__ == "__main__":
    main()
