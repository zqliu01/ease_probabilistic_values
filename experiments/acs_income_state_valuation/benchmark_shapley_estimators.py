"""Benchmark Shapley estimators on the ACSIncome state-source valuation game.

The experiment fixes one ACS train/evaluation split and repeats only the
estimator randomness.  This makes the comparison focus on estimator behavior,
not resampling noise in the ACS data split.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
EASESHAP_DIR = SCRIPT_DIR.parents[1] / "EaseSHAP"
MPLCONFIG_DIR = SCRIPT_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")
for path in (EASESHAP_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "benchmark_shapley_estimators"

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except ModuleNotFoundError:
    pass

import acs_data
import acs_model
import config as benchmark_config
import acs_state_game
from acs_state_game import ACSStateCoalitionGame
import easeshap.registry as easeshap_registry
from easeshap.registry import get_estimator_class


ACS_STATES = acs_data.US_STATES

READOUT_METHOD_BY_BACKEND = {
    "EaseSHAP": "_engine._readout_estimate",
    "RegressionMSR": "_run_kfold",
    "RegressionMSR_unbiased": "_run_kfold",
    "kernelSHAP": "_estimate",
    "LeverageSHAP": "_estimate",
    "LeverageSHAP_border": "_estimate",
    "PolySHAP_regression": "_run_polyshap_regression",
}

EXCLUDED_METHODS = {
    "LeverageSHAP_paired_border",
    "RegressionMSR_unbiased_no_replacement",
}


def load_method_specs() -> dict[str, dict[str, Any]]:
    """Load Shapley-compatible benchmark methods from the local ACS config."""

    specs: dict[str, dict[str, Any]] = {}
    for spec in benchmark_config.METHOD_SPECS:
        name = spec["name"]
        if name in EXCLUDED_METHODS:
            continue
        if spec.get("support") not in {"all", "shapley"}:
            continue

        backend = spec["backend"]
        available = True
        unavailable_error = ""
        try:
            get_estimator_class(backend)
        except Exception as exc:  # pragma: no cover - depends on optional installs.
            available = False
            unavailable_error = repr(exc)

        specs[name] = {
            "label": benchmark_config.METHOD_LABELS.get(name, spec.get("label", name)),
            "estimator": backend,
            "kwargs": dict(spec.get("estimator_kwargs", {})),
            "optional": bool(spec.get("optional", False)),
            "available": available,
            "unavailable_error": unavailable_error,
        }
    return specs


METHOD_SPECS: dict[str, dict[str, Any]] = load_method_specs()

_WORKER_CONTEXT: dict[str, Any] = {}


@dataclass(frozen=True)
class EstimatorTask:
    role: str
    method: str
    method_label: str
    estimator: str
    estimator_kwargs: dict[str, Any]
    seed_index: int
    estimator_seed: int
    nue_avg: int
    nue_track_avg: int
    nue_per_proc: int
    estimator_processes: int
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-year", default="2018")
    parser.add_argument("--target-state", type=str.upper, default="PA", choices=ACS_STATES)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--model-seed", type=int, default=2026)
    parser.add_argument("--estimator-seed-start", type=int, default=2026)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=acs_data.DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--encoder", default="semantic65", choices=["full", "semantic65"])
    parser.add_argument(
        "--utility-model",
        default="logistic",
        choices=["logistic", "xgboost", "gbm"],
        help="Model used inside each coalition utility evaluation.",
    )
    parser.add_argument(
        "--utility-cache-mode",
        default="memoize",
        choices=acs_state_game.UTILITY_CACHE_MODE_CHOICES,
        help=(
            "Repeated-coalition policy. 'memoize' reuses cached utilities; "
            "'recompute' evaluates every request while retaining the cache "
            "only to count unique coalitions."
        ),
    )
    parser.add_argument("--fixed-lambda", type=float, default=1.0)
    parser.add_argument(
        "--solver",
        default="liblinear",
        choices=["lbfgs", "liblinear", "newton-cholesky", "saga"],
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--xgb-n-estimators", type=int, default=5)
    parser.add_argument("--xgb-max-depth", type=int, default=5)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-tree-method", default="hist")
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--xgb-subsample", type=float, default=1.0)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=1.0)
    parser.add_argument(
        "--nue-avg",
        type=int,
        default=200,
        help="Budget per player. With 50 states, 200 means 10,000 utility evaluations.",
    )
    parser.add_argument("--num-checkpoints", type=int, default=20)
    parser.add_argument("--nue-per-proc", type=int, default=500)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--reference-method",
        default=None,
        choices=sorted(METHOD_SPECS),
        help=(
            "Optional high-budget method used as an approximate Shapley reference. "
            "Example: --reference-method OFA_fixed --reference-nue-avg 800."
        ),
    )
    parser.add_argument(
        "--reference-nue-avg",
        type=int,
        default=None,
        help="Budget per player for the approximate reference method.",
    )
    parser.add_argument(
        "--reference-num-seeds",
        type=int,
        default=None,
        help=(
            "Number of reference estimator seeds. Defaults to --num-seeds when "
            "--reference-method is set."
        ),
    )
    parser.add_argument(
        "--reference-seed-start",
        type=int,
        default=None,
        help=(
            "First estimator seed for reference runs. Defaults to "
            "--estimator-seed-start + 100000 to avoid reusing main-run seeds."
        ),
    )
    parser.add_argument(
        "--reference-num-checkpoints",
        type=int,
        default=1,
        help="Reference trajectory checkpoints. Use 1 when only the final reference is needed.",
    )
    parser.add_argument(
        "--reference-nue-per-proc",
        type=int,
        default=None,
        help="Reference run batch size. Defaults to --nue-per-proc.",
    )
    parser.add_argument(
        "--reference-estimator-processes",
        type=int,
        default=1,
        help=(
            "Internal multiprocessing workers for each reference estimator run. "
            "Use this for a single long reference run, e.g. "
            "--reference-nue-avg 4000 --reference-num-seeds 1 "
            "--reference-estimator-processes 20."
        ),
    )
    parser.add_argument("--ease-pilot-fraction", type=float, default=0.2)
    parser.add_argument(
        "--ease-fo-pilot-design-updates",
        type=int,
        default=1,
        help="Number of fixed-pilot design updates for EASE-FO only. Default: 1.",
    )
    parser.add_argument("--ease-num-folds", type=int, default=None)
    parser.add_argument("--ease-surrogate-ridge-lambda", type=float, default=None)
    parser.add_argument(
        "--ease-surrogate-ridge-schedule",
        default=None,
        choices=["fixed", "times_m"],
    )
    parser.add_argument("--regression-msr-num-folds", type=int, default=None)
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help=(
            "Include optional methods such as PolySHAP_regression_optional in "
            "the main method sweep. Disabled by default."
        ),
    )
    parser.add_argument(
        "--include-optional-reference",
        action="store_true",
        help="Allow an optional method to be used as the high-budget reference.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=sorted(METHOD_SPECS),
        help="Optional subset of benchmark method names. Defaults to all available Shapley-compatible methods.",
    )
    parser.add_argument(
        "--multiprocessing-start-method",
        default="fork",
        choices=["fork", "spawn", "forkserver"],
        help="Use fork on Slurm/Linux to avoid reloading the encoded ACS data in every worker.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def clone_method_spec(method: str) -> dict[str, Any]:
    spec = METHOD_SPECS[method]
    return {
        "name": method,
        "label": spec["label"],
        "estimator": spec["estimator"],
        "kwargs": dict(spec["kwargs"]),
        "optional": spec["optional"],
        "available": spec["available"],
        "unavailable_error": spec["unavailable_error"],
    }


def apply_method_overrides(spec: dict[str, Any], args: argparse.Namespace) -> None:
    if spec["estimator"] == "EaseSHAP":
        spec["kwargs"]["pilot_fraction"] = args.ease_pilot_fraction
        spec["kwargs"].pop("pilot_nue", None)
        if spec["name"] == "EaseSHAP_interaction_nonlinear":
            spec["kwargs"]["pilot_design_updates"] = args.ease_fo_pilot_design_updates
        if args.ease_num_folds is not None:
            spec["kwargs"]["num_folds"] = args.ease_num_folds
        if args.ease_surrogate_ridge_lambda is not None:
            spec["kwargs"]["surrogate_ridge_lambda"] = args.ease_surrogate_ridge_lambda
        if args.ease_surrogate_ridge_schedule is not None:
            spec["kwargs"]["surrogate_ridge_schedule"] = args.ease_surrogate_ridge_schedule
    if spec["estimator"] == "RegressionMSR_unbiased" and args.regression_msr_num_folds is not None:
        spec["kwargs"]["num_folds"] = args.regression_msr_num_folds


def validate_method_availability(specs: dict[str, dict[str, Any]]) -> None:
    unavailable = [
        f"{name}: {spec['unavailable_error']}"
        for name, spec in specs.items()
        if not spec["available"]
    ]
    if unavailable:
        raise RuntimeError(
            "Some requested methods are unavailable:\n" + "\n".join(unavailable)
        )


def configure_method_specs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    specs = {method: clone_method_spec(method) for method in METHOD_SPECS}

    if args.methods is None:
        selected = [
            name
            for name, spec in specs.items()
            if args.include_optional or not spec["optional"]
        ]
    else:
        selected = list(args.methods)
        optional_requested = [
            name for name in selected if specs[name]["optional"] and not args.include_optional
        ]
        if optional_requested:
            raise ValueError(
                "Optional main-sweep methods require --include-optional: "
                + ", ".join(optional_requested)
            )
    specs = {name: specs[name] for name in selected}

    validate_method_availability(specs)
    for spec in specs.values():
        apply_method_overrides(spec, args)
    return specs


def configure_reference_method_spec(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.reference_method is None:
        return {}
    spec = clone_method_spec(args.reference_method)
    if spec["optional"] and not args.include_optional_reference:
        raise ValueError(
            f"Reference method {args.reference_method!r} is optional. "
            "Pass --include-optional-reference to request an optional reference."
        )
    specs = {args.reference_method: spec}
    validate_method_availability(specs)
    apply_method_overrides(spec, args)
    return specs


def validate_and_complete_args(args: argparse.Namespace) -> None:
    positive_int_fields = [
        "train_size",
        "eval_size",
        "num_seeds",
        "nue_avg",
        "num_checkpoints",
        "nue_per_proc",
        "workers",
        "ease_fo_pilot_design_updates",
    ]
    for field in positive_int_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    args.utility_model = acs_state_game.normalize_utility_model(args.utility_model)
    if args.utility_model == "logistic":
        if args.fixed_lambda <= 0:
            raise ValueError("--fixed-lambda must be positive for logistic utility.")
        if args.max_iter <= 0:
            raise ValueError("--max-iter must be positive for logistic utility.")
    else:
        acs_state_game.require_xgb_classifier()
        for field in ["xgb_n_estimators", "xgb_max_depth", "xgb_n_jobs"]:
            if int(getattr(args, field)) <= 0:
                raise ValueError(f"--{field.replace('_', '-')} must be positive.")
        if args.xgb_learning_rate <= 0.0:
            raise ValueError("--xgb-learning-rate must be positive.")
        for field in ["xgb_subsample", "xgb_colsample_bytree"]:
            value = float(getattr(args, field))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"--{field.replace('_', '-')} must be in (0, 1].")
    if args.nue_avg % args.num_checkpoints != 0:
        raise ValueError("--nue-avg must be divisible by --num-checkpoints.")
    args.nue_track_avg = args.nue_avg // args.num_checkpoints

    reference_flags_without_method = []
    if args.reference_nue_avg is not None:
        reference_flags_without_method.append("--reference-nue-avg")
    if args.reference_num_seeds is not None:
        reference_flags_without_method.append("--reference-num-seeds")
    if args.reference_seed_start is not None:
        reference_flags_without_method.append("--reference-seed-start")
    if args.reference_num_checkpoints != 1:
        reference_flags_without_method.append("--reference-num-checkpoints")
    if args.reference_nue_per_proc is not None:
        reference_flags_without_method.append("--reference-nue-per-proc")
    if args.reference_estimator_processes != 1:
        reference_flags_without_method.append("--reference-estimator-processes")
    if args.include_optional_reference:
        reference_flags_without_method.append("--include-optional-reference")
    if args.reference_method is None and reference_flags_without_method:
        raise ValueError(
            "Reference-only options require --reference-method: "
            + ", ".join(reference_flags_without_method)
        )
    if args.reference_method is not None and args.reference_nue_avg is None:
        raise ValueError("--reference-method requires --reference-nue-avg.")

    if args.reference_method is not None:
        if args.reference_nue_avg <= 0:
            raise ValueError("--reference-nue-avg must be positive.")
        if args.reference_num_checkpoints <= 0:
            raise ValueError("--reference-num-checkpoints must be positive.")
        if args.reference_nue_avg % args.reference_num_checkpoints != 0:
            raise ValueError(
                "--reference-nue-avg must be divisible by --reference-num-checkpoints."
            )
        args.reference_num_seeds = (
            args.num_seeds if args.reference_num_seeds is None else args.reference_num_seeds
        )
        if args.reference_num_seeds <= 0:
            raise ValueError("--reference-num-seeds must be positive.")
        args.reference_seed_start = (
            args.estimator_seed_start + 100000
            if args.reference_seed_start is None
            else args.reference_seed_start
        )
        args.reference_nue_track_avg = args.reference_nue_avg // args.reference_num_checkpoints
        args.reference_nue_per_proc = (
            args.nue_per_proc
            if args.reference_nue_per_proc is None
            else args.reference_nue_per_proc
        )
        if args.reference_nue_per_proc <= 0:
            raise ValueError("--reference-nue-per-proc must be positive.")
        if args.reference_estimator_processes <= 0:
            raise ValueError("--reference-estimator-processes must be positive.")
    else:
        args.reference_num_seeds = 0
        args.reference_seed_start = None
        args.reference_nue_track_avg = None
        args.reference_nue_per_proc = None
        args.reference_estimator_processes = 1


def data_args_from(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        survey_year=args.survey_year,
        target_state=args.target_state,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.data_seed,
        data_dir=args.data_dir,
        download=args.download,
        encoder=args.encoder,
    )


def init_worker(context: dict[str, Any]) -> None:
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(context)


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


def default_readout_timing() -> dict[str, Any]:
    return {
        "elapsed_sec": 0.0,
        "regular_elapsed_sec": 0.0,
        "setup_sec": 0.0,
        "sampling_sec": 0.0,
        "utility_eval_sec": 0.0,
        "aggregate_sec": 0.0,
        "finalize_sec": 0.0,
        "worker_pool_setup_sec": 0.0,
        "worker_pool_teardown_sec": 0.0,
        "orchestration_sec": 0.0,
        "readout_elapsed_sec": 0.0,
        "final_readout_elapsed_sec": 0.0,
        "readout_call_count": 0,
        "readout_methods": [],
        "readout_hook_status": "not_configured",
        "timing_mode": "serial",
    }


def instrument_readout_timing(estimator: Any, backend: str) -> dict[str, Any]:
    timing = default_readout_timing()
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


def run_estimator_with_timing(
    *,
    backend: str,
    estimator_processes: int,
    game_args: dict[str, Any],
    nue_avg: int,
    nue_per_proc: int,
    nue_track_avg: int,
    estimator_seed: int,
    estimator_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one estimator while timing the same phases as the real-world benchmark.

    Serial runs time every sampling, utility-evaluation, and aggregation call
    independently. Parallel runs process one wave of at most
    ``estimator_processes`` batches at a time so those phase timings remain
    non-overlapping wall-clock measurements.
    """

    estimator_args = {
        "semivalue": "shapley",
        "semivalue_param": None,
        "game_func": ACSStateCoalitionGame,
        "game_args": game_args,
        "num_player": len(ACS_STATES),
        "nue_avg": nue_avg,
        "nue_per_proc": nue_per_proc,
        "nue_track_avg": nue_track_avg,
        "estimator_seed": estimator_seed,
    }

    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    estimator_cls = easeshap_registry.get_estimator_class(backend)
    estimator = estimator_cls(**estimator_args, **estimator_kwargs)
    setup_sec = time.perf_counter() - setup_start

    requires_serial_feedback = bool(getattr(estimator, "_requires_serial_feedback", False))
    effective_processes = (
        1 if requires_serial_feedback else max(1, int(estimator_processes))
    )
    if effective_processes == 1:
        timing = instrument_readout_timing(estimator, backend)
        timing["timing_mode"] = (
            "serial_feedback" if requires_serial_feedback and estimator_processes > 1 else "serial"
        )
    else:
        timing = default_readout_timing()
        timing["readout_hook_status"] = "disabled:parallel_estimator_processes"
        timing["timing_mode"] = "parallel_waves"

    sampling_sec = 0.0
    utility_eval_sec = 0.0
    aggregate_sec = 0.0
    worker_pool_setup_sec = 0.0
    worker_pool_teardown_sec = 0.0
    sampling_iter = iter(estimator.sampling())

    def next_samples() -> tuple[bool, Any]:
        nonlocal sampling_sec
        phase_start = time.perf_counter()
        try:
            samples = next(sampling_iter)
        except StopIteration:
            sampling_sec += time.perf_counter() - phase_start
            return False, None
        sampling_sec += time.perf_counter() - phase_start
        return True, samples

    if effective_processes == 1:
        while True:
            has_samples, samples = next_samples()
            if not has_samples:
                break

            phase_start = time.perf_counter()
            results = estimator.run(samples)
            utility_eval_sec += time.perf_counter() - phase_start

            phase_start = time.perf_counter()
            estimator.aggregate(results)
            aggregate_sec += time.perf_counter() - phase_start
    else:
        pool_setup_start = time.perf_counter()
        pool = mp.Pool(effective_processes)
        worker_pool_setup_sec = time.perf_counter() - pool_setup_start
        try:
            exhausted = False
            while not exhausted:
                sample_wave = []
                for _ in range(effective_processes):
                    has_samples, samples = next_samples()
                    if not has_samples:
                        exhausted = True
                        break
                    sample_wave.append(samples)
                if not sample_wave:
                    break

                phase_start = time.perf_counter()
                result_wave = pool.map(estimator.run, sample_wave)
                utility_eval_sec += time.perf_counter() - phase_start

                phase_start = time.perf_counter()
                for results in result_wave:
                    estimator.aggregate(results)
                aggregate_sec += time.perf_counter() - phase_start
        except BaseException:
            pool_teardown_start = time.perf_counter()
            pool.terminate()
            pool.join()
            worker_pool_teardown_sec = time.perf_counter() - pool_teardown_start
            raise
        else:
            pool_teardown_start = time.perf_counter()
            pool.close()
            pool.join()
            worker_pool_teardown_sec = time.perf_counter() - pool_teardown_start

    phase_start = time.perf_counter()
    values, trajectory = estimator.finalize()
    finalize_sec = time.perf_counter() - phase_start

    elapsed_sec = time.perf_counter() - total_start
    measured_phase_sec = (
        setup_sec
        + sampling_sec
        + utility_eval_sec
        + aggregate_sec
        + finalize_sec
        + worker_pool_setup_sec
        + worker_pool_teardown_sec
    )
    timing.update(
        {
            "elapsed_sec": elapsed_sec,
            "regular_elapsed_sec": max(
                0.0,
                elapsed_sec - float(timing["readout_elapsed_sec"]),
            ),
            "setup_sec": setup_sec,
            "sampling_sec": sampling_sec,
            "utility_eval_sec": utility_eval_sec,
            "aggregate_sec": aggregate_sec,
            "finalize_sec": finalize_sec,
            "worker_pool_setup_sec": worker_pool_setup_sec,
            "worker_pool_teardown_sec": worker_pool_teardown_sec,
            "orchestration_sec": max(0.0, elapsed_sec - measured_phase_sec),
            "effective_estimator_processes": effective_processes,
            "nue_per_proc_run": int(
                getattr(estimator, "nue_per_proc_run", nue_per_proc) or nue_per_proc
            ),
            "num_sample": int(getattr(estimator, "num_sample", 0) or 0),
            "batch_size": int(getattr(estimator, "batch_size", 0) or 0),
        }
    )
    return (
        np.asarray(values, dtype=float),
        np.asarray(trajectory, dtype=float),
        timing,
    )


def run_task(task: EstimatorTask) -> dict[str, Any]:
    output_dir = Path(task.output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if task.role == "main":
        task_stem = f"{task.method}_seed{task.estimator_seed}"
    else:
        task_stem = f"{task.role}_{task.method}_budget{task.nue_avg}x{len(ACS_STATES)}_seed{task.estimator_seed}"
    log_path = log_dir / f"{task_stem}.log"

    try:
        with log_path.open("w") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                start = time.perf_counter()
                utility_cache: dict[int, dict[str, float | int | str]] = {}
                game_args = {
                    "states": ACS_STATES,
                    "encoded_train": _WORKER_CONTEXT["encoded_train"],
                    "train_labels": _WORKER_CONTEXT["train_labels"],
                    "encoded_eval": _WORKER_CONTEXT["encoded_eval"],
                    "eval_y": _WORKER_CONTEXT["eval_y"],
                    "fixed_lambda": _WORKER_CONTEXT["fixed_lambda"],
                    "solver": _WORKER_CONTEXT["solver"],
                    "max_iter": _WORKER_CONTEXT["max_iter"],
                    "seed": _WORKER_CONTEXT["model_seed"],
                    "utility_cache": utility_cache,
                    "utility_cache_mode": _WORKER_CONTEXT["utility_cache_mode"],
                    "utility_model": _WORKER_CONTEXT["utility_model"],
                    "xgb_n_estimators": _WORKER_CONTEXT["xgb_n_estimators"],
                    "xgb_max_depth": _WORKER_CONTEXT["xgb_max_depth"],
                    "xgb_learning_rate": _WORKER_CONTEXT["xgb_learning_rate"],
                    "xgb_tree_method": _WORKER_CONTEXT["xgb_tree_method"],
                    "xgb_n_jobs": _WORKER_CONTEXT["xgb_n_jobs"],
                    "xgb_subsample": _WORKER_CONTEXT["xgb_subsample"],
                    "xgb_colsample_bytree": _WORKER_CONTEXT[
                        "xgb_colsample_bytree"
                    ],
                }
                values, trajectory, estimator_timing = run_estimator_with_timing(
                    backend=task.estimator,
                    estimator_processes=task.estimator_processes,
                    game_args=game_args,
                    nue_avg=task.nue_avg,
                    nue_per_proc=task.nue_per_proc,
                    nue_track_avg=task.nue_track_avg,
                    estimator_seed=task.estimator_seed,
                    estimator_kwargs=task.estimator_kwargs,
                )
                estimate_sec = float(estimator_timing["elapsed_sec"])
                checkpoint_readout_sec = max(
                    0.0,
                    float(estimator_timing["readout_elapsed_sec"])
                    - float(estimator_timing["final_readout_elapsed_sec"]),
                )
                paper_runtime_sec = max(
                    0.0,
                    estimate_sec - checkpoint_readout_sec,
                )

                game = ACSStateCoalitionGame(**game_args)
                empty_utility = game.evaluate(np.zeros(len(ACS_STATES), dtype=bool))
                grand_utility = game.evaluate(np.ones(len(ACS_STATES), dtype=bool))
                total_sec = time.perf_counter() - start

        result = {
            "status": "ok",
            "role": task.role,
            "method": task.method,
            "method_label": task.method_label,
            "estimator": task.estimator,
            "estimator_kwargs": task.estimator_kwargs,
            "seed_index": task.seed_index,
            "estimator_seed": task.estimator_seed,
            "nue_avg": task.nue_avg,
            "nue_track_avg": task.nue_track_avg,
            "nue_per_proc": task.nue_per_proc,
            "estimator_processes": task.estimator_processes,
            "estimate_sec": estimate_sec,
            "total_sec": total_sec,
            "paper_runtime_sec": paper_runtime_sec,
            "estimator_timing": estimator_timing,
            "setup_sec": float(estimator_timing["setup_sec"]),
            "sampling_sec": float(estimator_timing["sampling_sec"]),
            "utility_eval_sec": float(estimator_timing["utility_eval_sec"]),
            "aggregate_sec": float(estimator_timing["aggregate_sec"]),
            "finalize_sec": float(estimator_timing["finalize_sec"]),
            "worker_pool_setup_sec": float(estimator_timing["worker_pool_setup_sec"]),
            "worker_pool_teardown_sec": float(
                estimator_timing["worker_pool_teardown_sec"]
            ),
            "orchestration_sec": float(estimator_timing["orchestration_sec"]),
            "regular_elapsed_sec": float(estimator_timing["regular_elapsed_sec"]),
            "readout_elapsed_sec": float(estimator_timing["readout_elapsed_sec"]),
            "checkpoint_readout_elapsed_sec": checkpoint_readout_sec,
            "final_readout_elapsed_sec": float(
                estimator_timing["final_readout_elapsed_sec"]
            ),
            "readout_call_count": int(estimator_timing["readout_call_count"]),
            "readout_methods": estimator_timing["readout_methods"],
            "readout_hook_status": estimator_timing["readout_hook_status"],
            "timing_mode": estimator_timing["timing_mode"],
            "effective_estimator_processes": int(
                estimator_timing["effective_estimator_processes"]
            ),
            "nue_per_proc_run": int(estimator_timing["nue_per_proc_run"]),
            "num_sample": int(estimator_timing["num_sample"]),
            "batch_size": int(estimator_timing["batch_size"]),
            "utility_cache_mode": _WORKER_CONTEXT["utility_cache_mode"],
            "utility_cache_size": len(utility_cache),
            "empty_utility": float(empty_utility),
            "grand_utility": float(grand_utility),
            "total_gain": float(grand_utility - empty_utility),
            "sum_estimated_values": float(np.sum(values)),
            "log_path": str(log_path),
            "values": np.asarray(values, dtype=float).tolist(),
            "trajectory": np.asarray(trajectory, dtype=float).tolist(),
        }
        per_run_dir = output_dir / "per_run"
        per_run_dir.mkdir(parents=True, exist_ok=True)
        per_run_path = per_run_dir / f"{task_stem}.json"
        payload = dict(result)
        per_run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        result["per_run_path"] = str(per_run_path)
        return result
    except Exception:
        error = traceback.format_exc()
        log_path.write_text(error)
        return {
            "status": "error",
            "role": task.role,
            "method": task.method,
            "method_label": task.method_label,
            "estimator": task.estimator,
            "seed_index": task.seed_index,
            "estimator_seed": task.estimator_seed,
            "estimator_processes": task.estimator_processes,
            "log_path": str(log_path),
            "error": error,
        }


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reference_part = ""
        if args.reference_method is not None:
            reference_part = f"_ref{args.reference_method}_budget{args.reference_nue_avg}x{len(ACS_STATES)}"
        model_part = "xgb" if args.utility_model == "xgboost" else "logistic"
        run_name = (
            f"{args.target_state.lower()}_{args.encoder}_train{args.train_size}"
            f"_eval{args.eval_size}_{model_part}_budget{args.nue_avg}x{len(ACS_STATES)}"
            f"{reference_part}_seeds{args.num_seeds}_{timestamp}"
        )
    else:
        run_name = args.run_name
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_tasks(
    args: argparse.Namespace,
    output_dir: Path,
    method_specs: dict[str, dict[str, Any]],
    *,
    role: str,
    num_seeds: int,
    seed_start: int,
    nue_avg: int,
    nue_track_avg: int,
    nue_per_proc: int,
    estimator_processes: int,
) -> list[EstimatorTask]:
    tasks: list[EstimatorTask] = []
    seeds = [seed_start + i for i in range(num_seeds)]
    method_names = list(method_specs)
    for seed_index, seed in enumerate(seeds):
        for method in method_names:
            spec = method_specs[method]
            tasks.append(
                EstimatorTask(
                    role=role,
                    method=method,
                    method_label=spec["label"],
                    estimator=spec["estimator"],
                    estimator_kwargs=dict(spec["kwargs"]),
                    seed_index=seed_index,
                    estimator_seed=seed,
                    nue_avg=nue_avg,
                    nue_track_avg=nue_track_avg,
                    nue_per_proc=nue_per_proc,
                    estimator_processes=estimator_processes,
                    output_dir=str(output_dir),
                )
            )
    return tasks


def rank_order(values: np.ndarray) -> list[str]:
    order = np.argsort(-np.asarray(values, dtype=float))
    return [ACS_STATES[i] for i in order]


def overlap_at(values: np.ndarray, reference: np.ndarray, k: int) -> int:
    return len(set(rank_order(values)[:k]) & set(rank_order(reference)[:k]))


def finite_corr(values: np.ndarray, reference: np.ndarray, method: str) -> float:
    series = pd.Series(values)
    ref_series = pd.Series(reference)
    if series.nunique(dropna=True) <= 1 or ref_series.nunique(dropna=True) <= 1:
        return float("nan")
    return float(series.corr(ref_series, method=method))


def build_tables(
    *,
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    reference: dict[str, Any] | None = None,
) -> dict[str, Path]:
    run_rows = []
    value_rows = []
    trajectory_rows = []

    for result in results:
        run_record = {
            key: value
            for key, value in result.items()
            if key not in {"values", "trajectory", "estimator_kwargs", "estimator_timing"}
        }
        run_record["estimator_kwargs_json"] = json.dumps(
            result["estimator_kwargs"], sort_keys=True
        )
        run_rows.append(run_record)

        values = np.asarray(result["values"], dtype=float)
        for state, value in zip(ACS_STATES, values):
            value_rows.append(
                {
                    "method": result["method"],
                    "method_label": result["method_label"],
                    "seed_index": result["seed_index"],
                    "estimator_seed": result["estimator_seed"],
                    "state": state,
                    "value": float(value),
                }
            )

        trajectory = np.asarray(result["trajectory"], dtype=float)
        for checkpoint_idx, row in enumerate(trajectory, start=1):
            utility_evaluations = checkpoint_idx * int(result["nue_track_avg"]) * len(ACS_STATES)
            for state, value in zip(ACS_STATES, row):
                trajectory_rows.append(
                    {
                        "method": result["method"],
                        "method_label": result["method_label"],
                        "seed_index": result["seed_index"],
                        "estimator_seed": result["estimator_seed"],
                        "checkpoint": checkpoint_idx,
                        "utility_evaluations": utility_evaluations,
                        "state": state,
                        "value": float(value),
                    }
                )

    runs_df = pd.DataFrame(run_rows)
    values_df = pd.DataFrame(value_rows)
    trajectory_df = pd.DataFrame(trajectory_rows)

    final_matrix = {
        method: values_df[values_df["method"] == method]
        .pivot(index="estimator_seed", columns="state", values="value")
        .reindex(columns=ACS_STATES)
        for method in values_df["method"].drop_duplicates()
    }
    references = {
        method: matrix.mean(axis=0).to_numpy(dtype=float)
        for method, matrix in final_matrix.items()
    }
    reference_values = None
    if reference is not None:
        reference_values = np.asarray(reference["values"], dtype=float)

    metrics_rows = []
    grouped = trajectory_df.groupby(
        ["method", "method_label", "seed_index", "estimator_seed", "checkpoint"],
        sort=False,
    )
    total_gain_by_seed = runs_df.set_index(["method", "estimator_seed"])["total_gain"]
    for group_key, group in grouped:
        method, method_label, seed_index, estimator_seed, checkpoint = group_key
        values = group.set_index("state").reindex(ACS_STATES)["value"].to_numpy(dtype=float)
        own_reference = references[method]
        total_gain = float(total_gain_by_seed.loc[(method, estimator_seed)])
        utility_evaluations = int(group["utility_evaluations"].iloc[0])
        row = {
            "method": method,
            "method_label": method_label,
            "seed_index": seed_index,
            "estimator_seed": estimator_seed,
            "checkpoint": checkpoint,
            "utility_evaluations": utility_evaluations,
            "sum_estimated_values": float(np.nansum(values)),
            "efficiency_error": float(abs(np.nansum(values) - total_gain)),
            "rmse_to_own_final_mean": float(np.sqrt(np.nanmean((values - own_reference) ** 2))),
            "mae_to_own_final_mean": float(np.nanmean(np.abs(values - own_reference))),
            "pearson_to_own_final_mean": finite_corr(values, own_reference, "pearson"),
            "spearman_to_own_final_mean": finite_corr(values, own_reference, "spearman"),
        }
        for k in (5, 10, 15):
            row[f"top{k}_overlap_own_final_mean"] = overlap_at(values, own_reference, k)
        if reference_values is not None:
            row.update(
                {
                    "rmse_to_reference": float(
                        np.sqrt(np.nanmean((values - reference_values) ** 2))
                    ),
                    "mae_to_reference": float(np.nanmean(np.abs(values - reference_values))),
                    "pearson_to_reference": finite_corr(values, reference_values, "pearson"),
                    "spearman_to_reference": finite_corr(values, reference_values, "spearman"),
                }
            )
            for k in (5, 10, 15):
                row[f"top{k}_overlap_reference"] = overlap_at(values, reference_values, k)
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)

    metric_cols = [
        col
        for col in metrics_df.columns
        if col
        not in {
            "method",
            "method_label",
            "seed_index",
            "estimator_seed",
            "checkpoint",
            "utility_evaluations",
        }
    ]
    checkpoint_summary = (
        metrics_df.groupby(["method", "method_label", "checkpoint", "utility_evaluations"])[
            metric_cols
        ]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    checkpoint_summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        for col in checkpoint_summary.columns.to_flat_index()
    ]

    final_checkpoint = int(metrics_df["checkpoint"].max())
    final_metrics = metrics_df[metrics_df["checkpoint"] == final_checkpoint]
    runtime_cols = [
        "estimate_sec",
        "total_sec",
        "paper_runtime_sec",
        "regular_elapsed_sec",
        "setup_sec",
        "sampling_sec",
        "utility_eval_sec",
        "aggregate_sec",
        "finalize_sec",
        "readout_elapsed_sec",
        "checkpoint_readout_elapsed_sec",
        "final_readout_elapsed_sec",
        "worker_pool_setup_sec",
        "worker_pool_teardown_sec",
        "orchestration_sec",
        "utility_cache_size",
    ]
    runtime_summary = (
        runs_df.groupby(["method", "method_label"])[runtime_cols]
        .agg(["mean", "std", "sem", "min", "max"])
        .reset_index()
    )
    runtime_summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        for col in runtime_summary.columns.to_flat_index()
    ]
    final_metric_summary = (
        final_metrics.groupby(["method", "method_label"])[metric_cols]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    final_metric_summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        for col in final_metric_summary.columns.to_flat_index()
    ]
    method_summary = final_metric_summary.merge(
        runtime_summary, on=["method", "method_label"], how="left"
    )

    state_summary = (
        values_df.groupby(["method", "method_label", "state"])["value"]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    state_summary["rank_within_method"] = (
        state_summary.groupby("method")["mean"].rank(method="first", ascending=False).astype(int)
    )
    state_summary = state_summary.sort_values(["method", "rank_within_method"])

    paths = {
        "runs": output_dir / "runs.csv",
        "final_values_long": output_dir / "final_values_long.csv",
        "trajectory_long": output_dir / "trajectory_long.csv",
        "checkpoint_metrics": output_dir / "checkpoint_metrics.csv",
        "checkpoint_summary": output_dir / "checkpoint_summary.csv",
        "method_summary": output_dir / "method_summary.csv",
        "state_summary": output_dir / "state_summary.csv",
    }
    runs_df.to_csv(paths["runs"], index=False)
    values_df.to_csv(paths["final_values_long"], index=False)
    trajectory_df.to_csv(paths["trajectory_long"], index=False)
    metrics_df.to_csv(paths["checkpoint_metrics"], index=False)
    checkpoint_summary.to_csv(paths["checkpoint_summary"], index=False)
    method_summary.to_csv(paths["method_summary"], index=False)
    state_summary.to_csv(paths["state_summary"], index=False)

    return paths


def execute_tasks(
    *,
    tasks: list[EstimatorTask],
    context: dict[str, Any],
    args: argparse.Namespace,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    if not tasks:
        return results, errors, 0.0

    use_serial_outer_loop = args.workers == 1 or any(
        task.estimator_processes > 1 for task in tasks
    )

    if use_serial_outer_loop:
        init_worker(context)
        for idx, task in enumerate(tasks, start=1):
            result = run_task(task)
            if result["status"] == "ok":
                results.append(result)
                print(
                    f"[{label} {idx}/{len(tasks)}] {task.method} seed={task.estimator_seed} "
                    f"budget={task.nue_avg}n done in {result['total_sec']:.1f}s"
                )
            else:
                errors.append(result)
                print(
                    f"[{label} {idx}/{len(tasks)}] {task.method} seed={task.estimator_seed} "
                    f"budget={task.nue_avg}n failed"
                )
    else:
        mp_context = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_context,
            initializer=init_worker,
            initargs=(context,),
        ) as executor:
            future_to_task = {executor.submit(run_task, task): task for task in tasks}
            for idx, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                result = future.result()
                if result["status"] == "ok":
                    results.append(result)
                    print(
                        f"[{label} {idx}/{len(tasks)}] {task.method} seed={task.estimator_seed} "
                        f"budget={task.nue_avg}n done in {result['total_sec']:.1f}s"
                    )
                else:
                    errors.append(result)
                    print(
                        f"[{label} {idx}/{len(tasks)}] {task.method} seed={task.estimator_seed} "
                        f"budget={task.nue_avg}n failed"
                    )

    return results, errors, time.perf_counter() - total_start


def build_reference(
    *,
    reference_results: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Path]]:
    if not reference_results:
        return None, {}

    values = np.vstack(
        [np.asarray(result["values"], dtype=float) for result in reference_results]
    )
    mean_values = np.nanmean(values, axis=0)
    std_values = np.nanstd(values, axis=0, ddof=1) if len(reference_results) > 1 else np.zeros(values.shape[1])
    sem_values = std_values / np.sqrt(len(reference_results))
    first = reference_results[0]
    if len(reference_results) == 1:
        label = f"{first['method_label']} {first['nue_avg']}n single seed"
    else:
        label = (
            f"{first['method_label']} {first['nue_avg']}n "
            f"mean over {len(reference_results)} seeds"
        )

    reference = {
        "method": first["method"],
        "method_label": first["method_label"],
        "label": label,
        "nue_avg": int(first["nue_avg"]),
        "total_utility_budget": int(first["nue_avg"]) * len(ACS_STATES),
        "aggregate_total_utility_budget": (
            int(first["nue_avg"]) * len(ACS_STATES) * len(reference_results)
        ),
        "estimator_processes": int(first.get("estimator_processes", 1)),
        "num_seeds": len(reference_results),
        "estimator_seeds": [int(result["estimator_seed"]) for result in reference_results],
        "values": mean_values.tolist(),
    }

    reference_dir = output_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    values_path = reference_dir / "reference_values.csv"
    runs_path = reference_dir / "reference_runs.csv"
    summary_path = reference_dir / "reference_summary.json"

    reference_values_df = pd.DataFrame(
        {
            "state": ACS_STATES,
            "value": mean_values,
            "std": std_values,
            "sem": sem_values,
            "rank": pd.Series(mean_values).rank(method="first", ascending=False).astype(int),
        }
    ).sort_values("rank")
    reference_values_df.to_csv(values_path, index=False)

    run_rows = []
    for result in reference_results:
        run_rows.append(
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "values",
                    "trajectory",
                    "estimator_kwargs",
                    "estimator_timing",
                }
            }
            | {"estimator_kwargs_json": json.dumps(result["estimator_kwargs"], sort_keys=True)}
        )
    pd.DataFrame(run_rows).to_csv(runs_path, index=False)
    summary_path.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")

    return reference, {
        "reference_values": values_path,
        "reference_runs": runs_path,
        "reference_summary": summary_path,
    }


def main() -> None:
    args = parse_args()
    validate_and_complete_args(args)
    method_specs = configure_method_specs(args)
    reference_method_specs = configure_reference_method_spec(args)

    output_dir = make_output_dir(args)
    config_path = output_dir / "config.json"

    load_start = time.perf_counter()
    encoded_split = acs_model.prepare_encoded_acs_split(data_args_from(args))
    load_encode_sec = time.perf_counter() - load_start

    context = {
        "encoded_train": encoded_split.encoded_train,
        "train_labels": encoded_split.train_labels,
        "encoded_eval": encoded_split.encoded_eval,
        "eval_y": encoded_split.eval_y,
        "fixed_lambda": args.fixed_lambda,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "model_seed": args.model_seed,
        "utility_cache_mode": args.utility_cache_mode,
        "utility_model": args.utility_model,
        "xgb_n_estimators": args.xgb_n_estimators,
        "xgb_max_depth": args.xgb_max_depth,
        "xgb_learning_rate": args.xgb_learning_rate,
        "xgb_tree_method": args.xgb_tree_method,
        "xgb_n_jobs": args.xgb_n_jobs,
        "xgb_subsample": args.xgb_subsample,
        "xgb_colsample_bytree": args.xgb_colsample_bytree,
        "nue_avg": args.nue_avg,
        "nue_per_proc": args.nue_per_proc,
        "nue_track_avg": args.nue_track_avg,
    }
    config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "resolved_methods": list(method_specs),
        "resolved_reference_methods": list(reference_method_specs),
        "num_states": len(ACS_STATES),
        "total_utility_budget": args.nue_avg * len(ACS_STATES),
        "nue_track_avg": args.nue_track_avg,
        "reference_nue_track_avg": args.reference_nue_track_avg,
        "load_encode_sec": load_encode_sec,
        "eval_label_rate": float(np.mean(encoded_split.eval_y)),
        "train_label_rates": encoded_split.label_rates,
        "feature_block_dimensions": encoded_split.feature_block_dims,
        "n_features": int(encoded_split.feature_block_dims["total"]),
        "raw_data_dir": str(encoded_split.processed_split.data_dirs.raw_dir),
        "processed_data_dir": str(encoded_split.processed_split.data_dirs.processed_dir),
        "processed_split_dir": str(encoded_split.processed_split.split_dir),
        "sample_manifest_path": str(encoded_split.processed_split.manifest_path),
        "method_specs": method_specs,
        "reference_method_specs": reference_method_specs,
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n")

    tasks = make_tasks(
        args,
        output_dir,
        method_specs,
        role="main",
        num_seeds=args.num_seeds,
        seed_start=args.estimator_seed_start,
        nue_avg=args.nue_avg,
        nue_track_avg=args.nue_track_avg,
        nue_per_proc=args.nue_per_proc,
        estimator_processes=1,
    )
    reference_tasks = []
    if reference_method_specs:
        reference_tasks = make_tasks(
            args,
            output_dir,
            reference_method_specs,
            role="reference",
            num_seeds=args.reference_num_seeds,
            seed_start=args.reference_seed_start,
            nue_avg=args.reference_nue_avg,
            nue_track_avg=args.reference_nue_track_avg,
            nue_per_proc=args.reference_nue_per_proc,
            estimator_processes=args.reference_estimator_processes,
        )

    print(f"Output directory: {output_dir}")
    print(f"Target state: {args.target_state}")
    print(
        f"Encoder: {args.encoder} "
        f"({int(encoded_split.feature_block_dims['total'])} features)"
    )
    print(f"Utility model: {args.utility_model}")
    print(f"Train/eval sizes: {args.train_size}/{args.eval_size}")
    print(f"Budget: {args.nue_avg} x {len(ACS_STATES)} = {args.nue_avg * len(ACS_STATES)}")
    print(f"Checkpoints: {args.num_checkpoints} every {args.nue_track_avg}n evaluations")
    print(f"Methods: {', '.join(method_specs)}")
    print(f"Seeds per method: {args.num_seeds}")
    if reference_tasks:
        print(
            "Reference: "
            f"{args.reference_method} at {args.reference_nue_avg} x {len(ACS_STATES)} "
            f"= {args.reference_nue_avg * len(ACS_STATES)} utility evaluations, "
            f"{args.reference_num_seeds} seeds, "
            f"{args.reference_estimator_processes} estimator processes/run"
        )
    print(f"Workers: {args.workers}")
    print(f"Load/encode seconds: {load_encode_sec:.3f}")

    total_start = time.perf_counter()
    reference_results, reference_errors, reference_elapsed_sec = execute_tasks(
        tasks=reference_tasks,
        context=context,
        args=args,
        label="reference",
    )
    if reference_errors:
        error_path = output_dir / "reference_errors.json"
        error_path.write_text(json.dumps(reference_errors, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"{len(reference_errors)} reference runs failed; see {error_path}")
    reference, reference_paths = build_reference(
        reference_results=reference_results,
        output_dir=output_dir,
    )

    results, errors, main_elapsed_sec = execute_tasks(
        tasks=tasks,
        context=context,
        args=args,
        label="main",
    )

    elapsed_sec = time.perf_counter() - total_start
    if errors:
        error_path = output_dir / "errors.json"
        error_path.write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"{len(errors)} estimator runs failed; see {error_path}")

    paths = build_tables(results=results, args=args, output_dir=output_dir, reference=reference)
    paths.update(reference_paths)
    completion_path = output_dir / "completion.json"
    completion_path.write_text(
        json.dumps(
            {
                "completed_runs": len(results),
                "completed_reference_runs": len(reference_results),
                "reference_elapsed_sec_after_encoding": reference_elapsed_sec,
                "main_elapsed_sec_after_encoding": main_elapsed_sec,
                "pipeline_elapsed_sec_after_encoding": elapsed_sec,
                "elapsed_sec_after_encoding": main_elapsed_sec,
                "total_elapsed_sec": main_elapsed_sec + load_encode_sec,
                "pipeline_total_elapsed_sec": elapsed_sec + load_encode_sec,
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    method_summary = pd.read_csv(paths["method_summary"])
    print("\nFinal method summary:")
    display_cols = [
        "method_label",
        "efficiency_error_mean",
        "total_sec_mean",
    ]
    if reference is not None:
        display_cols[1:1] = [
            "rmse_to_reference_mean",
            "spearman_to_reference_mean",
            "top10_overlap_reference_mean",
        ]
    display_cols[1:1] = [
        "rmse_to_own_final_mean_mean",
        "spearman_to_own_final_mean_mean",
        "top10_overlap_own_final_mean_mean",
    ]
    print(method_summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
