"""Run the real-world semivalue benchmark.

The runner stores one pickle per
(dataset, semivalue, method, instance, run) task so SLURM arrays can retry
individual failures without rewriting successful jobs.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import json
import logging
import os
import pickle
import resource
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    CONFIGS_DIR,
    DATA_DIR,
    DATASET_SPECS,
    DEFAULT_N_RUNS,
    GROUNDTRUTH_DIR,
    METHOD_SPECS,
    OUT,
    PACKAGE_ROOT,
    RANDOM_STATE,
    SEMIVALUE_SPECS,
    SCRIPT_DIR,
    add_ease_switch_fractions_arg,
    add_results_config_args,
    benchmark_actual_budgets,
    checkpoint_interval_nue_avg,
    config_raw_dir,
    ease_pilot_nue_for_switch,
    ease_switch_budget,
    equal_interval_checkpoints,
    fraction_label,
    get_dataset,
    get_method,
    get_semivalue,
    iter_task_specs,
    max_budget,
    method_display_label,
    nearest_actual_checkpoint_indices,
    reference_path,
    result_path,
    resolve_config_dir_from_args,
    validate_results_config_args,
)
from games import BenchmarkGame, compute_reference_values


if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
_MPLCONFIGDIR = OUT / "mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))


READOUT_METHOD_BY_BACKEND = {
    "EaseSHAP": "_engine._readout_estimate",
    "RegressionMSR": "_run_kfold",
    "RegressionMSR_unbiased": "_run_kfold",
    "kernelSHAP": "_estimate",
    "LeverageSHAP": "_estimate",
    "LeverageSHAP_border": "_estimate",
    "PolySHAP_regression": "_run_polyshap_regression",
}


def get_estimator_class_from_registry(backend: str):
    from easeshap.registry import get_estimator_class

    return get_estimator_class(backend)


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
def silence_worker_output(enabled: bool):
    if not enabled:
        yield
        return
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


def ensure_dirs(*, config_dir: Path, raw_dir: Path, groundtruth_dir: Path) -> None:
    for path in (OUT, CONFIGS_DIR, config_dir, raw_dir, groundtruth_dir):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(*, config_dir: Path, raw_dir: Path, groundtruth_dir: Path) -> logging.Logger:
    ensure_dirs(config_dir=config_dir, raw_dir=raw_dir, groundtruth_dir=groundtruth_dir)
    mplconfig = config_dir / "mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mplconfig)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config_dir / "run.log", mode="a"),
        ],
        force=True,
    )
    return logging.getLogger("real_world_benchmark")


def atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}_{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def write_config_metadata(
    *,
    args: argparse.Namespace,
    config_dir: Path,
    raw_dir: Path,
    groundtruth_dir: Path,
) -> None:
    label = config_dir.name
    ease_switch_fractions = getattr(args, "ease_switch_fractions", None)
    if ease_switch_fractions:
        pilot_label = "pilots" + "_".join(fraction_label(fraction) for fraction in ease_switch_fractions)
    else:
        pilot_label = f"pilot{fraction_label(args.ease_switch_fraction)}"
    metadata = {
        "config_name": label,
        "budget_per_player": int(args.budget_per_player),
        "total_budget": f"{int(args.budget_per_player)}n",
        "ease_switch_fraction": None if ease_switch_fractions else float(args.ease_switch_fraction),
        "ease_switch_fractions": ease_switch_fractions or [float(args.ease_switch_fraction)],
        "ease_fo_pilot_design_updates": int(args.ease_fo_pilot_design_updates),
        "pilot_label": pilot_label,
        "excluded_methods": sorted(getattr(args, "exclude_method", None) or []),
        "tabular_game_mode": args.tabular_game_mode,
        "num_checkpoints": int(args.num_checkpoints),
        "checkpoint_interval_per_player": int(
            checkpoint_interval_nue_avg(
                budget_per_player=args.budget_per_player,
                num_checkpoints=args.num_checkpoints,
            )
        ),
        "raw_dir": os.path.relpath(raw_dir, start=SCRIPT_DIR),
        "groundtruth_dir": os.path.relpath(groundtruth_dir, start=SCRIPT_DIR),
        "data_dir": os.path.relpath(DATA_DIR, start=SCRIPT_DIR),
    }
    atomic_save_json(config_dir / "config.json", metadata)


def validate_method_specs(methods: list[dict[str, Any]]) -> None:
    names = set()
    for method in methods:
        name = method["name"]
        if name in names:
            raise ValueError(f"Duplicate method name: {name}")
        names.add(name)
        try:
            get_estimator_class_from_registry(method["backend"])
        except AttributeError as exc:
            raise ValueError(
                f"Backend {method['backend']!r} for method {name!r} is not available."
            ) from exc


def parse_int_set(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                start_text, end_text = part.split(":", 1)
                start = int(start_text)
                end = int(end_text)
                out.extend(range(start, end + 1))
            else:
                out.append(int(part))
    return sorted(set(out))


def select_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_id: int | None,
    num_tasks: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(tasks))
    if task_id is None and num_tasks is None:
        return indexed
    if task_id is None or num_tasks is None:
        raise ValueError("--task-id and --num-tasks must be provided together.")
    if num_tasks <= 0:
        raise ValueError("--num-tasks must be positive.")
    if task_id < 0 or task_id >= num_tasks:
        raise ValueError("--task-id must be in [0, num_tasks).")
    return [(idx, task) for idx, task in indexed if idx % num_tasks == task_id]


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_save_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}_{uuid.uuid4().hex}")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_or_compute_reference(
    *,
    task: dict[str, Any],
    semivalue_spec: dict[str, Any],
    random_state: int,
    tabular_game_mode: str,
    groundtruth_dir: Path,
    force: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = reference_path(
        task["dataset"],
        task["semivalue"],
        task["instance_id"],
        groundtruth_dir=groundtruth_dir,
    )
    if path.exists() and not force:
        data = np.load(path, allow_pickle=False)
        phi = np.asarray(data["phi"], dtype=np.float64)
        metadata = json.loads(str(data["metadata"].item()))
        return phi, metadata

    phi, metadata = compute_reference_values(
        dataset_name=task["dataset"],
        instance_id=int(task["instance_id"]),
        semivalue=semivalue_spec["semivalue"],
        semivalue_param=semivalue_spec["semivalue_param"],
        random_state=random_state,
        tabular_game_mode=tabular_game_mode,
    )
    metadata = {
        **metadata,
        "dataset": task["dataset"],
        "semivalue": task["semivalue"],
        "instance_id": int(task["instance_id"]),
        "tabular_game_mode": tabular_game_mode,
        "random_state": int(random_state),
        "computed_at_unix": time.time(),
    }
    atomic_save_npz(path, phi=np.asarray(phi, dtype=np.float64), metadata=json.dumps(metadata))
    return np.asarray(phi, dtype=np.float64), metadata


def estimator_kwargs_for_task(
    *,
    method_spec: dict[str, Any],
    n_players: int,
    budget_per_player: int,
    ease_switch_fraction: float,
    ease_fo_pilot_design_updates: int = 1,
) -> dict[str, Any]:
    kwargs = dict(method_spec.get("estimator_kwargs", {}))
    if method_spec["backend"] == "EaseSHAP":
        switch_budget = ease_switch_budget(
            n_players,
            budget_per_player=budget_per_player,
            switch_fraction=ease_switch_fraction,
        )
        kwargs["pilot_nue"] = ease_pilot_nue_for_switch(n_players, switch_budget)
        if method_spec["name"] == "EaseSHAP_interaction_nonlinear":
            kwargs["pilot_design_updates"] = int(ease_fo_pilot_design_updates)
    return kwargs


def run_estimator(
    *,
    backend: str,
    semivalue_spec: dict[str, Any],
    dataset_name: str,
    instance_id: int,
    tabular_game_mode: str,
    estimator_seed: int,
    n_players: int,
    nue_avg: int,
    nue_track_avg: int,
    nue_per_proc: int,
    estimator_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    estimator_args = dict(
        semivalue=semivalue_spec["semivalue"],
        semivalue_param=semivalue_spec["semivalue_param"],
        game_func=BenchmarkGame,
        game_args={
            "dataset_name": dataset_name,
            "instance_id": int(instance_id),
            "random_state": RANDOM_STATE,
            "tabular_game_mode": tabular_game_mode,
        },
        num_player=int(n_players),
        nue_avg=int(nue_avg),
        nue_per_proc=int(nue_per_proc),
        nue_track_avg=int(nue_track_avg),
        estimator_seed=int(estimator_seed),
    )

    total_start = time.perf_counter()
    rss_mb_start = current_rss_mb()
    peak_rss_mb_start = peak_rss_mb()

    setup_start = time.perf_counter()
    estimator_cls = get_estimator_class_from_registry(backend)
    estimator = estimator_cls(**estimator_args, **estimator_kwargs)
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
    timing["nue_per_proc_run"] = int(getattr(estimator, "nue_per_proc_run", nue_per_proc))
    timing["num_sample"] = int(getattr(estimator, "num_sample", 0) or 0)
    timing["batch_size"] = int(getattr(estimator, "batch_size", 0) or 0)

    return (
        np.asarray(values_final, dtype=np.float64),
        np.asarray(values_traj, dtype=np.float64),
        timing,
    )


def run_one_task(
    *,
    task_index: int,
    task: dict[str, Any],
    args: argparse.Namespace,
    config_dir: Path,
    raw_dir: Path,
    groundtruth_dir: Path,
    logger: logging.Logger,
) -> str:
    dataset_spec = get_dataset(task["dataset"])
    semivalue_spec = get_semivalue(task["semivalue"])
    base_method_name = task.get("base_method", task["method"])
    method_spec = get_method(base_method_name)
    n_players = int(dataset_spec["n_players"])
    task_ease_switch_fraction = float(task.get("ease_switch_fraction", args.ease_switch_fraction))
    nue_avg, nue_track_avg, actual_budgets = benchmark_actual_budgets(
        n_players,
        budget_per_player=args.budget_per_player,
        num_checkpoints=args.num_checkpoints,
    )
    requested_checkpoints = equal_interval_checkpoints(
        n_players,
        budget_per_player=args.budget_per_player,
        num_checkpoints=args.num_checkpoints,
    )
    checkpoint_indices = nearest_actual_checkpoint_indices(requested_checkpoints, actual_budgets)
    checkpoint_budgets = [actual_budgets[idx] for idx in checkpoint_indices]
    max_budget_total = max_budget(n_players, budget_per_player=args.budget_per_player)

    path = result_path(
        task["dataset"],
        task["semivalue"],
        task["method"],
        int(task["instance_id"]),
        int(task["run_idx"]),
        raw_dir=raw_dir,
    )
    if path.exists() and not args.force:
        try:
            with path.open("rb") as f:
                cached_payload = pickle.load(f)
            if cached_payload.get("status") == "ok":
                logger.info("task %d cached: %s", task_index, path)
                return "cached"
            logger.info("task %d rerun previous non-ok result: %s", task_index, path)
        except Exception:
            logger.info("task %d rerun unreadable cached result: %s", task_index, path)

    reference_phi, reference_metadata = load_or_compute_reference(
        task=task,
        semivalue_spec=semivalue_spec,
        random_state=RANDOM_STATE,
        tabular_game_mode=args.tabular_game_mode,
        groundtruth_dir=groundtruth_dir,
        force=args.force_reference,
    )
    if args.only_reference:
        logger.info("task %d reference ready", task_index)
        return "reference"

    estimator_kwargs = estimator_kwargs_for_task(
        method_spec=method_spec,
        n_players=n_players,
        budget_per_player=args.budget_per_player,
        ease_switch_fraction=task_ease_switch_fraction,
        ease_fo_pilot_design_updates=args.ease_fo_pilot_design_updates,
    )
    switch_budget = ease_switch_budget(
        n_players,
        budget_per_player=args.budget_per_player,
        switch_fraction=task_ease_switch_fraction,
    )
    seed = int(task["seed"]) + int(task["instance_id"]) * 10_003

    logger.info(
        "task %d start dataset=%s semivalue=%s method=%s instance=%03d run=%02d",
        task_index,
        task["dataset"],
        task["semivalue"],
        task["method"],
        int(task["instance_id"]),
        int(task["run_idx"]),
    )
    start = time.perf_counter()
    with silence_worker_output(not args.verbose_estimator):
        values_final, values_traj, timing = run_estimator(
            backend=method_spec["backend"],
            semivalue_spec=semivalue_spec,
            dataset_name=task["dataset"],
            instance_id=int(task["instance_id"]),
            tabular_game_mode=args.tabular_game_mode,
            estimator_seed=seed,
            n_players=n_players,
            nue_avg=nue_avg,
            nue_track_avg=nue_track_avg,
            nue_per_proc=args.nue_per_proc,
            estimator_kwargs=estimator_kwargs,
        )

    if len(values_traj) != len(actual_budgets):
        raise RuntimeError(
            "Estimator trajectory length does not match the benchmark checkpoint grid: "
            f"got len(values_traj)={len(values_traj)} and len(actual_budgets)={len(actual_budgets)} "
            f"for dataset={task['dataset']}, semivalue={task['semivalue']}, "
            f"method={task['method']}, instance={int(task['instance_id'])}."
        )
    checkpoint_estimates = values_traj[checkpoint_indices]
    l2_errors = np.linalg.norm(checkpoint_estimates - reference_phi[None, :], axis=1)
    method_payload = copy.deepcopy(method_spec)
    if base_method_name != task["method"]:
        method_payload["base_method"] = base_method_name
        method_payload["name"] = task["method"]
        method_payload["label"] = method_display_label(task["method"])

    payload = {
        "status": "ok",
        "task_index": int(task_index),
        "task": dict(task),
        "config_name": config_dir.name,
        "config_dir": str(config_dir),
        "dataset": dataset_spec,
        "semivalue": semivalue_spec,
        "method": method_payload,
        "n_players": n_players,
        "max_budget": int(max_budget_total),
        "budget_per_player": int(args.budget_per_player),
        "ease_switch_fraction": float(task_ease_switch_fraction),
        "ease_fo_pilot_design_updates": int(args.ease_fo_pilot_design_updates),
        "num_checkpoints": int(args.num_checkpoints),
        "ease_switch_budget": int(switch_budget),
        "ease_pilot_nue": (
            int(estimator_kwargs["pilot_nue"])
            if "pilot_nue" in estimator_kwargs
            else None
        ),
        "nue_avg": int(nue_avg),
        "nue_track_avg": int(nue_track_avg),
        "nue_per_proc": int(args.nue_per_proc),
        "requested_checkpoints": requested_checkpoints,
        "actual_budgets": actual_budgets,
        "checkpoint_indices": checkpoint_indices,
        "checkpoint_budgets": checkpoint_budgets,
        "checkpoint_estimates": checkpoint_estimates,
        "l2_errors": l2_errors,
        "reference_phi": reference_phi,
        "reference_metadata": reference_metadata,
        "values_final": values_final,
        "timing": timing,
        "estimator_kwargs": estimator_kwargs,
        "tabular_game_mode": args.tabular_game_mode,
        "elapsed_sec": time.perf_counter() - start,
    }
    if args.save_full_trajectory:
        payload["values_traj"] = values_traj
    atomic_save_pickle(path, payload)
    logger.info("task %d done in %.2fs -> %s", task_index, payload["elapsed_sec"], path)
    return "ok"


def save_failure(
    *,
    task_index: int,
    task: dict[str, Any],
    args: argparse.Namespace,
    config_dir: Path,
    raw_dir: Path,
    exc: BaseException,
) -> None:
    task_ease_switch_fraction = float(task.get("ease_switch_fraction", args.ease_switch_fraction))
    path = result_path(
        task["dataset"],
        task["semivalue"],
        task["method"],
        int(task["instance_id"]),
        int(task["run_idx"]),
        raw_dir=raw_dir,
    )
    payload = {
        "status": "failed",
        "task_index": int(task_index),
        "task": dict(task),
        "config_name": config_dir.name,
        "config_dir": str(config_dir),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "tabular_game_mode": args.tabular_game_mode,
        "budget_per_player": int(args.budget_per_player),
        "ease_switch_fraction": float(task_ease_switch_fraction),
        "ease_fo_pilot_design_updates": int(args.ease_fo_pilot_design_updates),
        "num_checkpoints": int(args.num_checkpoints),
        "failed_at_unix": time.time(),
    }
    atomic_save_pickle(path, payload)


def build_arg_parser() -> argparse.ArgumentParser:
    dataset_choices = [spec["name"] for spec in DATASET_SPECS]
    semivalue_choices = [spec["name"] for spec in SEMIVALUE_SPECS]
    method_choices = [spec["name"] for spec in METHOD_SPECS]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=dataset_choices)
    parser.add_argument("--semivalue", action="append", choices=semivalue_choices)
    parser.add_argument("--method", action="append", choices=method_choices)
    parser.add_argument(
        "--instance-id",
        action="append",
        help="Instance id, comma list, or inclusive range start:end. Can be repeated.",
    )
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--include-optional-polyshap", action="store_true")
    add_results_config_args(parser)
    add_ease_switch_fractions_arg(parser)
    parser.add_argument(
        "--exclude-method",
        action="append",
        choices=method_choices,
        default=[],
        help="Method to exclude from the task grid. Can be repeated.",
    )
    parser.add_argument("--nue-per-proc", type=int, default=5000)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--only-reference", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--verbose-estimator", action="store_true")
    parser.add_argument("--save-full-trajectory", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_results_config_args(args)

    config_dir = resolve_config_dir_from_args(args)
    raw_dir = config_raw_dir(config_dir)
    groundtruth_dir = GROUNDTRUTH_DIR

    if args.task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        args.task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        args.num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    instance_ids = parse_int_set(args.instance_id)
    tasks = iter_task_specs(
        datasets=args.dataset,
        semivalues=args.semivalue,
        methods=args.method,
        instance_ids=instance_ids,
        n_runs=args.n_runs,
        include_optional=args.include_optional_polyshap,
        ease_switch_fractions=args.ease_switch_fractions,
        exclude_methods=set(args.exclude_method or []),
    )
    if not tasks:
        raise ValueError(
            "Task grid is empty. Check --dataset, --semivalue, --method, "
            "--exclude-method, and --include-optional-polyshap."
        )
    selected = select_tasks(tasks, task_id=args.task_id, num_tasks=args.num_tasks)

    if args.list_tasks:
        print(f"config_dir={config_dir}")
        print(f"raw_dir={raw_dir}")
        print(f"groundtruth_dir={groundtruth_dir}")
        print(f"total_tasks={len(tasks)}")
        print(f"selected_tasks={len(selected)}")
        for task_index, task in selected[:20]:
            print(f"{task_index}: {task}")
        if len(selected) > 20:
            print(f"... {len(selected) - 20} more selected tasks")
        return 0

    logger = setup_logging(
        config_dir=config_dir,
        raw_dir=raw_dir,
        groundtruth_dir=groundtruth_dir,
    )
    write_config_metadata(
        args=args,
        config_dir=config_dir,
        raw_dir=raw_dir,
        groundtruth_dir=groundtruth_dir,
    )
    validate_method_specs(METHOD_SPECS)
    logger.info("total tasks=%d selected=%d", len(tasks), len(selected))
    status_counts: dict[str, int] = {}
    for task_index, task in selected:
        try:
            status = run_one_task(
                task_index=task_index,
                task=task,
                args=args,
                config_dir=config_dir,
                raw_dir=raw_dir,
                groundtruth_dir=groundtruth_dir,
                logger=logger,
            )
        except Exception as exc:
            logger.exception("task %d failed", task_index)
            status_counts["failed"] = status_counts.get("failed", 0) + 1
            if args.allow_failures:
                save_failure(
                    task_index=task_index,
                    task=task,
                    args=args,
                    config_dir=config_dir,
                    raw_dir=raw_dir,
                    exc=exc,
                )
                continue
            raise
        status_counts[status] = status_counts.get(status, 0) + 1

    logger.info("finished: %s", status_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
