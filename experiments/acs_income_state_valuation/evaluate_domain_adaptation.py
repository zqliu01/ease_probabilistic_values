"""Evaluate direct PA transfer scores and top-k state selection curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import log_loss, roc_auc_score

import acs_data
import acs_model
import acs_state_game


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "plot_config.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "domain_adaptation_scores"
ACS_STATES = acs_data.US_STATES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON file with primary_run_dir. Ignored when --benchmark-run-dir is passed.",
    )
    parser.add_argument(
        "--benchmark-run-dir",
        type=Path,
        default=None,
        help="Benchmark run directory containing reference/reference_values.csv.",
    )
    parser.add_argument("--survey-year", default=None)
    parser.add_argument("--target-state", type=str.upper, default=None, choices=ACS_STATES)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--encoder", default=None, choices=["full", "semantic65"])
    parser.add_argument(
        "--utility-model",
        default=None,
        choices=["logistic", "xgboost", "gbm"],
        help=(
            "Model used for direct-transfer and top-k utility evaluation. "
            "Defaults to the benchmark run config when present."
        ),
    )
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help=(
            "Allow top-k evaluation with a utility model different from the one "
            "used to produce the benchmark Shapley ranking."
        ),
    )
    parser.add_argument("--fixed-lambda", type=float, default=None)
    parser.add_argument(
        "--solver",
        default=None,
        choices=["lbfgs", "liblinear", "newton-cholesky", "saga"],
    )
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--xgb-n-estimators", type=int, default=None)
    parser.add_argument("--xgb-max-depth", type=int, default=None)
    parser.add_argument("--xgb-learning-rate", type=float, default=None)
    parser.add_argument("--xgb-tree-method", default=None)
    parser.add_argument("--xgb-n-jobs", type=int, default=None)
    parser.add_argument("--xgb-subsample", type=float, default=None)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=None)
    parser.add_argument("--random-num-seeds", type=int, default=20)
    parser.add_argument("--random-seed-start", type=int, default=2026)
    parser.add_argument(
        "--holdout-eval-size",
        type=int,
        default=0,
        help=(
            "If positive, also evaluate the same top-k selection orders on this "
            "many fresh target-state samples disjoint from the original "
            "evaluation set and target-state training rows."
        ),
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=3026,
        help="Random seed for sampling the alternate target-state evaluation set.",
    )
    parser.add_argument(
        "--holdout-num-seeds",
        type=int,
        default=1,
        help=(
            "Number of consecutive holdout seeds to evaluate. Seed j uses "
            "--holdout-seed + j."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def resolve_benchmark_run_dir(args: argparse.Namespace) -> Path:
    if args.benchmark_run_dir is not None:
        return args.benchmark_run_dir.resolve()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No --benchmark-run-dir was passed and config file does not exist: {config_path}"
        )
    config = json.loads(config_path.read_text())
    try:
        configured_run_dir = Path(config["primary_run_dir"])
    except KeyError as exc:
        raise KeyError(f"Missing primary_run_dir in {config_path}") from exc
    if configured_run_dir.is_absolute():
        return configured_run_dir
    return (config_path.parent / configured_run_dir).resolve()


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing benchmark config: {config_path}")
    return json.loads(config_path.read_text())


def choose_arg(args: argparse.Namespace, run_config: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return run_config.get(name, default)


def data_args_from(args: argparse.Namespace, run_config: dict[str, Any]) -> argparse.Namespace:
    if args.data_dir is None:
        # Historical benchmark configs may contain machine-specific absolute paths.
        # Use the current repo data directory unless the caller explicitly overrides it.
        data_dir = acs_data.DEFAULT_DATA_DIR
    else:
        data_dir = args.data_dir
    return argparse.Namespace(
        survey_year=choose_arg(args, run_config, "survey_year", "2018"),
        target_state=choose_arg(args, run_config, "target_state", "PA"),
        train_size=int(choose_arg(args, run_config, "train_size", 500)),
        eval_size=int(choose_arg(args, run_config, "eval_size", 1000)),
        seed=int(choose_arg(args, run_config, "data_seed", 2026)),
        data_dir=Path(data_dir),
        download=bool(args.download),
        encoder=choose_arg(args, run_config, "encoder", "semantic65"),
    )


def model_args_from(args: argparse.Namespace, run_config: dict[str, Any]) -> argparse.Namespace:
    benchmark_utility_model = acs_state_game.normalize_utility_model(
        run_config.get("utility_model", "logistic")
    )
    utility_model = acs_state_game.normalize_utility_model(
        args.utility_model
        if args.utility_model is not None
        else benchmark_utility_model
    )
    if utility_model != benchmark_utility_model and not args.allow_model_mismatch:
        raise ValueError(
            f"Benchmark Shapley values use {benchmark_utility_model!r}, but top-k "
            f"evaluation requested {utility_model!r}. Omit --utility-model to inherit "
            "the benchmark model, or pass --allow-model-mismatch for an intentional "
            "cross-model comparison."
        )
    fixed_lambda = float(choose_arg(args, run_config, "fixed_lambda", 1.0))
    max_iter = int(choose_arg(args, run_config, "max_iter", 5000))
    if utility_model == "logistic":
        if fixed_lambda <= 0.0:
            raise ValueError(
                "--fixed-lambda must be positive for LogisticRegression C=1/lambda."
            )
        if max_iter <= 0:
            raise ValueError("--max-iter must be positive for logistic utility.")
    xgb_learning_rate = float(choose_arg(args, run_config, "xgb_learning_rate", 0.1))
    xgb_subsample = float(choose_arg(args, run_config, "xgb_subsample", 1.0))
    xgb_colsample_bytree = float(
        choose_arg(args, run_config, "xgb_colsample_bytree", 1.0)
    )
    xgb_n_estimators = int(choose_arg(args, run_config, "xgb_n_estimators", 5))
    xgb_max_depth = int(choose_arg(args, run_config, "xgb_max_depth", 5))
    xgb_n_jobs = int(choose_arg(args, run_config, "xgb_n_jobs", 1))
    if utility_model == "xgboost":
        acs_state_game.require_xgb_classifier()
        if xgb_learning_rate <= 0.0:
            raise ValueError("--xgb-learning-rate must be positive.")
        for name, value in {
            "xgb_n_estimators": xgb_n_estimators,
            "xgb_max_depth": xgb_max_depth,
            "xgb_n_jobs": xgb_n_jobs,
        }.items():
            if value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")
        for name, value in {
            "xgb_subsample": xgb_subsample,
            "xgb_colsample_bytree": xgb_colsample_bytree,
        }.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1].")
    return argparse.Namespace(
        benchmark_utility_model=benchmark_utility_model,
        utility_model=utility_model,
        fixed_lambda=fixed_lambda,
        solver=choose_arg(args, run_config, "solver", "liblinear"),
        max_iter=max_iter,
        model_seed=int(choose_arg(args, run_config, "model_seed", 2026)),
        xgb_n_estimators=xgb_n_estimators,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        xgb_tree_method=choose_arg(args, run_config, "xgb_tree_method", "hist"),
        xgb_n_jobs=xgb_n_jobs,
        xgb_subsample=xgb_subsample,
        xgb_colsample_bytree=xgb_colsample_bytree,
    )


def make_output_dir(
    *,
    args: argparse.Namespace,
    data_args: argparse.Namespace,
    model_args: argparse.Namespace,
    benchmark_run_dir: Path,
) -> Path:
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_part = "xgb" if model_args.utility_model == "xgboost" else "logistic"
        run_name = (
            f"{data_args.target_state.lower()}_{data_args.encoder}"
            f"_train{data_args.train_size}_eval{data_args.eval_size}"
            f"_{model_part}_from_{benchmark_run_dir.name}_{timestamp}"
        )
    else:
        run_name = args.run_name
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_shapley_values(run_dir: Path) -> pd.DataFrame:
    values_path = run_dir / "reference" / "reference_values.csv"
    if not values_path.is_file():
        raise FileNotFoundError(f"Missing reference values: {values_path}")
    values = pd.read_csv(values_path)
    required = {"state", "value"}
    missing = required - set(values.columns)
    if missing:
        raise ValueError(f"{values_path} is missing columns: {sorted(missing)}")
    values = values[["state", "value"]].copy()
    missing_states = sorted(set(ACS_STATES) - set(values["state"]))
    if missing_states:
        raise ValueError(f"Reference values are missing states: {missing_states}")
    values["shapley_value"] = values["value"].astype(float)
    values = values.drop(columns=["value"])
    values["shapley_rank"] = values["shapley_value"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    return values.sort_values("shapley_rank").reset_index(drop=True)


def evaluate_state_set(
    *,
    encoded_split: acs_model.EncodedACSSplit,
    states: list[str],
    model_args: argparse.Namespace,
) -> dict[str, Any]:
    fit_start = time.perf_counter()
    x_train = sparse.vstack(
        [encoded_split.encoded_train[state] for state in states],
        format="csr",
    )
    y_train = np.concatenate([encoded_split.train_labels[state] for state in states])
    if len(np.unique(y_train)) < 2:
        p = float(np.mean(y_train))
        mode = "constant_train_prior"
    else:
        model, mode = acs_state_game.fit_utility_model(
            x_train,
            y_train,
            utility_model=model_args.utility_model,
            fixed_lambda=model_args.fixed_lambda,
            solver=model_args.solver,
            max_iter=model_args.max_iter,
            seed=model_args.model_seed,
            xgb_n_estimators=model_args.xgb_n_estimators,
            xgb_max_depth=model_args.xgb_max_depth,
            xgb_learning_rate=model_args.xgb_learning_rate,
            xgb_tree_method=model_args.xgb_tree_method,
            xgb_n_jobs=model_args.xgb_n_jobs,
            xgb_subsample=model_args.xgb_subsample,
            xgb_colsample_bytree=model_args.xgb_colsample_bytree,
        )
    fit_sec = time.perf_counter() - fit_start
    eval_start = time.perf_counter()
    if len(np.unique(y_train)) < 2:
        scores = np.full(
            len(encoded_split.eval_y),
            float(np.clip(p, 1e-6, 1 - 1e-6)),
        )
    else:
        scores = model.predict_proba(encoded_split.encoded_eval)[:, 1]
    loss = float(log_loss(encoded_split.eval_y, scores, labels=[0, 1]))
    try:
        auc = float(roc_auc_score(encoded_split.eval_y, scores))
    except ValueError:
        auc = float("nan")
    eval_sec = time.perf_counter() - eval_start
    return {
        "log_loss": loss,
        "utility": -loss,
        "auc": auc,
        "n_train": int(len(y_train)),
        "train_label_rate": float(np.mean(y_train)),
        "mode": mode,
        "fit_sec": fit_sec,
        "eval_sec": eval_sec,
        "fit_eval_sec": fit_sec + eval_sec,
    }


def evaluate_single_state_scores(
    *,
    encoded_split: acs_model.EncodedACSSplit,
    model_args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for state in ACS_STATES:
        metrics = evaluate_state_set(
            encoded_split=encoded_split,
            states=[state],
            model_args=model_args,
        )
        rows.append({"state": state, **metrics})
        print(
            f"[single-state] {state} log_loss={metrics['log_loss']:.6f} "
            f"auc={metrics['auc']:.4f}"
        )
    scores = pd.DataFrame(rows)
    scores["direct_transfer_rank"] = scores["utility"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    return scores.sort_values("direct_transfer_rank").reset_index(drop=True)


def topk_specs(
    *,
    shapley_values: pd.DataFrame,
    single_state_scores: pd.DataFrame,
    random_num_seeds: int,
    random_seed_start: int,
) -> list[dict[str, Any]]:
    shapley_order = shapley_values.sort_values("shapley_rank")["state"].to_list()
    direct_order = single_state_scores.sort_values("direct_transfer_rank")["state"].to_list()
    specs: list[dict[str, Any]] = []

    def append_order(
        *,
        strategy: str,
        strategy_label: str,
        states_order: list[str],
        selection_seed_index: int | None = None,
        selection_seed: int | None = None,
    ) -> None:
        for k in range(1, len(states_order) + 1):
            specs.append(
                {
                    "strategy": strategy,
                    "strategy_label": strategy_label,
                    "selection_seed_index": selection_seed_index,
                    "selection_seed": selection_seed,
                    "k": k,
                    "states": states_order[:k],
                }
            )

    append_order(
        strategy="shapley_topk",
        strategy_label="Top-k by Shapley value",
        states_order=shapley_order,
    )
    append_order(
        strategy="direct_transfer_topk",
        strategy_label="Top-k by direct transfer",
        states_order=direct_order,
    )
    for seed_index in range(random_num_seeds):
        seed = random_seed_start + seed_index
        rng = np.random.default_rng(seed)
        append_order(
            strategy="random",
            strategy_label="Random k states",
            states_order=list(rng.permutation(ACS_STATES)),
            selection_seed_index=seed_index,
            selection_seed=seed,
        )
    return specs


def metrics_from_scores(
    *,
    eval_y: np.ndarray,
    scores: np.ndarray,
    y_train: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    loss = float(log_loss(eval_y, scores, labels=[0, 1]))
    try:
        auc = float(roc_auc_score(eval_y, scores))
    except ValueError:
        auc = float("nan")
    return {
        "log_loss": loss,
        "utility": -loss,
        "auc": auc,
        "n_train": int(len(y_train)),
        "train_label_rate": float(np.mean(y_train)),
        "mode": mode,
    }


def evaluate_topk_specs_on_eval_splits(
    *,
    encoded_split: acs_model.EncodedACSSplit,
    model_args: argparse.Namespace,
    specs: list[dict[str, Any]],
    eval_splits: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs, start=1):
        states = list(spec["states"])
        fit_start = time.perf_counter()
        x_train = sparse.vstack(
            [encoded_split.encoded_train[state] for state in states],
            format="csr",
        )
        y_train = np.concatenate([encoded_split.train_labels[state] for state in states])
        if len(np.unique(y_train)) < 2:
            p = float(np.mean(y_train))
            model = None
            mode = "constant_train_prior"
        else:
            model, mode = acs_state_game.fit_utility_model(
                x_train,
                y_train,
                utility_model=model_args.utility_model,
                fixed_lambda=model_args.fixed_lambda,
                solver=model_args.solver,
                max_iter=model_args.max_iter,
                seed=model_args.model_seed,
                xgb_n_estimators=model_args.xgb_n_estimators,
                xgb_max_depth=model_args.xgb_max_depth,
                xgb_learning_rate=model_args.xgb_learning_rate,
                xgb_tree_method=model_args.xgb_tree_method,
                xgb_n_jobs=model_args.xgb_n_jobs,
                xgb_subsample=model_args.xgb_subsample,
                xgb_colsample_bytree=model_args.xgb_colsample_bytree,
            )
        fit_sec = time.perf_counter() - fit_start

        for eval_index, eval_info in enumerate(eval_splits):
            eval_start = time.perf_counter()
            if model is None:
                scores = np.full(
                    len(eval_info["eval_y"]),
                    float(np.clip(p, 1e-6, 1 - 1e-6)),
                )
            else:
                scores = model.predict_proba(eval_info["encoded_eval"])[:, 1]
            metrics = metrics_from_scores(
                eval_y=eval_info["eval_y"],
                scores=scores,
                y_train=y_train,
                mode=mode,
            )
            eval_sec = time.perf_counter() - eval_start
            # Charge the shared fit once so fit_eval_sec sums to actual executed work.
            charged_fit_sec = fit_sec if eval_index == 0 else 0.0
            rows.append(
                {
                    "strategy": spec["strategy"],
                    "strategy_label": spec["strategy_label"],
                    "selection_seed_index": spec["selection_seed_index"],
                    "selection_seed": spec["selection_seed"],
                    "evaluation_split": eval_info["name"],
                    "holdout_seed_index": eval_info["seed_index"],
                    "holdout_seed": eval_info["seed"],
                    "holdout_label_rate": eval_info["label_rate"],
                    "k": spec["k"],
                    "states": ",".join(states),
                    "model_fit_index": spec_index - 1,
                    "model_fit_sec": fit_sec,
                    "model_fit_reused": eval_index > 0,
                    "eval_sec": eval_sec,
                    "fit_eval_sec": charged_fit_sec + eval_sec,
                    **metrics,
                }
            )
        if spec_index % 50 == 0 or spec_index == len(specs):
            print(f"[top-k] evaluated {spec_index}/{len(specs)} models")
    return pd.DataFrame(rows)


def make_holdout_eval_split(
    *,
    encoded_split: acs_model.EncodedACSSplit,
    data_args: argparse.Namespace,
    target_x: pd.DataFrame,
    target_y: np.ndarray,
    holdout_eval_size: int,
    holdout_seed: int,
) -> tuple[acs_model.EncodedACSSplit, dict[str, Any]]:
    """Sample a second target-state eval set disjoint from benchmark eval/train rows."""

    if holdout_eval_size <= 0:
        raise ValueError("holdout_eval_size must be positive.")

    manifest = encoded_split.processed_split.manifest
    excluded_indices = set(int(index) for index in manifest["eval"]["indices"])
    excluded_indices.update(
        int(index)
        for index in manifest["train"][data_args.target_state]["indices"]
    )
    available = np.setdiff1d(
        np.arange(len(target_y), dtype=np.int64),
        np.asarray(sorted(excluded_indices), dtype=np.int64),
        assume_unique=False,
    )
    if len(available) < holdout_eval_size:
        raise ValueError(
            f"Need {holdout_eval_size} holdout {data_args.target_state} samples, "
            f"but only {len(available)} remain after excluding benchmark eval "
            "and target-state training rows."
        )

    rng = np.random.default_rng(holdout_seed)
    holdout_indices = rng.permutation(available)[:holdout_eval_size]
    holdout_x = target_x.iloc[holdout_indices].reset_index(drop=True)
    holdout_y = target_y[holdout_indices]
    encoded_eval = acs_model.transform_eval_frame(
        holdout_x,
        preprocessor=encoded_split.preprocessor,
        encoder=data_args.encoder,
        target_state=data_args.target_state,
    )
    holdout_split = replace(
        encoded_split,
        encoded_eval=encoded_eval,
        eval_y=holdout_y,
    )
    metadata = {
        "target_state": data_args.target_state,
        "eval_size": int(holdout_eval_size),
        "seed": int(holdout_seed),
        "indices": [int(index) for index in holdout_indices],
        "excluded_original_eval_and_target_train": True,
        "excluded_count": int(len(excluded_indices)),
        "available_count": int(len(available)),
        "label_rate": float(np.mean(holdout_y)),
    }
    return holdout_split, metadata


def summarize_topk(topk: pd.DataFrame) -> pd.DataFrame:
    summary = (
        topk.groupby(["strategy", "strategy_label", "k"], sort=False)
        .agg(
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            log_loss_sem=("log_loss", "sem"),
            log_loss_q10=("log_loss", lambda values: values.quantile(0.10)),
            log_loss_q90=("log_loss", lambda values: values.quantile(0.90)),
            utility_mean=("utility", "mean"),
            utility_std=("utility", "std"),
            utility_sem=("utility", "sem"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_sem=("auc", "sem"),
            count=("log_loss", "count"),
        )
        .reset_index()
    )
    fill_cols = [
        "log_loss_std",
        "log_loss_sem",
        "utility_std",
        "utility_sem",
        "auc_std",
        "auc_sem",
    ]
    summary[fill_cols] = summary[fill_cols].fillna(0.0)
    return summary


def write_outputs(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    data_args: argparse.Namespace,
    model_args: argparse.Namespace,
    benchmark_run_dir: Path,
    encoded_split: acs_model.EncodedACSSplit,
    shapley_values: pd.DataFrame,
    single_state_scores: pd.DataFrame,
    topk: pd.DataFrame,
    topk_summary: pd.DataFrame,
    holdout_topk: pd.DataFrame | None = None,
    holdout_topk_summary: pd.DataFrame | None = None,
    holdout_metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    if holdout_metadata is None:
        holdout_metadata_list: list[dict[str, Any]] = []
    elif isinstance(holdout_metadata, list):
        holdout_metadata_list = holdout_metadata
    else:
        holdout_metadata_list = [holdout_metadata]

    paths = {
        "shapley_values": output_dir / "shapley_values.csv",
        "single_state_scores": output_dir / "single_state_scores.csv",
        "topk_selection_long": output_dir / "topk_selection_long.csv",
        "topk_summary": output_dir / "topk_summary.csv",
        "config": output_dir / "config.json",
    }
    if holdout_topk is not None:
        paths["holdout_topk_selection_long"] = output_dir / "holdout_topk_selection_long.csv"
    if holdout_topk_summary is not None:
        paths["holdout_topk_summary"] = output_dir / "holdout_topk_summary.csv"
    if holdout_metadata_list:
        paths["holdout_eval_indices"] = output_dir / "holdout_eval_indices.csv"
    shapley_values.to_csv(paths["shapley_values"], index=False)
    single_state_scores.to_csv(paths["single_state_scores"], index=False)
    topk.to_csv(paths["topk_selection_long"], index=False)
    topk_summary.to_csv(paths["topk_summary"], index=False)
    if holdout_topk is not None:
        holdout_topk.to_csv(paths["holdout_topk_selection_long"], index=False)
    if holdout_topk_summary is not None:
        holdout_topk_summary.to_csv(paths["holdout_topk_summary"], index=False)
    if holdout_metadata_list:
        index_rows = []
        for metadata in holdout_metadata_list:
            for row_index in metadata["indices"]:
                index_rows.append(
                    {
                        "holdout_seed_index": metadata.get("seed_index"),
                        "holdout_seed": metadata["seed"],
                        "row_index": row_index,
                    }
                )
        pd.DataFrame(index_rows).to_csv(paths["holdout_eval_indices"], index=False)
    config = {
        "benchmark_run_dir": str(benchmark_run_dir),
        "benchmark_config_sha256": file_sha256(benchmark_run_dir / "config.json"),
        "output_dir": str(output_dir),
        "survey_year": data_args.survey_year,
        "target_state": data_args.target_state,
        "train_size": data_args.train_size,
        "eval_size": data_args.eval_size,
        "data_seed": data_args.seed,
        "data_dir": str(data_args.data_dir),
        "encoder": data_args.encoder,
        "benchmark_utility_model": model_args.benchmark_utility_model,
        "utility_model": model_args.utility_model,
        "allow_model_mismatch": bool(args.allow_model_mismatch),
        "fixed_lambda": model_args.fixed_lambda,
        "solver": model_args.solver,
        "max_iter": model_args.max_iter,
        "model_seed": model_args.model_seed,
        "xgb_n_estimators": model_args.xgb_n_estimators,
        "xgb_max_depth": model_args.xgb_max_depth,
        "xgb_learning_rate": model_args.xgb_learning_rate,
        "xgb_tree_method": model_args.xgb_tree_method,
        "xgb_n_jobs": model_args.xgb_n_jobs,
        "xgb_subsample": model_args.xgb_subsample,
        "xgb_colsample_bytree": model_args.xgb_colsample_bytree,
        "random_num_seeds": int(args.random_num_seeds),
        "random_seed_start": int(args.random_seed_start),
        "holdout_eval_size": int(args.holdout_eval_size),
        "holdout_seed": int(args.holdout_seed),
        "holdout_num_seeds": int(args.holdout_num_seeds),
        "states": ACS_STATES,
        "processed_split_dir": str(encoded_split.processed_split.split_dir),
        "sample_manifest_path": str(encoded_split.processed_split.manifest_path),
        "n_features": int(encoded_split.feature_block_dims["total"]),
        "eval_label_rate": float(np.mean(encoded_split.eval_y)),
    }
    config["holdout_eval"] = [
        {
            key: value
            for key, value in metadata.items()
            if key != "indices"
        }
        for metadata in holdout_metadata_list
    ]
    if holdout_metadata_list:
        config["holdout_eval_indices_path"] = str(paths["holdout_eval_indices"])
    paths["config"].write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return paths


def main() -> None:
    args = parse_args()
    benchmark_run_dir = resolve_benchmark_run_dir(args)
    run_config = load_run_config(benchmark_run_dir)
    data_args = data_args_from(args, run_config)
    model_args = model_args_from(args, run_config)
    if args.random_num_seeds <= 0:
        raise ValueError("--random-num-seeds must be positive.")
    if args.holdout_eval_size < 0:
        raise ValueError("--holdout-eval-size must be nonnegative.")
    if args.holdout_eval_size > 0 and args.holdout_num_seeds <= 0:
        raise ValueError("--holdout-num-seeds must be positive when holdout is enabled.")

    output_dir = make_output_dir(
        args=args,
        data_args=data_args,
        model_args=model_args,
        benchmark_run_dir=benchmark_run_dir,
    )

    start = time.perf_counter()
    encoded_split = acs_model.prepare_encoded_acs_split(data_args)
    shapley_values = load_shapley_values(benchmark_run_dir)
    print(f"Utility model: {model_args.utility_model}")
    single_state_scores = evaluate_single_state_scores(
        encoded_split=encoded_split,
        model_args=model_args,
    )
    specs = topk_specs(
        shapley_values=shapley_values,
        single_state_scores=single_state_scores,
        random_num_seeds=int(args.random_num_seeds),
        random_seed_start=int(args.random_seed_start),
    )
    eval_splits: list[dict[str, Any]] = [
        {
            "name": "validation",
            "seed_index": None,
            "seed": None,
            "label_rate": float(np.mean(encoded_split.eval_y)),
            "encoded_eval": encoded_split.encoded_eval,
            "eval_y": encoded_split.eval_y,
        }
    ]
    holdout_metadata: list[dict[str, Any]] = []
    if args.holdout_eval_size > 0:
        target_x, target_y = acs_data.load_state(
            raw_dir=encoded_split.processed_split.data_dirs.raw_dir,
            survey_year=data_args.survey_year,
            state=data_args.target_state,
            download=data_args.download,
        )
        for seed_index in range(int(args.holdout_num_seeds)):
            seed = int(args.holdout_seed) + seed_index
            holdout_split, metadata = make_holdout_eval_split(
                encoded_split=encoded_split,
                data_args=data_args,
                target_x=target_x,
                target_y=target_y,
                holdout_eval_size=int(args.holdout_eval_size),
                holdout_seed=seed,
            )
            metadata["seed_index"] = seed_index
            holdout_metadata.append(metadata)
            eval_splits.append(
                {
                    "name": "holdout",
                    "seed_index": seed_index,
                    "seed": seed,
                    "label_rate": metadata["label_rate"],
                    "encoded_eval": holdout_split.encoded_eval,
                    "eval_y": holdout_split.eval_y,
                }
            )
            print(
                f"[holdout] seed={seed} sampled {metadata['eval_size']} "
                f"{metadata['target_state']} rows "
                f"(label_rate={metadata['label_rate']:.4f})"
            )
    all_topk = evaluate_topk_specs_on_eval_splits(
        encoded_split=encoded_split,
        model_args=model_args,
        specs=specs,
        eval_splits=eval_splits,
    )
    validation_metadata_columns = [
        "evaluation_split",
        "holdout_seed_index",
        "holdout_seed",
        "holdout_label_rate",
    ]
    topk = (
        all_topk.loc[all_topk["evaluation_split"] == "validation"]
        .drop(columns=validation_metadata_columns)
        .reset_index(drop=True)
    )
    topk_summary = summarize_topk(topk)
    holdout_topk = None
    holdout_topk_summary = None
    if holdout_metadata:
        holdout_topk = (
            all_topk.loc[all_topk["evaluation_split"] == "holdout"]
            .drop(columns=["evaluation_split"])
            .reset_index(drop=True)
        )
        holdout_topk_summary = summarize_topk(holdout_topk)
    paths = write_outputs(
        output_dir=output_dir,
        args=args,
        data_args=data_args,
        model_args=model_args,
        benchmark_run_dir=benchmark_run_dir,
        encoded_split=encoded_split,
        shapley_values=shapley_values,
        single_state_scores=single_state_scores,
        topk=topk,
        topk_summary=topk_summary,
        holdout_topk=holdout_topk,
        holdout_topk_summary=holdout_topk_summary,
        holdout_metadata=holdout_metadata,
    )
    elapsed = time.perf_counter() - start
    print(f"Output directory: {output_dir}")
    print(f"Elapsed seconds: {elapsed:.1f}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
