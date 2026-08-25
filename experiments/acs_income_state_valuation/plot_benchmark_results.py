"""Plot paper-facing ACSIncome benchmark figures from saved outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = SCRIPT_DIR / "outputs" / "benchmark_shapley_estimators"
DEFAULT_DOMAIN_ADAPTATION_ROOT = SCRIPT_DIR / "outputs" / "domain_adaptation_scores"
DEFAULT_FIGURES_ROOT = SCRIPT_DIR / "figures"
FIGURE_STEMS = (
    "against_reference_log",
    "runtime_comparison",
    "topk_domain_adaptation",
    "topk_domain_adaptation_holdout",
)
FIGURE_SUFFIXES = (".png", ".svg", ".pdf")
DISPLAY_LABEL_BY_METHOD = {
    "LeverageSHAP_paired_implicit_polyshap2": "LeverageSHAP",
}
EXCLUDED_METHODS_FROM_METHOD_PLOTS = {
    "EaseSHAP_size_player",
    "OFA_baseline",
}
METHOD_STYLE_BY_METHOD = {
    "EaseSHAP_interaction_nonlinear": {"color": "#1F77B4", "marker": "o"},
    "GELS": {"color": "#2CA02C", "marker": "s"},
    "LeverageSHAP": {"color": "#D62728", "marker": "^"},
    "LeverageSHAP_paired_implicit_polyshap2": {"color": "#D62728", "marker": "^"},
    "OFA_fixed": {"color": "#8C564B", "marker": "D"},
    "RegressionMSR_unbiased": {"color": "#E377C2", "marker": "h"},
    "SHAP_IQ": {"color": "#7F7F7F", "marker": "X"},
    "complement": {"color": "#BCBD22", "marker": "v"},
    "group_testing": {"color": "#17BECF", "marker": "<"},
    "improved_AME": {"color": "#9467BD", "marker": "P"},
    "kernelSHAP": {"color": "#FF7F0E", "marker": ">"},
    "permutation": {"color": "#111111", "marker": "*"},
    "sampling_lift": {"color": "#A55194", "marker": "p"},
}
DOMAIN_ADAPTATION_STYLE = {
    "shapley_topk": {"color": "#4C78A8", "marker": "o"},
    "direct_transfer_topk": {"color": "#F58518", "marker": "s"},
    "random": {"color": "#54A24B", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Legacy JSON file selecting one benchmark and domain-adaptation run.",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        type=Path,
        action="append",
        help=(
            "Benchmark run directory to plot. May be repeated. Required unless "
            "--config or --all-runs is used."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Root containing benchmark run directories used by --all-runs.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Explicitly discover every structurally complete benchmark under "
            "--runs-root. This may include outputs from older code versions."
        ),
    )
    parser.add_argument(
        "--domain-adaptation-dir",
        dest="domain_adaptation_dirs",
        type=Path,
        action="append",
        help=(
            "Domain-adaptation run to associate through its benchmark fingerprint "
            "or exact recorded path. May be repeated. When omitted, runs under "
            "--domain-adaptation-root are discovered."
        ),
    )
    parser.add_argument(
        "--domain-adaptation-root",
        type=Path,
        default=DEFAULT_DOMAIN_ADAPTATION_ROOT,
        help="Root containing domain-adaptation runs for automatic discovery.",
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=DEFAULT_FIGURES_ROOT,
        help="Each run is written to <figures-root>/<benchmark-run-name>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Legacy exact output directory; valid only when plotting one run.",
    )
    parser.add_argument(
        "--runtime-band",
        "--band",
        dest="runtime_band",
        choices=["std", "sem"],
        default="std",
        help="Runtime error bar over estimator seeds. --band is kept as a compatibility alias.",
    )
    return parser.parse_args()


def load_plot_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    return json.loads(config_path.read_text())


def configure_matplotlib() -> None:
    mplconfig_dir = SCRIPT_DIR / ".mplconfig"
    mplconfig_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib

    matplotlib.use("Agg", force=True)


def resolve_configured_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def benchmark_input_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return (
        run_dir / "trajectory_long.csv",
        run_dir / "runs.csv",
        run_dir / "reference" / "reference_summary.json",
    )


def is_complete_benchmark_run(run_dir: Path) -> bool:
    return run_dir.is_dir() and all(
        path.is_file() for path in benchmark_input_paths(run_dir)
    )


def unique_resolved_paths(paths: list[Path]) -> list[Path]:
    unique: dict[Path, None] = {}
    for path in paths:
        unique[path.resolve()] = None
    return list(unique)


def resolve_run_dirs(
    args: argparse.Namespace,
    plot_config: dict[str, Any] | None,
) -> list[Path]:
    if args.run_dirs and plot_config is not None:
        raise ValueError("Use either --run-dir or --config, not both.")
    if args.all_runs and (args.run_dirs or plot_config is not None):
        raise ValueError("--all-runs cannot be combined with --run-dir or --config.")
    if args.run_dirs:
        run_dirs = unique_resolved_paths(args.run_dirs)
    elif plot_config is not None:
        try:
            configured_run = str(plot_config["primary_run_dir"])
        except KeyError as exc:
            raise KeyError(f"Missing primary_run_dir in {args.config}") from exc
        run_dirs = [resolve_configured_path(args.config.resolve(), configured_run)]
    elif args.all_runs:
        runs_root = args.runs_root.resolve()
        if not runs_root.is_dir():
            raise FileNotFoundError(f"Benchmark runs root does not exist: {runs_root}")
        run_dirs = sorted(
            path.resolve()
            for path in runs_root.iterdir()
            if is_complete_benchmark_run(path)
        )
    else:
        raise ValueError(
            "Select one or more runs with --run-dir, use the legacy --config option, "
            "or pass --all-runs for explicit directory discovery."
        )
    if not run_dirs:
        raise FileNotFoundError("No complete benchmark runs were found to plot.")
    return run_dirs


def resolve_domain_adaptation_dirs(
    args: argparse.Namespace,
    plot_config: dict[str, Any] | None,
) -> list[Path]:
    if args.domain_adaptation_dirs:
        return unique_resolved_paths(args.domain_adaptation_dirs)
    if plot_config is not None and plot_config.get("domain_adaptation_dir"):
        configured_domain = str(plot_config["domain_adaptation_dir"])
        return [resolve_configured_path(args.config.resolve(), configured_domain)]
    root = args.domain_adaptation_root.resolve()
    if not root.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir()
        and (path / "config.json").is_file()
        and (path / "topk_summary.csv").is_file()
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_domain_adaptation_runs(
    domain_dirs: list[Path],
    *,
    run_dirs: list[Path],
    require_all: bool,
) -> dict[Path, Path]:
    resolved_runs = [run_dir.resolve() for run_dir in run_dirs]
    runs_by_config_hash: dict[str, list[Path]] = {}
    for run_dir in resolved_runs:
        benchmark_config_path = run_dir / "config.json"
        if benchmark_config_path.is_file():
            runs_by_config_hash.setdefault(
                sha256(benchmark_config_path),
                [],
            ).append(run_dir)

    candidates: dict[Path, list[Path]] = {}
    for domain_dir in domain_dirs:
        config_path = domain_dir / "config.json"
        topk_path = domain_dir / "topk_summary.csv"
        for path in (config_path, topk_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing required domain-adaptation input: {path}")
        config = json.loads(config_path.read_text())
        benchmark_run_dir = config.get("benchmark_run_dir")
        if not benchmark_run_dir:
            raise KeyError(f"Missing benchmark_run_dir in {config_path}")

        benchmark_config_hash = config.get("benchmark_config_sha256")
        if benchmark_config_hash:
            matches = runs_by_config_hash.get(str(benchmark_config_hash), [])
            identity = f"config SHA-256 {benchmark_config_hash}"
        else:
            recorded_path = Path(str(benchmark_run_dir))
            if not recorded_path.is_absolute():
                recorded_path = config_path.parent / recorded_path
            recorded_path = recorded_path.resolve()
            matches = [run_dir for run_dir in resolved_runs if run_dir == recorded_path]
            identity = f"recorded path {recorded_path}"

        if not matches:
            if require_all:
                selected = ", ".join(str(path) for path in resolved_runs)
                raise ValueError(
                    f"Explicit domain-adaptation run {domain_dir} references {identity}, "
                    f"which does not match any selected benchmark run: {selected}"
                )
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Domain-adaptation run {domain_dir} references ambiguous {identity}: "
                + ", ".join(str(path) for path in matches)
            )
        candidates.setdefault(matches[0], []).append(domain_dir)

    selected: dict[Path, Path] = {}
    for benchmark_run, matches in candidates.items():
        if require_all and len(matches) > 1:
            raise ValueError(
                f"Multiple explicit domain-adaptation runs match {benchmark_run}: "
                + ", ".join(str(path) for path in matches)
            )
        selected_dir = max(
            matches,
            key=lambda path: ((path / "config.json").stat().st_mtime_ns, path.name),
        )
        selected[benchmark_run] = selected_dir
        if len(matches) > 1:
            print(
                f"[domain] {benchmark_run.name}: selected newest run "
                f"{selected_dir.name} from {len(matches)} candidates"
            )
    return selected


def relative_to_script(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path.resolve())


def display_label(method: str, method_label: str) -> str:
    return DISPLAY_LABEL_BY_METHOD.get(str(method), str(method_label))


def method_style(method: str) -> dict[str, str]:
    return METHOD_STYLE_BY_METHOD.get(str(method), {"color": "#4D4D4D", "marker": "o"})


def filter_method_plot_rows(table: Any) -> Any:
    return table[~table["method"].isin(EXCLUDED_METHODS_FROM_METHOD_PLOTS)].copy()


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.name == "manifest.json" or path.suffix in FIGURE_SUFFIXES:
            path.unlink()


def load_reference(run_dir: Path) -> tuple[str, Any]:
    summary_path = run_dir / "reference" / "reference_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing reference summary required for relative L2 plot: {summary_path}"
        )
    summary = json.loads(summary_path.read_text())
    try:
        values = summary["values"]
    except KeyError as exc:
        raise KeyError(f"Missing reference values in {summary_path}") from exc
    label = str(summary.get("label") or "reference")
    return label, values


def relative_sq_l2_by_checkpoint(trajectory: Any, reference_values: Any) -> Any:
    import numpy as np
    import pandas as pd

    state_order = list(trajectory["state"].drop_duplicates())
    reference = np.asarray(reference_values, dtype=np.float64)
    if len(state_order) != len(reference):
        raise ValueError(
            f"Trajectory has {len(state_order)} states but reference has {len(reference)} values."
        )
    denom = float(np.dot(reference, reference))
    if denom <= 0.0:
        denom = 1.0

    index_cols = [
        "method",
        "method_label",
        "seed_index",
        "estimator_seed",
        "checkpoint",
        "utility_evaluations",
    ]
    matrix = trajectory.pivot_table(
        index=index_cols,
        columns="state",
        values="value",
        aggfunc="first",
    )
    missing_states = [state for state in state_order if state not in matrix.columns]
    if missing_states:
        raise ValueError(f"Missing trajectory states: {missing_states}")
    values = matrix[state_order].to_numpy(dtype=np.float64)
    rel_sq_error = np.sum((values - reference[None, :]) ** 2, axis=1) / denom

    rows = matrix.index.to_frame(index=False)
    rows["relative_sq_l2_error"] = rel_sq_error
    summary = (
        rows.groupby(
            ["method", "method_label", "checkpoint", "utility_evaluations"],
            sort=False,
        )
        .agg(
            median=("relative_sq_l2_error", "median"),
            q10=("relative_sq_l2_error", lambda values: values.quantile(0.10)),
            q90=("relative_sq_l2_error", lambda values: values.quantile(0.90)),
            count=("relative_sq_l2_error", "count"),
        )
        .reset_index()
    )
    return rows, summary


def plot_against_reference_log(
    summary: Any,
    *,
    output_dir: Path,
) -> Path:
    configure_matplotlib()

    import matplotlib.pyplot as plt
    import numpy as np

    summary = filter_method_plot_rows(summary)
    if summary.empty:
        raise ValueError("No methods remain after applying method plot exclusions.")
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    positive_values = summary[["median", "q10"]].to_numpy(dtype=float).ravel()
    positive_values = positive_values[positive_values > 0]
    y_floor = float(np.min(positive_values) * 0.8) if len(positive_values) else np.finfo(float).tiny
    y_ceiling = 0.0
    for _, group in summary.groupby("method", sort=False):
        method = str(group["method"].iloc[0])
        style = method_style(method)
        label = display_label(method, group["method_label"].iloc[0])
        x = group["utility_evaluations"].to_numpy(dtype=float)
        y = group["median"].to_numpy(dtype=float)
        lower = np.maximum(group["q10"].to_numpy(dtype=float), y_floor)
        upper = np.maximum(group["q90"].to_numpy(dtype=float), y_floor)
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=4.0,
            label=label,
        )
        y_ceiling = max(y_ceiling, float(np.nanmax(upper)))
        ax.fill_between(x, lower, upper, color=style["color"], alpha=0.16, linewidth=0)

    ax.set_xlabel("Utility evaluations")
    ax.set_ylabel("Relative squared L2 error to reference")
    ax.set_yscale("log")
    if y_ceiling > 0.0:
        ax.set_ylim(bottom=y_floor, top=y_ceiling * 1.25)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()

    path = output_dir / "against_reference_log.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def runtime_summary(runs: Any, *, band: str) -> tuple[Any, str]:
    ok_runs = runs[(runs["status"] == "ok") & (runs["role"] == "main")].copy()
    ok_runs = filter_method_plot_rows(ok_runs)
    if ok_runs.empty:
        raise ValueError("runs.csv contains no successful main runs.")

    if "paper_runtime_sec" in ok_runs.columns:
        runtime_col = "paper_runtime_sec"
    elif "estimate_sec" in ok_runs.columns:
        runtime_col = "estimate_sec"
    else:
        raise KeyError("runs.csv needs paper_runtime_sec or estimate_sec.")
    ok_runs[runtime_col] = ok_runs[runtime_col].astype(float)
    summary = (
        ok_runs.groupby(["method", "method_label"], sort=False)[runtime_col]
        .agg(["mean", "std", "sem", "count"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )
    summary["spread"] = summary[band].fillna(0.0)
    summary["plot_label"] = [
        display_label(method, method_label)
        for method, method_label in zip(summary["method"], summary["method_label"])
    ]
    return summary, runtime_col


def plot_runtime_comparison(
    summary: Any,
    *,
    output_dir: Path,
    runtime_col: str,
) -> Path:
    configure_matplotlib()

    import matplotlib.pyplot as plt
    import numpy as np

    labels = summary["plot_label"].astype(str).to_list()
    y_pos = np.arange(len(summary))
    means = summary["mean"].to_numpy(dtype=float)
    spread = summary["spread"].to_numpy(dtype=float)
    colors = [method_style(method)["color"] for method in summary["method"]]

    height = max(4.8, 0.34 * len(summary) + 1.2)
    fig, ax = plt.subplots(figsize=(7.6, height))
    ax.barh(y_pos, means, xerr=spread, color=colors, alpha=0.88, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    if runtime_col == "paper_runtime_sec":
        xlabel = "Estimator runtime excluding checkpoint readouts (s)"
    else:
        xlabel = "Estimator runtime (s)"
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()

    path = output_dir / "runtime_comparison.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_topk_domain_adaptation(summary: Any, *, output_dir: Path) -> Path:
    return plot_topk_summary(
        summary,
        output_dir=output_dir,
        filename="topk_domain_adaptation.png",
        ylabel="PA evaluation log loss",
    )


def plot_topk_summary(
    summary: Any,
    *,
    output_dir: Path,
    filename: str,
    ylabel: str,
) -> Path:
    configure_matplotlib()

    import matplotlib.pyplot as plt
    import numpy as np

    required = {
        "strategy",
        "strategy_label",
        "k",
        "log_loss_mean",
        "log_loss_q10",
        "log_loss_q90",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"topk_summary.csv is missing columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    strategy_order = ["shapley_topk", "direct_transfer_topk", "random"]
    remaining = [
        strategy
        for strategy in summary["strategy"].drop_duplicates()
        if strategy not in strategy_order
    ]
    for strategy in [*strategy_order, *remaining]:
        group = summary[summary["strategy"] == strategy].sort_values("k")
        if group.empty:
            continue
        style = DOMAIN_ADAPTATION_STYLE.get(
            strategy,
            {"color": "#9D755D", "marker": "o"},
        )
        x = group["k"].to_numpy(dtype=float)
        y = group["log_loss_mean"].to_numpy(dtype=float)
        label = str(group["strategy_label"].iloc[0])
        ax.plot(
            x,
            y,
            label=label,
            color=style["color"],
            marker=style["marker"],
            markevery=max(1, int(np.ceil(len(group) / 12))),
            linewidth=2.1,
            markersize=4.5,
        )
        lower = group["log_loss_q10"].to_numpy(dtype=float)
        upper = group["log_loss_q90"].to_numpy(dtype=float)
        if np.any(np.abs(upper - lower) > 1e-12):
            alpha = 0.16 if strategy == "random" else 0.10
            ax.fill_between(x, lower, upper, color=style["color"], alpha=alpha, linewidth=0)

    ax.set_xlabel("Number of selected states")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def write_manifest(
    *,
    run_dir: Path,
    output_dir: Path,
    output_paths: list[Path],
    sources: dict[str, Path],
    runtime_band: str,
    runtime_col: str,
    domain_adaptation_dir: Path | None,
) -> Path:
    manifest = {
        "source_run": relative_to_script(run_dir),
        "domain_adaptation_dir": (
            None if domain_adaptation_dir is None else relative_to_script(domain_adaptation_dir)
        ),
        "reference_line": "median_relative_sq_l2_error",
        "reference_band": "q10_q90_relative_sq_l2_error",
        "domain_adaptation_line": "mean_pa_log_loss",
        "domain_adaptation_random_band": "q10_q90_pa_log_loss",
        "display_label_overrides": DISPLAY_LABEL_BY_METHOD,
        "excluded_method_plots": sorted(EXCLUDED_METHODS_FROM_METHOD_PLOTS),
        "method_style_by_method": METHOD_STYLE_BY_METHOD,
        "runtime_band": runtime_band,
        "runtime_column": runtime_col,
        "sources": {
            name: relative_to_script(path)
            for name, path in sources.items()
        },
        "figure_dir": relative_to_script(output_dir),
        "figures": [
            {
                "file": path.name,
                "sha256": sha256(path),
            }
            for path in output_paths
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def plot_run(
    *,
    run_dir: Path,
    domain_adaptation_dir: Path | None,
    output_dir: Path,
    runtime_band: str,
) -> tuple[Path, int]:
    trajectory_path, runs_path, reference_summary_path = benchmark_input_paths(run_dir)
    for path in (trajectory_path, runs_path, reference_summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")

    import pandas as pd

    clean_output_dir(output_dir)
    reference_label, reference_values = load_reference(run_dir)
    trajectory = pd.read_csv(trajectory_path)
    _, l2_summary = relative_sq_l2_by_checkpoint(trajectory, reference_values)
    runs = pd.read_csv(runs_path)
    runtime_table, runtime_col = runtime_summary(runs, band=runtime_band)

    sources = {
        "trajectory_long": trajectory_path,
        "runs": runs_path,
        "reference_summary": reference_summary_path,
    }
    output_paths = [
        plot_against_reference_log(l2_summary, output_dir=output_dir),
        plot_runtime_comparison(
            runtime_table,
            output_dir=output_dir,
            runtime_col=runtime_col,
        ),
    ]
    if domain_adaptation_dir is not None:
        topk_summary_path = domain_adaptation_dir / "topk_summary.csv"
        if not topk_summary_path.is_file():
            raise FileNotFoundError(f"Missing domain-adaptation summary: {topk_summary_path}")
        topk_summary = pd.read_csv(topk_summary_path)
        output_paths.append(
            plot_topk_domain_adaptation(topk_summary, output_dir=output_dir)
        )
        sources["domain_topk_summary"] = topk_summary_path
        holdout_topk_summary_path = domain_adaptation_dir / "holdout_topk_summary.csv"
        if holdout_topk_summary_path.is_file():
            holdout_topk_summary = pd.read_csv(holdout_topk_summary_path)
            output_paths.append(
                plot_topk_summary(
                    holdout_topk_summary,
                    output_dir=output_dir,
                    filename="topk_domain_adaptation_holdout.png",
                    ylabel="Holdout PA evaluation log loss",
                )
            )
            sources["domain_holdout_topk_summary"] = holdout_topk_summary_path
            holdout_selection_path = domain_adaptation_dir / "holdout_topk_selection_long.csv"
            if holdout_selection_path.is_file():
                sources["domain_holdout_topk_selection_long"] = holdout_selection_path
            holdout_indices_path = domain_adaptation_dir / "holdout_eval_indices.csv"
            if holdout_indices_path.is_file():
                sources["domain_holdout_eval_indices"] = holdout_indices_path
        for name in ("single_state_scores", "topk_selection_long", "config"):
            source_path = domain_adaptation_dir / f"{name}.csv"
            if name == "config":
                source_path = domain_adaptation_dir / "config.json"
            if source_path.is_file():
                sources[f"domain_{name}"] = source_path

    manifest_path = write_manifest(
        run_dir=run_dir,
        output_dir=output_dir,
        output_paths=output_paths,
        sources=sources,
        runtime_band=runtime_band,
        runtime_col=runtime_col,
        domain_adaptation_dir=domain_adaptation_dir,
    )
    print(f"[{run_dir.name}] reference: {reference_label}")
    print(f"[{run_dir.name}] wrote {len(output_paths)} figures to {output_dir}")
    print(f"[{run_dir.name}] wrote manifest to {manifest_path}")
    return manifest_path, len(output_paths)


def main() -> None:
    args = parse_args()
    plot_config = None
    if args.config is not None:
        plot_config = load_plot_config(args.config)
    run_dirs = resolve_run_dirs(args, plot_config)
    run_names = [run_dir.name for run_dir in run_dirs]
    if len(run_names) != len(set(run_names)):
        raise ValueError("Selected benchmark runs must have unique directory names.")
    if args.output_dir is not None and len(run_dirs) != 1:
        raise ValueError("--output-dir can only be used when plotting exactly one run.")

    domain_dirs = resolve_domain_adaptation_dirs(args, plot_config)
    explicit_domain_selection = bool(args.domain_adaptation_dirs) or bool(
        plot_config is not None and plot_config.get("domain_adaptation_dir")
    )
    domain_by_benchmark = map_domain_adaptation_runs(
        domain_dirs,
        run_dirs=run_dirs,
        require_all=explicit_domain_selection,
    )
    figures_root = args.figures_root.resolve()
    total_figures = 0
    for run_dir in run_dirs:
        if args.output_dir is not None:
            output_dir = args.output_dir.resolve()
        else:
            output_dir = figures_root / run_dir.name
        _, figure_count = plot_run(
            run_dir=run_dir,
            domain_adaptation_dir=domain_by_benchmark.get(run_dir.resolve()),
            output_dir=output_dir,
            runtime_band=args.runtime_band,
        )
        total_figures += figure_count
    print(f"Plotted {len(run_dirs)} benchmark runs and wrote {total_figures} figures.")


if __name__ == "__main__":
    main()
