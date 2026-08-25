"""Plot six Shapley-estimation trajectories with one shared legend.

The script reads only the frozen ``results/published*`` snapshots and creates a
2-by-3 paper figure containing:

1. ACSIncome with logistic regression (relative squared L2 error),
2. ACSIncome with XGBoost (relative squared L2 error),
3. CIFAR-10 (relative L2 error),
4. Breast Cancer (relative L2 error),
5. Communities and Crime (relative L2 error), and
6. NHANES I (relative L2 error).

The central lines are arithmetic means over estimator seeds or explained
instances. The paper-facing six-panel figure intentionally omits uncertainty
bands to reduce visual clutter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = EXPERIMENTS_DIR.parent.parent
ACS_RESULTS_ROOT = (
    EXPERIMENTS_DIR
    / "acs_income_state_valuation"
    / "results"
)
REAL_WORLD_RESULTS_ROOT = EXPERIMENTS_DIR / "real_world_benchmark" / "results"
DEFAULT_ACS_LOGISTIC_RUN = ACS_RESULTS_ROOT / "published_logistic"
DEFAULT_ACS_XGBOOST_RUN = ACS_RESULTS_ROOT / "published_xgboost"
DEFAULT_REAL_WORLD_INPUT = REAL_WORLD_RESULTS_ROOT / "published" / "l2_summary.csv"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "published" / "shapley_estimation_trajectories_6panel"

ACS_REQUIRED_FILES = (
    Path("checkpoint_metrics.csv"),
    Path("reference/reference_summary.json"),
)
REAL_WORLD_DATASETS = (
    "cifar10",
    "breast_cancer",
    "communities_crime",
    "nhanesi",
)
REAL_WORLD_N_PLAYERS = {
    "cifar10": 16,
    "breast_cancer": 30,
    "communities_crime": 101,
    "nhanesi": 79,
}
PANEL_TITLES = {
    "acs_logistic": "ACSIncome (Logistic Regression)",
    "acs_xgboost": "ACSIncome (XGBoost)",
    "cifar10": "CIFAR-10",
    "breast_cancer": "Breast Cancer",
    "communities_crime": "Communities and Crime",
    "nhanesi": "NHANES I",
}

# These are the methods retained by both existing paper-facing plotting scripts.
METHOD_ORDER = (
    "EaseSHAP_interaction_nonlinear",
    "OFA_fixed",
    "sampling_lift",
    "SHAP_IQ",
    "GELS",
    "improved_AME",
    "kernelSHAP",
    "LeverageSHAP",
    "permutation",
    "complement",
    "group_testing",
    "RegressionMSR_unbiased",
)
METHOD_LABELS = {
    "EaseSHAP_interaction_nonlinear": "EASE-FO",
    "OFA_fixed": "OFA",
    "sampling_lift": "Sampling lift",
    "SHAP_IQ": "SHAP-IQ",
    "GELS": "GELS",
    "improved_AME": "Improved AME",
    "kernelSHAP": "kernelSHAP",
    "LeverageSHAP": "LeverageSHAP",
    "permutation": "Permutation",
    "complement": "Complement",
    "group_testing": "Group testing",
    "RegressionMSR_unbiased": "RegressionMSR",
}

# One global style is necessary for a meaningful shared legend. The palette is
# the established ACS paper palette, with EASE-FO emphasized by line width.
METHOD_STYLES = {
    "EaseSHAP_interaction_nonlinear": {"color": "#1F77B4", "marker": "o"},
    "GELS": {"color": "#2CA02C", "marker": "s"},
    "LeverageSHAP": {"color": "#D62728", "marker": "^"},
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

# Match the typography in SOU_comparison_various_eta/customized_plots.py.
PAPER_TITLE_FONTSIZE = 14.0
PAPER_LABEL_FONTSIZE = 12.0
PAPER_LEGEND_FONTSIZE = 11.0
PAPER_TICK_FONTSIZE = 10.0

_PILOT_SUFFIX_RE = re.compile(r"^(?P<base>.+)_pilot[0-9]+p[0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acs-logistic-run",
        type=Path,
        default=DEFAULT_ACS_LOGISTIC_RUN,
        help="Published logistic-results directory.",
    )
    parser.add_argument(
        "--acs-xgboost-run",
        type=Path,
        default=DEFAULT_ACS_XGBOOST_RUN,
        help="Published XGBoost-results directory.",
    )
    parser.add_argument(
        "--real-world-input",
        type=Path,
        default=DEFAULT_REAL_WORLD_INPUT,
        help="Published real-world l2_summary.csv.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output path without an extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="One or more output formats.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def canonical_method(method: str) -> str:
    match = _PILOT_SUFFIX_RE.match(str(method))
    return match.group("base") if match else str(method)


def keep_paper_method(method: str) -> bool:
    method = str(method)
    base = canonical_method(method)
    if base not in METHOD_ORDER:
        return False
    if base == "EaseSHAP_interaction_nonlinear" and method != base:
        return method.endswith("_pilot0p20")
    return True


def validate_acs_run(run_dir: Path, utility_model: str) -> Path:
    run_dir = run_dir.expanduser().resolve()
    missing = [str(path) for path in ACS_REQUIRED_FILES if not (run_dir / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete ACS run {run_dir}; missing: {missing}")
    if utility_model not in run_dir.name:
        raise ValueError(
            f"Expected an ACS {utility_model} directory, got {run_dir.name!r}."
        )
    return run_dir


def validate_real_world_input(path: Path) -> Path:
    import pandas as pd

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Real-world checkpoint table does not exist: {path}")
    required = {
        "dataset",
        "semivalue",
        "method",
        "checkpoint_idx",
        "actual_budget",
        "mean_rel_l2",
    }
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return path


def resolve_sources(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    logistic_run = validate_acs_run(args.acs_logistic_run, "logistic")
    xgboost_run = validate_acs_run(args.acs_xgboost_run, "xgboost")
    real_world_input = validate_real_world_input(args.real_world_input)
    return logistic_run, xgboost_run, real_world_input


def summarize_acs_trajectory(
    run_dir: Path,
    *,
    central_stat: str = "median",
):
    import numpy as np
    import pandas as pd

    if central_stat not in {"mean", "median"}:
        raise ValueError(f"Unsupported central_stat={central_stat!r}")

    metrics_path = run_dir / "checkpoint_metrics.csv"
    rows = pd.read_csv(
        metrics_path,
        usecols=(
            "method",
            "seed_index",
            "estimator_seed",
            "checkpoint",
            "utility_evaluations",
            "rmse_to_reference",
        ),
    )
    rows = rows[rows["method"].astype(str).map(keep_paper_method)].copy()
    if rows.empty:
        raise ValueError(f"No paper methods remain in {metrics_path}")

    reference_summary = load_json(run_dir / "reference" / "reference_summary.json")
    reference = np.asarray(reference_summary.get("values", []), dtype=np.float64)
    denominator = float(np.dot(reference, reference))
    if denominator <= 0.0:
        raise ValueError(f"ACS reference has zero L2 norm: {run_dir}")

    # checkpoint_metrics.csv stores RMSE for each seed/checkpoint.  Therefore
    # n * RMSE^2 / ||reference||_2^2 is exactly the relative squared L2 error
    # used by the paper figure, without publishing the much larger state-level
    # trajectory_long.csv files.
    rows["error"] = (
        len(reference) * rows["rmse_to_reference"].to_numpy(dtype=np.float64) ** 2
        / denominator
    )
    rows["method"] = rows["method"].astype(str).map(canonical_method)
    summary = (
        rows.groupby(["method", "checkpoint", "utility_evaluations"], sort=False)["error"]
        .agg(
            center=central_stat,
            q10=lambda values: values.quantile(0.10),
            q90=lambda values: values.quantile(0.90),
            count="count",
        )
        .reset_index()
        .rename(columns={"utility_evaluations": "budget"})
    )
    summary.attrs["n_players"] = len(reference)
    return summary


def summarize_real_world_trajectories(
    path: Path,
    *,
    central_stat: str = "median",
):
    import pandas as pd

    if central_stat != "mean":
        raise ValueError("Published l2_summary.csv stores arithmetic means only")

    rows = pd.read_csv(
        path,
        usecols=(
            "dataset",
            "semivalue",
            "method",
            "checkpoint_idx",
            "actual_budget",
            "mean_rel_l2",
            "count",
        ),
    )
    rows = rows[
        rows["dataset"].isin(REAL_WORLD_DATASETS)
        & rows["semivalue"].eq("shapley")
        & rows["method"].astype(str).map(keep_paper_method)
    ].copy()
    if rows.empty:
        raise ValueError(f"No requested Shapley trajectory rows remain in {path}")
    missing_datasets = sorted(set(REAL_WORLD_DATASETS).difference(rows["dataset"].unique()))
    if missing_datasets:
        raise ValueError(f"{path} is missing Shapley rows for: {missing_datasets}")

    rows["method"] = rows["method"].astype(str).map(canonical_method)
    summary = rows.rename(
        columns={"actual_budget": "budget", "mean_rel_l2": "center"}
    )
    # The paper-facing figure has no uncertainty band.  Keeping these columns
    # equal to the mean preserves the common panel interface.
    summary["q10"] = summary["center"]
    summary["q90"] = summary["center"]
    return summary


def configure_matplotlib() -> None:
    mplconfig_dir = SCRIPT_DIR / ".mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mplconfig_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")


def plot_panel(
    ax: Any,
    summary: Any,
    *,
    title: str,
    panel_label: str,
    ylabel: str,
    n_players: int,
    title_fontsize: float = 11.5,
    panel_label_fontsize: float = 11.5,
    label_fontsize: float = 9.5,
    tick_fontsize: float = 8.5,
    minor_tick_fontsize: float = 7.5,
    show_band: bool = True,
) -> None:
    import numpy as np
    from matplotlib.ticker import StrMethodFormatter

    positive = summary.loc[
        summary[["center", "q10", "q90"]].gt(0.0).any(axis=1),
        ["center", "q10", "q90"],
    ].to_numpy(dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    floor = float(positive.min() * 0.75) if len(positive) else np.finfo(float).tiny

    for method in METHOD_ORDER:
        group = summary[summary["method"].eq(method)].sort_values("budget")
        if group.empty:
            continue
        style = METHOD_STYLES[method]
        x = group["budget"].to_numpy(dtype=np.float64) / float(n_players)
        y = np.maximum(group["center"].to_numpy(dtype=np.float64), floor)
        lower = np.maximum(group["q10"].to_numpy(dtype=np.float64), floor)
        upper = np.maximum(group["q90"].to_numpy(dtype=np.float64), floor)
        is_ease = method == "EaseSHAP_interaction_nonlinear"
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0 if is_ease else 1.35,
            markersize=3.2 if is_ease else 2.7,
            markeredgewidth=0.4,
            alpha=1.0 if is_ease else 0.90,
            zorder=4 if is_ease else 3,
        )
        if show_band:
            ax.fill_between(
                x,
                lower,
                upper,
                color=style["color"],
                alpha=0.14 if is_ease else 0.085,
                linewidth=0,
                zorder=2 if is_ease else 1,
            )

    ax.set_title(title, fontsize=title_fontsize, pad=8)
    ax.text(
        -0.13,
        1.07,
        panel_label,
        transform=ax.transAxes,
        fontsize=panel_label_fontsize,
        fontweight="bold",
        va="top",
    )
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 205.0)
    ax.set_xticks((0.0, 50.0, 100.0, 150.0, 200.0))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.tick_params(axis="both", which="minor", labelsize=minor_tick_fontsize)
    ax.grid(True, which="major", color="#d7d7d7", linewidth=0.65, alpha=0.8)
    ax.grid(True, which="minor", axis="y", color="#ededed", linewidth=0.45, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_figure(
    logistic_summary: Any,
    xgboost_summary: Any,
    real_world_summary: Any,
    *,
    output_stem: Path,
    formats: tuple[str, ...] | list[str],
    dpi: int,
) -> list[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.5), sharex=True)
    panels = (
        (
            logistic_summary,
            "acs_logistic",
            "Relative squared $L_2$ error",
            int(logistic_summary.attrs["n_players"]),
        ),
        (
            xgboost_summary,
            "acs_xgboost",
            "Relative squared $L_2$ error",
            int(xgboost_summary.attrs["n_players"]),
        ),
        (
            real_world_summary[real_world_summary["dataset"].eq("cifar10")],
            "cifar10",
            "Relative $L_2$ error",
            REAL_WORLD_N_PLAYERS["cifar10"],
        ),
        (
            real_world_summary[real_world_summary["dataset"].eq("breast_cancer")],
            "breast_cancer",
            "Relative $L_2$ error",
            REAL_WORLD_N_PLAYERS["breast_cancer"],
        ),
        (
            real_world_summary[real_world_summary["dataset"].eq("communities_crime")],
            "communities_crime",
            "Relative $L_2$ error",
            REAL_WORLD_N_PLAYERS["communities_crime"],
        ),
        (
            real_world_summary[real_world_summary["dataset"].eq("nhanesi")],
            "nhanesi",
            "Relative $L_2$ error",
            REAL_WORLD_N_PLAYERS["nhanesi"],
        ),
    )
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for ax, (summary, panel_name, ylabel, n_players), panel_label in zip(
        axes.flat,
        panels,
        panel_labels,
    ):
        plot_panel(
            ax,
            summary,
            title=PANEL_TITLES[panel_name],
            panel_label=panel_label,
            ylabel=ylabel,
            n_players=n_players,
            title_fontsize=PAPER_TITLE_FONTSIZE,
            panel_label_fontsize=PAPER_TITLE_FONTSIZE,
            label_fontsize=PAPER_LABEL_FONTSIZE,
            tick_fontsize=PAPER_TICK_FONTSIZE,
            minor_tick_fontsize=PAPER_TICK_FONTSIZE - 1.0,
            show_band=False,
        )

    handles = []
    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        is_ease = method == "EaseSHAP_interaction_nonlinear"
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0 if is_ease else 1.4,
                markersize=5.0,
                label=METHOD_LABELS[method],
            )
        )

    # Matplotlib fills multi-row legends down columns. Interleave the two rows
    # so the visible left-to-right order follows METHOD_ORDER.
    row_width = 6
    handles = [
        handles[index]
        for column in range(row_width)
        for index in (column, column + row_width)
        if index < len(handles)
    ]

    fig.supxlabel(
        "Utility evaluations / number of players ($m/n$)",
        fontsize=PAPER_LABEL_FONTSIZE,
        y=0.145,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
        fontsize=PAPER_LEGEND_FONTSIZE,
        handlelength=2.2,
        columnspacing=1.25,
        labelspacing=0.65,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.95,
        bottom=0.205,
        wspace=0.31,
        hspace=0.36,
    )

    output_stem = output_stem.expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for filetype in dict.fromkeys(formats):
        path = output_stem.with_suffix(f".{filetype}")
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if filetype == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def relative_to_workspace(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(WORKSPACE_DIR))
    except ValueError:
        return str(path)


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative_to_workspace(path),
        "size_bytes": stat.st_size,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    *,
    output_stem: Path,
    output_paths: list[Path],
    logistic_run: Path,
    xgboost_run: Path,
    real_world_input: Path,
) -> Path:
    manifest = {
        "selection_rule": "Explicit frozen results/published* snapshots.",
        "panels": [
            {
                "panel": "a",
                "title": PANEL_TITLES["acs_logistic"],
                "metric": "relative_squared_l2_error",
                "n_players": len(
                    load_json(logistic_run / "reference" / "reference_summary.json")[
                        "values"
                    ]
                ),
                "source": file_metadata(logistic_run / "checkpoint_metrics.csv"),
                "reference": file_metadata(
                    logistic_run / "reference" / "reference_summary.json"
                ),
            },
            {
                "panel": "b",
                "title": PANEL_TITLES["acs_xgboost"],
                "metric": "relative_squared_l2_error",
                "n_players": len(
                    load_json(xgboost_run / "reference" / "reference_summary.json")[
                        "values"
                    ]
                ),
                "source": file_metadata(xgboost_run / "checkpoint_metrics.csv"),
                "reference": file_metadata(
                    xgboost_run / "reference" / "reference_summary.json"
                ),
            },
            *[
                {
                    "panel": panel,
                    "title": PANEL_TITLES[dataset],
                    "dataset": dataset,
                    "semivalue": "shapley",
                    "metric": "relative_l2_error",
                    "n_players": REAL_WORLD_N_PLAYERS[dataset],
                    "source": file_metadata(real_world_input),
                }
                for panel, dataset in zip(("c", "d", "e", "f"), REAL_WORLD_DATASETS)
            ],
        ],
        "aggregation": {
            "line": "arithmetic mean in the original error scale",
            "band": "none",
            "x_axis": "utility_evaluations / number_of_players (m/n)",
            "log_axis_interpretation": "log(mean(error)), not mean(log(error))",
        },
        "method_order": list(METHOD_ORDER),
        "outputs": [
            {
                **file_metadata(path),
                "sha256": sha256(path),
            }
            for path in output_paths
        ],
    }
    manifest_path = output_stem.expanduser().resolve().with_name(
        f"{output_stem.name}_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    args = parse_args()
    logistic_run, xgboost_run, real_world_input = resolve_sources(args)
    print(f"ACS logistic source: {logistic_run}")
    print(f"ACS XGBoost source: {xgboost_run}")
    print(f"Real-world source: {real_world_input}")

    logistic_summary = summarize_acs_trajectory(logistic_run, central_stat="mean")
    xgboost_summary = summarize_acs_trajectory(xgboost_run, central_stat="mean")
    real_world_summary = summarize_real_world_trajectories(
        real_world_input,
        central_stat="mean",
    )
    output_paths = create_figure(
        logistic_summary,
        xgboost_summary,
        real_world_summary,
        output_stem=args.output_stem,
        formats=args.formats,
        dpi=args.dpi,
    )
    manifest_path = write_manifest(
        output_stem=args.output_stem,
        output_paths=output_paths,
        logistic_run=logistic_run,
        xgboost_run=xgboost_run,
        real_world_input=real_world_input,
    )
    for path in output_paths:
        print(f"Wrote {path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
