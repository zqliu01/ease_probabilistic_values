"""Create compact trajectory and runtime figures from the latest benchmark runs.

The script writes three paper-facing figures:

1. a 1-by-2 ACSIncome trajectory panel (logistic regression and XGBoost),
2. a 1-by-2 trajectory panel for CIFAR-10 and Breast Cancer, and
3. a standalone Breast Cancer runtime breakdown.

The trajectory figures use utility evaluations divided by the number of
players (m/n) on the x-axis. Their lines and uncertainty bands show the median
and 10th--90th percentiles, respectively. The runtime figure uses the same
component definitions as ``plot_runtime_breakdown.py``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from plot_runtime_breakdown import (
    COMPONENT_COLORS,
    COMPONENT_LABELS,
    RUNTIME_COMPONENTS,
    load_real_world_runtime,
    plot_runtime_panel,
    summarize_runtime,
)
from plot_shapley_estimation_trajectories import (
    ACS_RUNS_ROOT,
    METHOD_LABELS,
    METHOD_ORDER,
    METHOD_STYLES,
    PANEL_TITLES,
    REAL_WORLD_N_PLAYERS,
    REAL_WORLD_RESULTS_ROOT,
    configure_matplotlib,
    file_metadata,
    load_json,
    plot_panel,
    relative_to_workspace,
    resolve_sources,
    sha256,
    summarize_acs_trajectory,
    summarize_real_world_trajectories,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ACS_OUTPUT_NAME = "shapley_trajectories_acs_2panel"
REAL_WORLD_OUTPUT_NAME = "shapley_trajectories_cifar10_breast_cancer_2panel"
RUNTIME_OUTPUT_NAME = "runtime_breakdown_breast_cancer"
MANIFEST_NAME = "compact_figures_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acs-runs-root", type=Path, default=ACS_RUNS_ROOT)
    parser.add_argument("--acs-logistic-run", type=Path, default=None)
    parser.add_argument("--acs-xgboost-run", type=Path, default=None)
    parser.add_argument(
        "--real-world-results-root",
        type=Path,
        default=REAL_WORLD_RESULTS_ROOT,
    )
    parser.add_argument("--real-world-input", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory in which all compact figures are written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    parser.add_argument(
        "--band",
        choices=("std", "sem"),
        default="std",
        help="Variability displayed for total runtime.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def trajectory_legend_handles() -> list[Any]:
    from matplotlib.lines import Line2D

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
    return [
        handles[index]
        for column in range(row_width)
        for index in (column, column + row_width)
        if index < len(handles)
    ]


def save_figure(
    fig: Any,
    *,
    output_stem: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    output_stem = output_stem.expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for filetype in dict.fromkeys(formats):
        path = output_stem.with_suffix(f".{filetype}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if filetype == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        output_paths.append(path)
    return output_paths


def create_trajectory_pair(
    panels: tuple[tuple[Any, str, int], tuple[Any, str, int]],
    *,
    ylabel: str,
    output_stem: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharex=True)
    for index, (ax, (summary, panel_name, n_players)) in enumerate(zip(axes, panels)):
        plot_panel(
            ax,
            summary,
            title=PANEL_TITLES[panel_name],
            panel_label=f"({chr(ord('a') + index)})",
            ylabel=ylabel if index == 0 else "",
            n_players=n_players,
        )

    fig.supxlabel(
        "Utility evaluations / number of players ($m/n$)",
        fontsize=11.0,
        y=0.185,
    )
    fig.legend(
        handles=trajectory_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        frameon=False,
        fontsize=8.8,
        handlelength=2.0,
        columnspacing=1.0,
        labelspacing=0.60,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.91,
        bottom=0.29,
        wspace=0.23,
    )
    output_paths = save_figure(
        fig,
        output_stem=output_stem,
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return output_paths


def create_breast_cancer_runtime(
    summary: Any,
    *,
    output_stem: Path,
    formats: Iterable[str],
    dpi: int,
    band: str,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    plot_runtime_panel(
        ax,
        summary[summary["panel"].eq("breast_cancer")],
        title=PANEL_TITLES["breast_cancer"],
        panel_label="",
    )
    ax.invert_yaxis()

    handles = [
        Patch(
            facecolor=COMPONENT_COLORS[component],
            edgecolor="none",
            label=COMPONENT_LABELS[component],
        )
        for component in RUNTIME_COMPONENTS
    ]
    fig.supxlabel(
        "Estimator runtime (intermediate checkpoint readouts excluded)",
        fontsize=10.8,
        y=0.145,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        columnspacing=1.8,
        handlelength=1.8,
        labelspacing=0.65,
    )
    fig.text(
        0.5,
        0.008,
        f"Error bars show ±1 {band} of total runtime.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4d4d4d",
    )
    fig.subplots_adjust(
        left=0.20,
        right=0.975,
        top=0.92,
        bottom=0.27,
    )
    output_paths = save_figure(
        fig,
        output_stem=output_stem,
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return output_paths


def write_manifest(
    *,
    output_dir: Path,
    outputs: dict[str, list[Path]],
    runtime_summary_path: Path,
    logistic_run: Path,
    xgboost_run: Path,
    real_world_input: Path,
    real_world_raw_dir: Path,
    band: str,
) -> Path:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Newest structurally complete artifact by trajectory/checkpoint CSV mtime, "
            "unless overridden on the command line."
        ),
        "figures": {
            "acs_trajectory_pair": {
                "panels": [
                    {
                        "title": PANEL_TITLES["acs_logistic"],
                        "metric": "relative_squared_l2_error",
                        "utility_cache_mode": load_json(
                            logistic_run / "config.json"
                        ).get("utility_cache_mode", "not_recorded"),
                        "config": file_metadata(logistic_run / "config.json"),
                        "source": file_metadata(logistic_run / "trajectory_long.csv"),
                    },
                    {
                        "title": PANEL_TITLES["acs_xgboost"],
                        "metric": "relative_squared_l2_error",
                        "utility_cache_mode": load_json(
                            xgboost_run / "config.json"
                        ).get("utility_cache_mode", "not_recorded"),
                        "config": file_metadata(xgboost_run / "config.json"),
                        "source": file_metadata(xgboost_run / "trajectory_long.csv"),
                    },
                ],
                "aggregation": {
                    "line": "median",
                    "band": "10th--90th percentile",
                    "x_axis": "utility_evaluations / number_of_players (m/n)",
                },
            },
            "cifar10_breast_cancer_trajectory_pair": {
                "panels": [
                    {
                        "title": PANEL_TITLES["cifar10"],
                        "metric": "relative_l2_error",
                        "n_players": REAL_WORLD_N_PLAYERS["cifar10"],
                    },
                    {
                        "title": PANEL_TITLES["breast_cancer"],
                        "metric": "relative_l2_error",
                        "n_players": REAL_WORLD_N_PLAYERS["breast_cancer"],
                    },
                ],
                "source": file_metadata(real_world_input),
                "aggregation": {
                    "line": "median",
                    "band": "10th--90th percentile",
                    "x_axis": "utility_evaluations / number_of_players (m/n)",
                },
            },
            "breast_cancer_runtime": {
                "runtime_definition": (
                    "elapsed estimator time minus intermediate checkpoint readouts; "
                    "the final readout is retained"
                ),
                "components": {
                    "utility_eval_sec": "time in estimator.run(samples)",
                    "sampling_sec": "time advancing the estimator sampling iterator",
                    "final_readout_sec": (
                        "instrumented final estimator readout when available"
                    ),
                    "other_estimator_sec": (
                        "residual estimator time, including setup, aggregation, "
                        "finalization, orchestration, and uninstrumented readout work"
                    ),
                },
                "aggregation": {
                    "bar": "component-wise mean",
                    "error_bar": f"±1 {band} of total runtime",
                    "replicates": "explained instances and benchmark runs",
                },
                "source_anchor": file_metadata(real_world_input),
                "raw_dir": relative_to_workspace(real_world_raw_dir),
                "summary": file_metadata(runtime_summary_path),
            },
        },
        "method_order": list(METHOD_ORDER),
        "outputs": {
            figure: [
                {**file_metadata(path), "sha256": sha256(path)} for path in paths
            ]
            for figure, paths in outputs.items()
        },
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logistic_run, xgboost_run, real_world_input = resolve_sources(args)
    print(f"ACS logistic source: {logistic_run}")
    print(f"ACS XGBoost source: {xgboost_run}")
    print(f"Real-world source: {real_world_input}")

    logistic_summary = summarize_acs_trajectory(logistic_run)
    xgboost_summary = summarize_acs_trajectory(xgboost_run)
    real_world_summary = summarize_real_world_trajectories(real_world_input)

    outputs: dict[str, list[Path]] = {}
    outputs["acs_trajectory_pair"] = create_trajectory_pair(
        (
            (
                logistic_summary,
                "acs_logistic",
                int(logistic_summary.attrs["n_players"]),
            ),
            (
                xgboost_summary,
                "acs_xgboost",
                int(xgboost_summary.attrs["n_players"]),
            ),
        ),
        ylabel="Relative squared $L_2$ error",
        output_stem=output_dir / ACS_OUTPUT_NAME,
        formats=args.formats,
        dpi=args.dpi,
    )
    outputs["cifar10_breast_cancer_trajectory_pair"] = create_trajectory_pair(
        (
            (
                real_world_summary[real_world_summary["dataset"].eq("cifar10")],
                "cifar10",
                REAL_WORLD_N_PLAYERS["cifar10"],
            ),
            (
                real_world_summary[
                    real_world_summary["dataset"].eq("breast_cancer")
                ],
                "breast_cancer",
                REAL_WORLD_N_PLAYERS["breast_cancer"],
            ),
        ),
        ylabel="Relative $L_2$ error",
        output_stem=output_dir / REAL_WORLD_OUTPUT_NAME,
        formats=args.formats,
        dpi=args.dpi,
    )

    real_world_runtime, raw_dir = load_real_world_runtime(real_world_input)
    runtime_summary = summarize_runtime(real_world_runtime, band=args.band)
    breast_cancer_summary = runtime_summary[
        runtime_summary["panel"].eq("breast_cancer")
    ].copy()
    runtime_summary_path = output_dir / f"{RUNTIME_OUTPUT_NAME}_summary.csv"
    breast_cancer_summary.sort_values("method").to_csv(runtime_summary_path, index=False)
    outputs["breast_cancer_runtime"] = create_breast_cancer_runtime(
        runtime_summary,
        output_stem=output_dir / RUNTIME_OUTPUT_NAME,
        formats=args.formats,
        dpi=args.dpi,
        band=args.band,
    )

    manifest_path = write_manifest(
        output_dir=output_dir,
        outputs=outputs,
        runtime_summary_path=runtime_summary_path,
        logistic_run=logistic_run,
        xgboost_run=xgboost_run,
        real_world_input=real_world_input,
        real_world_raw_dir=raw_dir,
        band=args.band,
    )
    for paths in outputs.values():
        for path in paths:
            print(f"Wrote {path}")
    print(f"Wrote {runtime_summary_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
