"""Plot the paper runtime breakdown from its frozen published CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from plot_shapley_estimation_trajectories import (
    METHOD_LABELS,
    METHOD_ORDER,
    PANEL_TITLES,
    PAPER_LABEL_FONTSIZE,
    PAPER_LEGEND_FONTSIZE,
    PAPER_TICK_FONTSIZE,
    PAPER_TITLE_FONTSIZE,
    configure_matplotlib,
    file_metadata,
    sha256,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "real_world_benchmark"
    / "results"
    / "published"
    / "runtime_breakdown_3panel_summary.csv"
)
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "published" / "runtime_breakdown_3panel"
PANEL_NAMES = ("acs_xgboost", "communities_crime", "nhanesi")
DISPLAY_COMPONENTS = ("utility_eval_sec", "estimator_overhead_sec")
DISPLAY_COMPONENT_LABELS = {
    "utility_eval_sec": "Utility evaluation",
    "estimator_overhead_sec": "Estimator overhead",
}
DISPLAY_COMPONENT_COLORS = {
    "utility_eval_sec": "#4C78A8",
    "estimator_overhead_sec": "#B8B8B8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def create_figure(
    summary: pd.DataFrame,
    *,
    output_stem: Path,
    formats: tuple[str, ...] | list[str],
    dpi: int,
) -> list[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.ticker import EngFormatter, MaxNLocator

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.1), sharey=True)
    panel_labels = ("(a)", "(b)", "(c)")
    for ax, panel_name, panel_label in zip(axes, PANEL_NAMES, panel_labels):
        panel = (
            summary[summary["panel"].eq(panel_name)]
            .set_index("method")
            .reindex(METHOD_ORDER)
        )
        if panel[list(DISPLAY_COMPONENTS)].isna().any().any():
            missing = panel.index[panel["total_mean"].isna()].tolist()
            raise ValueError(
                f"Missing methods while plotting {PANEL_TITLES[panel_name]}: {missing}"
            )

        y = np.arange(len(METHOD_ORDER), dtype=float)
        left = np.zeros(len(METHOD_ORDER), dtype=float)
        for component in DISPLAY_COMPONENTS:
            values = panel[component].to_numpy(dtype=float)
            ax.barh(
                y,
                values,
                left=left,
                height=0.68,
                color=DISPLAY_COMPONENT_COLORS[component],
                edgecolor="white",
                linewidth=0.25,
                zorder=2,
            )
            left += values

        ax.set_title(
            PANEL_TITLES[panel_name],
            fontsize=PAPER_TITLE_FONTSIZE,
            pad=8,
        )
        ax.text(
            -0.13,
            1.07,
            panel_label,
            transform=ax.transAxes,
            fontsize=PAPER_TITLE_FONTSIZE,
            fontweight="bold",
            va="top",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [METHOD_LABELS[method] for method in METHOD_ORDER],
            fontsize=PAPER_TICK_FONTSIZE,
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.xaxis.set_major_formatter(
            EngFormatter(unit="s", places=0, sep="\N{NARROW NO-BREAK SPACE}")
        )
        ax.tick_params(axis="x", labelsize=PAPER_TICK_FONTSIZE)
        ax.grid(
            True,
            axis="x",
            color="#dddddd",
            linewidth=0.65,
            alpha=0.85,
            zorder=0,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()

    handles = [
        Patch(
            facecolor=DISPLAY_COMPONENT_COLORS[component],
            edgecolor="none",
            label=DISPLAY_COMPONENT_LABELS[component],
        )
        for component in DISPLAY_COMPONENTS
    ]
    fig.supxlabel(
        "Estimator runtime (intermediate checkpoint readouts excluded)",
        fontsize=PAPER_LABEL_FONTSIZE,
        y=0.155,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
        fontsize=PAPER_LEGEND_FONTSIZE,
        columnspacing=1.8,
        handlelength=1.8,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.91,
        bottom=0.24,
        wspace=0.25,
    )

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
    plt.close(fig)
    return output_paths


def write_manifest(
    *,
    output_stem: Path,
    output_paths: list[Path],
    input_path: Path,
) -> Path:
    output_stem = output_stem.expanduser().resolve()
    manifest = {
        "panels": [
            {
                "panel": label,
                "name": panel_name,
                "title": PANEL_TITLES[panel_name],
            }
            for label, panel_name in zip(("a", "b", "c"), PANEL_NAMES)
        ],
        "runtime_definition": (
            "elapsed estimator time minus intermediate checkpoint readouts; "
            "the final readout is retained"
        ),
        "components": {
            "utility_eval_sec": "time in estimator.run(samples)",
            "estimator_overhead_sec": (
                "sampling_sec + final_readout_sec + other_estimator_sec"
            ),
        },
        "aggregation": "Frozen component-wise means; no error bars.",
        "source": {**file_metadata(input_path), "sha256": sha256(input_path)},
        "outputs": [
            {**file_metadata(path), "sha256": sha256(path)} for path in output_paths
        ],
    }
    manifest_path = output_stem.with_name(f"{output_stem.name}_manifest.json")
    import json

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    summary = pd.read_csv(input_path)
    required = {"panel", "method", *DISPLAY_COMPONENTS, "stack_total"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"{input_path} is missing columns: {missing}")
    max_stack_delta = (
        summary["utility_eval_sec"]
        + summary["estimator_overhead_sec"]
        - summary["stack_total"]
    ).abs().max()
    if max_stack_delta > 1e-9:
        raise AssertionError(
            "Utility evaluation and estimator overhead do not sum to total runtime."
        )
    output_paths = create_figure(
        summary,
        output_stem=args.output_stem,
        formats=args.formats,
        dpi=args.dpi,
    )
    manifest_path = write_manifest(
        output_stem=args.output_stem,
        output_paths=output_paths,
        input_path=input_path,
    )
    for path in output_paths:
        print(f"Wrote {path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
