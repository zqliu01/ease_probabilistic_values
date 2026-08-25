"""Create the five-panel SOU comparison figure used in the paper.

The top row compares EASE with RegressionMSR, OFA, and PolySHAP at
eta = 0.25.  The centered bottom row compares EASE with RegressionMSR at
eta = 0.5 and eta = 0.75.

Plotting and aggregation are delegated to ``customized_plots.py`` so each
panel uses the original mean/std aggregation, colors, labels, fonts, and axis
settings.  The data paths below point only to the frozen ``published`` CSVs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SOU_DIR = SCRIPT_DIR.parent / "SOU_comparison_various_eta"
if str(SOU_DIR) not in sys.path:
    sys.path.insert(0, str(SOU_DIR))

import customized_plots as sou_plots


PUBLISHED_RESULTS_DIR = SOU_DIR / "results" / "published"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "published" / "sou_comparison_various_eta_5panel"

PUBLISHED_COMPARISONS = (
    replace(
        sou_plots.COMPARISONS[0],
        summary_csv=PUBLISHED_RESULTS_DIR / "regressionmsr" / "summary.csv",
    ),
    replace(
        sou_plots.COMPARISONS[1],
        summary_csv=PUBLISHED_RESULTS_DIR / "ofa" / "summary.csv",
    ),
    replace(
        sou_plots.COMPARISONS[2],
        summary_csv=PUBLISHED_RESULTS_DIR / "polyshap" / "summary.csv",
    ),
)


@dataclass(frozen=True)
class Panel:
    comparison: sou_plots.ComparisonSpec
    alpha: float
    eta_label: str

    @property
    def titled_comparison(self) -> sou_plots.ComparisonSpec:
        title = f"{self.comparison.title},\n" + rf"$\eta = {self.eta_label}$"
        return replace(self.comparison, title=title)


PANELS = (
    Panel(PUBLISHED_COMPARISONS[0], sou_plots.ALPHAS[0], sou_plots.ETA_LABELS[0]),
    Panel(PUBLISHED_COMPARISONS[1], sou_plots.ALPHAS[0], sou_plots.ETA_LABELS[0]),
    Panel(PUBLISHED_COMPARISONS[2], sou_plots.ALPHAS[0], sou_plots.ETA_LABELS[0]),
    Panel(PUBLISHED_COMPARISONS[0], sou_plots.ALPHAS[1], sou_plots.ETA_LABELS[1]),
    Panel(PUBLISHED_COMPARISONS[0], sou_plots.ALPHAS[2], sou_plots.ETA_LABELS[2]),
)
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output path without a file extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def build_figure():
    plt = sou_plots.plt
    fig = plt.figure(
        figsize=(sou_plots.PANEL_WIDTH * 3, sou_plots.FIG_HEIGHT * 2)
    )
    grid = fig.add_gridspec(2, 6)
    slots = (
        grid[0, 0:2],
        grid[0, 2:4],
        grid[0, 4:6],
        grid[1, 1:3],
        grid[1, 3:5],
    )

    rows_by_comparison = {
        comparison: sou_plots.read_rows(comparison.summary_csv)
        for comparison in {panel.comparison for panel in PANELS}
    }

    axes = []
    for index, (panel, slot) in enumerate(zip(PANELS, slots)):
        ax = fig.add_subplot(slot, sharey=axes[0] if axes else None)
        sou_plots.plot_comparison_panel(
            ax,
            rows_by_comparison[panel.comparison],
            panel.titled_comparison,
            panel.alpha,
        )
        ax.text(
            -0.11,
            1.18,
            PANEL_LABELS[index],
            transform=ax.transAxes,
            fontsize=sou_plots.TITLE_FONTSIZE,
            fontweight="bold",
            ha="left",
            va="top",
        )
        show_y_axis = index in (0, 3)
        ax.tick_params(axis="y", labelleft=show_y_axis)
        if show_y_axis:
            ax.set_ylabel(
                sou_plots.Y_AXIS_LABEL,
                fontsize=sou_plots.LABEL_FONTSIZE,
            )
        axes.append(ax)

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    for extension in args.formats:
        output_path = args.output_stem.with_suffix(f".{extension}")
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {output_path}")
    sou_plots.plt.close(fig)


if __name__ == "__main__":
    main()
