"""
Customized per-eta comparison plots for the SOU experiments.

This script reads the existing summary.csv files and writes one figure per eta.
Each figure has three panels: EASE vs RegressionMSR, EASE vs PolySHAP, and
EASE vs OFA.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "customized_plots"

ALPHAS = (0.25**0.5, 0.5**0.5, 0.75**0.5)
ETA_LABELS = ("0.25", "0.5", "0.75")

X_AXIS_LABEL = "Avg. Utility Evals per Player"
Y_AXIS_LABEL = "relative squared error"
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11
PANEL_WIDTH = 3.5
FIG_HEIGHT = 3.35

LABELS = {
    "EaseSHAP": "EASE",
    "EaseSHAP_boundary_size_player": "EASE",
    "EaseSHAP_order2": "EASE",
    "RegressionMSR_unbiased": "RegressionMSR",
    "OFA_fixed": "OFA",
    "PolySHAP_regression": "PolySHAP",
}

COLORS = {
    "EaseSHAP": "tab:blue",
    "EaseSHAP_boundary_size_player": "tab:blue",
    "EaseSHAP_order2": "tab:blue",
    "RegressionMSR_unbiased": "tab:orange",
    "OFA_fixed": "tab:orange",
    "PolySHAP_regression": "tab:orange",
}


@dataclass(frozen=True)
class ComparisonSpec:
    title: str
    summary_csv: Path
    algorithms: tuple[str, str]


COMPARISONS = (
    ComparisonSpec(
        title="EASE vs RegressionMSR",
        summary_csv=RESULTS_DIR / "regmsr_unpaired_vs_easeshap" / "summary.csv",
        algorithms=("EaseSHAP", "RegressionMSR_unbiased"),
    ),
    ComparisonSpec(
        title="EASE vs OFA",
        summary_csv=RESULTS_DIR / "ofa_vs_easeshap" / "summary.csv",
        algorithms=("EaseSHAP_boundary_size_player", "OFA_fixed"),
    ),
    ComparisonSpec(
        title="EASE vs PolySHAP",
        summary_csv=RESULTS_DIR / "polyshap_vs_easeshap" / "summary.csv",
        algorithms=("EaseSHAP_order2", "PolySHAP_regression"),
    ),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def aggregate(
    rows: list[dict[str, str]], alpha: float, algorithm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_nue: dict[int, list[float]] = {}
    alpha_key = f"{alpha:g}"
    for row in rows:
        if row["alpha"] != alpha_key or row["algorithm"] != algorithm:
            continue
        by_nue.setdefault(int(row["nue"]), []).append(float(row["rel_sq_error"]))

    if not by_nue:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    xs = np.array(sorted(by_nue), dtype=int)
    means = np.array([np.mean(by_nue[x]) for x in xs])
    stds = np.array([np.std(by_nue[x]) for x in xs])
    return xs, means, stds


def plot_comparison_panel(ax: plt.Axes, rows: list[dict[str, str]], spec: ComparisonSpec, alpha: float) -> None:
    plotted = False
    for algorithm in spec.algorithms:
        xs, means, stds = aggregate(rows, alpha, algorithm)
        if xs.size == 0:
            continue

        color = COLORS.get(algorithm)
        lower = np.maximum(means - stds, np.finfo(float).tiny)
        ax.plot(xs, means, marker="o", label=LABELS.get(algorithm, algorithm), color=color)
        ax.fill_between(xs, lower, means + stds, alpha=0.18, color=color)
        plotted = True

    if not plotted:
        raise RuntimeError(f"No rows found for alpha={alpha:g} in {spec.summary_csv}")

    ax.set_title(spec.title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.legend(loc="best", frameon=False, fontsize=LEGEND_FONTSIZE)


def eta_filename(eta_label: str) -> str:
    return f"eta_{eta_label.replace('.', 'p')}.png"


def plot_eta(alpha: float, eta_label: str, rows_by_comparison: dict[ComparisonSpec, list[dict[str, str]]]) -> Path:
    fig, axes = plt.subplots(
        1,
        len(COMPARISONS),
        figsize=(PANEL_WIDTH * len(COMPARISONS), FIG_HEIGHT),
        sharey=True,
    )

    for ax, spec in zip(axes, COMPARISONS):
        plot_comparison_panel(ax, rows_by_comparison[spec], spec, alpha)

    axes[0].set_ylabel(Y_AXIS_LABEL, fontsize=LABEL_FONTSIZE)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / eta_filename(eta_label)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    rows_by_comparison = {spec: read_rows(spec.summary_csv) for spec in COMPARISONS}
    for alpha, eta_label in zip(ALPHAS, ETA_LABELS):
        output_path = plot_eta(alpha, eta_label, rows_by_comparison)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
