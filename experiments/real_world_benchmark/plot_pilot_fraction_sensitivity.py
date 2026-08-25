"""Plot EASE-FO's sensitivity to the pilot-switch fraction.

Reads the pre-aggregated ``l2_summary.csv`` from the
``m200n_pilots0p05_0p10_0p20_0p40_updates3_ckpt20`` sweep (same total budget,
pilot-design-update count, and checkpoint grid as the main-text real-world
benchmark) and plots relative-L2 convergence trajectories for
EaseSHAP_interaction_nonlinear (EASE-FO) at each of the four swept pilot
fractions, one panel per feature-attribution dataset, Shapley value only.

``vit4by4`` is intentionally excluded: it is a separate, unused image
benchmark that does not correspond to the "CIFAR-10" dataset reported in the
paper (that figure is keyed off ``cifar10``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR / "results" / "configs" / "m200n_pilots0p05_0p10_0p20_0p40_updates3_ckpt20"
SUMMARY_CSV = CONFIG_DIR / "l2_summary.csv"
OUT_DIR = SCRIPT_DIR / "figures" / "pilot_fraction_sensitivity"
MPLCONFIG_DIR = OUT_DIR / "mplconfig"

DATASETS = ["cifar10", "breast_cancer", "communities_crime", "nhanesi"]
DATASET_TITLES = {
    "cifar10": "CIFAR-10",
    "breast_cancer": "Breast Cancer",
    "communities_crime": "Communities and Crime",
    "nhanesi": "NHANES I",
}
# actual_budget in l2_summary.csv is the total number of utility evaluations m,
# not m/n; divide by n_players to match the main-text convention (m/n on the
# x-axis), as in figures/plot_shapley_estimation_trajectories.py.
N_PLAYERS = {
    "cifar10": 16,
    "breast_cancer": 30,
    "communities_crime": 101,
    "nhanesi": 79,
}
SEMIVALUE = "shapley"
PILOT_FRACTIONS = ["0p05", "0p10", "0p20", "0p40"]
METHODS = [f"EaseSHAP_interaction_nonlinear_pilot{p}" for p in PILOT_FRACTIONS]
PILOT_LABELS = {
    f"EaseSHAP_interaction_nonlinear_pilot{p}": f"pilot = {p.replace('p', '.')}"
    for p in PILOT_FRACTIONS
}
# Sequential colormap: lighter = smaller pilot fraction, darker = larger.
CMAP = ["#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
PILOT_COLORS = {method: CMAP[i] for i, method in enumerate(METHODS)}


def main() -> None:
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
    import matplotlib.pyplot as plt

    df = pd.read_csv(
        SUMMARY_CSV,
        usecols=[
            "dataset",
            "semivalue",
            "method",
            "checkpoint_idx",
            "actual_budget",
            "mean_rel_l2",
            "sem_rel_l2",
        ],
    )
    df = df[
        df["dataset"].isin(DATASETS)
        & (df["semivalue"] == SEMIVALUE)
        & df["method"].isin(METHODS)
    ]

    title_fontsize = 19
    label_fontsize = 17
    tick_fontsize = 15
    legend_fontsize = 16

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), sharey=False)
    axes = axes.flatten()
    for ax, dataset in zip(axes, DATASETS):
        panel = df[df["dataset"] == dataset]
        for method in METHODS:
            rows = panel[panel["method"] == method].sort_values("checkpoint_idx")
            if rows.empty:
                continue
            x = rows["actual_budget"].to_numpy(dtype=np.float64) / float(N_PLAYERS[dataset])
            y = rows["mean_rel_l2"].to_numpy(dtype=np.float64)
            sem = rows["sem_rel_l2"].to_numpy(dtype=np.float64)
            color = PILOT_COLORS[method]
            ax.plot(x, y, marker="o", markersize=5.5, linewidth=2.6, color=color, label=PILOT_LABELS[method])
            ax.fill_between(x, np.maximum(y - sem, 1e-12), y + sem, color=color, alpha=0.15, linewidth=0)
        ax.set_title(DATASET_TITLES[dataset], fontsize=title_fontsize, pad=8)
        ax.set_xlabel("Utility evaluations per player, $m/n$", fontsize=label_fontsize)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    axes[0].set_ylabel("Relative $L_2$ error", fontsize=label_fontsize)
    axes[2].set_ylabel("Relative $L_2$ error", fontsize=label_fontsize)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"pilot_fraction_sensitivity_4panel.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
