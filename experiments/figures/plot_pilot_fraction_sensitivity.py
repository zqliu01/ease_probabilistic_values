"""Plot pilot-fraction sensitivity from the frozen published summary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "real_world_benchmark"
    / "results"
    / "published"
    / "l2_summary.csv"
)
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "published" / "pilot_fraction_sensitivity_4panel"

DATASETS = ("cifar10", "breast_cancer", "communities_crime", "nhanesi")
DATASET_TITLES = {
    "cifar10": "CIFAR-10",
    "breast_cancer": "Breast Cancer",
    "communities_crime": "Communities and Crime",
    "nhanesi": "NHANES I",
}
N_PLAYERS = {
    "cifar10": 16,
    "breast_cancer": 30,
    "communities_crime": 101,
    "nhanesi": 79,
}
PILOT_FRACTIONS = ("0p05", "0p10", "0p20", "0p40")
METHODS = tuple(
    f"EaseSHAP_interaction_nonlinear_pilot{fraction}"
    for fraction in PILOT_FRACTIONS
)
PILOT_LABELS = {
    method: f"pilot = {fraction.replace('p', '.')}"
    for method, fraction in zip(METHODS, PILOT_FRACTIONS)
}
PILOT_COLORS = dict(zip(METHODS, ("#fcae91", "#fb6a4a", "#de2d26", "#a50f15")))


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
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mplconfig_dir = SCRIPT_DIR / ".mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_csv(
        args.input,
        usecols=(
            "dataset",
            "semivalue",
            "method",
            "checkpoint_idx",
            "actual_budget",
            "mean_rel_l2",
            "sem_rel_l2",
        ),
    )
    frame = frame[
        frame["dataset"].isin(DATASETS)
        & frame["semivalue"].eq("shapley")
        & frame["method"].isin(METHODS)
    ]
    missing = sorted(set(DATASETS).difference(frame["dataset"].unique()))
    if missing:
        raise ValueError(f"Published sensitivity input is missing: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), sharey=False)
    for ax, dataset in zip(axes.flat, DATASETS):
        panel = frame[frame["dataset"].eq(dataset)]
        for method in METHODS:
            rows = panel[panel["method"].eq(method)].sort_values("checkpoint_idx")
            x = rows["actual_budget"].to_numpy(dtype=np.float64) / N_PLAYERS[dataset]
            y = rows["mean_rel_l2"].to_numpy(dtype=np.float64)
            sem = rows["sem_rel_l2"].to_numpy(dtype=np.float64)
            color = PILOT_COLORS[method]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=5.5,
                linewidth=2.6,
                color=color,
                label=PILOT_LABELS[method],
            )
            ax.fill_between(
                x,
                np.maximum(y - sem, 1e-12),
                y + sem,
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        ax.set_title(DATASET_TITLES[dataset], fontsize=19, pad=8)
        ax.set_xlabel("Utility evaluations per player, $m/n$", fontsize=17)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(axis="both", labelsize=15)

    axes[0, 0].set_ylabel("Relative $L_2$ error", fontsize=17)
    axes[1, 0].set_ylabel("Relative $L_2$ error", fontsize=17)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=16,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in dict.fromkeys(args.formats):
        path = args.output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
