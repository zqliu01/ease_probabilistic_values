"""
Generate all SOU full benchmark figures from post-training CSV outputs.

The run_benchmark_n*.py scripts own training and CSV generation. This script
owns figure generation for every n and writes all PNGs to one figures/ folder.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
MPLCONFIG_DIR = SCRIPT_DIR / ".mplconfig"
CACHE_DIR = SCRIPT_DIR / ".cache"

MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BENCHMARK_MODULES = {
    40: "run_benchmark_n40",
    80: "run_benchmark_n80",
    160: "run_benchmark_n160",
}

AUCC_SEMIVALUE_ORDER = [
    ("shapley", "Shapley"),
    ("beta4_1", "BS(4,1)"),
    ("beta1_4", "BS(1,4)"),
    ("wb0p25", "WB(0.25)"),
    ("wb0p5", "WB(0.5)"),
    ("wb0p75", "WB(0.75)"),
]

AUCC_METHOD_LABELS = {
    "EaseSHAP_interaction_nonlinear": "EASE-FO",
    "EaseSHAP_size_player": "EASE-SP",
    "OFA_fixed": "OFA",
    "OFA_baseline": "OFA baseline",
    "sampling_lift": "Sampling lift",
    "SHAP_IQ": "SHAP-IQ",
    "GELS": "GELS",
    "improved_AME": "Improved AME",
    "kernelSHAP": "kernelSHAP",
    "LeverageSHAP": "LeverageSHAP",
    "permutation": "Permutation",
    "complement": "Complement",
    "group_testing": "Group testing",
    "WSL": "WSL",
    "weighted_permutation": "Weighted permutation",
    "OFA_optimal": "OFA optimal",
    "WGELS_shapley": "WGELS",
    "AME": "AME",
    "RegressionMSR_unbiased": "RegressionMSR",
    "PolySHAP_regression": "PolySHAP",
}

AUCC_METHOD_COLORS = {
    "EaseSHAP_interaction_nonlinear": "#c7413b",
    "EaseSHAP_size_player": "#437cb3",
    "OFA_fixed": "#f2ac69",
    "OFA_baseline": "#ff7f0e",
    "sampling_lift": "#f7d1a7",
    "SHAP_IQ": "#82bc78",
    "GELS": "#c3e8b4",
    "improved_AME": "#d62728",
    "kernelSHAP": "#d47370",
    "LeverageSHAP": "#f4bcb9",
    "permutation": "#b199ce",
    "complement": "#d4c8df",
    "group_testing": "#ab8d86",
    "WSL": "#d3bdb7",
    "weighted_permutation": "#e1a5d2",
    "OFA_optimal": "#c7c7c7",
    "WGELS_shapley": "#f2cfdf",
    "AME": "#a8a8a8",
    "RegressionMSR_unbiased": "#5f6368",
    "PolySHAP_regression": "#9edae5",
}

AUCC_METHOD_MARKERS = {
    "EaseSHAP_interaction_nonlinear": "o",
    "EaseSHAP_size_player": "s",
    "OFA_fixed": "^",
    "OFA_baseline": "D",
    "sampling_lift": "D",
    "SHAP_IQ": "v",
    "GELS": "P",
    "improved_AME": "*",
    "kernelSHAP": "X",
    "LeverageSHAP": "<",
    "permutation": ">",
    "complement": "h",
    "group_testing": "p",
    "WSL": "*",
    "weighted_permutation": "H",
    "OFA_optimal": "x",
    "WGELS_shapley": "p",
    "AME": "d",
    "RegressionMSR_unbiased": "1",
    "PolySHAP_regression": "^",
}

AUCC_DEFAULT_METHOD_ORDER = [
    "EaseSHAP_interaction_nonlinear",
    "EaseSHAP_size_player",
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
    "WSL",
    "weighted_permutation",
    "WGELS_shapley",
    "RegressionMSR_unbiased",
    "PolySHAP_regression",
]

DEFAULT_OMITTED_METHODS = {
    "AME",
}

AUCC_HIDDEN_BY_DEFAULT = {
    *DEFAULT_OMITTED_METHODS,
    "OFA_baseline",
    "OFA_optimal",
}

AUCC_EASE_METHODS = {
    "EaseSHAP_interaction_nonlinear",
    "EaseSHAP_size_player",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        nargs="*",
        choices=sorted(BENCHMARK_MODULES),
        default=None,
        help="Optional n values to plot. Defaults to all available n values.",
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        choices=["all", "summary", "aucc", "custom-aucc", "runtime"],
        default=["all"],
        help="Figure families to generate.",
    )
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method ids to plot, in the requested order.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a selected figure family cannot be generated from the available CSVs.",
    )
    return parser.parse_args()


def selected_figures(args: argparse.Namespace) -> set[str]:
    if "all" in args.figures:
        return {"summary", "aucc", "custom-aucc", "runtime"}
    return set(args.figures)


def load_benchmark(n: int) -> ModuleType:
    return importlib.import_module(BENCHMARK_MODULES[n])


def read_rows(path: Path, *, required: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing CSV: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_matrix(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    if not path.exists():
        return [], {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], {}
        methods = [field for field in reader.fieldnames if field != "setting"]
        table: dict[str, dict[str, float]] = {}
        for row in reader:
            setting = row.get("setting", "")
            if not setting:
                continue
            values = {}
            for method in methods:
                value = row.get(method, "")
                if value != "":
                    values[method] = float(value)
            table[setting] = values
        return methods, table


def select_benchmark_methods(benchmark: ModuleType, requested_methods: list[str] | None) -> list[dict[str, Any]]:
    if requested_methods is None:
        return [
            method
            for method in benchmark.METHOD_SPECS
            if method["name"] not in DEFAULT_OMITTED_METHODS
        ]
    return benchmark.select_methods(benchmark.METHOD_SPECS, requested_methods)


def select_aucc_methods(available: list[str], requested_methods: list[str] | None) -> list[str]:
    available_set = set(available)
    if requested_methods:
        missing = [method for method in requested_methods if method not in available_set]
        if missing:
            raise ValueError(f"Methods not found in plot data: {missing}")
        return requested_methods

    visible = [method for method in available if method not in AUCC_HIDDEN_BY_DEFAULT]
    ordered = [method for method in AUCC_DEFAULT_METHOD_ORDER if method in visible]
    ordered.extend(method for method in visible if method not in ordered)
    return ordered


def style_for_aucc_methods(methods: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
    color_cycle = list(plt.get_cmap("tab20").colors)
    colors = {
        method: AUCC_METHOD_COLORS.get(method, color_cycle[idx % len(color_cycle)])
        for idx, method in enumerate(methods)
    }

    markers_base = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*", "8", "H", "d", "1", "2", "3", "4"]
    markers = {
        method: AUCC_METHOD_MARKERS.get(method, markers_base[idx % len(markers_base)])
        for idx, method in enumerate(methods)
    }
    return colors, markers


def plot_summary_figures(
    *,
    benchmark: ModuleType,
    rows: list[dict[str, str]],
    methods: list[dict[str, Any]],
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    if not rows:
        return []

    color_map = plt.get_cmap("tab20")
    colors = {method["name"]: color_map(i % 20) for i, method in enumerate(methods)}

    row_lookup: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["semivalue_name"],
            benchmark.row_eta_text(row),
            row["algorithm"],
            int(row["nue"]),
        )
        row_lookup[key].append(float(row["rel_l2_error"]))

    settings = benchmark.build_settings()
    selected_setting_labels = {setting["label"] for setting in settings}
    selected_sv_names = list(dict.fromkeys(setting["name"] for setting in settings))
    selected_eta_labels = [benchmark.eta_text(alpha) for alpha in benchmark.GAME_ALPHAS]
    available_sv_names = {row["semivalue_name"] for row in rows}

    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for sv_spec in benchmark.SEMIVALUE_SPECS:
        if sv_spec["name"] not in selected_sv_names:
            continue
        if sv_spec["name"] not in available_sv_names:
            continue

        fig, axes = plt.subplots(
            1,
            len(benchmark.GAME_ALPHAS),
            figsize=(5.2 * len(benchmark.GAME_ALPHAS), 4.2),
            sharey=True,
        )
        if len(benchmark.GAME_ALPHAS) == 1:
            axes = [axes]

        plotted_algorithms = []
        for ax, alpha, eta_str in zip(axes, benchmark.GAME_ALPHAS, selected_eta_labels):
            setting_cur = benchmark.setting_label(alpha, sv_spec)
            if setting_cur not in selected_setting_labels:
                ax.set_visible(False)
                continue

            for method in methods:
                if not benchmark.is_compatible(method, {**sv_spec, "alpha": alpha, "label": setting_cur}):
                    continue
                xs = []
                means = []
                stds = []
                for nue in benchmark.NUE_BUDGETS:
                    vals = row_lookup.get((sv_spec["name"], eta_str, method["name"], int(nue)), [])
                    if not vals:
                        continue
                    xs.append(int(nue) * benchmark.N)
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                if not xs:
                    continue

                xs_arr = np.asarray(xs, dtype=int)
                means_arr = np.maximum(np.asarray(means, dtype=float), 1e-16)
                stds_arr = np.asarray(stds, dtype=float)
                lower = np.maximum(means_arr - stds_arr, 1e-16)
                upper = np.maximum(means_arr + stds_arr, 1e-16)
                ax.plot(xs_arr, means_arr, linewidth=1.2, label=method["name"], color=colors[method["name"]])
                ax.fill_between(xs_arr, lower, upper, alpha=0.13, color=colors[method["name"]])
                if method["name"] not in plotted_algorithms:
                    plotted_algorithms.append(method["name"])

            ax.set_title(rf"$\eta = {eta_str}$", fontsize=10)
            ax.set_xlabel("Total NUE", fontsize=9)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel("Relative L2 error", fontsize=9)
        handles = [
            plt.Line2D([0], [0], color=colors[name], linewidth=1.5, label=name)
            for name in plotted_algorithms
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(4, len(handles)),
                bbox_to_anchor=(0.5, -0.24),
                fontsize=7,
                frameon=True,
            )

        fig.suptitle(f"{sv_spec['title']}, mean +/- 1 std over {benchmark.N_RUNS} runs", fontsize=12, y=1.02)
        fig.tight_layout()
        path = out_dir / f"{sv_spec['name']}_rel_l2_n{benchmark.N}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def compute_plot_aucc(
    benchmark: ModuleType,
    rows: list[dict[str, str]],
    metric: str = "rel_l2_error",
    aucc_mode: str = "mean",
) -> dict[tuple[str, str, str], list[float]]:
    by_run: dict[tuple[str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if not row.get(metric):
            continue
        key = (benchmark.row_eta_text(row), row["semivalue_name"], row["algorithm"], row["run_idx"])
        x_value = int(row.get("total_nue") or int(row["nue"]) * benchmark.N)
        by_run[key].append((x_value, float(row[metric])))

    by_setting_method: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (eta, semivalue, algorithm, _run_idx), curve in by_run.items():
        curve_sorted = sorted(curve)
        xs = np.asarray([point[0] for point in curve_sorted], dtype=float)
        ys = np.asarray([point[1] for point in curve_sorted], dtype=float)
        if len(ys) == 0:
            continue
        if aucc_mode == "trapz" and len(ys) > 1 and xs[-1] > xs[0]:
            aucc = float(np.trapz(ys, xs) / (xs[-1] - xs[0]))
        else:
            aucc = float(np.mean(ys))
        by_setting_method[(eta, semivalue, algorithm)].append(aucc)
    return by_setting_method


def write_aucc_panel_figure(
    *,
    benchmark: ModuleType,
    rows: list[dict[str, str]],
    requested_methods: list[str] | None,
    eta_labels: list[str],
    out_dir: Path,
    output_stem: str,
    figsize: tuple[float, float],
    dpi: int = 300,
    legend_position: str = "bottom",
    font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    show_title: bool = True,
) -> Path | None:
    if not rows:
        return None

    available = list(dict.fromkeys(row["algorithm"] for row in rows))
    methods = select_aucc_methods(available, requested_methods)
    if not methods:
        return None

    by_setting_method = compute_plot_aucc(benchmark, rows)
    return write_aucc_panel_from_values(
        by_setting_method=by_setting_method,
        methods=methods,
        eta_labels=eta_labels,
        out_dir=out_dir,
        output_stem=output_stem,
        figsize=figsize,
        dpi=dpi,
        legend_position=legend_position,
        font_scale=font_scale,
        legend_font_scale=legend_font_scale,
        show_title=show_title,
    )


def setting_lookup(benchmark: ModuleType) -> dict[str, tuple[str, str]]:
    return {
        setting["label"]: (benchmark.eta_text(setting["alpha"]), setting["name"])
        for setting in benchmark.build_settings()
    }


def write_aucc_panel_figure_from_tables(
    *,
    benchmark: ModuleType,
    mean_path: Path,
    std_path: Path,
    requested_methods: list[str] | None,
    eta_labels: list[str],
    out_dir: Path,
    output_stem: str,
    figsize: tuple[float, float],
    dpi: int = 300,
    legend_position: str = "bottom",
    font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    show_title: bool = True,
) -> Path | None:
    available_methods, means_by_setting = read_matrix(mean_path)
    _std_methods, stds_by_setting = read_matrix(std_path)
    if not means_by_setting:
        return None

    methods = select_aucc_methods(available_methods, requested_methods)
    if not methods:
        return None

    lookup = setting_lookup(benchmark)
    by_setting_method: dict[tuple[str, str, str], tuple[float, float]] = {}
    for setting, means in means_by_setting.items():
        if setting not in lookup:
            continue
        eta_label, semivalue = lookup[setting]
        stds = stds_by_setting.get(setting, {})
        for method in methods:
            if method not in means:
                continue
            by_setting_method[(eta_label, semivalue, method)] = (float(means[method]), float(stds.get(method, 0.0)))

    if not by_setting_method:
        return None

    return write_aucc_panel_from_table_values(
        by_setting_method=by_setting_method,
        methods=methods,
        eta_labels=eta_labels,
        out_dir=out_dir,
        output_stem=output_stem,
        figsize=figsize,
        dpi=dpi,
        legend_position=legend_position,
        font_scale=font_scale,
        legend_font_scale=legend_font_scale,
        show_title=show_title,
    )


def write_aucc_panel_from_values(
    *,
    by_setting_method: dict[tuple[str, str, str], list[float]],
    methods: list[str],
    eta_labels: list[str],
    out_dir: Path,
    output_stem: str,
    figsize: tuple[float, float],
    dpi: int,
    legend_position: str,
    font_scale: float,
    legend_font_scale: float,
    show_title: bool,
) -> Path | None:
    table_values: dict[tuple[str, str, str], tuple[float, float]] = {}
    for key, values in by_setting_method.items():
        if values:
            table_values[key] = (float(np.mean(values)), float(np.std(values)))
    return write_aucc_panel_from_table_values(
        by_setting_method=table_values,
        methods=methods,
        eta_labels=eta_labels,
        out_dir=out_dir,
        output_stem=output_stem,
        figsize=figsize,
        dpi=dpi,
        legend_position=legend_position,
        font_scale=font_scale,
        legend_font_scale=legend_font_scale,
        show_title=show_title,
    )


def write_aucc_panel_from_table_values(
    *,
    by_setting_method: dict[tuple[str, str, str], tuple[float, float]],
    methods: list[str],
    eta_labels: list[str],
    out_dir: Path,
    output_stem: str,
    figsize: tuple[float, float],
    dpi: int,
    legend_position: str,
    font_scale: float,
    legend_font_scale: float,
    show_title: bool,
) -> Path | None:
    if not by_setting_method:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    colors, markers = style_for_aucc_methods(methods)
    x_base = np.arange(len(AUCC_SEMIVALUE_ORDER), dtype=float)
    jitter = 0.08
    offsets = np.linspace(-jitter, jitter, len(methods)) if len(methods) > 1 else np.zeros(len(methods))

    positive_values = [mean for mean, _std in by_setting_method.values() if mean > 0.0]
    min_positive = min(positive_values) if positive_values else 1e-8

    fig, axes = plt.subplots(1, len(eta_labels), figsize=figsize, sharey=False)
    if len(eta_labels) == 1:
        axes = [axes]

    handles = []
    for method in methods:
        label = AUCC_METHOD_LABELS.get(method, method)
        linewidth = 1.6 if method in AUCC_EASE_METHODS else 1.05
        alpha_line = 0.95 if method in AUCC_EASE_METHODS else 0.68
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=colors[method],
                marker=markers[method],
                linewidth=linewidth,
                markersize=4.3,
                label=label,
                alpha=alpha_line,
            )
        )

    for ax, eta_label in zip(axes, eta_labels):
        for method_idx, method in enumerate(methods):
            xs = []
            means = []
            stds = []
            for semivalue_idx, (semivalue, _label) in enumerate(AUCC_SEMIVALUE_ORDER):
                value = by_setting_method.get((eta_label, semivalue, method))
                if value is None:
                    continue
                mean, std = value
                xs.append(x_base[semivalue_idx] + offsets[method_idx])
                means.append(mean)
                stds.append(std)

            if not xs:
                continue

            xs_arr = np.asarray(xs, dtype=float)
            means_arr = np.asarray(means, dtype=float)
            stds_arr = np.asarray(stds, dtype=float)
            lower = np.maximum(means_arr - stds_arr, min_positive / 5.0)
            upper = np.maximum(means_arr + stds_arr, min_positive / 5.0)
            linewidth = 1.6 if method in AUCC_EASE_METHODS else 1.05
            markersize = 4.3 if method in AUCC_EASE_METHODS else 3.3
            alpha_line = 0.95 if method in AUCC_EASE_METHODS else 0.68

            if len(xs_arr) > 1:
                ax.plot(
                    xs_arr,
                    means_arr,
                    color=colors[method],
                    marker=markers[method],
                    linewidth=linewidth,
                    markersize=markersize,
                    alpha=alpha_line,
                    zorder=3 if method in AUCC_EASE_METHODS else 2,
                )
                ax.fill_between(
                    xs_arr,
                    lower,
                    upper,
                    color=colors[method],
                    alpha=0.22 if method in AUCC_EASE_METHODS else 0.14,
                    linewidth=0,
                    zorder=1,
                )
            else:
                ax.errorbar(
                    xs_arr,
                    means_arr,
                    yerr=stds_arr,
                    fmt=markers[method],
                    color=colors[method],
                    markersize=markersize,
                    alpha=alpha_line,
                    capsize=2.2,
                    linewidth=linewidth,
                    zorder=3 if method in AUCC_EASE_METHODS else 2,
                )

        if show_title:
            ax.set_title(rf"$\eta = {eta_label}$", fontsize=10.5 * font_scale, pad=7)
        ax.set_xticks(x_base)
        ax.set_xticklabels(
            [label for _name, label in AUCC_SEMIVALUE_ORDER],
            rotation=28,
            ha="right",
            fontsize=10.0 * font_scale,
        )
        ax.set_yscale("log")
        ax.tick_params(axis="y", which="major", labelsize=10.0 * font_scale)
        ax.tick_params(axis="y", which="minor", labelsize=8.0 * font_scale)
        ax.grid(True, axis="y", which="major", color="#d9d9d9", linewidth=0.65)
        ax.grid(True, axis="y", which="minor", color="#eeeeee", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.07)

    axes[0].set_ylabel("AUCC", fontsize=12.0 * font_scale)
    if legend_position == "right":
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.79, 0.5),
            ncol=1,
            frameon=False,
            fontsize=7.4 * font_scale * legend_font_scale,
            handlelength=1.9,
            columnspacing=1.1,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0), w_pad=1.0)
    elif legend_position == "bottom":
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=min(5, len(handles)),
            frameon=False,
            fontsize=7.4 * font_scale * legend_font_scale,
            handlelength=1.9,
            columnspacing=1.1,
        )
        fig.tight_layout(rect=(0.0, 0.285, 1.0, 1.0), w_pad=1.0)
    else:
        raise ValueError(f"Unknown legend_position: {legend_position}")

    path = out_dir / f"{output_stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def runtime_without_checkpoint_readouts_sec(row: dict[str, str]) -> float:
    regular = row.get("regular_elapsed_sec", "")
    final_readout = _as_float(row, "final_readout_elapsed_sec")
    if regular != "":
        return _as_float(row, "regular_elapsed_sec") + final_readout

    elapsed = _as_float(row, "elapsed_sec")
    readout = _as_float(row, "readout_elapsed_sec")
    return max(0.0, elapsed - readout) + final_readout


def select_runtime_methods(timing_rows: list[dict[str, str]], requested_methods: list[str] | None) -> list[str]:
    available = list(dict.fromkeys(row["algorithm"] for row in timing_rows if row.get("status", "ok") == "ok"))
    if requested_methods:
        missing = [method for method in requested_methods if method not in available]
        if missing:
            raise ValueError(f"Methods not found in timing rows: {missing}")
        return requested_methods

    visible = [method for method in available if method not in DEFAULT_OMITTED_METHODS]
    ordered = [method for method in AUCC_DEFAULT_METHOD_ORDER if method in visible]
    ordered.extend(method for method in visible if method not in ordered)
    return ordered


def write_runtime_no_checkpoint_figure(
    *,
    benchmark: ModuleType,
    timing_rows: list[dict[str, str]],
    requested_methods: list[str] | None,
    out_dir: Path,
    output_stem: str,
    dpi: int = 300,
) -> Path | None:
    if not timing_rows:
        return None

    methods = select_runtime_methods(timing_rows, requested_methods)
    if not methods:
        return None

    by_semivalue_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in timing_rows:
        if row.get("status", "ok") != "ok":
            continue
        method = row["algorithm"]
        if method not in methods:
            continue
        semivalue_name = row["semivalue_name"]
        runtime_min = runtime_without_checkpoint_readouts_sec(row) / 60.0
        if np.isfinite(runtime_min) and runtime_min > 0.0:
            by_semivalue_method[(semivalue_name, method)].append(runtime_min)

    if not by_semivalue_method:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    colors, _markers = style_for_aucc_methods(methods)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), sharex=True)
    axes_flat = axes.ravel()
    positive_means = []

    for ax, (semivalue_name, semivalue_label) in zip(axes_flat, AUCC_SEMIVALUE_ORDER):
        present_methods = [method for method in methods if by_semivalue_method.get((semivalue_name, method))]
        if not present_methods:
            ax.set_visible(False)
            continue

        means = np.asarray(
            [np.mean(by_semivalue_method[(semivalue_name, method)]) for method in present_methods],
            dtype=float,
        )
        stds = np.asarray(
            [np.std(by_semivalue_method[(semivalue_name, method)]) for method in present_methods],
            dtype=float,
        )
        positive_means.extend(float(value) for value in means if value > 0.0)

        y_pos = np.arange(len(present_methods), dtype=float)
        ax.barh(
            y_pos,
            means,
            color=[colors[method] for method in present_methods],
            alpha=0.86,
            height=0.66,
            edgecolor="white",
            linewidth=0.4,
        )
        if np.any(stds > 0.0):
            lower = np.minimum(stds, means * 0.9)
            ax.errorbar(
                means,
                y_pos,
                xerr=np.vstack([lower, stds]),
                fmt="none",
                ecolor="#333333",
                elinewidth=0.65,
                capsize=1.8,
                alpha=0.72,
            )

        labels = [AUCC_METHOD_LABELS.get(method, method) for method in present_methods]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7.0)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_title(semivalue_label, fontsize=10.5)
        ax.grid(True, axis="x", which="both", alpha=0.28)
        ax.tick_params(axis="x", labelsize=8)

    if positive_means:
        xmin = min(positive_means) * 0.7
        xmax = max(positive_means) * 1.35
        for ax in axes_flat:
            if ax.get_visible():
                ax.set_xlim(xmin, xmax)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Runtime without checkpoint readouts (minutes)", fontsize=9)

    fig.suptitle(
        f"Runtime for one final estimate, n={benchmark.N}, total NUE={benchmark.TOTAL_NUE:,}",
        fontsize=13,
        y=0.995,
    )
    fig.text(0.5, 0.955, "Bars show mean over eta values and seeds; error bars show +/- 1 std.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935))

    path = out_dir / f"{output_stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def append_written(written: list[Path], path: Path | None) -> bool:
    if path is None:
        return False
    written.append(path)
    return True


def plot_for_benchmark(
    *,
    benchmark: ModuleType,
    figure_names: set[str],
    out_dir: Path,
    dpi: int,
    requested_methods: list[str] | None,
    strict: bool,
) -> tuple[list[Path], list[str]]:
    written: list[Path] = []
    skipped: list[str] = []
    methods = select_benchmark_methods(benchmark, requested_methods)
    summary_path = benchmark.OUT / "summary.csv"
    timing_path = benchmark.OUT / "run_timing.csv"
    aucc_mean_path = benchmark.OUT / "aucc_mean.csv"
    aucc_std_path = benchmark.OUT / "aucc_std.csv"
    rows = read_rows(summary_path, required=False)

    if "summary" in figure_names:
        paths = plot_summary_figures(
            benchmark=benchmark,
            rows=rows,
            methods=methods,
            out_dir=out_dir,
            dpi=dpi,
        )
        written.extend(paths)
        if not paths:
            skipped.append(f"summary_n{benchmark.N}: missing or empty {summary_path}")

    if "aucc" in figure_names:
        path = write_aucc_panel_figure_from_tables(
            benchmark=benchmark,
            mean_path=aucc_mean_path,
            std_path=aucc_std_path,
            requested_methods=requested_methods,
            eta_labels=[benchmark.eta_text(alpha) for alpha in benchmark.GAME_ALPHAS],
            out_dir=out_dir,
            output_stem=f"aucc_by_eta_n{benchmark.N}",
            figsize=(8.8, 3.65),
            dpi=dpi,
        )
        if path is None and rows:
            path = write_aucc_panel_figure(
                benchmark=benchmark,
                rows=rows,
                requested_methods=requested_methods,
                eta_labels=[benchmark.eta_text(alpha) for alpha in benchmark.GAME_ALPHAS],
                out_dir=out_dir,
                output_stem=f"aucc_by_eta_n{benchmark.N}",
                figsize=(8.8, 3.65),
                dpi=dpi,
            )
        if not append_written(written, path):
            skipped.append(f"aucc_n{benchmark.N}: missing or empty AUCC/summary CSVs")

    if "custom-aucc" in figure_names:
        single_path = write_aucc_panel_figure_from_tables(
            benchmark=benchmark,
            mean_path=aucc_mean_path,
            std_path=aucc_std_path,
            requested_methods=requested_methods,
            eta_labels=["0.25"],
            out_dir=out_dir,
            output_stem=f"aucc_eta_0p25_n{benchmark.N}",
            figsize=(7.0, 4.2),
            dpi=dpi,
            legend_position="right",
            font_scale=1.35,
            legend_font_scale=1.12,
            show_title=False,
        )
        pair_path = write_aucc_panel_figure_from_tables(
            benchmark=benchmark,
            mean_path=aucc_mean_path,
            std_path=aucc_std_path,
            requested_methods=requested_methods,
            eta_labels=["0.5", "0.75"],
            out_dir=out_dir,
            output_stem=f"aucc_eta_0p5_0p75_n{benchmark.N}",
            figsize=(6.1, 3.65),
            dpi=dpi,
        )

        if single_path is None and rows:
            single_path = write_aucc_panel_figure(
                benchmark=benchmark,
                rows=rows,
                requested_methods=requested_methods,
                eta_labels=["0.25"],
                out_dir=out_dir,
                output_stem=f"aucc_eta_0p25_n{benchmark.N}",
                figsize=(7.0, 4.2),
                dpi=dpi,
                legend_position="right",
                font_scale=1.35,
                legend_font_scale=1.12,
                show_title=False,
            )
        if pair_path is None and rows:
            pair_path = write_aucc_panel_figure(
                benchmark=benchmark,
                rows=rows,
                requested_methods=requested_methods,
                eta_labels=["0.5", "0.75"],
                out_dir=out_dir,
                output_stem=f"aucc_eta_0p5_0p75_n{benchmark.N}",
                figsize=(6.1, 3.65),
                dpi=dpi,
            )

        custom_written = 0
        custom_written += int(append_written(written, single_path))
        custom_written += int(append_written(written, pair_path))
        if custom_written != 2:
            skipped.append(f"custom_aucc_n{benchmark.N}: missing or empty AUCC/summary CSVs")

    if "runtime" in figure_names:
        timing_rows = read_rows(timing_path, required=False)
        path = write_runtime_no_checkpoint_figure(
            benchmark=benchmark,
            timing_rows=timing_rows,
            requested_methods=requested_methods,
            out_dir=out_dir,
            output_stem=f"runtime_no_checkpoints_n{benchmark.N}",
            dpi=dpi,
        )
        if not append_written(written, path):
            skipped.append(f"runtime_n{benchmark.N}: missing or empty {timing_path}")

    if strict and skipped:
        raise RuntimeError("Could not generate selected figures:\n" + "\n".join(skipped))
    return written, skipped


def main() -> None:
    args = parse_args()
    ns = args.n or sorted(BENCHMARK_MODULES)
    figure_names = selected_figures(args)

    all_written: list[Path] = []
    all_skipped: list[str] = []
    for n in ns:
        benchmark = load_benchmark(n)
        written, skipped = plot_for_benchmark(
            benchmark=benchmark,
            figure_names=figure_names,
            out_dir=args.out_dir,
            dpi=args.dpi,
            requested_methods=args.methods,
            strict=args.strict,
        )
        all_written.extend(written)
        all_skipped.extend(skipped)

    for path in all_written:
        print(f"wrote {path}")
    for item in all_skipped:
        print(f"skipped {item}")
    if not all_written:
        raise RuntimeError("No figures were written.")


if __name__ == "__main__":
    main()
