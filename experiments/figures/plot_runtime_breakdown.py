"""Plot six estimator-runtime breakdowns from the latest benchmark artifacts.

Each horizontal bar is the mean runtime for one method. The stack separates
utility evaluation, sampling, the instrumented final readout, and remaining
estimator work. Intermediate checkpoint-readout time is excluded so the total
represents the cost of producing a final estimate. Error bars show variability
in the total runtime over estimator seeds (ACSIncome) or explained instances
(the four real-world benchmarks).
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from plot_shapley_estimation_trajectories import (
    ACS_RUNS_ROOT,
    METHOD_LABELS,
    METHOD_ORDER,
    PANEL_TITLES,
    REAL_WORLD_DATASETS,
    REAL_WORLD_RESULTS_ROOT,
    WORKSPACE_DIR,
    canonical_method,
    configure_matplotlib,
    file_metadata,
    keep_paper_method,
    load_json,
    relative_to_workspace,
    resolve_sources,
    sha256,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "runtime_breakdown_6panel"
RUNTIME_COMPONENTS = (
    "utility_eval_sec",
    "sampling_sec",
    "final_readout_sec",
    "other_estimator_sec",
)
COMPONENT_LABELS = {
    "utility_eval_sec": "Utility evaluation",
    "sampling_sec": "Sampling",
    "final_readout_sec": "Final readout",
    "other_estimator_sec": "Other estimator work",
}
COMPONENT_COLORS = {
    "utility_eval_sec": "#4C78A8",
    "sampling_sec": "#F58518",
    "final_readout_sec": "#54A24B",
    "other_estimator_sec": "#B8B8B8",
}


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
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
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
        help="Variability shown for total runtime.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def finite_nonnegative(value: Any, *, field: str, source: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"Invalid {field}={value!r} in {source}")
    return number


def timing_components(
    timing: Mapping[str, Any],
    *,
    source: str,
    paper_runtime_sec: float | None = None,
) -> dict[str, float]:
    required = ("elapsed_sec", "sampling_sec", "utility_eval_sec")
    missing = [field for field in required if field not in timing]
    if missing:
        raise KeyError(f"Missing timing fields {missing} in {source}")

    elapsed = finite_nonnegative(timing["elapsed_sec"], field="elapsed_sec", source=source)
    readout = finite_nonnegative(
        timing.get("readout_elapsed_sec", 0.0),
        field="readout_elapsed_sec",
        source=source,
    )
    final_readout = finite_nonnegative(
        timing.get("final_readout_elapsed_sec", 0.0),
        field="final_readout_elapsed_sec",
        source=source,
    )
    checkpoint_readout = max(0.0, readout - final_readout)
    if paper_runtime_sec is None:
        total = max(0.0, elapsed - checkpoint_readout)
    else:
        total = finite_nonnegative(
            paper_runtime_sec,
            field="paper_runtime_sec",
            source=source,
        )

    utility_eval = finite_nonnegative(
        timing["utility_eval_sec"],
        field="utility_eval_sec",
        source=source,
    )
    sampling = finite_nonnegative(
        timing["sampling_sec"],
        field="sampling_sec",
        source=source,
    )
    residual = total - utility_eval - sampling - final_readout
    tolerance = max(1e-6, total * 1e-6)
    if residual < -tolerance:
        raise ValueError(
            f"Recorded components exceed paper runtime by {-residual:.6g} seconds in {source}"
        )
    residual = max(0.0, residual)
    return {
        "utility_eval_sec": utility_eval,
        "sampling_sec": sampling,
        "final_readout_sec": final_readout,
        "other_estimator_sec": residual,
        "total_runtime_sec": utility_eval + sampling + final_readout + residual,
        "checkpoint_readout_excluded_sec": checkpoint_readout,
    }


def load_acs_runtime(run_dir: Path, *, panel: str) -> pd.DataFrame:
    runs_path = run_dir / "runs.csv"
    if not runs_path.is_file():
        raise FileNotFoundError(f"Missing ACS runtime table: {runs_path}")
    runs = pd.read_csv(runs_path)
    required = {
        "status",
        "role",
        "method",
        "seed_index",
        "paper_runtime_sec",
        "estimate_sec",
        "sampling_sec",
        "utility_eval_sec",
        "readout_elapsed_sec",
        "final_readout_elapsed_sec",
    }
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"{runs_path} is missing required columns: {missing}")
    runs = runs[
        runs["status"].eq("ok")
        & runs["role"].eq("main")
        & runs["method"].astype(str).map(keep_paper_method)
    ].copy()
    if runs.empty:
        raise ValueError(f"No successful paper-method runs remain in {runs_path}")

    records: list[dict[str, Any]] = []
    for row in runs.to_dict(orient="records"):
        source = f"{runs_path}: method={row['method']} seed_index={row['seed_index']}"
        timing = {
            "elapsed_sec": row["estimate_sec"],
            "sampling_sec": row["sampling_sec"],
            "utility_eval_sec": row["utility_eval_sec"],
            "readout_elapsed_sec": row["readout_elapsed_sec"],
            "final_readout_elapsed_sec": row["final_readout_elapsed_sec"],
        }
        records.append(
            {
                "panel": panel,
                "method": canonical_method(str(row["method"])),
                "replicate": int(row["seed_index"]),
                **timing_components(
                    timing,
                    source=source,
                    paper_runtime_sec=float(row["paper_runtime_sec"]),
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def load_real_world_runtime(real_world_input: Path) -> tuple[pd.DataFrame, Path]:
    raw_dir = real_world_input.parent / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Missing raw timing artifacts next to {real_world_input}: {raw_dir}"
        )

    records: list[dict[str, Any]] = []
    for dataset in REAL_WORLD_DATASETS:
        semivalue_dir = raw_dir / dataset / "shapley"
        if not semivalue_dir.is_dir():
            raise FileNotFoundError(f"Missing Shapley runtime directory: {semivalue_dir}")
        for method_dir in sorted(path for path in semivalue_dir.iterdir() if path.is_dir()):
            method_from_path = method_dir.name
            if not keep_paper_method(method_from_path):
                continue
            for path in sorted(method_dir.glob("*.pkl")):
                with path.open("rb") as handle:
                    payload = pickle.load(handle)
                if str(payload.get("status", "")) != "ok":
                    continue
                timing = payload.get("timing")
                if not isinstance(timing, Mapping):
                    raise TypeError(f"Missing timing mapping in {path}")
                method_payload = payload.get("method", {})
                method = (
                    str(method_payload.get("name", method_from_path))
                    if isinstance(method_payload, Mapping)
                    else method_from_path
                )
                if not keep_paper_method(method):
                    continue
                task = payload.get("task", {})
                instance_id = (
                    int(task.get("instance_id", -1))
                    if isinstance(task, Mapping)
                    else -1
                )
                run_idx = int(task.get("run_idx", 0)) if isinstance(task, Mapping) else 0
                records.append(
                    {
                        "panel": dataset,
                        "method": canonical_method(method),
                        "replicate": f"instance_{instance_id:03d}_run{run_idx:02d}",
                        **timing_components(timing, source=str(path)),
                    }
                )

    runtime = pd.DataFrame.from_records(records)
    if runtime.empty:
        raise ValueError(f"No usable real-world timing records were found under {raw_dir}")
    for dataset in REAL_WORLD_DATASETS:
        present = set(runtime.loc[runtime["panel"].eq(dataset), "method"])
        missing = [method for method in METHOD_ORDER if method not in present]
        if missing:
            raise ValueError(f"Missing runtime methods for {dataset}: {missing}")
    return runtime, raw_dir


def summarize_runtime(runtime: pd.DataFrame, *, band: str) -> pd.DataFrame:
    grouped = runtime.groupby(["panel", "method"], sort=False)
    component_means = grouped[list(RUNTIME_COMPONENTS)].mean().reset_index()
    total_stats = (
        grouped["total_runtime_sec"]
        .agg(total_mean="mean", total_std="std", count="count")
        .reset_index()
    )
    summary = component_means.merge(total_stats, on=["panel", "method"], how="inner")
    summary["total_std"] = summary["total_std"].fillna(0.0)
    summary["total_sem"] = summary["total_std"] / np.sqrt(summary["count"].clip(lower=1))
    summary["uncertainty"] = summary[f"total_{band}"]
    summary["stack_total"] = summary[list(RUNTIME_COMPONENTS)].sum(axis=1)
    if not np.allclose(summary["stack_total"], summary["total_mean"], rtol=1e-9, atol=1e-9):
        raise AssertionError("Mean runtime components do not sum to mean total runtime.")
    return summary


def plot_runtime_panel(
    ax: Any,
    summary: pd.DataFrame,
    *,
    title: str,
    panel_label: str,
    title_fontsize: float = 11.5,
    panel_label_fontsize: float = 11.5,
    ytick_fontsize: float = 8.0,
    xtick_fontsize: float = 8.2,
) -> None:
    from matplotlib.ticker import EngFormatter, MaxNLocator

    panel = summary.set_index("method").reindex(METHOD_ORDER)
    if panel[list(RUNTIME_COMPONENTS)].isna().any().any():
        missing = panel.index[panel["total_mean"].isna()].tolist()
        raise ValueError(f"Missing methods while plotting {title}: {missing}")
    y = np.arange(len(METHOD_ORDER), dtype=float)
    left = np.zeros(len(METHOD_ORDER), dtype=float)
    for component in RUNTIME_COMPONENTS:
        values = panel[component].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.68,
            color=COMPONENT_COLORS[component],
            edgecolor="white",
            linewidth=0.25,
            label=COMPONENT_LABELS[component],
            zorder=2,
        )
        left += values
    ax.errorbar(
        panel["total_mean"].to_numpy(dtype=float),
        y,
        xerr=panel["uncertainty"].to_numpy(dtype=float),
        fmt="none",
        ecolor="#252525",
        elinewidth=0.75,
        capsize=1.8,
        capthick=0.75,
        zorder=4,
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
    ax.set_yticks(y)
    ax.set_yticklabels(
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        fontsize=ytick_fontsize,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.xaxis.set_major_formatter(
        EngFormatter(unit="s", places=0, sep="\N{NARROW NO-BREAK SPACE}")
    )
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.65, alpha=0.85, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def create_figure(
    summary: pd.DataFrame,
    *,
    output_stem: Path,
    formats: tuple[str, ...] | list[str],
    dpi: int,
    band: str,
) -> list[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.8), sharey=True)
    panels = (
        ("acs_logistic", PANEL_TITLES["acs_logistic"]),
        ("acs_xgboost", PANEL_TITLES["acs_xgboost"]),
        ("cifar10", PANEL_TITLES["cifar10"]),
        ("breast_cancer", PANEL_TITLES["breast_cancer"]),
        ("communities_crime", PANEL_TITLES["communities_crime"]),
        ("nhanesi", PANEL_TITLES["nhanesi"]),
    )
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for ax, (panel_name, title), panel_label in zip(axes.flat, panels, panel_labels):
        plot_runtime_panel(
            ax,
            summary[summary["panel"].eq(panel_name)],
            title=title,
            panel_label=panel_label,
        )
    axes[0, 0].invert_yaxis()

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
        fontsize=11.5,
        y=0.115,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.8,
        handlelength=1.8,
    )
    fig.text(
        0.5,
        0.006,
        f"Error bars show ±1 {band} of total runtime.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4d4d4d",
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.95,
        bottom=0.17,
        wspace=0.26,
        hspace=0.30,
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


def write_outputs(
    summary: pd.DataFrame,
    *,
    output_stem: Path,
    output_paths: list[Path],
    logistic_run: Path,
    xgboost_run: Path,
    real_world_input: Path,
    real_world_raw_dir: Path,
    band: str,
) -> tuple[Path, Path]:
    output_stem = output_stem.expanduser().resolve()
    summary_path = output_stem.with_name(f"{output_stem.name}_summary.csv")
    summary.sort_values(["panel", "method"]).to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_definition": (
            "elapsed estimator time minus intermediate checkpoint readouts; "
            "the final readout is retained"
        ),
        "components": {
            "utility_eval_sec": "time in estimator.run(samples)",
            "sampling_sec": "time advancing the estimator sampling iterator",
            "final_readout_sec": "instrumented final estimator readout when available",
            "other_estimator_sec": (
                "residual estimator time, including setup, aggregation, finalization, "
                "orchestration, and uninstrumented readout work"
            ),
        },
        "aggregation": {
            "bar": "component-wise mean",
            "error_bar": f"±1 {band} of total runtime",
            "acs_replicates": "estimator seeds",
            "real_world_replicates": "explained instances and benchmark runs",
        },
        "runtime_caveat": {
            "cifar10": (
                "utility access is an indexed lookup in the precomputed coalition-value "
                "array rather than end-to-end image-model inference"
            )
        },
        "sources": {
            "acs_logistic": file_metadata(logistic_run / "runs.csv"),
            "acs_xgboost": file_metadata(xgboost_run / "runs.csv"),
            "acs_logistic_config": file_metadata(logistic_run / "config.json"),
            "acs_xgboost_config": file_metadata(xgboost_run / "config.json"),
            "acs_utility_cache_mode": {
                "logistic": load_json(logistic_run / "config.json").get(
                    "utility_cache_mode",
                    "not_recorded",
                ),
                "xgboost": load_json(xgboost_run / "config.json").get(
                    "utility_cache_mode",
                    "not_recorded",
                ),
            },
            "real_world_checkpoint_anchor": file_metadata(real_world_input),
            "real_world_raw_dir": relative_to_workspace(real_world_raw_dir),
        },
        "summary": file_metadata(summary_path),
        "outputs": [
            {**file_metadata(path), "sha256": sha256(path)} for path in output_paths
        ],
    }
    manifest_path = output_stem.with_name(f"{output_stem.name}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return summary_path, manifest_path


def main() -> int:
    args = parse_args()
    logistic_run, xgboost_run, real_world_input = resolve_sources(args)
    print(f"ACS logistic source: {logistic_run / 'runs.csv'}")
    print(f"ACS XGBoost source: {xgboost_run / 'runs.csv'}")
    print(f"Real-world source anchor: {real_world_input}")

    runtime_frames = [
        load_acs_runtime(logistic_run, panel="acs_logistic"),
        load_acs_runtime(xgboost_run, panel="acs_xgboost"),
    ]
    real_world_runtime, raw_dir = load_real_world_runtime(real_world_input)
    runtime_frames.append(real_world_runtime)
    runtime = pd.concat(runtime_frames, ignore_index=True)
    summary = summarize_runtime(runtime, band=args.band)
    output_paths = create_figure(
        summary,
        output_stem=args.output_stem,
        formats=args.formats,
        dpi=args.dpi,
        band=args.band,
    )
    summary_path, manifest_path = write_outputs(
        summary,
        output_stem=args.output_stem,
        output_paths=output_paths,
        logistic_run=logistic_run,
        xgboost_run=xgboost_run,
        real_world_input=real_world_input,
        real_world_raw_dir=raw_dir,
        band=args.band,
    )
    for path in output_paths:
        print(f"Wrote {path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
