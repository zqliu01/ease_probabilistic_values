"""Build the four paper AUCC figures from frozen published summaries."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent / "SOU_full_benchmark"
PUBLISHED_ROOT = BENCHMARK_DIR / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "published"

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import plot_figures as sou_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def eta_label_from_setting(setting: str) -> tuple[str, str]:
    try:
        alpha_part, semivalue = setting.split("__", 1)
        alpha = float(alpha_part.removeprefix("alpha_").replace("p", "."))
    except ValueError as exc:
        raise ValueError(f"Invalid AUCC setting label: {setting!r}") from exc

    eta = alpha**2
    for expected in (0.25, 0.5, 0.75):
        if math.isclose(eta, expected, rel_tol=0.0, abs_tol=1e-5):
            return f"{expected:g}", semivalue
    raise ValueError(f"Unexpected alpha in AUCC setting label: {setting!r}")


def read_published_tables(n_players: int):
    result_dir = PUBLISHED_ROOT / f"published_n{n_players}"
    available_methods, means = sou_plots.read_matrix(result_dir / "aucc_mean.csv")
    _std_methods, stds = sou_plots.read_matrix(result_dir / "aucc_std.csv")
    if not means:
        raise ValueError(f"No AUCC values found in {result_dir}")

    methods = sou_plots.select_aucc_methods(available_methods, None)
    values: dict[tuple[str, str, str], tuple[float, float]] = {}
    for setting, means_by_method in means.items():
        eta_label, semivalue = eta_label_from_setting(setting)
        stds_by_method = stds.get(setting, {})
        for method in methods:
            if method in means_by_method:
                values[(eta_label, semivalue, method)] = (
                    float(means_by_method[method]),
                    float(stds_by_method.get(method, 0.0)),
                )
    return methods, values


def write_figure(
    *,
    n_players: int,
    eta_labels: list[str],
    output_dir: Path,
    output_stem: str,
    figsize: tuple[float, float],
    dpi: int,
    legend_position: str = "bottom",
    font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    show_title: bool = True,
) -> Path:
    methods, values = read_published_tables(n_players)
    path = sou_plots.write_aucc_panel_from_table_values(
        by_setting_method=values,
        methods=methods,
        eta_labels=eta_labels,
        out_dir=output_dir,
        output_stem=output_stem,
        figsize=figsize,
        dpi=dpi,
        legend_position=legend_position,
        font_scale=font_scale,
        legend_font_scale=legend_font_scale,
        show_title=show_title,
    )
    if path is None:
        raise RuntimeError(f"Could not generate {output_stem}")
    return path


def main() -> int:
    args = parse_args()
    specs = (
        {
            "n_players": 40,
            "eta_labels": ["0.25"],
            "output_stem": "aucc_eta_0p25_n40",
            "figsize": (7.0, 4.2),
            "legend_position": "right",
            "font_scale": 1.35,
            "legend_font_scale": 1.12,
            "show_title": False,
        },
        {
            "n_players": 40,
            "eta_labels": ["0.5", "0.75"],
            "output_stem": "aucc_eta_0p5_0p75_n40",
            "figsize": (6.1, 3.65),
        },
        {
            "n_players": 80,
            "eta_labels": ["0.25", "0.5", "0.75"],
            "output_stem": "aucc_eta_all_n80",
            "figsize": (8.8, 3.65),
        },
        {
            "n_players": 160,
            "eta_labels": ["0.25", "0.5", "0.75"],
            "output_stem": "aucc_eta_all_n160",
            "figsize": (8.8, 3.65),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        path = write_figure(output_dir=args.output_dir, dpi=args.dpi, **spec)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
