"""
Draw customized AUCC panel figures for the SOU full benchmark.

The script reuses the AUCC data aggregation and styling helpers from
run_benchmark.py, then writes two standalone PNGs:
- eta = 0.5
- eta = 0.25 and eta = 0.75
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import run_benchmark as benchmark


DEFAULT_OUT_DIR = benchmark.OUT / "customized_plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=benchmark.OUT / "summary.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method ids to plot, in the requested order.",
    )
    return parser.parse_args()


def read_rows(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    with open(summary_path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary)

    written = []
    single_path = benchmark.write_aucc_panel_figure(
        rows=rows,
        requested_methods=args.methods,
        eta_labels=["0.25"],
        out_dir=args.out_dir,
        output_stem="aucc_eta_0p25",
        figsize=(7.0, 4.2),
        dpi=args.dpi,
        legend_position="right",
        font_scale=1.35,
        legend_font_scale=1.12,
        show_title=False,
    )
    if single_path is not None:
        written.append(single_path)

    pair_path = benchmark.write_aucc_panel_figure(
        rows=rows,
        requested_methods=args.methods,
        eta_labels=["0.5", "0.75"],
        out_dir=args.out_dir,
        output_stem="aucc_eta_0p5_0p75",
        figsize=(6.1, 3.65),
        dpi=args.dpi,
    )
    if pair_path is not None:
        written.append(pair_path)

    if len(written) != 2:
        raise RuntimeError("Expected to write two customized AUCC plots.")

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
