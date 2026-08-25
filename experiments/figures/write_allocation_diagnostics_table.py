"""Write the paper allocation-diagnostics TeX table from its published CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR / "data" / "allocation_tv_diagnostics.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "published"
LEVERAGE_COLUMN = "TV(q_hat, LeverageSHAP)"
INITIAL_COLUMN = "TV(q_hat, q_init)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def main() -> int:
    args = parse_args()
    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", LEVERAGE_COLUMN, INITIAL_COLUMN}
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{args.input} is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No diagnostic rows found in {args.input}")

    latex_lines = [
        r"% Values are mean (standard error).",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Task & $\mathrm{TV}(\hat q,\mathrm{LeverageSHAP})$ & "
        r"$\mathrm{TV}(\hat q,q^{\mathrm{init}})$ \\",
        r"\midrule",
    ]
    for row in rows:
        dataset = str(row["dataset"])
        leverage = str(row[LEVERAGE_COLUMN])
        initial = str(row[INITIAL_COLUMN])
        latex_lines.append(
            f"{latex_escape(dataset)} & {leverage} & {initial} " + r"\\"
        )
    latex_lines.extend((r"\bottomrule", r"\end{tabular}"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latex_path = args.output_dir / "allocation_tv_diagnostics.tex"
    latex_path.write_text("\n".join(latex_lines) + "\n")
    print(f"wrote {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
