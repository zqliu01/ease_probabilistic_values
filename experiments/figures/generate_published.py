"""Regenerate every published paper figure and table from frozen CSVs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS = (
    "plot_sou_comparison_various_eta.py",
    "plot_sou_aucc.py",
    "plot_shapley_estimation_trajectories.py",
    "plot_runtime_breakdown_3panel.py",
    "plot_pilot_fraction_sensitivity.py",
    "write_allocation_diagnostics_table.py",
)


def main() -> int:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
    for script in SCRIPTS:
        command = [sys.executable, str(SCRIPT_DIR / script)]
        print("running", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=SCRIPT_DIR, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
