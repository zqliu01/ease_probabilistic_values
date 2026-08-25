"""
Structured Gaussian SOU Shapley experiment: EaseSHAP vs RegressionMSR.

Setup
-----
- Individual Shapley values only.
- gameSOUStructuredGaussianBitset with alpha in {sqrt(0.25), sqrt(0.5), sqrt(0.75)}.
- n = 40 players, n^2 random high-order unanimity terms with requested maximum
  order set by SOU_MAX_HIGH_ORDER_SIZE (default 20), clipped at n - 1, and
  exact SOU semivalue ground truth.

Algorithms
----------
- EaseSHAP:
    * no exact boundary handling (boundary_policy="none"),
    * no complement sampling,
    * degree <= 1 interaction surrogate,
    * no nonlinear |S| terms,
    * no size-player basis,
    * 2-fold crossfit readout.
- RegressionMSR_unbiased:
    * sampling with replacement,
    * no complement sampling,
    * 2-fold cross-fitting,
    * default special LeverageSHAP / Kernel Banzhaf surrogates enabled.
- RegressionMSR_unbiased_plain:
    * sampling with replacement,
    * no complement sampling,
    * 2-fold cross-fitting,
    * special LeverageSHAP / Kernel Banzhaf surrogates disabled.

Ground truth
------------
- Analytic SOU Shapley values from gameSOUStructuredGaussianBitset.get_semivalue().

Runs
----
- 5_000 NUE, tracked every 500 NUE, 10 random seeds.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Keep BLAS/OpenMP from oversubscribing shared servers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parents[1]
PACKAGE_ROOT = PAPER_DIR / "EaseSHAP"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from easeshap import runEstimator
from easeshap.utilityFuncs import gameSOUStructuredGaussianBitset


OUT = SCRIPT_DIR / "results" / "regmsr_unpaired_vs_easeshap"
GROUNDTRUTH_DIR = OUT / "groundtruth"
RUNS_DIR = OUT / "runs"
PLOTS_DIR = OUT / "plots"
GAME_DIR = OUT / "game"
PLOT_OUTPUT_FILENAME = "sou_vs_regressionmsr.png"

N = 40
GAME_SEED = 42
BASE_SEED = 2026
RUN_SEEDS = [BASE_SEED + i * 137 for i in range(10)]

ALPHAS = [0.25**0.5, 0.5**0.5, 0.75**0.5]
NUM_HIGH_ORDER = N ** 2
DEFAULT_MAX_HIGH_ORDER_SIZE = 20
try:
    MAX_HIGH_ORDER_SIZE = int(
        os.environ.get("SOU_MAX_HIGH_ORDER_SIZE", str(DEFAULT_MAX_HIGH_ORDER_SIZE))
    )
except ValueError as exc:
    raise ValueError("SOU_MAX_HIGH_ORDER_SIZE must be an integer.") from exc
if MAX_HIGH_ORDER_SIZE < 3:
    raise ValueError("SOU_MAX_HIGH_ORDER_SIZE must be at least 3.")
EFFECTIVE_MAX_HIGH_ORDER_SIZE = min(MAX_HIGH_ORDER_SIZE, N - 1)
SIGMA2 = None
RUN_NUE = 5_000
TRACK_STEP = 500
NUE_BUDGETS = np.arange(TRACK_STEP, RUN_NUE + 1, TRACK_STEP, dtype=int)

SEMIVALUE = "shapley"
SEMIVALUE_PARAM = None

GAME_TAG = f"kmax{MAX_HIGH_ORDER_SIZE}"
GROUNDTRUTH_TAG = f"exact_sou_{GAME_TAG}"

RUN_EASESHAP_KWARGS = {
    "pilot_design_updates": 3,
    "exact_boundary_handling": False,
    "boundary_policy": "none",
    "boundary_order": 1,
    "use_complement_sampling": False,
    "surrogate_basis": 1,
    "include_nonlinear_size_terms": False,
    "num_folds": 2,
    "surrogate_readout_mode": "crossfit",
}

REGMSR_KWARGS = {
    "sampling_with_replacement": True,
    "paired_sampling": False,
    "num_folds": 2,
}

REGMSR_PLAIN_KWARGS = {
    "sampling_with_replacement": True,
    "paired_sampling": False,
    "num_folds": 2,
    "use_special_surrogates": False,
}

ALGORITHM_SPECS = [
    {
        "name": "EaseSHAP",
        "backend": "EaseSHAP",
        "estimator_kwargs": RUN_EASESHAP_KWARGS,
    },
    {
        "name": "RegressionMSR_unbiased",
        "backend": "RegressionMSR_unbiased",
        "estimator_kwargs": REGMSR_KWARGS,
    },
    {
        "name": "RegressionMSR_unbiased_plain",
        "backend": "RegressionMSR_unbiased",
        "estimator_kwargs": REGMSR_PLAIN_KWARGS,
    },
]

ETA_LABELS = ("0.25", "0.5", "0.75")
X_AXIS_LABEL = "Avg. Utility Evals per Player"
Y_AXIS_LABEL = "relative squared error"
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11
PANEL_WIDTH = 3.5
FIG_HEIGHT = 3.35

PLOT_LABELS = {
    "EaseSHAP": "EASE",
    "EaseSHAP_boundary_size_player": "EASE",
    "EaseSHAP_order2": "EASE",
    "RegressionMSR_unbiased": "RegressionMSR",
    "OFA_fixed": "OFA",
    "PolySHAP_regression": "PolySHAP",
}

PLOT_COLORS = {
    "EaseSHAP": "tab:blue",
    "EaseSHAP_boundary_size_player": "tab:blue",
    "EaseSHAP_order2": "tab:blue",
    "RegressionMSR_unbiased": "tab:orange",
    "OFA_fixed": "tab:orange",
    "PolySHAP_regression": "tab:orange",
}

DROP_PLOT_ALGORITHMS = {"RegressionMSR_unbiased_plain"}


def alpha_label(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace(".", "p")


def ensure_dirs() -> None:
    for path in (GROUNDTRUTH_DIR, RUNS_DIR, PLOTS_DIR, GAME_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUT / "run.log", mode="a"),
        ],
        force=True,
    )
    return logging.getLogger("regmsr_unpaired_vs_easeshap")


def game_args(alpha: float) -> dict:
    return {
        "num_player": N,
        "alpha": float(alpha),
        "num_high_order": NUM_HIGH_ORDER,
        "max_high_order_size": MAX_HIGH_ORDER_SIZE,
        "sigma2": SIGMA2,
        "path": str(GAME_DIR),
        "seed": GAME_SEED,
    }


@contextlib.contextmanager
def silence_worker_output():
    """Suppress estimator progress bars and prints inside parallel workers."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)


def run_estimator(
    *,
    backend: str,
    estimator_seed: int,
    nue_avg: int,
    nue_track_avg: int,
    alpha: float,
    estimator_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    estimator = runEstimator(
        estimator=backend,
        n_process=1,
        semivalue=SEMIVALUE,
        semivalue_param=SEMIVALUE_PARAM,
        game_func=gameSOUStructuredGaussianBitset,
        game_args=game_args(alpha),
        num_player=N,
        nue_avg=int(nue_avg),
        nue_per_proc=min(int(nue_avg), 20_000),
        nue_track_avg=int(nue_track_avg),
        estimator_seed=int(estimator_seed),
        **estimator_kwargs,
    )
    values_final, values_traj = estimator.run()
    return np.asarray(values_final, dtype=np.float64), np.asarray(values_traj, dtype=np.float64)


def groundtruth_path(alpha: float) -> Path:
    return GROUNDTRUTH_DIR / f"{alpha_label(alpha)}_{GROUNDTRUTH_TAG}.npy"


def run_path(alpha: float, run_idx: int) -> Path:
    return RUNS_DIR / f"{alpha_label(alpha)}_{GAME_TAG}_run{run_idx:02d}.pkl"


def groundtruth_worker(alpha: float, force: bool):
    path = groundtruth_path(alpha)
    if path.exists() and not force:
        return alpha, "cached", 0.0

    start = time.time()
    game = gameSOUStructuredGaussianBitset(**game_args(alpha))
    values_exact = game.get_semivalue(semivalue=SEMIVALUE, semivalue_param=SEMIVALUE_PARAM)
    np.save(path, np.asarray(values_exact, dtype=np.float64))
    return alpha, "computed", time.time() - start


def run_worker(
    alpha: float,
    run_idx: int,
    seed: int,
    force: bool,
):
    path = run_path(alpha, run_idx)
    if path.exists() and not force:
        return alpha, run_idx, "cached", 0.0

    if not groundtruth_path(alpha).exists():
        raise FileNotFoundError(f"Missing ground truth: {groundtruth_path(alpha)}")

    results = {}
    start = time.time()
    with silence_worker_output():
        for spec in ALGORITHM_SPECS:
            alg_start = time.time()
            values_final, values_traj = run_estimator(
                backend=spec["backend"],
                estimator_seed=seed,
                nue_avg=RUN_NUE,
                nue_track_avg=TRACK_STEP,
                alpha=alpha,
                estimator_kwargs=spec["estimator_kwargs"],
            )
            results[spec["name"]] = {
                "values_final": values_final,
                "values_traj": values_traj,
                "elapsed": time.time() - alg_start,
            }

    payload = {
        "alpha": float(alpha),
        "run_idx": int(run_idx),
        "seed": int(seed),
        "nue_budgets": NUE_BUDGETS.copy(),
        "algorithms": results,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return alpha, run_idx, "computed", time.time() - start


def run_parallel(tasks, worker, n_proc: int, logger: logging.Logger, label: str) -> None:
    if not tasks:
        logger.info("%s: nothing to do.", label)
        return
    if n_proc <= 1:
        for args in tasks:
            logger.info("%s: %s", label, worker(*args))
        return
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futures = {pool.submit(worker, *args): args for args in tasks}
        for fut in as_completed(futures):
            logger.info("%s: %s", label, fut.result())


def compute_error_rows() -> list[dict]:
    rows = []
    for alpha in ALPHAS:
        truth_path = groundtruth_path(alpha)
        if not truth_path.exists():
            continue
        truth = np.load(truth_path)
        denom = float(np.dot(truth, truth))
        if denom <= 0.0:
            denom = 1.0
        for run_idx in range(len(RUN_SEEDS)):
            path = run_path(alpha, run_idx)
            if not path.exists():
                continue
            with open(path, "rb") as f:
                payload = pickle.load(f)
            for algorithm, result in payload["algorithms"].items():
                traj = np.asarray(result["values_traj"], dtype=np.float64)
                for idx, nue in enumerate(payload["nue_budgets"]):
                    err = traj[idx] - truth
                    sq_error = float(np.dot(err, err))
                    rows.append(
                        {
                            "alpha": f"{alpha:g}",
                            "run_idx": str(run_idx),
                            "algorithm": algorithm,
                            "nue": str(int(nue)),
                            "sq_error": f"{sq_error:.17g}",
                            "rel_sq_error": f"{sq_error / denom:.17g}",
                            "rmse": f"{float(np.sqrt(np.mean(err * err))):.17g}",
                            "elapsed_sec": f"{float(result['elapsed']):.6g}",
                        }
                    )
    return rows


def write_summary(logger: logging.Logger) -> Path:
    rows = compute_error_rows()
    path = OUT / "summary.csv"
    fieldnames = [
        "alpha",
        "run_idx",
        "algorithm",
        "nue",
        "sq_error",
        "rel_sq_error",
        "rmse",
        "elapsed_sec",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("summary: wrote %s rows to %s", len(rows), path)
    return path


def aggregate_plot_rows(
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


def plot_alpha_panel(ax: plt.Axes, rows: list[dict[str, str]], alpha: float, eta_label: str) -> None:
    plotted = False
    for algorithm in [spec["name"] for spec in ALGORITHM_SPECS]:
        if algorithm in DROP_PLOT_ALGORITHMS:
            continue

        xs, means, stds = aggregate_plot_rows(rows, alpha, algorithm)
        if xs.size == 0:
            continue

        color = PLOT_COLORS.get(algorithm)
        lower = np.maximum(means - stds, np.finfo(float).tiny)
        ax.plot(xs, means, marker="o", label=PLOT_LABELS.get(algorithm, algorithm), color=color)
        ax.fill_between(xs, lower, means + stds, alpha=0.18, color=color)
        plotted = True

    if not plotted:
        raise RuntimeError(f"No rows found for alpha={alpha:g} in {OUT / 'summary.csv'}")

    ax.set_title(rf"$\eta = {eta_label}$", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def plot_summary(logger: logging.Logger) -> None:
    rows = compute_error_rows()
    if not rows:
        logger.info("plots: no rows available.")
        return

    fig, axes = plt.subplots(
        1,
        len(ALPHAS),
        figsize=(PANEL_WIDTH * len(ALPHAS), FIG_HEIGHT),
        sharey=True,
    )
    if len(ALPHAS) == 1:
        axes = [axes]

    for ax, alpha, eta_label in zip(axes, ALPHAS, ETA_LABELS):
        plot_alpha_panel(ax, rows, alpha, eta_label)

    axes[0].set_ylabel(Y_AXIS_LABEL, fontsize=LABEL_FONTSIZE)
    axes[-1].legend(loc="best", frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()

    path = PLOTS_DIR / PLOT_OUTPUT_FILENAME
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("plots: wrote %s", path)


def write_config() -> None:
    config = {
        "n": N,
        "game_seed": GAME_SEED,
        "base_seed": BASE_SEED,
        "run_seeds": RUN_SEEDS,
        "alphas": ALPHAS,
        "num_high_order": NUM_HIGH_ORDER,
        "max_high_order_size": MAX_HIGH_ORDER_SIZE,
        "effective_max_high_order_size": EFFECTIVE_MAX_HIGH_ORDER_SIZE,
        "sigma2": SIGMA2,
        "run_nue": RUN_NUE,
        "track_step": TRACK_STEP,
        "nue_budgets": NUE_BUDGETS.tolist(),
        "semivalue": SEMIVALUE,
        "semivalue_param": SEMIVALUE_PARAM,
        "groundtruth_tag": GROUNDTRUTH_TAG,
        "groundtruth_backend": "analytic_sou",
        "game_backend": "gameSOUStructuredGaussianBitset",
        "algorithm_specs": ALGORITHM_SPECS,
    }
    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["all", "groundtruth", "runs", "summary", "plots"],
        default="all",
        help="Which phase to run.",
    )
    parser.add_argument("--n-proc", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Recompute existing groundtruth/run files.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting in phase=all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    logger = setup_logging()
    write_config()

    logger.info("output: %s", OUT)
    logger.info(
        "game: n=%d, alphas=%s, num_high_order=%d, "
        "max_high_order_size=%d, effective_max_high_order_size=%d, sigma2=%s",
        N,
        ALPHAS,
        NUM_HIGH_ORDER,
        MAX_HIGH_ORDER_SIZE,
        EFFECTIVE_MAX_HIGH_ORDER_SIZE,
        SIGMA2,
    )
    logger.info("budgets: run_nue=%d, track_step=%d", RUN_NUE, TRACK_STEP)
    logger.info("n_proc=%d, force=%s", args.n_proc, args.force)

    if args.phase in {"all", "groundtruth"}:
        tasks = [(alpha, args.force) for alpha in ALPHAS]
        run_parallel(tasks, groundtruth_worker, args.n_proc, logger, "groundtruth")

    if args.phase in {"all", "runs"}:
        missing_truth = [alpha for alpha in ALPHAS if not groundtruth_path(alpha).exists()]
        if missing_truth:
            raise RuntimeError(f"Missing ground truth for alphas: {missing_truth}")
        tasks = [
            (alpha, run_idx, seed, args.force)
            for alpha in ALPHAS
            for run_idx, seed in enumerate(RUN_SEEDS)
        ]
        run_parallel(tasks, run_worker, args.n_proc, logger, "runs")

    if args.phase in {"all", "summary"}:
        write_summary(logger)

    if args.phase in {"all", "plots"} and not args.no_plots:
        plot_summary(logger)


if __name__ == "__main__":
    main()
