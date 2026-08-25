"""Compute the cross-experiment three-update EASE-FO allocation diagnostics.

The computation combines ACSIncome utility data with the Breast Cancer,
NHANES I, and Communities and Crime feature-attribution games. It reconstructs
the fixed-pilot optimization used by EASE-FO and evaluates the learned
allocation after three updates on disjoint held-out complement pairs.
Following the paper, the learned law is denoted ``q_hat`` and the
initialization law is denoted ``q_init``.

The comparison is restricted to the random-sampling support induced by
``boundary_order=1``: coalition sizes 2, ..., n-2.  All reported design risks
are the held-out, uncentered first-order allocation criterion, normalized by
the held-out residual oracle for the same fitted surrogate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
BENCHMARK_DIR = EXPERIMENTS_DIR / "real_world_benchmark"
PAPER_DIR = EXPERIMENTS_DIR.parent
EASESHAP_DIR = PAPER_DIR / "EaseSHAP"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(EASESHAP_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

from games import load_runtime  # noqa: E402
from easeshap.ease import (  # noqa: E402
    _EmpiricalDenseStats,
    _FeatureBuilder,
    _FullSemivalueTarget,
    _SizeStrata,
    _StratifiedLaw,
)


FEATURE_DATASETS = {
    "Breast Cancer": "breast_cancer",
    "NHANES I": "nhanesi",
    "Communities and Crime": "communities_crime",
}
DEFAULT_ACS_UTILITY_CSV = (
    SCRIPT_DIR / "data" / "allocation_diagnostics_acs_utilities.csv"
)
METRIC_COLUMNS = [
    "tv_q_hat_to_leverage",
    "tv_q_hat_to_q_init",
    "tv_q_hat_to_oracle",
    "risk_leverage_over_oracle",
    "risk_q_init_over_oracle",
    "risk_q_hat_over_oracle",
    "heldout_pair_residual_mse",
    "tv_update_1_to_update_2",
    "tv_update_2_to_q_hat",
]


@dataclass
class PairData:
    masks: np.ndarray
    utility: np.ndarray
    complement_utility: np.ndarray

    def __post_init__(self) -> None:
        self.masks = np.asarray(self.masks, dtype=bool)
        self.utility = np.asarray(self.utility, dtype=float).reshape(-1)
        self.complement_utility = np.asarray(
            self.complement_utility, dtype=float
        ).reshape(-1)
        if self.masks.ndim != 2:
            raise ValueError("masks must be a two-dimensional Boolean array")
        if not (
            len(self.masks) == len(self.utility) == len(self.complement_utility)
        ):
            raise ValueError("Pair arrays have inconsistent lengths")


def sample_unique_masks(
    n: int,
    size: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    target = min(int(count), math.comb(int(n), int(size)))
    found: set[int] = set()
    masks: list[np.ndarray] = []
    while len(masks) < target:
        idx = np.sort(rng.choice(n, size=size, replace=False))
        key = sum(1 << int(i) for i in idx)
        if n % 2 == 0 and size == n // 2:
            full = (1 << n) - 1
            key = min(key, full ^ key)
        if key in found:
            continue
        found.add(key)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        if n % 2 == 0 and size == n // 2:
            complement = ~mask
            complement_key = sum(1 << int(i) for i in np.flatnonzero(complement))
            if complement_key < sum(1 << int(i) for i in idx):
                mask = complement
        masks.append(mask)
    return np.asarray(masks, dtype=bool)


def sampled_pair_masks(n: int, pairs_per_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = [
        sample_unique_masks(n, size, pairs_per_size, rng)
        for size in range(1, n // 2 + 1)
    ]
    return np.concatenate(chunks, axis=0)


def stratified_pair_split(
    masks: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    strata = np.minimum(masks.sum(axis=1), masks.shape[1] - masks.sum(axis=1))
    train: list[int] = []
    test: list[int] = []
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        rng.shuffle(indices)
        n_test = max(1, int(round(test_fraction * len(indices))))
        if n_test >= len(indices):
            n_test = max(1, len(indices) - 1)
        test.extend(indices[:n_test].tolist())
        train.extend(indices[n_test:].tolist())
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def evaluate_masks(runtime, masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if (
        runtime.model is not None
        and runtime.baseline is not None
        and runtime.explicand is not None
    ):
        baseline = np.asarray(runtime.baseline, dtype=float).reshape(-1)
        explicand = np.asarray(runtime.explicand, dtype=float).reshape(-1)
        design = np.repeat(baseline[None, :], len(masks), axis=0)
        rows, columns = np.nonzero(masks)
        design[rows, columns] = explicand[columns]
        return np.asarray(runtime.model.predict(design), dtype=float).reshape(-1)
    values = runtime.value_array()
    if values is not None:
        powers = 1 << np.arange(masks.shape[1], dtype=np.int64)
        indices = masks.astype(np.int64) @ powers
        return np.asarray(values, dtype=float)[indices]
    return np.asarray([runtime.evaluate(mask) for mask in masks], dtype=float)


def redirect_shap_dataset_cache() -> None:
    """Keep optional SHAP dataset downloads in the ignored figures cache."""
    try:
        import shap.datasets as shap_datasets
    except ImportError:
        return
    cache_dir = SCRIPT_DIR / ".data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def local_cache(url: str, file_name: str | None = None) -> str:
        name = file_name or url.rsplit("/", 1)[-1]
        path = cache_dir / name
        if not path.is_file():
            urlretrieve(url, path)
        return str(path)

    shap_datasets.cache = local_cache


@dataclass
class EaseComponents:
    n: int
    strata: _SizeStrata
    feature_builder: _FeatureBuilder
    target: _FullSemivalueTarget
    law: _StratifiedLaw
    q_init: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default="0:9")
    parser.add_argument("--pairs-per-size", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--budget-per-player", type=int, default=200)
    parser.add_argument("--pilot-fraction", type=float, default=0.2)
    parser.add_argument("--pilot-design-updates", type=int, default=3)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--random-state", type=int, default=40)
    parser.add_argument("--acs-utility-csv", type=Path, default=DEFAULT_ACS_UTILITY_CSV)
    parser.add_argument("--skip-acs", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(FEATURE_DATASETS.values()),
        help="Run only selected feature dataset(s); may be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "three_update_allocation_active_support",
    )
    return parser.parse_args()


def parse_instances(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, stop = map(int, part.split(":", 1))
            values.extend(range(start, stop + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def mask_to_array(mask: int, n: int) -> np.ndarray:
    return np.asarray([(int(mask) >> bit) & 1 for bit in range(n)], dtype=bool)


def load_acs_pairs(path: Path, n: int = 50) -> PairData:
    frame = pd.read_csv(path, usecols=["mask", "utility"])
    utility = dict(zip(frame["mask"].astype(object).map(int), frame["utility"].astype(float)))
    full = (1 << n) - 1
    selected: list[int] = []
    for mask in utility:
        size = int(mask).bit_count()
        comp = full ^ int(mask)
        if comp not in utility or not 2 <= size <= n - 2:
            continue
        if size < n - size or (size == n - size and int(mask) < comp):
            selected.append(int(mask))
    selected.sort(key=lambda value: (value.bit_count(), value))
    masks = np.asarray([mask_to_array(mask, n) for mask in selected], dtype=bool)
    values = np.asarray([utility[mask] for mask in selected], dtype=float)
    complements = np.asarray([utility[full ^ mask] for mask in selected], dtype=float)
    return PairData(masks, values, complements)


def load_feature_pairs(
    *,
    dataset: str,
    instance_id: int,
    pairs_per_size: int,
    seed: int,
    random_state: int,
):
    runtime = load_runtime(
        dataset_name=dataset,
        instance_id=instance_id,
        random_state=random_state,
        tabular_game_mode="baseline_tree",
    )
    masks = sampled_pair_masks(runtime.n_players, pairs_per_size, seed)
    canonical_sizes = np.minimum(masks.sum(axis=1), runtime.n_players - masks.sum(axis=1))
    keep = canonical_sizes >= 2
    masks = masks[keep]
    return runtime, PairData(
        masks=masks,
        utility=evaluate_masks(runtime, masks),
        complement_utility=evaluate_masks(runtime, ~masks),
    )


def build_ease_components(n: int) -> EaseComponents:
    sampling_mask = np.ones(n + 1, dtype=bool)
    sampling_mask[[0, 1, n - 1, n]] = False
    strata = _SizeStrata(n, sampling_mask)
    feature_builder = _FeatureBuilder(
        n=n,
        surrogate_basis=1,
        include_nonlinear_size_terms=True,
    )
    boundary_x = np.empty((0, n), dtype=bool)
    boundary_context = strata.context_from_X(boundary_x)
    target = _FullSemivalueTarget(
        n=n,
        semivalue="shapley",
        semivalue_param=None,
        feature_builder=feature_builder,
        boundary_X=boundary_x,
        boundary_context=boundary_context,
    )
    law = _StratifiedLaw(n=n, strata=strata, is_paired=True)
    q_init = law.normalize_density(target.initial_design_factor(strata))
    return EaseComponents(n, strata, feature_builder, target, law, q_init)


def pilot_num_rows(n: int, budget_per_player: int, pilot_fraction: float) -> int:
    boundary_evaluations = 2 + 2 * n
    random_budget = max(0, int(budget_per_player) * n - boundary_evaluations)
    rows = int(round(float(pilot_fraction) * random_budget))
    return (rows // 2) * 2


def _canonical_sizes(masks: np.ndarray) -> np.ndarray:
    sizes = masks.sum(axis=1).astype(int)
    return np.minimum(sizes, masks.shape[1] - sizes)


def _pair_key(mask: np.ndarray) -> bytes:
    mask = np.asarray(mask, dtype=bool)
    comp = ~mask
    left = np.packbits(mask, bitorder="little").tobytes()
    right = np.packbits(comp, bitorder="little").tobytes()
    return min(left, right)


def heldout_indices_excluding_pilot(
    *,
    data: PairData,
    pilot_x: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    forbidden = {_pair_key(mask) for mask in pilot_x[0::2]}
    canonical = _canonical_sizes(data.masks)
    selected: list[int] = []
    for size in np.unique(canonical):
        all_idx = np.flatnonzero(canonical == size)
        candidates = np.asarray(
            [idx for idx in all_idx if _pair_key(data.masks[idx]) not in forbidden],
            dtype=int,
        )
        target = max(1, int(round(float(test_fraction) * len(all_idx))))
        if len(candidates) < target:
            raise ValueError(
                f"Only {len(candidates)} non-pilot pairs remain at canonical size {size}; "
                f"need {target} held-out pairs"
            )
        rng.shuffle(candidates)
        selected.extend(candidates[:target].tolist())
    return np.asarray(selected, dtype=int)


def sample_fixed_pilot(
    *,
    data: PairData,
    train_indices: np.ndarray,
    components: EaseComponents,
    num_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = components.n
    if num_rows <= 0 or num_rows % 2:
        raise ValueError("The paired pilot row count must be a positive even integer")
    canonical = _canonical_sizes(data.masks)
    pools = {
        int(size): train_indices[canonical[train_indices] == size]
        for size in np.unique(canonical[train_indices])
    }
    required = list(range(2, n // 2 + 1))
    missing = [size for size in required if size not in pools or len(pools[size]) == 0]
    if missing:
        raise ValueError(f"Missing active pilot pair strata: {missing}")

    ordered_mass = components.strata.counts * components.q_init
    pair_mass = np.asarray(
        [
            ordered_mass[size]
            if 2 * size == n
            else ordered_mass[size] + ordered_mass[n - size]
            for size in required
        ],
        dtype=float,
    )
    pair_mass /= pair_mass.sum()
    sampled_sizes = rng.choice(required, size=num_rows // 2, replace=True, p=pair_mass)

    x = np.empty((num_rows, n), dtype=bool)
    y = np.empty(num_rows, dtype=float)
    q0 = np.empty(num_rows, dtype=float)
    for pair_id, size in enumerate(sampled_sizes):
        source = int(rng.choice(pools[int(size)]))
        mask = data.masks[source]
        row = 2 * pair_id
        x[row] = mask
        x[row + 1] = ~mask
        y[row] = data.utility[source]
        y[row + 1] = data.complement_utility[source]
        q0[row] = components.q_init[int(mask.sum())]
        q0[row + 1] = components.q_init[int((~mask).sum())]
    return x, y, q0


def update_size_law(
    *,
    beta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    components: EaseComponents,
    floor: float = 1e-8,
    minimum_count: int = 2,
) -> np.ndarray:
    context = components.strata.context_from_X(x)
    ids = components.strata.ids_from_context(context)
    prediction = components.feature_builder.predict_from_rows(beta, x, context)
    residual = y - prediction
    pair_difference_sq = (residual[0::2] - residual[1::2]) ** 2
    left = ids[0::2]
    right = ids[1::2]
    rss = np.zeros(components.n + 1, dtype=float)
    counts = np.zeros(components.n + 1, dtype=np.int64)
    np.add.at(rss, left, pair_difference_sq)
    np.add.at(rss, right, pair_difference_sq)
    np.add.at(counts, left, 1)
    np.add.at(counts, right, 1)
    if counts.sum() == 0:
        second_moment = np.ones(components.n + 1, dtype=float)
    else:
        global_mse = max(float(rss.sum() / counts.sum()), 1e-12)
        second_moment = np.full(components.n + 1, global_mse, dtype=float)
        strong = counts >= int(minimum_count)
        second_moment[strong] = rss[strong] / counts[strong]
        second_moment = np.maximum(second_moment, 1e-12)
        second_moment = 0.5 * (second_moment + second_moment[::-1])
    factor = components.target.initial_design_factor(components.strata) * np.sqrt(second_moment)
    return components.law.mix_with_initial_mass(factor, components.q_init, floor)


def run_three_updates(
    *,
    x: np.ndarray,
    y: np.ndarray,
    q0: np.ndarray,
    components: EaseComponents,
    ridge_lambda: float,
    updates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    backend = _EmpiricalDenseStats(
        target=components.target,
        strata=components.strata,
        feature_builder=components.feature_builder,
        ridge_lambda=ridge_lambda,
        ridge_schedule="times_m",
        num_folds=2,
    )
    folds = np.repeat(rng.integers(2, size=len(y) // 2), 2)
    backend.append(x, y, q0, folds)
    fit = backend.fit_all()
    q_current = components.q_init.copy()
    q_history = [q_current.copy()]
    for update in range(updates):
        if update > 0:
            ids = components.strata.ids_from_context(components.strata.context_from_X(x))
            fit = backend.fit_candidate(x, y, q0, q_current[ids])
        q_current = update_size_law(
            beta=fit.beta,
            x=x,
            y=y,
            components=components,
        )
        q_history.append(q_current.copy())
    return fit.beta, q_history


def size_mass(q_density: np.ndarray, components: EaseComponents) -> np.ndarray:
    mass = components.strata.counts * np.asarray(q_density, dtype=float)
    mass = np.where(components.strata.sampling_mask, mass, 0.0)
    total = mass.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Allocation has no positive finite mass on the active support")
    return mass / total


def total_variation(left: np.ndarray, right: np.ndarray, active: np.ndarray) -> float:
    return float(0.5 * np.abs(left[active] - right[active]).sum())


def evaluate_repeat(
    *,
    data: PairData,
    dataset_label: str,
    instance_id: int,
    repeat: int,
    seed: int,
    test_fraction: float,
    budget_per_player: int,
    pilot_fraction: float,
    updates: int,
    ridge_lambda: float,
    feature_runtime=None,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    n = data.masks.shape[1]
    components = build_ease_components(n)
    split_seed = seed + 104729 * instance_id + 1009 * repeat
    rng = np.random.default_rng(split_seed + 37)
    rows = pilot_num_rows(n, budget_per_player, pilot_fraction)
    if feature_runtime is None:
        train, test = stratified_pair_split(data.masks, test_fraction, split_seed)
        x_pilot, y_pilot, q0 = sample_fixed_pilot(
            data=data,
            train_indices=train,
            components=components,
            num_rows=rows,
            rng=rng,
        )
    else:
        sampled = components.law.sample_batch(rng, rows, components.q_init)
        x_pilot = sampled[:, :n].astype(bool)
        q0 = sampled[:, n].astype(float)
        y_pilot = evaluate_masks(feature_runtime, x_pilot)
        test = heldout_indices_excluding_pilot(
            data=data,
            pilot_x=x_pilot,
            test_fraction=test_fraction,
            rng=rng,
        )
    beta, q_history = run_three_updates(
        x=x_pilot,
        y=y_pilot,
        q0=q0,
        components=components,
        ridge_lambda=ridge_lambda,
        updates=updates,
        rng=rng,
    )

    masks = data.masks[test]
    context = components.strata.context_from_X(masks)
    comp_context = components.strata.context_from_X(~masks)
    pred = components.feature_builder.predict_from_rows(beta, masks, context)
    pred_comp = components.feature_builder.predict_from_rows(beta, ~masks, comp_context)
    residual_difference = (data.utility[test] - pred) - (data.complement_utility[test] - pred_comp)
    canonical = _canonical_sizes(masks)
    rho_sq = components.target.true_stratum_weight(components.strata)
    moments = np.zeros(n + 1, dtype=float)
    counts = np.zeros(n + 1, dtype=np.int64)
    for size in range(2, n - 1):
        take = canonical == min(size, n - size)
        if not np.any(take):
            raise ValueError(f"No held-out pairs for active size {size}")
        moments[size] = rho_sq[size] * float(np.mean(residual_difference[take] ** 2))
        counts[size] = int(take.sum())

    oracle_factor = np.sqrt(np.maximum(moments, 0.0))
    q_oracle = components.law.normalize_density(oracle_factor)
    oracle = size_mass(q_oracle, components)
    q_init = size_mass(components.q_init, components)
    q_masses = [size_mass(q, components) for q in q_history]
    learned = q_masses[-1]
    active = components.strata.sampling_mask
    leverage = np.zeros(n + 1, dtype=float)
    leverage[active] = 1.0 / int(active.sum())

    design_terms = components.strata.counts**2 * moments

    def design_risk(mass: np.ndarray) -> float:
        if np.any(mass[active] <= 0):
            return float("inf")
        return float(np.sum(design_terms[active] / mass[active]))

    risk_oracle = design_risk(oracle)
    risk_leverage = design_risk(leverage)
    risk_q_init = design_risk(q_init)
    risk_q_hat = design_risk(learned)
    if not np.isfinite(risk_oracle) or risk_oracle <= 0:
        raise ValueError("Held-out oracle risk is not positive and finite")

    metrics: dict[str, float | int | str] = {
        "dataset": dataset_label,
        "instance_id": int(instance_id),
        "repeat": int(repeat),
        "n_players": int(n),
        "pilot_rows": int(rows),
        "pilot_pairs": int(rows // 2),
        "test_pairs": int(len(test)),
        "tv_q_hat_to_leverage": total_variation(learned, leverage, active),
        "tv_q_hat_to_q_init": total_variation(learned, q_init, active),
        "tv_q_hat_to_oracle": total_variation(learned, oracle, active),
        "risk_oracle": risk_oracle,
        "risk_leverage": risk_leverage,
        "risk_q_init": risk_q_init,
        "risk_q_hat": risk_q_hat,
        "risk_leverage_over_oracle": risk_leverage / risk_oracle,
        "risk_q_init_over_oracle": risk_q_init / risk_oracle,
        "risk_q_hat_over_oracle": risk_q_hat / risk_oracle,
        "heldout_pair_residual_mse": float(
            np.mean(
                [
                    np.mean(residual_difference[canonical == size] ** 2)
                    for size in np.unique(canonical)
                ]
            )
        ),
        "tv_update_1_to_update_2": total_variation(q_masses[1], q_masses[2], active),
        "tv_update_2_to_q_hat": total_variation(q_masses[2], q_masses[3], active),
    }
    profile_rows: list[dict[str, float | int | str]] = []
    allocations = {
        "leverage": leverage,
        "q_init": q_masses[0],
        "q_after_update_1": q_masses[1],
        "q_after_update_2": q_masses[2],
        "q_hat": q_masses[3],
        "heldout_oracle": oracle,
    }
    for size in np.flatnonzero(active):
        for allocation, mass in allocations.items():
            profile_rows.append(
                {
                    "dataset": dataset_label,
                    "instance_id": int(instance_id),
                    "repeat": int(repeat),
                    "size": int(size),
                    "allocation": allocation,
                    "size_mass": float(mass[size]),
                    "heldout_size_pairs": int(counts[size]),
                    "heldout_rho_weighted_pair_moment": float(moments[size]),
                }
            )
    return metrics, profile_rows


def summarize_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dataset, frame in raw.groupby("dataset", sort=False):
        if frame["instance_id"].nunique() > 1:
            units = frame.groupby("instance_id", as_index=False)[METRIC_COLUMNS].mean()
            uncertainty = "SE across instance means"
        else:
            units = frame[["repeat", *METRIC_COLUMNS]].copy()
            uncertainty = "Monte Carlo SE across repeated pilot/split draws"
        row: dict[str, float | int | str] = {
            "dataset": dataset,
            "n_instances": int(frame["instance_id"].nunique()),
            "repeats_per_instance": int(frame["repeat"].nunique()),
            "uncertainty_basis": uncertainty,
        }
        for metric in METRIC_COLUMNS:
            values = units[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_se"] = (
                float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def display_table(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary[["dataset", "n_instances", "repeats_per_instance"]].copy()
    selected = {
        "TV(q_hat, LeverageSHAP)": "tv_q_hat_to_leverage",
        "TV(q_hat, q_init)": "tv_q_hat_to_q_init",
    }
    for label, metric in selected.items():
        out[label] = [
            f"{mean:.3f} ({se:.3f})"
            for mean, se in zip(summary[f"{metric}_mean"], summary[f"{metric}_se"])
        ]
    return out


def main() -> None:
    args = parse_args()
    if args.pilot_design_updates != 3:
        raise ValueError("This diagnostic is intended to evaluate exactly three pilot design updates")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    instances = parse_instances(args.instances)
    if not instances and not args.skip_features:
        raise ValueError("No feature-attribution instances selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    metrics_rows: list[dict[str, float | int | str]] = []
    profile_rows: list[dict[str, float | int | str]] = []

    if not args.skip_acs:
        acs_data = load_acs_pairs(args.acs_utility_csv)
        for repeat in range(args.repeats):
            metrics, profiles = evaluate_repeat(
                data=acs_data,
                dataset_label="ACSIncome",
                instance_id=0,
                repeat=repeat,
                seed=args.seed,
                test_fraction=args.test_fraction,
                budget_per_player=args.budget_per_player,
                pilot_fraction=args.pilot_fraction,
                updates=args.pilot_design_updates,
                ridge_lambda=args.ridge_lambda,
            )
            metrics_rows.append(metrics)
            profile_rows.extend(profiles)
        print(f"completed ACSIncome: {len(acs_data.masks)} cached active complement pairs")

    if not args.skip_features:
        redirect_shap_dataset_cache()
        selected = set(args.dataset or FEATURE_DATASETS.values())
        for label, dataset in FEATURE_DATASETS.items():
            if dataset not in selected:
                continue
            for instance_id in instances:
                runtime, data = load_feature_pairs(
                    dataset=dataset,
                    instance_id=instance_id,
                    pairs_per_size=args.pairs_per_size,
                    seed=args.seed + 1009 * instance_id,
                    random_state=args.random_state,
                )
                for repeat in range(args.repeats):
                    metrics, profiles = evaluate_repeat(
                        data=data,
                        dataset_label=label,
                        instance_id=instance_id,
                        repeat=repeat,
                        seed=args.seed,
                        test_fraction=args.test_fraction,
                        budget_per_player=args.budget_per_player,
                        pilot_fraction=args.pilot_fraction,
                        updates=args.pilot_design_updates,
                        ridge_lambda=args.ridge_lambda,
                        feature_runtime=runtime,
                    )
                    metrics_rows.append(metrics)
                    profile_rows.extend(profiles)
                print(f"completed {label} instance {instance_id}: {len(data.masks)} active pairs")

    raw = pd.DataFrame(metrics_rows)
    profiles = pd.DataFrame(profile_rows)
    summary = summarize_metrics(raw)
    table = display_table(summary)
    raw.to_csv(args.output_dir / "raw_metrics.csv", index=False)
    profiles.to_csv(args.output_dir / "allocation_profiles.csv", index=False)
    summary.to_csv(args.output_dir / "summary_with_se.csv", index=False)
    table.to_csv(args.output_dir / "report_table.csv", index=False)
    metadata = {
        "active_sizes": "2,...,n-2 (boundary_order=1)",
        "allocation_reported": "q_hat after three fixed-pilot updates",
        "budget_per_player": args.budget_per_player,
        "pilot_fraction": args.pilot_fraction,
        "pilot_design_updates": args.pilot_design_updates,
        "ridge_lambda": args.ridge_lambda,
        "ridge_schedule": "times_m",
        "surrogate_basis": "intercept + player indicators + log1p(size) + (size/n)^2",
        "stage2_size_floor": 1e-8,
        "stage2_min_size_count": 2,
        "pairs_per_size": args.pairs_per_size,
        "test_fraction": args.test_fraction,
        "repeats": args.repeats,
        "feature_instances": instances,
        "acs_utility_csv": str(args.acs_utility_csv.resolve()),
        "risk_definition": (
            "held-out uncentered first-order allocation criterion; ratios normalize the "
            "held-out residual oracle to one"
        ),
        "elapsed_sec": time.perf_counter() - start,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\nThree-update active-support allocation diagnostics")
    print(table.to_string(index=False))
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
