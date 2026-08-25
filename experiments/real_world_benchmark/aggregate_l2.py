"""Aggregate raw benchmark outputs into L2-error CSV summaries."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    add_ease_switch_fractions_arg,
    add_results_config_args,
    config_raw_dir,
    resolve_config_dir_from_args,
    validate_results_config_args,
)


def _std_sem(values: np.ndarray) -> tuple[float, float]:
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if values.size else math.nan
    return std, sem


def _reference_norms(payload: dict[str, Any]) -> tuple[float, float]:
    reference_phi = payload.get("reference_phi")
    if reference_phi is None:
        return math.nan, math.nan
    reference_phi = np.asarray(reference_phi, dtype=np.float64)
    denom_sq = float(np.dot(reference_phi, reference_phi))
    if denom_sq <= 0.0:
        denom_sq = 1.0
    return float(math.sqrt(denom_sq)), denom_sq


def _metadata_value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key, "")
    return "" if value is None else value


def load_result_rows(raw_dir: Path, *, config_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*/*/*/*.pkl")):
        with path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("status") != "ok":
            continue
        task = payload["task"]
        errors = np.asarray(payload["l2_errors"], dtype=np.float64)
        reference_l2_norm, reference_sq_norm = _reference_norms(payload)
        requested = list(payload["requested_checkpoints"])
        actual = list(payload["checkpoint_budgets"])
        for idx, error in enumerate(errors):
            sq_error = float(error * error)
            rows.append(
                {
                    # The selected result folder is the source of truth. This
                    # keeps derived CSVs coherent when a final configuration
                    # combines payloads produced by separate runs.
                    "config_name": config_name,
                    "budget_per_player": _metadata_value(payload, "budget_per_player"),
                    "ease_switch_fraction": _metadata_value(payload, "ease_switch_fraction"),
                    "num_checkpoints": _metadata_value(payload, "num_checkpoints"),
                    "ease_switch_budget": _metadata_value(payload, "ease_switch_budget"),
                    "ease_pilot_nue": _metadata_value(payload, "ease_pilot_nue"),
                    "tabular_game_mode": _metadata_value(payload, "tabular_game_mode"),
                    "dataset": task["dataset"],
                    "semivalue": task["semivalue"],
                    "method": task["method"],
                    "instance_id": int(task["instance_id"]),
                    "run_idx": int(task["run_idx"]),
                    "checkpoint_idx": idx + 1,
                    "requested_budget": int(requested[idx]),
                    "actual_budget": int(actual[idx]),
                    "l2_error": float(error),
                    "sq_error": sq_error,
                    "rel_l2_error": (
                        float(error / reference_l2_norm) if math.isfinite(reference_l2_norm) else math.nan
                    ),
                    "rel_sq_error": (
                        float(sq_error / reference_sq_norm) if math.isfinite(reference_sq_norm) else math.nan
                    ),
                    "reference_l2_norm": reference_l2_norm,
                    "elapsed_sec": float(payload.get("elapsed_sec", math.nan)),
                    "path": str(path),
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["config_name"],
            row["budget_per_player"],
            row["ease_switch_fraction"],
            row["num_checkpoints"],
            row["ease_switch_budget"],
            str(row["ease_pilot_nue"]),
            row["tabular_game_mode"],
            row["dataset"],
            row["semivalue"],
            row["method"],
            row["checkpoint_idx"],
            row["actual_budget"],
        )
        grouped[key].append(row)

    out = []
    for key, items in sorted(grouped.items()):
        (
            config_name,
            budget_per_player,
            ease_switch_fraction,
            num_checkpoints,
            ease_switch_budget,
            _ease_pilot_nue_sort_key,
            tabular_game_mode,
            dataset,
            semivalue,
            method,
            checkpoint_idx,
            actual_budget,
        ) = key
        requested = np.array([item["requested_budget"] for item in items], dtype=np.float64)
        l2_values = np.array([item["l2_error"] for item in items], dtype=np.float64)
        sq_values = np.array([item["sq_error"] for item in items], dtype=np.float64)
        rel_l2_values = np.array([item["rel_l2_error"] for item in items], dtype=np.float64)
        rel_sq_values = np.array([item["rel_sq_error"] for item in items], dtype=np.float64)
        reference_l2_norms = np.array([item["reference_l2_norm"] for item in items], dtype=np.float64)
        std_l2, sem_l2 = _std_sem(l2_values)
        std_sq, sem_sq = _std_sem(sq_values)
        std_rel_l2, sem_rel_l2 = _std_sem(rel_l2_values)
        std_rel_sq, sem_rel_sq = _std_sem(rel_sq_values)
        out.append(
            {
                "config_name": config_name,
                "budget_per_player": budget_per_player,
                "ease_switch_fraction": ease_switch_fraction,
                "num_checkpoints": num_checkpoints,
                "ease_switch_budget": ease_switch_budget,
                "ease_pilot_nue": items[0]["ease_pilot_nue"],
                "tabular_game_mode": tabular_game_mode,
                "dataset": dataset,
                "semivalue": semivalue,
                "method": method,
                "checkpoint_idx": int(checkpoint_idx),
                "requested_budget": int(round(float(np.mean(requested)))),
                "actual_budget": int(actual_budget),
                "mean_l2": float(np.mean(l2_values)),
                "std_l2": std_l2,
                "sem_l2": sem_l2,
                "mean_sq_error": float(np.mean(sq_values)),
                "std_sq_error": std_sq,
                "sem_sq_error": sem_sq,
                "mean_rel_l2": float(np.mean(rel_l2_values)),
                "std_rel_l2": std_rel_l2,
                "sem_rel_l2": sem_rel_l2,
                "mean_rel_sq": float(np.mean(rel_sq_values)),
                "std_rel_sq": std_rel_sq,
                "sem_rel_sq": sem_rel_sq,
                "mean_reference_l2_norm": float(np.mean(reference_l2_norms)),
                "count": int(len(l2_values)),
            }
        )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_results_config_args(parser)
    add_ease_switch_fractions_arg(parser)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--summary-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_results_config_args(args)
    config_dir = resolve_config_dir_from_args(args)
    raw_dir = args.raw_dir or config_raw_dir(config_dir)
    summary_dir = args.summary_dir or config_dir
    rows = load_result_rows(raw_dir, config_name=config_dir.name)
    summary = summarize_rows(rows)

    write_rows(summary_dir / "l2_by_checkpoint.csv", rows)
    write_rows(summary_dir / "l2_summary.csv", summary)
    print(f"config_dir={config_dir}")
    print(f"raw_dir={raw_dir}")
    print(f"summary_dir={summary_dir}")
    print(f"loaded_results={len(rows)}")
    print(f"summary_rows={len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
