"""Data preparation for the ACSIncome state-source valuation experiment.

The public pipeline keeps downloaded ACS PUMS files under ``data/raw`` and the
deterministic experiment split under ``data/processed``.  The processed split
contains unencoded ACSIncome rows, so different encoders can share the same
train/evaluation split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome, state_list
from folktables.load_acs import _STATE_CODES


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR_NAME = "raw"
PROCESSED_DIR_NAME = "processed"
HORIZON = "1-Year"
SURVEY = "person"
LABEL_COLUMN = "label"
DOWNLOAD_ATTEMPTS = 3

US_STATES = [state for state in state_list if state != "PR"]
LOAD_COLUMNS = ACSIncome.features + [ACSIncome.target, "PWGTP"]


@dataclass(frozen=True)
class ACSDataDirs:
    root_dir: Path
    raw_dir: Path
    processed_dir: Path


@dataclass(frozen=True)
class ACSProcessedSplit:
    train_frames: dict[str, pd.DataFrame]
    train_labels: dict[str, np.ndarray]
    eval_x: pd.DataFrame
    eval_y: np.ndarray
    label_rates: dict[str, float]
    manifest: dict[str, Any]
    split_dir: Path
    manifest_path: Path
    data_dirs: ACSDataDirs


def resolve_data_dirs(data_dir: Path | str, survey_year: str) -> ACSDataDirs:
    """Resolve public data-root layout while accepting legacy raw-cache paths."""

    root_dir = Path(data_dir)
    if root_dir.resolve() == DEFAULT_DATA_DIR.resolve():
        raw_dir = root_dir / RAW_DIR_NAME
    else:
        legacy_survey_dir = root_dir / str(survey_year) / HORIZON
        raw_dir = root_dir if legacy_survey_dir.exists() else root_dir / RAW_DIR_NAME
    return ACSDataDirs(
        root_dir=root_dir,
        raw_dir=raw_dir,
        processed_dir=root_dir / PROCESSED_DIR_NAME,
    )


def state_file_path(raw_dir: Path, survey_year: str, state: str) -> Path:
    state_code = _STATE_CODES[state]
    return Path(raw_dir) / str(survey_year) / HORIZON / f"psam_p{state_code}.csv"


def state_zip_path(raw_dir: Path, survey_year: str, state: str) -> Path:
    return Path(raw_dir) / str(survey_year) / HORIZON / f"csv_p{state.lower()}.zip"


def make_data_source(raw_dir: Path, survey_year: str) -> ACSDataSource:
    return ACSDataSource(
        survey_year=str(survey_year),
        horizon=HORIZON,
        survey=SURVEY,
        root_dir=str(raw_dir),
    )


def download_state(raw_dir: Path, survey_year: str, state: str) -> pd.DataFrame:
    """Download one state, removing partial Folktables zip files before retrying."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = state_zip_path(raw_dir, survey_year, state)
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return make_data_source(raw_dir, survey_year).get_data(
                states=[state],
                download=True,
            )
        except Exception as exc:
            last_error = exc
            if zip_path.exists():
                zip_path.unlink()
                print(f"Deleted incomplete download {zip_path}")
            if attempt < DOWNLOAD_ATTEMPTS:
                print(
                    f"Retrying {state} download "
                    f"({attempt + 1}/{DOWNLOAD_ATTEMPTS})..."
                )
    raise RuntimeError(
        f"Failed to download ACS data for {state} after {DOWNLOAD_ATTEMPTS} attempts."
    ) from last_error


def load_state(
    *,
    raw_dir: Path,
    survey_year: str,
    state: str,
    download: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load one state's ACSIncome frame from cache, or download when allowed."""

    file_path = state_file_path(raw_dir, survey_year, state)
    if file_path.exists():
        raw = pd.read_csv(
            file_path,
            usecols=LOAD_COLUMNS,
            dtype={ACSIncome.target: np.float64},
        )
    else:
        if not download:
            raise FileNotFoundError(
                f"Missing raw ACS file {file_path}. Run "
                "`python3 acs_data.py prepare --download` to populate data/raw, "
                "or pass --data-dir pointing at an existing Folktables cache."
            )
        raw = download_state(raw_dir, survey_year, state)
    x, y, _ = ACSIncome.df_to_pandas(raw)
    return x.reset_index(drop=True), np.asarray(y, dtype=int).ravel()


def sample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if n < k:
        raise ValueError(f"Need at least {k} records, found {n}.")
    return rng.permutation(n)[:k]


def split_id(
    *,
    survey_year: str,
    target_state: str,
    train_size: int,
    eval_size: int,
    seed: int,
) -> str:
    return (
        f"acsincome_{survey_year}_target{target_state}"
        f"_train{train_size}_eval{eval_size}_seed{seed}"
    )


def processed_split_dir(
    *,
    processed_dir: Path,
    survey_year: str,
    target_state: str,
    train_size: int,
    eval_size: int,
    seed: int,
) -> Path:
    return processed_dir / split_id(
        survey_year=survey_year,
        target_state=target_state,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )


def processed_split_exists(split_dir: Path) -> bool:
    train_dir = split_dir / "train"
    return (
        (split_dir / "sample_manifest.json").exists()
        and (split_dir / "eval.csv").exists()
        and all((train_dir / f"{state}.csv").exists() for state in US_STATES)
    )


def _with_label(frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    result[LABEL_COLUMN] = np.asarray(labels, dtype=int)
    return result


def _split_features_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    labels = frame[LABEL_COLUMN].to_numpy(dtype=int)
    features = frame.drop(columns=[LABEL_COLUMN]).reset_index(drop=True)
    return features, labels


def load_processed_split(
    *,
    split_dir: Path,
    data_dirs: ACSDataDirs,
) -> ACSProcessedSplit:
    train_frames: dict[str, pd.DataFrame] = {}
    train_labels: dict[str, np.ndarray] = {}
    train_dir = split_dir / "train"
    for state in US_STATES:
        features, labels = _split_features_labels(pd.read_csv(train_dir / f"{state}.csv"))
        train_frames[state] = features
        train_labels[state] = labels

    eval_x, eval_y = _split_features_labels(pd.read_csv(split_dir / "eval.csv"))
    label_rates = {
        state: float(np.mean(train_labels[state]))
        for state in US_STATES
    }
    manifest_path = split_dir / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return ACSProcessedSplit(
        train_frames=train_frames,
        train_labels=train_labels,
        eval_x=eval_x,
        eval_y=eval_y,
        label_rates=label_rates,
        manifest=manifest,
        split_dir=split_dir,
        manifest_path=manifest_path,
        data_dirs=data_dirs,
    )


def prepare_processed_split(
    *,
    survey_year: str,
    target_state: str,
    train_size: int,
    eval_size: int,
    seed: int,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    download: bool = False,
    force: bool = False,
) -> ACSProcessedSplit:
    data_dirs = resolve_data_dirs(data_dir, survey_year)
    split_dir = processed_split_dir(
        processed_dir=data_dirs.processed_dir,
        survey_year=survey_year,
        target_state=target_state,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )
    if processed_split_exists(split_dir) and not force:
        return load_processed_split(split_dir=split_dir, data_dirs=data_dirs)

    split_dir.mkdir(parents=True, exist_ok=True)
    train_dir = split_dir / "train"
    train_dir.mkdir(exist_ok=True)

    target_x, target_y = load_state(
        raw_dir=data_dirs.raw_dir,
        survey_year=survey_year,
        state=target_state,
        download=download,
    )
    target_rng = np.random.default_rng(seed)
    target_order = target_rng.permutation(len(target_y))
    eval_idx = target_order[:eval_size]
    target_train_pool = target_order[eval_size:]
    if len(target_train_pool) < train_size:
        raise ValueError(
            f"Not enough {target_state} records for disjoint train/eval samples."
        )

    eval_x = target_x.iloc[eval_idx].reset_index(drop=True)
    eval_y = target_y[eval_idx]
    _with_label(eval_x, eval_y).to_csv(split_dir / "eval.csv", index=False)

    train_indices: dict[str, list[int]] = {}
    label_rates: dict[str, float] = {}
    for state_index, state in enumerate(US_STATES):
        if state == target_state:
            x, y = target_x, target_y
            train_idx = target_train_pool[:train_size]
        else:
            x, y = load_state(
                raw_dir=data_dirs.raw_dir,
                survey_year=survey_year,
                state=state,
                download=download,
            )
            rng = np.random.default_rng(seed + 1000 + state_index)
            train_idx = sample_indices(len(y), train_size, rng)

        train_indices[state] = [int(index) for index in train_idx]
        train_y = y[train_idx]
        label_rates[state] = float(np.mean(train_y))
        train_x = x.iloc[train_idx].reset_index(drop=True)
        _with_label(train_x, train_y).to_csv(train_dir / f"{state}.csv", index=False)

    manifest = {
        "schema_version": 1,
        "dataset": "ACSIncome",
        "survey_year": str(survey_year),
        "horizon": HORIZON,
        "survey": SURVEY,
        "target_state": target_state,
        "states": US_STATES,
        "train_size": int(train_size),
        "eval_size": int(eval_size),
        "seed": int(seed),
        "data_root": str(data_dirs.root_dir),
        "raw_dir": str(data_dirs.raw_dir),
        "processed_dir": str(data_dirs.processed_dir),
        "split_dir": str(split_dir),
        "eval": {
            "state": target_state,
            "indices": [int(index) for index in eval_idx],
            "label_rate": float(np.mean(eval_y)),
        },
        "train": {
            state: {
                "indices": indices,
                "label_rate": label_rates[state],
            }
            for state, indices in train_indices.items()
        },
    }
    (split_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return load_processed_split(split_dir=split_dir, data_dirs=data_dirs)


def load_or_prepare_processed_split(
    *,
    survey_year: str,
    target_state: str,
    train_size: int,
    eval_size: int,
    seed: int,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    download: bool = False,
) -> ACSProcessedSplit:
    return prepare_processed_split(
        survey_year=survey_year,
        target_state=target_state,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
        data_dir=data_dir,
        download=download,
        force=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="Download/cache raw ACS files if needed and materialize the split.",
    )
    prepare.add_argument("--survey-year", default="2018")
    prepare.add_argument("--target-state", type=str.upper, default="PA", choices=US_STATES)
    prepare.add_argument("--train-size", type=int, default=500)
    prepare.add_argument("--eval-size", type=int, default=1000)
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        split = prepare_processed_split(
            survey_year=args.survey_year,
            target_state=args.target_state,
            train_size=args.train_size,
            eval_size=args.eval_size,
            seed=args.seed,
            data_dir=args.data_dir,
            download=args.download,
            force=args.force,
        )
        print(f"Processed split: {split.split_dir}")
        print(f"Sample manifest: {split.manifest_path}")
        print(f"Raw data directory: {split.data_dirs.raw_dir}")


if __name__ == "__main__":
    main()
