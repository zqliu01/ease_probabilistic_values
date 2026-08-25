"""Seed shapiq's cache with PolySHAP's precomputed ViT4by4 utility tables."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from config import (
    DEFAULT_INSTANCE_COUNT,
    POLYSHAP_ARCHIVE_URL,
    POLYSHAP_CACHE_DIR,
    POLYSHAP_VIT4BY4_GAME_NAME,
    POLYSHAP_VIT4BY4_N_PLAYERS,
)


def shapiq_target_dir() -> Path:
    try:
        from shapiq.benchmark.precompute import SHAPIQ_DATA_DIR
    except ImportError as exc:
        raise ImportError(
            "prepare_polyshap_vit4by4.py requires shapiq==1.3.0 with benchmark support."
        ) from exc
    return Path(SHAPIQ_DATA_DIR) / POLYSHAP_VIT4BY4_GAME_NAME / str(POLYSHAP_VIT4BY4_N_PLAYERS)


def download_archive(url: str, archive_path: Path, *, force: bool) -> None:
    if archive_path.exists() and not force:
        print(f"using cached archive: {archive_path}")
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(f"{archive_path.suffix}.tmp")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp_path.replace(archive_path)
    print(f"saved archive: {archive_path}")


def extract_vit4by4_tables(archive_path: Path, target_dir: Path) -> int:
    source_fragment = (
        f"/data/precomputed_games/{POLYSHAP_VIT4BY4_GAME_NAME}/"
        f"{POLYSHAP_VIT4BY4_N_PLAYERS}/"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.endswith(".npz"):
                continue
            if source_fragment not in f"/{info.filename}":
                continue
            target_path = target_dir / Path(info.filename).name
            with zf.open(info) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    if count == 0:
        raise FileNotFoundError(
            "Did not find PolySHAP ViT4by4 precomputed files in the downloaded archive. "
            f"Expected archive members containing {source_fragment!r}."
        )
    return count


def verify_loader(expected_count: int) -> None:
    from shapiq.benchmark import load_games_from_configuration

    games = list(
        load_games_from_configuration(
            game_class="ImageClassifierLocalXAI",
            n_player_id=2,
            config_id=1,
            n_games=expected_count,
        )
    )
    if len(games) != expected_count:
        raise RuntimeError(f"Expected {expected_count} shapiq games, loaded {len(games)}.")
    n_players = int(getattr(games[0], "n_players", -1))
    if n_players != POLYSHAP_VIT4BY4_N_PLAYERS:
        raise RuntimeError(
            f"Expected {POLYSHAP_VIT4BY4_N_PLAYERS} players, loaded game has {n_players}."
        )
    print(f"verified shapiq loader: {len(games)} games, n_players={n_players}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install PolySHAP ViT4by4 precomputed games into shapiq's local cache."
    )
    parser.add_argument("--url", default=POLYSHAP_ARCHIVE_URL, help="PolySHAP source archive URL.")
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=POLYSHAP_CACHE_DIR / "PolySHAP-main.zip",
        help="Local path used to cache the downloaded archive.",
    )
    parser.add_argument("--force-download", action="store_true", help="Redownload the archive.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that shapiq can load the cached precomputed games.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=DEFAULT_INSTANCE_COUNT,
        help="Expected number of precomputed games.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_dir = shapiq_target_dir()
    print(f"shapiq target dir: {target_dir}")
    if not args.verify_only:
        download_archive(args.url, args.archive_path, force=args.force_download)
        extracted = extract_vit4by4_tables(args.archive_path, target_dir)
        print(f"extracted {extracted} files")
        if extracted < args.expected_count:
            raise RuntimeError(
                f"Expected at least {args.expected_count} files, extracted {extracted}."
            )
    verify_loader(args.expected_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
