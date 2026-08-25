"""Precompute CIFAR10 4x4-superpixel utility tables on a GPU.

This mirrors the PolySHAP CIFAR10 setup:
  - CIFAR10 test split.
  - Deterministic shuffle with random_state=40.
  - 4x4 image grid, 16 players.
  - Gray baseline value 128.
  - ViT model aaraki/vit-base-patch16-224-in21k-finetuned-cifar10.

The output is one NPZ per explained instance with `values[mask]` indexed by
little-endian coalition bitmask.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np

from config import CIFAR10_PRECOMPUTED_DIR, DATA_DIR, DEFAULT_INSTANCE_COUNT, RANDOM_STATE


MODEL_NAME = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
LABEL_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_int_set(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                start_text, end_text = part.split(":", 1)
                out.extend(range(int(start_text), int(end_text) + 1))
            else:
                out.append(int(part))
    return sorted(set(out))


def coalition_batch(indices: np.ndarray, n_players: int) -> np.ndarray:
    bits = np.arange(n_players, dtype=np.uint32)
    return ((indices[:, None].astype(np.uint32) >> bits[None, :]) & 1).astype(bool)


def make_superpixel_masks(height: int, width: int, grid_size: int) -> np.ndarray:
    h_step = height // grid_size
    w_step = width // grid_size
    masks = []
    for row in range(grid_size):
        for col in range(grid_size):
            mask = np.zeros((height, width), dtype=bool)
            mask[row * h_step : (row + 1) * h_step, col * w_step : (col + 1) * w_step] = True
            masks.append(mask)
    return np.stack(masks, axis=0)


def mask_images(x_explain, coalitions: np.ndarray, masks: np.ndarray, baseline_value: int):
    from PIL import Image

    img_np = np.asarray(x_explain)
    masked_images = []
    for coalition in coalitions:
        if np.any(coalition):
            keep_mask = masks[coalition].any(axis=0)
        else:
            keep_mask = np.zeros(masks.shape[1:], dtype=bool)
        masked = img_np.copy()
        masked[~keep_mask] = baseline_value
        masked_images.append(Image.fromarray(masked))
    return masked_images


def load_processor(model_name: str):
    try:
        from transformers import AutoImageProcessor

        return AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        from transformers import AutoFeatureExtractor

        return AutoFeatureExtractor.from_pretrained(model_name)


def precompute_instance(
    *,
    id_explain: int,
    output_dir: Path,
    torchvision_root: Path,
    model_name: str,
    batch_size: int,
    random_state: int,
    device_name: str,
    baseline_value: int,
    force: bool,
) -> Path:
    import torch
    from torchvision.datasets import CIFAR10
    from transformers import AutoModelForImageClassification

    path = output_dir / f"instance_{int(id_explain):03d}.npz"
    if path.exists() and not force:
        print(f"cached {path}")
        return path

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    test_dataset = CIFAR10(root=str(torchvision_root), train=False, download=True)
    rng = random.Random(int(random_state))
    indices = list(range(len(test_dataset)))
    rng.shuffle(indices)
    dataset_index = indices[int(id_explain)]
    x_explain, true_label = test_dataset[dataset_index]

    processor = load_processor(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    model.eval()

    original_inputs = processor(images=x_explain, return_tensors="pt").to(device)
    with torch.no_grad():
        original_logits = model(**original_inputs).logits
    explained_class = int(torch.argmax(original_logits, dim=-1).item())

    grid_size = 4
    n_players = grid_size * grid_size
    img_np = np.asarray(x_explain)
    masks = make_superpixel_masks(img_np.shape[0], img_np.shape[1], grid_size)
    values = np.empty(1 << n_players, dtype=np.float32)

    try:
        from tqdm import tqdm

        iterator = tqdm(
            range(0, 1 << n_players, batch_size),
            desc=f"cifar10 instance {id_explain}",
        )
    except Exception:
        iterator = range(0, 1 << n_players, batch_size)

    for start in iterator:
        end = min(start + int(batch_size), 1 << n_players)
        mask_indices = np.arange(start, end, dtype=np.uint32)
        coalitions = coalition_batch(mask_indices, n_players=n_players)
        images = mask_images(x_explain, coalitions, masks, baseline_value=baseline_value)
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, explained_class]
        values[start:end] = logits.detach().cpu().numpy().astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}.npz")
    np.savez_compressed(
        tmp,
        values=values,
        n_players=np.array(n_players, dtype=np.int64),
        id_explain=np.array(int(id_explain), dtype=np.int64),
        dataset_index=np.array(int(dataset_index), dtype=np.int64),
        random_state=np.array(int(random_state), dtype=np.int64),
        true_label=np.array(int(true_label), dtype=np.int64),
        explained_class=np.array(int(explained_class), dtype=np.int64),
        model_name=np.array(model_name),
        grid_size=np.array(grid_size, dtype=np.int64),
        baseline_value=np.array(int(baseline_value), dtype=np.int64),
        label_names=np.array(LABEL_NAMES),
    )
    os.replace(tmp, path)
    print(
        f"wrote {path} true={LABEL_NAMES[int(true_label)]} "
        f"pred={LABEL_NAMES[int(explained_class)]}"
    )
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--n-instances", type=int, default=DEFAULT_INSTANCE_COUNT)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=CIFAR10_PRECOMPUTED_DIR)
    parser.add_argument("--torchvision-root", type=Path, default=DATA_DIR / "torchvision")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline-value", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        args.task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        args.num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    instance_ids = parse_int_set(args.instance_id)
    if instance_ids is None:
        instance_ids = list(range(int(args.n_instances)))
    if args.task_id is not None or args.num_tasks is not None:
        if args.task_id is None or args.num_tasks is None:
            raise ValueError("--task-id and --num-tasks must be provided together.")
        instance_ids = [
            instance_id
            for pos, instance_id in enumerate(instance_ids)
            if pos % int(args.num_tasks) == int(args.task_id)
        ]

    for id_explain in instance_ids:
        precompute_instance(
            id_explain=int(id_explain),
            output_dir=args.output_dir,
            torchvision_root=args.torchvision_root,
            model_name=args.model_name,
            batch_size=args.batch_size,
            random_state=args.random_state,
            device_name=args.device,
            baseline_value=args.baseline_value,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
