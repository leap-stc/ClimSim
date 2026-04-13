from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.climsim_smoke_common import load_v4_norms

INPUT_DIM = 1525
OUTPUT_DIM = 368


def _make_inputs(
    rng: np.random.Generator,
    n: int,
    input_sub: np.ndarray,
    input_div: np.ndarray,
) -> np.ndarray:
    x = np.tile(input_sub.reshape(1, -1), (n, 1)).astype(np.float32)
    nonzero = input_div != 0
    x[:, nonzero] += (
        rng.normal(loc=0.0, scale=0.05, size=(n, int(nonzero.sum()))).astype(np.float32)
        * input_div[nonzero].reshape(1, -1)
    )
    # Keep the qc/qi-related sections small and positive so the exp transform is stable.
    x[:, 120:180] = rng.uniform(0.0, 1e-8, size=(n, 60)).astype(np.float32)
    x[:, 180:240] = rng.uniform(0.0, 1e-8, size=(n, 60)).astype(np.float32)
    # For zero-div features, stay exactly at the normalization mean to avoid inf/nan before sanitization.
    x[:, ~nonzero] = input_sub[~nonzero]
    return x


def _make_targets(rng: np.random.Generator, n: int, out_scale: np.ndarray) -> np.ndarray:
    safe_scale = np.where(out_scale == 0, 1.0, out_scale)
    desired_scaled = rng.normal(loc=0.0, scale=0.05, size=(n, OUTPUT_DIM)).astype(np.float32)
    return desired_scaled / safe_scale.reshape(1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic ClimSim smoke-test data.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    shard_dir = output_dir / "shard_000"
    shard_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    repo_root = Path(__file__).resolve().parent.parent
    input_sub, input_div, out_scale, _, _ = load_v4_norms(repo_root)
    train_x = _make_inputs(rng, args.train_samples, input_sub, input_div)
    train_y = _make_targets(rng, args.train_samples, out_scale)
    val_x = _make_inputs(rng, args.val_samples, input_sub, input_div)
    val_y = _make_targets(rng, args.val_samples, out_scale)

    with h5py.File(shard_dir / "train_input.h5", "w") as f:
        f.create_dataset("data", data=train_x)
    with h5py.File(shard_dir / "train_target.h5", "w") as f:
        f.create_dataset("data", data=train_y)

    np.save(output_dir / "val_input.npy", val_x)
    np.save(output_dir / "val_target.npy", val_y)

    metadata = {
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "seed": args.seed,
    }
    (output_dir / "README.txt").write_text(
        "Synthetic ClimSim smoke dataset\n" + "\n".join(f"{k}={v}" for k, v in metadata.items()) + "\n"
    )

    print(output_dir)


if __name__ == "__main__":
    main()
