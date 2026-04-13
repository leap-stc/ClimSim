from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from online_testing.baseline_models.MLP_v2rh.training.climsim_datapip import climsim_dataset
from scripts.climsim_smoke_common import load_v4_norms


def evaluate(model_dir: Path, dataset_dir: Path, batch_size: int = 8) -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    input_sub, input_div, out_scale, qc_lbd, qi_lbd = load_v4_norms(repo_root)

    dataset = climsim_dataset(
        str(dataset_dir / "val_input.npy"),
        str(dataset_dir / "val_target.npy"),
        input_sub,
        input_div,
        out_scale,
        qinput_prune=True,
        output_prune=True,
        strato_lev=15,
        qc_lbd=qc_lbd,
        qi_lbd=qi_lbd,
        decouple_cloud=False,
        aggressive_pruning=False,
        strato_lev_qc=30,
        strato_lev_qinput=-1,
        strato_lev_tinput=-1,
        strato_lev_out=12,
        input_clip=True,
        input_clip_rhonly=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model_path = model_dir / "model.pt"
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x).cpu().numpy())
            targets.append(y.cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    mse = float(np.mean((pred - target) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred - target)))
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum((target - np.mean(target, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "num_samples": int(pred.shape[0]),
        "num_features": int(pred.shape[1]),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ClimSim MLP_v2rh smoke run.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(args.model_dir.resolve(), args.dataset_dir.resolve())
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
