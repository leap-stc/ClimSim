from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from online_testing.baseline_models.MLP_v2rh.training.climsim_datapip import climsim_dataset
from online_testing.baseline_models.MLP_v2rh.training.climsim_datapip_h5 import climsim_dataset_h5
from scripts.climsim_smoke_common import load_v4_norms


class SimpleMLP(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, hidden_dim: int = 256, layers: int = 4):
        super().__init__()
        blocks = []
        prev = in_dims
        for _ in range(layers):
            blocks.append(nn.Linear(prev, hidden_dim))
            blocks.append(nn.ReLU())
            prev = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(self.backbone(x))
        y = y.clone()
        y[:, -8:] = torch.relu(y[:, -8:].clone())
        return y


def build_data(repo_root: Path, dataset_dir: Path):
    input_sub, input_div, out_scale, qc_lbd, qi_lbd = load_v4_norms(repo_root)

    common = dict(
        input_sub=input_sub,
        input_div=input_div,
        out_scale=out_scale,
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

    train_ds = climsim_dataset_h5(str(dataset_dir), **common)
    val_ds = climsim_dataset(
        str(dataset_dir / "val_input.npy"),
        str(dataset_dir / "val_target.npy"),
        **common,
    )
    return train_ds, val_ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal ClimSim MLP smoke baseline.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=2)
    parser.add_argument("--max-val-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_data(repo_root, dataset_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(in_dims=1525, out_dims=368).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _epoch in range(args.epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            if step >= args.max_train_steps:
                break
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(val_loader):
                if step >= args.max_val_steps:
                    break
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_losses.append(float(loss.detach().cpu().item()))

    scripted = torch.jit.script(model.to("cpu").eval())
    model_path = output_dir / "model.pt"
    scripted.save(str(model_path))

    metrics = {
        "train_loss_last": train_losses[-1],
        "val_loss_last": val_losses[-1],
        "train_loss_mean": float(np.mean(train_losses)),
        "val_loss_mean": float(np.mean(val_losses)),
        "train_steps": len(train_losses),
        "val_steps": len(val_losses),
        "device": str(device),
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
