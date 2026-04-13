from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "climsim-train"
LOCAL_REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = "/root/climsim-src"
SMOKE_DATA_REL = "smoke_data/v4"
RUNS_ROOT = "/tmp/climsim_modal_runs"

IGNORE_PATTERNS = [
    ".git",
    ".venv",
    "venv",
    "ENV",
    "env",
    "__pycache__",
    ".pytest_cache",
    "wandb",
    ".mypy_cache",
    ".hermes-loop",
    "artifacts",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "build-essential",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "numpy",
        "pandas",
        "matplotlib",
        "xarray",
        "netCDF4",
        "h5py",
        "tqdm",
        "omegaconf",
        "hydra-core",
        "torch",
    )
    .add_local_dir(str(LOCAL_REPO_ROOT), remote_path=REPO_ROOT, ignore=IGNORE_PATTERNS)
)

app = modal.App(APP_NAME)


def _run(command: list[str], cwd: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)


@app.function(image=image, gpu="T4", timeout=60 * 60)
def run_training_and_eval(
    expname: str = "mlp_v2rh_modal_smoke",
    smoke: bool = True,
    batch_size: int = 64,
    epochs: int = 1,
    use_wandb: bool = False,
) -> str:
    dataset_dir = f"{REPO_ROOT}/{SMOKE_DATA_REL}"
    save_root = f"{RUNS_ROOT}/{expname}"
    os.makedirs(save_root, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if use_wandb:
        env.setdefault("WANDB_PROJECT", APP_NAME)
    else:
        env["WANDB_MODE"] = "disabled"

    model_dir = Path(save_root) / expname
    model_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "python",
        f"{REPO_ROOT}/scripts/train_mlp_v2rh_smoke.py",
        "--repo-root",
        REPO_ROOT,
        "--dataset-dir",
        dataset_dir,
        "--output-dir",
        str(model_dir),
        "--batch-size",
        str(min(batch_size, 16) if smoke else batch_size),
        "--epochs",
        str(1 if smoke else epochs),
        "--max-train-steps",
        "2" if smoke else "20",
        "--max-val-steps",
        "2" if smoke else "10",
    ]
    train_proc = _run(train_cmd, cwd=REPO_ROOT, env=env)

    summary: dict[str, object] = {
        "train_exit_code": train_proc.returncode,
        "train_stdout_tail": train_proc.stdout[-12000:],
        "train_stderr_tail": train_proc.stderr[-12000:],
        "expname": expname,
        "save_root": save_root,
        "dataset_dir": dataset_dir,
    }

    if train_proc.returncode != 0:
        raise RuntimeError(json.dumps(summary, indent=2, sort_keys=True))

    eval_json = model_dir / "smoke_eval.json"
    eval_cmd = [
        "python",
        f"{REPO_ROOT}/scripts/eval_mlp_v2rh_smoke.py",
        "--model-dir",
        str(model_dir),
        "--dataset-dir",
        dataset_dir,
        "--output-json",
        str(eval_json),
    ]
    eval_proc = _run(eval_cmd, cwd=REPO_ROOT, env=env)
    summary["eval_exit_code"] = eval_proc.returncode
    summary["eval_stdout_tail"] = eval_proc.stdout[-12000:]
    summary["eval_stderr_tail"] = eval_proc.stderr[-12000:]

    if eval_proc.returncode != 0:
        raise RuntimeError(json.dumps(summary, indent=2, sort_keys=True))

    summary["eval_json"] = str(eval_json)
    summary["eval"] = json.loads(eval_proc.stdout)
    return json.dumps(summary, indent=2, sort_keys=True)


@app.local_entrypoint()
def main(
    expname: str = "mlp_v2rh_modal_smoke",
    smoke: bool = True,
    batch_size: int = 64,
    epochs: int = 1,
    use_wandb: bool = False,
) -> None:
    result = run_training_and_eval.remote(
        expname=expname,
        smoke=smoke,
        batch_size=batch_size,
        epochs=epochs,
        use_wandb=use_wandb,
    )
    print(result)
