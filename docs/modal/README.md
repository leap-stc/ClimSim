# Modal onboarding for ClimSim autoresearch

This branch adds a first-pass Modal launcher scaffold so the official ClimSim repo can move away from pure SLURM assumptions.

Added files:
- `scripts/modal_train_climsim.py`
- `docs/research/climsim-autoresearch-brief.md`
- `todos.md`

## What this is trying to solve

The official repo already has strong training code under:
- `online_testing/baseline_models/MLP_v2rh/training/`
- `online_testing/baseline_models/Unet_v4/training/`
- `online_testing/baseline_models/Unet_v5/training/`

But those stacks still assume:
- hard-coded HPC paths
- SLURM launch scripts
- hand-managed environments

The immediate goal is not to redesign the training code. It is to make one path cloud-runnable with launch-time overrides.

## Recommended first target

Use `MLP_v2rh` first.

Why:
- simplest strong online-oriented baseline in the repo
- easier than the U-Net paths for the first smoke test
- already uses Hydra + Modulus + checkpointing/logging logic

## What the launcher does

`scripts/modal_train_climsim.py` is a batch-job wrapper around the existing training script.

Design choices:
- use a CUDA base image because the repo depends on NVIDIA Modulus
- mount separate persistent Modal volumes for:
  - data
  - checkpoints
  - outputs
- treat W&B as optional
- pass all repo-specific paths via Hydra overrides instead of editing training code
- support a cheap smoke mode and a fuller training mode

## Important caveat

The launcher is a first scaffold. The most likely remaining sharp edge is the exact runtime environment for `nvidia-modulus`.

Before trusting full runs, verify in order:
1. image builds successfully
2. container can import `torch` and `modulus`
3. smoke dataset paths are valid
4. smoke run reaches at least one train and validation step
5. checkpoint and output volumes persist expected artifacts

## Suggested secret / volume names

Default names in the scaffold:
- secret: `wandb`
- data volume: `climsim-data`
- checkpoint volume: `climsim-checkpoints`
- output volume: `climsim-outputs`

You can rename these with CLI flags.

## Smoke-run idea

The right first remote milestone is not “full training works.”
It is:
- image builds
- one small job starts
- one or two train steps run
- one or two val steps run
- outputs are persisted

Suggested smoke overrides are already embedded in the script:
- `epochs=1`
- `early_stop_step=2`
- `num_workers=0`
- smaller batch size
- offline MLflow-style logging by default

## Example workflow

Authenticate Modal locally first.

Then run a smoke job:

```bash
modal run scripts/modal_train_climsim.py --smoke
```

For a fuller run:

```bash
modal run scripts/modal_train_climsim.py \
  --dataset-subdir preprocessing/v4 \
  --expname mlp_v2rh_modal_full \
  --gpu A100 \
  --batch-size 512 \
  --epochs 10
```

To enable W&B:

1. create a Modal secret named `wandb` that contains `WANDB_API_KEY`
2. run with `--use-wandb`

## If you want to run your exact local branch remotely

The current repo-smoke launchers use local source inclusion in the Modal image via `add_local_dir(...)`.
That is convenient for rapid iteration on the exact working tree, but it is not as reproducible as cloning a pinned upstream commit.

If we want stronger provenance later, the next refinement is to switch the repo-mode launcher to a pinned Git commit or tag and record that revision in the returned JSON.

## Next engineering tasks

1. create or stage a tiny smoke dataset subset in the data volume
2. validate the current `nvidia-modulus` install path
3. run the smoke job to completion
4. convert that smoke path into a bounded iteration-loop goal
5. only then expand to U-Net and Kaggle-style sequence baselines
