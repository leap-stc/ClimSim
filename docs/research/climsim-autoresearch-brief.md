# ClimSim autoresearch research brief

Date: 2026-04-12
Repo: `~/Documents/vscode/climsim-autoresearch`
Base upstream: `https://github.com/leap-stc/ClimSim`

## Why this repo is worth building on

The official ClimSim repo is not just a dataset helper package. It already contains:

- the core `climsim_utils` package for preprocessing and metrics
- offline baseline model folders under `baseline_models/`
- newer online-testing-oriented training stacks under `online_testing/baseline_models/`
- preprocessing notebooks and precomputed normalization assets
- documentation tying together the Hugging Face datasets, paper baselines, and online workflow

That means the right strategy is not “invent a new benchmark repo from scratch.”
The right strategy is:

1. keep the official repo as the substrate
2. add operational scaffolding for reproducible remote execution
3. make one path measurable and loopable
4. only then widen the modeling frontier

## Repo audit summary

Important files and what they imply:

- `README.md`
  - positions ClimSim as the main benchmark and points to docs, demo notebooks, evaluation, and online testing
- `website/quickstart.md`
  - confirms the original quickstart uses the subsampled low-res pre-normalized NPY data
- `online_testing/README.md`
  - says the online task should use the low-res or expanded low-res datasets, not the quickstart dataset, because online work needs cloud and wind tendencies too
  - documents stronger baselines tied to the `Stable Machine-Learning Parameterization` paper
- `online_testing/baseline_models/MLP_v2rh/training/`
  - already has a reasonably complete PyTorch/Hydra/Modulus training stack
- `online_testing/baseline_models/Unet_v4` and `Unet_v5`
  - encode stronger official online baselines and constraints-oriented variants
- `setup.py`
  - only packages `climsim_utils`; the training stacks themselves are not operationally packaged for cloud execution

Main operational weakness:
- configs still contain hard-coded HPC paths like `/global/...` and `/pscratch/...`
- launch is still thought of in terms of SLURM scripts, not portable batch runners

## Dataset landscape

### 1. ClimSim
Primary benchmark for subgrid physics emulation.

Relevant sources:
- repo: https://github.com/leap-stc/ClimSim
- high-res dataset: https://huggingface.co/datasets/LEAP/ClimSim_high-res
- low-res dataset: https://huggingface.co/datasets/LEAP/ClimSim_low-res
- low-res aquaplanet: https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet

Practical implications:
- the natural supervised object is an atmospheric column at a timestep
- there are both flattened/tabular and structured-per-level ways to represent the same example
- preprocessing and normalization are a huge part of the benchmark, not an implementation detail

### 2. ClimSim-Online
Key source for the offline-vs-online gap.

- paper: https://arxiv.org/abs/2306.08754

Practical implication:
- an autoresearch loop cannot trust offline pointwise metrics alone if the long-term goal is a stable coupled emulator

### 3. Stable ML parameterization in E3SM-MMF
The most directly useful “what actually worked online” paper for the current repo.

- paper: https://arxiv.org/abs/2407.00124

Practical implication:
- the official repo’s `MLP_v2rh`, `Unet_v4`, and `Unet_v5` paths are not toy examples; they are the closest thing to a reusable production baseline inside this repo

### 4. Confidence-guided mixing
- paper: https://arxiv.org/abs/2510.08107

Practical implication:
- once the harness exists, hybrid/fallback policies are a high-value ablation family

### 5. WeatherBench 2 / ClimateBench / WxC-Bench
These matter more as benchmark-design references than as immediate implementation targets.

- WeatherBench 2 repo: https://github.com/google-research/weatherbench2
- ClimateBench repo: https://github.com/duncanwp/ClimateBench
- ClimateBench data: https://zenodo.org/records/5196512
- WxC-Bench paper: https://www.nature.com/articles/s41597-026-06839-7

Practical implication:
- use them as architectural inspiration for configs, evaluators, and multi-benchmark abstractions later
- do not let them distract from the first ClimSim smoke/full training path now

## What to treat as the first strong baseline

## Recommendation: `online_testing/baseline_models/MLP_v2rh`

Why start here:
- already uses PyTorch + Hydra + Modulus, which is much closer to a batch-cloud workflow than the older notebook baselines
- simpler than the U-Net paths, so it is the best first target for Modal smoke execution
- close enough to the official online literature to matter scientifically

Why not start with U-Net first:
- stronger but more operationally complex
- not ideal for the first “can this repo run remotely and checkpoint cleanly?” milestone

Why not start with a fresh Squeezeformer repo immediately:
- it would create a second source of truth before the official repo is operationally stable
- we still need a reproducible data contract, smoke target, and experiment runner
- Kaggle-derived architectures are phase 2 or 3 work, not phase 0

## Current practical model hierarchy

1. `MLP_v2rh`
- best first operational target
- likely first Modal smoke and first iteration-loop target

2. `Unet_v4` / `Unet_v5`
- best official next-step baselines once the remote harness works
- especially relevant if the goal shifts from “remote training works” to “online robustness and stronger official baselines”

3. Kaggle-style sequence models
- likely include Squeezeformer/Conformer-like vertical encoders, strong target normalization, head splitting, and clipping/conservation tricks
- should be added only after the repo has a stable harness

## Why Kaggle-style sequence models still matter

The practical baseline memo from research synthesis suggests a consistent pattern:

- strong public leaderboard solutions often use vertical-sequence encoders
- Squeezeformer-like blocks are a very plausible next architecture family for ClimSim
- but the actual win is often more about preserving vertical structure, target scaling, and post-processing than about “transformers” in the abstract

This suggests the right expansion path is:

1. operationalize official repo baselines first
2. pull or adapt a battle-tested Squeezeformer implementation from the best public Kaggle winner / reproduction code rather than rebuilding it blindly
   - current reference repo: `https://github.com/shlomoron/LEAP-solution`
3. compare against the official MLP/U-Net baselines with the same data contract and diagnostics
4. treat architecture search as a controlled process:
   - first fix data + metric + evaluation
   - then swap the trunk architecture only
   - then tune loss weighting, clipping, and output heads

The key idea is to avoid mixing “new architecture” and “new training recipe” in the same experiment if we want autoresearch to learn anything cumulative.

## Main technical blockers to autonomous research today

### Hard-coded path assumptions
Example from `online_testing/baseline_models/MLP_v2rh/training/conf/config_single.yaml`:
- `climsim_path: '/global/u2/z/zeyuanhu/nvidia_codes/Climsim_private/'`
- `data_path: '/pscratch/sd/z/zeyuanhu/hugging/E3SM-MMF_ne4/preprocessing/v4/'`
- `save_path: '/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/saved_models/'`

This is the single biggest operational blocker.

### No explicit smoke-test contract
There is no small, canonical, machine-checkable command that says:
- import works
- data paths are valid
- one train step works
- one validation step works
- checkpoint/log writing works

Without this, an iteration loop has nothing sharp to optimize against.

### SLURM-era launch assumptions
The repo contains many `sbatch` scripts, but no cloud-native batch launcher abstraction.

### Weak loop state
There was no repo-local `todos.md` or loop anchor before this work.

## Modal direction

A Modal-ready path should:

- keep the existing training script intact
- use launch-time Hydra overrides instead of editing code for paths
- mount persistent storage for:
  - datasets
  - checkpoints
  - outputs/logs
- treat W&B as optional and disable or route offline for smoke runs
- start from a CUDA-compatible base image because of `nvidia-modulus`

Added in this branch:
- `docs/modal/README.md`
- `scripts/modal_train_climsim.py`

Those files are intended as the first cloud-launch scaffold, not the final production launcher.

## Best next bounded goal

Get a true smoke run working for `MLP_v2rh` with all paths overridden at launch time.

A good smoke target is:
- 1 epoch
- batch size small enough to run cheaply
- `early_stop_step=2`
- `num_workers=0`
- logger routed to offline MLflow or otherwise non-networked mode
- writes outputs and checkpoints to persistent storage

That goal is concrete enough for a fresh-context loop.

## Suggested future metric for autoresearch onboarding

Candidate bounded goal:
- “Modal smoke run for `MLP_v2rh` exits 0 and writes at least one checkpoint and one metrics artifact.”

Candidate verification command once implemented:
- a repo-local wrapper script that launches the smoke config and then checks the output/checkpoint paths

Why this is the right first metric:
- it is operationally essential
- it is binary and mechanically verifiable
- it unlocks future modeling iterations

## Reference links

- ClimSim repo: https://github.com/leap-stc/ClimSim
- ClimSim high-res dataset: https://huggingface.co/datasets/LEAP/ClimSim_high-res
- ClimSim paper / online benchmark framing: https://arxiv.org/abs/2306.08754
- Stable online parameterization paper: https://arxiv.org/abs/2407.00124
- Confidence-guided mixing: https://arxiv.org/abs/2510.08107
- WeatherBench 2: https://github.com/google-research/weatherbench2
- ClimateBench: https://github.com/duncanwp/ClimateBench
- ClimateBench data: https://zenodo.org/records/5196512
- WxC-Bench paper: https://www.nature.com/articles/s41597-026-06839-7
