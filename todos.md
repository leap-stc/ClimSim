# ClimSim autoresearch TODO

Branch: `hermes/research-onboarding`
Base repo: `https://github.com/leap-stc/ClimSim`
Primary intent: turn the official ClimSim repo into a repo Nathan can do real autoresearch on, with a strong research foundation, a reproducible offline baseline path, a Modal-ready remote training path, and a clear next bounded loop.

## Current read on the repo

- This repo already contains a lot more than the original paper baselines:
  - offline baseline models under `baseline_models/`
  - online-training-oriented models under `online_testing/baseline_models/`
  - Hydra + Weights & Biases + NVIDIA Modulus training code for `MLP_v2rh`, `Unet_v4`, and `Unet_v5`
  - data preprocessing notebooks and normalization assets
- The strongest immediately usable training code for remote execution is in `online_testing/baseline_models/`, not the older notebook-first baselines.
- The biggest blockers for autonomous iteration are operational, not scientific:
  - hard-coded HPC paths in YAML configs
  - SLURM-oriented launch assumptions
  - no small reproducible smoke-test harness
  - no repo-local experiment/todo system for fresh-context loops

## Research-backed starting point

1. Start from the official ClimSim repo rather than a blank repo.
2. Use official online-testing baselines as the first reproducible target:
   - `MLP_v2rh` for the simplest strong baseline
   - `Unet_v4` / `Unet_v5` as the official stronger online-oriented baselines
3. Treat Kaggle-style sequence models such as Squeezeformer as a second wave, after the repo can:
   - ingest a compact reproducible dataset split
   - run smoke tests remotely
   - log metrics and artifacts consistently
   - use the best public LEAP Kaggle solution as a concrete reference point: `https://github.com/shlomoron/LEAP-solution`
4. Separate offline progress from online progress:
   - offline metric improvements are necessary
   - online stability needs separate evaluation and should not be collapsed into a single number too early

## Staged plan

### Phase 0 — repo audit and research foundation
- [x] Clone official `leap-stc/ClimSim` into `~/Documents/vscode/climsim-autoresearch`
- [x] Create a working branch for Hermes onboarding work
- [x] Audit repo structure, official docs, training entrypoints, and configs
- [x] Write an explicit research/onboarding memo in `docs/research/`
- [x] Create this `todos.md` so future iterations have a stable anchor

### Phase 1 — make one training path operationally legible
- [ ] Choose one first-class baseline path for reproducible work
  - current default choice: `online_testing/baseline_models/MLP_v2rh/training/train_mlp_h5loader.py`
- [ ] Replace hard-coded HPC paths with launch-time overrides only
- [ ] Define a tiny smoke dataset contract
- [ ] Define one canonical full-run config contract
- [ ] Add a documented command that can run 1-2 batches end-to-end

### Phase 2 — Modalize the first path
- [x] Add a first-pass Modal launcher scaffold: `scripts/modal_train_climsim.py`
- [x] Add Modal onboarding notes: `docs/modal/README.md`
- [ ] Validate the image build assumptions around `nvidia-modulus`
- [ ] Validate the launcher with a true smoke dataset
- [ ] Confirm checkpoint/output persistence layout works in practice

### Phase 3 — make the repo autoresearchable
- [ ] Create a small deterministic verification target for iteration loops
  - likely target: smoke training exits 0 and writes checkpoints/logs
- [ ] Add a repo-local runner script for that verification target
- [ ] Initialize `.hermes-loop/` or autoresearch harness around that bounded goal
- [ ] Record first baseline metric and artifact locations
- [ ] Run repeated fresh-context improvement iterations against the bounded metric

### Phase 4 — expand modeling surface after the harness works
- [ ] Add a clean sequence-model baseline directory for Kaggle-style architectures
  - first candidate: Squeezeformer-style vertical-sequence model
  - strongly consider pulling a minimal, battle-tested Squeezeformer implementation from the Kaggle winner / public reproduction repo instead of reimplementing from scratch
- [ ] Define an architecture-search process rather than one-off model swaps
  - stage 1: fixed data contract + fixed eval + swap trunk only
  - stage 2: compare MLP vs U-Net vs Squeezeformer/Conformer-style vertical encoder
  - stage 3: tune loss weighting / clipping / head structure only after the trunk comparison is stable
- [ ] Add better grouped validation splits and per-target diagnostics
- [ ] Add confidence / clipping / conservation correction ablations
- [ ] Add a short-list of online-readiness proxies before full coupled evaluation

## Immediate next bounded goal

Get the official repo `MLP_v2rh` trainer running on Modal in smoke mode, with the launcher modeled directly after the repo's own training entrypoint:
- call `online_testing/baseline_models/MLP_v2rh/training/train_mlp_h5loader.py`
- keep launch-time path overrides
- keep the same smoke dataset contract for now
- produce a trained model artifact that can be evaluated with the existing smoke eval script

That is the right next iteration-loop target because it moves the harness back toward the repo's real training stack instead of a custom fallback trainer.

## Notes for future Hermes iterations

- Prefer editing around `online_testing/baseline_models/` first. That is where the most mature PyTorch/Hydra training stack already lives.
- Do not start with full coupled online evaluation on Modal. First win the smoke/offline path.
- Do not overfit to Kaggle ideas before the official repo’s strongest reusable training path is runnable and measurable.
- If a better first bounded goal appears after a real smoke test, update this file rather than keeping plans implicit.
