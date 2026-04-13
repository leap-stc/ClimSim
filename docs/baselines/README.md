# Baselines and reproduction status

## Current operational smoke baseline

The current working operational smoke baseline is the official repo `MLP_v2rh` trainer running on Modal in smoke mode via:

- trainer: `online_testing/baseline_models/MLP_v2rh/training/train_mlp_h5loader.py`
- launcher: `scripts/modal_train_climsim_repo.py`
- verifier: `scripts/run_modal_mlp_v2rh_repo_smoke.sh`

This is not a reproduction of the published paper numbers yet.
It is a repo-native smoke baseline proving that the official training stack can run remotely, export artifacts, and be evaluated.

The current smoke numbers below are synthetic-data operational checks. They demonstrate that the repo-native training/eval harness runs end-to-end, but they are not directly comparable to the published offline metrics in `website/evaluating.md`.

### Latest successful repo-native smoke results

Source artifact on the local machine at time of measurement: `artifacts/modal_repo_smoke/latest.json` (gitignored; not part of the committed baseline record)

- train exit code: 0
- eval exit code: 0
- train loss: `1.620e-03`
- valid loss: `1.406e-03`
- MAE: `0.03939143568277359`
- MSE: `0.0028124661184847355`
- RMSE: `0.05303268915003967`
- R2: `-0.4287988854121101`

### Produced artifacts

- `model.pt`
- `ckpt_epoch_1_metric_0.0014.mdlus`
- `ckpt_epoch_1_metric_0.0014.pt`

## Published offline baseline reference from the repo docs

The original repo documents offline baseline metrics in `website/evaluating.md` for the older baseline suite (CNN / ED / HSR / MLP / RPN / cVAE) on the real low-resolution quickstart dataset.

For the documented offline MLP baseline, the repo reports for example:

- RMSE
  - dT/dt: `4.421`
  - dq/dt: `7.322`
  - NETSW: `26.71`
  - FLWDS: `6.969`
- MAE
  - dT/dt: `2.683`
  - dq/dt: `4.495`
  - NETSW: `13.36`
  - FLWDS: `5.224`
- R2
  - dT/dt: `0.589`
  - NETSW: `0.983`
  - FLWDS: `0.924`

These are the numbers we should compare against when we reproduce the original offline baseline regime on the real `LEAP/subsampled_low_res` dataset.

## Current gap to close

We still need to:

1. download the real quickstart dataset `LEAP/subsampled_low_res`
2. run the repo-native offline baseline/evaluation flow on that real dataset
3. compare the resulting metrics to the documented table in `website/evaluating.md`

## External architecture-search reference

For Kaggle-style sequence-model work, we are using this public reference repo:

- `https://github.com/shlomoron/LEAP-solution`

This should inform a future `Squeezeformer` / architecture-search branch after offline baseline reproduction is in place.
