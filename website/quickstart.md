# Quickstart

**Step 1**

The first step is to download the subsampled low-resolution real-geography version of the data [here](https://huggingface.co/datasets/LEAP/subsampled_low_res/tree/main). This contains subsampled and prenormalized data that was used for training, validation, and metrics for the ClimSim paper. It can be reproduced with the full version of the [dataset](https://huggingface.co/datasets/LEAP/ClimSim_low-res) using the [`preprocessing/create_npy_data_splits.ipynb`](https://github.com/leap-stc/ClimSim/blob/main/preprocessing/create_npy_data_splits.ipynb) notebook.

System requirements can be found [here](https://github.com/leap-stc/ClimSim/blob/main/system_requirements.md). This was contributed by [STAT GU4243/GR 5243 Applied Data Science - Fall 2023 at Columbia University](https://github.com/TZstatsADS/ADS_Teaching/blob/master/16-Fall2023/Projects_StarterCodes/Project3-ClimSim/asset/system_requirements.md).

Training data corresponds to **train_input.npy** and **train_target.npy**. Validation data corresponds to **val_input.npy** and **val_target.npy**. Scoring data (which can be treated as a test set) corresponds to **scoring_input.npy** and **scoring_target.npy**. We have an additional held-out test set that we are using for our [Kaggle competition](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim).

**Step 2**

Install the `climsim_utils` python tools, by running the following code from the root of the [GitHub repo](https://github.com/leap-stc/ClimSim/tree/main):

```
pip install .
```

If you already have all `climsim_utils` dependencies (`tensorflow`, `xarray`, etc.) installed in your local environment, you can alternatively run:

```
pip install . --no-deps
```

**Step 3**

Train your model on the training data and validate using the validation data. If you wish to use something like a CNN, you will probably want to separate the variables into channels and broadcast scalars into vectors of the same dimension as vertically-resolved variables. Methods to do this can be found in the [`climsim_utils/data_utils.py`](https://github.com/leap-stc/ClimSim/blob/main/climsim_utils/data_utils.py) script.

**Step 4**

Evaluation time! Use the [`evaluation/main_figure_generation.ipynb`](https://github.com/leap-stc/ClimSim/blob/main/evaluation/main_figure_generation.ipynb) notebook to see how your model does! Use the **calc_MAE**, **calc_RMSE**, and **calc_R2** methods in the [`climsim_utils/data_utils.py`](https://github.com/leap-stc/ClimSim/blob/main/climsim_utils/data_utils.py) script to see how your model does on point estimates and use the `calc_CRPS` method to check how well-calibrated your model is if it's stochastic. 😊

