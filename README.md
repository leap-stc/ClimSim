# ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators

This repository contains the code necessary to download and preprocess the data, and create, train, and evaluate the baseline models in the [paper](https://arxiv.org/abs/2306.08754).

![fig_1](./website/fig_1.png)


## Quickstart

This method is intended for those who wish to quickly try out new methods on a manageable subset of the data uploaded to HuggingFace.

**Step 1**

The first step is to download the subsampled low-resolution real-geography version of the data [here](https://huggingface.co/datasets/LEAP/subsampled_low_res/tree/main). This contains subsampled and prenormalized data that was used for training, validation, and metrics for the ClimSim paper. It can be reproduced with the full version of the [dataset](https://huggingface.co/datasets/LEAP/ClimSim_low-res) using the [preprocessing/create_npy_data_splits.ipynb](https://github.com/leap-stc/ClimSim/blob/main/preprocessing/create_npy_data_splits.ipynb) notebook.

Training data corresponds to **train_input.npy** and **train_target.npy**. Validation data corresponds to **val_input.npy** and **val_target.npy**. Scoring data (which can be treated as a test set) corresponds to **scoring_input.npy** and **scoring_target.npy**. We have an additional held-out test set that we will use for an upcoming online competition. Keep an eye out! 😉

**Step 2**

Install the `climsim_utils` python tools, by running the following code from the root of this repo:

```
pip install .
```

If you already have all `climsim_utils` dependencies (`tensorflow`, `xarray`, etc.) installed in your local environment, you can alternatively run:

```
pip install . --no-deps
```

**Step 3**

Train your model on the training data and validate using the validation data. If you wish to use something like a CNN, you will probably want to separate the variables into channels and broadcast scalars into vectors of the same dimension as vertically-resolved variables. Methods to do this can be found in the [climsim_utils/data_utils.py](https://github.com/leap-stc/ClimSim/blob/main/climsim_utils/data_utils.py) script.

**Step 4**

Evaluation time! Use the [evaluation/main_figure_generation.ipynb](https://github.com/leap-stc/ClimSim/blob/main/evaluation/main_figure_generation.ipynb) notebook to see how your model does! Use the **calc_MAE**, **calc_RMSE**, and **calc_R2** methods in the [climsim_utils/data_utils.py](https://github.com/leap-stc/ClimSim/blob/main/climsim_utils/data_utils.py) script to see how your model does on point estimates and use the calc_CRPS method to check how well-calibrated your model is if it's stochastic. 😊


## Dataset Information

[![Dataset: E3SM-MMF High-Resolution Real Geography](https://img.shields.io/badge/Dataset-%20High%20Resolution%20Real%20Geography-yellow?logo=🤗&style=flat-square)](https://huggingface.co/datasets/LEAP/ClimSim_high-res)
[![Dataset: E3SM-MMF Low-Resolution Real Geography](https://img.shields.io/badge/Dataset-%20Low%20Resolution%20Real%20Geography-yellow?logo=🤗&style=flat-square)](https://huggingface.co/datasets/LEAP/ClimSim_low-res)
[![Dataset: E3SM-MMF Low-Resolution Aquaplanet](https://img.shields.io/badge/Dataset-%20Low%20Resolution%20Aquaplanet-yellow?logo=🤗&style=flat-square)](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet)


Data from multi-scale climate model (E3SM-MMF) simulations were saved at 20-minute intervals for 10 simulated years. Two netCDF files--input and output (target)--are produced at each timestep, totaling 525,600 files for each configuration. 3 configurations of E3SM-MMF were run and can be downloaded from Hugging Face:

1. [**High-Resolution Real Geography**](https://huggingface.co/datasets/LEAP/ClimSim_high-res)
    - 1.5&deg; x 1.5&deg; horizontal resolution (21,600 grid columns)
    - 5.7 billion total samples (41.2 TB)
    - 102 MB per input file, 61 MB per output file
2. [**Low-Resolution Real Geography**](https://huggingface.co/datasets/LEAP/ClimSim_low-res)
    - 11.5&deg; x 11.5&deg; horizontal resolution (384 grid columns)
    - 100 million total samples (744 GB)
    - 1.9 MB per input file, 1.1 MB per output file
3. [**Low-Resolution Aquaplanet**](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet)
    - 11.5&deg; x 11.5&deg; horizontal resolution (384 grid columns)
    - 100 million total samples (744 GB)
    - 1.9 MB per input file, 1.1 MB per output file

Input files are labeled ```E3SM-MMF.mli.YYYY-MM-DD-SSSSS.nc```, where ```YYYY-MM-DD-SSS``` corresponds to the simulation year (```YYYY```), month (``MM``), day of the month (``DD``), and seconds of the day (```SSSSS```), with timesteps being spaced 1,200 seconds (20 minutes) apart. Target files are labeled the same way, except ```mli``` is replaced by ```mlo```. 
Scalar variables vary in time and "horizontal" grid ("ncol"), while vertically-resolved variables vary additionally in vertical space ("lev").  For vertically-resolved variables, lower indices of "lev" corresponds to higher levels in the atmosphere. This is because pressure decreases monotonically with altitude.   

The full list of variables can be found in [Supplementary Information](https://arxiv.org/pdf/2306.08754.pdf), Table 1.

There is also a quickstart dataset that contains subsampled and prenormalized data. This data was used for training, validation, and metrics for the ClimSim paper and can be reproduced using the ```preprocessing/create_npy_data_splits.ipynb``` notebook.
- [Quickstart dataset](https://huggingface.co/datasets/LEAP/subsampled_low_res)


## Installation & setup

For preprocessing and evaluation, please install the `climsim_utils` python tools, by running the following code from the root of this repo:

```
pip install .
```

If you already have all `climsim_utils` dependencies (`tensorflow`, `xarray`, etc.) installed in your local environment, you can alternatively run:

```
pip install . --no-deps
```


## Baseline Models

Six different baseline models were created and trained:
1. Convolutional neural network (CNN)
2. Encoder-decoder (ED)
3. Heteroskedastic regression (HSR)
4. Multi-layer perceptron (MLP)
5. Randomized prior network (RPN)
6. Conditional variational autoencoder (cVAE)

Jupyter Notebooks describing how to load and train the simple CNN and MLP models are found in the ```demo_notebooks/``` folder. The environments and code used to train each model, as well as the pre-trained models, are found in the ```baseline_models/``` folder.

The dataset used for the baseline models corresponds to the **Low-Resolution Real Geography** dataset. The subset of variables used to train our models is shown below:


| Input | Target | Variable | Description | Units | Dimensions |
| :---: | :----: | :------: | :---------: | :---: | :--------: |
| X |  | *T* | Air temperature | K | (lev, ncol) |
| X |  | *q* | Specific humidity | kg/kg | (lev, ncol) |
| X |  | PS | Surface pressure | Pa | (ncol) |
| X |  | SOLIN | Solar insolation | W/m&#x00B2; | (ncol) |
| X |  | LHFLX | Surface latent heat flux | W/m&#x00B2; | (ncol) |
| X |  | SHFLX | Surface sensible heat flux | W/m&#x00B2; | (ncol) |
|  | X | *dT/dt* | Heating tendency | K/s | (lev, ncol) |
|  | X | *dq/dt* | Moistening tendency | kg/kg/s | (lev, ncol) |
|  | X | NETSW | Net surface shortwave flux | W/m&#x00B2; | (ncol) |
|  | X | FLWDS | Downward surface longwave flux | W/m&#x00B2; | (ncol) |
|  | X | PRECSC | Snow rate | m/s | (ncol) |
|  | X | PRECC | Rain rate | m/s | (ncol) |
|  | X | SOLS | Visible direct solar flux | W/m&#x00B2; | (ncol) |
|  | X | SOLL | Near-IR direct solar flux | W/m&#x00B2; | (ncol) |
|  | X | SOLSD | Visible diffuse solar flux | W/m&#x00B2; | (ncol) |
|  | X | SOLLD | Near-IR diffuse solar flux | W/m&#x00B2; | (ncol) |


## Evaluation

Four different evaluation metrics were calculated:
1. Mean absolute error (MAE)
2. Coefficient of determination (R&#x00B2;)
3. Root mean squared error (RMSE)
4. Continuous ranked probability score (CRPS)

Evaluation and comparison of the different baseline models are found in the ```evaluation/``` folder. All variables are converted to a common energy unit (i.e., W/m&#x00B2;) for scoring. The scoring is done using the functions in ```evaluation/data_utils.py```. 

The ```evaluation/main_figure_generation.ipynb``` notebook calculates and plots MAE, R&#x00B2;, RMSE, and CRPS scores for each baseline model. The separate R&#x00B2; for *longitudinally-averaged* and time-averaged 3D variables is found in ```plot_R2_analysis.ipynb```.
