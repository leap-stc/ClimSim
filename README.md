[![Dataset: E3SM-MMF High-Resolution Real Geography](https://img.shields.io/badge/Dataset-%20High%20Resolution%20Real%20Geography-yellow?logo=🤗&style=flat-square)](https://huggingface.co/datasets/LEAP/ClimSim_high-res)
[![Dataset: E3SM-MMF Low-Resolution Real Geography](https://img.shields.io/badge/Dataset-%20Low%20Resolution%20Real%20Geography-yellow?logo=🤗&style=flat-square)](https://huggingface.co/datasets/LEAP/ClimSim_low-res)
[![Dataset: E3SM-MMF Low-Resolution Aquaplanet](https://img.shields.io/badge/Dataset-%20Low%20Resolution%20Aquaplanet-yellow?logo=🤗&style=flat-square)](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet)

# ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators

This repository contains the code necessary to download and preprocess the data, and create, train, and evaluate the baseline models in the paper.

![fig_1](./fig_1.png)

Introductory video:

https://www.youtube.com/watch?v=M3Vz0zR1Auc

## Dataset Information

Data from multi-scale climate model (E3SM-MMF) simulations were saved at 20-minute intervals for 10 simulated years. Two netCDF files---input and ouput (target)---are produced at each timestep, totaling 525,600 files for each configuration. 3 configurations of E3SM-MMF were run:

1. **High-Resolution Real Geography**
    - 1.5&deg; x 1.5&deg; horizontal resolution (21,600 grid columns)
    - 5.7 billion total samples (41.2 TB)
    - 102 MB per input file, 61 MB per output file
2. **Low-Resolution Real Geography**
    - 11.5&deg; x 11.5&deg; horizontal resolution (384 grid columns)
    - 100 million total samples (744 GB)
    - 1.9 MB per input file, 1.1 MB per output file
3. **Low-Resolution Aquaplanet**
    - 11.5&deg; x 11.5&deg; horizontal resolution (384 grid columns)
    - 100 million total samples (744 GB)
    - 1.9 MB per input file, 1.1 MB per output file

Scalar variables vary in time and horizontal space ("ncol"), while vertically-resolved variables vary additionally in vertical space ("lev"). The full list of variables can be found in Supplmentary Information Table 1. The subset of variables used in our experiments is shown below:

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

## Download the Data

The input ("mli") and target ("mlo") data for all E3SM-MMF configurations can be downloaded from Hugging Face:
- [High-Resolution Real Geography dataset](https://huggingface.co/datasets/LEAP/ClimSim_high-res)
- [Low-Resolution Real Geography dataset](https://huggingface.co/datasets/LEAP/ClimSim_low-res)
- [Low-Resolution Aquaplanet dataset](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet)

## Installation & setup

For preprocessing and evaluation, please install the `climsim_utils` python tools, by running the following code from the root of this repo:

```
pip install .
```

If you already have all `climsim_utils` dependencies (`tensorflow`, `xarray`, etc.) installed in your local environment, you can alternatively run:

```
pip install . --no-deps
```


## Preprocess the Data

The default preprocessing workflow takes folders of monthly data from the climate model simulations, and creates normalized NumPy arrays for input and target data for training, validation, and scoring. These NumPy arrays are called ```train_input.npy```, ```train_target.npy```, ```val_input.npy```, ```val_target.npy```, ```scoring_input.npy```, and ```scoring_target.npy```. An option to strictly use a data loader and avoid converting into NumPy arrays is available in ```data_utils.py```; however, this can slow down training because of increased I/O.

The data comes in the form of folders labeled ```YYYY-MM```, which corresponds to the simulation year (```YYYY```) and month (``MM``). Within each of these folders are netCDF (.nc) files that represent inputs and targets for individual timesteps. Input files are labeled ```E3SM-MMF.mli.YYYY-MM-DD-SSSSS.nc``` where ```DD-SSSSS``` corresponds to the day of the month (``DD``) and seconds of the day (```SSSSS```), with timesteps being spaced 1,200 seconds (20 minutes) apart. Target files are labeled the same way, except ```mli``` is replaced by ```mlo```. For vertically-resolved variables, lower indices corresponds to higher levels in the atmosphere. This is because pressure decreases monotonically with altitude.   

The files containing the default normalization factors for the input and target data are found in the ```norm_factors/``` folder, precomputed for convenience. However, one can use their own normalization factors if desired. The file containing the E3SM-MMF grid information is found in the ```grid_info/``` folder. This corresponds to the netCDF file ending in ```grid-info.nc``` on Hugging Face.

The environment needed for preprocessing can be found in the ```/preprocessing/env/requirements.txt``` file. A class designed for preprocessing and metrics can be imported from the ```data_utils.py``` script. This script is used in the ```preprocessing/create_npy_data_splits.ipynb``` notebook, which creates training, validation, and scoring datasets.

By default, training and validation data subsample every 7$^{\text{th}}$ timestep while scoring data subsamples every 6$^{\text{th}}$  timestep to enable daily-averaged metrics. Training data is taken from the second month of simulation year 1 through the first month of simulation year 8 (i.e., 0001-02 through 0008-01). Both validation and scoring data are taken from 0008-02 through 0009-01. However, the ```data_utils.py``` allows the user to easily change these defaults assuming knowledge of regular expressions. To see how this works, please reference ```preprocessing/create_npy_data_splits.ipynb```.

## Baseline Models

Six different baseline models were created and trained:
1. Convolutional neural network (CNN)
2. Encoder-decoder (ED)
3. Heteroskedastic regression (HSR)
4. Multi-layer perceptron (MLP)
5. Randomized prior network (RPN)
6. Conditional variational autoencoder (cVAE)

Jupyter Notebooks describing how to load and train simple CNN and MLP models are found in the ```demo_notebooks/``` folder. The environments and code used to train each model, as well as the pre-trained models, are found in the ```baseline_models/``` folder.

## Evaluation

Four different evaluation metrics were calculated:
1. Mean absolute error (MAE)
2. Coefficient of determination (R&#x00B2;)
3. Root mean squared error (RMSE)
4. Continuous ranked probability score (CRPS)

Evaluation and comparison of the different baseline models are found in the ```metrics_and_figures/``` folder. All variables are converted to a common energy unit (i.e., W/m&#x00B2;) for scoring. The scoring is done using the functions in ```metrics_and_figures/data_utils.py```. 

Evaluation metrics are computed separately for each horizontally-averaged, vertically-averaged, and time-averaged target variable. The performance for each baseline model for all four metrics is shown below:

| **MAE (W/m&#x00B2;)** | CNN | ED | HSR | MLP | RPN | cVAE |
| --------------------- | --- | --- | --- | --- | --- | ---- |
| *dT/dt* | **2.585** | 2.684 | 2.845 | 2.683 | 2.685 | 2.732 |
| *dq/dt* | **4.401** | 4.673 | 4.784 | 4.495 | 4.592 | 4.680 |
| NETSW | 18.85 | 14.968 | 19.82 | **13.36** | 18.88 | 19.73 |
| FLWDS | 8.598 | 6.894 | 6.267 | **5.224** | 6.018 | 6.588 |
| PRECSC | 3.364 | 3.046 | 3.511 | **2.684** | 3.328 | 3.322 |
| PRECC | 37.83 | 37.250 | 42.38 | **34.33** | 37.46 | 38.81 |
| SOLS | 10.83 | 8.554 | 11.31 | **7.97** | 10.36 | 10.94 |
| SOLL | 13.15 | 10.924 | 13.60 | **10.30** | 12.96 | 13.46 |
| SOLSD | 5.817 | 5.075 | 6.331 | **4.533** | 5.846 | 6.159 |
| SOLLD | 5.679 | 5.136 | 6.215 | **4.806** | 5.702 | 6.066 |

| **R&#x00B2;** | CNN | ED | HSR | MLP | RPN | cVAE |
| --------------------- | --- | --- | --- | --- | --- | ---- |
| *dT/dt* | **0.627** | 0.542 | 0.568 | 0.589 | 0.617 | 0.590 |
| *dq/dt* | -- | -- | -- | -- | -- | -- |
| NETSW | 0.944 | 0.980 | 0.959 | **0.983** | 0.968 | 0.957 |
| FLWDS | 0.828 | 0.802 | 0.904 | **0.924** | 0.912 | 0.883 |
| PRECSC | -- | -- | -- | -- | -- | -- |
| PRECC | **0.077** | -17.909 | -68.35 | -38.69 | -67.94 | -0.926 |
| SOLS | 0.927 | 0.960 | 0.929 | **0.961** | 0.943 | 0.929 |
| SOLL | 0.916 | 0.945 | 0.916 | **0.948** | 0.928 | 0.915 |
| SOLSD | 0.927 | 0.951 | 0.923 | **0.956** | 0.940 | 0.921 |
| SOLLD | 0.813 | 0.857 | 0.797 | **0.866** | 0.837 | 0.796 |

| **RMSE (W/m&#x00B2;)** | CNN | ED | HSR | MLP | RPN | cVAE |
| ---------------------- | --- | --- | --- | --- | --- | ---- |
| *dT/dt* | **4.369** | 4.696 | 4.825 | 4.421 | 4.482 | 4.721 |
| *dq/dt* | **7.284** | 7.643 | 7.896 | 7.322 | 7.518 | 7.780 |
| NETSW | 36.91 | 28.537 | 37.77 | **26.71** | 33.60 | 38.36 |
| FLWDS | 10.86 | 9.070 | 8.220 | **6.969** | 7.914 | 8.530 |
| PRECSC | 6.001 | 5.078 | 6.095 | **4.734** | 5.511 | 6.182 |
| PRECC | 85.31 | 76.682 | 90.64 | **72.88** | 76.58 | 88.71 |
| SOLS | 22.92 | 17.999 | 23.61 | **17.40** | 20.61 | 23.27 |
| SOLL | 27.25 | 22.540 | 27.78 | **21.95** | 25.22 | 27.81 |
| SOLSD | 12.13 | 9.917 | 12.40 | **9.420** | 11.00 | 12.64 |
| SOLLD | 12.10 | 10.417 | 12.47 | **10.12** | 11.25 | 12.63 |

| **CRPS (W/m&#x00B2;)** | CNN | ED | HSR | MLP | RPN | cVAE |
| ---------------------- | --- | --- | --- | --- | --- | ---- |
| *dT/dt* | -- | -- | 3.284 | -- | **2.580** | 2.795 |
| *dq/dt* | -- | -- | 4.899 | -- | **4.022** | 4.372 |
| NETSW | -- | -- | 0.055 | -- | **0.053** | 0.057 |
| FLWDS | -- | -- | 0.018 | -- | **0.016** | 0.018 |
| PRECSC | -- | -- | 0.011 | -- | **0.008** | 0.009 |
| PRECC | -- | -- | 0.122 | -- | **0.085** | 0.097 |
| SOLS  | -- | -- | 0.031 | -- | **0.028** | 0.033 |
| SOLL  | -- | -- | 0.038 | -- | **0.035** | 0.040 |
| SOLSD | -- | -- | 0.018 | -- | **0.015** | 0.016 |
| SOLLD | -- | -- | 0.017 | -- | **0.015** | 0.016 |

The ```metrics_and_figures/ClimSim_metrics.ipynb``` and ```metrics_and_figures/crps_clean.py``` scripts calculate and plot MAE, R&#x00B2;, RMSE, and CRPS scores for each baseline model. The separate R&#x00B2; for *longitudinally-averaged* and time-averaged 3D variables is found in ```plot_R2_analysis.ipynb```.
