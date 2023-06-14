# ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators

This repository is the official implementation of "ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators". It contains all the code for downloading and processing the data, as well as code for the baseline models in the paper.

![fig_1](./fig_1.png)

## Dataset Information

Data from the mutli-scale climate model (E3SM-MMF) simulations were saved at 20 minute intervals for 10 simulated years. Two netCDF files (input and output) are produced at each timestep, totaling 525,600 files for each configuration. We ran 3 configurations of E3SM-MMF:

1. **High-Resolution Real Geography**
    - Horizontal Resolution: 1.5&deg; x 1.5&deg; (21,600 grid columns)
    - Total Samples: 5.7 billion
    - Total Data Volume: 41.2 TB
    - File Sizes: 102 MB per input file, 61 MB per output file
2. **Low-Resolution Real Geography**
    - Horizontal Resolution: 1.5&deg; x 1.5&deg; (384 grid columns)
    - Total Samples: 100 million
    - Total Data Volume: 744 GB
    - File Sizes: 1.9 MB per input file, 1.1 MB per output file
3. **Low-Resolution Aquaplanet**
    - Horizontal Resolution: 11.5&deg; x 11.5&deg; (384 grid columns)
    - Total Samples: 100 million
    - Total Data Volume: 744 GB
    - File Sizes: 1.9 MB per input file, 1.1 MB per output file

2D variables vary in horizontal space, referred to as "grid columns" (ncol), and 3D variables vary additionally in vertical space (lev). The full list of variables can be found [here](https://docs.google.com/spreadsheets/d/1ljRfHq6QB36u0TuoxQXcV4_DSQUR0X4UimZ4QHR8f9M/edit#gid=0). The subset of variables used in the experiments is shown below:

| Input | Target | Variable | Description | Units | Dimensions |
| :---: | :----: | :------: | :---------: | :---: | :--------: |
| X |  | T | Air temperature | K | (ncol, lev) |
| X |  | q | Specific humidity | kg/kg | (ncol, lev) |
| X |  | p&#x209B; | Surface pressure | Pa | (ncol) |
| X |  | SOLIN | Solar insolation | W/m&#x00B2; | (ncol) |
| X |  | LHFLX | Surface latent heat flux | W/m&#x00B2; | (ncol) |
| X |  | SHFLX | Surface sensible heat flux | W/m&#x00B2; | (ncol) |
|  | X | dT/dt | Heating tendency | K/s | (ncol, lev) |
|  | X | dq/dt | Moistening tendency | kg/kg/s | (ncol, lev) |
|  | X | NETSW | Net surface shortwave flux | W/m&#x00B2; | (ncol) |
|  | X | FLWDS | Downward surface longwave flux | W/m&#x00B2; | (ncol) |
|  | X | PRECSC | Snow rate | m/s | (ncol) |
|  | X | PRECC | Rain rate | m/s | (ncol) |
|  | X | SOLS | Visible direct solar flux | W/m&#x00B2; | (ncol) |
|  | X | SOLL | Near-IR direct solar flux | W/m&#x00B2; | (ncol) |
|  | X | SOLSD | Visible diffused solar flux | W/m&#x00B2; | (ncol) |
|  | X | SOLLD | Near-IR diffused solar flux | W/m&#x00B2; | (ncol) |

## Download the Data

The data for all E3SM-MMF configurations can be downloaded from Hugging Face:
- [High-resolution Real Geography dataset](https://huggingface.co/datasets/LEAP/ClimSim_high-res)
- [Low-resolution Real Geography dataset](https://huggingface.co/datasets/LEAP/ClimSim_low-res)
- [Low-Resolution Aquaplanet dataset](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet)

## Step 1: Preprocessing

The preprocessing workflow is as follows:
- Downsample in time by using every seventh sample
- Collapse horizontal location and time into a single sample dimension
- Normalize variables by subtracting the mean and dividing by the range, with these statistics calculated separately at each of the 60 vertical levels for 3D variables
- Concatenate variables into multi-variate input and output vectors for each sample

Install the requirements needed for preprocessing from the ```/preprocessing/env/requirements.txt``` file. Make the training dataset using ```preprocessing/make_train_npy.ipynb```, the validation dataset using ```preprocessing/make_val_npy.ipynb```, and the scoring dataset using ``/preprocessing/make_val_stride6.ipynb```.

The files containing the normalization factors for the input and output data are found in the ```norm_factors/``` folder. The file containing the E3SM-MMF grid information is found in the ```grid_info/``` folder.

## Step 2: Training

We trained a total of 5 different models:
1. Convolutional neural network (CNN)
2. Heteroskedastic regression (HSR)
3. Multi-layer perceptron (MLP)
4. Randomized prior network (RPN)
5. Conditional variational autoencoder (cVAE)

Jupyter Notebooks describing how to load and train simple CNN and MLP models can be found in the ```demo_notebooks/``` folder. The environments and code used to train each model, as well as the  pre-trained models, can be found in the repsective model folders within the ```baseline_models/``` folder.

## Step 3: Evaluation

Evaluation metrics are computed separately for each global-mean, time-mean target variable. The performance for each baseline model is shown below:

| **Variable** | **MAE (W/m&#x00B2;)** | **R&#x00B2;** |
|--------------|-----------------|---------|
|              | CNN | HSR | MLP | RPN | cVAE | CNN | HSR | MLP | RPN | cVAE |
|--------------|-----|-----|-----|-----|------|-----|-----|-----|-----|------|
| dT/dt        | **2.585** | 2.845 | 2.683 | 2.685 | 2.732 | **0.627** | 0.568 | 0.589 | 0.617 | 0.590 |
| dq/dt        | **4.401** | 4.784 | 4.495 | 4.592 | 4.680 | -- | -- | -- | -- | -- |
| NETSW        | 18.85 | 19.82 | **13.36** | 18.88 | 19.73 | 0.944 | 0.959 | **0.983** | 0.968 | 0.957 |
| FLWDS        | 8.598 | 6.267 | **5.224** | 6.018 | 6.588 | 0.828 | 0.904 | **0.924** | 0.912 | 0.883 |
| PRECSC       | 3.364 | 3.511 | **2.684** | 3.328 | 3.322 | -- | -- | -- | -- | -- |
| PRECC        | 37.83 | 42.38 | **34.33** | 37.46 | 38.81 | **0.077** | -68.35 | -38.69 | -67.94 | -0.926 |
| SOLS         | 10.83 | 11.31 | **7.97** | 10.36 | 10.94 | 0.927 | 0.929 | **0.961** | 0.943 | 0.929 |
| SOLL         | 13.15 | 13.60 | **10.30** | 12.96 | 13.46 | 0.916 | 0.916 | **0.948** | 0.928 | 0.915 |
| SOLSD        | 5.817 | 6.331 | **4.533** | 5.846 | 6.159 | 0.927 | 0.923 | **0.956** | 0.940 | 0.921 |
| SOLLD        | 5.679 | 6.215 | **4.806** | 5.702 | 6.066 | 0.813 | 0.797 | **0.866** | 0.837 | 0.796 |

