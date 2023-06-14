# ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators

This repository is the official implementation of "ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators". It contains all the code for downloding and processing the data as well as code for the baseline models in the paper.

![fig_1](./fig_1.png)


## Dataset Information

Data from the climate model simulations were saved at 20 minute intervals for 10 simulated years. Two netCDF files (input and output) are produced at each timestep, totaling 525,600 files for each configuration. We ran 3 configurations of the E3SM-MMF multi-scale climate model:

1. E3SM-MMF High-Resolution Real Geography
    - Horizontal Resolution: 1.5&deg; x 1.5&deg; (21,600 grid columns)
    - Total Samples: 5.7 billion
    - Total Data Volume: 41.2 TB
    - File Sizes: 102 MB per input file, 61 MB per output file
2. E3SM-MMF Low-Resolution Real Geography
    - Horizontal Resolution: 1.5&deg; x 1.5&deg; (384 grid columns)
    - Total Samples: 100 million
    - Total Data Volume: 744 GB
    - File Sizes: 1.9 MB per input file, 1.1 MB per output file
3. E3SM-MMF Low-Resolution Aquaplanet
    - Horizontal Resolution: 11.5&deg; x 11.5&deg; (384 grid columns)
    - Total Samples: 100 million
    - Total Data Volume: 744 GB
    - File Sizes: 1.9 MB per input file, 1.1 MB per output file

At each timestep, 2D variables vary in horizontal space, referred to as ``grid columns'' (ncol), and 3D variables vary additionally in vertical space (lev). The full list of variables can be found [here](https://docs.google.com/spreadsheets/d/1ljRfHq6QB36u0TuoxQXcV4_DSQUR0X4UimZ4QHR8f9M/edit#gid=0). The subset of variables used in our experiments is shown below:

| Input | Target | Variable | Description | Units | Dimensions |
| :---: | :----: | :------: | :---------: | :---: | :--------: |
| X |  | $T$ | Air temperature | $\text{K}$ | (ncol, lev) |
| X |  | $q$ | Specific humidity | $\text{kg/kg}$ | (ncol, lev) |
| X |  | $p_s$ | Surface pressure |$\text{Pa}$ | (ncol) |
| X |  | $\text{SOLIN}$ | Solar insolation | $\text{W/m<sup>2</sup>}$ | (ncol) |
| X |  | $\text{LHFLX}$ | Surface latent heat flux | W/m&#x00B2; | (ncol) |
| X |  | $\text{SHFLX}$ | Surface sensible heat flux | $\text{W/m^2}$ | (ncol) |
|  | X | $\text{dT/dt}$ | Heating tendency | $\text{K/s}$ | (ncol, lev) |
|  | X | $\text{dq/dt}$ | Moistening tendency | $\text{kg/kg/s}$ | (ncol, lev) |
|  | X | $\text{NETSW}$ | Net surface shortwave flux | $\text{W/m^2}$ | (ncol) |
|  | X | $\text{FLWDS}$ | Downward surface longwave flux | $\text{W/m^2}$ | (ncol) |
|  | X | $\text{PRECSC}$ | Snow rate | $\text{m/s}$ | (ncol) |
|  | X | $\text{PRECC}$ | Rain rate | $\text{m/s}$ | (ncol) |
|  | X | $\text{SOLS}$ | Visible direct solar flux | $\text{W/m^2}$ | (ncol) |
|  | X | $\text{SOLL}$ | Near-IR direct solar flux | $\text{W/m^2}$ | (ncol) |
|  | X | $\text{SOLSD}$ | Visible diffused solar flux | $\text{W/m^2}$ | (ncol) |
|  | X | $\text{SOLLd}$ | Near-IR diffused solar flux | $\text{W/m^2}$ | (ncol) |


## Download the Data

The data for all configurations of the multi-scale climate model (E3SM-MMF) can be downloaded from [Hugging Face](https://huggingface.co/sungduk):
- [High-resolution real geography dataset](https://huggingface.co/datasets/LEAP/ClimSim_high-res)
- [Low-resolution real geography dataset](https://huggingface.co/datasets/LEAP/ClimSim_low-res)
- [Low-resolution aquaplanet dataset](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet)

The files containing the normalization factors for the input and output data are found in the ```norm_factors/``` folder.
The file containing the grid information for E3SM-MMF is found in the ```grid_info/``` folder.


## Step 1: Preprocessing

Install the requirements needed for preprocessing from the ```/preprocessing/env/requirements.txt``` file.
Make the training dataset using ```/preprocessing/make_train_npy.ipynb```.
Make the validation dataset using ```/preprocessing/make_val_npy.ipynb```.
Make the scoring dataset using ```/preprocessing/make_val_stride6.ipynb```.


## Step 2: Training

We trained a total of 5 different models: 
- CNN
- HSP
- MLP
- RPN
- cVAE

Jupyter Notebooks describing how to load and train simple CNN and MLP models can be found in the ```demo_notebooks/``` folder.
The envrionments used to train each model, the code used to train each model, and the pre-trained models can be found in the repsecitve model folders within the ```basline_models``` folder.


## Step 3: Evaluation

To evaluate the trained model on benhcmarks reported in the paper,


## Step 4: Results

Evaluation metrics are computed separately for each global-mean, time-mean tagret variable. The performance for each of our basleine models is shown below:

\begin{table}[ht]
\centering
\small
\begin{tabular}{l|ccccc|ccccc}
\toprule
\multicolumn{1}{c|}{\multirow{2}{*}{\textbf{Variable}}} & \multicolumn{5}{c|}{\textbf{MAE [W/m$^2$]}} & \multicolumn{5}{c}{\textbf{R$^2$}} \\
\cmidrule{2-11}
\multicolumn{1}{c|}{} & CNN & HSR & MLP & RPN & cVAE & CNN & HSR & MLP & RPN & cVAE \\
\midrule
dT/dt & \textbf{2.585} & 2.845 & 2.683 & 2.685 & 2.732 & \textbf{0.627} & 0.568 & 0.589 & 0.617 & 0.590 \\
dq/dt & \textbf{4.401} & 4.784 & 4.495 & 4.592 & 4.680 & -- & -- & -- & -- & -- \\
NETSW & 18.85 & 19.82 & \textbf{13.36} & 18.88 & 19.73 & 0.944 & 0.959 & \textbf{0.983} & 0.968 & 0.957 \\
FLWDS & 8.598 & 6.267 & \textbf{5.224} & 6.018 & 6.588 & 0.828 & 0.904 & \textbf{0.924} & 0.912 & 0.883 \\
PRECSC & 3.364 & 3.511 & \textbf{2.684} & 3.328 & 3.322 & -- & -- & -- & -- & -- \\
PRECC & 37.83 & 42.38 & \textbf{34.33} & 37.46 & 38.81 & \textbf{0.077} & -68.35 & -38.69 & -67.94 & -0.926 \\
SOLS & 10.83 & 11.31 & \textbf{7.97} & 10.36 & 10.94 & 0.927 & 0.929 & \textbf{0.961} & 0.943 & 0.929 \\
SOLL & 13.15 & 13.60 & \textbf{10.30} & 12.96 & 13.46 & 0.916 & 0.916 & \textbf{0.948} & 0.928 & 0.915 \\
SOLSD & 5.817 & 6.331 & \textbf{4.533} & 5.846 & 6.159 & 0.927 & 0.923 & \textbf{0.956} & 0.940 & 0.921 \\
SOLLD & 5.679 & 6.215 & \textbf{4.806} & 5.702 & 6.066 & 0.813 & 0.797 & \textbf{0.866} & 0.837 & 0.796 \\
\bottomrule
\end{tabular}
\vspace{1mm}
\caption{\centering Summary statistics of global-mean, time-mean target variables for each baseline architecture.}
\label{tab:summarystats}
\end{table}



|  Model |  MAE  |  RMSE  |  $R^2$  |
| ------ | ----- | ------ | ------- |
|  cVAE  |       |        |         |
|  HSR   |       |        |         |
|  RPN   |       |        |         |
|  CNN   |       |        |         |
|  MLP   |       |        |         |
