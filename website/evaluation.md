# Model evaluation

Six different baseline models were created and trained:
1. Convolutional neural network (CNN)
2. Encoder-decoder (ED)
3. Heteroskedastic regression (HSR)
4. Multi-layer perceptron (MLP)
5. Randomized prior network (RPN)
6. Conditional variational autoencoder (cVAE)

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

