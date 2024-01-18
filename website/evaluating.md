# Evaluation

Four different evaluation metrics were calculated:
1. Mean absolute error (MAE)
2. Coefficient of determination (R&#x00B2;)
3. Root mean squared error (RMSE)
4. Continuous ranked probability score (CRPS)

Evaluation and comparison of the different baseline models are found in the [```evaluation/```](https://github.com/leap-stc/ClimSim/tree/main/evaluation) folder on GitHub. All variables are converted to a common energy unit (i.e., W/m&#x00B2;) for scoring. The scoring is done using the functions in [`climsim_utils/data_utils.py`](https://github.com/leap-stc/ClimSim/tree/main/climsim_utils). 

[This notebook](./evaluation/main_figure_generation.ipynb) calculates and plots MAE, R&#x00B2;, RMSE, and CRPS scores for each baseline model. The separate R&#x00B2; for *longitudinally-averaged* and time-averaged 3D variables is found in [this notebook](./evaluation/plot_R2_analysis.ipynb).

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
| *dT/dt* | -- | -- | **2.158** | -- | 2.305 | 2.708 |
| *dq/dt* | -- | -- | **3.645** | -- | 4.100 | 4.565 |
| NETSW | -- | -- | **14.62** | -- | 14.82 | 20.53 |
| FLWDS | -- | -- | 4.561 | -- | **4.430** | 6.732 |
| PRECSC | -- | -- | 2.905 | -- | **2.729** | 3.513 |
| PRECC | -- | -- | 34.30 | -- | **30.08** | 40.17 |
| SOLS  | -- | -- | 8.369 | -- | **8.309** | 11.91 |
| SOLL  | -- | -- | **10.14** | -- | 10.49 | 14.42 |
| SOLSD | -- | -- | 4.773 | -- | **4.649** | 5.945 |
| SOLLD | -- | -- | **4.599** | -- | 4.682 | 5.925 |

