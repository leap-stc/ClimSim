# Baseline Models

Six different baseline models were created and trained:
1. Convolutional neural network (CNN)
2. Encoder-decoder (ED)
3. Heteroskedastic regression (HSR)
4. Multi-layer perceptron (MLP)
5. Randomized prior network (RPN)
6. Conditional variational autoencoder (cVAE)

There are Jupyter Notebooks that describe how to load and train the simple [CNN](./demo_notebooks/CNN) and [MLP](./demo_notebooks/MLP) models. The environments and code used to train each model, as well as the pre-trained models, are found in the [```baseline_models/```](https://github.com/leap-stc/ClimSim/tree/main/baseline_models) folder on GitHub.

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


