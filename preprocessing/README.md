# Preprocessing the Data


The default preprocessing workflow takes folders of monthly data from the climate model simulations, and creates normalized NumPy arrays for input and target data for training, validation, and scoring. These NumPy arrays are called ```train_input.npy```, ```train_target.npy```, ```val_input.npy```, ```val_target.npy```, ```scoring_input.npy```, and ```scoring_target.npy```. An option to strictly use a data loader and avoid converting into NumPy arrays is available in ```data_utils.py```; however, this can slow down training because of increased I/O.

The files containing the default normalization factors for the input and target data are found in the ```normalizations/``` folder, precomputed for convenience. However, one can use their own normalization factors if desired. The file containing the E3SM-MMF grid information is found in the ```../grid_info/``` folder. This corresponds to the netCDF file ending in ```grid-info.nc``` on Hugging Face.

The environment needed for preprocessing can be found in the ```requirements-lock.txt``` file. A class designed for preprocessing and metrics can be imported from the ```../climsim_utils/data_utils.py``` script. This script is used in the ```create_npy_data_splits.ipynb``` notebook, which creates training, validation, and scoring datasets.

By default, training and validation data subsample every $7^{th}$ timestep while scoring data subsamples every $6^{th}$  timestep to enable daily-averaged metrics. Training data is taken from the second month of simulation year 1 through the first month of simulation year 8 (i.e., 0001-02 through 0008-01). Both validation and scoring data are taken from 0008-02 through 0009-01. However, the ```../climsim_utils/data_utils.py``` allows the user to easily change these defaults assuming knowledge of regular expressions. To see how this works, please reference ```create_npy_data_splits.ipynb```.


