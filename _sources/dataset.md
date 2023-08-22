# Dataset Information

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
Scalar variables vary in time and "horizontal" grid (`ncol`), while vertically-resolved variables vary additionally in vertical space (`lev`).  For vertically-resolved variables, lower indices of `lev` corresponds to higher levels in the atmosphere. This is because pressure decreases monotonically with altitude.   

The full list of variables can be found in [Supplementary Information](https://arxiv.org/pdf/2306.08754.pdf), Table 1.

There is also a [**Quickstart dataset**](https://huggingface.co/datasets/LEAP/subsampled_low_res) that contains subsampled and prenormalized data. This data was used for training, validation, and metrics for the ClimSim paper and can be reproduced using the [`preprocessing/create_npy_data_splits.ipynb`](https://github.com/leap-stc/ClimSim/tree/main/preprocessing/create_npy_data_splits.ipynb) notebook.


