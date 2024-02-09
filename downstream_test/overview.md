# Downstream task overview

Once we train a ML model, we want to couple it to the host E3SM climate simulator and evaluate the performance of such hybrid ML-physics simulation, e.g., whether the hybrid simulation can reproduce the statistics of the pure physics simulation. This task is called the downstream task. We provided a few baseline models that we trained and optimized on the downstream task. Here we show how to reproduce these models and how to do the downstream evaluation.

## Expanding input features

[This notebook](./expand_feature/adding_input_feature.ipynb) expand the input data file to incorporate additional features from previous time steps. For each sample, the input file was originally saved in a file named as '.mli.'. This notebook will create a new file for each sample that include both the original features and the additional features listed in the table below. This notebooks require that the data is continuous in time instead of subsampled with a stride > 1. You can download the full data here: [Low-Resolution Real Geography](https://huggingface.co/datasets/LEAP/ClimSim_low-res).

| **Variable**                   | **Units**      | **Description**                               | **Normalization**          |
|--------------------------------|----------------|-----------------------------------------------|----------------------------|
| $RH(z)$                        |                | Relative humidity                             |                            |
| $liq\_partition(z)$            |                | Fraction of liquid cloud                      |                            |
| $q_n(z)$                       | kg/kg          | Total cloud (liquid + ice) mixing ratio       | 1 - exp(-$\lambda x$)      |
| $dT_{adv}(z,t_0,t_{-1})$       | K/s            | Large-scale forcing of temperature            | x/(max-min)                |
| $dq_{T,adv}(z,t_0,t_{-1})$     | kg/kg/s        | Large-scale forcing of total water            | x/(max-min)                |
| $du_{adv}(z,t_0,t_{-1})$       | m/s$^2$ | Large-scale forcing of zonal wind        | x/(max-min)                |
| $dT(z,t_{-1},t_{-2})$          | K/s            | Temperature tendency                          | x/std                      |
| $dq_v(z,t_{-1},t_{-2})$        | kg/kg/s        | Water vapor tendency                          | x/std                      |
| $dq_n(z,t_{-1},t_{-2})$        | kg/kg/s        | Total cloud tendency                          | x/std                      |
| $dq_c(z,t_{-1},t_{-2})$        | kg/kg/s        | Liquid cloud tendency                         | x/std                      |
| $dq_i(z,t_{-1},t_{-2})$        | kg/kg/s        | Ice cloud tendency                            | x/std                      |
| $du(z,t_{-1},t_{-2})$          | m/s$^2$ | Zonal wind tendency                      | x/std                      |
| cos(lat)                       |                | Cosine of latitude                            |                            |
| sin(lat)                       |                | Sine of latitude                              |                            |

Below is a list of other variables that are used and already exist in the original input files.
| **Variable**                   | **Units**      | **Description**                               | **Normalization**          |
|--------------------------------|----------------|-----------------------------------------------|----------------------------|
| $T(z)$                         | K              | Temperature                                   | (x-mean)/(max-min)         |
| $q_c(z)$                       | kg/kg          | Liquid cloud mixing ratio                     | 1 - exp(-$\lambda x$)      |
| $q_i(z)$                       | kg/kg          | Ice cloud mixing ratio                        | 1 - exp(-$\lambda x$)      |
| $u(z)$                         | m/s            | Zonal wind                                    | (x-mean)/(max-min)         |
| $v(z)$                         | m/s            | Meridional wind                               | (x-mean)/(max-min)         |
| O3$(z)$                        | mol/mol        | Ozone volume mixing ratio                     | (x-mean)/(max-min)         |
| CH4$(z)$                       | mol/mol        | Methane volume mixing ratio                   | (x-mean)/(max-min)         |
| N2O$(z)$                       | mol/mol        | Nitrous volume mixing ratio                   | (x-mean)/(max-min)         |
| PS                             | Pa             | Surface pressure                              | (x-mean)/(max-min)         |
| SOLIN                          | W/m$^2$ | Solar insolation                         | x/(max-min)                |
| LHFLX                          | W/m$^2$ | Surface latent heat flux                  | x/(max-min)                |
| SHFLX                          | W/m$^2$ | Surface sensible heat flux                 | x/(max-min)                |
| TAUX                           | W/m$^2$ | Zonal surface stress                      | (x-mean)/(max-min)         |
| TAUY                           | W/m$^2$ | Meridional surface stress                 | (x-mean)/(max-min)         |
| COSZRS                         |                | Cosine of solar zenith angle                  | (x-mean)/(max-min)         |
| ALDIF                          |                | Albedo for diffuse longwave radiation         | (x-mean)/(max-min)         |
| ALDIR                          |                | Albedo for direct longwave radiation          | (x-mean)/(max-min)         |
| ASDIF                          |                | Albedo for diffuse shortwave radiation        | (x-mean)/(max-min)         |
| ASDIR                          |                | Albedo for direct shortwave radiation         | (x-mean)/(max-min)         |
| LWUP                           | W/m$^2$ | Upward longwave flux                      | (x-mean)/(max-min)         |
| ICEFRAC                        |                | Sea-ice area fraction                         |                            |
| LANDFRAC                       |                | Land area fraction                            |                            |
| OCNFRAC                        |                | Ocean area fraction                           |                            |
| SNOWHLAND                      | m              | Snow depth over land                          | (x-mean)/(max-min)         |

## Aggregating and Preprocessing Sample Files

[This notebook](./create_dataset/create_dataset_example_v4.ipynb) integrate individual sample files into one/a few files containing data array for the Unet model without cloud physics constraints (Unet_v4). [This notebook](./create_dataset/create_dataset_example_v5.ipynb) shows how to prepare input/output files for the Unet with cloud physics constraints (Unet_v5). The difference is that Unet_v4 predict liquid and ice cloud separately and does use total cloud features like $q_n$ and $dq_n(z,t_{-1},t_{-2})$ and $liq\_partition(z)$. For the Unet_v5, we only predict total cloud and do not $q_c$, $q_i$, $dq_c$, and $dq_i$. In case you have access to the Perlmutter cluster, the v4 and v5 training data is also currently located at ```/pscratch/sd/z/zeyuanhu/hugging/E3SM-MMF_ne4/preprocessing/v4_full/``` and ```/pscratch/sd/z/zeyuanhu/hugging/E3SM-MMF_ne4/preprocessing/v5_full/```.

## Normalization

Data preprocessing is required for the raw data before we send them to a neural net. The exact normalization we used for each input feature is listed in the two tables above. This requires as to calculate the mean, min, and max for each variable at each level. [This notebook](./normalization/input_scaling.ipynb) shows how we generate input scaling files that contain the mean/max/min for each variable. One thing to note is that, most but not all the input features are normalized by (x-mean)/(max-min). For variables using (x-mean)/(max-min), we calculate mean,max,min per-level and save as usual. For variables with blank normalization, we simply set mean=0, max=1, min=0. For variables using x/std, we set mean=0, max=1/std, min=0. For variables using x/(max-min), we set mean = 0 and save max/min as usual. For cloud (liquid, ice, and total cloud) input, we set mean=0, max=1, min=0. We have a separate exponential transformation for cloud mass features. See [This notebook](./normalization/cloud_exponential_transformation.ipynb) for how to do the expoennetial transformation).

[This notebook](./normalization/output_scaling.ipynb) shows how we generate output scaling files. The output target features are normalized by per-level standard deviation, $y' = y/max(std, \epsilon)$, where $\epsilon$ is a small threshold to prevent dividing by tiny values. 

For reproducibility purpose, these scaling files are provided under ```preprocessing/normalizations```.

## Training script

After we prepared the preprocessed training data and the normalization files, we can start to run the training code. The training scripts can be found in ```downstream_test/baseline_models/```. We provide MLP_v2rh (MLP using v2 input features, i.e., no previous step information, and replace specific humidity with relative humidity in the input), Unet_v4 (Unet without cloud physics constraint), and Unet_v5 (Unet with cloud physics constraint). The training was done on the Perlmutter computing cluster from National Energy Research Scientific Computing Center (NERSC). Under the slurm folder (e.g., ```downstream_test/baseline_models/Unet_v5/training/slurm/```) in each model directory, we contains the slurm scripts to run the training job on the Perlmutter cluster. The training will read in the default configuration arguments listed in ```training/conf/config_single.yaml```. Don't forget to change a few path argument in the config_single.yaml to the paths on your machine, or you can also overwrite those paths in the slurm job scripts.

The training requires to use the [modulus library](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html). We used the modulus container image for the training environment. You could download the latest version by following the instructions on [modulus website](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html). For reproducibility information, we used version ```nvcr.io/nvidia/modulus/modulus:24.01```. For computing clusters that do not support docker image, you may need to use singularity or other available tools. Please refer to your cluster information page for how you can use a container image. If you don't want to use a container, you could also ```pip install nvidia-modulus``` on any system but we recommend the container for best results.

Once you obtain the container image, you would also need to open the image and install tensorflow. Tensorflow is only used in the data_utils.py and is not used in training. On the Perlmutter cluster, we run the container image and do the pip install as below:
```
shifter --image=nvcr.io/nvidia/modulus/modulus:24.01 /bin/bash
pip install tensorflow
```
The Perlmutter cluster used "shifter" to manage containers and may not be your case. Please refer to your cluster information for how to use a container image. pip will by default install the package under ```$HOME/.local/bin/```. You can add ```export PATH=$PATH:/path_to_your_home_directory/.local/bin``` in your .bashrc to ensure that the installed packages are accessible every time you use the container, without needing to reinstall them. If you find other libraries are missing but required, please report to us, and you may also use pip to install any missing libraries. The current training script will sync the training log to wandb. You'll be asked your wandb information the first time you run the training script.