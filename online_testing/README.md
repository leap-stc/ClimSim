# Hybrid E3SM-MMF-NN-Emulator Simulation and Online Evaluation

## Table of Contents

1. [Problem overview](#1-problem-overview)
2. [Data preparation](#2-data-preparation)
    1. [Data download](#21-data-download)
    2. [Combine raw data into a few single files](#22-combine-raw-data-into-a-few-single-files)
3. [Model training](#3-model-training)
    1. [General requirement](#31-general-requirement)
    2. [Training scripts of our baseline online models](#32-training-scripts-of-our-baseline-online-models)
4. [Model post-processing: create wrapper for the trained model to include any normalization and de-normalization](#4-model-post-processing-create-wrapper-for-the-trained-model-to-include-any-normalization-and-de-normalization)
5. [Run hybrid E3SM MMF-NN-Emulator simulation](#5-run-hybrid-e3sm-mmf-nn-emulator-simulation)
6. [Evaluation of hybrid simulation](#6-evaluation-of-hybrid-simulation)

## 1. Problem overview
The ultimate goal of training a ML model emulator (of the cloud-resolving model embedded in the E3SM-MMF climate simulator) using the ClimSim dataset is to couple it to the host E3SM climate simulator and evaluate the performance of such hybrid ML-physics simulation, e.g., whether the hybrid simulation can reproduce the statistics of the pure physics simulation. Here we use "online" to denote this task of performing and evaluating hybrid simulation, in contrast to the "offline" task in which we focus on training a ML model. Here we describe the entire workflow of training these baseline models, running and evaluating the hybrid simulation. We provided a few baseline models that we trained and optimized on the online task. These pretrained models include the MLP models and U-Net models from [Stable Machine-Learning Parameterization](https://arxiv.org/abs/2407.00124) paper.

Refer to the [ClimSim-Online paper](https://arxiv.org/abs/2306.08754) for more details on the online task overview and the [Stable Machine-Learning Parameterization](https://arxiv.org/abs/2407.00124) paper for more details on the example baseline models we provide.

---

## 2. Data preparation

### 2.1 Data download

We take the low-resolution dataset as example. Dowload either the [Low-Resolution Real Geography](https://huggingface.co/datasets/LEAP/ClimSim_low-res) or [Low-Resolution Real Geography Expanded](https://huggingface.co/datasets/LEAP/ClimSim_low-res-expanded) dataset from Hugging Face. The expanded version includes additional input features such as large-scale forcings and convection memory (previous steps state tendencies) that we used in our pretrained U-Net models (refer to [this paper](https://arxiv.org/abs/2407.00124) for more details). 

Please don't use the current preprocessed [Subsampled Low-Resolution Data](https://huggingface.co/datasets/LEAP/subsampled_low_res) which does not include cloud and wind tendencies in target variables. For online testing, we need the ML model to predict not only temperature and moisture tendencies but also these cloud and wind tendencies.

If you would like to work on the [High-Resolution Dataset]((https://huggingface.co/datasets/LEAP/ClimSim_high-res)) and also want to expand the input feature, you can follow [this notebook](./online_testing/data_preparation/adding_input_feature.ipynb) which illustrates how we created the expanded input features from the original low-resolution dataset.

### 2.2 Combine raw data into a few single files

The raw data contains a large number of individual data files outputted at each E3SM model time step. We need to aggregate these individual files into a few files containing data array for efficient training.

Take our MLP baseline model (from the [Stable Machine-Learning Parameterization](https://arxiv.org/abs/2407.00124) paper) for example. Run the [create_dataset_example_v2rh.ipynb](./data_preparation/create_dataset/create_dataset_example_v2rh.ipynb) notebook to prepare the input/output files for the MLP_v2rh model. 

If you want to reproduce the U-Net models from [Stable Machine-Learning Parameterization](https://arxiv.org/abs/2407.00124) paper, run the [create_dataset_example_v4.ipynb](./data_preparation/create_dataset/create_dataset_example_v4.ipynb) notebook to prepare the input/output files for the Unet_v4 model. Or run the [create_dataset_example_v5.ipynb](./data_preparation/create_dataset/create_dataset_example_v5.ipynb) notebook to prepare the input/output files for the Unet_v5 model. 'v4' is the unconstrained U-Net, while 'v5' is the constrained U-Net, please refer to original paper for more details.

**Note:** For both 'v4' and 'v5' input features in the [data_utils.py](../climsim_utils/data_utils.py), the code includes the following variables into the preprocessed input data array: ```'tm_state_ps', 'tm_pbuf_SOLIN', 'tm_pbuf_LHFLX', 'tm_pbuf_SHFLX', 'tm_pbuf_COSZRS', 'icol'```. These features are not yet implemented in the online E3SM code, so during the U-Net training, these input features were set to 0 at the beginning of the forward method of the U-Net model (i.e., they are not used in U-Net training). To be comopatible with the E3SM code, you should exclude these features from the input feature array when you train your own model.

---

## 3. Model training

### 3.1 General requirement

To be able to couple your trained NN model to E3SM seeminglessly, you need to be aware of the following requirements before training your NN model:

- Your NN model must be saved in TorchScript format. Converting a pytorch model into TorchScript is straightforward. Our training scripts include the code to save the model in TorchScript format. You can also refer to the [Official Torchscript Documentation](https://pytorch.org/docs/stable/jit.html) for more details.
- Your NN model's forward method should take an input tensor with shape (batch_size, num_input_features) and return an output tensor with shape (batch_size, num_output_features). The output feature dimension should have a length of ```num_output_features = 368``` and contain the following variables in the same order as: ```'ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD'```. The ptend variables are vertical profiles of tendencies of atmospheric states and have a length of 60.

### 3.2 Training scripts of our baseline online models

We provide the training scripts under the ```online_testing/baseline_models/``` directory. Under the folder of each baseline model, we provide the slurm scripts under the ```slurm``` folder to run the training job.

For example, to train the MLP model (with a huber loss and a 'step' lr scheduler), you can run the following command:
```bash
cd online_testing/baseline_models/MLP_v2rh/training/slurm/
sbatch v2rh_mlp_nonaggressive_cliprh_huber_step_3l_lr1em3.sbatch
```

The training will read in the default configuration arguments listed in ```training/conf/config_single.yaml```. You need to change a few path argument in the config_single.yaml to the paths on your machine, or you can also overwrite those paths in the slurm job scripts. By default, the training slurm scripts requested to use 4 GPUs. You can change the number of GPUs in the slurm scripts.

The training requires to use the [modulus library](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html). We used the modulus container image for the training environment. You could download the latest version by following the instructions on [modulus website](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html). For reproducibility information, we used version ```nvcr.io/nvidia/modulus/modulus:24.01```. If you don't want to use a container, you could also use

```bash
pip install nvidia-modulus
``` 
to install on any system but we recommend the container for best results.

---

## 4 Model post-processing: create wrapper for the trained model to include any normalization and de-normalization

The E3SM MMF-NN-Emulator code expects the NN model to take un-normalized input features and output un-normalized output features. Notebooks provided in ```./model_postprocessing``` directory show how to create a wrapper for our pretrained MLP and U-Net models to include pre/post-processing such as normalization and de-normalization inside the forward method of the TorchScript model. 

For example, the [v5_nn_wrapper.ipynb](./model_postprocessing/v5_nn_wrapper.ipynb) notebook shows how to create a wrapper for the U-Net model to read raw input features, calculate additional needed input features, normalize the input, clip input values, pass them to the U-Net model, de-normalize the output features, and apply the temperature-based liquid-ice cloud partitioning.

Currently two input configurations are supported in the E3SM MMF-NN Emulator code: 'v2rh' and 'v4' (see the v2_rh_inputs and v4_inputs in [data_utils.py](../climsim_utils/data_utils.py)). If you did not use these two configurations during training, your wrapper function should internally convert the input array first to be compatible with your input features. For example, in the [v5_nn_wrapper.ipynb](./model_postprocessing/v5_nn_wrapper.ipynb) notebook, the wrapper NN function will still take an v4 input array but will internally convert it to a v5 input array before passing it to the saved U-Net model. 

Again, note that in the 'v4' input configuration, the input features include the following variables: ```'tm_state_ps', 'tm_pbuf_SOLIN', 'tm_pbuf_LHFLX', 'tm_pbuf_SHFLX', 'tm_pbuf_COSZRS', 'icol'```. These features are not yet implemented in the online E3SM code, you should exclude these features from the input feature array when you train your own model. During our U-Net training, these input features were set to 0 at the beginning of the forward method of the U-Net model (i.e., they are not used in U-Net training)

---

## 5. Run hybrid E3SM MMF-NN-Emulator simulations

Please follow the instructions in the [ClimSim-Online repository](https://github.com/leap-stc/climsim-online/tree/main) to set up the container environment and run the hybrid simulation.

Please check the [NVlabs/E3SM MMF-NN-Emulator repository](https://github.com/zyhu-hu/E3SM_nvlab/tree/cleaner_workflow_tomerge/climsim_scripts) to learn about the configurations and namelist variables of the E3SM MMF-NN-Emulator version.

---

## 6. Evaluation of hybrid simulations

The notebooks in the ```./evaluation``` directory show how to reproduce the plots in the [Stable Machine-Learning Parameterization](https://arxiv.org/abs/2407.00124) paper. Data required by these evaluation/visualization notebooks can be downloaded at [Stable Machine-Learning Parameterization: Zenodo Data](https://zenodo.org/records/12797811).

---

## Author
- Zeyuan Hu, Harvard University

## References

- [ClimSim-Online: A Large Multi-scale Dataset and Framework for Hybrid ML-physics Climate Emulation](https://arxiv.org/abs/2306.08754)
- [Stable Machine-Learning Parameterization of Subgrid Processes with Real Geography and Full-physics Emulation](https://arxiv.org/abs/2407.00124)