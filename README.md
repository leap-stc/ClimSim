# ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators

This repository is the official implementation of "ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators" (add URL). It contains all the code for downloding and processing the data as well as code for the baseline models in the paper.

![fig_1](./fig_1.png)

## Requirements

To install requirements:
```
pip install -r requirements.txt
```
For more information on how to set up the environment, see see **this folder/file**. 

## Download the Data

The data for all model configuration is hosted on [Hugging Face](https://huggingface.co/sungduk):
- [High-resolution real geography dataset](https://huggingface.co/datasets/sungduk/E3SM-MMF_ne30)
- [Low-resolution real geography dataset](https://huggingface.co/datasets/sungduk/E3SM-MMF_ne4)
- [Low-resolution aquaplanet dataset](https://huggingface.co/datasets/sungduk/E3SM-MMF_ne4_aq)

Download each dataset using:
```
wget '<dataset URL>'
```
For more information about the data itself and how to download it, see **this folder/file**.

## Training

To train the model(s) in the paper, run this command:
```
train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```
For more information about how our models were trained, the full training procedure, and appropriate hyperparameters, see **this folder/file**.

## Evaluation

To evaluate **<model name>**, run:
```
eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```
For more information on how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-Trained Models

You can download pretrained models here:

- **[My model](https://drive.google.com/mymodel.pth)** trained on ClimSim using parameters **<X, Y, Z>**. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our models achieve the following performance on:

### Model Evaluation ClimSim (add URL)

|  Model |  MAE  |  RMSE  |  $R^2$  |
| ------ | ----- | ------ | ------- |
|  cVAE  |       |        |         |
|  HSR   |       |        |         |
|  RPN   |       |        |         |
|  CNN   |       |        |         |
|  MLP   |       |        |         |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributing

>📋  Pick a license and describe how to contribute to your code repository. 
