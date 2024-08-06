[![Dataset: E3SM-MMF High-Resolution Real Geography](https://img.shields.io/badge/Dataset-%20High%20Resolution%20Real%20Geography-yellow?logo=🤗&style=flat-square)](https://huggingface.co/datasets/LEAP/ClimSim_high-res)
[![Dataset: E3SM-MMF Low-Resolution Real Geography](https://img.shields.io/badge/Dataset-%20Low%20Resolution%20Real%20Geography-yellow?logo=🤗&style=flat-square)](https://huggingface.co/datasets/LEAP/ClimSim_low-res)
[![Dataset: E3SM-MMF Low-Resolution Aquaplanet](https://img.shields.io/badge/Dataset-%20Low%20Resolution%20Aquaplanet-yellow?logo=🤗&style=flat-square)](https://huggingface.co/datasets/LEAP/ClimSim_low-res_aqua-planet)

# ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators

ClimSim is the largest-ever dataset designed for hybrid ML-physics research. It comprises multi-scale climate simulations, developed by a consortium of climate scientists and ML researchers. It consists of 5.7 billion pairs of multivariate input and output vectors that isolate the influence of locally-nested, high-resolution, high-fidelity physics on a host climate simulator’s macro-scale physical state.
The dataset is global in coverage, spans multiple years at high sampling frequency, and is designed such that resulting emulators are compatible with downstream coupling into operational climate simulators. 
We implement a range of deterministic and stochastic regression baselines to highlight the ML challenges and their scoring. 

![fig_1](figures/fig_1.png)

## Getting Started

* [Quickstart](https://leap-stc.github.io/ClimSim/quickstart.html)
* [Dataset Information](https://leap-stc.github.io/ClimSim/dataset.html)
* [Code Installation](https://leap-stc.github.io/ClimSim/installation.html)

## Models and Evaluation

* [Baseline Models](https://leap-stc.github.io/ClimSim/models.html)
* [Evaluation](https://leap-stc.github.io/ClimSim/evaluating.html)

## Demo Notebooks

* [Multi-Layer Perceptron (MLP) Example](https://leap-stc.github.io/ClimSim/demo_notebooks/mlp_example.html)
* [Convolutional Neural Network (CNN) Example](https://leap-stc.github.io/ClimSim/demo_notebooks/cnn_example.html)
* [Water Conservation Example](https://leap-stc.github.io/ClimSim/demo_notebooks/water_conservation.html)

## Online Testing

* [Online Testing](./online_testing/README.md)

## Project Structure

[![Code Repository](https://img.shields.io/badge/-Code%20Repository-181717?logo=github&style=for-the-badge)](https://github.com/leap-stc/ClimSim/tree/main)

* [GitHub Repository Structure](./ARCHITECTURE.md)


## References

* [ClimSim paper](https://arxiv.org/abs/2306.08754)
* [Recorded NeurIPS 2023 talk](https://www.youtube.com/watch?v=Wa1HXB_chYg)
[![YouTube video](https://img.youtube.com/vi/Wa1HXB_chYg/0.jpg)](https://www.youtube.com/watch?v=Wa1HXB_chYg)

* [Contributor Guide](https://leap-stc.github.io/ClimSim/CONTRIBUTING.html)


## Legal

ClimSim uses the Apache 2.0 license for code found on the associated GitHub repo and the Creative Commons Attribution 4.0 license for data hosted on HuggingFace. The [LICENSE file](https://github.com/leap-stc/ClimSim/blob/main/LICENSE) for the repo can be found in the top-level directory.
