# ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators

[![Code Repository](https://img.shields.io/badge/-Code%20Repository-181717?logo=github&style=for-the-badge)](https://github.com/leap-stc/ClimSim/tree/main)

ClimSim is the largest-ever dataset designed for hybrid ML-physics research. It comprises multi-scale climate simulations, developed by a consortium of climate scientists and ML researchers. It consists of 5.7 billion pairs of multivariate input and output vectors that isolate the influence of locally-nested, high-resolution, high-fidelity physics on a host climate simulator’s macro-scale physical state.
The dataset is global in coverage, spans multiple years at high sampling frequency, and is designed such that resulting emulators are compatible with downstream coupling into operational climate simulators. 
We implement a range of deterministic and stochastic regression baselines to highlight the ML challenges and their scoring. The [data](https://huggingface.co/datasets/LEAP/ClimSim_high-res2) and [code](https://leap-stc.github.io/ClimSim) are released openly to support the development of hybrid ML-physics and high-fidelity climate simulations for the benefit of science and society.

![fig_1](./fig_1.png)

## Introductory video:

https://www.youtube.com/watch?v=M3Vz0zR1Auc

[![Alt text](https://img.youtube.com/vi/M3Vz0zR1Auc/0.jpg)](https://www.youtube.com/watch?v=M3Vz0zR1Auc)


