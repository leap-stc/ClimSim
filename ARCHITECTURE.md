## Project structure

```
├── .github                   <- Github Actions workflows
│
├── baseline_models           <- Pretrained baseline models, as well as python scripts and dependencies
|                                for re-training the models 
│   ├── CNN                      <- Convolutional Neural Network
│   ├── ED                       <- Encoder Decoder
│   ├── HSR                      <- Heteroskedastic Regression
│   ├── MLP                      <- Multilayer Perceptron
│   ├── RPN                      <- Randomized Prior Network
│   └── cVAE                     <- Conditional Variational Autoencoder
│
├── climsim_utils             <- Installable python package for data preprocessing and model evaluation
|
├── dataset_statistics        <- Precomputed statistics of input and output variables in dataset
│
├── demo_notebooks            <- Jupyter notebooks with examples of how to use the dataset for ML 
│
├── evaluation                <- Jupyter notebooks for evaluating ML models
│
├── grid_info                 <- Grid information for low-resolution dataset
│
├── preprocessing             <- Jupyter notebooks for preprocessing
│   ├── normalizations           <- Precomputed normalization factors for input and output data 
│
├── tests                     <- Tests of any kind
│
├── website                   <- Documentation
|
├── .gitignore                <- List of files ignored by git
|
|── LICENSE                   <- License
|
├── setup.py                  <- File for installing climsim_utils as a package
|
└── README.md
```

