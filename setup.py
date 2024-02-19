#!/usr/bin/env python

import os
from setuptools import find_packages, setup

# Check environment variable for the backend choice
ml_backend = os.getenv('ML_BACKEND', 'tensorflow').lower()

# Base requirements
install_requires = [
    "xarray",
    "numpy",
    "pandas",
    "matplotlib",
    "netCDF4",
    "h5py",
    "tqdm",
]

# Conditional requirements based on the backend choice
if ml_backend == 'pytorch':
    install_requires.append('torch')
elif ml_backend == 'tensorflow':
    install_requires.append('tensorflow')
else:
    raise ValueError(f"Unsupported ML_BACKEND value: {ml_backend}. Choose 'tensorflow' or 'pytorch'.")

setup(
    name="climsim_utils",
    version="0.0.1",
    description="""
    Tools for working with ClimSim, an open large-scale dataset for training
    high-resolution physics emulators in hybrid multi-scale climate simulators.
    """,
    author="Jerry Lin",
    author_email="jerryl9@uci.edu",
    url="https://github.com/leap-stc/ClimSim",
    python_requires=">=3.9",
    install_requires=install_requires,
    packages=find_packages(),
)
