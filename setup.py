#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="climsim_utils",
    version="0.0.1",
    description="",
    author="",
    author_email="",
    url="https://github.com/leap-stc/ClimSim",
    python_requires=">=3.9",
    install_requires=[
        "xarray",
        "numpy",
        "pandas",
        "matplotlib",
        "tensorflow",
        "netCDF4",
        "h5py",
        "tqdm",
    ],
    packages=find_packages(),
)
