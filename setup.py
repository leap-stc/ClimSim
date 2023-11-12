#!/usr/bin/env python

from setuptools import find_packages, setup

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
