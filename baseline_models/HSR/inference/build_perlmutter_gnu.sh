#!/bin/bash

export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0
export PY_SITE_PATH=/global/common/software/nersc/pm-stable/sw/pytorch/2.1.0/lib/python3.10/site-packages
export CMAKE_PREFIX_PATH="${PY_SITE_PATH}/torch/share/cmake;${PY_SITE_PATH}/pybind11/share/cmake"
export PYTORCH_FORTRAN_PATH=/global/homes/a/akshay13/Codes/pytorch-fortran/gnu/install
echo "Trying to use Python libraries from $PY_SITE_PATH"

CONFIG=Release
OPENACC=0


BUILD_PATH=$(pwd -P)/gnu
mkdir -p $BUILD_PATH/build

# build inference example
(
    set -x
    cd $BUILD_PATH/build
    cmake \
        -Dpytorch_proxy_ROOT=${PYTORCH_FORTRAN_PATH} \
        -Dpytorch_fort_proxy_ROOT=${PYTORCH_FORTRAN_PATH} \
        -DOPENACC=${OPENACC} -DCMAKE_BUILD_TYPE=${CONFIG} \
        -DCMAKE_Fortran_COMPILER=gfortran ../..
    cmake --build . --parallel
)
