#!/bin/bash

export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7
export NVPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/bin
export PY_SITE_PATH=/global/common/software/nersc/pm-2022q4/sw/pytorch/2.0.1/lib/python3.9/site-packages
export CMAKE_PREFIX_PATH="${PY_SITE_PATH}/torch/share/cmake;${PY_SITE_PATH}/pybind11/share/cmake"
export PYTORCH_FORTRAN_PATH=/global/cfs/cdirs/m4331/shared/pytorch-fortran-nvhpc22.7/nvhpc/install
echo "Trying to use Python libraries from $PY_SITE_PATH"

CONFIG=Release
OPENACC=1


BUILD_PATH=$(pwd -P)/nvhpc
mkdir -p $BUILD_PATH/build

# build inference example
(
    set -x
    cd $BUILD_PATH/build
    cmake \
        -Dpytorch_proxy_ROOT=${PYTORCH_FORTRAN_PATH} \
        -Dpytorch_fort_proxy_ROOT=${PYTORCH_FORTRAN_PATH} \
        -DOPENACC=${OPENACC} -DCMAKE_BUILD_TYPE=${CONFIG} \
        -DCMAKE_Fortran_COMPILER=nvfortran ../..
    cmake --build . --parallel
)
