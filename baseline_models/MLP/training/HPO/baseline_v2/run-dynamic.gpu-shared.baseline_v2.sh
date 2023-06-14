#!/bin/sh
echo "--- run-dynamic.sh ---"
echo SLURM_LOCALID $SLURM_LOCALID
echo SLURMD_NODENAME $SLURMD_NODENAME
#echo $LD_LIBRARY_PATH

#export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/

module load cudnn cudatoolkit
source ~/.bashrc
conda activate tf_conda
echo " "
echo "Current conda env:"
echo $CONDA_PREFIX
echo " "

# python_script=production-tuning.gpu-shared.py
python_script=$1
proj_name=$2
python ${python_script} -p ${proj_name} > logs/${proj_name}/keras-tuner-dynamic-shared-$SLURM_JOBID-$SLURMD_NODENAME-$SLURM_LOCALID.log 2>&1
