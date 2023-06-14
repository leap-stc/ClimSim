#!/bin/sh
echo "--- run-dynamic.sh ---"
echo SLURM_LOCALID $SLURM_LOCALID
echo SLURMD_NODENAME $SLURMD_NODENAME
#echo $LD_LIBRARY_PATH

#export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate bair
echo " "
echo "Current conda env:"
echo $CONDA_PREFIX
echo " "

# python_script=production-tuning.gpu-shared.py
python_script=$1
python ${python_script} > logs/bair/keras-tuner-dynamic-shared-$SLURM_JOBID-$SLURMD_NODENAME-$SLURM_LOCALID.log 2>&1
