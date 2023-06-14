#!/bin/bash
#SBATCH --job-name="Ritwik_HPO_v1"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:4
#SBATCH --ntasks=5
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ritwikgupta@berkeley.edu
#SBATCH -t 12:00:00


[[ -d ./logs/bair ]] || mkdir ./logs/bair

# $1: python script name
# $2: Keras Tuner "project name"
source /ocean/projects/atm200007p/rgupta7/miniconda3/bin/activate e3sm
module load cuda/11.7
srun --mpi=pmi2 --wait=0 bash run-dynamic.gpu-shared.baseline_v1.sh $1
