#!/bin/bash
#SBATCH --job-name="make_train"
#SBATCH --output="logs_train.out"
#SBATCH --partition=RM
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 8:00:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate preprocessing
python make_train.py
