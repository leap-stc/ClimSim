#!/bin/bash
#SBATCH --job-name="make_test"
#SBATCH --output="logs_test.out"
#SBATCH --partition=RM
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryl9@uci.edu
#SBATCH -t 4:00:00

source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate preprocessing
python make_test.py
