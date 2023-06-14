#!/bin/bash
#SBATCH --job-name="tend_state_q0002_14_lev"
#SBATCH --output="logs/tend_state_q0002_14_lev.%j.out"
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --export=ALL
#SBATCH --account=m3312
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liranp@uci.edu
#SBATCH -t 05:00:00

module load python
./tend_state_q0002_14.py

