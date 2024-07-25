#!/bin/bash
#SBATCH -A m4331
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 4
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
##SBATCH --mail-user=zeyuanh@nvidia.com
##SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cmd="python train_mlp_h5loader.py --config-name=config_single \
        data_path='/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_full/'\
        expname='v2rh_mlp_nonaggressive_cliprh_mae_step_3l_lr1em3' \
	variable_subsets='v2_rh' \
	mlp_hidden_dims=[384,1024,640] \
        mlp_layers=3 \
	batch_size=1024 \
        num_workers=32 \
	qinput_prune=True \
        output_prune=True \
	input_clip=True \
	strato_lev=15 \
	strato_lev_out=12 \
	strato_lev_qinput=22 \
	input_clip_rhonly=True \
	aggressive_pruning=False \
        epochs=28 \
	loss='mae' \
	dropout=0.0 \
        save_top_ckpts=15 \
        learning_rate=0.001 \
	logger='wandb' \
        wandb.project='v4plus_unet' \
	scheduler_name='step' \
        scheduler.step.step_size=7 \
        scheduler.step.gamma=0.3162 "

cd /global/homes/z/zeyuanhu/nvidia_codes/Climsim_private/downstream_test/baseline_models/MLP_v2rh/training
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"
