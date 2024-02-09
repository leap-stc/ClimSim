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

cmd="python train_unet_h5loader.py --config-name=config_single \
        data_path='/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/preprocessing/v4_full/'\
        expname='v4plus_unet_nonaggressive_cliprh_huber_rop2' \
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
        epochs=20 \
	loss='huber' \
	dropout=0.0 \
        save_top_ckpts=15 \
        learning_rate=0.0001 \
	logger='wandb' \
        wandb.project='v4plus_unet' \
	scheduler_name='plateau' \
        scheduler.plateau.patience=2 \
        scheduler.plateau.factor=0.3162 \
        unet_num_blocks=2 \
        unet_attn_resolutions=[0] \
        unet_model_channels=128 "

cd /global/homes/z/zeyuanhu/nvidia_codes/Climsim_private/downstream_test/baseline_models/Unet_v4/training
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"
