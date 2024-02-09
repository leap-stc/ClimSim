#!/bin/bash
#SBATCH -A m4331
#SBATCH -C gpu
#SBATCH -q shared_interactive 
#SBATCH -t 04:00:00
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 2
#SBATCH -n 2
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
##SBATCH --mail-user=zeyuanh@nvidia.com
##SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cmd="python train_unet_h5loader_classifier_gradout.py --config-name=config_single \
        data_path='/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/preprocessing/v5_full/'\
        expname='v5_classifier_lr3em4_qnlog_thred1013_smaller2_clip' \
	batch_size=2024 \
        num_workers=32 \
	qinput_prune=True \
        output_prune=True \
	input_clip=True \
	strato_lev=15 \
	strato_lev_out=15 \
	strato_lev_qinput=15 \
	input_clip_rhonly=True \
	aggressive_pruning=True \
	threshold_class1=1e-10 \
	threshold_class2=1e-13 \
	qn_logtransform=True \
        epochs=10 \
	dropout=0.1 \
	drop_extreme_samples=False \
        save_top_ckpts=15 \
        learning_rate=0.0003 \
	clip_grad=True \
	clip_grad_norm=1.0 \
        scheduler_warmup.enable=False \
        scheduler_warmup.init_lr=1e-7 \
        scheduler_warmup.warmup_steps=5 \
	logger='wandb' \
        wandb.project='v5_unet_class' \
        scheduler_name='step' \
        scheduler.step.step_size=3 \
	scheduler.step.gamma=0.316 \
        unet_num_blocks=1 \
        unet_attn_resolutions=[0] \
        unet_model_channels=32 "

cd /global/homes/z/zeyuanhu/nvidia_codes/Climsim_private/downstream_test/baseline_models/Unet_v5/training
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"

