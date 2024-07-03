import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
import modulus
from modulus.metrics.general.mse import mse
from loss_energy import loss_energy
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from omegaconf import DictConfig
from modulus.launch.logging import (
    PythonLogger,
    LaunchLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
    initialize_mlflow,
)
from climsim_utils.data_utils import *
from climsim_datapip import climsim_dataset
from climsim_datapip_h5 import climsim_dataset_h5
from mlp import MLP
import mlp as mlp

import hydra
from collections.abc import Iterable
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler
import gc

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:

    DistributedManager.initialize()
    dist = DistributedManager()

    grid_path = cfg.climsim_path+'/grid_info/ClimSim_low-res_grid-info.nc'
    norm_path = cfg.climsim_path+'/preprocessing/normalizations/'
    grid_info = xr.open_dataset(grid_path)
    input_mean = xr.open_dataset(norm_path + cfg.input_mean)
    input_max = xr.open_dataset(norm_path + cfg.input_max)
    input_min = xr.open_dataset(norm_path + cfg.input_min)
    output_scale = xr.open_dataset(norm_path + cfg.output_scale)
    # qc_lbd = xr.open_dataset(norm_path + cfg.qc_lbd)
    # qi_lbd = xr.open_dataset(norm_path + cfg.qi_lbd)

    lbd_qc = np.loadtxt(norm_path + cfg.qc_lbd, delimiter=',')
    lbd_qi = np.loadtxt(norm_path + cfg.qi_lbd, delimiter=',')

    data = data_utils(grid_info = grid_info, 
                  input_mean = input_mean, 
                  input_max = input_max, 
                  input_min = input_min, 
                  output_scale = output_scale)

    # set variables to subset
    if cfg.variable_subsets == 'v1': 
        data.set_to_v1_vars()
    elif cfg.variable_subsets == 'v1_dyn':
        data.set_to_v1_dyn_vars()
    elif cfg.variable_subsets == 'v2':
        data.set_to_v2_vars()
    elif cfg.variable_subsets == 'v2_dyn':
        data.set_to_v2_dyn_vars()
    elif cfg.variable_subsets == 'v2_rh':
        data.set_to_v2_rh_vars()
    elif cfg.variable_subsets == 'v3':
        data.set_to_v3_vars()
    elif cfg.variable_subsets == 'v4':
        data.set_to_v4_vars()
    else:
        raise ValueError('Unknown variable subset')

    input_size = data.input_feature_len
    output_size = data.target_feature_len

    input_sub, input_div, out_scale = data.save_norm(write=False)


    # Create dataset instances
    # check if cfg.data_path + cfg.train_input exist
    # if os.path.exists(cfg.data_path + cfg.train_input):
    #     train_input_path = cfg.data_path + cfg.train_input
    #     train_target_path = cfg.data_path + cfg.train_target
    # else:
    #     #make train_input_path a list of all paths of cfg.data_path +'/*/'+cfg.train_input
    #     train_input_path = [f for f in glob.glob(cfg.data_path +'/*/'+cfg.train_input)]
    #     train_target_path = [f for f in glob.glob(cfg.data_path +'/*/'+cfg.train_target)]

    # print(train_input_path)

    val_input_path = cfg.data_path + cfg.val_input
    val_target_path = cfg.data_path + cfg.val_target
    if not os.path.exists(cfg.data_path + cfg.val_input):
        raise ValueError('Validation input path does not exist')

    #choose dataset class based on cfg.lazy_load
    # if cfg.lazy_load:
    #     if isinstance(train_input_path, list):
    #         dataset_class = climsim_dataset_lazy_list
    #     else:
    #         dataset_class = climsim_dataset_lazy
    # else:
    #     dataset_class = climsim_dataset
    
    #train_dataset = dataset_class(train_input_path, train_target_path, input_sub, input_div, out_scale, cfg.qinput_prune, cfg.output_prune, cfg.strato_lev, lbd_qc, lbd_qi)
    val_dataset = climsim_dataset(val_input_path, val_target_path, input_sub, input_div, out_scale, cfg.qinput_prune, cfg.output_prune, \
                                  cfg.strato_lev, lbd_qc, lbd_qi, cfg.decouple_cloud, cfg.aggressive_pruning, \
                                    cfg.strato_lev_qc, cfg.strato_lev_qinput, cfg.strato_lev_tinput, cfg.strato_lev_out, cfg.input_clip, cfg.input_clip_rhonly)

    #train_sampler = DistributedSampler(train_dataset) if dist.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.distributed else None
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.batch_size, 
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=cfg.num_workers)
    
    train_dataset = climsim_dataset_h5(cfg.data_path, \
                                    input_sub, input_div, out_scale, cfg.qinput_prune, cfg.output_prune, \
                                        cfg.strato_lev, lbd_qc, lbd_qi, cfg.decouple_cloud, cfg.aggressive_pruning, \
                                            cfg.strato_lev_qc, cfg.strato_lev_qinput, cfg.strato_lev_tinput, cfg.strato_lev_out, cfg.input_clip, cfg.input_clip_rhonly)
            
    train_sampler = DistributedSampler(train_dataset) if dist.distributed else None
    
    train_loader = DataLoader(train_dataset, 
                                batch_size=cfg.batch_size, 
                                shuffle=False if dist.distributed else True,
                                sampler=train_sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(),
                                num_workers=cfg.num_workers)
    # Create dataloaders
    # train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # train_loader = DataLoader(train_dataset, 
    #                           batch_size=cfg.batch_size, 
    #                           shuffle=False,
    #                           sampler=train_sampler,
    #                           drop_last=True,
    #                           pin_memory=torch.cuda.is_available(),
    #                           num_workers=cfg.num_workers)
                              
    

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print('debug: output_size', output_size, output_size//60, output_size%60)

    tmp_output_prune = cfg.output_prune
    tmp_strato_lev = cfg.strato_lev_out
    tmp_mlp_hidden_dims = cfg.mlp_hidden_dims
    if isinstance(tmp_mlp_hidden_dims, Iterable) and not isinstance(tmp_mlp_hidden_dims, list):
        print('Input is iterable but not a list. Converting to list...')
        tmp_mlp_hidden_dims = list(tmp_mlp_hidden_dims)
    else:
        print('Input is already a list or not iterable')

    print(f"Type of tmp_mlp_hidden_dims: {type(tmp_mlp_hidden_dims)}")
    if isinstance(tmp_mlp_hidden_dims, list):
        print('input is list')
    else:
        print('input is not list')
    tmp_mlp_layers = cfg.mlp_layers
    print('MLP init arguments: ', input_size, output_size, tmp_mlp_hidden_dims, tmp_mlp_layers, tmp_output_prune, tmp_strato_lev)
    model = MLP(
        in_dims = input_size,
        out_dims = output_size,
        hidden_dims = tmp_mlp_hidden_dims,
        layers = tmp_mlp_layers,
        output_prune = tmp_output_prune,
        strato_lev_out = tmp_strato_lev,
    ).to(dist.device)

    if len(cfg.restart_path) > 0:
        print("Restarting from checkpoint: " + cfg.restart_path)
        if dist.distributed:
            model_restart = modulus.Module.from_checkpoint(cfg.restart_path).to(dist.device)
            if dist.rank == 0:
                model.load_state_dict(model_restart.state_dict())
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                model.load_state_dict(model_restart.state_dict())
        else:
            model_restart = modulus.Module.from_checkpoint(cfg.restart_path).to(dist.device)
            model.load_state_dict(model_restart.state_dict())

    # Set up DistributedDataParallel if using more than a single process.
    # The `distributed` property of DistributedManager can be used to
    # check this.
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],  # Set the device_id to be
                                               # the local rank of this process on
                                               # this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # create optimizer
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    else:
        raise ValueError('Optimizer not implemented')
    
    # create scheduler
    if cfg.scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step.step_size, gamma=cfg.scheduler.step.gamma)
    elif cfg.scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler.plateau.factor, patience=cfg.scheduler.plateau.patience, verbose=True)
    elif cfg.scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler.cosine.T_max, eta_min=cfg.scheduler.cosine.eta_min)
    else:
        raise ValueError('Scheduler not implemented')
    
    # create loss function
    if cfg.loss == 'mse':
        loss_fn = mse
        criterion = nn.MSELoss()
    elif cfg.loss == 'mae':
        loss_fn = nn.L1Loss()
        criterion = nn.L1Loss()
    elif cfg.loss == 'huber':
        loss_fn = nn.SmoothL1Loss()
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError('Loss function not implemented')
    
    def loss_weighted(pred, target):
        if cfg.variable_subsets in ['v1','v1_dyn']:
            raise ValueError('Weighted loss not implemented for v1/v1_dyn')
        # dt_weight = 1.0
        # dq1_weight = 1.0
        # dq2_weight = 1.0
        # dq3_weight = 1.0
        # du_weight = 1.0
        # dv_weight = 1.0
        # d2d_weight = 1.0

        # pred should be of shape (batch_size, 368)
        # target should be of shape (batch_size, 368)
        # 0-60: dt, 60-120 dq1, 120-180 dq2, 180-240 dq3, 240-300 du, 300-360 dv, 360-368 d2d
        #only do the calculation if any of the weights are not 1.0
        if cfg.dt_weight == 1.0 and cfg.dq1_weight == 1.0 and cfg.dq2_weight == 1.0 and cfg.dq3_weight == 1.0 and cfg.du_weight == 1.0 and cfg.dv_weight == 1.0 and cfg.d2d_weight == 1.0:
            return criterion(pred, target)
        pred[:,0:60] = pred[:,0:60] * cfg.dt_weight
        pred[:,60:120] = pred[:,60:120] * cfg.dq1_weight
        pred[:,120:180] = pred[:,120:180] * cfg.dq2_weight
        pred[:,180:240] = pred[:,180:240] * cfg.dq3_weight
        pred[:,240:300] = pred[:,240:300] * cfg.du_weight
        pred[:,300:360] = pred[:,300:360] * cfg.dv_weight
        pred[:,360:368] = pred[:,360:368] * cfg.d2d_weight
        target[:,0:60] = target[:,0:60] * cfg.dt_weight
        target[:,60:120] = target[:,60:120] * cfg.dq1_weight
        target[:,120:180] = target[:,120:180] * cfg.dq2_weight
        target[:,180:240] = target[:,180:240] * cfg.dq3_weight
        target[:,240:300] = target[:,240:300] * cfg.du_weight
        target[:,300:360] = target[:,300:360] * cfg.dv_weight
        target[:,360:368] = target[:,360:368] * cfg.d2d_weight
        return criterion(pred, target)
    
    # Initialize the console logger
    logger = PythonLogger("main")  # General python logger

    if cfg.logger == 'wandb':
        # Initialize the MLFlow logger
        initialize_wandb(
            project=cfg.wandb.project,
            name=cfg.expname,
            entity="zeyuan_hu",
            mode="online",
        )
        LaunchLogger.initialize(use_wandb=True)
    else:
        # Initialize the MLFlow logger
        initialize_mlflow(
            experiment_name=cfg.mlflow.project,
            experiment_desc="Modulus launch development",
            run_name=cfg.expname,
            run_desc="Modulus Training",
            user_name="Modulus User",
            mode="offline",
        )
        LaunchLogger.initialize(use_mlflow=True)

    if cfg.save_top_ckpts<=0:
        logger.info("Checkpoints should be set >0, setting to 1")
        num_top_ckpts = 1
    else:
        num_top_ckpts = cfg.save_top_ckpts

    if cfg.top_ckpt_mode == 'min':
        top_checkpoints = [(float('inf'), None)] * num_top_ckpts
    elif cfg.top_ckpt_mode == 'max':
        top_checkpoints = [(-float('inf'), None)] * num_top_ckpts
    else:
        raise ValueError('Unknown top_ckpt_mode')
    
    if dist.rank == 0:
        save_path = os.path.join(cfg.save_path, cfg.expname) #cfg.save_path + cfg.expname
        save_path_ckpt = os.path.join(save_path, 'ckpt')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_ckpt):
            os.makedirs(save_path_ckpt)
    
    if dist.world_size > 1:
        torch.distributed.barrier()
      

    hyai = data.grid_info['hyai'].values
    hybi = data.grid_info['hybi'].values
    hyai = torch.tensor(hyai, dtype=torch.float32).to(device)
    hybi = torch.tensor(hybi, dtype=torch.float32).to(device)
    # input_sub, input_div, out_scale = data.save_norm(write=False)
    input_sub_device = torch.tensor(input_sub, dtype=torch.float32).to(device)
    input_div_device = torch.tensor(input_div, dtype=torch.float32).to(device)
    out_scale_device = torch.tensor(out_scale, dtype=torch.float32).to(device)

    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        # cuda_graph_warmup=11,
    )
    def training_step(model, data_input, target):
        output = model(data_input)
        loss = loss_weighted(output, target)
        return loss
    @StaticCaptureEvaluateNoGrad(model=model, use_graphs=False)
    def eval_step_forward(my_model, invar):
        return my_model(invar)
    #training block
    logger.info("Starting Training!")
    # Basic training block with tqdm for progress tracking
    for epoch in range(cfg.epochs):
        if dist.distributed:
            train_sampler.set_epoch(epoch)
        # idx_train_loader = epoch % len(train_input_path)
        # if epoch >0:
        #     #free the memory of previously defined train_dataset and train_loader
        #     del train_dataset.inputs
        #     del train_dataset.targets
        #     del train_dataset
        #     del train_loader
        #     torch.cuda.empty_cache()
        #     gc.collect()
        # logger.info(f"Training epoch {epoch+1}/{cfg.epochs} with train_input_path: {train_input_path[idx_train_loader]}")
        # train_dataset = climsim_dataset(train_input_path[idx_train_loader], train_target_path[idx_train_loader], \
        #                                 input_sub, input_div, out_scale, cfg.qinput_prune, cfg.output_prune, \
        #                                     cfg.strato_lev, lbd_qc, lbd_qi, cfg.decouple_cloud, cfg.aggressive_pruning, \
        #                                         cfg.strato_lev_qc, cfg.strato_lev_qinput, cfg.strato_lev_tinput, cfg.input_clip, cfg.input_clip_rhonly)
                
        # train_sampler = DistributedSampler(train_dataset) if dist.distributed else None
        # if dist.distributed:
        #     train_sampler.set_epoch(epoch)
        # train_loader = DataLoader(train_dataset, 
        #                           batch_size=cfg.batch_size, 
        #                           shuffle=False,
        #                           sampler=train_sampler,
        #                           drop_last=True,
        #                           pin_memory=torch.cuda.is_available(),
        #                           num_workers=cfg.num_workers)
        # wrap the epoch in launch logger to control frequency of output for console logs
        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=10) as launchlog:
            # model.train()
            # Wrap train_loader with tqdm for a progress bar
            train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            current_step = 0
            for data_input, target in train_loop:
                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                    break
                # if cfg.output_prune: # this is currently done in the dataset class
                #     # the following code only works for the v2/v3 output cases!
                #     target[:,60:60+cfg.strato_lev] = 0
                #     target[:,120:120+cfg.strato_lev] = 0
                #     target[:,180:180+cfg.strato_lev] = 0
                data_input, target = data_input.to(device), target.to(device)
                # optimizer.zero_grad()
                # output = model(data_input)
                # if cfg.do_energy_loss:
                #     ps_raw = data_input[:,1500]*input_div[1500]+input_sub[1500]
                #     loss_energy_train = loss_energy(output, target, ps_raw, hyai, hybi, out_scale_device)*cfg.energy_loss_weight
                #     loss_orig = loss_weighted(output, target)
                #     loss = loss_orig + loss_energy_train
                # else:
                #     loss = loss_weighted(output, target)
                # loss.backward()
                loss = training_step(model, data_input, target)
                # max_grad = max(p.grad.abs().max() for p in model.parameters() if p.grad is not None)
                # # Initialize a list to store the L2 norms of each parameter's gradient
                # l2_norms = []

                # for p in model.parameters():
                #     if p.grad is not None:
                #         # Calculate the L2 norm for each parameter's gradient and add it to the list
                #         l2_norms.append(torch.norm(p.grad, p=2))

                # # Calculate the mean of the L2 norms
                # mean_l2_norm = torch.mean(torch.stack(l2_norms))

                #optimizer.step()
                # del data_input, target, output
                #loss = training_step(data_input, target)
                # scheduler.step()
                #launchlog.log_minibatch({"loss_train": loss.detach().cpu().numpy()})
                #if dist.rank == 0:
                if cfg.do_energy_loss:
                    launchlog.log_minibatch({"loss_train": loss.detach().cpu().numpy(), "loss_energy_train": loss_energy_train.detach().cpu().numpy(), "lr": optimizer.param_groups[0]["lr"], "loss_orig": loss_orig.detach().cpu().numpy()})
                else:
                    launchlog.log_minibatch({"loss_train": loss.detach().cpu().numpy(), "lr": optimizer.param_groups[0]["lr"]})
                # Update the progress bar description with the current loss
                train_loop.set_description(f'Epoch {epoch+1}')
                train_loop.set_postfix(loss=loss.item())
                current_step += 1
            #launchlog.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            
            # model.eval()
            val_loss = 0.0
            if cfg.do_energy_loss:
                val_energy_loss = 0.0
                val_orig = 0.0
            num_samples_processed = 0
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/1 [Validation]')
            current_step = 0
            for data_input, target in val_loop:
                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                    break
                # if cfg.output_prune:
                #     # the following code only works for the v2/v3 output cases!
                #     target[:,60:60+cfg.strato_lev] = 0
                #     target[:,120:120+cfg.strato_lev] = 0
                #     target[:,180:180+cfg.strato_lev] = 0
                # Move data to the device
                data_input, target = data_input.to(device), target.to(device)

                output = eval_step_forward(model, data_input)
                if cfg.do_energy_loss:
                    ps_raw = data_input[:,1500]*input_div[1500]+input_sub[1500]
                    loss_energy_train = loss_energy(output, target, ps_raw, hyai, hybi, out_scale_device)*cfg.energy_loss_weight
                    loss_orig = loss_weighted(output, target)
                    loss = loss_orig + loss_energy_train
                else:
                    loss = loss_weighted(output, target)
                val_loss += loss.item() * data_input.size(0)
                num_samples_processed += data_input.size(0)

                # Calculate and update the current average loss
                current_val_loss_avg = val_loss / num_samples_processed
                val_loop.set_postfix(loss=current_val_loss_avg)
                current_step += 1
                if cfg.do_energy_loss:
                    val_energy_loss += loss_energy_train.item() * data_input.size(0)
                    val_orig += loss_orig.item() * data_input.size(0)
                    current_val_loss_avg_energy = val_energy_loss / num_samples_processed
                    current_val_loss_avg_orig = val_orig / num_samples_processed
                del data_input, target, output
                    
            
            # if dist.rank == 0:
                #all reduce the loss
            if dist.world_size > 1:
                current_val_loss_avg = torch.tensor(current_val_loss_avg, device=dist.device)
                torch.distributed.all_reduce(current_val_loss_avg)
                current_val_loss_avg = current_val_loss_avg.item() / dist.world_size

            if dist.rank == 0:
                if cfg.do_energy_loss:
                    launchlog.log_epoch({"loss_valid": current_val_loss_avg, "loss_energy_valid": current_val_loss_avg_energy, "loss_orig_valid": current_val_loss_avg_orig})
                else:
                    launchlog.log_epoch({"loss_valid": current_val_loss_avg})

                current_metric = current_val_loss_avg
                # Save the top checkpoints
                if cfg.top_ckpt_mode == 'min':
                    is_better = current_metric < max(top_checkpoints, key=lambda x: x[0])[0]
                elif cfg.top_ckpt_mode == 'max':
                    is_better = current_metric > min(top_checkpoints, key=lambda x: x[0])[0]
                
                #print('debug: is_better', is_better, current_metric, top_checkpoints)
                if len(top_checkpoints) == 0 or is_better:
                    ckpt_path = os.path.join(save_path_ckpt, f'ckpt_epoch_{epoch+1}_metric_{current_metric:.4f}.mdlus')
                    if dist.distributed:
                        model.module.save(ckpt_path)
                    else:
                        model.save(ckpt_path)
                    top_checkpoints.append((current_metric, ckpt_path))
                    # Sort and keep top 5 based on max/min goal at the beginning
                    if cfg.top_ckpt_mode == 'min':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=False)
                    elif cfg.top_ckpt_mode == 'max':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                    # delete the worst checkpoint
                    if len(top_checkpoints) > num_top_ckpts:
                        worst_ckpt = top_checkpoints.pop()
                        print(f"Removing worst checkpoint: {worst_ckpt[1]}")
                        if worst_ckpt[1] is not None:
                            os.remove(worst_ckpt[1])
                            
            if cfg.scheduler_name == 'plateau':
                scheduler.step(current_val_loss_avg)
            else:
                scheduler.step()
            
            if dist.world_size > 1:
                torch.distributed.barrier()
                
    if dist.rank == 0:
        logger.info("Start recovering the model from the top checkpoint to do torchscript conversion")         
        #recover the model weight to the top checkpoint
        model = modulus.Module.from_checkpoint(top_checkpoints[0][1]).to(device)

        # Save the model
        save_file = os.path.join(save_path, 'model.mdlus')
        model.save(save_file)
        # convert the model to torchscript
        mlp.device = "cpu"
        device = torch.device("cpu")
        model_inf = modulus.Module.from_checkpoint(save_file).to(device)
        scripted_model = torch.jit.script(model_inf)
        scripted_model = scripted_model.eval()
        save_file_torch = os.path.join(save_path, 'model.pt')
        scripted_model.save(save_file_torch)
        # save input and output normalizations
        data.save_norm(save_path, True)
        logger.info("saved input/output normalizations and model to: " + save_path)

        mdlus_directory = os.path.join(save_path, 'ckpt')
        for filename in os.listdir(mdlus_directory):
            print(filename)
            if filename.endswith(".mdlus"):
                full_path = os.path.join(mdlus_directory, filename)
                print(full_path)
                model = modulus.Module.from_checkpoint(full_path).to("cpu")
                scripted_model = torch.jit.script(model)
                scripted_model = scripted_model.eval()

                # Save the TorchScript model
                save_path_torch = os.path.join(mdlus_directory, filename.replace('.mdlus', '.pt'))
                scripted_model.save(save_path_torch)
                print('save path for ckpt torchscript:', save_path_torch)


        logger.info("Training complete!")

    return current_val_loss_avg

if __name__ == "__main__":
    main()