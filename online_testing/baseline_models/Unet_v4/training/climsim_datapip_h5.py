# #import xarray as xr
# from torch.utils.data import Dataset
# import numpy as np
# import torch

#import xarray as xr
from torch.utils.data import Dataset
import numpy as np
import torch
import glob
import h5py

class climsim_dataset_h5(Dataset):
    def __init__(self, 
                 parent_path, 
                 input_sub, 
                 input_div, 
                 out_scale, 
                 qinput_prune, 
                 output_prune, 
                 strato_lev,
                 qc_lbd,
                 qi_lbd, 
                 decouple_cloud=False, 
                 aggressive_pruning=False,
                 strato_lev_qc=30,
                 strato_lev_qinput=None, 
                 strato_lev_tinput=None,
                 strato_lev_out = 12,
                 input_clip=False,
                 input_clip_rhonly=False,):
        """
        Args:
            parent_path (str): Path to the .zarr file containing the inputs and targets.
            input_sub (np.ndarray): Input data mean.
            input_div (np.ndarray): Input data standard deviation.
            out_scale (np.ndarray): Output data standard deviation.
            qinput_prune (bool): Whether to prune the input data.
            output_prune (bool): Whether to prune the output data.
            strato_lev (int): Number of levels in the stratosphere.
            qc_lbd (np.ndarray): Coefficients for the exponential transformation of qc.
            qi_lbd (np.ndarray): Coefficients for the exponential transformation of qi.
        """
        self.parent_path = parent_path
        self.input_paths = glob.glob(f'{parent_path}/**/train_input.h5', recursive=True)
        print('input paths:', self.input_paths)
        if not self.input_paths:
            raise FileNotFoundError("No 'train_input.h5' files found under the specified parent path.")
        self.target_paths = [path.replace('train_input.h5', 'train_target.h5') for path in self.input_paths]

        # Initialize lists to hold the samples count per file
        self.samples_per_file = []
        for input_path in self.input_paths:
            with h5py.File(input_path, 'r') as file:  # Open the file to read the number of samples
                # Assuming dataset is named 'data', adjust if different
                self.samples_per_file.append(file['data'].shape[0])
                
        self.cumulative_samples = np.cumsum([0] + self.samples_per_file)
        self.total_samples = self.cumulative_samples[-1]

        self.input_files = {}
        self.target_files = {}
        for input_path, target_path in zip(self.input_paths, self.target_paths):
            self.input_files[input_path] = h5py.File(input_path, 'r')
            self.target_files[target_path] = h5py.File(target_path, 'r')

        # for input_path, target_path in zip(self.input_paths, self.target_paths):
        #     # Lazily open zarr files and keep the reference
        #     self.input_zarrs[input_path] = zarr.open(input_path, mode='r')
        #     self.target_zarrs[target_path] = zarr.open(target_path, mode='r')
        
        self.input_sub = input_sub
        self.input_div = input_div
        self.out_scale = out_scale
        self.qinput_prune = qinput_prune
        self.output_prune = output_prune
        self.strato_lev = strato_lev
        self.qc_lbd = qc_lbd
        self.qi_lbd = qi_lbd
        self.decouple_cloud = decouple_cloud
        self.aggressive_pruning = aggressive_pruning
        self.strato_lev_qc = strato_lev_qc
        self.strato_lev_out = strato_lev_out
        self.input_clip = input_clip
        if strato_lev_qinput <0:
            self.strato_lev_qinput = strato_lev
        else:
            self.strato_lev_qinput = strato_lev_qinput
        self.strato_lev_tinput = strato_lev_tinput
        self.input_clip_rhonly = input_clip_rhonly

        if self.strato_lev_qinput <self.strato_lev:
            raise ValueError('strato_lev_qinput should be greater than or equal to strato_lev, otherwise inconsistent with E3SM')


    def __len__(self):
        return self.total_samples
    
    def _find_file_and_index(self, idx):
        file_idx = np.searchsorted(self.cumulative_samples, idx+1) - 1
        local_idx = idx - self.cumulative_samples[file_idx]
        return file_idx, local_idx

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of bounds")
        # Find which file the index falls into
        # file_idx = np.searchsorted(self.cumulative_samples, idx+1) - 1
        # local_idx = idx - self.cumulative_samples[file_idx]

        # x = zarr.open(self.input_paths[file_idx], mode='r')[local_idx]
        # y = zarr.open(self.target_paths[file_idx], mode='r')[local_idx]
        file_idx, local_idx = self._find_file_and_index(idx)


        # x = self.input_zarrs[self.input_paths[file_idx]][local_idx]
        # y = self.target_zarrs[self.target_paths[file_idx]][local_idx]
        # Open the HDF5 files and read the data for the given index
        input_file = self.input_files[self.input_paths[file_idx]]
        target_file = self.target_files[self.target_paths[file_idx]]
        x = input_file['data'][local_idx]
        y = target_file['data'][local_idx]

        # with h5py.File(self.input_paths[file_idx], 'r') as input_file:
        #     x = input_file['data'][local_idx]  # Adjust 'data' if your dataset has a different name
        
        # with h5py.File(self.target_paths[file_idx], 'r') as target_file:
        #     y = target_file['data'][local_idx]  # Adjust 'data' if your dataset has a different name

        # x = np.load(self.input_paths,mmap_mode='r')[idx]
        # y = np.load(self.target_paths,mmap_mode='r')[idx]
        x[120:180] = 1 - np.exp(-x[120:180] * self.qc_lbd)
        x[180:240] = 1 - np.exp(-x[180:240] * self.qi_lbd)
        # Avoid division by zero in input_div and set corresponding x to 0
        # input_div_nonzero = self.input_div != 0
        # x = np.where(input_div_nonzero, (x - self.input_sub) / self.input_div, 0)
        x = (x - self.input_sub) / self.input_div
        #make all inf and nan values 0
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0

        y = y * self.out_scale
        if self.decouple_cloud:
            x[120:240] = 0
            x[60*14:60*16] =0
            x[60*19:60*21] =0
        elif self.aggressive_pruning:
            # for profiles, only keep stratosphere temperature. prune all other profiles in stratosphere
            x[60:60+self.strato_lev_qinput] = 0 # prune RH
            x[120:120+self.strato_lev_qc] = 0
            x[180:180+self.strato_lev_qinput] = 0
            x[240:240+self.strato_lev] = 0 # prune u
            x[300:300+self.strato_lev] = 0 # prune v
            x[360:360+self.strato_lev] = 0
            x[420:420+self.strato_lev] = 0
            x[480:480+self.strato_lev] = 0
            x[540:540+self.strato_lev] = 0
            x[600:600+self.strato_lev] = 0
            x[660:660+self.strato_lev] = 0
            x[720:720+self.strato_lev] = 0
            x[780:780+self.strato_lev_qinput] = 0
            x[840:840+self.strato_lev_qc] = 0 # prune qc_phy
            x[900:900+self.strato_lev_qinput] = 0
            x[960:960+self.strato_lev] = 0
            x[1020:1020+self.strato_lev] = 0
            x[1080:1080+self.strato_lev_qinput] = 0
            x[1140:1140+self.strato_lev_qc] = 0 # prune qc_phy in previous time step
            x[1200:1200+self.strato_lev_qinput] = 0
            x[1260:1260+self.strato_lev] = 0
            x[1515] = 0 #SNOWHICE
        elif self.qinput_prune:
            # x[:,60:60+self.strato_lev] = 0
            x[120:120+self.strato_lev] = 0
            x[180:180+self.strato_lev] = 0

        if self.strato_lev_tinput >0:
            x[0:self.strato_lev_tinput] = 0
        
        if self.input_clip:
            if self.input_clip_rhonly:
                x[60:120] = np.clip(x[60:120], 0, 1.2)
            else:
                x[60:120] = np.clip(x[60:120], 0, 1.2) # for RH, clip to (0,1.2)
                x[360:720] = np.clip(x[360:720], -0.5, 0.5) # for dyn forcing, clip to (-0.5,0.5)
                x[720:1320] = np.clip(x[720:1320], -3, 3) # for phy tendencies  clip to (-3,3)

        
        if self.output_prune:
            y[60:60+self.strato_lev_out] = 0
            y[120:120+self.strato_lev_out] = 0
            y[180:180+self.strato_lev_out] = 0
            y[240:240+self.strato_lev_out] = 0
            y[300:300+self.strato_lev_out] = 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)