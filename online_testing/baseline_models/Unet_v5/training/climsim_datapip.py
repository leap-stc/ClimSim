# #import xarray as xr
# from torch.utils.data import Dataset
# import numpy as np
# import torch

#import xarray as xr
from torch.utils.data import Dataset
import numpy as np
import torch

class climsim_dataset(Dataset):
    def __init__(self, 
                 input_paths, 
                 target_paths, 
                 input_sub, 
                 input_div, 
                 out_scale, 
                 qinput_prune, 
                 output_prune, 
                 strato_lev,
                 strato_lev_out,
                 qn_lbd,
                 decouple_cloud=False, 
                 aggressive_pruning=False,
                #  strato_lev_qc=30,
                 strato_lev_qinput=None, 
                 strato_lev_tinput=None,
                 input_clip=False,
                 input_clip_rhonly=False,
                 qn_tscaled=False,
                 qn_logtransform=False):
        """
        Args:
            input_paths (str): Path to the .npy file containing the inputs.
            target_paths (str): Path to the .npy file containing the targets.
            input_sub (np.ndarray): Input data mean.
            input_div (np.ndarray): Input data standard deviation.
            out_scale (np.ndarray): Output data standard deviation.
            qinput_prune (bool): Whether to prune the input data.
            output_prune (bool): Whether to prune the output data.
            strato_lev (int): Number of levels in the stratosphere.
            qc_lbd (np.ndarray): Coefficients for the exponential transformation of qc.
            qi_lbd (np.ndarray): Coefficients for the exponential transformation of qi.
        """
        self.inputs = np.load(input_paths)
        self.targets = np.load(target_paths)
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.input_sub = input_sub
        self.input_div = input_div
        self.out_scale = out_scale
        self.qinput_prune = qinput_prune
        self.output_prune = output_prune
        self.strato_lev = strato_lev
        self.strato_lev_out = strato_lev_out
        self.qn_lbd = qn_lbd
        self.decouple_cloud = decouple_cloud
        self.aggressive_pruning = aggressive_pruning
        self.input_clip = input_clip
        if strato_lev_qinput <0:
            self.strato_lev_qinput = strato_lev
        else:
            self.strato_lev_qinput = strato_lev_qinput
        self.strato_lev_tinput = strato_lev_tinput
        self.input_clip_rhonly = input_clip_rhonly
        self.qn_tscaled = qn_tscaled
        self.qn_logtransform = qn_logtransform

        if self.strato_lev_qinput <self.strato_lev:
            raise ValueError('strato_lev_qinput should be greater than or equal to strato_lev, otherwise inconsistent with E3SM')

    def t_scaled_weight(self, t):
        # Polynomial coefficients
        a = 1.043084e-12
        b = -4.028800e-10
        c = 4.128325e-08
        # Evaluate polynomial
        y = a * t**2 + b * t + c
        # Set the boundary values
        t_min = 190
        t_max = 290
        # Polynomial values at the boundaries
        y_min = 2.39141e-09  #a * t_min**2 + b * t_min + c
        y_max = 1.21714e-08 #a * t_max**2 + b * t_max + c
        # Apply boundary conditions using np.where
        y = np.where(t < t_min, y_min, y)
        y = np.where(t > t_max, y_max, y)
        return y_max/y

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]

        if self.qn_tscaled:
            # use temperature to generate weights for scaling qn
            qn_scale_weight = self.t_scaled_weight(x[0:60])

        if not self.qn_logtransform:
            x[120:180] = 1 - np.exp(-x[120:180] * self.qn_lbd)
        x = (x - self.input_sub) / self.input_div

        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0

        y = y * self.out_scale
        if self.decouple_cloud:
            x[120:180] = 0
            x[60*14:60*15] =0
            x[60*18:60*19] =0

        if self.aggressive_pruning:
            # for profiles, only keep stratosphere temperature. prune all other profiles in stratosphere
            x[60:60+self.strato_lev_qinput] = 0 # prune RH
            x[120:120+self.strato_lev_qinput] = 0
            # x[180:180+self.strato_lev] = 0 # should be liq_partition
            x[240:240+self.strato_lev] = 0 # prune u
            x[300:300+self.strato_lev] = 0 # prune v
            x[360:360+self.strato_lev] = 0
            x[420:420+self.strato_lev] = 0
            x[480:480+self.strato_lev] = 0
            x[540:540+self.strato_lev] = 0
            x[600:600+self.strato_lev] = 0
            x[660:660+self.strato_lev] = 0
            x[720:720+self.strato_lev] = 0
            x[780:780+self.strato_lev_qinput] = 0 # prune qv_phy
            x[840:840+self.strato_lev_qinput] = 0 # prune qn_phy
            x[900:900+self.strato_lev] = 0
            x[960:960+self.strato_lev] = 0
            x[1020:1020+self.strato_lev_qinput] = 0 # prune qv_phy
            x[1080:1080+self.strato_lev_qinput] = 0 # prune qn_phy in previous time step
            x[1140:1140+self.strato_lev] = 0
            x[1395] = 0 #SNOWHICE
        elif self.qinput_prune:
            #raise NotImplementedError('should use aggressive_pruning! instead of qinput_prune!')
            # x[:,60:60+self.strato_lev] = 0
            x[120:120+self.strato_lev] = 0
            # x[180:180+self.strato_lev] = 0

        if self.strato_lev_tinput >0:
            x[0:self.strato_lev_tinput] = 0
        
        if self.input_clip:
            if self.input_clip_rhonly:
                x[60:120] = np.clip(x[60:120], 0, 1.2)
            else:
                x[60:120] = np.clip(x[60:120], 0, 1.2) # for RH, clip to (0,1.2)
                x[360:720] = np.clip(x[360:720], -0.5, 0.5) # for dyn forcing, clip to (-0.5,0.5)
                x[720:1200] = np.clip(x[720:1200], -3, 3) # for phy tendencies  clip to (-3,3)

        
        if self.output_prune:
            y[60:60+self.strato_lev_out] = 0
            y[120:120+self.strato_lev_out] = 0
            y[180:180+self.strato_lev_out] = 0
            y[240:240+self.strato_lev_out] = 0

        if self.qn_tscaled:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(qn_scale_weight, dtype=torch.float32)
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)