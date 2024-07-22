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
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
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