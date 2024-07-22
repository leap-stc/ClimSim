import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus

"""
Contains the code for the MLP and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class MLPMetaData(modulus.ModelMetaData):
    name: str = "MLP"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    
    
class MLP(modulus.Module):
    """
    MLP Estimator
    """
    def __init__(self, in_dims, out_dims, hidden_dims, layers, dropout=0., output_prune=False, strato_lev_out=15):
        super().__init__(meta=MLPMetaData())
        # check if hidden_dims is a list of hidden_dims
        if isinstance(hidden_dims, list):
            # print('input is list')
            assert len(hidden_dims) == layers, "Length of hidden_dims should be equal to layers"
        else:
            hidden_dims = [hidden_dims] * layers
        
        self.output_prune = output_prune
        self.strato_lev_out = strato_lev_out

        self.linears = []
        for i in range(layers):
            self.linears += [torch.nn.Sequential(
                torch.nn.Linear(in_dims if i == 0 else hidden_dims[i-1], hidden_dims[i]),
                # torch.nn.LayerNorm(hidden_dims),
                torch.nn.Dropout(p=dropout))
            ]
            # self.add_module('linear%d' % i, self.linears[-1])
        self.linears = torch.nn.ModuleList(self.linears)
        self.final_linear = torch.nn.Linear(hidden_dims[-1], out_dims)

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        for linear in self.linears:
            x = torch.nn.functional.relu(linear(x))
        x = self.final_linear(x)

        if self.output_prune:
            x = x.clone()
            x[:, 60:60+self.strato_lev_out] = x[:, 60:60+self.strato_lev_out].clone().zero_()
            x[:, 120:120+self.strato_lev_out] = x[:, 120:120+self.strato_lev_out].clone().zero_()
            x[:, 180:180+self.strato_lev_out] = x[:, 180:180+self.strato_lev_out].clone().zero_()
            x[:, 240:240+self.strato_lev_out] = x[:, 240:240+self.strato_lev_out].clone().zero_()
        
        # do relu for the last 8 elements
        # x[:,-8:] = torch.nn.functional.relu(x[:,-8:])
        x = x.clone()
        x[:,-8:] = torch.nn.functional.relu(x[:,-8:].clone())
        return x