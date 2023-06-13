#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:20:34 2023

@author: mohamedazizbhouri
"""

import numpy as np

import pickle
from matplotlib import pyplot as plt
plt.close('all')

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 32,
                     'lines.linewidth': 2,
                     'axes.labelsize': 32,
                     'axes.titlesize': 32,
                     'xtick.labelsize': 32,
                     'ytick.labelsize': 32,
                     'legend.fontsize': 32,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex"
                     })
######################################
############ User's input ############
######################################

# please provide your model_name. The latter will be used in saving the results
# files (npy and plots)
model_name_l = ['HSR']

######################################
############# Test steup #############
######################################

mlo_scale = np.array( np.load('mlo_scale.npy'), dtype=np.float32)

# Please specify the temporal stride used in generating validation/test dataset
# and original GCM time-step (dt_GCM) in minutes
stride = 6
dt_GCM = 20

test_y = np.load('training_data/val_target_stride6.npy')#[:,ind_output]
test_y = test_y / mlo_scale
pred_y_l = []

nrun = 4
import h5py    
filename = 'hsr_samples_bestcrps.h5' 

f = h5py.File(filename, 'r')
samples = f['pred'][:]
if nrun<4:
    pred_y = np.swapaxes(np.swapaxes(samples, 0, 2), 1, 2)[nrun*32:(nrun+1)*32,:,:]
else:
    pred_y = np.swapaxes(np.swapaxes(samples, 0, 2), 1, 2)
print(pred_y.shape)
#bof = aaa

pred_y = pred_y / mlo_scale
print(pred_y.shape)

pred_y_l.append(pred_y)

# file 'pressures_val_stride6_60lvls.npy' contains pressure levels in Pa
pressure = np.load('pressures_val_stride6_60lvls.npy')/100
N_pressure = pressure.shape[0] # Number of pressure levels: 60

# load dictionary of latitude-longitude coordinates of GCM grid on which data is saved
with open('indextolatlons.pkl', 'rb') as f:
    data = pickle.load(f)
N_lat_long = len(data) # = 384 There are 384 points in latitude-longitude GCM grid

mweightpre = np.load('ne4.val-stride-6.dp-over-g.npy').T # 60x384 ==> 384x60
area = np.load('ne4.area.npy')[:,None]/4/np.pi # 384 ==> 384x1

Lv = 2.26e6
cp = 1.00464e3
rho = 997

weight = np.concatenate( (cp*mweightpre,Lv*mweightpre,area,area,rho*Lv*area,rho*Lv*area,area,area,area,area), axis=1 )
# 384 x 128

test_y = np.array(test_y,dtype=np.float64)

for i in range(1):
    pred_y = pred_y_l[i]
    model_name = model_name_l[i]
    # numpy return erronous errors when summing over "large" number of points with float32 format, 
    # so we transform arrays to float64 type before computing coefficient of determination R2
    pred_y = np.array(pred_y,dtype=np.float64) # Npts(time discret \times long-lat discret) x 128
    
    dim_y = test_y.shape[1]
    
    # Ndata is total number of data points in validation / test dataset
    # N_time_steps is the number of time-steps considered in validation / test dataset
    Ndata = test_y.shape[0]
    N_time_steps = Ndata//N_lat_long
    
    ###########################################
    # Global metrics for 128 output variables #
    ###########################################
    
    def reshape_npy(y):
        # reshape true data into: N_time_steps x N_lat_long x N_pressure
        return y.reshape( (N_time_steps, N_lat_long, dim_y) )
        
    test_daily = weight * reshape_npy(test_y) # N_time_steps x N_lat_long x N_pressure
    test_daily = test_daily.reshape( (N_time_steps*N_lat_long, dim_y) ) # Npts(time discret \times long-lat discret) x 128
    
    pred_daily = np.zeros((test_y.shape[0], dim_y, pred_y.shape[0]))
    # batch_size, num_variables, num_samples
    for i in range(pred_y.shape[0]):
        pred_daily[:,:,i] =  (weight * reshape_npy(pred_y[i,:,:])).reshape( (N_time_steps*N_lat_long, dim_y) )
    
    def crps(outputs, target, weights=None):
        """
        Computes the Continuous Ranked Probability Score (CRPS) between the target and the ecdf for each output variable and then takes a weighted average over them.
    
        Input
        -----
        outputs - float[B, F, S] samples from the model
        target - float[B, F] ground truth target
        """
        n = outputs.shape[2]
        y_hats = np.sort(outputs, axis=-1)
        # E[Y - y]
        mae = np.abs(target[..., None] - y_hats).mean(axis=(0, -1))
        # E[Y - Y'] ~= sum_i sum_j |Y_i - Y_j| / (2 * n * (n-1))
        diff = y_hats[..., 1:] - y_hats[..., :-1]
        count = np.arange(1, n) * np.arange(n - 1, 0, -1)
        crps = mae - (diff * count).sum(axis=-1).mean(axis=0) / (2 * n * (n-1))
        return crps#.average(weights=weights)
    
    crps_f = crps(pred_daily, test_daily)
    print(crps_f.shape)
    print(np.array(crps_f).shape)
    np.save(model_name+'_CRPS_Mike_'+str(nrun)+'.npy',crps_f)
    