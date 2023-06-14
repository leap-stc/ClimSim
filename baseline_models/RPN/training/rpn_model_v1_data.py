#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:19:00 2022

@author: mohamedazizbhouri
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from jax import numpy as np
from jax import vmap, jit, random
import time
import numpy as onp

case_norm = 'rpn_params_all_Gauss'

is_gins = 1

case_var = 'all_zeros_' 
n_run_param = 0
ensemble_size = 32 
if case_var == 'all_zeros_':
    ind_output = np.arange(128)

dim_x = 124
dim_y = ind_output.shape[0]

layers = [dim_x, 768, 640, 512, 640, 640, dim_y]

##########################################################
##########################################################
##########################################################
from functools import partial
from jax.nn import relu, gelu
def leakyRELU(x):
    return np.where(x > 0, x, x * 0.15)
def RELU(x):
    return np.where(x > 0, x, x * 0)

def MLP(layers, activation=leakyRELU): # np.tanh
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev*random.normal(k1, (d_in, d_out)) 
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs
    return init, apply
        
##########################################################
##########################################################
##########################################################
    
from jax import grad
from jax.example_libraries import optimizers
import itertools

# Define the model
class EnsembleRegression:
    def __init__(self, layers, ensemble_size, rng_key = random.PRNGKey(0)):  
        # Network initialization and evaluation functions
        self.init, self.apply = MLP(layers)
        self.init_prior, self.apply_prior = MLP(layers)
        
        # Random keys
        k1, k2, k3 = random.split(rng_key, 3)
        keys_1 = random.split(k1, ensemble_size)
        keys_2 = random.split(k2, ensemble_size)
        keys_3 = random.split(k3, ensemble_size)
        
        # Initialize
        params = vmap(self.init)(keys_1)
        params_prior = vmap(self.init_prior)(keys_2)
                
        lr = optimizers.exponential_decay(5e-4, decay_steps=1000, decay_rate=0.99) 
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
        
        self.opt_state = vmap(self.opt_init)(params)
        self.prior_opt_state = vmap(self.opt_init)(params_prior)
        self.key_opt_state = vmap(self.opt_init)(keys_3)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

    # Define the forward pass
    def net_forward(self, params, params_prior, inputs):
        Y_pred = self.apply(params, inputs) + self.apply_prior(params_prior, inputs) 
        return Y_pred

    def loss(self, params, params_prior, batch):
        inputs, targets = batch
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, 0))(params, params_prior, inputs)
        # Compute loss
        loss = np.mean((targets - outputs)**2)
        return loss

    # Define the update step
    def step(self, i, opt_state, prior_opt_state, key_opt_state, batch):
        params = self.get_params(opt_state)
        params_prior = self.get_params(prior_opt_state)
        g = grad(self.loss)(params, params_prior, batch)
        return self.opt_update(i, g, opt_state)
    
    def monitor_loss(self, opt_state, prior_opt_state, batch):
        params = self.get_params(opt_state)
        params_prior = self.get_params(prior_opt_state)
        loss_value = self.loss(params, params_prior, batch)
        return loss_value

    # Optimize parameters in a loop
    def train(self, nIter = 1000):
        # Define vectorized SGD step across the entire ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0)))
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0)))

        # Main training loop
        tt = time.time()
        for it in range(nIter):
            inds_loc = inds[:,onp.random.choice(N_rpn, batch_size, replace=False)]
            inputs0 = X_train[inds_loc[0,:],:][None,:,:]
            targets0 = y_train[inds_loc[0,:],:] [None,:,:]
            for ii in range(ensemble_size-1):
                inputs0 = onp.concatenate( (inputs0,X_train[inds_loc[ii+1,:],:][None,:,:]), axis=0)
                targets0 = onp.concatenate( (targets0,y_train[inds_loc[ii+1,:],:][None,:,:]), axis=0)
            
            batch = (inputs0-mu_XH)/sigma_XH, (targets0-mu_yH)/sigma_yH
            
            self.opt_state = v_step(it, self.opt_state, self.prior_opt_state, self.key_opt_state, batch)
            # Logger
            if it % nloss == 0: # 500
                loss_value = v_monitor_loss(self.opt_state, self.prior_opt_state, batch)
                self.loss_log.append(loss_value)
                print(case_norm, case_var, n_run_param, it, nIter, time.time() - tt, loss_value.max())
                tt = time.time()
            if (it+1) % nsave == 0: # 10000
                params = vmap(self.get_params)(self.opt_state)
                np.save(pp+case_var+case_norm+'/run_'+str(n_run_param)+'/loss.npy',np.array(self.loss_log))
                for i in range(len(layers)-1):
                    for j in range(2):
                        np.save(pp+case_var+case_norm+'/run_'+str(n_run_param)+'/params_'+str(i)+'_'+str(j),params[i][j])
                            
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def posterior(self, params, inputs):
        params, params_prior = params
        samples = vmap(self.net_forward, (0, 0, 0))(params, params_prior, inputs)
        return samples
    
##########################################################
##########################################################
##########################################################

# Helper functions
normalize = vmap(lambda x, mu, std: (x-mu)/std, in_axes=(0,0,0))
denormalize = vmap(lambda x, mu, std: x*std + mu, in_axes=(0,0,0))

if is_gins == 1:
    current_dirs_parent = os.path.dirname(os.getcwd())
    current_dirs_parent = os.path.dirname(current_dirs_parent)
    current_dirs_parent = os.path.dirname(current_dirs_parent)
    pp = current_dirs_parent+'/glab/projects/GCM_data/'
else:
    #bridges
    current_dirs_parent = os.path.dirname(os.getcwd()) 
    pp = current_dirs_parent+'/GCM_data/'

# Training data
print('loading data')
tt = time.time()
X_train = onp.load(pp+'training_data/train_input.npy')
y_train = onp.load(pp+'training_data/train_target.npy')[:,ind_output]

N_tot = X_train.shape[0]

model = EnsembleRegression(layers, ensemble_size, rng_key = random.PRNGKey(n_run_param))


batch_size = 3072

nb = 2628
fraction = 0.8
N_rpn = 8073216
 
nsave = 50000
nloss = 500
nepoch = 5 

onp.random.seed(n_run_param)

inds = []
for i in range(ensemble_size):
    inds.append(onp.random.choice(N_tot, N_rpn, replace=False))
inds = onp.array(inds) # N_rpn x ensemble_size

if case_norm == 'rpn_params_all_Gauss_tune_frac' or case_norm == 'rpn_params_all_Gauss_from_norm':
    if 1 == 0: # to compute mu and sigma 
        mu_XH, sigma_XH = [], []
        for i in range(X_train.shape[1]):
            print(i)
            mu_XH.append(onp.mean(onp.array(X_train,dtype=onp.float64)[:,i]))
            sigma_XH.append(onp.std(onp.array(X_train,dtype=onp.float64)[:,i]))
        mu_XH = onp.array(onp.array(mu_XH),dtype=onp.float32)
        sigma_XH = onp.array(onp.array(sigma_XH),dtype=onp.float32)
        np.save('mu_X_final.npy',mu_XH)
        np.save('sigma_X_final.npy',sigma_XH)
        
        mu_yH, sigma_yH = [], []
        for i in range(y_train.shape[1]):
            print(i)
            mu_yH.append(onp.mean(onp.array(y_train,dtype=onp.float64)[:,i]))
            sigma_yH.append(onp.std(onp.array(y_train,dtype=onp.float64)[:,i]))
        mu_yH = onp.array(onp.array(mu_yH),dtype=onp.float32)
        sigma_yH = onp.array(onp.array(sigma_yH),dtype=onp.float32)
        print(sigma_yH[64:72])
        if case_var == 'all_zeros_':
            mu_yH[64:72] = 0
            sigma_yH[64:72] = 1
        print(sigma_yH[64:72])
        np.save('mu_y_final.npy',mu_yH)
        np.save('sigma_y_final.npy',sigma_yH)
        
        bof = aaa
    else:
        
        mu_XH = onp.load('mu_X_final.npy')
        sigma_XH = onp.load('sigma_X_final.npy')
        mu_yH = onp.load('mu_y_final.npy')
        sigma_yH = onp.load('sigma_y_final.npy')

params_prior = vmap(model.get_params)(model.prior_opt_state)
print('saving parameters')
for i in range(len(layers)-1):
    for j in range(2):
        np.save(pp+case_var+case_norm+'/run_'+str(n_run_param)+'/params_prior_'+str(i)+'_'+str(j),params_prior[i][j])
print('finished saving')

# Train model
model.train(nIter=nepoch*nb)
params = vmap(model.get_params)(model.opt_state)
for i in range(len(layers)-1):
    for j in range(2):
        np.save(pp+case_var+case_norm+'/run_'+str(n_run_param)+'/params_'+str(i)+'_'+str(j),params[i][j])
   
