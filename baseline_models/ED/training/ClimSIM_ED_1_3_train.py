from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler,CSVLogger


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
import tensorflow.keras as ke

"""
Author: Gunnar Behrens

Contains the code to construct ED and to train ED

hyperparameters are based on a hyperparameter tuning of ED for SPCAM version3 (similar to E3SM-MMF),
if not stated else

more information about ED can be found here: Behrens2022 (https://doi.org/10.1029/2022MS003130)
"""


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) # print number of available GPUs


original_dim_input=124  # MMF input node size

original_dim_output=int(128) # MMF target node size 


# network hyperparameters:
latent_dim=5 # define the latent space width
input_shape = (original_dim_input,) # define input shape of Encoder
decoder_input_shape=(latent_dim,) # define input shape of Decoder
out_shape=(original_dim_output,) # define target shape of Decoder
intermediate_dim = 463 # define node size of first and last hidden layers of Encoder or Decoder (Behrens2022)
batchsize=714 # define batchsize of training (Behrens2022)

# load input and target data of training and validation date set

# training data
train_input=np.load('$PROJECT_DATA_path/train_input.npy') # specify path to train_input.npy file
train_output=np.load('$PROJECT_DATA_path/train_target.npy') # specify path to train_target.npy fie

# validation data
val_input=np.load('$PROJECT_DATA_path/val_input.npy') # specify path to val_input.npy file
val_output=np.load('$PROJECT_DATA_path/val_target.npy') # specify path to val_target.npy file

# Construct Encoder model based on Behrens2022

input_lay=Input(shape=input_shape, name='encoder_input')
x_0 =Dense(intermediate_dim, activation='relu')(input_lay)
x_1 =Dense(intermediate_dim, activation='relu')(x_0)
x_2 =Dense(intermediate_dim/2, activation='relu')(x_1)
x_3 =Dense(intermediate_dim/4, activation='relu')(x_2)
x_4 =Dense(intermediate_dim/8, activation='relu')(x_3)
x_5 =Dense(intermediate_dim/16, activation='relu')(x_4)
x_6 =Dense(latent_dim, activation='relu')(x_5)
    
encoder=Model(input_lay,x_6, name='encoder') # build Encoder model 
encoder.summary() # show structure of Encoder
    

# Construct Decoder model based on Behrens2022

input_decoder=Input(shape=decoder_input_shape, name='decoder_input')
x_0 =Dense(intermediate_dim/16, activation='relu')(input_decoder)
x_1 =Dense(intermediate_dim/8, activation='relu')(x_0)
x_2 =Dense(intermediate_dim/4, activation='relu')(x_1)
x_3 =Dense(intermediate_dim/2, activation='relu')(x_2)
x_4 =Dense(intermediate_dim, activation='relu')(x_3)
x_5 =Dense(intermediate_dim, activation='relu')(x_4)
output_lay= Dense(original_dim_output,activation='elu')(x_5)

decoder = Model(input_decoder, output_lay, name='decoder') #build Decoder
decoder.summary() # show structure of Decoder

# Connect Encoder and Decoder
decoder_outputs=decoder(encoder(input_lay))

# build ED
ED=Model(input_lay,decoder_outputs, name='ED')
ED.summary() # show structure of ED


# compile ED with learning rate of 0.0001 (adjusted to ClimSim data set) and define further metrics
ED.compile(ke.optimizers.Adam(lr=0.0001), loss=mse, metrics=['mse','mae','accuracy']) # add. metrics = mse, mae and accuracy


lr_init= 0.0001 # set initial learning rate for learning rate scheduler

# set learning rate schedule
def schedule(epoch):
    """
    learning rate scheduler based on Behrens 2022, divides lr after each 7th epoch by factor 5
    """
        

    if epoch < 7:
            
        return lr_init
    elif epoch < 14:
            
        return lr_init/5

    elif epoch < 21:
            
        return lr_init/25
        
    elif epoch < 28:
        return lr_init/125
        
    elif epoch < 35:
        return lr_init/625

    elif epoch<42:
        return lr_init/3125



        

callback_lr=LearningRateScheduler(schedule,verbose=1) # use callback for lr scheduler

csv_logger = CSVLogger('../model/ED_ClimSIM_1_3.csv') # define path where history of training is stored
 
# model training over 40 epochs based on Behrens2022, shuffle set to true    
ED.fit(x=train_input,y=train_output,validation_data=(val_input,val_output),epochs=40,shuffle=True,
        callbacks=[callback_lr,csv_logger])

# save model as .h5 file
ED.save('../model/ED_ClimSIM_1_3_model.h5')

