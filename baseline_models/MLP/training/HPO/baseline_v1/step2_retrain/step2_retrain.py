import xarray as xr
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras import callbacks
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner import RandomSearch
import os
import tensorflow_addons as tfa
import sys
import glob
import random
from pathlib import Path

### Make sure to change this parameters ###
###########################################
input_length = 124
output_length = 128
output_length_lin = 120
output_length_relu = 8
f_trial_info = '../step1_analysis/step1_results.csv'
###########################################

def read_args():
    # argv[1]: lot id
    # argv[2]: trial id
    # argv[3]: continueed training if "continue";
    #          ohterwise, training starts from the checkpoint saved during step 1 HPO.
    return sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)==4 else "none"

def set_environment():
    ## Part 1: Set CUDA_VISIBLE_DEVICES
    # Find the least used gpu id
    fgpu = 'logs/gpuid.%s.%s.txt'%(os.environ['SLURM_JOB_ID'], os.environ['HOSTNAME'])
    if not os.path.exists(fgpu):
        open(fgpu, "w").close()
        print(f"{fgpu} has been created.")

    # initialize a dictionary to count the occurrence of each number
    count = {0: 0, 1: 0, 2: 0, 3: 0}

    # read the file and count the occurrence of each number
    with open(fgpu, 'r') as f:
        for line in f:
            for num in line.split():
                count[int(num)] += 1

    # find the number with the least occurrence
    gpu_id = min(count, key=count.get)
    print("fgpu: ", count)
    print("GPU ID for training: ", gpu_id)

    # set CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # update fgpu
    with open(fgpu, 'a') as f:
        f.write(f"{gpu_id} ")

    ## Part 2: Limit memory preallocation
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    ## Part 3: Query available GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # If there are multiple GPUs, you can iterate over them
        for gpu in gpus:
            print("GPU:", gpu)

def build_model(lot_id:str, trial_id:str, n_samples:int, sw_continue:str):
   
    # read trial info
    results_step1 =  pd.read_csv(f_trial_info)
    trial_info = results_step1.loc[(results_step1['lot']==lot_id)*
                                   (results_step1['trial']==trial_id)]
    hp_act        = trial_info['activation'].values.item()
    hp_batch_size = trial_info['batch_size'].values.item()
    hp_num_layers = trial_info['num_layers'].values.item()
    hp_optimizer  = trial_info['optimizer'].values.item()
    hp_units      = np.fromstring(trial_info['units'].values.item()[1:-1], 
                                  sep=' ', 
                                  dtype=int)

    # constrcut a model
    # input layer
    x = keras.layers.Input(shape=(input_length), name='input')
    input_layer = x

    # hidden layers
    if hp_num_layers != len(hp_units): 
        raise Exception("the number of layers (hp) does not match the number of units (hp).")
    else:
        for klayer in range(hp_num_layers):
            n_units = hp_units[klayer]
            x = keras.layers.Dense(n_units)(x)
            if hp_act=='relu':
                x = keras.layers.ReLU()(x)
            elif hp_act=='elu':
                x = keras.layers.ELU()(x)
            elif hp_act=='leakyrelu':
                x = keras.layers.LeakyReLU(alpha=.15)(x)

    # output layer (upper)
    x = keras.layers.Dense(output_length)(x)
    if   hp_act == 'relu':
        x = keras.layers.ReLU()(x)
    elif hp_act == 'elu':
        x = keras.layers.ELU()(x)
    elif hp_act == 'leakyrelu':
        x = keras.layers.LeakyReLU(alpha=.15)(x)

    # output layer (lower)
    output_lin   = keras.layers.Dense(output_length_lin,activation='linear')(x)
    output_relu  = keras.layers.Dense(output_length_relu,activation='relu')(x)
    output_layer = keras.layers.Concatenate()([output_lin, output_relu])

    model = keras.Model(input_layer, output_layer, name='retrained_model')

    # load weights
    if sw_continue.lower() == "continue":
        fn_model = f'retrained_models/step2_{lot_id}_{trial_id}.last.h5'
        model = models.load_model(fn_model, compile=False)
        print(f'model weights are initialized with: {fn_model}')
    else:
        f_wgts = f'../results/{lot_id}/{trial_id}/checkpoint'
        model.load_weights(f_wgts)
        print(f'model weights are initizlized with: {f_wgts}')

    # Optimizer
    # Set up cyclic learning rate
    INIT_LR = 2.5e-4
    MAX_LR  = 2.5e-3
    steps_per_epoch = n_samples // hp_batch_size
    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
                                              maximal_learning_rate=MAX_LR,
                                              scale_fn = lambda x: 1/(2.**(x-1)),
                                              step_size = 2 * steps_per_epoch,
                                              scale_mode = 'cycle'
                                             )

    if  hp_optimizer == "Adam":
        my_optimizer = keras.optimizers.Adam(learning_rate=clr)
    elif hp_optimizer == "RAdam":
        my_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=clr)
    elif hp_optimizer == "RMSprop":
        my_optimizer = keras.optimizers.RMSprop(learning_rate=clr)
    elif hp_optimizer == "SGD":
        my_optimizer = keras.optimizers.SGD(learning_rate=clr)

    # compile
    model.compile(optimizer=my_optimizer, #optimizer=keras.optimizers.Adam(learning_rate=clr),
                  loss='mse',
                  metrics=['mse','mae','accuracy'])

    # model summary
    print(model.summary())

    return model, hp_batch_size

def main(lot_id:str, trial_id:str, sw_continue:str):

    ### Make sure to change this parameters ###
    ###########################################
    fn_retrained_best = f'retrained_models/step2_{lot_id}_{trial_id}.best.h5'
    fn_retrained_last = f'retrained_models/step2_{lot_id}_{trial_id}.last.h5'
    fn_metrics        = f'retrained_models/metrics/step2_{lot_id}_{trial_id}.metrics.csv'
    Path('./retrained_models/metrics').mkdir(parents=True, exist_ok=True)
    num_epochs = 18
    ###########################################

    ### Dataset prep ###
    # in/out variable lists
    vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
    vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC',
                'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

    # normalization/scaling factors
    mli_mean  = xr.open_dataset('../../../norm_factors/mli_mean.nc',  engine='netcdf4')
    mli_min   = xr.open_dataset('../../../norm_factors/mli_min.nc',   engine='netcdf4')
    mli_max   = xr.open_dataset('../../../norm_factors/mli_max.nc',   engine='netcdf4')
    mlo_scale = xr.open_dataset('../../../norm_factors/mlo_scale.nc', engine='netcdf4')

    # train dataset for HPO
    # (subsampling id done here by "stride_sample")
    stride_sample = 7 # prime number to sample all 'tod'
    f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.000[1234567]-*-*-*.nc')
    f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0008-01-*-*.nc')
    f_mli = sorted([*f_mli1, *f_mli2])
    random.shuffle(f_mli) # to reduce IO bottleneck
    f_mli = f_mli[::stride_sample]

    # validation dataset for HPO
    f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0008-0[23456789]-*-*.nc')
    f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0008-1[012]-*-*.nc')
    f_mli3 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0009-01-*-*.nc')
    f_mli_val = sorted([*f_mli1, *f_mli2, *f_mli3])
    random.shuffle(f_mli_val)
    f_mli_val = f_mli_val[::stride_sample]

    # data generator
    # (also includes data preprocessing)
    def load_nc_dir_with_generator(filelist:list):
        def gen():
            for file in filelist:
                # read mli
                ds = xr.open_dataset(file, engine='netcdf4')
                ds = ds[vars_mli]

                # read mlo
                dso = xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4')

                # make mlo variales: ptend_t and ptend_q0001
                dso['ptend_t'] = (dso['state_t'] - ds['state_t'])/1200 # T tendency [K/s]
                dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
                dso = dso[vars_mlo]

                # normalizatoin, scaling
                ds = (ds-mli_mean)/(mli_max-mli_min)
                dso = dso*mlo_scale

                # stack
                #ds = ds.stack({'batch':{'sample','ncol'}}) # this line was for data files that include 'sample' dimension
                ds = ds.stack({'batch':{'ncol'}})
                ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
                #dso = dso.stack({'batch':{'sample','ncol'}})
                dso = dso.stack({'batch':{'ncol'}})
                dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')

                yield (ds.values, dso.values) # generating a tuple of (input, output)

        return tf.data.Dataset.from_generator(gen,
                                              output_types=(tf.float64, tf.float64),
                                              output_shapes=((None,input_length),(None,output_length))
                                             )

    ### build model ###
    model, batch_size = build_model(lot_id=lot_id,
                                    trial_id=trial_id,
                                    n_samples=len(f_mli),
                                    sw_continue=sw_continue)

    ### fit ###
    # callbacks
    checkpoint_best = keras.callbacks.ModelCheckpoint(filepath=fn_retrained_best,
                                                      save_weights_only=False,
                                                      verbose=1,
                                                      monitor='val_loss',
                                                      save_best_only=True) # first checkpoint for best model
    checkpoint_last = keras.callbacks.ModelCheckpoint(filepath=fn_retrained_last,
                                                      save_weights_only=False,
                                                      verbose=1,
                                                      save_best_only=False) # second checkpoint for continuing training
    csv_logger = keras.callbacks.CSVLogger(fn_metrics, append=True)
    earlystop = keras.callbacks.EarlyStopping('val_loss', patience=8)

    # instantiate generators
    tds_shuffle_buffer = 384*30 # 30 day equivalent num_samples
    tds = load_nc_dir_with_generator(f_mli)
    tds = tds.unbatch()
    tds = tds.shuffle(buffer_size=tds_shuffle_buffer, reshuffle_each_iteration=True)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=int(np.ceil(tds_shuffle_buffer/batch_size))) # in realtion to the batch size

    tds_val = load_nc_dir_with_generator(f_mli_val)
    tds_val = tds_val.unbatch()
    tds_val = tds_val.shuffle(buffer_size=tds_shuffle_buffer, reshuffle_each_iteration=True)
    tds_val = tds_val.batch(batch_size)
    tds_val = tds_val.prefetch(buffer_size=int(np.ceil(tds_shuffle_buffer/batch_size))) # in realtion to the batch size

    # fit
    history = model.fit(tds,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_data=tds_val,
                        verbose=2,
                        callbacks=[checkpoint_best, checkpoint_last, csv_logger, earlystop])

if __name__ == '__main__':

    lot_id, trial_id, sw_continue = read_args()
    set_environment()
    main(lot_id, trial_id, sw_continue)
