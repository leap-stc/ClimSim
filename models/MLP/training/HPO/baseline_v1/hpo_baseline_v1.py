import xarray as xr
import numpy as np
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
import argparse
import glob
import random

def set_environment(workers_per_node, workers_per_gpu):
    print('<< set_environment START >>')
    nodename = os.environ['SLURMD_NODENAME']
    procid = os.environ['SLURM_LOCALID']
    print(f'node name: {nodename}')
    print(f'procid:    {procid}')
    # stream = os.popen('scontrol show hostname $SLURM_NODELIST')
    # output = stream.read()
    # oracle = output.split("\n")[0]
    oracle_ip = os.environ["NERSC_NODE_HSN_IP"]
    print(f'oracle ip: {oracle_ip}')
    if procid==str(workers_per_node): # This takes advantage of the fact that procid numbering starts with ZERO
        os.environ["KERASTUNER_TUNER_ID"] = "chief"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Keras Tuner Oracle has been assigned.")
    else:
        os.environ["KERASTUNER_TUNER_ID"] = "tuner-" + str(nodename) + "-" + str(procid)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{int(int(procid)//workers_per_gpu)}"
    print(f'SY DEBUG: procid-{procid} / GPU-ID-{os.environ["CUDA_VISIBLE_DEVICES"]}')

    os.environ["KERASTUNER_ORACLE_IP"] = oracle_ip
    os.environ["KERASTUNER_ORACLE_PORT"] = "8000"
    print("KERASTUNER_TUNER_ID:    %s"%os.environ["KERASTUNER_TUNER_ID"])
    print("KERASTUNER_ORACLE_IP:   %s"%os.environ["KERASTUNER_ORACLE_IP"])
    print("KERASTUNER_ORACLE_PORT: %s"%os.environ["KERASTUNER_ORACLE_PORT"])
    #print(os.environ)
    print('<< set_environment END >>')

class MyHyperModel(HyperModel):
    def __init__(self, n_samples:int, tds, tds_val,
                 name=None, tunable=True):
        self.name = name
        self.tunable = tunable
        self._build = self.build
        self.build = self._build_wrapper
        
        self.n_samples = n_samples
        self.tds = tds
        self.tds_val = tds_val
        self.tds_shuffle_buffer = 384*30
        
        self.input_length = 124 # 60 + 60 + 1 +1 +1 + 1
        self.output_length_lin  = 120 # 60 + 60
        self.output_length_relu = 8
        self.output_length = self.output_length_lin + self.output_length_relu
        self.batch_size = -999 # updated in 'def build'

    def build(self, hp):        
        # hyperparameters to be tuned:
        n_layers = hp.Int("num_layers", 2, 12, default=2)
        hp_act = hp.Choice("activation", ['relu', 'elu', 'leakyrelu'], default='relu')
        hp_batch_size = hp.Choice("batch_size",
                                  [  48,   96,  192,  384,  768, 1152, 1536, 2304, 3072],
                                  default=3072)
        hp_optimizer = hp.Choice("optimizer", ['Adam', 'RAdam', 'RMSprop', 'SGD'], default='Adam')
        
        # constrcut a model
        # input layer
        x = keras.layers.Input(shape=(self.input_length,), name='input')
        input_layer = x
        
        # hidden layers
        for klayer in range(n_layers):
            n_units = hp.Int(f"units_{klayer}", min_value=128, max_value=1024, step=128, default=128)
            x = keras.layers.Dense(n_units)(x)
            if hp_act=='relu':
                x = keras.layers.ReLU()(x) 
            elif hp_act=='elu':
                x = keras.layers.ELU()(x)
            elif hp_act=='leakyrelu':
                x = keras.layers.LeakyReLU(alpha=.15)(x)
                
        # output layer (upper)
        x = keras.layers.Dense(self.output_length)(x)
        if   hp_act == 'relu':
            x = keras.layers.ReLU()(x) 
        elif hp_act == 'elu':
            x = keras.layers.ELU()(x)
        elif hp_act == 'leakyrelu':
            x = keras.layers.LeakyReLU(alpha=.15)(x)
        
        # output layer (lower)
        output_lin   = keras.layers.Dense(self.output_length_lin,activation='linear')(x)
        output_relu  = keras.layers.Dense(self.output_length_relu,activation='relu')(x)
        output_layer = keras.layers.Concatenate()([output_lin, output_relu])

        model = keras.Model(input_layer, output_layer, name='trial_model')
        
        # Optimizer
        # Set up cyclic learning rate
        INIT_LR = 2.5e-4
        MAX_LR  = 2.5e-3
        steps_per_epoch = self.n_samples // self.batch_size
        clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
                                                  maximal_learning_rate=MAX_LR,
                                                  scale_fn=lambda x: 1/(2.**(x-1)),
                                                  step_size= 2 * steps_per_epoch,
                                                  scale_mode = 'cycle'
                                                 )
        
        if   hp_optimizer == "Adam":
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
                                 
        # update self.batch_size for .fit
        self.batch_size = hp_batch_size
        
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        self.tds = self.tds.unbatch()
        self.tds = self.tds.shuffle(buffer_size=self.tds_shuffle_buffer, reshuffle_each_iteration=True)
        self.tds = self.tds.batch(self.batch_size)
        self.tds = self.tds.prefetch(buffer_size=int(np.ceil(self.tds_shuffle_buffer/self.batch_size))) # in realtion to the batch size

        self.tds_val = self.tds_val.unbatch()
        self.tds_val = self.tds_val.shuffle(buffer_size=self.tds_shuffle_buffer, reshuffle_each_iteration=True)
        self.tds_val = self.tds_val.batch(self.batch_size)
        self.tds_val = self.tds_val.prefetch(buffer_size=int(np.ceil(self.tds_shuffle_buffer/self.batch_size))) # in realtion to the batch size

        return model.fit(*args,x=self.tds,validation_data=self.tds_val,**kwargs)

def main():
    # in/out variable lists
    vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
    vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC',
                'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']
    
    # normalization/scaling factors
    mli_mean  = xr.open_dataset('../../norm_factors/mli_mean.nc', engine='netcdf4')
    mli_min   = xr.open_dataset('../../norm_factors/mli_min.nc', engine='netcdf4')
    mli_max   = xr.open_dataset('../../norm_factors/mli_max.nc', engine='netcdf4')
    mlo_scale = xr.open_dataset('../../norm_factors/mlo_scale.nc', engine='netcdf4')

    # train dataset for HPO
    # (subsampling id done here by "stride_sample")
    stride_sample = 37 # about ~20% assuming we will use 1/7 subsampled dataset for full training.
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
                                              output_shapes=((None,124),(None,128))
                                             )
    
    # instantiate generators
    tds = load_nc_dir_with_generator(f_mli)
    tds_val = load_nc_dir_with_generator(f_mli_val)

    # set up a search algorithm
    # note that train and val datasets (i.e., tds and tds_val) are a part of tuner's argument
    samples_per_file = 384
    n_epochs = 12
    max_trials = 1000
    project_name = args.proj_name
    tuner = RandomSearch(MyHyperModel(n_samples=len(f_mli)*samples_per_file,
                                      tds=tds,
                                      tds_val=tds_val),
                         objective='val_loss',
                         max_trials=max_trials,
                         max_retries_per_trial = 1,
                         executions_per_trial = 1,
                         directory='results/',
                         overwrite=False,
                         project_name=project_name
                        )

    print("---SEARCH SPACE---")
    tuner.search_space_summary()

    # search
    if os.environ["KERASTUNER_TUNER_ID"] != "chief":
        tuner.search(epochs=n_epochs,
                     verbose = 2)

if __name__ == '__main__':

    # command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--proj_name')
    args = parser.parse_args()

    # assign GPUs for workers
    gpus_per_node = 4 # NERSC Perlmutter
    ntasks = int(os.environ['SLURM_NTASKS']) # "total number of workers" + 1 (for oracle)
    nnodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    workers_per_node = int((ntasks - 1) / nnodes)
    workers_per_gpu  = int(workers_per_node / gpus_per_node)
    set_environment(workers_per_node=workers_per_node, workers_per_gpu=workers_per_gpu)

    # limit memory preallocation
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # only using a single GPU per trial

    # query available GPU (as debugging info only)
    print(f'({args.proj_name}) after-set_env')
    print(tf.config.list_physical_devices('GPU'))

    # run main program
    main()
