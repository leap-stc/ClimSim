import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import models
from keras import layers
from keras import callbacks
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner import RandomSearch
import os
import tensorflow_addons as tfa

# from qhoptim.tf import QHAdamOptimizer
import sys
import argparse
import glob
import random

from pathlib import Path
import time

from keras.layers.convolutional import Conv1D
import logging


def get_filenames(dataroot, filenames):
    '''
    Create list of filenames
    
    Args:
        dataroot: Relative or global path to directory of filenames
        filenames: List of filename wildcards
        stride_sample int: pick every nth sample
    '''
    
    filepaths = []
    for filename in filenames:
        filepaths.extend(dataroot.glob('**/' + filename))
    # f_mli = sorted([*f_mli1, *f_mli2]) # I commented this out. It seems unecessary to sort the list if it will be shuffled
    random.shuffle(filepaths)
    # f_mli = f_mli[0:72*5] # debugging
    # random.shuffle(f_mli) # I commented this out. It seems unnecessary to shuffle twice.

    return filepaths


def set_environment(num_gpus_per_node=1, oracle_port="32768"):
    """
    This function sets up the environment variables for the Keras Tuner Oracle.
    It should be called at the beginning of the main function.
    The default oracle port is 8000, but 8000 is also very popular.
    When running into GPU issues, scanning for alternative ports is recommended.
    """

    print("<< set_environment START >>")
    num_gpus_per_node = str(num_gpus_per_node)
    nodename = "charybdis"
    procid = "0"
    print(f"node name: {nodename}")
    print(f"procid:    {procid}")
    if procid == str(
        num_gpus_per_node
    ):  # This takes advantage of the fact that procid numbering starts with ZERO
        os.environ["KERASTUNER_TUNER_ID"] = "chief"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Keras Tuner Oracle has been assigned.")
    else:
        os.environ["KERASTUNER_TUNER_ID"] = "tuner-" + str(nodename) + "-" + str(procid)
        # os.environ["CUDA_VISIBLE_DEVICES"] = f"{int(int(procid)//workers_per_gpu)}"
        os.environ["CUDA_VISIBLE_DEVICES"] = procid
    print(f'SY DEBUG: procid-{procid} / GPU-ID-{os.environ["CUDA_VISIBLE_DEVICES"]}')

    os.environ["KERASTUNER_ORACLE_IP"] = "localhost"  # Use full hostname
    os.environ["KERASTUNER_ORACLE_PORT"] = oracle_port
    print("KERASTUNER_TUNER_ID:    %s" % os.environ["KERASTUNER_TUNER_ID"])
    print("KERASTUNER_ORACLE_IP:   %s" % os.environ["KERASTUNER_ORACLE_IP"])
    print("KERASTUNER_ORACLE_PORT: %s" % os.environ["KERASTUNER_ORACLE_PORT"])
    # print(os.environ)
    print("<< set_environment END >>")

def continuous_ranked_probability_score(y_true, y_pred):
    """Continuous Ranked Probability Score.

    This implementation is based on the identity:
    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.
    We've closly followed the aproach of 
    https://github.com/TheClimateCorporation/properscoring for
    for the actual implementation.

    Reference
    ---------
    Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
        prediction, and estimation, 2005. University of Washington Department of
        Statistics Technical Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    Args:
    y_true: tf.Tensor.
    y_pred: tf.Tensor.
        Tensors of same shape and type.
    """
    score = tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)), axis=-1)
    diff = tf.subtract(tf.expand_dims(y_pred, -1), tf.expand_dims(y_pred, -2))
    score = tf.add(score, tf.multiply(tf.constant(-0.5, dtype=diff.dtype),tf.reduce_mean(tf.abs(diff),axis=(-2, -1))))

    return tf.reduce_mean(score)


def mse_adjusted(y_true, y_pred):
    se = K.square(y_pred - y_true)
    return K.mean(se[:,:,0:2])*(120/128) + K.mean(se[:,:,2:10])*(8/128)
    

def mae_adjusted(y_true, y_pred):
    ae = K.abs(y_pred - y_true)
    return K.mean(ae[:,:,0:2])*(120/128) + K.mean(ae[:,:,2:10])*(8/128)


class CNNHyperModel():
    def build(self):
        """
        Create a ResNet-style 1D CNN. The data is of shape (batch, lev, vars)
        where lev is treated as the spatial dimension. The architecture
        consists of residual blocks with each two conv layers.
        """
        # Define output shapes
        in_shape = (60, 6)
        out_shape = (60, 10)
        output_length_lin = 2
        output_length_relu = out_shape[-1] - 2

        hp_depth = 12
        hp_channel_width = 406
        hp_kernel_width = 3
        hp_activation = "relu"
        hp_pre_out_activation = "elu"
        hp_norm = False
        hp_dropout = 0.175
        hp_optimizer = "Adam"
        hp_loss = "mean_absolute_error"

        channel_dims = [hp_channel_width] * hp_depth
        kernels = [hp_kernel_width] * hp_depth

        # Initialize special layers
        norm_layer = self.get_normalization_layer(hp_norm)
        if len(channel_dims) != len(kernels):
            print(
                f"[WARNING] Length of channel_dims and kernels does not match. Using 1st argument in kernels, {kernels[0]}, for every layer"
            )
            kernels = [kernels[0]] * len(channel_dims)

        # Initialize model architecture
        input_layer = keras.Input(shape=in_shape)
        x = input_layer  # Set aside input layer
        previous_block_activation = x  # Set aside residual
        for filters, kernel_size in zip(channel_dims, kernels):
            # First conv layer in block
            # 'same' applies zero padding.
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
            # todo: add se_block
            if norm_layer:
                x = norm_layer(x)
            x = keras.layers.Activation(hp_activation)(x)
            x = keras.layers.Dropout(hp_dropout)(x)

            # Second convolution layer
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
            if norm_layer:
                x = norm_layer(x)
            x = keras.layers.Activation(hp_activation)(x)
            x = keras.layers.Dropout(hp_dropout)(x)

            # Project residual
            residual = Conv1D(
                filters=filters, kernel_size=1, strides=1, padding="same"
            )(previous_block_activation)
            x = keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Output layers.
        # x = keras.layers.Dense(filters[-1], activation='gelu')(x) # Add another last layer.
        x = Conv1D(
            out_shape[-1],
            kernel_size=1,
            activation=hp_pre_out_activation,
            padding="same",
        )(x)
        # Assume that vertically resolved variables follow no particular range.
        output_lin = keras.layers.Dense(output_length_lin, activation="linear")(x)
        # Assume that all globally resolved variables are positive.
        output_relu = keras.layers.Dense(output_length_relu, activation="relu")(x)
        output_layer = keras.layers.Concatenate()([output_lin, output_relu])

        model = keras.Model(input_layer, output_layer, name="cnn")

        # Optimizer
        # Set up cyclic learning rate
        INIT_LR = 1e-4
        MAX_LR = 1e-3
        steps_per_epoch = 10091520 // hp_depth
        clr = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=INIT_LR,
            maximal_learning_rate=MAX_LR,
            scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
            step_size=2 * steps_per_epoch,
            scale_mode="cycle",
        )

        # Set up optimizer
        if hp_optimizer == "Adam":
            my_optimizer = keras.optimizers.Adam(learning_rate=clr)
        elif hp_optimizer == "SGD":
            my_optimizer = keras.optimizers.SGD(learning_rate=clr)

        if hp_loss == "mse":
            loss = mse_adjusted
        elif hp_loss == "mean_absolute_error":
            loss = mae_adjusted
        elif hp_loss == "kl_divergence":
            loss = tf.keras.losses.KLDivergence()
        # compile
        model.compile(
            optimizer=my_optimizer,
            loss=loss,
            metrics=["mse", "mae", "accuracy", mse_adjusted, mae_adjusted, continuous_ranked_probability_score],
        )

        print(model.summary())

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )

    def get_normalization_layer(self, norm=None, axis=1):
        """
        Return normalization layer given string
        Args:
            norm string
            axis indices for layer normalization. todo: don't hard-code
        """
        if norm == "layer_norm":
            norm_layer = tf.keras.layers.LayerNormalization(axis=axis)
        elif norm == "batch_norm":
            norm_layer = tf.keras.layers.BatchNormalization()
        else:
            norm_layer = None
        return norm_layer


def decode_fn(record_bytes):
    record = tf.io.parse_single_example(
        record_bytes,
        {
            'X': tf.io.FixedLenFeature([360], dtype=tf.float32),
            'Y': tf.io.FixedLenFeature([600], dtype=tf.float32)
        }
    )
    x = tf.reshape(record["X"], (60, 6))
    #x = x[None, :, :]
    y = tf.reshape(record["Y"], (60, 10))
    #y = y[None, :, :]
    return (x, y)

def data_generator(inputs, targets):
    def generator():
        num_samples = inputs.shape[0]
        for i in range(num_samples):
            yield inputs[i], targets[i]
    
    return generator

def main():
    logging.basicConfig(level=logging.DEBUG)
    #training_data_path = "/shared/ritwik/dev/E3SM-MMF_baseline/data/tfrecords/"
    #train_match = ['E3SM-MMF.mli.000[1234567]-*-*-*.tfrecord', 'E3SM-MMF.mli.0008-01-*-*.tfrecord']
    #val_match = ['E3SM-MMF.mli.0008-0[23456789]-*-*.tfrecord', 'E3SM-MMF.mli.0008-1[012]-*-*.tfrecord', 'E3SM-MMF.mli.0009-01-*-*.tfrecord']
    #train_fnames = get_filenames(Path(training_data_path), train_match)
    #val_fnames = get_filenames(Path(training_data_path), val_match)
    #batch_size = 256

    #ignore_order = tf.data.Options()
    #ignore_order.experimental_deterministic = False

    batch_size = 512  # Adjust the batch size according to your available GPU memory
    shuffle_buffer = 2000
    max_epochs = 15

    start = time.time()
    train_input = np.load("/home/ritwik/e3sm-np/train_input_cnn.npy")
    end = time.time()
    print(f"Took {end - start} seconds to load train_input")

    start = time.time()
    train_target = np.load("/home/ritwik/e3sm-np/train_target_cnn.npy")
    end = time.time()
    print(f"Took {end - start} seconds to load train_target")

    start = time.time()
    val_input = np.load("/home/ritwik/e3sm-np/val_input_cnn.npy")
    end = time.time()
    print(f"Took {end - start} seconds to load val_input")

    start = time.time()
    val_target = np.load("/home/ritwik/e3sm-np/val_target_cnn.npy")
    end = time.time()
    print(f"Took {end - start} seconds to load val_target")


    train_input_generator = data_generator(train_input, train_target)
    val_input_generator = data_generator(val_input, val_target)

    train_ds = tf.data.Dataset.from_generator(train_input_generator,
                                                output_signature=(
                                                tf.TensorSpec(shape=(60, 6), dtype=tf.float32),
                                                tf.TensorSpec(shape=(60, 10), dtype=tf.float32)))
    val_ds = tf.data.Dataset.from_generator(val_input_generator,
                                                output_signature=(
                                                tf.TensorSpec(shape=(60, 6), dtype=tf.float32),
                                                tf.TensorSpec(shape=(60, 10), dtype=tf.float32)))

    train_ds = train_ds.repeat(max_epochs) \
        .shuffle(shuffle_buffer) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size) \
                .prefetch(tf.data.AUTOTUNE)

    #train_ds = tf.data.TFRecordDataset(train_fnames) \
    #    .map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE) \
    #    .with_options(ignore_order) \
    #    .batch(batch_size, drop_remainder=True) \
    #    .prefetch(buffer_size=tf.data.AUTOTUNE) \
    #    .shuffle(50)
    #val_ds = tf.data.TFRecordDataset(val_fnames) \
    #    .map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE) \
    #    .with_options(ignore_order) \
    #    .batch(batch_size, drop_remainder=True) \
    #    .prefetch(buffer_size=tf.data.AUTOTUNE)
    
    #print(train_ds, val_ds)
    logging.debug("Data loaded")

    model = CNNHyperModel().build()

    model.fit(
        train_ds,
        epochs=15,
        steps_per_epoch=10091520//batch_size,
        validation_data=val_ds,
        verbose=1,
        shuffle=True,
        use_multiprocessing=True,
        workers=4,
        callbacks=[
            callbacks.EarlyStopping("val_loss", patience=10),
            callbacks.ModelCheckpoint("/shared/ritwik/dev/E3SM-MMF_baseline/HPO/baseline_v1/results/bair_deep_sweep_final_{epoch}", verbose=1),
        ]
    )


if __name__ == "__main__":
    main()
