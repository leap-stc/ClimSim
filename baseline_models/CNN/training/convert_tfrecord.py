from tqdm import tqdm

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import random

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
import os

import multiprocessing as mp

def pad_and_stack_layers_and_vars_1d(ds, dso):
    """
    Pads and stack all variables into (batch, n_vertical_levels, n_variables),
    e.g., input: (batch, 60, 6) and output: (batch, 60, 10)
    Args:
        ds xarray.Dataset(lev, ncol) with vars_mli of shapes (lev, ncol) and (ncol)
        dso xarray.Dataset(lev, ncol) with vars_mlo of shapes (lev, ncol) and (ncol)
    Returns:
        arr xarray.DataArray(batch, lev, variable)
        arro xarray.DataArray(batch, lev, variable)
    """
    ds = ds.stack({"batch": {"ncol"}})
    (ds,) = xr.broadcast(ds)  # repeat global variables across levels
    arr = ds.to_array("mlvar", name="mli")
    arr = arr.transpose("batch", "lev", "mlvar")

    dso = dso.stack({"batch": {"ncol"}})
    (dso,) = xr.broadcast(dso)
    arro = dso.to_array("mlvar", name="mlo")
    arro = arro.transpose("batch", "lev", "mlvar")

    return arr, arro

norm_root = Path("/jet/home/rgupta7/E3SM-MMF_baseline/norm_factors/")
mli_mean = xr.open_dataset(norm_root / "mli_mean.nc")
mli_min = xr.open_dataset(norm_root / "mli_min.nc")
mli_max = xr.open_dataset(norm_root / "mli_max.nc")
mlo_scale = xr.open_dataset(norm_root / "mlo_scale.nc")

def _convert_record(f):
    ds = xr.open_dataset(f, engine="netcdf4")
    ds = ds[['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']]

    # read mlo
    dso = xr.open_dataset(Path(str(f).replace(".mli.", ".mlo.")), engine="netcdf4")

    # make mlo variales: ptend_t and ptend_q0001
    dso["ptend_t"] = (
        dso["state_t"] - ds["state_t"]
    ) / 1200  # T tendency [K/s]
    dso["ptend_q0001"] = (
        dso["state_q0001"] - ds["state_q0001"]
    ) / 1200  # Q tendency [kg/kg/s]
    dso = dso[['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']]

    # normalization, scaling
    ds = (ds - mli_mean) / (mli_max - mli_min)
    dso = dso * mlo_scale

    ds, dso = pad_and_stack_layers_and_vars_1d(ds, dso)

    folder = "-".join(f.stem.split(".")[-1].split("-")[:2])
    writer = tf.io.TFRecordWriter(f"/ocean/projects/atm200007p/rgupta7/e3sm_train_tfrecord/{f.stem}.tfrecord")

    for inp, out in zip(ds, dso):
        record = tf.train.Features(feature={
            'X': tf.train.Feature(float_list=tf.train.FloatList(value=inp.values.reshape(-1))),
            'Y': tf.train.Feature(float_list=tf.train.FloatList(value=out.values.reshape(-1)))
        })
        example = tf.train.Example(features=record)
        writer.write(example.SerializeToString())

fnames = Path("/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train/").glob("**/*mli*.nc")
pool = mp.Pool(64)
r = list(tqdm(pool.imap(_convert_record, fnames), total=210240))
