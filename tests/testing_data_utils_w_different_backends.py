from climsim_utils.data_utils import *
import argparse
import xarray as xr
from typing import Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MLBackendType = Literal["tensorflow", "pytorch"]

def setup_data_utils(grid_path: str, norm_path: str, train_filelist_directory: str, ml_backend: MLBackendType):  # type: ignore
    # Load data and normalizations

    grid_info = xr.open_dataset(grid_path, engine="netcdf4")
    input_mean = xr.open_dataset(norm_path + "inputs/input_mean.nc", engine="netcdf4")
    input_max = xr.open_dataset(norm_path + "inputs/input_max.nc", engine="netcdf4")
    input_min = xr.open_dataset(norm_path + "inputs/input_min.nc", engine="netcdf4")
    output_scale = xr.open_dataset(norm_path + "outputs/output_scale.nc", engine="netcdf4")

    data = data_utils(
        grid_info=grid_info,
        input_mean=input_mean,
        input_max=input_max,
        input_min=input_min,
        output_scale=output_scale,
        ml_backend=ml_backend,
    )

    # Get all .nc files in the directory at any depth using os
    data.data_path = train_filelist_directory
    data.set_regexps(data_split="train", regexps=["E3SM-MMF.mli.0001-02-01*.nc"])
    data.set_stride_sample(data_split="train", stride_sample=1)
    data.set_filelist(data_split="train")

    data.set_to_v2_vars()

    return data

if __name__ == "__main__":
    # Argument parsing

    parser = argparse.ArgumentParser(description="ClimSim quickstart testing")
    parser.add_argument("--grid_path", type=str, help="Path to grid info")
    parser.add_argument("--norm_path", type=str, help="Path to normalizations")
    parser.add_argument("--train_filelist_directory", type=str, help="List of .nc files for training")
    parser.add_argument("--numpy_save_directory", type=str, help="Directory to save numpy files")

    args = parser.parse_args()

    # Load data and normalizations

    grid_path = args.grid_path
    norm_path = args.norm_path
    train_filelist_directory = args.train_filelist_directory
    numpy_save_directory = args.numpy_save_directory

    # Testing pytorch

    data = setup_data_utils(grid_path=grid_path, norm_path=norm_path, train_filelist_directory=train_filelist_directory, ml_backend="pytorch")
    data.backend = "pytorch"

    torch_data_loader = data.load_ncdata_with_generator(data_split="train")
    torch_data_loader_as_list = list(torch_data_loader)

    input_example, target_example = torch_data_loader_as_list[0]

    logging.info("Pytorch input shape: %s", input_example.shape)
    logging.info("Pytorch target shape: %s", target_example.shape)

    data.save_as_npy(save_path=numpy_save_directory + "train_pytorch_backend", data_split="train")

    # Testing tensorflow

    data = setup_data_utils(grid_path=grid_path, norm_path=norm_path, train_filelist_directory=train_filelist_directory, ml_backend="tensorflow")

    tensforflow_dataset = data.load_ncdata_with_generator(data_split="train")
    tensorflow_dataset_as_list = list(tensforflow_dataset)

    input_example, target_example = tensorflow_dataset_as_list[0]

    logging.info("Tensorflow input shape: %s", input_example.shape)
    logging.info("Tensorflow target shape: %s", target_example.shape)

    # Save dataset as npy
    data.save_as_npy(save_path=numpy_save_directory + "train_tensorflow_backend", data_split="train")

    logging.info("Things look reasonable!")