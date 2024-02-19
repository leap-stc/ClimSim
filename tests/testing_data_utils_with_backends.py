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
    parser.add_argument("--ml_backends", nargs="+", help="ML backends (tensorflow or pytorch)")

    args = parser.parse_args()

    # Load data and normalizations

    grid_path = args.grid_path
    norm_path = args.norm_path
    train_filelist_directory = args.train_filelist_directory
    numpy_save_directory = args.numpy_save_directory
    ml_backends = args.ml_backends

    if len(ml_backends) == 0:
        raise ValueError("No ML backends provided. Please provide at least one backend.")

    if len(ml_backends) > 2:
        raise ValueError("More than two ML backends provided. Please provide a correct number of backends.")

    for ml_backend in ml_backends:
        logging.info(f"Testing {ml_backend} backend")

        data = setup_data_utils(grid_path=grid_path, norm_path=norm_path, train_filelist_directory=train_filelist_directory, ml_backend=ml_backend)

        dataset = data.load_ncdata_with_generator(data_split="train")
        dataset_as_list = list(dataset)

        input_example, target_example = dataset_as_list[0]

        logging.info(f"{ml_backend} input shape: {input_example.shape}")
        logging.info(f"{ml_backend} target shape: {target_example.shape}")

        data.save_as_npy(save_path=numpy_save_directory + f"{ml_backend}_backend_", data_split="train")
        logging.info(f"Saved {ml_backend} backend data as numpy files")

    if ml_backends == ["tensorflow", "pytorch"] or ml_backends == ["pytorch", "tensorflow"]:
        logging.info("Comparing numpy files from different backends")

        # Load NumPy files and assert they are the same
        tensorflow_train_input = np.load(numpy_save_directory + "tensorflow_backend_train_input.npy")
        tensorflow_train_target = np.load(numpy_save_directory + "tensorflow_backend_train_target.npy")

        pytorch_train_input = np.load(numpy_save_directory + "pytorch_backend_train_input.npy")
        pytorch_train_target = np.load(numpy_save_directory + "pytorch_backend_train_target.npy")

        assert np.array_equal(tensorflow_train_input, pytorch_train_input), "Tensorflow and PyTorch input data are not equal."
        assert np.array_equal(tensorflow_train_target, pytorch_train_target), "Tensorflow and PyTorch target data are not equal."

    logging.info("Things look reasonable!")