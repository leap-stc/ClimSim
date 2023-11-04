from climsim_utils.data_utils import *

grid_path = './grid_info/ClimSim_low-res_grid-info.nc'
norm_path = './preprocessing/normalizations/'

grid_info = xr.open_dataset(grid_path)
input_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc')
input_max = xr.open_dataset(norm_path + 'inputs/input_max.nc')
input_min = xr.open_dataset(norm_path + 'inputs/input_min.nc')
output_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc')

data = data_utils(grid_info = grid_info, 
                  input_mean = input_mean, 
                  input_max = input_max, 
                  input_min = input_min, 
                  output_scale = output_scale)

data.set_to_v2_vars()

# set data path for train data
data.data_path = '/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train/'

# set regular expressions for selecting training data
data.set_regexps(data_split = 'train', 
                 regexps = ['E3SM-MMF.mli.000[1234567]-*-*-*.nc', # years 1 through 7
                            'E3SM-MMF.mli.0008-01-*-*.nc']) # first month of year 8

# set temporal subsampling
data.set_stride_sample(data_split = 'train', stride_sample = 7)

# create list of files to extract data from
data.set_filelist(data_split = 'train')

# do not normalize
data.normalize = False

# save numpy files of training data
data.save_as_npy(data_split = 'train', save_path = '/ocean/projects/ees210014p/jlin96/iclr_npy_arrays/')

# open a file for writing
with open('confirmation_train.txt', 'w') as f:
    # write some text to the file
    f.write('numpy array created!\n')
