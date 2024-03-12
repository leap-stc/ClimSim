# import necessary packages

from climsim_utils.data_utils import *

# set variable names

v2_inputs = ['state_t',
             'state_q0001',
             'state_q0002',
             'state_q0003',
             'state_u',
             'state_v',
             'state_ps',
             'pbuf_SOLIN',
             'pbuf_LHFLX',
             'pbuf_SHFLX',
             'pbuf_TAUX',
             'pbuf_TAUY',
             'pbuf_COSZRS',
             'cam_in_ALDIF',
             'cam_in_ALDIR',
             'cam_in_ASDIF',
             'cam_in_ASDIR',
             'cam_in_LWUP',
             'cam_in_ICEFRAC',
             'cam_in_LANDFRAC',
             'cam_in_OCNFRAC',
             'cam_in_SNOWHICE',
             'cam_in_SNOWHLAND',
             'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
             'pbuf_CH4',
             'pbuf_N2O']

v2_outputs = ['ptend_t',
              'ptend_q0001',
              'ptend_q0002',
              'ptend_q0003',
              'ptend_u',
              'ptend_v',
              'cam_out_NETSW',
              'cam_out_FLWDS',
              'cam_out_PRECSC',
              'cam_out_PRECC',
              'cam_out_SOLS',
              'cam_out_SOLL',
              'cam_out_SOLSD',
              'cam_out_SOLLD']

vertically_resolved = ['state_t', 
                       'state_q0001', 
                       'state_q0002', 
                       'state_q0003', 
                       'state_u', 
                       'state_v', 
                       'pbuf_ozone', 
                       'pbuf_CH4', 
                       'pbuf_N2O', 
                       'ptend_t', 
                       'ptend_q0001', 
                       'ptend_q0002', 
                       'ptend_q0003', 
                       'ptend_u', 
                       'ptend_v']

ablated_vars = ['ptend_q0001',
                'ptend_q0002',
                'ptend_q0003',
                'ptend_u',
                'ptend_v']

v2_vars = v2_inputs + v2_outputs

train_col_names = []
ablated_col_names = []
for var in v2_vars:
    if var in vertically_resolved:
        for i in range(60):
            train_col_names.append(var + '_' + str(i))
            if i < 12 and var in ablated_vars:
                ablated_col_names.append(var + '_' + str(i))
    else:
        train_col_names.append(var)

input_col_names = []
for var in v2_inputs:
    if var in vertically_resolved:
        for i in range(60):
            input_col_names.append(var + '_' + str(i))
    else:
        input_col_names.append(var)

output_col_names = []
for var in v2_outputs:
    if var in vertically_resolved:
        for i in range(60):
            output_col_names.append(var + '_' + str(i))
    else:
        output_col_names.append(var)

assert(len(train_col_names) == 17 + 60*9 + 60*6 + 8)
assert(len(input_col_names) == 17 + 60*9)
assert(len(output_col_names) == 60*6 + 8)
assert(len(set(output_col_names).intersection(set(ablated_col_names))) == len(ablated_col_names))

# initialize data_utils object

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

# do not normalize
data.normalize = False

# create training data

# set data path for training data
data.data_path = '/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train/'

# set regular expressions for selecting training data
data.set_regexps(data_split = 'train', 
                 regexps = ['E3SM-MMF.mli.000[1234567]-*-*-*.nc', # years 1 through 7
                            'E3SM-MMF.mli.0008-01-*-*.nc']) # first month of year 8

# set temporal subsampling
data.set_stride_sample(data_split = 'train', stride_sample = 7)

# create list of files to extract data from
data.set_filelist(data_split = 'train')

# save numpy files of training data
data_loader = data.load_ncdata_with_generator(data_split = 'train')
npy_iterator = list(data_loader.as_numpy_iterator())
npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
npy_output = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
train_npy = np.concatenate([npy_input, npy_output], axis = 1)
train_index = ["train_" + str(x) for x in range(train_npy.shape[0])]

train = pd.DataFrame(train_npy, index = train_index, columns = train_col_names)
train.index.name = 'sample_id'
print('dropping cam_in_SNOWHICE because of strange values')
train.drop('cam_in_SNOWHICE', axis=1, inplace=True)

# ASSERT, SHAPE, CSV, PRINT
assert sum(train.isnull().any()) == 0
print(train.shape)
train.to_csv('recreated/user_data/train.csv')
print('finished creating train data')


# create train output weighting

train_output_weighting = train[output_col_names].copy()
weighting_dict = {}
for col_name in output_col_names:
    weighting_dict[col_name] = 1/max(train_output_weighting[col_name].std(), 1e-15)
    train_output_weighting[col_name] = weighting_dict[col_name]
train_output_weighting[ablated_col_names] = 0

# ASSERT, SHAPE, CSV, PRINT
assert sum(train_output_weighting.isnull().any()) == 0
print(train_output_weighting.shape)
train_output_weighting.to_csv('recreated/user_data/train_output_weighting.csv')
print('finished creating train_output_weighting data')

# creating sample submission
sample_submission = pd.DataFrame(index = shuffled_test_index, columns = output_col_names)
sample_submission.index.name = 'sample_id'
for col_name in output_col_names:
    sample_submission[col_name] = weighting_dict[col_name]
sample_submission[ablated_col_names] = 0

# ASSERT, SHAPE, CSV, PRINT
assert sum(sample_submission.isnull().any()) == 0
print(sample_submission.shape)
sample_submission.head(625000).to_csv('recreated/user_data/sample_submission.csv')
print('finished creating sample_submission data')