import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob, os
import re
import tensorflow as tf
import netCDF4
import h5py
from tqdm import tqdm

class data_utils:
    
    def __init__(self,
                 data_path, 
                 input_vars, 
                 target_vars, 
                 grid_info,
                 inp_mean,
                 inp_max,
                 inp_min,
                 out_scale):
        self.data_path = data_path
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.grid_info = grid_info
        self.level_name = 'lev'
        self.sample_name = 'sample'
        self.latlonnum = len(self.grid_info['ncol']) # number of unique lat/lon grid points
        # make area-weights
        self.grid_info['area_wgt'] = self.grid_info['area']/self.grid_info['area'].mean(dim = 'ncol')
        # map ncol to nsamples dimension
        # to_xarray = {'area_wgt':(self.sample_name,np.tile(self.grid_info['area_wgt'], int(n_samples/len(self.grid_info['ncol']))))}
        # to_xarray = xr.Dataset(to_xarray)
        self.inp_mean = inp_mean
        self.inp_max = inp_max
        self.inp_min = inp_min
        self.out_scale = out_scale
        self.lats, self.lats_indices = np.unique(self.grid_info['lat'].values, return_index=True)
        self.lons, self.lons_indices = np.unique(self.grid_info['lon'].values, return_index=True)
        self.sort_lat_key = np.argsort(self.grid_info['lat'].values[np.sort(self.lats_indices)])
        self.sort_lon_key = np.argsort(self.grid_info['lon'].values[np.sort(self.lons_indices)])
        self.indextolatlon = {i: (self.grid_info['lat'].values[i%self.latlonnum], self.grid_info['lon'].values[i%self.latlonnum]) for i in range(self.latlonnum)}
        
        def find_keys(dictionary, value):
            keys = []
            for key, val in dictionary.items():
                if val[0] == value:
                    keys.append(key)
            return keys
        indices_list = []
        for lat in self.lats:
            indices = find_keys(self.indextolatlon, lat)
            indices_list.append(indices)
        indices_list.sort(key = lambda x: x[0])
        self.lat_indices_list = indices_list

        self.hyam = self.grid_info['hyam'].values
        self.hybm = self.grid_info['hybm'].values
        self.p0 = 1e5 # code assumes this will always be a scalar
        self.train_regexps = None
        self.train_stride_sample = None
        self.train_filelist = None
        self.val_regexps = None
        self.val_stride_sample = None
        self.val_filelist = None
        self.scoring_regexps = None
        self.scoring_stride_sample = None
        self.scoring_filelist = None
        self.test_regexps = None
        self.test_stride_sample = None
        self.test_filelist = None
        # physical constants from E3SM_ROOT/share/util/shr_const_mod.F90
        self.grav    = 9.80616    # acceleration of gravity ~ m/s^2
        self.cp      = 1.00464e3  # specific heat of dry air   ~ J/kg/K
        self.lv      = 2.501e6    # latent heat of evaporation ~ J/kg
        self.lf      = 3.337e5    # latent heat of fusion      ~ J/kg
        self.lsub    = self.lv + self.lf    # latent heat of sublimation ~ J/kg
        self.rho_air = 101325/(6.02214e26*1.38065e-23/28.966)/273.15 # density of dry air at STP  ~ kg/m^3
                                                                    # ~ 1.2923182846924677
                                                                    # SHR_CONST_PSTD/(SHR_CONST_RDAIR*SHR_CONST_TKFRZ)
                                                                    # SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
                                                                    # SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
        self.rho_h20 = 1.e3       # density of fresh water     ~ kg/m^ 3
        self.target_var_len = {'ptend_t':60,
                               'ptend_q0001':60,
                               'cam_out_NETSW':1,
                               'cam_out_FLWDS':1,
                               'cam_out_PRECSC':1,
                               'cam_out_PRECC':1,
                               'cam_out_SOLS':1,
                               'cam_out_SOLL':1,
                               'cam_out_SOLSD':1,
                               'cam_out_SOLLD':1
                               }
        self.target_energy_conv = {'ptend_t':self.cp,
                                   'ptend_q0001':self.lv,
                                   'cam_out_NETSW':1.,
                                   'cam_out_FLWDS':1.,
                                   'cam_out_PRECSC':self.lv*self.rho_h20,
                                   'cam_out_PRECC':self.lv*self.rho_h20,
                                   'cam_out_SOLS':1.,
                                   'cam_out_SOLL':1.,
                                   'cam_out_SOLSD':1.,
                                   'cam_out_SOLLD':1.
                                  }
        self.target_short_name = {'ptend_t': 'dT/dt', 
                                  'ptend_q0001':'dq/dt', 
                                  'cam_out_NETSW':  'NETSW',
                                  'cam_out_FLWDS':  'FLWDS',
                                  'cam_out_PRECSC': 'PRECSC',
                                  'cam_out_PRECC': 'PRECC',
                                  'cam_out_SOLS': 'SOLS',
                                  'cam_out_SOLL': 'SOLL',
                                  'cam_out_SOLSD': 'SOLSD',
                                  'cam_out_SOLLD': 'SOLLD',
                                  }
        # for V1 output (limited subset)
        self.var_idx = {}
        self.var_idx['ptend_t'] = (0,60)
        self.var_idx['ptend_q0001'] = (60, 120)
        self.var_idx['surface_vars'] = (120, 128)

        # for metrics
        self.crps_compatible = ["HSR", "RPN", "cVAE"]
        self.samples_scoring = None
        self.preds_scoring = None
        self.input_scoring = None
        self.target_scoring = None
        self.model_names = None
        self.metric_names = None
        self.linecolors = {'CNN':  '#0072B2', 
                           'HSR':  '#E69F00', 
                           'MLP':  '#882255', 
                           'RPN':  '#009E73', 
                           'cVAE': '#D55E00' 
                           }

    def get_xrdata(self, file, file_vars = None):
        '''
        This function reads in a file and returns an xarray dataset with the variables specified.
        file_vars must be a list of strings.
        '''
        ds = xr.open_dataset(file, engine = 'netcdf4')
        if file_vars is not None:
            ds = ds[file_vars]
        ds = ds.merge(self.grid_info[['lat','lon']])
        ds = ds.where((ds['lat']>-999)*(ds['lat']<999), drop=True)
        ds = ds.where((ds['lon']>-999)*(ds['lon']<999), drop=True)
        return ds

    def get_input(self, input_file):
        '''
        This function reads in a file and returns an xarray dataset with the input variables for the emulator.
        '''
        # read inputs
        return self.get_xrdata(input_file, self.input_vars)

    def get_target(self, input_file):
        '''
        This function reads in a file and returns an xarray dataset with the target variables for the emulator.
        '''
        # read inputs
        ds_input = self.get_input(input_file)
        ds_target = self.get_xrdata(input_file.replace('.mli.','.mlo.'))
        # each timestep is 20 minutes which corresponds to 1200 seconds
        ds_target['ptend_t'] = (ds_target['state_t'] - ds_input['state_t'])/1200 # T tendency [K/s]
        ds_target['ptend_q0001'] = (ds_target['state_q0001'] - ds_input['state_q0001'])/1200 # Q tendency [kg/kg/s]
        ds_target = ds_target[self.target_vars]
        return ds_target
    
    def set_regexps(self, data_split, regexps):
        '''
        This function sets the regular expressions used for getting the filelist for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            self.train_regexps = regexps
        elif data_split == 'val':
            self.val_regexps = regexps
        elif data_split == 'scoring':
            self.scoring_regexps = regexps
        elif data_split == 'test':
            self.test_regexps = regexps
    
    def set_stride_sample(self, data_split, stride_sample):
        '''
        This function sets the stride_sample for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            self.train_stride_sample = stride_sample
        elif data_split == 'val':
            self.val_stride_sample = stride_sample
        elif data_split == 'scoring':
            self.scoring_stride_sample = stride_sample
        elif data_split == 'test':
            self.test_stride_sample = stride_sample
    
    def set_filelist(self, data_split):
        '''
        This function sets the filelists corresponding to data splits for train, val, scoring, and test.
        '''
        filelist = []
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            assert self.train_regexps is not None, 'regexps for train is not set.'
            assert self.train_stride_sample is not None, 'stride_sample for train is not set.'
            for regexp in self.train_regexps:
                filelist = filelist + glob.glob(self.data_path + "*/" + regexp)
            self.train_filelist = sorted(filelist)[::self.train_stride_sample]
        elif data_split == 'val':
            assert self.val_regexps is not None, 'regexps for val is not set.'
            assert self.val_stride_sample is not None, 'stride_sample for val is not set.'
            for regexp in self.val_regexps:
                filelist = filelist + glob.glob(self.data_path + "*/" + regexp)
            self.val_filelist = sorted(filelist)[::self.val_stride_sample]
        elif data_split == 'scoring':
            assert self.scoring_regexps is not None, 'regexps for scoring is not set.'
            assert self.scoring_stride_sample is not None, 'stride_sample for scoring is not set.'
            for regexp in self.scoring_regexps:
                filelist = filelist + glob.glob(self.data_path + "*/" + regexp)
            self.scoring_filelist = sorted(filelist)[::self.scoring_stride_sample]
        elif data_split == 'test':
            assert self.test_regexps is not None, 'regexps for test is not set.'
            assert self.test_stride_sample is not None, 'stride_sample for test is not set.'
            for regexp in self.test_regexps:
                filelist = filelist + glob.glob(self.data_path + "*/" + regexp)
            self.test_filelist = sorted(filelist)[::self.test_stride_sample]

    def get_filelist(self, data_split):
        '''
        This function returns the filelist corresponding to data splits for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            assert self.train_filelist is not None, 'filelist for train is not set.'
            return self.train_filelist
        elif data_split == 'val':
            assert self.val_filelist is not None, 'filelist for val is not set.'
            return self.val_filelist
        elif data_split == 'scoring':
            assert self.scoring_filelist is not None, 'filelist for scoring is not set.'
            return self.scoring_filelist
        elif data_split == 'test':
            assert self.test_filelist is not None, 'filelist for test is not set.'
            return self.test_filelist

    def get_pressure_grid(self, data_split):
        '''
        This function creates the temporally and zonally averaged pressure grid corresponding to a given data split.
        '''
        filelist = self.get_filelist(data_split)
        ps = np.concatenate([self.get_xrdata(file, ['state_ps'])['state_ps'].values[np.newaxis, :] for file in tqdm(filelist)], axis = 0)[:, :, np.newaxis]
        hyam_component = self.hyam[np.newaxis, np.newaxis, :]*self.p0
        hybm_component = self.hybm[np.newaxis, np.newaxis, :]*ps
        pressures = np.mean(hyam_component + hybm_component, axis = 0)
        pg_lats = []
        def find_keys(dictionary, value):
            keys = []
            for key, val in dictionary.items():
                if val[0] == value:
                    keys.append(key)
            return keys
        for lat in self.lats:
            indices = find_keys(self.indextolatlon, lat)
            pg_lats.append(np.mean(pressures[indices, :], axis = 0)[:, np.newaxis])
        pressure_grid = np.concatenate(pg_lats, axis = 1)
        return pressure_grid
    
    def load_ncdata_with_generator(self, data_split):
        '''
        This function works as a dataloader when training the emulator with raw netCDF files.
        This can be used as a dataloader during training or it can be used to create entire datasets.
        When used as a dataloader for training, I/O can slow down training considerably.
        This function also normalizes the data.
        mli corresponds to input
        mlo corresponds to target
        '''
        filelist = self.get_filelist(data_split)
        def gen():
            for file in filelist:
                # read inputs
                ds_input = self.get_input(file)
                # read targets
                ds_target = self.get_target(file)
                
                # normalization, scaling
                ds_input = (ds_input - self.inp_mean)/(self.inp_max - self.inp_min)
                ds_target = ds_target*self.out_scale

                # stack
                # ds = ds.stack({'batch':{'sample','ncol'}})
                ds_input = ds_input.stack({'batch':{'ncol'}})
                ds_input = ds_input.to_stacked_array('mlvar', sample_dims=['batch'], name='mli')
                # dso = dso.stack({'batch':{'sample','ncol'}})
                ds_target = ds_target.stack({'batch':{'ncol'}})
                ds_target = ds_target.to_stacked_array('mlvar', sample_dims=['batch'], name='mlo')
                yield (ds_input.values, ds_target.values)

        return tf.data.Dataset.from_generator(
            gen,
            output_types = (tf.float64, tf.float64),
            output_shapes = ((None,124),(None,128))
        )
    
    def save_as_npy(self,
                 data_split, 
                 save_path = '', 
                 save_latlontime_dict = False):
        '''
        This function saves the training data as a .npy file. Prefix should be train or val.
        '''
        prefix = save_path + data_split
        data_loader = self.load_ncdata_with_generator(data_split)
        npy_iterator = list(data_loader.as_numpy_iterator())
        npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
        npy_target = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
        with open(save_path + prefix + '_input.npy', 'wb') as f:
            np.save(f, np.float32(npy_input))
        with open(save_path + prefix + '_target.npy', 'wb') as f:
            np.save(f, np.float32(npy_target))
        if data_split == 'train':
            data_files = self.train_filelist
        elif data_split == 'val':
            data_files = self.val_filelist
        elif data_split == 'scoring':
            data_files = self.scoring_filelist
        elif data_split == 'test':
            data_files = self.test_filelist
        if save_latlontime_dict:
            dates = [re.sub('^.*mli\.', '', x) for x in data_files]
            dates = [re.sub('\.nc$', '', x) for x in dates]
            repeat_dates = []
            for date in dates:
                for i in range(self.latlonnum):
                    repeat_dates.append(date)
            latlontime = {i: [(self.grid_info['lat'].values[i%self.latlonnum], self.grid_info['lon'].values[i%self.latlonnum]), repeat_dates[i]] for i in range(npy_input.shape[0])}
            with open(save_path + prefix + '_indextolatlontime.pkl', 'wb') as f:
                pickle.dump(latlontime, f)
    
    def reshape_npy(self, var_arr, var_arr_dim):
        '''
        This function reshapes the a variable in numpy such that time gets its own axis (instead of being num_samples x num_levels).
        Shape of target would be (timestep, lat/lon combo, num_levels)
        '''
        var_arr = var_arr.reshape((int(var_arr.shape[0]/self.latlonnum), self.latlonnum, var_arr_dim))
        return var_arr

    @staticmethod
    def ls(dir_path = ''):
        '''
        You can treat this as a Python wrapper for the bash command "ls".
        '''
        return os.popen(' '.join(['ls', dir_path])).read().splitlines()
    
    @staticmethod
    def set_plot_params():
        '''
        This function sets the plot parameters for matplotlib.
        '''
        plt.close('all')
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rc('font', family='sans')
        plt.rcParams.update({'font.size': 32,
                            'lines.linewidth': 2,
                            'axes.labelsize': 32,
                            'axes.titlesize': 32,
                            'xtick.labelsize': 32,
                            'ytick.labelsize': 32,
                            'legend.fontsize': 32,
                            'axes.linewidth': 2,
                            "pgf.texsystem": "pdflatex"
                            })
        # %config InlineBackend.figure_format = 'retina'
        # use the above line when working in a jupyter notebook

    @staticmethod
    def get_pred_npy(load_path = ''):
        '''
        This function loads the prediction .npy file.
        '''
        with open(load_path, 'rb') as f:
            pred = np.load(f)
        return pred
    
    @staticmethod
    def get_pred_h5(load_path = ''):
        '''
        This function loads the prediction .h5 file.
        '''
        hf = h5py.File(load_path, 'r')
        pred = np.array(hf.get('pred'))
        return pred

    def reshape_daily(self, output):
        '''
        This function returns two numpy arrays, one for each vertically resolved variable (heating and moistening).
        Dimensions of expected input are num_samples by 128 (number of target features).
        Data is expected to use a stride_sample of 6. (12 samples per day, 20 min timestep)
        '''
        num_samples = output.shape[0]
        heating = output[:,:60].reshape((int(num_samples/self.latlonnum), self.latlonnum, 60))
        moistening = output[:,60:120].reshape((int(num_samples/self.latlonnum), self.latlonnum, 60))
        heating_daily = np.mean(heating.reshape((heating.shape[0]//12, 12, self.latlonnum, 60)), axis = 1) # Nday x lotlonnum x 60
        moistening_daily = np.mean(moistening.reshape((moistening.shape[0]//12, 12, self.latlonnum, 60)), axis = 1) # Nday x lotlonnum x 60
        heating_daily_long = []
        moistening_daily_long = []
        for i in range(len(self.lats)):
            heating_daily_long.append(np.mean(heating_daily[:,self.lat_indices_list[i],:],axis=1))
            moistening_daily_long.append(np.mean(moistening_daily[:,self.lat_indices_list[i],:],axis=1))
        heating_daily_long = np.array(heating_daily_long) # lat x Nday x 60
        moistening_daily_long = np.array(moistening_daily_long) # lat x Nday x 60
        return heating_daily_long, moistening_daily_long
    
    # def update_grid_info(self, output):
    #     '''
    #     This function updates grid_info such that the num_samples dimension of the numpy array gets mapped to ncol from grid_info
    #     '''
    #     num_samples = output.shape[0]
    #     to_xarray = {'area_wgt':(self.sample_name,np.tile(self.grid_info['area_wgt'], int(num_samples/len(self.grid_info['ncol']))))}
    #     to_xarray = xr.Dataset(to_xarray)
    #     self.grid_info = xr.merge([self.grid_info[['P0','hyai','hyam','hybi','hybm','lat','lon','area']],
    #                                to_xarray[['area_wgt']]])
        
    # def make_xarray(self, input, output):
    #     '''
    #     This turns numpy output into xarray output, area weights, makes appropriate unit conversions.
    #     Output can either be target or prediction.
    #     This function only works with V1 variables.
    #     Needs updating for V2.
    #     '''
    #     num_samples = output.shape[0]
    #     to_xarray = {}
    #     for k, kvar in enumerate(self.target_vars):
    #         # length of variable (ie number of levels)
    #         kvar_len = self.target_var_len[kvar]
    #         # set dimensions of variable
    #         if kvar_len == 60:
    #             kvar_dims = (self.sample_name, self.level_name)
    #         elif kvar_len == 1:
    #             kvar_dims = self.sample_name

    #         # set start and end indices of variable in the loaded numpy array
    #         # then, add 'kvar':(kvar_dims, <np_array>) to dictionary
    #         if k == 0:
    #             ind1 = 0
    #         ind2 = ind1 + kvar_len

    #         # scaled output
    #         kvar_data = np.squeeze(output[:,ind1:ind2])
    #         # unscaled output
    #         kvar_data = kvar_data/self.out_scale[kvar].values
    #         to_xarray[kvar] = [kvar_dims, kvar_data]
    #         ind1 = ind2
        
    #     # convert dict to xarray dataset
    #     xr_output = xr.Dataset(to_xarray)
    #     # add surface pressure ('state_ps') from ml input
    #     # normalized ps
    #     # this assumes state_ps is the third variable in the input, after heating and moistening tendencies
    #     state_ps = xr.DataArray(input[:, 120], dims = ('sample'), name = 'state_ps')
    #     # denormalized ps
    #     state_ps = state_ps * (self.inp_max['state_ps'] - self.inp_min['state_ps']) + self.inp_mean['state_ps']
    #     xr_output['state_ps'] = state_ps

    #     # add grid info
    #     xr_output = xr.merge([xr_output, self.grid_info])

    #     # add pressure thickness of each level, dp
    #     # FYI, in a hybrid sigma vertical coordinate system, pressure at level z is
    #     # P[x, z] = hyam[z] * P0 + hybm[z] * PS[x,z]
    #     tmp = xr_output['P0']*xr_output['hyai'] + xr_output['state_ps']*xr_output['hybi']
    #     tmp = tmp.isel(ilev = slice(1,61)).values - tmp.isel(ilev = slice(0,60)).values
    #     tmp = tmp.transpose()
    #     xr_output['dp'] = xr.DataArray(tmp, dims = ('sample', 'lev'))
        
    #     # break (sample) to (ncol, time)
    #     num_timesteps = int(num_samples/self.latlonnum)
    #     dim_ncol = np.arange(self.latlonnum)
    #     dim_timestep = np.arange(num_timesteps)
    #     new_ind = pd.MultiIndex.from_product([dim_timestep, dim_ncol], names = ['time', 'ncol'])
    #     xr_output = xr_output.assign_coords(sample = new_ind).unstack('sample')

    #     for kvar in self.target_vars:
    #         # weight vertical levels by dp/g
    #         # only for vertically-resolved variables, e.g. ptend_{t, q0001}
    #         # dp/g = - \rho * dz
    #         if self.target_var_len[kvar] == 60:
    #             xr_output[kvar] = xr_output[kvar] * xr_output['dp']/self.grav

    #         # weight area for all variables
    #         xr_output[kvar] = xr_output[kvar] * xr_output['area_wgt']

    #         # convert units to W/m2
    #         xr_output[kvar] = self.target_energy_conv[kvar] * xr_output[kvar]
    #     return xr_output
    
    # def calc_MAE(self, pred, target):
    #     '''
    #     This function takes in two xarray objects, prediction and target, and returns the mean absolute error. 
    #     '''
    #     metric = (np.abs(target - pred)).mean(dim = 'time')
    #     return metric.mean(dim = 'ncol')
    
    # def calc_RMSE(self, pred, target):
    #     '''
    #     This function takes in two xarray objects, prediction and target, and returns the root mean squared error.
    #     '''
    #     metric = np.sqrt(((target - pred)**2.).mean(dim = 'time'))
    #     return metric.mean(dim = 'ncol')
    
    # def calc_R2(self, pred, target):
    #     '''
    #     This function takes in two xarray objects, prediction and target, and returns the R2 score.
    #     '''
    #     sum_squared_residuals = ((target - pred)**2).sum(dim = 'time')
    #     sum_squared_diffs = ((target - target.mean(dim = 'time'))**2).sum(dim = 'time')
    #     metric = 1 - sum_squared_residuals/sum_squared_diffs
    #     return metric.mean(dim = 'ncol')
    
    # def calc_CRPS(self, pred, target):
    #     '''
    #     This function takes in prediction and target and returns the CRPS
    #     '''
    #     pass
    
    # def stack_metric(self, metric, metric_name):
    #     '''
    #     This function takes in a metric xarray object and the name of the metric and flattens the level dimension.
    #     '''
    #     if metric != 'CRPS':
    #         metric_stacked = metric.to_stacked_array('ml_out_idx', sample_dims = '', name = metric_name)
    #         return metric_stacked.values
    #     else:
    #         pass
    
    # def df_stack_metric(self, metrics):
    #     '''
    #     This function takes in a list of metric xarray objects and the names of the metrics and creates a pandas Dataframe.
    #     Index is output_idx.
    #     '''
    #     assert len(metrics) == len(self.metric_names), 'Number of metrics and metric names must be the same.'
    #     df = pd.DataFrame(dict(zip(self.metric_names, metrics)))
    #     df.index.name = 'output_idx'
    #     return df
    
    # def vert_avg_metric(self, metric):
    #     '''
    #     This function takes in a metric xarray object and the name of the metric and averages over the vertical dimension.
    #     '''
    #     if metric != "CRPS":
    #         metric_vert_avg = metric.mean(dim = 'lev')
    #         metric_vert_avg = metric_vert_avg.mean(dim = 'ilev') # removing dummy dimension
    #         return metric_vert_avg.to_pandas()
    #     else:
    #         pass

    # def df_vert_avg_metric(self, metrics):
    #     '''
    #     This function takes in a list of metric xarray objects and the names of the metrics and creates a pandas Dataframe.
    #     Index is variable name.
    #     '''
    #     assert len(metrics) == len(self.metric_names), 'Number of metrics and metric names must be the same.'
    #     df = pd.DataFrame(dict(zip(self.metric_names, metrics)))
    #     df.index.name = 'Variable'
    #     return df
    
    # def plot_metrics_lev_agg(self):
    #     '''
    #     This function plots level aggregated metrics. Included in main text.
    #     '''
    #     self.set_plot_params()
    #     assert self.preds_scoring is not None, 'No predictions for scoring.'
    #     assert self.target_scoring is not None, 'Target not set.'
    #     model_metrics = {}
    #     for model_name in self.model_names:
    #         metrics = []
    #         for metric_name in tqdm(self.metric_names):
    #             if metric_name == 'MAE':
    #                 metrics.append(self.vert_avg_metric(self.calc_MAE(self.preds_scoring[model_name], self.target_scoring)))
    #             elif metric_name == 'RMSE':
    #                 metrics.append(self.vert_avg_metric(self.calc_RMSE(self.preds_scoring[model_name], self.target_scoring)))
    #             elif metric_name == 'R2':
    #                 metrics.append(self.vert_avg_metric(self.calc_R2(self.preds_scoring[model_name], self.target_scoring)))
    #             elif metric_name == 'CRPS':
    #                 if model_name in self.crps_compatible:
    #                     metrics.append(self.vert_avg_metric(self.calc_CRPS(self.preds_scoring[model_name], self.target_scoring)))
    #         df_vert_avg = self.df_vert_avg_metric(metrics)
    #         model_metrics[model_name] = df_vert_avg
    #     plotting_dict = {}
    #     for metric_name in self.metric_names:
    #         if metric_name == 'CRPS':
    #             plot_index = self.crps_compatible
    #         else:
    #             plot_index = self.model_names
    #         plotting_dict[metric_name] = pd.DataFrame([model_metrics[model_name] for model_name in plot_index], index = plot_index)

    #     abc='abcdefg'
    #     fig, _ax = plt.subplots(nrows  = len(self.metric_names), 
    #                             sharex = True)

    #     for k, kmetric in enumerate(self.metric_names):
    #         ax = _ax[k]
    #         plotdata = plotting_dict[kmetric]
    #         plotdata = plotdata.rename(columns=self.target_short_name)
    #         plotdata = plotdata.transpose()
    #         plotdata.plot.bar(color = [lc_model[kmodel] for kmodel in plotdata.keys()],
    #                         # width = .2,
    #                         legend = False,
    #                         ax=ax)

    #         ax.set_title(f'({abc[k]}) {kmetric}')
    #         ax.set_xlabel('Output variable')
    #         ax.set_xticklabels(plotdata.index, rotation=0, ha='center')

    #         # no units for R2
    #         # log y scale
    #         if kmetric != 'R2':
    #             ax.set_ylabel('W/m2')
    #             ax.set_yscale('log')
            
    #         # not plotting negative R2 values
    #         if kmetric == 'R2':
    #             ax.set_ylim(0,1)

    #         fig.set_size_inches(7, 8)

    #     _ax[0].legend(ncols=3, columnspacing=.9, labelspacing=.3,
    #                 handleheight=.07, handlelength=1.5, handletextpad=.2,
    #                 borderpad=.2,
    #                 loc='upper right')
        
    #     fig.tight_layout()
    #     return
    
    # def plot_metrics_vert_prof(self):
    #     '''
    #     This function plots vertical profile of metrics for tendency variables. Included in SI.
    #     '''
    #     self.set_plot_params()
    #     assert self.preds_scoring is not None, 'No predictions for scoring.'
    #     assert self.target_scoring is not None, 'Target not set.'
    #     model_metrics = {}
    #     for model_name in self.model_names:
    #         metrics = []
    #         for metric_name in tqdm(self.metric_names):
    #             if metric_name == "MAE":
    #                 metrics.append(self.stack_metric(self.calc_MAE(self.preds_scoring[model_name], self.target_scoring), metric_name = metric_name))
    #             elif metric_name == "RMSE":
    #                 metrics.append(self.stack_metric(self.calc_RMSE(self.preds_scoring[model_name], self.target_scoring), metric_name = metric_name))
    #             elif metric_name == "R2":
    #                 metrics.append(self.stack_metric(self.calc_R2(self.preds_scoring[model_name], self.target_scoring), metric_name = metric_name))
    #             elif metric_name == "CRPS":
    #                 if model_name in self.crps_compatible:
    #                     metrics.append(self.stack_metric(self.calc_CRPS(self.preds_scoring[model_name], self.target_scoring), metric_name = metric_name))
    #         df_stack = self.df_stack_metric(metrics)
    #         model_metrics[model_name] = df_stack
    #     plotting_dict = {}
    #     for metric_name in self.metric_names:
    #         if metric_name == "CRPS":
    #             plot_index = self.crps_compatible
    #         else:
    #             plot_index = self.model_names
    #         plotting_dict[metric_name] = pd.DataFrame([model_metrics[model_name] for model_name in plot_index], index = plot_index)

    #     abc='abcdefg'
    #     for kvar in ['ptend_t','ptend_q0001']:
    #         fig, _ax = plt.subplots(ncols=2, nrows=2)
    #         _ax = _ax.flatten()
    #         for k, kmetric in enumerate(self.metric_names):
    #             ax = _ax[k]
    #             idx_start = self.var_idx[kvar][0]
    #             idx_end = self.var_idx[kvar][1]
    #             plotdata = plotting_dict[kmetric].iloc[:,idx_start:idx_end]
    #             if kvar == 'ptend_q0001':
    #                 plotdata.columns = plotdata.columns - 60
    #             if kvar=='ptend_q0001': # this is to keep the right x axis range.
    #                 plotdata = plotdata.where(~np.isinf(plotdata),-999)
    #             plotdata = plotdata.transpose()
    #             plotdata.plot(color = [self.linecolors[kmodel] for kmodel in plotdata.keys()],
    #                         legend=False,
    #                         ax=ax,
    #                         )

    #             ax.set_xlabel('Level index')
    #             ax.set_title(f'({abc[k]}) {kmetric} ({self.target_short_name[kvar]})')
    #             if kmetric != 'R2':
    #                 ax.set_ylabel('W/m2')

    #             # R2 ylim
    #             if  (kmetric=='R2'):
    #                 ax.set_ylim(0,1.05)

    #         # legend
    #         _ax[0].legend(ncols=1, labelspacing=.3,
    #                 handleheight=.07, handlelength=1.5, handletextpad=.2,
    #                 borderpad=.3,
    #                 loc='upper left')
            
    #         fig.tight_layout()
    #         fig.set_size_inches(7,4.5)
    #     return
    
    # def plot_combined(self):
    #     '''
    #     This function plots vertical profiles and level aggregated metrics (MAE and R2). Figure is in main text.
    #     '''
    #     self.set_plot_params()
    #     assert self.preds_scoring is not None, 'No predictions for scoring.'
    #     assert self.target_scoring is not None, 'Target not set.'
    #     sw_log = False

    #     abc='abbcdefghij'
    #     # fig, _ax = plt.subplots(ncols=2, nrows=4,
    #     #                         gridspec_kw={'height_ratios': [1.6,1,1,1]})

    #     fig, _ax = plt.subplots(ncols=2, nrows=3)
    #     gs = _ax[0, 0].get_gridspec()
    #     # remove the underlying axes
    #     for ax in _ax[0, :]:
    #         ax.remove()
    #     axbig = fig.add_subplot(gs[0, 0:])

    #     # top rows

    #     model_metrics = {}
    #     for model_name in self.model_names:
    #         # fn_metrics
    #         # ./metrics/CNN.metrics.lev-avg.csv
    #         metrics = []
    #         for metric_name in tqdm(self.metric_names):
    #             if metric_name == "MAE":
    #                 metrics.append(self.vert_avg_metric(self.calc_MAE(self.preds_scoring[model_name], self.target_scoring)))
    #             elif metric_name == "RMSE":
    #                 metrics.append(self.vert_avg_metric(self.calc_RMSE(self.preds_scoring[model_name], self.target_scoring)))
    #             elif metric_name == "R2":
    #                 metrics.append(self.vert_avg_metric(self.calc_R2(self.preds_scoring[model_name], self.target_scoring)))
    #             elif metric_name == "CRPS":
    #                 if model_name in self.crps_compatible:
    #                     metrics.append(self.vert_avg_metric(self.calc_CRPS(self.preds_scoring[model_name], self.target_scoring)))
    #         df_vert_avg = self.df_vert_avg_metric(metrics)
    #         model_metrics[model_name] = df_vert_avg

    #     plotting_dict = {}
    #     for metric_name in self.metric_names:
    #         if metric_name == 'CRPS':
    #             plot_index = self.crps_compatible
    #         else:
    #             plot_index = self.model_names
    #         plotting_dict[metric_name] = pd.DataFrame([model_metrics[model_name] for model_name in plot_index], index = plot_index)

    #     ax = axbig
    #     plotdata = plotting_dict['MAE']
    #     plotdata = plotdata.rename(columns=self.target_short_name)
    #     plotdata = plotdata.transpose()
    #     plotdata.plot.bar(color=[self.linecolors[model_name] for model_name in self.model_names],
    #                     legend = False,
    #                     ax=ax)
    #     ax.set_xticklabels(plotdata.index, rotation=0, ha='center')
    #     ax.set_xlabel('')
    #     # ax.set_title(f'({abc[k]}) {kmetric}')
    #     ax.set_ylabel('W/m2')

    #     ax.text(0.03, 0.93, f'(a) {kmetric}', horizontalalignment='left',
    #         verticalalignment='top', transform=ax.transAxes,
    #         fontweight='demi')

    #     if sw_log:
    #         ax.set_yscale('log')

    #     ax.legend(ncols=2,
    #             columnspacing=.8,
    #             labelspacing=.3,
    #             handleheight=.1,
    #             handlelength=1.5,
    #             handletextpad=.2,
    #             borderpad=.4,
    #             frameon=True,
    #             loc='upper right')

    #     # bottom rows

    #     rel_metrics = ['MAE', 'R2']

    #     model_metrics = {}
    #     for model_name in self.model_names:
    #         metrics = []
    #         for metric_name in tqdm(rel_metrics):
    #             if metric_name == "MAE":
    #                 metrics.append(self.calc_MAE(self.preds_scoring[model_name], self.target_scoring))
    #             elif metric_name == "RMSE":
    #                 metrics.append(self.calc_RMSE(self.preds_scoring[model_name], self.target_scoring))
    #             elif metric_name == "R2":
    #                 metrics.append(self.calc_R2(self.preds_scoring[model_name], self.target_scoring))
    #             elif metric_name == "CRPS":
    #                 if model_name in self.crps_compatible:
    #                     metrics.append(self.calc_CRPS(self.preds_scoring[model_name], self.target_scoring))
    #         df_stack = self.df_stack_metric(metrics)
    #         model_metrics[model_name] = df_stack

    #     plotting_dict = {}
    #     for metric_name in rel_metrics:
    #         plotting_dict[kmetric] = pd.DataFrame([plotting_dict[model_name][metric_name] for model_name in self.model_names], 
    #                                               index=self.model_names)

    #     for kk, kvar in enumerate(['ptend_t','ptend_q0001']):
    #         for k, kmetric in enumerate(rel_metrics):
    #             ax = _ax[k+1, 0 if kvar=='ptend_t' else 1]
    #             idx_start = self.var_idx[kvar][0]
    #             idx_end = self.var_idx[kvar][1]
    #             plotdata = plotting_dict[kmetric].iloc[:,idx_start:idx_end]
    #             if kvar == 'ptend_q0001':
    #                 plotdata.columns = plotdata.columns - 60
    #             if kvar=='ptend_q0001': # this is to keep the right x axis range.
    #                 plotdata = plotdata.where(~np.isinf(plotdata),-999)
    #             plotdata.transpose().plot(color=[self.linecolors[model_name] for model_name in self.model_names], 
    #                                       legend=False, 
    #                                       ax=ax)
    #             #ax.set_title(f'({abc[k]}) {kmetric}')
    #             if k==0:
    #                 ax.set_ylabel(f'W/m2')
    #                 ax.set_xlabel('')
    #                 ax.set_xticklabels('')
    #             elif k==1:
    #                 ax.set_xlabel('Level index')


    #             ax.text(0.03, 0.93, f'({abc[kk+2*k+2]}) {kmetric}, {self.target_short_name[kvar]}', horizontalalignment='left',
    #                 verticalalignment='top', transform=ax.transAxes,
    #                 fontweight='demi')

    #             if sw_log:
    #                 ax.set_yscale('log')

    #             # R2 ylim
    #             if  (kmetric=='R2'):
    #                 ax.set_ylim(0,1.05)

    #     fig.set_size_inches(9,5.25)
    #     return

    def plot_r2_analysis(self, pressure_grid, save_path = ''):
        '''
        This function plots the R2 pressure latitude figure shown in the SI.
        '''
        self.set_plot_params()
        n_model = len(self.model_names)
        fig, ax = plt.subplots(2,n_model, figsize=(n_model*12,18))
        y = np.array(range(60))
        X, Y = np.meshgrid(np.sin(self.lats*np.pi/180), y)
        Y = pressure_grid/100
        test_heat_daily_long, test_moist_daily_long = self.reshape_daily(self.target_scoring)
        for i, model_name in enumerate(self.model_names):
            pred_heat_daily_long, pred_moist_daily_long = self.reshape_daily(self.preds_scoring[model_name])
            coeff = 1 - np.sum( (pred_heat_daily_long-test_heat_daily_long)**2, axis=1)/np.sum( (test_heat_daily_long-np.mean(test_heat_daily_long, axis=1)[:,None,:])**2, axis=1)
            coeff = coeff[self.sort_lat_key,:]
            coeff = coeff.T
            
            contour_plot = ax[0,i].pcolor(X, Y, coeff,cmap='Blues', vmin = 0, vmax = 1) # pcolormesh
            ax[0,i].contour(X, Y, coeff, [0.7], colors='orange', linewidths=[4])
            ax[0,i].contour(X, Y, coeff, [0.9], colors='yellow', linewidths=[4])
            ax[0,i].set_ylim(ax[0,i].get_ylim()[::-1])
            ax[0,i].set_title(self.model_names[i] + " - Heating")
            ax[0,i].set_xticks([])
            
            coeff = 1 - np.sum( (pred_moist_daily_long-test_moist_daily_long)**2, axis=1)/np.sum( (test_moist_daily_long-np.mean(test_moist_daily_long, axis=1)[:,None,:])**2, axis=1)
            coeff = coeff[self.sort_lat_key,:]
            coeff = coeff.T
            
            contour_plot = ax[1,i].pcolor(X, Y, coeff,cmap='Blues', vmin = 0, vmax = 1) # pcolormesh
            ax[1,i].contour(X, Y, coeff, [0.7], colors='orange', linewidths=[4])
            ax[1,i].contour(X, Y, coeff, [0.9], colors='yellow', linewidths=[4])
            ax[1,i].set_ylim(ax[1,i].get_ylim()[::-1])
            ax[1,i].set_title(self.model_names[i] + " - Moistening")
            ax[1,i].xaxis.set_ticks([np.sin(-50/180*np.pi), 0, np.sin(50/180*np.pi)])
            ax[1,i].xaxis.set_ticklabels(['50$^\circ$S', '0$^\circ$', '50$^\circ$N'])
            ax[1,i].xaxis.set_tick_params(width = 2)
            
            if i != 0:
                ax[0,i].set_yticks([])
                ax[1,i].set_yticks([])
                
        # lines below for x and y label axes are valid if 3 models are considered
        # we want to put only one label for each axis
        # if nbr of models is different from 3 please adjust label location to center it

        #ax[1,1].xaxis.set_label_coords(-0.10,-0.10)

        ax[0,0].set_ylabel("Pressure [hPa]")
        ax[0,0].yaxis.set_label_coords(-0.2,-0.09) # (-1.38,-0.09)
        ax[0,0].yaxis.set_ticks([1000,800,600,400,200,0])
        ax[1,0].yaxis.set_ticks([1000,800,600,400,200,0])
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.12, 0.02, 0.76])
        cb = fig.colorbar(contour_plot, cax=cbar_ax)
        cb.set_label("Skill Score "+r'$\left(\mathrm{R^{2}}\right)$',labelpad=50.1)
        plt.suptitle("Baseline Models Skill for Vertically Resolved Tendencies", y = 0.97)
        plt.subplots_adjust(hspace=0.13)
        plt.show()
        plt.savefig(save_path + 'press_lat_diff_models.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    
    @staticmethod
    def reshape_input_for_cnn(npy_input, save_path = ''):
        '''
        This function reshapes a numpy input array to be compatible with CNN training.
        Each variable becomes its own channel.
        For the input there are 6 channels, each with 60 vertical levels.
        The last 4 channels correspond to scalars repeated across all 60 levels.
        This is for V1 data only! (V2 data has more variables)
        '''
        npy_input_cnn = np.stack([
            npy_input[:, 0:60],
            npy_input[:, 60:120],
            np.repeat(npy_input[:, 120][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_input[:, 121][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_input[:, 122][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_input[:, 123][:, np.newaxis], 60, axis = 1)], axis = 2)
        
        if save_path != '':
            with open(save_path + 'train_input_cnn.npy', 'wb') as f:
                np.save(f, np.float32(npy_input_cnn))
        return npy_input_cnn
    
    @staticmethod
    def reshape_target_for_cnn(npy_target, save_path = ''):
        '''
        This function reshapes a numpy target array to be compatible with CNN training.
        Each variable becomes its own channel.
        For the input there are 6 channels, each with 60 vertical levels.
        The last 4 channels correspond to scalars repeated across all 60 levels.
        This is for V1 data only! (V2 data has more variables)
        '''
        npy_target_cnn = np.stack([
            npy_target[:, 0:60],
            npy_target[:, 60:120],
            np.repeat(npy_target[:, 120][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 121][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 122][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 123][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 124][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 125][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 126][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 127][:, np.newaxis], 60, axis = 1)], axis = 2)
        
        if save_path != '':
            with open(save_path + 'train_target_cnn.npy', 'wb') as f:
                np.save(f, np.float32(npy_target_cnn))
        return npy_target_cnn
    
    @staticmethod
    def reshape_target_from_cnn(npy_predict_cnn, save_path = ''):
        '''
        This function reshapes CNN target to (num_samples, 128) for standardized metrics.
        This is for V1 data only! (V2 data has more variables)
        '''
        npy_predict_cnn_reshaped = np.concatenate([
            npy_predict_cnn[:,:,0],
            npy_predict_cnn[:,:,1],
            np.mean(npy_predict_cnn[:,:,2], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,3], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,4], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,5], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,6], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,7], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,8], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,9], axis = 1)[:, np.newaxis]], axis = 1)
        
        if save_path != '':
            with open(save_path + 'cnn_predict_reshaped.npy', 'wb') as f:
                np.save(f, np.float32(npy_predict_cnn_reshaped))
        return npy_predict_cnn_reshaped




  




