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
                 grid_info,
                 input_mean,
                 input_max,
                 input_min,
                 output_scale):
        self.data_path = None
        self.input_vars = []
        self.target_vars = []
        self.input_feature_len = None
        self.target_feature_len = None
        self.grid_info = grid_info
        self.level_name = 'lev'
        self.sample_name = 'sample'
        self.latlonnum = len(self.grid_info['ncol']) # number of unique lat/lon grid points
        # make area-weights
        self.grid_info['area_wgt'] = self.grid_info['area']/self.grid_info['area'].mean(dim = 'ncol')
        self.area_wgt = self.grid_info['area_wgt'].values
        # map ncol to nsamples dimension
        # to_xarray = {'area_wgt':(self.sample_name,np.tile(self.grid_info['area_wgt'], int(n_samples/len(self.grid_info['ncol']))))}
        # to_xarray = xr.Dataset(to_xarray)
        self.input_mean = input_mean
        self.input_max = input_max
        self.input_min = input_min
        self.output_scale = output_scale
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
        self.pressure_grid = None
        self.dp = None
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
        
        self.v1_inputs = ['state_t',
                          'state_q0001',
                          'state_ps',
                          'pbuf_SOLIN',
                          'pbuf_LHFLX',
                          'pbuf_SHFLX']
        
        self.v1_outputs = ['ptend_t',
                           'ptend_q0001',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD']

        self.var_lens = {#inputs
                         'state_t':60,
                         'state_q0001':60,
                         'state_ps':1,
                         'pbuf_SOLIN':1,
                         'pbuf_LHFLX':1,
                         'pbuf_SHFLX':1,
                         #outputs
                         'ptend_t':60,
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

        # for metrics
    
        self.input_train = None
        self.target_train = None
        self.preds_train = None
        self.samples_train = None
        self.target_weighted_train = {}
        self.preds_weighted_train = {}
        self.samples_weighted_train = {}
        self.metrics_train = []
        self.metrics_idx_train = {}
        self.metrics_var_train = {}

        self.input_val = None
        self.target_val = None
        self.preds_val = None
        self.samples_val = None
        self.target_weighted_val = {}
        self.preds_weighted_val = {}
        self.samples_weighted_val = {}
        self.metrics_val = []
        self.metrics_idx_val = {}
        self.metrics_var_val = {}
        
        self.input_scoring = None
        self.target_scoring = None
        self.preds_scoring = None
        self.samples_scoring = None
        self.target_weighted_scoring = {}
        self.preds_weighted_scoring = {}
        self.samples_weighted_scoring = {}
        self.metrics_scoring = []
        self.metrics_idx_scoring = {}
        self.metrics_var_scoring = {}

        self.input_test = None
        self.target_test = None
        self.preds_test = None
        self.samples_test = None
        self.target_weighted_test = {}
        self.preds_weighted_test = {}
        self.samples_weighted_test = {}
        self.metrics_test = []
        self.metrics_idx_test = {}
        self.metrics_var_test = {}

        self.model_names = []
        self.metrics_names = []
        self.metrics_dict = {'MAE': self.calc_MAE,
                             'RMSE': self.calc_RMSE,
                             'R2': self.calc_R2,
                             'CRPS': self.calc_CRPS
                            }
        self.linecolors = ['#0072B2', 
                           '#E69F00', 
                           '#882255', 
                           '#009E73', 
                           '#D55E00'
                           ]

    def set_to_v1_vars(self):
        '''
        This function sets the inputs and outputs to the V1 subset.
        '''
        self.input_vars = self.v1_inputs
        self.target_vars = self.v1_outputs
        self.input_feature_len = 124
        self.target_feature_len = 128

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
                ds_input = (ds_input - self.input_mean)/(self.input_max - self.input_min)
                ds_target = ds_target*self.output_scale

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
    def load_npy_file(load_path = ''):
        '''
        This function loads the prediction .npy file.
        '''
        with open(load_path, 'rb') as f:
            pred = np.load(f)
        return pred
    
    @staticmethod
    def load_h5_file(load_path = ''):
        '''
        This function loads the prediction .h5 file.
        '''
        hf = h5py.File(load_path, 'r')
        pred = np.array(hf.get('pred'))
        return pred
    
    def set_pressure_grid(self, input_arr):
        '''
        This function sets the pressure weighting for metrics.
        '''
        state_ps = input_arr[:,120]*(self.input_max['state_ps'].values - self.input_min['state_ps'].values) + self.input_mean['state_ps'].values
        state_ps = np.reshape(state_ps, (-1, self.latlonnum))
        pressure_grid_p1 = np.array(self.grid_info['P0']*self.grid_info['hyai'])[:,np.newaxis,np.newaxis]
        pressure_grid_p2 = self.grid_info['hybi'].values[:, np.newaxis, np.newaxis] * state_ps[np.newaxis, :, :]
        self.pressure_grid = pressure_grid_p1 + pressure_grid_p2
        self.dp = self.pressure_grid[1:61,:,:] - self.pressure_grid[0:60,:,:]
        self.dp = self.dp.transpose((1,2,0))
    
    def get_pressure_grid_plotting(self, data_split):
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
        pressure_grid_plotting = np.concatenate(pg_lats, axis = 1)
        return pressure_grid_plotting

    def output_weighting(self, output):
        '''
        This function does four transformations, and assumes we are using V1 variables:
        [0] Undos the output scaling
        [1] Weight vertical levels by dp/g
        [2] Weight horizontal area of each grid cell by a[x]/mean(a[x])
        [3] Unit conversion to a common energy unit
        '''
        num_samples = output.shape[0]
        heating = output[:,:60].reshape((int(num_samples/self.latlonnum), self.latlonnum, 60))
        moistening = output[:,60:120].reshape((int(num_samples/self.latlonnum), self.latlonnum, 60))
        netsw = output[:,120].reshape((int(num_samples/self.latlonnum), self.latlonnum))
        flwds = output[:,121].reshape((int(num_samples/self.latlonnum), self.latlonnum))
        precsc = output[:,122].reshape((int(num_samples/self.latlonnum), self.latlonnum))
        precc = output[:,123].reshape((int(num_samples/self.latlonnum), self.latlonnum))
        sols = output[:,124].reshape((int(num_samples/self.latlonnum), self.latlonnum))
        soll = output[:,125].reshape((int(num_samples/self.latlonnum), self.latlonnum))
        solsd = output[:,126].reshape((int(num_samples/self.latlonnum), self.latlonnum))
        solld = output[:,127].reshape((int(num_samples/self.latlonnum), self.latlonnum))
        
        # heating = heating.transpose((2,0,1))
        # moistening = moistening.transpose((2,0,1))
        # scalar_outputs = scalar_outputs.transpose((2,0,1))

        # [0] Undo output scaling
        heating = heating/self.output_scale['ptend_t'].values[np.newaxis, np.newaxis, :]
        moistening = moistening/self.output_scale['ptend_q0001'].values[np.newaxis, np.newaxis, :]
        netsw = netsw/self.output_scale['cam_out_NETSW'].values
        flwds = flwds/self.output_scale['cam_out_FLWDS'].values
        precsc = precsc/self.output_scale['cam_out_PRECSC'].values
        precc = precc/self.output_scale['cam_out_PRECC'].values
        sols = sols/self.output_scale['cam_out_SOLS'].values
        soll = soll/self.output_scale['cam_out_SOLL'].values
        solsd = solsd/self.output_scale['cam_out_SOLSD'].values
        solld = solld/self.output_scale['cam_out_SOLLD'].values

        # [1] Weight vertical levels by dp/g
        # only for vertically-resolved variables, e.g. ptend_{t,q0001}
        # dp/g = -\rho * dz
        heating = heating * self.dp/self.grav
        moistening = moistening * self.dp/self.grav

        # [2] weight by area
        heating = heating * self.area_wgt[np.newaxis, :, np.newaxis]
        moistening = moistening * self.area_wgt[np.newaxis, :, np.newaxis]
        netsw = netsw * self.area_wgt[np.newaxis, :]
        flwds = flwds * self.area_wgt[np.newaxis, :]
        precsc = precsc * self.area_wgt[np.newaxis, :]
        precc = precc * self.area_wgt[np.newaxis, :]
        sols = sols * self.area_wgt[np.newaxis, :]
        soll = soll * self.area_wgt[np.newaxis, :]
        solsd = solsd * self.area_wgt[np.newaxis, :]
        solld = solld * self.area_wgt[np.newaxis, :]

        # [3] unit conversion
        heating = heating * self.target_energy_conv['ptend_t']
        moistening = moistening * self.target_energy_conv['ptend_q0001']
        netsw = netsw * self.target_energy_conv['cam_out_NETSW']
        flwds = flwds * self.target_energy_conv['cam_out_FLWDS']
        precsc = precsc * self.target_energy_conv['cam_out_PRECSC']
        precc = precc * self.target_energy_conv['cam_out_PRECC']
        sols = sols * self.target_energy_conv['cam_out_SOLS']
        soll = soll * self.target_energy_conv['cam_out_SOLL']
        solsd = solsd * self.target_energy_conv['cam_out_SOLSD']
        solld = solld * self.target_energy_conv['cam_out_SOLLD']

        return {'ptend_t':heating,
                'ptend_q0001':moistening,
                'cam_out_NETSW':netsw,
                'cam_out_FLWDS':flwds,
                'cam_out_PRECSC':precsc,
                'cam_out_PRECC':precc,
                'cam_out_SOLS':sols,
                'cam_out_SOLL':soll,
                'cam_out_SOLSD':solsd,
                'cam_out_SOLLD':solld}
    
    def reweight_target(self, data_split):
        '''
        data_split should be train, val, scoring, or test
        weights target variables assuming V1 outputs using the output_weighting function
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            assert self.target_train is not None
            self.target_weighted_train = self.output_weighting(self.target_train)
        elif data_split == 'val':
            assert self.target_val is not None
            self.target_weighted_val = self.output_weighting(self.target_val)
        elif data_split == 'scoring':
            assert self.target_scoring is not None
            self.target_weighted_scoring = self.output_weighting(self.target_scoring)
        elif data_split == 'test':
            assert self.target_test is not None
            self.target_weighted_test = self.output_weighting(self.target_test)

    def reweight_preds(self, data_split):
        '''
        weights predictions assuming V1 outputs using the output_weighting function
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        assert self.model_names is not None

        if data_split == 'train':
            assert self.preds_train is not None
            for model_name in self.model_names:
                self.preds_weighted_train[model_name] = self.output_weighting(self.preds_train[model_name])
        elif data_split == 'val':
            assert self.preds_val is not None
            for model_name in self.model_names:
                self.preds_weighted_val[model_name] = self.output_weighting(self.preds_val[model_name])
        elif data_split == 'scoring':
            assert self.preds_scoring is not None
            for model_name in self.model_names:
                self.preds_weighted_scoring[model_name] = self.output_weighting(self.preds_scoring[model_name])
        elif data_split == 'test':
            assert self.preds_test is not None
            for model_name in self.model_names:
                self.preds_weighted_test[model_name] = self.output_weighting(self.preds_test[model_name])

    def calc_MAE(self, pred, target):
        '''
        calculate 'globally averaged' mean absolute error 
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.latlonnum
        assert pred.shape == target.shape
        mae = np.abs(pred - target).mean(axis = 0)
        return mae.mean(axis = 0) # we decided to average globally at end
    
    def calc_RMSE(self, pred, target):
        '''
        calculate 'globally averaged' root mean squared error 
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.latlonnum
        assert pred.shape == target.shape
        sq_diff = (pred - target)**2
        rmse = np.sqrt(sq_diff.mean(axis = 0)) # mean over time
        return rmse.mean(axis = 0) # we decided to separately average globally at end

    def calc_R2(self, pred, target):
        '''
        calculate 'globally averaged' R-squared
        for vertically-resolved variables, input shape should be time x grid x level
        for scalars, input shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.latlonnum
        assert pred.shape == target.shape
        sq_diff = (pred - target)**2
        tss_time = (target - target.mean(axis = 0)[np.newaxis, ...])**2 # mean over time
        r_squared = 1 - sq_diff.sum(axis = 0)/tss_time.sum(axis = 0) # sum over time
        return r_squared.mean(axis = 0) # we decided to separately average globally at end

    def calc_CRPS(self, preds, target):
        '''
        calculate 'globally averaged' continuous ranked probability score
        for vertically-resolved variables, input shape should be time x grid x level x num_crps_samples
        for scalars, input shape should be time x grid x num_crps_samples

        returns vector of length level or 1
        '''
        assert preds.shape[1] == self.latlonnum
        num_crps = preds.shape[-1]
        mae = np.mean(np.abs(preds - target[..., np.newaxis]), axis = (0, -1)) # mean over time and crps samples
        diff = preds[..., 1:] - preds[..., :-1]
        count = np.arange(1, num_crps) * np.arange(num_crps - 1, 0, -1)
        spread = (diff * count[np.newaxis, np.newaxis, np.newaxis, :]).mean(axis = (0, -1)) # mean over time and crps samples
        crps = mae - spread/(num_crps*(num_crps-1))
        # already divided by two in spread by exploiting symmetry
        return crps.mean(axis = 0) # we decided to separately average globally at end

    def create_metrics_df(self, data_split):
        '''
        creates a dataframe of metrics for each model
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], \
            'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        assert len(self.model_names) != 0
        assert len(self.metrics_names) != 0
        assert len(self.target_vars) != 0
        assert self.target_feature_len is not None

        if data_split == 'train':
            assert len(self.preds_weighted_train) != 0
            assert len(self.target_weighted_train) != 0
            for model_name in self.model_names:
                df_var = pd.DataFrame(columns = self.metrics_names, index = self.target_vars)
                df_var.index.name = 'variable'
                df_idx = pd.DataFrame(columns = self.metrics_names, index = range(self.target_feature_len))
                df_idx.index.name = 'output_idx'
                for metric_name in self.metrics_names:
                    current_idx = 0
                    for target_var in self.target_vars:
                        metric = self.metrics_dict[metric_name](self.preds_weighted_train[model_name][target_var], self.target_weighted_train[target_var])
                        df_var.loc[target_var, metric_name] = np.mean(metric)
                        df_idx.loc[current_idx:current_idx + self.var_lens[target_var] - 1, metric_name] = np.atleast_1d(metric)
                        current_idx += self.var_lens[target_var]
                self.metrics_var_train[model_name] = df_var
                self.metrics_idx_train[model_name] = df_idx

        elif data_split == 'val':
            assert len(self.preds_weighted_val) != 0
            assert len(self.target_weighted_val) != 0
            for model_name in self.model_names:
                df_var = pd.DataFrame(columns = self.metrics_names, index = self.target_vars)
                df_var.index.name = 'variable'
                df_idx = pd.DataFrame(columns = self.metrics_names, index = range(self.target_feature_len))
                df_idx.index.name = 'output_idx'
                for metric_name in self.metrics_names:
                    current_idx = 0
                    for target_var in self.target_vars:
                        metric = self.metrics_dict[metric_name](self.preds_weighted_val[model_name][target_var], self.target_weighted_val[target_var])
                        df_var.loc[target_var, metric_name] = np.mean(metric)
                        df_idx.loc[current_idx:current_idx + self.var_lens[target_var] - 1, metric_name] = np.atleast_1d(metric)
                        current_idx += self.var_lens[target_var]
                self.metrics_var_val[model_name] = df_var
                self.metrics_idx_val[model_name] = df_idx

        elif data_split == 'scoring':
            assert len(self.preds_weighted_scoring) != 0
            assert len(self.target_weighted_scoring) != 0
            for model_name in self.model_names:
                df_var = pd.DataFrame(columns = self.metrics_names, index = self.target_vars)
                df_var.index.name = 'variable'
                df_idx = pd.DataFrame(columns = self.metrics_names, index = range(self.target_feature_len))
                df_idx.index.name = 'output_idx'
                for metric_name in self.metrics_names:
                    current_idx = 0
                    for target_var in self.target_vars:
                        metric = self.metrics_dict[metric_name](self.preds_weighted_scoring[model_name][target_var], self.target_weighted_scoring[target_var])
                        df_var.loc[target_var, metric_name] = np.mean(metric)
                        df_idx.loc[current_idx:current_idx + self.var_lens[target_var] - 1, metric_name] = np.atleast_1d(metric)
                        current_idx += self.var_lens[target_var]
                self.metrics_var_scoring[model_name] = df_var
                self.metrics_idx_scoring[model_name] = df_idx

        elif data_split == 'test':
            assert len(self.preds_weighted_test) != 0
            assert len(self.target_weighted_test) != 0
            for model_name in self.model_names:
                df_var = pd.DataFrame(columns = self.metrics_names, index = self.target_vars)
                df_var.index.name = 'variable'
                df_idx = pd.DataFrame(columns = self.metrics_names, index = range(self.target_feature_len))
                df_idx.index.name = 'output_idx'
                for metric_name in self.metrics_names:
                    current_idx = 0
                    for target_var in self.target_vars:
                        metric = self.metrics_dict[metric_name](self.preds_weighted_test[model_name][target_var], self.target_weighted_test[target_var])
                        df_var.loc[target_var, metric_name] = np.mean(metric)
                        df_idx.loc[current_idx:current_idx + self.var_lens[target_var] - 1, metric_name] = np.atleast_1d(metric)
                        current_idx += self.var_lens[target_var]
                self.metrics_var_test[model_name] = df_var
                self.metrics_idx_test[model_name] = df_idx

    def reshape_daily(self, output):
        '''
        This function returns two numpy arrays, one for each vertically resolved variable (heating and moistening).
        Dimensions of expected input are num_samples by 128 (number of target features).
        Output argument is espected to be have dimensions of num_samples by features.
        Heating is expected to be the first feature, and moistening is expected to be the second feature.
        Data is expected to use a stride_sample of 6. (12 samples per day, 20 min timestep).
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

    def plot_r2_analysis(self, pressure_grid_plotting, save_path = ''):
        '''
        This function plots the R2 pressure latitude figure shown in the SI.
        '''
        self.set_plot_params()
        n_model = len(self.model_names)
        fig, ax = plt.subplots(2,n_model, figsize=(n_model*12,18))
        y = np.array(range(60))
        X, Y = np.meshgrid(np.sin(self.lats*np.pi/180), y)
        Y = pressure_grid_plotting/100
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




  




