import os
import glob
import xarray as xr
import numpy as np

def process_one_file(args):
    """
    Process a single NetCDF file by updating its dataset with information from previous files.
    
    Args:
        i: int
            The index of the current file in the full file list.
        nc_files_in: list of str
            List of the full filenames.
        lat: xarray.DataArray
            DataArray of latitude.
        lon: xarray.DataArray
            DataArray of longitude.
        input_abbrev: str
            The input file name abbreviation, the default input data should be 'mli'.
        output_abbrev: str
            The output file name abbreviation, the default output data should be 'mlo'.
        input_abbrev_new: str
            The abbreviation for the new input file name.
    
    Returns:
        None
    """
    i, nc_files_in, lat, lon, input_abbrev, output_abbrev, input_abbrev_new = args
    dsin = xr.open_dataset(nc_files_in[i])
    dsin_prev = xr.open_dataset(nc_files_in[i-1])
    dsin_prev2 = xr.open_dataset(nc_files_in[i-2])
    dsout_prev = xr.open_dataset(nc_files_in[i-1].replace(input_abbrev, output_abbrev))
    dsout_prev2 = xr.open_dataset(nc_files_in[i-2].replace(input_abbrev, output_abbrev))
    dsin['tm_state_t'] = dsin_prev['state_t']
    dsin['tm_state_q0001'] = dsin_prev['state_q0001']
    dsin['tm_state_q0002'] = dsin_prev['state_q0002']
    dsin['tm_state_q0003'] = dsin_prev['state_q0003']
    dsin['tm_state_u'] = dsin_prev['state_u']
    dsin['tm_state_v'] = dsin_prev['state_v']

    dsin['state_t_prvphy'] = (dsout_prev['state_t'] - dsin_prev['state_t'])/1200.
    dsin['state_q0001_prvphy'] = (dsout_prev['state_q0001'] - dsin_prev['state_q0001'])/1200.
    dsin['state_q0002_prvphy'] = (dsout_prev['state_q0002'] - dsin_prev['state_q0002'])/1200.
    dsin['state_q0003_prvphy'] = (dsout_prev['state_q0003'] - dsin_prev['state_q0003'])/1200.
    dsin['state_u_prvphy'] = (dsout_prev['state_u'] - dsin_prev['state_u'])/1200.

    dsin['tm_state_t_prvphy'] = (dsout_prev2['state_t'] - dsin_prev2['state_t'])/1200.
    dsin['tm_state_q0001_prvphy'] = (dsout_prev2['state_q0001'] - dsin_prev2['state_q0001'])/1200.
    dsin['tm_state_q0002_prvphy'] = (dsout_prev2['state_q0002'] - dsin_prev2['state_q0002'])/1200.
    dsin['tm_state_q0003_prvphy'] = (dsout_prev2['state_q0003'] - dsin_prev2['state_q0003'])/1200.
    dsin['tm_state_u_prvphy'] = (dsout_prev2['state_u'] - dsin_prev2['state_u'])/1200.

    dsin['state_t_dyn'] = (dsin['state_t'] - dsout_prev['state_t'])/1200.
    dsin['state_q0_dyn'] = (dsin['state_q0001'] - dsout_prev['state_q0001'] + dsin['state_q0002'] - dsout_prev['state_q0002'] + dsin['state_q0003'] - dsout_prev['state_q0003'])/1200.
    dsin['state_u_dyn'] = (dsin['state_u'] - dsout_prev['state_u'])/1200.

    dsin['tm_state_t_dyn'] = (dsin_prev['state_t'] - dsout_prev2['state_t'])/1200.
    dsin['tm_state_q0_dyn'] = (dsin_prev['state_q0001'] - dsout_prev2['state_q0001'] + dsin_prev['state_q0002'] - dsout_prev2['state_q0002'] + dsin_prev['state_q0003'] - dsout_prev2['state_q0003'])/1200.
    dsin['tm_state_u_dyn'] = (dsin_prev['state_u'] - dsout_prev2['state_u'])/1200.

    dsin['tm_state_ps'] = dsin_prev['state_ps']
    dsin['tm_pbuf_SOLIN'] = dsin_prev['pbuf_SOLIN']
    dsin['tm_pbuf_SHFLX'] = dsin_prev['pbuf_SHFLX']
    dsin['tm_pbuf_LHFLX'] = dsin_prev['pbuf_LHFLX']
    dsin['tm_pbuf_COSZRS'] = dsin_prev['pbuf_COSZRS']

    dsin['lat'] = lat
    dsin['lon'] = lon
    clat = lat.copy()
    slat = lat.copy()
    icol = lat.copy()
    clat[:] = np.cos(lat*2.*np.pi/360.)
    slat[:] = np.sin(lat*2.*np.pi/360.)
    icol[:] = np.arange(1,385)
    dsin['clat'] = clat
    dsin['slat'] = slat
    dsin['icol'] = icol

    new_file_path = nc_files_in[i].replace(input_abbrev, input_abbrev_new)
    dsin.to_netcdf(new_file_path)

    return None