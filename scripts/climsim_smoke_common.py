from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

NUM_LEVELS = 60

V4_INPUTS = [
    "state_t",
    "state_rh",
    "state_q0002",
    "state_q0003",
    "state_u",
    "state_v",
    "state_t_dyn",
    "state_q0_dyn",
    "state_u_dyn",
    "tm_state_t_dyn",
    "tm_state_q0_dyn",
    "tm_state_u_dyn",
    "state_t_prvphy",
    "state_q0001_prvphy",
    "state_q0002_prvphy",
    "state_q0003_prvphy",
    "state_u_prvphy",
    "tm_state_t_prvphy",
    "tm_state_q0001_prvphy",
    "tm_state_q0002_prvphy",
    "tm_state_q0003_prvphy",
    "tm_state_u_prvphy",
    "pbuf_ozone",
    "pbuf_CH4",
    "pbuf_N2O",
    "state_ps",
    "pbuf_SOLIN",
    "pbuf_LHFLX",
    "pbuf_SHFLX",
    "pbuf_TAUX",
    "pbuf_TAUY",
    "pbuf_COSZRS",
    "cam_in_ALDIF",
    "cam_in_ALDIR",
    "cam_in_ASDIF",
    "cam_in_ASDIR",
    "cam_in_LWUP",
    "cam_in_ICEFRAC",
    "cam_in_LANDFRAC",
    "cam_in_OCNFRAC",
    "cam_in_SNOWHICE",
    "cam_in_SNOWHLAND",
    "tm_state_ps",
    "tm_pbuf_SOLIN",
    "tm_pbuf_LHFLX",
    "tm_pbuf_SHFLX",
    "tm_pbuf_COSZRS",
    "clat",
    "slat",
    "icol",
]

V4_OUTPUTS = [
    "ptend_t",
    "ptend_q0001",
    "ptend_q0002",
    "ptend_q0003",
    "ptend_u",
    "ptend_v",
    "cam_out_NETSW",
    "cam_out_FLWDS",
    "cam_out_PRECSC",
    "cam_out_PRECC",
    "cam_out_SOLS",
    "cam_out_SOLL",
    "cam_out_SOLSD",
    "cam_out_SOLLD",
]

PROFILE_VARS = {
    "state_t",
    "state_rh",
    "state_q0001",
    "state_q0002",
    "state_q0003",
    "state_u",
    "state_v",
    "state_t_dyn",
    "state_q0_dyn",
    "state_u_dyn",
    "state_v_dyn",
    "state_t_prvphy",
    "state_q0001_prvphy",
    "state_q0002_prvphy",
    "state_q0003_prvphy",
    "state_u_prvphy",
    "tm_state_t_dyn",
    "tm_state_q0_dyn",
    "tm_state_u_dyn",
    "tm_state_t_prvphy",
    "tm_state_q0001_prvphy",
    "tm_state_q0002_prvphy",
    "tm_state_q0003_prvphy",
    "tm_state_u_prvphy",
    "pbuf_ozone",
    "pbuf_CH4",
    "pbuf_N2O",
    "ptend_t",
    "ptend_q0001",
    "ptend_q0002",
    "ptend_q0003",
    "ptend_u",
    "ptend_v",
}


def var_len(name: str) -> int:
    return NUM_LEVELS if name in PROFILE_VARS else 1


def load_v4_norms(repo_root: Path):
    norm_path = repo_root / "preprocessing/normalizations"
    input_mean = xr.open_dataset(norm_path / "inputs/input_mean_v4_pervar.nc")
    input_max = xr.open_dataset(norm_path / "inputs/input_max_v4_pervar.nc")
    input_min = xr.open_dataset(norm_path / "inputs/input_min_v4_pervar.nc")
    output_scale_path = norm_path / "outputs/output_scale_std_lowerthred.nc"
    if not output_scale_path.exists():
        output_scale_path = norm_path / "outputs/output_scale.nc"
    output_scale = xr.open_dataset(output_scale_path)

    input_sub = []
    input_div = []
    for var in V4_INPUTS:
        if var_len(var) == 1:
            input_sub.append(float(input_mean[var].values))
            input_div.append(float(input_max[var].values - input_min[var].values))
        else:
            input_sub.extend(np.asarray(input_mean[var].values).tolist())
            input_div.extend((np.asarray(input_max[var].values) - np.asarray(input_min[var].values)).tolist())

    out_scale = []
    for var in V4_OUTPUTS:
        if var_len(var) == 1:
            out_scale.append(float(output_scale[var].values))
        else:
            out_scale.extend(np.asarray(output_scale[var].values).tolist())

    qc_lbd = np.loadtxt(norm_path / "inputs/qc_exp_lambda_large.txt", delimiter=",")
    qi_lbd = np.loadtxt(norm_path / "inputs/qi_exp_lambda_large.txt", delimiter=",")

    return (
        np.asarray(input_sub, dtype=np.float32),
        np.asarray(input_div, dtype=np.float32),
        np.asarray(out_scale, dtype=np.float32),
        qc_lbd.astype(np.float32),
        qi_lbd.astype(np.float32),
    )
