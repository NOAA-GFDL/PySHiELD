import numpy as np
import xarray as xr


def set_sfc_arrays(data_dir: str):
    isl = xr.open_dataset(f"{data_dir}/isl.nc")
    islmsk = np.round(isl.values).astype(int)
    return islmsk
