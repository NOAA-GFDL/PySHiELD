import numpy as np
import xarray as xr


def set_sfc_arrays(data_dir: str):
    isl = xr.open_dataset(data_dir)
    islmsk = np.round(isl.values).astype(int)
    return islmsk
