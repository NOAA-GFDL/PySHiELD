from pathlib import Path
from pyshield.stencils.surface import SurfaceLayer, SurfaceState
from pyshield import PhysicsConfig

import numpy as np
import xarray as xr

def states_from_fortran_restarts(
    dycore_datafile: Path,
    phys_datafile: Path,
    tracer_datafile: Path,
    sfc_datafile: Path
    ak: Quantity,
    quantity_factory: QuantityFactory,
    schemes: PHYSICS_PACKAGES,
):
    dycore_data = xr.open_dataset(dycore_datafile)
    phys_data = xr.open_dataset(phys_datafile)
    tracer_data = xr.open_dataset(tracer_datafile)
    state = PhysicsState.init_zeros(quantity_factory, schemes)
    sstate = SurfaceState.init_zeros(quantity_factory)

def test_sfc_runs(restart_path):
    config = PhysicsConfig()
    state, sstate = states_from_fortran_restarts()
    sfc = SurfaceLayer()
    sfc()
