import datetime
from pathlib import Path

import numpy as np
import pytest

from examples.notebook.utilities import (
    fortran_restart_to_radstate,
    setup_infrastructure,
)
from pyshield.physics_state import SurfaceState
from pyshield.radiation import RadiationConfig, RadiationState, RTE_RRTMGPDriver
from pyshield.stencils.physics import calc_sigma


@pytest.mark.parametrize("datapath", [Path("test_data/radtest/")])
def test_rte_rrtmgp(datapath: Path):
    dycore_data = Path(datapath.joinpath("fv_core.res.tile1.nc"))
    phys_data = Path(datapath.joinpath("phy_data.tile1.nc"))
    tracer_data = Path(datapath.joinpath("fv_tracer.res.tile1.nc"))
    etafile = Path(datapath.joinpath("eta91.nc"))

    nx = 48
    ny = 48
    nz = 91

    date = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc)
    quantity_factory, stencil_factory, grid_data = setup_infrastructure(
        nx, ny, nz, etafile
    )
    grid_data.lon_agrid.field[:] = grid_data.lon.field[:-1, :-1]
    grid_data.lat_agrid.field[:] = grid_data.lat.field[:-1, :-1]
    radconf = RadiationConfig(
        deltsw=3600.0,
        delt_rad=3600.0,
        date=date,
        fhswr=1.0,
        fhlwr=1.0,
        isolar=10,
        icmphys=4,
        ico2flg=0,
        ioznflg=1,
        ictmflg=-1,
        ialbflg=-1,
        iemsflg=0,
        ldisable_radiation_quasi_sea_ice=False,
        solar_constant_file=Path("global_solarconstant_noaa_an.txt"),
        input_dir=Path(datapath.joinpath("test_data/")),
        aerosol_file=Path(datapath.joinpath("test_data/")),
        sollat=0.0,
        nstp=6,
        ivflip=1,
        lcnorm=False,
        lcrick=False,
        gfs_cloud_overlap=False,
    )
    sigma = calc_sigma(grid_data.ak.data, grid_data.bk.data, 0)
    gridlon = grid_data.lon_agrid
    gridlat = grid_data.lat_agrid

    state = RadiationState.init_zeros(quantity_factory, np)
    sstate = SurfaceState.init_zeros(quantity_factory)
    fortran_restart_to_radstate(
        dycore_datafile=dycore_data,
        phys_datafile=phys_data,
        tracer_datafile=tracer_data,
        ak=grid_data.ak,
        state=state,
    )

    rad = RTE_RRTMGPDriver(
        config=radconf,
        gridlon=gridlon,
        gridlat=gridlat,
        sigma=sigma,
        quantity_factory=quantity_factory,
        stencil_factory=stencil_factory,
    )
    rad.step_radiation(state, sstate, date)
