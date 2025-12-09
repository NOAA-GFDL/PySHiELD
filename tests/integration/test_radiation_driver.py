import datetime
from pathlib import Path

import numpy as np
import pytest

from examples.notebook.utilities import (
    fortran_restart_to_radstate,
    setup_infrastructure,
    states_from_fortran_restarts,
)
from pyshield._config import PhysicsConfig
from pyshield.physics_state import SurfaceState
from pyshield.radiation import RTE_RRTMGPConfig, RTE_RRTMGPDriver, RTE_RRTMGPState
from pyshield.stencils.physics import Physics, calc_sigma


@pytest.mark.parametrize("datapath", [Path("test_data/RESTART/")])
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
    radconf = RTE_RRTMGPConfig(
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

    state = RTE_RRTMGPState.init_zeros(quantity_factory, np)
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


@pytest.mark.parametrize("datapath", [Path("test_data/radtest/")])
def test_rte_rrtmgp_in_physics(datapath: Path):
    dycore_data = Path(datapath.joinpath("fv_core.res.tile1.nc"))
    phys_data = Path(datapath.joinpath("phy_data.tile1.nc"))
    tracer_data = Path(datapath.joinpath("fv_tracer.res.tile1.nc"))
    etafile = Path(datapath.joinpath("eta91.nc"))

    nx = 48
    ny = 48
    nz = 91
    npx = nx + 1
    npy = ny + 1
    npz = nz + 1

    date = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc)
    quantity_factory, stencil_factory, grid_data = setup_infrastructure(
        nx, ny, nz, etafile
    )
    grid_data.lon_agrid.field[:] = grid_data.lon.field[:-1, :-1]
    grid_data.lat_agrid.field[:] = grid_data.lat.field[:-1, :-1]
    conf = PhysicsConfig(
        dt_atmos=225.0,
        fhswr=225.0,
        fhlwr=225.0,
        npx=npx,
        npy=npy,
        npz=npz,
        nwat=6,
        prescribe_sst=False,
        schemes=["GFS_microphysics", "RTE_RRTMGP"],
        hydro_delp=False,
    )
    radconf = RTE_RRTMGPConfig(
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

    state, radstate, sstate = states_from_fortran_restarts(
        dycore_data,
        phys_data,
        tracer_data,
        grid_data.ak,
        quantity_factory,
        conf.schemes,
    )

    physics = Physics(
        stencil_factory,
        quantity_factory,
        grid_data,
        conf,
        radconf,
    )
    physics(state, radstate, sstate, date, conf.dt_atmos)
