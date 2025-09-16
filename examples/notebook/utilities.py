from pathlib import Path

import numpy as np
import xarray as xr

from ndsl import (
    CompilationConfig,
    GridIndexing,
    NullComm,
    Quantity,
    QuantityFactory,
    StencilConfig,
    StencilFactory,
    SubtileGridSizer,
    TileCommunicator,
)
from ndsl.dsl.typing import Int
from ndsl.grid import (
    AngleGridData,
    ContravariantGridData,
    GridData,
    HorizontalGridData,
    MetricTerms,
    VerticalGridData,
)
from pyshield.physics_state import PHYSICS_PACKAGES, PhysicsState, SurfaceState
from pyshield.radiation import RTE_RRTMGPState


def states_from_fortran_restarts(
    dycore_datafile: Path,
    phys_datafile: Path,
    tracer_datafile: Path,
    ak: Quantity,
    quantity_factory: QuantityFactory,
    schemes: PHYSICS_PACKAGES,
):
    dycore_data = xr.open_dataset(dycore_datafile)
    phys_data = xr.open_dataset(phys_datafile)
    tracer_data = xr.open_dataset(tracer_datafile)

    state = PhysicsState.init_zeros(quantity_factory, schemes)
    radstate = RTE_RRTMGPState.init_zeros(quantity_factory, np)
    sstate = SurfaceState.init_zeros(quantity_factory)

    buff_3d = np.zeros_like(state.prsi.field)
    for k in range(buff_3d.shape[2]):
        if k == 0:
            buff_3d[:, :, k] = ak.data[0]
        else:
            buff_3d[:, :, k] = (
                buff_3d[:, :, k - 1] + dycore_data.delp.data[0, k - 1, :, :]
            )
    state.prsi.field[:] = buff_3d[:, :, :]

    state.delp.field[:] = dycore_data.delp.data[0, :, :, :].transpose(2, 1, 0)
    state.pt.field[:] = dycore_data.T.data[0, :, :, :].transpose(2, 1, 0)
    state.tsfc.field[:] = phys_data.ts_clim_iano.data[0, :, :].transpose()
    state.qvapor.field[:] = tracer_data.sphum.data[0, :, :, :].transpose(2, 1, 0)
    state.qliquid.view[:] = tracer_data.liq_wat.data[0, :, :, :].transpose(2, 1, 0)
    state.qice.view[:] = tracer_data.ice_wat.data[0, :, :, :].transpose(2, 1, 0)
    state.qcld.view[:] = tracer_data.cld_amt.data[0, :, :, :].transpose(2, 1, 0)
    state.qo3mr.view[:] = tracer_data.o3mr.data[0, :, :, :].transpose(2, 1, 0)
    state.delz.field[:] = dycore_data.DZ.data[0, :, :, :].transpose(2, 1, 0)
    return state, radstate, sstate


def fortran_restart_to_radstate(
    dycore_datafile: Path,
    phys_datafile: Path,
    tracer_datafile: Path,
    ak: Quantity,
    state: RTE_RRTMGPState,
):
    dycore_data = xr.open_dataset(dycore_datafile)
    phys_data = xr.open_dataset(phys_datafile)
    tracer_data = xr.open_dataset(tracer_datafile)
    buff_3d = np.zeros_like(state.prsi.field)
    for k in range(buff_3d.shape[2]):
        if k == 0:
            buff_3d[:, :, k] = ak.data[0]
        else:
            buff_3d[:, :, k] = (
                buff_3d[:, :, k - 1] + dycore_data.delp.data[0, k - 1, :, :]
            )
    state.prsi.field[:] = buff_3d[:, :, ::-1]
    state.prsl.field[:] = (
        state.prsi.field[:, :, 1:] - state.prsi.field[:, :, :-1]
    ) / np.log(state.prsi.field[:, :, 1:] / state.prsi.field[:, :, :-1])
    state.tlyr.field[:] = dycore_data.T.data[0, ::-1, :, :].transpose(1, 2, 0)
    state.tlvl.field[:, :, 1:-1] = state.tlyr.field[:, :, :-1] + (
        state.tlyr.field[:, :, 1:] - state.tlyr.field[:, :, :-1]
    ) * (np.log(state.prsi.field[:, :, 1:-1]) - np.log(state.prsl.field[:, :, :-1])) / (
        np.log(state.prsl.field[:, :, 1:]) - np.log(state.prsl.field[:, :, :-1])
    )
    state.tlvl.field[:, :, -1] = state.tlyr.field[:, :, -1]
    state.tlvl.field[:, :, 0] = state.tlyr.field[:, :, 0]
    state.tsfc.field[:] = phys_data.ts_clim_iano.data[0, :, :]
    state.qvapor.field[:] = tracer_data.sphum.data[0, ::-1, :, :].transpose(1, 2, 0)
    state.qliquid.view[:] = tracer_data.liq_wat.data[0, ::-1, :, :].transpose(1, 2, 0)
    state.qice.view[:] = tracer_data.ice_wat.data[0, ::-1, :, :].transpose(1, 2, 0)
    state.qcld.view[:] = tracer_data.cld_amt.data[0, ::-1, :, :].transpose(1, 2, 0)
    state.qo3mr.view[:] = tracer_data.o3mr.data[0, ::-1, :, :].transpose(1, 2, 0)


def setup_infrastructure(nx: Int, ny: Int, nz: Int, etafile: Path):
    n_halo = 3

    rank = 0

    comm = NullComm(rank, 1)
    communicator = TileCommunicator.from_layout(comm=comm, layout=(1, 1))

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx,
        ny_tile=ny,
        nz=nz,
        n_halo=n_halo,
        extra_dim_lengths={},
        layout=(1, 1),
        tile_partitioner=communicator.partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    quantity_factory = QuantityFactory.from_backend(sizer, backend="numpy")

    comconf = CompilationConfig()
    comconf.validate_args = False
    sconf = StencilConfig(compilation_config=comconf)
    grid_indexing = GridIndexing.from_sizer_and_communicator(
        sizer=sizer, comm=communicator
    )
    stencil_factory = StencilFactory(
        config=sconf, grid_indexing=grid_indexing, comm=comm
    )

    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        grid_type=4,
        eta_file=etafile,
    )
    horizontal_data = HorizontalGridData.new_from_metric_terms(metric_terms)
    vertical_data = VerticalGridData.new_from_metric_terms(metric_terms)
    contravariant_data = ContravariantGridData.new_from_metric_terms(metric_terms)
    angle_data = AngleGridData.new_from_metric_terms(metric_terms)
    grid_data = GridData(
        horizontal_data=horizontal_data,
        vertical_data=vertical_data,
        contravariant_data=contravariant_data,
        angle_data=angle_data,
    )
    return quantity_factory, stencil_factory, grid_data
