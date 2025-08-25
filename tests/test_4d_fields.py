from ndsl.dsl.gt4py import (
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    sqrt,
)

import ndsl.constants as constants
from ndsl.constants import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM, Z_INTERFACE_DIM
import pyshield.constants as physcons

# from pace.dsl.dace.orchestration import orchestrate
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
from ndsl.dsl.typing import (
    Bool,
    BoolField,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntField,
    IntFieldIJ,
)
from pyshield._config import TRACER_DIM, FloatFieldTracer


def setup_infrastructure(nx: Int, ny: Int, nz: Int):
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
    return quantity_factory, stencil_factory


def sample_4d_stencil(
    q_in: FloatFieldTracer,
    q_out: FloatField,
):
    from __externals__ import ntke
    with computation(PARALLEL), interval(...):
        q_out = max(q_in[0, 0, 0][ntke], physcons.TKMIN)

class SampleCalculation:
    def __init__(
        self,
        stencil_factory: StencilFactory,
    ):
        self._test_calc = stencil_factory.from_dims_halo(
            func=sample_4d_stencil,
            externals={
                "ntke": 8,
            },
            compute_dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM],
        )
    
    def __call__(
        self,
        q_in: FloatFieldTracer,
        q_out: FloatField,
    ):
        self._test_calc(q_in, q_out)

def test_4d_stencil_call():
    ntracers = 9
    quantity_factory, stencil_factory = setup_infrastructure(24, 24, 91)
    quantity_factory.set_extra_dim_lengths(
        **{
            TRACER_DIM: ntracers,
        }
    )
    q_out = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_INTERFACE_DIM],
        units="unknown",
        dtype=Float,
    )
    q_in = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, TRACER_DIM],
            units="unknown",
            dtype=Float,
        )
    calc = SampleCalculation(stencil_factory)
    calc(q_in, q_out)
