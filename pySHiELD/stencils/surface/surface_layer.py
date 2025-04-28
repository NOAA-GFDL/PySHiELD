from gt4py.cartesian.gtscript import PARALLEL, computation, interval

import ndsl.constants as constants
from ndsl.constants import X_DIM, Y_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    Bool,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity
from pySHiELD._config import SurfaceConfig
from pySHiELD.functions.set_sfc_params import set_sfc_arrays
from pySHiELD.stencils.surface.sfc_diff import SurfaceExchange
from pySHiELD.stencils.surface.sfc_ocean import SurfaceOcean
from pySHiELD.stencils.surface.sfc_sice import SurfaceSeaIce
from pySHiELD.stencils.surface.sfc_state import SurfaceState


def init_step_vars(
    tsfc: FloatFieldIJ,
    phil: FloatField,
    tsurf: FloatFieldIJ,
    flag_guess: BoolFieldIJ,
    flag_iter: BoolFieldIJ,
    drain: FloatFieldIJ,
    ep1d: FloatFieldIJ,
    runof: FloatFieldIJ,
    hflx: FloatFieldIJ,
    evap: FloatFieldIJ,
    evbs: FloatFieldIJ,
    evcw: FloatFieldIJ,
    trans: FloatFieldIJ,
    sbsno: FloatFieldIJ,
    snowc: FloatFieldIJ,
    snohf: FloatFieldIJ,
    qss: FloatFieldIJ,
    gflx: FloatFieldIJ,
    zlvl: FloatFieldIJ,
    smcwlt2: FloatFieldIJ,
    smcref2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(-1, None):
        tsurf = tsfc
        flag_guess = False
        flag_iter = True
        drain = 0.0
        ep1d = 0.0
        runof = 0.0
        hflx = 0.0
        evap = 0.0
        evbs = 0.0
        evcw = 0.0
        trans = 0.0
        sbsno = 0.0
        snowc = 0.0
        snohf = 0.0
        qss = 0.0
        gflx = 0.0
        zlvl = phil * constants.RGRAV
        smcwlt2 = 0.0
        smcref2 = 0.0


def update_guess_0(
    wind: FloatFieldIJ,
    iteration: Int,
    flag_guess: BoolFieldIJ,
):
    with computation(PARALLEL), interval(0, 1):
        if (iteration == 0) and (wind < 2.0):
            flag_guess[0, 0] = True


def update_guess_1(
    wind: FloatFieldIJ,
    iteration: Int,
    flag_guess: BoolFieldIJ,
    flag_iter: BoolFieldIJ,
    islmsk: IntFieldIJ,
):
    from __externals__ import nsstm_coupling

    with computation(PARALLEL), interval(0, 1):
        flag_iter = False
        flag_guess = False

        if (iteration == 0) and (wind < 2.0):
            if (islmsk == 1) or ((islmsk == 0) and (nsstm_coupling > 0)):
                flag_iter = True


class SurfaceLayer:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: SurfaceConfig,
    ):
        grid_indexing = stencil_factory.grid_indexing

        islmsk = set_sfc_arrays(config.sfc_data)
        self._islmsk = quantity_factory.from_array(
            islmsk,
            [X_DIM, Y_DIM],
            units="None",
            dtype=Int,
        )

        def make_quantity_2d() -> Quantity:
            return quantity_factory.zeros(
                [X_DIM, Y_DIM],
                units="unknown",
                dtype=Float,
            )

        self._cdq = make_quantity_2d()

        self._flag_guess = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="None",
            dtype=Bool,
        )

        self._flag_iter = quantity_factory.ones(
            [X_DIM, Y_DIM],
            units="None",
            dtype=Bool,
        )

        self._exchange = SurfaceExchange(
            stencil_factory=stencil_factory,
            ivegsrc=config.ivegsrc,
            do_z0_hwrf15=config.do_z0_hwrf15,
            do_z0_hwrf17=config.do_z0_hwrf17,
            do_z0_hwrf17_hwonly=config.do_z0_hwrf17_hwonly,
            do_z0_moon=config.do_z0_moon,
            redrag=config.redrag,
            wind_th_hwrf=config.wind_th_hwrf
        )
        self._update_guess_0 = stencil_factory.from_origin_domain(
            update_guess_0,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._sfc_ocean = SurfaceOcean(
            stencil_factory=stencil_factory,
        )
        self._sfc_sice = SurfaceSeaIce(
            stencil_factory=stencil_factory,
            mom4ice=config.mom4ice,
            lsm=config.lsm,
            dt_atmos=config.dt_atmos,
        )
        self._update_guess_1 = stencil_factory.from_origin_domain(
            update_guess_1,
            externals={"nsstm_coupling": config.nstf_name[0]},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(self, state: SurfaceState):
        for iteration in range(2):
            self._exchange()

            self._update_guess_0(state.wind, iteration, self._flag_guess)

            self._sfc_ocean()

            #  TODO: LSM here

            self._sfc_sice()

            self._update_guess_1(
                state.wind,
                iteration,
                self._flag_guess,
                self._flag_iter,
                self._islmsk,
            )
