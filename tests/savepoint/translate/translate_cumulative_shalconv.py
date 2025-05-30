from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
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
from ndsl.initialization.sizer import SubtileGridSizer
from gt4py.cartesian.gtscript import FORWARD, computation, interval
from pySHiELD.stencils.shallow_convection.samfshalconv import (
    stencil_static1, stencil_static0
    stencil_update_kbcon1_cnvflg,
    stencil_static9, stencil_static12,
    feedback_control_update_mass_flux,
    comp_tendencies, stencil_static13,
    stencil_ntrstatic0, init_col_arr,
    init_final, init_kbm_kmax, init_par_and_arr,
    init_tracers, pa_to_cb
)
from pySHiELD._config import ShallowConvectionConfig
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py
import numpy as np

def set_pfld_kbcon(
    kbcon: IntFieldIJ,
    k_mask: IntField,
    pfld: FloatField,
    pfld_kbcon: FloatFieldIJ
):
    with computation(FORWARD), interval(...):
        if k_mask == kbcon:
            pfld_kbcon = pfld

class InitCols:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: ShallowConvectionConfig,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._ntk = config.ntke
        self._ntiw = config.ntiw
        self._ntcw = config.ntcw
        self._ntr = config.nsamftrac
        self._ncloud = config.ncld
        self._dt2 = config.dt_atmos

        # Determine whether to perform aerosol transport #
        self._do_aerosols = (config.itc > 0) and (config.ntchm > 0) and (self._ntr > 0)
        if self._do_aerosols:
            self._do_aerosols = self._ntr >= config.itc
        if self._do_aerosols:
            raise NotImplementedError(
                "Shallow convection of aerosols is not implemented yet"
            )

        self._clam = config.clam_shal
        self._c0s = config.c0s_shal
        self._c1 = config.c1_shal
        self._pgcon = config.pgcon_shal
        self._asolfac = config.asolfac_shal

        self._km = grid_indexing.domain[2]
        self._km1 = grid_indexing.domain[2] - 1
        self.TRACER_DIM = TRACER_DIM

        self.quantity_factory = quantity_factory
        self.quantity_factory.set_extra_dim_lengths(
            **{
                self.TRACER_DIM: int(self._ntr + 2),
            }
        )

        # Configure stencils
        self._pa_to_cb = stencil_factory.from_origin_domain(
            func=pa_to_cb,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_col_arr = stencil_factory.from_origin_domain(
            func=init_col_arr,
            externals={"km": self._km},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
    
    def __call__(
        self,
        delp,
        prslp,
        psp,
        phil,
        qtr,
        q1,
        t1,
        rn,
        kbot,
        ktop,
        kcnv,
        islimsk,
        dot,
        hpbl,
        ud_mf,
        dt_mf,
        u1,
        v1,
        garea,
        cnvw,
        cnvc,
        cnvflg,
        kbcon,
        kb,
        ktcon,
        ktconn,
        pdot,
        qlko_ktcon,
        edt,
        aa1,
        cina,
        vshear,
        gdx,
        ps,
        prsl,
        del0,
    ):
    # Convert input Pa terms to Cb terms
        self._pa_to_cb(
            psp,
            prslp,
            delp,
            ps,
            prsl,
            del0,
        )

        self._init_col_arr(
            kcnv,
            cnvflg,
            kbot,
            ktop,
            kbcon,
            kb,
            ktcon,
            ktconn,
            pdot,
            rn,
            qlko_ktcon,
            edt,
            aa1,
            cina,
            vshear,
            gdx,
            garea,
        )

class Static1:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: ShallowConvectionConfig,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._ntk = config.ntke
        self._ntiw = config.ntiw
        self._ntcw = config.ntcw
        self._ntr = config.nsamftrac
        self._ncloud = config.ncld
        self._dt2 = config.dt_atmos

        # Determine whether to perform aerosol transport #
        self._do_aerosols = (config.itc > 0) and (config.ntchm > 0) and (self._ntr > 0)
        if self._do_aerosols:
            self._do_aerosols = self._ntr >= config.itc
        if self._do_aerosols:
            raise NotImplementedError(
                "Shallow convection of aerosols is not implemented yet"
            )

        self._clam = config.clam_shal
        self._c0s = config.c0s_shal
        self._c1 = config.c1_shal
        self._pgcon = config.pgcon_shal
        self._asolfac = config.asolfac_shal

        self._km = grid_indexing.domain[2]
        self._km1 = grid_indexing.domain[2] - 1
        self.TRACER_DIM = TRACER_DIM

        self.quantity_factory = quantity_factory
        self.quantity_factory.set_extra_dim_lengths(
            **{
                self.TRACER_DIM: int(self._ntr + 2),
            }
        )

        # Tracers are kind of borked right now. In Fortran the water vapor is passed in
        # separately, while all others come in via the variable "qtr" which has
        # dimensions (i, j, k, n_tracers - 1), ice and liquid water are stored in
        # qtr[:, :, :, 0] and qtr[:, :, :, 1] and everything is reorganized to
        # accomodate that. Aerosols live in qtr[:, :, :, itc:].
        # If we're being straightforward about it we'd have our 4D tracer array with
        # special handling for ntvapor, ntiw, and ntcw (like we do for TKE), have an
        # `if n not in [ntvap, ntiw, ntcw]:` around the other tracer calls, and then
        # do something similar with the aerosols.
        # A better solution would be to have the attributes we want accessible easily
        # so we can send the aerosols into the aerosol calculations by attribute, and
        # similarly except (or invoke) vapor etc. from the other calculations.

        def make_quantity():
            return quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )

        def make_quantity_2D(type=Float):
            return quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=type)

        # Allocate arrays

        # Layer mask:
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k

        self._heo_kb = make_quantity_2D()
        self._drag = make_quantity()
        self._ps = make_quantity_2D()
        self._prsl = make_quantity()
        self._del0 = make_quantity()
        self._ktcon = make_quantity_2D(Int)
        self._ktconn = make_quantity_2D(Int)
        self._pdot = make_quantity_2D()
        self._qlko_ktcon = make_quantity_2D()
        self._edt = make_quantity_2D()
        self._aa1 = make_quantity_2D()
        self._cina = make_quantity_2D()
        self._vshear = make_quantity_2D()
        self._gdx = make_quantity_2D()
        self._c0 = make_quantity_2D()
        self._c0t = make_quantity()
        self._tx1 = make_quantity_2D()
        self._kpbl = make_quantity_2D(Int)
        self._zo = make_quantity()
        self._zi = make_quantity()
        self._pfld = make_quantity()
        self._eta = make_quantity()
        self._hcko = make_quantity()
        self._qcko = make_quantity()
        self._qrcko = make_quantity()
        self._ucko = make_quantity()
        self._vcko = make_quantity()
        self._dbyo = make_quantity()
        self._pwo = make_quantity()
        self._dellal = make_quantity()
        self._to = make_quantity()
        self._qo = make_quantity()
        self._uo = make_quantity()
        self._vo = make_quantity()
        self._wu2 = make_quantity()
        self._buo = make_quantity()
        self._cnvwt = make_quantity()
        self._qeso = make_quantity()
        self._hmax = make_quantity_2D()
        self._po = make_quantity()

        # Configure stencils
        self._pa_to_cb = stencil_factory.from_origin_domain(
            func=pa_to_cb,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_col_arr = stencil_factory.from_origin_domain(
            func=init_col_arr,
            externals={"km": self._km},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_par_and_arr = stencil_factory.from_origin_domain(
            func=init_par_and_arr,
            externals={
                "asolfac": self._asolfac,
                "c0s": self._c0s,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_kbm_kmax = stencil_factory.from_origin_domain(
            func=init_kbm_kmax,
            externals={"km": self._km},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_final = stencil_factory.from_origin_domain(
            func=init_final,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_tracers = stencil_factory.from_origin_domain(
            func=init_tracers,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_static0 = stencil_factory.from_origin_domain(
            func=stencil_static0,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_static1 = stencil_factory.from_origin_domain(
            func=stencil_static1,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_ntrstatic0 = stencil_factory.from_origin_domain(
            func=stencil_ntrstatic0,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
    
    def __call__(
        self,
        delp,
        prslp,
        psp,
        phil,
        qtr,
        q1,
        t1,
        rn,
        kbot,
        ktop,
        kcnv,
        islimsk,
        dot,
        hpbl,
        ud_mf,
        dt_mf,
        u1,
        v1,
        garea,
        cnvw,
        cnvc,
        cnvflg,
        flg,
        kbcon,
        kmax,
        kbm,
        kb,
        heo,
        heso,
    ):
        # Convert input Pa terms to Cb terms
        self._pa_to_cb(
            psp,
            prslp,
            delp,
            self._ps,
            self._prsl,
            self._del0,
        )

        self._init_col_arr(
            kcnv,
            cnvflg,
            kbot,
            ktop,
            kbcon,
            kb,
            self._ktcon,
            self._ktconn,
            self._pdot,
            rn,
            self._qlko_ktcon,
            self._edt,
            self._aa1,
            self._cina,
            self._vshear,
            self._gdx,
            garea,
        )
        if exit_routine(cnvflg.view[:]):
            return

        conv_a = copy.deepcopy(cnvflg.view[:])
        conv_b = np.ones_like(conv_a)

        cols = col_diffs(conv_a, conv_b)
        print("Post-init: ", cols)

        self._init_par_and_arr(
            islimsk,
            self._c0,
            t1,
            self._c0t,
            cnvw,
            cnvc,
            ud_mf,
            dt_mf,
        )
        self._init_kbm_kmax(
            kbm,
            kmax,
            self._tx1,
            self._ps,
            self._prsl,
            self._k_mask,
        )
        self._init_final(
            kbm,
            kmax,
            flg,
            cnvflg,
            self._kpbl,
            self._prsl,
            self._zo,
            phil,
            self._zi,
            self._pfld,
            self._eta,
            self._hcko,
            self._qcko,
            self._qrcko,
            self._ucko,
            self._vcko,
            self._dbyo,
            self._pwo,
            self._dellal,
            self._to,
            self._qo,
            self._uo,
            self._vo,
            self._wu2,
            self._buo,
            self._drag,
            self._cnvwt,
            self._qeso,
            heo,
            heso,
            hpbl,
            t1,
            q1,
            u1,
            v1,
            self._k_mask,
        )

        # Init tracers
        for n_tracer in range(self._ntr):
            if (n_tracer != self._ntiw) and (n_tracer != self._ntcw):
                self._init_tracers(
                    cnvflg,
                    self._k_mask,
                    kmax,
                    self._ctr,
                    self._ctro,
                    self._ecko,
                    qtr,
                    n_tracer,
                )

        self._stencil_static0(
            cnvflg,
            self._hmax,
            heo,
            kb,
            self._k_mask,
            self._kpbl,
            kmax,
            self._zo,
            self._to,
            self._qeso,
            self._qo,
            self._po,
            self._uo,
            self._vo,
            heso,
            self._pfld,
        )
        for n_tracer in range(self._ntr):
            if (n_tracer != self._ntiw) and (n_tracer != self._ntcw):
                self._stencil_ntrstatic0(
                    cnvflg,
                    self._k_mask,
                    kmax,
                    self._ctro,
                    n_tracer,
                )

        self._static1(
            cnvflg,
            flg,
            kbcon,
            kmax,
            self._k_mask,
            kbm,
            kb,
            self._heo_kb,
            heo,
            heso,
        )

class Static2:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: ShallowConvectionConfig,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._ntk = config.ntke
        self._ntiw = config.ntiw
        self._ntcw = config.ntcw
        self._ntr = config.nsamftrac
        self._ncloud = config.ncld
        self._dt2 = config.dt_atmos

        # Determine whether to perform aerosol transport #
        self._do_aerosols = (config.itc > 0) and (config.ntchm > 0) and (self._ntr > 0)
        if self._do_aerosols:
            self._do_aerosols = self._ntr >= config.itc
        if self._do_aerosols:
            raise NotImplementedError(
                "Shallow convection of aerosols is not implemented yet"
            )

        self._clam = config.clam_shal
        self._c0s = config.c0s_shal
        self._c1 = config.c1_shal
        self._pgcon = config.pgcon_shal
        self._asolfac = config.asolfac_shal

        self._km = grid_indexing.domain[2]
        self._km1 = grid_indexing.domain[2] - 1
        self.TRACER_DIM = TRACER_DIM

        self.quantity_factory = quantity_factory
        self.quantity_factory.set_extra_dim_lengths(
            **{
                self.TRACER_DIM: int(self._ntr + 2),
            }
        )

        # Tracers are kind of borked right now. In Fortran the water vapor is passed in
        # separately, while all others come in via the variable "qtr" which has
        # dimensions (i, j, k, n_tracers - 1), ice and liquid water are stored in
        # qtr[:, :, :, 0] and qtr[:, :, :, 1] and everything is reorganized to
        # accomodate that. Aerosols live in qtr[:, :, :, itc:].
        # If we're being straightforward about it we'd have our 4D tracer array with
        # special handling for ntvapor, ntiw, and ntcw (like we do for TKE), have an
        # `if n not in [ntvap, ntiw, ntcw]:` around the other tracer calls, and then
        # do something similar with the aerosols.
        # A better solution would be to have the attributes we want accessible easily
        # so we can send the aerosols into the aerosol calculations by attribute, and
        # similarly except (or invoke) vapor etc. from the other calculations.

        def make_quantity():
            return quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )

        def make_quantity_2D(type=Float):
            return quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=type)

        # Allocate arrays

        # Layer mask:
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k

        self._cnvflg = make_quantity_2D(Bool)
        self._heo_kb = make_quantity_2D()
        self._drag = make_quantity()
        self._ps = make_quantity_2D()
        self._prsl = make_quantity()
        self._del0 = make_quantity()
        self._kbcon = make_quantity_2D(Int)
        self._kbcon1 = make_quantity_2D(Int)
        self._kb = make_quantity_2D(Int)
        self._ktcon = make_quantity_2D(Int)
        self._ktconn = make_quantity_2D(Int)
        self._pdot = make_quantity_2D()
        self._qlko_ktcon = make_quantity_2D()
        self._edt = make_quantity_2D()
        self._aa1 = make_quantity_2D()
        self._cina = make_quantity_2D()
        self._vshear = make_quantity_2D()
        self._gdx = make_quantity_2D()
        self._c0 = make_quantity_2D()
        self._c0t = make_quantity()
        self._kbm = make_quantity_2D(Int)
        self._kmax = make_quantity_2D(Int)
        self._tx1 = make_quantity_2D()
        self._kpbl = make_quantity_2D(Int)
        self._flg = make_quantity_2D(Bool)
        self._zo = make_quantity()
        self._zi = make_quantity()
        self._pfld = make_quantity()
        self._eta = make_quantity()
        self._hcko = make_quantity()
        self._qcko = make_quantity()
        self._qrcko = make_quantity()
        self._ucko = make_quantity()
        self._vcko = make_quantity()
        self._dbyo = make_quantity()
        self._pwo = make_quantity()
        self._dellal = make_quantity()
        self._to = make_quantity()
        self._qo = make_quantity()
        self._uo = make_quantity()
        self._vo = make_quantity()
        self._wu2 = make_quantity()
        self._buo = make_quantity()
        self._cnvwt = make_quantity()
        self._qeso = make_quantity()
        self._heo = make_quantity()
        self._heso = make_quantity()
        self._hmax = make_quantity_2D()
        self._po = make_quantity()
        self._pfld_kb = make_quantity_2D()
        self._pfld_kbcon = make_quantity_2D()
        self._pfld_kbcon1 = make_quantity_2D()
        self._sumx = make_quantity_2D()
        self._wc = make_quantity_2D()
        self._tkemean = make_quantity_2D()
        self._clamt = make_quantity_2D()
        self._xlamue = make_quantity()
        self._xlamud = make_quantity_2D()
        self._xmbmax = make_quantity_2D()
        self._ktcon1 = make_quantity_2D(Int)
        self._zi_kb = make_quantity_2D()
        self._zi_ktcon = make_quantity_2D()
        self._zi_kbcon = make_quantity_2D()
        self._dellah = make_quantity()
        self._dellaq = make_quantity()
        self._dellau = make_quantity()
        self._dellav = make_quantity()
        self._dtconv = make_quantity_2D()
        self._tauadv = make_quantity_2D()
        self._xmb = make_quantity_2D()
        self._sigmagfm = make_quantity_2D()
        self._scaldfunc = make_quantity_2D()
        self._umean = make_quantity_2D()
        self._delhbar = make_quantity_2D()
        self._delqbar = make_quantity_2D()
        self._deltbar = make_quantity_2D()
        self._delubar = make_quantity_2D()
        self._delvbar = make_quantity_2D()
        self._qcond = make_quantity_2D()
        self._rntot = make_quantity_2D()
        self._delqev = make_quantity_2D()
        self._delq2 = make_quantity_2D()
        self._deltv = make_quantity_2D()
        self._delq = make_quantity_2D()
        self._qevap = make_quantity_2D()

        self._ctr = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.TRACER_DIM],
            units="unknown",
            dtype=Float,
        )

        self._ctro = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.TRACER_DIM],
            units="unknown",
            dtype=Float,
        )

        self._ecko = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.TRACER_DIM],
            units="unknown",
            dtype=Float,
        )

        self._dellae = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.TRACER_DIM],
            units="unknown",
            dtype=Float,
        )

        self._delebar = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.TRACER_DIM],
            units="unknown",
            dtype=Float,
        )

        # Configure stencils
        self._pa_to_cb = stencil_factory.from_origin_domain(
            func=pa_to_cb,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_col_arr = stencil_factory.from_origin_domain(
            func=init_col_arr,
            externals={"km": self._km},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_par_and_arr = stencil_factory.from_origin_domain(
            func=init_par_and_arr,
            externals={
                "asolfac": self._asolfac,
                "c0s": self._c0s,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_kbm_kmax = stencil_factory.from_origin_domain(
            func=init_kbm_kmax,
            externals={"km": self._km},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_final = stencil_factory.from_origin_domain(
            func=init_final,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._init_tracers = stencil_factory.from_origin_domain(
            func=init_tracers,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_static0 = stencil_factory.from_origin_domain(
            func=stencil_static0,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_static1 = stencil_factory.from_origin_domain(
            func=stencil_static1,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_static2 = stencil_factory.from_origin_domain(
            func=stencil_static2,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
    ):
        # Convert input Pa terms to Cb terms
        self._pa_to_cb(
            psp,
            prslp,
            delp,
            self._ps,
            self._prsl,
            self._del0,
        )

        self._init_col_arr(
            kcnv,
            self._cnvflg,
            kbot,
            ktop,
            self._kbcon,
            self._kb,
            self._ktcon,
            self._ktconn,
            self._pdot,
            rn,
            self._qlko_ktcon,
            self._edt,
            self._aa1,
            self._cina,
            self._vshear,
            self._gdx,
            garea,
        )
        if exit_routine(self._cnvflg.view[:]):
            return

        conv_a = copy.deepcopy(self._cnvflg.view[:])
        conv_b = np.ones_like(conv_a)

        cols = col_diffs(conv_a, conv_b)
        print("Post-init: ", cols)

        self._init_par_and_arr(
            islimsk,
            self._c0,
            t1,
            self._c0t,
            cnvw,
            cnvc,
            ud_mf,
            dt_mf,
        )
        self._init_kbm_kmax(
            self._kbm,
            self._kmax,
            self._tx1,
            self._ps,
            self._prsl,
            self._k_mask,
        )
        self._init_final(
            self._kbm,
            self._kmax,
            self._flg,
            self._cnvflg,
            self._kpbl,
            self._prsl,
            self._zo,
            phil,
            self._zi,
            self._pfld,
            self._eta,
            self._hcko,
            self._qcko,
            self._qrcko,
            self._ucko,
            self._vcko,
            self._dbyo,
            self._pwo,
            self._dellal,
            self._to,
            self._qo,
            self._uo,
            self._vo,
            self._wu2,
            self._buo,
            self._drag,
            self._cnvwt,
            self._qeso,
            self._heo,
            self._heso,
            hpbl,
            t1,
            q1,
            u1,
            v1,
            self._k_mask,
        )

        # Init tracers
        for n_tracer in range(self._ntr):
            if (n_tracer != self._ntiw) and (n_tracer != self._ntcw):
                self._init_tracers(
                    self._cnvflg,
                    self._k_mask,
                    self._kmax,
                    self._ctr,
                    self._ctro,
                    self._ecko,
                    qtr,
                    n_tracer,
                )

        self._stencil_static0(
            self._cnvflg,
            self._hmax,
            self._heo,
            self._kb,
            self._k_mask,
            self._kpbl,
            self._kmax,
            self._zo,
            self._to,
            self._qeso,
            self._qo,
            self._po,
            self._uo,
            self._vo,
            self._heso,
            self._pfld,
        )
        for n_tracer in range(self._ntr):
            if (n_tracer != self._ntiw) and (n_tracer != self._ntcw):
                self._stencil_ntrstatic0(
                    self._cnvflg,
                    self._k_mask,
                    self._kmax,
                    self._ctro,
                    n_tracer,
                )

        self._stencil_static1(
            self._cnvflg,
            self._flg,
            self._kbcon,
            self._kmax,
            self._k_mask,
            self._kbm,
            self._kb,
            self._heo_kb,
            self._heo,
            self._heso,
        )
        if exit_routine(self._cnvflg.view[:]):
            return

        conv_b = copy.deepcopy(self._cnvflg.view[:])

        columns = col_diffs(conv_a, conv_b)
        print("after static1: ", columns)

        self._stencil_static2(
            cnvflg,
            pdot,
            dot,
            islimsk,
            self._k_mask,
            kbcon,
            kb,
            pfld,
            pfld_kb,
            pfld_kbcon,
        )

class UpdateKB9:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._pfld_kbcon = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=Float)
        self._pfld_kbcon1 = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=Float)
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k
        
        self._set_pfld_kbcon = stencil_factory.from_origin_domain(
            func=set_pfld_kbcon,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_update_kbcon1_cnvflg = stencil_factory.from_origin_domain(
            func=stencil_update_kbcon1_cnvflg,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_static9 = stencil_factory.from_origin_domain(
            func=stencil_static9,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
    
    def __call__(
        self,
        dbyo: FloatField,
        cnvflg: BoolFieldIJ,
        kmax: IntFieldIJ,
        kbm: IntFieldIJ,
        kbcon: IntFieldIJ,
        kbcon1: IntFieldIJ,
        flg: BoolFieldIJ,
        pfld: FloatField,
    ):
        self._set_pfld_kbcon(
            kbcon,
            self._k_mask,
            pfld,
            self._pfld_kbcon
        )
        self._stencil_update_kbcon1_cnvflg(
            dbyo,
            cnvflg,
            kmax,
            kbm,
            kbcon,
            kbcon1,
            flg,
            self._k_mask,
        )
        self._stencil_static9(
            cnvflg,
            pfld,
            self._pfld_kbcon,
            self._pfld_kbcon1,
            self._k_mask,
            kbcon1,
        )

class Static12:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        c1: Float,
        ncloud: Int,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._heo_kb = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=Float)
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k
        
        self._static12 = stencil_factory.from_origin_domain(
            func=stencil_static12,
            externals={"c1": c1, "ncloud": ncloud},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
    
    def __call__(
        self,
        cnvflg: BoolFieldIJ,
        aa1: FloatFieldIJ,
        flg: BoolFieldIJ,
        ktcon1: IntFieldIJ,
        kbm: IntFieldIJ,
        ktcon: IntFieldIJ,
        zo: FloatField,
        qeso: FloatField,
        to: FloatField,
        dbyo: FloatField,
        zi: FloatField,
        xlamue: FloatField,
        xlamud: FloatFieldIJ,
        qcko: FloatField,
        qrcko: FloatField,
        qo: FloatField,
        eta: FloatField,
        del0: FloatField,
        c0t: FloatField,
        pwo: FloatField,
        cnvwt: FloatField,
        buo: FloatField,
        wu2: FloatField,
        wc: FloatFieldIJ,
        sumx: FloatFieldIJ,
        kbcon1: IntFieldIJ,
        drag: FloatField,
        dellal: FloatField,
    ):
        self._static12(
            cnvflg,
            aa1,
            flg,
            ktcon1,
            kbm,
            self._k_mask,
            ktcon,
            zo,
            qeso,
            to,
            dbyo,
            zi,
            xlamue,
            xlamud,
            qcko,
            qrcko,
            qo,
            eta,
            del0,
            c0t,
            pwo,
            cnvwt,
            buo,
            wu2,
            wc,
            sumx,
            kbcon1,
            drag,
            dellal,
        )

class FeedbackCtrl:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        dt2: Float,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._ud_mf = quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )
        self._dt_mf = quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k
        
        self._feedback_control_update_mass_flux = stencil_factory.from_origin_domain(
            func=feedback_control_update_mass_flux,
            externals={"dt2": dt2},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
    
    def __call__(
        self,
        cnvflg: BoolFieldIJ,
        kmax: IntFieldIJ,
        kb: IntFieldIJ,
        ktcon: IntFieldIJ,
        flg: BoolFieldIJ,
        islimsk: IntFieldIJ,
        ktop: IntFieldIJ,
        kbot: IntFieldIJ,
        kbcon: IntFieldIJ,
        kcnv: IntFieldIJ,
        qeso: FloatField,
        pfld: FloatField,
        delhbar: FloatFieldIJ,
        delqbar: FloatFieldIJ,
        deltbar: FloatFieldIJ,
        delubar: FloatFieldIJ,
        delvbar: FloatFieldIJ,
        qcond: FloatFieldIJ,
        dellah: FloatField,
        dellaq: FloatField,
        t1: FloatField,
        xmb: FloatFieldIJ,
        q1: FloatField,
        u1: FloatField,
        dellau: FloatField,
        v1: FloatField,
        dellav: FloatField,
        del0: FloatField,
        rntot: FloatFieldIJ,
        delqev: FloatFieldIJ,
        delq2: FloatFieldIJ,
        pwo: FloatField,
        deltv: FloatFieldIJ,
        delq: FloatFieldIJ,
        qevap: FloatFieldIJ,
        rn: FloatFieldIJ,
        edt: FloatFieldIJ,
        cnvw: FloatField,
        cnvwt: FloatField,
        cnvc: FloatField,
        eta: FloatField,
    ):
        self._feedback_control_update_mass_flux(
            cnvflg,
            self._k_mask,
            kmax,
            kb,
            ktcon,
            flg,
            islimsk,
            ktop,
            kbot,
            kbcon,
            kcnv,
            qeso,
            pfld,
            delhbar,
            delqbar,
            deltbar,
            delubar,
            delvbar,
            qcond,
            dellah,
            dellaq,
            t1,
            xmb,
            q1,
            u1,
            dellau,
            v1,
            dellav,
            del0,
            rntot,
            delqev,
            delq2,
            pwo,
            deltv,
            delq,
            qevap,
            rn,
            edt,
            cnvw,
            cnvwt,
            cnvc,
            self._ud_mf,
            self._dt_mf,
            eta,
        )

class SC13:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        ncloud: Int,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._ncloud = ncloud
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k
        self._stencil_static13 = stencil_factory.from_origin_domain(
                func=stencil_static13,
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )

    def __call__(
        self,
        cnvflg,
        ktcon,
        qeso,
        to,
        dbyo,
        qcko,
        qlko_ktcon,
    ):
        if self._ncloud > 0:
            self._stencil_static13(
                cnvflg,
                self._k_mask,
                ktcon,
                qeso,
                to,
                dbyo,
                qcko,
                qlko_ktcon,
            )

class CompTendencies:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        dt2: Float,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._dt2=dt2
        self._zi_ktcon = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=Float)
        self._zi_kbcon = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=Float)
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k
        self._comp_tendencies = stencil_factory.from_origin_domain(
            func=comp_tendencies,
            externals={"dt2": self._dt2},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        pass
    def __call__(
        self,
        cnvflg,
        kbcon1,
        kbcon,
        ktcon1,
        ktcon,
        kb,
        kmax,
        dellah,
        dellaq,
        dellau,
        dellav,
        del0,
        zi,
        heo,
        qo,
        xlamue,
        xlamud,
        eta,
        hcko,
        qrcko,
        uo,
        ucko,
        vo,
        vcko,
        qcko,
        dellal,
        qlko_ktcon,
        wc,
        gdx,
        dtconv,
        u1,
        v1,
        po,
        to,
        tauadv,
        xmb,
        sigmagfm,
        garea,
        scaldfunc,
        xmbmax,
        sumx,
        umean,
    ):
        self._comp_tendencies(
            cnvflg,
            self._k_mask,
            kmax,
            kb,
            ktcon,
            ktcon1,
            kbcon1,
            kbcon,
            dellah,
            dellaq,
            dellau,
            dellav,
            del0,
            zi,
            self._zi_ktcon,
            self._zi_kbcon,
            heo,
            qo,
            xlamue,
            xlamud,
            eta,
            hcko,
            qrcko,
            uo,
            ucko,
            vo,
            vcko,
            qcko,
            dellal,
            qlko_ktcon,
            wc,
            gdx,
            dtconv,
            u1,
            v1,
            po,
            to,
            tauadv,
            xmb,
            sigmagfm,
            garea,
            scaldfunc,
            xmbmax,
            sumx,
            umean,
        )


class TranslateStatic1(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "cnvflg": {"serialname": "sc1_cnvflg", "shield": True},
            "flg": {"serialname": "sc1_flg", "shield": True},
            "kbcon": {"serialname": "sc1_kbcon", "shield": True, "index_variable": True},
            "kmax": {"serialname": "sc1_kmax", "shield": True, "index_variable": True},
            "kbm": {"serialname": "sc1_kbm", "shield": True, "index_variable": True},
            "kb": {"serialname": "sc1_kb", "shield": True, "index_variable": True},
            "heo": {"serialname": "sc1_heo", "shield": True},
            "heso": {"serialname": "sc1_heso", "shield": True},
            "delp": {"serialname": "sc1_delta", "shield": True},
            "prslp": {"serialname": "sc1_prsl", "shield": True},
            "psp": {"serialname": "sc1_pgr", "shield": True},
            "phil": {"serialname": "sc1_phil", "shield": True},
            "qtr": {"serialname": "sc1_clw", "shield": True},
            "q1": {"serialname": "sc1_gq0", "shield": True},
            "t1": {"serialname": "sc1_gt0", "shield": True},
            "u1": {"serialname": "sc1_gu0", "shield": True},
            "v1": {"serialname": "sc1_gv0", "shield": True},
            "rn": {"serialname": "sc1_rain1", "shield": True},
            "kbot": {"serialname": "sc1_kbot", "shield": True, "index_variable": True},
            "ktop": {"serialname": "sc1_ktop", "shield": True, "index_variable": True},
            "kcnv": {"serialname": "sc1_kcnv", "shield": True, "index_variable": True},
            "islimsk": {"serialname": "sc1_islmsk", "shield": True},
            "garea": {"serialname": "sc1_garea", "shield": True},
            "dot": {"serialname": "sc1_vvl", "shield": True},
            "hpbl": {"serialname": "sc1_hpbl", "shield": True},
            "ud_mf": {"serialname": "sc1_ud_mf", "shield": True},
            "dt_mf": {"serialname": "sc1_dt_mf", "shield": True},
            "cnvw": {"serialname": "sc1_cnvw", "shield": True},
            "cnvc": {"serialname": "sc1_cnvc", "shield": True},
        }
        self.in_vars["parameters"] = [
            "sc1_clam_shal",
            "sc1_c0s_shal",
            "sc1_c1_shal",
            "sc1_ncld",
            "sc1_pgcon_shal",
            "sc1_asolfac_shal",
            "sc1_dtp",
            "sc1_itc",
            "sc1_ntchm",
            "sc1_ntk",
            "sc1_nsamftrac",
            "sc1_ser_fscav",
        ]
        self.out_vars = {
            "cnvflg": {"serialname": "sc1_cnvflg", "shield": True},
            "flg": {"serialname": "sc1_flg", "shield": True},
            "kbcon": {"serialname": "sc1_kbcon", "shield": True, "index_variable": True},
            "kmax": {"serialname": "sc1_kmax", "shield": True, "index_variable": True},
            "kbm": {"serialname": "sc1_kbm", "shield": True, "index_variable": True},
            "kb": {"serialname": "sc1_kb", "shield": True, "index_variable": True},
            "heo": {"serialname": "sc1_heo", "shield": True},
            "heso": {"serialname": "sc1_heso", "shield": True},
        }
        self.stencil_factory = stencil_factory

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.grid_indexing = stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        config = ShallowConvectionConfig(
            dt_atmos=inputs.pop("sc_dtp"),
            ntke=int(inputs.pop("sc_ntk") - 1),
            nsamftrac=int(inputs.pop("sc_nsamftrac")),
            ncld=int(inputs.pop("sc_ncld")),
            ntchm=int(inputs.pop("sc_ntchm")),
            ntiw=0,
            ntcw=1,
            itc=int(inputs.pop("sc_itc") - 1),
            clam_shal=inputs.pop("sc_clam_shal"),
            c0s_shal=inputs.pop("sc_c0s_shal"),
            c1_shal=inputs.pop("sc_c1_shal"),
            pgcon_shal=inputs.pop("sc_pgcon_shal"),
            asolfac_shal=inputs.pop("sc_asolfac_shal"),
            fscav=inputs.pop("sc_ser_fscav"),
        )
        self.compute_func = Static1(
            self.stencil_factory,
            self.quantity_factory,
            config,
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class TranslateUpdateKb9(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "dbyo": {"serialname": "uk9_dbyo", "shield": True},
            "cnvflg": {"serialname": "uk9_cnvflg", "shield": True},
            "kmax": {"serialname": "uk9_kmax", "shield": True, "index_variable": True},
            "kbm": {"serialname": "uk9_kbm", "shield": True, "index_variable": True},
            "kbcon": {"serialname": "uk9_kbcon", "shield": True, "index_variable": True},
            "kbcon1": {"serialname": "uk9_kbcon1", "shield": True, "index_variable": True},
            "flg": {"serialname": "uk9_flg", "shield": True},
            "pfld": {"serialname": "uk9_pfld", "shield": True},
        }
        self.out_vars = {
            "dbyo": {"serialname": "uk9_dbyo", "shield": True},
            "cnvflg": {"serialname": "uk9_cnvflg", "shield": True},
            "kmax": {"serialname": "uk9_kmax", "shield": True, "index_variable": True},
            "kbm": {"serialname": "uk9_kbm", "shield": True, "index_variable": True},
            "kbcon": {"serialname": "uk9_kbcon", "shield": True, "index_variable": True},
            "kbcon1": {"serialname": "uk9_kbcon1", "shield": True, "index_variable": True},
            "flg": {"serialname": "uk9_flg", "shield": True},
            "pfld": {"serialname": "uk9_pfld", "shield": True},
        }
        self.stencil_factory = stencil_factory

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.grid_indexing = stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func = UpdateKB9(
            self.stencil_factory,
            self.quantity_factory,
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class TranslateStatic12(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "cnvflg": {"serialname": "s12_cnvflg", "shield": True},
            "aa1": {"serialname": "s12_aa1", "shield": True},
            "flg": {"serialname": "s12_flg", "shield": True},
            "ktcon1": {"serialname": "s12_ktcon1", "shield": True, "index_variable": True},
            "kbm": {"serialname": "s12_kbm", "shield": True, "index_variable": True},
            "ktcon": {"serialname": "s12_ktcon", "shield": True, "index_variable": True},
            "zo": {"serialname": "s12_zo", "shield": True},
            "qeso": {"serialname": "s12_qeso", "shield": True},
            "to": {"serialname": "s12_to", "shield": True},
            "dbyo": {"serialname": "s12_dbyo", "shield": True},
            "zi": {"serialname": "s12_zi", "shield": True},
            "xlamue": {"serialname": "s12_xlamue", "shield": True},
            "xlamud": {"serialname": "s12_xlamud", "shield": True},
            "qcko": {"serialname": "s12_qcko", "shield": True},
            "qrcko": {"serialname": "s12_qrcko", "shield": True},
            "qo": {"serialname": "s12_qo", "shield": True},
            "eta": {"serialname": "s12_eta", "shield": True},
            "del0": {"serialname": "s12_del0", "shield": True},
            "c0t": {"serialname": "s12_c0t", "shield": True},
            "pwo": {"serialname": "s12_pwo", "shield": True},
            "cnvwt": {"serialname": "s12_cnvwt", "shield": True},
            "buo": {"serialname": "s12_buo", "shield": True},
            "wu2": {"serialname": "s12_wu2", "shield": True},
            "wc": {"serialname": "s12_wc", "shield": True},
            "sumx": {"serialname": "s12_sumx", "shield": True},
            "kbcon1": {"serialname": "s12_kbcon1", "shield": True, "index_variable": True},
            "drag": {"serialname": "s12_drag", "shield": True},
            "dellal": {"serialname": "s12_dellal", "shield": True},
        }
        self.in_vars["parameters"] = [
            "s12_c1",
            "s12_ncloud",
        ]
        self.out_vars = {
            "cnvflg": {"serialname": "s12_cnvflg", "shield": True},
            "aa1": {"serialname": "s12_aa1", "shield": True},
            "flg": {"serialname": "s12_flg", "shield": True},
            "ktcon1": {"serialname": "s12_ktcon1", "shield": True, "index_variable": True},
            "kbm": {"serialname": "s12_kbm", "shield": True, "index_variable": True},
            "ktcon": {"serialname": "s12_ktcon", "shield": True, "index_variable": True},
            "zo": {"serialname": "s12_zo", "shield": True},
            "qeso": {"serialname": "s12_qeso", "shield": True},
            "to": {"serialname": "s12_to", "shield": True},
            "dbyo": {"serialname": "s12_dbyo", "shield": True},
            "zi": {"serialname": "s12_zi", "shield": True},
            "xlamue": {"serialname": "s12_xlamue", "shield": True},
            "xlamud": {"serialname": "s12_xlamud", "shield": True},
            "qcko": {"serialname": "s12_qcko", "shield": True},
            "qrcko": {"serialname": "s12_qrcko", "shield": True},
            "qo": {"serialname": "s12_qo", "shield": True},
            "eta": {"serialname": "s12_eta", "shield": True},
            "del0": {"serialname": "s12_del0", "shield": True},
            "c0t": {"serialname": "s12_c0t", "shield": True},
            "pwo": {"serialname": "s12_pwo", "shield": True},
            "cnvwt": {"serialname": "s12_cnvwt", "shield": True},
            "buo": {"serialname": "s12_buo", "shield": True},
            "wu2": {"serialname": "s12_wu2", "shield": True},
            "wc": {"serialname": "s12_wc", "shield": True},
            "sumx": {"serialname": "s12_sumx", "shield": True},
            "kbcon1": {"serialname": "s12_kbcon1", "shield": True, "index_variable": True},
            "drag": {"serialname": "s12_drag", "shield": True},
            "dellal": {"serialname": "s12_dellal", "shield": True},
        }
        self.stencil_factory = stencil_factory

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.grid_indexing = stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func = Static12(
            self.stencil_factory,
            self.quantity_factory,
            inputs.pop("s12_c1"),
            int(inputs.pop("s12_ncloud")),
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class TranslateFeedbackCtrl(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "cnvflg": {"serialname": "fc_cnvflg", "shield": True},
            "kmax": {"serialname": "fc_kmax", "shield": True, "index_variable": True},
            "kb": {"serialname": "fc_kb", "shield": True, "index_variable": True},
            "ktcon": {"serialname": "fc_ktcon", "shield": True, "index_variable": True},
            "flg": {"serialname": "fc_flg", "shield": True},
            "islimsk": {"serialname": "fc_islimsk", "shield": True},
            "ktop": {"serialname": "fc_ktop", "shield": True, "index_variable": True},
            "kbot": {"serialname": "fc_kbot", "shield": True, "index_variable": True},
            "kbcon": {"serialname": "fc_kbcon", "shield": True, "index_variable": True},
            "kcnv": {"serialname": "fc_kcnv", "shield": True, "index_variable": True},
            "qeso": {"serialname": "fc_qeso", "shield": True},
            "pfld": {"serialname": "fc_pfld", "shield": True},
            "delhbar": {"serialname": "fc_delhbar", "shield": True},
            "delqbar": {"serialname": "fc_delqbar", "shield": True},
            "deltbar": {"serialname": "fc_deltbar", "shield": True},
            "delubar": {"serialname": "fc_delubar", "shield": True},
            "delvbar": {"serialname": "fc_delvbar", "shield": True},
            "qcond": {"serialname": "fc_qcond", "shield": True},
            "dellah": {"serialname": "fc_dellah", "shield": True},
            "dellaq": {"serialname": "fc_dellaq", "shield": True},
            "t1": {"serialname": "fc_t1", "shield": True},
            "xmb": {"serialname": "fc_xmb", "shield": True},
            "q1": {"serialname": "fc_q1", "shield": True},
            "u1": {"serialname": "fc_u1", "shield": True},
            "dellau": {"serialname": "fc_dellau", "shield": True},
            "v1": {"serialname": "fc_v1", "shield": True},
            "dellav": {"serialname": "fc_dellav", "shield": True},
            "del0": {"serialname": "fc_del", "shield": True},
            "rntot": {"serialname": "fc_rntot", "shield": True},
            "delqev": {"serialname": "fc_delqev", "shield": True},
            "delq2": {"serialname": "fc_delq2", "shield": True},
            "pwo": {"serialname": "fc_pwo", "shield": True},
            "deltv": {"serialname": "fc_deltv", "shield": True},
            "delq": {"serialname": "fc_delq", "shield": True},
            "qevap": {"serialname": "fc_qevap", "shield": True},
            "rn": {"serialname": "fc_rn", "shield": True},
            "edt": {"serialname": "fc_edt", "shield": True},
            "cnvw": {"serialname": "fc_cnvw", "shield": True},
            "cnvwt": {"serialname": "fc_cnvwt", "shield": True},
            "cnvc": {"serialname": "fc_cnvc", "shield": True},
            "eta": {"serialname": "fc_eta", "shield": True},
        }
        self.in_vars["parameters"] = [
            "fc_dt2",
        ]
        self.out_vars = {
            "cnvflg": {"serialname": "fc_cnvflg", "shield": True},
            "kmax": {"serialname": "fc_kmax", "shield": True, "index_variable": True},
            "kb": {"serialname": "fc_kb", "shield": True, "index_variable": True},
            "ktcon": {"serialname": "fc_ktcon", "shield": True, "index_variable": True},
            "flg": {"serialname": "fc_flg", "shield": True},
            "islimsk": {"serialname": "fc_islimsk", "shield": True},
            "ktop": {"serialname": "fc_ktop", "shield": True, "index_variable": True},
            "kbot": {"serialname": "fc_kbot", "shield": True, "index_variable": True},
            "kbcon": {"serialname": "fc_kbcon", "shield": True, "index_variable": True},
            "kcnv": {"serialname": "fc_kcnv", "shield": True, "index_variable": True},
            "qeso": {"serialname": "fc_qeso", "shield": True},
            "pfld": {"serialname": "fc_pfld", "shield": True},
            "delhbar": {"serialname": "fc_delhbar", "shield": True},
            "delqbar": {"serialname": "fc_delqbar", "shield": True},
            "deltbar": {"serialname": "fc_deltbar", "shield": True},
            "delubar": {"serialname": "fc_delubar", "shield": True},
            "delvbar": {"serialname": "fc_delvbar", "shield": True},
            "qcond": {"serialname": "fc_qcond", "shield": True},
            "dellah": {"serialname": "fc_dellah", "shield": True},
            "dellaq": {"serialname": "fc_dellaq", "shield": True},
            "t1": {"serialname": "fc_t1", "shield": True},
            "xmb": {"serialname": "fc_xmb", "shield": True},
            "q1": {"serialname": "fc_q1", "shield": True},
            "u1": {"serialname": "fc_u1", "shield": True},
            "dellau": {"serialname": "fc_dellau", "shield": True},
            "v1": {"serialname": "fc_v1", "shield": True},
            "dellav": {"serialname": "fc_dellav", "shield": True},
            "rntot": {"serialname": "fc_rntot", "shield": True},
            "delqev": {"serialname": "fc_delqev", "shield": True},
            "delq2": {"serialname": "fc_delq2", "shield": True},
            "pwo": {"serialname": "fc_pwo", "shield": True},
            "deltv": {"serialname": "fc_deltv", "shield": True},
            "delq": {"serialname": "fc_delq", "shield": True},
            "qevap": {"serialname": "fc_qevap", "shield": True},
            "rn": {"serialname": "fc_rn", "shield": True},
            "edt": {"serialname": "fc_edt", "shield": True},
            "cnvw": {"serialname": "fc_cnvw", "shield": True},
            "cnvwt": {"serialname": "fc_cnvwt", "shield": True},
            "cnvc": {"serialname": "fc_cnvc", "shield": True},
            "eta": {"serialname": "fc_eta", "shield": True},
        }
        self.stencil_factory = stencil_factory

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.grid_indexing = stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func = FeedbackCtrl(
            self.stencil_factory,
            self.quantity_factory,
            inputs.pop("fc_dt2"),
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class TranslateSC13(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "cnvflg": {"serialname": "s13_cnvflg", "shield": True},
            "ktcon": {"serialname": "s13_ktcon", "shield": True, "index_variable": True},
            "qeso": {"serialname": "s13_qeso", "shield": True},
            "to": {"serialname": "s13_to", "shield": True},
            "dbyo": {"serialname": "s13_dbyo", "shield": True},
            "qcko": {"serialname": "s13_qcko", "shield": True},
            "qlko_ktcon": {"serialname": "s13_qlko_ktcon", "shield": True},
        }
        self.in_vars["parameters"] = [
            "s13_ncloud",
        ]
        self.out_vars = {
            "cnvflg": {"serialname": "s13_cnvflg", "shield": True},
            "ktcon": {"serialname": "s13_ktcon", "shield": True, "index_variable": True},
            "qeso": {"serialname": "s13_qeso", "shield": True},
            "to": {"serialname": "s13_to", "shield": True},
            "dbyo": {"serialname": "s13_dbyo", "shield": True},
            "qcko": {"serialname": "s13_qcko", "shield": True},
            "qlko_ktcon": {"serialname": "s13_qlko_ktcon", "shield": True},
        }
        self.stencil_factory = stencil_factory

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.grid_indexing = stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func = SC13(
            self.stencil_factory,
            self.quantity_factory,
            int(inputs.pop("s13_ncloud")),
        )
        self.compute_func(**inputs)
        print(np.argwhere(inputs['cnvflg']))
        return self.slice_output(inputs)

class TranslateCompTendencies(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "cnvflg": {"serialname": "sct_cnvflg", "shield": True},
            "kbcon1": {"serialname": "sct_kbcon1", "shield": True, "index_variable": True},
            "kbcon": {"serialname": "sct_kbcon", "shield": True, "index_variable": True},
            "ktcon1": {"serialname": "sct_ktcon1", "shield": True, "index_variable": True},
            "ktcon": {"serialname": "sct_ktcon", "shield": True, "index_variable": True},
            "kb": {"serialname": "sct_kb", "shield": True, "index_variable": True},
            "kmax": {"serialname": "sct_kmax", "shield": True, "index_variable": True},
            "dellah": {"serialname": "sct_dellah", "shield": True},
            "dellaq": {"serialname": "sct_dellaq", "shield": True},
            "dellau": {"serialname": "sct_dellau", "shield": True},
            "dellav": {"serialname": "sct_dellav", "shield": True},
            "del0": {"serialname": "sct_del", "shield": True},
            "zi": {"serialname": "sct_zi", "shield": True},
            "heo": {"serialname": "sct_heo", "shield": True},
            "qo": {"serialname": "sct_qo", "shield": True},
            "xlamue": {"serialname": "sct_xlamue", "shield": True},
            "xlamud": {"serialname": "sct_xlamud", "shield": True},
            "eta": {"serialname": "sct_eta", "shield": True},
            "hcko": {"serialname": "sct_hcko", "shield": True},
            "qrcko": {"serialname": "sct_qrcko", "shield": True},
            "uo": {"serialname": "sct_uo", "shield": True},
            "ucko": {"serialname": "sct_ucko", "shield": True},
            "vo": {"serialname": "sct_vo", "shield": True},
            "vcko": {"serialname": "sct_vcko", "shield": True},
            "qcko": {"serialname": "sct_qcko", "shield": True},
            "dellal": {"serialname": "sct_dellal", "shield": True},
            "qlko_ktcon": {"serialname": "sct_qlko_ktcon", "shield": True},
            "wc": {"serialname": "sct_wc", "shield": True},
            "gdx": {"serialname": "sct_gdx", "shield": True},
            "dtconv": {"serialname": "sct_dtconv", "shield": True},
            "u1": {"serialname": "sct_u1", "shield": True},
            "v1": {"serialname": "sct_v1", "shield": True},
            "po": {"serialname": "sct_po", "shield": True},
            "to": {"serialname": "sct_to", "shield": True},
            "tauadv": {"serialname": "sct_tauadv", "shield": True},
            "xmb": {"serialname": "sct_xmb", "shield": True},
            "sigmagfm": {"serialname": "sct_sigmagfm", "shield": True},
            "garea": {"serialname": "sct_garea", "shield": True},
            "scaldfunc": {"serialname": "sct_scaldfunc", "shield": True},
            "xmbmax": {"serialname": "sct_xmbmax", "shield": True},
            "sumx": {"serialname": "sct_sumx", "shield": True},
            "umean": {"serialname": "sct_umean", "shield": True},
        }
        self.in_vars["parameters"] = [
            "sct_dt2",
        ]
        self.out_vars = {
            "cnvflg": {"serialname": "sct_cnvflg", "shield": True},
            "kbcon1": {"serialname": "sct_kbcon1", "shield": True, "index_variable": True},
            "kbcon": {"serialname": "sct_kbcon", "shield": True, "index_variable": True},
            "ktcon1": {"serialname": "sct_ktcon1", "shield": True, "index_variable": True},
            "ktcon": {"serialname": "sct_ktcon", "shield": True, "index_variable": True},
            "kb": {"serialname": "sct_kb", "shield": True, "index_variable": True},
            "kmax": {"serialname": "sct_kmax", "shield": True, "index_variable": True},
            "dellah": {"serialname": "sct_dellah", "shield": True},
            "dellaq": {"serialname": "sct_dellaq", "shield": True},
            "dellau": {"serialname": "sct_dellau", "shield": True},
            "dellav": {"serialname": "sct_dellav", "shield": True},
            "del0": {"serialname": "sct_del", "shield": True},
            "zi": {"serialname": "sct_zi", "shield": True},
            "heo": {"serialname": "sct_heo", "shield": True},
            "qo": {"serialname": "sct_qo", "shield": True},
            "xlamue": {"serialname": "sct_xlamue", "shield": True},
            "xlamud": {"serialname": "sct_xlamud", "shield": True},
            "eta": {"serialname": "sct_eta", "shield": True},
            "hcko": {"serialname": "sct_hcko", "shield": True},
            "qrcko": {"serialname": "sct_qrcko", "shield": True},
            "uo": {"serialname": "sct_uo", "shield": True},
            "ucko": {"serialname": "sct_ucko", "shield": True},
            "vo": {"serialname": "sct_vo", "shield": True},
            "vcko": {"serialname": "sct_vcko", "shield": True},
            "qcko": {"serialname": "sct_qcko", "shield": True},
            "dellal": {"serialname": "sct_dellal", "shield": True},
            "qlko_ktcon": {"serialname": "sct_qlko_ktcon", "shield": True},
            "wc": {"serialname": "sct_wc", "shield": True},
            "gdx": {"serialname": "sct_gdx", "shield": True},
            "dtconv": {"serialname": "sct_dtconv", "shield": True},
            "u1": {"serialname": "sct_u1", "shield": True},
            "v1": {"serialname": "sct_v1", "shield": True},
            "po": {"serialname": "sct_po", "shield": True},
            "to": {"serialname": "sct_to", "shield": True},
            "tauadv": {"serialname": "sct_tauadv", "shield": True},
            "xmb": {"serialname": "sct_xmb", "shield": True},
            "sigmagfm": {"serialname": "sct_sigmagfm", "shield": True},
            "garea": {"serialname": "sct_garea", "shield": True},
            "scaldfunc": {"serialname": "sct_scaldfunc", "shield": True},
            "xmbmax": {"serialname": "sct_xmbmax", "shield": True},
            "sumx": {"serialname": "sct_sumx", "shield": True},
            "umean": {"serialname": "sct_umean", "shield": True},
        }
        self.stencil_factory = stencil_factory

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.grid_indexing = stencil_factory.grid_indexing
        self.ignore_near_zero_errors = {"sct_tauadv": True}
        self.near_zero = 1e-30

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func = CompTendencies(
            self.stencil_factory,
            self.quantity_factory,
            inputs.pop("sct_dt2"),
        )
        print(np.argwhere(inputs['cnvflg']))
        print(len(np.argwhere(inputs['cnvflg'])))
        self.compute_func(**inputs)
        return self.slice_output(inputs)
