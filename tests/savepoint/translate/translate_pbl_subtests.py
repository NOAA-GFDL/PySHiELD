from gt4py.cartesian.gtscript import FORWARD, computation, interval

from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD._config import FloatFieldTracer
from pySHiELD.stencils.pbl.satmedmfvdiff import (
    compute_asymptotic_mixing_length,
    enhance_pbl_height_thermal,
    init_turbulence,
    mrf_pbl_2_thermal_excess,
    mrf_pbl_scheme_part1,
    stratocumulus,
    thermal_pbl_calc,
    tke_tridiag_matrix_ele_comp,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


def set_thlvx_0(
    thlvx: FloatField,
    thlvx_0: FloatFieldIJ,
):

    with computation(FORWARD):
        with interval(0, 1):
            thlvx_0 = thlvx[0, 0, 0]


def set_pbot_ptop(
    phii: FloatField,
    pbot: FloatFieldIJ,
    ptop: FloatFieldIJ,
):
    with computation(FORWARD):
        with interval(0, 1):
            ptop = phii
        with interval(-1, None):
            pbot = phii


class InitTurb:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config,
    ):
        idx = stencil_factory.grid_indexing
        km1 = idx.domain[2] - 1
        self._dt_atmos = config.dt_atmos
        self._ntiw = config.ntiw
        self._ntcw = config.ntcw
        self._ntke = config.ntke

        self._kmpbl = idx.domain[2] // 2 + 1
        self._kmscu = idx.domain[2] // 2 + 1
        self._ptop = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="Pa",
            dtype=Float,
        )
        self._pbot = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="Pa",
            dtype=Float,
        )
        self._k_mask = quantity_factory.zeros(
            [Z_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(idx.domain[2]):
            self._k_mask.data[k] = k

        self._init_turbulence = stencil_factory.from_origin_domain(
            func=init_turbulence,
            externals={
                "cap_k0_land": config.cap_k0_land,
                "do_dk_hb19": config.do_dk_hb19,
                "dt2": self._dt_atmos,
                "km1": km1,
                "ntcw": self._ntcw,
                "ntiw": self._ntiw,
                "ntke": self._ntke,
                "xkzm_hi": config.xkzm_hi,
                "xkzm_hl": config.xkzm_hl,
                "xkzm_ho": config.xkzm_ho,
                "xkzm_lim": config.xkzm_lim,
                "xkzm_mi": config.xkzm_mi,
                "xkzm_ml": config.xkzm_ml,
                "xkzm_mo": config.xkzm_mo,
                "xkzm_s": config.xkzm_s,
            },
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(0, 0, 1)),
        )

    def __call__(
        self,
        zi: FloatField,
        zl: FloatField,
        zm: FloatField,
        phii: FloatField,
        phil: FloatField,
        chz: FloatField,
        ckz: FloatField,
        area: FloatFieldIJ,
        gdx: FloatFieldIJ,
        tke: FloatField,
        q1: FloatFieldTracer,
        rdzt: FloatField,
        prn: FloatField,
        kx1: IntFieldIJ,
        prsi: FloatField,
        kinver: IntFieldIJ,
        tx1: FloatFieldIJ,
        tx2: FloatFieldIJ,
        xkzo: FloatField,
        xkzmo: FloatField,
        kpblx: IntFieldIJ,
        hpblx: FloatFieldIJ,
        pblflg: BoolFieldIJ,
        sfcflg: BoolFieldIJ,
        pcnvflg: BoolFieldIJ,
        scuflg: BoolFieldIJ,
        zorl: FloatFieldIJ,
        dusfc: FloatFieldIJ,
        dvsfc: FloatFieldIJ,
        dtsfc: FloatFieldIJ,
        dqsfc: FloatFieldIJ,
        kpbl: IntFieldIJ,
        hpbl: FloatFieldIJ,
        rbsoil: FloatFieldIJ,
        radmin: FloatFieldIJ,
        mrad: IntFieldIJ,
        krad: IntFieldIJ,
        lcld: IntFieldIJ,
        kcld: IntFieldIJ,
        theta: FloatField,
        prslk: FloatField,
        psk: FloatFieldIJ,
        t1: FloatField,
        pix: FloatField,
        qlx: FloatField,
        slx: FloatField,
        thvx: FloatField,
        qtx: FloatField,
        thlx: FloatField,
        thlvx: FloatField,
        svx: FloatField,
        thetae: FloatField,
        gotvx: FloatField,
        prsl: FloatField,
        plyr: FloatField,
        rhly: FloatField,
        qstl: FloatField,
        bf: FloatField,
        cfly: FloatField,
        crb: FloatFieldIJ,
        dtdz1: FloatFieldIJ,
        evap: FloatFieldIJ,
        heat: FloatFieldIJ,
        hlw: FloatField,
        radx: FloatField,
        sflux: FloatFieldIJ,
        shr2: FloatField,
        stress: FloatFieldIJ,
        hsw: FloatField,
        thermal: FloatFieldIJ,
        tsea: FloatFieldIJ,
        ustar: FloatFieldIJ,
        u1: FloatField,
        v1: FloatField,
        u10m: FloatFieldIJ,
        v10m: FloatFieldIJ,
        xmu: FloatFieldIJ,
        islimsk: FloatFieldIJ,
    ):
        self._init_turbulence(
            zi,
            zl,
            zm,
            phii,
            phil,
            chz,
            ckz,
            area,
            gdx,
            tke,
            q1,
            rdzt,
            prn,
            kx1,
            prsi,
            self._k_mask,
            kinver,
            tx1,
            tx2,
            xkzo,
            xkzmo,
            kpblx,
            hpblx,
            pblflg,
            sfcflg,
            pcnvflg,
            scuflg,
            zorl,
            dusfc,
            dvsfc,
            dtsfc,
            dqsfc,
            kpbl,
            hpbl,
            rbsoil,
            radmin,
            mrad,
            krad,
            lcld,
            kcld,
            theta,
            prslk,
            psk,
            t1,
            pix,
            qlx,
            slx,
            thvx,
            qtx,
            thlx,
            thlvx,
            svx,
            thetae,
            gotvx,
            prsl,
            plyr,
            rhly,
            qstl,
            bf,
            cfly,
            crb,
            dtdz1,
            evap,
            heat,
            hlw,
            radx,
            sflux,
            shr2,
            stress,
            hsw,
            thermal,
            tsea,
            ustar,
            u1,
            v1,
            u10m,
            v10m,
            xmu,
            islimsk,
            self._ptop,
            self._pbot,
        )


class MRFScheme:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        idx = stencil_factory.grid_indexing
        km1 = idx.domain[2] - 1
        self._kmpbl = idx.domain[2] // 2 + 1
        self._kmscu = idx.domain[2] // 2 + 1
        self._mrf_pbl_scheme_part1 = stencil_factory.from_origin_domain(
            func=mrf_pbl_scheme_part1,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, self._kmpbl),
        )

        self._mrf_pbl_2_thermal_excess = stencil_factory.from_origin_domain(
            func=mrf_pbl_2_thermal_excess,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )
        self._k_mask = quantity_factory.zeros(
            [Z_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(idx.domain[2]):
            self._k_mask.data[k] = k

        self._thlvx_0 = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )

    def __call__(
        self,
        crb,
        flg,
        kpblx,
        rbdn,
        rbup,
        rbsoil,
        thermal,
        thlvx,
        u1,
        v1,
        zl,
        evap,
        fh,
        fm,
        gotvx,
        zol,
        heat,
        hpbl,
        hpblx,
        kpbl,
        pblflg,
        pcnvflg,
        phih,
        phim,
        sfcflg,
        sflux,
        theta,
        ustar,
        vpert,
        zi,
    ):
        self._mrf_pbl_scheme_part1(
            crb,
            flg,
            kpblx,
            self._k_mask,
            rbdn,
            rbup,
            rbsoil,
            thermal,
            thlvx,
            self._thlvx_0,
            u1,
            v1,
            zl,
        )
        self._mrf_pbl_2_thermal_excess(
            crb,
            evap,
            fh,
            flg,
            fm,
            gotvx,
            heat,
            hpbl,
            hpblx,
            kpbl,
            kpblx,
            self._k_mask,
            pblflg,
            pcnvflg,
            phih,
            phim,
            rbdn,
            rbup,
            rbsoil,
            sfcflg,
            sflux,
            thermal,
            theta,
            ustar,
            vpert,
            zi,
            zl,
            zol,
        )


class ThermalPBL:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        idx = stencil_factory.grid_indexing
        km1 = idx.domain[2] - 1
        self._kmpbl = idx.domain[2] // 2 + 1
        self._kmscu = idx.domain[2] // 2 + 1
        self._k_mask = quantity_factory.zeros(
            [Z_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(idx.domain[2]):
            self._k_mask.data[k] = k

        self._thlvx_0 = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )

        self._thermal_pbl_calc = stencil_factory.from_origin_domain(
            func=thermal_pbl_calc,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, self._kmpbl),
        )

        self._set_thlvx_0 = stencil_factory.from_origin_domain(
            func=set_thlvx_0,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._enhance_pbl_height_thermal = stencil_factory.from_origin_domain(
            func=enhance_pbl_height_thermal,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        crb,
        flg,
        kpbl,
        rbdn,
        rbup,
        thermal,
        thlvx,
        u1,
        v1,
        zl,
        hpbl,
        pblflg,
        pcnvflg,
        zi,
    ):
        self._set_thlvx_0(
            thlvx,
            self._thlvx_0,
        )

        self._thermal_pbl_calc(
            crb,
            flg,
            kpbl,
            self._k_mask,
            rbdn,
            rbup,
            thermal,
            thlvx,
            self._thlvx_0,
            u1,
            v1,
            zl,
        )

        self._enhance_pbl_height_thermal(
            crb,
            hpbl,
            kpbl,
            self._k_mask,
            pblflg,
            pcnvflg,
            rbdn,
            rbup,
            zi,
            zl,
        )


class Stratocumulus:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        idx = stencil_factory.grid_indexing
        km1 = idx.domain[2] - 1
        self._kmpbl = idx.domain[2] // 2 + 1
        self._kmscu = idx.domain[2] // 2 + 1
        self._k_mask = quantity_factory.zeros(
            [Z_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(idx.domain[2]):
            self._k_mask.data[k] = k

        self._stratocumulus = stencil_factory.from_origin_domain(
            func=stratocumulus,
            externals={"km1": km1},
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, self._kmscu),
        )

    def __call__(
        self,
        flg,
        kcld,
        krad,
        lcld,
        radmin,
        radx,
        qlx,
        scuflg,
        zl,
    ):
        self._stratocumulus(
            flg,
            kcld,
            krad,
            lcld,
            self._k_mask,
            radmin,
            radx,
            qlx,
            scuflg,
            zl,
        )


class PBLAML:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        idx = stencil_factory.grid_indexing
        km1 = idx.domain[2] - 1
        self._kmpbl = idx.domain[2] // 2 + 1
        self._kmscu = idx.domain[2] // 2 + 1

        self._lev = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Int,
        )
        self._ptop = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )
        self._pbot = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )

        self._compute_asymptotic_mixing_length = stencil_factory.from_origin_domain(
            func=compute_asymptotic_mixing_length,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(0, 0, -1)),
        )

        self._set_pbot_ptop = stencil_factory.from_origin_domain(
            func=set_pbot_ptop,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        zldn,
        thvx,
        tke,
        gotvx,
        zl,
        tsea,
        q1,
        zi,
        rlam,
        ele,
        zol,
        gdx,
        phii,
    ):
        self._set_pbot_ptop(
            phii,
            self._pbot,
            self._ptop,
        )

        self._compute_asymptotic_mixing_length(
            zldn,
            thvx,
            tke,
            gotvx,
            zl,
            tsea,
            q1,
            zi,
            rlam,
            ele,
            zol,
            gdx,
            phii,
            self._ptop,
            self._pbot,
            self._lev,
        )


class TKETridiag:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config,
    ):
        idx = stencil_factory.grid_indexing
        km1 = idx.domain[2] - 1
        self._kmpbl = idx.domain[2] // 2 + 1
        self._kmscu = idx.domain[2] // 2 + 1
        self._dt_atmos = config.dt_atmos
        self._k_mask = quantity_factory.zeros(
            [Z_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(idx.domain[2]):
            self._k_mask.data[k] = k

        self._cu = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._rt = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._f1_p1 = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )
        self._ad_p1 = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )

        self._tke_tridiag_matrix_ele_comp = stencil_factory.from_origin_domain(
            func=tke_tridiag_matrix_ele_comp,
            externals={"dt2": self._dt_atmos},
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        ad,
        al,
        au,
        delta,
        dkq,
        f1,
        kpbl,
        krad,
        mrad,
        pcnvflg,
        prsl,
        qcdo,
        qcko,
        rdzt,
        scuflg,
        tke,
        xmf,
        xmfd,
    ):
        self._tke_tridiag_matrix_ele_comp(
            ad,
            self._ad_p1,
            al,
            au,
            delta,
            dkq,
            f1,
            self._f1_p1,
            kpbl,
            krad,
            self._k_mask,
            mrad,
            pcnvflg,
            prsl,
            qcdo,
            qcko,
            rdzt,
            scuflg,
            tke,
            xmf,
            xmfd,
            self._cu,
            self._rt,
        )
        pass


class TranslatePBLInit(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "zi": {"shield": True, "kend": namelist.npz + 1},
            "zl": {
                "shield": True,
            },
            "zm": {
                "shield": True,
            },
            "phii": {"shield": True, "kend": namelist.npz + 1},
            "phil": {
                "shield": True,
            },
            "chz": {
                "shield": True,
            },
            "ckz": {
                "shield": True,
            },
            "area": {
                "shield": True,
            },
            "gdx": {
                "shield": True,
            },
            "tke": {
                "shield": True,
            },
            "q1": {
                "shield": True,
            },
            "rdzt": {"shield": True, "kend": namelist.npz - 1},
            "prn": {"shield": True, "kend": namelist.npz - 1},
            "kx1": {"shield": True, "index_variable": True},
            "prsi": {"shield": True, "kend": namelist.npz + 1},
            "kinver": {
                "shield": True,
            },
            "tx1": {
                "shield": True,
            },
            "tx2": {
                "shield": True,
            },
            "xkzo": {"shield": True, "kend": namelist.npz - 1},
            "xkzmo": {"shield": True, "kend": namelist.npz - 1},
            "kpblx": {"shield": True, "index_variable": True},
            "hpblx": {
                "shield": True,
            },
            "pblflg": {
                "shield": True,
            },
            "sfcflg": {
                "shield": True,
            },
            "pcnvflg": {
                "shield": True,
            },
            "scuflg": {
                "shield": True,
            },
            "zorl": {
                "shield": True,
            },
            "dusfc": {
                "shield": True,
            },
            "dvsfc": {
                "shield": True,
            },
            "dtsfc": {
                "shield": True,
            },
            "dqsfc": {
                "shield": True,
            },
            "kpbl": {"shield": True, "index_variable": True},
            "hpbl": {
                "shield": True,
            },
            "rbsoil": {
                "shield": True,
            },
            "radmin": {
                "shield": True,
            },
            "mrad": {
                "shield": True,
            },
            "krad": {"shield": True, "index_variable": True},
            "lcld": {"shield": True, "index_variable": True},
            "kcld": {"shield": True, "index_variable": True},
            "theta": {
                "shield": True,
            },
            "prslk": {
                "shield": True,
            },
            "psk": {
                "shield": True,
            },
            "t1": {
                "shield": True,
            },
            "pix": {
                "shield": True,
            },
            "qlx": {
                "shield": True,
            },
            "slx": {
                "shield": True,
            },
            "thvx": {
                "shield": True,
            },
            "qtx": {
                "shield": True,
            },
            "thlx": {
                "shield": True,
            },
            "thlvx": {
                "shield": True,
            },
            "svx": {
                "shield": True,
            },
            "thetae": {
                "shield": True,
            },
            "gotvx": {
                "shield": True,
            },
            "prsl": {
                "shield": True,
            },
            "plyr": {
                "shield": True,
            },
            "rhly": {
                "shield": True,
            },
            "qstl": {
                "shield": True,
            },
            "bf": {"shield": True, "kend": namelist.npz - 1},
            "cfly": {
                "shield": True,
            },
            "crb": {
                "shield": True,
            },
            "dtdz1": {
                "shield": True,
            },
            "evap": {
                "shield": True,
            },
            "heat": {
                "shield": True,
            },
            "hlw": {
                "shield": True,
            },
            "radx": {"shield": True, "kend": namelist.npz - 1},
            "sflux": {
                "shield": True,
            },
            "shr2": {"shield": True, "kend": namelist.npz - 1},
            "stress": {
                "shield": True,
            },
            "hsw": {
                "shield": True,
            },
            "thermal": {
                "shield": True,
            },
            "tsea": {
                "shield": True,
            },
            "u10m": {
                "shield": True,
            },
            "ustar": {
                "shield": True,
            },
            "u1": {
                "shield": True,
            },
            "v1": {
                "shield": True,
            },
            "v10m": {
                "shield": True,
            },
            "xmu": {
                "shield": True,
            },
            "islimsk": {"shield": True},
        }
        self.in_vars["parameters"] = [
            "ntcw",
            "ntiw",
            "ntke",
        ]
        self.out_vars = {
            "zi": {"shield": True, "kend": namelist.npz + 1},
            "zl": {
                "shield": True,
            },
            "zm": {
                "shield": True,
            },
            "phii": {"shield": True, "kend": namelist.npz + 1},
            "phil": {
                "shield": True,
            },
            "chz": {
                "shield": True,
            },
            "ckz": {
                "shield": True,
            },
            "area": {
                "shield": True,
            },
            "gdx": {
                "shield": True,
            },
            "tke": {
                "shield": True,
            },
            "q1": {
                "shield": True,
            },
            "rdzt": {"shield": True, "kend": namelist.npz - 1},
            "prn": {"shield": True, "kend": namelist.npz - 1},
            "kx1": {"shield": True, "index_variable": True},
            "prsi": {"shield": True, "kend": namelist.npz + 1},
            "kinver": {
                "shield": True,
            },
            "tx1": {
                "shield": True,
            },
            "tx2": {
                "shield": True,
            },
            "xkzo": {"shield": True, "kend": namelist.npz - 1},
            "xkzmo": {"shield": True, "kend": namelist.npz - 1},
            "kpblx": {"shield": True, "index_variable": True},
            "hpblx": {
                "shield": True,
            },
            "pblflg": {
                "shield": True,
            },
            "sfcflg": {
                "shield": True,
            },
            "pcnvflg": {
                "shield": True,
            },
            "scuflg": {
                "shield": True,
            },
            "zorl": {
                "shield": True,
            },
            "dusfc": {
                "shield": True,
            },
            "dvsfc": {
                "shield": True,
            },
            "dtsfc": {
                "shield": True,
            },
            "dqsfc": {
                "shield": True,
            },
            "kpbl": {"shield": True, "index_variable": True},
            "hpbl": {
                "shield": True,
            },
            "rbsoil": {
                "shield": True,
            },
            "radmin": {
                "shield": True,
            },
            "mrad": {
                "shield": True,
            },
            "krad": {"shield": True, "index_variable": True},
            "lcld": {"shield": True, "index_variable": True},
            "kcld": {"shield": True, "index_variable": True},
            "theta": {
                "shield": True,
            },
            "prslk": {
                "shield": True,
            },
            "psk": {
                "shield": True,
            },
            "t1": {
                "shield": True,
            },
            "pix": {
                "shield": True,
            },
            "qlx": {
                "shield": True,
            },
            "slx": {
                "shield": True,
            },
            "thvx": {
                "shield": True,
            },
            "qtx": {
                "shield": True,
            },
            "thlx": {
                "shield": True,
            },
            "thlvx": {
                "shield": True,
            },
            "svx": {
                "shield": True,
            },
            "thetae": {
                "shield": True,
            },
            "gotvx": {
                "shield": True,
            },
            "prsl": {
                "shield": True,
            },
            "plyr": {
                "shield": True,
            },
            "rhly": {
                "shield": True,
            },
            "qstl": {
                "shield": True,
            },
            "bf": {"shield": True, "kend": namelist.npz - 1},
            "cfly": {
                "shield": True,
            },
            "crb": {
                "shield": True,
            },
            "dtdz1": {
                "shield": True,
            },
            "evap": {
                "shield": True,
            },
            "heat": {
                "shield": True,
            },
            "hlw": {
                "shield": True,
            },
            "radx": {"shield": True, "kend": namelist.npz - 1},
            "sflux": {
                "shield": True,
            },
            "shr2": {"shield": True, "kend": namelist.npz - 1},
            "stress": {
                "shield": True,
            },
            "hsw": {
                "shield": True,
            },
            "thermal": {
                "shield": True,
            },
            "tsea": {
                "shield": True,
            },
            "u10m": {
                "shield": True,
            },
            "ustar": {
                "shield": True,
            },
            "u1": {
                "shield": True,
            },
            "v1": {
                "shield": True,
            },
            "v10m": {
                "shield": True,
            },
            "xmu": {
                "shield": True,
            },
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.make_storage_data_input_vars(inputs)

        config = self.namelist.pbl
        config.ntke = int(inputs.pop("ntke") - 1)
        config.ntcw = int(inputs.pop("ntcw") - 1)
        config.ntiw = int(inputs.pop("ntiw") - 1)
        inputs["kpbl"] = inputs["kpbl"].astype(int)
        inputs["krad"] = inputs["krad"].astype(int)
        inputs["lcld"] = inputs["lcld"].astype(int)
        inputs["kcld"] = inputs["kcld"].astype(int)
        inputs["kpblx"] = inputs["kpblx"].astype(int)

        compute_func = InitTurb(
            self.stencil_factory,
            quantity_factory,
            config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateMRF(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "crb": {"serialname": "crb", "shield": True},
            "flg": {"serialname": "flg", "shield": True},
            "kpblx": {"serialname": "kpblx", "shield": True, "index_variable": True},
            "rbdn": {"serialname": "rbdn", "shield": True},
            "rbup": {"serialname": "rbup", "shield": True},
            "rbsoil": {"serialname": "rbsoil", "shield": True},
            "thermal": {"serialname": "thermal", "shield": True},
            "thlvx": {"serialname": "thlvx", "shield": True},
            "u1": {"serialname": "u1", "shield": True},
            "v1": {"serialname": "v1", "shield": True},
            "zl": {"serialname": "zl", "shield": True},
            "evap": {"serialname": "evap", "shield": True},
            "fh": {"serialname": "fh", "shield": True},
            "fm": {"serialname": "fm", "shield": True},
            "gotvx": {"serialname": "gotvx", "shield": True},
            "zol": {"serialname": "zol", "shield": True},
            "heat": {"serialname": "heat", "shield": True},
            "hpbl": {"serialname": "hpbl", "shield": True},
            "hpblx": {"serialname": "hpblx", "shield": True},
            "kpbl": {"serialname": "kpbl", "shield": True, "index_variable": True},
            "pblflg": {"serialname": "pblflg", "shield": True},
            "pcnvflg": {"serialname": "pcnvflg", "shield": True},
            "phih": {"serialname": "phih", "shield": True},
            "phim": {"serialname": "phim", "shield": True},
            "sfcflg": {"serialname": "sfcflg", "shield": True},
            "sflux": {"serialname": "sflux", "shield": True},
            "theta": {"serialname": "theta", "shield": True},
            "ustar": {"serialname": "ustar", "shield": True},
            "vpert": {"serialname": "vpert", "shield": True},
            "zi": {"serialname": "zi", "shield": True, "kend": namelist.npz + 1},
        }

        self.out_vars = {
            "crb": {"serialname": "crb", "shield": True},
            "flg": {"serialname": "flg", "shield": True},
            "kpblx": {"serialname": "kpblx", "shield": True, "index_variable": True},
            "rbdn": {"serialname": "rbdn", "shield": True},
            "rbup": {"serialname": "rbup", "shield": True},
            "rbsoil": {"serialname": "rbsoil", "shield": True},
            "thermal": {"serialname": "thermal", "shield": True},
            "thlvx": {"serialname": "thlvx", "shield": True},
            "u1": {"serialname": "u1", "shield": True},
            "v1": {"serialname": "v1", "shield": True},
            "zl": {"serialname": "zl", "shield": True},
            "evap": {"serialname": "evap", "shield": True},
            "fh": {"serialname": "fh", "shield": True},
            "fm": {"serialname": "fm", "shield": True},
            "gotvx": {"serialname": "gotvx", "shield": True},
            "zol": {"serialname": "zol", "shield": True},
            "heat": {"serialname": "heat", "shield": True},
            "hpbl": {"serialname": "hpbl", "shield": True},
            "hpblx": {"serialname": "hpblx", "shield": True},
            "kpbl": {"serialname": "kpbl", "shield": True, "index_variable": True},
            "pblflg": {"serialname": "pblflg", "shield": True},
            "pcnvflg": {"serialname": "pcnvflg", "shield": True},
            "phih": {"serialname": "phih", "shield": True},
            "phim": {"serialname": "phim", "shield": True},
            "sfcflg": {"serialname": "sfcflg", "shield": True},
            "sflux": {"serialname": "sflux", "shield": True},
            "theta": {"serialname": "theta", "shield": True},
            "ustar": {"serialname": "ustar", "shield": True},
            "vpert": {"serialname": "vpert", "shield": True},
            "zi": {"serialname": "zi", "shield": True, "kend": namelist.npz + 1},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.make_storage_data_input_vars(inputs)

        config = self.namelist.pbl
        inputs["kpbl"] = inputs["kpbl"].astype(int)
        inputs["kpblx"] = inputs["kpblx"].astype(int)

        compute_func = MRFScheme(
            self.stencil_factory,
            quantity_factory,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateThermalPBL(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "crb": {"serialname": "crb", "shield": True},
            "flg": {"serialname": "flg", "shield": True},
            "kpbl": {"serialname": "kpbl", "shield": True, "index_variable": True},
            "rbdn": {"serialname": "rbdn", "shield": True},
            "rbup": {"serialname": "rbup", "shield": True},
            "thermal": {"serialname": "thermal", "shield": True},
            "thlvx": {"serialname": "thlvx", "shield": True},
            "u1": {"serialname": "u1", "shield": True},
            "v1": {"serialname": "v1", "shield": True},
            "zl": {"serialname": "zl", "shield": True},
            "hpbl": {"serialname": "hpbl", "shield": True},
            "pblflg": {"serialname": "pblflg", "shield": True},
            "pcnvflg": {"serialname": "pcnvflg", "shield": True},
            "zi": {"serialname": "zi", "shield": True, "kend": namelist.npz + 1},
        }

        self.out_vars = {
            "crb": {"serialname": "crb", "shield": True},
            "flg": {"serialname": "flg", "shield": True},
            "kpbl": {"serialname": "kpbl", "shield": True, "index_variable": True},
            "rbdn": {"serialname": "rbdn", "shield": True},
            "rbup": {"serialname": "rbup", "shield": True},
            "thermal": {"serialname": "thermal", "shield": True},
            "thlvx": {"serialname": "thlvx", "shield": True},
            "u1": {"serialname": "u1", "shield": True},
            "v1": {"serialname": "v1", "shield": True},
            "zl": {"serialname": "zl", "shield": True},
            "hpbl": {"serialname": "hpbl", "shield": True},
            "pblflg": {"serialname": "pblflg", "shield": True},
            "pcnvflg": {"serialname": "pcnvflg", "shield": True},
            "zi": {"serialname": "zi", "shield": True, "kend": namelist.npz + 1},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.make_storage_data_input_vars(inputs)

        config = self.namelist.pbl
        inputs["kpbl"] = inputs["kpbl"].astype(int)

        compute_func = ThermalPBL(
            self.stencil_factory,
            quantity_factory,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateStratocumulus(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "flg": {"shield": True},
            "kcld": {"shield": True, "index_variable": True},
            "krad": {"shield": True, "index_variable": True},
            "lcld": {"shield": True, "index_variable": True},
            "radmin": {"shield": True},
            "radx": {"shield": True, "kend": namelist.npz - 1},
            "qlx": {"shield": True},
            "scuflg": {"shield": True},
            "zl": {"shield": True},
        }

        self.out_vars = {
            "flg": {"shield": True},
            "kcld": {"shield": True, "index_variable": True},
            "krad": {"shield": True, "index_variable": True},
            "lcld": {"shield": True, "index_variable": True},
            "radmin": {"shield": True},
            "radx": {"shield": True, "kend": namelist.npz - 1},
            "qlx": {"shield": True},
            "scuflg": {"shield": True},
            "zl": {"shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.make_storage_data_input_vars(inputs)

        config = self.namelist.pbl
        inputs["kcld"] = inputs["kcld"].astype(int)
        inputs["krad"] = inputs["krad"].astype(int)
        inputs["lcld"] = inputs["lcld"].astype(int)

        compute_func = Stratocumulus(
            self.stencil_factory,
            quantity_factory,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslatePBLAML(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "zldn": {"shield": True},
            "thvx": {"shield": True},
            "tke": {"shield": True},
            "gotvx": {"shield": True},
            "zl": {"shield": True},
            "tsea": {"shield": True},
            "q1": {"shield": True},
            "zi": {"shield": True, "kend": namelist.npz + 1},
            "rlam": {"shield": True, "kend": namelist.npz - 1},
            "ele": {"shield": True},
            "zol": {"shield": True},
            "gdx": {"shield": True},
            "phii": {"shield": True, "kend": namelist.npz + 1},
        }

        self.out_vars = {
            "zldn": {"shield": True},
            "thvx": {"shield": True},
            "tke": {"shield": True},
            "gotvx": {"shield": True},
            "zl": {"shield": True},
            "tsea": {"shield": True},
            "q1": {"shield": True},
            "zi": {"shield": True, "kend": namelist.npz + 1},
            "rlam": {"shield": True, "kend": namelist.npz - 1},
            "ele": {"shield": True},
            "zol": {"shield": True},
            "gdx": {"shield": True},
            "phii": {"shield": True, "kend": namelist.npz + 1},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.make_storage_data_input_vars(inputs)

        config = self.namelist.pbl

        compute_func = PBLAML(
            self.stencil_factory,
            quantity_factory,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateTKETridiagEle(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "ad": {"serialname": "ad", "shield": True},
            "al": {"serialname": "al", "shield": True, "kend": namelist.npz - 1},
            "au": {"serialname": "au", "shield": True, "kend": namelist.npz - 1},
            "delta": {"serialname": "del", "shield": True},
            "dkq": {"serialname": "dkq", "shield": True, "kend": namelist.npz - 1},
            "f1": {"serialname": "f1", "shield": True},
            "kpbl": {"serialname": "kpbl", "shield": True, "index_variable": True},
            "krad": {"serialname": "krad", "shield": True, "index_variable": True},
            "mrad": {"serialname": "mrad", "shield": True},
            "pcnvflg": {"serialname": "pcnvflg", "shield": True},
            "prsl": {"serialname": "prsl", "shield": True},
            "qcdo": {"serialname": "qcdo", "shield": True},
            "qcko": {"serialname": "qcko", "shield": True},
            "rdzt": {"serialname": "rdzt", "shield": True, "kend": namelist.npz - 1},
            "scuflg": {"serialname": "scuflg", "shield": True},
            "tke": {"serialname": "tke", "shield": True},
            "xmf": {"serialname": "xmf", "shield": True},
            "xmfd": {"serialname": "xmfd", "shield": True},
        }

        self.out_vars = {
            "ad": {"serialname": "ad", "shield": True},
            "al": {"serialname": "al", "shield": True, "kend": namelist.npz - 1},
            "au": {"serialname": "au", "shield": True, "kend": namelist.npz - 1},
            "delta": {"serialname": "del", "shield": True},
            "dkq": {"serialname": "dkq", "shield": True, "kend": namelist.npz - 1},
            "f1": {"serialname": "f1", "shield": True},
            "kpbl": {"serialname": "kpbl", "shield": True, "index_variable": True},
            "krad": {"serialname": "krad", "shield": True, "index_variable": True},
            "mrad": {"serialname": "mrad", "shield": True},
            "pcnvflg": {"serialname": "pcnvflg", "shield": True},
            "prsl": {"serialname": "prsl", "shield": True},
            "qcdo": {"serialname": "qcdo", "shield": True},
            "qcko": {"serialname": "qcko", "shield": True},
            "rdzt": {"serialname": "rdzt", "shield": True, "kend": namelist.npz - 1},
            "scuflg": {"serialname": "scuflg", "shield": True},
            "tke": {"serialname": "tke", "shield": True},
            "xmf": {"serialname": "xmf", "shield": True},
            "xmfd": {"serialname": "xmfd", "shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.make_storage_data_input_vars(inputs)

        config = self.namelist.pbl
        inputs["kpbl"] = inputs["kpbl"].astype(int)
        inputs["krad"] = inputs["krad"].astype(int)

        compute_func = TKETridiag(
            self.stencil_factory,
            quantity_factory,
            config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
