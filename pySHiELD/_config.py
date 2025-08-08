import dataclasses
import math
from enum import Enum, unique
from typing import List, Optional, Tuple

import f90nml

import ndsl.constants as constants
import pyshield.constants as physcons
from ndsl.dsl.typing import Bool, Float, Int
from ndsl.namelist import Namelist, NamelistDefaults
from ndsl.utils import MetaEnumStr


DEFAULT_INT = Int(0)
DEFAULT_FLOAT = Float(0.0)
DEFAULT_BOOL = False
DEFAULT_SCHEMES = ["GFS_microphysics"]


@unique
class PHYSICS_PACKAGES(Enum, metaclass=MetaEnumStr):
    GFS_microphysics = "GFS_microphysics"


@dataclasses.dataclass
class AdjustNegativeTracerConfig:
    ntimes: Float
    c1_ice: Float
    c1_liq: Float
    c1_vap: Float
    d1_ice: Float
    d1_vap: Float
    li00: Float
    li20: Float
    lv00: Float
    t_wfr: Float


@dataclasses.dataclass
class FastMPConfig:
    do_warm_rain_mp: Bool
    do_wbf: Bool
    c1_vap: Float
    c1_liq: Float
    c1_ice: Float
    lv00: Float
    li00: Float
    li20: Float
    d1_vap: Float
    d1_ice: Float
    t_wfr: Float
    ql_mlt: Float
    qs_mlt: Float
    tau_imlt: Float
    tice_mlt: Float
    do_cond_timescale: Bool
    do_evap_timescale: Bool
    do_hail: Bool
    rh_fac_evap: Float
    rh_fac_cond: Float
    rhc_cevap: Float
    tau_l2v: Float
    tau_v2l: Float
    tau_r2g: Float
    tau_smlt: Float
    tau_gmlt: Float
    tau_l2r: Float
    use_rhc_cevap: Bool
    qi0_crt: Float
    qi0_max: Float
    ql0_max: Float
    tau_wbf: Float
    do_psd_water_num: Bool
    do_psd_ice_num: Bool
    muw: Float
    mui: Float
    mur: Float
    mus: Float
    mug: Float
    muh: Float
    pcaw: Float
    pcbw: Float
    pcai: Float
    pcbi: Float
    prog_ccn: Float
    inflag: Int
    igflag: Int
    qi_lim: Float
    t_sub: Float
    is_fac: Float
    tau_i2s: Float
    fast_fr_mlt: Bool
    fast_dep_sub: Bool
    delay_cond_evap: Bool
    nconds: Int


@dataclasses.dataclass
class MicroPhysicsConfig:
    dt_full: Float
    dt_split: Float = dataclasses.field(init=False)
    ntimes: Int
    nconds: Int
    hydrostatic: Bool
    npx: Int
    npy: Int
    npz: Int
    nwat: Int
    do_qa: Bool
    do_inline_mp: Bool
    c_cracw: Float
    c_paut: Float
    c_pracs: Float
    c_psacr: Float
    c_pgacr: Float
    c_pgacs: Float
    c_psacw: Float
    c_psaci: Float
    c_pracw: Float
    c_praci: Float
    c_pgacw: Float
    c_pgaci: Float
    ccn_l: Float
    ccn_o: Float
    const_vg: Bool
    const_vi: Bool
    const_vr: Bool
    const_vw: Bool
    const_vs: Bool
    vw_fac: Float
    vs_fac: Float
    vg_fac: Float
    vi_fac: Float
    vr_fac: Float
    de_ice: Bool
    layout: Tuple[Int, Int]
    # gfdl_cloud_microphys.F90
    tau_r2g: Float
    tau_smlt: Float
    tau_gmlt: Float
    tau_g2r: Float
    tau_imlt: Float
    tau_i2s: Float
    tau_l2r: Float
    tau_g2v: Float
    tau_v2g: Float
    ql_mlt: Float
    ql0_max: Float
    qs_mlt: Float
    t_sub: Float
    t_min: Float
    qi_gen: Float
    qi_lim: Float
    qi0_max: Float
    rad_snow: Bool
    rad_graupel: Bool
    rad_rain: Bool
    do_cld_adj: Bool
    dw_ocean: Float
    dw_land: Float
    icloud_f: Int
    cld_min: Float
    tau_l2v: Float
    tau_v2l: Float
    tau_revp: Float
    tau_wbf: Float
    c2l_ord: Int
    do_sedi_heat: Bool
    do_sedi_melt: Bool
    do_sedi_uv: Bool
    do_sedi_w: Bool
    fast_sat_adj: Bool
    qc_crt: Float
    fix_negative: Bool
    do_cond_timescale: Bool
    do_evap_timescale: Bool
    do_hail: Bool
    consv_checker: Bool
    do_warm_rain_mp: Bool
    do_wbf: Bool
    do_psd_water_fall: Bool
    do_psd_ice_fall: Bool
    do_psd_water_num: Bool
    do_psd_ice_num: Bool
    do_new_acc_water: Bool
    do_new_acc_ice: Bool
    cp_heating: Bool
    fast_fr_mlt: Bool
    fast_dep_sub: Bool
    delay_cond_evap: Bool
    nconds: Int
    mp_time: Float
    prog_ccn: Bool
    qi0_crt: Float
    qs0_crt: Float
    xr_a: Float
    xr_b: Float
    xr_c: Float
    te_err: Float
    tw_err: Float
    rh_thres: Float
    rhc_cevap: Float
    rhc_revap: Float
    f_dq_p: Float
    f_dq_m: Float
    fi2s_fac: Float
    fi2g_fac: Float
    fs2g_fac: Float
    is_fac: Float
    ss_fac: Float
    gs_fac: Float
    rh_fac_evap: Float
    rh_fac_cond: Float
    sed_fac: Float
    rh_inc: Float
    rh_inr: Float
    # rh_ins: Any
    rthresh: Float
    sedi_transport: Bool
    # use_ccn: Any
    use_ppm: Bool
    use_rhc_cevap: Bool
    use_rhc_revap: Bool
    vw_max: Float
    vg_max: Float
    vi_max: Float
    vr_max: Float
    vs_max: Float
    z_slope_ice: Bool
    z_slope_liq: Bool
    tice: Float
    tice_mlt: Float
    alin: Float
    alinw: Float
    alini: Float
    alinr: Float
    alins: Float
    aling: Float
    alinh: Float
    blinw: Float
    blini: Float
    blinr: Float
    blins: Float
    bling: Float
    blinh: Float
    clin: Float
    n0w_sig: Float
    n0i_sig: Float
    n0r_sig: Float
    n0s_sig: Float
    n0g_sig: Float
    n0h_sig: Float
    n0w_exp: Float
    n0i_exp: Float
    n0r_exp: Float
    n0s_exp: Float
    n0g_exp: Float
    n0h_exp: Float
    muw: Float
    mui: Float
    mur: Float
    mus: Float
    mug: Float
    muh: Float
    cfflag: Float
    irain_f: Int
    inflag: Int
    igflag: Int
    ifflag: Int
    sedflag: Int
    vdiffflag: Int
    do_mp_table_emulation: Bool
    c_air: Float = dataclasses.field(init=False)
    c_vap: Float = dataclasses.field(init=False)
    d0_vap: Float = dataclasses.field(init=False)
    lv00: Float = dataclasses.field(init=False)
    li00: Float = dataclasses.field(init=False)
    li20: Float = dataclasses.field(init=False)
    d1_vap: Float = dataclasses.field(init=False)
    d1_ice: Float = dataclasses.field(init=False)
    c1_vap: Float = dataclasses.field(init=False)
    c1_liq: Float = dataclasses.field(init=False)
    c1_ice: Float = dataclasses.field(init=False)
    n_min: Int = dataclasses.field(init=False)
    delt: Float = dataclasses.field(init=False)
    esbasw: Float = dataclasses.field(init=False)
    tbasw: Float = dataclasses.field(init=False)
    esbasi: Float = dataclasses.field(init=False)
    tmin: Float = dataclasses.field(init=False)
    t_wfr: Float = dataclasses.field(init=False)
    pcaw: Float = dataclasses.field(init=False)
    pcbw: Float = dataclasses.field(init=False)
    pcai: Float = dataclasses.field(init=False)
    pcbi: Float = dataclasses.field(init=False)
    pcar: Float = dataclasses.field(init=False)
    pcbr: Float = dataclasses.field(init=False)
    pcas: Float = dataclasses.field(init=False)
    pcbs: Float = dataclasses.field(init=False)
    pcag: Float = dataclasses.field(init=False)
    pcbg: Float = dataclasses.field(init=False)
    pcah: Float = dataclasses.field(init=False)
    pcbh: Float = dataclasses.field(init=False)
    edaw: Float = dataclasses.field(init=False)
    edbw: Float = dataclasses.field(init=False)
    edai: Float = dataclasses.field(init=False)
    edbi: Float = dataclasses.field(init=False)
    edar: Float = dataclasses.field(init=False)
    edbr: Float = dataclasses.field(init=False)
    edas: Float = dataclasses.field(init=False)
    edbs: Float = dataclasses.field(init=False)
    edag: Float = dataclasses.field(init=False)
    edbg: Float = dataclasses.field(init=False)
    edah: Float = dataclasses.field(init=False)
    edbh: Float = dataclasses.field(init=False)
    oeaw: Float = dataclasses.field(init=False)
    oebw: Float = dataclasses.field(init=False)
    oeai: Float = dataclasses.field(init=False)
    oebi: Float = dataclasses.field(init=False)
    oear: Float = dataclasses.field(init=False)
    oebr: Float = dataclasses.field(init=False)
    oeas: Float = dataclasses.field(init=False)
    oebs: Float = dataclasses.field(init=False)
    oeag: Float = dataclasses.field(init=False)
    oebg: Float = dataclasses.field(init=False)
    oeah: Float = dataclasses.field(init=False)
    oebh: Float = dataclasses.field(init=False)
    rraw: Float = dataclasses.field(init=False)
    rrbw: Float = dataclasses.field(init=False)
    rrai: Float = dataclasses.field(init=False)
    rrbi: Float = dataclasses.field(init=False)
    rrar: Float = dataclasses.field(init=False)
    rrbr: Float = dataclasses.field(init=False)
    rras: Float = dataclasses.field(init=False)
    rrbs: Float = dataclasses.field(init=False)
    rrag: Float = dataclasses.field(init=False)
    rrbg: Float = dataclasses.field(init=False)
    rrah: Float = dataclasses.field(init=False)
    rrbh: Float = dataclasses.field(init=False)
    tvai: Float = dataclasses.field(init=False)
    tvbi: Float = dataclasses.field(init=False)
    tvar: Float = dataclasses.field(init=False)
    tvbr: Float = dataclasses.field(init=False)
    tvas: Float = dataclasses.field(init=False)
    tvbs: Float = dataclasses.field(init=False)
    tvag: Float = dataclasses.field(init=False)
    tvbg: Float = dataclasses.field(init=False)
    tvah: Float = dataclasses.field(init=False)
    tvbh: Float = dataclasses.field(init=False)
    crevp_1: Float = dataclasses.field(init=False)
    crevp_2: Float = dataclasses.field(init=False)
    crevp_3: Float = dataclasses.field(init=False)
    crevp_4: Float = dataclasses.field(init=False)
    crevp_5: Float = dataclasses.field(init=False)
    cssub_1: Float = dataclasses.field(init=False)
    cssub_2: Float = dataclasses.field(init=False)
    cssub_3: Float = dataclasses.field(init=False)
    cssub_4: Float = dataclasses.field(init=False)
    cssub_5: Float = dataclasses.field(init=False)
    cgsub_1: Float = dataclasses.field(init=False)
    cgsub_2: Float = dataclasses.field(init=False)
    cgsub_3: Float = dataclasses.field(init=False)
    cgsub_4: Float = dataclasses.field(init=False)
    cgsub_5: Float = dataclasses.field(init=False)
    csmlt_1: Float = dataclasses.field(init=False)
    csmlt_2: Float = dataclasses.field(init=False)
    csmlt_3: Float = dataclasses.field(init=False)
    csmlt_4: Float = dataclasses.field(init=False)
    cgmlt_1: Float = dataclasses.field(init=False)
    cgmlt_2: Float = dataclasses.field(init=False)
    cgmlt_3: Float = dataclasses.field(init=False)
    cgmlt_4: Float = dataclasses.field(init=False)
    cgfr_1: Float = dataclasses.field(init=False)
    cgfr_2: Float = dataclasses.field(init=False)
    normw: Float = dataclasses.field(init=False)
    normr: Float = dataclasses.field(init=False)
    normi: Float = dataclasses.field(init=False)
    norms: Float = dataclasses.field(init=False)
    normg: Float = dataclasses.field(init=False)
    expow: Float = dataclasses.field(init=False)
    expor: Float = dataclasses.field(init=False)
    expoi: Float = dataclasses.field(init=False)
    expos: Float = dataclasses.field(init=False)
    expog: Float = dataclasses.field(init=False)
    cracw: Float = dataclasses.field(init=False)
    craci: Float = dataclasses.field(init=False)
    csacw: Float = dataclasses.field(init=False)
    csaci: Float = dataclasses.field(init=False)
    cgacw: Float = dataclasses.field(init=False)
    cgaci: Float = dataclasses.field(init=False)
    cracs: Float = dataclasses.field(init=False)
    csacr: Float = dataclasses.field(init=False)
    cgacr: Float = dataclasses.field(init=False)
    cgacs: Float = dataclasses.field(init=False)
    acc: List[Float] = dataclasses.field(init=False)
    acco: List[List[Float]] = dataclasses.field(init=False)

    def __post_init__(self):
        if self.hydrostatic:
            self.c_air = constants.CP_AIR
            self.c_vap = constants.CP_VAP
            self.do_sedi_w = False
        else:
            self.c_air = constants.CV_AIR
            self.c_vap = constants.CV_VAP
        self.d0_vap = self.c_vap - physcons.C_LIQ

        # scaled constants to reduce 32 bit floating errors
        self.lv00 = (constants.HLV - self.d0_vap * physcons.TICE0) / self.c_air
        self.li00 = (constants.HLF - constants.DC_ICE * physcons.TICE0) / self.c_air
        self.li20 = self.lv00 + self.li00

        self.d1_vap = self.d0_vap / self.c_air
        self.d1_ice = constants.DC_ICE / self.c_air

        self.c1_vap = self.c_vap / self.c_air
        self.c1_liq = physcons.C_LIQ / self.c_air
        self.c1_ice = physcons.C_ICE / self.c_air

        self._calculate_particle_parameters()

        self._calculate_slope_parameters()

        self._calculate_evaporation_and_sublimation_constants()

        self._calculate_accretion_parameters()

        self._calculate_melting_and_freezing_constants()

        self._set_timestepping()

        self.n_min = 1600
        self.delt = 0.1
        self.esbasw = 1013246.0
        self.tbasw = physcons.TICE0 + 100.0
        self.esbasi = 6107.1
        self.tmin = physcons.TICE0 - self.n_min * self.delt
        if self.do_warm_rain_mp:  # unsupported
            self.t_wfr = self.tmin
        else:
            self.t_wfr = physcons.TICE0 - 40.0

    @property
    def adjustnegative(self) -> AdjustNegativeTracerConfig:
        return AdjustNegativeTracerConfig(
            ntimes=self.ntimes,
            c1_ice=self.c1_ice,
            c1_liq=self.c1_liq,
            c1_vap=self.c1_vap,
            d1_ice=self.d1_ice,
            d1_vap=self.d1_vap,
            li00=self.li00,
            li20=self.li20,
            lv00=self.lv00,
            t_wfr=self.t_wfr,
        )

    @property
    def fastmp(self) -> FastMPConfig:
        return FastMPConfig(
            do_warm_rain_mp=self.do_warm_rain_mp,
            do_wbf=self.do_wbf,
            c1_vap=self.c1_vap,
            c1_liq=self.c1_liq,
            c1_ice=self.c1_ice,
            lv00=self.lv00,
            li00=self.li00,
            li20=self.li20,
            d1_vap=self.d1_vap,
            d1_ice=self.d1_ice,
            t_wfr=self.t_wfr,
            ql_mlt=self.ql_mlt,
            qs_mlt=self.qs_mlt,
            tau_imlt=self.tau_imlt,
            tice_mlt=self.tice_mlt,
            do_cond_timescale=self.do_cond_timescale,
            do_evap_timescale=self.do_evap_timescale,
            do_hail=self.do_hail,
            rh_fac_evap=self.rh_fac_evap,
            rh_fac_cond=self.rh_fac_cond,
            rhc_cevap=self.rhc_cevap,
            tau_l2v=self.tau_l2v,
            tau_v2l=self.tau_v2l,
            tau_r2g=self.tau_r2g,
            tau_smlt=self.tau_smlt,
            tau_gmlt=self.tau_gmlt,
            tau_l2r=self.tau_l2r,
            use_rhc_cevap=self.use_rhc_cevap,
            qi0_crt=self.qi0_crt,
            qi0_max=self.qi0_max,
            ql0_max=self.ql0_max,
            tau_wbf=self.tau_wbf,
            do_psd_water_num=self.do_psd_water_num,
            do_psd_ice_num=self.do_psd_ice_num,
            muw=self.muw,
            mui=self.mui,
            mur=self.mur,
            mus=self.mus,
            mug=self.mug,
            muh=self.muh,
            pcaw=self.pcaw,
            pcbw=self.pcbw,
            pcai=self.pcai,
            pcbi=self.pcbi,
            prog_ccn=self.prog_ccn,
            inflag=self.inflag,
            igflag=self.igflag,
            qi_lim=self.qi_lim,
            t_sub=self.t_sub,
            is_fac=self.is_fac,
            tau_i2s=self.tau_i2s,
            fast_fr_mlt=self.fast_fr_mlt,
            fast_dep_sub=self.fast_dep_sub,
            delay_cond_evap=self.delay_cond_evap,
            nconds=self.nconds,
        )

    def _set_timestepping(self):
        """
        Set split timestepping info
        full_timestep is equivalent to dtm
        split_timestep is equivalent to dts
        """
        self.ntimes = Int(
            max(self.ntimes, self.dt_full / min(self.dt_full, self.mp_time))
        )
        self.dt_split = self.dt_full / self.ntimes

    def _calculate_particle_parameters(self):
        """
        Calculate parameters for particle concentration, effective diameter,
        optical extinction, radar reflectivity, and terminal velocity
        for each tracer species
        """
        muw = self.muw
        mui = self.mui
        mur = self.mur
        mus = self.mus
        mug = self.mug
        muh = self.muh
        n0w_exp = self.n0w_exp
        n0i_exp = self.n0i_exp
        n0r_exp = self.n0r_exp
        n0s_exp = self.n0s_exp
        n0g_exp = self.n0g_exp
        n0h_exp = self.n0h_exp
        n0w_sig = self.n0w_sig
        n0i_sig = self.n0i_sig
        n0r_sig = self.n0r_sig
        n0s_sig = self.n0s_sig
        n0g_sig = self.n0g_sig
        n0h_sig = self.n0h_sig
        alinw = self.alinw
        alini = self.alini
        alinr = self.alinr
        alins = self.alins
        aling = self.aling
        alinh = self.alinh
        blinw = self.blinw
        blini = self.blini
        blinr = self.blinr
        blins = self.blins
        bling = self.bling
        blinh = self.blinh

        # TODO: Refactor these to be functions, holy cow

        # Particle Concentration:
        self.pcaw = (
            math.exp(3 / (muw + 3) * math.log(n0w_sig))
            * math.gamma(muw)
            * math.exp(3 * n0w_exp / (muw + 3) * math.log(10.0))
        )
        self.pcbw = math.exp(
            muw
            / (muw + 3)
            * math.log(constants.PI * physcons.RHO_W * math.gamma(muw + 3))
        )
        self.pcai = (
            math.exp(3 / (mui + 3) * math.log(n0i_sig))
            * math.gamma(mui)
            * math.exp(3 * n0i_exp / (mui + 3) * math.log(10.0))
        )
        self.pcbi = math.exp(
            mui
            / (mui + 3)
            * math.log(constants.PI * physcons.RHO_I * math.gamma(mui + 3))
        )

        self.pcar = (
            math.exp(3 / (mur + 3) * math.log(n0r_sig))
            * math.gamma(mur)
            * math.exp(3 * n0r_exp / (mur + 3) * math.log(10.0))
        )
        self.pcbr = math.exp(
            mur
            / (mur + 3)
            * math.log(constants.PI * physcons.RHO_R * math.gamma(mur + 3))
        )

        self.pcas = (
            math.exp(3 / (mus + 3) * math.log(n0s_sig))
            * math.gamma(mus)
            * math.exp(3 * n0s_exp / (mus + 3) * math.log(10.0))
        )
        self.pcbs = math.exp(
            mus
            / (mus + 3)
            * math.log(constants.PI * physcons.RHO_S * math.gamma(mus + 3))
        )

        self.pcag = (
            math.exp(3 / (mug + 3) * math.log(n0g_sig))
            * math.gamma(mug)
            * math.exp(3 * n0g_exp / (mug + 3) * math.log(10.0))
        )
        self.pcbg = math.exp(
            mug
            / (mug + 3)
            * math.log(constants.PI * physcons.RHO_G * math.gamma(mug + 3))
        )

        self.pcah = (
            math.exp(3 / (muh + 3) * math.log(n0h_sig))
            * math.gamma(muh)
            * math.exp(3 * n0h_exp / (muh + 3) * math.log(10.0))
        )
        self.pcbh = math.exp(
            muh
            / (muh + 3)
            * math.log(constants.PI * physcons.RHO_H * math.gamma(muh + 3))
        )

        # Effective Diameter
        self.edaw = (
            math.exp(-1.0 / (muw + 3) * math.log(n0w_sig))
            * (muw + 2)
            * math.exp(-n0w_exp / (muw + 3) * math.log(10.0))
        )
        self.edbw = math.exp(
            1.0
            / (muw + 3)
            * math.log(constants.PI * physcons.RHO_W * math.gamma(muw + 3))
        )

        self.edai = (
            math.exp(-1.0 / (mui + 3) * math.log(n0i_sig))
            * (mui + 2)
            * math.exp(-n0i_exp / (mui + 3) * math.log(10.0))
        )
        self.edbi = math.exp(
            1.0
            / (mui + 3)
            * math.log(constants.PI * physcons.RHO_I * math.gamma(mui + 3))
        )

        self.edar = (
            math.exp(-1.0 / (mur + 3) * math.log(n0r_sig))
            * (mur + 2)
            * math.exp(-n0r_exp / (mur + 3) * math.log(10.0))
        )
        self.edbr = math.exp(
            1.0
            / (mur + 3)
            * math.log(constants.PI * physcons.RHO_R * math.gamma(mur + 3))
        )

        self.edas = (
            math.exp(-1.0 / (mus + 3) * math.log(n0s_sig))
            * (mus + 2)
            * math.exp(-n0s_exp / (mus + 3) * math.log(10.0))
        )
        self.edbs = math.exp(
            1.0
            / (mus + 3)
            * math.log(constants.PI * physcons.RHO_S * math.gamma(mus + 3))
        )

        self.edag = (
            math.exp(-1.0 / (mug + 3) * math.log(n0g_sig))
            * (mug + 2)
            * math.exp(-n0g_exp / (mug + 3) * math.log(10.0))
        )
        self.edbg = math.exp(
            1.0
            / (mug + 3)
            * math.log(constants.PI * physcons.RHO_G * math.gamma(mug + 3))
        )

        self.edah = (
            math.exp(-1.0 / (muh + 3) * math.log(n0h_sig))
            * (muh + 2)
            * math.exp(-n0h_exp / (muh + 3) * math.log(10.0))
        )
        self.edbh = math.exp(
            1.0
            / (muh + 3)
            * math.log(constants.PI * physcons.RHO_H * math.gamma(muh + 3))
        )

        # Optical Extinction
        self.oeaw = (
            math.exp(1.0 / (muw + 3) * math.log(n0w_sig))
            * constants.PI
            * math.gamma(muw + 2)
            * math.exp(n0w_exp / (muw + 3) * math.log(10.0))
        )
        self.oebw = 2 * math.exp(
            (muw + 2)
            / (muw + 3)
            * math.log(constants.PI * physcons.RHO_W * math.gamma(muw + 3))
        )

        self.oeai = (
            math.exp(1.0 / (mui + 3) * math.log(n0i_sig))
            * constants.PI
            * math.gamma(mui + 2)
            * math.exp(n0i_exp / (mui + 3) * math.log(10.0))
        )
        self.oebi = 2 * math.exp(
            (mui + 2)
            / (mui + 3)
            * math.log(constants.PI * physcons.RHO_I * math.gamma(mui + 3))
        )

        self.oear = (
            math.exp(1.0 / (mur + 3) * math.log(n0r_sig))
            * constants.PI
            * math.gamma(mur + 2)
            * math.exp(n0r_exp / (mur + 3) * math.log(10.0))
        )
        self.oebr = 2 * math.exp(
            (mur + 2)
            / (mur + 3)
            * math.log(constants.PI * physcons.RHO_R * math.gamma(mur + 3))
        )

        self.oeas = (
            math.exp(1.0 / (mus + 3) * math.log(n0s_sig))
            * constants.PI
            * math.gamma(mus + 2)
            * math.exp(n0s_exp / (mus + 3) * math.log(10.0))
        )
        self.oebs = 2 * math.exp(
            (mus + 2)
            / (mus + 3)
            * math.log(constants.PI * physcons.RHO_S * math.gamma(mus + 3))
        )

        self.oeag = (
            math.exp(1.0 / (mug + 3) * math.log(n0g_sig))
            * constants.PI
            * math.gamma(mug + 2)
            * math.exp(n0g_exp / (mug + 3) * math.log(10.0))
        )
        self.oebg = 2 * math.exp(
            (mug + 2)
            / (mug + 3)
            * math.log(constants.PI * physcons.RHO_G * math.gamma(mug + 3))
        )

        self.oeah = (
            math.exp(1.0 / (muh + 3) * math.log(n0h_sig))
            * constants.PI
            * math.gamma(muh + 2)
            * math.exp(n0h_exp / (muh + 3) * math.log(10.0))
        )
        self.oebh = 2 * math.exp(
            (muh + 2)
            / (muh + 3)
            * math.log(constants.PI * physcons.RHO_H * math.gamma(muh + 3))
        )

        # Radar Reflectivity
        self.rraw = (
            math.exp(-3 / (muw + 3) * math.log(n0w_sig))
            * math.gamma(muw + 6)
            * math.exp(-3 * n0w_exp / (muw + 3) * math.log(10.0))
        )
        self.rrbw = math.exp(
            (muw + 6)
            / (muw + 3)
            * math.log(constants.PI * physcons.RHO_W * math.gamma(muw + 3))
        )

        self.rrai = (
            math.exp(-3 / (mui + 3) * math.log(n0i_sig))
            * math.gamma(mui + 6)
            * math.exp(-3 * n0i_exp / (mui + 3) * math.log(10.0))
        )
        self.rrbi = math.exp(
            (mui + 6)
            / (mui + 3)
            * math.log(constants.PI * physcons.RHO_I * math.gamma(mui + 3))
        )

        self.rrar = (
            math.exp(-3 / (mur + 3) * math.log(n0r_sig))
            * math.gamma(mur + 6)
            * math.exp(-3 * n0r_exp / (mur + 3) * math.log(10.0))
        )
        self.rrbr = math.exp(
            (mur + 6)
            / (mur + 3)
            * math.log(constants.PI * physcons.RHO_R * math.gamma(mur + 3))
        )

        self.rras = (
            math.exp(-3 / (mus + 3) * math.log(n0s_sig))
            * math.gamma(mus + 6)
            * math.exp(-3 * n0s_exp / (mus + 3) * math.log(10.0))
        )
        self.rrbs = math.exp(
            (mus + 6)
            / (mus + 3)
            * math.log(constants.PI * physcons.RHO_S * math.gamma(mus + 3))
        )

        self.rrag = (
            math.exp(-3 / (mug + 3) * math.log(n0g_sig))
            * math.gamma(mug + 6)
            * math.exp(-3 * n0g_exp / (mug + 3) * math.log(10.0))
        )
        self.rrbg = math.exp(
            (mug + 6)
            / (mug + 3)
            * math.log(constants.PI * physcons.RHO_G * math.gamma(mug + 3))
        )

        self.rrah = (
            math.exp(-3 / (muh + 3) * math.log(n0h_sig))
            * math.gamma(muh + 6)
            * math.exp(-3 * n0h_exp / (muh + 3) * math.log(10.0))
        )
        self.rrbh = math.exp(
            (muh + 6)
            / (muh + 3)
            * math.log(constants.PI * physcons.RHO_H * math.gamma(muh + 3))
        )

        # Terminal Velocity
        self.tvaw = (
            math.exp(-blinw / (muw + 3) * math.log(n0w_sig))
            * alinw
            * math.gamma(muw + blinw + 3)
            * math.exp(-blinw * n0w_exp / (muw + 3) * math.log(10.0))
        )
        self.tvbw = math.exp(
            blinw
            / (muw + 3)
            * math.log(constants.PI * physcons.RHO_W * math.gamma(muw + 3))
        ) * math.gamma(muw + 3)

        self.tvai = (
            math.exp(-blini / (mui + 3) * math.log(n0i_sig))
            * alini
            * math.gamma(mui + blini + 3)
            * math.exp(-blini * n0i_exp / (mui + 3) * math.log(10.0))
        )
        self.tvbi = math.exp(
            blini
            / (mui + 3)
            * math.log(constants.PI * physcons.RHO_I * math.gamma(mui + 3))
        ) * math.gamma(mui + 3)

        self.tvar = (
            math.exp(-blinr / (mur + 3) * math.log(n0r_sig))
            * alinr
            * math.gamma(mur + blinr + 3)
            * math.exp(-blinr * n0r_exp / (mur + 3) * math.log(10.0))
        )
        self.tvbr = math.exp(
            blinr
            / (mur + 3)
            * math.log(constants.PI * physcons.RHO_R * math.gamma(mur + 3))
        ) * math.gamma(mur + 3)

        self.tvas = (
            math.exp(-blins / (mus + 3) * math.log(n0s_sig))
            * alins
            * math.gamma(mus + blins + 3)
            * math.exp(-blins * n0s_exp / (mus + 3) * math.log(10.0))
        )
        self.tvbs = math.exp(
            blins
            / (mus + 3)
            * math.log(constants.PI * physcons.RHO_S * math.gamma(mus + 3))
        ) * math.gamma(mus + 3)

        self.tvag = (
            math.exp(-bling / (mug + 3) * math.log(n0g_sig))
            * aling
            * math.gamma(mug + bling + 3)
            * math.exp(-bling * n0g_exp / (mug + 3) * math.log(10.0))
        ) * physcons.GCON
        self.tvbg = math.exp(
            bling
            / (mug + 3)
            * math.log(constants.PI * physcons.RHO_G * math.gamma(mug + 3))
        ) * math.gamma(mug + 3)

        self.tvah = (
            math.exp(-blinh / (muh + 3) * math.log(n0h_sig))
            * alinh
            * math.gamma(muh + blinh + 3)
            * math.exp(-blinh * n0h_exp / (muh + 3) * math.log(10.0))
        ) * physcons.HCON
        self.tvbh = math.exp(
            blinh
            / (muh + 3)
            * math.log(constants.PI * physcons.RHO_H * math.gamma(muh + 3))
        ) * math.gamma(muh + 3)

    def _calculate_slope_parameters(self):
        """
        Calculates slope parameters used for other variables
        """
        self.normw = (
            constants.PI * physcons.RHO_W * self.n0w_sig * math.gamma(self.muw + 3)
        )
        self.normi = (
            constants.PI * physcons.RHO_I * self.n0i_sig * math.gamma(self.mui + 3)
        )
        self.normr = (
            constants.PI * physcons.RHO_R * self.n0r_sig * math.gamma(self.mur + 3)
        )
        self.norms = (
            constants.PI * physcons.RHO_S * self.n0s_sig * math.gamma(self.mus + 3)
        )
        self.normg = (
            constants.PI * physcons.RHO_G * self.n0g_sig * math.gamma(self.mug + 3)
        )
        self.normh = (
            constants.PI * physcons.RHO_H * self.n0h_sig * math.gamma(self.muh + 3)
        )

        self.expow = math.exp(self.n0w_exp / (self.muw + 3) * math.log(10.0))
        self.expoi = math.exp(self.n0i_exp / (self.mui + 3) * math.log(10.0))
        self.expor = math.exp(self.n0r_exp / (self.mur + 3) * math.log(10.0))
        self.expos = math.exp(self.n0s_exp / (self.mus + 3) * math.log(10.0))
        self.expog = math.exp(self.n0g_exp / (self.mug + 3) * math.log(10.0))
        self.expoh = math.exp(self.n0h_exp / (self.muh + 3) * math.log(10.0))

    def _calculate_evaporation_and_sublimation_constants(self):
        """
        calculates crevp, cssub, and cgsub constants for rain evaporation,
        snow sublimation, and graupel or hail sublimation, Lin et al. (1983)
        """

        # TODO: These should also be made functions

        self.crevp_1 = (
            2.0
            * constants.PI
            * physcons.VDIFU
            * physcons.TCOND
            * constants.RVGAS
            * self.n0r_sig
            * math.gamma(1 + self.mur)
            / math.exp((1 + self.mur) / (self.mur + 3) * math.log(self.normr))
            * math.exp(2.0 * math.log(self.expor))
        )
        self.crevp_2 = 0.78
        self.crevp_3 = (
            0.31
            * physcons.SCM3
            * math.sqrt(self.alinr / physcons.VISK)
            * math.gamma((3 + 2 * self.mur + self.blinr) / 2)
            / math.exp(
                (3 + 2 * self.mur + self.blinr)
                / (self.mur + 3)
                / 2
                * math.log(self.normr)
            )
            * math.exp((1 + self.mur) / (self.mur + 3) * math.log(self.normr))
            / math.gamma(1 + self.mur)
            * math.exp((-1 - self.blinr) / 2.0 * math.log(self.expor))
        )
        self.crevp_4 = physcons.TCOND * constants.RVGAS
        self.crevp_5 = physcons.VDIFU

        self.cssub_1 = (
            2.0
            * constants.PI
            * physcons.VDIFU
            * physcons.TCOND
            * constants.RVGAS
            * self.n0s_sig
            * math.gamma(1 + self.mus)
            / math.exp((1 + self.mus) / (self.mus + 3) * math.log(self.norms))
            * math.exp(2.0 * math.log(self.expos))
        )
        self.cssub_2 = 0.78
        self.cssub_3 = (
            0.31
            * physcons.SCM3
            * math.sqrt(self.alins / physcons.VISK)
            * math.gamma((3 + 2 * self.mus + self.blins) / 2)
            / math.exp(
                (3 + 2 * self.mus + self.blins)
                / (self.mus + 3)
                / 2
                * math.log(self.norms)
            )
            * math.exp((1 + self.mus) / (self.mus + 3) * math.log(self.norms))
            / math.gamma(1 + self.mus)
            * math.exp((-1 - self.blins) / 2.0 * math.log(self.expos))
        )
        self.cssub_4 = physcons.TCOND * constants.RVGAS
        self.cssub_5 = physcons.VDIFU

        if self.do_hail:
            self.cgsub_1 = (
                2.0
                * constants.PI
                * physcons.VDIFU
                * physcons.TCOND
                * constants.RVGAS
                * self.n0h_sig
                * math.gamma(1 + self.muh)
                / math.exp((1 + self.muh) / (self.muh + 3) * math.log(self.normh))
                * math.exp(2.0 * math.log(self.expoh))
            )
            self.cgsub_2 = 0.78
            self.cgsub_3 = (
                0.31
                * physcons.SCM3
                * math.sqrt(self.alinh * physcons.HCON / physcons.VISK)
                * math.gamma((3 + 2 * self.muh + self.blinh) / 2)
                / math.exp(
                    1.0
                    / (self.muh + 3)
                    * (3 + 2 * self.muh + self.blinh)
                    / 2
                    * math.log(self.normh)
                )
                * math.exp(1.0 / (self.muh + 3) * (1 + self.muh) * math.log(self.normh))
                / math.gamma(1 + self.muh)
                * math.exp((-1 - self.blinh) / 2.0 * math.log(self.expoh))
            )
        else:
            self.cgsub_1 = (
                2.0
                * constants.PI
                * physcons.VDIFU
                * physcons.TCOND
                * constants.RVGAS
                * self.n0g_sig
                * math.gamma(1 + self.mug)
                / math.exp((1 + self.mug) / (self.mug + 3) * math.log(self.normg))
                * math.exp(2.0 * math.log(self.expog))
            )
            self.cgsub_2 = 0.78
            self.cgsub_3 = (
                0.31
                * physcons.SCM3
                * math.sqrt(self.aling * physcons.GCON / physcons.VISK)
                * math.gamma((3 + 2 * self.mug + self.bling) / 2)
                / math.exp(
                    (3 + 2 * self.mug + self.bling)
                    / (self.mug + 3)
                    / 2
                    * math.log(self.normg)
                )
                * math.exp((1 + self.mug) / (self.mug + 3) * math.log(self.normg))
                / math.gamma(1 + self.mug)
                * math.exp((-1 - self.bling) / 2.0 * math.log(self.expog))
            )
        self.cgsub_4 = physcons.TCOND * constants.RVGAS
        self.cgsub_5 = physcons.VDIFU

    def _calculate_accretion_parameters(self):
        """
        Accretion between cloud water, cloud ice, rain,snow, and graupel or hail,
        Lin et al. (1983)
        """
        self.cracw = (
            constants.PI
            * self.n0r_sig
            * self.alinr
            * math.gamma(2 + self.mur + self.blinr)
            / (
                4.0
                * math.exp(
                    (2 + self.mur + self.blinr) / (self.mur + 3) * math.log(self.normr)
                )
            )
            * math.exp((1 - self.blinr) * math.log(self.expor))
        )
        self.craci = self.cracw
        self.csacw = (
            constants.PI
            * self.n0s_sig
            * self.alins
            * math.gamma(2 + self.mus + self.blins)
            / (
                4.0
                * math.exp(
                    (2 + self.mus + self.blins) / (self.mus + 3) * math.log(self.norms)
                )
            )
            * math.exp((1 - self.blins) * math.log(self.expos))
        )
        self.csaci = self.csacw
        if self.do_hail:
            self.cgacw = (
                constants.PI
                * self.n0h_sig
                * self.alinh
                * math.gamma(2 + self.muh + self.blinh)
                * physcons.HCON
                / (
                    4.0
                    * math.exp(
                        (2 + self.muh + self.blinh)
                        / (self.muh + 3)
                        * math.log(self.normh)
                    )
                )
                * math.exp((1 - self.blinh) * math.log(self.expoh))
            )
            self.cgaci = self.cgacw
        else:
            self.cgacw = (
                constants.PI
                * self.n0g_sig
                * self.aling
                * math.gamma(2 + self.mug + self.bling)
                * physcons.GCON
                / (
                    4.0
                    * math.exp(
                        (2 + self.mug + self.bling)
                        / (self.mug + 3)
                        * math.log(self.normg)
                    )
                )
                * math.exp((1 - self.bling) * math.log(self.expog))
            )
            self.cgaci = self.cgacw

        if self.do_new_acc_water:
            self.cracw = (
                constants.PI ** 2 * self.n0r_sig * self.n0w_sig * physcons.RHO_W / 24.0
            )
            self.csacw = (
                constants.PI ** 2 * self.n0s_sig * self.n0w_sig * physcons.RHO_W / 24.0
            )
            if self.do_hail:
                self.cgacw = (
                    constants.PI ** 2
                    * self.n0h_sig
                    * self.n0w_sig
                    * physcons.RHO_W
                    / 24.0
                )
            else:
                self.cgacw = (
                    constants.PI ** 2
                    * self.n0g_sig
                    * self.n0w_sig
                    * physcons.RHO_W
                    / 24.0
                )

        if self.do_new_acc_ice:
            self.craci = (
                constants.PI ** 2 * self.n0r_sig * self.n0i_sig * physcons.RHO_I / 24.0
            )
            self.csaci = (
                constants.PI ** 2 * self.n0s_sig * self.n0i_sig * physcons.RHO_I / 24.0
            )
            if self.do_hail:
                self.cgaci = (
                    constants.PI ** 2
                    * self.n0h_sig
                    * self.n0i_sig
                    * physcons.RHO_I
                    / 24.0
                )
            else:
                self.cgaci = (
                    constants.PI ** 2
                    * self.n0g_sig
                    * self.n0i_sig
                    * physcons.RHO_I
                    / 24.0
                )
        else:
            pass

        self.cracw = self.cracw * self.c_pracw
        self.craci = self.craci * self.c_praci
        self.csacw = self.csacw * self.c_psacw
        self.csaci = self.csaci * self.c_psaci
        self.cgacw = self.cgacw * self.c_pgacw
        self.cgaci = self.cgaci * self.c_pgaci

        self.cracs = (
            constants.PI ** 2 * self.n0r_sig * self.n0s_sig * physcons.RHO_S / 24.0
        )
        self.csacr = (
            constants.PI ** 2 * self.n0s_sig * self.n0r_sig * physcons.RHO_R / 24.0
        )
        if self.do_hail:
            self.cgacs = (
                constants.PI ** 2 * self.n0h_sig * self.n0s_sig * physcons.RHO_S / 24.0
            )
            self.cgacr = (
                constants.PI ** 2 * self.n0h_sig * self.n0r_sig * physcons.RHO_R / 24.0
            )
        else:
            self.cgacs = (
                constants.PI ** 2 * self.n0g_sig * self.n0s_sig * physcons.RHO_S / 24.0
            )
            self.cgacr = (
                constants.PI ** 2 * self.n0g_sig * self.n0r_sig * physcons.RHO_R / 24.0
            )

        self.cracs *= self.c_pracs
        self.csacr *= self.c_psacr
        self.cgacs *= self.c_pgacs
        self.cgacr *= self.c_pgacr

        """
        act / ace / acc:
         0 -  1: racs (s - r)
         2 -  3: sacr (r - s)
         4 -  5: gacr (r - g)
         6 -  7: gacs (s - g)
         8 -  9: racw (w - r)
        10 - 11: raci (i - r)
        12 - 13: sacw (w - s)
        14 - 15: saci (i - s)
        16 - 17: gacw (w - g)
        18 - 19: gaci (i - g)
        """
        act = []
        act.append(self.norms)
        act.append(self.normr)
        act.append(act[1])
        act.append(act[0])
        act.append(act[1])
        if self.do_hail:
            act.append(self.normh)
        else:
            act.append(self.normg)
        act.append(act[0])
        act.append(act[5])
        act.append(self.normw)
        act.append(act[1])
        act.append(self.normi)
        act.append(act[1])
        act.append(act[8])
        act.append(act[0])
        act.append(act[10])
        act.append(act[0])
        act.append(act[8])
        act.append(act[5])
        act.append(act[10])
        act.append(act[5])

        ace = []
        ace.append(self.expos)
        ace.append(self.expor)
        ace.append(ace[1])
        ace.append(ace[0])
        ace.append(ace[1])
        if self.do_hail:
            ace.append(self.expoh)
        else:
            ace.append(self.expog)
        ace.append(ace[0])
        ace.append(ace[5])
        ace.append(self.expow)
        ace.append(ace[1])
        ace.append(self.expoi)
        ace.append(ace[1])
        ace.append(ace[8])
        ace.append(ace[0])
        ace.append(ace[10])
        ace.append(ace[0])
        ace.append(ace[8])
        ace.append(ace[5])
        ace.append(ace[10])
        ace.append(ace[5])

        acc = []
        acc.append(self.mus)
        acc.append(self.mur)
        acc.append(acc[1])
        acc.append(acc[0])
        acc.append(acc[1])
        if self.do_hail:
            acc.append(self.muh)
        else:
            acc.append(self.mug)
        acc.append(acc[0])
        acc.append(acc[5])
        acc.append(self.muw)
        acc.append(acc[1])
        acc.append(self.mui)
        acc.append(acc[1])
        acc.append(acc[8])
        acc.append(acc[0])
        acc.append(acc[10])
        acc.append(acc[0])
        acc.append(acc[8])
        acc.append(acc[5])
        acc.append(acc[10])
        acc.append(acc[5])

        self.acc = acc

        acco = []
        occ = [1.0, 2.0, 1.0]
        for i in range(3):
            accoi = []
            for k in range(10):
                accoi.append(
                    occ[i]
                    * math.gamma(6 + acc[2 * k] - (i + 1))
                    * math.gamma(acc[2 * k + 1] + i)
                    / (
                        math.exp(
                            (6 + acc[2 * k] - (i + 1))
                            / (acc[2 * k] + 3)
                            * math.log(act[2 * k])
                        )
                        * math.exp(
                            (acc[2 * k + 1] + i)
                            / (acc[2 * k + 1] + 3)
                            * math.log(act[2 * k + 1])
                        )
                    )
                    * math.exp((i - 2) * math.log(ace[2 * k]))
                    * math.exp((4 - (i + 1)) * math.log(ace[2 * k + 1]))
                )
            acco.append(accoi)

        self.acco = acco

    def _calculate_melting_and_freezing_constants(self):
        """
        Calculates parameters for snow and graupel melting,
        and rain freezing
        """

        # Snow melting, Lin et al. (1983)
        self.csmlt_1 = (
            2.0
            * constants.PI
            * physcons.TCOND
            * self.n0s_sig
            * math.gamma(1 + self.mus)
            / math.exp((1 + self.mus) / (self.mus + 3) * math.log(self.norms))
            * math.exp(2.0 * math.log(self.expos))
        )
        self.csmlt_2 = (
            2.0
            * constants.PI
            * physcons.VDIFU
            * self.n0s_sig
            * math.gamma(1 + self.mus)
            / math.exp((1 + self.mus) / (self.mus + 3) * math.log(self.norms))
            * math.exp(2.0 * math.log(self.expos))
        )
        self.csmlt_3 = self.cssub_2
        self.csmlt_4 = self.cssub_3

        # Graupel or hail melting, Lin et al. (1983)
        if self.do_hail:
            self.cgmlt_1 = (
                2.0
                * constants.PI
                * physcons.TCOND
                * self.n0h_sig
                * math.gamma(1 + self.muh)
                / math.exp((1 + self.muh) / (self.muh + 3) * math.log(self.normh))
                * math.exp(2.0 * math.log(self.expoh))
            )
            self.cgmlt_2 = (
                2.0
                * constants.PI
                * physcons.VDIFU
                * self.n0h_sig
                * math.gamma(1 + self.muh)
                / math.exp((1 + self.muh) / (self.muh + 3) * math.log(self.normh))
                * math.exp(2.0 * math.log(self.expoh))
            )
        else:
            self.cgmlt_1 = (
                2.0
                * constants.PI
                * physcons.TCOND
                * self.n0g_sig
                * math.gamma(1 + self.mug)
                / math.exp((1 + self.mug) / (self.mug + 3) * math.log(self.normg))
                * math.exp(2.0 * math.log(self.expog))
            )
            self.cgmlt_2 = (
                2.0
                * constants.PI
                * physcons.VDIFU
                * self.n0g_sig
                * math.gamma(1 + self.mug)
                / math.exp((1 + self.mug) / (self.mug + 3) * math.log(self.normg))
                * math.exp(2.0 * math.log(self.expog))
            )
        self.cgmlt_3 = self.cgsub_2
        self.cgmlt_4 = self.cgsub_3

        # Rain freezing, Lin et al. (1983)
        self.cgfr_1 = (
            1.0e2
            / 36
            * constants.PI ** 2
            * self.n0r_sig
            * physcons.RHO_R
            * math.gamma(6 + self.mur)
            / math.exp((6 + self.mur) / (self.mur + 3) * math.log(self.normr))
            * math.exp(-3.0 * math.log(self.expor))
        )
        self.cgfr_2 = 0.66


@dataclasses.dataclass
class PhysicsConfig:
    dt_atmos: Float = DEFAULT_FLOAT
    ntimes: Int = NamelistDefaults.ntimes
    nconds: Int = Int(1)
    hydrostatic: Bool = DEFAULT_BOOL
    scheme: str = "GFS"  # Which physics schemes to use
    npx: Int = DEFAULT_INT
    npy: Int = DEFAULT_INT
    npz: Int = DEFAULT_INT
    nwat: Int = DEFAULT_INT
    schemes: List = None
    do_qa: Bool = DEFAULT_BOOL
    do_inline_mp: Bool = NamelistDefaults.do_inline_mp
    c_cracw: Float = NamelistDefaults.c_cracw
    c_paut: Float = NamelistDefaults.c_paut
    c_pracs: Float = NamelistDefaults.c_pracs
    c_psacr: Float = NamelistDefaults.c_psacr
    c_pgacr: Float = NamelistDefaults.c_pgacr
    c_pgacs: Float = NamelistDefaults.c_pgacs
    c_psacw: Float = NamelistDefaults.c_psacw
    c_psaci: Float = NamelistDefaults.c_psaci
    c_pracw: Float = NamelistDefaults.c_pracw
    c_praci: Float = NamelistDefaults.c_praci
    c_pgacw: Float = NamelistDefaults.c_pgacw
    c_pgaci: Float = NamelistDefaults.c_pgaci
    ccn_l: Float = NamelistDefaults.ccn_l
    ccn_o: Float = NamelistDefaults.ccn_o
    const_vg: Bool = NamelistDefaults.const_vg
    const_vi: Bool = NamelistDefaults.const_vi
    const_vr: Bool = NamelistDefaults.const_vr
    const_vw: Bool = NamelistDefaults.const_vw
    const_vs: Bool = NamelistDefaults.const_vs
    vw_fac: Float = NamelistDefaults.vw_fac
    vs_fac: Float = NamelistDefaults.vs_fac
    vg_fac: Float = NamelistDefaults.vg_fac
    vi_fac: Float = NamelistDefaults.vi_fac
    vr_fac: Float = NamelistDefaults.vr_fac
    de_ice: Bool = NamelistDefaults.de_ice
    layout: Tuple[Int, Int] = NamelistDefaults.layout
    # gfdl_cloud_microphys.F90
    tau_r2g: Float = NamelistDefaults.tau_r2g  # rain freezing during fast_sat
    tau_smlt: Float = NamelistDefaults.tau_smlt  # snow melting timescale
    tau_gmlt: Float = NamelistDefaults.tau_gmlt  # graupel melting timescale
    tau_g2r: Float = NamelistDefaults.tau_g2r  # graupel melting to rain
    tau_imlt: Float = NamelistDefaults.tau_imlt  # cloud ice melting
    tau_i2s: Float = NamelistDefaults.tau_i2s  # cloud ice to snow auto - conversion
    tau_l2r: Float = NamelistDefaults.tau_l2r  # cloud water to rain auto - conversion
    tau_g2v: Float = NamelistDefaults.tau_g2v  # graupel sublimation
    tau_v2g: Float = (
        NamelistDefaults.tau_v2g
    )  # graupel deposition -- make it a slow process
    ql_mlt: Float = (
        NamelistDefaults.ql_mlt
    )  # max value of cloud water allowed from melted cloud ice
    ql0_max: Float = (
        NamelistDefaults.ql0_max
    )  # max cloud water value (auto converted to rain)
    qs_mlt: Float = NamelistDefaults.qs_mlt  # max cloud water due to snow melt
    t_min: Float = (
        NamelistDefaults.t_min
    )  # minimum temperature to freeze - dry all water vapor (K)
    t_sub: Float = NamelistDefaults.t_sub  # min temp for sublimation of cloud ice
    qi_gen: Float = (
        NamelistDefaults.qi_gen
    )  # max cloud ice generation during remapping step
    qi_lim: Float = (
        NamelistDefaults.qi_lim
    )  # cloud ice limiter to prevent large ice build up
    qi0_max: Float = NamelistDefaults.qi0_max  # max cloud ice value (by other sources)
    rad_snow: Bool = (
        NamelistDefaults.rad_snow
    )  # consider snow in cloud fraction calculation
    rad_graupel: Bool = (
        NamelistDefaults.rad_graupel
    )  # consider graupel in cloud fraction calculation
    rad_rain: Bool = (
        NamelistDefaults.rad_rain
    )  # consider rain in cloud fraction calculation
    do_cld_adj: Bool = NamelistDefaults.do_cld_adj  # do cloud fraction adjustment
    dw_ocean: Float = NamelistDefaults.dw_ocean  # base value for ocean
    dw_land: Float = (
        NamelistDefaults.dw_land
    )  # base value for subgrid deviation / variability over land
    # cloud scheme 0 - ?
    # 1: old fvgfs gfdl) mp implementation
    # 2: binary cloud scheme (0 / 1)
    icloud_f: Int = NamelistDefaults.icloud_f
    cld_min: Float = NamelistDefaults.cld_min  # minimum cloud fraction
    tau_l2v: Float = (
        NamelistDefaults.tau_l2v
    )  # cloud water to water vapor (evaporation)
    tau_v2l: Float = (
        NamelistDefaults.tau_v2l
    )  # water vapor to cloud water (condensation)
    tau_revp: Float = NamelistDefaults.tau_revp
    tau_wbf: Float = NamelistDefaults.tau_wbf
    c2l_ord: Int = NamelistDefaults.c2l_ord
    do_sedi_heat: Bool = NamelistDefaults.do_sedi_heat
    do_sedi_melt: Bool = NamelistDefaults.do_sedi_melt
    do_sedi_uv: Bool = NamelistDefaults.do_sedi_uv
    do_sedi_w: Bool = NamelistDefaults.do_sedi_w
    fast_sat_adj: Bool = NamelistDefaults.fast_sat_adj
    qc_crt: Float = NamelistDefaults.qc_crt
    fix_negative: Bool = NamelistDefaults.fix_negative
    do_cond_timescale: Bool = NamelistDefaults.do_cond_timescale
    do_evap_timescale: Bool = True
    do_hail: Bool = NamelistDefaults.do_hail
    consv_checker: Bool = NamelistDefaults.consv_checker
    do_warm_rain_mp: Bool = NamelistDefaults.do_warm_rain_mp
    do_wbf: Bool = NamelistDefaults.do_wbf
    do_psd_water_fall: Bool = NamelistDefaults.do_psd_water_fall
    do_psd_ice_fall: Bool = NamelistDefaults.do_psd_ice_fall
    do_psd_water_num: Bool = NamelistDefaults.do_psd_water_num
    do_psd_ice_num: Bool = NamelistDefaults.do_psd_ice_num
    do_new_acc_water: Bool = NamelistDefaults.do_new_acc_water
    do_new_acc_ice: Bool = NamelistDefaults.do_new_acc_ice
    cp_heating: Bool = NamelistDefaults.cp_heating
    fast_fr_mlt: Bool = True
    fast_dep_sub: Bool = True
    delay_cond_evap: Bool = True
    nconds: Int = 1
    mp_time: Float = NamelistDefaults.mp_time
    prog_ccn: Bool = NamelistDefaults.prog_ccn
    qi0_crt: Float = NamelistDefaults.qi0_crt
    qs0_crt: Float = NamelistDefaults.qs0_crt
    xr_a: Float = NamelistDefaults.xr_a
    xr_b: Float = NamelistDefaults.xr_b
    xr_c: Float = NamelistDefaults.xr_c
    te_err: Float = NamelistDefaults.te_err
    tw_err: Float = NamelistDefaults.tw_err
    rh_thres: Float = NamelistDefaults.rh_thres
    rhc_cevap: Float = NamelistDefaults.rhc_cevap
    rhc_revap: Float = NamelistDefaults.rhc_revap
    f_dq_p: Float = NamelistDefaults.f_dq_p
    f_dq_m: Float = NamelistDefaults.f_dq_m
    fi2s_fac: Float = NamelistDefaults.fi2s_fac
    fi2g_fac: Float = NamelistDefaults.fi2g_fac
    fs2g_fac: Float = NamelistDefaults.fs2g_fac
    is_fac: Float = NamelistDefaults.is_fac
    ss_fac: Float = NamelistDefaults.ss_fac
    gs_fac: Float = NamelistDefaults.gs_fac
    rh_fac_evap: Float = NamelistDefaults.rh_fac_evap
    rh_fac_cond: Float = NamelistDefaults.rh_fac_cond
    sed_fac: Float = NamelistDefaults.sed_fac
    rh_inc: Float = NamelistDefaults.rh_inc
    rh_inr: Float = NamelistDefaults.rh_inr
    # rh_ins: Any
    rthresh: Float = NamelistDefaults.rthresh
    sedi_transport: Bool = NamelistDefaults.sedi_transport
    # use_ccn: Any
    use_ppm: Bool = NamelistDefaults.use_ppm
    use_rhc_cevap: Bool = NamelistDefaults.use_rhc_cevap
    use_rhc_revap: Bool = NamelistDefaults.use_rhc_revap
    vw_max: Float = NamelistDefaults.vw_max
    vg_max: Float = NamelistDefaults.vg_max
    vi_max: Float = NamelistDefaults.vi_max
    vr_max: Float = NamelistDefaults.vr_max
    vs_max: Float = NamelistDefaults.vs_max
    z_slope_ice: Bool = NamelistDefaults.z_slope_ice
    z_slope_liq: Bool = NamelistDefaults.z_slope_liq
    tice: Float = NamelistDefaults.tice
    tice_mlt: Float = NamelistDefaults.tice_mlt
    alin: Float = NamelistDefaults.alin
    alinw: Float = NamelistDefaults.alinw
    alini: Float = NamelistDefaults.alini
    alinr: Float = NamelistDefaults.alinr
    alins: Float = NamelistDefaults.alins
    aling: Float = NamelistDefaults.aling
    alinh: Float = NamelistDefaults.alinh
    blinw: Float = NamelistDefaults.blinw
    blini: Float = NamelistDefaults.blini
    blinr: Float = NamelistDefaults.blinr
    blins: Float = NamelistDefaults.blins
    bling: Float = NamelistDefaults.bling
    blinh: Float = NamelistDefaults.blinh
    clin: Float = NamelistDefaults.clin
    n0w_sig: Float = NamelistDefaults.n0w_sig
    n0i_sig: Float = NamelistDefaults.n0i_sig
    n0r_sig: Float = NamelistDefaults.n0r_sig
    n0s_sig: Float = NamelistDefaults.n0s_sig
    n0g_sig: Float = NamelistDefaults.n0g_sig
    n0h_sig: Float = NamelistDefaults.n0h_sig
    n0w_exp: Float = NamelistDefaults.n0w_exp
    n0i_exp: Float = NamelistDefaults.n0i_exp
    n0r_exp: Float = NamelistDefaults.n0r_exp
    n0s_exp: Float = NamelistDefaults.n0s_exp
    n0g_exp: Float = NamelistDefaults.n0g_exp
    n0h_exp: Float = NamelistDefaults.n0h_exp
    muw: Float = NamelistDefaults.muw
    mui: Float = NamelistDefaults.mui
    mur: Float = NamelistDefaults.mur
    mus: Float = NamelistDefaults.mus
    mug: Float = NamelistDefaults.mug
    muh: Float = NamelistDefaults.muh
    cfflag: Int = NamelistDefaults.cfflag
    irain_f: Int = NamelistDefaults.irain_f
    inflag: Int = NamelistDefaults.inflag
    igflag: Int = NamelistDefaults.igflag
    ifflag: Int = NamelistDefaults.ifflag
    sedflag: Int = NamelistDefaults.sedflag
    vdiffflag: Int = NamelistDefaults.vdiffflag
    do_mp_table_emulation: Bool = NamelistDefaults.do_mp_table_emulation

    namelist_override: Optional[str] = None

    def __post_init__(self):
        if self.schemes is None:
            self.schemes = DEFAULT_SCHEMES
        package_schemes = []
        for scheme in self.schemes:
            if scheme not in PHYSICS_PACKAGES:
                raise NotImplementedError(f"{scheme} physics scheme not implemented")
            package_schemes.append(PHYSICS_PACKAGES[scheme])
        self.schemes = package_schemes
        if self.namelist_override is not None:
            try:
                f90_nml = f90nml.read(self.namelist_override)
            except FileNotFoundError:
                print(f"{self.namelist_override} does not exist")
            physics_config = self.from_f90nml(f90_nml)
            for var in physics_config.__dict__.keys():
                setattr(self, var, physics_config.__dict__[var])

    @classmethod
    def from_f90nml(self, f90_namelist: f90nml.Namelist) -> "PhysicsConfig":
        namelist = Namelist.from_f90nml(f90_namelist)
        return self.from_namelist(namelist)

    @classmethod
    def from_namelist(cls, namelist: Namelist) -> "PhysicsConfig":
        return cls(
            dt_atmos=namelist.dt_atmos,
            hydrostatic=namelist.hydrostatic,
            npx=namelist.npx,
            npy=namelist.npy,
            npz=namelist.npz,
            nwat=namelist.nwat,
            do_qa=namelist.do_qa,
            c_cracw=namelist.c_cracw,
            c_paut=namelist.c_paut,
            c_pracs=namelist.c_pracs,
            c_psacr=namelist.c_psacr,
            c_pgacr=namelist.c_pgacr,
            c_pgacs=namelist.c_pgacs,
            c_psacw=namelist.c_psacw,
            c_psaci=namelist.c_psaci,
            c_pracw=namelist.c_pracw,
            c_praci=namelist.c_praci,
            c_pgacw=namelist.c_pgacw,
            c_pgaci=namelist.c_pgaci,
            ccn_l=namelist.ccn_l,
            ccn_o=namelist.ccn_o,
            const_vg=namelist.const_vg,
            const_vi=namelist.const_vi,
            const_vr=namelist.const_vr,
            const_vw=namelist.const_vw,
            const_vs=namelist.const_vs,
            vw_fac=namelist.vw_fac,
            vs_fac=namelist.vs_fac,
            vg_fac=namelist.vg_fac,
            vi_fac=namelist.vi_fac,
            vr_fac=namelist.vr_fac,
            de_ice=namelist.de_ice,
            layout=namelist.layout,
            tau_r2g=namelist.tau_r2g,
            tau_smlt=namelist.tau_smlt,
            tau_gmlt=namelist.tau_gmlt,
            tau_g2r=namelist.tau_g2r,
            tau_imlt=namelist.tau_imlt,
            tau_i2s=namelist.tau_i2s,
            tau_l2r=namelist.tau_l2r,
            tau_g2v=namelist.tau_g2v,
            tau_v2g=namelist.tau_v2g,
            tau_wbf=namelist.tau_wbf,
            ql_mlt=namelist.ql_mlt,
            ql0_max=namelist.ql0_max,
            qs_mlt=namelist.qs_mlt,
            t_min=namelist.t_min,
            t_sub=namelist.t_sub,
            qi_gen=namelist.qi_gen,
            qi_lim=namelist.qi_lim,
            qi0_max=namelist.qi0_max,
            rad_snow=namelist.rad_snow,
            rad_graupel=namelist.rad_graupel,
            rad_rain=namelist.rad_rain,
            do_cld_adj=namelist.do_cld_adj,
            dw_ocean=namelist.dw_ocean,
            dw_land=namelist.dw_land,
            icloud_f=namelist.icloud_f,
            cld_min=namelist.cld_min,
            tau_l2v=namelist.tau_l2v,
            tau_v2l=namelist.tau_v2l,
            tau_revp=namelist.tau_revp,
            c2l_ord=namelist.c2l_ord,
            do_sedi_heat=namelist.do_sedi_heat,
            do_sedi_melt=namelist.do_sedi_melt,
            do_sedi_uv=namelist.do_sedi_uv,
            do_sedi_w=namelist.do_sedi_w,
            fast_sat_adj=namelist.fast_sat_adj,
            qc_crt=namelist.qc_crt,
            fix_negative=namelist.fix_negative,
            do_cond_timescale=namelist.do_cond_timescale,
            do_evap_timescale=namelist.do_evap_timescale,
            do_hail=namelist.do_hail,
            consv_checker=namelist.consv_checker,
            do_warm_rain_mp=namelist.do_warm_rain_mp,
            do_wbf=namelist.do_wbf,
            do_psd_water_fall=namelist.do_psd_water_fall,
            do_psd_ice_fall=namelist.do_psd_ice_fall,
            do_psd_water_num=namelist.do_psd_water_num,
            do_psd_ice_num=namelist.do_psd_ice_num,
            do_new_acc_water=namelist.do_new_acc_water,
            do_new_acc_ice=namelist.do_new_acc_ice,
            cp_heating=namelist.cp_heating,
            fast_fr_mlt=namelist.fast_fr_mlt,
            fast_dep_sub=namelist.fast_dep_sub,
            delay_cond_evap=namelist.delay_cond_evap,
            nconds=namelist.nconds,
            mp_time=namelist.mp_time,
            prog_ccn=namelist.prog_ccn,
            qi0_crt=namelist.qi0_crt,
            qs0_crt=namelist.qs0_crt,
            xr_a=namelist.xr_a,
            xr_b=namelist.xr_b,
            xr_c=namelist.xr_c,
            te_err=namelist.te_err,
            tw_err=namelist.tw_err,
            rh_thres=namelist.rh_thres,
            rhc_cevap=namelist.rhc_cevap,
            rhc_revap=namelist.rhc_revap,
            f_dq_p=namelist.f_dq_p,
            f_dq_m=namelist.f_dq_m,
            fi2s_fac=namelist.fi2s_fac,
            fi2g_fac=namelist.fi2g_fac,
            fs2g_fac=namelist.fs2g_fac,
            is_fac=namelist.is_fac,
            ss_fac=namelist.ss_fac,
            gs_fac=namelist.gs_fac,
            rh_fac_evap=namelist.rh_fac_evap,
            rh_fac_cond=namelist.rh_fac_cond,
            sed_fac=namelist.sed_fac,
            rh_inc=namelist.rh_inc,
            rh_inr=namelist.rh_inr,
            rthresh=namelist.rthresh,
            sedi_transport=namelist.sedi_transport,
            use_ppm=namelist.use_ppm,
            use_rhc_cevap=namelist.use_rhc_cevap,
            use_rhc_revap=namelist.use_rhc_revap,
            vw_max=namelist.vw_max,
            vg_max=namelist.vg_max,
            vi_max=namelist.vi_max,
            vr_max=namelist.vr_max,
            vs_max=namelist.vs_max,
            z_slope_ice=namelist.z_slope_ice,
            z_slope_liq=namelist.z_slope_liq,
            tice=namelist.tice,
            tice_mlt=namelist.tice_mlt,
            alin=namelist.alin,
            alinw=namelist.alinw,
            alini=namelist.alini,
            alinr=namelist.alinr,
            alins=namelist.alins,
            aling=namelist.aling,
            alinh=namelist.alinh,
            blinw=namelist.blinw,
            blini=namelist.blini,
            blinr=namelist.blinr,
            blins=namelist.blins,
            bling=namelist.bling,
            blinh=namelist.blinh,
            clin=namelist.clin,
            n0w_sig=namelist.n0w_sig,
            n0i_sig=namelist.n0i_sig,
            n0r_sig=namelist.n0r_sig,
            n0s_sig=namelist.n0s_sig,
            n0g_sig=namelist.n0g_sig,
            n0h_sig=namelist.n0h_sig,
            n0w_exp=namelist.n0w_exp,
            n0i_exp=namelist.n0i_exp,
            n0r_exp=namelist.n0r_exp,
            n0s_exp=namelist.n0s_exp,
            n0g_exp=namelist.n0g_exp,
            n0h_exp=namelist.n0h_exp,
            muw=namelist.muw,
            mui=namelist.mui,
            mur=namelist.mur,
            mus=namelist.mus,
            mug=namelist.mug,
            muh=namelist.muh,
            cfflag=namelist.cfflag,
            irain_f=namelist.irain_f,
            inflag=namelist.inflag,
            igflag=namelist.igflag,
            ifflag=namelist.ifflag,
            sedflag=namelist.sedflag,
            vdiffflag=namelist.vdiffflag,
            ntimes=namelist.ntimes,
            do_inline_mp=namelist.do_inline_mp,
            do_mp_table_emulation=namelist.do_mp_table_emulation,
        )

    @property
    def microphysics(self) -> MicroPhysicsConfig:
        return MicroPhysicsConfig(
            dt_full=self.dt_atmos,
            ntimes=self.ntimes,
            nconds=self.nconds,
            hydrostatic=self.hydrostatic,
            npx=self.npx,
            npy=self.npy,
            npz=self.npz,
            nwat=self.nwat,
            do_qa=self.do_qa,
            do_inline_mp=self.do_inline_mp,
            c_cracw=self.c_cracw,
            c_paut=self.c_paut,
            c_pracs=self.c_pracs,
            c_psacr=self.c_psacr,
            c_pgacr=self.c_pgacr,
            c_pgacs=self.c_pgacs,
            c_psacw=self.c_psacw,
            c_psaci=self.c_psaci,
            c_pracw=self.c_pracw,
            c_praci=self.c_praci,
            c_pgacw=self.c_pgacw,
            c_pgaci=self.c_pgaci,
            ccn_l=self.ccn_l,
            ccn_o=self.ccn_o,
            const_vg=self.const_vg,
            const_vi=self.const_vi,
            const_vr=self.const_vr,
            const_vw=self.const_vw,
            const_vs=self.const_vs,
            vw_fac=self.vw_fac,
            vs_fac=self.vs_fac,
            vg_fac=self.vg_fac,
            vi_fac=self.vi_fac,
            vr_fac=self.vr_fac,
            de_ice=self.de_ice,
            layout=self.layout,
            tau_r2g=self.tau_r2g,
            tau_smlt=self.tau_smlt,
            tau_gmlt=self.tau_gmlt,
            tau_g2r=self.tau_g2r,
            tau_imlt=self.tau_imlt,
            tau_i2s=self.tau_i2s,
            tau_l2r=self.tau_l2r,
            tau_g2v=self.tau_g2v,
            tau_v2g=self.tau_v2g,
            ql_mlt=self.ql_mlt,
            ql0_max=self.ql0_max,
            qs_mlt=self.qs_mlt,
            t_min=self.t_min,
            t_sub=self.t_sub,
            qi_gen=self.qi_gen,
            qi_lim=self.qi_lim,
            qi0_max=self.qi0_max,
            rad_snow=self.rad_snow,
            rad_graupel=self.rad_graupel,
            rad_rain=self.rad_rain,
            do_cld_adj=self.do_cld_adj,
            dw_ocean=self.dw_ocean,
            dw_land=self.dw_land,
            icloud_f=self.icloud_f,
            cld_min=self.cld_min,
            tau_l2v=self.tau_l2v,
            tau_v2l=self.tau_v2l,
            tau_revp=self.tau_revp,
            tau_wbf=self.tau_wbf,
            c2l_ord=self.c2l_ord,
            do_sedi_heat=self.do_sedi_heat,
            do_sedi_melt=self.do_sedi_melt,
            do_sedi_uv=self.do_sedi_uv,
            do_sedi_w=self.do_sedi_w,
            fast_sat_adj=self.fast_sat_adj,
            qc_crt=self.qc_crt,
            fix_negative=self.fix_negative,
            do_cond_timescale=self.do_cond_timescale,
            do_evap_timescale=self.do_evap_timescale,
            do_hail=self.do_hail,
            consv_checker=self.consv_checker,
            do_warm_rain_mp=self.do_warm_rain_mp,
            do_wbf=self.do_wbf,
            do_psd_water_fall=self.do_psd_water_fall,
            do_psd_ice_fall=self.do_psd_ice_fall,
            do_psd_water_num=self.do_psd_water_num,
            do_psd_ice_num=self.do_psd_ice_num,
            do_new_acc_water=self.do_new_acc_water,
            do_new_acc_ice=self.do_new_acc_ice,
            cp_heating=self.cp_heating,
            fast_fr_mlt=self.fast_fr_mlt,
            fast_dep_sub=self.fast_dep_sub,
            delay_cond_evap=self.delay_cond_evap,
            mp_time=self.mp_time,
            prog_ccn=self.prog_ccn,
            qi0_crt=self.qi0_crt,
            qs0_crt=self.qs0_crt,
            xr_a=self.xr_a,
            xr_b=self.xr_b,
            xr_c=self.xr_c,
            te_err=self.te_err,
            tw_err=self.tw_err,
            rh_thres=self.rh_thres,
            rhc_cevap=self.rhc_cevap,
            rhc_revap=self.rhc_revap,
            f_dq_p=self.f_dq_p,
            f_dq_m=self.f_dq_m,
            fi2s_fac=self.fi2s_fac,
            fi2g_fac=self.fi2g_fac,
            fs2g_fac=self.fs2g_fac,
            is_fac=self.is_fac,
            ss_fac=self.ss_fac,
            gs_fac=self.gs_fac,
            rh_fac_evap=self.rh_fac_evap,
            rh_fac_cond=self.rh_fac_cond,
            sed_fac=self.sed_fac,
            rh_inc=self.rh_inc,
            rh_inr=self.rh_inr,
            rthresh=self.rthresh,
            sedi_transport=self.sedi_transport,
            use_ppm=self.use_ppm,
            use_rhc_cevap=self.use_rhc_cevap,
            use_rhc_revap=self.use_rhc_revap,
            vw_max=self.vw_max,
            vg_max=self.vg_max,
            vi_max=self.vi_max,
            vr_max=self.vr_max,
            vs_max=self.vs_max,
            z_slope_ice=self.z_slope_ice,
            z_slope_liq=self.z_slope_liq,
            tice=self.tice,
            tice_mlt=self.tice_mlt,
            alin=self.alin,
            alinw=self.alinw,
            alini=self.alini,
            alinr=self.alinr,
            alins=self.alins,
            aling=self.aling,
            alinh=self.alinh,
            blinw=self.blinw,
            blini=self.blini,
            blinr=self.blinr,
            blins=self.blins,
            bling=self.bling,
            blinh=self.blinh,
            clin=self.clin,
            n0w_sig=self.n0w_sig,
            n0i_sig=self.n0i_sig,
            n0r_sig=self.n0r_sig,
            n0s_sig=self.n0s_sig,
            n0g_sig=self.n0g_sig,
            n0h_sig=self.n0h_sig,
            n0w_exp=self.n0w_exp,
            n0i_exp=self.n0i_exp,
            n0r_exp=self.n0r_exp,
            n0s_exp=self.n0s_exp,
            n0g_exp=self.n0g_exp,
            n0h_exp=self.n0h_exp,
            muw=self.muw,
            mui=self.mui,
            mur=self.mur,
            mus=self.mus,
            mug=self.mug,
            muh=self.muh,
            cfflag=self.cfflag,
            irain_f=self.irain_f,
            inflag=self.inflag,
            igflag=self.igflag,
            ifflag=self.ifflag,
            sedflag=self.sedflag,
            vdiffflag=self.vdiffflag,
            do_mp_table_emulation=self.do_mp_table_emulation,
        )
