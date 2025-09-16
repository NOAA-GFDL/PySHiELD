import dataclasses
from enum import Enum, unique
from typing import List, Optional, Tuple

import f90nml

import ndsl.namelist as nml
from ndsl import MetaEnumStr
from pyshield.stencils.gfdl_cld_microphysics import (
    GFDLCloudMPConfig,
    GFDLCloudMPDefaults,
)


DEFAULT_INT = 0
DEFAULT_FLOAT = 0.0
DEFAULT_STR = ""
DEFAULT_BOOL = False
DEFAULT_SCHEMES = ["GFS_microphysics"]


@unique
class PHYSICS_PACKAGES(Enum, metaclass=MetaEnumStr):
    GFS_microphysics = "GFS_microphysics"
    GFDL_cloud_microphysics = "GFDL_cloud_microphysics"


@dataclasses.dataclass
class PhysicsConfig:
    dt_atmos: float = DEFAULT_FLOAT
    ntimes: int = GFDLCloudMPDefaults.ntimes
    nconds: int = 1
    hydrostatic: bool = DEFAULT_BOOL
    npx: int = DEFAULT_INT
    npy: int = DEFAULT_INT
    npz: int = DEFAULT_INT
    nwat: int = DEFAULT_INT
    schemes: List = None
    do_qa: bool = DEFAULT_BOOL
    do_inline_mp: bool = GFDLCloudMPDefaults.do_inline_mp
    c_cracw: float = GFDLCloudMPDefaults.c_cracw
    c_paut: float = GFDLCloudMPDefaults.c_paut
    c_pracs: float = GFDLCloudMPDefaults.c_pracs
    c_psacr: float = GFDLCloudMPDefaults.c_psacr
    c_pgacr: float = GFDLCloudMPDefaults.c_pgacr
    c_pgacs: float = GFDLCloudMPDefaults.c_pgacs
    c_psacw: float = GFDLCloudMPDefaults.c_psacw
    c_psaci: float = GFDLCloudMPDefaults.c_psaci
    c_pracw: float = GFDLCloudMPDefaults.c_pracw
    c_praci: float = GFDLCloudMPDefaults.c_praci
    c_pgacw: float = GFDLCloudMPDefaults.c_pgacw
    c_pgaci: float = GFDLCloudMPDefaults.c_pgaci
    ccn_l: float = GFDLCloudMPDefaults.ccn_l
    ccn_o: float = GFDLCloudMPDefaults.ccn_o
    const_vg: bool = GFDLCloudMPDefaults.const_vg
    const_vi: bool = GFDLCloudMPDefaults.const_vi
    const_vr: bool = GFDLCloudMPDefaults.const_vr
    const_vw: bool = GFDLCloudMPDefaults.const_vw
    const_vs: bool = GFDLCloudMPDefaults.const_vs
    vw_fac: float = GFDLCloudMPDefaults.vw_fac
    vs_fac: float = GFDLCloudMPDefaults.vs_fac
    vg_fac: float = GFDLCloudMPDefaults.vg_fac
    vi_fac: float = GFDLCloudMPDefaults.vi_fac
    vr_fac: float = GFDLCloudMPDefaults.vr_fac
    de_ice: bool = GFDLCloudMPDefaults.de_ice
    layout: Tuple[int, int] = nml.GFDLCloudMPDefaults.layout
    # gfdl_cloud_microphys.F90
    tau_r2g: float = GFDLCloudMPDefaults.tau_r2g
    """rain freezing during fast_sat"""
    tau_smlt: float = GFDLCloudMPDefaults.tau_smlt
    """snow melting timescale"""
    tau_gmlt: float = GFDLCloudMPDefaults.tau_gmlt
    """graupel melting timescale"""
    tau_g2r: float = GFDLCloudMPDefaults.tau_g2r
    """graupel melting to rain"""
    tau_imlt: float = GFDLCloudMPDefaults.tau_imlt
    """cloud ice melting"""
    tau_i2s: float = GFDLCloudMPDefaults.tau_i2s
    """cloud ice to snow auto - conversion"""
    tau_l2r: float = GFDLCloudMPDefaults.tau_l2r
    """cloud water to rain auto - conversion"""
    tau_g2v: float = GFDLCloudMPDefaults.tau_g2v
    """graupel sublimation"""
    tau_v2g: float = GFDLCloudMPDefaults.tau_v2g
    """graupel deposition -- make it a slow process"""
    ql_mlt: float = GFDLCloudMPDefaults.ql_mlt
    """max value of cloud water allowed from melted cloud ice"""
    ql0_max: float = GFDLCloudMPDefaults.ql0_max
    """max cloud water value (auto converted to rain)"""
    qs_mlt: float = GFDLCloudMPDefaults.qs_mlt
    """max cloud water due to snow melt"""
    t_min: float = GFDLCloudMPDefaults.t_min
    """minimum temperature to freeze - dry all water vapor (K)"""
    t_sub: float = GFDLCloudMPDefaults.t_sub
    """min temp for sublimation of cloud ice"""
    qi_gen: float = GFDLCloudMPDefaults.qi_gen
    """max cloud ice generation during remapping step"""
    qi_lim: float = GFDLCloudMPDefaults.qi_lim
    """cloud ice limiter to prevent large ice build up"""
    qi0_max: float = GFDLCloudMPDefaults.qi0_max
    """max cloud ice value (by other sources)"""
    rad_snow: bool = GFDLCloudMPDefaults.rad_snow
    """consider snow in cloud fraction calculation"""
    rad_graupel: bool = GFDLCloudMPDefaults.rad_graupel
    """consider graupel in cloud fraction calculation"""
    rad_rain: bool = GFDLCloudMPDefaults.rad_rain
    """consider rain in cloud fraction calculation"""
    do_cld_adj: bool = GFDLCloudMPDefaults.do_cld_adj
    """do cloud fraction adjustment"""
    dw_ocean: float = GFDLCloudMPDefaults.dw_ocean
    """base value for ocean"""
    dw_land: float = GFDLCloudMPDefaults.dw_land
    """base value for subgrid deviation / variability over land
     - cloud scheme 0 - ?
     - 1: old fvgfs gfdl) mp implementation
     - 2: binary cloud scheme (0 / 1)"""
    icloud_f: int = GFDLCloudMPDefaults.icloud_f
    cld_min: float = GFDLCloudMPDefaults.cld_min
    """minimum cloud fraction"""
    tau_l2v: float = GFDLCloudMPDefaults.tau_l2v
    """cloud water to water vapor (evaporation)"""
    tau_v2l: float = GFDLCloudMPDefaults.tau_v2l
    """water vapor to cloud water (condensation)"""
    tau_revp: float = GFDLCloudMPDefaults.tau_revp
    tau_wbf: float = GFDLCloudMPDefaults.tau_wbf
    c2l_ord: int = GFDLCloudMPDefaults.c2l_ord
    do_sedi_heat: bool = GFDLCloudMPDefaults.do_sedi_heat
    do_sedi_melt: bool = GFDLCloudMPDefaults.do_sedi_melt
    do_sedi_uv: bool = GFDLCloudMPDefaults.do_sedi_uv
    do_sedi_w: bool = GFDLCloudMPDefaults.do_sedi_w
    fast_sat_adj: bool = GFDLCloudMPDefaults.fast_sat_adj
    qc_crt: float = GFDLCloudMPDefaults.qc_crt
    fix_negative: bool = GFDLCloudMPDefaults.fix_negative
    do_cond_timescale: bool = GFDLCloudMPDefaults.do_cond_timescale
    do_evap_timescale: bool = True
    do_hail: bool = GFDLCloudMPDefaults.do_hail
    consv_checker: bool = GFDLCloudMPDefaults.consv_checker
    do_warm_rain_mp: bool = GFDLCloudMPDefaults.do_warm_rain_mp
    do_wbf: bool = GFDLCloudMPDefaults.do_wbf
    do_psd_water_fall: bool = GFDLCloudMPDefaults.do_psd_water_fall
    do_psd_ice_fall: bool = GFDLCloudMPDefaults.do_psd_ice_fall
    do_psd_water_num: bool = GFDLCloudMPDefaults.do_psd_water_num
    do_psd_ice_num: bool = GFDLCloudMPDefaults.do_psd_ice_num
    do_new_acc_water: bool = GFDLCloudMPDefaults.do_new_acc_water
    do_new_acc_ice: bool = GFDLCloudMPDefaults.do_new_acc_ice
    cp_heating: bool = GFDLCloudMPDefaults.cp_heating
    fast_fr_mlt: bool = True
    fast_dep_sub: bool = True
    delay_cond_evap: bool = True
    mp_time: float = GFDLCloudMPDefaults.mp_time
    prog_ccn: bool = GFDLCloudMPDefaults.prog_ccn
    qi0_crt: float = GFDLCloudMPDefaults.qi0_crt
    qs0_crt: float = GFDLCloudMPDefaults.qs0_crt
    xr_a: float = GFDLCloudMPDefaults.xr_a
    xr_b: float = GFDLCloudMPDefaults.xr_b
    xr_c: float = GFDLCloudMPDefaults.xr_c
    te_err: float = GFDLCloudMPDefaults.te_err
    tw_err: float = GFDLCloudMPDefaults.tw_err
    rh_thres: float = GFDLCloudMPDefaults.rh_thres
    rhc_cevap: float = GFDLCloudMPDefaults.rhc_cevap
    rhc_revap: float = GFDLCloudMPDefaults.rhc_revap
    f_dq_p: float = GFDLCloudMPDefaults.f_dq_p
    f_dq_m: float = GFDLCloudMPDefaults.f_dq_m
    fi2s_fac: float = GFDLCloudMPDefaults.fi2s_fac
    fi2g_fac: float = GFDLCloudMPDefaults.fi2g_fac
    fs2g_fac: float = GFDLCloudMPDefaults.fs2g_fac
    is_fac: float = GFDLCloudMPDefaults.is_fac
    ss_fac: float = GFDLCloudMPDefaults.ss_fac
    gs_fac: float = GFDLCloudMPDefaults.gs_fac
    rh_fac_evap: float = GFDLCloudMPDefaults.rh_fac_evap
    rh_fac_cond: float = GFDLCloudMPDefaults.rh_fac_cond
    sed_fac: float = GFDLCloudMPDefaults.sed_fac
    rh_inc: float = GFDLCloudMPDefaults.rh_inc
    rh_inr: float = GFDLCloudMPDefaults.rh_inr
    # rh_ins: Any
    rthresh: float = GFDLCloudMPDefaults.rthresh
    sedi_transport: bool = GFDLCloudMPDefaults.sedi_transport
    # use_ccn: Any
    use_ppm: bool = GFDLCloudMPDefaults.use_ppm
    use_rhc_cevap: bool = GFDLCloudMPDefaults.use_rhc_cevap
    use_rhc_revap: bool = GFDLCloudMPDefaults.use_rhc_revap
    vw_max: float = GFDLCloudMPDefaults.vw_max
    vg_max: float = GFDLCloudMPDefaults.vg_max
    vi_max: float = GFDLCloudMPDefaults.vi_max
    vr_max: float = GFDLCloudMPDefaults.vr_max
    vs_max: float = GFDLCloudMPDefaults.vs_max
    z_slope_ice: bool = GFDLCloudMPDefaults.z_slope_ice
    z_slope_liq: bool = GFDLCloudMPDefaults.z_slope_liq
    tice: float = GFDLCloudMPDefaults.tice
    tice_mlt: float = GFDLCloudMPDefaults.tice_mlt
    alin: float = GFDLCloudMPDefaults.alin
    alinw: float = GFDLCloudMPDefaults.alinw
    alini: float = GFDLCloudMPDefaults.alini
    alinr: float = GFDLCloudMPDefaults.alinr
    alins: float = GFDLCloudMPDefaults.alins
    aling: float = GFDLCloudMPDefaults.aling
    alinh: float = GFDLCloudMPDefaults.alinh
    blinw: float = GFDLCloudMPDefaults.blinw
    blini: float = GFDLCloudMPDefaults.blini
    blinr: float = GFDLCloudMPDefaults.blinr
    blins: float = GFDLCloudMPDefaults.blins
    bling: float = GFDLCloudMPDefaults.bling
    blinh: float = GFDLCloudMPDefaults.blinh
    clin: float = GFDLCloudMPDefaults.clin
    n0w_sig: float = GFDLCloudMPDefaults.n0w_sig
    n0i_sig: float = GFDLCloudMPDefaults.n0i_sig
    n0r_sig: float = GFDLCloudMPDefaults.n0r_sig
    n0s_sig: float = GFDLCloudMPDefaults.n0s_sig
    n0g_sig: float = GFDLCloudMPDefaults.n0g_sig
    n0h_sig: float = GFDLCloudMPDefaults.n0h_sig
    n0w_exp: float = GFDLCloudMPDefaults.n0w_exp
    n0i_exp: float = GFDLCloudMPDefaults.n0i_exp
    n0r_exp: float = GFDLCloudMPDefaults.n0r_exp
    n0s_exp: float = GFDLCloudMPDefaults.n0s_exp
    n0g_exp: float = GFDLCloudMPDefaults.n0g_exp
    n0h_exp: float = GFDLCloudMPDefaults.n0h_exp
    muw: float = GFDLCloudMPDefaults.muw
    mui: float = GFDLCloudMPDefaults.mui
    mur: float = GFDLCloudMPDefaults.mur
    mus: float = GFDLCloudMPDefaults.mus
    mug: float = GFDLCloudMPDefaults.mug
    muh: float = GFDLCloudMPDefaults.muh
    cfflag: int = GFDLCloudMPDefaults.cfflag
    irain_f: int = GFDLCloudMPDefaults.irain_f
    inflag: int = GFDLCloudMPDefaults.inflag
    igflag: int = GFDLCloudMPDefaults.igflag
    ifflag: int = GFDLCloudMPDefaults.ifflag
    sedflag: int = GFDLCloudMPDefaults.sedflag
    vdiffflag: int = GFDLCloudMPDefaults.vdiffflag
    do_mp_table_emulation: bool = GFDLCloudMPDefaults.do_mp_table_emulation
    daily_mean: bool = DEFAULT_BOOL
    """flag to replace cosz with daily mean value"""

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
        namelist = nml.Namelist.from_f90nml(f90_namelist)
        return self.from_namelist(namelist)

    @classmethod
    def from_namelist(cls, namelist: nml.Namelist) -> "PhysicsConfig":
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
            c_pgacs=namelist.c_pgacs,
            c_psaci=namelist.c_psaci,
            ccn_l=namelist.ccn_l,
            ccn_o=namelist.ccn_o,
            const_vg=namelist.const_vg,
            const_vi=namelist.const_vi,
            const_vr=namelist.const_vr,
            const_vs=namelist.const_vs,
            vs_fac=namelist.vs_fac,
            vg_fac=namelist.vg_fac,
            vi_fac=namelist.vi_fac,
            vr_fac=namelist.vr_fac,
            de_ice=namelist.de_ice,
            layout=namelist.layout,
            tau_r2g=namelist.tau_r2g,
            tau_smlt=namelist.tau_smlt,
            tau_g2r=namelist.tau_g2r,
            tau_imlt=namelist.tau_imlt,
            tau_i2s=namelist.tau_i2s,
            tau_l2r=namelist.tau_l2r,
            tau_g2v=namelist.tau_g2v,
            tau_v2g=namelist.tau_v2g,
            ql_mlt=namelist.ql_mlt,
            ql0_max=namelist.ql0_max,
            qs_mlt=namelist.qs_mlt,
            t_sub=namelist.t_sub,
            qi_gen=namelist.qi_gen,
            qi_lim=namelist.qi_lim,
            qi0_max=namelist.qi0_max,
            rad_snow=namelist.rad_snow,
            rad_graupel=namelist.rad_graupel,
            rad_rain=namelist.rad_rain,
            dw_ocean=namelist.dw_ocean,
            dw_land=namelist.dw_land,
            icloud_f=namelist.icloud_f,
            cld_min=namelist.cld_min,
            tau_l2v=namelist.tau_l2v,
            tau_v2l=namelist.tau_v2l,
            c2l_ord=namelist.c2l_ord,
            do_sedi_heat=namelist.do_sedi_heat,
            do_sedi_w=namelist.do_sedi_w,
            fast_sat_adj=namelist.fast_sat_adj,
            qc_crt=namelist.qc_crt,
            fix_negative=namelist.fix_negative,
            mp_time=namelist.mp_time,
            prog_ccn=namelist.prog_ccn,
            qi0_crt=namelist.qi0_crt,
            qs0_crt=namelist.qs0_crt,
            rh_inc=namelist.rh_inc,
            rh_inr=namelist.rh_inr,
            rthresh=namelist.rthresh,
            sedi_transport=namelist.sedi_transport,
            use_ppm=namelist.use_ppm,
            vg_max=namelist.vg_max,
            vi_max=namelist.vi_max,
            vr_max=namelist.vr_max,
            vs_max=namelist.vs_max,
            z_slope_ice=namelist.z_slope_ice,
            z_slope_liq=namelist.z_slope_liq,
            tice=namelist.tice,
            alin=namelist.alin,
            clin=namelist.clin,
            irain_f=namelist.irain_f,
            daily_mean=namelist.daily_mean,
        )

    @property
    def microphysics(self) -> GFDLCloudMPConfig:
        return GFDLCloudMPConfig(
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
            t_sub=self.t_sub,
            t_min=self.t_min,
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
