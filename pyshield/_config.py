import dataclasses
import math
from enum import Enum, unique
from typing import List, Optional, Tuple

import f90nml

import ndsl.constants as constants
import ndsl.namelist as nml
import pyshield.constants as physcons
from ndsl import MetaEnumStr


DEFAULT_INT = 0
DEFAULT_FLOAT = 0.0
DEFAULT_STR = ""
DEFAULT_BOOL = False
DEFAULT_SCHEMES = ["GFS_microphysics"]


@unique
class PHYSICS_PACKAGES(Enum, metaclass=MetaEnumStr):
    GFS_microphysics = "GFS_microphysics"
    SHiELD_microphysics = "SHiELD_microphysics"


# Global set of physics namelist defaults
# attached to class for namespacing and static typing
class NamelistDefaults:
    tau_r2g = 900.0
    """rain freezing during fast_sat"""
    tau_smlt = 900.0
    """snow melting timescale"""
    tau_gmlt = 600.0
    """snow melting timescale"""
    tau_g2r = 600.0
    """graupel melting to rain"""
    tau_imlt = 1200.0
    """cloud ice melting"""
    tau_i2s = 1000.0
    """cloud ice to snow auto - conversion"""
    tau_l2r = 900.0
    """cloud water to rain auto - conversion"""
    tau_l2v = 300.0
    """cloud water to water vapor (evaporation)"""
    tau_v2l = 150.0
    """water vapor to cloud water (condensation)"""
    tau_revp = 0.0
    """rain evaporation time scale (s)"""
    tau_g2v = 1200.0
    """graupel sublimation"""
    tau_v2g = 21600.0
    """graupel deposition -- make it a slow process"""
    tau_wbf = 300.0
    """wegener bergeron findeisen timescale"""
    sat_adj0 = 0.90
    """adjustment factor (0: no, 1: full) during fast_sat_adj"""
    ql_gen = 1.0e-3
    """max new cloud water during remapping step if fast_sat_adj = .t."""
    ql_mlt = 2.0e-3
    """max value of cloud water allowed from melted cloud ice"""
    qs_mlt = 1.0e-6
    """max cloud water due to snow melt"""
    ql0_max = 2.0e-3
    """max cloud water value (auto converted to rain)"""
    t_min = 178.0
    """minimum temperature to freeze - dry all water vapor (K)"""
    t_sub = 184.0
    """min temp for sublimation of cloud ice"""
    qi_gen = 1.82e-6
    """max cloud ice generation during remapping step"""
    qi_lim = 1.0
    """cloud ice limiter to prevent large ice build up"""
    qi0_max = 1.0e-4
    """max cloud ice value (by other sources)"""
    rad_snow = True
    """consider snow in cloud fraciton calculation"""
    rad_rain = True
    """consider rain in cloud fraction calculation"""
    rad_graupel = True
    """consider graupel in cloud fraction calculation"""
    do_cld_adj = False
    """do cloud fraction adjustment"""
    tintqs = False
    """use temperature in the saturation mixing in PDF"""
    dw_ocean = 0.10
    """base value for ocean"""
    dw_land = 0.20
    """
    base value for subgrid deviation / variability over land
     - cloud scheme 0 - ?
     - 1: old fvgfs gfdl) mp implementation
     - 2: binary cloud scheme (0 / 1)
    """
    icloud_f = 0
    """
    GFDL cloud scheme
     - 0: subgrid variability based scheme
     - 1: same as 0, but for old fvgfs implementation
     - 2: binary cloud scheme
     - 3: extension of 0
    """
    cld_min = 0.05
    """minimum cloud fraction"""
    c2l_ord = 4
    regional = False
    m_split = 0
    convert_ke = False
    breed_vortex_inline = False
    use_old_omega = True
    use_logp = False
    rf_fast = False
    p_ref = 1e5
    """Surface pressure used to construct a horizontally-uniform reference"""
    adiabatic = False
    nf_omega = 1
    fv_sg_adj = -1
    n_sponge = 1
    fast_sat_adj = True
    qc_crt = 5.0e-8
    """Minimum condensate mixing ratio to allow partial cloudiness"""
    c_cracw = 0.8
    """Rain accretion efficiency"""
    c_paut = 0.55
    """Autoconversion cloud water to rain (use 0.5 to reduce autoconversion)"""
    c_pracs = 1.0
    """snow to rain accretion efficiency"""
    c_psacr = 1.0
    """rain to snow accretion efficiency"""
    c_pgacr = 1.0
    """rain to graupel accretion efficiency"""
    c_pgacs = 0.01
    """Snow to graupel "accretion" eff. (was 0.1 in zetac)"""
    c_psacw = 1.0
    """Cloud water to snow accretion efficiency"""
    c_psaci = 0.05
    """Accretion: cloud ice to snow (was 0.1 in zetac)"""
    c_pracw = 0.8
    """Cloud water to rain accretion efficiency"""
    c_praci = 1.0
    """Cloud ice to rain accretion efficiency"""
    c_pgacw = 1.0
    """Cloud water to graupel accretion efficiency"""
    c_pgaci = 0.05
    """Cloud ice to graupel accretion efficiency (was 0.1 in ZETAC)"""
    ccn_l = 270.0
    """CCN over land (cm^-3)"""
    ccn_o = 90.0
    """CCN over ocean (cm^-3)"""
    use_rhc_cevap = False
    """cap of rh for cloud water evaporation"""
    use_rhc_revap = False
    """cap of rh for rain evaporation"""
    const_vw = False
    """Fall velocity tuning constant of cloud water"""
    const_vg = False
    """Fall velocity tuning constant of graupel"""
    const_vi = False
    """Fall velocity tuning constant of ice"""
    const_vr = False
    """Fall velocity tuning constant of rain water"""
    const_vs = False
    """Fall velocity tuning constant of snow"""
    is_fac = 0.2
    """Cloud ice sublimation temperature factor"""
    ss_fac = 0.2
    """Snow sublimation temperature factor"""
    gs_fac = 0.2
    """Graupel sublimation temperature factor"""
    rh_fac_evap = 10.0
    """cloud water evaporation relative humidity factor"""
    rh_fac_cond = 10.0
    """cloud water condensation relative humidity factor"""
    sed_fac = 1.0
    """coefficient for sedimentation fall,
    scale from 1.0 (implicit) to 0.0 (lagrangian)"""
    xr_a = 0.25
    """p value in Xu and Randall (1996)"""
    xr_b = 100.0
    """alpha_0 value in Xu and Randall (1996)"""
    xr_c = 0.49
    """gamma value in Xu and Randall (1996)"""
    te_err = 1.0e-5
    """64bit: 1.e-14, 32bit: 1.e-7; turn off to save computer time"""
    tw_err = 1.0e-8
    """64bit: 1.e-14, 32bit: 1.e-7; turn off to save computer time"""
    rh_thres = 0.75
    """minimum relative humidity for cloud fraction"""
    rhc_cevap = 0.85
    """maximum relative humidity for cloud water evaporation"""
    rhc_revap = 0.85
    """maximum relative humidity for rain evaporation"""
    f_dq_p = 1.0
    """cloud fraction adjustment for supersaturation"""
    f_dq_m = 1.0
    """cloud fraction adjustment for undersaturation"""
    fi2s_fac = 1.0
    """maximum sink of cloud ice to form snow: 0-1"""
    fi2g_fac = 1.0
    """maximum sink of cloud ice to form graupel: 0-1"""
    fs2g_fac = 1.0
    """maximum sink of snow to form graupel: 0-1"""
    vw_fac = 1.0
    vi_fac = 1.0
    """if const_vi: 1/3"""
    vs_fac = 1.0
    """if const_vs: 1."""
    vg_fac = 1.0
    """if const_vg: 2."""
    vr_fac = 1.0
    """if const_vr: 4."""
    de_ice = False
    """To prevent excessive build-up of cloud ice from external sources"""
    do_qa = True
    """Do inline cloud fraction"""
    do_sedi_heat = True
    """Transport of heat in sedimentation"""
    do_sedi_melt = True
    """Melt cloud ice, snow, and graupel during sedimentation"""
    do_sedi_uv = True
    """Transport of horizontal momentum in sedimentation"""
    do_sedi_w = True
    """Transport of vertical motion in sedimentation"""
    fix_negative = True
    """Fix negative water species"""
    do_cond_timescale = False
    """Whether to apply a timescale to condensation"""
    do_hail = False
    """Use hail parameters instead of graupel"""
    consv_checker = False
    """Turn on energy and water conservation check in microphysics"""
    do_warm_rain_mp = False
    """Do only warm rain microphysics"""
    do_wbf = False
    """Do Wegener Bergeron Findeisen process"""
    do_psd_water_fall = False
    """Calculate cloud water terminal velocity based on PSD"""
    do_psd_ice_fall = False
    """Calculate cloud ice terminal velocity based on PSD"""
    do_psd_water_num = False
    """Calculate cloud water number concentration based on PSD"""
    do_psd_ice_num = False
    """Calculate cloud ice number concentration based on PSD"""
    do_new_acc_water = False
    """Perform the new accretion for cloud water"""
    do_new_acc_ice = False
    """Perform the new accretion for cloud water"""
    cp_heating = False
    """update temperature based on constant pressure"""
    mono_prof = False
    """Perform terminal fall with mono ppm scheme"""
    mp_time = 150.0
    """Maximum microphysics timestep (sec)"""
    prog_ccn = False
    """Do prognostic ccn (yi ming's method)"""
    qi0_crt = 1.0e-04
    """Cloud ice to snow autoconversion threshold"""
    qs0_crt = 1.0e-3
    """Snow to graupel density threshold (0.6e-3 in purdue lin scheme)"""
    rh_inc = 0.25
    """RH increment for complete evaporation of cloud water and cloud ice"""
    rh_inr = 0.25
    """RH increment for minimum evaporation of rain"""
    rthresh = 10.0e-6
    """Critical cloud drop radius (micrometers)"""
    sedi_transport = True
    """Transport of momentum in sedimentation"""
    use_ppm = False
    """Use ppm fall scheme"""
    vw_max = 0.01
    """Maximum fall speed for cloud water"""
    vg_max = 8.0
    """Maximum fall speed for graupel"""
    vi_max = 0.5
    """Maximum fall speed for ice"""
    vr_max = 12.0
    """Maximum fall speed for rain"""
    vs_max = 5.0
    """Maximum fall speed for snow"""
    z_slope_ice = True
    """Use linear mono slope for autoconversions"""
    z_slope_liq = True
    """Use linear mono slope for autoconversions"""
    tice = 273.16
    """set tice = 165. to turn off ice - phase phys (kessler emulator)"""
    tice_mlt = 273.16
    """Can set ice melting temperature to 268
    based on observation (Kay et al. 2016) (K)"""
    alin = 842.0
    """'a' in lin1983"""
    alinw = 3.0e7
    """'a' in Lin et al. (1983) for cloud water (Ikawa and Saito 1990)"""
    alini = 7.0e2
    """'a' in Lin et al. (1983) for cloud ice (Ikawa and Saita 1990)"""
    alinr = 842.0
    """'a' in Lin et al. (1983) for rain (Liu and Orville 1969)"""
    alins = 4.8
    """'a' in Lin et al. (1983) for snow (straka 2009)"""
    aling = 1.0
    """'a' in Lin et al. (1983) for graupel (Pruppacher and Klett 2010)"""
    alinh = 1.0
    """'a' in Lin et al. (1983) for hail (Pruppacher and Klett 2010)"""
    blinw = 2.0
    """'b' in Lin et al. (1983) for cloud water (Ikawa and Saito 1990)"""
    blini = 1.0
    """'b' in Lin et al. (1983) for cloud ice (Ikawa and Saita 1990)"""
    blinr = 0.8
    """'b' in Lin et al. (1983) for rain (Liu and Orville 1969)"""
    blins = 0.25
    """'b' in Lin et al. (1983) for snow (straka 2009)"""
    bling = 0.5
    """'b' in Lin et al. (1983) for graupel (Pruppacher and Klett 2010)"""
    blinh = 0.5
    """'b' in Lin et al. (1983) for hail (Pruppacher and Klett 2010)"""
    clin = 4.8
    """'c' in lin 1983, 4.8 -- > 6. (to ehance ql -- > qs)"""
    ntimes = 1
    """Number of cloud microphysics sub cycles"""
    do_inline_mp = False
    """Whether the microphsyics is called inside of the dycore"""
    do_mp_table_emulation = False
    """Whether lookup tables should be emulated instead of
    directly calculating values in microphysics, useful for validation"""
    n0w_sig = 1.1
    """cwater significand (Lin et al. 1983) (m^-4) (Martin et al. 1994)"""
    n0i_sig = 1.3
    """cice significand (Lin et al. 1983) (m^-4) (McFarquhar et al. 2015)"""
    n0r_sig = 8.0
    """rain significand (Lin et al. 1983) (m^-4) (Marshall and Palmer 1948)"""
    n0s_sig = 3.0
    """snow significand (Lin et al. 1983) (m^-4) (Gunn and Marshall 1958)"""
    n0g_sig = 4.0
    """graupel significand (Rutledge and Hobbs 1984) (m^-4) (Houze et al. 1979)"""
    n0h_sig = 4.0
    """hail significand (Lin et al. 1983) (m^-4) (Federer and Waldvogel 1975)"""
    n0w_exp = 41
    """cwater exponent (Lin et al. 1983) (m^-4) (Martin et al. 1994)"""
    n0i_exp = 18
    """cice exponent (Lin et al. 1983) (m^-4) (McFarquhar et al. 2015)"""
    n0r_exp = 6
    """rain exponent (Lin et al. 1983) (m^-4) (Marshall and Palmer 1948)"""
    n0s_exp = 6
    """snow exponent (Lin et al. 1983) (m^-4) (Gunn and Marshall 1958)"""
    n0g_exp = 6
    """graupel exponent (Rutledge and Hobbs 1984) (m^-4) (Houze et al. 1979)"""
    n0h_exp = 4
    """hail exponent (Lin et al. 1983) (m^-4) (Federer and Waldvogel 1975)"""
    muw = 6.0
    """shape parameter of cloud water in Gamma distribution (Martin et al. 1994)"""
    mui = 3.35
    """Gamma shape parameter of cloud ice (McFarquhar et al. 2015)"""
    mur = 1.0
    """shape parameter of rain in Gamma distribution (Marshall and Palmer 1948)"""
    mus = 1.0
    """shape parameter of snow in Gamma distribution (Gunn and Marshall 1958)"""
    mug = 1.0
    """shape parameter of graupel in Gamma distribution (Houze et al. 1979)"""
    muh = 1.0
    """shape parameter of hail in Gamma distribution (Federer and Waldvogel 1975)"""
    cfflag = 1
    """
    cloud fraction scheme
     - 1: GFDL cloud scheme
     - 2: Xu and Randall (1996)
     - 3: Park et al. (2016)
     - 4: Gultepe and Isaac (2007)
    """
    irain_f = 0
    """
    cloud water to rain auto conversion scheme
     - 0: subgrid variability based scheme
     - 1: no subgrid varaibility
    """
    inflag = 1
    """
    Ice nucleation scheme:
     - 1: Hong et al. (2004)
     - 2: Meyers et al. (1992)
     - 3: Meyers et al. (1992)
     - 4: Cooper (1986)
     - 5: Fletcher (1962)
    """
    igflag = 3
    """
    Ice generation scheme
     - 1: WSM6
     - 2: WSM6 with 0 at 0 C
     - 3: WSM6 with 0 at 0 C and fixed value at - 10 C
     - 4: combination of 1 and 3
     """
    ifflag = 1
    """
    Ice fall scheme
     - 1: Deng and Mace (2008)
     - 2: Heymsfield and Donner (1990)
    """
    sedflag = 1
    """
    sedimentation scheme
     - 1: implicit scheme
     - 2: explicit scheme
     - 3: lagrangian scheme
     - 4: combined implicit and lagrangian scheme
    """
    vdiffflag = 1
    """
    wind difference scheme in accretion
     - 1: Wisner et al. (1972)
     - 2: Mizuno (1990)
     - 3: Murakami (1990)
    """


@dataclasses.dataclass
class AdjustNegativeTracerConfig:
    ntimes: float
    c1_ice: float
    c1_liq: float
    c1_vap: float
    d1_ice: float
    d1_vap: float
    li00: float
    li20: float
    lv00: float
    t_wfr: float


@dataclasses.dataclass
class FastMPConfig:
    do_warm_rain_mp: bool
    do_wbf: bool
    c1_vap: float
    c1_liq: float
    c1_ice: float
    lv00: float
    li00: float
    li20: float
    d1_vap: float
    d1_ice: float
    t_wfr: float
    ql_mlt: float
    qs_mlt: float
    tau_imlt: float
    tice_mlt: float
    do_cond_timescale: bool
    do_evap_timescale: bool
    do_hail: bool
    rh_fac_evap: float
    rh_fac_cond: float
    rhc_cevap: float
    tau_l2v: float
    tau_v2l: float
    tau_r2g: float
    tau_smlt: float
    tau_gmlt: float
    tau_l2r: float
    use_rhc_cevap: bool
    qi0_crt: float
    qi0_max: float
    ql0_max: float
    tau_wbf: float
    do_psd_water_num: bool
    do_psd_ice_num: bool
    muw: float
    mui: float
    mur: float
    mus: float
    mug: float
    muh: float
    pcaw: float
    pcbw: float
    pcai: float
    pcbi: float
    prog_ccn: float
    inflag: int
    igflag: int
    qi_lim: float
    t_sub: float
    is_fac: float
    tau_i2s: float
    fast_fr_mlt: bool
    fast_dep_sub: bool
    delay_cond_evap: bool
    nconds: int


@dataclasses.dataclass
class MicroPhysicsConfig:
    dt_full: float
    dt_split: float = dataclasses.field(init=False)
    ntimes: int
    nconds: int
    hydrostatic: bool
    npx: int
    npy: int
    npz: int
    nwat: int
    do_qa: bool
    do_inline_mp: bool
    c_cracw: float
    c_paut: float
    c_pracs: float
    c_psacr: float
    c_pgacr: float
    c_pgacs: float
    c_psacw: float
    c_psaci: float
    c_pracw: float
    c_praci: float
    c_pgacw: float
    c_pgaci: float
    ccn_l: float
    ccn_o: float
    const_vg: bool
    const_vi: bool
    const_vr: bool
    const_vw: bool
    const_vs: bool
    vw_fac: float
    vs_fac: float
    vg_fac: float
    vi_fac: float
    vr_fac: float
    de_ice: bool
    layout: Tuple[int, int]
    # gfdl_cloud_microphys.F90
    tau_r2g: float
    tau_smlt: float
    tau_gmlt: float
    tau_g2r: float
    tau_imlt: float
    tau_i2s: float
    tau_l2r: float
    tau_g2v: float
    tau_v2g: float
    ql_mlt: float
    ql0_max: float
    qs_mlt: float
    t_sub: float
    t_min: float
    qi_gen: float
    qi_lim: float
    qi0_max: float
    rad_snow: bool
    rad_graupel: bool
    rad_rain: bool
    do_cld_adj: bool
    dw_ocean: float
    dw_land: float
    icloud_f: int
    cld_min: float
    tau_l2v: float
    tau_v2l: float
    tau_revp: float
    tau_wbf: float
    c2l_ord: int
    do_sedi_heat: bool
    do_sedi_melt: bool
    do_sedi_uv: bool
    do_sedi_w: bool
    fast_sat_adj: bool
    qc_crt: float
    fix_negative: bool
    do_cond_timescale: bool
    do_evap_timescale: bool
    do_hail: bool
    consv_checker: bool
    do_warm_rain_mp: bool
    do_wbf: bool
    do_psd_water_fall: bool
    do_psd_ice_fall: bool
    do_psd_water_num: bool
    do_psd_ice_num: bool
    do_new_acc_water: bool
    do_new_acc_ice: bool
    cp_heating: bool
    fast_fr_mlt: bool
    fast_dep_sub: bool
    delay_cond_evap: bool
    mp_time: float
    prog_ccn: bool
    qi0_crt: float
    qs0_crt: float
    xr_a: float
    xr_b: float
    xr_c: float
    te_err: float
    tw_err: float
    rh_thres: float
    rhc_cevap: float
    rhc_revap: float
    f_dq_p: float
    f_dq_m: float
    fi2s_fac: float
    fi2g_fac: float
    fs2g_fac: float
    is_fac: float
    ss_fac: float
    gs_fac: float
    rh_fac_evap: float
    rh_fac_cond: float
    sed_fac: float
    rh_inc: float
    rh_inr: float
    # rh_ins: Any
    rthresh: float
    sedi_transport: bool
    # use_ccn: Any
    use_ppm: bool
    use_rhc_cevap: bool
    use_rhc_revap: bool
    vw_max: float
    vg_max: float
    vi_max: float
    vr_max: float
    vs_max: float
    z_slope_ice: bool
    z_slope_liq: bool
    tice: float
    tice_mlt: float
    alin: float
    alinw: float
    alini: float
    alinr: float
    alins: float
    aling: float
    alinh: float
    blinw: float
    blini: float
    blinr: float
    blins: float
    bling: float
    blinh: float
    clin: float
    n0w_sig: float
    n0i_sig: float
    n0r_sig: float
    n0s_sig: float
    n0g_sig: float
    n0h_sig: float
    n0w_exp: float
    n0i_exp: float
    n0r_exp: float
    n0s_exp: float
    n0g_exp: float
    n0h_exp: float
    muw: float
    mui: float
    mur: float
    mus: float
    mug: float
    muh: float
    cfflag: float
    irain_f: int
    inflag: int
    igflag: int
    ifflag: int
    sedflag: int
    vdiffflag: int
    do_mp_table_emulation: bool
    c_air: float = dataclasses.field(init=False)
    c_vap: float = dataclasses.field(init=False)
    d0_vap: float = dataclasses.field(init=False)
    lv00: float = dataclasses.field(init=False)
    li00: float = dataclasses.field(init=False)
    li20: float = dataclasses.field(init=False)
    d1_vap: float = dataclasses.field(init=False)
    d1_ice: float = dataclasses.field(init=False)
    c1_vap: float = dataclasses.field(init=False)
    c1_liq: float = dataclasses.field(init=False)
    c1_ice: float = dataclasses.field(init=False)
    n_min: int = dataclasses.field(init=False)
    delt: float = dataclasses.field(init=False)
    esbasw: float = dataclasses.field(init=False)
    tbasw: float = dataclasses.field(init=False)
    esbasi: float = dataclasses.field(init=False)
    tmin: float = dataclasses.field(init=False)
    t_wfr: float = dataclasses.field(init=False)
    pcaw: float = dataclasses.field(init=False)
    pcbw: float = dataclasses.field(init=False)
    pcai: float = dataclasses.field(init=False)
    pcbi: float = dataclasses.field(init=False)
    pcar: float = dataclasses.field(init=False)
    pcbr: float = dataclasses.field(init=False)
    pcas: float = dataclasses.field(init=False)
    pcbs: float = dataclasses.field(init=False)
    pcag: float = dataclasses.field(init=False)
    pcbg: float = dataclasses.field(init=False)
    pcah: float = dataclasses.field(init=False)
    pcbh: float = dataclasses.field(init=False)
    edaw: float = dataclasses.field(init=False)
    edbw: float = dataclasses.field(init=False)
    edai: float = dataclasses.field(init=False)
    edbi: float = dataclasses.field(init=False)
    edar: float = dataclasses.field(init=False)
    edbr: float = dataclasses.field(init=False)
    edas: float = dataclasses.field(init=False)
    edbs: float = dataclasses.field(init=False)
    edag: float = dataclasses.field(init=False)
    edbg: float = dataclasses.field(init=False)
    edah: float = dataclasses.field(init=False)
    edbh: float = dataclasses.field(init=False)
    oeaw: float = dataclasses.field(init=False)
    oebw: float = dataclasses.field(init=False)
    oeai: float = dataclasses.field(init=False)
    oebi: float = dataclasses.field(init=False)
    oear: float = dataclasses.field(init=False)
    oebr: float = dataclasses.field(init=False)
    oeas: float = dataclasses.field(init=False)
    oebs: float = dataclasses.field(init=False)
    oeag: float = dataclasses.field(init=False)
    oebg: float = dataclasses.field(init=False)
    oeah: float = dataclasses.field(init=False)
    oebh: float = dataclasses.field(init=False)
    rraw: float = dataclasses.field(init=False)
    rrbw: float = dataclasses.field(init=False)
    rrai: float = dataclasses.field(init=False)
    rrbi: float = dataclasses.field(init=False)
    rrar: float = dataclasses.field(init=False)
    rrbr: float = dataclasses.field(init=False)
    rras: float = dataclasses.field(init=False)
    rrbs: float = dataclasses.field(init=False)
    rrag: float = dataclasses.field(init=False)
    rrbg: float = dataclasses.field(init=False)
    rrah: float = dataclasses.field(init=False)
    rrbh: float = dataclasses.field(init=False)
    tvai: float = dataclasses.field(init=False)
    tvbi: float = dataclasses.field(init=False)
    tvar: float = dataclasses.field(init=False)
    tvbr: float = dataclasses.field(init=False)
    tvas: float = dataclasses.field(init=False)
    tvbs: float = dataclasses.field(init=False)
    tvag: float = dataclasses.field(init=False)
    tvbg: float = dataclasses.field(init=False)
    tvah: float = dataclasses.field(init=False)
    tvbh: float = dataclasses.field(init=False)
    crevp_1: float = dataclasses.field(init=False)
    crevp_2: float = dataclasses.field(init=False)
    crevp_3: float = dataclasses.field(init=False)
    crevp_4: float = dataclasses.field(init=False)
    crevp_5: float = dataclasses.field(init=False)
    cssub_1: float = dataclasses.field(init=False)
    cssub_2: float = dataclasses.field(init=False)
    cssub_3: float = dataclasses.field(init=False)
    cssub_4: float = dataclasses.field(init=False)
    cssub_5: float = dataclasses.field(init=False)
    cgsub_1: float = dataclasses.field(init=False)
    cgsub_2: float = dataclasses.field(init=False)
    cgsub_3: float = dataclasses.field(init=False)
    cgsub_4: float = dataclasses.field(init=False)
    cgsub_5: float = dataclasses.field(init=False)
    csmlt_1: float = dataclasses.field(init=False)
    csmlt_2: float = dataclasses.field(init=False)
    csmlt_3: float = dataclasses.field(init=False)
    csmlt_4: float = dataclasses.field(init=False)
    cgmlt_1: float = dataclasses.field(init=False)
    cgmlt_2: float = dataclasses.field(init=False)
    cgmlt_3: float = dataclasses.field(init=False)
    cgmlt_4: float = dataclasses.field(init=False)
    cgfr_1: float = dataclasses.field(init=False)
    cgfr_2: float = dataclasses.field(init=False)
    normw: float = dataclasses.field(init=False)
    normr: float = dataclasses.field(init=False)
    normi: float = dataclasses.field(init=False)
    norms: float = dataclasses.field(init=False)
    normg: float = dataclasses.field(init=False)
    expow: float = dataclasses.field(init=False)
    expor: float = dataclasses.field(init=False)
    expoi: float = dataclasses.field(init=False)
    expos: float = dataclasses.field(init=False)
    expog: float = dataclasses.field(init=False)
    cracw: float = dataclasses.field(init=False)
    craci: float = dataclasses.field(init=False)
    csacw: float = dataclasses.field(init=False)
    csaci: float = dataclasses.field(init=False)
    cgacw: float = dataclasses.field(init=False)
    cgaci: float = dataclasses.field(init=False)
    cracs: float = dataclasses.field(init=False)
    csacr: float = dataclasses.field(init=False)
    cgacr: float = dataclasses.field(init=False)
    cgacs: float = dataclasses.field(init=False)
    acc: List[float] = dataclasses.field(init=False)
    acco: List[List[float]] = dataclasses.field(init=False)

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
        self.li00 = (constants.HLF - physcons.DC_ICE * physcons.TICE0) / self.c_air
        self.li20 = self.lv00 + self.li00

        self.d1_vap = self.d0_vap / self.c_air
        self.d1_ice = physcons.DC_ICE / self.c_air

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
        self.ntimes = int(
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
                constants.PI**2 * self.n0r_sig * self.n0w_sig * physcons.RHO_W / 24.0
            )
            self.csacw = (
                constants.PI**2 * self.n0s_sig * self.n0w_sig * physcons.RHO_W / 24.0
            )
            if self.do_hail:
                self.cgacw = (
                    constants.PI**2
                    * self.n0h_sig
                    * self.n0w_sig
                    * physcons.RHO_W
                    / 24.0
                )
            else:
                self.cgacw = (
                    constants.PI**2
                    * self.n0g_sig
                    * self.n0w_sig
                    * physcons.RHO_W
                    / 24.0
                )

        if self.do_new_acc_ice:
            self.craci = (
                constants.PI**2 * self.n0r_sig * self.n0i_sig * physcons.RHO_I / 24.0
            )
            self.csaci = (
                constants.PI**2 * self.n0s_sig * self.n0i_sig * physcons.RHO_I / 24.0
            )
            if self.do_hail:
                self.cgaci = (
                    constants.PI**2
                    * self.n0h_sig
                    * self.n0i_sig
                    * physcons.RHO_I
                    / 24.0
                )
            else:
                self.cgaci = (
                    constants.PI**2
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
            constants.PI**2 * self.n0r_sig * self.n0s_sig * physcons.RHO_S / 24.0
        )
        self.csacr = (
            constants.PI**2 * self.n0s_sig * self.n0r_sig * physcons.RHO_R / 24.0
        )
        if self.do_hail:
            self.cgacs = (
                constants.PI**2 * self.n0h_sig * self.n0s_sig * physcons.RHO_S / 24.0
            )
            self.cgacr = (
                constants.PI**2 * self.n0h_sig * self.n0r_sig * physcons.RHO_R / 24.0
            )
        else:
            self.cgacs = (
                constants.PI**2 * self.n0g_sig * self.n0s_sig * physcons.RHO_S / 24.0
            )
            self.cgacr = (
                constants.PI**2 * self.n0g_sig * self.n0r_sig * physcons.RHO_R / 24.0
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
            * constants.PI**2
            * self.n0r_sig
            * physcons.RHO_R
            * math.gamma(6 + self.mur)
            / math.exp((6 + self.mur) / (self.mur + 3) * math.log(self.normr))
            * math.exp(-3.0 * math.log(self.expor))
        )
        self.cgfr_2 = 0.66


@dataclasses.dataclass
class PhysicsConfig:
    dt_atmos: float = DEFAULT_FLOAT
    ntimes: int = NamelistDefaults.ntimes
    nconds: int = 1
    hydrostatic: bool = DEFAULT_BOOL
    npx: int = DEFAULT_INT
    npy: int = DEFAULT_INT
    npz: int = DEFAULT_INT
    nwat: int = DEFAULT_INT
    schemes: List = None
    do_qa: bool = DEFAULT_BOOL
    do_inline_mp: bool = NamelistDefaults.do_inline_mp
    c_cracw: float = NamelistDefaults.c_cracw
    c_paut: float = NamelistDefaults.c_paut
    c_pracs: float = NamelistDefaults.c_pracs
    c_psacr: float = NamelistDefaults.c_psacr
    c_pgacr: float = NamelistDefaults.c_pgacr
    c_pgacs: float = NamelistDefaults.c_pgacs
    c_psacw: float = NamelistDefaults.c_psacw
    c_psaci: float = NamelistDefaults.c_psaci
    c_pracw: float = NamelistDefaults.c_pracw
    c_praci: float = NamelistDefaults.c_praci
    c_pgacw: float = NamelistDefaults.c_pgacw
    c_pgaci: float = NamelistDefaults.c_pgaci
    ccn_l: float = NamelistDefaults.ccn_l
    ccn_o: float = NamelistDefaults.ccn_o
    const_vg: bool = NamelistDefaults.const_vg
    const_vi: bool = NamelistDefaults.const_vi
    const_vr: bool = NamelistDefaults.const_vr
    const_vw: bool = NamelistDefaults.const_vw
    const_vs: bool = NamelistDefaults.const_vs
    vw_fac: float = NamelistDefaults.vw_fac
    vs_fac: float = NamelistDefaults.vs_fac
    vg_fac: float = NamelistDefaults.vg_fac
    vi_fac: float = NamelistDefaults.vi_fac
    vr_fac: float = NamelistDefaults.vr_fac
    de_ice: bool = NamelistDefaults.de_ice
    layout: Tuple[int, int] = nml.NamelistDefaults.layout
    # gfdl_cloud_microphys.F90
    tau_r2g: float = NamelistDefaults.tau_r2g
    """rain freezing during fast_sat"""
    tau_smlt: float = NamelistDefaults.tau_smlt
    """snow melting timescale"""
    tau_gmlt: float = NamelistDefaults.tau_gmlt
    """graupel melting timescale"""
    tau_g2r: float = NamelistDefaults.tau_g2r
    """graupel melting to rain"""
    tau_imlt: float = NamelistDefaults.tau_imlt
    """cloud ice melting"""
    tau_i2s: float = NamelistDefaults.tau_i2s
    """cloud ice to snow auto - conversion"""
    tau_l2r: float = NamelistDefaults.tau_l2r
    """cloud water to rain auto - conversion"""
    tau_g2v: float = NamelistDefaults.tau_g2v
    """graupel sublimation"""
    tau_v2g: float = NamelistDefaults.tau_v2g
    """graupel deposition -- make it a slow process"""
    ql_mlt: float = NamelistDefaults.ql_mlt
    """max value of cloud water allowed from melted cloud ice"""
    ql0_max: float = NamelistDefaults.ql0_max
    """max cloud water value (auto converted to rain)"""
    qs_mlt: float = NamelistDefaults.qs_mlt
    """max cloud water due to snow melt"""
    t_min: float = NamelistDefaults.t_min
    """minimum temperature to freeze - dry all water vapor (K)"""
    t_sub: float = NamelistDefaults.t_sub
    """min temp for sublimation of cloud ice"""
    qi_gen: float = NamelistDefaults.qi_gen
    """max cloud ice generation during remapping step"""
    qi_lim: float = NamelistDefaults.qi_lim
    """cloud ice limiter to prevent large ice build up"""
    qi0_max: float = NamelistDefaults.qi0_max
    """max cloud ice value (by other sources)"""
    rad_snow: bool = NamelistDefaults.rad_snow
    """consider snow in cloud fraction calculation"""
    rad_graupel: bool = NamelistDefaults.rad_graupel
    """consider graupel in cloud fraction calculation"""
    rad_rain: bool = NamelistDefaults.rad_rain
    """consider rain in cloud fraction calculation"""
    do_cld_adj: bool = NamelistDefaults.do_cld_adj
    """do cloud fraction adjustment"""
    dw_ocean: float = NamelistDefaults.dw_ocean
    """base value for ocean"""
    dw_land: float = NamelistDefaults.dw_land
    """base value for subgrid deviation / variability over land
     - cloud scheme 0 - ?
     - 1: old fvgfs gfdl) mp implementation
     - 2: binary cloud scheme (0 / 1)"""
    icloud_f: int = NamelistDefaults.icloud_f
    cld_min: float = NamelistDefaults.cld_min
    """minimum cloud fraction"""
    tau_l2v: float = NamelistDefaults.tau_l2v
    """cloud water to water vapor (evaporation)"""
    tau_v2l: float = NamelistDefaults.tau_v2l
    """water vapor to cloud water (condensation)"""
    tau_revp: float = NamelistDefaults.tau_revp
    tau_wbf: float = NamelistDefaults.tau_wbf
    c2l_ord: int = NamelistDefaults.c2l_ord
    do_sedi_heat: bool = NamelistDefaults.do_sedi_heat
    do_sedi_melt: bool = NamelistDefaults.do_sedi_melt
    do_sedi_uv: bool = NamelistDefaults.do_sedi_uv
    do_sedi_w: bool = NamelistDefaults.do_sedi_w
    fast_sat_adj: bool = NamelistDefaults.fast_sat_adj
    qc_crt: float = NamelistDefaults.qc_crt
    fix_negative: bool = NamelistDefaults.fix_negative
    do_cond_timescale: bool = NamelistDefaults.do_cond_timescale
    do_evap_timescale: bool = True
    do_hail: bool = NamelistDefaults.do_hail
    consv_checker: bool = NamelistDefaults.consv_checker
    do_warm_rain_mp: bool = NamelistDefaults.do_warm_rain_mp
    do_wbf: bool = NamelistDefaults.do_wbf
    do_psd_water_fall: bool = NamelistDefaults.do_psd_water_fall
    do_psd_ice_fall: bool = NamelistDefaults.do_psd_ice_fall
    do_psd_water_num: bool = NamelistDefaults.do_psd_water_num
    do_psd_ice_num: bool = NamelistDefaults.do_psd_ice_num
    do_new_acc_water: bool = NamelistDefaults.do_new_acc_water
    do_new_acc_ice: bool = NamelistDefaults.do_new_acc_ice
    cp_heating: bool = NamelistDefaults.cp_heating
    fast_fr_mlt: bool = True
    fast_dep_sub: bool = True
    delay_cond_evap: bool = True
    mp_time: float = NamelistDefaults.mp_time
    prog_ccn: bool = NamelistDefaults.prog_ccn
    qi0_crt: float = NamelistDefaults.qi0_crt
    qs0_crt: float = NamelistDefaults.qs0_crt
    xr_a: float = NamelistDefaults.xr_a
    xr_b: float = NamelistDefaults.xr_b
    xr_c: float = NamelistDefaults.xr_c
    te_err: float = NamelistDefaults.te_err
    tw_err: float = NamelistDefaults.tw_err
    rh_thres: float = NamelistDefaults.rh_thres
    rhc_cevap: float = NamelistDefaults.rhc_cevap
    rhc_revap: float = NamelistDefaults.rhc_revap
    f_dq_p: float = NamelistDefaults.f_dq_p
    f_dq_m: float = NamelistDefaults.f_dq_m
    fi2s_fac: float = NamelistDefaults.fi2s_fac
    fi2g_fac: float = NamelistDefaults.fi2g_fac
    fs2g_fac: float = NamelistDefaults.fs2g_fac
    is_fac: float = NamelistDefaults.is_fac
    ss_fac: float = NamelistDefaults.ss_fac
    gs_fac: float = NamelistDefaults.gs_fac
    rh_fac_evap: float = NamelistDefaults.rh_fac_evap
    rh_fac_cond: float = NamelistDefaults.rh_fac_cond
    sed_fac: float = NamelistDefaults.sed_fac
    rh_inc: float = NamelistDefaults.rh_inc
    rh_inr: float = NamelistDefaults.rh_inr
    # rh_ins: Any
    rthresh: float = NamelistDefaults.rthresh
    sedi_transport: bool = NamelistDefaults.sedi_transport
    # use_ccn: Any
    use_ppm: bool = NamelistDefaults.use_ppm
    use_rhc_cevap: bool = NamelistDefaults.use_rhc_cevap
    use_rhc_revap: bool = NamelistDefaults.use_rhc_revap
    vw_max: float = NamelistDefaults.vw_max
    vg_max: float = NamelistDefaults.vg_max
    vi_max: float = NamelistDefaults.vi_max
    vr_max: float = NamelistDefaults.vr_max
    vs_max: float = NamelistDefaults.vs_max
    z_slope_ice: bool = NamelistDefaults.z_slope_ice
    z_slope_liq: bool = NamelistDefaults.z_slope_liq
    tice: float = NamelistDefaults.tice
    tice_mlt: float = NamelistDefaults.tice_mlt
    alin: float = NamelistDefaults.alin
    alinw: float = NamelistDefaults.alinw
    alini: float = NamelistDefaults.alini
    alinr: float = NamelistDefaults.alinr
    alins: float = NamelistDefaults.alins
    aling: float = NamelistDefaults.aling
    alinh: float = NamelistDefaults.alinh
    blinw: float = NamelistDefaults.blinw
    blini: float = NamelistDefaults.blini
    blinr: float = NamelistDefaults.blinr
    blins: float = NamelistDefaults.blins
    bling: float = NamelistDefaults.bling
    blinh: float = NamelistDefaults.blinh
    clin: float = NamelistDefaults.clin
    n0w_sig: float = NamelistDefaults.n0w_sig
    n0i_sig: float = NamelistDefaults.n0i_sig
    n0r_sig: float = NamelistDefaults.n0r_sig
    n0s_sig: float = NamelistDefaults.n0s_sig
    n0g_sig: float = NamelistDefaults.n0g_sig
    n0h_sig: float = NamelistDefaults.n0h_sig
    n0w_exp: float = NamelistDefaults.n0w_exp
    n0i_exp: float = NamelistDefaults.n0i_exp
    n0r_exp: float = NamelistDefaults.n0r_exp
    n0s_exp: float = NamelistDefaults.n0s_exp
    n0g_exp: float = NamelistDefaults.n0g_exp
    n0h_exp: float = NamelistDefaults.n0h_exp
    muw: float = NamelistDefaults.muw
    mui: float = NamelistDefaults.mui
    mur: float = NamelistDefaults.mur
    mus: float = NamelistDefaults.mus
    mug: float = NamelistDefaults.mug
    muh: float = NamelistDefaults.muh
    cfflag: int = NamelistDefaults.cfflag
    irain_f: int = NamelistDefaults.irain_f
    inflag: int = NamelistDefaults.inflag
    igflag: int = NamelistDefaults.igflag
    ifflag: int = NamelistDefaults.ifflag
    sedflag: int = NamelistDefaults.sedflag
    vdiffflag: int = NamelistDefaults.vdiffflag
    do_mp_table_emulation: bool = NamelistDefaults.do_mp_table_emulation
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
