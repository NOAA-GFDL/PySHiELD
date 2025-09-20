import dataclasses
from enum import Enum, unique
from typing import List, Optional, Tuple

import f90nml

from ndsl import MetaEnumStr
from ndsl.namelist import Namelist


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
    hydrostatic: bool = DEFAULT_BOOL
    npx: int = DEFAULT_INT
    npy: int = DEFAULT_INT
    npz: int = DEFAULT_INT
    nwat: int = DEFAULT_INT
    schemes: List = DEFAULT_SCHEMES
    do_qa: bool = DEFAULT_BOOL
    do_inline_mp: bool = False
    """Whether microphysics is inlined in the dycore"""
    c_cracw: float = 0.8
    """Rain accretion efficiency"""
    c_paut: float = 0.5
    """Autoconversion cloud water to rain (use 0.5 to reduce autoconversion"""
    c_pgacs: float = 0.01
    """Snow to graupel "accretion" eff. (was 0.1 in zetac)"""
    c_psaci: float = 0.05
    """Accretion: cloud ice to snow (was 0.1 in zetac)"""
    ccn_l: float = 300.0
    """CCN over land (cm^-3)"""
    ccn_o: float = 100.0
    """CCN over ocean (cm^-3)"""
    const_vg: bool = False
    """Fall velocity tuning constant of graupel"""
    const_vi: bool = False
    """Fall velocity tuning constant of ice"""
    const_vr: bool = False
    """Fall velocity tuning constant of rain water"""
    const_vs: bool = False
    """Fall velocity tuning constant of snow"""
    vs_fac: float = 1.0
    """if const_vs: 1."""
    vg_fac: float = 1.0
    """if const_vg: 2."""
    vi_fac: float = 1.0
    """if const_vi: 1/3"""
    vr_fac: float = 1.0
    """if const_vr: 4."""
    de_ice: bool = False
    """To prevent excessive build-up of cloud ice from external sources"""
    layout: Tuple[int, int] = (1, 1)
    # gfdl_cloud_microphys.F90
    tau_imlt: float = 600.0
    """cloud ice melting"""
    tau_i2s: float = 1000.0
    """cloud ice to snow auto - conversion"""
    tau_g2v: float = 1200.0
    """graupel sublimation"""
    tau_v2g: float = 21600.0
    """graupel deposition -- make it a slow process"""
    ql_mlt: float = 2.0e-3
    """max value of cloud water allowed from melted cloud ice"""
    qs_mlt: float = 1.0e-6
    """max cloud water due to snow melt"""
    t_sub: float = 184.0
    """min temp for sublimation of cloud ice"""
    qi_gen: float = 1.82e-6
    """max cloud ice generation during remapping step"""
    qi_lim: float = 1.0
    """cloud ice limiter to prevent large ice build up"""
    qi0_max: float = 1.0e-4
    """max cloud ice value (by other sources)"""
    rad_snow: bool = True
    """consider snow in cloud fraction calculation"""
    rad_rain: bool = True
    """consider rain in cloud fraction calculation"""
    dw_ocean: float = 0.10
    """base value for ocean"""
    dw_land: float = 0.15
    """base value for subgrid deviation / variability over land"""
    # cloud scheme 0 - ?
    # 1: old fvgfs gfdl) mp implementation
    # 2: binary cloud scheme (0 / 1)
    tau_l2v: float = 300.0
    """cloud water to water vapor (evaporation)"""
    c2l_ord: int = 4
    do_sedi_heat: bool = False
    """Transport of heat in sedimentation"""
    do_sedi_w: bool = True
    """Transport of vertical motion in sedimentation"""
    fast_sat_adj: bool = True
    qc_crt: float = 5.0e-8
    """Minimum condensate mixing ratio to allow partial cloudiness"""
    fix_negative: bool = True
    """Fix negative water species"""
    irain_f: int = 0
    """Cloud water to rain auto conversion scheme"""
    mp_time: float = 225.0
    """Maximum microphysics timestep (sec)"""
    prog_ccn: bool = False
    """Do prognostic ccn (yi ming's method)"""
    qi0_crt: float = 8e-05
    """Cloud ice to snow autoconversion threshold"""
    qs0_crt: float = 0.003
    """Snow to graupel density threshold (0.6e-3 in purdue lin scheme)"""
    rh_inc: float = 0.2
    """RH increment for complete evaporation of cloud water and cloud ice"""
    rh_inr: float = 0.3
    """RH increment for minimum evaporation of rain"""
    # rh_ins: Any
    rthresh: float = 1e-05
    """Critical cloud drop radius (micrometers)"""
    sedi_transport: bool = True
    """Transport of momentum in sedimentation"""
    # use_ccn: Any
    use_ppm: bool = False
    """Use ppm fall scheme"""
    vg_max: float = 16.0
    """Maximum fall speed for graupel"""
    vi_max: float = 1.0
    """Maximum fall speed for ice"""
    vr_max: float = 16.0
    """Maximum fall speed for rain"""
    vs_max: float = 2.0
    """Maximum fall speed for snow"""
    z_slope_ice: bool = True
    """Use linear mono slope for autoconversions"""
    z_slope_liq: bool = True
    """Use linear mono slope for autoconversions"""
    tice: float = 273.16
    """set tice = 165. to turn off ice - phase phys (kessler emulator)"""
    alin: float = 842.0
    """value for 'a' in lin1983"""
    clin: float = 4.8
    """"c" in lin 1983, 4.8 -- > 6. (to enhance ql -- > qs)"""
    namelist_override: Optional[str] = None
    daily_mean: bool = DEFAULT_BOOL
    """flag to replace cosz with daily mean value"""

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
            tau_imlt=namelist.tau_imlt,
            tau_i2s=namelist.tau_i2s,
            tau_g2v=namelist.tau_g2v,
            tau_v2g=namelist.tau_v2g,
            ql_mlt=namelist.ql_mlt,
            qs_mlt=namelist.qs_mlt,
            t_sub=namelist.t_sub,
            qi_gen=namelist.qi_gen,
            qi_lim=namelist.qi_lim,
            qi0_max=namelist.qi0_max,
            rad_snow=namelist.rad_snow,
            rad_rain=namelist.rad_rain,
            dw_ocean=namelist.dw_ocean,
            dw_land=namelist.dw_land,
            tau_l2v=namelist.tau_l2v,
            c2l_ord=namelist.c2l_ord,
            do_sedi_heat=namelist.do_sedi_heat,
            do_sedi_w=namelist.do_sedi_w,
            fast_sat_adj=namelist.fast_sat_adj,
            qc_crt=namelist.qc_crt,
            fix_negative=namelist.fix_negative,
            irain_f=namelist.irain_f,
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
            daily_mean=namelist.daily_mean,
        )
