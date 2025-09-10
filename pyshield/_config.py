import dataclasses
from enum import Enum, unique
from typing import List, Optional, Sequence, Tuple

import f90nml

from ndsl import MetaEnumStr
from ndsl.namelist import Namelist, NamelistDefaults


DEFAULT_FLOAT = 0.0
DEFAULT_INT = 0
DEFAULT_BOOL = False
DEFAULT_SCHEMES = ["GFS_microphysics"]


@unique
class PHYSICS_PACKAGES(Enum, metaclass=MetaEnumStr):
    GFS_microphysics = "GFS_microphysics"


@dataclasses.dataclass
class SurfaceConfig:
    do_z0_hwrf15: bool = DEFAULT_BOOL
    """flag to use z0 scheme from 2015 HWRF"""
    do_z0_hwrf17: bool = DEFAULT_BOOL
    """flag to use z0 scheme from 2017 HWRF"""
    do_z0_hwrf17_hwonly: bool = DEFAULT_BOOL
    """flag to use z0 scheme from 2017 HWRF only under high wind"""
    do_z0_moon: bool = DEFAULT_BOOL
    """flag to use z0 scheme from Moon et al. 2007"""
    dt_atmos: float = DEFAULT_FLOAT
    mom4ice: bool = DEFAULT_BOOL
    """Flag to enable mom4 sea-ice"""
    ivegsrc: int = DEFAULT_INT
    """
    Source of vegetation data:
     - 0: USGS
     - 1: IGBP (20 category)
     - 2: UMD (13 category)
    """
    lsm: int = DEFAULT_INT
    """LSM selection. 1=NOAH, 2=NOAH MP"""
    redrag: bool = DEFAULT_BOOL
    """flag for reduced drag coefficient over sea"""
    wind_th_hwrf: float = DEFAULT_FLOAT
    """Wind speed threshold when z0 level off as in HWRF"""
    lsoil: int = 4
    """Number of soil levels"""
    nstf_name: Sequence[int] = (0, 0, 1, 0, 5)
    """
    nstf_name contains the NSSTM related parameters:
    nstf_name(1) : 0 = NSSTM off, 1 = NSSTM on but uncoupled, 2 = NSSTM on and coupled
    nstf_name(2) : 1 = NSSTM spin up on, 0 = NSSTM spin up off
    nstf_name(3) : 1 = NSSTM analysis on, 0 = NSSTM analysis off
    nstf_name(4) : zsea1 in mm
    nstf_name(5) : zsea2 in mm
    TODO: implement via namelist?
    """


@dataclasses.dataclass
class PhysicsConfig:
    dt_atmos: float = DEFAULT_FLOAT
    hydrostatic: bool = DEFAULT_BOOL
    npx: int = DEFAULT_INT
    npy: int = DEFAULT_INT
    npz: int = DEFAULT_INT
    nwat: int = DEFAULT_INT
    schemes: List = None
    do_qa: bool = DEFAULT_BOOL
    do_z0_hwrf15: bool = DEFAULT_BOOL
    do_z0_hwrf17: bool = DEFAULT_BOOL
    do_z0_hwrf17_hwonly: bool = True
    do_z0_moon: bool = DEFAULT_BOOL
    c_cracw: float = NamelistDefaults.c_cracw
    c_paut: float = NamelistDefaults.c_paut
    c_pgacs: float = NamelistDefaults.c_pgacs
    c_psaci: float = NamelistDefaults.c_psaci
    ccn_l: float = NamelistDefaults.ccn_l
    ccn_o: float = NamelistDefaults.ccn_o
    const_vg: bool = NamelistDefaults.const_vg
    const_vi: bool = NamelistDefaults.const_vi
    const_vr: bool = NamelistDefaults.const_vr
    const_vs: bool = NamelistDefaults.const_vs
    vs_fac: float = NamelistDefaults.vs_fac
    vg_fac: float = NamelistDefaults.vg_fac
    vi_fac: float = NamelistDefaults.vi_fac
    vr_fac: float = NamelistDefaults.vr_fac
    de_ice: bool = NamelistDefaults.de_ice
    layout: Tuple[int, int] = NamelistDefaults.layout
    # gfdl_cloud_microphys.F90
    tau_imlt: float = NamelistDefaults.tau_imlt  # cloud ice melting
    tau_i2s: float = NamelistDefaults.tau_i2s  # cloud ice to snow auto - conversion
    tau_g2v: float = NamelistDefaults.tau_g2v  # graupel sublimation
    tau_v2g: float = (
        NamelistDefaults.tau_v2g
    )  # graupel deposition -- make it a slow process
    ql_mlt: float = (
        NamelistDefaults.ql_mlt
    )  # max value of cloud water allowed from melted cloud ice
    qs_mlt: float = NamelistDefaults.qs_mlt  # max cloud water due to snow melt
    t_sub: float = NamelistDefaults.t_sub  # min temp for sublimation of cloud ice
    qi_gen: float = (
        NamelistDefaults.qi_gen
    )  # max cloud ice generation during remapping step
    qi_lim: float = (
        NamelistDefaults.qi_lim
    )  # cloud ice limiter to prevent large ice build up
    qi0_max: float = NamelistDefaults.qi0_max  # max cloud ice value (by other sources)
    rad_snow: bool = (
        NamelistDefaults.rad_snow
    )  # consider snow in cloud fraction calculation
    rad_rain: bool = (
        NamelistDefaults.rad_rain
    )  # consider rain in cloud fraction calculation
    dw_ocean: float = NamelistDefaults.dw_ocean  # base value for ocean
    dw_land: float = (
        NamelistDefaults.dw_land
    )  # base value for subgrid deviation / variability over land
    # cloud scheme 0 - ?
    # 1: old fvgfs gfdl) mp implementation
    # 2: binary cloud scheme (0 / 1)
    tau_l2v: float = (
        NamelistDefaults.tau_l2v
    )  # cloud water to water vapor (evaporation)
    c2l_ord: int = NamelistDefaults.c2l_ord
    do_sedi_heat: bool = NamelistDefaults.do_sedi_heat
    do_sedi_w: bool = NamelistDefaults.do_sedi_w
    fast_sat_adj: bool = NamelistDefaults.fast_sat_adj
    qc_crt: float = NamelistDefaults.qc_crt
    fix_negative: bool = NamelistDefaults.fix_negative
    irain_f: int = NamelistDefaults.irain_f
    mp_time: float = NamelistDefaults.mp_time
    prog_ccn: bool = NamelistDefaults.prog_ccn
    qi0_crt: float = NamelistDefaults.qi0_crt
    qs0_crt: float = NamelistDefaults.qs0_crt
    rh_inc: float = NamelistDefaults.rh_inc
    rh_inr: float = NamelistDefaults.rh_inr
    # rh_ins: Any
    rthresh: float = NamelistDefaults.rthresh
    sedi_transport: bool = NamelistDefaults.sedi_transport
    # use_ccn: Any
    use_ppm: bool = NamelistDefaults.use_ppm
    vg_max: float = NamelistDefaults.vg_max
    vi_max: float = NamelistDefaults.vi_max
    vr_max: float = NamelistDefaults.vr_max
    vs_max: float = NamelistDefaults.vs_max
    z_slope_ice: bool = NamelistDefaults.z_slope_ice
    z_slope_liq: bool = NamelistDefaults.z_slope_liq
    tice: float = NamelistDefaults.tice
    alin: float = NamelistDefaults.alin
    clin: float = NamelistDefaults.clin
    mom4ice: bool = NamelistDefaults.mom4ice
    lsm: int = NamelistDefaults.lsm
    redrag: bool = NamelistDefaults.redrag
    wind_th_hwrf: float = 33.0
    lsoil: int = 4
    ivegsrc: int = 2
    """
    Source for veg and soil categories:
    ivegsrc = 0 => USGS
    ivegsrc = 1 => IGBP (20 category)
    ivegsrc = 2 => UMD (13 category)
    """
    nstf_name: Sequence[int] = (0, 0, 1, 0, 5)
    """
    nstf_name contains the NSSTM related parameters:
    nstf_name(1) : 0 = NSSTM off, 1 = NSSTM on but uncoupled, 2 = NSSTM on and coupled
    nstf_name(2) : 1 = NSSTM spin up on, 0 = NSSTM spin up off
    nstf_name(3) : 1 = NSSTM analysis on, 0 = NSSTM analysis off
    nstf_name(4) : zsea1 in mm
    nstf_name(5) : zsea2 in mm
    TODO: implement via namelist?
    """
    namelist_override: Optional[str] = None
    daily_mean: bool = DEFAULT_BOOL  # flag to replace cosz with daily mean value

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
        if not isinstance(self.nstf_name, tuple):
            self.nstf_name = tuple(self.nstf_name)
        if len(self.nstf_name) != 5:
            raise IndexError(f"nstf_name must have 5 elements, got {self.nstf_name}")

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

    @property
    def surface(self) -> SurfaceConfig:
        return SurfaceConfig(
            do_z0_hwrf15=self.do_z0_hwrf15,
            do_z0_hwrf17=self.do_z0_hwrf17,
            do_z0_hwrf17_hwonly=self.do_z0_hwrf17_hwonly,
            do_z0_moon=self.do_z0_moon,
            dt_atmos=self.dt_atmos,
            mom4ice=self.mom4ice,
            lsm=self.lsm,
            redrag=self.redrag,
            wind_th_hwrf=self.wind_th_hwrf,
            ivegsrc=self.ivegsrc,
            nstf_name=self.nstf_name,
        )
