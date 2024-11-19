import dataclasses
from enum import Enum, unique
from typing import List, Optional, Tuple

import f90nml

from ndsl import MetaEnumStr
from ndsl.dsl.gt4py_utils import tracer_variables
from ndsl.dsl.typing import Float, set_4d_field_size
from ndsl.namelist import Namelist, NamelistDefaults


# TODO: This is a hack
FloatFieldTracer = set_4d_field_size(9, Float)

DEFAULT_INT = 0
DEFAULT_BOOL = False
DEFAULT_SCHEMES = ["GFS_microphysics"]
TRACER_DIM = "n_tracers"


# TODO: Should we have an enum for each class of parameterization
# microphysics, PBL, shallow convection, etc?
@unique
class PHYSICS_PACKAGES(Enum, metaclass=MetaEnumStr):
    GFS_microphysics = "GFS_microphysics"
    SATMED_EDMF = "SATM_EDMF"


@dataclasses.dataclass
class PBLConfig:
    dt_atmos: int = DEFAULT_INT
    hydrostatic: bool = DEFAULT_BOOL
    isatmedmf: int = NamelistDefaults.isatmedmf
    xkzm_h: float = NamelistDefaults.xkzm_h
    xkzm_m: float = NamelistDefaults.xkzm_m
    xkzm_hl: float = NamelistDefaults.xkzm_hl
    xkzm_ml: float = NamelistDefaults.xkzm_ml
    xkzm_hi: float = NamelistDefaults.xkzm_hi
    xkzm_mi: float = NamelistDefaults.xkzm_mi
    xkzminv: float = NamelistDefaults.xkzminv
    xkzm_s: float = NamelistDefaults.xkzm_s
    xkzm_lim: float = NamelistDefaults.xkzm_lim
    xkgdx: float = NamelistDefaults.xkgdx
    do_dk_hb19: bool = DEFAULT_BOOL
    rlmn: float = NamelistDefaults.rlmn
    rlmx: float = NamelistDefaults.rlmx
    ntracers: int = int(len(tracer_variables))
    ntiw: int = DEFAULT_INT
    ntcw: int = DEFAULT_INT
    ntke: int = DEFAULT_INT
    dspheat: bool = NamelistDefaults.dspheat
    cap_k0_land: bool = DEFAULT_BOOL

    def __post_init__(self):
        if self.isatmedmf != 0:
            raise NotImplementedError(
                f"PBL Config: isatmedmf == {self.isatmedmf} not implemented"
            )
        self.ntiw = tracer_variables.index("qice")
        self.ntiw = tracer_variables.index("qliquid")
        self.ntke = tracer_variables.index("qsgs_tke")


@dataclasses.dataclass
class PhysicsConfig:
    dt_atmos: int = DEFAULT_INT
    hydrostatic: bool = DEFAULT_BOOL
    npx: int = DEFAULT_INT
    npy: int = DEFAULT_INT
    npz: int = DEFAULT_INT
    nwat: int = DEFAULT_INT
    schemes: List = None
    ntracers: int = int(len(tracer_variables))
    ntiw: int = DEFAULT_INT
    ntcw: int = DEFAULT_INT
    ntke: int = DEFAULT_INT
    do_qa: bool = DEFAULT_BOOL
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
    isatmedmf: int = NamelistDefaults.isatmedmf
    dspheat: bool = NamelistDefaults.dspheat
    xkzm_h: float = NamelistDefaults.xkzm_h
    xkzm_m: float = NamelistDefaults.xkzm_m
    xkzm_hl: float = NamelistDefaults.xkzm_hl
    xkzm_ml: float = NamelistDefaults.xkzm_ml
    xkzm_hi: float = NamelistDefaults.xkzm_hi
    xkzm_mi: float = NamelistDefaults.xkzm_mi
    xkzminv: float = NamelistDefaults.xkzminv
    xkzm_lim: float = NamelistDefaults.xkzm_lim
    xkgdx: float = NamelistDefaults.xkgdx
    do_dk_hb19: bool = DEFAULT_BOOL
    rlmn: float = NamelistDefaults.rlmn
    rlmx: float = NamelistDefaults.rlmx
    cap_k0_land: bool = DEFAULT_BOOL
    xkzm_s: float = NamelistDefaults.xkzm_s
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
        self.ntiw = tracer_variables.index("qice")
        self.ntiw = tracer_variables.index("qliquid")
        self.ntke = tracer_variables.index("qsgs_tke")
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
            isatmedmf=namelist.isatmedmf,
            dspheat=namelist.dspheat,
            xkzm_h=namelist.xkzm_h,
            xkzm_m=namelist.xkzm_m,
            xkzm_s=namelist.xkzm_s,
        )

    @property
    def pbl(self) -> PBLConfig:
        return PBLConfig(
            dt_atmos=self.dt_atmos,
            hydrostatic=self.hydrostatic,
            isatmedmf=self.isatmedmf,
            xkzm_h=self.xkzm_h,
            xkzm_m=self.xkzm_m,
            xkzm_hl=self.xkzm_hl,
            xkzm_ml=self.xkzm_ml,
            xkzm_hi=self.xkzm_hi,
            xkzm_mi=self.xkzm_mi,
            xkzminv=self.xkzminv,
            xkzm_s=self.xkzm_s,
            xkzm_lim=self.xkzm_lim,
            xkgdx=self.xkgdx,
            do_dk_hb19=self.do_dk_hb19,
            rlmn=self.rlmn,
            rlmx=self.rlmx,
            ntracers=self.ntracers,
            ntiw=self.ntiw,
            ntcw=self.ntcw,
            ntke=self.ntke,
            dspheat=self.dspheat,
            cap_k0_land=self.cap_k0_land,
        )
