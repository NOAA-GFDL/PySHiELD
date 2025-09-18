import dataclasses
from enum import Enum, unique

import f90nml

from ndsl import MetaEnumStr
from ndsl.dsl.gt4py_utils import tracer_variables
from ndsl.namelist import Namelist, NamelistDefaults


DEFAULT_INT = 0
DEFAULT_BOOL = False
DEFAULT_SCHEMES = ["GFS_microphysics"]
TRACER_DIM = "n_tracers"


# TODO: Should we have an enum for each class of parameterization
# microphysics, PBL, shallow convection, etc?
@unique
class PHYSICS_PACKAGES(Enum, metaclass=MetaEnumStr):
    GFS_microphysics = "GFS_microphysics"
    SATM_EDMF = "SATM_EDMF"


@dataclasses.dataclass
class PBLConfig:
    dt_atmos: int = DEFAULT_INT
    hydrostatic: bool = DEFAULT_BOOL
    isatmedmf: int = NamelistDefaults.isatmedmf
    """flag for scale-aware turbulent moist edmf scheme"""
    xkzm_h: float = NamelistDefaults.xkzm_h
    """Background vertical diffusion for heat q over ocean"""
    xkzm_m: float = NamelistDefaults.xkzm_m
    """Background vertical diffusion for momentum over ocean"""
    xkzm_hl: float = NamelistDefaults.xkzm_hl
    """Background vertical diffusion for heat q over land"""
    xkzm_ml: float = NamelistDefaults.xkzm_ml
    """Background vertical diffusion for momentum over land"""
    xkzm_hi: float = NamelistDefaults.xkzm_hi
    """Background vertical diffusion for heat q over ice"""
    xkzm_mi: float = NamelistDefaults.xkzm_mi
    """Background vertical diffusion for momentum over ice"""
    xkzm_ho: float = NamelistDefaults.xkzm_ho
    """Background vertical diffusion for heat q over ocean"""
    xkzm_mo: float = NamelistDefaults.xkzm_mo
    """Background vertical diffusion for momentum over ocean"""
    xkzminv: float = NamelistDefaults.xkzminv
    """Diffusivity in inversion layers"""
    xkzm_s: float = NamelistDefaults.xkzm_s
    """Sigma threshold for background momentum diffusion"""
    xkzm_lim: float = NamelistDefaults.xkzm_lim
    """Background diffusion limit"""
    xkgdx: float = NamelistDefaults.xkgdx
    """Background vertical diffusion threshold"""
    do_dk_hb19: bool = DEFAULT_BOOL
    """Flag to use HB19 background diffusion formula in satmedmf"""
    rlmn: float = NamelistDefaults.rlmn
    """Lower limit on aymptotic mixing length in satmedmf"""
    rlmx: float = NamelistDefaults.rlmx
    """Upper limit on aymptotic mixing length in satmedmf"""
    ntracers: int = int(len(tracer_variables))
    """Number of tracers"""
    ntiw: int = DEFAULT_INT
    """Tracer index of ice water"""
    ntcw: int = DEFAULT_INT
    """Tracer index of cloud water"""
    ntke: int = DEFAULT_INT
    """Tracer index of subgrid turbulent kinetic energy"""
    dspheat: bool = NamelistDefaults.dspheat
    """Flag for dissipative heating"""
    cap_k0_land: bool = NamelistDefaults.cap_k0_land
    """Flag to apply limiter on background diffusivity in inversion layer over land"""

    def __post_init__(self):
        if self.isatmedmf != 0:
            raise NotImplementedError(
                f"PBL Config: isatmedmf == {self.isatmedmf} not implemented"
            )
        self.ntiw = tracer_variables.index("qice")
        self.ntcw = tracer_variables.index("qliquid")
        self.ntke = tracer_variables.index("qsgs_tke")

    @classmethod
    def from_f90nml(self, f90_namelist: f90nml.Namelist) -> "PBLConfig":
        namelist = Namelist.from_f90nml(f90_namelist)
        return self.from_namelist(namelist)

    @classmethod
    def from_namelist(cls, namelist: Namelist) -> "PBLConfig":
        return cls(
            dt_atmos=namelist.dt_atmos,
            hydrostatic=namelist.hydrostatic,
            isatmedmf=namelist.isatmedmf,
            dspheat=namelist.dspheat,
            xkzm_h=namelist.xkzm_h,
            xkzm_m=namelist.xkzm_m,
            xkzm_s=namelist.xkzm_s,
            xkzm_hl=namelist.xkzm_hl,
            xkzm_ml=namelist.xkzm_ml,
            xkzm_ho=namelist.xkzm_ho,
            xkzm_mo=namelist.xkzm_mo,
            xkzm_hi=namelist.xkzm_hi,
            xkzm_mi=namelist.xkzm_mi,
            xkzminv=namelist.xkzminv,
            xkzm_lim=namelist.xkzm_lim,
            xkgdx=namelist.xkgdx,
            do_dk_hb19=namelist.do_dk_hb19,
            rlmn=namelist.rlmn,
            rlmx=namelist.rlmx,
            ntracers=namelist.ntracers,
            cap_k0_land=namelist.cap_k0_land,
        )
