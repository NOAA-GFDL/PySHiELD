import dataclasses
from typing import Sequence

import f90nml

from ndsl.namelist import Namelist, NamelistDefaults


DEFAULT_FLOAT = 0.0
DEFAULT_INT = 0
DEFAULT_BOOL = False
DEFAULT_SCHEMES = ["GFS_microphysics"]


@dataclasses.dataclass
class SurfaceConfig:
    do_z0_hwrf15: bool = DEFAULT_BOOL
    """flag to use z0 scheme from 2015 HWRF"""
    do_z0_hwrf17: bool = DEFAULT_BOOL
    """flag to use z0 scheme from 2017 HWRF"""
    do_z0_hwrf17_hwonly: bool = True
    """flag to use z0 scheme from 2017 HWRF only under high wind"""
    do_z0_moon: bool = DEFAULT_BOOL
    """flag to use z0 scheme from Moon et al. 2007"""
    dt_atmos: float = DEFAULT_FLOAT
    mom4ice: bool = NamelistDefaults.mom4ice
    """Flag to enable mom4 sea-ice"""
    ivegsrc: int = 2
    """
    Source of vegetation data:
     - 0: USGS
     - 1: IGBP (20 category)
     - 2: UMD (13 category)
    """
    lsm: int = NamelistDefaults.lsm
    """LSM selection. 1=NOAH, 2=NOAH MP"""
    redrag: bool = NamelistDefaults.redrag
    """flag for reduced drag coefficient over sea"""
    wind_th_hwrf: float = 33.0
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

    def __post_init__(self):
        if not isinstance(self.nstf_name, tuple):
            self.nstf_name = tuple(self.nstf_name)
        if len(self.nstf_name) != 5:
            raise IndexError(f"nstf_name must have 5 elements, got {self.nstf_name}")

    @classmethod
    def from_f90nml(self, f90_namelist: f90nml.Namelist) -> "SurfaceConfig":
        namelist = Namelist.from_f90nml(f90_namelist)
        return self.from_namelist(namelist)

    @classmethod
    def from_namelist(cls, namelist: Namelist) -> "SurfaceConfig":
        return cls(
            do_z0_hwrf15=namelist.do_z0_hwrf15,
            do_z0_hwrf17=namelist.do_z0_hwrf17,
            do_z0_hwrf17_hwonly=namelist.do_z0_hwrf17_hwonly,
            do_z0_moon=namelist.do_z0_moon,
            dt_atmos=namelist.dt_atmos,
            mom4ice=namelist.mom4ice,
            ivegsrc=namelist.ivegsrc,
            lsm=namelist.lsm,
            redrag=namelist.redrag,
            wind_th_hwrf=namelist.wind_th_hwrf,
            lsoil=namelist.lsoil,
            nstf_name=namelist.nstf_name,
        )
