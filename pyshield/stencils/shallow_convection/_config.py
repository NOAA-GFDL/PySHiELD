import dataclasses

from ndsl.dsl.gt4py_utils import tracer_variables
from ndsl.dsl.typing import Float, set_4d_field_size


# TODO: This should be handled by tracer functionality when ready
FloatFieldShalConv = set_4d_field_size(7, Float)

_DEFAULT_INT = 0
DEFAULT_BOOL = False
DEFAULT_FLOAT = 0.0
SC_TRACER_DIM = "n_tracers_shal"


@dataclasses.dataclass
class ShallowConvectionConfig:
    dt_atmos: float = DEFAULT_FLOAT
    """timestep length (s)"""
    ntke: int = -1
    """index of tke tracer"""
    nsamftrac: int = 7
    """number of tracers convected, excluding humidity"""
    ncld: int = 1
    """Choice of cloud scheme"""
    ntchm: int = _DEFAULT_INT
    """number of chemical tracers"""
    ntcw: int = -1
    """index of cloud water tracer"""
    ntiw: int = -1
    """index pf cloud ice tracer"""
    itc: int = _DEFAULT_INT
    """index of first chemical tracer"""
    clam_shal: float = 0.3
    """c_e for shallow convection (Han and Pan, 2011, eq(6))"""
    c0s_shal: float = 0.002
    """conversion parameter of detrainment from liquid water
    into convetive precipitaiton (1/m)"""
    c1_shal: float = 5.0e-4
    """conversion parameter of detrainment from liquid water
    into grid-scale cloud water (1/m)"""
    pgcon_shal: float = 0.55
    """reduction factor in momentum transport
    due to convection induced pressure gradient force
    0.7 : Gregory et al. (1997, QJRMS)
    0.55: Zhang & Wu (2003, JAS)
    """
    cthk: float = 200.0
    """max cloud top for shallow convection"""
    top_shal: float = 0.7
    """max cloud height for shallow convection (P/Ps < top_shal)"""
    limit_shal_conv: bool = False
    """flag for constraining shal conv based on diagnosed cloud depth/top"""
    asolfac_shal: float = 0.89
    """aerosol-aware parameter inversely proportional to CCN number concentraion
    based on Lim & Hong (2012)"""
    fscav: list = dataclasses.field(default_factory=list)
    """aerosol scavenging coefficients"""

    def __post_init__(self):
        if self.ntiw == -1:
            self.ntiw = tracer_variables.index("qice") - 1
        if self.ntcw == -1:
            self.ntcw = tracer_variables.index("qliquid") - 1
        if self.ntke == -1:
            self.ntke = tracer_variables.index("qsgs_tke") - 1
        # TODO: the -1 is because currently samfshalconv expects qvapor to be a
        # separate array from the rest of the tracers. This is pretty awkward and
        # should be improved...
