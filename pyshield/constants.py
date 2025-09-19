import ndsl.constants as constants
from ndsl.dsl.typing import Float


# Driver constants
HOCP = Float(constants.HLV / constants.CP_AIR)
QMIN = Float(1.0e-10)
RAINMIN = Float(1.0e-13)
P850 = Float(85000.0)
EPSQ = Float(1.0e-20)
HSUB = Float(constants.HLV + constants.HLF)
CZMIN = Float(0.0001)
"""cos(89.994)"""
ZERO = Float(0.0e0)
ALBDF = Float(0.06)
CON_P001 = Float(0.001e0)
CON_DAY = Float(86400.0)
TF = Float(258.16)
TCRF = Float(1.0 / (constants.TICE - TF))
CONT = Float(constants.CP_AIR / constants.GRAV)
CONQ = Float(constants.HLV / constants.GRAV)
P00 = Float(1.0e5)
"""idealized surface atmospheric pressure (Pa)"""

# Alternative Units and Precision in Physics:
CPH2O1 = Float(4.218e3)
"""Specific heat of water in J/kg/K"""
CPH2O2 = Float(4.2e6)

# Constants used in interpolating radiation onto physics timestep:
F_EPS = Float(0.0001)
