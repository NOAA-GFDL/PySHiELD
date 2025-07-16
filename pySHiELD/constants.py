from ndsl import constants


# Driver constants
HOCP = constants.HLV / constants.CP_AIR
QMIN = 1.0e-10
RAINMIN = 1.0e-13
P850 = 85000.0
EPSQ = 1.0e-20
HSUB = constants.HLV + constants.HLF
CZMIN = 0.0001  # cos(89.994)
ZERO = 0.0e0
ALBDF = 0.06
CON_P001 = 0.001e0
CON_DAY = 86400.0
TF = 258.16
TCRF = 1.0 / (constants.TICE - TF)
CONT = constants.CP_AIR / constants.GRAV
CONQ = constants.HLV / constants.GRAV

# Alternative Units and Precision in Physics:
CPH2O1 = 4.218e3  # Specific heat of water in J/kg/K
CPH2O2 = 4.2e6

# Constants used in interpolating radiation onto physics timestep:
F_EPS = 0.0001

TICE = 2.7120e2
"""temp freezing sea (K)"""
