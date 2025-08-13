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

C_LIQ0 = 4.2180e3  # Specific heat of liquid H2O

# Shallow Convection
ELOCP = constants.HLV / constants.CP_AIR
EL2ORC = constants.HLV * constants.HLV / (constants.RVGAS * constants.CP_AIR)
CM = 1.0
DELTA = constants.RVGAS / constants.RDGAS - 1.0
FACT1 = (constants.CP_VAP - C_LIQ0) / constants.RVGAS
FACT2 = constants.HLV / constants.RVGAS - FACT1 * constants.TICE0
CLAMD = 0.1
TKEMX = 0.65
TKEMN = 0.05
DTKE = TKEMX - TKEMN
DTHK = 25.0
D0_SHAL = 0.001
CINPCRMX = 180.0
CINPCRMN = 120.0
CINACRMX = -120.0
CINACRMN = -80.0
CRTLAMD = 3.0e-4
DTMAX = 10800.0
DTMIN = 600.0
BET1 = 1.875
CD1 = 0.506
F1 = 2.0
GAM1 = 0.5
BETAW = 0.03
DXCRT = 15.0e3
H1 = 0.33333333
SHAL_TF = 233.16
SHAL_TCR = 263.16
SHAL_TCRF = 1.0 / (SHAL_TCR - SHAL_TF)

AAFAC = 0.05
EVFACT = 0.3
EVFACTL = 0.3

W1L = -8.0e-3
W2L = -4.0e-2
W3L = -5.0e-3
W4L = -5.0e-4
W1S = -2.0e-4
W2S = -2.0e-3
W3S = -1.0e-3
W4S = -2.0e-5

# Constants used in interpolating radiation onto physics timestep:
F_EPS = 0.0001
