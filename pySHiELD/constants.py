import ndsl.constants as constants

C_LIQ0 = 4.2180e+3  # Specific heat of liquid H2O

# Shallow Convection
ELOCP = constants.HLV / constants.CP_AIR,
EL2ORC = constants.HLV * constants.HLV / (
    constants.RVGAS * constants.CP_AIR
)
CM = 1.0
DELTA = constants.RVGAS / constants.RDGAS - 1.
FACT1 = (constants.CP_VAP - C_LIQ0) / constants.RVGAS
FACT2 = constants.HLV / constants.RVGAS - FACT1 * constants.TICE0
CLAMD = 0.1
TKEMX = 0.65
TKEMN = 0.05
DTKE = TKEMX - TKEMN
DTHK = 25.
CINPCRMX = 180.
CINPCRMN = 120.
CINACRMX = -120.
CINACRMN = -80.
CRTLAMD = 3.e-4
DTMAX = 10800.
DTMIN = 600.
BET1 = 1.875
CD1 = .506
F1 = 2.0
GAM1 = .5
BETAW = .03
DXCRT = 15.e3
H1 = 0.33333333

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
