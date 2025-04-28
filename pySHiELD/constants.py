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

# sfc_sice constants
FLOAT_EPS = 1.0e-8
HIMAX = 8.0  # maximum ice thickness allowed
HIMIN = 0.1  # minimum ice thickness required
HSMAX = 2.0  # maximum snow depth allowed
TIMIN = 173.0  # minimum temperature allowed for snow/ice
ALBFW = 0.06  # albedo for lead
DSI = 1.0 / 0.33
TSICE = 2.7120e2  # temp freezing sea (K)
TFW = -1.8  # TFW - seawater freezing temp (c)
CIMIN = 0.15  # Minimum ice concentration
RHO_SNO = 330.0  # Density of snow over sea ice (kg/m^3)
RHO_ICE = 917.0  # Density of ice   (kg/m^3)
DSDW = RHO_SNO / constants.RHO_H2O  # Ratio of snow to water density
DIDW = RHO_ICE / constants.RHO_H2O  # Ratio of ice to water density
CI = 2054.0  # heat capacity of fresh ice (j/kg/k)
LI = 3.34e5  # latent heat of fusion of fresh ice (j/kg-ice)
SI = 1.0  # salinity of sea ice
MU = 0.054  # relates freezing temp to salinity
KI = 2.03  # conductivity of ice  (w/mk)
KS = 0.31  # conductivity of snow   (w/mk)
I0 = 0.3  # ice surface penetrating solar fraction
TFI = -MU * SI  # sea ice freezing temp = -MU*salinity
DICI = RHO_ICE * CI  # Ice density times heat capacity (j/m^3/k)
DILI = RHO_ICE * LI  # Ice density times latent heat of fusion (j/m^3)
DSLI = RHO_SNO * LI  # Snow density times latent heat of fusion (j/m^3)
# Alternative Units and Precision in Physics:
CPH2O1 = 4.218e3  # Specific heat of water in J/kg/K
CPH2O2 = 4.2e6

# Constants used in interpolating radiation onto physics timestep:
F_EPS = 0.0001

# SFC_DIFF Constants
ALPHA = 5.0
A0 = -3.975
CA = 0.4
A1 = 12.32
ALPHA4 = 4.0 * ALPHA
B1 = -7.755
B2 = 6.041
ALPHA2 = ALPHA + ALPHA
BETA = 1.0
A0P = -7.941
A1P = 24.75
B1P = -8.705
B2P = 7.899
ZTMIN = -999.0
CHARNOCK = 0.014
VIS = 1.4e-5
RNU = 1.51e-5
VISI = 1.0 / VIS
A2 = 17.2693882
A3 = 273.16
A4 = 35.86
A23M4 = A2 * (A3 - A4)
NSOLD = 4
GS1 = 9.8
GS2 = 9.81
LSUBC = 2.501e6
LSUBF = 3.335e5
LSUBS = 2.83e6
ELCP = 2.4888e3
RD1 = 287.04
CP1 = 1004.5
CP2 = 1004.0
CPH2O1 = 4.218e3
CPH2O2 = 4.2e6
CPVAP = 1.8460e3
CPICE = 2.1060e3
CPICE1 = 2.106e6
SIGMA1 = 5.67e-8
ZBOT = -8.0
SALP = 4.0
CFACTR = 0.5
CMCMAX = 0.5e-3
SBETA = -2.0
RSMAX = 5000.0
TOPT = 298.0
REFDK = 2.0e-6
FRZK = 0.15
FXEXP = 2.0
REFKDT = 3.0
CZIL = 0.075
CSOIL = 2.00e6
PSAT = 6.1078e2  # Pressure at H20 3pt (Pa)
