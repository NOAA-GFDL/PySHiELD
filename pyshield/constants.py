import math

import ndsl.constants as constants


TICE0 = constants.TICE - 0.01

C_ICE = 2.106e3  # Heat capacity of ice at 0 degrees Celsius
C_LIQ = 4.218e3  # Heat capacity of water at 0 degrees Celsius
DC_ICE = C_LIQ - C_ICE  # Isobaric heating / cooling (J/kg/K)
DC_VAP = constants.CP_VAP - C_LIQ  # Isobaric heating / cooling (J/kg/K)
D2ICE = DC_VAP + DC_ICE  # Isobaric heating / cooling (J/kg/K)
LV0 = (
    constants.HLV - DC_VAP * TICE0
)  # 3148711.3338762247, evaporation latent heat coefficient at 0 degrees Kelvin
LI00 = (
    constants.HLF - DC_ICE * TICE0
)  # -242413.92000000004, fusion latent heat coefficient at 0 degrees Kelvin
LI2 = (
    LV0 + LI00
)  # 2906297.413876225, sublimation latent heat coefficient at 0 degrees Kelvin
LI0 = constants.HLF - DC_ICE * TICE0

RHO_0 = 1.0  # reference air density (kg/m^3), ref: IFS
RHO_W = 1.0e3  # density of cloud water (kg/m^3)
RHO_I = 9.17e2  # density of cloud ice (kg/m^3)
RHO_R = 1.0e3  # density of rain (Lin et al. 1983) (kg/m^3)
RHO_S = 1.0e2  # density of snow (Lin et al. 1983) (kg/m^3)
RHO_G = 4.0e2  # density of graupel (Rutledge and Hobbs 1984) (kg/m^3)
RHO_H = 9.17e2  # density of hail (Lin et al. 1983) (kg/m^3)

VISD = 1.717e-5  # dynamics viscosity of air at 0 deg C and 1000 hPa
# (Mason, 1971) (kg/m/s)
VISK = 1.35e-5  # kinematic viscosity of air at 0 deg C  and 1000 hPa
# (Mason, 1971) (m^2/s)
VDIFU = 2.25e-5  # diffusivity of water vapor in air at 0 deg C  and 1000 hPa
# (Mason, 1971) (m^2/s)
TCOND = 2.40e-2  # thermal conductivity of air at 0 C and 1000 hPa
# (Mason, 1971) (J/m/s/K)
SCM3 = math.exp(1.0 / 3 * math.log(VISK / VDIFU))  # Schmidt number, Sc ** (1 / 3)
# Lin et al. (1983)

QCMIN = 1.0e-15  # min value for cloud condensates (kg/kg)
QFMIN = 1.0e-8  # min value for sedimentation (kg/kg)
DT_FR = 8.0  # t_wfr - dt_fr: minimum temperature water can exist
# (Moore and Molinero 2011)
CDG = 3.15121  # drag coefficient of graupel (Locatelli and Hobbs, 1974)
CDH = 0.5  # drag coefficient of hail (Heymsfield and Wright, 2014)

DZ_MIN_FLIP = 1.0e-2  # used for correcting flipped height (m)

# Terminal Velocity Parameters, Lin et al. (1983)
GCON = (4.0 * constants.GRAV * RHO_G / (3.0 * CDG * RHO_0)) ** 0.5
HCON = (4.0 * constants.GRAV * RHO_H / (3.0 * CDH * RHO_0)) ** 0.5
