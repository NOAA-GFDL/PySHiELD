import copy
import typing

import numpy as np

import ndsl.constants as constants
import pySHiELD.functions.microphysics_funcs as functions
from ndsl import Quantity, QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval, sin, cos, acos, min, max
from ndsl.dsl.typing import Float, FloatFieldIJ, BoolFieldIJ

CCR = 1.3e-6  # iteration limit
CYEAR = 365.25  # days of year
SVT6  = 78.035  # days between perihelion passage and march equinox of 1900
TPP = 1.55  # days between epoch and perihelion passage of 1900
CZLIMT = 0.0001  # ~ cos(89.99427)
JDOR  = 2415020  # jd of epoch which is january 0, 1900 at 12 hours ut

CON_SOLR = 1.3608e+3
SMON_SAV = [CON_SOLR for i in range(12)]

def solar(
    jd: int,
    fjd: float,
):
    """
    !  ===================================================================  !
    !                                                                       !
    !  solar computes radius vector, declination and right ascension of     !
    !  sun, and equation of time.                                           !
    !                                                                       !
    !  inputs:                                                              !
    !    jd       - julian day                                              !
    !    fjd      - fraction of the julian day                              !
    !                                                                       !
    !  outputs:                                                             !
    !    r1       - earth-sun radius vector                                 !
    !    dlt      - declination of sun in radians                           !
    !    alp      - right ascension of sun in radians                       !
    !                                                                       !
    !  module variables:                                                    !
    !    sollag   - equation of time in radians                             !
    !    sindec   - sine of declination angle                               !
    !    cosdec   - cosine of declination angle                             !
    !                                                                       !
    !  usage:    call solar                                                 !
    !                                                                       !
    !  external subroutines called: none                                    !
    !                                                                       !
    !  ===================================================================  !
    """
    # computes time in julian centuries after epoch

    t1 = float(jd - JDOR) / 36525.0

    # computes length of anomalistic and tropical years (minus 365 days)

    year = 0.25964134e0 + 0.304e-5 * t1
    tyear= 0.24219879E0 - 0.614e-5 * t1

    # computes orbit eccentricity and angle of earth's inclination from t

    ec   = 0.01675104e0 - (0.418e-4 + 0.126e-6 * t1) * t1
    angin= 23.452294e0 - (0.0130125e0 + 0.164e-5 * t1) * t1

    ador = JDOR
    jdoe = ador + (SVT6 * CYEAR) / (year - tyear)

    # deleqn is updated svt6 for current date

    deleqn= float(jdoe - jd) * (year - tyear) / CYEAR
    year  = year + 365.0
    sni   = np.sin( angin / (180. / constants.PI) )
    tini  = 1.0 / np.tan( angin / (180. / constants.PI) )
    er    = np.sqrt( (1.0 + ec) / (1.0 - ec) )
    qq    = deleqn * 2. * constants.PI / year

    # determine true anomaly at equinox
    e1 = 1.0
    cd = 1.0
    iter = 0

    while cd > CCR:
        ep = e1 - (e1 - ec*np.sin(e1) - qq) / (1.0 - ec*np.cos(e1))
        cd = abs(e1 - ep)
        e1 = ep
        iter = iter + 1
        if iter > 10:
            print(f"ITERATION COUNT FOR LOOP 32 = {iter}")
            print(f"E, EP, CD =', {e1}, {ep}, {cd}")
            break

    eq   = 2.0 * np.atan( er * np.tan( 0.5*e1 ) )

    # date is days since last perihelion passage

    dat  = float(jd - JDOR) - TPP + fjd
    date = dat % year

    # solve orbit equations by newton's method

    em   = 2. * constants.PI * date / year
    e1   = 1.0
    cr   = 1.0
    iter = 0

    while cr > CCR:
        ep   = e1 - (e1 - ec*np.sin(e1) - em) / (1.0 - ec*np.cos(e1))
        cr   = abs(e1 - ep)
        e1   = ep
        iter = iter + 1

        if iter > 10:
            print(f"ITERATION COUNT FOR LOOP 31 = {iter}")
            break

    w1   = 2.0 * np.atan( er * np.tan( 0.5*e1 ) )

    r1   = 1.0 - ec*np.cos(e1)

    sindec = sni * sin(w1 - eq)
    cosdec = np.sqrt( 1.0 - sindec*sindec )

    dlt  = np.asin( sindec )
    alp  = np.asin( np.tan(dlt)*tini )

    tst  = cos( w1 - eq )
    if tst < 0.0:
        alp = constants.PI - alp
    if alp < 0.0:
        alp = alp + 2. * constants.PI

    sun  = 2. * constants.PI * (date - deleqn) / year
    if sun < 0.0:
        sun = sun + 2. * constants.PI
    sollag = sun - alp - 0.03255e0
    return r1, dlt, alp, sollag, sindec, cosdec

def solar_update(
    jdate: list[int],
    sdate: list[int],
    kyear: int,
    deltsw: float,
    deltim: float,
    lsol_chg: bool,
    iyr_sav: int,
    isolflg: bool,
):
    """
    !  ===================================================================  !
    !                                                                       !
    !  sol_update computes solar parameters at forecast time                !
    !                                                                       !
    !  inputs:                                                              !
    !     jdate(8)- ncep absolute date and time at fcst time                !
    !                (yr, mon, day, t-zone, hr, min, sec, mil-sec)          !
    !     sdate(8)- absolute date and time for solar calculations           !
    !                (yr, mon, day, t-zone, hr, min, sec, mil-sec)          !
    !     kyear   - usually kyear=jdate(1). if not, it is for hindcast mode,!
    !               and it is usually the init cond time and serves as the  !
    !               upper limit of data can be used.                        !
    !     deltsw  - time duration in seconds per sw calculation             !
    !     deltim  - timestep in seconds                                     !
    !     lsol_chg- logical flags for change solar constant                 !
    !     me      - print message control flag                              !
    !                                                                       !
    !  outputs:                                                             !
    !    slag          - equation of time in radians                        !
    !    sdec, cdec    - sin and cos of the solar declination angle         !
    !    solcon        - sun-earth distance adjusted solar constant (w/m2)  !
    !                                                                       !
    !                                                                       !
    !  module variable:                                                     !
    !   solc0   - solar constant  (w/m**2) not adjusted by earth-sun dist   !
    !   isolflg - solar constant control flag                               !
    !             = 0: use the old fixed solar constant                     !
    !             =10: use the new fixed solar constant                     !
    !             = 1: use noaa ann-mean tsi tbl abs-scale with cycle apprx !
    !             = 2: use noaa ann-mean tsi tbl tim-scale with cycle apprx !
    !             = 3: use cmip5 ann-mean tsi tbl tim-scale with cycle apprx!
    !             = 4: use cmip5 mon-mean tsi tbl tim-scale with cycle apprx!
    !   solar_fname-external solar constant data table                      !
    !   sindec  - sine of the solar declination angle                       !
    !   cosdec  - cosine of the solar declination angle                     !
    !   anginc  - solar angle increment per iteration for cosz calc         !
    !   nstp    - total number of zenith angle iterations                   !
    !   smon_sav- saved monthly solar constants (isolflg=4 only)            !
    !   iyr_sav - saved year  of data previously used                       !
    !                                                                       !
    !  usage:    call sol_update                                            !
    !                                                                       !
    !  subprograms called:  solar, prtime                                   !
    !                                                                       !
    !  external functions called: iw3jdn                                    !
    !                                                                       !
    !  ===================================================================  !
    """
    iyear = sdate[0]
    imon  = sdate[1]
    iday  = sdate[2]
    ihr   = sdate[4]
    imin  = sdate[5]
    isec  = sdate[6]
    if lsol_chg:  # get solar constant from data table
        if iyear == iyr_sav: # same year, no new reading necessary
            if isolflg:
                solc0 = SMON_SAV[imon]
        else:  # need to read in new data
            iyr_sav = iyear
            # TODO finish all of this
    
    # calculate forecast julian day and fraction of julian day
    jd1 = iw3jdn(iyear, imon, iday) # TODO!!!!

    pass
    return slag, sdec, cdec, solcon


def coszmn(
    xlon: FloatFieldIJ,
    sinlat: FloatFieldIJ,
    coslat: FloatFieldIJ,
    solhr: Float,
    sollat: Float,
    sollag: Float,
    sindec: Float,
    cosdec: Float,
    anginc: Float,
    coszen: FloatFieldIJ,
    coszdg: FloatFieldIJ,
    daymask: BoolFieldIJ
):
    """
    !  ===================================================================  !
    !                                                                       !
    !  coszmn computes mean cos solar zenith angle over sw calling interval !
    !                                                                       !
    !  inputs:                                                              !
    !    xlon  (IM)    - grids' longitudes in radians, work both on zonal   !
    !                    0->2pi and -pi->+pi arrangements                   !
    !    sinlat(IM)    - sine of the corresponding latitudes                !
    !    coslat(IM)    - cosine of the corresponding latitudes              !
    !    solhr         - time after 00z in hours                            !
    !    IM            - num of grids in horizontal dimension               !
    !    me            - print message control flag                         !
    !    daily_mean    - replace cosz with daily mean value                 !
    !    fixed_sollat  - fix solar latitude                                 !
    !    sollat        - latitude the solar position fixed to (-90. to 90.) !
    !                                                                       !
    !  outputs:                                                             !
    !    coszen(IM)    - average of cosz for daytime only in sw call interval
    !    coszdg(IM)    - average of cosz over entire sw call interval       !
    !                                                                       !
    !  module variables:                                                    !
    !    sollag        - equation of time                                   !
    !    sindec        - sine of the solar declination angle                !
    !    cosdec        - cosine of the solar declination angle              !
    !    anginc        - solar angle increment per iteration for cosz calc  !
    !    nstp          - total number of zenith angle iterations            !
    !                                                                       !
    !  usage:    call comzmn                                                !
    !                                                                       !
    !  external subroutines called: none                                    !
    !                                                                       !
    !  ===================================================================  !
    """
    from __externals__ import daily_mean, fixed_sollat, nstp

    with computation(PARALLEL), interval(0, 1):
        daymask = False
        
        solar_angle = (constants.PI / 12.0) * (solhr - 12.0) #  solar angle at present time
        rstp = 1. / nstp

        coszen = 0.0
        istsun = 0

        iter = 0
        while iter > nstp:
            cns = solar_angle + (iter-1.0)* anginc + sollag
            if fixed_sollat:
                ss = sin(sollat * constants.PI / 180.0) * sindec
                cc = cos(sollat * constants.PI / 180.0) * cosdec
            else:
                ss = sinlat * sindec
                cc = coslat * cosdec

            if iter == 0:
                h = acos(min(max(-ss/cc, -1.), 1.))
                coszenm = ss * h / constants.PI + cc * (
                    sin(h) - sin(-h)
                ) / (2 * constants.PI)
            coszn = ss + cc * cos(cns + xlon)
            coszen = coszen + max(0.0, coszn)
            if (coszn > CZLIMT):
                istsun = istsun + 1
            iter += 1

        coszdg = coszen * rstp
        if istsun > 0:
            coszen = coszen / istsun
        
        if daily_mean:
            coszdg = coszenm
            coszen = coszenm
        if coszen >= 0.0001:
            daymask = True

