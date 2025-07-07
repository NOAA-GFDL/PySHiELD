from pathlib import Path

import numpy as np

import ndsl.constants as constants
from ndsl.dsl.gt4py import PARALLEL, acos, computation, cos, interval, max, min, sin
from ndsl.dsl.typing import BoolFieldIJ, Float, FloatFieldIJ, Int
from ndsl.logging import ndsl_log

NDAYS_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 30]
MINYEAR = 1957  # earlist year 2-d co2 data available
NF_VGAS = 10  # number of gas species
IMXCO2  = 24  # input co2 dat lon points
JMXCO2  = 12  # input co2 data lat points
RESCO2=15.0  # horizontal resolution in degree
PRSCO2=788.0  # pressure limitation for 2-d co2 (mb)

def gas_update(iyear, imon, iday, ihour, ioznflg, loz1st, ldoco2, ictmflg, co2dat_file, co2gbl_file):
    """
    !  ===================================================================  !
    !                                                                       !
    !  gas_update reads in 2-d monthly co2 data set for a specified year.   !
    !  data are in a 15 degree lat/lon horizontal resolution.               !
    !                                                                       !
    !  inputs:                                               dimemsion      !
    !     iyear   - year of the requested data for fcst         1           !
    !     imon    - month of the year                           1           !
    !     iday    - day of the month                            1           !
    !     ihour   - hour of the day                             1           !
    !     loz1st  - clim ozone 1st time update control flag     1           !
    !     ldoco2  - co2 update control flag                     1           !
    !     me      - print message control flag                  1           !
    !                                                                       !
    !  outputs: (to the module variables)                                   !
    !    ( none )                                                           !
    !                                                                       !
    !  external module variables:  (in physparam)                           !
    !     ico2flg    - co2 data source control flag                         !
    !                   =0: use prescribed co2 global mean value            !
    !                   =1: use input global mean co2 value (co2_glb)       !
    !                   =2: use input 2-d monthly co2 value (co2vmr_sav)    !
    !     ictmflg    - =yyyy#, data ic time/date control flag               !
    !                  =   -2: same as 0, but superimpose seasonal cycle    !
    !                          from climatology data set.                   !
    !                  =   -1: use user provided external data for the fcst !
    !                          time, no extrapolation.                      !
    !                  =    0: use data at initial cond time, if not existed!
    !                          then use latest, without extrapolation.      !
    !                  =    1: use data at the forecast time, if not existed!
    !                          then use latest and extrapolate to fcst time.!
    !                  =yyyy0: use yyyy data for the forecast time, no      !
    !                          further data extrapolation.                  !
    !                  =yyyy1: use yyyy data for the fcst. if needed, do    !
    !                          extrapolation to match the fcst time.        !
    !     ioznflg    - ozone data control flag                              !
    !                   =0: use climatological ozone profile                !
    !                   >0: use interactive ozone profile                   !
    !     ivflip     - vertical profile indexing flag                       !
    !     co2dat_file- external co2 2d monthly obsv data table              !
    !     co2gbl_file- external co2 global annual mean data table           !
    !                                                                       !
    !  internal module variables:                                           !
    !     co2vmr_sav - monthly co2 volume mixing ratio     IMXCO2*JMXCO2*12 !
    !     co2cyc_sav - monthly cycle co2 vol mixing ratio  IMXCO2*JMXCO2*12 !
    !     co2_glb    - global annual mean co2 mixing ratio                  !
    !     gco2cyc    - global monthly mean co2 variation       12           !
    !     k1oz,k2oz,facoz                                                   !
    !                - climatology ozone parameters             1           !
    !                                                                       !
    !  usage:    call gas_update                                            !
    !                                                                       !
    !  subprograms called:  none                                            !
    !                                                                       !
    !  ===================================================================  !
    """
    if not ioznflg:
        midmon = NDAYS_MONTH[imon] // 2 + 1
        change = loz1st or ((iday==midmon) and (ihour==0))

        if change:
            if iday < midmon:
                k1oz = ((imon+10) % 12) + 1
                midm = NDAYS_MONTH[k1oz]/2 + 1
                k2oz = imon
                midp = NDAYS_MONTH[k1oz] + midmon
            else:
                k1oz = imon
                midm = midmon
                k2oz = (imon % 12) + 1
                midp = NDAYS_MONTH[k2oz]/2 + 1 + NDAYS_MONTH[k1oz]
        if (iday < midmon):
            id = iday + NDAYS_MONTH[k1oz]
        else:
            id = iday
        facoz = float(id - midm) / float(midp - midm)

    if ictmflg < 0:  # use user provided external data
        lextpl = False  # no time extrapolation
        idyr = iyear  # use the model year
    else:  # use historically observed data
        lextpl = ( ictmflg%10) == 1  # flag for data extrapolation
        idyr   = ictmflg / 10  # year of data source used
        if idyr == 0:  # not specified, use model year
            idyr = iyear 

    # for data earlier than MINYEAR (1957), the data are in
    # the form of semi-yearly global mean values.  otherwise,
    # data are monthly mean in horizontal 2-d map.
    if idyr < MINYEAR and ictmflg > 0:
        raise NotImplementedError(f"")

def read_global_annual_co2(co2gbl_file: Path):
    """
    Function to read a text file of CO2 global half-yearly means and growth rates into a model.
    Assumes a format of:
    HEADER
    yyyy     co2  co2   growth_rate   growth_rate
    Returns a dictionary mapping year to data
    """
    global_annual_co2_data = {}
    co2_file = open(co2gbl_file)
    for i, line in enumerate(co2_file):
        if line.split():  # skip blank lines
            table_dat = line.split()
            if i == 0:
                continue  # skip the header
            elif table_dat[0] == "---":
                break  # end at the dashes
            else:
                year = int(table_dat[0])
                global_annual_co2_data[year] = [float(data) for data in table_dat[1:]]
    co2_file.close()
    return global_annual_co2_data

def read_monthly_resolved_co2(co2dat_file: Path):
    """
    Function to read a text file of 15-degree CO2 monthly means into a model.
    Assumes a format of:
    HEADER
    24 values (15 degree resolution longitudinally)
    There should be 144 lines (12 months, resolved 15 degrees latitudinally)
    Assuming here that it's grouped by month, so 12 lines of January,
    12 of February, and so on?

    Returns a dictionary mapping month to 2D data
    """
    resolved_monthly_co2_data = {}
    co2_file = open(co2dat_file)
    imonth = 1
    for i, line in enumerate(co2_file):
        if line.split():  # skip blank lines
            table_dat = line.split()
            if i == 0:
                continue  # skip the header
            else:
                ilat = (i - 1) % 12
                if ilat == 0:
                    co2_data = []
                co2_data.append([float(data) for data in table_dat])
                if ilat == 11:
                    resolved_monthly_co2_data[imonth] = co2_data
                    imonth += 1
    co2_file.close()
    return resolved_monthly_co2_data

def read_monthly_cycle_co2(co2cyc_file: Path):
    """
    Function to read a text file of 15-degree CO2 monthly deviations into a model.
    Assumes a format of:
    HEADER
    MONTH
    12 lines of 24 values (15 degree resolution longitudinally)
    repeated for each month of the year

    Returns a dictionary mapping month to the global mean and 2D data
    """
    resolved_monthly_co2_cycle = {}
    co2_file = open(co2cyc_file)
    for i, line in enumerate(co2_file):
        if line.split():  # skip blank lines
            table_dat = line.split()
            if i < 13:
                continue  # skip the header and mean values
            else:
                if table_dat[0] == "MONTH":
                    imonth = int(table_dat[2])
                    resolved_monthly_co2_cycle[imonth] = {}
                    resolved_monthly_co2_cycle[imonth]["mean"] = float(table_dat[-1])
                    resolved_monthly_co2_cycle[imonth]["data"] = []
                else:
                    resolved_monthly_co2_cycle[imonth]["data"].append([float(data) for data in table_dat])
    co2_file.close()
    return resolved_monthly_co2_cycle

def get_co2(iyear, imon, iday):
    if imon < 7 or (imon == 7 and iday  <= 2):  # Assumes July 2 is the yearly midpoint
        midyear = False
    else:
        midyear = True

    