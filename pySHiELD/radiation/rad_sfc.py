import os
import re
from pathlib import Path

import numpy as np

import ndsl.constants as constants
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int
from ndsl.logging import ndsl_log

EMS_REF = [0.97, 0.95, 0.94, 0.90, 0.93, 0.96, 0.96, 0.99]
IMXEMS = 360
JMXEMS = 180

def read_sfcemis_file(datafile: Path):
    pass

def map_sfc_to_grid(
    sfc_data: np.ndarray,
    gridlon: np.ndarray,
    gridlat: np.ndarray,
) -> np.ndarray:
    """
    Takes an input array and assigns values to the grid resolution based on lon/lat.
    note: this is a simple mapping method, an upgrade is needed if
    the model grid is much coarser than the 1-deg data resolution.
    """
    sfc_grid_data = np.zeros_like(gridlon)
    nx = gridlon.shape[0]
    ny = gridlon.shape[1]
    degres_i = sfc_data.shape[0] / 360.0
    degres_j = sfc_data.shape[1] / 180.0
    for i, j in np.nindex(sfc_grid_data.shape):
        lon = gridlon[i, j] if gridlon[i, j] >= 0 else gridlon[i, j] + (2.0 * constants.PI)
        lat = constants.PI - gridlat[i, j]
        i2 = int((lon * (180./constants.PI)) // degres_i)
        j2 = int((lat * (180./constants.PI)) // degres_j)
        sfc_grid_data[i, j] = sfc_grid_data[i2, j2]
    return sfc_grid_data

def sfc_init(
    ialbflg: Int,
    iemsflg: Int,
    ldisable_radiation_quasi_sea_ice: bool,
    sfcemis_datafile: Path,
    sfcemis_data: np.ndarray,
):
    # Initialization of surface albedo section
    # physparam::ialbflg 
    # - 0: using climatology surface albedo scheme for SW
    # - 1: using MODIS based land surface albedo for SW
    # - 2: using land surface model albedo for SW 
    if ialbflg == 0:
        ndsl_log.info("Using climatology surface albedo scheme for sw")
    elif ialbflg == 1:
        ndsl_log.info("Using MODIS based land surface albedo for sw")
    elif ialbflg == 2:
        ndsl_log.info("Using Albedo From Land Model")
    else:
        raise ValueError(f"ialbflg must be 0, 1, or 2, got {ialbflg}")
    # physparam::ldisable_radiation_quasi_sea_ice 
    # - = .false.: use a sea-ice-like albedo and emissivity for below
    #     freezing ocean grid cells.
    # - = .true.:  treat all ocean grid cells as if they were
    #     above freezing when determing the albedo and emissivity
    if ldisable_radiation_quasi_sea_ice:
        ndsl_log.info("Disabling radiation quasi-sea-ice")
    else:
        ndsl_log.info("Enable radiation quasi-sea-ice")

    # Initialization of surface emissivity section
    # physparam::iemsflg
    # - = 0: fixed SFC emissivity at 1.0
    # - = 1: input SFC emissivity type map from "semis_file"
    # - = 2: using SFC emissivity from land model
    iemslw = iemsflg % 10  # emissivity control
    if iemslw == 0:
        ndsl_log.info("Using Fixed Surface Emissivity = 1.0 for lw")
    elif iemslw == 1:
        ndsl_log.info(f"Using Varying Surface Emissivity for lw from {sfcemis_datafile}")
        if not sfcemis_datafile.is_file():
            raise FileExistsError(f"{sfcemis_datafile} does not exist")
        sfcemis_data = np.genfromtxt(sfcemis_datafile, dtype=int, delimiter=1, skip_header=1).reshape(IMXEMS, JMXEMS)
    elif iemslw == 2:
        ndsl_log.info("Using Surface Emissivity From Land Model")
    else:
        raise ValueError(f"iemslw must be 0, 1, or 2, got {iemslw}")

def set_albedo():
    pass

def set_sfcemis(
    gridlon: np.ndarray,
    gridlat: np.ndarray,
    islmsk: np.ndarray,
    snowf: np.ndarray,
    sncovr: np.ndarray,
    zorlf: np.ndarray,
    tskin: np.ndarray,
    tairf: np.ndarray,
    hprif: np.ndarray,
    iemslw: Int,
    ialbflg: Int,
    ldisable_radiation_quasi_sea_ice: bool,
    sfcemis: np.ndarray,
    sfcemis_data: np.ndarray,
    sfcemis_lsm: np.ndarray,
):
    """
    !  ===================================================================  !
    !                                                                       !
    !  this program computes surface emissivity for lw radiation.           !
    !                                                                       !
    !  usage:         call setemis                                          !
    !                                                                       !
    !  subprograms called:  none                                            !
    !                                                                       !
    !  ====================  defination of variables  ====================  !
    !                                                                       !
    !  inputs:                                                              !
    !     xlon  (IMAX)  - longitude in radiance, ok for both 0->2pi or      !
    !                     -pi -> +pi ranges                                 !
    !     xlat  (IMAX)  - latitude  in radiance, default to pi/2 -> -pi/2   !
    !                     range, otherwise see in-line comment              !
    !     slmsk (IMAX)  - sea(0),land(1),ice(2) mask on fcst model grid     !
    !     snowf (IMAX)  - snow depth water equivalent in mm                 !
    !     sncovr(IMAX)  - ialbflg=1: snow cover over land in fraction       !
    !     zorlf (IMAX)  - surface roughness in cm                           !
    !     tsknf (IMAX)  - ground surface temperature in k                   !
    !     tairf (IMAX)  - lowest model layer air temperature in k           !
    !     hprif (IMAX)  - topographic sdv in m                              !
    !     lsmemiss(IMAX)- emissivity from lsm                               !
    !                                                                       ! 
    !     IMAX          - array horizontal dimension                        !
    !                                                                       !
    !  outputs:                                                             !
    !     sfcemis(IMAX) - surface emissivity                                !
    !                                                                       !
    !  -------------------------------------------------------------------  !
    !                                                                       !
    !  surface type definations:                                            !
    !     1. open water                   2. grass/wood/shrub land          !
    !     3. tundra/bare soil             4. sandy desert                   !
    !     5. rocky desert                 6. forest                         !
    !     7. ice                          8. snow                           !
    !                                                                       !
    !  input index data lon from 0 towards east, lat from n to s            !
    !                                                                       !
    !  ====================    end of description    =====================  !
    """
    if iemslw == 0:
        sfcemis = Float(1.0)
        return
    
    sfcemis[islmsk == 0] = Float(EMS_REF[0])  # sea
    sfcemis[islmsk == 2] = Float(EMS_REF[6])  # sea-ice
    if iemslw == 1:
        sfcemis_grid_data = map_sfc_to_grid(sfcemis_data, gridlon, gridlat)
        for i, j in np.nindex(sfcemis.shape):
            if islmsk[i, j] == 1:
                idx = max(1, sfcemis_grid_data[i, j])
                if idx >= 6: idx = 2
                sfcemis[i, j] = Float(EMS_REF[idx])
            if ialbflg == 1 and islmsk[i, j] == 1:  # input land area snow cover
                fsno = 1. - sncovr[i, j]
                sfcemis[i, j] = Float(sfcemis[i, j] * fsno) + Float(EMS_REF[7] * sncovr[i, j])
            else:  # compute snow cover from snow depth
                if snowf[i, j] > 0.0:
                    asnow = 0.02*snowf[i, j]
                    argh  = min(0.50, max(.025, 0.01*zorlf[i, j]))
                    hrgh  = min(1.0, max(0.20, 1.0577-1.1538e-3*hprif[i, j] ) )
                    fsno0 = asnow / (argh + asnow) * hrgh
                    if islmsk[i, j] == 0 and (tskin[i, j] > 271.2 or ldisable_radiation_quasi_sea_ice):
                        fsno0 = 0.0
                    fsno1 = 1.0 - fsno0
                    sfcemis[i, j] = Float(sfcemis[i, j] * fsno1) + Float(EMS_REF[7] * fsno0)
    else:  # iemslw == 2
        sfcemis[islmsk == 1] = sfcemis_lsm[islmsk == 1]  # land from LSM
    pass