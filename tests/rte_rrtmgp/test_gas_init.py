import pytest
import numpy as np
from pathlib import Path
from ...pySHiELD.radiation.rad_gases import read_co2_files, gas_init, co2_update

@pytest.mark.parameterize("datapath", ["path/to/data.nc"])
def test_dataread(datapath):
    co2_glb_data, co2_mvr_data, co2_cyc_data = read_co2_files(Path(datapath))
    pass

@pytest.mark.parameterize("datapath", ["path/to/data.nc"])
@pytest.mark.parameterize(
    "ico2flg, ictmflg, iyear, imonth",
    pytest.param(0, 0, 2000, 1, id="0_0_jan_2000",),
)
def test_gas_init(
    datapath,
    ico2flg,
    ictmflg,
    iyear,
    imonth,
):
    gridlon = np.zeros((20, 20))
    gridlat = np.zeros((20, 20))
    arrays = gas_init(
        Path(datapath),
        ico2flg,
        0,
        ictmflg,
        iyear,
        imonth,
        gridlon,
        gridlat,
    )
    pass

@pytest.mark.parameterize("datapath", ["path/to/data.nc"])
@pytest.mark.parameterize(
    "ico2flg, ictmflg, iyear, imon, saved_month,",
    pytest.param(0, 0, 2000, 1, 2, id="0_0_jan_feb_2000",),
)
def test_co2_update(
    datapath,
    iyear,
    imon,
    saved_month,
    ico2flg,
    ictmflg,
):
    gridlon = np.zeros((20, 20))
    gridlat = np.zeros((20, 20))
    (
        _, _, _, _, _, _, _, _, _,
        co2_glb,
        co2_arr,
        co2_cyc,
        co2_mvr_data,
        co2_glb_data,
        co2_cyc_data,
    ) = gas_init(
        Path(datapath),
        ico2flg,
        0,
        ictmflg,
        iyear,
        imon,
        gridlon,
        gridlat,
    )
    ldoco2 = imon == saved_month
    co2_update(
        iyear,
        imon,
        ico2flg,
        ldoco2,
        ictmflg,
        co2_glb,
        co2_arr,
        co2_cyc,
        gridlon,
        gridlat,
        co2_glb_data,
        co2_mvr_data,
        co2_cyc_data,
    )
    pass
