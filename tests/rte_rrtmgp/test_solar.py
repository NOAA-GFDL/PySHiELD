from pathlib import Path

import pytest

from ...pySHiELD.radiation.rad_astro import read_NOAA_solar_file, sol_init


@pytest.mark.parameterize("datapath", ["path/to/data.nc"])
@pytest.mark.parameterize(
    "year, expected",
    pytest.param(
        2000,
        0.0,
        id="2000",
    ),
    pytest.param(
        1837,
        0.0,
        id="2000",
    ),
)
def test_read_data(datapath, year, expected):
    data_dict = read_NOAA_solar_file(datapath)
    assert data_dict["constants"][year] == expected


@pytest.mark.parameterize("datapath", ["path/to/data.nc"])
@pytest.mark.parameterize(
    "isolar, year, expected",
    pytest.param(
        0,
        1990,
        1.3660e3,
        id="old_constant",
    ),
    pytest.param(
        10,
        1990,
        1.3608e3,
        id="new_constant",
    ),
    pytest.param(
        2,
        1990,
        0.0,
        id="data_read",
    ),
)
def test_solinit(datapath, isolar, year, expected):
    isolflg, sol_const_data, solc0 = sol_init(isolar, Path(datapath), year)
    assert isinstance(sol_const_data, dict)
    assert solc0 == expected


def test_solar_update():
    pass
