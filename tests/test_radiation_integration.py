import pytest

@pytest.mark.parameterize("datapath", ["path/to/data.nc"])
def test_ingest_data(datapath):
    pass