import dataclasses

import pyrte_rrtmgp as rad
import xarray as xr

from ndsl import Float
from pySHiELD import PhysicsState


@dataclasses.dataclass
class RadiationConfig:
    dt_atmos: Float
    fhswr: Float
    fhlwr: Float
    aerosol_file: str


class RadiationDriver:
    def __init__(self, config: RadiationConfig):
        pass

    def _accumulate_radiation_inputs(state: PhysicsState):
        """
        For RTE-RRTMGP we need profiles of temperature, pressure, and the species used
        for the spectral calculations (humidity, C02, etc.).
        Here we extract that info from the model state.
        """
        pass

    def step_radiation(self, state: PhysicsState):
        pass
