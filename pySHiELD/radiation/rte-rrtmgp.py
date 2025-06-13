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
        For RTE-RRTMGP we need level and layer profiles of temperature and pressure,
        the species used for the spectral calculations:
            humidity, cloud water (and size), cloud ice (and size),
            CO2, O3, N2O, N2, O2, CH4, CO
        albedo and surface emissivities, the solar zenith angle, and what points are
        in daylight (to determine if the SW solver is called for them)
        Here we extract that info from the model state and time.
        """
        pass

    def _prep_outputs(self):
        pass

    def step_radiation(self, state: PhysicsState):
        pass
