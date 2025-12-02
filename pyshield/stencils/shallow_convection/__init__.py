from ._config import SC_TRACER_DIM, FloatFieldShalConv, ShallowConvectionConfig
from .samfshalconv import ScaleAwareMassFluxShallowConvection
from .state import SAMFShalConvState


"""
ShallowConvectionConfig: Configuration class for the shallow convection
ScaleAwareMassFluxShallowConvection: Shallow convection class
SAMFShalConvState: Class containing arguments for the SAMF shallow convection
FloatFieldShalConv: Type for 4D fields containing tracers for shallow convection
SC_TRACER_DIM: Description for shallow convection tracers to use in a QuantityFactory
"""

__all__ = [
    "ShallowConvectionConfig",
    "ScaleAwareMassFluxShallowConvection",
    "SAMFShalConvState",
    "FloatFieldShalConv",
    "SC_TRACER_DIM",
]
