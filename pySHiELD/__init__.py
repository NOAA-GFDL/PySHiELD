from ._config import PHYSICS_PACKAGES, PhysicsConfig
from .physics_state import PhysicsState
from .stencils.physics import Physics


__all__ = list(key for key in locals().keys() if not key.startswith("_"))
__version__ = "0.2.0"

"""
PHYSICS_PACKAGES: Enum of available physics packages
PhysicsConfig: Class containing physics variables and model configurations
PhysicsState: Class containing the state for the physics schemes
Physics: Class that specifies and executes the subgrid physics schemes
"""
