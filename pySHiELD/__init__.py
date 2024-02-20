from ._config import PHYSICS_PACKAGES, PhysicsConfig
from .physics_state import PhysicsState
from .stencils.physics import Physics


__all__ = list(key for key in locals().keys() if not key.startswith("_"))
__version__ = "0.2.0"

"""
PHYSICS_PACKAGES: Enum of available physics packages
PhysicsConfig: Dataclass containing physics variables as defined by configuration yaml
PhysicsState: Dataclass which initializes defines physics variable types
Physics: Class containing methods for updating physics and interactions
"""
