"""Built-in SpatialCraft dataset adapters."""

from .erqa import ERQAAdapter
from .omni3d import Omni3DAdapter
from .robospatial import RoboSpatialAdapter
from .sat import SATAdapter
from .viewspatial import ViewSpatialAdapter

__all__ = [
    "ERQAAdapter",
    "Omni3DAdapter",
    "RoboSpatialAdapter",
    "SATAdapter",
    "ViewSpatialAdapter",
]
