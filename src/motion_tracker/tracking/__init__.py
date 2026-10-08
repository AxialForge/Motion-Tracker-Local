from .base import PointSpec, TrackerPlugin, get_plugin, register_plugin, track_points
from . import dot  # noqa: F401  (registers the dot-tracking plugin)

__all__ = ["PointSpec", "TrackerPlugin", "get_plugin", "register_plugin", "track_points"]
