"""Core data model: tracks, per-sample flags, events, scale calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Per-sample flag bits. Interpolated values are flagged, never passed off as measured.
LOW_CONF = 1
LOST = 2
INTERPOLATED = 4
RECOVERED = 8  # colour-lock re-acquired the point on this frame


@dataclass
class Track:
    """One tracked item on the common video clock (real frame timestamps)."""

    name: str
    kind: str = "dot"  # dot | billet | zone | operator
    mode: str = "dot_tracking"
    params: dict = field(default_factory=dict)
    frame: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    conf: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    flags: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    id: int | None = None

    def __len__(self) -> int:
        return len(self.frame)

    def measured(self) -> np.ndarray:
        """Mask of samples that are real measurements (not lost, not interpolated)."""
        return (self.flags & (LOST | INTERPOLATED)) == 0

    def copy(self) -> Track:
        return Track(
            self.name, self.kind, self.mode, dict(self.params),
            self.frame.copy(), self.t.copy(), self.x.copy(), self.y.copy(),
            self.conf.copy(), self.flags.copy(), self.id,
        )


@dataclass(frozen=True)
class Event:
    name: str
    group: str  # billet | ram | operator | cycle | custom
    item: str  # track name the event came from
    t: float  # seconds on the common clock (sub-frame where possible)
    frame: float  # fractional frame index
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Scale:
    """Pixels to real units, from two clicked points a known distance apart.

    Measurements are in the image plane: the camera must face the motion squarely.
    """

    units_per_px: float
    unit: str = "mm"

    @classmethod
    def from_points(cls, p1, p2, distance: float, unit: str = "mm") -> Scale:
        px = math.dist(p1, p2)
        if px <= 0 or distance <= 0:
            raise ValueError("points must differ and distance must be positive")
        return cls(distance / px, unit)
