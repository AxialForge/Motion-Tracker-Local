"""Tracking plugin interface and the multi-point tracking run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..ingest import FrameReader
from ..model import Track


@dataclass(frozen=True)
class PointSpec:
    """A point the user placed on a paused frame."""

    name: str
    x: float
    y: float
    seed_frame: int = 0
    algo: str = "csrt"  # csrt (precision) | kcf (speed)
    box: int = 32  # tracked box size in px
    mode: str = "dot_tracking"
    options: dict = field(default_factory=dict)  # mode-specific settings (see each plugin)


@dataclass(frozen=True)
class Sample:
    x: float
    y: float
    conf: float
    flags: int


class TrackerPlugin(Protocol):
    """One tracking mode. Instances are per point and stateful."""

    def init(self, frame: np.ndarray) -> Sample | None:
        """Seed on the clicked frame. May return a refined seed sample (else the click is used)."""
        ...
    def update(self, frame: np.ndarray) -> Sample: ...


class FrameConsumer(Protocol):
    """Something that watches every frame and yields tracks at the end (e.g. the billet detector)."""

    def feed(self, index: int, t: float, image: np.ndarray) -> None: ...
    def finish(self) -> list[Track]: ...


_PLUGINS: dict[str, Callable[[PointSpec], TrackerPlugin]] = {}
_KINDS: dict[str, str] = {}


def register_plugin(mode: str, factory: Callable[[PointSpec], TrackerPlugin], kind: str = "dot") -> None:
    _PLUGINS[mode] = factory
    _KINDS[mode] = kind


def get_plugin(mode: str) -> Callable[[PointSpec], TrackerPlugin]:
    try:
        return _PLUGINS[mode]
    except KeyError:
        raise ValueError(f"unknown tracking mode {mode!r}; available: {sorted(_PLUGINS)}") from None


def track_points(
    reader: FrameReader,
    specs: list[PointSpec],
    start: int = 0,
    stop: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    consumers: list[FrameConsumer] | None = None,
) -> list[Track]:
    """Track any number of points in one decode pass, each with its own algorithm.

    Runs over [start, stop) so long clips can be processed in segments; `cancel()` returning
    True ends the run cleanly and returns what was tracked so far. `consumers` see every frame in
    the same decode pass and add their own tracks (billet blobs need no clicks).
    """
    total = (stop if stop is not None else reader.info.frame_count) - start
    plugins = {s.name: get_plugin(s.mode)(s) for s in specs}
    rows: dict[str, list[tuple]] = {s.name: [] for s in specs}
    for fr in reader.frames(start, stop):
        for s in specs:
            if fr.index < s.seed_frame:
                continue
            p = plugins[s.name]
            if fr.index == s.seed_frame:
                sm = p.init(fr.image) or Sample(s.x, s.y, 1.0, 0)  # seeded on the clicked frame
            else:
                sm = p.update(fr.image)
            rows[s.name].append((fr.index, fr.t, sm.x, sm.y, sm.conf, sm.flags))
        for c in consumers or ():
            c.feed(fr.index, fr.t, fr.image)
        if progress:
            progress(fr.index - start + 1, total)
        if cancel and cancel():
            break
    tracks = []
    for s in specs:
        r = rows[s.name]
        a = np.array(r, dtype=float) if r else np.empty((0, 6))
        tracks.append(Track(
            name=s.name, kind=_KINDS.get(s.mode, "dot"), mode=s.mode,
            params={"algo": s.algo, "box": s.box, "seed_frame": s.seed_frame, "seed_xy": [s.x, s.y], **s.options},
            frame=a[:, 0].astype(np.int32), t=a[:, 1], x=a[:, 2], y=a[:, 3],
            conf=a[:, 4].astype(np.float32), flags=a[:, 5].astype(np.uint8),
        ))
    for c in consumers or ():
        tracks += c.finish()
    return tracks
