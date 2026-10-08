import numpy as np
import pytest

from motion_tracker.ingest import FrameReader
from motion_tracker.model import Track
from motion_tracker.synthetic import RamSpec, write_clip
from motion_tracker.tracking import PointSpec, track_points


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clips") / "ram.mp4"
    spec = write_clip(path, RamSpec())
    return path, spec


@pytest.fixture(scope="session")
def tracked(clip):
    """One CSRT run over the main clip, shared by tests (tracking is the slow part)."""
    path, spec = clip
    (tr,) = track_points(FrameReader(str(path)), [PointSpec("ram", spec.x, spec.y_top)])
    return tr


@pytest.fixture(scope="session")
def small_spec():
    return RamSpec(width=640, height=360, x=320, y_top=100, y_bottom=260, dot_radius=7, seconds=5)


@pytest.fixture(scope="session")
def small_clip(tmp_path_factory, small_spec):
    path = tmp_path_factory.mktemp("small") / "small.mp4"
    write_clip(path, small_spec)
    return path, small_spec


def truth_track(spec: RamSpec, hz: float = 2000.0) -> Track:
    """Noise-free analytic ram track on a dense clock, used to derive ground-truth event times."""
    t = np.arange(0, spec.seconds, 1 / hz)
    y = spec.y_at(t)
    n = len(t)
    return Track("truth", frame=np.arange(n, dtype=np.int32), t=t, x=np.full(n, spec.x), y=y,
                 conf=np.ones(n, np.float32), flags=np.zeros(n, np.uint8))
