from dataclasses import replace

import numpy as np
import pytest

from conftest import truth_track
from motion_tracker.events import detect_ram
from motion_tracker.ingest import FrameReader
from motion_tracker.model import LOST
from motion_tracker.synthetic import RamSpec, write_clip
from motion_tracker.tracking import PointSpec, track_points


@pytest.fixture(scope="module")
def edge_run(tmp_path_factory):
    spec = RamSpec(dot=False, seconds=12)
    p = tmp_path_factory.mktemp("edge") / "edge.mp4"
    write_clip(p, spec)
    pt = PointSpec("ram edge", spec.x, spec.y_top + spec.edge_offset, mode="edge_tracking",
                   options={"orientation": "horizontal", "length": 80, "search": 40})
    (tr,) = track_points(FrameReader(str(p)), [pt])
    return spec, tr


def test_edge_tracking_needs_no_dot(edge_run):
    spec, tr = edge_run
    err = np.abs(tr.y - (spec.y_at(tr.t) + spec.edge_offset))
    print(f"edge: error mean {err.mean():.3f}px max {err.max():.3f}px")
    assert tr.kind == "edge" and tr.mode == "edge_tracking" and tr.params["orientation"] == "horizontal"
    assert err.mean() < 0.15 and err.max() < 1.5 and not (tr.flags & LOST).any()
    assert np.ptp(tr.x) == 0  # measures one axis; the other stays at the click


def test_ram_events_from_an_edge_track_match_truth(edge_run):
    spec, tr = edge_run
    truth = detect_ram(truth_track(spec), smooth_window=1)
    got = detect_ram(tr)
    assert len(got.strokes) == len(truth.strokes) == 3
    errs = [abs(getattr(a, f) - getattr(b, f)) for a, b in zip(truth.strokes, got.strokes)
            for f in ("down_start", "bottom_reach", "up_start", "top_reach")]
    print(f"edge ram events: max error {max(errs) * 1000:.1f} ms")
    assert max(errs) < 0.5 / spec.fps


def test_seed_snaps_to_the_edge_and_vertical_orientation(tmp_path):
    spec = RamSpec(dot=False, seconds=2)
    p = tmp_path / "e.mp4"
    write_clip(p, spec)
    # click 6 px off the true edge: the seed sample is the refined edge, not the click
    (tr,) = track_points(FrameReader(str(p)), [PointSpec("e", spec.x, spec.y_top + spec.edge_offset + 6, mode="edge_tracking")], stop=3)
    assert tr.y[0] == pytest.approx(spec.y_top + spec.edge_offset, abs=0.5)
    # the column's left side is a vertical edge, static: tracks X
    (tv,) = track_points(FrameReader(str(p)), [PointSpec("v", spec.x - 60, 60, mode="edge_tracking",
                                                          options={"orientation": "vertical", "length": 40})], stop=20)
    assert np.abs(tv.x - (spec.x - 60)).max() < 1.0 and np.ptp(tv.y) == 0


def test_edge_vanishing_is_flagged_lost(tmp_path):
    spec = RamSpec(dot=False, seconds=2, noise_sigma=0.0)
    p = tmp_path / "e.mp4"
    write_clip(p, spec)
    far = PointSpec("flat", 100, 500, mode="edge_tracking", options={"search": 10})  # nothing but flat background
    (tr,) = track_points(FrameReader(str(p)), [far], stop=30)
    assert tr.conf[1:].max() < 0.5 or (tr.flags[1:] != 0).any()


def test_edge_in_a_saved_run(tmp_path):
    from motion_tracker.runs import Run, apply_run
    spec = RamSpec(dot=False, seconds=5)
    p = tmp_path / "e.mp4"
    write_clip(p, spec)
    run = Run(name="edge", points=[{"name": "ram edge", "mode": "edge_tracking", "x": spec.x, "y": spec.y_top + spec.edge_offset,
                                    "orientation": "horizontal", "length": 80, "search": 40}],
              analyses=[{"type": "ram", "item": "ram edge"}])
    res = apply_run(run, str(p), tmp_path / "e.mtp")
    assert res.warnings == [] and {e.name for e in res.events} >= {"Ram starts downstroke", "Ram reaches bottom"}
