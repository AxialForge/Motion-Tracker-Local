import numpy as np
import pytest

from conftest import truth_track
from motion_tracker import analysis
from motion_tracker.events import detect_ram, extrema, line_cross, moving_state, zone_transitions
from motion_tracker.model import Track
from motion_tracker.summary import durations_between, stats


def test_ram_events_match_ground_truth_within_half_a_frame(tracked, clip):
    tr, spec = tracked, clip[1]
    truth = detect_ram(truth_track(spec), smooth_window=1)
    got = detect_ram(tr)
    assert len(got.strokes) == len(truth.strokes) == 3
    names = {e.name for e in got.events}
    assert {"Ram starts downstroke", "Ram reaches bottom", "Ram starts upstroke", "Ram reaches top"} <= names
    errs = []
    for a, b in zip(truth.strokes, got.strokes):
        for f in ("down_start", "bottom_reach", "up_start", "top_reach"):
            errs.append(abs(getattr(a, f) - getattr(b, f)))
    print(f"event time error: max {max(errs) * 1000:.1f} ms (frame = 33.3 ms)")
    assert max(errs) < 0.5 / spec.fps


def test_cycle_time_and_stroke_metrics(tracked, clip):
    tr, spec = tracked, clip[1]
    ram = detect_ram(tr)
    assert ram.cycle_times == pytest.approx([spec.cycle] * 2, abs=0.02)
    s = ram.strokes[0]
    assert s.bottom_dwell > spec.dwell  # the band edges add time on each side of the true dwell
    assert 0 < s.downstroke_time < spec.down and s.down_peak_speed > s.down_mean_speed > 0
    assert all(i > 0 for i in ram.idle_top)


def test_flat_signal_yields_no_events():
    n = 100
    tr = Track("f", frame=np.arange(n, dtype=np.int32), t=np.arange(n) / 30, x=np.zeros(n), y=np.full(n, 100.0) + np.random.default_rng(0).normal(0, .1, n),
               conf=np.ones(n, np.float32), flags=np.zeros(n, np.uint8))
    assert detect_ram(tr).events == []


def _line(n=90):
    t = np.arange(n) / 30
    x = np.concatenate([np.full(30, 10.0), np.linspace(10, 200, 30), np.full(30, 200.0)])
    return Track("p", frame=np.arange(n, dtype=np.int32), t=t, x=x, y=np.full(n, 50.0),
                 conf=np.ones(n, np.float32), flags=np.zeros(n, np.uint8))


def test_line_cross_subframe():
    (e,) = line_cross(_line(), axis="x", value=100.0, direction="up")
    assert e.t == pytest.approx(30 / 30 + (100 - 10) / (190 / 29) / 30 * 1, abs=0.02)
    assert line_cross(_line(), axis="x", value=100.0, direction="down") == []


def test_zone_transitions():
    ev = zone_transitions(_line(), (90, 0, 150, 100), "S1")
    assert [e.name for e in ev] == ["p enters S1", "p leaves S1"]


def test_moving_state():
    ev = moving_state(_line(), speed_threshold=50.0)
    assert [e.name for e in ev] == ["p starts moving", "p stops moving"]
    assert ev[0].t == pytest.approx(1.0, abs=0.1) and ev[1].t == pytest.approx(2.0, abs=0.1)


def test_extrema_subframe(tracked):
    ev = extrema(analysis.smooth(tracked, 5), axis="y")
    assert any("max" in e.name for e in ev) and any("min" in e.name for e in ev)


def test_stats_and_outliers():
    s = stats([3.1, 3.1, 3.2, 3.1, 3.15, 6.0])
    assert s.n == 6 and s.outliers == [5] and s.max == 6.0
    assert stats([]).n == 0


def test_durations_between_user_picked_events(tracked):
    ram = detect_ram(tracked)
    d = durations_between(ram.events, "Ram starts downstroke", "Ram reaches bottom")
    # the clip ends mid-stroke: a 4th partial stroke still has start and bottom events
    assert len(d) == 4 and d[:3] == pytest.approx([s.downstroke_time for s in ram.strokes], abs=1e-9)
