from dataclasses import replace

import numpy as np
import pytest

from motion_tracker import analysis
from motion_tracker.ingest import FrameReader
from motion_tracker.model import INTERPOLATED, LOST, LOW_CONF, RECOVERED
from motion_tracker.synthetic import RamSpec, write_clip
from motion_tracker.tracking import PointSpec, track_points


def _errors(track, spec):
    return np.hypot(track.x - spec.x, track.y - spec.y_at(track.t))


@pytest.mark.parametrize("algo,limit", [("csrt", 0.5), ("kcf", 0.5)])
def test_dot_accuracy_vs_ground_truth(clip, tracked, algo, limit):
    path, spec = clip
    tr = tracked if algo == "csrt" else track_points(
        FrameReader(str(path)), [PointSpec("ram", spec.x, spec.y_top, algo=algo)])[0]
    err = _errors(tr, spec)
    print(f"{algo}: mean {err.mean():.3f}px  max {err.max():.3f}px")
    assert len(tr) == 360 and not np.isnan(tr.x).any()
    assert err.mean() < limit and err.max() < 2.0
    assert tr.conf.min() > 0.7


def test_multi_point_each_with_own_algorithm(small_clip):
    path, spec = small_clip
    trs = track_points(FrameReader(str(path)), [
        PointSpec("a", spec.x, spec.y_top, algo="csrt"), PointSpec("b", spec.x, spec.y_top, algo="kcf")])
    assert [t.params["algo"] for t in trs] == ["csrt", "kcf"] and all(len(t) == 150 for t in trs)
    assert all(_errors(t, spec).max() < 2.0 for t in trs)


def test_segment_and_cancel(small_clip):
    path, spec = small_clip
    r = FrameReader(str(path))
    (tr,) = track_points(r, [PointSpec("ram", spec.x, spec.y_top)], start=0, stop=50)
    assert len(tr) == 50
    n = {"c": 0}

    def cancel():
        n["c"] += 1
        return n["c"] >= 20

    (tr,) = track_points(r, [PointSpec("ram", spec.x, spec.y_top)], cancel=cancel)
    assert len(tr) == 20


def test_late_seed_frame(small_clip):
    path, spec = small_clip
    f = 40
    (tr,) = track_points(FrameReader(str(path)), [PointSpec("ram", spec.x, float(spec.y_at(f / 30)), seed_frame=f)], stop=100)
    assert tr.frame[0] == f and len(tr) == 60 and _errors(tr, spec).max() < 2.0


def test_occlusion_is_flagged_and_colour_lock_recovers(small_spec, tmp_path):
    spec = replace(small_spec, seconds=6, occlude=[(2.3, 2.7)])  # dot hidden while the ram is moving
    p = tmp_path / "occ.mp4"
    write_clip(p, spec)
    (tr,) = track_points(FrameReader(str(p)), [PointSpec("ram", spec.x, spec.y_top)])
    hidden = (tr.t >= 2.35) & (tr.t < 2.7)
    assert (tr.flags[hidden] & (LOST | LOW_CONF | RECOVERED)).any()
    assert ((tr.flags[hidden] & (LOST | LOW_CONF)) > 0).mean() > 0.5  # never passed off as good data
    after = tr.t >= 2.8
    assert _errors(tr, spec)[after].max() < 2.0  # re-acquired
    assert (tr.flags & RECOVERED).any()


def _gappy(n=60, gaps=((20, 25),)):
    from motion_tracker.model import Track
    t = np.arange(n) / 30
    x = t * 100.0
    flags = np.zeros(n, np.uint8)
    for a, b in gaps:
        x[a:b] = np.nan
        flags[a:b] = LOST
    return Track("t", frame=np.arange(n, dtype=np.int32), t=t, x=x, y=x * 0.5, conf=np.ones(n, np.float32), flags=flags)


def test_gap_fill_interpolates_and_flags():
    tr = _gappy()
    out = analysis.fill_gaps(tr, max_gap=10)
    np.testing.assert_allclose(out.x[20:25], tr.t[20:25] * 100.0)  # linear motion is recovered exactly
    assert ((out.flags[20:25] & INTERPOLATED) > 0).all() and ((out.flags[20:25] & LOST) == 0).all()
    assert not out.measured()[20:25].any() and out.measured()[:20].all()
    assert np.isnan(tr.x[20:25]).all()  # input untouched


def test_gap_fill_leaves_long_and_edge_gaps():
    out = analysis.fill_gaps(_gappy(gaps=((0, 3), (20, 40))), max_gap=10)
    assert np.isnan(out.x[0:3]).all() and np.isnan(out.x[20:40]).all()  # no data on one side / too long


def test_smooth_keeps_nan_gaps():
    from motion_tracker.model import Track
    n = 40
    x = np.arange(n, dtype=float)
    x[20:25] = np.nan
    tr = Track("t", frame=np.arange(n, dtype=np.int32), t=np.arange(n) / 30, x=x, y=x.copy(),
               conf=np.ones(n, np.float32), flags=np.zeros(n, np.uint8))
    out = analysis.smooth(tr, 5)
    assert np.isnan(out.x[20:25]).all() and not np.isnan(out.x[:20]).any()


@pytest.mark.parametrize("name,bgr", [("blue", (220, 80, 30)), ("yellow", (30, 220, 230)),
                                      ("white", (250, 250, 250)), ("black", (15, 15, 15))])
def test_dot_colour_varies_per_video(small_spec, tmp_path, name, bgr):
    """Colour is read from each seed patch. Colourless dots fall back to template matching."""
    spec = replace(small_spec, seconds=4, dot_bgr=bgr, occlude=[(1.2, 1.5)])
    p = tmp_path / f"{name}.mp4"
    write_clip(p, spec)
    (tr,) = track_points(FrameReader(str(p)), [PointSpec("d", spec.x, spec.y_top)])
    err = _errors(tr, spec)
    assert not np.isnan(err[tr.t > 1.7]).any()  # re-acquired after the occlusion
    assert err[tr.t > 1.7].max() < 4.0 and err[tr.t < 1.1].max() < 3.0
    hidden = (tr.t >= 1.25) & (tr.t < 1.5)
    assert ((tr.flags[hidden] & (LOST | LOW_CONF | RECOVERED)) > 0).any()


def test_mixed_dot_colours_in_one_video(small_spec, tmp_path):
    """Each point carries its own colour model, so two differently coloured dots do not interfere."""
    import cv2
    from motion_tracker.synthetic import _background, render_frame
    spec = replace(small_spec, seconds=3)
    p = tmp_path / "mixed.mp4"
    # render a second, static green dot alongside the ram dot
    import subprocess
    bg, rng = _background(spec), np.random.default_rng(3)
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "640x360", "-r", "30",
                             "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(p)], stdin=subprocess.PIPE)
    for i in range(90):
        f = render_frame(spec, bg, i / 30, rng)
        cv2.circle(f, (500, 120), 7, (60, 200, 60), -1, cv2.LINE_AA)
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    a, b = track_points(FrameReader(str(p)), [PointSpec("ram", spec.x, spec.y_top), PointSpec("fixed", 500, 120, algo="kcf")])
    assert _errors(a, spec).max() < 2.0
    assert np.hypot(b.x - 500, b.y - 120).max() < 1.5
