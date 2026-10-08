import subprocess

import numpy as np
import pytest

from motion_tracker.ingest import FrameReader, check_timing, exposure_warning, frame_timestamps, probe


def test_probe_reads_metadata_automatically(clip):
    path, spec = clip
    info = probe(str(path))
    assert (info.width, info.height) == (spec.width, spec.height)
    assert info.fps == pytest.approx(spec.fps)
    assert info.frame_count == round(spec.seconds * spec.fps)
    assert info.duration_s == pytest.approx(spec.seconds, abs=0.05)
    assert info.codec == "h264" and info.rotation == 0 and not info.is_vfr


def test_timestamps_are_real_and_monotonic(clip):
    ts = frame_timestamps(str(clip[0]))
    assert ts[0] == 0 and np.all(np.diff(ts) > 0)
    assert np.diff(ts).mean() == pytest.approx(1 / 30, rel=1e-3)
    assert check_timing(ts).clean


def test_frames_carry_timestamps(clip):
    r = FrameReader(str(clip[0]))
    frames = list(r.frames(10, 14))
    assert [f.index for f in frames] == [10, 11, 12, 13]
    assert frames[0].t == pytest.approx(10 / 30, abs=1e-6)
    assert frames[0].image.shape == (540, 960, 3)


def test_dropped_frames_are_flagged(clip, tmp_path):
    out = tmp_path / "dropped.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip[0]), "-vf", r"select='not(eq(n\,50)+eq(n\,51))'",
                    "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "20", str(out)], check=True)
    rep = check_timing(frame_timestamps(str(out)))
    assert rep.dropped == [(50, 2)]  # frame index 50 in the output follows a 2-frame gap


def test_rotation_metadata_is_applied(clip, tmp_path):
    out = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-display_rotation", "90", "-i", str(clip[0]), "-c", "copy", str(out)],
                   check=True)
    info = probe(str(out))
    assert info.rotation in (90, 270)
    assert (info.width, info.height) == (540, 960)  # display size is swapped
    first = next(FrameReader(str(out), info).frames(0, 1))
    assert first.image.shape[:2] == (960, 540)


def test_exposure_warning():
    dark = np.full((100, 100, 3), 60, np.uint8)
    blown = dark.copy()
    blown[:30] = 255
    assert exposure_warning(dark) is None
    assert exposure_warning(blown) == pytest.approx(0.3)
