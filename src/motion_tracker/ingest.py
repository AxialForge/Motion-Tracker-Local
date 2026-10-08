"""Video ingest: metadata, real per-frame timestamps, rotation, frame reading."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, asdict
from fractions import Fraction

import cv2
import numpy as np


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: str
    codec: str
    width: int  # display size, after rotation
    height: int
    rotation: int  # degrees clockwise to apply for display (0/90/180/270)
    fps: float  # nominal (average) frame rate
    duration_s: float
    frame_count: int
    pix_fmt: str
    is_vfr: bool
    start_time_s: float  # container timestamp of the first frame (tracks start at t=0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TimingReport:
    median_dt: float
    dropped: list[tuple[int, int]]  # (frame index after the gap, frames missing)
    irregular: list[int]  # frames arriving much sooner than expected

    @property
    def clean(self) -> bool:
        return not self.dropped and not self.irregular


@dataclass(frozen=True)
class Frame:
    index: int
    t: float
    image: np.ndarray  # BGR, already rotated for display


def _ffprobe(*args: str) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise IngestError("ffprobe not found on PATH; install FFmpeg")
    r = subprocess.run([exe, "-v", "error", "-of", "json", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise IngestError(f"ffprobe failed: {r.stderr.strip()}")
    return json.loads(r.stdout or "{}")


def _fraction(s: str | None) -> float:
    try:
        f = Fraction(s or "0")
        return float(f)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation_cw(stream: dict) -> int:
    """Clockwise degrees to apply, from display-matrix side data or the legacy rotate tag."""
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            return int(round(-float(sd["rotation"]))) % 360 // 90 * 90
    tag = stream.get("tags", {}).get("rotate")
    if tag is not None:
        return int(round(float(tag))) % 360 // 90 * 90
    return 0


def frame_timestamps(path: str) -> np.ndarray:
    """Presentation timestamps (s) of every video frame, sorted, relative to the first frame."""
    data = _ffprobe("-select_streams", "v:0", "-show_entries", "packet=pts_time,dts_time", "-i", path)
    out = []
    for p in data.get("packets", []):
        v = p.get("pts_time", p.get("dts_time"))
        if v not in (None, "N/A"):
            out.append(float(v))
    if not out:
        raise IngestError("no video packets found")
    ts = np.sort(np.asarray(out))
    return ts - ts[0]


def probe(path: str) -> VideoInfo:
    """Read frame rate, resolution, duration, rotation and codec. The user never types these."""
    data = _ffprobe(
        "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration,pix_fmt,start_time"
        ":stream_side_data=rotation:stream_tags=rotate",
        "-i", path,
    )
    streams = data.get("streams", [])
    if not streams:
        raise IngestError(f"no video stream in {path}")
    s = streams[0]
    ts = frame_timestamps(path)
    dts = np.diff(ts)
    med = float(np.median(dts)) if len(dts) else 0.0
    fps = _fraction(s.get("avg_frame_rate")) or _fraction(s.get("r_frame_rate")) or (1 / med if med else 0.0)
    rot = _rotation_cw(s)
    w, h = int(s["width"]), int(s["height"])
    if rot in (90, 270):
        w, h = h, w
    spread = float(np.percentile(dts, 99) - np.percentile(dts, 1)) if len(dts) > 10 else 0.0
    dur = float(s.get("duration") or 0) or (ts[-1] + med)
    return VideoInfo(
        path=path, codec=s.get("codec_name", "?"), width=w, height=h, rotation=rot,
        fps=fps, duration_s=dur, frame_count=len(ts), pix_fmt=s.get("pix_fmt", "?"),
        is_vfr=bool(med and spread / med > 0.2), start_time_s=float(s.get("start_time") or 0),
    )


def check_timing(ts: np.ndarray) -> TimingReport:
    """Flag dropped frames (gaps) and irregular frames (too early) from real timestamps."""
    if len(ts) < 3:
        return TimingReport(0.0, [], [])
    dts = np.diff(ts)
    med = float(np.median(dts))
    dropped = [(i + 1, int(round(d / med)) - 1) for i, d in enumerate(dts) if d > 1.5 * med]
    irregular = [i + 1 for i, d in enumerate(dts) if d < 0.5 * med]
    return TimingReport(med, dropped, irregular)


def exposure_warning(image: np.ndarray, level: int = 250, max_fraction: float = 0.02) -> float | None:
    """Fraction of blown-out pixels if it is large enough to warn about, else None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frac = float((gray >= level).mean())
    return frac if frac > max_fraction else None


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class FrameReader:
    """Sequential frame reader that stamps each frame with its real timestamp."""

    def __init__(self, path: str, info: VideoInfo | None = None, timestamps: np.ndarray | None = None):
        self.path = path
        self.info = info or probe(path)
        self.timestamps = timestamps if timestamps is not None else frame_timestamps(path)

    def _t(self, i: int) -> float:
        ts = self.timestamps
        if i < len(ts):
            return float(ts[i])
        step = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / max(self.info.fps, 1)
        return float(ts[-1]) + step * (i - len(ts) + 1)

    def frames(self, start: int = 0, stop: int | None = None) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise IngestError(f"cannot open {self.path}")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # we apply rotation ourselves, deterministically
        rot = _ROT.get(self.info.rotation)
        try:
            i = 0
            while stop is None or i < stop:
                if i < start:
                    if not cap.grab():
                        return
                    i += 1
                    continue
                ok, img = cap.read()
                if not ok:
                    return
                if rot is not None:
                    img = cv2.rotate(img, rot)
                yield Frame(i, self._t(i), img)
                i += 1
        finally:
            cap.release()
