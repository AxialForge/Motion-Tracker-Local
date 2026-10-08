"""Synthetic press-line video with known ground truth, for objective accuracy testing.

A coloured dot rides on a ram that does: idle at top, downstroke, dwell at bottom, upstroke.
Ground truth is the analytic position, so tracker error and event-time error are measurable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np


def smootherstep(u: np.ndarray | float):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


@dataclass
class RamSpec:
    width: int = 960
    height: int = 540
    fps: float = 30.0
    seconds: float = 12.0
    x: float = 480.0
    y_top: float = 150.0
    y_bottom: float = 390.0
    idle_top: float = 0.8
    down: float = 0.9
    dwell: float = 0.5
    up: float = 0.9
    dot_radius: float = 9.0
    dot_bgr: tuple = (40, 60, 230)  # orange-red craft paper
    noise_sigma: float = 2.0
    occlude: list = field(default_factory=list)  # [(t0, t1)] seconds the dot is hidden
    seed: int = 1

    @property
    def cycle(self) -> float:
        return self.idle_top + self.down + self.dwell + self.up

    def y_at(self, t):
        """Analytic ram Y(t) in pixels (vectorised)."""
        t = np.asarray(t, dtype=float)
        ph = np.mod(t, self.cycle)
        a, b, c = self.idle_top, self.idle_top + self.down, self.idle_top + self.down + self.dwell
        span = self.y_bottom - self.y_top
        y = np.full_like(ph, self.y_top)
        y = np.where((ph >= a) & (ph < b), self.y_top + span * smootherstep((ph - a) / self.down), y)
        y = np.where((ph >= b) & (ph < c), self.y_bottom, y)
        y = np.where(ph >= c, self.y_bottom - span * smootherstep((ph - c) / self.up), y)
        return y


def _background(spec: RamSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    bg = np.full((spec.height, spec.width, 3), 90, np.uint8)
    for _ in range(60):  # textured clutter so the tracker has something to ignore
        x0, y0 = int(rng.integers(0, spec.width)), int(rng.integers(0, spec.height))
        col = int(rng.integers(50, 140))
        cv2.rectangle(bg, (x0, y0), (x0 + int(rng.integers(10, 80)), y0 + int(rng.integers(10, 60))), (col,) * 3, -1)
    cv2.rectangle(bg, (spec.width // 2 - 90, 40), (spec.width // 2 + 90, spec.height - 40), (70, 70, 75), 6)
    return bg


def render_frame(spec: RamSpec, bg: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    img = bg.copy()
    y = float(spec.y_at(t))
    cv2.rectangle(img, (int(spec.x - 60), int(y - 40)), (int(spec.x + 60), int(y + 40)), (130, 130, 135), -1)  # ram block
    if not any(a <= t < b for a, b in spec.occlude):
        s = 4  # sub-pixel dot placement
        cv2.circle(img, (int(round(spec.x * (1 << s))), int(round(y * (1 << s)))),
                   int(round(spec.dot_radius * (1 << s))), spec.dot_bgr, -1, cv2.LINE_AA, s)
    else:
        cv2.rectangle(img, (int(spec.x - 25), int(y - 25)), (int(spec.x + 25), int(y + 25)), (30, 30, 30), -1)  # tooling
    if spec.noise_sigma:
        img = np.clip(img + rng.normal(0, spec.noise_sigma, img.shape), 0, 255).astype(np.uint8)
    return img


def write_clip(path: str | Path, spec: RamSpec | None = None) -> RamSpec:
    """Encode an H.264 MP4 (via FFmpeg) and a `<name>.truth.json` next to it."""
    spec = spec or RamSpec()
    path = Path(path)
    bg = _background(spec)
    rng = np.random.default_rng(spec.seed + 1)
    n = int(round(spec.seconds * spec.fps))
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{spec.width}x{spec.height}",
           "-r", str(spec.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p"]
    cmd.append(str(path))
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(render_frame(spec, bg, i / spec.fps, rng).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    path.with_suffix(".truth.json").write_text(json.dumps(asdict(spec)))
    return spec
