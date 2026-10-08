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
    dot: bool = True  # False: a hard-edged ram column with no dot (track its bottom edge)
    dot_radius: float = 9.0
    dot_bgr: tuple = (40, 60, 230)  # orange-red craft paper
    noise_sigma: float = 2.0
    occlude: list = field(default_factory=list)  # [(t0, t1)] seconds the dot is hidden
    seed: int = 1

    @property
    def cycle(self) -> float:
        return self.idle_top + self.down + self.dwell + self.up

    @property
    def edge_offset(self) -> float:
        """Bottom edge of the no-dot ram column sits this far below the ram reference Y."""
        return 40.0

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


def _render_edge_ram(spec: RamSpec, bg: np.ndarray, y: float) -> np.ndarray:
    """A ram column hanging from the top of the frame; its bottom edge is anti-aliased at a sub-pixel Y."""
    edge = y + spec.edge_offset
    rows = np.arange(spec.height, dtype=np.float32)[:, None]
    a = np.clip(edge - rows + 0.5, 0.0, 1.0)  # fraction of each pixel row covered by the column
    c0, c1 = int(spec.x - 60), int(spec.x + 60)
    img = bg.astype(np.float32).copy()
    img[:, c0:c1] = img[:, c0:c1] * (1 - a[..., None]) + 150.0 * a[..., None]
    return img


def render_frame(spec: RamSpec, bg: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    y = float(spec.y_at(t))
    if not spec.dot:
        img = _render_edge_ram(spec, bg, y)
        if spec.noise_sigma:
            img = img + rng.normal(0, spec.noise_sigma, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)
    img = bg.copy()
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


def _encode(path: Path, n: int, w: int, h: int, fps: float, frames) -> None:
    """Encode an H.264 MP4 through FFmpeg from an iterator of BGR frames."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def write_clip(path: str | Path, spec: RamSpec | None = None) -> RamSpec:
    """Encode an H.264 MP4 (via FFmpeg) and a `<name>.truth.json` next to it."""
    spec = spec or RamSpec()
    path = Path(path)
    bg = _background(spec)
    rng = np.random.default_rng(spec.seed + 1)
    n = int(round(spec.seconds * spec.fps))
    _encode(path, n, spec.width, spec.height, spec.fps, (render_frame(spec, bg, i / spec.fps, rng) for i in range(n)))
    path.with_suffix(".truth.json").write_text(json.dumps(asdict(spec)))
    return spec


@dataclass
class BilletSpec:
    """Glowing billets travelling heater exit -> station 1 -> station 2 -> out of frame.

    Each billet pops into existence at the heater exit, so first appearance is frame-quantised.
    """

    width: int = 960
    height: int = 540
    fps: float = 30.0
    seconds: float = 14.0
    y: float = 300.0
    heater_x: float = 80.0
    s1_x: float = 400.0
    s2_x: float = 700.0
    exit_x: float = 1060.0  # off-frame
    starts: list = field(default_factory=lambda: [0.5, 6.5])
    move1: float = 2.0
    dwell1: float = 1.0
    move2: float = 1.0
    dwell2: float = 1.5
    move3: float = 1.0
    half_len: float = 35.0
    half_h: float = 13.0
    noise_sigma: float = 2.0
    hide: list = field(default_factory=list)  # [(t0, t1)] seconds all billets are hidden (tooling)
    seed: int = 2

    @property
    def total(self) -> float:
        return self.move1 + self.dwell1 + self.move2 + self.dwell2 + self.move3

    @property
    def keyframes(self) -> list[tuple[float, float]]:
        t1 = self.move1
        t2 = t1 + self.dwell1
        t3 = t2 + self.move2
        t4 = t3 + self.dwell2
        return [(0.0, self.heater_x), (t1, self.s1_x), (t2, self.s1_x), (t3, self.s2_x), (t4, self.s2_x), (t4 + self.move3, self.exit_x)]

    def x_at(self, local_t):
        """Billet X at time since it appeared (vectorised); NaN outside its lifetime."""
        lt = np.asarray(local_t, dtype=float)
        kf = self.keyframes
        x = np.full_like(lt, np.nan)
        for (ta, xa), (tb, xb) in zip(kf, kf[1:]):
            m = (lt >= ta) & (lt <= tb)
            x = np.where(m, xa + (xb - xa) * smootherstep((lt - ta) / (tb - ta)) if xb != xa else xa, x)
        return x


def write_billet_clip(path: str | Path, spec: BilletSpec | None = None) -> BilletSpec:
    spec = spec or BilletSpec()
    path = Path(path)
    rng = np.random.default_rng(spec.seed)
    bg = np.full((spec.height, spec.width, 3), 28, np.uint8)
    for _ in range(40):  # dim clutter: machinery, never as bright as a billet
        x0, y0 = int(rng.integers(0, spec.width)), int(rng.integers(0, spec.height))
        cv2.rectangle(bg, (x0, y0), (x0 + int(rng.integers(20, 120)), y0 + int(rng.integers(10, 70))), (int(rng.integers(35, 120)),) * 3, -1)
    n = int(round(spec.seconds * spec.fps))
    sh = 4

    def frames():
        for i in range(n):
            t = i / spec.fps
            img = bg.astype(np.float32).copy()
            if not any(a <= t < b for a, b in spec.hide):
                halo = np.zeros_like(img)
                core = np.zeros_like(img)
                for t0 in spec.starts:
                    x = float(spec.x_at(t - t0)) if t >= t0 else float("nan")
                    if np.isnan(x):
                        continue
                    c = (int(round(x * (1 << sh))), int(round(spec.y * (1 << sh))))
                    cv2.ellipse(halo, c, (int((spec.half_len + 12) * (1 << sh)), int((spec.half_h + 10) * (1 << sh))), 0, 0, 360, (40, 120, 255), -1, cv2.LINE_AA, sh)
                    cv2.ellipse(core, c, (int(spec.half_len * (1 << sh)), int(spec.half_h * (1 << sh))), 0, 0, 360, (190, 235, 255), -1, cv2.LINE_AA, sh)
                halo = cv2.GaussianBlur(halo, (0, 0), 8)
                img = np.maximum(img, halo * 0.8)
                img = np.maximum(img, core)
            if spec.noise_sigma:
                img = img + rng.normal(0, spec.noise_sigma, img.shape)
            yield np.clip(img, 0, 255).astype(np.uint8)

    _encode(path, n, spec.width, spec.height, spec.fps, frames())
    path.with_suffix(".truth.json").write_text(json.dumps(asdict(spec)))
    return spec
