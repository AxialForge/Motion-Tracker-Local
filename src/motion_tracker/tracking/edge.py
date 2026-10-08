"""Hard-edge tracking: follow one straight edge along its normal, no dots needed.

A horizontal edge (for example the bottom of the ram) is tracked in Y; a vertical edge in X. The
brightness profile is averaged along a strip of `length` px across the edge, so noise averages out
and the edge position is found to sub-pixel precision from the gradient peak.

Options (PointSpec.options): orientation ("horizontal" | "vertical"), length (strip px, 60),
search (px each side of the predicted edge, 40), polarity ("auto" | "rising" | "falling" | "either").
Needs a visible straight edge with contrast; it measures along one axis only. The other coordinate
stays at the click.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..model import LOST, LOW_CONF
from .base import PointSpec, Sample, register_plugin

_NAN = float("nan")
LOW_CONF_THRESHOLD = 0.4
LOST_THRESHOLD = 0.1


class EdgeTracker:
    def __init__(self, spec: PointSpec):
        o = spec.options
        self.horizontal = o.get("orientation", "horizontal") == "horizontal"
        self.length = int(o.get("length", 60))
        self.search = int(o.get("search", 40))
        self.polarity = o.get("polarity", "auto")
        self.sigma = float(o.get("sigma", 1.2))
        if self.polarity not in ("auto", "rising", "falling", "either"):
            raise ValueError(f"bad polarity {self.polarity!r}")
        self.spec = spec
        # `along` runs across the edge (what we measure); `fixed` runs along it (stays at the click)
        self.along, self.fixed = (spec.y, spec.x) if self.horizontal else (spec.x, spec.y)
        self._vel = 0.0
        self._miss = 0
        k = int(max(3, round(self.sigma * 4)) | 1)
        self._kernel = cv2.getGaussianKernel(k, self.sigma).ravel()

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return g if self.horizontal else g.T  # rows always run across the edge

    def _score(self, g: np.ndarray, centre: float, half: float):
        """Edge strength along the normal inside [centre-half, centre+half]. Returns (idx0, signed gradient)."""
        h, w = g.shape
        lo, hi = max(int(centre - half) - 2, 0), min(int(centre + half) + 3, h)
        c0 = int(max(self.fixed - self.length / 2, 0))
        c1 = int(min(self.fixed + self.length / 2 + 1, w))
        if hi - lo < 5 or c1 <= c0:
            return lo, np.zeros(max(hi - lo, 0))
        prof = g[lo:hi, c0:c1].mean(axis=1)
        prof = np.convolve(np.pad(prof, len(self._kernel) // 2, mode="edge"), self._kernel, mode="valid")
        return lo, np.gradient(prof)

    def _pick(self, lo: int, grad: np.ndarray, predicted: float, half: float, sign: float | None):
        """Strongest edge near the prediction, refined to sub-pixel with a parabola."""
        score = np.abs(grad) if sign is None else sign * grad
        idx = np.arange(len(score)) + lo
        w = np.exp(-0.5 * ((idx - predicted) / max(0.75 * half, 1.0)) ** 2)  # mild preference for the predicted spot
        k = int(np.argmax(score * w))
        pos = float(idx[k])
        if 0 < k < len(score) - 1:
            a, b, c = score[k - 1], score[k], score[k + 1]
            den = a - 2 * b + c
            if den < 0:
                pos += 0.5 * (a - c) / den
        return pos, float(score[k]), float(np.sign(grad[k]))

    def init(self, frame: np.ndarray) -> Sample | None:
        g = self._gray(frame)
        lo, grad = self._score(g, self.along, self.search)
        want = {"rising": 1.0, "falling": -1.0}.get(self.polarity)
        pos, strength, sgn = self._pick(lo, grad, self.along, self.search, want)
        self._strength0 = max(strength, 1e-6)
        self._sign = want if want is not None else (None if self.polarity == "either" else sgn)
        self.along = pos
        return self._out(pos, 1.0 if strength > 1.0 else 0.0, 0)

    def update(self, frame: np.ndarray) -> Sample:
        g = self._gray(frame)
        half = min(self.search * (1 + 0.5 * self._miss), self.search * 3)
        predicted = self.along + self._vel
        lo, grad = self._score(g, predicted, half)
        if len(grad) == 0:
            self._miss += 1
            return self._out(_NAN, 0.0, LOST)
        pos, strength, _ = self._pick(lo, grad, predicted, half, self._sign)
        conf = float(np.clip(strength / self._strength0, 0.0, 1.0))
        if conf < LOST_THRESHOLD:
            self._miss += 1
            return self._out(_NAN, conf, LOST)
        self._miss = 0
        self._vel = 0.6 * self._vel + 0.4 * (pos - self.along)
        self.along = pos
        return self._out(pos, conf, LOW_CONF if conf < LOW_CONF_THRESHOLD else 0)

    def _out(self, pos: float, conf: float, flags: int) -> Sample:
        # horizontal edge: measured value is Y; vertical edge: measured value is X
        if self.horizontal:
            return Sample(self.fixed if flags != LOST else _NAN, pos, conf, flags)
        return Sample(pos, self.fixed if flags != LOST else _NAN, conf, flags)


def _init_wrapper(spec: PointSpec) -> EdgeTracker:
    return EdgeTracker(spec)


register_plugin("edge_tracking", _init_wrapper, kind="edge")
