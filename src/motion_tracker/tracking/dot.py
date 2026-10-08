"""Dot tracking: OpenCV CSRT (precision) or KCF (speed) with colour-lock recovery.

CSRT/KCF run on CPU. Confidence is appearance similarity to the seed patch (hue/saturation
histogram correlation for coloured dots, grey template correlation otherwise); it is a health
indicator, not a calibrated probability.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..model import LOST, LOW_CONF, RECOVERED
from .base import PointSpec, Sample, register_plugin

LOW_CONF_THRESHOLD = 0.4
RECOVER_THRESHOLD = 0.25
TEMPLATE_RECOVER_SCORE = 0.6
_SAT_MIN, _VAL_MIN = 60, 50
_NAN = float("nan")


def _make_cv_tracker(algo: str):
    makers = {"csrt": "TrackerCSRT_create", "kcf": "TrackerKCF_create"}
    if algo not in makers:
        raise ValueError(f"unknown algorithm {algo!r}; use csrt or kcf")
    fn = getattr(cv2, makers[algo], None) or getattr(getattr(cv2, "legacy", None), makers[algo], None)
    if fn is None:
        raise RuntimeError("OpenCV tracker missing: install opencv-contrib-python")
    return fn()


def _clip_box(cx: float, cy: float, size: float, shape) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    half = size / 2
    x0, y0 = max(int(round(cx - half)), 0), max(int(round(cy - half)), 0)
    x1, y1 = min(int(round(cx + half)), w), min(int(round(cy + half)), h)
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


class DotTracker:
    def __init__(self, spec: PointSpec, color_lock: bool = True, refine: bool = True):
        self.spec = spec
        self.color_lock = color_lock
        self.refine = refine
        self._tr = None
        self._last = (spec.x, spec.y)  # last good position
        self._lost_frames = 0

    # -- setup ---------------------------------------------------------------------------------
    def init(self, frame: np.ndarray) -> None:
        s = self.spec
        x, y, w, h = _clip_box(s.x, s.y, s.box, frame.shape)
        self._bbox0 = (x, y, w, h)
        patch = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, _SAT_MIN, _VAL_MIN), (180, 255, 255))
        self._color_ok = self.color_lock and mask.mean() / 255 > 0.05
        if self._color_ok:
            self._hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
            cv2.normalize(self._hist, self._hist, 0, 255, cv2.NORM_MINMAX)
            self._area0 = float(max((mask > 0).sum(), 1))
        else:
            self._gray0 = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self._start(frame, (x, y, w, h))

    def _start(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        self._tr = _make_cv_tracker(self.spec.algo)
        self._tr.init(frame, tuple(int(v) for v in bbox))

    # -- appearance ----------------------------------------------------------------------------
    def _similarity(self, frame: np.ndarray, bbox) -> float:
        x, y, w, h = bbox
        patch = frame[max(y, 0):y + h, max(x, 0):x + w]
        if patch.size == 0:
            return 0.0
        if self._color_ok:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, _SAT_MIN, _VAL_MIN), (180, 255, 255))
            if mask.mean() / 255 < 0.02:
                return 0.0
            hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            return float(np.clip(cv2.compareHist(self._hist, hist, cv2.HISTCMP_CORREL), 0, 1))
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.resize(g, (self._gray0.shape[1], self._gray0.shape[0]))
        a, b = g - g.mean(), self._gray0 - self._gray0.mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float(np.clip((a * b).sum() / d, 0, 1)) if d > 1e-6 else 0.0

    def _backproject(self, frame: np.ndarray, rect) -> tuple[np.ndarray, tuple[int, int]]:
        x, y, w, h = rect
        roi = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bp = cv2.calcBackProject([hsv], [0, 1], self._hist, [0, 180, 0, 256], 1)
        valid = cv2.inRange(hsv, (0, _SAT_MIN, _VAL_MIN), (180, 255, 255))
        bp = cv2.bitwise_and(bp, valid)
        _, m = cv2.threshold(bp, 40, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), (x, y)

    # -- colour-lock recovery --------------------------------------------------------------------
    def _recover(self, frame: np.ndarray) -> tuple[float, float] | None:
        base = self.spec.box * 3
        radius = min(base + 25 * self._lost_frames, max(frame.shape[:2]))
        lx, ly = self._last
        rect = _clip_box(lx, ly, 2 * radius, frame.shape)
        if not self._color_ok:
            return self._recover_by_template(frame, rect)
        mask, (ox, oy) = self._backproject(frame, rect)
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        best, best_d = None, 1e18
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < 0.3 * self._area0:
                continue
            cx, cy = cents[k][0] + ox, cents[k][1] + oy
            d = (cx - lx) ** 2 + (cy - ly) ** 2
            if d < best_d:
                best, best_d = (float(cx), float(cy)), d
        return best

    def _recover_by_template(self, frame: np.ndarray, rect) -> tuple[float, float] | None:
        """White, grey and black dots have no hue to lock onto: search for the seed patch itself."""
        x, y, w, h = rect
        th, tw = self._gray0.shape
        if w <= tw or h <= th:
            return None
        gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        res = cv2.matchTemplate(gray, self._gray0, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < TEMPLATE_RECOVER_SCORE:
            return None
        return float(x + loc[0] + tw / 2), float(y + loc[1] + th / 2)

    def _refined_center(self, frame: np.ndarray, bbox) -> tuple[float, float]:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        if not (self.refine and self._color_ok):
            return cx, cy
        rect = _clip_box(cx, cy, max(w, h) * 1.5, frame.shape)
        mask, (ox, oy) = self._backproject(frame, rect)
        if (mask > 0).sum() < 0.4 * self._area0:
            return cx, cy
        m = cv2.moments(mask, binaryImage=True)
        return m["m10"] / m["m00"] + ox, m["m01"] / m["m00"] + oy

    # -- per-frame update --------------------------------------------------------------------------
    def update(self, frame: np.ndarray) -> Sample:
        ok, bbox = self._tr.update(frame)
        bbox = tuple(int(round(v)) for v in bbox) if ok else None
        conf = self._similarity(frame, bbox) if ok else 0.0
        flags = 0
        if (not ok) or conf < RECOVER_THRESHOLD:
            found = self._recover(frame)
            if found is not None:
                nb = _clip_box(found[0], found[1], self.spec.box, frame.shape)
                self._start(frame, nb)
                bbox, flags = nb, RECOVERED
                conf = self._similarity(frame, nb)
            elif not ok:
                self._lost_frames += 1
                return Sample(_NAN, _NAN, 0.0, LOST)
        self._lost_frames = 0
        cx, cy = self._refined_center(frame, bbox)
        if conf < LOW_CONF_THRESHOLD:
            flags |= LOW_CONF
        else:
            self._last = (cx, cy)
        return Sample(cx, cy, conf, flags)


register_plugin("dot_tracking", lambda spec: DotTracker(spec))
