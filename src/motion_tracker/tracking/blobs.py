"""Bright-blob billet tracking: glowing, overexposed billets found by brightness, no clicks needed.

Runs continuously: every frame is thresholded on brightness (max of B, G, R, so blown-out orange
still counts), blobs are found, and each new blob gets a billet ID. Blobs are matched from frame to
frame with a constant-velocity prediction, and a billet hidden for a few frames is carried through
the gap (flagged LOST) before being dropped. Billets that touch and merge into one blob are not split.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..model import LOST, LOW_CONF, Track

Rect = tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass(frozen=True)
class BlobConfig:
    threshold: int | None = 220  # brightness (0-255); None = per-frame Otsu, never below `min_auto`
    min_auto: int = 180
    min_area: int = 150  # px; smaller blobs are noise or sparks
    max_area: int | None = None
    roi: Rect | None = None  # detect only inside this region
    exclude: tuple[Rect, ...] = ()  # ignore bright things here (furnace window, lights)
    max_jump: float = 120.0  # px per frame a billet may move between detections
    max_missing: int = 15  # frames a billet may vanish (hidden by tooling) and still keep its ID
    min_life: int = 5  # detections needed before a track counts as a billet
    open_px: int = 3  # morphological opening size, removes speckle

    @classmethod
    def from_dict(cls, d: dict) -> BlobConfig:
        d = dict(d)
        for k in ("roi",):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        d["exclude"] = tuple(tuple(r) for r in d.get("exclude", ()))
        try:
            return cls(**d)
        except TypeError as e:
            raise ValueError(f"bad blob settings: {e}") from None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["exclude"] = [list(r) for r in self.exclude]
        return d


@dataclass
class _Live:
    x: float
    y: float
    last_frame: int
    vx: float = 0.0
    vy: float = 0.0
    missing: int = 0
    rows: list = field(default_factory=list)  # (frame, t, x, y, conf, flags)
    areas: list = field(default_factory=list)
    detections: int = 0


class BlobDetector:
    """FrameConsumer: feed every frame, then `finish()` returns one Track per billet."""

    def __init__(self, cfg: BlobConfig | None = None):
        self.cfg = cfg or BlobConfig()
        self._live: list[_Live] = []
        self._done: list[_Live] = []
        self._kernel = np.ones((self.cfg.open_px, self.cfg.open_px), np.uint8)

    # -- detection ---------------------------------------------------------------------------------
    def _detect(self, image: np.ndarray) -> list[tuple[float, float, float]]:
        c = self.cfg
        v = image.max(axis=2)
        thr = c.threshold if c.threshold is not None else max(c.min_auto, int(cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]))
        mask = (v >= thr).astype(np.uint8) * 255
        if c.roi is not None:
            x0, y0, x1, y1 = c.roi
            keep = np.zeros_like(mask)
            keep[y0:y1, x0:x1] = 255
            mask = cv2.bitwise_and(mask, keep)
        for x0, y0, x1, y1 in c.exclude:
            mask[y0:y1, x0:x1] = 0
        if c.open_px > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < c.min_area or (c.max_area and area > c.max_area):
                continue
            bx, by, bw, bh = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            sel = labels[by:by + bh, bx:bx + bw] == k
            w = (v[by:by + bh, bx:bx + bw].astype(np.float32) - thr + 1.0) * sel  # brightness-weighted centroid
            ys, xs = np.mgrid[by:by + bh, bx:bx + bw]
            tot = float(w.sum())
            out.append((float((w * xs).sum() / tot), float((w * ys).sum() / tot), float(area)))
        return out

    # -- association -------------------------------------------------------------------------------
    def feed(self, index: int, t: float, image: np.ndarray) -> None:
        dets = self._detect(image)
        live = self._live
        matched_dets: set[int] = set()
        if live and dets:
            cost = np.full((len(live), len(dets)), 1e9)
            gates = []
            for i, b in enumerate(live):
                gap = index - b.last_frame
                px, py = b.x + b.vx * gap, b.y + b.vy * gap
                gate = self.cfg.max_jump * (1 + 0.5 * b.missing)
                gates.append(gate)
                for j, (dx, dy, _) in enumerate(dets):
                    d = float(np.hypot(dx - px, dy - py))
                    if d <= gate:
                        cost[i, j] = d
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e9:
                    self._update(live[i], index, t, *dets[j])
                    matched_dets.add(j)
        for j, (dx, dy, area) in enumerate(dets):
            if j not in matched_dets:  # a new blob: a new billet ID
                b = _Live(dx, dy, index)
                self._update(b, index, t, dx, dy, area, first=True)
                live.append(b)
        keep = []
        for b in live:
            if b.last_frame == index:  # matched or newly created this frame
                keep.append(b)
                continue
            b.missing += 1
            b.rows.append((index, t, float("nan"), float("nan"), 0.0, LOST))
            if b.missing > self.cfg.max_missing:
                self._done.append(b)
            else:
                keep.append(b)
        self._live = keep

    def _update(self, b: _Live, index: int, t: float, x: float, y: float, area: float, first: bool = False) -> None:
        gap = max(index - b.last_frame, 1)
        if not first:
            b.vx = 0.5 * b.vx + 0.5 * (x - b.x) / gap
            b.vy = 0.5 * b.vy + 0.5 * (y - b.y) / gap
        b.x, b.y, b.last_frame, b.missing = x, y, index, 0
        b.areas.append(area)
        ref = float(np.median(b.areas[-30:]))
        conf = float(min(area, ref) / max(area, ref))  # a billet partly hidden shows a shrunken blob
        b.rows.append((index, t, x, y, conf, LOW_CONF if conf < 0.4 else 0))
        b.detections += 1

    # -- output ------------------------------------------------------------------------------------
    def finish(self) -> list[Track]:
        all_b = [b for b in self._done + self._live if b.detections >= self.cfg.min_life]
        all_b.sort(key=lambda b: b.rows[0][0])
        tracks = []
        for n, b in enumerate(all_b, 1):
            rows = list(b.rows)
            while rows and rows[-1][5] & LOST:  # trailing missing frames are just the billet leaving
                rows.pop()
            a = np.array(rows, dtype=float)
            tracks.append(Track(
                name=f"Billet {n}", kind="billet", mode="bright_blob", params=self.cfg.to_dict(),
                frame=a[:, 0].astype(np.int32), t=a[:, 1], x=a[:, 2], y=a[:, 3],
                conf=a[:, 4].astype(np.float32), flags=a[:, 5].astype(np.uint8)))
        self._done, self._live = [], []
        return tracks
