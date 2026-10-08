"""Event detection from tracks: ram strokes plus reusable generic event types.

Times are sub-frame: threshold crossings are linearly interpolated between samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .model import Event, Track


def _cross_t(t0: float, t1: float, v0: float, v1: float, level: float) -> float:
    return t0 + (level - v0) / (v1 - v0) * (t1 - t0) if v1 != v0 else t0


def _frame_at(track: Track, t: float) -> float:
    return float(np.interp(t, track.t, track.frame))


@dataclass(frozen=True)
class Stroke:
    index: int
    down_start: float
    bottom_reach: float
    up_start: float
    top_reach: float
    down_peak_speed: float  # px/s
    up_peak_speed: float
    down_mean_speed: float
    up_mean_speed: float

    @property
    def downstroke_time(self) -> float:
        return self.bottom_reach - self.down_start

    @property
    def upstroke_time(self) -> float:
        return self.top_reach - self.up_start

    @property
    def total_time(self) -> float:
        return self.top_reach - self.down_start

    @property
    def bottom_dwell(self) -> float:
        return self.up_start - self.bottom_reach


@dataclass(frozen=True)
class RamAnalysis:
    events: list[Event]
    strokes: list[Stroke]
    idle_top: list[float]  # time near top between strokes (top_reach_i to down_start_{i+1})
    cycle_times: list[float]  # down_start_i to down_start_{i+1}


def detect_ram(track: Track, *, axis: str = "y", down_is_positive: bool = True,
               band: float = 0.05, smooth_window: int = 5, min_range_px: float = 10.0) -> RamAnalysis:
    """Ram events from one point on the ram.

    The ram is "at top" or "at bottom" when within `band` (fraction of its travel) of that extreme.
    Image Y grows downward, so by default the bottom of the stroke is the maximum Y.
    """
    t = track.t
    v = (track.y if axis == "y" else track.x).astype(float).copy()
    ok = ~np.isnan(v)
    t, v, frames = t[ok], v[ok], track.frame[ok]
    if len(v) < 10:
        return RamAnalysis([], [], [], [])
    if not down_is_positive:
        v = -v
    w = smooth_window + (smooth_window + 1) % 2
    vs = savgol_filter(v, w, 2) if smooth_window > 2 and len(v) > w else v
    lo, hi = np.percentile(vs, 2), np.percentile(vs, 98)
    rng = hi - lo
    if rng < min_range_px:
        return RamAnalysis([], [], [], [])
    top_thr, bot_thr = lo + band * rng, hi - band * rng
    state = np.where(vs <= top_thr, 0, np.where(vs >= bot_thr, 2, 1))  # 0 top, 1 middle, 2 bottom

    # runs of top (0) / bottom (2) samples, in order
    runs: list[tuple[int, int, int]] = []  # (state, first_idx, last_idx)
    i = 0
    while i < len(state):
        if state[i] == 1:
            i += 1
            continue
        j = i
        while j + 1 < len(state) and state[j + 1] == state[i]:
            j += 1
        runs.append((int(state[i]), i, j))
        i = j + 1

    def entry(k: int) -> float | None:  # time the band is entered
        s, a, _ = runs[k]
        if a == 0:
            return None
        thr = top_thr if s == 0 else bot_thr
        return _cross_t(t[a - 1], t[a], vs[a - 1], vs[a], thr)

    def exit_(k: int) -> float | None:  # time the band is left
        s, _, b = runs[k]
        if b == len(vs) - 1:
            return None
        thr = top_thr if s == 0 else bot_thr
        return _cross_t(t[b], t[b + 1], vs[b], vs[b + 1], thr)

    item = track.name
    events: list[Event] = []

    def add(name: str, when: float, **d):
        events.append(Event(name, "ram", item, float(when), _frame_at(track, when), d))

    strokes: list[Stroke] = []
    n = 0
    for k in range(len(runs) - 1):
        s, nxt = runs[k][0], runs[k + 1][0]
        if s == 0 and nxt == 2:
            if (ds := exit_(k)) is not None:
                add("Ram starts downstroke", ds)
            if (br := entry(k + 1)) is not None:
                add("Ram reaches bottom", br)
        elif s == 2 and nxt == 0:
            if (us := exit_(k)) is not None:
                add("Ram starts upstroke", us)
            if (tr := entry(k + 1)) is not None:
                add("Ram reaches top", tr)
        if k + 2 < len(runs) and (s, runs[k + 1][0], runs[k + 2][0]) == (0, 2, 0):
            ds, br, us, tr = exit_(k), entry(k + 1), exit_(k + 1), entry(k + 2)
            if None in (ds, br, us, tr):
                continue
            sp = np.abs(np.gradient(vs, t))
            dmask, umask = (t >= ds) & (t <= br), (t >= us) & (t <= tr)
            dpk = float(sp[dmask].max()) if dmask.any() else float("nan")
            upk = float(sp[umask].max()) if umask.any() else float("nan")
            strokes.append(Stroke(
                n, ds, br, us, tr, dpk, upk,
                rng / (br - ds) if br > ds else float("nan"), rng / (tr - us) if tr > us else float("nan"),
            ))
            n += 1
    events.sort(key=lambda e: e.t)
    idle = [b.down_start - a.top_reach for a, b in zip(strokes, strokes[1:])]
    cyc = [b.down_start - a.down_start for a, b in zip(strokes, strokes[1:])]
    return RamAnalysis(events, strokes, idle, cyc)


# ---- generic, reusable event types -------------------------------------------------------------

def line_cross(track: Track, *, axis: str, value: float, direction: str = "either",
               name: str | None = None, group: str = "custom") -> list[Event]:
    """Point crosses a line (axis = value). direction: 'up' (increasing), 'down' (decreasing), 'either'."""
    a = track.x if axis == "x" else track.y
    out = []
    for i in range(len(a) - 1):
        a0, a1 = a[i], a[i + 1]
        if np.isnan(a0) or np.isnan(a1):
            continue
        rising, falling = a0 < value <= a1, a0 > value >= a1
        if (rising and direction in ("up", "either")) or (falling and direction in ("down", "either")):
            when = _cross_t(track.t[i], track.t[i + 1], a0, a1, value)
            out.append(Event(name or f"{track.name} crosses {axis}={value:g}", group, track.name,
                             float(when), _frame_at(track, when), {"direction": "up" if rising else "down"}))
    return out


def zone_transitions(track: Track, rect: tuple[float, float, float, float], zone: str) -> list[Event]:
    """Point enters or leaves a rectangular zone (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    inside = (track.x >= x0) & (track.x <= x1) & (track.y >= y0) & (track.y <= y1) & ~np.isnan(track.x)
    out = []
    for i in range(1, len(inside)):
        if inside[i] != inside[i - 1]:
            when = float((track.t[i - 1] + track.t[i]) / 2)  # frame-level resolution
            nm = f"{track.name} enters {zone}" if inside[i] else f"{track.name} leaves {zone}"
            out.append(Event(nm, "custom", track.name, when, _frame_at(track, when), {"zone": zone}))
    return out


def moving_state(track: Track, *, speed_threshold: float, min_hold: int = 3, smooth_window: int = 5) -> list[Event]:
    """Item starts or stops moving: speed (px/s) crosses a threshold and holds for `min_hold` samples."""
    ok = ~np.isnan(track.x)
    t, x, y = track.t[ok], track.x[ok], track.y[ok]
    if len(t) < smooth_window + 2:
        return []
    w = smooth_window + (smooth_window + 1) % 2
    sp = np.hypot(np.gradient(savgol_filter(x, w, 2), t), np.gradient(savgol_filter(y, w, 2), t))
    moving = sp > speed_threshold
    out, cur, i = [], bool(moving[0]), 1
    while i < len(moving):
        if moving[i] != cur and (moving[i:i + min_hold] == moving[i]).all():
            cur = bool(moving[i])
            when = _cross_t(t[i - 1], t[i], sp[i - 1], sp[i], speed_threshold)
            out.append(Event(f"{track.name} {'starts' if cur else 'stops'} moving", "custom", track.name,
                             float(when), float(np.interp(when, t, track.frame[ok])), {"speed_threshold": speed_threshold}))
        i += 1
    return out


def extrema(track: Track, *, axis: str = "y", prominence_frac: float = 0.3) -> list[Event]:
    """Item reaches position min or max; sub-frame time from a parabola through the extreme."""
    from scipy.signal import find_peaks

    a = (track.x if axis == "x" else track.y).astype(float)
    ok = ~np.isnan(a)
    t, a, fr = track.t[ok], a[ok], track.frame[ok]
    if len(a) < 5:
        return []
    prom = prominence_frac * (a.max() - a.min())
    out = []
    for sign, label in ((1, "max"), (-1, "min")):
        pk, _ = find_peaks(sign * a, prominence=prom)
        for p in pk:
            when = float(t[p])
            if 0 < p < len(a) - 1:
                y0, y1, y2 = sign * a[p - 1], sign * a[p], sign * a[p + 1]
                den = y0 - 2 * y1 + y2
                if den != 0:
                    off = 0.5 * (y0 - y2) / den
                    when = float(t[p] + off * (t[p + 1] - t[p] if off > 0 else t[p] - t[p - 1]))
            out.append(Event(f"{track.name} reaches {axis} {label}", "custom", track.name, when,
                             float(np.interp(when, t, fr)), {}))
    return sorted(out, key=lambda e: e.t)
