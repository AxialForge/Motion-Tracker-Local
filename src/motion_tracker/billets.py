"""Billet events and timelines from billet tracks (bright-blob mode).

Covers: exits induction heater, enters/leaves frame, starts/stops moving, placed in / leaves a
station, dwell per station, station-to-station transfer, leaves press, billet count and gap between
billets, and picked up by an operator or manipulator (proximity to a tool track).

Stations are rectangles you draw. A billet hidden by tooling is only carried through short gaps here;
presence while hidden is the die-station zone mode (Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import moving_state, rect_crossings
from .model import Event, Track

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class Station:
    name: str
    rect: Rect


@dataclass(frozen=True)
class StationVisit:
    station: str
    enter: float | None  # None: the billet was already there when first seen
    leave: float | None  # None: still there when last seen

    @property
    def dwell(self) -> float | None:
        return None if self.enter is None or self.leave is None else self.leave - self.enter


@dataclass
class BilletAnalysis:
    events: list[Event]
    visits: dict[str, list[StationVisit]]  # billet name -> visits in time order
    first_seen: dict[str, float]
    last_seen: dict[str, float]
    count: int
    gaps: list[float]  # seconds between consecutive billets first appearing
    dwell: dict[str, list[float]] = field(default_factory=dict)  # station -> dwell per complete visit
    transfer: dict[str, list[float]] = field(default_factory=dict)  # "S1 -> S2" -> seconds, per billet


def _in(rect: Rect, x: float, y: float) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _valid(tr: Track) -> np.ndarray:
    return np.where(~(np.isnan(tr.x) | np.isnan(tr.y)))[0]


def analyze_billets(billets: list[Track], frame_size: tuple[int, int], *, stations: list[Station] | None = None,
                    exit_zone: Rect | None = None, edge_margin: float = 30.0,
                    speed_threshold: float | None = None) -> BilletAnalysis:
    """Billet events and timelines. `frame_size` is (width, height) of the displayed frame."""
    stations = stations or []
    w, h = frame_size
    near_edge = lambda x, y: x < edge_margin or x > w - edge_margin or y < edge_margin or y > h - edge_margin  # noqa: E731
    events: list[Event] = []
    visits: dict[str, list[StationVisit]] = {}
    first_seen: dict[str, float] = {}
    last_seen: dict[str, float] = {}
    dwell: dict[str, list[float]] = {s.name: [] for s in stations}
    transfer: dict[str, list[float]] = {}

    def add(tr, name, when, frame, **d):
        events.append(Event(name, "billet", tr.name, float(when), float(frame), d))

    for tr in billets:
        idx = _valid(tr)
        if len(idx) == 0:
            continue
        i0, i1 = idx[0], idx[-1]
        first_seen[tr.name], last_seen[tr.name] = float(tr.t[i0]), float(tr.t[i1])
        x0, y0 = tr.x[i0], tr.y[i0]
        if exit_zone is not None and _in(exit_zone, x0, y0):
            add(tr, f"{tr.name} exits induction heater", tr.t[i0], tr.frame[i0])
        elif near_edge(x0, y0):
            add(tr, f"{tr.name} enters frame", tr.t[i0], tr.frame[i0])
        else:
            add(tr, f"{tr.name} appears", tr.t[i0], tr.frame[i0])
        x1, y1 = tr.x[i1], tr.y[i1]
        add(tr, f"{tr.name} leaves frame" if near_edge(x1, y1) else f"{tr.name} disappears", tr.t[i1], tr.frame[i1])

        vs: list[StationVisit] = []
        for st in stations:
            open_visit = _in(st.rect, x0, y0)  # already inside when first seen
            cur_enter = None
            for when, fr, kind in rect_crossings(tr, st.rect):
                if kind == "enter":
                    cur_enter, open_visit = when, True
                    add(tr, f"{tr.name} placed in {st.name}", when, fr, station=st.name)
                else:
                    vs.append(StationVisit(st.name, cur_enter if open_visit and cur_enter is not None else None, when))
                    cur_enter, open_visit = None, False
                    add(tr, f"{tr.name} leaves {st.name}", when, fr, station=st.name)
            if open_visit:
                vs.append(StationVisit(st.name, cur_enter, None))
        vs.sort(key=lambda v: (v.enter if v.enter is not None else (v.leave or 0.0)))
        visits[tr.name] = vs
        for v in vs:
            if v.dwell is not None:
                dwell[v.station].append(v.dwell)
        for a, b in zip(vs, vs[1:]):  # transfer: leave of one station to entry of the next
            if a.leave is not None and b.enter is not None and a.station != b.station:
                transfer.setdefault(f"{a.station} -> {b.station}", []).append(b.enter - a.leave)
        if stations:
            last_st = stations[-1].name
            for v in vs:
                if v.station == last_st and v.leave is not None:
                    add(tr, f"{tr.name} leaves press", v.leave, float(np.interp(v.leave, tr.t, tr.frame)))
        if speed_threshold is not None:
            events += moving_state(tr, speed_threshold=speed_threshold)
    firsts = sorted(first_seen.values())
    events = [e if e.group == "billet" else Event(e.name, "billet", e.item, e.t, e.frame, e.details) for e in events]  # moving_state events are custom
    events.sort(key=lambda e: e.t)
    return BilletAnalysis(events, visits, first_seen, last_seen, len(first_seen),
                          [b - a for a, b in zip(firsts, firsts[1:])], dwell, transfer)


def pickup_events(billet: Track, tool: Track, *, max_dist: float, min_hold: int = 3) -> list[Event]:
    """Billet picked up / released: the billet stays within `max_dist` px of a tool or hand track for
    at least `min_hold` consecutive frames. Proximity only; it does not prove the billet is held."""
    common, bi, ti = np.intersect1d(billet.frame, tool.frame, return_indices=True)
    d = np.hypot(billet.x[bi] - tool.x[ti], billet.y[bi] - tool.y[ti])
    near = np.nan_to_num(d, nan=np.inf) <= max_dist
    out, i = [], 0
    while i < len(near):
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(near) and near[j + 1]:
            j += 1
        if j - i + 1 >= min_hold:
            for k, label in ((i, "picked up by"), (j, "released by")):
                out.append(Event(f"{billet.name} {label} {tool.name}", "billet", billet.name,
                                 float(billet.t[bi][k]), float(common[k]), {"tool": tool.name}))
        i = j + 1
    return out
