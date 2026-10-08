"""Exports: track data (CSV/XLSX), event log (CSV/XLSX), cycle summary (CSV/XLSX)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from openpyxl import Workbook

from .billets import BilletAnalysis
from .events import RamAnalysis
from .model import INTERPOLATED, LOST, LOW_CONF, RECOVERED, Event, Scale, Track
from .summary import stats


def flag_text(flags: int) -> str:
    names = [(LOST, "lost"), (INTERPOLATED, "interpolated"), (LOW_CONF, "low_confidence"), (RECOVERED, "recovered")]
    return "|".join(n for bit, n in names if flags & bit) or "measured"


def _num(v: float, nd: int = 4):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def track_rows(tracks: list[Track], scale: Scale | None = None) -> tuple[list[str], list[list]]:
    """One row per sample per item. Units are pixels unless a scale is given (then both are written)."""
    head = ["item", "frame", "timestamp_s", "x_px", "y_px"]
    if scale:
        head += [f"x_{scale.unit}", f"y_{scale.unit}"]
    head += ["confidence", "status"]
    rows = []
    for tr in tracks:
        for f, t, x, y, c, fl in zip(tr.frame, tr.t, tr.x, tr.y, tr.conf, tr.flags):
            r = [tr.name, int(f), _num(t, 6), _num(x), _num(y)]
            if scale:
                r += [_num(x * scale.units_per_px), _num(y * scale.units_per_px)]
            rows.append(r + [_num(c, 3), flag_text(int(fl))])
    return head, rows


def event_rows(events: list[Event]) -> tuple[list[str], list[list]]:
    head = ["timestamp_s", "frame", "event", "group", "item", "since_previous_s"]
    rows, prev = [], None
    for e in sorted(events, key=lambda e: e.t):
        rows.append([_num(e.t, 6), _num(e.frame, 2), e.name, e.group, e.item, "" if prev is None else _num(e.t - prev, 6)])
        prev = e.t
    return head, rows


def summary_rows(ram: RamAnalysis) -> tuple[list[str], list[list]]:
    head = ["metric", "count", "min_s", "max_s", "mean_s", "std_s", "outlier_indices"]
    series = {
        "Cycle time (down-start to down-start)": ram.cycle_times,
        "Stroke time (down-start to top)": [s.total_time for s in ram.strokes],
        "Downstroke time": [s.downstroke_time for s in ram.strokes],
        "Upstroke time": [s.upstroke_time for s in ram.strokes],
        "Dwell at bottom": [s.bottom_dwell for s in ram.strokes],
        "Idle at top": ram.idle_top,
    }
    rows = []
    for name, vals in series.items():
        s = stats(vals)
        rows.append([name, s.n, _num(s.min, 4), _num(s.max, 4), _num(s.mean, 4), _num(s.std, 4), " ".join(map(str, s.outliers))])
    for name, vals in {"Peak downstroke speed (px/s)": [s.down_peak_speed for s in ram.strokes],
                       "Peak upstroke speed (px/s)": [s.up_peak_speed for s in ram.strokes]}.items():
        s = stats(vals)
        rows.append([name, s.n, _num(s.min, 1), _num(s.max, 1), _num(s.mean, 1), _num(s.std, 1), " ".join(map(str, s.outliers))])
    return head, rows


def write_table(path: str | Path, head: list[str], rows: list[list]) -> Path:
    """Write CSV or XLSX depending on the file extension."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        wb = Workbook()
        ws = wb.active
        ws.append(head)
        for r in rows:
            ws.append(r)
        wb.save(path)
    else:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rows)
    return path


def billet_rows(b: BilletAnalysis) -> tuple[list[str], list[list]]:
    """One row per billet: first/last seen, gap since the previous billet, and each station visit."""
    stations = list(b.dwell)
    transfers = list(b.transfer)
    head = ["billet", "first_seen_s", "last_seen_s", "gap_since_previous_s"]
    for st in stations:
        head += [f"{st}_enter_s", f"{st}_leave_s", f"{st}_dwell_s"]
    head += [f"transfer_{k.replace(' -> ', '_to_')}_s" for k in transfers]
    rows, prev = [], None
    for name in sorted(b.first_seen, key=b.first_seen.get):
        r = [name, _num(b.first_seen[name], 4), _num(b.last_seen[name], 4),
             "" if prev is None else _num(b.first_seen[name] - prev, 4)]
        prev = b.first_seen[name]
        visits = {v.station: v for v in b.visits.get(name, [])}
        for st in stations:
            v = visits.get(st)
            r += ["", "", ""] if v is None else [_num(v.enter, 4) if v.enter is not None else "", _num(v.leave, 4) if v.leave is not None else "",
                                                  _num(v.dwell, 4) if v.dwell is not None else ""]
        vs = b.visits.get(name, [])
        for k in transfers:
            a, c = k.split(" -> ")
            va, vc = visits.get(a), visits.get(c)
            r.append(_num(vc.enter - va.leave, 4) if va and vc and va.leave is not None and vc.enter is not None else "")
        rows.append(r)
    return head, rows
