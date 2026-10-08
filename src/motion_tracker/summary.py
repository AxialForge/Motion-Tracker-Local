"""Cycle statistics: count, min, max, average, spread, outlier flags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Event


@dataclass(frozen=True)
class Stats:
    n: int
    min: float
    max: float
    mean: float
    std: float
    outliers: list[int]  # indices flagged by the IQR rule (needs n >= 4)


def stats(values) -> Stats:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return Stats(0, float("nan"), float("nan"), float("nan"), float("nan"), [])
    out: list[int] = []
    if len(v) >= 4:
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        out = [int(i) for i in np.where((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr))[0]]
    return Stats(len(v), float(v.min()), float(v.max()), float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, out)


def durations_between(events: list[Event], start_name: str, end_name: str) -> list[float]:
    """Cycle/segment times: each `start_name` event paired with the next `end_name` event after it.

    The user picks any two events as the bounds.
    """
    starts = sorted(e.t for e in events if e.name == start_name)
    ends = sorted(e.t for e in events if e.name == end_name)
    out = []
    for s in starts:
        nxt = next((e for e in ends if e > s), None)
        if nxt is not None:
            out.append(nxt - s)
    return out
