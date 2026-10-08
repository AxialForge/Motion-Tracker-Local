"""Optional smoothing and gap filling. Interpolated values are flagged, never passed off as measured."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .model import INTERPOLATED, LOST, Track


def fill_gaps(track: Track, max_gap: int = 10) -> Track:
    """Linearly interpolate LOST runs of at most `max_gap` frames; flag them INTERPOLATED."""
    out = track.copy()
    bad = np.isnan(out.x) | np.isnan(out.y)
    good = ~bad
    if good.sum() < 2:
        return out
    i = 0
    n = len(out)
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:  # only fill gaps with measured data on both sides
            idx = np.arange(i, j)
            out.x[idx] = np.interp(out.t[idx], [out.t[i - 1], out.t[j]], [out.x[i - 1], out.x[j]])
            out.y[idx] = np.interp(out.t[idx], [out.t[i - 1], out.t[j]], [out.y[i - 1], out.y[j]])
            out.flags[idx] = (out.flags[idx] & (0xFF ^ LOST)) | INTERPOLATED
        i = j
    return out


def smooth(track: Track, window: int = 5, polyorder: int = 2) -> Track:
    """Savitzky-Golay smoothing of X and Y over measured/interpolated stretches (NaN gaps untouched)."""
    out = track.copy()
    if window <= polyorder:
        return out
    window += (window + 1) % 2  # savgol needs an odd window
    ok = ~(np.isnan(out.x) | np.isnan(out.y))
    i, n = 0, len(out)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= window:
            out.x[i:j] = savgol_filter(out.x[i:j], window, polyorder)
            out.y[i:j] = savgol_filter(out.y[i:j], window, polyorder)
        i = j
    return out
