"""Plots: position versus time per item with events marked; PNG or PDF by file extension."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .model import INTERPOLATED, LOST, LOW_CONF, Event, Scale, Track  # noqa: E402


def plot_tracks(tracks: list[Track], events: list[Event], path: str | Path, *, axis: str = "auto",
                scale: Scale | None = None, title: str | None = None) -> Path:
    """One lane per item (shared time axis). Low-confidence and interpolated stretches are marked,
    lost stretches are left as gaps, events are vertical lines labelled by name."""
    k = scale.units_per_px if scale else 1.0
    unit = scale.unit if scale else "px"
    fig, axes = plt.subplots(len(tracks), 1, sharex=True, figsize=(11, 2.6 * len(tracks) + 0.8), squeeze=False)
    for ax, tr in zip(axes[:, 0], tracks):
        an = axis if axis != "auto" else ("x" if tr.kind == "billet" else "y")  # billets travel along X
        v = (tr.x if an == "x" else tr.y) * k
        ax.plot(tr.t, v, lw=1.3, color="#1f4e9c", label=tr.name)
        weak = ((tr.flags & (LOW_CONF | INTERPOLATED)) > 0) & ~np.isnan(v)
        ax.plot(tr.t[weak], v[weak], ".", ms=4, color="#d9822b", label="low confidence / interpolated")
        lost = (tr.flags & LOST) > 0
        for t0, t1 in _runs(tr.t, lost):
            ax.axvspan(t0, t1, color="#c0392b", alpha=0.12, lw=0)
        for e in events:
            if e.item == tr.name:
                ax.axvline(e.t, color="#777", lw=0.6, ls="--")
        ax.set_ylabel(f"{tr.name}  {an.upper()} ({unit})")
        if an == "y":
            ax.invert_yaxis()  # image Y grows downward; plot so up means up
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel("time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _runs(t: np.ndarray, mask: np.ndarray):
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        yield t[i], t[min(j + 1, len(t) - 1)]
        i = j + 1
