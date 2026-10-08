"""Saved, reusable analysis runs: tracked points plus event analyses, applied to one clip or a batch.

A run is a small JSON file. Point coordinates are in the displayed frame, so a run only makes sense
for clips filmed with the same fixed framing; batch results flag clips where a point looks wrong.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import analysis, export, plots
from .events import RamAnalysis, detect_ram, extrema, line_cross, moving_state, zone_transitions
from .ingest import FrameReader, probe
from .model import Event, Scale
from .session import Session
from .summary import Stats, durations_between, stats
from .tracking import PointSpec, track_points

LOW_HEALTH = 0.5  # mean confidence below this suggests the point was seeded in the wrong place


@dataclass
class Run:
    name: str
    points: list[dict]
    analyses: list[dict] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)  # [{"name", "start", "end"}] event-name bounds
    fill_gaps: int = 0
    smooth: int = 5
    scale: dict | None = None  # {"p1": [x, y], "p2": [x, y], "distance": 500, "unit": "mm"}

    @classmethod
    def load(cls, path: str | Path) -> Run:
        d = json.loads(Path(path).read_text())
        run = cls(**d)
        run.validate()
        return run

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    def validate(self) -> None:
        names = {p["name"] for p in self.points}
        if not names or len(names) != len(self.points):
            raise ValueError("run needs at least one point, with unique names")
        kinds = {"ram", "line_cross", "zone", "moving", "extrema"}
        for a in self.analyses:
            if a.get("type") not in kinds:
                raise ValueError(f"unknown analysis type {a.get('type')!r}; use one of {sorted(kinds)}")
            if a.get("item") not in names:
                raise ValueError(f"analysis refers to unknown point {a.get('item')!r}")

    def point_specs(self) -> list[PointSpec]:
        return [PointSpec(p["name"], p["x"], p["y"], p.get("seed_frame", 0), p.get("algo", "csrt"), p.get("box", 32))
                for p in self.points]

    def scale_obj(self) -> Scale | None:
        s = self.scale
        return Scale.from_points(s["p1"], s["p2"], s["distance"], s.get("unit", "mm")) if s else None


@dataclass
class RunResult:
    clip: str
    events: list[Event]
    rams: dict[str, RamAnalysis]
    segment_stats: dict[str, Stats]
    cycle_stats: dict[str, Stats]
    warnings: list[str]


def _analyse(run: Run, tracks) -> tuple[list[Event], dict[str, RamAnalysis]]:
    by_name = {t.name: analysis.smooth(t, run.smooth) if run.smooth else t for t in tracks}
    events: list[Event] = []
    rams: dict[str, RamAnalysis] = {}
    for a in run.analyses:
        tr = by_name[a["item"]]
        kind = a["type"]
        if kind == "ram":
            r = detect_ram(tr, axis=a.get("axis", "y"), down_is_positive=a.get("down_is_positive", True),
                           band=a.get("band", 0.05), smooth_window=1)
            rams[tr.name] = r
            events += r.events
        elif kind == "line_cross":
            events += line_cross(tr, axis=a["axis"], value=a["value"], direction=a.get("direction", "either"),
                                 name=a.get("name"), group=a.get("group", "custom"))
        elif kind == "zone":
            events += zone_transitions(tr, tuple(a["rect"]), a["zone"])
        elif kind == "moving":
            events += moving_state(tr, speed_threshold=a["speed_threshold"], min_hold=a.get("min_hold", 3))
        elif kind == "extrema":
            events += extrema(tr, axis=a.get("axis", "y"), prominence_frac=a.get("prominence_frac", 0.3))
    return sorted(events, key=lambda e: e.t), rams


def apply_run(run: Run, clip: str, session_path: str | Path, *, plot_path: str | Path | None = None,
              progress=None, cancel=None) -> RunResult:
    """Track, analyse, and save everything for one clip into its own session file."""
    info = probe(clip)
    tracks = track_points(FrameReader(clip, info), run.point_specs(), progress=progress, cancel=cancel)
    warnings = []
    for tr in tracks:
        if len(tr) and float(tr.conf.mean()) < LOW_HEALTH:
            warnings.append(f"{tr.name}: mean confidence {tr.conf.mean():.2f}; check the seed position")
        if len(tr) and (tr.flags & 2 > 0).mean() > 0.2:
            warnings.append(f"{tr.name}: lost on {(tr.flags & 2 > 0).mean():.0%} of frames")
    if run.fill_gaps:
        tracks = [analysis.fill_gaps(t, run.fill_gaps) for t in tracks]
    events, rams = _analyse(run, tracks)
    seg = {s["name"]: stats(durations_between(events, s["start"], s["end"])) for s in run.segments}
    cyc = {}
    for item, r in rams.items():
        cyc[f"{item}: cycle time"] = stats(r.cycle_times)
        cyc[f"{item}: stroke time"] = stats([s.total_time for s in r.strokes])
    with Session(session_path) as s:
        s.set_video(info)
        s.set_meta("run", run.__dict__)
        for t in tracks:
            s.save_track(t)
        s.save_events(events, replace_group=None)
    if plot_path:
        plots.plot_tracks(tracks, events, plot_path, scale=run.scale_obj(), title=Path(clip).name)
    return RunResult(clip, events, rams, seg, cyc, warnings)


def run_batch(run: Run, clips: list[str], outdir: str | Path, *, xlsx: bool = False) -> Path:
    """Apply one saved run to many clips; per-clip sessions, exports and plots, plus one combined summary."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ext = "xlsx" if xlsx else "csv"
    rows = []
    for clip in clips:
        stem = Path(clip).stem
        d = out / stem
        d.mkdir(exist_ok=True)
        res = apply_run(run, clip, d / f"{stem}.mtp", plot_path=d / f"{stem}.png")
        with Session(d / f"{stem}.mtp") as s:
            export.write_table(d / f"tracks.{ext}", *export.track_rows(s.load_tracks(), run.scale_obj()))
            export.write_table(d / f"events.{ext}", *export.event_rows(res.events))
        for item, r in res.rams.items():
            export.write_table(d / f"cycle_summary_{item}.{ext}", *export.summary_rows(r))
        for name, st in {**res.cycle_stats, **res.segment_stats}.items():
            rows.append([stem, name, st.n, _r(st.mean), _r(st.min), _r(st.max), _r(st.std), "; ".join(res.warnings)])
        if not (res.cycle_stats or res.segment_stats):
            rows.append([stem, "(no metrics)", 0, "", "", "", "", "; ".join(res.warnings)])
    path = out / f"batch_summary.{ext}"
    export.write_table(path, ["clip", "metric", "n", "mean_s", "min_s", "max_s", "std_s", "warnings"], rows)
    return path


def _r(v: float):
    return "" if v != v else round(v, 4)
