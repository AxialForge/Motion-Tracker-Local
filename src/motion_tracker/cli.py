"""Command line: probe a clip, track points, detect ram events, export. (The UI comes later.)"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import analysis, export
from .events import detect_ram
from .ingest import FrameReader, check_timing, frame_timestamps, probe
from .model import Scale
from .session import Session
from .tracking import PointSpec, track_points


def _point(s: str) -> PointSpec:
    """name,x,y[,algo[,seed_frame[,box]]]"""
    p = s.split(",")
    if len(p) < 3:
        raise argparse.ArgumentTypeError("use name,x,y[,algo[,seed_frame[,box]]]")
    return PointSpec(p[0], float(p[1]), float(p[2]), int(p[4]) if len(p) > 4 else 0,
                     p[3] if len(p) > 3 else "csrt", int(p[5]) if len(p) > 5 else 32)


def cmd_probe(a) -> int:
    info = probe(a.clip)
    ts = frame_timestamps(a.clip)
    rep = check_timing(ts)
    print(f"{info.codec}  {info.width}x{info.height}  {info.fps:.3f} fps  {info.duration_s:.2f} s  "
          f"{info.frame_count} frames  rotation {info.rotation}  {'VFR' if info.is_vfr else 'CFR'}")
    print(f"dropped gaps: {rep.dropped[:10] or 'none'}   irregular: {rep.irregular[:10] or 'none'}")
    return 0


def cmd_track(a) -> int:
    info = probe(a.clip)
    reader = FrameReader(a.clip, info)
    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"\r{done}/{total} frames  {done / (time.perf_counter() - t0):.0f} fps", end="", file=sys.stderr)

    tracks = track_points(reader, a.point, a.start, a.stop, progress)
    print(file=sys.stderr)
    with Session(a.session) as s:
        s.set_video(info)
        for tr in tracks:
            s.save_track(analysis.fill_gaps(tr, a.fill_gaps) if a.fill_gaps else tr)
    for tr in tracks:
        lost = int((tr.flags & 2 > 0).sum())
        print(f"{tr.name}: {len(tr)} samples, {lost} lost, mean confidence {tr.conf.mean():.2f}")
    return 0


def cmd_events(a) -> int:
    with Session(a.session) as s:
        tr = analysis.smooth(s.get_track(a.item), a.smooth)
        ram = detect_ram(tr, band=a.band, smooth_window=1)
        s.save_events(ram.events, replace_group="ram")
    head, rows = export.event_rows(ram.events)
    print(f"{len(ram.strokes)} strokes, {len(ram.events)} events")
    h2, r2 = export.summary_rows(ram)
    for r in r2:
        print(f"  {r[0]:<42} n={r[1]} mean={r[4]} min={r[2]} max={r[3]}")
    return 0


def cmd_export(a) -> int:
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    scale = Scale.from_points(a.scale[:2], a.scale[2:4], a.scale[4]) if a.scale else None
    ext = "xlsx" if a.xlsx else "csv"
    with Session(a.session) as s:
        tracks, events = s.load_tracks(), s.load_events()
        written = [export.write_table(out / f"tracks.{ext}", *export.track_rows(tracks, scale)),
                   export.write_table(out / f"events.{ext}", *export.event_rows(events))]
        if events:
            tr = analysis.smooth(next(t for t in tracks if t.name == a.item), a.smooth) if a.item else None
            if tr is not None:
                written.append(export.write_table(out / f"cycle_summary.{ext}", *export.summary_rows(detect_ram(tr, smooth_window=1))))
    for w in written:
        print(w)
    return 0


def cmd_plot(a) -> int:
    from . import plots

    with Session(a.session) as s:
        tracks, events = s.load_tracks(), s.load_events()
    print(plots.plot_tracks(tracks, events, a.out, axis=a.axis))
    return 0


def cmd_run(a) -> int:
    from .runs import Run, run_batch

    print(run_batch(Run.load(a.run), a.clips, a.outdir, xlsx=a.xlsx))
    return 0


def cmd_synth(a) -> int:
    from .synthetic import RamSpec, write_clip

    write_clip(a.out, RamSpec(fps=a.fps, seconds=a.seconds))
    print(a.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mt", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="read clip metadata and timing")
    p.add_argument("clip")
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("track", help="track points and save to a session file")
    p.add_argument("clip")
    p.add_argument("-p", "--point", action="append", type=_point, required=True, metavar="NAME,X,Y[,ALGO[,SEED[,BOX]]]")
    p.add_argument("-s", "--session", default="session.mtp")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int)
    p.add_argument("--fill-gaps", type=int, default=0, help="interpolate lost runs up to N frames (flagged)")
    p.set_defaults(fn=cmd_track)
    p = sub.add_parser("events", help="detect ram events from a tracked point")
    p.add_argument("session")
    p.add_argument("item")
    p.add_argument("--band", type=float, default=0.05)
    p.add_argument("--smooth", type=int, default=5)
    p.set_defaults(fn=cmd_events)
    p = sub.add_parser("export", help="write track data, event log, cycle summary")
    p.add_argument("session")
    p.add_argument("outdir")
    p.add_argument("--item", help="ram track name for the cycle summary")
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--xlsx", action="store_true")
    p.add_argument("--scale", type=float, nargs=5, metavar=("X1", "Y1", "X2", "Y2", "DIST_MM"))
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("plot", help="position-vs-time plot (PNG or PDF) with events marked")
    p.add_argument("session")
    p.add_argument("out")
    p.add_argument("--axis", choices=["x", "y"], default="y")
    p.set_defaults(fn=cmd_plot)
    p = sub.add_parser("run", help="apply a saved run (JSON) to one or many clips")
    p.add_argument("run")
    p.add_argument("clips", nargs="+")
    p.add_argument("-o", "--outdir", default="out")
    p.add_argument("--xlsx", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("synth", help="generate a synthetic press clip with known ground truth")
    p.add_argument("out")
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--seconds", type=float, default=12)
    p.set_defaults(fn=cmd_synth)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
