# Motion Tracker (local)

Local desktop tool that turns phone video of a manufacturing process into timed X/Y data and
cycle events (press line: ram strokes, cycle times). Offline, CPU + GPU. See the feature list for
the full plan; this repo is built in phases.

**Status: Milestone 1, headless engine.** No UI yet. Everything is scored against synthetic video
with known ground truth.

## What works now

- Ingest: auto metadata (ffprobe), real per-frame timestamps (VFR-safe), dropped/irregular frame flags,
  rotation from metadata, blown-out exposure check.
- Dot tracking: CSRT (precision) or KCF (speed) per point, any number of points in one decode pass,
  colour-lock recovery after brief loss, per-sample confidence, segment runs and cancel.
- Smoothing and gap filling (interpolated values are flagged, never passed off as measured).
- Ram events with sub-frame times: start/reach of downstroke and upstroke, per-stroke times, dwell,
  idle at top, peak/mean speed, cycle times, min/max/mean with outlier flags.
- Generic reusable events: line crossing, zone enter/leave, start/stop moving, min/max.
- Session file (SQLite `.mtp`), export of track data, event log, cycle summary (CSV/XLSX), optional scale.

Not yet: any UI, billet/zone/operator modes, NVDEC decode, annotated video, PDF/PNG outputs,
backward re-tracking, stabilization, multi-phone sync.

## Setup (Windows)

Needs Python 3.11 or 3.12 and FFmpeg (`ffmpeg`, `ffprobe`) on PATH.

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Try it on a clip

```
mt probe clips\press7.mov
mt track clips\press7.mov -p ram,812,430 -s press7.mtp      # name,x,y[,algo[,seed_frame[,box]]]
mt events press7.mtp ram
mt export press7.mtp out --item ram --xlsx
```

`mt track` prints frames/second as it runs: that is your benchmark number on your laptop.
Click coordinates are in the displayed (rotated) frame, in pixels.

No footage handy: `mt synth clips\fake.mp4` makes a fake press clip.
Keep real footage in `clips\` (git-ignored).
