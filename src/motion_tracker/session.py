"""Session file: a local SQLite project holding video reference, tracks, events and settings."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .ingest import VideoInfo
from .model import Event, Track

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, mode TEXT NOT NULL, params TEXT NOT NULL);
CREATE TABLE samples (
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    frame INTEGER NOT NULL, t REAL NOT NULL, x REAL, y REAL, conf REAL NOT NULL, flags INTEGER NOT NULL,
    PRIMARY KEY (track_id, frame)) WITHOUT ROWID;
CREATE TABLE events (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, grp TEXT NOT NULL, item TEXT NOT NULL,
    t REAL NOT NULL, frame REAL NOT NULL, details TEXT NOT NULL);
"""


class Session:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        new = not self.path.exists()
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA foreign_keys = ON")
        if new:
            self.db.executescript(_SCHEMA)
            self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.db.commit()
        elif self.db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise RuntimeError(f"{path}: unsupported session version")

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- meta ----------------------------------------------------------------------------------
    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    def get_meta(self, key: str, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_video(self, info: VideoInfo) -> None:
        self.set_meta("video", info.to_dict())

    def video(self) -> VideoInfo | None:
        d = self.get_meta("video")
        return VideoInfo(**d) if d else None

    # -- tracks --------------------------------------------------------------------------------
    def save_track(self, tr: Track) -> int:
        """Insert or replace (by name) a track and all its samples."""
        self.db.execute("DELETE FROM tracks WHERE name = ?", (tr.name,))
        cur = self.db.execute("INSERT INTO tracks (name, kind, mode, params) VALUES (?,?,?,?)",
                              (tr.name, tr.kind, tr.mode, json.dumps(tr.params)))
        tid = cur.lastrowid
        nan = lambda v: None if np.isnan(v) else float(v)  # noqa: E731
        self.db.executemany(
            "INSERT INTO samples VALUES (?,?,?,?,?,?,?)",
            ((tid, int(f), float(t), nan(x), nan(y), float(c), int(fl))
             for f, t, x, y, c, fl in zip(tr.frame, tr.t, tr.x, tr.y, tr.conf, tr.flags)),
        )
        self.db.commit()
        tr.id = tid
        return tid

    def load_tracks(self) -> list[Track]:
        out = []
        for tid, name, kind, mode, params in self.db.execute("SELECT * FROM tracks ORDER BY id").fetchall():
            rows = self.db.execute(
                "SELECT frame, t, x, y, conf, flags FROM samples WHERE track_id = ? ORDER BY frame", (tid,)).fetchall()
            a = np.array([[r[0], r[1], np.nan if r[2] is None else r[2], np.nan if r[3] is None else r[3],
                           r[4], r[5]] for r in rows], dtype=float).reshape(-1, 6)
            out.append(Track(name, kind, mode, json.loads(params), a[:, 0].astype(np.int32), a[:, 1], a[:, 2], a[:, 3],
                             a[:, 4].astype(np.float32), a[:, 5].astype(np.uint8), tid))
        return out

    def get_track(self, name: str) -> Track:
        for tr in self.load_tracks():
            if tr.name == name:
                return tr
        raise KeyError(name)

    # -- events --------------------------------------------------------------------------------
    def save_events(self, events: list[Event], replace_group: str | None = None) -> None:
        if replace_group:
            self.db.execute("DELETE FROM events WHERE grp = ?", (replace_group,))
        self.db.executemany(
            "INSERT INTO events (name, grp, item, t, frame, details) VALUES (?,?,?,?,?,?)",
            ((e.name, e.group, e.item, e.t, e.frame, json.dumps(e.details)) for e in events))
        self.db.commit()

    def load_events(self) -> list[Event]:
        return [Event(n, g, i, t, f, json.loads(d)) for n, g, i, t, f, d in self.db.execute(
            "SELECT name, grp, item, t, frame, details FROM events ORDER BY t")]
