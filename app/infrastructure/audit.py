"""Idempotent background projection into SQLite, never read by the controller."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.infrastructure.job_history import journal_events


def sync_events(
    state_root: Path, history_root: Path, database: Path, busy_timeout: float
) -> int:
    database.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with sqlite3.connect(database, timeout=busy_timeout) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, job_id TEXT NOT NULL,
                revision INTEGER NOT NULL, event_type TEXT NOT NULL,
                recorded_at REAL NOT NULL, snapshot TEXT NOT NULL,
                UNIQUE(job_id, revision)
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                status TEXT NOT NULL, first_seen_at REAL NOT NULL,
                deadline_at REAL NOT NULL, primary_returned_at REAL,
                reason TEXT, snapshot TEXT NOT NULL
            );
            """
        )
        revisions = dict(connection.execute("SELECT job_id, revision FROM jobs"))
        for event in journal_events(state_root, history_root, after=revisions):
            snapshot = event["snapshot"]
            encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
            with connection:
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event["event_id"],
                        event["job_id"],
                        event["revision"],
                        event["event_type"],
                        event["recorded_at"],
                        encoded,
                    ),
                ).rowcount
                connection.execute(
                    """INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        revision=excluded.revision, status=excluded.status,
                        first_seen_at=excluded.first_seen_at,
                        deadline_at=excluded.deadline_at,
                        primary_returned_at=excluded.primary_returned_at,
                        reason=excluded.reason, snapshot=excluded.snapshot
                    WHERE excluded.revision > jobs.revision""",
                    (
                        snapshot["job_id"],
                        snapshot["revision"],
                        snapshot["status"],
                        snapshot["first_seen_at"],
                        snapshot["deadline_at"],
                        snapshot["primary_returned_at"],
                        snapshot["reason"],
                        encoded,
                    ),
                )
            count += inserted
            revisions[event["job_id"]] = event["revision"]
    return count
