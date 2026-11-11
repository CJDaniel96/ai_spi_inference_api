"""Idempotent background projection into SQLite, never read by the controller."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.infrastructure.job_history import (
    history_conflicts,
    history_folder,
    journal_events,
    read_receipt,
)


def sync_events(
    state_root: Path, history_root: Path, database: Path, busy_timeout: float
) -> int:
    database.parent.mkdir(parents=True, exist_ok=True)
    conflicts = history_conflicts(state_root, history_root)
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
                processing_started_at REAL, processing_finished_at REAL,
                primary_returned_at REAL, delivery_status TEXT,
                reason TEXT, snapshot TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS history_conflicts (
                job_id TEXT PRIMARY KEY, reason TEXT NOT NULL,
                archived_snapshot TEXT NOT NULL, working_snapshot TEXT NOT NULL
            );
            """
        )
        # Rebuild the old deadline-based projection transactionally. Immutable
        # snapshots/events retain their original contents for historical review.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        if "deadline_at" in columns:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("ALTER TABLE jobs RENAME TO jobs_legacy")
                connection.execute("""CREATE TABLE jobs (
                    job_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                    status TEXT NOT NULL, first_seen_at REAL NOT NULL,
                    processing_started_at REAL, processing_finished_at REAL,
                    primary_returned_at REAL, delivery_status TEXT,
                    reason TEXT, snapshot TEXT NOT NULL)""")
                connection.execute("""INSERT INTO jobs
                    SELECT job_id, revision, status, first_seen_at, NULL, NULL,
                    primary_returned_at, CASE WHEN primary_returned_at IS NULL
                    THEN 'pending' ELSE 'returned' END, reason, snapshot
                    FROM jobs_legacy""")
                connection.execute("DROP TABLE jobs_legacy")
        revisions = dict(connection.execute("SELECT job_id, revision FROM jobs"))
        for event in journal_events(
            state_root, history_root, after=revisions, conflicts=conflicts
        ):
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
                    """INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        revision=excluded.revision, status=excluded.status,
                        first_seen_at=excluded.first_seen_at,
                        processing_started_at=excluded.processing_started_at,
                        processing_finished_at=excluded.processing_finished_at,
                        delivery_status=excluded.delivery_status,
                        primary_returned_at=excluded.primary_returned_at,
                        reason=excluded.reason, snapshot=excluded.snapshot
                    WHERE excluded.revision > jobs.revision""",
                    (
                        snapshot["job_id"],
                        snapshot["revision"],
                        snapshot["status"],
                        snapshot["first_seen_at"],
                        snapshot.get("processing_started_at"),
                        snapshot.get("processing_finished_at"),
                        snapshot["primary_returned_at"],
                        snapshot.get(
                            "delivery_status",
                            "returned"
                            if snapshot["primary_returned_at"]
                            else "pending",
                        ),
                        snapshot["reason"],
                        encoded,
                    ),
                )
            count += inserted
            revisions[event["job_id"]] = event["revision"]
        # A legacy duplicate execution can reuse revisions already committed by
        # the canonical archive. Preserve both snapshots separately; never merge
        # its active events into that archive's revision sequence.
        with connection:
            for job_id, reason in conflicts.items():
                archived = read_receipt(history_folder(history_root, job_id))[
                    "snapshot"
                ]
                events = sorted(
                    (state_root / "jobs" / job_id / "events").glob("*.json")
                )
                working = json.loads(events[-1].read_text(encoding="utf-8"))["snapshot"]
                connection.execute(
                    """INSERT INTO history_conflicts VALUES (?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET reason=excluded.reason,
                        archived_snapshot=excluded.archived_snapshot,
                        working_snapshot=excluded.working_snapshot""",
                    (job_id, reason, json.dumps(archived), json.dumps(working)),
                )
                row = connection.execute(
                    "SELECT snapshot FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if row:
                    snapshot = json.loads(row[0])
                    snapshot.update(
                        status="HISTORY_CONFLICT",
                        reason="history_conflict",
                        last_error=reason,
                    )
                    connection.execute(
                        "UPDATE jobs SET status=?, reason=?, snapshot=? WHERE job_id=?",
                        (
                            "HISTORY_CONFLICT",
                            "history_conflict",
                            json.dumps(snapshot),
                            job_id,
                        ),
                    )
    return count
