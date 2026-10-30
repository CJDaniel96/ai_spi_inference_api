"""Export offline SPI evidence to Excel-friendly CSVs using only the stdlib.

Run ``python app/tools/spi_statistics.py --help`` or see docs/spi_statistics.md.
Config mode reads configured locations; folder mode reads only collected evidence.
Database contents are never modified and embedded job paths are never followed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

UTC = timezone.utc  # noqa: UP017 -- standalone tool also supports Python 3.10
TZ8 = timezone(timedelta(hours=8))
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
STATE_TIMES = (
    "first_seen_at",
    "csv_ready_at",
    "expired_at",
    "ready_at",
    "inference_started_at",
    "result_ready_at",
    "primary_returned_at",
    "completed_at",
    "deadline_at",
)
EXTERNAL_TIMES = ("spi_sent_at", "spi_read_at", "op_confirm_at")
TIME_FIELDS = (
    *STATE_TIMES,
    "manifest_created_at",
    "request_start_at",
    "request_end_at",
    *EXTERNAL_TIMES,
)
JOB_COLUMNS = [
    "machine_id",
    "job_id",
    "execution_mode",
    "report_date",
    "report_date_source",
    "selection_time_field",
    "selection_at",
    "status",
    "state_reason",
    "deadline_start_source",
    "input_mode",
    "image_backup_status",
    "publication_unknown",
    "published_csv_count",
    "publication_evidence",
    "manifest_selection",
    "manifest_fingerprint",
    "manifest_variants",
    "source_csvs",
    "outcome",
    "reason",
    "errors",
    "row_count",
    "pass_count",
    "fail_count",
    "image_count",
    "image_count_source",
    "spi_alarm_count",
    "image_threshold",
    "threshold_source",
    "image_threshold_exceeded",
    "threshold_evidence",
    "image_bucket",
    *TIME_FIELDS,
    "pre_ready_s",
    "waiting_s",
    "inference_processing_s",
    "publish_s",
    "internal_total_s",
    "end_to_end_s",
    "end_to_end_budget_s",
    "request_total_s",
    "machine_wait_s",
    "read_after_write_s",
    "late_after_op_s",
    "late_after_op",
    "deadline_met",
    "ingest_attempts",
    "inference_attempts",
    "publish_attempts",
    "legacy_request_count",
    "manifest_sources",
    "state_sources",
    "log_sources",
    "event_sources",
    "field_sources",
    "missing_fields",
    "data_issues",
]
ISSUE_COLUMNS = ["severity", "job_id", "source", "message"]
MODEL_COLUMNS = [
    "machine_id",
    "job_id",
    "execution_mode",
    "model",
    "request_ms",
    "timing_source",
    "manifest_selection",
]


def number(value: Any) -> float | None:
    """Accept finite numbers, never turn missing or malformed values into zero."""
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite number: {value!r}")
    return result


def integer(value: Any) -> int | None:
    result = number(value)
    if result is not None and (result < 0 or not result.is_integer()):
        raise ValueError(f"Expected nonnegative integer: {value!r}")
    return int(result) if result is not None else None


def timestamp(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "None", "null"}:
        return None
    try:
        result = number(value)
    except ValueError:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError(f"Timestamp requires timezone: {value!r}") from None
        result = parsed.timestamp()
    if result is not None:
        datetime.fromtimestamp(result, UTC)  # reject out-of-range epoch values
    return result


def path_text(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/")


def job_from_folder(value: str) -> str:
    return path_text(value).split("/")[-1]


def bucket(count: int | None) -> str:
    if count is None:
        return "unknown"
    for upper, label in (
        (0, "0"),
        (10, "1-10"),
        (30, "11-30"),
        (60, "31-60"),
        (100, "61-100"),
    ):
        if count <= upper:
            return label
    return "101+"


def log_values(line: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z_]\w*)=", line))
    return {
        match[1]: line[
            match.end() : matches[i + 1].start() if i + 1 < len(matches) else len(line)
        ].strip()
        for i, match in enumerate(matches)
    }


class Evidence:
    """Collect one machine's evidence without loading any inference dependencies."""

    def __init__(
        self,
        root: Path,
        output: Path,
        machine_id: str,
        threshold: int | None = None,
        threshold_source: str = "",
        extensions: set[str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.output = output.resolve()
        self.machine_id = machine_id
        self.threshold = threshold
        self.threshold_source = threshold_source
        self.extensions = extensions or DEFAULT_EXTENSIONS
        self.issues: list[dict] = []
        self.manifests: dict[str, dict[str, dict]] = defaultdict(dict)
        self.states: dict[str, list[tuple[dict, str]]] = defaultdict(list)
        self.logs: dict[str, list[dict]] = defaultdict(list)
        self.legacy: dict[str, list[tuple[dict, str]]] = defaultdict(list)
        self.events: dict[str, list[tuple[dict, str]]] = defaultdict(list)
        self.files: list[Path] = []
        self.raw_roots: dict[str, set[Path]] = defaultdict(set)
        self.image_counts: dict[Path, int] = defaultdict(int)
        self.source_jobs: dict[str, str] = {}
        self.log_files: set[Path] = set()
        self.request_log_files: set[Path] = set()
        self.database_files: set[Path] = set()
        self.live_database = False

    def issue(
        self, source: Any, message: str, job_id: str = "", severity: str = "warning"
    ) -> None:
        self.issues.append(
            dict(severity=severity, job_id=job_id, source=str(source), message=message)
        )

    def discover(self, sources: list[Path] | None = None) -> None:
        found: set[Path] = set()
        for source in sources if sources is not None else [self.root]:
            if source.is_symlink():
                self.issue(source, "Symbolic-link evidence source ignored")
                continue
            if source.is_file():
                found.add(source.resolve())
                continue
            if not source.is_dir():
                self.issue(source, "Configured source does not exist")
                continue
            for directory, dirs, names in os.walk(
                source,
                followlinks=False,
                onerror=lambda exc: self.issue(
                    exc.filename, f"Cannot scan directory: {exc}", severity="error"
                ),
            ):
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if not (Path(directory) / d).is_symlink()
                    and (Path(directory) / d).resolve() != self.output
                )
                found.update(
                    (Path(directory) / n).resolve()
                    for n in names
                    if not (Path(directory) / n).is_symlink()
                )
        self.files = sorted(found)
        for path in self.files:
            try:
                lower = path.name.lower()
                if path in self.database_files:
                    self.read_database(path)
                elif path in self.log_files:
                    self.read_log(path)
                elif path in self.request_log_files:
                    self.read_csv(path)
                elif path.suffix.lower() == ".json":
                    self.read_json(path)
                elif path.suffix.lower() in {".sqlite3", ".sqlite", ".db"}:
                    self.read_database(path)
                elif lower.startswith("log.csv") or path.suffix.lower() == ".csv":
                    self.read_csv(path)
                elif lower.startswith("system") or path.suffix.lower() in {
                    ".log",
                    ".txt",
                }:
                    self.read_log(path)
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                OverflowError,
                sqlite3.Error,
            ) as exc:
                self.issue(
                    path,
                    f"Cannot read evidence: {exc}",
                    self.source_jobs.get(str(path), ""),
                    "error",
                )
        # Index once: avoid walking every image again for every production job.
        for path in self.files:
            if path.suffix.lower() == ".csv" and "csv_inputs" not in path.parts:
                self.raw_roots[path.parent.name].add(path.parent)
        roots = set().union(*self.raw_roots.values()) if self.raw_roots else set()
        for path in self.files:
            if path.suffix.lower() in self.extensions:
                for parent in path.parents:
                    if (
                        parent in roots
                        and "ai_result" not in path.relative_to(parent).parts
                    ):
                        self.image_counts[parent] += 1

    def read_json(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not data.get("job_id"):
            return
        job_id = str(data["job_id"])
        self.source_jobs[str(path)] = job_id
        if "csv_results" in data:
            # Keep a known job in the denominator even if its manifest is invalid.
            self.manifests.setdefault(job_id, {})
            if data.get("schema_version") != 1:
                raise ValueError("Unsupported manifest schema_version")
            if data.get("outcome") not in {"normal", "fallback"}:
                raise ValueError("Invalid manifest outcome")
            if not isinstance(data["csv_results"], list) or not data["csv_results"]:
                raise ValueError("Manifest has no CSV results")
            if not isinstance(data.get("counts", {}), dict) or not isinstance(
                data.get("model_timings_ms", {}), dict
            ):
                raise ValueError("Manifest counts/model_timings_ms must be objects")
            if timestamp(data["created_at"]) is None:
                raise ValueError("Manifest created_at is missing")
            if not isinstance(data.get("errors", []), list) or any(
                not isinstance(error, str) for error in data.get("errors", [])
            ):
                raise ValueError("Manifest errors must be a list of strings")
            names = set()
            for item in data["csv_results"]:
                if not isinstance(item, dict) or not isinstance(
                    item.get("result_codes"), list
                ):
                    raise ValueError(
                        "Manifest CSV result must contain a result_codes list"
                    )
                name = str(item["source_csv"])
                for relative in (name, item["processed_csv"]):
                    normalized = path_text(relative)
                    if (
                        normalized.startswith("/")
                        or ":" in normalized
                        or ".." in normalized.split("/")
                    ):
                        raise ValueError("Manifest contains an unsafe artifact path")
                if name in names:
                    raise ValueError("Duplicate source_csv in manifest")
                names.add(name)
                if any(
                    type(v) is not int or v not in {22, 23}
                    for v in item["result_codes"]
                ):
                    raise ValueError("Manifest result codes must be 22 or 23")
            for value in data.get("model_timings_ms", {}).values():
                if number(value) is None or number(value) < 0:
                    raise ValueError("Invalid model timing")
            token = hashlib.sha256(
                json.dumps(
                    data,
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            item = self.manifests[job_id].setdefault(
                token,
                {"data": data, "paths": [], "token": token},
            )
            item["paths"].append(path)
        elif "status" in data and "ready_at" in data:
            self.states[job_id].append((data, str(path)))

    def read_database(self, path: Path) -> None:
        # Live sources require SQLite backup, not separate copies of changing
        # DB/WAL files. Collected offline evidence keeps its no-sidecar-write mode.
        with tempfile.TemporaryDirectory(prefix="spi-statistics-") as temporary:
            copied = Path(temporary) / path.name
            if self.live_database:
                source = sqlite3.connect(
                    f"{path.as_uri()}?mode=ro", uri=True, timeout=5
                )
                try:
                    target = sqlite3.connect(copied)
                    try:
                        deadline = time.monotonic() + 30

                        def progress(_status, _remaining, _total):
                            if time.monotonic() > deadline:
                                raise TimeoutError(
                                    "SQLite snapshot exceeded 30 seconds"
                                )

                        source.backup(target, pages=256, progress=progress, sleep=0.05)
                    finally:
                        target.close()
                finally:
                    source.close()
            else:
                shutil.copyfile(path, copied)
                wal = path.with_name(path.name + "-wal")
                if wal.is_file() and not wal.is_symlink():
                    shutil.copyfile(wal, copied.with_name(copied.name + "-wal"))
            connection = sqlite3.connect(
                f"{copied.as_uri()}?mode=ro", uri=True, timeout=5
            )
            try:
                if not connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='pipeline_jobs'"
                ).fetchone():
                    return
                connection.row_factory = sqlite3.Row
                has_publications = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='pipeline_publications'"
                ).fetchone()
                for row in connection.execute("SELECT * FROM pipeline_jobs").fetchall():
                    data = dict(row)
                    if has_publications and data.get("first_seen_at") is not None:
                        data["published_csv_count"] = connection.execute(
                            "SELECT COUNT(*) FROM pipeline_publications "
                            "WHERE job_id = ?",
                            (data["job_id"],),
                        ).fetchone()[0]
                    self.states[str(data["job_id"])].append((data, str(path)))
            finally:
                connection.close()

    def read_csv(self, path: Path, *, events_only: bool = False) -> None:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            event_file = (
                events_only
                or path.name.lower() == "events.csv"
                or path.name.lower().endswith("_events.csv")
            )
            is_event = (
                event_file
                and "job_id" in fields
                and bool(
                    fields
                    & {
                        *EXTERNAL_TIMES,
                        "spi_alarm_count",
                        "image_count",
                        "image_threshold",
                    }
                )
            )
            is_legacy = {"job_folder", "request_start_at"} <= fields
            if event_file and not is_event:
                raise ValueError("Events CSV requires job_id and event/count columns")
            if not is_event and not is_legacy:
                return
            for index, row in enumerate(reader, 2):
                source = f"{path}:{index}"
                if None in row:
                    self.issue(source, "CSV has extra fields", severity="error")
                    continue
                if is_event:
                    if row.get("machine_id") not in {None, "", self.machine_id}:
                        self.issue(
                            source,
                            "machine_id differs from --machine-id",
                            str(row.get("job_id", "")),
                            "error",
                        )
                        continue
                    job_id = str(row.get("job_id") or "").strip()
                    if job_id:
                        self.events[job_id].append((row, source))
                else:
                    job_id = job_from_folder(row["job_folder"])
                    if job_id:
                        self.legacy[job_id].append((row, source))

    def read_log(self, path: Path) -> None:
        with path.open(encoding="utf-8-sig", errors="replace") as handle:
            for index, line in enumerate(handle, 1):
                if "event=" not in line:
                    continue
                values = log_values(line)
                if values.get("event") not in {
                    "pipeline.ingest.enqueued",
                    "pipeline.primary_returned",
                    "process.summary",
                    "process.skip",
                    "process.fallback",
                }:
                    continue
                job_id = values.get("job_id") or job_from_folder(
                    values.get("job_folder", "")
                )
                if job_id:
                    values["_source"] = f"{path}:{index}"
                    self.logs[job_id].append(values)

    def choose_manifest(
        self, job_id: str, state: dict, published: dict
    ) -> tuple[dict | None, str]:
        variants = list(self.manifests.get(job_id, {}).values())
        if not variants:
            return None, "missing"
        eligible = variants
        if published:
            eligible = [
                v
                for v in variants
                if v["data"]["outcome"] == published.get("outcome")
                and (v["data"].get("reason") or "-") == published.get("reason", "-")
                and timestamp(v["data"]["created_at"])
                <= timestamp(published["returned_at"])
            ]
        target = path_text(state.get("result_manifest_path"))
        if target:
            matches = [
                v
                for v in eligible
                if any(
                    target == path_text(p)
                    or target.endswith("/" + "/".join(p.parts[-3:]))
                    for p in v["paths"]
                )
            ]
            if len(matches) == 1:
                return matches[0], "state_reference"
        backups = [
            v for v in eligible if any(p.parent.name == "ai_result" for p in v["paths"])
        ]
        if len(backups) == 1:
            return backups[0], "published_backup"
        if published:
            if len(eligible) == 1:
                return eligible[0], "publisher_log_match"
        if len(variants) == 1 and not target and not published:
            return variants[0], "single_unverified"
        # Do not guess that the newest inference attempt was actually published.
        self.issue(
            "manifest",
            "Cannot identify the final manifest; counts and "
            "model timings left blank. See manifest_candidates.csv.",
            job_id,
        )
        return None, "ambiguous"

    def images(self, job_id: str, names: list[str]) -> int | None:
        if not names:
            return None
        directories = self.raw_roots.get(job_id, set())
        counts = set()
        for root in directories:
            if not all((root / name).is_file() for name in names):
                continue
            counts.add(self.image_counts[root])
        if len(counts) == 1:
            return counts.pop()
        if counts:
            self.issue(
                "raw_backup", "Copied raw folders have different image counts", job_id
            )
        return None

    def build(self) -> tuple[list[dict], list[dict], list[dict]]:
        rows, models, candidates = [], [], []
        jobs = sorted(
            set(self.manifests)
            | set(self.states)
            | set(self.logs)
            | set(self.legacy)
            | set(self.events)
        )
        for job_id in jobs:
            try:
                row, model_rows = self.build_job(job_id)
                rows.append(row)
                models.extend(model_rows)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                self.issue("job", f"Cannot normalize job: {exc}", job_id, "error")
                # Retain the job in the denominator, even when its data is invalid.
                rows.append(
                    dict(
                        machine_id=self.machine_id,
                        job_id=job_id,
                        report_date="unknown",
                        image_bucket="unknown",
                        data_issues=str(exc),
                    )
                )
            for variant in self.manifests.get(job_id, {}).values():
                data = variant["data"]
                candidates.append(
                    dict(
                        job_id=job_id,
                        fingerprint=variant["token"],
                        outcome=data["outcome"],
                        reason=data.get("reason"),
                        manifest_created_at=format_time(timestamp(data["created_at"])),
                        sources=" | ".join(map(str, variant["paths"])),
                    )
                )
        return rows, models, candidates

    def build_job(self, job_id: str) -> tuple[dict, list[dict]]:
        row: dict[str, Any] = dict(machine_id=self.machine_id, job_id=job_id)
        provenance = {}

        def assign(field: str, value: Any, source: str) -> None:
            if value is not None and value != "":
                previous = row.get(field)
                different = previous is not None and previous != value
                if (
                    different
                    and isinstance(previous, (int, float))
                    and isinstance(value, (int, float))
                ):
                    different = abs(previous - value) > 0.00001
                if different:
                    self.issue(
                        source, f"Conflicting {field}; selected this source", job_id
                    )
                row[field] = value
                provenance[field] = source

        states = self.states.get(job_id, [])
        state, state_source = max(
            states,
            key=lambda pair: timestamp(pair[0].get("updated_at")) or 0,
            default=({}, ""),
        )
        row["state_sources"] = " | ".join(sorted({s for _, s in states}))
        assign("state_reason", state.get("reason"), state_source)
        for field in STATE_TIMES:
            assign(field, timestamp(state.get(field)), state_source)
        for field in (
            "status",
            "ingest_attempts",
            "inference_attempts",
            "publish_attempts",
            "input_mode",
            "image_backup_status",
            "publication_unknown",
            "published_csv_count",
        ):
            assign(field, state.get(field), state_source)
        logs = self.logs.get(job_id, [])
        row["log_sources"] = " | ".join(sorted({v["_source"] for v in logs}))
        published = max(
            (v for v in logs if v["event"] == "pipeline.primary_returned"),
            key=lambda v: timestamp(v["returned_at"]),
            default={},
        )
        for log in logs:
            if log["event"] == "pipeline.ingest.enqueued":
                for field in ("ready_at", "deadline_at"):
                    if field not in row:
                        assign(field, timestamp(log.get(field)), log["_source"])
        if published:
            for field, log_key in (
                ("primary_returned_at", "returned_at"),
                ("ready_at", "ready_at"),
                ("deadline_at", "deadline_at"),
            ):
                assign(field, timestamp(published.get(log_key)), published["_source"])
            assign("outcome", published.get("outcome"), published["_source"])
            assign(
                "reason",
                published.get("reason") if published.get("reason") != "-" else None,
                published["_source"],
            )

        variant, selection = self.choose_manifest(job_id, state, published)
        row["manifest_selection"] = selection
        row["manifest_variants"] = len(self.manifests.get(job_id, {}))
        model_values = {}
        model_source = ""
        if variant:
            data = variant["data"]
            row["manifest_fingerprint"] = variant["token"]
            model_source = " | ".join(map(str, variant["paths"]))
            row["manifest_sources"] = model_source
            for field in ("outcome", "reason"):
                value = data.get(field)
                if field in row and row[field] != value:
                    self.issue(
                        model_source,
                        f"Published {field} conflicts with manifest",
                        job_id,
                    )
                else:
                    assign(field, value, model_source)
            assign("manifest_created_at", timestamp(data["created_at"]), model_source)
            codes = [
                code for item in data["csv_results"] for code in item["result_codes"]
            ]
            for field, value in (
                ("row_count", len(codes)),
                ("pass_count", codes.count(22)),
                ("fail_count", codes.count(23)),
            ):
                assign(field, value, model_source)
                if field in data.get("counts", {}) and data["counts"][field] != value:
                    self.issue(
                        model_source,
                        f"Manifest {field} differs from result_codes",
                        job_id,
                    )
            names = [item["source_csv"] for item in data["csv_results"]]
            row["source_csvs"] = " | ".join(names)
            image_count = self.images(job_id, names)
            if image_count is not None and image_count < len(codes):
                self.issue(
                    "raw_backup",
                    "Fewer copied images than manifest rows; "
                    "image count ignored (copy may be incomplete)",
                    job_id,
                )
            else:
                assign("image_count", image_count, "copied_raw_folder")
            row["errors"] = json.dumps(data.get("errors", []), ensure_ascii=False)
            model_values = data.get("model_timings_ms", {})
            for error in data.get("errors", []):
                match = re.search(r"Image count (\d+) exceeds threshold (\d+)", error)
                if match:
                    assign("image_count", int(match[1]), "manifest_error")
                    assign("image_threshold", int(match[2]), "manifest_error")

        legacy = self.legacy.get(job_id, [])
        if legacy:
            unique = {json.dumps(v, sort_keys=True): (v, s) for v, s in legacy}
            row["legacy_request_count"] = len(unique)
            if states or self.manifests.get(job_id) or published:
                self.issue(
                    "legacy_log",
                    "Both durable and /process evidence exist; "
                    "legacy data not merged into this pipeline job",
                    job_id,
                )
            else:
                latest, source = max(
                    unique.values(),
                    key=lambda p: timestamp(p[0]["request_start_at"]) or 0,
                )
                for field in ("request_start_at", "request_end_at"):
                    assign(field, timestamp(latest.get(field)), source)
                for field, old in (
                    ("image_count", "img_numbers"),
                    ("pass_count", "pass_count"),
                    ("fail_count", "fail_count"),
                ):
                    assign(field, integer(latest.get(old)), source)
                if "pass_count" in row and "fail_count" in row:
                    row["row_count"] = row["pass_count"] + row["fail_count"]
                assign(
                    "request_total_s",
                    seconds_ms(latest.get("request_latency_ms")),
                    source,
                )
                model_values = {
                    name: latest[f"{name}_request_ms"]
                    for name in ("anomaly", "paste", "distance")
                    if latest.get(f"{name}_request_ms") not in {None, ""}
                    and number(latest[f"{name}_request_ms"]) > 0
                }
                model_source = source
                row["execution_mode"] = "legacy"
        row.setdefault(
            "execution_mode",
            "pipeline"
            if states or self.manifests.get(job_id) or published
            else "legacy"
            if any(v["event"].startswith("process.") for v in logs)
            else "unknown",
        )

        if row["execution_mode"] == "legacy":
            summaries = [v for v in logs if v["event"].startswith("process.")]
            # Only join the request whose primary latency belongs to this CSV row.
            # A time anchor is absent in text events, so multiple req_ids are ambiguous.
            requests = {v.get("req_id") for v in summaries}
            if len(requests) == 1 and row.get("legacy_request_count", 1) == 1:
                for log in summaries:
                    assign(
                        "internal_total_s",
                        seconds_ms(log.get("primary_return_latency_ms")),
                        log["_source"],
                    )
                    assign("reason", log.get("reason"), log["_source"])
                    if log["event"] == "process.summary":
                        assign("outcome", "normal", log["_source"])
                    elif log["event"] in {"process.skip", "process.fallback"}:
                        assign("outcome", "fallback", log["_source"])
                    if "threshold" in log:
                        assign(
                            "image_threshold", integer(log["threshold"]), log["_source"]
                        )
                    if "images" in log and "image_count" not in row:
                        assign("image_count", integer(log["images"]), log["_source"])
            elif summaries:
                self.issue(
                    "legacy_log",
                    "Multiple requests cannot be safely matched "
                    "to primary-return log evidence",
                    job_id,
                )

        for field in (
            *EXTERNAL_TIMES,
            "spi_alarm_count",
            "image_count",
            "image_threshold",
        ):
            values = {}
            for event, source in self.events.get(job_id, []):
                raw = event.get(field)
                if raw not in {None, ""}:
                    try:
                        value = (
                            timestamp(raw) if field in EXTERNAL_TIMES else integer(raw)
                        )
                    except (ValueError, TypeError, OverflowError) as exc:
                        self.issue(
                            source, f"Invalid {field}; ignored: {exc}", job_id, "error"
                        )
                        continue
                    values.setdefault(value, []).append(source)
            if len(values) == 1:
                value, sources = next(iter(values.items()))
                if field in row and row[field] != value:
                    self.issue(
                        " | ".join(sources),
                        f"{field} conflicts with "
                        "system evidence; system value retained",
                        job_id,
                    )
                else:
                    assign(field, value, " | ".join(sources))
            elif values:
                self.issue(
                    "events", f"Conflicting {field}; external value ignored", job_id
                )
        row["event_sources"] = " | ".join(s for _, s in self.events.get(job_id, []))

        if "image_threshold" not in row:
            assign("image_threshold", self.threshold, self.threshold_source)
        row["image_count_source"] = provenance.get("image_count", "")
        row["threshold_source"] = provenance.get("image_threshold", "")
        count, threshold = row.get("image_count"), row.get("image_threshold")
        if count is not None and threshold is not None:
            row["image_threshold_exceeded"] = int(count > threshold)
            row["threshold_evidence"] = "image_count_comparison"
        if row.get("reason") == "images_exceed_threshold":
            if row.get("image_threshold_exceeded") == 0:
                self.issue(
                    "threshold",
                    "Threshold comparison conflicts with skip "
                    "reason; reason confirms exceeded historical threshold",
                    job_id,
                )
            row["image_threshold_exceeded"] = 1
            row["threshold_evidence"] = "images_exceed_threshold_reason"
        row["image_bucket"] = bucket(count)
        if row.get("reason") == "images_exceed_threshold":
            model_values = {}
        row["publication_evidence"] = (
            "primary_timestamp"
            if "primary_returned_at" in row
            else "published_backup"
            if variant and any(p.parent.name == "ai_result" for p in variant["paths"])
            else "legacy_primary_latency"
            if "internal_total_s" in row
            else "not_recorded"
        )

        for field, start, end in (
            ("pre_ready_s", "spi_sent_at", "ready_at"),
            ("waiting_s", "ready_at", "inference_started_at"),
            ("inference_processing_s", "inference_started_at", "result_ready_at"),
            ("publish_s", "result_ready_at", "primary_returned_at"),
            ("internal_total_s", "ready_at", "primary_returned_at"),
            ("end_to_end_s", "first_seen_at", "primary_returned_at"),
            ("end_to_end_budget_s", "first_seen_at", "deadline_at"),
            ("machine_wait_s", "spi_sent_at", "spi_read_at"),
            ("read_after_write_s", "primary_returned_at", "spi_read_at"),
        ):
            if start in row and end in row:
                duration = row[end] - row[start]
                if duration < 0:
                    self.issue("timestamps", f"Negative {field}; left blank", job_id)
                    row.pop(field, None)
                else:
                    row[field] = round(duration, 6)
        if "spi_read_at" in row and "op_confirm_at" in row:
            row["late_after_op_s"] = round(row["spi_read_at"] - row["op_confirm_at"], 6)
            row["late_after_op"] = int(row["late_after_op_s"] > 0)
        if "primary_returned_at" in row and "deadline_at" in row:
            row["deadline_met"] = int(
                row["primary_returned_at"] < row["deadline_at"]
                if "first_seen_at" in row
                else row["primary_returned_at"] <= row["deadline_at"]
            )
        elif row.get("status") == "EXPIRED":
            row["deadline_met"] = 0
        row["deadline_start_source"] = (
            "folder_inspection"
            if row.get("state_reason") == "input_inspection_timeout"
            else "first_csv_observed"
            if "first_seen_at" in row
            else ""
        )
        date_field = next(
            (
                f
                for f in (
                    "ready_at",
                    "first_seen_at",
                    "request_start_at",
                    "spi_sent_at",
                    "manifest_created_at",
                )
                if f in row
            ),
            None,
        )
        row["report_date"] = (
            datetime.fromtimestamp(row[date_field], TZ8).date().isoformat()
            if date_field
            else "unknown"
        )
        row["report_date_source"] = date_field or ""
        for field in TIME_FIELDS:
            if field in row:
                row[field] = format_time(row[field])
        row["field_sources"] = json.dumps(
            provenance, ensure_ascii=False, sort_keys=True
        )
        row["missing_fields"] = " | ".join(
            field
            for field in (
                *STATE_TIMES,
                "image_count",
                "image_threshold",
                *EXTERNAL_TIMES,
            )
            if field not in row
        )
        row["data_issues"] = " | ".join(
            i["message"] for i in self.issues if i["job_id"] == job_id
        )
        models = [
            dict(
                machine_id=self.machine_id,
                job_id=job_id,
                execution_mode=row["execution_mode"],
                model=name,
                request_ms=number(ms),
                timing_source=model_source,
                manifest_selection=selection,
            )
            for name, ms in sorted(model_values.items())
        ]
        return row, models


def seconds_ms(value: Any) -> float | None:
    result = number(value)
    if result is not None and result < 0:
        raise ValueError("Negative duration")
    return result / 1000 if result is not None else None


def format_time(value: float | None) -> str:
    return (
        datetime.fromtimestamp(value, TZ8).isoformat(timespec="microseconds")
        if value is not None
        else ""
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 6)


def rate(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 6) if denominator else None


def summarize(rows: list[dict], **labels: Any) -> dict:
    known = [r for r in rows if r.get("image_threshold_exceeded") is not None]
    exceeded = sum(r["image_threshold_exceeded"] == 1 for r in known)
    result = dict(
        **labels,
        total_jobs=len(rows),
        publication_evidence_jobs=sum(
            r.get("publication_evidence") not in {None, "not_recorded"} for r in rows
        ),
        normal_result_jobs=sum(r.get("outcome") == "normal" for r in rows),
        fallback_result_jobs=sum(r.get("outcome") == "fallback" for r in rows),
        unresolved_manifest_jobs=sum(
            r.get("manifest_selection") == "ambiguous" for r in rows
        ),
        threshold_known_jobs=len(known),
        threshold_unknown_jobs=len(rows) - len(known),
        threshold_exceeded_jobs=exceeded,
        threshold_exceeded_pct=rate(exceeded, len(known)),
        threshold_coverage_pct=rate(len(known), len(rows)),
        threshold_skip_result_jobs=sum(
            r.get("reason") == "images_exceed_threshold" for r in rows
        ),
        threshold_skip_published_jobs=sum(
            r.get("reason") == "images_exceed_threshold"
            and r.get("publication_evidence") not in {None, "not_recorded"}
            for r in rows
        ),
        jobs_with_issues=sum(bool(r.get("data_issues")) for r in rows),
        expired_jobs=sum(r.get("status") == "EXPIRED" for r in rows),
        partial_publication_jobs=sum(
            (r.get("published_csv_count") or 0) > 0 and not r.get("primary_returned_at")
            for r in rows
        ),
        publication_unknown_jobs=sum(bool(r.get("publication_unknown")) for r in rows),
        image_backup_incomplete_jobs=sum(
            r.get("image_backup_status") in {"incomplete", "failed"} for r in rows
        ),
    )
    end_to_end_jobs = [r for r in rows if r.get("first_seen_at")]
    met = sum(r.get("deadline_met") == 1 for r in end_to_end_jobs)
    result["end_to_end_jobs"] = len(end_to_end_jobs)
    result["end_to_end_met_jobs"] = met
    result["end_to_end_met_pct"] = rate(met, len(end_to_end_jobs))
    image_known = [r for r in known if r.get("image_count") is not None]
    image_total = sum(r["image_count"] for r in image_known)
    result["image_weighted_known_jobs"] = len(image_known)
    result["image_weighted_exceeded_pct"] = rate(
        sum(r["image_count"] for r in image_known if r["image_threshold_exceeded"]),
        image_total,
    )
    for field in (
        "waiting_s",
        "inference_processing_s",
        "publish_s",
        "internal_total_s",
        "end_to_end_s",
        "machine_wait_s",
    ):
        values = [r[field] for r in rows if r.get(field) is not None]
        result.update(
            {
                f"{field}_n": len(values),
                f"{field}_mean": round(sum(values) / len(values), 6)
                if values
                else None,
                f"{field}_p50": percentile(values, 0.5),
                f"{field}_p95": percentile(values, 0.95),
                f"{field}_p99": percentile(values, 0.99),
                f"{field}_max": max(values) if values else None,
            }
        )
        if field in {"internal_total_s", "machine_wait_s"}:
            for limit in (3, 5):
                n = sum(v > limit for v in values)
                result[f"{field}_over_{limit}s_jobs"] = n
                result[f"{field}_over_{limit}s_pct"] = rate(n, len(values))
    op_known = [r for r in rows if r.get("late_after_op") is not None]
    result["op_comparison_known_jobs"] = len(op_known)
    result["late_after_op_jobs"] = sum(r["late_after_op"] for r in op_known)
    result["late_after_op_pct"] = rate(result["late_after_op_jobs"], len(op_known))
    return result


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def range_boundary(value: str) -> float:
    """CLI boundaries without a timezone use the report's UTC+8 timezone."""
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ8)
    return parsed.timestamp()


def selection_time(row: dict, field: str) -> tuple[str, float | None]:
    if field == "auto":
        field = row.get("report_date_source") or "unknown"
    if field == "job_timestamp":
        job_id = row["job_id"]
        if not re.fullmatch(r"\d{14}", job_id):
            return field, None
        try:
            return field, datetime.strptime(job_id, "%Y%m%d%H%M%S").replace(
                tzinfo=TZ8
            ).timestamp()
        except ValueError:
            return field, None
    return field, timestamp(row.get(field))


def export(
    evidence: Evidence,
    *,
    start: float | None = None,
    end: float | None = None,
    time_field: str = "auto",
) -> dict:
    if start is not None and end is not None and start >= end:
        raise ValueError("--start must be earlier than --end (end is exclusive)")
    rows, models, candidates = evidence.build()
    available_jobs = len(rows)
    selected, outside_range, missing_time = [], 0, 0
    for row in rows:
        field, instant = selection_time(row, time_field)
        row["selection_time_field"] = field
        row["selection_at"] = format_time(instant)
        row["report_date"] = (
            datetime.fromtimestamp(instant, TZ8).date().isoformat()
            if instant is not None
            else "unknown"
        )
        row["report_date_source"] = field if field != "unknown" else ""
        if start is not None or end is not None:
            if instant is None:
                missing_time += 1
                evidence.issue(
                    "time_filter",
                    f"Excluded: no valid {field} timestamp",
                    row["job_id"],
                )
                continue
            if (start is not None and instant < start) or (
                end is not None and instant >= end
            ):
                outside_range += 1
                continue
        selected.append(row)
    rows = selected
    job_ids = {r["job_id"] for r in rows}
    models = [r for r in models if r["job_id"] in job_ids]
    candidates = [r for r in candidates if r["job_id"] in job_ids]
    filter_labels = dict(
        filter_time_field=time_field,
        filter_start_at=format_time(start),
        filter_end_at=format_time(end),
        filter_available_jobs=available_jobs,
        filter_outside_range_jobs=outside_range,
        filter_missing_time_jobs=missing_time,
    )
    evidence.output.mkdir(parents=True, exist_ok=True)
    summaries = [summarize(rows, scope="all", report_date="all", **filter_labels)]
    days = sorted({r["report_date"] for r in rows})
    summaries += [
        summarize(
            [r for r in rows if r["report_date"] == day],
            scope="day",
            report_date=day,
            **filter_labels,
        )
        for day in days
    ]
    buckets = [
        summarize(
            [r for r in rows if r["image_bucket"] == label],
            image_bucket=label,
            **filter_labels,
        )
        for label in ("0", "1-10", "11-30", "31-60", "61-100", "101+", "unknown")
    ]
    for filename, data, columns in (
        ("jobs.csv", rows, JOB_COLUMNS),
        ("model_timings.csv", models, MODEL_COLUMNS),
        ("summary.csv", summaries, list(summaries[0])),
        ("image_buckets.csv", buckets, list(buckets[0])),
        ("issues.csv", evidence.issues, ISSUE_COLUMNS),
        (
            "manifest_candidates.csv",
            candidates,
            [
                "job_id",
                "fingerprint",
                "outcome",
                "reason",
                "manifest_created_at",
                "sources",
            ],
        ),
    ):
        write_csv(evidence.output / filename, data, columns)
    return {
        "jobs": len(rows),
        "available_jobs": available_jobs,
        "excluded_outside_range_jobs": outside_range,
        "excluded_missing_time_jobs": missing_time,
        "time_filter": filter_labels,
        "issues": len(evidence.issues),
        "errors": sum(i["severity"] == "error" for i in evidence.issues),
        "output": str(evidence.output),
    }


def project_root() -> Path:
    script = Path(__file__).resolve()
    return (
        script.parents[2]
        if script.parent.name == "tools" and script.parent.parent.name == "app"
        else script.parent
    )


def configured_path(value: str, base: Path) -> Path:
    """Use the same project-relative path convention as the pipeline workers."""
    text = str(value).strip()
    if not text:
        raise ValueError("Configured evidence path must not be blank")
    if os.name != "nt" and PureWindowsPath(text).drive:
        raise ValueError(
            f"Cannot read Windows path on this OS: {text}. "
            "Use --input for copied evidence or update the statistics config paths."
        )
    candidate = Path(text)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def configured_sources(config: dict, base: Path, evidence: Evidence) -> list[Path]:
    pipeline, logging = config.get("pipeline", {}), config.get("logging", {})
    sources = [
        configured_path(pipeline.get("result_root", "data/pipeline/results"), base)
    ]
    database = configured_path(
        pipeline.get("database_path", "data/pipeline/jobs.sqlite3"), base
    )
    sources.append(database)
    evidence.database_files.add(database)
    evidence.live_database = True
    for value in (
        config.get("paths", {}).get("backup_output_root"),
        pipeline.get("staging_root"),
    ):
        if value:
            sources.append(configured_path(value, base))
    log_dir = configured_path(logging.get("log_dir", "log"), base)
    system = Path(logging.get("system_log_file", "system"))
    bases = [log_dir / system]
    bases += [
        log_dir / f"{system.stem}.{stage}{system.suffix}"
        for stage in ("ingest", "inference", "publisher")
    ]
    request = log_dir / logging.get("request_log_file", "log.csv")
    for base_file, destination in [(p, evidence.log_files) for p in bases] + [
        (request, evidence.request_log_files)
    ]:
        matching = [base_file] if base_file.is_file() else []
        matching += [
            p
            for p in base_file.parent.glob(base_file.name + ".*")
            if p.name[len(base_file.name) + 1 :].isdigit() and p.is_file()
        ]
        for path in matching:
            if not path.is_symlink():
                resolved = path.resolve()
                destination.add(resolved)
                sources.append(resolved)
    if not evidence.log_files and not evidence.request_log_files:
        evidence.issue(log_dir, "No configured system/request logs found")
    return sorted(set(sources))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Collected evidence for ONE machine; omit to read config locations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Report folder; defaults to <project-root>/statistics_output",
    )
    parser.add_argument(
        "--machine-id", help="Machine label; defaults to input/project folder name"
    )
    parser.add_argument(
        "--image-threshold",
        type=int,
        help="Explicit fallback threshold; no implicit default of 60",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config JSON; otherwise AI_CONFIG_PATH or config/ai_server.json",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Base for config-relative paths; defaults to this tool's project root",
    )
    parser.add_argument(
        "--events",
        type=Path,
        help="Optional external event CSV; in-folder CSVs are auto-detected",
    )
    parser.add_argument(
        "--start",
        help="Inclusive start date/time; no timezone means UTC+8",
    )
    parser.add_argument(
        "--end",
        help="Exclusive end date/time; no timezone means UTC+8",
    )
    parser.add_argument(
        "--time-field",
        choices=["auto", *TIME_FIELDS, "job_timestamp"],
        default="auto",
        help="Timestamp field; auto prefers ready, request, SPI-sent, then manifest",
    )
    args = parser.parse_args(argv)
    try:
        start = range_boundary(args.start) if args.start is not None else None
        end = range_boundary(args.end) if args.end is not None else None
        if start is not None and end is not None and start >= end:
            raise ValueError("--start must be earlier than --end (end is exclusive)")
        base = (args.project_root or project_root()).resolve()
        root = args.input.resolve() if args.input else base
        output = (args.output or base / "statistics_output").resolve()
        if args.input and not root.is_dir():
            raise ValueError(f"Input directory does not exist: {root}")
        config_path = args.config
        if not args.input and config_path is None:
            config_path = Path(
                os.environ.get("AI_CONFIG_PATH") or base / "config/ai_server.json"
            )
        threshold, source, extensions = args.image_threshold, "cli", DEFAULT_EXTENSIONS
        config = {}
        if config_path:
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            processing = config.get("processing", {})
            if threshold is None:
                threshold = integer(processing.get("folder_images_num_threshold"))
                source = str(config_path.resolve())
            extensions = {
                str(e).lower()
                for e in processing.get("image_extensions", sorted(DEFAULT_EXTENSIONS))
            }
        threshold = integer(threshold)
        evidence = Evidence(
            root,
            output,
            args.machine_id or root.name,
            threshold,
            source if threshold is not None else "",
            extensions,
        )
        sources = [root] if args.input else configured_sources(config, base, evidence)
        if any(path.is_relative_to(output) for path in sources):
            raise ValueError("Output cannot equal a source or be its ancestor")
        evidence.discover(sources)
        if args.events and args.events.resolve() not in evidence.files:
            evidence.read_csv(args.events.resolve(), events_only=True)
        result = export(evidence, start=start, end=end, time_field=args.time_field)
        result.update(
            input_mode="folder" if args.input else "config",
            config=str(config_path.resolve()) if config_path else None,
            sources=[str(p) for p in sources],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["errors"] else 0 if result["available_jobs"] else 1
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        sqlite3.Error,
    ) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
