"""Durable FIFO processing and bounded delivery; SQLite is an audit projection."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from app.application.inference.obb_service import ObbInspectionService, build_fallback
from app.application.ingest import io as ingest_io
from app.application.ingest.service import (
    IngestCollisionError,
    IngestStorageError,
    IngestValidationError,
    SourceChangedError,
)
from app.application.operations import (
    backup_result,
    export_result,
    prepare_publications,
    publish_csv,
)
from app.core.config import AppConfig
from app.core.logging import get_logger
from app.domain.entities.pipeline_result import PipelineResultManifest
from app.infrastructure.async_io import call_io
from app.infrastructure.audit import sync_events
from app.infrastructure.deadline_io import DeadlineExceeded, DeadlineIO
from app.infrastructure.input.sinic_folder_input import FileSettleTracker
from app.infrastructure.job_history import (
    HistoryConflictError,
    archive_job,
    cleanup_committed,
    is_archived,
)
from app.infrastructure.local_store import ControllerLock, JobState, LocalJobStore
from app.infrastructure.output.sinic_csv_output import PublishCompletionUnknown


class StatePersistenceError(Exception):
    """A controller decision could not be recorded safely."""


class Controller:
    def __init__(self, config: AppConfig, runtime, *, io_factory=DeadlineIO):
        self.config = config.resolved()
        self.runtime = runtime
        self.io_factory = io_factory
        self.store = LocalJobStore(Path(self.config.paths.state_root))
        self.history_root = Path(self.config.paths.history_root)
        self.jobs: dict[str, JobState] = {}
        self._once_results = None
        self.artifacts = self.store.root / "artifacts"
        self.inference = ObbInspectionService(
            config=self.config,
            result_root=self.artifacts,
            runtime=runtime,
            io_factory=io_factory,
        )
        self.log = get_logger("controller")
        self._tasks = set()
        self._process_tasks = {}
        self._retiring = set()
        self._closed_jobs = set()
        self._history_conflicts = {}
        self._elapsed = {}
        self._warm_task = None
        self._warm_error = None
        self._closing = False
        self._fatal_error = None
        self._background_busy = set()
        self._discovery_lane = None
        self._scan_logged_at = 0.0
        self._scan_warning_at = 0.0

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)
        return task

    def _task_finished(self, task):
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.log.error("controller task failed", exc_info=task.exception())
            self._fatal_error = task.exception()
            for pending in list(self._tasks):
                pending.cancel()

    def _save(self, job, event):
        if job.job_id in self._elapsed:
            baseline, started = self._elapsed[job.job_id]
            job.processing_elapsed_ms = baseline + (time.monotonic() - started) * 1000
        try:
            self.store.save(job, event)
        except Exception as exc:
            raise StatePersistenceError(
                f"Cannot persist {event} for {job.job_id}"
            ) from exc
        self.log.info(
            "event=%s job_id=%s status=%s reason=%s last_error=%s",
            event,
            job.job_id,
            job.status,
            job.reason,
            job.last_error,
        )

    def _admission_blocked(self, job_id):
        return (
            job_id in self.jobs
            or job_id in self._closed_jobs
            or job_id in self._retiring
            or job_id in self._history_conflicts
        )

    def register(self, source, first_seen_at, *, observation="csv_observation"):
        if self._fatal_error is not None:
            raise self._fatal_error
        # No await between the final ownership check and its durable claim.
        if self._admission_blocked(source.name):
            return None
        backup = (
            Path(self.config.paths.backup_output_root)
            / datetime.strptime(source.name, "%Y%m%d%H%M%S").date().isoformat()
            / source.name
        )
        job = JobState(
            job_id=source.name,
            source_folder=str(source),
            original_backup_folder=str(backup),
            first_seen_at=first_seen_at,
            first_seen_source=observation,
            config_snapshot=self.config.model_dump(),
            export_status="pending" if self.config.minifi.enabled else "disabled",
        )
        self._save(job, "job.queued")
        self.jobs[job.job_id] = job
        if self._once_results is not None:
            self._once_results[job.job_id] = job
        return job

    def _mark_history_conflict(self, job_id, reason):
        if job_id not in self._history_conflicts:
            self.log.error(
                "event=job.history_conflict job_id=%s detail=%s; "
                "both journals retained; automatic processing stopped",
                job_id,
                reason,
            )
        self._history_conflicts[job_id] = reason
        job = self.jobs.get(job_id)
        if job is not None:
            job.status = "HISTORY_CONFLICT"
            task = self._process_tasks.get(job_id)
            if task is not None and task is not asyncio.current_task():
                task.cancel()

    def recover(self):
        archived = {
            folder.name
            for folder in (self.store.root / "jobs").iterdir()
            if folder.is_dir()
            and folder.name not in self._history_conflicts
            and is_archived(self.history_root, folder.name)
        }
        self.jobs = self.store.load(exclude=archived)
        if self._once_results is not None:
            self._once_results.update(self.jobs)
        for job in self.jobs.values():
            if job.job_id in self._history_conflicts:
                self._mark_history_conflict(
                    job.job_id, self._history_conflicts[job.job_id]
                )
                continue
            if job.primary_terminal:
                continue
            changed = False
            for record in job.publications.values():
                if record["status"] == "PUBLISHING":
                    record.update(
                        status="UNKNOWN",
                        error="Process stopped before delivery acknowledgement",
                    )
                    if record.get("attempt_log"):
                        record["attempt_log"][-1].update(
                            status="UNKNOWN",
                            finished_at=time.time(),
                            error=record["error"],
                        )
                    changed = True
            expected = set()
            if job.result_manifest_path:
                expected = {
                    item.source_csv
                    for item in PipelineResultManifest.read(
                        Path(job.result_manifest_path)
                    ).csv_results
                }
            if (
                expected
                and set(job.publications) == expected
                and all(r["status"] == "RETURNED" for r in job.publications.values())
            ):
                self._finish_delivery(job)
            elif job.config_snapshot != self.config.model_dump():
                self._fail(
                    job,
                    "configuration_changed",
                    status="PUBLICATION_UNKNOWN"
                    if any(r["status"] == "UNKNOWN" for r in job.publications.values())
                    else "FAILED",
                )
            elif changed:
                job.status = "RESULT_READY"
                self._save(job, "publication.recovered_unknown")

    async def _new_lane(self):
        lane = self.io_factory()
        try:
            await call_io(
                lane,
                ingest_io.ping,
                deadline_at=time.time()
                + self.config.runtime.io_startup_timeout_seconds,
            )
            return lane
        except BaseException:
            lane.close()
            raise

    async def _discover(self, target_date: date):
        prefix = target_date.strftime("%Y%m%d")
        self._closed_jobs = {j for j in self._closed_jobs if j.startswith(prefix)}
        if self._discovery_lane is None:
            self._discovery_lane = await self._new_lane()
            self.log.info(
                "event=discovery.ready watch_root=%s date=%s scan_timeout_seconds=%s",
                self.config.paths.watch_root,
                target_date,
                self.config.runtime.discovery_timeout_seconds,
            )
        lane = self._discovery_lane

        async def operation(function, *args):
            return await call_io(
                lane,
                function,
                *args,
                deadline_at=time.time() + self.config.runtime.discovery_timeout_seconds,
            )

        try:
            folders = await operation(
                ingest_io.candidates,
                self.config.processing.image_extensions,
                Path(self.config.paths.watch_root),
                target_date,
                self.history_root,
            )
            registered = 0
            for folder in folders:
                if self._fatal_error is not None:
                    raise self._fatal_error
                if self._admission_blocked(folder.name):
                    continue
                if await operation(is_archived, self.history_root, folder.name):
                    self._closed_jobs.add(folder.name)
                    continue
                if self._admission_blocked(folder.name):
                    continue
                observed = time.time()
                files, _, _ = await operation(
                    ingest_io.inspect_source,
                    self.config.processing.image_extensions,
                    folder,
                )
                if files.csv_files and self.register(folder, observed) is not None:
                    registered += 1
            if registered or time.monotonic() - self._scan_logged_at >= 30:
                self.log.info(
                    "event=discovery.scan candidates=%s queued=%s pending=%s",
                    len(folders),
                    registered,
                    sum(not j.primary_terminal for j in self.jobs.values()),
                )
                self._scan_logged_at = time.monotonic()
        except BaseException:
            lane.close()
            self._discovery_lane = None
            raise

    async def _worker(self):
        try:
            await asyncio.shield(self._warm_task)
        except Exception as exc:
            self._warm_error = exc
        while not self._closing:
            pending = sorted(
                (j for j in self.jobs.values() if not j.primary_terminal),
                key=lambda j: (j.first_seen_at, j.job_id),
            )
            if not pending:
                await asyncio.sleep(self.config.runtime.poll_interval_seconds)
                continue
            job = pending[0]
            task = self._spawn(self._process(job))
            self._process_tasks[job.job_id] = task
            try:
                await task
            except asyncio.CancelledError:
                if job.job_id not in self._history_conflicts:
                    raise
            finally:
                self._process_tasks.pop(job.job_id, None)

    def _fail(self, job, reason, *, status="INPUT_FAILED"):
        job.status, job.reason = status, reason
        job.delivery_status = "unknown" if status == "PUBLICATION_UNKNOWN" else "failed"
        job.export_status = "disabled"
        job.processing_finished_at = time.time()
        self._save(job, "job.failed")

    async def _prepare_input(self, job, lane):
        if job.image_backup_status == "complete" and job.ready_at is not None:
            return True
        csv_settle = FileSettleTracker(
            settle_seconds=self.config.processing.source_settle_seconds
        )
        image_settle = FileSettleTracker(
            settle_seconds=self.config.processing.source_settle_seconds
        )
        source = Path(job.source_folder)
        stop = time.monotonic() + self.config.processing.input_ready_timeout_seconds
        while time.monotonic() < stop:

            def deadline():
                return time.time() + min(
                    self.config.runtime.io_timeout_seconds,
                    max(0.001, stop - time.monotonic()),
                )

            try:
                files, csv_sig, image_sig = await call_io(
                    lane,
                    ingest_io.inspect_source,
                    self.config.processing.image_extensions,
                    source,
                    deadline_at=deadline(),
                )
                job.image_count = files.image_count
                csv_stable = csv_settle.observe_snapshot(
                    str(source), csv_sig, bool(files.csv_files)
                )
                images_stable = image_settle.observe_snapshot(
                    str(source), image_sig, bool(files.image_files)
                )
                if job.csv_snapshot_folder is None and csv_stable:
                    destination = (
                        self.store.folder(job.job_id) / "csv" / uuid.uuid4().hex
                    )
                    await call_io(
                        lane,
                        ingest_io.snapshot_csv,
                        self.config,
                        source,
                        destination,
                        deadline_at=deadline(),
                    )
                    job.csv_snapshot_folder = str(destination)
                    job.csv_ready_at = time.time()
                    job.status = "CSV_READY"
                    self._save(job, "input.csv_ready")
                if job.csv_snapshot_folder and images_stable:
                    validated = await call_io(
                        lane,
                        ingest_io.validate_source,
                        self.config,
                        source,
                        Path(job.csv_snapshot_folder),
                        deadline_at=deadline(),
                    )
                    job.status = "BACKING_UP"
                    self._save(job, "input.raw_backup_started")
                    await call_io(
                        lane,
                        ingest_io.archive_source,
                        self.config,
                        validated,
                        Path(self.config.paths.backup_output_root),
                        Path(job.csv_snapshot_folder),
                        deadline_at=time.time()
                        + self.config.runtime.io_timeout_seconds,
                    )
                    job.ready_at = time.time()
                    job.image_backup_status = "complete"
                    job.status = "READY"
                    self._save(job, "input.raw_backup_ready")
                    return True
            except IngestCollisionError as exc:
                job.last_error = str(exc)
                self._fail(job, "raw_backup_conflict")
                return False
            except SourceChangedError as exc:
                job.csv_snapshot_folder = None
                job.csv_ready_at = None
                job.last_error = str(exc)
                csv_settle.forget(str(source))
                image_settle.forget(str(source))
            except (
                IngestValidationError,
                IngestStorageError,
                OSError,
                ValueError,
            ) as exc:
                job.last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, DeadlineExceeded):
                    break
            await asyncio.sleep(self.config.runtime.poll_interval_seconds)
        if job.csv_snapshot_folder:
            await self._choose_fallback(job, "input_ready_timeout")
        else:
            self._fail(job, "input_ready_timeout")
        return False

    async def _process(self, job):
        self._elapsed[job.job_id] = (job.processing_elapsed_ms, time.monotonic())
        if job.processing_started_at is None:
            job.processing_started_at = time.time()
            job.queue_ms = max(
                0, (job.processing_started_at - job.first_seen_at) * 1000
            )
        self._save(job, "job.processing_started")
        lane = None
        try:
            lane = await self._new_lane()
            if not job.result_manifest_path:
                ready = await self._prepare_input(job, lane)
                if ready:
                    if self._warm_error is not None:
                        job.last_error = (
                            f"{type(self._warm_error).__name__}: {self._warm_error}"
                        )
                        await self._choose_fallback(
                            job,
                            "required_model_timeout"
                            if isinstance(self._warm_error, TimeoutError)
                            else "required_model_failure",
                        )
                    else:
                        job.status = "INFERENCING"
                        job.inference_started_at = time.time()
                        self._save(job, "inference.started")
                        manifest, path = await self.inference.prepare(
                            job_id=job.job_id,
                            staged_folder=Path(job.original_backup_folder),
                        )
                        job.inference_finished_at = time.time()
                        self._select(job, manifest, path)
            if job.result_manifest_path and not job.primary_terminal:
                await self._publish(job, lane)
        except StatePersistenceError:
            raise
        except Exception as exc:
            job.last_error = f"{type(exc).__name__}: {exc}"
            self._fail(job, "processing_failure", status="FAILED")
        finally:
            if lane is not None:
                lane.close()
            if job.job_id not in self._history_conflicts and job.primary_terminal:
                job.processing_finished_at = time.time()
                self._save(job, "job.processing_finished")
            self._elapsed.pop(job.job_id, None)

    async def _choose_fallback(self, job, reason):
        if (
            job.primary_terminal
            or job.result_manifest_path
            or not job.csv_snapshot_folder
        ):
            return
        lane = await self._new_lane()
        try:
            manifest, path = await call_io(
                lane,
                build_fallback,
                self.config,
                self.artifacts,
                job.job_id,
                Path(job.csv_snapshot_folder),
                reason,
                job.last_error or "",
                deadline_at=time.time() + self.config.runtime.io_timeout_seconds,
            )
            self._select(job, manifest, path)
        finally:
            lane.close()

    def _select(self, job, manifest: PipelineResultManifest, path):
        job.result_manifest_path = str(path)
        job.result_ready_at = time.time()
        job.reason = manifest.reason
        job.last_error = "; ".join(manifest.errors) if manifest.errors else None
        job.status = "RESULT_READY"
        self._save(job, "result.selected")

    def _finish_delivery(self, job):
        states = {r["status"] for r in job.publications.values()}
        if states == {"RETURNED"}:
            job.status = "PRIMARY_RETURNED"
            job.delivery_status = "returned"
            job.primary_returned_at = max(
                r["returned_at"] for r in job.publications.values()
            )
        else:
            job.status = (
                "PUBLICATION_UNKNOWN"
                if "UNKNOWN" in states
                else "PARTIAL_RETURN"
                if "RETURNED" in states
                else "RETURN_FAILED"
            )
            job.delivery_status = (
                "unknown"
                if "UNKNOWN" in states
                else "partial"
                if "RETURNED" in states
                else "failed"
            )
            job.export_status = "disabled"
            job.last_error = "; ".join(
                f"{name}: {record.get('error', record['status'])}"
                for name, record in job.publications.items()
                if record["status"] != "RETURNED"
            )
            job.reason = (
                "publish_completion_unknown"
                if "UNKNOWN" in states
                else "delivery_attempts_exhausted"
            )
        job.processing_finished_at = time.time()
        self._save(
            job,
            "job.primary_returned"
            if job.status == "PRIMARY_RETURNED"
            else "job.delivery_failed",
        )

    async def _publish(self, job, lane):
        if not job.publications:
            job.publications = await call_io(
                lane,
                prepare_publications,
                Path(job.csv_snapshot_folder),
                Path(job.result_manifest_path),
                deadline_at=time.time() + self.config.runtime.io_timeout_seconds,
            )
            self._save(job, "publication.prepared")
        settings = self.config.delivery
        for name, record in job.publications.items():
            if record["status"] in {"RETURNED", "UNKNOWN"}:
                continue
            while record["attempts"] < settings.max_attempts:
                last = record.get("attempt_log", [])
                if last and last[-1].get("finished_at"):
                    delay = max(
                        0,
                        last[-1]["finished_at"]
                        + settings.retry_interval_seconds
                        - time.time(),
                    )
                    if delay:
                        await asyncio.sleep(delay)
                record["attempts"] += 1
                attempt = {
                    "number": record["attempts"],
                    "started_at": time.time(),
                    "finished_at": None,
                    "elapsed_ms": None,
                    "status": "PUBLISHING",
                    "error": None,
                }
                record.setdefault("attempt_log", []).append(attempt)
                record["status"] = "PUBLISHING"
                job.status = "PUBLISHING"
                self._save(job, "publication.started")  # durable intent before SPI IO
                started = time.monotonic()
                try:
                    result = await call_io(
                        lane,
                        publish_csv,
                        self.config,
                        Path(record["result_path"]),
                        record["result_sha256"],
                        time.time() + settings.attempt_timeout_seconds,
                        deadline_at=time.time() + settings.attempt_timeout_seconds,
                        job_id=job.job_id,
                    )
                except StatePersistenceError:
                    raise
                except (
                    PublishCompletionUnknown,
                    DeadlineExceeded,
                    BrokenPipeError,
                    EOFError,
                ) as exc:
                    uncertain = not isinstance(exc, DeadlineExceeded) or exc.uncertain
                    record.update(
                        status="UNKNOWN" if uncertain else "FAILED", error=str(exc)
                    )
                    attempt.update(status=record["status"], error=str(exc))
                except (OSError, ValueError) as exc:
                    record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                    attempt.update(status="FAILED", error=record["error"])
                except Exception as exc:
                    # An unclassified worker failure cannot establish that SPI
                    # saw no side effect. Preserve it for manual reconciliation.
                    record.update(
                        status="UNKNOWN", error=f"{type(exc).__name__}: {exc}"
                    )
                    attempt.update(status="UNKNOWN", error=record["error"])
                else:
                    record.update(
                        status="RETURNED",
                        destination=str(result.destination_csv),
                        returned_at=time.time(),
                        error=None,
                    )
                    record["processing_to_return_ms"] = max(
                        0, (record["returned_at"] - job.processing_started_at) * 1000
                    )
                    attempt.update(
                        status="RETURNED", destination=str(result.destination_csv)
                    )
                attempt.update(
                    finished_at=time.time(),
                    elapsed_ms=(time.monotonic() - started) * 1000,
                )
                self._save(
                    job,
                    "publication.returned"
                    if record["status"] == "RETURNED"
                    else "publication.unknown"
                    if record["status"] == "UNKNOWN"
                    else "publication.failed",
                )
                self.log.info(
                    "event=delivery.attempt job_id=%s source_csv=%s attempt=%s "
                    "status=%s elapsed_ms=%.3f processing_to_return_ms=%s error=%s",
                    job.job_id,
                    name,
                    attempt["number"],
                    record["status"],
                    attempt["elapsed_ms"],
                    record.get("processing_to_return_ms"),
                    attempt["error"],
                )
                if record["status"] in {"RETURNED", "UNKNOWN"}:
                    break
        self._finish_delivery(job)

    async def _background_operation(self, job, kind, function, *args):
        key = (job.job_id, kind)
        if key in self._background_busy:
            return False, None
        self._background_busy.add(key)
        lane = self.io_factory()
        try:
            result = await call_io(
                lane,
                function,
                *args,
                deadline_at=time.time()
                + self.config.runtime.background_timeout_seconds,
            )
            if job.job_id in self._history_conflicts:
                return False, None
            job.background_errors.pop(kind, None)
            return True, result
        except Exception as exc:
            if job.job_id in self._history_conflicts:
                return False, None
            job.background_attempts += 1
            job.background_errors[kind] = str(exc)
            job.next_background_at = (
                time.time() + self.config.runtime.background_retry_seconds
            )
            self._save(job, f"background.{kind}_failed")
            self.log.warning("Background %s failed for %s: %s", kind, job.job_id, exc)
            return False, None
        finally:
            lane.close()
            self._background_busy.discard(key)

    async def _backups_once(self):
        for job in list(self.jobs.values()):
            if (
                not job.primary_terminal
                or job.job_id in self._history_conflicts
                or job.job_id in self._retiring
                or job.job_id in self._process_tasks
                or time.time() < job.next_background_at
            ):
                continue
            if job.backup_status != "complete":
                success, _ = await self._background_operation(
                    job,
                    "backup",
                    backup_result,
                    Path(job.csv_snapshot_folder) if job.csv_snapshot_folder else None,
                    Path(job.original_backup_folder),
                    Path(job.result_manifest_path)
                    if job.result_manifest_path
                    else None,
                    asdict(job),
                )
                if success:
                    job.backup_status = "complete"
                    if job.delivery_status == "returned":
                        job.status = "DONE"
                    job.completed_at = time.time()
                    self._save(job, "backup.completed")

    async def _raw_backups_once(self):
        for job in list(self.jobs.values()):
            if (
                not job.primary_terminal
                or job.job_id in self._history_conflicts
                or job.job_id in self._retiring
                or job.job_id in self._process_tasks
                or job.image_backup_status != "pending"
                or time.time() < job.next_background_at
            ):
                continue
            success, result = await self._background_operation(
                job,
                "raw_backup",
                ingest_io.backup_images,
                Path(job.source_folder),
                Path(job.original_backup_folder) / "raw",
                Path(job.csv_snapshot_folder) if job.csv_snapshot_folder else None,
            )
            if success:
                job.raw_backup_attempts += 1
                job.image_backup_status = (
                    result
                    if result == "complete" or job.raw_backup_attempts >= 5
                    else "pending"
                )
                if result != "complete":
                    job.next_background_at = (
                        time.time() + self.config.runtime.background_retry_seconds
                    )
                self._save(job, "backup.raw_completed")

    async def _exports_once(self):
        for job in list(self.jobs.values()):
            if (
                job.backup_status != "complete"
                or job.delivery_status != "returned"
                or job.job_id in self._history_conflicts
                or job.job_id in self._retiring
                or job.job_id in self._process_tasks
                or job.export_status in {"complete", "disabled"}
                or time.time() < job.next_background_at
            ):
                continue
            success, _ = await self._background_operation(
                job,
                "export",
                export_result,
                job.config_snapshot,
                Path(job.original_backup_folder),
            )
            if success:
                job.export_status = "complete"
                self._save(job, "export.completed")

    async def _audit_once(self):
        if not self.config.database.enabled:
            return
        lane = self.io_factory()
        try:
            await call_io(
                lane,
                sync_events,
                self.store.root,
                self.history_root,
                Path(self.config.database.path),
                self.config.database.busy_timeout_seconds,
                deadline_at=time.time()
                + self.config.runtime.background_timeout_seconds,
            )
        except Exception as exc:
            self.log.warning(
                "Audit database unavailable; local events retained: %s", exc
            )
        finally:
            lane.close()

    async def _cleanup_once(self):
        # Complete an interrupted deletion even if no active journal remains.
        lane = self.io_factory()
        try:
            conflicts = await call_io(
                lane,
                cleanup_committed,
                self.store.root,
                self.history_root,
                deadline_at=time.time()
                + self.config.runtime.background_timeout_seconds,
            )
            for job_id, reason in conflicts.items():
                self._mark_history_conflict(job_id, reason)
        except Exception as exc:
            self.log.warning("Committed workspace cleanup deferred: %s", exc)
        finally:
            lane.close()
        for job in list(self.jobs.values()):
            tasks = (self._process_tasks.get(job.job_id),)
            if (
                not job.cleanup_ready
                or job.job_id in self._retiring
                or job.job_id in self._process_tasks
                or time.time() < job.next_background_at
                or any(task is not None and not task.done() for task in tasks)
                or any(key[0] == job.job_id for key in self._background_busy)
            ):
                continue
            self._retiring.add(job.job_id)
            lane = self.io_factory()
            try:
                archived = await call_io(
                    lane,
                    archive_job,
                    self.store.root,
                    self.history_root,
                    job,
                    deadline_at=time.time()
                    + self.config.runtime.background_timeout_seconds,
                )
                if self._once_results is not None:
                    self._once_results[job.job_id] = archived
                self._closed_jobs.add(job.job_id)
                self.jobs.pop(job.job_id)
                self._process_tasks.pop(job.job_id, None)
                self.log.info("event=job.workspace_removed job_id=%s", job.job_id)
            except HistoryConflictError as exc:
                self._mark_history_conflict(job.job_id, str(exc))
            except Exception as exc:
                # The archive might already be committed. Do not append a new
                # active event after that receipt, or retry could conflict.
                job.next_background_at = (
                    time.time() + self.config.runtime.background_retry_seconds
                )
                self.log.warning(
                    "Workspace cleanup deferred for %s: %s", job.job_id, exc
                )
            finally:
                lane.close()
                self._retiring.discard(job.job_id)

    async def _repeat(self, operation, interval):
        while not self._closing:
            await operation()
            await asyncio.sleep(interval)

    async def _warm(self):
        try:
            await asyncio.wait_for(
                self.runtime.start(), self.config.inference.obb.startup_timeout_seconds
            )
        except Exception as exc:
            self.runtime.close()
            self.log.error(
                "Model unavailable; complete CSV jobs will use all-23 protection: %s",
                exc,
            )
            raise

    async def run(self, *, once=False, target_date: date | None = None):
        self._once_results = {} if once else None
        with ControllerLock(self.store.root):
            self._warm_task = asyncio.create_task(self._warm())
            try:
                await self._cleanup_once()
                self.recover()
                self._spawn(self._worker())
                if not once:
                    for operation, interval in (
                        (
                            self._backups_once,
                            self.config.runtime.background_retry_seconds,
                        ),
                        (
                            self._raw_backups_once,
                            self.config.runtime.background_retry_seconds,
                        ),
                        (self._exports_once, self.config.minifi.poll_seconds),
                        (self._audit_once, self.config.database.poll_seconds),
                        (
                            self._cleanup_once,
                            self.config.runtime.background_retry_seconds,
                        ),
                    ):
                        self._spawn(self._repeat(operation, interval))
                while True:
                    try:
                        await self._discover(target_date or datetime.now().date())
                    except (OSError, DeadlineExceeded) as exc:
                        if time.monotonic() - self._scan_warning_at >= 5:
                            self.log.warning(
                                "Discovery deferred watch_root=%s: %s",
                                self.config.paths.watch_root,
                                exc,
                            )
                            self._scan_warning_at = time.monotonic()
                        if once:
                            raise
                    if once:
                        while (
                            any(not job.primary_terminal for job in self.jobs.values())
                            or self._process_tasks
                        ):
                            if self._fatal_error:
                                raise self._fatal_error
                            await asyncio.sleep(
                                self.config.runtime.poll_interval_seconds
                            )
                        await self._backups_once()
                        await self._raw_backups_once()
                        await self._exports_once()
                        await self._cleanup_once()
                        await self._audit_once()
                        if self._fatal_error:
                            raise self._fatal_error
                        return self._once_results
                    if self._fatal_error:
                        raise self._fatal_error
                    await asyncio.sleep(self.config.runtime.scan_interval_seconds)
            finally:
                await self.close()

    async def close(self):
        self._closing = True
        if self._discovery_lane is not None:
            self._discovery_lane.close()
        for task in list(self._tasks):
            task.cancel()
        if self._warm_task is not None:
            self._warm_task.cancel()
        await asyncio.gather(
            *list(self._tasks),
            *([self._warm_task] if self._warm_task else []),
            return_exceptions=True,
        )
        self.runtime.close()
