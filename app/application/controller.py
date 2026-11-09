"""One owner for intake, deadlines and SPI publication; no database dependency."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import date, datetime
from pathlib import Path

from app.application.inference.obb_service import ObbInspectionService
from app.application.ingest import io as ingest_io
from app.application.ingest.service import (
    IngestCollisionError,
    IngestStorageError,
    IngestValidationError,
    SourceChangedError,
)
from app.application.operations import backup_result, export_result, publish_csv
from app.application.shared.fallback_result import FallbackResultBuilder
from app.core.config import AppConfig
from app.core.logging import get_logger
from app.domain.entities.pipeline_result import PipelineResultManifest
from app.infrastructure.async_io import DeadlineGate, call_io
from app.infrastructure.audit import sync_events
from app.infrastructure.deadline_io import DeadlineExceeded, DeadlineIO
from app.infrastructure.input.sinic_folder_input import FileSettleTracker
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
        self.jobs: dict[str, JobState] = {}
        self.artifacts = self.store.root / "artifacts"
        self.fallback = FallbackResultBuilder(
            config=self.config, result_root=self.artifacts
        )
        self.inference = ObbInspectionService(
            config=self.config, result_root=self.artifacts, runtime=runtime
        )
        self.log = get_logger("controller")
        self._tasks: set[asyncio.Task] = set()
        self._process_tasks: dict[str, asyncio.Task] = {}
        self._publish_tasks: dict[str, asyncio.Task] = {}
        self._stops: dict[str, tuple[float, float]] = {}
        self._decision_locks: dict[str, asyncio.Lock] = {}
        self._input_slots = asyncio.Semaphore(self.config.runtime.max_active_jobs)
        self._gpu_slot = asyncio.Lock()
        self._publish_gate = DeadlineGate()
        self._warm_task = None
        self._closing = False
        self._fatal_error = None
        self._background_busy: set[tuple[str, str]] = set()

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)
        return task

    def _task_finished(self, task):
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            # A failed durable write is fatal: never continue with an unrecorded
            # SPI side effect or silently lose the ability to recover.
            self.log.error("controller task failed", exc_info=task.exception())
            self._fatal_error = task.exception()
            for pending in list(self._tasks):
                pending.cancel()

    def _save(self, job: JobState, event: str):
        try:
            self.store.save(job, event)
        except Exception as exc:
            raise StatePersistenceError(
                f"Cannot persist {event} for {job.job_id}"
            ) from exc
        self.log.info(
            "event=%s job_id=%s status=%s reason=%s",
            event,
            job.job_id,
            job.status,
            job.reason,
        )

    def register(
        self, source: Path, first_seen_at: float, *, observation="csv_observation"
    ):
        if self._fatal_error is not None:
            raise self._fatal_error
        if source.name in self.jobs:
            return self.jobs[source.name]
        total = self.config.reliability.end_to_end_timeout_seconds
        deadline = first_seen_at + total
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
            deadline_at=deadline,
            inference_cutoff_at=deadline
            - self.config.reliability.primary_publish_reserve_seconds,
            config_snapshot=self.config.model_dump(),
            export_status="pending" if self.config.minifi.enabled else "disabled",
        )
        self._save(job, "job.discovered")
        self.jobs[job.job_id] = job
        self._schedule(job)
        return job

    def _schedule(self, job):
        if job.primary_terminal:
            return
        now, monotonic = time.time(), time.monotonic()
        self._stops[job.job_id] = (
            monotonic + max(0, job.deadline_at - now),
            monotonic + max(0, job.inference_cutoff_at - now),
        )
        self._decision_locks[job.job_id] = asyncio.Lock()
        self._spawn(self._monitor(job))
        if job.result_manifest_path:
            self._publish_tasks[job.job_id] = self._spawn(self._publish(job))
        else:
            self._process_tasks[job.job_id] = self._spawn(self._process(job))

    def _remaining(self, job, *, inference=False):
        absolute = job.inference_cutoff_at if inference else job.deadline_at
        stop = self._stops[job.job_id][1 if inference else 0]
        return min(absolute - time.time(), stop - time.monotonic())

    def _io_deadline(self, job, *, inference=False):
        return time.time() + max(0, self._remaining(job, inference=inference))

    def recover(self):
        self.jobs = self.store.load()
        for job in self.jobs.values():
            if job.primary_terminal:
                continue
            if job.result_manifest_path:
                manifest = PipelineResultManifest.read(Path(job.result_manifest_path))
                if all(
                    job.publications.get(item.source_csv, {}).get("status")
                    == "RETURNED"
                    for item in manifest.csv_results
                ):
                    job.status = "PRIMARY_RETURNED"
                    job.primary_returned_at = max(
                        item["returned_at"] for item in job.publications.values()
                    )
                    self._save(job, "job.primary_returned_recovered")
                    continue
            if any(
                item["status"] == "PUBLISHING" for item in job.publications.values()
            ):
                # A rename may have succeeded before its acknowledgement was
                # recorded. Never infer failure from a missing machine CSV.
                self._expire(job, "publish_completion_unknown", uncertain=True)
            elif time.time() >= job.deadline_at:
                self._expire(job, "expired_during_downtime")
            elif job.config_snapshot != self.config.model_dump():
                # Do not silently rerun an unfinished job with different weights,
                # output destinations or judgement settings after a restart.
                self._expire(job, "configuration_changed")
            else:
                self._schedule(job)

    def _expire(self, job, reason, *, uncertain=False, failed=False):
        if job.primary_terminal:
            return
        job.status = (
            "PUBLICATION_UNKNOWN" if uncertain else "FAILED" if failed else "EXPIRED"
        )
        job.reason = reason
        for item in job.publications.values():
            if item["status"] == "PUBLISHING":
                item["status"] = "UNKNOWN" if uncertain else "EXPIRED"
        event = (
            "job.publication_unknown"
            if uncertain
            else "job.failed"
            if failed
            else "job.expired"
        )
        self._save(job, event)
        task = self._process_tasks.get(job.job_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        publishing = self._publish_tasks.get(job.job_id)
        if publishing is not None and publishing is not asyncio.current_task():
            publishing.cancel()

    async def _discover(self, target_date: date):
        lane = self.io_factory()
        try:
            folders = await call_io(
                lane,
                ingest_io.candidates,
                self.config.processing.image_extensions,
                Path(self.config.paths.watch_root),
                target_date,
                deadline_at=time.time() + self.config.runtime.discovery_timeout_seconds,
            )
            for folder in folders:
                if self._fatal_error is not None:
                    raise self._fatal_error
                if folder.name in self.jobs:
                    continue
                observed = time.time()
                try:
                    files, _, _ = await call_io(
                        lane,
                        ingest_io.inspect_source,
                        self.config.processing.image_extensions,
                        folder,
                        self.config.processing.folder_images_num_threshold,
                        deadline_at=observed
                        + min(
                            self.config.runtime.discovery_timeout_seconds,
                            self.config.reliability.end_to_end_timeout_seconds,
                        ),
                    )
                    if files.csv_files:
                        self.register(folder, observed)
                except DeadlineExceeded:
                    job = self.register(
                        folder, observed, observation="folder_inspection"
                    )
                    self._expire(job, "folder_inspection_timeout")
                except OSError as exc:
                    self.log.warning("Cannot inspect %s: %s", folder, exc)
        finally:
            lane.close()

    async def _process(self, job):
        lane = self.io_factory()
        csv_settle = FileSettleTracker(
            settle_seconds=self.config.processing.source_settle_seconds
        )
        image_settle = FileSettleTracker(
            settle_seconds=self.config.processing.source_settle_seconds
        )
        source = Path(job.source_folder)
        archived = job.image_backup_status == "complete" and job.ready_at is not None
        try:
            async with self._input_slots:
                while (
                    not archived
                    and not job.primary_terminal
                    and job.result_manifest_path is None
                ):
                    try:
                        files, csv_signature, image_signature = await call_io(
                            lane,
                            ingest_io.inspect_source,
                            self.config.processing.image_extensions,
                            source,
                            self.config.processing.folder_images_num_threshold,
                            deadline_at=self._io_deadline(job),
                        )
                        job.image_count = files.image_count
                        csv_stable = csv_settle.observe_snapshot(
                            str(source), csv_signature, bool(files.csv_files)
                        )
                        images_stable = image_settle.observe_snapshot(
                            str(source), image_signature, bool(files.image_files)
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
                                deadline_at=self._io_deadline(job),
                            )
                            job.csv_snapshot_folder = str(destination)
                            job.csv_ready_at = time.time()
                            job.status = "CSV_READY"
                            self._save(job, "input.csv_ready")
                        if job.csv_snapshot_folder:
                            if (
                                files.image_count
                                > self.config.processing.folder_images_num_threshold
                            ):
                                await self._choose_fallback(
                                    job, "images_exceed_threshold"
                                )
                                return
                            if self._remaining(job, inference=True) <= 0:
                                await self._choose_fallback(
                                    job, "publish_reserve_reached"
                                )
                                return
                            if images_stable:
                                validated = await call_io(
                                    lane,
                                    ingest_io.validate_source,
                                    self.config,
                                    source,
                                    Path(job.csv_snapshot_folder),
                                    deadline_at=self._io_deadline(job),
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
                                    deadline_at=self._io_deadline(job),
                                )
                                if job.result_manifest_path or job.primary_terminal:
                                    return
                                job.ready_at = time.time()
                                job.image_backup_status = "complete"
                                job.status = "READY"
                                self._save(job, "input.raw_backup_ready")
                                break
                    except IngestCollisionError as exc:
                        job.last_error = str(exc)
                        self._expire(job, "raw_backup_conflict", failed=True)
                        return
                    except SourceChangedError as exc:
                        # Preserve the original deadline while taking a new CSV
                        # generation; old snapshots remain immutable for audit.
                        job.csv_snapshot_folder = None
                        job.csv_ready_at = None
                        job.reason = str(exc)
                        csv_settle.forget(source)
                        image_settle.forget(source)
                        self._save(job, "input.changed")
                    except (
                        IngestStorageError,
                        IngestValidationError,
                        ValueError,
                        OSError,
                    ) as exc:
                        message = f"input_not_ready: {exc}"
                        if job.last_error != message:
                            job.last_error = message
                            self._save(job, "input.not_ready")
                    await asyncio.sleep(self.config.runtime.poll_interval_seconds)
                if job.result_manifest_path or job.primary_terminal:
                    return

            remaining = self._remaining(job, inference=True)
            if remaining <= 0:
                await self._choose_fallback(job, "publish_reserve_reached")
                return
            try:
                await asyncio.wait_for(self._gpu_slot.acquire(), remaining)
            except TimeoutError:
                await self._choose_fallback(job, "inference_queue_timeout")
                return
            try:
                if self._warm_task is not None and not self._warm_task.done():
                    await asyncio.wait_for(
                        asyncio.shield(self._warm_task),
                        max(0, self._remaining(job, inference=True)),
                    )
                elif (
                    self._warm_task is not None
                    and self._warm_task.exception() is not None
                ):
                    await self.runtime.start(
                        deadline_at=self._io_deadline(job, inference=True)
                    )
                if self._remaining(job, inference=True) <= 0:
                    await self._choose_fallback(job, "required_model_timeout")
                    return
                job.status = "INFERENCING"
                job.inference_started_at = time.time()
                self._save(job, "inference.started")
                manifest, path = await self.inference.prepare(
                    job_id=job.job_id,
                    staged_folder=Path(job.original_backup_folder),
                    deadline_at=self._io_deadline(job),
                )
                async with self._decision_locks[job.job_id]:
                    if (
                        not job.primary_terminal
                        and job.result_manifest_path is None
                        and self._remaining(job, inference=True) > 0
                    ):
                        self._select(job, manifest, path)
            except (TimeoutError, RuntimeError) as exc:
                job.last_error = str(exc)
                await self._choose_fallback(job, "required_model_failure")
            finally:
                self._gpu_slot.release()
        except DeadlineExceeded:
            if not job.primary_terminal:
                self._expire(job, "input_timeout")
        finally:
            lane.close()

    async def _choose_fallback(self, job, reason):
        async with self._decision_locks[job.job_id]:
            if (
                job.primary_terminal
                or job.result_manifest_path is not None
                or job.csv_snapshot_folder is None
            ):
                return
            if self._remaining(job) <= 0:
                self._expire(job, "deadline_exhausted")
                return
            manifest, path = await asyncio.to_thread(
                self.fallback.build_csv_fallback,
                job_id=job.job_id,
                csv_folder=Path(job.csv_snapshot_folder),
                reason=reason,
                errors=(job.last_error,) if job.last_error else (),
            )
            if not job.primary_terminal and self._remaining(job) > 0:
                self._select(job, manifest, path)

    def _select(self, job, manifest: PipelineResultManifest, path: Path):
        job.result_manifest_path = str(path)
        job.result_ready_at = time.time()
        job.reason = manifest.reason
        job.status = "RESULT_READY"
        self._save(job, "result.selected")
        self._publish_tasks[job.job_id] = self._spawn(self._publish(job))
        task = self._process_tasks.get(job.job_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _monitor(self, job):
        while not job.primary_terminal:
            if self._remaining(job) <= 0:
                in_progress = any(
                    item["status"] == "PUBLISHING" for item in job.publications.values()
                )
                self._expire(
                    job,
                    "publish_completion_unknown"
                    if in_progress
                    else "deadline_exhausted",
                    uncertain=in_progress,
                )
                return
            if (
                self._remaining(job, inference=True) <= 0
                and job.csv_snapshot_folder
                and not job.result_manifest_path
            ):
                try:
                    await asyncio.wait_for(
                        self._choose_fallback(job, "publish_reserve_reached"),
                        timeout=max(0, self._remaining(job)),
                    )
                except TimeoutError:
                    self._expire(job, "fallback_preparation_timeout")
                    return
            await asyncio.sleep(self.config.runtime.poll_interval_seconds)

    async def _publish(self, job):
        lane = self.io_factory()
        try:
            async with self._publish_gate.claim(job.job_id, self._stops[job.job_id][0]):
                if job.primary_terminal:
                    return
                manifest = PipelineResultManifest.read(Path(job.result_manifest_path))
                for item in manifest.csv_results:
                    previous = job.publications.get(item.source_csv, {})
                    if previous.get("status") == "RETURNED":
                        continue
                    while not job.primary_terminal:
                        if self._remaining(job) <= 0:
                            self._expire(job, "publish_timeout")
                            return
                        record = {
                            "status": "PUBLISHING",
                            "attempts": previous.get("attempts", 0) + 1,
                        }
                        job.publications[item.source_csv] = record
                        job.status = "PUBLISHING"
                        # Persist intent BEFORE submitting a network side effect.
                        self._save(job, "publication.started")
                        try:
                            result = await call_io(
                                lane,
                                publish_csv,
                                self.config,
                                Path(job.csv_snapshot_folder) / item.source_csv,
                                item.result_codes,
                                self._io_deadline(job),
                                deadline_at=self._io_deadline(job),
                                job_id=job.job_id,
                            )
                            if job.primary_terminal:
                                return
                            record.update(
                                status="RETURNED",
                                destination=str(result.destination_csv),
                                returned_at=time.time(),
                            )
                            self._save(job, "publication.returned")
                            break
                        except (
                            PublishCompletionUnknown,
                            DeadlineExceeded,
                            BrokenPipeError,
                            EOFError,
                        ) as exc:
                            uncertain = (
                                not isinstance(exc, DeadlineExceeded) or exc.uncertain
                            )
                            self._expire(
                                job,
                                "publish_completion_unknown"
                                if uncertain
                                else "publish_timeout",
                                uncertain=uncertain,
                            )
                            return
                        except OSError as exc:
                            record.update(status="FAILED", error=str(exc))
                            self._save(job, "publication.retry")
                            previous = record
                            await asyncio.sleep(min(0.05, max(0, self._remaining(job))))
                if not job.primary_terminal:
                    job.status = "PRIMARY_RETURNED"
                    job.primary_returned_at = max(
                        item["returned_at"] for item in job.publications.values()
                    )
                    self._save(job, "job.primary_returned")
        finally:
            lane.close()

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
            job.background_errors.pop(kind, None)
            return True, result
        except Exception as exc:
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
                job.status not in {"PRIMARY_RETURNED", "DONE"}
                or time.time() < job.next_background_at
            ):
                continue
            if job.backup_status != "complete":
                success, _ = await self._background_operation(
                    job,
                    "backup",
                    backup_result,
                    Path(job.csv_snapshot_folder),
                    Path(job.original_backup_folder),
                    Path(job.result_manifest_path),
                )
                if success:
                    job.backup_status = "complete"
                    job.status = "DONE"
                    job.completed_at = time.time()
                    self._save(job, "backup.completed")

    async def _raw_backups_once(self):
        for job in list(self.jobs.values()):
            if (
                job.status not in {"PRIMARY_RETURNED", "DONE"}
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
                Path(job.csv_snapshot_folder),
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
        if not self.config.minifi.enabled:
            return
        for job in list(self.jobs.values()):
            if (
                job.backup_status != "complete"
                or job.export_status in {"complete", "disabled"}
                or time.time() < job.next_background_at
            ):
                continue
            success, _ = await self._background_operation(
                job,
                "export",
                export_result,
                AppConfig.model_validate(job.config_snapshot),
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

    async def _repeat(self, operation, interval):
        while not self._closing:
            await operation()
            await asyncio.sleep(interval)

    def _warm_finished(self, task):
        if not task.cancelled() and task.exception() is not None:
            self.log.error(
                "Model unavailable; complete CSV jobs will use all-23 protection: %s",
                task.exception(),
            )

    async def run(self, *, once=False, target_date: date | None = None):
        self._fatal_error = None
        with ControllerLock(self.store.root):
            self._warm_task = asyncio.create_task(self.runtime.start())
            # Warm-up failure is observed by inference; it must not stop fallback.
            self._warm_task.add_done_callback(self._warm_finished)
            try:
                self.recover()
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
                    ):
                        self._spawn(self._repeat(operation, interval))
                while True:
                    try:
                        await self._discover(target_date or datetime.now().date())
                    except (OSError, DeadlineExceeded) as exc:
                        self.log.warning("Discovery deferred: %s", exc)
                    if once:
                        while any(
                            not job.primary_terminal for job in self.jobs.values()
                        ):
                            if self._fatal_error:
                                raise self._fatal_error
                            await asyncio.sleep(
                                self.config.runtime.poll_interval_seconds
                            )
                        await self._backups_once()
                        await self._exports_once()
                        await self._raw_backups_once()
                        await self._audit_once()
                        if self._fatal_error:
                            raise self._fatal_error
                        return self.jobs
                    if self._fatal_error:
                        raise self._fatal_error
                    await asyncio.sleep(self.config.runtime.scan_interval_seconds)
            finally:
                await self.close()

    async def close(self):
        self._closing = True
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
