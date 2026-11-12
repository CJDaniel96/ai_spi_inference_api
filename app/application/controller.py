"""Scan unfinished CSVs, back up, infer, return, mark DONE and export MiNiFi."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
import time
import uuid
from contextlib import ExitStack
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
    commit_result,
    export_result,
    mark_done,
    publish_csv,
)
from app.core.config import AppConfig
from app.core.logging import get_logger
from app.infrastructure.async_io import call_io
from app.infrastructure.deadline_io import DeadlineExceeded, DeadlineIO
from app.infrastructure.input.sinic_folder_input import FileSettleTracker
from app.infrastructure.local_store import ControllerLock
from app.infrastructure.output.sinic_csv_output import PublishCompletionUnknown
from app.infrastructure.run_record import InspectionRun, write_report


class ReportPersistenceError(Exception):
    """Stop rather than silently losing the execution's JSON report."""


class Controller:
    def __init__(self, config: AppConfig, runtime, *, io_factory=DeadlineIO):
        self.config = config.resolved()
        self.runtime = runtime
        self.io_factory = io_factory
        self.artifacts = Path(tempfile.mkdtemp(prefix="spi-ai-results-"))
        self.inference = ObbInspectionService(
            config=self.config,
            result_root=self.artifacts,
            runtime=runtime,
            io_factory=io_factory,
        )
        self.log = get_logger("controller")
        self.jobs: dict[str, InspectionRun] = {}
        self._busy_folders = set()
        self._queue = asyncio.Queue()
        self._deferred = {}
        self._background_runs = {}
        self._once_results = None
        self._discovery_lane = None
        self._warm_task = None
        self._worker_task = None
        self._maintenance_task = None
        self._warm_error = None
        self._scan_logged_at = 0
        self._scan_warning_at = 0

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

    async def _io(self, lane, function, *args, timeout=None, **kwargs):
        return await call_io(
            lane,
            function,
            *args,
            deadline_at=time.time()
            + (timeout or self.config.runtime.io_timeout_seconds),
            **kwargs,
        )

    async def _save(self, run, lane, event):
        try:
            await self._io(lane, write_report, run)
        except Exception as exc:
            raise ReportPersistenceError(
                f"Cannot write run report: {run.backup_folder}"
            ) from exc
        method = (
            self.log.warning
            if event.endswith("failed") or run.last_error
            else self.log.info
        )
        method(
            "event=%s job_id=%s run_id=%s status=%s reason=%s last_error=%s",
            event,
            run.job_id,
            run.run_id,
            run.status,
            run.reason,
            run.last_error,
        )

    async def _discover(self, target_date: date | None = None):
        if self._discovery_lane is None:
            self._discovery_lane = await self._new_lane()
            self.log.info(
                "event=discovery.ready watch_root=%s scan_timeout_seconds=%s",
                self.config.paths.watch_root,
                self.config.runtime.discovery_timeout_seconds,
            )
        lane = self._discovery_lane
        timeout = self.config.runtime.discovery_timeout_seconds
        registered = 0
        try:
            folders = await self._io(
                lane,
                ingest_io.candidates,
                self.config.processing.image_extensions,
                Path(self.config.paths.watch_root),
                target_date,
                timeout=timeout,
            )
            for key in list(self._deferred):
                if key[0] not in folders:
                    del self._deferred[key]
            for folder in folders:
                if folder in self._busy_folders or any(
                    run.source_folder == str(folder)
                    for run in self._background_runs.values()
                ):
                    continue
                _, signatures, _ = await self._io(
                    lane,
                    ingest_io.inspect_source,
                    self.config.processing.image_extensions,
                    folder,
                    timeout=timeout,
                )
                current = {signature[0]: signature for signature in signatures}
                for key in list(self._deferred):
                    if key[0] == folder and key[1] not in current:
                        del self._deferred[key]
                names = []
                for name, signature in current.items():
                    previous = self._deferred.get((folder, name))
                    if (
                        previous
                        and previous[0] == signature
                        and (previous[1] is None or time.time() < previous[1])
                    ):
                        continue
                    names.append(name)
                if not names:
                    continue
                run_id = uuid.uuid4().hex
                backup = (
                    Path(self.config.paths.backup_output_root)
                    / datetime.strptime(folder.name, "%Y%m%d%H%M%S").date().isoformat()
                    / folder.name
                )
                run = InspectionRun(
                    folder.name,
                    run_id,
                    str(folder),
                    str(backup),
                    tuple(names),
                    time.time(),
                    self.config.model_dump(),
                    workspace_folder=str(self.artifacts / run_id),
                    source_signatures={name: current[name] for name in names},
                    export_status="pending"
                    if self.config.minifi.enabled
                    else "disabled",
                )
                self._busy_folders.add(folder)
                self.jobs[run_id] = run
                try:
                    await self._save(run, lane, "job.queued")
                except BaseException:
                    self.jobs.pop(run_id)
                    self._busy_folders.discard(folder)
                    raise
                self._queue.put_nowait((run, current))
                registered += 1
                if self._once_results is not None:
                    self._once_results[run_id] = run
            if registered or time.monotonic() - self._scan_logged_at >= 30:
                self.log.info(
                    "event=discovery.scan candidates=%s queued=%s pending=%s",
                    len(folders),
                    registered,
                    len(self.jobs),
                )
                self._scan_logged_at = time.monotonic()
        except BaseException:
            lane.close()
            self._discovery_lane = None
            raise

    async def _prepare_input(self, run, lane):
        csv_settle = FileSettleTracker(
            settle_seconds=self.config.processing.source_settle_seconds
        )
        image_settle = FileSettleTracker(
            settle_seconds=self.config.processing.source_settle_seconds
        )
        source = Path(run.source_folder)
        stop = time.monotonic() + self.config.processing.input_ready_timeout_seconds
        while time.monotonic() < stop:
            try:
                files, csv_sig, image_sig = await self._io(
                    lane,
                    ingest_io.inspect_source,
                    self.config.processing.image_extensions,
                    source,
                    run.csv_names,
                    timeout=max(
                        0.001,
                        min(
                            self.config.runtime.io_timeout_seconds,
                            stop - time.monotonic(),
                        ),
                    ),
                )
                run.image_count = files.image_count
                csv_stable = csv_settle.observe_snapshot(
                    str(source), csv_sig, len(files.csv_files) == len(run.csv_names)
                )
                images_stable = image_settle.observe_snapshot(
                    str(source), image_sig, bool(files.image_files)
                )
                if run.csv_ready_at is None and csv_stable:
                    await self._io(
                        lane,
                        ingest_io.snapshot_csv,
                        self.config,
                        source,
                        Path(run.csv_snapshot_folder),
                        run.csv_names,
                        expected_signature=csv_sig,
                    )
                    run.csv_ready_at = time.time()
                    run.source_signatures = {
                        signature[0]: signature for signature in csv_sig
                    }
                    await self._io(
                        lane,
                        ingest_io.preserve_flat,
                        Path(run.csv_snapshot_folder),
                        Path(run.backup_folder),
                        self.config.processing.image_extensions,
                    )
                    run.status = "CSV_READY"
                    await self._save(run, lane, "input.csv_ready")
                if (
                    run.csv_ready_at
                    and run.image_count
                    > self.config.processing.folder_images_num_threshold
                ):
                    await self._fallback(run, lane, "images_exceed_threshold")
                    return False
                if run.csv_ready_at and images_stable:
                    validated = await self._io(
                        lane,
                        ingest_io.validate_source,
                        self.config,
                        source,
                        Path(run.csv_snapshot_folder),
                    )
                    run.status = "BACKING_UP"
                    run.backup_started_at = time.time()
                    await self._save(run, lane, "input.raw_backup_started")
                    await self._io(
                        lane,
                        ingest_io.archive_source,
                        self.config,
                        validated,
                        Path(run.inference_folder),
                        Path(run.csv_snapshot_folder),
                        backup_folder=Path(run.backup_folder),
                    )
                    run.ready_at = time.time()
                    run.backup_finished_at = run.ready_at
                    run.status = "READY"
                    run.last_error = None
                    run.image_backup_status = "complete"
                    await self._save(run, lane, "input.raw_backup_ready")
                    return True
            except IngestCollisionError as exc:
                run.status, run.reason = "INPUT_FAILED", "raw_backup_conflict"
                run.last_error = str(exc)
                return False
            except SourceChangedError as exc:
                run.last_error = str(exc)
                run.csv_ready_at = None
                # Discard only this attempt's unused CSV snapshot.
                await self._io(
                    lane,
                    shutil.rmtree,
                    Path(run.csv_snapshot_folder),
                    ignore_errors=True,
                )
                await self._io(
                    lane, shutil.rmtree, Path(run.inference_folder), ignore_errors=True
                )
                csv_settle.forget(str(source))
                image_settle.forget(str(source))
            except (
                IngestValidationError,
                IngestStorageError,
                OSError,
                ValueError,
            ) as exc:
                run.last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, DeadlineExceeded):
                    break
            await asyncio.sleep(self.config.runtime.poll_interval_seconds)
        if run.csv_ready_at:
            await self._fallback(run, lane, "input_ready_timeout")
        else:
            run.status, run.reason = "INPUT_FAILED", "input_ready_timeout"
            run.image_backup_status = await self._io(
                lane,
                ingest_io.backup_images,
                source,
                Path(run.inference_folder),
                None,
                backup_folder=Path(run.backup_folder),
                image_extensions=self.config.processing.image_extensions,
            )
        return False

    async def _fallback(self, run, lane, reason):
        manifest, path = await self._io(
            lane,
            build_fallback,
            self.config,
            self.artifacts,
            run.job_id,
            Path(run.csv_snapshot_folder),
            reason,
            run.last_error or "",
        )
        run.result_manifest_path = str(path)
        run.reason = manifest.reason
        if run.image_backup_status != "complete":
            run.backup_started_at = time.time()
            run.image_backup_status = await self._io(
                lane,
                ingest_io.backup_images,
                Path(run.source_folder),
                Path(run.inference_folder),
                Path(run.csv_snapshot_folder),
                backup_folder=Path(run.backup_folder),
                image_extensions=self.config.processing.image_extensions,
            )
            run.backup_finished_at = time.time()

    async def _publish(self, run, lane):
        settings = self.config.delivery
        for name, record in run.publications.items():
            for number in range(1, settings.max_attempts + 1):
                if number > 1:
                    await asyncio.sleep(settings.retry_interval_seconds)
                record.update(status="PUBLISHING", attempts=number)
                attempt = {
                    "number": number,
                    "started_at": time.time(),
                    "status": "PUBLISHING",
                }
                record["attempt_log"].append(attempt)
                await self._save(run, lane, "delivery.started")
                started = time.monotonic()
                try:
                    result = await self._io(
                        lane,
                        publish_csv,
                        self.config,
                        Path(record["result_path"]),
                        record["result_sha256"],
                        time.time() + settings.attempt_timeout_seconds,
                        job_id=run.job_id,
                        timeout=settings.attempt_timeout_seconds,
                    )
                except (PublishCompletionUnknown, DeadlineExceeded) as exc:
                    record.update(
                        status="UNKNOWN"
                        if getattr(exc, "uncertain", True)
                        else "FAILED",
                        error=str(exc),
                    )
                except Exception as exc:
                    record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                else:
                    record.update(
                        status="RETURNED",
                        returned_at=time.time(),
                        destination=str(result.destination_csv),
                        error=None,
                    )
                    record["processing_to_return_ms"] = (
                        record["returned_at"] - run.processing_started_at
                    ) * 1000
                attempt.update(
                    status=record["status"],
                    finished_at=time.time(),
                    elapsed_ms=(time.monotonic() - started) * 1000,
                    error=record.get("error"),
                )
                await self._save(run, lane, "delivery.attempt")
                if record["status"] in {"RETURNED", "UNKNOWN"}:
                    break
            if record["status"] == "RETURNED":
                await self._mark(run, name, record, lane)
        statuses = {r["status"] for r in run.publications.values()}
        run.status = (
            "PUBLICATION_UNKNOWN"
            if "UNKNOWN" in statuses
            else "DONE"
            if statuses == {"RETURNED"}
            else "PARTIAL_RETURN"
            if "RETURNED" in statuses
            else "RETURN_FAILED"
        )
        returned = [
            r["returned_at"]
            for r in run.publications.values()
            if r["status"] == "RETURNED"
        ]
        if returned:
            run.primary_returned_at = max(returned)
        if run.status != "DONE":
            run.last_error = "; ".join(
                f"{name}: {r.get('error')}"
                for name, r in run.publications.items()
                if r["status"] != "RETURNED"
            )

    async def _mark(self, run, name, record, lane):
        record["marker_attempted_at"] = time.time()
        record["marker_attempts"] = record.get("marker_attempts", 0) + 1
        try:
            destination = await self._io(
                lane,
                mark_done,
                Path(run.source_folder) / name,
                record["source_sha256"],
                expected_signature=run.source_signatures.get(name),
            )
        except Exception as exc:
            record.update(
                marker_status="failed", marker_error=f"{type(exc).__name__}: {exc}"
            )
            self.log.warning(
                "event=source.mark_failed job_id=%s source_csv=%s error=%s",
                run.job_id,
                name,
                record["marker_error"],
            )
        else:
            record.update(
                marker_status="done",
                marked_at=time.time(),
                marked_path=destination,
                marker_error=None,
            )
            self._deferred.pop((Path(run.source_folder), name), None)
        await self._save(
            run,
            lane,
            "source.marked"
            if record["marker_status"] == "done"
            else "source.mark_failed",
        )

    async def _finish(self, run, lane):
        for name, record in run.publications.items():
            if record["status"] == "RETURNED" and record["marker_status"] != "done":
                await self._mark(run, name, record, lane)
        returned = tuple(
            name
            for name, record in run.publications.items()
            if record["status"] == "RETURNED"
        )
        if not returned:
            run.export_status = "disabled"
        elif run.export_status == "pending":
            run.export_started_at = time.time()
            try:
                await self._io(
                    lane,
                    export_result,
                    self.config.model_dump(),
                    Path(run.backup_folder),
                    returned,
                    timeout=self.config.runtime.background_timeout_seconds,
                )
            except Exception as exc:
                run.export_error = f"{type(exc).__name__}: {exc}"
                self.log.warning(
                    "event=export.failed run_id=%s error=%s",
                    run.run_id,
                    run.export_error,
                )
            else:
                run.export_status, run.export_error = "complete", None
            finally:
                run.export_finished_at = time.time()
        run.next_background_at = (
            time.time() + self.config.runtime.background_retry_seconds
        )
        pending_marker = any(
            r["status"] == "RETURNED" and r["marker_status"] != "done"
            for r in run.publications.values()
        )
        if run.status in {"DONE", "MARK_PENDING"}:
            run.status = "MARK_PENDING" if pending_marker else "DONE"
        await self._save(run, lane, "job.report_updated")
        if pending_marker or run.export_status == "pending":
            self._background_runs[run.run_id] = run
        else:
            self._background_runs.pop(run.run_id, None)

    async def _process(self, run, signatures):
        lane = await self._new_lane()
        started = time.monotonic()
        run.processing_started_at = time.time()
        run.status = "PROCESSING"
        run.queue_ms = (run.processing_started_at - run.first_seen_at) * 1000
        try:
            await self._save(run, lane, "job.processing_started")
            ready = await self._prepare_input(run, lane)
            if ready:
                if self._warm_error:
                    run.last_error = str(self._warm_error)
                    await self._fallback(
                        run,
                        lane,
                        "required_model_timeout"
                        if isinstance(self._warm_error, TimeoutError)
                        else "required_model_failure",
                    )
                else:
                    run.status = "INFERENCING"
                    run.inference_started_at = time.time()
                    await self._save(run, lane, "inference.started")
                    manifest, path = await self.inference.prepare(
                        job_id=run.job_id,
                        staged_folder=Path(run.inference_folder),
                    )
                    run.inference_finished_at = time.time()
                    run.result_manifest_path, run.reason = str(path), manifest.reason
                    run.last_error = "; ".join(manifest.errors) or None
            if run.result_manifest_path:
                run.status = "RESULT_READY"
                run.result_ready_at = time.time()
                run.result_manifest_path, run.publications = await self._io(
                    lane,
                    commit_result,
                    Path(run.csv_snapshot_folder),
                    Path(run.result_manifest_path),
                    Path(run.backup_folder),
                    run.run_id,
                )
                await self._publish(run, lane)
                await self._finish(run, lane)
        except ReportPersistenceError:
            raise
        except asyncio.CancelledError:
            run.status, run.reason = "INTERRUPTED", "process_stopped"
            raise
        except Exception as exc:
            run.status, run.reason, run.last_error = (
                "FAILED",
                "processing_failure",
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            try:
                run.processing_finished_at = time.time()
                run.processing_elapsed_ms = (time.monotonic() - started) * 1000
                await self._save(run, lane, "job.processing_finished")
                for name in run.csv_names:
                    record = run.publications.get(name, {})
                    if record.get("marker_status") == "done":
                        self._deferred.pop((Path(run.source_folder), name), None)
                        continue
                    # Avoid resending uncertain or returned-but-unmarked CSVs.
                    retry = (
                        None
                        if record.get("status") in {"UNKNOWN", "RETURNED"}
                        else time.time() + self.config.runtime.background_retry_seconds
                    )
                    self._deferred[(Path(run.source_folder), name)] = (
                        run.source_signatures.get(name, signatures[name]),
                        retry,
                    )
                await self._io(
                    lane, shutil.rmtree, self.artifacts / run.job_id, ignore_errors=True
                )
                await self._io(
                    lane, shutil.rmtree, Path(run.workspace_folder), ignore_errors=True
                )
            finally:
                lane.close()
                self._busy_folders.discard(Path(run.source_folder))
                self.jobs.pop(run.run_id, None)

    async def _warm(self):
        try:
            await asyncio.wait_for(
                self.runtime.start(), self.config.inference.obb.startup_timeout_seconds
            )
        except Exception as exc:
            self._warm_error = exc
            self.runtime.close()
            self.log.error(
                "Model unavailable; valid CSVs will use all-23 protection: %s", exc
            )

    async def _worker(self):
        await self._warm_task
        while True:
            run, signatures = await self._queue.get()
            try:
                await self._process(run, signatures)
            finally:
                self._queue.task_done()

    async def _maintenance(self, *, once=False):
        while True:
            for run in list(self._background_runs.values()):
                if run.run_id in self.jobs:
                    continue
                if not once and time.time() < run.next_background_at:
                    continue
                lane = await self._new_lane()
                try:
                    await self._finish(run, lane)
                finally:
                    lane.close()
            if once:
                return
            await asyncio.sleep(self.config.runtime.background_retry_seconds)

    def _check_tasks(self):
        for task in (self._worker_task, self._maintenance_task):
            if task is not None and task.done():
                task.result()

    async def run(self, *, once=False, target_date=None):
        self._once_results = {} if once else None
        identity = os.path.normcase(str(Path(self.config.paths.watch_root).resolve()))
        lock_root = (
            Path(tempfile.gettempdir())
            / "spi-ai-locks"
            / hashlib.sha256(identity.encode()).hexdigest()
        )
        with ExitStack() as ownership:
            try:
                ownership.enter_context(ControllerLock(lock_root))
                self._warm_task = asyncio.create_task(self._warm())
                self._worker_task = asyncio.create_task(self._worker())
                if not once:
                    self._maintenance_task = asyncio.create_task(self._maintenance())
                while True:
                    self._check_tasks()
                    try:
                        await self._discover(target_date)
                    except (OSError, DeadlineExceeded) as exc:
                        if once:
                            raise
                        if time.monotonic() - self._scan_warning_at >= 5:
                            self.log.warning(
                                "Discovery deferred watch_root=%s: %s",
                                self.config.paths.watch_root,
                                exc,
                            )
                            self._scan_warning_at = time.monotonic()
                    if once:
                        while self.jobs:
                            self._check_tasks()
                            await asyncio.sleep(
                                self.config.runtime.poll_interval_seconds
                            )
                        self._check_tasks()
                        await self._maintenance(once=True)
                        return self._once_results
                    await asyncio.sleep(self.config.runtime.scan_interval_seconds)
            finally:
                await self.close()

    async def close(self):
        tasks = [
            t
            for t in (self._worker_task, self._maintenance_task, self._warm_task)
            if t is not None
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._discovery_lane is not None:
            self._discovery_lane.close()
        self.runtime.close()
        shutil.rmtree(self.artifacts, ignore_errors=True)
