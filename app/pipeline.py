"""CLI entry point for the three independently runnable pipeline workers."""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import tempfile
import time
from datetime import date
from pathlib import Path, PureWindowsPath

import httpx

from app.application.services.pipeline_inference_service import (
    PipelineInferenceService,
)
from app.application.services.pipeline_publisher_service import (
    PipelinePublisherService,
)
from app.core.config import AppConfig, load_config, resolve_under_project_root
from app.core.logging import get_logger, setup_logging
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one durable SPI pipeline stage independently."
    )
    parser.add_argument("stage", choices=("ingest", "inference", "publisher"))
    parser.add_argument(
        "--once", action="store_true", help="Process at most one claim and exit."
    )
    parser.add_argument("--worker-id", help="Stable worker id used for SQLite leases.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Config JSON path (otherwise AI_CONFIG_PATH/default is used).",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="Manual ingest date (YYYY-MM-DD); requires 'ingest --once'.",
    )
    return parser


def _configured_path(value: str | Path) -> Path:
    """Resolve local relative paths while preserving Windows drive paths."""
    text = str(value)
    candidate = Path(text)
    if candidate.is_absolute() or PureWindowsPath(text).drive:
        return candidate
    return resolve_under_project_root(candidate)


def _worker_id(stage: str, configured: str | None) -> str:
    return configured or f"{socket.gethostname()}:{os.getpid()}:{stage}"


def _repository(config: AppConfig) -> PipelineJobRepository:
    return PipelineJobRepository(_configured_path(config.pipeline.database_path))


def _configure_logging(config: AppConfig, stage: str) -> None:
    base_name = Path(config.logging.system_log_file)
    stage_log_name = f"{base_name.stem}.{stage}{base_name.suffix}"
    setup_logging(
        log_dir=_configured_path(config.logging.log_dir),
        system_log_file=stage_log_name,
    )


def _run_ingest(
    config: AppConfig,
    *,
    worker_id: str,
    once: bool,
    target_date: date | None = None,
) -> None:
    # Lazy import keeps the inference/publisher processes independent from cv2.
    from app.application.workers.ingest_worker import IngestWorker

    watch_root_value = config.pipeline.watch_root
    if not watch_root_value:
        raise ValueError("pipeline.watch_root is required for the ingest worker")
    worker = IngestWorker(
        config=config,
        repository=_repository(config),
        worker_id=worker_id,
        watch_root=_configured_path(watch_root_value),
        backup_root=_configured_path(config.paths.backup_output_root),
        staging_root=(
            _configured_path(config.pipeline.staging_root)
            if config.pipeline.staging_root
            else None
        ),
    )
    if once:
        # A fresh folder needs two stable observations.  Keep one-shot/manual
        # ingest bounded while allowing the configured settle window to elapse.
        stop_at = time.monotonic() + config.pipeline.source_settle_seconds + 1.0
        while True:
            if worker.run_once(target_date=target_date):
                break
            if time.monotonic() >= stop_at:
                break
            time.sleep(config.pipeline.ingest_poll_interval_seconds)
    else:
        worker.run_forever()


async def _run_inference(config: AppConfig, *, worker_id: str, once: bool) -> None:
    from app.application.workers.inference_worker import InferenceWorker

    async with httpx.AsyncClient() as http_client:
        service = PipelineInferenceService(
            config=config,
            result_root=_configured_path(config.pipeline.result_root),
            http_client=http_client,
        )
        worker = InferenceWorker(
            config=config,
            repository=_repository(config),
            service=service,
            worker_id=worker_id,
        )
        if once:
            await worker.run_once()
        else:
            await worker.run_forever()


def _run_publisher(config: AppConfig, *, worker_id: str, once: bool) -> None:
    from app.application.workers.publisher_worker import PublisherWorker

    result_root = _configured_path(config.pipeline.result_root)
    worker = PublisherWorker(
        config=config,
        repository=_repository(config),
        publisher=PipelinePublisherService(
            config=config,
            machine_output_root=_configured_path(config.paths.external_output_root),
        ),
        fallback_builder=PipelineInferenceService(
            config=config,
            result_root=result_root,
        ),
        worker_id=worker_id,
    )
    if once:
        worker.run_once()
    else:
        worker.run_forever()


def _ensure_readable_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"{label} is not an accessible directory: {path}")
    try:
        next(path.iterdir(), None)
    except OSError as exc:
        raise RuntimeError(f"{label} is not readable: {path}: {exc}") from exc


def _ensure_writable_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, probe_name = tempfile.mkstemp(
            dir=path,
            prefix=".pipeline-write-probe-",
            suffix=".tmp",
        )
        os.close(descriptor)
        Path(probe_name).unlink()
    except OSError as exc:
        raise RuntimeError(f"{label} is not writable: {path}: {exc}") from exc


def _ensure_output_directory(path: Path, label: str) -> None:
    """Check a machine-facing directory without creating a probe file in it."""
    _ensure_readable_directory(path, label)
    if not os.access(path, os.W_OK):
        raise RuntimeError(f"{label} is not writable: {path}")


def _preflight(config: AppConfig, stage: str) -> None:
    # Opening the repository validates/creates the local SQLite state path.
    _repository(config)
    if stage == "ingest":
        assert config.pipeline.watch_root is not None
        _ensure_readable_directory(
            _configured_path(config.pipeline.watch_root), "pipeline.watch_root"
        )
        _ensure_writable_directory(
            _configured_path(config.paths.backup_output_root),
            "paths.backup_output_root",
        )
        if config.pipeline.staging_root:
            _ensure_writable_directory(
                _configured_path(config.pipeline.staging_root),
                "pipeline.staging_root",
            )
    elif stage == "inference":
        _ensure_writable_directory(
            _configured_path(config.pipeline.result_root), "pipeline.result_root"
        )
    else:
        _ensure_output_directory(
            _configured_path(config.paths.external_output_root),
            "paths.external_output_root",
        )
        _ensure_writable_directory(
            _configured_path(config.paths.backup_output_root),
            "paths.backup_output_root",
        )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.date is not None and (args.stage != "ingest" or not args.once):
        parser.error("--date requires: ingest --once")
    config = load_config(args.config)
    _configure_logging(config, args.stage)
    logger = get_logger("pipeline.cli")
    if not config.pipeline.enabled:
        logger.error("event=pipeline.startup_failed err=pipeline.enabled must be true")
        return 2
    worker_id = _worker_id(args.stage, args.worker_id)
    try:
        _preflight(config, args.stage)
    except Exception as exc:
        logger.exception(
            "event=pipeline.startup_preflight_failed stage=%s err=%s",
            args.stage,
            exc,
        )
        return 2
    logger.info(
        "event=pipeline.worker_start stage=%s worker_id=%s once=%s database=%s",
        args.stage,
        worker_id,
        args.once,
        _configured_path(config.pipeline.database_path),
    )
    try:
        if args.stage == "ingest":
            _run_ingest(
                config,
                worker_id=worker_id,
                once=args.once,
                target_date=args.date,
            )
        elif args.stage == "inference":
            asyncio.run(_run_inference(config, worker_id=worker_id, once=args.once))
        else:
            _run_publisher(config, worker_id=worker_id, once=args.once)
    except KeyboardInterrupt:
        logger.info(
            "event=pipeline.worker_stop stage=%s worker_id=%s reason=interrupt",
            args.stage,
            worker_id,
        )
    except Exception as exc:
        logger.exception(
            "event=pipeline.worker_fatal stage=%s worker_id=%s err=%s",
            args.stage,
            worker_id,
            exc,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
