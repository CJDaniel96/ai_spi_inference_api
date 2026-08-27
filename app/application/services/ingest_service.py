"""Stage-one validation and immutable archival for timestamp-folder jobs.

The service deliberately separates :meth:`validate` from :meth:`archive`.
Validation establishes the soft-real-time ``ready_at`` timestamp as soon as
the CSV contract, every expected image, and a stable source snapshot have been
verified.  A worker can therefore persist/enqueue that deadline before the
potentially slower copy to local storage begins.

Archival copies the complete source tree into temporary directories on the
destination filesystem, verifies their byte content, and atomically renames
them into place.  Existing byte-identical targets are reused, making retries
idempotent.  By default the immutable raw backup is also the staging folder,
so the shared-folder payload is read only once.  A distinct staging root may be
provided when an installation explicitly needs two local copies.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.core.errors import AppError, CsvSchemaError
from app.domain.services.csv_merger import CsvMerger

_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_TIMESTAMP_PATTERN = re.compile(r"\d{14}")
_CSV_SUFFIX = ".csv"
_HASH_CHUNK_SIZE = 1024 * 1024


class IngestValidationError(AppError, ValueError):
    """The source job does not satisfy the machine-input contract."""

    status_code = 400


class SourceChangedError(IngestValidationError):
    """The source tree changed during validation or archival."""


class IngestStorageError(AppError):
    """A local staging or backup operation failed."""

    status_code = 500


class IngestCollisionError(IngestStorageError):
    """An existing job target has different content and cannot be overwritten."""

    status_code = 409


@dataclass(frozen=True)
class FileSnapshotEntry:
    """One stat-only source entry used for fast stability comparisons."""

    relative_path: str
    kind: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ContentManifestEntry:
    """One byte-verified tree entry used for idempotency checks."""

    relative_path: str
    kind: str
    size: int
    sha256: str


TreeSnapshot = tuple[FileSnapshotEntry, ...]
ContentManifest = tuple[ContentManifestEntry, ...]


@dataclass(frozen=True)
class ValidatedIngest:
    """Stable, decoded source job ready to be queued and archived."""

    source_folder: Path
    job_id: str
    expected_image_keys: tuple[str, ...]
    ready_at: datetime
    deadline_at: datetime
    source_snapshot: TreeSnapshot


@dataclass(frozen=True)
class ArchivePlan:
    """Deterministic local paths that may be persisted before copying starts."""

    staged_folder: Path
    original_backup_folder: Path


@dataclass(frozen=True)
class IngestResult:
    """Local immutable paths and deadline carried to the inference worker."""

    source_folder: Path
    staged_folder: Path
    original_backup_folder: Path
    expected_image_keys: tuple[str, ...]
    ready_at: datetime
    deadline_at: datetime
    staging_reused: bool
    backup_reused: bool

    @property
    def reused(self) -> bool:
        """Return whether every required local target already existed."""
        return self.staging_reused and self.backup_reused


@dataclass(frozen=True)
class _PreparedCopy:
    target: Path
    temporary: Path | None
    reused: bool


class IngestService:
    """Validate one timestamp job and create immutable local raw copies."""

    def __init__(
        self,
        *,
        image_name_source_column: str | None = None,
        image_name_template: str | None = None,
        primary_return_deadline_seconds: float = 30.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if primary_return_deadline_seconds <= 0:
            raise ValueError("primary_return_deadline_seconds must be positive")
        self._source_column = image_name_source_column
        self._template = image_name_template
        self._deadline_seconds = primary_return_deadline_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._csv_merger = CsvMerger()

    def validate(self, source_folder: str | Path) -> ValidatedIngest:
        """Validate CSV/image completeness and establish the job deadline.

        ``ready_at`` is sampled immediately after the post-validation snapshot
        matches the pre-validation snapshot.  Copy time therefore consumes the
        same 30-second end-to-end budget as inference and publication.
        """
        source = self._resolve_source(source_folder)
        job_id = self._validate_job_id(source.name)
        initial_snapshot = self._snapshot_tree(source)
        csv_files = self._csv_files(source)
        if not csv_files:
            raise IngestValidationError(f"No CSV files found in job: {source}")

        image_keys: list[str] = []
        seen_keys: dict[str, tuple[Path, int]] = {}
        for csv_path in csv_files:
            frame = self._read_csv(csv_path)
            if frame.empty:
                raise IngestValidationError(f"CSV contains no data rows: {csv_path}")
            try:
                frame = self._csv_merger.add_image_name_column(
                    frame,
                    source_column=self._source_column,
                    template=self._template,
                    csv_stem=csv_path.stem,
                )
            except CsvSchemaError as exc:
                raise IngestValidationError(
                    f"Invalid image-name contract in {csv_path.name}: {exc}"
                ) from exc

            for row_number, raw_key in enumerate(frame["img_name"], start=2):
                image_key = self._safe_image_key(raw_key, csv_path, row_number)
                collision_key = image_key.casefold()
                previous = seen_keys.get(collision_key)
                if previous is not None:
                    previous_csv, previous_row = previous
                    raise IngestValidationError(
                        "Duplicate expected image key "
                        f"{image_key!r}: {previous_csv.name} row {previous_row} and "
                        f"{csv_path.name} row {row_number}"
                    )
                seen_keys[collision_key] = (csv_path, row_number)
                self._validate_image(source / image_key, image_key)
                image_keys.append(image_key)

        self._assert_snapshot_unchanged(source, initial_snapshot, "validation")
        ready_at = self._aware_now()
        return ValidatedIngest(
            source_folder=source,
            job_id=job_id,
            expected_image_keys=tuple(image_keys),
            ready_at=ready_at,
            deadline_at=ready_at + timedelta(seconds=self._deadline_seconds),
            source_snapshot=initial_snapshot,
        )

    def archive(
        self,
        validated: ValidatedIngest,
        *,
        backup_root: str | Path,
        staging_root: str | Path | None = None,
    ) -> IngestResult:
        """Atomically archive a validated source tree to local storage.

        When ``staging_root`` is omitted, the immutable backup is returned as
        ``staged_folder``.  The same optimization is applied when staging and
        backup roots resolve to the same directory.
        """
        source = validated.source_folder
        self._assert_snapshot_unchanged(source, validated.source_snapshot, "copy")
        plan = self.plan_archive(
            validated, backup_root=backup_root, staging_root=staging_root
        )
        backup_target = plan.original_backup_folder
        staging_target = plan.staged_folder

        source_manifest = self._content_manifest(source, source=True)
        self._assert_snapshot_unchanged(source, validated.source_snapshot, "copy")

        # Commit the original backup before any optional staging copy.  Publisher
        # requires this directory before Primary can become machine-visible.
        targets = tuple(dict.fromkeys((backup_target, staging_target)))
        prepared: dict[Path, _PreparedCopy] = {}
        committed_targets: set[Path] = set()
        try:
            for target in targets:
                prepared[target] = self._prepare_copy(source, target, source_manifest)
            self._assert_snapshot_unchanged(source, validated.source_snapshot, "copy")
            for target in targets:
                before_commit = prepared[target]
                prepared[target] = self._commit_copy(before_commit, source_manifest)
                if before_commit.temporary is not None and not prepared[target].reused:
                    committed_targets.add(target)
            self._assert_snapshot_unchanged(source, validated.source_snapshot, "copy")
        except SourceChangedError:
            # These directories were atomically committed by this call but have
            # not been returned/enqueued as READY.  Removing only our own new
            # targets lets a fresh stable version of the same job id retry;
            # pre-existing idempotent targets are never touched.
            for target in committed_targets:
                shutil.rmtree(target, ignore_errors=True)
            raise
        finally:
            for item in prepared.values():
                if item.temporary is not None:
                    shutil.rmtree(item.temporary, ignore_errors=True)

        staging_reused = prepared[staging_target].reused
        backup_reused = prepared[backup_target].reused
        return IngestResult(
            source_folder=source,
            staged_folder=staging_target,
            original_backup_folder=backup_target,
            expected_image_keys=validated.expected_image_keys,
            ready_at=validated.ready_at,
            deadline_at=validated.deadline_at,
            staging_reused=staging_reused,
            backup_reused=backup_reused,
        )

    def plan_archive(
        self,
        validated: ValidatedIngest,
        *,
        backup_root: str | Path,
        staging_root: str | Path | None = None,
    ) -> ArchivePlan:
        """Return immutable target paths without touching local storage.

        The ingest worker can persist these paths with ``ready_at`` and
        ``deadline_at`` before starting the archive, so a process crash during
        copy is recoverable from the durable ``INGESTING`` state.
        """
        source = validated.source_folder
        backup_base = Path(backup_root).expanduser().resolve(strict=False)
        backup_target = (
            backup_base
            / self._job_date(validated.job_id).isoformat()
            / validated.job_id
        )
        if staging_root is None:
            staging_target = backup_target
        else:
            staging_base = Path(staging_root).expanduser().resolve(strict=False)
            if staging_base == backup_base:
                staging_target = backup_target
            else:
                staging_target = staging_base / validated.job_id

        self._validate_target(source, backup_target)
        self._validate_target(source, staging_target)
        return ArchivePlan(
            staged_folder=staging_target,
            original_backup_folder=backup_target,
        )

    def ingest(
        self,
        source_folder: str | Path,
        *,
        backup_root: str | Path,
        staging_root: str | Path | None = None,
    ) -> IngestResult:
        """Validate and archive in one call for simple worker deployments."""
        validated = self.validate(source_folder)
        return self.archive(
            validated, backup_root=backup_root, staging_root=staging_root
        )

    @staticmethod
    def _resolve_source(source_folder: str | Path) -> Path:
        source = Path(source_folder).expanduser()
        try:
            source = source.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise IngestValidationError(
                f"Source job folder does not exist: {source_folder}"
            ) from exc
        if not source.is_dir():
            raise IngestValidationError(
                f"Source job path is not a directory: {source_folder}"
            )
        return source

    @staticmethod
    def _validate_job_id(folder_name: str) -> str:
        if _TIMESTAMP_PATTERN.fullmatch(folder_name) is None:
            raise IngestValidationError(
                f"Job folder must be a 14-digit timestamp: {folder_name!r}"
            )
        try:
            datetime.strptime(folder_name, _TIMESTAMP_FORMAT)
        except ValueError as exc:
            raise IngestValidationError(
                f"Job folder has an invalid timestamp: {folder_name!r}"
            ) from exc
        return folder_name

    @staticmethod
    def _job_date(job_id: str):
        return datetime.strptime(job_id, _TIMESTAMP_FORMAT).date()

    @staticmethod
    def _csv_files(source: Path) -> tuple[Path, ...]:
        return tuple(
            sorted(
                child
                for child in source.iterdir()
                if child.suffix.lower() == _CSV_SUFFIX
                and stat.S_ISREG(child.lstat().st_mode)
            )
        )

    @staticmethod
    def _read_csv(csv_path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(csv_path)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise IngestValidationError(
                f"Unable to parse CSV {csv_path.name}: {exc}"
            ) from exc

    @staticmethod
    def _safe_image_key(raw_key: object, csv_path: Path, row_number: int) -> str:
        image_key = str(raw_key).strip().replace("\\", "/")
        if (
            not image_key
            or image_key in {".", ".."}
            or "/" in image_key
            or Path(image_key).name != image_key
        ):
            raise IngestValidationError(
                f"Unsafe image filename at {csv_path.name} row {row_number}: "
                f"{raw_key!r}"
            )
        return image_key

    @staticmethod
    def _validate_image(image_path: Path, image_key: str) -> None:
        try:
            metadata = image_path.lstat()
        except FileNotFoundError as exc:
            raise IngestValidationError(
                f"Expected image does not exist: {image_key}"
            ) from exc
        except OSError as exc:
            raise IngestValidationError(
                f"Unable to inspect expected image {image_key}: {exc}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise IngestValidationError(
                f"Expected image is not a regular file: {image_key}"
            )
        if metadata.st_size <= 0:
            raise IngestValidationError(f"Expected image is empty: {image_key}")
        try:
            encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        except (OSError, cv2.error) as exc:
            raise IngestValidationError(
                f"Unable to decode expected image {image_key}: {exc}"
            ) from exc
        if decoded is None or decoded.size == 0:
            raise IngestValidationError(f"Expected image is not decodable: {image_key}")

    def _aware_now(self) -> datetime:
        current = self._clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError(
                "IngestService clock must return a timezone-aware datetime"
            )
        return current

    @classmethod
    def _snapshot_tree(cls, root: Path) -> TreeSnapshot:
        try:
            entries: list[FileSnapshotEntry] = []
            for path in cls._walk_tree(root):
                metadata = path.lstat()
                relative = path.relative_to(root).as_posix()
                if stat.S_ISDIR(metadata.st_mode):
                    kind = "directory"
                    size = 0
                elif stat.S_ISREG(metadata.st_mode):
                    kind = "file"
                    size = metadata.st_size
                else:
                    raise IngestValidationError(
                        f"Source entry is not a regular file or directory: {relative}"
                    )
                entries.append(
                    FileSnapshotEntry(
                        relative_path=relative,
                        kind=kind,
                        size=size,
                        mtime_ns=metadata.st_mtime_ns,
                    )
                )
            return tuple(sorted(entries, key=lambda item: item.relative_path))
        except IngestValidationError:
            raise
        except OSError as exc:
            raise SourceChangedError(
                f"Unable to capture stable source snapshot for {root}: {exc}"
            ) from exc

    @staticmethod
    def _walk_tree(root: Path) -> tuple[Path, ...]:
        paths: list[Path] = []
        for directory, dir_names, file_names in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            for name in sorted((*dir_names, *file_names)):
                paths.append(directory_path / name)
        return tuple(paths)

    @classmethod
    def _content_manifest(cls, root: Path, *, source: bool = False) -> ContentManifest:
        entries: list[ContentManifestEntry] = []
        try:
            for path in cls._walk_tree(root):
                metadata = path.lstat()
                relative = path.relative_to(root).as_posix()
                if stat.S_ISDIR(metadata.st_mode):
                    entries.append(ContentManifestEntry(relative, "directory", 0, ""))
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    error_type = IngestValidationError if source else IngestStorageError
                    raise error_type(f"Tree contains a non-regular entry: {relative}")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                        digest.update(chunk)
                entries.append(
                    ContentManifestEntry(
                        relative,
                        "file",
                        metadata.st_size,
                        digest.hexdigest(),
                    )
                )
        except (IngestValidationError, IngestStorageError):
            raise
        except OSError as exc:
            if source:
                raise SourceChangedError(
                    f"Source changed while calculating content manifest: {root}"
                ) from exc
            raise IngestStorageError(
                f"Unable to verify local ingest target {root}: {exc}"
            ) from exc
        return tuple(sorted(entries, key=lambda item: item.relative_path))

    @classmethod
    def _assert_snapshot_unchanged(
        cls, root: Path, expected: TreeSnapshot, phase: str
    ) -> None:
        current = cls._snapshot_tree(root)
        if current != expected:
            raise SourceChangedError(
                f"Source job changed during {phase}; retry after it becomes stable: "
                f"{root}"
            )

    @staticmethod
    def _validate_target(source: Path, target: Path) -> None:
        if target == source or target.is_relative_to(source):
            raise IngestStorageError(
                f"Ingest target must not be inside the source job: {target}"
            )

    @classmethod
    def _prepare_copy(
        cls,
        source: Path,
        target: Path,
        source_manifest: ContentManifest,
    ) -> _PreparedCopy:
        if target.exists():
            if not target.is_dir():
                raise IngestCollisionError(
                    f"Ingest target exists and is not a directory: {target}"
                )
            if cls._content_manifest(target) != source_manifest:
                raise IngestCollisionError(
                    f"Ingest target already exists with different content: {target}"
                )
            return _PreparedCopy(target=target, temporary=None, reused=True)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
                )
            )
            try:
                shutil.copytree(
                    source,
                    temporary,
                    dirs_exist_ok=True,
                    copy_function=shutil.copy2,
                )
                if cls._content_manifest(temporary) != source_manifest:
                    raise SourceChangedError(
                        f"Source changed while copying job: {source}"
                    )
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            return _PreparedCopy(target=target, temporary=temporary, reused=False)
        except (IngestValidationError, IngestStorageError):
            raise
        except OSError as exc:
            raise IngestStorageError(
                f"Unable to prepare local ingest target {target}: {exc}"
            ) from exc

    @classmethod
    def _commit_copy(
        cls,
        prepared: _PreparedCopy,
        source_manifest: ContentManifest,
    ) -> _PreparedCopy:
        if prepared.temporary is None:
            return prepared
        try:
            prepared.temporary.rename(prepared.target)
            return _PreparedCopy(target=prepared.target, temporary=None, reused=False)
        except OSError as exc:
            # A concurrent idempotent worker may have won the rename race.
            if prepared.target.is_dir():
                if cls._content_manifest(prepared.target) == source_manifest:
                    shutil.rmtree(prepared.temporary, ignore_errors=True)
                    return _PreparedCopy(
                        target=prepared.target, temporary=None, reused=True
                    )
                raise IngestCollisionError(
                    f"Concurrent ingest target has different content: {prepared.target}"
                ) from exc
            raise IngestStorageError(
                f"Unable to atomically publish ingest target {prepared.target}: {exc}"
            ) from exc
