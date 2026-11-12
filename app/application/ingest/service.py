"""Validate input and atomically preserve one immutable raw backup.

The controller owns all timestamps and deadlines. This service checks source
stability, copies source bytes once with hashes, validates the local images,
and publishes the verified backup by renaming its temporary directory.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.core.errors import AppError, CsvSchemaError
from app.domain.services.csv_merger import CsvMerger
from app.infrastructure.input.sinic_folder_input import is_pending_csv
from app.infrastructure.local_store import ensure_directory, fsync_directory
from app.infrastructure.output.sinic_csv_output import _parse_document
from app.infrastructure.repositories.csv_repository import CsvRepository

_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_TIMESTAMP_PATTERN = re.compile(r"\d{14}")
_HASH_CHUNK_SIZE = 1024 * 1024


class IngestValidationError(AppError, ValueError):
    """The source job does not satisfy the machine-input contract."""


class SourceChangedError(IngestValidationError):
    """The source tree changed during validation or archival."""


class IngestStorageError(AppError):
    """A local raw-backup operation failed."""


class IngestCollisionError(IngestStorageError):
    """An existing job target has different content and cannot be overwritten."""


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
    """Source snapshot and expected image keys; decoding may be deferred locally."""

    source_folder: Path
    job_id: str
    expected_image_keys: tuple[str, ...]
    source_snapshot: TreeSnapshot
    images_decoded: bool = True
    csv_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class IngestResult:
    """The verified backup destination and whether identical data was reused."""

    original_backup_folder: Path
    reused: bool


@dataclass(frozen=True)
class _PreparedCopy:
    target: Path
    temporary: Path | None
    reused: bool


class IngestService:
    """Validate one timestamp job and create its immutable local raw backup."""

    def __init__(
        self,
        *,
        image_name_template: str = (
            "{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg"
        ),
    ) -> None:
        self._template = image_name_template
        self._csv_merger = CsvMerger()

    def snapshot_csv(
        self,
        source_folder: Path,
        destination: Path,
        csv_names=None,
        *,
        expected_signature=None,
    ) -> Path:
        """Validate/atomically snapshot only CSV bytes; never inspect images."""
        source = self._resolve_source(source_folder)
        self._validate_job_id(source.name)
        self._validate_target(source, destination.resolve(strict=False))
        csv_files = tuple(
            p
            for p in self._csv_files(source)
            if csv_names is None or p.name in csv_names
        )
        if csv_names is not None and len(csv_files) != len(csv_names):
            raise SourceChangedError("Selected CSV disappeared")
        if not csv_files:
            raise IngestValidationError("No CSV files found")

        def snapshot():
            return tuple(
                (p.name, p.stat().st_size, p.stat().st_mtime_ns)
                for p in self._csv_files(source)
                if csv_names is None or p.name in csv_names
            )

        initial = snapshot()
        if expected_signature is not None and initial != expected_signature:
            raise SourceChangedError("CSV changed after the stability check")
        payloads = {}
        seen: set[str] = set()
        for path in csv_files:
            payload = path.read_bytes()
            document = _parse_document(payload, path.name)
            if "is_pass" not in document.fieldnames or not document.rows:
                raise IngestValidationError(
                    f"Missing is_pass or data rows: {path.name}"
                )
            if len(set(document.fieldnames)) != len(document.fieldnames):
                raise IngestValidationError(f"Duplicate CSV headers: {path.name}")
            if any(
                None in row or any(v is None for v in row.values())
                for row in document.rows
            ):
                raise IngestValidationError(f"Incomplete CSV row: {path.name}")
            frame = pd.DataFrame(document.rows)
            frame = self._csv_merger.add_image_name_column(
                frame,
                template=self._template,
                csv_stem=path.stem,
            )
            for number, raw_key in enumerate(frame["img_name"], 2):
                key = self._safe_image_key(raw_key, path, number).casefold()
                if key in seen:
                    raise IngestValidationError(f"Duplicate expected image key: {key}")
                seen.add(key)
            payloads[path.name] = payload
        if snapshot() != initial:
            raise SourceChangedError("CSV set changed during snapshot")
        ensure_directory(destination.parent)
        temporary = Path(tempfile.mkdtemp(prefix=".csv-", dir=destination.parent))
        try:
            for name, payload in payloads.items():
                with (temporary / name).open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            if destination.exists():
                if {p.name: p.read_bytes() for p in destination.iterdir()} != payloads:
                    raise IngestCollisionError("Persisted CSV snapshot differs")
            else:
                os.replace(temporary, destination)
                fsync_directory(destination.parent)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        if snapshot() != initial:
            raise SourceChangedError("CSV set changed while committing snapshot")
        return destination

    def validate(
        self,
        source_folder: str | Path,
        *,
        decode_images: bool = True,
        csv_snapshot: Path | None = None,
    ) -> ValidatedIngest:
        """Validate completeness against a stable source snapshot.

        With decode_images=False, decoding happens on the verified local copy.
        Timing and the decision to proceed remain with the controller.
        """
        source = self._resolve_source(source_folder)
        job_id = self._validate_job_id(source.name)
        csv_files = self._csv_files(csv_snapshot or source)
        csv_names = tuple(p.name for p in csv_files) if csv_snapshot else None
        initial_snapshot = self._snapshot_tree(source, csv_names)
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
                if decode_images:
                    self._validate_image(source / image_key, image_key)
                else:
                    self._validate_image_metadata(source / image_key, image_key)
                image_keys.append(image_key)

        self._assert_snapshot_unchanged(
            source, initial_snapshot, "validation", csv_names
        )
        return ValidatedIngest(
            source_folder=source,
            job_id=job_id,
            expected_image_keys=tuple(image_keys),
            source_snapshot=initial_snapshot,
            images_decoded=decode_images,
            csv_names=csv_names,
        )

    def archive(
        self,
        validated: ValidatedIngest,
        *,
        backup_root: str | Path,
        csv_snapshot: Path | None = None,
        target_folder: Path | None = None,
    ) -> IngestResult:
        """Copy and verify a single raw backup without touching the source."""
        source = validated.source_folder
        backup_target = target_folder or (
            Path(backup_root).expanduser().resolve(strict=False)
            / self._job_date(validated.job_id).isoformat()
            / validated.job_id
        )
        self._validate_target(source, backup_target)
        self._assert_snapshot_unchanged(
            source, validated.source_snapshot, "copy", validated.csv_names
        )
        prepared = None
        committed = False
        try:
            prepared, manifest = self._prepare_streamed_copy(
                source, backup_target, validated.source_snapshot, validated.csv_names
            )
            local_source = prepared.temporary or prepared.target
            if csv_snapshot is not None:
                expected = {
                    p.name: p.read_bytes() for p in self._csv_files(csv_snapshot)
                }
                actual = {p.name: p.read_bytes() for p in self._csv_files(local_source)}
                if expected != actual:
                    raise SourceChangedError(
                        "Copied CSV differs from immutable snapshot"
                    )
            if not validated.images_decoded:
                for image_key in validated.expected_image_keys:
                    self._validate_image(local_source / image_key, image_key)
            self._assert_snapshot_unchanged(
                source, validated.source_snapshot, "copy", validated.csv_names
            )
            before_commit = prepared
            prepared = self._commit_copy(prepared, manifest)
            committed = before_commit.temporary is not None and not prepared.reused
            self._assert_snapshot_unchanged(
                source, validated.source_snapshot, "copy", validated.csv_names
            )
        except SourceChangedError:
            # Roll back only the destination this attempt created, never an
            # existing, byte-identical backup reused during recovery.
            if committed:
                shutil.rmtree(backup_target, ignore_errors=True)
            raise
        finally:
            if prepared is not None and prepared.temporary is not None:
                shutil.rmtree(prepared.temporary, ignore_errors=True)
        return IngestResult(backup_target, prepared.reused)

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
                if is_pending_csv(child) and stat.S_ISREG(child.lstat().st_mode)
            )
        )

    @staticmethod
    def _read_csv(csv_path: Path) -> pd.DataFrame:
        try:
            return CsvRepository().read_for_output(csv_path)
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
    def _validate_image_metadata(image_path: Path, image_key: str) -> None:
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

    @classmethod
    def _validate_image(cls, image_path: Path, image_key: str) -> None:
        cls._validate_image_metadata(image_path, image_key)
        try:
            encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        except (OSError, cv2.error) as exc:
            raise IngestValidationError(
                f"Unable to decode expected image {image_key}: {exc}"
            ) from exc
        if decoded is None or decoded.size == 0:
            raise IngestValidationError(f"Expected image is not decodable: {image_key}")

    @classmethod
    def _snapshot_tree(cls, root: Path, csv_names=None) -> TreeSnapshot:
        try:
            entries: list[FileSnapshotEntry] = []
            for path in cls._walk_tree(root):
                if (
                    csv_names is not None
                    and path.suffix.lower() == ".csv"
                    and path.relative_to(root).as_posix() not in csv_names
                ):
                    continue
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
    def _content_manifest(cls, root: Path) -> ContentManifest:
        entries: list[ContentManifestEntry] = []
        try:
            for path in cls._walk_tree(root):
                metadata = path.lstat()
                relative = path.relative_to(root).as_posix()
                if stat.S_ISDIR(metadata.st_mode):
                    entries.append(ContentManifestEntry(relative, "directory", 0, ""))
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise IngestStorageError(
                        f"Tree contains a non-regular entry: {relative}"
                    )
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
        except IngestStorageError:
            raise
        except OSError as exc:
            raise IngestStorageError(
                f"Unable to verify local ingest target {root}: {exc}"
            ) from exc
        return tuple(sorted(entries, key=lambda item: item.relative_path))

    @classmethod
    def _assert_snapshot_unchanged(
        cls, root: Path, expected: TreeSnapshot, phase: str, csv_names=None
    ) -> None:
        current = cls._snapshot_tree(root, csv_names)
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
    def _prepare_streamed_copy(
        cls, source: Path, target: Path, snapshot: TreeSnapshot, csv_names=None
    ) -> tuple[_PreparedCopy, ContentManifest]:
        """Read source bytes once, hashing during copy and verifying locally."""
        ensure_directory(target.parent)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
        )
        expected = {entry.relative_path: entry for entry in snapshot}
        entries = [
            ContentManifestEntry(entry.relative_path, "directory", 0, "")
            for entry in snapshot
            if entry.kind == "directory"
        ]

        def copy_and_hash(src, dst):
            path = Path(src)
            relative = path.relative_to(source).as_posix()
            entry = expected.get(relative)
            metadata = path.lstat()
            if (
                entry is None
                or entry.kind != "file"
                or not stat.S_ISREG(metadata.st_mode)
                or (metadata.st_size, metadata.st_mtime_ns)
                != (entry.size, entry.mtime_ns)
            ):
                raise SourceChangedError(f"Source changed during copy: {relative}")
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as reader, Path(dst).open("wb") as writer:
                for chunk in iter(lambda: reader.read(_HASH_CHUNK_SIZE), b""):
                    writer.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if size != entry.size:
                raise SourceChangedError(f"Source changed during copy: {relative}")
            shutil.copystat(path, dst)
            entries.append(
                ContentManifestEntry(relative, "file", size, digest.hexdigest())
            )
            return dst

        try:
            shutil.copytree(
                source,
                temporary,
                dirs_exist_ok=True,
                copy_function=copy_and_hash,
                ignore=lambda directory, names: [
                    name
                    for name in names
                    if (Path(directory) / name).relative_to(source).as_posix()
                    not in expected
                ],
            )
            cls._assert_snapshot_unchanged(source, snapshot, "copy", csv_names)
            manifest = tuple(sorted(entries, key=lambda item: item.relative_path))
            if set(expected) != {entry.relative_path for entry in manifest}:
                raise SourceChangedError("Source file set changed during copy")
            if cls._content_manifest(temporary) != manifest:
                raise IngestStorageError("Local copy failed content verification")
            if target.exists():
                if not target.is_dir() or cls._content_manifest(target) != manifest:
                    raise IngestCollisionError(
                        f"Ingest target already exists with different content: {target}"
                    )
                shutil.rmtree(temporary)
                return _PreparedCopy(target, None, True), manifest
            return _PreparedCopy(target, temporary, False), manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @classmethod
    def _commit_copy(
        cls,
        prepared: _PreparedCopy,
        source_manifest: ContentManifest,
    ) -> _PreparedCopy:
        if prepared.temporary is None:
            return prepared
        try:
            for folder in sorted(
                (entry for entry in prepared.temporary.rglob("*") if entry.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                fsync_directory(folder)
            fsync_directory(prepared.temporary)
            prepared.temporary.rename(prepared.target)
            fsync_directory(prepared.target.parent)
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
