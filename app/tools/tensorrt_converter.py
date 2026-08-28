"""Convert Ultralytics YOLO and anomalib PatchCore models to TensorRT.

The module intentionally imports CUDA, TensorRT, Ultralytics, anomalib, ONNX,
and PyTorch lazily.  Production workers can therefore import the application
without pulling converter-only dependencies into their startup path.

TensorRT plans are hardware- and runtime-specific artifacts.  Build them on
the target AIPC (or an identical GPU/software image), then validate them with
the runtime that will serve production inference.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

ModelKind = Literal["yolo", "patchcore"]
Precision = Literal["fp32", "fp16"]
Shape = tuple[int, ...]

_SUPPORTED_TRT_MAJOR = 10
_UNSUPPORTED_TRT_VERSIONS = {(10, 1)}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ConversionError(RuntimeError):
    """Raised when a model cannot be converted without violating its contract."""


@dataclass(frozen=True)
class ShapeProfile:
    """TensorRT optimization-profile shapes for a single NCHW input."""

    minimum: Shape
    optimum: Shape
    maximum: Shape

    def __post_init__(self) -> None:
        ranks = {len(self.minimum), len(self.optimum), len(self.maximum)}
        if len(ranks) != 1:
            raise ValueError("minimum/optimum/maximum shapes must have equal rank")
        if not self.minimum:
            raise ValueError("shape profile cannot be empty")
        for label, shape in (
            ("minimum", self.minimum),
            ("optimum", self.optimum),
            ("maximum", self.maximum),
        ):
            if any(value <= 0 for value in shape):
                raise ValueError(f"{label} shape must contain only positive values")
        for low, opt, high in zip(
            self.minimum, self.optimum, self.maximum, strict=True
        ):
            if not low <= opt <= high:
                raise ValueError(
                    "shape profile must satisfy minimum <= optimum <= maximum"
                )

    @property
    def dynamic(self) -> bool:
        return not (self.minimum == self.optimum == self.maximum)

    @classmethod
    def nchw(
        cls,
        *,
        channels: int,
        height: int,
        width: int,
        minimum_batch: int,
        optimum_batch: int,
        maximum_batch: int,
    ) -> ShapeProfile:
        return cls(
            minimum=(minimum_batch, channels, height, width),
            optimum=(optimum_batch, channels, height, width),
            maximum=(maximum_batch, channels, height, width),
        )


@dataclass(frozen=True)
class TensorInfo:
    """Serializable TensorRT I/O description."""

    name: str
    mode: Literal["input", "output"]
    shape: Shape
    dtype: str
    profile_minimum: Shape | None = None
    profile_optimum: Shape | None = None
    profile_maximum: Shape | None = None


@dataclass(frozen=True)
class EngineInspection:
    """Information read back from a deserialized TensorRT plan."""

    tensors: tuple[TensorInfo, ...]

    @property
    def inputs(self) -> tuple[TensorInfo, ...]:
        return tuple(tensor for tensor in self.tensors if tensor.mode == "input")

    @property
    def outputs(self) -> tuple[TensorInfo, ...]:
        return tuple(tensor for tensor in self.tensors if tensor.mode == "output")


@dataclass(frozen=True)
class BuildResult:
    """Raw TensorRT plan plus its verified runtime inspection."""

    plan: bytes
    inspection: EngineInspection
    tensorrt_version: str
    builder_precision_mode: Precision


@dataclass(frozen=True)
class PatchCoreTransform:
    """Relevant preprocessing recovered from anomalib export metadata."""

    resize: tuple[int, int] | None
    center_crop: tuple[int, int] | None
    mean: tuple[float, float, float] | None
    std: tuple[float, float, float] | None
    unsupported: tuple[str, ...]


def _require_module(name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise ConversionError(
            f"required module '{name}' is unavailable: {type(exc).__name__}: "
            f"{exc}. {install_hint}"
        ) from exc


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _load_tensorrt() -> Any:
    trt = _require_module(
        "tensorrt",
        "Install NVIDIA TensorRT in the AIPC Python 3.12 environment.",
    )
    if not hasattr(trt, "Logger") or not hasattr(trt, "Builder"):
        raise ConversionError(
            "the imported 'tensorrt' is not NVIDIA TensorRT. The repository also "
            "contains a legacy directory named 'tensorrt'; install NVIDIA's "
            "TensorRT Python package in this environment."
        )
    version = str(getattr(trt, "__version__", ""))
    parsed = _version_tuple(version)
    if not parsed:
        raise ConversionError("NVIDIA TensorRT did not report a valid version")
    if parsed[0] != _SUPPORTED_TRT_MAJOR or parsed[:2] in _UNSUPPORTED_TRT_VERSIONS:
        raise ConversionError(
            "this project requires TensorRT 10.x, excluding 10.1.x; "
            f"found {version}. Build and run the engine with the same supported "
            "TensorRT version on the target AIPC."
        )
    return trt


def _parse_metadata_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(stripped)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return value


def load_onnx_metadata(path: Path) -> dict[str, Any]:
    """Read metadata properties without loading large external tensor files."""

    onnx = _require_module(
        "onnx",
        "Install converter dependencies from requirements-tensorrt-export.txt.",
    )
    try:
        try:
            model = onnx.load(str(path), load_external_data=False)
        except TypeError:
            model = onnx.load(str(path))
    except Exception as exc:
        raise ConversionError(
            f"failed to read ONNX metadata from {path}: {exc}"
        ) from exc
    return {
        item.key: _parse_metadata_value(item.value)
        for item in getattr(model, "metadata_props", ())
    }


def _onnx_model_tensors(model: Any) -> tuple[Any, ...]:
    """Collect initializers and attribute tensors from every ONNX subgraph."""

    tensors: list[Any] = []

    def message_field(message: Any, field: str) -> Any | None:
        if not hasattr(message, field):
            return None
        has_field = getattr(message, "HasField", None)
        if callable(has_field):
            try:
                if not bool(has_field(field)):
                    return None
            except (TypeError, ValueError):
                pass
        return getattr(message, field)

    def add_tensor(tensor: Any | None) -> None:
        if tensor is not None:
            tensors.append(tensor)

    def add_sparse(sparse: Any | None) -> None:
        if sparse is None:
            return
        add_tensor(getattr(sparse, "values", None))
        add_tensor(getattr(sparse, "indices", None))

    def visit_attributes(attributes: Any) -> None:
        for attribute in attributes:
            add_tensor(message_field(attribute, "t"))
            for tensor in getattr(attribute, "tensors", ()):
                add_tensor(tensor)
            add_sparse(message_field(attribute, "sparse_tensor"))
            for sparse in getattr(attribute, "sparse_tensors", ()):
                add_sparse(sparse)
            graph = message_field(attribute, "g")
            if graph is not None:
                visit_graph(graph)
            for nested_graph in getattr(attribute, "graphs", ()):
                visit_graph(nested_graph)

    def visit_nodes(nodes: Any) -> None:
        for node in nodes:
            visit_attributes(getattr(node, "attribute", ()))

    def visit_graph(graph: Any) -> None:
        for tensor in getattr(graph, "initializer", ()):
            add_tensor(tensor)
        for sparse in getattr(graph, "sparse_initializer", ()):
            add_sparse(sparse)
        visit_nodes(getattr(graph, "node", ()))

    visit_graph(model.graph)
    for training_info in getattr(model, "training_info", ()):
        for field in ("initialization", "algorithm"):
            graph = message_field(training_info, field)
            if graph is not None:
                visit_graph(graph)
    for function in getattr(model, "functions", ()):
        visit_nodes(getattr(function, "node", ()))
        visit_attributes(getattr(function, "attribute_proto", ()))
    return tuple(tensors)


def _onnx_external_data_files(path: Path) -> tuple[tuple[Path, Path], ...]:
    """Return safe ``(source, relative_location)`` external initializer files."""

    onnx = _require_module(
        "onnx",
        "Install converter dependencies from requirements-tensorrt-export.txt.",
    )
    try:
        try:
            model = onnx.load(str(path), load_external_data=False)
        except TypeError:
            model = onnx.load(str(path))
    except Exception as exc:
        raise ConversionError(f"failed to inspect ONNX external data: {exc}") from exc

    tensors = _onnx_model_tensors(model)
    external_marker = getattr(getattr(onnx, "TensorProto", object), "EXTERNAL", 1)
    source_root = path.parent.resolve()
    locations: dict[Path, Path] = {}
    for tensor in tensors:
        if getattr(tensor, "data_location", None) != external_marker:
            continue
        entries = {
            entry.key: entry.value for entry in getattr(tensor, "external_data", ())
        }
        location = entries.get("location")
        if not location:
            raise ConversionError("ONNX external tensor has no location")
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ConversionError(
                f"unsafe ONNX external-data location is not allowed: {location}"
            )
        source_file = (source_root / relative).resolve()
        try:
            source_file.relative_to(source_root)
        except ValueError as exc:
            raise ConversionError(
                f"ONNX external-data location escapes its model directory: {location}"
            ) from exc
        if not source_file.is_file():
            raise ConversionError(
                f"ONNX external-data file does not exist: {source_file}"
            )
        locations[relative] = source_file
    return tuple((source_file, relative) for relative, source_file in locations.items())


def encode_ultralytics_engine(plan: bytes, metadata: dict[str, Any]) -> bytes:
    """Add the metadata prefix expected by Ultralytics 8.3.187 AutoBackend."""

    try:
        encoded = json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"YOLO metadata is not JSON serializable: {exc}") from exc
    if len(encoded) >= 2**31:
        raise ConversionError("YOLO metadata is unexpectedly large")
    return len(encoded).to_bytes(4, byteorder="little", signed=True) + encoded + plan


def decode_ultralytics_engine(payload: bytes) -> tuple[dict[str, Any], bytes]:
    """Decode an Ultralytics metadata-prefixed engine (used by tests/inspection)."""

    if len(payload) < 5:
        raise ConversionError("Ultralytics engine payload is too short")
    metadata_length = int.from_bytes(payload[:4], byteorder="little", signed=True)
    if metadata_length <= 0 or metadata_length > len(payload) - 4:
        raise ConversionError("invalid Ultralytics engine metadata length")
    try:
        metadata = json.loads(payload[4 : 4 + metadata_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError("invalid Ultralytics engine metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise ConversionError("Ultralytics engine metadata must be a JSON object")
    return metadata, payload[4 + metadata_length :]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ConversionError(f"output already exists (use --force): {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=f"{path.suffix}.tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        if path.exists() and not overwrite:
            raise ConversionError(f"output appeared during build: {path}")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded, overwrite=overwrite)


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=f"{path.suffix}.staged",
    )
    staged = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _stage_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=f"{destination.suffix}.staged",
    )
    staged = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination_file:
            with source.open("rb") as source_file:
                shutil.copyfileobj(
                    source_file,
                    destination_file,
                    length=1024 * 1024,
                )
                destination_file.flush()
                os.fsync(destination_file.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _unused_backup_path(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".rollback",
    )
    os.close(descriptor)
    backup = Path(temporary_name)
    backup.unlink()
    return backup


def _commit_staged_files(
    staged: dict[Path, Path],
    *,
    overwrite: bool,
    artifact_label: str,
) -> None:
    backups: dict[Path, Path] = {}
    published: set[Path] = set()
    try:
        if overwrite:
            for destination in staged:
                if destination.exists():
                    backup = _unused_backup_path(destination)
                    # Register before the syscall so a Ctrl-C delivered after
                    # the rename can still discover and restore this backup.
                    backups[destination] = backup
                    os.replace(destination, backup)
        else:
            for destination in staged:
                if destination.exists():
                    raise ConversionError(
                        f"output appeared during build: {destination}"
                    )

        for destination, staged_path in staged.items():
            if overwrite:
                # Pre-registering is safe: rollback tolerates an absent target
                # when interruption happens before os.replace starts.
                published.add(destination)
                os.replace(staged_path, destination)
            else:
                try:
                    try:
                        os.link(staged_path, destination)
                    finally:
                        # A Python signal may be delivered after the hard-link
                        # syscall succeeds but before its next bytecode. Infer
                        # success by inode so rollback does not miss the file.
                        try:
                            if os.path.samefile(staged_path, destination):
                                published.add(destination)
                        except OSError:
                            pass
                except FileExistsError as exc:
                    raise ConversionError(
                        f"output appeared during build: {destination}"
                    ) from exc
                if destination not in published:
                    raise ConversionError(
                        f"failed to verify published output: {destination}"
                    )
                staged_path.unlink()
    except BaseException as exc:
        # Reconcile non-force hard links from filesystem state. This closes
        # the final bookkeeping window if interruption occurs while checking
        # samefile() immediately after a successful os.link().
        for destination, staged_path in staged.items():
            if destination in backups or destination in published:
                continue
            try:
                staged_stat = staged_path.stat()
                destination_stat = destination.stat()
            except OSError:
                continue
            if (
                staged_stat.st_dev == destination_stat.st_dev
                and staged_stat.st_ino == destination_stat.st_ino
            ):
                published.add(destination)
        rollback_errors: list[str] = []
        for destination in published - backups.keys():
            try:
                destination.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"remove {destination}: {rollback_exc}")
        for destination, backup in backups.items():
            if backup.exists():
                try:
                    os.replace(backup, destination)
                except OSError as rollback_exc:
                    rollback_errors.append(
                        f"restore {destination} from {backup}: {rollback_exc}"
                    )
        for staged_path in staged.values():
            try:
                staged_path.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"remove {staged_path}: {rollback_exc}")
        if rollback_errors:
            raise ConversionError(
                f"{artifact_label} publish failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, ConversionError):
            raise
        raise ConversionError(
            f"failed to publish {artifact_label}: {type(exc).__name__}: {exc}"
        ) from exc
    else:
        for backup in backups.values():
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                # The published set is already consistent. A stale hidden
                # rollback file is safer than undoing a successful release.
                pass
        for staged_path in staged.values():
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                pass


def _publish_engine_and_manifest(
    engine_path: Path,
    engine_payload: bytes,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Publish the engine/manifest pair with exception-safe rollback.

    Two filesystem names cannot be switched in one portable atomic syscall.
    Both files are therefore staged and fsynced first; intentional replacement
    keeps same-directory rollback names until both switches succeed.  A normal
    Python/filesystem error restores the previous pair instead of leaving a new
    engine next to stale provenance.
    """

    if engine_path.resolve() == manifest_path.resolve():
        raise ConversionError("engine and manifest paths must be different")
    if engine_path.parent.resolve() != manifest_path.parent.resolve():
        raise ConversionError("engine and manifest must share one output directory")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        for candidate in (engine_path, manifest_path):
            if candidate.exists():
                raise ConversionError(
                    f"output already exists (use --force): {candidate}"
                )

    try:
        manifest_payload = (
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConversionError(
            f"engine manifest is not JSON serializable: {exc}"
        ) from exc
    staged: dict[Path, Path] = {}
    try:
        staged[engine_path] = _stage_bytes(engine_path, engine_payload)
        staged[manifest_path] = _stage_bytes(manifest_path, manifest_payload)
    except BaseException:
        for staged_path in staged.values():
            staged_path.unlink(missing_ok=True)
        raise
    _commit_staged_files(
        staged,
        overwrite=overwrite,
        artifact_label="engine/manifest pair",
    )


def _normalize_two_dimensions(values: Sequence[int], label: str) -> tuple[int, int]:
    if len(values) == 1:
        height = width = int(values[0])
    elif len(values) == 2:
        height, width = (int(value) for value in values)
    else:
        raise ConversionError(f"{label} accepts one value or HEIGHT WIDTH")
    if height <= 0 or width <= 0:
        raise ConversionError(f"{label} values must be positive")
    return height, width


def _build_profile(
    *,
    channels: int,
    height: int,
    width: int,
    batch: int,
    dynamic: bool,
    minimum_batch: int | None,
    maximum_batch: int | None,
) -> ShapeProfile:
    if batch <= 0:
        raise ConversionError("--batch must be positive")
    if dynamic:
        minimum = 1 if minimum_batch is None else minimum_batch
        maximum = batch if maximum_batch is None else maximum_batch
    else:
        if minimum_batch is not None or maximum_batch is not None:
            raise ConversionError("--min-batch/--max-batch require --dynamic")
        minimum = maximum = batch
    try:
        return ShapeProfile.nchw(
            channels=channels,
            height=height,
            width=width,
            minimum_batch=minimum,
            optimum_batch=batch,
            maximum_batch=maximum,
        )
    except ValueError as exc:
        raise ConversionError(str(exc)) from exc


def _validate_runtime_batch(profile: ShapeProfile, *, allow_nonstandard: bool) -> None:
    runtime_batch = 8
    if allow_nonstandard:
        return
    if not profile.minimum[0] <= runtime_batch <= profile.maximum[0]:
        raise ConversionError(
            "the current SPI model services pad inference to batch 8; the engine "
            "profile must include batch 8. Use --allow-nonstandard-batch only for "
            "a different consumer."
        )
    if not profile.dynamic and profile.optimum[0] != runtime_batch:
        raise ConversionError(
            "the current SPI model services require fixed batch 8; use --batch 8 "
            "or --allow-nonstandard-batch for a different consumer."
        )


def _tensor_shape(tensor: Any) -> Shape:
    return tuple(int(dimension) for dimension in tensor.shape)


def _network_tensors(network: Any) -> tuple[list[Any], list[Any]]:
    inputs = [network.get_input(index) for index in range(network.num_inputs)]
    outputs = [network.get_output(index) for index in range(network.num_outputs)]
    return inputs, outputs


def _validate_network_input(input_tensor: Any, profile: ShapeProfile) -> None:
    source_shape = _tensor_shape(input_tensor)
    if len(source_shape) != len(profile.optimum):
        raise ConversionError(
            f"ONNX input '{input_tensor.name}' has rank {len(source_shape)}, "
            f"but profile rank is {len(profile.optimum)}"
        )
    for index, (source, minimum, optimum, maximum) in enumerate(
        zip(
            source_shape,
            profile.minimum,
            profile.optimum,
            profile.maximum,
            strict=True,
        )
    ):
        if source > 0 and not minimum == optimum == maximum == source:
            raise ConversionError(
                f"ONNX input '{input_tensor.name}' dimension {index} is fixed at "
                f"{source}, but requested profile is {minimum}/{optimum}/{maximum}. "
                "A static batch-1 ONNX cannot safely be changed to batch 8; export "
                "a dynamic ONNX from the source .pt model instead."
            )


def _resolve_patchcore_output_names(
    outputs: Sequence[TensorInfo],
    score_selector: str | None,
) -> tuple[str, str]:
    """Return ``(anomaly_map_source, score_source)`` or fail on ambiguity."""

    if len(outputs) < 2:
        raise ConversionError(
            "PatchCore ONNX must expose both anomaly_map and pred_score outputs"
        )
    by_name = {output.name: output for output in outputs}

    score: TensorInfo | None = None
    if score_selector is not None:
        if score_selector.isdecimal():
            index = int(score_selector)
            if index >= len(outputs):
                raise ConversionError(
                    f"--score-output index {index} is outside the ONNX outputs"
                )
            score = outputs[index]
        else:
            score = by_name.get(score_selector)
            if score is None:
                raise ConversionError(
                    f"--score-output '{score_selector}' is not an ONNX output"
                )
    else:
        aliases = ("pred_score", "pred_scores", "anomaly_score", "anomaly_scores")
        alias_matches = [by_name[name] for name in aliases if name in by_name]
        if len(alias_matches) == 1:
            score = alias_matches[0]
        elif len(alias_matches) > 1:
            raise ConversionError(
                "PatchCore score output is ambiguous; select it with --score-output"
            )
        else:
            low_rank = [output for output in outputs if len(output.shape) <= 2]
            if len(low_rank) == 1:
                score = low_rank[0]

    if score is None:
        raise ConversionError(
            "cannot identify the PatchCore score output; use --score-output NAME"
        )

    map_aliases = ("anomaly_map", "anomaly_maps")
    map_matches = [by_name[name] for name in map_aliases if name in by_name]
    if len(map_matches) == 1:
        anomaly_map = map_matches[0]
    elif len(map_matches) > 1:
        raise ConversionError("PatchCore anomaly-map output is ambiguous")
    else:
        map_candidates = [
            output
            for output in outputs
            if output.name != score.name and len(output.shape) >= 3
        ]
        if len(map_candidates) != 1:
            raise ConversionError(
                "cannot identify the PatchCore anomaly-map output by shape/name"
            )
        anomaly_map = map_candidates[0]

    if anomaly_map.name == score.name:
        raise ConversionError("PatchCore anomaly-map and score outputs must differ")
    return anomaly_map.name, score.name


def _prepare_network_contract(
    network: Any,
    *,
    kind: ModelKind,
    profile: ShapeProfile,
    score_selector: str | None,
) -> None:
    inputs, outputs = _network_tensors(network)
    if len(inputs) != 1:
        raise ConversionError(
            f"{kind} converter supports exactly one ONNX input; found {len(inputs)}"
        )
    input_tensor = inputs[0]
    input_shape = _tensor_shape(input_tensor)
    if len(input_shape) != 4:
        raise ConversionError(
            f"{kind} ONNX input must be NCHW rank 4; found {input_shape}"
        )
    if input_shape[1] not in (-1, 3):
        raise ConversionError(
            f"{kind} ONNX input channel dimension must be 3; found {input_shape[1]}"
        )
    if not _is_supported_runtime_float_dtype(str(input_tensor.dtype)):
        raise ConversionError(
            f"{kind} ONNX input dtype must be float32/float16; "
            f"found {input_tensor.dtype}"
        )
    _validate_network_input(input_tensor, profile)

    if kind == "yolo":
        input_tensor.name = "images"
        if not outputs:
            raise ConversionError("YOLO ONNX has no output tensors")
        return

    input_tensor.name = "input"
    output_infos = [
        TensorInfo(
            name=str(output.name),
            mode="output",
            shape=_tensor_shape(output),
            dtype=str(output.dtype),
        )
        for output in outputs
    ]
    anomaly_map_name, score_name = _resolve_patchcore_output_names(
        output_infos,
        score_selector,
    )
    for output in outputs:
        if output.name == anomaly_map_name:
            output.name = "anomaly_map"
        elif output.name == score_name:
            output.name = "pred_score"


def _parser_error_text(parser: Any) -> str:
    errors = []
    for index in range(int(getattr(parser, "num_errors", 0))):
        errors.append(str(parser.get_error(index)))
    return "\n".join(errors) or "TensorRT ONNX parser did not provide details"


def _set_workspace(config: Any, trt: Any, workspace_gib: float) -> None:
    if not math.isfinite(workspace_gib) or workspace_gib <= 0:
        raise ConversionError("--workspace-gib must be finite and greater than zero")
    try:
        workspace_bytes = int(workspace_gib * (1 << 30))
    except (OverflowError, ValueError) as exc:
        raise ConversionError(
            "--workspace-gib resolves outside the supported byte range"
        ) from exc
    if workspace_bytes < 1 or workspace_bytes > sys.maxsize:
        raise ConversionError(
            "--workspace-gib resolves outside the supported byte range "
            f"1..{sys.maxsize}"
        )
    if hasattr(config, "set_memory_pool_limit"):
        try:
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                workspace_bytes,
            )
            if hasattr(config, "get_memory_pool_limit"):
                configured = int(
                    config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)
                )
                if configured != workspace_bytes:
                    raise ConversionError(
                        "TensorRT did not accept the requested workspace limit: "
                        f"requested {workspace_bytes} bytes, configured {configured}"
                    )
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(
                f"TensorRT rejected the workspace limit: {exc}"
            ) from exc
    elif hasattr(config, "max_workspace_size"):
        try:
            config.max_workspace_size = workspace_bytes
        except Exception as exc:
            raise ConversionError(
                f"TensorRT rejected the workspace limit: {exc}"
            ) from exc
    else:
        raise ConversionError("TensorRT builder config has no workspace API")


def _engine_tensor_info(trt: Any, engine: Any, index: int) -> TensorInfo:
    if hasattr(engine, "num_io_tensors"):
        name = str(engine.get_tensor_name(index))
        shape = tuple(int(value) for value in engine.get_tensor_shape(name))
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        dtype = str(engine.get_tensor_dtype(name))
        profile_shapes: tuple[Shape | None, Shape | None, Shape | None] = (
            None,
            None,
            None,
        )
        if is_input and -1 in shape and hasattr(engine, "get_tensor_profile_shape"):
            values = engine.get_tensor_profile_shape(name, 0)
            profile_shapes = tuple(
                tuple(int(item) for item in value) for value in values
            )  # type: ignore[assignment]
    else:
        name = str(engine.get_binding_name(index))
        shape = tuple(int(value) for value in engine.get_binding_shape(index))
        is_input = bool(engine.binding_is_input(index))
        dtype = str(engine.get_binding_dtype(index))
        profile_shapes = (None, None, None)
        if is_input and -1 in shape and hasattr(engine, "get_profile_shape"):
            values = engine.get_profile_shape(0, index)
            profile_shapes = tuple(
                tuple(int(item) for item in value) for value in values
            )  # type: ignore[assignment]
    return TensorInfo(
        name=name,
        mode="input" if is_input else "output",
        shape=shape,
        dtype=dtype,
        profile_minimum=profile_shapes[0],
        profile_optimum=profile_shapes[1],
        profile_maximum=profile_shapes[2],
    )


def _inspect_deserialized_engine(trt: Any, engine: Any) -> EngineInspection:
    count = (
        int(engine.num_io_tensors)
        if hasattr(engine, "num_io_tensors")
        else int(engine.num_bindings)
    )
    return EngineInspection(
        tensors=tuple(_engine_tensor_info(trt, engine, index) for index in range(count))
    )


def _is_supported_runtime_float_dtype(dtype: str) -> bool:
    normalized = dtype.strip().lower().replace(" ", "")
    return normalized in {
        "datatype.float",
        "float",
        "float32",
        "fp32",
        "datatype.half",
        "half",
        "float16",
        "fp16",
    }


def _validate_patchcore_output_contract(
    inspection: EngineInspection,
) -> None:
    outputs = {tensor.name: tensor for tensor in inspection.outputs}
    anomaly_map = outputs.get("anomaly_map")
    score = outputs.get("pred_score")
    if anomaly_map is None or score is None:
        raise ConversionError(
            "verified PatchCore engine must contain anomaly_map and pred_score outputs"
        )
    if len(anomaly_map.shape) != 4 or anomaly_map.shape[1] != 1:
        raise ConversionError(
            "PatchCore anomaly_map must have batch-first shape [B,1,H,W]"
        )
    if anomaly_map.shape[2] <= 0 or anomaly_map.shape[3] <= 0:
        raise ConversionError(
            "current PatchCore runtime requires fixed anomaly-map C/H/W dimensions"
        )
    if len(score.shape) not in (1, 2) or (
        len(score.shape) == 2 and score.shape[1] != 1
    ):
        raise ConversionError("PatchCore pred_score must have shape [B] or [B,1]")
    if not _is_supported_runtime_float_dtype(
        anomaly_map.dtype
    ) or not _is_supported_runtime_float_dtype(score.dtype):
        raise ConversionError("PatchCore outputs must use floating-point dtypes")

    expected_batch = inspection.inputs[0].shape[0]
    if anomaly_map.shape[0] != expected_batch:
        raise ConversionError(
            "PatchCore anomaly_map does not preserve the engine batch dimension"
        )
    if score.shape[0] != expected_batch:
        raise ConversionError(
            "PatchCore pred_score does not preserve the engine batch dimension"
        )


def _validate_engine_contract(
    inspection: EngineInspection,
    *,
    kind: ModelKind,
    profile: ShapeProfile,
) -> None:
    if len(inspection.inputs) != 1:
        raise ConversionError(
            f"verified TensorRT engine has {len(inspection.inputs)} inputs; expected 1"
        )
    expected_input = "images" if kind == "yolo" else "input"
    if inspection.inputs[0].name != expected_input:
        raise ConversionError(
            f"verified {kind} engine input is '{inspection.inputs[0].name}', "
            f"expected '{expected_input}'"
        )
    if not _is_supported_runtime_float_dtype(inspection.inputs[0].dtype):
        raise ConversionError(
            f"verified {kind} engine input must be float32/float16; "
            f"found {inspection.inputs[0].dtype}"
        )
    output_names = {output.name for output in inspection.outputs}
    if kind == "patchcore":
        if any(dimension <= 0 for dimension in inspection.inputs[0].shape[1:]):
            raise ConversionError(
                "current PatchCore runtime supports a dynamic batch only; input "
                "C/H/W dimensions must be fixed"
            )
        _validate_patchcore_output_contract(inspection)
    elif not output_names:
        raise ConversionError("verified YOLO engine has no outputs")
    input_info = inspection.inputs[0]
    if -1 in input_info.shape:
        if input_info.profile_minimum != profile.minimum:
            raise ConversionError("verified TensorRT minimum profile does not match")
        if input_info.profile_optimum != profile.optimum:
            raise ConversionError("verified TensorRT optimum profile does not match")
        if input_info.profile_maximum != profile.maximum:
            raise ConversionError("verified TensorRT maximum profile does not match")
    elif input_info.shape != profile.optimum:
        raise ConversionError(
            f"verified TensorRT input shape {input_info.shape} does not match "
            f"requested {profile.optimum}"
        )


def build_tensorrt_plan(
    onnx_path: Path,
    *,
    kind: ModelKind,
    precision: Precision,
    workspace_gib: float,
    profile: ShapeProfile,
    score_selector: str | None = None,
    verbose: bool = False,
    allow_fp32_fallback: bool = False,
) -> BuildResult:
    """Parse ONNX, build a raw TensorRT plan, deserialize it, and verify I/O."""

    if not onnx_path.is_file():
        raise ConversionError(f"ONNX file does not exist: {onnx_path}")
    trt = _load_tensorrt()
    severity = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    _set_workspace(config, trt, workspace_gib)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    if hasattr(parser, "parse_from_file"):
        parsed = bool(parser.parse_from_file(str(onnx_path)))
    else:
        parsed = bool(parser.parse(onnx_path.read_bytes()))
    if not parsed:
        raise ConversionError(
            f"TensorRT failed to parse ONNX {onnx_path}:\n{_parser_error_text(parser)}"
        )

    _prepare_network_contract(
        network,
        kind=kind,
        profile=profile,
        score_selector=score_selector,
    )
    inputs, _ = _network_tensors(network)
    if any(dimension == -1 for dimension in _tensor_shape(inputs[0])):
        optimization_profile = builder.create_optimization_profile()
        accepted = optimization_profile.set_shape(
            inputs[0].name,
            profile.minimum,
            profile.optimum,
            profile.maximum,
        )
        if accepted is False:
            raise ConversionError(
                "TensorRT rejected the requested optimization profile"
            )
        profile_index = config.add_optimization_profile(optimization_profile)
        if isinstance(profile_index, int) and profile_index < 0:
            raise ConversionError("TensorRT rejected the optimization profile")

    builder_precision_mode: Precision = precision
    if precision == "fp16":
        if not bool(builder.platform_has_fast_fp16):
            if not allow_fp32_fallback:
                raise ConversionError(
                    "GPU does not report fast FP16 support; use --precision fp32 or "
                    "explicitly pass --allow-fp32-fallback"
                )
            builder_precision_mode = "fp32"
        else:
            config.set_flag(trt.BuilderFlag.FP16)
    if (
        builder_precision_mode == "fp32"
        and hasattr(
            getattr(trt, "BuilderFlag", object),
            "TF32",
        )
        and hasattr(config, "clear_flag")
    ):
        config.clear_flag(trt.BuilderFlag.TF32)

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise ConversionError("TensorRT engine build returned no serialized plan")
        plan = bytes(serialized)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise ConversionError("TensorRT engine build returned no engine")
        plan = bytes(engine.serialize())
    if not plan:
        raise ConversionError("TensorRT produced an empty engine plan")

    runtime = trt.Runtime(logger)
    verified_engine = runtime.deserialize_cuda_engine(plan)
    if verified_engine is None:
        raise ConversionError("TensorRT could not deserialize the engine it just built")
    inspection = _inspect_deserialized_engine(trt, verified_engine)
    _validate_engine_contract(
        inspection,
        kind=kind,
        profile=profile,
    )
    return BuildResult(
        plan=plan,
        inspection=inspection,
        tensorrt_version=str(trt.__version__),
        builder_precision_mode=builder_precision_mode,
    )


def _normalize_yolo_names(value: Any) -> dict[int, str]:
    if isinstance(value, (list, tuple)):
        items = list(enumerate(value))
    elif isinstance(value, dict):
        items = []
        for key, name in value.items():
            if isinstance(key, bool):
                raise ConversionError("YOLO class-name indices must be integers")
            if isinstance(key, int):
                index = key
            elif isinstance(key, str) and key.strip().isdigit():
                index = int(key.strip())
            else:
                raise ConversionError(f"YOLO class-name index is invalid: {key!r}")
            items.append((index, name))
    else:
        raise ConversionError("YOLO class-name metadata must be a list or mapping")
    if not items:
        raise ConversionError("YOLO class-name metadata cannot be empty")
    normalized: dict[int, str] = {}
    for index, name in items:
        if not isinstance(name, str) or not name.strip():
            raise ConversionError(
                f"YOLO class name at index {index} must be a non-empty string"
            )
        if index in normalized:
            raise ConversionError(f"YOLO class-name index is duplicated: {index}")
        normalized[index] = name
    expected = list(range(len(normalized)))
    if sorted(normalized) != expected:
        raise ConversionError(
            "YOLO class-name indices must be contiguous and start at zero"
        )
    return {index: normalized[index] for index in expected}


def _yolo_metadata_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConversionError(f"YOLO {field} metadata must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ConversionError(f"YOLO {field} metadata must be an integer")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    raise ConversionError(f"YOLO {field} metadata must be an integer")


def _prepare_yolo_metadata(
    metadata: dict[str, Any],
    *,
    task: str,
    class_names: Sequence[str] | None,
    stride: int,
    batch: int,
    image_size: tuple[int, int],
    dynamic: bool,
    nms: bool | None,
) -> dict[str, Any]:
    prepared = dict(metadata)
    detected_task = prepared.get("task")
    supported_tasks = {"detect", "segment", "pose", "classify", "obb"}
    if task == "auto":
        if not isinstance(detected_task, str) or detected_task not in supported_tasks:
            raise ConversionError(
                "YOLO ONNX task metadata is missing or unsupported; pass --task "
                "explicitly"
            )
    else:
        if task not in supported_tasks:
            raise ConversionError(f"unsupported YOLO task: {task}")
        prepared["task"] = task

    if class_names:
        prepared["names"] = {index: name for index, name in enumerate(class_names)}
    if "names" not in prepared:
        raise ConversionError(
            "YOLO ONNX has no class-name metadata; pass --class-names NAME [NAME ...]"
        )
    prepared["names"] = _normalize_yolo_names(prepared["names"])

    prepared_stride = _yolo_metadata_integer(
        prepared["stride"]
        if "stride" in prepared and prepared["stride"] is not None
        else stride,
        "stride",
    )
    channels = _yolo_metadata_integer(
        prepared["channels"]
        if "channels" in prepared and prepared["channels"] is not None
        else 3,
        "channels",
    )
    if prepared_stride <= 0:
        raise ConversionError("YOLO stride metadata must be positive")
    if channels != 3:
        raise ConversionError(
            "current YOLO runtime requires three input channels; "
            f"metadata has {channels}"
        )
    prepared["stride"] = prepared_stride
    prepared["batch"] = batch
    prepared["imgsz"] = [image_size[0], image_size[1]]
    prepared["channels"] = channels
    export_args = prepared.get("args")
    if not isinstance(export_args, dict):
        export_args = {}
    export_args = dict(export_args)
    export_args["dynamic"] = dynamic
    if nms is not None:
        if "nms" in export_args:
            if not isinstance(export_args["nms"], bool):
                raise ConversionError("YOLO args.nms metadata must be boolean")
            if export_args["nms"] != nms:
                raise ConversionError(
                    "requested YOLO output contract conflicts with ONNX metadata"
                )
        export_args["nms"] = nms
    elif "nms" not in export_args:
        raise ConversionError(
            "YOLO ONNX metadata does not identify raw versus embedded-NMS output; "
            "pass --onnx-output-contract raw or end2end"
        )
    else:
        if not isinstance(export_args["nms"], bool):
            raise ConversionError("YOLO args.nms metadata must be boolean")
    prepared["args"] = export_args
    try:
        # JSON round-tripping normalizes integer class-name keys before the
        # exact metadata is prefixed to the engine and later compared/read.
        return json.loads(json.dumps(prepared, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"YOLO metadata is not JSON serializable: {exc}") from exc


def _export_yolo_pt_to_onnx(
    source: Path,
    target: Path,
    *,
    task: str,
    image_size: tuple[int, int],
    batch: int,
    dynamic: bool,
    device: str,
    opset: int,
    simplify: bool,
    nms: bool,
) -> None:
    ultralytics = _require_module(
        "ultralytics",
        "Install the project CUDA dependencies before exporting YOLO.",
    )
    version = str(getattr(ultralytics, "__version__", ""))
    if version != "8.3.187":
        raise ConversionError(
            "YOLO conversion is validated against ultralytics==8.3.187; "
            f"found {version or 'unknown'}"
        )
    torch = _require_module(
        "torch",
        "Install torch==2.7.1 with the CUDA 12.8 wheel.",
    )
    if not bool(torch.cuda.is_available()):
        raise ConversionError("YOLO TensorRT conversion requires CUDA on the AIPC")

    target.parent.mkdir(parents=True, exist_ok=True)
    copied_source = target.parent / source.name
    shutil.copy2(source, copied_source)
    model_task = None if task == "auto" else task
    model = ultralytics.YOLO(str(copied_source), task=model_task)
    try:
        exported = model.export(
            format="onnx",
            imgsz=list(image_size),
            batch=batch,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            device=device,
            half=False,
            nms=nms,
        )
    except Exception as exc:
        raise ConversionError(f"Ultralytics failed to export YOLO ONNX: {exc}") from exc
    exported_path = Path(str(exported))
    if not exported_path.is_file() or exported_path.stat().st_size == 0:
        raise ConversionError(
            f"Ultralytics did not produce a usable ONNX file: {exported_path}"
        )
    if exported_path.resolve() != target.resolve():
        shutil.move(str(exported_path), target)


def _torch_load_checkpoint(torch: Any, source: Path, *, trust_pickle: bool) -> Any:
    try:
        return torch.jit.load(str(source), map_location="cpu")
    except Exception:
        pass

    try:
        return torch.load(str(source), map_location="cpu", weights_only=True)
    except Exception as safe_error:
        if not trust_pickle:
            raise ConversionError(
                "PatchCore .pt is not a weights-only/TorchScript artifact. Loading "
                "a serialized anomalib module uses Python pickle and can execute "
                "code. Re-run with --trust-pickle only for a trusted model file."
            ) from safe_error
    try:
        return torch.load(str(source), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(source), map_location="cpu")
    except Exception as exc:
        raise ConversionError(f"failed to load PatchCore checkpoint: {exc}") from exc


def _looks_like_state_dict(value: Any, torch: Any) -> bool:
    return (
        bool(value)
        and isinstance(value, dict)
        and all(
            isinstance(key, str) and isinstance(tensor, torch.Tensor)
            for key, tensor in value.items()
        )
    )


def _instantiate_patchcore_from_state_dict(
    torch: Any,
    state_dict: dict[str, Any],
    *,
    input_size: tuple[int, int],
    backbone: str | None,
    layers: Sequence[str] | None,
    num_neighbors: int,
) -> Any:
    if not backbone or not layers:
        raise ConversionError(
            "a PatchCore state_dict cannot describe its architecture; pass both "
            "--backbone and --layers from the training configuration"
        )
    try:
        anomalib_models = importlib.import_module("anomalib.models")
        patchcore_class = anomalib_models.Patchcore
    except Exception as exc:
        raise ConversionError(
            "failed to import anomalib.models.Patchcore. Use the same anomalib/timm/"
            f"torch environment as training: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        lightning_model = patchcore_class(
            input_size=input_size,
            backbone=backbone,
            layers=list(layers),
            pre_trained=False,
            coreset_sampling_ratio=0.1,
            num_neighbors=num_neighbors,
        )
    except Exception as exc:
        raise ConversionError(
            f"failed to reconstruct PatchCore architecture: {exc}"
        ) from exc

    try:
        if any(key.startswith("model.") for key in state_dict):
            lightning_model.load_state_dict(state_dict, strict=True)
            model = lightning_model.model
        else:
            model = lightning_model.model
            model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ConversionError(
            "PatchCore checkpoint does not strictly match the supplied architecture "
            f"(input_size={input_size}, backbone={backbone}, layers={list(layers)}): "
            f"{exc}"
        ) from exc
    if not isinstance(model, torch.nn.Module):
        raise ConversionError("reconstructed PatchCore model is not a torch module")
    return model


def _load_patchcore_model(
    source: Path,
    *,
    trust_pickle: bool,
    model_input_size: tuple[int, int],
    backbone: str | None,
    layers: Sequence[str] | None,
    num_neighbors: int,
) -> tuple[Any, dict[str, Any]]:
    torch = _require_module(
        "torch",
        "Install torch==2.7.1 in the converter environment.",
    )
    artifact = _torch_load_checkpoint(torch, source, trust_pickle=trust_pickle)
    metadata: dict[str, Any] = {}
    if isinstance(artifact, torch.nn.Module):
        model = artifact
    elif isinstance(artifact, dict) and isinstance(
        artifact.get("model"), torch.nn.Module
    ):
        model = artifact["model"]
        if isinstance(artifact.get("metadata"), dict):
            metadata = dict(artifact["metadata"])
    else:
        if isinstance(artifact, dict) and _looks_like_state_dict(
            artifact.get("state_dict"), torch
        ):
            state_dict = artifact["state_dict"]
        elif _looks_like_state_dict(artifact, torch):
            state_dict = artifact
        else:
            raise ConversionError(
                "unsupported PatchCore .pt layout. Expected TorchScript, anomalib "
                "export {'model': module}, Lightning {'state_dict': ...}, or a raw "
                "state_dict. For a newer anomalib artifact, first export ONNX in "
                "the original training environment."
            )
        model = _instantiate_patchcore_from_state_dict(
            torch,
            state_dict,
            input_size=model_input_size,
            backbone=backbone,
            layers=layers,
            num_neighbors=num_neighbors,
        )

    underlying = getattr(model, "model", None)
    if isinstance(underlying, torch.nn.Module) and not hasattr(model, "memory_bank"):
        model = underlying
    memory_bank = getattr(model, "memory_bank", None)
    if memory_bank is None or not isinstance(memory_bank, torch.Tensor):
        raise ConversionError("PatchCore model has no tensor memory_bank")
    if memory_bank.numel() == 0:
        raise ConversionError(
            "PatchCore memory_bank is empty; export a fitted/trained checkpoint"
        )
    existing_input_size = getattr(model, "input_size", None)
    if existing_input_size is not None:
        existing = tuple(int(value) for value in existing_input_size)
        if existing != model_input_size:
            raise ConversionError(
                f"PatchCore model input_size is {existing}, but requested model "
                f"input size is {model_input_size}"
            )
    model.eval()
    return model, metadata


def _patchcore_output_mode(output: Any, torch: Any) -> tuple[str, Any, Any]:
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        anomaly_map, score = output[0], output[1]
        mode = "sequence"
    elif isinstance(output, dict):
        map_key = next(
            (key for key in ("anomaly_map", "anomaly_maps") if key in output),
            None,
        )
        score_key = next(
            (
                key
                for key in (
                    "pred_score",
                    "pred_scores",
                    "anomaly_score",
                    "anomaly_scores",
                )
                if key in output
            ),
            None,
        )
        if map_key is None or score_key is None:
            raise ConversionError(
                "PatchCore dict output lacks anomaly-map/score fields"
            )
        anomaly_map, score = output[map_key], output[score_key]
        mode = f"dict:{map_key}:{score_key}"
    else:
        anomaly_map = next(
            (
                getattr(output, name)
                for name in ("anomaly_map", "anomaly_maps")
                if hasattr(output, name)
            ),
            None,
        )
        score = next(
            (
                getattr(output, name)
                for name in (
                    "pred_score",
                    "pred_scores",
                    "anomaly_score",
                    "anomaly_scores",
                )
                if hasattr(output, name)
            ),
            None,
        )
        mode = "attributes"
    if not isinstance(anomaly_map, torch.Tensor) or not isinstance(score, torch.Tensor):
        raise ConversionError(
            "PatchCore eval forward must return tensor anomaly_map and pred_score"
        )
    return mode, anomaly_map, score


def _make_patchcore_export_wrapper(
    torch: Any,
    model: Any,
    *,
    engine_input_size: tuple[int, int],
    model_input_size: tuple[int, int],
    center_crop: tuple[int, int] | None,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
    device: str,
) -> Any:
    functional = torch.nn.functional

    class PatchCoreExportWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            if mean is None or std is None:
                self.register_buffer("mean", torch.empty(0), persistent=False)
                self.register_buffer("std", torch.empty(0), persistent=False)
            else:
                self.register_buffer(
                    "mean",
                    torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1),
                    persistent=False,
                )
                self.register_buffer(
                    "std",
                    torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1),
                    persistent=False,
                )
            self.output_mode = ""

        def _preprocess(self, value: Any) -> Any:
            if center_crop is not None:
                crop_height, crop_width = center_crop
                source_height, source_width = engine_input_size
                top = (source_height - crop_height) // 2
                left = (source_width - crop_width) // 2
                value = value[
                    ...,
                    top : top + crop_height,
                    left : left + crop_width,
                ]
            elif engine_input_size != model_input_size:
                value = functional.interpolate(
                    value,
                    size=model_input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if self.mean.numel():
                value = (value - self.mean) / self.std
            return value

        def forward(self, value: Any) -> tuple[Any, Any]:
            output = self.model(self._preprocess(value))
            if self.output_mode == "sequence":
                return output[0], output[1]
            if self.output_mode.startswith("dict:"):
                _, map_key, score_key = self.output_mode.split(":", 2)
                return output[map_key], output[score_key]
            anomaly_map = getattr(
                output,
                "anomaly_map",
                getattr(output, "anomaly_maps", None),
            )
            score = getattr(
                output,
                "pred_score",
                getattr(
                    output,
                    "pred_scores",
                    getattr(
                        output,
                        "anomaly_score",
                        getattr(output, "anomaly_scores", None),
                    ),
                ),
            )
            return anomaly_map, score

    wrapper = PatchCoreExportWrapper().to(device).eval()
    dummy = torch.zeros((1, 3, *engine_input_size), dtype=torch.float32, device=device)
    with torch.no_grad():
        preprocessed = wrapper._preprocess(dummy)
        output = wrapper.model(preprocessed)
    mode, anomaly_map, score = _patchcore_output_mode(output, torch)
    if anomaly_map.shape[0] != 1 or score.shape[0] != 1:
        raise ConversionError(
            "PatchCore outputs do not preserve the input batch dimension"
        )
    wrapper.output_mode = mode
    return wrapper


def _patchcore_preprocessing(
    mode: str,
    custom_mean: Sequence[float] | None,
    custom_std: Sequence[float] | None,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    if mode == "none":
        if custom_mean is not None or custom_std is not None:
            raise ConversionError("--mean/--std require --preprocess custom")
        return None, None
    if mode == "imagenet":
        if custom_mean is not None or custom_std is not None:
            raise ConversionError("--mean/--std require --preprocess custom")
        return _IMAGENET_MEAN, _IMAGENET_STD
    if mode != "custom":
        raise ConversionError(f"unsupported PatchCore preprocessing mode: {mode}")
    if custom_mean is None or custom_std is None:
        raise ConversionError("custom preprocessing requires both --mean and --std")
    if len(custom_mean) != 3 or len(custom_std) != 3:
        raise ConversionError("--mean and --std each require exactly three values")
    mean = tuple(float(value) for value in custom_mean)
    std = tuple(float(value) for value in custom_std)
    if not all(math.isfinite(value) for value in (*mean, *std)):
        raise ConversionError("--mean and --std values must be finite")
    if any(value <= 0 for value in std):
        raise ConversionError("all --std values must be positive")
    return mean, std  # type: ignore[return-value]


def _metadata_transform_size(
    transform: dict[str, Any],
    transform_name: str,
) -> tuple[int, int]:
    try:
        height = int(transform["height"])
        width = int(transform["width"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ConversionError(
            f"PatchCore artifact {transform_name} has invalid height/width metadata"
        ) from exc
    if height <= 0 or width <= 0:
        raise ConversionError(
            f"PatchCore artifact {transform_name} height/width must be positive"
        )
    return height, width


def _metadata_transform_triplet(
    transform: dict[str, Any],
    field: str,
) -> tuple[float, float, float]:
    values = transform.get(field)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ConversionError(
            f"PatchCore artifact Normalize.{field} must contain three values"
        )
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConversionError(
            f"PatchCore artifact Normalize.{field} contains invalid values"
        ) from exc
    if not all(math.isfinite(value) for value in result):
        raise ConversionError(
            f"PatchCore artifact Normalize.{field} values must be finite"
        )
    return result  # type: ignore[return-value]


def _patchcore_transform_from_metadata(
    metadata: dict[str, Any],
) -> PatchCoreTransform | None:
    """Recover the eval transform serialized by anomalib 0.7.

    A canonical anomalib Torch export treats ``metadata['transform']`` as the
    inference contract.  Only operations that can be represented by the
    current SPI runtime/export wrapper are accepted; silently ignoring another
    evaluation transform would change anomaly scores.
    """

    serialized = metadata.get("transform")
    if serialized is None:
        return None
    if not isinstance(serialized, (dict, list, tuple)):
        raise ConversionError("PatchCore artifact transform metadata is malformed")

    transforms: list[tuple[str, dict[str, Any]]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            class_name = value.get("__class_fullname__")
            if isinstance(class_name, str) and class_name:
                transforms.append((class_name.rsplit(".", 1)[-1], value))
            for nested in value.values():
                if isinstance(nested, (dict, list, tuple)):
                    walk(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                walk(nested)

    walk(serialized)
    if not transforms:
        raise ConversionError(
            "PatchCore artifact transform metadata contains no serialized transforms"
        )

    allowed_containers = {"Compose", "ToTensorV2"}
    supported_operations = {"Resize", "CenterCrop", "Normalize", "ToFloat"}
    unsupported = [
        name
        for name, _ in transforms
        if name not in allowed_containers | supported_operations
    ]
    for name, transform in transforms:
        if name not in allowed_containers | supported_operations:
            continue
        try:
            probability = float(transform.get("p", 1.0))
        except (TypeError, ValueError, OverflowError):
            unsupported.append(f"{name}(invalid p)")
            continue
        always_apply = transform.get("always_apply", False) is True
        if not always_apply and (
            not math.isfinite(probability) or not math.isclose(probability, 1.0)
        ):
            unsupported.append(f"{name}(p={probability})")

    def exactly_one(name: str) -> dict[str, Any] | None:
        matches = [value for found, value in transforms if found == name]
        if len(matches) > 1:
            raise ConversionError(
                f"PatchCore artifact contains multiple {name} transforms"
            )
        return matches[0] if matches else None

    resize_node = exactly_one("Resize")
    crop_node = exactly_one("CenterCrop")
    normalize_node = exactly_one("Normalize")
    to_float_node = exactly_one("ToFloat")
    resize = _metadata_transform_size(resize_node, "Resize") if resize_node else None
    if resize_node is not None:
        try:
            interpolation = int(resize_node.get("interpolation", 1))
        except (TypeError, ValueError, OverflowError):
            unsupported.append("Resize(invalid interpolation)")
        else:
            if interpolation != 1:
                unsupported.append(f"Resize(interpolation={interpolation})")
    center_crop = (
        _metadata_transform_size(crop_node, "CenterCrop") if crop_node else None
    )
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    if normalize_node is not None:
        mean = _metadata_transform_triplet(normalize_node, "mean")
        std = _metadata_transform_triplet(normalize_node, "std")
        if any(value <= 0 for value in std):
            raise ConversionError(
                "PatchCore artifact Normalize.std values must be positive"
            )
        try:
            max_pixel_value = float(normalize_node.get("max_pixel_value", 255.0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConversionError(
                "PatchCore artifact Normalize.max_pixel_value is invalid"
            ) from exc
        if not math.isfinite(max_pixel_value) or not math.isclose(
            max_pixel_value,
            255.0,
        ):
            unsupported.append(f"Normalize(max_pixel_value={max_pixel_value})")
    if to_float_node is not None:
        try:
            max_value = float(to_float_node.get("max_value", 255.0))
        except (TypeError, ValueError, OverflowError):
            unsupported.append("ToFloat(invalid max_value)")
        else:
            if not math.isfinite(max_value) or not math.isclose(max_value, 255.0):
                unsupported.append(f"ToFloat(max_value={max_value})")
    if normalize_node is not None and to_float_node is not None:
        unsupported.append("Normalize+ToFloat")
    elif normalize_node is None and to_float_node is None:
        unsupported.append("missing Normalize/ToFloat")

    return PatchCoreTransform(
        resize=resize,
        center_crop=center_crop,
        mean=mean,
        std=std,
        unsupported=tuple(unsupported),
    )


def _validate_patchcore_artifact_transform(
    metadata: dict[str, Any],
    *,
    engine_input_size: tuple[int, int],
    model_input_size: tuple[int, int],
    center_crop: tuple[int, int] | None,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
    ignore_artifact_transform: bool,
) -> PatchCoreTransform | None:
    if "transform" not in metadata:
        return None
    if ignore_artifact_transform:
        return None

    transform = _patchcore_transform_from_metadata(metadata)
    if transform is None:
        return None
    if transform.unsupported:
        raise ConversionError(
            "PatchCore artifact contains unsupported evaluation transforms: "
            + ", ".join(transform.unsupported)
            + ". Use --ignore-artifact-transform only after independently "
            "validating preprocessing parity."
        )
    if transform.resize is not None and transform.resize != engine_input_size:
        raise ConversionError(
            f"PatchCore artifact Resize is {transform.resize}, but --input-size "
            f"is {engine_input_size}"
        )
    if transform.center_crop != center_crop:
        raise ConversionError(
            f"PatchCore artifact CenterCrop is {transform.center_crop}, but the "
            f"requested --center-crop is {center_crop}"
        )
    artifact_model_size = transform.center_crop or transform.resize or engine_input_size
    if artifact_model_size != model_input_size:
        raise ConversionError(
            f"PatchCore artifact preprocessing produces {artifact_model_size}, "
            f"but --model-input-size is {model_input_size}"
        )

    def triplets_match(
        left: tuple[float, float, float] | None,
        right: tuple[float, float, float] | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        return all(
            math.isclose(left_value, right_value, rel_tol=1e-7, abs_tol=1e-9)
            for left_value, right_value in zip(left, right, strict=True)
        )

    if not triplets_match(transform.mean, mean) or not triplets_match(
        transform.std,
        std,
    ):
        raise ConversionError(
            "PatchCore artifact Normalize mean/std does not match --preprocess; "
            "use the training transform values or explicitly pass "
            "--ignore-artifact-transform after a parity test"
        )
    return transform


def _export_patchcore_pt_to_onnx(
    source: Path,
    target: Path,
    *,
    trust_pickle: bool,
    engine_input_size: tuple[int, int],
    model_input_size: tuple[int, int],
    center_crop: tuple[int, int] | None,
    preprocess: str,
    custom_mean: Sequence[float] | None,
    custom_std: Sequence[float] | None,
    backbone: str | None,
    layers: Sequence[str] | None,
    num_neighbors: int,
    dynamic: bool,
    opset: int,
    export_device: str,
    ignore_artifact_transform: bool,
) -> dict[str, Any]:
    torch = _require_module(
        "torch",
        "Install torch==2.7.1 and converter dependencies.",
    )
    if export_device.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise ConversionError(
            f"requested --export-device {export_device}, but CUDA is unavailable"
        )
    mean, std = _patchcore_preprocessing(preprocess, custom_mean, custom_std)
    model, artifact_metadata = _load_patchcore_model(
        source,
        trust_pickle=trust_pickle,
        model_input_size=model_input_size,
        backbone=backbone,
        layers=layers,
        num_neighbors=num_neighbors,
    )
    validated_transform = _validate_patchcore_artifact_transform(
        artifact_metadata,
        engine_input_size=engine_input_size,
        model_input_size=model_input_size,
        center_crop=center_crop,
        mean=mean,
        std=std,
        ignore_artifact_transform=ignore_artifact_transform,
    )
    artifact_metadata = dict(artifact_metadata)
    artifact_metadata["converter_transform_validation"] = {
        "artifact_transform_present": "transform" in artifact_metadata,
        "ignored_by_operator": bool(
            ignore_artifact_transform and "transform" in artifact_metadata
        ),
        "validated_contract": (
            asdict(validated_transform) if validated_transform is not None else None
        ),
    }
    model = model.to(export_device).eval()
    wrapper = _make_patchcore_export_wrapper(
        torch,
        model,
        engine_input_size=engine_input_size,
        model_input_size=model_input_size,
        center_crop=center_crop,
        mean=mean,
        std=std,
        device=export_device,
    )
    dummy = torch.zeros(
        (1, 3, *engine_input_size),
        dtype=torch.float32,
        device=export_device,
    )
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch"},
            "anomaly_map": {0: "batch"},
            "pred_score": {0: "batch"},
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["anomaly_map", "pred_score"],
        "dynamic_axes": dynamic_axes,
    }
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(target),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise ConversionError(
                f"PyTorch failed to export PatchCore ONNX: {exc}"
            ) from exc
        try:
            torch.onnx.export(wrapper, dummy, str(target), **export_kwargs)
        except Exception as fallback_exc:
            raise ConversionError(
                f"PyTorch failed to export PatchCore ONNX: {fallback_exc}"
            ) from fallback_exc
    except Exception as exc:
        raise ConversionError(
            f"PyTorch failed to export PatchCore ONNX: {exc}"
        ) from exc
    if not target.is_file() or target.stat().st_size == 0:
        raise ConversionError("PyTorch did not produce a usable PatchCore ONNX file")
    return artifact_metadata


def _environment_manifest(build: BuildResult) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "tensorrt": build.tensorrt_version,
    }
    for module_name in ("torch", "ultralytics", "anomalib", "onnx"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        environment[module_name] = str(getattr(module, "__version__", "unknown"))
    try:
        torch = importlib.import_module("torch")
        environment["cuda"] = str(getattr(torch.version, "cuda", None))
        if torch.cuda.is_available():
            environment["gpu"] = torch.cuda.get_device_name(0)
            environment["compute_capability"] = list(
                torch.cuda.get_device_capability(0)
            )
    except Exception:
        pass
    return environment


def _manifest_payload(
    *,
    kind: ModelKind,
    source: Path,
    output: Path,
    profile: ShapeProfile,
    requested_precision: Precision,
    build: BuildResult,
    workspace_gib: float,
    engine_format: str,
    preprocessing: dict[str, Any],
    onnx_metadata: dict[str, Any] | None,
    engine_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "kind": kind,
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "engine": str(output.resolve()),
        "engine_sha256": engine_sha256 or _sha256(output),
        "engine_format": engine_format,
        "requested_precision": requested_precision,
        "builder_precision_mode": build.builder_precision_mode,
        "precision_note": (
            "TensorRT builder mode; individual layers may use a supported "
            "higher-precision fallback"
        ),
        "workspace_gib": workspace_gib,
        "profile": asdict(profile),
        "io_tensors": [asdict(tensor) for tensor in build.inspection.tensors],
        "preprocessing": preprocessing,
        "onnx_metadata": onnx_metadata or {},
        "environment": _environment_manifest(build),
    }


def _copy_optional_onnx(
    source: Path,
    target: Path | None,
    *,
    force: bool,
    forbidden_targets: Sequence[Path] = (),
) -> None:
    if target is None:
        return
    if target.suffix.lower() != ".onnx":
        raise ConversionError("--save-onnx path must end in .onnx")
    external_files = _onnx_external_data_files(source)
    target_root = target.parent.resolve()
    forbidden = {path.resolve() for path in forbidden_targets}
    copies = [
        (source, target),
        *(
            (external_source, target.parent / relative)
            for external_source, relative in external_files
        ),
    ]
    destinations: dict[Path, Path] = {}
    for copy_source, copy_target in copies:
        destination = copy_target.resolve()
        try:
            destination.relative_to(target_root)
        except ValueError as exc:
            raise ConversionError(
                f"ONNX retained output escapes its destination directory: {copy_target}"
            ) from exc
        if destination in forbidden:
            raise ConversionError(
                "ONNX external-data destination conflicts with another output: "
                f"{copy_target}"
            )
        previous_source = destinations.get(destination)
        if previous_source is not None:
            raise ConversionError(
                "ONNX retained outputs resolve to the same destination: "
                f"{copy_target} (also used by {previous_source})"
            )
        destinations[destination] = copy_source
        if not force and copy_target.exists() and destination != copy_source.resolve():
            raise ConversionError(f"output already exists (use --force): {copy_target}")

    # Stage every external initializer before changing any retained artifact.
    # Publish the ONNX protobuf last so readers never observe a graph whose
    # external initializers have not been copied yet.
    publish_order = [
        *(
            (external_source, target.parent / relative)
            for external_source, relative in external_files
        ),
        (source, target),
    ]
    staged: dict[Path, Path] = {}
    try:
        for copy_source, copy_target in publish_order:
            if copy_source.resolve() == copy_target.resolve():
                continue
            staged[copy_target] = _stage_copy(copy_source, copy_target)
    except BaseException:
        for staged_path in staged.values():
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    _commit_staged_files(
        staged,
        overwrite=force,
        artifact_label="retained ONNX set",
    )


def _verify_yolo_inference(
    engine_path: Path,
    *,
    test_image: Path,
    task: str,
    batch: int,
) -> None:
    if not test_image.is_file():
        raise ConversionError(f"--test-image does not exist: {test_image}")
    ultralytics = _require_module(
        "ultralytics",
        "Install ultralytics==8.3.187 before YOLO inference verification.",
    )
    try:
        model_task = None if task == "auto" else task
        model = ultralytics.YOLO(str(engine_path), task=model_task)
        sources = [str(test_image)] * batch
        results = model.predict(sources, batch=batch, verbose=False, device=0)
    except Exception as exc:
        raise ConversionError(
            f"YOLO engine inference verification failed: {exc}"
        ) from exc
    if len(results) != batch:
        raise ConversionError(
            f"YOLO verification returned {len(results)} results for batch {batch}"
        )


def convert_yolo(args: argparse.Namespace) -> tuple[Path, Path]:
    source = args.input.resolve()
    output = args.output.resolve()
    if source.suffix.lower() not in {".pt", ".onnx"}:
        raise ConversionError("YOLO input must be .pt or .onnx")
    if not source.is_file():
        raise ConversionError(f"YOLO input does not exist: {source}")
    if output.suffix.lower() != ".engine":
        raise ConversionError("YOLO output must end in .engine")
    manifest_path = output.with_suffix(output.suffix + ".json")
    if not args.force:
        for candidate in (output, manifest_path):
            if candidate.exists():
                raise ConversionError(
                    f"output already exists (use --force): {candidate}"
                )

    image_size = _normalize_two_dimensions(args.imgsz, "--imgsz")
    profile = _build_profile(
        channels=3,
        height=image_size[0],
        width=image_size[1],
        batch=args.batch,
        dynamic=args.dynamic,
        minimum_batch=args.min_batch,
        maximum_batch=args.max_batch,
    )
    _validate_runtime_batch(
        profile,
        allow_nonstandard=args.allow_nonstandard_batch,
    )

    with tempfile.TemporaryDirectory(prefix="spi-yolo-export-") as directory:
        temporary_root = Path(directory)
        if source.suffix.lower() == ".pt":
            if args.onnx_output_contract is not None:
                raise ConversionError(
                    "--onnx-output-contract applies only to an input .onnx model"
                )
            onnx_path = temporary_root / f"{source.stem}.onnx"
            _export_yolo_pt_to_onnx(
                source,
                onnx_path,
                task=args.task,
                image_size=image_size,
                batch=args.batch,
                dynamic=args.dynamic,
                device=args.device,
                opset=args.opset,
                simplify=args.simplify,
                nms=bool(args.nms),
            )
            nms_contract: bool | None = bool(args.nms)
        else:
            if args.nms:
                raise ConversionError(
                    "--nms changes a .pt export graph and cannot rewrite an existing "
                    "ONNX; use --onnx-output-contract to declare its actual output"
                )
            onnx_path = source
            contracts = {"raw": False, "end2end": True}
            nms_contract = (
                contracts[args.onnx_output_contract]
                if args.onnx_output_contract is not None
                else None
            )

        metadata = load_onnx_metadata(onnx_path)
        metadata = _prepare_yolo_metadata(
            metadata,
            task=args.task,
            class_names=args.class_names,
            stride=args.stride,
            batch=args.batch,
            image_size=image_size,
            dynamic=args.dynamic,
            nms=nms_contract,
        )
        build = build_tensorrt_plan(
            onnx_path,
            kind="yolo",
            precision=args.precision,
            workspace_gib=args.workspace_gib,
            profile=profile,
            verbose=args.verbose,
            allow_fp32_fallback=args.allow_fp32_fallback,
        )
        payload = encode_ultralytics_engine(build.plan, metadata)
        decoded_metadata, decoded_plan = decode_ultralytics_engine(payload)
        if decoded_metadata != metadata or decoded_plan != build.plan:
            raise ConversionError("built YOLO engine failed metadata verification")
        if args.test_image is not None:
            verification_engine = temporary_root / "verification.engine"
            verification_engine.write_bytes(payload)
            verify_batch = args.batch if not args.dynamic else profile.optimum[0]
            _verify_yolo_inference(
                verification_engine,
                test_image=args.test_image.resolve(),
                task=str(metadata["task"]),
                batch=verify_batch,
            )
        _copy_optional_onnx(
            onnx_path,
            args.save_onnx.resolve() if args.save_onnx else None,
            force=args.force,
            forbidden_targets=(output, manifest_path),
        )
        manifest = _manifest_payload(
            kind="yolo",
            source=source,
            output=output,
            profile=profile,
            requested_precision=args.precision,
            build=build,
            workspace_gib=args.workspace_gib,
            engine_format="ultralytics-metadata+raw-tensorrt-plan",
            preprocessing={
                "owner": "ultralytics-runtime",
                "image_size": list(image_size),
            },
            onnx_metadata=metadata,
            engine_sha256=_sha256_bytes(payload),
        )
        _publish_engine_and_manifest(
            output,
            payload,
            manifest_path,
            manifest,
            overwrite=args.force,
        )
    return output, manifest_path


def convert_patchcore(args: argparse.Namespace) -> tuple[Path, Path]:
    source = args.input.resolve()
    output = args.output.resolve()
    if source.suffix.lower() not in {".pt", ".pth", ".ckpt", ".onnx"}:
        raise ConversionError("PatchCore input must be .pt, .pth, .ckpt, or .onnx")
    if not source.is_file():
        raise ConversionError(f"PatchCore input does not exist: {source}")
    if output.suffix.lower() != ".engine":
        raise ConversionError("PatchCore output must end in .engine")
    manifest_path = output.with_suffix(output.suffix + ".json")
    if not args.force:
        for candidate in (output, manifest_path):
            if candidate.exists():
                raise ConversionError(
                    f"output already exists (use --force): {candidate}"
                )

    engine_input_size = _normalize_two_dimensions(args.input_size, "--input-size")
    center_crop = (
        _normalize_two_dimensions(args.center_crop, "--center-crop")
        if args.center_crop
        else None
    )
    if center_crop is not None and (
        center_crop[0] > engine_input_size[0] or center_crop[1] > engine_input_size[1]
    ):
        raise ConversionError("--center-crop cannot exceed --input-size")
    default_model_size = center_crop or engine_input_size
    model_input_size = (
        _normalize_two_dimensions(args.model_input_size, "--model-input-size")
        if args.model_input_size
        else default_model_size
    )
    if center_crop is not None and center_crop != model_input_size:
        raise ConversionError(
            "when --center-crop is used, --model-input-size must equal the crop size"
        )
    _patchcore_preprocessing(args.preprocess, args.mean, args.std)

    profile = _build_profile(
        channels=3,
        height=engine_input_size[0],
        width=engine_input_size[1],
        batch=args.batch,
        dynamic=args.dynamic,
        minimum_batch=args.min_batch,
        maximum_batch=args.max_batch,
    )
    _validate_runtime_batch(
        profile,
        allow_nonstandard=args.allow_nonstandard_batch,
    )

    with tempfile.TemporaryDirectory(prefix="spi-patchcore-export-") as directory:
        temporary_root = Path(directory)
        artifact_metadata: dict[str, Any] = {}
        if source.suffix.lower() == ".onnx":
            if args.ignore_artifact_transform:
                raise ConversionError(
                    "--ignore-artifact-transform applies only to a PatchCore .pt/"
                    ".pth/.ckpt artifact that contains anomalib transform metadata"
                )
            if args.preprocess != "none":
                raise ConversionError(
                    "an existing PatchCore ONNX graph cannot be safely rewritten "
                    "to add ImageNet/custom normalization. It must already accept "
                    "the runtime's RGB [0,1] input; use --preprocess none, or use "
                    "the .pt route to embed preprocessing."
                )
            if args.center_crop is not None or args.model_input_size is not None:
                raise ConversionError(
                    "--center-crop/--model-input-size can only be embedded while "
                    "exporting PatchCore .pt; an input ONNX must already contain "
                    "its complete preprocessing graph"
                )
            onnx_path = source
        else:
            onnx_path = temporary_root / f"{source.stem}.onnx"
            artifact_metadata = _export_patchcore_pt_to_onnx(
                source,
                onnx_path,
                trust_pickle=args.trust_pickle,
                engine_input_size=engine_input_size,
                model_input_size=model_input_size,
                center_crop=center_crop,
                preprocess=args.preprocess,
                custom_mean=args.mean,
                custom_std=args.std,
                backbone=args.backbone,
                layers=args.layers,
                num_neighbors=args.num_neighbors,
                dynamic=args.dynamic,
                opset=args.opset,
                export_device=args.export_device,
                ignore_artifact_transform=args.ignore_artifact_transform,
            )

        build = build_tensorrt_plan(
            onnx_path,
            kind="patchcore",
            precision=args.precision,
            workspace_gib=args.workspace_gib,
            profile=profile,
            score_selector=args.score_output,
            verbose=args.verbose,
            allow_fp32_fallback=args.allow_fp32_fallback,
        )
        _copy_optional_onnx(
            onnx_path,
            args.save_onnx.resolve() if args.save_onnx else None,
            force=args.force,
            forbidden_targets=(output, manifest_path),
        )
        mean, std = _patchcore_preprocessing(
            args.preprocess,
            args.mean,
            args.std,
        )
        preprocessing = {
            "runtime_input": "NCHW RGB float32 [0,1]",
            "mode": args.preprocess,
            "mean": list(mean) if mean else None,
            "std": list(std) if std else None,
            "engine_input_size": list(engine_input_size),
            "model_input_size": list(model_input_size),
            "center_crop": list(center_crop) if center_crop else None,
            "embedded_by_tool": source.suffix.lower() != ".onnx",
        }
        manifest = _manifest_payload(
            kind="patchcore",
            source=source,
            output=output,
            profile=profile,
            requested_precision=args.precision,
            build=build,
            workspace_gib=args.workspace_gib,
            engine_format="raw-tensorrt-plan",
            preprocessing=preprocessing,
            onnx_metadata=artifact_metadata,
            engine_sha256=_sha256_bytes(build.plan),
        )
        _publish_engine_and_manifest(
            output,
            build.plan,
            manifest_path,
            manifest,
            overwrite=args.force,
        )
    return output, manifest_path


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="trusted source model (.pt/.ckpt/TorchScript may execute code)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="target .engine path",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp16",
        help="TensorRT builder precision (INT8 intentionally requires a separate "
        "calibrated workflow)",
    )
    parser.add_argument(
        "--workspace-gib",
        type=float,
        default=4.0,
        help="TensorRT builder workspace limit in GiB",
    )
    parser.add_argument("--batch", type=int, default=8, help="fixed/optimum batch")
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="build a dynamic batch profile",
    )
    parser.add_argument("--min-batch", type=int, help="dynamic minimum batch")
    parser.add_argument("--max-batch", type=int, help="dynamic maximum batch")
    parser.add_argument(
        "--allow-nonstandard-batch",
        action="store_true",
        help="allow a profile that is incompatible with this repo's batch-8 APIs",
    )
    parser.add_argument(
        "--allow-fp32-fallback",
        action="store_true",
        help="allow an FP16 request to build as FP32 when the GPU lacks fast FP16",
    )
    parser.add_argument(
        "--save-onnx",
        type=Path,
        help="optionally retain the canonical/intermediate ONNX",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing output/retained ONNX",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose TRT logs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert_to_tensorrt",
        description=(
            "Build verified TensorRT 10 engines for this SPI system. Run on the "
            "target NVIDIA AIPC; engines are not portable artifacts."
        ),
    )
    subparsers = parser.add_subparsers(dest="kind", required=True)

    yolo = subparsers.add_parser(
        "yolo",
        help="convert an Ultralytics YOLO .pt or .onnx model",
    )
    _add_common_arguments(yolo)
    yolo.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[640, 640],
        metavar="PIXELS",
        help="one square size or HEIGHT WIDTH",
    )
    yolo.add_argument(
        "--task",
        choices=("auto", "detect", "segment", "pose", "classify", "obb"),
        default="auto",
        help="model task; ONNX without metadata requires an explicit value",
    )
    yolo.add_argument(
        "--class-names",
        nargs="+",
        help="class names required when a YOLO ONNX lacks metadata",
    )
    yolo.add_argument("--stride", type=int, default=32)
    yolo.add_argument("--device", default="0", help="CUDA device for .pt export")
    yolo.add_argument("--opset", type=int, default=17)
    yolo.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="simplify the intermediate Ultralytics ONNX",
    )
    yolo.add_argument(
        "--nms",
        action="store_true",
        help="embed Ultralytics NMS while exporting an input .pt model",
    )
    yolo.add_argument(
        "--onnx-output-contract",
        choices=("raw", "end2end"),
        help="existing ONNX output is raw predictions or embedded-NMS/end-to-end",
    )
    yolo.add_argument(
        "--test-image",
        type=Path,
        help="run one padded batch through the completed YOLO engine",
    )
    yolo.set_defaults(handler=convert_yolo)

    patchcore = subparsers.add_parser(
        "patchcore",
        help="convert an anomalib PatchCore .pt/.ckpt or canonical .onnx",
    )
    _add_common_arguments(patchcore)
    patchcore.set_defaults(dynamic=True)
    patchcore.add_argument(
        "--input-size",
        nargs="+",
        type=int,
        default=[256, 256],
        metavar="PIXELS",
        help="engine input size: one square value or HEIGHT WIDTH",
    )
    patchcore.add_argument(
        "--model-input-size",
        nargs="+",
        type=int,
        metavar="PIXELS",
        help="underlying .pt model size after optional crop/resize",
    )
    patchcore.add_argument(
        "--center-crop",
        nargs="+",
        type=int,
        metavar="PIXELS",
        help="embed a centered crop while exporting .pt",
    )
    patchcore.add_argument(
        "--preprocess",
        required=True,
        choices=("none", "imagenet", "custom"),
        help=".pt: preprocessing embedded in ONNX; existing .onnx accepts only "
        "none and must already consume runtime RGB [0,1]",
    )
    patchcore.add_argument("--mean", nargs=3, type=float)
    patchcore.add_argument("--std", nargs=3, type=float)
    patchcore.add_argument(
        "--trust-pickle",
        action="store_true",
        help="allow a trusted Python-pickled anomalib module checkpoint",
    )
    patchcore.add_argument(
        "--ignore-artifact-transform",
        action="store_true",
        help=(
            "override anomalib .pt transform metadata only after an independent "
            "preprocessing/decision-parity test"
        ),
    )
    patchcore.add_argument(
        "--backbone",
        help="required for a state_dict checkpoint (for example efficientnet_b5)",
    )
    patchcore.add_argument(
        "--layers",
        nargs="+",
        help="required for a state_dict checkpoint (for example blocks.2 blocks.4)",
    )
    patchcore.add_argument("--num-neighbors", type=int, default=9)
    patchcore.add_argument("--opset", type=int, default=17)
    patchcore.add_argument(
        "--export-device",
        default="cpu",
        help="PyTorch device for .pt to ONNX export",
    )
    patchcore.add_argument(
        "--score-output",
        help=(
            "ONNX score output name or zero-based index when auto-detection "
            "is ambiguous"
        ),
    )
    patchcore.set_defaults(handler=convert_patchcore)
    return parser


def _print_result(output: Path, manifest: Path) -> None:
    print("TensorRT conversion completed")
    print(f"  engine   : {output}")
    print(f"  manifest : {manifest}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.workspace_gib) or args.workspace_gib <= 0:
        parser.error("--workspace-gib must be finite and greater than zero")
    if args.opset <= 0:
        parser.error("--opset must be positive")
    if getattr(args, "stride", 1) <= 0:
        parser.error("--stride must be positive")
    if getattr(args, "num_neighbors", 1) <= 0:
        parser.error("--num-neighbors must be positive")
    try:
        output, manifest = args.handler(args)
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: conversion interrupted", file=sys.stderr)
        return 130
    _print_result(output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
