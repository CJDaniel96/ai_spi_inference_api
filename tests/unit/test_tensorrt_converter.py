"""CUDA-free unit tests for the TensorRT conversion contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.tools import tensorrt_converter as converter


def _tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    mode: str = "output",
    dtype: str = "float32",
) -> converter.TensorInfo:
    return converter.TensorInfo(
        name=name,
        mode=mode,  # type: ignore[arg-type]
        shape=shape,
        dtype=dtype,
    )


def test_shape_profile_reports_static_and_dynamic_profiles() -> None:
    static = converter.ShapeProfile.nchw(
        channels=3,
        height=256,
        width=320,
        minimum_batch=8,
        optimum_batch=8,
        maximum_batch=8,
    )
    dynamic = converter.ShapeProfile.nchw(
        channels=3,
        height=256,
        width=320,
        minimum_batch=1,
        optimum_batch=8,
        maximum_batch=16,
    )

    assert static.minimum == (8, 3, 256, 320)
    assert static.dynamic is False
    assert dynamic.optimum == (8, 3, 256, 320)
    assert dynamic.dynamic is True


@pytest.mark.parametrize(
    ("minimum", "optimum", "maximum", "message"),
    [
        ((1, 3), (1, 3, 8), (1, 3, 8), "equal rank"),
        ((), (), (), "cannot be empty"),
        ((0, 3, 8), (1, 3, 8), (8, 3, 8), "positive"),
        ((2, 3, 8), (1, 3, 8), (8, 3, 8), "minimum <= optimum"),
        ((1, 3, 8), (9, 3, 8), (8, 3, 8), "minimum <= optimum"),
    ],
)
def test_shape_profile_rejects_invalid_shapes(
    minimum: tuple[int, ...],
    optimum: tuple[int, ...],
    maximum: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        converter.ShapeProfile(minimum, optimum, maximum)


def test_build_profile_applies_static_and_dynamic_batch_rules() -> None:
    static = converter._build_profile(
        channels=3,
        height=640,
        width=640,
        batch=8,
        dynamic=False,
        minimum_batch=None,
        maximum_batch=None,
    )
    dynamic = converter._build_profile(
        channels=3,
        height=640,
        width=640,
        batch=8,
        dynamic=True,
        minimum_batch=None,
        maximum_batch=16,
    )

    assert static.minimum == static.optimum == static.maximum == (8, 3, 640, 640)
    assert dynamic.minimum == (1, 3, 640, 640)
    assert dynamic.optimum == (8, 3, 640, 640)
    assert dynamic.maximum == (16, 3, 640, 640)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch": 0, "dynamic": False, "minimum_batch": None, "maximum_batch": None},
        {"batch": 8, "dynamic": False, "minimum_batch": 1, "maximum_batch": None},
        {"batch": 8, "dynamic": True, "minimum_batch": 9, "maximum_batch": 16},
        {"batch": 8, "dynamic": True, "minimum_batch": 1, "maximum_batch": 7},
    ],
)
def test_build_profile_rejects_invalid_batch_ranges(kwargs: dict[str, Any]) -> None:
    with pytest.raises(converter.ConversionError):
        converter._build_profile(channels=3, height=256, width=256, **kwargs)


@pytest.mark.parametrize(
    "profile",
    [
        converter.ShapeProfile((8, 3, 256, 256), (8, 3, 256, 256), (8, 3, 256, 256)),
        converter.ShapeProfile((1, 3, 256, 256), (4, 3, 256, 256), (8, 3, 256, 256)),
        converter.ShapeProfile((1, 3, 256, 256), (8, 3, 256, 256), (16, 3, 256, 256)),
    ],
)
def test_runtime_batch_accepts_profiles_that_can_execute_batch_eight(
    profile: converter.ShapeProfile,
) -> None:
    converter._validate_runtime_batch(profile, allow_nonstandard=False)


@pytest.mark.parametrize(
    "profile",
    [
        converter.ShapeProfile((4, 3, 256, 256), (4, 3, 256, 256), (4, 3, 256, 256)),
        converter.ShapeProfile((1, 3, 256, 256), (4, 3, 256, 256), (7, 3, 256, 256)),
    ],
)
def test_runtime_batch_rejects_profiles_incompatible_with_spi_services(
    profile: converter.ShapeProfile,
) -> None:
    with pytest.raises(converter.ConversionError, match="batch 8"):
        converter._validate_runtime_batch(profile, allow_nonstandard=False)

    converter._validate_runtime_batch(profile, allow_nonstandard=True)


def test_ultralytics_engine_metadata_round_trip_preserves_plan_bytes() -> None:
    metadata = {
        "task": "detect",
        "names": {0: "焊點", 1: "component"},
        "imgsz": [640, 640],
        "args": {"dynamic": True, "nms": False},
    }
    plan = b"\x00\xffraw-tensorrt-plan\x00"

    payload = converter.encode_ultralytics_engine(plan, metadata)
    decoded_metadata, decoded_plan = converter.decode_ultralytics_engine(payload)

    encoded_length = int.from_bytes(payload[:4], "little", signed=True)
    assert encoded_length == len(payload) - len(plan) - 4
    # JSON object keys are strings after decoding, matching Ultralytics' format.
    assert decoded_metadata["names"] == {"0": "焊點", "1": "component"}
    assert decoded_metadata["task"] == "detect"
    assert decoded_plan == plan


def test_ultralytics_engine_rejects_nonserializable_metadata() -> None:
    with pytest.raises(converter.ConversionError, match="not JSON serializable"):
        converter.encode_ultralytics_engine(b"plan", {"bad": object()})


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00\x00\x00\x00x",
        (100).to_bytes(4, "little", signed=True) + b"{}plan",
        (1).to_bytes(4, "little", signed=True) + b"\xffplan",
        (2).to_bytes(4, "little", signed=True) + b"[]plan",
    ],
)
def test_ultralytics_engine_rejects_malformed_metadata_prefix(payload: bytes) -> None:
    with pytest.raises(converter.ConversionError):
        converter.decode_ultralytics_engine(payload)


def test_prepare_yolo_metadata_fills_runtime_contract_fields() -> None:
    prepared = converter._prepare_yolo_metadata(
        {"task": "detect", "names": ["old"], "stride": 16},
        task="segment",
        class_names=["solder", "bridge"],
        stride=32,
        batch=8,
        image_size=(640, 512),
        dynamic=True,
        nms=False,
    )

    assert prepared["task"] == "segment"
    assert prepared["names"] == {"0": "solder", "1": "bridge"}
    assert prepared["stride"] == 16
    assert prepared["batch"] == 8
    assert prepared["imgsz"] == [640, 512]
    assert prepared["channels"] == 3
    assert prepared["args"] == {"dynamic": True, "nms": False}


def _canonical_patchcore_metadata() -> dict[str, Any]:
    return {
        "transform": {
            "__class_fullname__": "albumentations.core.composition.Compose",
            "transforms": [
                {
                    "__class_fullname__": (
                        "albumentations.augmentations.geometric.resize.Resize"
                    ),
                    "height": 256,
                    "width": 320,
                },
                {
                    "__class_fullname__": (
                        "albumentations.augmentations.crops.transforms.CenterCrop"
                    ),
                    "height": 224,
                    "width": 288,
                },
                {
                    "__class_fullname__": (
                        "albumentations.augmentations.transforms.Normalize"
                    ),
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "max_pixel_value": 255.0,
                },
                {
                    "__class_fullname__": (
                        "albumentations.pytorch.transforms.ToTensorV2"
                    )
                },
            ],
        }
    }


def test_parses_canonical_anomalib_transform_metadata() -> None:
    parsed = converter._patchcore_transform_from_metadata(
        _canonical_patchcore_metadata()
    )

    assert parsed == converter.PatchCoreTransform(
        resize=(256, 320),
        center_crop=(224, 288),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        unsupported=(),
    )


def test_patchcore_cli_transform_contract_accepts_matching_metadata() -> None:
    validated = converter._validate_patchcore_artifact_transform(
        _canonical_patchcore_metadata(),
        engine_input_size=(256, 320),
        model_input_size=(224, 288),
        center_crop=(224, 288),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        ignore_artifact_transform=False,
    )

    assert validated is not None
    assert validated.resize == (256, 320)
    assert validated.center_crop == (224, 288)


def test_patchcore_cli_transform_contract_accepts_canonical_none_normalization() -> (
    None
):
    metadata = _canonical_patchcore_metadata()
    transforms = metadata["transform"]["transforms"]  # type: ignore[index]
    transforms[2] = {  # type: ignore[index]
        "__class_fullname__": "albumentations.augmentations.transforms.ToFloat",
        "max_value": 255.0,
    }

    validated = converter._validate_patchcore_artifact_transform(
        metadata,
        engine_input_size=(256, 320),
        model_input_size=(224, 288),
        center_crop=(224, 288),
        mean=None,
        std=None,
        ignore_artifact_transform=False,
    )

    assert validated is not None
    assert validated.mean is None
    assert validated.std is None
    assert validated.unsupported == ()


def test_patchcore_transform_rejects_missing_unit_range_conversion() -> None:
    metadata = _canonical_patchcore_metadata()
    transforms = metadata["transform"]["transforms"]  # type: ignore[index]
    del transforms[2]  # type: ignore[index]

    with pytest.raises(converter.ConversionError, match="missing Normalize/ToFloat"):
        converter._validate_patchcore_artifact_transform(
            metadata,
            engine_input_size=(256, 320),
            model_input_size=(224, 288),
            center_crop=(224, 288),
            mean=None,
            std=None,
            ignore_artifact_transform=False,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"engine_input_size": (256, 256)}, "artifact Resize"),
        ({"center_crop": None}, "artifact CenterCrop"),
        ({"model_input_size": (200, 200)}, "preprocessing produces"),
        ({"mean": (0.1, 0.2, 0.3)}, "Normalize mean/std"),
    ],
)
def test_patchcore_cli_transform_contract_rejects_mismatches(
    overrides: dict[str, Any],
    message: str,
) -> None:
    arguments: dict[str, Any] = {
        "engine_input_size": (256, 320),
        "model_input_size": (224, 288),
        "center_crop": (224, 288),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "ignore_artifact_transform": False,
    }
    arguments.update(overrides)

    with pytest.raises(converter.ConversionError, match=message):
        converter._validate_patchcore_artifact_transform(
            _canonical_patchcore_metadata(),
            **arguments,
        )


def test_patchcore_cli_transform_contract_allows_explicit_ignore() -> None:
    metadata = _canonical_patchcore_metadata()
    metadata["transform"]["transforms"].append(  # type: ignore[index]
        {
            "__class_fullname__": (
                "albumentations.augmentations.transforms.RandomBrightnessContrast"
            )
        }
    )

    validated = converter._validate_patchcore_artifact_transform(
        metadata,
        engine_input_size=(640, 640),
        model_input_size=(640, 640),
        center_crop=None,
        mean=None,
        std=None,
        ignore_artifact_transform=True,
    )
    parsed_args = converter.build_parser().parse_args(
        [
            "patchcore",
            "--input",
            "model.pt",
            "--output",
            "model.engine",
            "--preprocess",
            "none",
            "--ignore-artifact-transform",
        ]
    )

    assert validated is None
    assert parsed_args.ignore_artifact_transform is True


def _prepare_yolo_for_test(
    metadata: dict[str, Any],
    *,
    task: str = "auto",
    nms: bool | None = None,
) -> dict[str, Any]:
    return converter._prepare_yolo_metadata(
        metadata,
        task=task,
        class_names=None,
        stride=32,
        batch=8,
        image_size=(640, 640),
        dynamic=False,
        nms=nms,
    )


def test_yolo_onnx_metadata_requires_explicit_nms_contract_when_missing() -> None:
    metadata = {"task": "detect", "names": ["component"], "stride": 32}

    with pytest.raises(converter.ConversionError, match="raw versus embedded-NMS"):
        _prepare_yolo_for_test(metadata)


def test_yolo_onnx_metadata_rejects_non_boolean_nms() -> None:
    metadata = {
        "task": "detect",
        "names": ["component"],
        "stride": 32,
        "args": {"nms": "false"},
    }

    with pytest.raises(converter.ConversionError, match="nms metadata must be boolean"):
        _prepare_yolo_for_test(metadata)


@pytest.mark.parametrize(
    ("metadata", "task", "message"),
    [
        (
            {"names": ["component"], "args": {"nms": False}},
            "auto",
            "task metadata is missing or unsupported",
        ),
        (
            {"task": "caption", "names": ["component"], "args": {"nms": False}},
            "auto",
            "task metadata is missing or unsupported",
        ),
        (
            {"task": "detect", "names": ["component"], "args": {"nms": False}},
            "caption",
            "unsupported YOLO task",
        ),
    ],
)
def test_yolo_onnx_metadata_rejects_missing_or_invalid_task(
    metadata: dict[str, Any],
    task: str,
    message: str,
) -> None:
    with pytest.raises(converter.ConversionError, match=message):
        _prepare_yolo_for_test(metadata, task=task)


@pytest.mark.parametrize(
    "names",
    [
        [],
        [""],
        [123],
        {1: "component"},
        {True: "component"},
        {"not-an-index": "component"},
    ],
)
def test_yolo_onnx_metadata_rejects_invalid_class_names(names: Any) -> None:
    metadata = {
        "task": "detect",
        "names": names,
        "args": {"nms": False},
    }

    with pytest.raises(converter.ConversionError, match="class-name|class name"):
        _prepare_yolo_for_test(metadata)


@pytest.mark.parametrize("stride", [0, -1, "not-a-number", True, 32.5])
def test_yolo_onnx_metadata_rejects_invalid_stride(stride: Any) -> None:
    metadata = {
        "task": "detect",
        "names": ["component"],
        "stride": stride,
        "args": {"nms": False},
    }

    with pytest.raises(converter.ConversionError, match="stride"):
        _prepare_yolo_for_test(metadata)


@pytest.mark.parametrize("channels", [0, 1, 4, "not-a-number", True, 3.5])
def test_yolo_onnx_metadata_rejects_invalid_channels(channels: Any) -> None:
    metadata = {
        "task": "detect",
        "names": ["component"],
        "channels": channels,
        "args": {"nms": False},
    }

    with pytest.raises(converter.ConversionError, match="channels|input channels"):
        _prepare_yolo_for_test(metadata)


def test_patchcore_output_mapping_prefers_canonical_names() -> None:
    outputs = [
        _tensor("anomaly_map", (-1, 1, 256, 256)),
        _tensor("pred_score", (-1,)),
    ]

    assert converter._resolve_patchcore_output_names(outputs, None) == (
        "anomaly_map",
        "pred_score",
    )


def test_patchcore_output_mapping_uses_shape_fallback_and_explicit_selector() -> None:
    outputs = [
        _tensor("heatmap", (-1, 1, 256, 256)),
        _tensor("classification", (-1,)),
    ]

    assert converter._resolve_patchcore_output_names(outputs, None) == (
        "heatmap",
        "classification",
    )
    assert converter._resolve_patchcore_output_names(outputs, "1") == (
        "heatmap",
        "classification",
    )
    assert converter._resolve_patchcore_output_names(outputs, "classification") == (
        "heatmap",
        "classification",
    )


@pytest.mark.parametrize(
    ("outputs", "selector", "message"),
    [
        ([_tensor("only", (-1, 1, 8, 8))], None, "both anomaly_map"),
        (
            [_tensor("map_a", (-1, 1, 8, 8)), _tensor("map_b", (-1, 1, 8, 8))],
            None,
            "score output",
        ),
        (
            [_tensor("anomaly_map", (-1, 1, 8, 8)), _tensor("score", (-1,))],
            "9",
            "outside",
        ),
        (
            [_tensor("anomaly_map", (-1, 1, 8, 8)), _tensor("score", (-1,))],
            "missing",
            "not an ONNX output",
        ),
        (
            [
                _tensor("anomaly_map", (-1, 1, 8, 8)),
                _tensor("anomaly_maps", (-1, 1, 8, 8)),
                _tensor("pred_score", (-1,)),
            ],
            None,
            "anomaly-map output is ambiguous",
        ),
    ],
)
def test_patchcore_output_mapping_rejects_ambiguous_or_invalid_contracts(
    outputs: list[converter.TensorInfo],
    selector: str | None,
    message: str,
) -> None:
    with pytest.raises(converter.ConversionError, match=message):
        converter._resolve_patchcore_output_names(outputs, selector)


class _NetworkTensor:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape
        self.dtype = "float32"


class _Network:
    def __init__(
        self,
        inputs: list[_NetworkTensor],
        outputs: list[_NetworkTensor],
    ) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self.num_inputs = len(inputs)
        self.num_outputs = len(outputs)

    def get_input(self, index: int) -> _NetworkTensor:
        return self._inputs[index]

    def get_output(self, index: int) -> _NetworkTensor:
        return self._outputs[index]


def test_prepare_patchcore_contract_renames_input_and_mapped_outputs() -> None:
    network = _Network(
        [_NetworkTensor("image", (-1, 3, 256, 256))],
        [
            _NetworkTensor("heatmap", (-1, 1, 256, 256)),
            _NetworkTensor("score", (-1,)),
        ],
    )
    profile = converter.ShapeProfile(
        (1, 3, 256, 256),
        (8, 3, 256, 256),
        (16, 3, 256, 256),
    )

    converter._prepare_network_contract(
        network,
        kind="patchcore",
        profile=profile,
        score_selector="score",
    )

    assert network.get_input(0).name == "input"
    assert [network.get_output(index).name for index in range(2)] == [
        "anomaly_map",
        "pred_score",
    ]


def test_validate_engine_contract_checks_patchcore_names_and_profile() -> None:
    profile = converter.ShapeProfile(
        (1, 3, 256, 256),
        (8, 3, 256, 256),
        (16, 3, 256, 256),
    )
    inspection = converter.EngineInspection(
        tensors=(
            converter.TensorInfo(
                name="input",
                mode="input",
                shape=(-1, 3, 256, 256),
                dtype="float32",
                profile_minimum=profile.minimum,
                profile_optimum=profile.optimum,
                profile_maximum=profile.maximum,
            ),
            _tensor("anomaly_map", (-1, 1, 256, 256)),
            _tensor("pred_score", (-1,)),
        )
    )

    converter._validate_engine_contract(
        inspection,
        kind="patchcore",
        profile=profile,
    )

    missing_score = converter.EngineInspection(tensors=inspection.tensors[:-1])
    with pytest.raises(converter.ConversionError, match="anomaly_map and pred_score"):
        converter._validate_engine_contract(
            missing_score,
            kind="patchcore",
            profile=profile,
        )


def _patchcore_inspection(
    *,
    input_shape: tuple[int, ...] = (-1, 3, 256, 256),
    input_dtype: str = "float32",
    anomaly_map_shape: tuple[int, ...] = (-1, 1, 256, 256),
    anomaly_map_dtype: str = "float32",
    score_shape: tuple[int, ...] = (-1,),
    score_dtype: str = "float32",
) -> tuple[converter.EngineInspection, converter.ShapeProfile]:
    profile = converter.ShapeProfile(
        (1, 3, 256, 256),
        (8, 3, 256, 256),
        (16, 3, 256, 256),
    )
    return (
        converter.EngineInspection(
            tensors=(
                converter.TensorInfo(
                    name="input",
                    mode="input",
                    shape=input_shape,
                    dtype=input_dtype,
                    profile_minimum=profile.minimum,
                    profile_optimum=profile.optimum,
                    profile_maximum=profile.maximum,
                ),
                _tensor(
                    "anomaly_map",
                    anomaly_map_shape,
                    dtype=anomaly_map_dtype,
                ),
                _tensor("pred_score", score_shape, dtype=score_dtype),
            )
        ),
        profile,
    )


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((-1, -1, 256, 256), r"\[B,1,H,W\]"),
        ((-1, 1, -1, 256), "fixed anomaly-map C/H/W"),
        ((-1, 1, 256, -1), "fixed anomaly-map C/H/W"),
    ],
)
def test_patchcore_engine_rejects_nonfixed_output_channel_or_spatial_dimensions(
    shape: tuple[int, ...],
    message: str,
) -> None:
    inspection, profile = _patchcore_inspection(anomaly_map_shape=shape)

    with pytest.raises(converter.ConversionError, match=message):
        converter._validate_engine_contract(
            inspection,
            kind="patchcore",
            profile=profile,
        )


def test_patchcore_engine_rejects_vector_score_per_sample() -> None:
    inspection, profile = _patchcore_inspection(score_shape=(-1, 2))

    with pytest.raises(converter.ConversionError, match=r"\[B\] or \[B,1\]"):
        converter._validate_engine_contract(
            inspection,
            kind="patchcore",
            profile=profile,
        )


@pytest.mark.parametrize(
    ("dtype_overrides", "message"),
    [
        ({"input_dtype": "int8"}, "engine input must be float32/float16"),
        ({"anomaly_map_dtype": "int32"}, "outputs must use floating-point"),
        ({"score_dtype": "bool"}, "outputs must use floating-point"),
    ],
)
def test_patchcore_engine_rejects_nonfloating_runtime_tensors(
    dtype_overrides: dict[str, str],
    message: str,
) -> None:
    inspection, profile = _patchcore_inspection(**dtype_overrides)

    with pytest.raises(converter.ConversionError, match=message):
        converter._validate_engine_contract(
            inspection,
            kind="patchcore",
            profile=profile,
        )


def test_cli_rejects_invalid_builder_values_before_calling_handler(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    def fail_if_called(_: Any) -> tuple[Path, Path]:
        nonlocal called
        called = True
        raise AssertionError("handler must not be called")

    monkeypatch.setattr(converter, "convert_yolo", fail_if_called)

    with pytest.raises(SystemExit) as exc_info:
        converter.main(
            [
                "yolo",
                "--input",
                "model.onnx",
                "--output",
                "model.engine",
                "--workspace-gib",
                "0",
            ]
        )

    assert exc_info.value.code == 2
    assert "--workspace-gib must be finite and greater than zero" in (
        capsys.readouterr().err
    )
    assert called is False


def test_cli_reports_existing_output_without_loading_tensorrt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "model.onnx"
    output = tmp_path / "model.engine"
    source.write_bytes(b"onnx")
    output.write_bytes(b"keep-me")

    def fail_if_called() -> Any:
        raise AssertionError("TensorRT must not be loaded for an existing output")

    monkeypatch.setattr(converter, "_load_tensorrt", fail_if_called)

    exit_code = converter.main(
        [
            "yolo",
            "--input",
            str(source),
            "--output",
            str(output),
            "--task",
            "detect",
            "--class-names",
            "component",
        ]
    )

    assert exit_code == 2
    assert "output already exists" in capsys.readouterr().err
    assert output.read_bytes() == b"keep-me"


def test_atomic_write_refuses_to_overwrite_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "model.engine"
    output.write_bytes(b"existing")

    with pytest.raises(converter.ConversionError, match="output already exists"):
        converter._atomic_write_bytes(output, b"replacement", overwrite=False)

    assert output.read_bytes() == b"existing"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_preserves_file_that_appears_during_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "model.engine"

    def competing_writer(_: int) -> None:
        output.write_bytes(b"competitor")

    monkeypatch.setattr(converter.os, "fsync", competing_writer)

    with pytest.raises(converter.ConversionError, match="appeared during build"):
        converter._atomic_write_bytes(output, b"new-engine", overwrite=False)

    assert output.read_bytes() == b"competitor"
    assert list(tmp_path.glob(".model.*.engine.tmp")) == []


def test_engine_manifest_publish_restores_old_pair_if_second_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = tmp_path / "model.engine"
    manifest = tmp_path / "model.engine.json"
    old_engine = b"old-engine"
    old_manifest = b'{"release":"old"}\n'
    engine.write_bytes(old_engine)
    manifest.write_bytes(old_manifest)
    real_replace = converter.os.replace

    def fail_new_manifest(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == manifest and source_path.suffix == ".staged":
            raise OSError("simulated second publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(converter.os, "replace", fail_new_manifest)

    with pytest.raises(converter.ConversionError, match="failed to publish"):
        converter._publish_engine_and_manifest(
            engine,
            b"new-engine",
            manifest,
            {"release": "new"},
            overwrite=True,
        )

    assert engine.read_bytes() == old_engine
    assert manifest.read_bytes() == old_manifest
    assert list(tmp_path.glob("*.rollback")) == []
    assert list(tmp_path.glob("*.staged")) == []


def test_engine_manifest_publish_restores_old_pair_on_ctrl_c_after_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = tmp_path / "model.engine"
    manifest = tmp_path / "model.engine.json"
    engine.write_bytes(b"old-engine")
    manifest.write_bytes(b'{"release":"old"}\n')
    real_replace = converter.os.replace

    def interrupt_after_backup(source: str | Path, destination: str | Path) -> None:
        real_replace(source, destination)
        if Path(source) == engine and Path(destination).suffix == ".rollback":
            raise KeyboardInterrupt

    monkeypatch.setattr(converter.os, "replace", interrupt_after_backup)

    with pytest.raises(KeyboardInterrupt):
        converter._publish_engine_and_manifest(
            engine,
            b"new-engine",
            manifest,
            {"release": "new"},
            overwrite=True,
        )

    assert engine.read_bytes() == b"old-engine"
    assert manifest.read_bytes() == b'{"release":"old"}\n'


def test_engine_manifest_publish_removes_partial_pair_on_ctrl_c_after_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = tmp_path / "model.engine"
    manifest = tmp_path / "model.engine.json"
    real_link = converter.os.link

    def interrupt_after_link(source: str | Path, destination: str | Path) -> None:
        real_link(source, destination)
        if Path(destination) == engine:
            raise KeyboardInterrupt

    monkeypatch.setattr(converter.os, "link", interrupt_after_link)

    with pytest.raises(KeyboardInterrupt):
        converter._publish_engine_and_manifest(
            engine,
            b"new-engine",
            manifest,
            {"release": "new"},
            overwrite=False,
        )

    assert not engine.exists()
    assert not manifest.exists()


def test_engine_manifest_publish_reconciles_ctrl_c_during_link_bookkeeping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = tmp_path / "model.engine"
    manifest = tmp_path / "model.engine.json"
    real_samefile = converter.os.path.samefile

    def interrupt_after_samefile(first: str | Path, second: str | Path) -> bool:
        result = real_samefile(first, second)
        if Path(second) == engine:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(converter.os.path, "samefile", interrupt_after_samefile)

    with pytest.raises(KeyboardInterrupt):
        converter._publish_engine_and_manifest(
            engine,
            b"new-engine",
            manifest,
            {"release": "new"},
            overwrite=False,
        )

    assert not engine.exists()
    assert not manifest.exists()


def test_copy_optional_onnx_rejects_external_destination_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "retained"
    outside_root = tmp_path / "outside"
    source_root.mkdir()
    destination_root.mkdir()
    outside_root.mkdir()
    source = source_root / "model.onnx"
    external_source = source_root / "weights.bin"
    source.write_bytes(b"onnx")
    external_source.write_bytes(b"weights")
    link = destination_root / "linked"
    try:
        link.symlink_to(outside_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    monkeypatch.setattr(
        converter,
        "_onnx_external_data_files",
        lambda _: ((external_source, Path("linked/weights.bin")),),
    )

    with pytest.raises(converter.ConversionError, match="escapes its destination"):
        converter._copy_optional_onnx(
            source,
            destination_root / "model.onnx",
            force=False,
        )

    assert not (destination_root / "model.onnx").exists()
    assert not (outside_root / "weights.bin").exists()


def test_onnx_tensor_discovery_includes_subgraphs_and_attribute_tensors() -> None:
    top_level = SimpleNamespace(name="top")
    nested = SimpleNamespace(name="nested")
    attribute_tensor = SimpleNamespace(name="attribute")
    training_tensor = SimpleNamespace(name="training")
    function_tensor = SimpleNamespace(name="function")

    def graph(*, initializers: list[Any], nodes: list[Any]) -> Any:
        return SimpleNamespace(
            initializer=initializers,
            sparse_initializer=[],
            node=nodes,
        )

    nested_graph = graph(initializers=[nested], nodes=[])
    nested_attribute = SimpleNamespace(
        g=nested_graph,
        tensors=[attribute_tensor],
        graphs=[],
        sparse_tensors=[],
    )
    model = SimpleNamespace(
        graph=graph(
            initializers=[top_level],
            nodes=[SimpleNamespace(attribute=[nested_attribute])],
        ),
        training_info=[
            SimpleNamespace(
                algorithm=graph(initializers=[training_tensor], nodes=[]),
            )
        ],
        functions=[
            SimpleNamespace(
                node=[],
                attribute_proto=[SimpleNamespace(t=function_tensor)],
            )
        ],
    )

    assert converter._onnx_model_tensors(model) == (
        top_level,
        attribute_tensor,
        nested,
        training_tensor,
        function_tensor,
    )


@pytest.mark.parametrize("conflicting_name", ["model.engine", "model.engine.json"])
def test_copy_optional_onnx_rejects_external_filename_output_conflict(
    conflicting_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "retained"
    source_root.mkdir()
    destination_root.mkdir()
    source = source_root / "model.onnx"
    external_source = source_root / "external.bin"
    source.write_bytes(b"onnx")
    external_source.write_bytes(b"weights")
    engine = destination_root / "model.engine"
    manifest = destination_root / "model.engine.json"
    monkeypatch.setattr(
        converter,
        "_onnx_external_data_files",
        lambda _: ((external_source, Path(conflicting_name)),),
    )

    with pytest.raises(
        converter.ConversionError,
        match="conflicts with another output",
    ):
        converter._copy_optional_onnx(
            source,
            destination_root / "model.onnx",
            force=False,
            forbidden_targets=(engine, manifest),
        )

    assert not (destination_root / "model.onnx").exists()
    assert not (destination_root / conflicting_name).exists()


def test_copy_optional_onnx_restores_existing_set_on_force_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "retained"
    source_root.mkdir()
    destination_root.mkdir()
    source = source_root / "model.onnx"
    external_source = source_root / "weights.bin"
    target = destination_root / "model.onnx"
    external_target = destination_root / "weights.bin"
    source.write_bytes(b"new-onnx")
    external_source.write_bytes(b"new-weights")
    target.write_bytes(b"old-onnx")
    external_target.write_bytes(b"old-weights")
    monkeypatch.setattr(
        converter,
        "_onnx_external_data_files",
        lambda _: ((external_source, Path("weights.bin")),),
    )
    real_replace = converter.os.replace

    def fail_new_graph(source_path: str | Path, destination: str | Path) -> None:
        source_candidate = Path(source_path)
        destination_candidate = Path(destination)
        if destination_candidate == target and source_candidate.suffix == ".staged":
            raise OSError("simulated retained graph publish failure")
        real_replace(source_path, destination)

    monkeypatch.setattr(converter.os, "replace", fail_new_graph)

    with pytest.raises(converter.ConversionError, match="retained ONNX set"):
        converter._copy_optional_onnx(source, target, force=True)

    assert target.read_bytes() == b"old-onnx"
    assert external_target.read_bytes() == b"old-weights"


class _FakeConfig:
    def __init__(self) -> None:
        self.workspace: tuple[object, int] | None = None
        self.flags: list[object] = []
        self.profiles: list[_FakeOptimizationProfile] = []

    def set_memory_pool_limit(self, pool: object, value: int) -> None:
        self.workspace = (pool, value)

    def add_optimization_profile(self, profile: _FakeOptimizationProfile) -> None:
        self.profiles.append(profile)

    def set_flag(self, flag: object) -> None:
        self.flags.append(flag)


@pytest.mark.parametrize("workspace_gib", [float("nan"), float("inf"), 1e-300, 1e300])
def test_workspace_rejects_nonfinite_or_out_of_range_values(
    workspace_gib: float,
) -> None:
    config = _FakeConfig()
    trt = SimpleNamespace(MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"))

    with pytest.raises(converter.ConversionError, match="workspace-gib"):
        converter._set_workspace(config, trt, workspace_gib)


def test_workspace_wraps_tensorrt_binding_errors() -> None:
    class RejectingConfig:
        def set_memory_pool_limit(self, pool: object, value: int) -> None:
            raise OverflowError(f"cannot set {pool} to {value}")

    trt = SimpleNamespace(MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"))

    with pytest.raises(converter.ConversionError, match="rejected the workspace"):
        converter._set_workspace(RejectingConfig(), trt, 4.0)


class _FakeOptimizationProfile:
    def __init__(self) -> None:
        self.shape: (
            tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None
        ) = None

    def set_shape(
        self,
        name: str,
        minimum: tuple[int, ...],
        optimum: tuple[int, ...],
        maximum: tuple[int, ...],
    ) -> bool:
        self.shape = (name, minimum, optimum, maximum)
        return True


class _FakeEngine:
    def __init__(
        self,
        trt: Any,
        network: _Network,
        profile: _FakeOptimizationProfile,
    ) -> None:
        self._trt = trt
        self._network = network
        self._profile = profile
        self._tensors = [
            *network._inputs,  # noqa: SLF001 - intentionally tiny TensorRT fake
            *network._outputs,  # noqa: SLF001 - intentionally tiny TensorRT fake
        ]
        self.num_io_tensors = len(self._tensors)

    def get_tensor_name(self, index: int) -> str:
        return self._tensors[index].name

    def _by_name(self, name: str) -> _NetworkTensor:
        return next(tensor for tensor in self._tensors if tensor.name == name)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self._by_name(name).shape

    def get_tensor_mode(self, name: str) -> object:
        return (
            self._trt.TensorIOMode.INPUT
            if self._by_name(name) in self._network._inputs  # noqa: SLF001
            else self._trt.TensorIOMode.OUTPUT
        )

    def get_tensor_dtype(self, name: str) -> str:
        return self._by_name(name).dtype

    def get_tensor_profile_shape(
        self, name: str, profile_index: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert name == "input"
        assert profile_index == 0
        assert self._profile.shape is not None
        return self._profile.shape[1:]


def _fake_tensorrt_10() -> Any:
    state = SimpleNamespace(network=None, config=None, optimization_profile=None)
    trt = SimpleNamespace(
        __version__="10.8.0",
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
        NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0),
        BuilderFlag=SimpleNamespace(FP16="fp16"),
        TensorIOMode=SimpleNamespace(INPUT="input", OUTPUT="output"),
    )

    class Logger:
        INFO = "info"
        VERBOSE = "verbose"

        def __init__(self, severity: str) -> None:
            self.severity = severity

    class Builder:
        platform_has_fast_fp16 = True

        def __init__(self, logger: Logger) -> None:
            self.logger = logger

        def create_builder_config(self) -> _FakeConfig:
            state.config = _FakeConfig()
            return state.config

        def create_network(self, flags: int) -> _Network:
            assert flags == 1
            state.network = _Network(
                [_NetworkTensor("source_input", (-1, 3, 256, 256))],
                [
                    _NetworkTensor("heatmap", (-1, 1, 256, 256)),
                    _NetworkTensor("score", (-1,)),
                ],
            )
            return state.network

        def create_optimization_profile(self) -> _FakeOptimizationProfile:
            state.optimization_profile = _FakeOptimizationProfile()
            return state.optimization_profile

        def build_serialized_network(
            self, network: _Network, config: _FakeConfig
        ) -> bytes:
            assert network is state.network
            assert config is state.config
            return b"fake-tensorrt-10-plan"

    class OnnxParser:
        num_errors = 0

        def __init__(self, network: _Network, logger: Logger) -> None:
            assert network is state.network

        def parse_from_file(self, path: str) -> bool:
            return Path(path).is_file()

    class Runtime:
        def __init__(self, logger: Logger) -> None:
            self.logger = logger

        def deserialize_cuda_engine(self, plan: bytes) -> _FakeEngine:
            assert plan == b"fake-tensorrt-10-plan"
            assert state.network is not None
            assert state.optimization_profile is not None
            return _FakeEngine(trt, state.network, state.optimization_profile)

    trt.Logger = Logger
    trt.Builder = Builder
    trt.OnnxParser = OnnxParser
    trt.Runtime = Runtime
    trt._test_state = state
    return trt


def test_fake_tensorrt_10_build_enforces_profile_precision_and_io_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx_path = tmp_path / "patchcore.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    trt = _fake_tensorrt_10()
    monkeypatch.setattr(converter, "_load_tensorrt", lambda: trt)
    profile = converter.ShapeProfile(
        (1, 3, 256, 256),
        (8, 3, 256, 256),
        (16, 3, 256, 256),
    )

    result = converter.build_tensorrt_plan(
        onnx_path,
        kind="patchcore",
        precision="fp16",
        workspace_gib=2,
        profile=profile,
        score_selector="score",
    )

    state = trt._test_state
    assert result.plan == b"fake-tensorrt-10-plan"
    assert result.tensorrt_version == "10.8.0"
    assert result.builder_precision_mode == "fp16"
    assert [tensor.name for tensor in result.inspection.tensors] == [
        "input",
        "anomaly_map",
        "pred_score",
    ]
    assert state.config.workspace == ("workspace", 2 * (1 << 30))
    assert state.config.flags == ["fp16"]
    assert state.config.profiles == [state.optimization_profile]
    assert state.optimization_profile.shape == (
        "input",
        profile.minimum,
        profile.optimum,
        profile.maximum,
    )
