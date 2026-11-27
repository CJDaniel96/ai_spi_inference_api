"""Component rules, confidence filtering and native result publication."""

import asyncio
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from app.application.inference.obb_service import ObbInspectionService
from app.application.operations import backup_result, prepare_publications, publish_csv
from app.core.component_rules import DEFAULT_COMPONENT_RULES_PATH, load_component_rules
from app.core.config import AppConfig
from app.core.inspection_config import ObbConfig
from app.domain.entities.obb_result import OBB_CLASSES, ObbBatchResult, ObbDetection
from app.domain.entities.pipeline_result import PipelineOutcome, PipelineResultManifest
from app.domain.services.appearance_classifier import AppearanceClassifier
from app.domain.services.component_judge import ComponentJudge
from tests.helpers import AsyncTestIO

ROOT = Path(__file__).resolve().parents[2]
POLYGON = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def box(name="Solder_Paste", confidence=0.8):
    return ObbDetection(name, confidence, POLYGON)


def judge():
    return ComponentJudge(
        ObbConfig().class_thresholds,
        load_component_rules(ROOT / DEFAULT_COMPONENT_RULES_PATH),
    )


def row(**changes):
    return {
        "component_name": "R0001",
        "insp_vol": "100",
        "insp_area": "100",
        "insp_hei": "100",
        "insp_x": "0",
        "insp_y": "0",
        "vol_l_ng": "50",
        "vol_h_ng": "150",
        "area_l_ng": "50",
        "area_h_ng": "150",
        "hei_l_ng": "50",
        "hei_h_ng": "150",
        "shift_x_ng": "0.05",
        "shift_y_ng": "0.05",
        **changes,
    }


@pytest.mark.parametrize(
    ("detections", "passed", "ok", "ng"),
    [
        ((box(confidence=0.71), box(confidence=0.3), box(confidence=0.5)), True, 1, 0),
        ((box(confidence=0.7),), True, 1, 0),
        ((box(confidence=0.699),), False, 0, 0),
        ((box(), box(confidence=0.75)), False, 2, 0),
        ((box(), box("Bridge", 0.25)), False, 1, 1),
        ((box(), box("Bridge", 0.249)), True, 1, 0),
        ((), False, 0, 0),
    ],
)
def test_confidence_filtering_happens_before_counting(detections, passed, ok, ng):
    result = judge().appearance.classify(detections, image_center=(5.0, 5.0))
    assert (result.passed, result.ok_count, result.ng_count) == (passed, ok, ng)


@pytest.mark.parametrize("name", OBB_CLASSES[1:])
def test_each_ng_class_blocks_appearance(name):
    result = judge().judge(row(), (box(), box(name, 0.3)), image_center=(5.0, 5.0))
    assert result.failed_rule == "yolo_obb" and result.defect == name


def test_ng_thresholds_are_independent():
    thresholds = ObbConfig().class_thresholds
    thresholds["Bridge"] = 0.9
    classifier = AppearanceClassifier(thresholds)
    assert classifier.classify(
        (box(), box("Bridge", 0.8)), image_center=(5.0, 5.0)
    ).passed
    assert not classifier.classify(
        (box(), box("Stain", 0.3)), image_center=(5.0, 5.0)
    ).passed


@pytest.mark.parametrize(
    ("name", "measurement", "low", "high"),
    [("volume", "insp_vol", 1, 180), ("area", "insp_area", 10, 160)],
)
@pytest.mark.parametrize("position", ["lower", "upper", "below", "above"])
def test_component_intervals_include_boundaries(name, measurement, low, high, position):
    value = {"lower": low, "upper": high, "below": low - 0.001, "above": high + 0.001}[
        position
    ]
    decision = judge().judge(
        row(component_name="U0500", **{measurement: value}),
        (box(),),
        image_center=(5.0, 5.0),
    )
    assert decision.passed == (position in ("lower", "upper"))
    if not decision.passed:
        assert decision.failed_rule == name
        assert decision.defect == f"{'low' if position == 'below' else 'high'} {name}"
    check = next(check for check in decision.checks if check["rule"] == name)
    assert (check["low"], check["high"]) == (low, high)
    assert decision.component_rule == "U0500"


@pytest.mark.parametrize(
    ("name", "volume", "area", "matched", "failed"),
    [
        ("U0500", 1, 10, "U0500", ""),
        (" U0900 ", 190, 170, "U0900", "volume"),
        ("C1174", 180, 59, "C1174", "area"),
        ("C1178", 180, 160, "C1178", ""),
        ("C1110", 180, 55, "C1110", "volume"),
        ("RS0501", 170, 60, "RS0501", ""),
        ("R0001", 170, 50, "others", ""),
        ("U05001", 100, 12, "others", "area"),
    ],
)
def test_component_matching_and_default_record_both_checks(
    name, volume, area, matched, failed
):
    decision = judge().judge(
        row(component_name=name, insp_vol=volume, insp_area=area),
        (box(),),
        image_center=(5.0, 5.0),
    )
    assert decision.component_rule == matched
    assert decision.failed_rule == failed
    assert [check["rule"] for check in decision.checks] == [
        "yolo_obb",
        "volume",
        "area",
    ]


@pytest.mark.parametrize("name", ["", "  ", None])
def test_missing_component_is_invalid_after_appearance_passes(name):
    data = row(component_name=name, insp_vol="")
    with pytest.raises(ValueError, match="component_name"):
        judge().judge(data, (box(),), image_center=(5.0, 5.0))
    assert judge().judge(data, (), image_center=(5.0, 5.0)).failed_rule == "yolo_obb"


@pytest.mark.parametrize("column", ["insp_vol", "insp_area"])
@pytest.mark.parametrize("value", ["", "NaN", "inf", "invalid"])
def test_invalid_measurement_is_never_a_pass(column, value):
    with pytest.raises(ValueError, match="measurement"):
        judge().judge(row(**{column: value}), (box(),), image_center=(5.0, 5.0))


@pytest.mark.parametrize(
    "changes",
    [
        {"conf": 0.8},
        {"class_thresholds": {"Solder_Paste": 0.7}},
        {"class_thresholds": {**ObbConfig().class_thresholds, "Bridge": float("nan")}},
        {"class_thresholds": {**ObbConfig().class_thresholds, "Bridge": 0.1}},
        {"model_path": "unsupported.bin"},
    ],
)
def test_invalid_inference_settings_are_rejected(changes):
    with pytest.raises(ValidationError):
        ObbConfig(**changes)


@pytest.mark.parametrize("label", ["Foreign_object", "Foreign_Object"])
def test_threshold_spelling_preserves_custom_ng_confidence(label):
    thresholds = ObbConfig().class_thresholds
    del thresholds["Foreign_object"]
    thresholds[label] = 0.65
    original = dict(thresholds)

    settings = ObbConfig(class_thresholds=thresholds)

    assert thresholds == original
    assert settings.model_dump()["class_thresholds"]["Foreign_object"] == 0.65
    assert set(settings.class_thresholds) == set(OBB_CLASSES)
    classifier = AppearanceClassifier(settings.class_thresholds)
    assert classifier.classify(
        (box(), box("Foreign_object", 0.649)), image_center=(5.0, 5.0)
    ).passed
    decision = classifier.classify(
        (box(), box("Foreign_object", 0.65)), image_center=(5.0, 5.0)
    )
    assert not decision.passed
    assert decision.reason == "Foreign_object"


@pytest.mark.parametrize("alias_first", [False, True])
@pytest.mark.parametrize("alias_threshold", [0.25, 0.9])
def test_duplicate_threshold_aliases_are_rejected(alias_first, alias_threshold):
    thresholds = ObbConfig().class_thresholds
    if alias_first:
        thresholds = {"Foreign_Object": alias_threshold, **thresholds}
    else:
        thresholds["Foreign_Object"] = alias_threshold

    with pytest.raises(ValidationError, match="Duplicate OBB class threshold") as exc:
        ObbConfig(class_thresholds=thresholds)

    assert "Foreign_object" in str(exc.value)
    assert "Foreign_Object" in str(exc.value)


@pytest.mark.parametrize(
    ("removed", "added", "missing", "unexpected"),
    [
        ("Bridge", None, ["Bridge"], []),
        (None, "Unknown", [], ["Unknown"]),
        ("Foreign_object", "Foreign_Objects", ["Foreign_object"], ["Foreign_Objects"]),
    ],
)
def test_invalid_threshold_names_report_missing_and_unexpected(
    removed, added, missing, unexpected
):
    thresholds = ObbConfig().class_thresholds
    if removed is not None:
        del thresholds[removed]
    if added is not None:
        thresholds[added] = 0.25

    with pytest.raises(ValidationError) as exc:
        ObbConfig(class_thresholds=thresholds)

    assert f"missing={missing}" in str(exc.value)
    assert f"unexpected={unexpected}" in str(exc.value)


class FakeRuntime:
    def __init__(self, *, detections=None, error=None, missing=False):
        self.detections = (box(),) if detections is None else detections
        self.error = error
        self.missing = missing
        self.calls = 0

    def close(self):
        pass

    async def infer(self, images, *, deadline_at):
        self.calls += 1
        if self.error:
            raise self.error
        results = {key: self.detections for key in images}
        if self.missing:
            results.pop(next(iter(results)))
        return ObbBatchResult(
            results,
            1.5,
            "test-weight-sha256",
            "cpu",
            image_centers={key: (5.0, 5.0) for key in images},
        )


def service_case(tmp_path, rows, runtime=None):
    folder = tmp_path / "20261106120000"
    folder.mkdir()
    for i, value in enumerate(rows):
        value["image"] = f"image{i}.jpg"
        value["is_pass"] = "1"
        (folder / value["image"]).write_bytes(b"fake image; inference stub")
    with (folder / "board.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = json.loads((ROOT / "config/ai_server.json").read_text())
    settings["paths"] = {
        "watch_root": str(tmp_path / "input"),
        "external_output_root": str(tmp_path / "return"),
        "backup_output_root": str(tmp_path / "backup"),
    }
    settings["processing"]["image_name_template"] = "{image}"
    settings["minifi"]["output_path"] = str(tmp_path / "minifi")
    config = AppConfig.model_validate(settings)
    runtime = runtime or FakeRuntime()
    service = ObbInspectionService(
        config=config,
        result_root=tmp_path / "results",
        runtime=runtime,
        io_factory=AsyncTestIO,
    )
    return config, folder, service, runtime


def test_native_result_preserves_machine_contract_and_metadata(tmp_path):
    config, folder, service, runtime = service_case(
        tmp_path, [row(insp_hei="invalid", insp_x="999"), row(insp_area="49")]
    )
    original = (folder / "board.csv").read_bytes()
    manifest, path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.NORMAL
    assert manifest.csv_results[0].result_codes == (22, 23)
    assert manifest.inspection_metadata["site_id"] == "SX"
    assert manifest.inspection_metadata["limit_source"] == "component_rules"
    assert manifest.inspection_metadata["rule_order"] == [
        "yolo_obb",
        "volume",
        "area",
    ]
    assert PipelineResultManifest.read(path) == manifest
    processed = pd.read_csv(path.parent / manifest.csv_results[0].processed_csv)
    assert processed["ai_failed_rule"].fillna("").tolist() == ["", "area"]
    assert (
        json.loads(processed.iloc[0]["obb_retained_detections"])[0]["confidence"] == 0.8
    )
    records = prepare_publications(folder, path)
    for record in records.values():
        publish_csv(
            config,
            Path(record["result_path"]),
            record["result_sha256"],
            time.time() + 10,
            job_id=folder.name,
        )
    backup_result(folder, folder, path)
    returned = pd.read_csv(tmp_path / "return/board.csv")
    assert (
        returned.columns.tolist() == pd.read_csv(folder / "board.csv").columns.tolist()
    )
    assert returned["is_pass"].tolist() == [22, 23]
    assert (folder / "board.csv").read_bytes() == original
    assert PipelineResultManifest.read(folder / "ai_result/manifest.json") == manifest
    assert runtime.calls == 1


@pytest.mark.parametrize(
    "runtime",
    [
        FakeRuntime(error=TimeoutError("timeout")),
        FakeRuntime(error=RuntimeError("GPU failure")),
        FakeRuntime(missing=True),
    ],
)
def test_model_failure_or_missing_results_returns_all_23(tmp_path, runtime):
    _config, folder, service, _runtime = service_case(tmp_path, [row(), row()], runtime)
    manifest, _path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.FALLBACK
    assert manifest.csv_results[0].result_codes == (23, 23)
    assert manifest.inspection_metadata["site_id"] == "SX"


def test_empty_successful_detections_are_normal_ng(tmp_path):
    _config, folder, service, _runtime = service_case(
        tmp_path, [row()], FakeRuntime(detections=())
    )
    manifest, _path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.NORMAL
    assert manifest.csv_results[0].result_codes == (23,)


def test_center_with_disjoint_neighbors_continues_component_rules_and_records_selection(
    tmp_path,
):
    central = ObbDetection("Solder_Paste", 0.8, ((3, 3), (7, 3), (7, 7), (3, 7)))
    neighbors = (
        ObbDetection("Solder_Paste", 0.99, ((0, 0), (2, 0), (2, 2), (0, 2))),
        ObbDetection("Solder_Paste", 0.9, ((8, 8), (10, 8), (10, 10), (8, 10))),
    )
    _config, folder, service, _ = service_case(
        tmp_path,
        [row(), row(insp_vol=29)],
        FakeRuntime(detections=(*neighbors, central)),
    )
    manifest, path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.NORMAL
    assert manifest.csv_results[0].result_codes == (22, 23)
    assert manifest.counts == {"pass_count": 1, "fail_count": 1, "low volume": 1}
    policy = manifest.inspection_metadata["policy"]["appearance"]
    assert policy["center"] == "original_image_midpoint"
    processed = pd.read_csv(path.parent / manifest.csv_results[0].processed_csv)
    assert processed["ai_failed_rule"].fillna("").tolist() == ["", "volume"]
    assert processed["obb_ok_count"].tolist() == [3, 3]
    checks = json.loads(processed.iloc[0]["ai_rule_checks"])
    assert [check["rule"] for check in checks] == [
        "yolo_obb",
        "volume",
        "area",
    ]
    appearance = checks[0]
    assert appearance["image_center"] == [5.0, 5.0]
    assert appearance["center_count"] == 1
    assert appearance["overlap_count"] == 0
    assert appearance["ignored_solder_paste_count"] == 2
    assert appearance["selected_solder_paste"] == central.to_dict()
    assert len(json.loads(processed.iloc[0]["obb_raw_detections"])) == 3
    assert len(json.loads(processed.iloc[0]["obb_retained_detections"])) == 3


@pytest.mark.parametrize(
    "centers",
    [
        {},
        {"image0.jpg": (5, 5)},
        {"image0.jpg": (5, 5), "unexpected.jpg": (5, 5)},
        {"image0.jpg": (5, 5), "image1.jpg": (0, 5)},
        {"image0.jpg": (5, 5), "image1.jpg": (float("nan"), 5)},
        {"image0.jpg": (5, 5), "image1.jpg": None},
    ],
    ids=["missing", "partial", "wrong-key", "zero", "non-finite", "invalid-type"],
)
def test_missing_or_invalid_original_centers_use_whole_job_protection(
    tmp_path, centers
):
    class InvalidCenterRuntime(FakeRuntime):
        async def infer(self, images, *, deadline_at):
            result = await super().infer(images, deadline_at=deadline_at)
            return replace(result, image_centers=centers)

    _config, folder, service, _ = service_case(
        tmp_path, [row(), row()], InvalidCenterRuntime()
    )
    manifest, _path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.FALLBACK
    assert manifest.reason == "required_model_failure"
    assert manifest.csv_results[0].result_codes == (23, 23)


@pytest.mark.parametrize(
    "center,expected_code,expected_reason",
    [
        ((5.0, 5.0), 22, ""),
        ((15.0, 5.0), 23, "missing_center_solder_paste"),
    ],
)
def test_judgment_uses_each_original_image_center_from_model(
    tmp_path, center, expected_code, expected_reason
):
    class SizedRuntime(FakeRuntime):
        async def infer(self, images, *, deadline_at):
            result = await super().infer(images, deadline_at=deadline_at)
            return replace(result, image_centers={key: center for key in images})

    _config, folder, service, _ = service_case(tmp_path, [row()], SizedRuntime())
    manifest, path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.NORMAL
    assert manifest.csv_results[0].result_codes == (expected_code,)
    processed = pd.read_csv(path.parent / manifest.csv_results[0].processed_csv)
    assert processed["ai_reason"].fillna("").tolist() == [expected_reason]
    assert json.loads(processed.iloc[0]["ai_rule_checks"])[0]["image_center"] == list(
        center
    )


def test_missing_measurement_uses_whole_job_fail_safe(tmp_path):
    _config, folder, service, _runtime = service_case(
        tmp_path, [row(), row(insp_area="")]
    )
    manifest, _path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.FALLBACK
    assert manifest.reason == "invalid_measurement"
    assert manifest.csv_results[0].result_codes == (23, 23)


def test_supplied_measurements_apply_component_rules(tmp_path):
    # Seven supplied SPI measurement rows, omitting board identifiers.
    samples = [
        ("C0626", 184.474, 110.748, 49.971, 0.008, 0),
        ("R0700", 46.744, 67.543, 20.762, 0.014, -0.007),
        ("U0702", 47.032, 50.634, 27.866, -0.003, -0.020),
        ("U1200", 35.040, 32.268, 32.578, 0.031, -0.016),
        ("U0702", 48.180, 49.748, 29.054, 0, -0.031),
        ("U0702", 41.094, 40.525, 30.421, 0.004, -0.028),
        ("U0702", 48.753, 59.242, 24.688, -0.004, -0.028),
    ]
    rows = [
        row(
            component_name=c,
            insp_vol=v,
            insp_area=a,
            insp_hei=h,
            insp_x=x,
            insp_y=y,
            vol_l_ng=150,
            vol_h_ng=50,
            area_l_ng=150,
            area_h_ng=50,
            hei_l_ng=150,
            hei_h_ng=50,
        )
        for c, v, a, h, x, y in samples
    ]
    _config, folder, service, _runtime = service_case(tmp_path, rows)
    manifest, path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.csv_results[0].result_codes == (23, 22, 22, 23, 23, 23, 22)
    processed = pd.read_csv(path.parent / manifest.csv_results[0].processed_csv)
    assert processed["ai_failed_rule"].fillna("").tolist() == [
        "volume",
        "",
        "",
        "area",
        "area",
        "area",
        "",
    ]


def test_same_component_ignores_machine_limits_and_preserves_csv(tmp_path):
    rows = [
        row(component_name="U0500", vol_l_ng=200, vol_h_ng=300),
        row(component_name="U0500", vol_l_ng=150, vol_h_ng=50),
    ]
    _config, folder, service, _runtime = service_case(tmp_path, rows)
    original = (folder / "board.csv").read_bytes()
    manifest, path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.csv_results[0].result_codes == (22, 22)
    processed = pd.read_csv(path.parent / manifest.csv_results[0].processed_csv)
    assert processed["ai_limit_source"].tolist() == [
        "component_rules",
        "component_rules",
    ]
    assert processed["ai_component_rule"].tolist() == ["U0500", "U0500"]
    checks = json.loads(processed.iloc[1]["ai_rule_checks"])
    assert (checks[1]["low"], checks[1]["high"]) == (1, 180)
    assert processed["vol_l_ng"].tolist() == [200, 150]
    assert processed["vol_h_ng"].tolist() == [300, 50]
    assert (folder / "board.csv").read_bytes() == original


def test_measurement_judge_only_requires_component_name_volume_and_area(tmp_path):
    _config, folder, service, _runtime = service_case(
        tmp_path,
        [
            {"component_name": "R0001", "insp_vol": "30", "insp_area": "50"},
        ],
    )
    manifest, _path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.NORMAL
    assert manifest.csv_results[0].result_codes == (22,)


def test_saved_backup_above_image_limit_returns_all_23_without_loading_model_inputs(
    tmp_path, monkeypatch
):
    import app.application.inference.obb_service as module

    config, folder, service, runtime = service_case(tmp_path, [row(), row()])
    config.processing.folder_images_num_threshold = 1

    def unexpected(*args):
        pytest.fail("An oversized saved backup must bypass inference input loading")

    monkeypatch.setattr(module, "load_inputs", unexpected)
    manifest, path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.FALLBACK
    assert manifest.reason == "images_exceed_threshold" and runtime.calls == 0
    assert manifest.csv_results[0].result_codes == (23, 23)
    assert PipelineResultManifest.read(path) == manifest


@pytest.mark.parametrize("error", [None, RuntimeError("model unavailable")])
def test_result_metadata_records_crop_ratio_for_normal_and_protection_results(
    tmp_path, error
):
    config, folder, service, _runtime = service_case(
        tmp_path, [row()], FakeRuntime(error=error)
    )
    manifest, path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.inspection_metadata["preprocessing"] == {
        "center_crop_ratio": config.inference.obb.center_crop_ratio,
        "crop_size_rounding": "floor_min_1",
        "crop_alignment": "center_floor",
        "obb_coordinate_space": "original_image",
    }
    assert (
        PipelineResultManifest.read(path).inspection_metadata
        == manifest.inspection_metadata
    )


def test_different_crop_ratios_produce_distinct_rule_versions(tmp_path):
    config, folder, cropped, _runtime = service_case(tmp_path, [row()])
    config = config.model_copy(deep=True)
    config.inference.obb.center_crop_ratio = 1.0
    full = ObbInspectionService(
        config=config,
        result_root=tmp_path / "full-results",
        runtime=FakeRuntime(),
        io_factory=AsyncTestIO,
    )
    cropped_result, _ = asyncio.run(
        cropped.prepare(job_id=folder.name, staged_folder=folder)
    )
    full_result, _ = asyncio.run(full.prepare(job_id=folder.name, staged_folder=folder))
    assert (
        cropped_result.inspection_metadata["preprocessing"]["center_crop_ratio"] == 0.6
    )
    assert full_result.inspection_metadata["preprocessing"]["center_crop_ratio"] == 1.0
    assert (
        cropped_result.inspection_metadata["rule_version"]
        != full_result.inspection_metadata["rule_version"]
    )


def test_configured_tolerance_uses_original_size_and_is_audited_and_versioned(tmp_path):
    target = ObbDetection("Solder_Paste", 0.8, ((5.4, 4), (7, 4), (7, 6), (5.4, 6)))
    config, folder, tolerant, _ = service_case(
        tmp_path, [row()], FakeRuntime(detections=(target,))
    )
    strict_config = config.model_copy(deep=True)
    strict_config.inference.obb.center_tolerance_ratio = 0
    strict = ObbInspectionService(
        config=strict_config,
        result_root=tmp_path / "strict-results",
        runtime=FakeRuntime(detections=(target,)),
        io_factory=AsyncTestIO,
    )
    tolerant_result, path = asyncio.run(
        tolerant.prepare(job_id=folder.name, staged_folder=folder)
    )
    strict_result, _ = asyncio.run(
        strict.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert tolerant_result.outcome is strict_result.outcome is PipelineOutcome.NORMAL
    assert tolerant_result.csv_results[0].result_codes == (22,)
    assert strict_result.csv_results[0].result_codes == (23,)
    metadata = tolerant_result.inspection_metadata
    assert metadata["policy"]["appearance"]["center_tolerance_ratio"] == 0.05
    assert (
        metadata["policy"]["appearance"]["center_tolerance_basis"]
        == "original_image_shorter_side"
    )
    assert (
        strict_result.inspection_metadata["policy"]["appearance"][
            "center_tolerance_ratio"
        ]
        == 0
    )
    assert metadata["rule_version"] != strict_result.inspection_metadata["rule_version"]
    assert metadata["preprocessing"]["center_crop_ratio"] == 0.6
    processed = pd.read_csv(path.parent / tolerant_result.csv_results[0].processed_csv)
    check = json.loads(processed.iloc[0]["ai_rule_checks"])[0]
    assert check["image_center"] == [5.0, 5.0]
    # Original size is 10x10: tolerance is 0.5 px, not the cropped 6x6's 0.3 px.
    assert check["center_tolerance_px"] == 0.5
    assert check["center_selection_mode"] == "within_tolerance"
    assert check["selected_solder_paste"] == target.to_dict()
