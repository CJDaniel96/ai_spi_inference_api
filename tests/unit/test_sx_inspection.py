"""Per-row machine limits, confidence filtering and native result publication."""

import asyncio
import csv
import json
import time
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from app.application.inference.obb_service import ObbInspectionService
from app.application.operations import backup_result, prepare_publications, publish_csv
from app.core.config import AppConfig
from app.core.inspection_config import ObbConfig
from app.domain.entities.obb_result import OBB_CLASSES, ObbBatchResult, ObbDetection
from app.domain.entities.pipeline_result import PipelineOutcome, PipelineResultManifest
from app.domain.services.appearance_classifier import AppearanceClassifier
from app.domain.services.component_judge import MEASUREMENTS, ComponentJudge
from tests.helpers import AsyncTestIO

ROOT = Path(__file__).resolve().parents[2]
POLYGON = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def box(name="Solder_Paste", confidence=0.8):
    return ObbDetection(name, confidence, POLYGON)


def judge():
    return ComponentJudge(ObbConfig().class_thresholds)


def row(**changes):
    return {
        "component_name": "others",
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
    result = judge().appearance.classify(detections)
    assert (result.passed, result.ok_count, result.ng_count) == (passed, ok, ng)


@pytest.mark.parametrize("name", OBB_CLASSES[1:])
def test_each_ng_class_blocks_appearance(name):
    result = judge().judge(row(), (box(), box(name, 0.3)))
    assert result.failed_rule == "yolo_obb" and result.defect == name


def test_ng_thresholds_are_independent():
    thresholds = ObbConfig().class_thresholds
    thresholds["Bridge"] = 0.9
    classifier = AppearanceClassifier(thresholds)
    assert classifier.classify((box(), box("Bridge", 0.8))).passed
    assert not classifier.classify((box(), box("Stain", 0.3))).passed


@pytest.mark.parametrize(
    ("name", "measurement", "low_column", "high_column"),
    MEASUREMENTS,
)
@pytest.mark.parametrize(("left", "right"), [(40, 160), (160, 40), (75, 75)])
@pytest.mark.parametrize("position", ["lower", "upper", "below", "above"])
def test_csv_intervals_normalize_both_orders_and_include_boundaries(
    name, measurement, low_column, high_column, left, right, position
):
    low, high = min(left, right), max(left, right)
    value = {"lower": low, "upper": high, "below": low - 0.001, "above": high + 0.001}[
        position
    ]
    data = row(**{measurement: value, low_column: left, high_column: right})
    original = data.copy()
    decision = judge().judge(data, (box(),))
    assert decision.passed == (position in ("lower", "upper"))
    if not decision.passed:
        assert decision.failed_rule == name
        assert decision.defect == f"{'low' if position == 'below' else 'high'} {name}"
    check = next(check for check in decision.checks if check["rule"] == name)
    assert (check["low"], check["high"]) == (low, high)
    assert check["limits_reversed"] == (left > right)
    assert check["source_limits"] == {low_column: left, high_column: right}
    assert data == original


@pytest.mark.parametrize(
    ("changes", "rule"),
    [
        (
            {"insp_vol": "20", "insp_area": "20", "insp_hei": "20", "insp_x": "1"},
            "volume",
        ),
        ({"insp_area": "20", "insp_hei": "20", "insp_x": "1"}, "area"),
        ({"insp_hei": "20", "insp_x": "1"}, "height"),
        ({"insp_x": "-.051"}, "shift"),
        ({"insp_y": ".051"}, "shift"),
    ],
)
def test_rule_order_and_signed_displacements(changes, rule):
    decision = judge().judge(row(**changes), (box(),))
    assert decision.failed_rule == rule
    order = [check["rule"] for check in decision.checks]
    assert order == ["yolo_obb", "volume", "area", "height", "shift"][: len(order)]


def test_csv_limits_override_previous_component_specific_values():
    data = row(
        component_name="U0500",
        insp_vol=55,
        insp_area=55,
        insp_hei=55,
        vol_l_ng=65,
        vol_h_ng=55,
        area_l_ng=55,
        area_h_ng=65,
        hei_l_ng=65,
        hei_h_ng=55,
        shift_x_ng=0.06,
        shift_y_ng=0.07,
        insp_x=-0.06,
        insp_y=0.07,
    )
    assert judge().judge(data, (box(),)).passed


@pytest.mark.parametrize(
    "column",
    [
        "vol_l_ng",
        "vol_h_ng",
        "area_l_ng",
        "area_h_ng",
        "hei_l_ng",
        "hei_h_ng",
        "shift_x_ng",
        "shift_y_ng",
    ],
)
@pytest.mark.parametrize("value", ["", "NaN", "inf", "invalid", None])
def test_invalid_csv_limit_cannot_become_a_pass(column, value):
    with pytest.raises(ValueError, match="limit"):
        judge().judge(row(**{column: value}), (box(),))


@pytest.mark.parametrize("column", ["shift_x_ng", "shift_y_ng"])
def test_negative_shift_limit_is_invalid(column):
    with pytest.raises(ValueError, match="limits"):
        judge().judge(row(**{column: -0.05}), (box(),))


@pytest.mark.parametrize(
    ("x", "y", "passed"),
    [
        (-0.03, 0.07, True),
        (0.03, -0.07, True),
        (-0.031, 0, False),
        (0, -0.071, False),
    ],
)
def test_shift_axes_use_their_own_csv_limits(x, y, passed):
    assert (
        judge()
        .judge(row(insp_x=x, insp_y=y, shift_x_ng=0.03, shift_y_ng=0.07), (box(),))
        .passed
        == passed
    )


def test_zero_shift_limits_only_accept_zero_displacement():
    assert judge().judge(row(shift_x_ng=0, shift_y_ng=0), (box(),)).passed
    assert (
        not judge()
        .judge(row(shift_x_ng=0, shift_y_ng=0, insp_y=0.001), (box(),))
        .passed
    )


@pytest.mark.parametrize(
    "column", ["insp_vol", "insp_area", "insp_hei", "insp_x", "insp_y"]
)
@pytest.mark.parametrize("value", ["", "NaN", "inf", "invalid"])
def test_invalid_measurement_is_never_a_pass(column, value):
    with pytest.raises(ValueError, match="measurement"):
        judge().judge(row(**{column: value}), (box(),))


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
    assert classifier.classify((box(), box("Foreign_object", 0.649))).passed
    decision = classifier.classify((box(), box("Foreign_object", 0.65)))
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
        return ObbBatchResult(results, 1.5, "test-weight-sha256", "cpu")


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
        "state_root": str(tmp_path / "state"),
        "external_output_root": str(tmp_path / "return"),
        "backup_output_root": str(tmp_path / "backup"),
        "history_root": str(tmp_path / "history"),
    }
    settings["processing"]["image_name_template"] = "{image}"
    settings["database"]["path"] = str(tmp_path / "records/audit.sqlite3")
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
        tmp_path, [row(), row(insp_hei="49")]
    )
    original = (folder / "board.csv").read_bytes()
    manifest, path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.NORMAL
    assert manifest.csv_results[0].result_codes == (22, 23)
    assert manifest.inspection_metadata["site_id"] == "SX"
    assert manifest.inspection_metadata["limit_source"] == "csv"
    assert manifest.inspection_metadata["rule_order"] == [
        "yolo_obb",
        "volume",
        "area",
        "height",
        "shift",
    ]
    assert PipelineResultManifest.read(path) == manifest
    processed = pd.read_csv(path.parent / manifest.csv_results[0].processed_csv)
    assert processed["ai_failed_rule"].fillna("").tolist() == ["", "height"]
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


def test_supplied_measurements_fail_volume_before_other_checks(tmp_path):
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
    assert manifest.csv_results[0].result_codes == (23,) * 7
    processed = pd.read_csv(path.parent / manifest.csv_results[0].processed_csv)
    assert processed["ai_failed_rule"].tolist() == ["volume"] * 7
    assert processed["ai_defect_name"].tolist() == ["high volume"] + ["low volume"] * 6


def test_same_component_uses_each_rows_limits_and_preserves_csv(tmp_path):
    rows = [
        row(component_name="U0500", vol_l_ng=200, vol_h_ng=300),
        row(component_name="U0500", vol_l_ng=150, vol_h_ng=50),
    ]
    _config, folder, service, _runtime = service_case(tmp_path, rows)
    original = (folder / "board.csv").read_bytes()
    manifest, path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.csv_results[0].result_codes == (23, 22)
    processed = pd.read_csv(path.parent / manifest.csv_results[0].processed_csv)
    assert processed["ai_limit_source"].tolist() == ["csv", "csv"]
    checks = json.loads(processed.iloc[1]["ai_rule_checks"])
    assert checks[1]["limits_reversed"] is True
    assert checks[1]["source_limits"] == {"vol_l_ng": 150, "vol_h_ng": 50}
    assert processed["vol_l_ng"].tolist() == [200, 150]
    assert processed["vol_h_ng"].tolist() == [300, 50]
    assert (folder / "board.csv").read_bytes() == original


def test_missing_csv_limit_uses_whole_job_fail_safe_without_excel_fallback(tmp_path):
    _config, folder, service, _runtime = service_case(
        tmp_path, [row(), row(vol_h_ng="")]
    )
    manifest, _path = asyncio.run(
        service.prepare(job_id=folder.name, staged_folder=folder)
    )
    assert manifest.outcome is PipelineOutcome.FALLBACK
    assert manifest.csv_results[0].result_codes == (23, 23)
    assert "limit" in manifest.errors[0]
