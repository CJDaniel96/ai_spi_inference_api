"""Unit tests for the CSV merger domain service."""

from __future__ import annotations

import pandas as pd
import pytest

from app.domain.entities.inference_result import ModelInferenceResult
from app.domain.services.csv_merger import CsvMerger, build_image_name


def test_build_image_name_basic() -> None:
    assert build_image_name(1, 1426) == "0_1426.jpg"


def test_build_image_name_preserves_pad_string() -> None:
    assert build_image_name(2, "001") == "1_001.jpg"


def test_build_image_name_raises_on_non_numeric_array_id() -> None:
    with pytest.raises(ValueError, match="Array_id"):
        build_image_name("abc", 1)


def test_add_image_name_column_missing_array_id_raises() -> None:
    df = pd.DataFrame({"Pad_no": [1, 2]})
    with pytest.raises(ValueError, match="Array_id"):
        CsvMerger().add_image_name_column(df)


def test_add_image_name_column_missing_pad_no_raises() -> None:
    df = pd.DataFrame({"Array_id": [1, 2]})
    with pytest.raises(ValueError, match="Pad_no"):
        CsvMerger().add_image_name_column(df)


def test_add_image_name_column_builds_expected() -> None:
    df = pd.DataFrame({"Array_id": [1, 2], "Pad_no": ["001", 1426]})
    out = CsvMerger().add_image_name_column(df)
    assert out["img_name"].tolist() == ["0_001.jpg", "1_1426.jpg"]


def _df_with_images() -> pd.DataFrame:
    """Return a 2-row frame with img_name ``0_100.jpg`` / ``0_101.jpg``."""
    df = pd.DataFrame({"Array_id": [1, 1], "Pad_no": [100, 101]})
    return CsvMerger().add_image_name_column(df)


def test_merge_anomaly_result() -> None:
    result = ModelInferenceResult(
        name="anomaly",
        target_column="anomaly_score",
        results={"0_100.jpg": 0.95, "0_101.jpg": 0.1},
    )
    out = CsvMerger().merge_model_results(_df_with_images(), [result])
    assert out["anomaly_score"].tolist() == [0.95, 0.1]


def test_merge_distance_result() -> None:
    result = ModelInferenceResult(
        name="distance",
        target_column="min_pad_distance",
        results={"0_100.jpg": 5.0, "0_101.jpg": 7.0},
    )
    out = CsvMerger().merge_model_results(_df_with_images(), [result])
    assert out["min_pad_distance"].tolist() == [5.0, 7.0]


def test_merge_missing_result_fills_nan() -> None:
    result = ModelInferenceResult(
        name="distance", target_column="min_pad_distance", results={}
    )
    out = CsvMerger().merge_model_results(_df_with_images(), [result])
    assert "min_pad_distance" in out.columns
    assert out["min_pad_distance"].isna().all()


def test_merge_partial_result_fills_nan_for_unmatched_rows() -> None:
    result = ModelInferenceResult(
        name="anomaly", target_column="anomaly_score", results={"0_100.jpg": 0.95}
    )
    out = CsvMerger().merge_model_results(_df_with_images(), [result])
    assert out["anomaly_score"].iloc[0] == 0.95
    assert pd.isna(out["anomaly_score"].iloc[1])


def test_merge_unknown_image_does_not_add_rows() -> None:
    result = ModelInferenceResult(
        name="anomaly",
        target_column="anomaly_score",
        results={"0_100.jpg": 0.95, "9_999.jpg": 0.5},
    )
    out = CsvMerger().merge_model_results(_df_with_images(), [result])
    assert len(out) == 2
    assert out["anomaly_score"].iloc[0] == 0.95
    assert pd.isna(out["anomaly_score"].iloc[1])


def test_merge_requires_img_name_column() -> None:
    df = pd.DataFrame({"Array_id": [1]})
    result = ModelInferenceResult(name="x", target_column="y", results={})
    with pytest.raises(ValueError, match="img_name"):
        CsvMerger().merge_model_results(df, [result])
