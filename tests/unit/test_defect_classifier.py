"""Unit tests for the defect classifier domain service."""

from __future__ import annotations

import pandas as pd

from app.core.config import DefectRuleConfig
from app.domain.services.defect_classifier import DefectClassifier, add_is_pass


def _config() -> DefectRuleConfig:
    return DefectRuleConfig(
        anomaly_threshold=0.9,
        high_cover_threshold=180.0,
        short_distance_threshold=6.8,
        low_vol_offset=-10.0,
        high_vol_offset=20.0,
        high_paste_height_threshold=200.0,
    )


def _classify(df: pd.DataFrame) -> list[str]:
    return DefectClassifier(_config()).classify(df)["ai_defect_name"].tolist()


def test_anomaly_threshold() -> None:
    df = pd.DataFrame({"anomaly_score": [0.95, 0.5]})
    assert _classify(df) == ["FM/color", ""]


def test_high_vol() -> None:
    # insp_vol 130 > vol_h_ng 100 + high_vol_offset 20 (= 120)
    df = pd.DataFrame({"insp_vol": [130.0], "vol_l_ng": [50.0], "vol_h_ng": [100.0]})
    assert _classify(df) == ["high vol"]


def test_low_vol() -> None:
    # insp_vol 30 < vol_l_ng 50 + low_vol_offset -10 (= 40)
    df = pd.DataFrame({"insp_vol": [30.0], "vol_l_ng": [50.0], "vol_h_ng": [100.0]})
    assert _classify(df) == ["low vol"]


def test_high_cover() -> None:
    df = pd.DataFrame({"cover%": [200.0]})
    assert _classify(df) == ["high cover"]


def test_short_distance_uses_less_than() -> None:
    # 5.0 < 6.8 -> short distance; 7.0 is not < 6.8 -> no label
    df = pd.DataFrame({"min_pad_distance": [5.0, 7.0]})
    assert _classify(df) == ["short distance", ""]


def test_high_paste() -> None:
    df = pd.DataFrame({"insp_height": [250.0]})
    assert _classify(df) == ["high paste"]


def test_priority_anomaly_beats_high_vol() -> None:
    df = pd.DataFrame(
        {
            "anomaly_score": [0.95],
            "insp_vol": [130.0],
            "vol_l_ng": [50.0],
            "vol_h_ng": [100.0],
        }
    )
    assert _classify(df) == ["FM/color"]


def test_no_relevant_columns_yields_empty_labels() -> None:
    df = pd.DataFrame({"other": [1, 2]})
    assert _classify(df) == ["", ""]


def test_add_is_pass_empty_defect_is_22() -> None:
    df = pd.DataFrame({"ai_defect_name": ["", "  "]})
    assert add_is_pass(df)["is_pass"].tolist() == [22, 22]


def test_add_is_pass_non_empty_defect_is_23() -> None:
    df = pd.DataFrame({"ai_defect_name": ["FM/color", "high vol"]})
    assert add_is_pass(df)["is_pass"].tolist() == [23, 23]


def test_add_is_pass_missing_column_defaults_to_22() -> None:
    df = pd.DataFrame({"x": [1]})
    assert add_is_pass(df)["is_pass"].tolist() == [22]
