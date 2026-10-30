"""Unit tests for the derived-metrics domain service."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.domain.services.derived_metrics import DerivedMetricsCalculator


def test_pad_area_computed_when_width_and_length_present() -> None:
    df = pd.DataFrame({"Width": [2.0], "Length": [4.0]})
    result = DerivedMetricsCalculator().add_derived_columns(df)
    assert "pad_area" in result.df.columns
    assert result.df["pad_area"].iloc[0] == math.pi * 2.0 * 4.0 / 4.0


def test_cover_percent_when_paste_pixels_and_pad_area_present() -> None:
    df = pd.DataFrame({"Width": [2.0], "Length": [4.0], "paste_pixels": [100.0]})
    result = DerivedMetricsCalculator().add_derived_columns(df)
    assert "cover%" in result.df.columns
    pad_area = math.pi * 2.0 * 4.0 / 4.0
    assert result.df["cover%"].iloc[0] == 100.0 * 0.8246 * 100.0 / pad_area


def test_missing_paste_pixels_does_not_crash_and_warns() -> None:
    df = pd.DataFrame({"Width": [2.0], "Length": [4.0]})
    result = DerivedMetricsCalculator().add_derived_columns(df)
    assert "pad_area" in result.df.columns
    assert "cover%" not in result.df.columns
    assert any("cover%" in warning for warning in result.warnings)


def test_missing_width_length_does_not_crash_and_warns() -> None:
    df = pd.DataFrame({"paste_pixels": [100.0]})
    result = DerivedMetricsCalculator().add_derived_columns(df)
    assert "pad_area" not in result.df.columns
    assert "cover%" not in result.df.columns
    assert any("pad_area" in warning for warning in result.warnings)


def test_segmentation_ratio_preserved_without_legacy_conversion() -> None:
    frame = pd.DataFrame(
        {"Width": [2], "Length": [4], "paste_pixels": [999], "cover%": [185.0]}
    )
    result = DerivedMetricsCalculator().add_derived_columns(
        frame, cover_from_model=True
    )
    assert result.df["cover%"].iloc[0] == 185.0


def test_old_ratio_is_recomputed_when_not_provided_by_model() -> None:
    frame = pd.DataFrame({"pad_area": [10], "paste_pixels": [10], "cover%": [999]})
    result = DerivedMetricsCalculator().add_derived_columns(frame)
    assert result.df["cover%"].iloc[0] == pytest.approx(82.46)
