"""Unit tests for the derived-metrics domain service."""

from __future__ import annotations

import math

import pandas as pd

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
