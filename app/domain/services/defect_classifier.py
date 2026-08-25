"""Domain service: rule-based defect classification and pass/fail decision.

Pure pandas logic. Thresholds come from a config object; there is no filesystem,
HTTP, or FastAPI dependency.
"""

from __future__ import annotations

import pandas as pd

from app.core.config import DefectRuleConfig, DefectRuleName
from app.domain.entities.defect import FAIL_CODE, PASS_CODE, DefectLabel

_DEFECT_COLUMN = "ai_defect_name"
_IS_PASS_COLUMN = "is_pass"

# Input columns consulted by the rules.
_ANOMALY_SCORE_COLUMN = "anomaly_score"
_INSP_VOL_COLUMN = "insp_vol"
_VOL_L_NG_COLUMN = "vol_l_ng"
_VOL_H_NG_COLUMN = "vol_h_ng"
_COVER_COLUMN = "cover%"
_MIN_PAD_DISTANCE_COLUMN = "min_pad_distance"
_INSP_HEI_COLUMN = "insp_hei"


class DefectClassifier:
    """Assigns ``ai_defect_name`` using ordered, first-match-wins rules.

    Rule priority comes from ``DefectRuleConfig.rule_order``. Only the first
    matching enabled rule per row assigns a label. The legacy default order is:

    1. ``anomaly_score > anomaly_threshold`` -> ``FM/color``
    2. ``insp_vol > vol_h_ng + high_vol_offset`` -> ``high vol``
    3. ``insp_vol < vol_l_ng + low_vol_offset`` -> ``low vol``
    4. ``cover% > high_cover_threshold`` -> ``high cover``
    5. ``min_pad_distance < short_distance_threshold`` -> ``short distance``
    6. ``insp_hei > high_paste_height_threshold`` -> ``high paste``

    Rules whose input columns are absent are skipped without error.
    """

    def __init__(self, config: DefectRuleConfig) -> None:
        """Initialize with the defect thresholds/offsets.

        Args:
            config: Threshold configuration (see :class:`DefectRuleConfig`).
        """
        self._config = config

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with ``ai_defect_name`` populated.

        Args:
            df: Frame with any subset of the rule input columns.

        Returns:
            A new frame with the ``ai_defect_name`` column.
        """
        out = df.copy()
        if _DEFECT_COLUMN not in out.columns:
            out[_DEFECT_COLUMN] = ""

        for rule_name in self._config.rule_order:
            self._apply_rule(out, rule_name)
        return out

    def _apply_rule(self, out: pd.DataFrame, rule_name: DefectRuleName) -> None:
        """Apply one configured rule to rows which are still unassigned."""
        if rule_name == "anomaly":
            self._apply_above(
                out,
                _ANOMALY_SCORE_COLUMN,
                self._config.anomaly_threshold,
                DefectLabel.FM_COLOR,
            )
        elif rule_name == "high_vol":
            self._apply_volume_rule(out, high=True)
        elif rule_name == "low_vol":
            self._apply_volume_rule(out, high=False)
        elif rule_name == "high_cover":
            self._apply_above(
                out,
                _COVER_COLUMN,
                self._config.high_cover_threshold,
                DefectLabel.HIGH_COVER,
            )
        elif rule_name == "short_distance":
            self._apply_below(
                out,
                _MIN_PAD_DISTANCE_COLUMN,
                self._config.short_distance_threshold,
                DefectLabel.SHORT_DISTANCE,
            )
        elif rule_name == "high_paste":
            self._apply_above(
                out,
                _INSP_HEI_COLUMN,
                self._config.high_paste_height_threshold,
                DefectLabel.HIGH_PASTE,
            )

    @staticmethod
    def _unassigned(out: pd.DataFrame) -> pd.Series:
        """Return a boolean mask of rows without a defect label yet."""
        return out[_DEFECT_COLUMN].astype(str).str.len() == 0

    def _apply_above(
        self,
        out: pd.DataFrame,
        column: str,
        threshold: float,
        label: DefectLabel,
    ) -> None:
        """Label unassigned rows where ``column > threshold`` (in place)."""
        if column not in out.columns:
            return
        values = pd.to_numeric(out[column], errors="coerce")
        mask = self._unassigned(out) & (values > threshold)
        out.loc[mask, _DEFECT_COLUMN] = label.value

    def _apply_below(
        self,
        out: pd.DataFrame,
        column: str,
        threshold: float,
        label: DefectLabel,
    ) -> None:
        """Label unassigned rows where ``column < threshold`` (in place)."""
        if column not in out.columns:
            return
        values = pd.to_numeric(out[column], errors="coerce")
        mask = self._unassigned(out) & (values < threshold)
        out.loc[mask, _DEFECT_COLUMN] = label.value

    def _apply_volume_rule(self, out: pd.DataFrame, *, high: bool) -> None:
        """Apply one configured high- or low-volume rule (in place)."""
        needed = {_INSP_VOL_COLUMN, _VOL_L_NG_COLUMN, _VOL_H_NG_COLUMN}
        if not needed.issubset(out.columns):
            return
        insp_vol = pd.to_numeric(out[_INSP_VOL_COLUMN], errors="coerce")
        if high:
            threshold = (
                pd.to_numeric(out[_VOL_H_NG_COLUMN], errors="coerce")
                + self._config.high_vol_offset
            )
            mask = self._unassigned(out) & (insp_vol > threshold)
            label = DefectLabel.HIGH_VOL
        else:
            threshold = (
                pd.to_numeric(out[_VOL_L_NG_COLUMN], errors="coerce")
                + self._config.low_vol_offset
            )
            mask = self._unassigned(out) & (insp_vol < threshold)
            label = DefectLabel.LOW_VOL
        out.loc[mask, _DEFECT_COLUMN] = label.value


def add_is_pass(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``is_pass`` derived from ``ai_defect_name``.

    ``is_pass`` is :data:`PASS_CODE` (22) when ``ai_defect_name`` is empty or
    whitespace, otherwise :data:`FAIL_CODE` (23).

    Args:
        df: Frame that may contain an ``ai_defect_name`` column.

    Returns:
        A new frame with the ``is_pass`` column.
    """
    out = df.copy()
    if _DEFECT_COLUMN not in out.columns:
        out[_DEFECT_COLUMN] = ""
    is_pass = pd.Series(PASS_CODE, index=out.index, dtype="Int64")
    has_defect = out[_DEFECT_COLUMN].astype(str).str.strip() != ""
    is_pass.loc[has_defect] = FAIL_CODE
    out[_IS_PASS_COLUMN] = is_pass
    return out
