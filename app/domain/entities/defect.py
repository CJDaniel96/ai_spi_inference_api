"""Domain entity: defect classification labels and pass/fail codes."""

from __future__ import annotations

from enum import Enum


class DefectLabel(str, Enum):
    """Canonical ``ai_defect_name`` values produced by classification.

    Values must stay byte-compatible with the legacy pipeline and downstream
    analytics (see ``ai_server_fastapi.add_ai_defect_name``).
    """

    FM_COLOR = "FM/color"
    HIGH_VOL = "high vol"
    LOW_VOL = "low vol"
    HIGH_COVER = "high cover"
    SHORT_DISTANCE = "short distance"
    HIGH_PASTE = "high paste"


# ``is_pass`` codes (legacy contract): 22 = pass (no defect), 23 = fail (defect).
PASS_CODE = 22
FAIL_CODE = 23
