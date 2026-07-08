"""Domain service: rule-based defect classification (skeleton)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from app.core.config import AppConfig


class DefectClassifier:
    """Assigns ``ai_defect_name`` and ``is_pass`` using ordered rules.

    TODO(phase-3): Migrate logic from ``ai_server_fastapi.add_ai_defect_name``
    and ``add_is_pass``. Keep the rule order and thresholds identical to
    preserve behavior.
    """

    def __init__(self, config: "AppConfig") -> None:
        """Initialize with the thresholds/offsets from config."""
        self._config = config

    def classify(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Return ``df`` with ``ai_defect_name`` and ``is_pass`` populated."""
        raise NotImplementedError("Migrated in a later phase.")
