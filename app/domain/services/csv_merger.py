"""Domain service: merge model results into CSV frames (skeleton)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from app.domain.entities.inference_result import InferenceResult


class CsvMerger:
    """Merges scalar model results into a CSV DataFrame.

    TODO(phase-3): Migrate ``ai_server_fastapi.merge_scalar_result`` and
    ``build_img_name_column`` here.
    """

    def merge(
        self,
        df: pd.DataFrame,
        result: InferenceResult,
    ) -> pd.DataFrame:
        """Return ``df`` with the result's scalar column merged in by image name."""
        raise NotImplementedError("Migrated in a later phase.")
