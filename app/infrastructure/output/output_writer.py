"""Infrastructure: primary/backup/processed CSV output writer and path policy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.infrastructure.output.sinic_csv_output import SinicCsvOutput
from app.infrastructure.repositories.csv_repository import CsvRepository

_AI_SUBFOLDER = "AI"
_PROCESSED_SUFFIX = "_processed"
_IS_PASS_COLUMN = "is_pass"
_MODE_FULL_AI_COLUMNS = "full_ai_columns"
_LAYOUT_MACHINE_RETURN = "machine_return"


class OutputWriter:
    """Writes the primary, backup, and processed CSVs for one input CSV.

    Path policy:
      * primary:   ``{external_output_root}/{job_name}/AI/{csv_name}``
      * backup:    ``{backup_output_root}/{year}/{month}/{job_name}/{csv_name}``
      * processed: ``{backup_output_root}/{year}/{month}/{job_name}/``
        ``{stem}_processed{suffix}``

    ``primary_csv_mode`` selects the primary CSV contents:
      * ``is_pass_only``    -> original columns with only ``is_pass`` updated
      * ``full_ai_columns`` -> the full processed frame (all AI columns)

    The backup CSV is always the ``is_pass_only`` frame and the processed CSV is
    always the full frame, preserving existing behavior.
    """

    def __init__(
        self,
        *,
        external_output_root: str,
        backup_output_root: str,
        primary_csv_mode: str,
        csv_repository: CsvRepository,
        primary_path_layout: str = "legacy_ai_subfolder",
        preserve_job_folder: bool = False,
        require_existing_is_pass: bool = True,
    ) -> None:
        """Initialize with output roots, the primary mode, and a CSV writer."""
        self._external_root = Path(external_output_root)
        self._backup_root = Path(backup_output_root)
        self._mode = primary_csv_mode
        self._path_layout = primary_path_layout
        self._csv = csv_repository
        self._machine_output = SinicCsvOutput(
            self._external_root,
            preserve_job_folder=preserve_job_folder,
            require_existing_result_column=require_existing_is_pass,
        )

    def write(
        self,
        *,
        job_name: str,
        csv_name: str,
        year: str,
        month: str,
        output_frame: pd.DataFrame,
        processing_frame: pd.DataFrame,
        source_csv: Path | None = None,
    ) -> list[str]:
        """Write the three output CSVs and return their paths.

        Args:
            job_name: The job folder name.
            csv_name: The input CSV filename.
            year: 4-digit year for the backup path.
            month: 2-digit month for the backup path.
            output_frame: The string-typed original frame (for ``is_pass_only``).
            processing_frame: The full enriched frame (carrying ``is_pass``).

        Returns:
            The written paths ``[primary, backup, processed]``.
        """
        is_pass_frame = output_frame.copy()
        is_pass_frame[_IS_PASS_COLUMN] = processing_frame[_IS_PASS_COLUMN].astype(str)

        primary_frame = (
            processing_frame if self._mode == _MODE_FULL_AI_COLUMNS else is_pass_frame
        )

        backup_dir = self._backup_root / year / month / job_name
        backup_path = backup_dir / csv_name
        processed_path = backup_dir / self._processed_name(csv_name)

        if self._path_layout == _LAYOUT_MACHINE_RETURN:
            if source_csv is None:
                raise ValueError("source_csv is required for machine_return output")
            machine_result = self._machine_output.write(
                source_csv,
                processing_frame[_IS_PASS_COLUMN].astype(str).tolist(),
            )
            primary_path = machine_result.destination_csv
        else:
            primary_path = self._external_root / job_name / _AI_SUBFOLDER / csv_name
            self._csv.write(primary_frame, primary_path)
        self._csv.write(is_pass_frame, backup_path)
        self._csv.write(processing_frame, processed_path)
        return [str(primary_path), str(backup_path), str(processed_path)]

    @staticmethod
    def _processed_name(csv_name: str) -> str:
        """Return the ``{stem}_processed{suffix}`` filename for ``csv_name``."""
        path = Path(csv_name)
        return f"{path.stem}{_PROCESSED_SUFFIX}{path.suffix}"
