"""Application-level exception hierarchy.

These errors carry a suggested HTTP ``status_code`` so the API layer can map
them to responses consistently. Status codes are chosen to match the legacy
``ai_server_fastapi.py`` contract (missing folder/CSV -> 400) so wiring them in
during later phases stays backward compatible.
"""

from __future__ import annotations

from typing import Optional


class AppError(Exception):
    """Base class for all application errors.

    Attributes:
        message: Human-readable error description.
        status_code: Suggested HTTP status code when surfaced via the API.
    """

    status_code: int = 500

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error description.
            status_code: Optional override for the class-level status code.
        """
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class ConfigError(AppError):
    """Raised when configuration is missing or invalid."""

    status_code = 500


class JobFolderNotFoundError(AppError):
    """Raised when the requested job folder does not exist or is not a directory."""

    status_code = 400


class CsvFileNotFoundError(AppError):
    """Raised when no CSV files are found in the job folder."""

    status_code = 400


class ModelClientError(AppError):
    """Raised when a call to an inference model endpoint fails."""

    status_code = 502
