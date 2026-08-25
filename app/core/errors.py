"""Application-level exception hierarchy and error-handling contract.

Every layer (domain, application, infrastructure) raises these typed errors
instead of leaking framework or library exceptions. Each carries a suggested
HTTP ``status_code``; the API layer maps them centrally (see
``app.main.create_app``) so clients receive a consistent, concise ``{"detail":
...}`` payload. Unexpected (non-``AppError``) exceptions are logged with a full
traceback and returned as a generic 500 without leaking internals.

HTTP mapping:
    * ConfigError               -> 500
    * JobFolderNotFoundError    -> 400
    * CsvFileNotFoundError      -> 400
    * CsvSchemaError            -> 400
    * OutputWriteError          -> 500
    * ModelClientError          -> 502 (client adapters normally return typed
      error results; required failures become an HTTP-200 all-23 fallback)
    * any other Exception       -> 500 (generic message, traceback logged)
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors.

    Attributes:
        message: Human-readable error description.
        status_code: Suggested HTTP status code when surfaced via the API.
    """

    status_code: int = 500

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
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


class CsvSchemaError(AppError, ValueError):
    """Raised when a CSV lacks required columns or has unusable values.

    Also a :class:`ValueError` so the legacy ``ai_server_fastapi.py`` route,
    which maps ``ValueError`` to HTTP 400, keeps returning 400 unchanged.
    """

    status_code = 400


class ModelClientError(AppError):
    """Raised when a call to an inference model endpoint fails."""

    status_code = 502


class OutputWriteError(AppError):
    """Raised when writing an output CSV to disk fails."""

    status_code = 500
