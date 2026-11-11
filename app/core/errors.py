"""Typed errors shared by configuration, CSV parsing and local IO."""


class AppError(Exception):
    pass


class ConfigError(AppError):
    pass


class CsvSchemaError(AppError, ValueError):
    pass


class OutputWriteError(AppError):
    pass
