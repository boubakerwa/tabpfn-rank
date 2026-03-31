"""Custom exceptions for the project."""


class RecPFNError(Exception):
    """Base project exception."""


class OptionalDependencyNotAvailable(RecPFNError):
    """Raised when an optional model dependency is required but missing."""


class DatasetConfigurationError(RecPFNError):
    """Raised when a dataset cannot be loaded or normalized."""

