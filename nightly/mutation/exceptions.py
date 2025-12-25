"""Custom exceptions for mutation package"""


class MutationError(Exception):
    """Base exception for mutation package"""
    pass


class HyperparameterError(MutationError):
    """Raised when hyperparameter mutation fails"""
    pass


class CommandExecutionError(MutationError):
    """Raised when command execution fails"""
    pass


class MetricParsingError(MutationError):
    """Raised when metric parsing fails"""
    pass


class ExperimentError(MutationError):
    """Raised when experiment execution fails"""
    pass


class ConfigurationError(MutationError):
    """Raised when configuration is invalid"""
    pass
