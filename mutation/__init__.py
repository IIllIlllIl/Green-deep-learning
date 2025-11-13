"""
Energy-Efficient Training Mutation Tool

A framework for automated hyperparameter mutation experiments
with energy consumption monitoring.
"""

__version__ = "2.0.0"

# Public API
from .session import ExperimentSession
from .runner import MutationRunner
from .hyperparams import (
    mutate_hyperparameter,
    generate_mutations,
)
from .command_runner import CommandRunner
from .energy import (
    check_training_success,
    extract_performance_metrics,
    parse_energy_metrics,
)
from .utils import setup_logger, set_governor
from .exceptions import (
    MutationError,
    HyperparameterError,
    CommandExecutionError,
    MetricParsingError,
    ExperimentError,
    ConfigurationError,
)

__all__ = [
    "ExperimentSession",
    "MutationRunner",
    "CommandRunner",
    "mutate_hyperparameter",
    "generate_mutations",
    "check_training_success",
    "extract_performance_metrics",
    "parse_energy_metrics",
    "setup_logger",
    "set_governor",
    "MutationError",
    "HyperparameterError",
    "CommandExecutionError",
    "MetricParsingError",
    "ExperimentError",
    "ConfigurationError",
]
