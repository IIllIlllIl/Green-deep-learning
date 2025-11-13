"""
Hyperparameter Mutation Functions

This module provides pure functions for mutating hyperparameters in deep learning experiments.
It implements various mutation strategies including log-uniform, uniform, and zero-probability
distributions to generate diverse hyperparameter configurations.

Constants:
    FLOAT_PRECISION: Number of decimal places for float hyperparameters
    MAX_MUTATION_ATTEMPTS: Maximum attempts to generate unique mutations
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union


# Constants
FLOAT_PRECISION = 6
MAX_MUTATION_ATTEMPTS = 1000


def _format_hyperparam_value(value: Any, param_type: str) -> str:
    """Format hyperparameter value based on type

    Args:
        value: Parameter value to format
        param_type: Type of the parameter ('int' or 'float')

    Returns:
        Formatted string representation
    """
    if param_type == "int":
        return str(int(value))
    elif param_type == "float":
        return f"{value:.{FLOAT_PRECISION}f}"
    else:
        return str(value)


def _normalize_mutation_key(mutation: Dict[str, Any]) -> tuple:
    """Create normalized, hashable key for mutation uniqueness check

    Normalizes float values to fixed precision and sorts parameters
    deterministically to create a unique key for each mutation.

    Args:
        mutation: Hyperparameter mutation dictionary

    Returns:
        Sorted tuple of (param, normalized_value) pairs
    """
    normalized_items = []
    for param, value in mutation.items():
        # Normalize float values to fixed precision
        if isinstance(value, float):
            normalized_value = f"{value:.{FLOAT_PRECISION}f}"
        elif isinstance(value, int):
            normalized_value = str(int(value))
        else:
            normalized_value = str(value)

        normalized_items.append((param, normalized_value))

    # Sort by parameter name for deterministic order
    return tuple(sorted(normalized_items))


def _build_hyperparam_args(
    supported_params: Dict,
    hyperparams: Dict[str, Any],
    as_list: bool = True
) -> Union[List[str], str]:
    """Build hyperparameter arguments for command line

    Args:
        supported_params: Dictionary of supported hyperparameters from config
        hyperparams: Dictionary of hyperparameter values
        as_list: If True, return list of args; if False, return space-separated string

    Returns:
        List of argument strings or single space-separated string
    """
    args = []
    for param, value in hyperparams.items():
        if param in supported_params:
            flag = supported_params[param]["flag"]
            param_type = supported_params[param]["type"]
            formatted_value = _format_hyperparam_value(value, param_type)
            args.extend([flag, formatted_value])

    return args if as_list else " ".join(args)


def mutate_hyperparameter(
    param_config: Dict,
    param_name: str = "",
    random_seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Any:
    """Mutate a single hyperparameter with advanced strategies

    Implements parameter-specific mutation strategies:
    - Epochs: Log-uniform distribution [default×0.5, default×2.0]
    - Learning Rate: Log-uniform distribution [default×0.1, default×10.0]
    - Weight Decay: 30% zero + 70% log-uniform [0.0, default×100]
    - Dropout: Uniform distribution [0.0, 0.7]
    - Seed: Uniform integer [0, 9999]

    Args:
        param_config: Configuration for the hyperparameter (contains type, range, default, etc.)
        param_name: Name of the parameter (used to determine strategy)
        random_seed: Random seed for reproducibility (only used if provided)
        logger: Logger instance for debug messages

    Returns:
        Mutated value
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    param_type = param_config["type"]
    param_range = param_config["range"]
    default_value = param_config.get("default")

    # Get mutation distribution strategy (default to "uniform")
    distribution = param_config.get("distribution", "uniform")
    zero_probability = param_config.get("zero_probability", 0.0)

    # Handle zero probability for parameters like weight_decay
    if zero_probability > 0 and random.random() < zero_probability:
        logger.debug(f"Parameter '{param_name}' set to zero (zero_probability={zero_probability})")
        return 0.0 if param_type == "float" else 0

    # Determine the range
    min_val, max_val = param_range[0], param_range[1]

    # Handle different distribution strategies
    if distribution == "log_uniform":
        # Log-uniform distribution (for exponentially-sensitive parameters)
        if min_val <= 0:
            raise ValueError(f"Log-uniform distribution requires min_val > 0, got {min_val}")

        log_min = math.log(min_val)
        log_max = math.log(max_val)
        log_value = random.uniform(log_min, log_max)
        value = math.exp(log_value)

        if param_type == "int":
            result = max(min_val, min(max_val, int(round(value))))
        else:
            result = max(min_val, min(max_val, value))

        logger.debug(f"Generated log-uniform value for '{param_name}': {result}")
        return result

    elif distribution == "uniform":
        # Standard uniform distribution
        if param_type == "int":
            result = random.randint(min_val, max_val)
        else:
            result = random.uniform(min_val, max_val)

        logger.debug(f"Generated uniform value for '{param_name}': {result}")
        return result

    else:
        raise ValueError(f"Unknown distribution type: {distribution}")


def generate_mutations(
    supported_params: Dict,
    mutate_params: List[str],
    num_mutations: int = 1,
    random_seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """Generate mutated hyperparameter sets with uniqueness guarantee

    Args:
        supported_params: Dictionary of supported hyperparameters from config
        mutate_params: List of hyperparameters to mutate
        num_mutations: Number of unique mutation sets to generate
        random_seed: Random seed for reproducibility (only used if provided)
        logger: Logger instance for debug messages

    Returns:
        List of mutated hyperparameter dictionaries (all unique)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Determine which parameters to mutate
    if "all" in mutate_params:
        params_to_mutate = list(supported_params.keys())
    else:
        params_to_mutate = [p for p in mutate_params if p in supported_params]

    if not params_to_mutate:
        raise ValueError(
            f"No valid hyperparameters to mutate. "
            f"Available: {list(supported_params.keys())}"
        )

    print(f"📊 Generating {num_mutations} unique mutation(s) for parameters: {params_to_mutate}")

    mutations = []
    seen_mutations = set()  # Track unique mutations using normalized keys
    attempts = 0

    while len(mutations) < num_mutations and attempts < MAX_MUTATION_ATTEMPTS:
        attempts += 1

        # Generate new mutation
        mutation = {}
        for param in params_to_mutate:
            param_config = supported_params[param]
            mutation[param] = mutate_hyperparameter(
                param_config,
                param,
                random_seed=random_seed,
                logger=logger
            )

        # Use normalized key for uniqueness check (handles float precision issues)
        mutation_key = _normalize_mutation_key(mutation)

        # Check if this mutation is unique
        if mutation_key not in seen_mutations:
            seen_mutations.add(mutation_key)
            mutations.append(mutation)
            print(f"   Mutation {len(mutations)}: {mutation}")

    # Warning if we couldn't generate enough unique mutations
    if len(mutations) < num_mutations:
        logger.warning(
            f"Could only generate {len(mutations)} unique mutations after {attempts} attempts. "
            f"Requested: {num_mutations}, Generated: {len(mutations)}. "
            f"Consider widening hyperparameter ranges or reducing num_mutations."
        )
        print(f"⚠️  Warning: Could only generate {len(mutations)} unique mutations after {attempts} attempts")
        print(f"   Requested: {num_mutations}, Generated: {len(mutations)}")
        print(f"   Consider widening hyperparameter ranges or reducing num_mutations")

    return mutations
