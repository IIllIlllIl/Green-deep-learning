"""
Hyperparameter Deduplication Module

This module provides functions to extract historical hyperparameter combinations
from CSV files and build deduplication sets to prevent inter-round duplication.

Functions:
    extract_mutations_from_csv: Extract hyperparameter combinations from a single CSV
    load_historical_mutations: Load mutations from multiple CSV files
    build_dedup_set: Build a set of mutation keys for deduplication
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .hyperparams import _normalize_mutation_key, FLOAT_PRECISION


# Mapping of CSV column names to hyperparameter names
HYPERPARAM_COLUMNS = {
    "hyperparam_alpha": "alpha",
    "hyperparam_batch_size": "batch_size",
    "hyperparam_dropout": "dropout",
    "hyperparam_epochs": "epochs",
    "hyperparam_kfold": "kfold",
    "hyperparam_learning_rate": "learning_rate",
    "hyperparam_max_iter": "max_iter",
    "hyperparam_seed": "seed",
    "hyperparam_weight_decay": "weight_decay",
}


def _parse_hyperparam_value(value_str: str) -> Optional[float]:
    """Parse hyperparameter value from CSV string

    Args:
        value_str: String value from CSV

    Returns:
        Parsed numeric value, or None if empty/invalid
    """
    if not value_str or value_str.strip() == "":
        return None

    try:
        return float(value_str)
    except ValueError:
        return None


def extract_mutations_from_csv(
    csv_path: Path,
    filter_by_repo: Optional[str] = None,
    filter_by_model: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Dict], Dict]:
    """Extract hyperparameter combinations from a CSV file

    Reads a summary CSV file and extracts all hyperparameter combinations,
    grouping them by repository/model for targeted deduplication.

    Args:
        csv_path: Path to CSV file
        filter_by_repo: Optional repository name to filter by
        filter_by_model: Optional model name to filter by
        logger: Optional logger instance

    Returns:
        Tuple of (mutations_list, statistics_dict)
        - mutations_list: List of mutation dicts with only non-None values
        - statistics_dict: Statistics about extraction (total, filtered, etc.)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return [], {"total": 0, "filtered": 0, "extracted": 0}

    mutations = []
    stats = {
        "total": 0,
        "filtered": 0,
        "extracted": 0,
        "by_model": {}
    }

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                stats["total"] += 1

                # Apply filters if specified
                repo = row.get("repository", "")
                model = row.get("model", "")

                if filter_by_repo and repo != filter_by_repo:
                    stats["filtered"] += 1
                    continue

                if filter_by_model and model != filter_by_model:
                    stats["filtered"] += 1
                    continue

                # Extract hyperparameters (only non-None values)
                mutation = {}
                for csv_col, param_name in HYPERPARAM_COLUMNS.items():
                    value = _parse_hyperparam_value(row.get(csv_col, ""))
                    if value is not None:
                        mutation[param_name] = value

                # Only add if we extracted at least one hyperparameter
                if mutation:
                    mutations.append(mutation)
                    stats["extracted"] += 1

                    # Track by model
                    model_key = f"{repo}/{model}"
                    stats["by_model"][model_key] = stats["by_model"].get(model_key, 0) + 1

        logger.info(
            f"Extracted {stats['extracted']} mutations from {csv_path.name} "
            f"(total: {stats['total']}, filtered: {stats['filtered']})"
        )

    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
        return [], stats

    return mutations, stats


def load_historical_mutations(
    csv_paths: List[Path],
    filter_by_repo: Optional[str] = None,
    filter_by_model: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Dict], Dict]:
    """Load historical hyperparameter combinations from multiple CSV files

    Args:
        csv_paths: List of paths to CSV files
        filter_by_repo: Optional repository name to filter by
        filter_by_model: Optional model name to filter by
        logger: Optional logger instance

    Returns:
        Tuple of (all_mutations, combined_statistics)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    all_mutations = []
    combined_stats = {
        "total_files": len(csv_paths),
        "successful_files": 0,
        "total_rows": 0,
        "total_filtered": 0,
        "total_extracted": 0,
        "by_model": {}
    }

    for csv_path in csv_paths:
        mutations, stats = extract_mutations_from_csv(
            csv_path,
            filter_by_repo=filter_by_repo,
            filter_by_model=filter_by_model,
            logger=logger
        )

        if mutations:
            all_mutations.extend(mutations)
            combined_stats["successful_files"] += 1
            combined_stats["total_rows"] += stats["total"]
            combined_stats["total_filtered"] += stats["filtered"]
            combined_stats["total_extracted"] += stats["extracted"]

            # Merge by_model statistics
            for model_key, count in stats["by_model"].items():
                combined_stats["by_model"][model_key] = \
                    combined_stats["by_model"].get(model_key, 0) + count

    logger.info(
        f"Loaded {combined_stats['total_extracted']} total mutations from "
        f"{combined_stats['successful_files']}/{combined_stats['total_files']} CSV files"
    )

    return all_mutations, combined_stats


def build_dedup_set(
    mutations: List[Dict],
    logger: Optional[logging.Logger] = None
) -> Set[tuple]:
    """Build a set of normalized mutation keys for deduplication

    Converts a list of mutation dictionaries into a set of normalized keys
    that can be used to initialize the seen_mutations set in generate_mutations().

    Args:
        mutations: List of mutation dictionaries
        logger: Optional logger instance

    Returns:
        Set of normalized mutation keys (tuples)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    dedup_set = set()

    for mutation in mutations:
        key = _normalize_mutation_key(mutation)
        dedup_set.add(key)

    logger.info(f"Built deduplication set with {len(dedup_set)} unique mutations")

    return dedup_set


def print_dedup_statistics(
    stats: Dict,
    dedup_set: Optional[Set[tuple]] = None
) -> None:
    """Print human-readable statistics about loaded historical data

    Args:
        stats: Statistics dictionary from load_historical_mutations
        dedup_set: Optional deduplication set to show unique count
    """
    print("\n" + "=" * 80)
    print("Historical Hyperparameter Loading Statistics")
    print("=" * 80)
    print(f"CSV Files Processed: {stats['successful_files']}/{stats['total_files']}")
    print(f"Total Rows: {stats['total_rows']}")
    print(f"Filtered Rows: {stats['total_filtered']}")
    print(f"Extracted Mutations: {stats['total_extracted']}")

    if dedup_set:
        print(f"Unique Mutations: {len(dedup_set)}")

    if stats.get("by_model"):
        print("\nBreakdown by Model:")
        for model_key, count in sorted(stats["by_model"].items()):
            print(f"  {model_key}: {count}")

    print("=" * 80)
    print()
