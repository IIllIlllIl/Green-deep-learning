"""
Energy and Performance Metric Parsing Functions

This module provides pure functions for parsing energy consumption metrics and
performance metrics from training logs and energy monitoring outputs.

Constants:
    EMPTY_STATS_DICT: Template for empty statistics
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional


# Constants
EMPTY_STATS_DICT = {"avg": None, "max": None, "min": None, "sum": None}


def check_training_success(
    log_file: str,
    repo: str,
    min_log_file_size_bytes: int = 1000,
    logger: Optional[logging.Logger] = None
) -> tuple[bool, str]:
    """Check if training completed successfully

    Args:
        log_file: Path to training log file
        repo: Repository name
        min_log_file_size_bytes: Minimum expected log file size
        logger: Logger instance for debug messages

    Returns:
        Tuple of (success, error_message)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    log_path = Path(log_file)

    if not log_path.exists():
        logger.warning(f"Log file not found: {log_file}")
        return False, "Log file not found"

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()

    # IMPORTANT: Check for success indicators FIRST
    # Some repos may have warnings/tracebacks but still complete successfully
    success_patterns = [
        r"Training completed successfully",
        r"训练完成",
        r"训练成功完成",
        r"\[SUCCESS\]",
        r"✓.*训练成功",
        r"Evaluation completed",
        r"All.*completed",
        r"Rank@1:",  # Person_reID success indicator
        r"mAP:",     # Person_reID success indicator
    ]

    for pattern in success_patterns:
        if re.search(pattern, log_content, re.IGNORECASE):
            logger.debug(f"Training success detected with pattern: {pattern}")
            return True, "Training completed successfully"

    # Only check for error patterns if no success indicators found
    # More specific error patterns to avoid false positives
    error_patterns = [
        r"CUDA out of memory",
        r"RuntimeError:(?!.*DeprecationWarning).*",  # Exclude DeprecationWarnings
        r"AssertionError",
        r"FileNotFoundError",
        r"KeyboardInterrupt",
        r"Training.*FAILED",
        r"Fatal error:",
        r"致命错误:",
    ]

    for pattern in error_patterns:
        if re.search(pattern, log_content, re.IGNORECASE):
            logger.warning(f"Error pattern found in log: {pattern}")
            return False, f"Error pattern found: {pattern}"

    # If no clear indicators, check file size
    if log_path.stat().st_size < min_log_file_size_bytes:
        logger.warning(f"Log file too small: {log_path.stat().st_size} bytes")
        return False, "Log file too small, training likely failed"

    logger.debug("No errors detected in log file")
    return True, "No errors detected"


def extract_performance_metrics(
    log_file: str,
    repo: str,
    log_patterns: Dict[str, str],
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """Extract performance metrics from training log

    Args:
        log_file: Path to training log file
        repo: Repository name
        log_patterns: Dictionary of metric_name -> regex_pattern
        logger: Logger instance for debug messages

    Returns:
        Dictionary of performance metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    log_path = Path(log_file)
    metrics = {}

    if not log_path.exists():
        logger.warning(f"Log file not found for performance metric extraction: {log_path}")
        print(f"⚠️  Warning: Log file not found for performance metric extraction: {log_path}")
        return metrics

    if not log_patterns:
        logger.warning(f"No performance metric patterns defined for repo: {repo}")
        print(f"⚠️  Warning: No performance metric patterns defined for repo: {repo}")
        return metrics

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()

    for metric_name, pattern in log_patterns.items():
        matches = re.findall(pattern, log_content, re.IGNORECASE)
        if matches:
            try:
                last_match = matches[-1]
                # Handle both tuple (groups) and string (no groups)
                if isinstance(last_match, tuple):
                    value_str = last_match[0] if last_match else None
                else:
                    value_str = last_match

                if value_str:
                    metrics[metric_name] = float(value_str)
                    logger.debug(f"Extracted {metric_name}: {metrics[metric_name]}")
                else:
                    logger.warning(f"Empty match for metric '{metric_name}'")
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse metric '{metric_name}': {e}")
            except IndexError as e:
                logger.error(f"Index error parsing metric '{metric_name}': {e}")
        else:
            logger.debug(f"Metric '{metric_name}' pattern not found in log")

    return metrics


def _parse_csv_metric_streaming(
    csv_file: Path,
    field_name: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Optional[float]]:
    """Parse CSV file and compute statistics in streaming fashion

    Args:
        csv_file: Path to CSV file
        field_name: Name of the field to extract
        logger: Logger instance for debug messages

    Returns:
        Dictionary with avg, max, min, sum statistics (or None if file doesn't exist)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not csv_file.exists():
        logger.debug(f"CSV file not found: {csv_file}")
        return EMPTY_STATS_DICT.copy()

    try:
        count = 0
        total = 0.0
        max_val = float('-inf')
        min_val = float('inf')

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)

            # Validate field exists in header
            if field_name not in (reader.fieldnames or []):
                logger.warning(
                    f"Field '{field_name}' not found in {csv_file.name}. "
                    f"Available fields: {', '.join(reader.fieldnames or [])}"
                )
                return EMPTY_STATS_DICT.copy()

            for row_num, row in enumerate(reader, start=1):
                try:
                    value = float(row[field_name])
                    count += 1
                    total += value
                    max_val = max(max_val, value)
                    min_val = min(min_val, value)
                except ValueError as e:
                    logger.warning(
                        f"Invalid value in {csv_file.name} row {row_num}: "
                        f"{row.get(field_name)} - {e}"
                    )
                except KeyError as e:
                    logger.error(
                        f"Missing field '{field_name}' in {csv_file.name} row {row_num}: {e}"
                    )

        if count == 0:
            logger.warning(f"No valid data in {csv_file.name} for field '{field_name}'")
            return EMPTY_STATS_DICT.copy()

        return {
            "avg": total / count,
            "max": max_val,
            "min": min_val,
            "sum": total
        }

    except Exception as e:
        logger.error(f"Error parsing {csv_file}: {e}", exc_info=True)
        return EMPTY_STATS_DICT.copy()


def parse_energy_metrics(
    energy_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Parse energy consumption metrics

    Args:
        energy_dir: Directory containing energy monitoring outputs
        logger: Logger instance for debug messages

    Returns:
        Dictionary of energy metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    metrics = {
        "cpu_energy_pkg_joules": None,
        "cpu_energy_ram_joules": None,
        "cpu_energy_total_joules": None,
        "gpu_power_avg_watts": None,
        "gpu_power_max_watts": None,
        "gpu_power_min_watts": None,
        "gpu_energy_total_joules": None,
        "gpu_temp_avg_celsius": None,
        "gpu_temp_max_celsius": None,
        "gpu_util_avg_percent": None,
        "gpu_util_max_percent": None,
    }

    # Parse CPU energy
    cpu_file = energy_dir / "cpu_energy.txt"
    if cpu_file.exists():
        try:
            with open(cpu_file, 'r') as f:
                content = f.read()
                pkg_match = re.search(r"Package Energy.*:\s*([0-9.]+)", content)
                ram_match = re.search(r"RAM Energy.*:\s*([0-9.]+)", content)
                total_match = re.search(r"Total CPU Energy.*:\s*([0-9.]+)", content)

                if pkg_match:
                    metrics["cpu_energy_pkg_joules"] = float(pkg_match.group(1))
                    logger.debug(f"CPU Package Energy: {metrics['cpu_energy_pkg_joules']} J")
                if ram_match:
                    metrics["cpu_energy_ram_joules"] = float(ram_match.group(1))
                    logger.debug(f"CPU RAM Energy: {metrics['cpu_energy_ram_joules']} J")
                if total_match:
                    metrics["cpu_energy_total_joules"] = float(total_match.group(1))
                    logger.debug(f"CPU Total Energy: {metrics['cpu_energy_total_joules']} J")
        except Exception as e:
            logger.error(f"Error parsing CPU energy file: {e}", exc_info=True)
    else:
        logger.debug(f"CPU energy file not found: {cpu_file}")

    # Parse GPU power (streaming)
    gpu_power_file = energy_dir / "gpu_power.csv"
    power_stats = _parse_csv_metric_streaming(gpu_power_file, 'power_draw_w', logger=logger)
    metrics["gpu_power_avg_watts"] = power_stats["avg"]
    metrics["gpu_power_max_watts"] = power_stats["max"]
    metrics["gpu_power_min_watts"] = power_stats["min"]
    # Approximate energy (sum of power samples, assuming 1 second intervals)
    metrics["gpu_energy_total_joules"] = power_stats["sum"]

    # Parse GPU temperature (streaming)
    gpu_temp_file = energy_dir / "gpu_temperature.csv"
    temp_stats = _parse_csv_metric_streaming(gpu_temp_file, 'gpu_temp_c', logger=logger)
    metrics["gpu_temp_avg_celsius"] = temp_stats["avg"]
    metrics["gpu_temp_max_celsius"] = temp_stats["max"]

    # Parse GPU utilization (streaming)
    gpu_util_file = energy_dir / "gpu_utilization.csv"
    util_stats = _parse_csv_metric_streaming(gpu_util_file, 'gpu_util_percent', logger=logger)
    metrics["gpu_util_avg_percent"] = util_stats["avg"]
    metrics["gpu_util_max_percent"] = util_stats["max"]

    return metrics
