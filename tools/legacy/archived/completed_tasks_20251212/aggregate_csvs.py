#!/usr/bin/env python3
"""
CSV Aggregation Script for Mutation-Based Training Energy Profiler

Purpose:
    Aggregate summary CSV files from multiple experiment runs into a single
    consolidated CSV file for comprehensive analysis.

Features:
    - Combines data from default, 1x mutation, and 2x mutation experiments
    - Adds 'experiment_source' column to track data origin
    - Makes experiment IDs unique by prepending source name (format: source__original_id)
    - Filters out mnist_ff experiments (due to batch_size configuration changes)
    - Validates data consistency and reports statistics
    - For 2x mutation run, uses only safe data (pre-orphan process)

Usage:
    python3 scripts/aggregate_csvs.py [OPTIONS]

Options:
    --output PATH       Output CSV file path (default: results/summary_all.csv)
    --keep-mnist-ff     Keep mnist_ff experiments (default: filter out)
    --verbose           Print detailed processing information
    --dry-run          Show what would be done without writing output

Input Files (relative to project root):
    - results/defualt/summary.csv           (default experiments)
    - results/mutation_1x/summary.csv       (1x mutation experiments)
    - results/mutation_2x_20251122_175401/summary_safe.csv  (2x mutation safe data)

Output:
    - results/summary_all.csv (or custom path)
    - Consolidated CSV with additional 'experiment_source' column

Author: Mutation Energy Profiler Team
Created: 2025-11-26
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Configuration
DEFAULT_OUTPUT = "results/summary_all.csv"

# Input file configurations
INPUT_CONFIGS = [
    {
        "name": "default",
        "path": "results/defualt/summary.csv",
        "description": "Default hyperparameter experiments"
    },
    {
        "name": "mutation_1x",
        "path": "results/mutation_1x/summary.csv",
        "description": "1x mutation experiments (1 run per config)"
    },
    {
        "name": "mutation_2x_safe",
        "path": "results/mutation_2x_20251122_175401/summary_safe.csv",
        "description": "2x mutation experiments (safe data, pre-orphan process)"
    }
]


def validate_csv_structure(file_path: Path, expected_columns: set) -> Tuple[bool, str]:
    """
    Validate that CSV file exists and has expected column structure.

    Args:
        file_path: Path to CSV file
        expected_columns: Set of expected column names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            actual_columns = set(reader.fieldnames or [])

            if not expected_columns.issubset(actual_columns):
                missing = expected_columns - actual_columns
                return False, f"Missing columns: {missing}"

            return True, ""

    except Exception as e:
        return False, f"Error reading file: {e}"


def read_csv_with_source(
    file_path: Path,
    source_name: str,
    filter_mnist_ff: bool = True,
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Read CSV file and add experiment_source column.

    Args:
        file_path: Path to CSV file
        source_name: Name to use for experiment_source column
        filter_mnist_ff: Whether to filter out mnist_ff experiments
        verbose: Print detailed information

    Returns:
        Tuple of (rows, statistics)
    """
    rows = []
    stats = {
        "total": 0,
        "filtered_mnist_ff": 0,
        "success": 0,
        "failed": 0
    }

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1

            # Filter mnist_ff if requested
            if filter_mnist_ff and row.get("model") == "mnist_ff":
                stats["filtered_mnist_ff"] += 1
                if verbose:
                    print(f"  Filtered: {row.get('experiment_id')} (mnist_ff)")
                continue

            # Add source column
            row["experiment_source"] = source_name

            # Make experiment_id unique by prepending source name
            # Format: source_name__original_experiment_id
            original_id = row.get("experiment_id", "")
            row["experiment_id"] = f"{source_name}__{original_id}"

            # Update statistics
            if row.get("training_success") == "True":
                stats["success"] += 1
            else:
                stats["failed"] += 1

            rows.append(row)

    return rows, stats


def aggregate_csvs(
    output_path: Path,
    filter_mnist_ff: bool = True,
    verbose: bool = False,
    dry_run: bool = False
) -> int:
    """
    Aggregate multiple CSV files into a single output file.

    Args:
        output_path: Path for output CSV file
        filter_mnist_ff: Whether to filter out mnist_ff experiments
        verbose: Print detailed information
        dry_run: Don't write output file

    Returns:
        Exit code (0 for success, 1 for error)
    """
    project_root = Path(__file__).parent.parent

    print("=" * 80)
    print("CSV Aggregation Script")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Output file: {output_path}")
    print(f"Filter mnist_ff: {filter_mnist_ff}")
    print(f"Dry run: {dry_run}")
    print("=" * 80)
    print()

    # Step 1: Validate all input files
    print("Step 1: Validating input files...")
    print("-" * 80)

    expected_columns = {
        "experiment_id", "timestamp", "repository", "model", "training_success"
    }

    all_valid = True
    for config in INPUT_CONFIGS:
        input_path = project_root / config["path"]
        valid, error = validate_csv_structure(input_path, expected_columns)

        status = "✓" if valid else "✗"
        print(f"{status} {config['name']:20s} {input_path}")
        if not valid:
            print(f"  Error: {error}")
            all_valid = False
        else:
            print(f"  Description: {config['description']}")

    if not all_valid:
        print("\n✗ Validation failed. Please check input files.")
        return 1

    print("\n✓ All input files validated successfully")
    print()

    # Step 2: Read and aggregate data
    print("Step 2: Reading and aggregating data...")
    print("-" * 80)

    all_rows = []
    all_stats = {}

    for config in INPUT_CONFIGS:
        input_path = project_root / config["path"]
        print(f"\nProcessing: {config['name']}")
        print(f"  Source: {input_path}")

        rows, stats = read_csv_with_source(
            input_path,
            config["name"],
            filter_mnist_ff,
            verbose
        )

        all_rows.extend(rows)
        all_stats[config["name"]] = stats

        print(f"  Total rows: {stats['total']}")
        if filter_mnist_ff and stats['filtered_mnist_ff'] > 0:
            print(f"  Filtered (mnist_ff): {stats['filtered_mnist_ff']}")
        print(f"  Kept: {len(rows)} ({stats['success']} success, {stats['failed']} failed)")

    print()
    print("-" * 80)
    print(f"Total aggregated rows: {len(all_rows)}")
    print()

    # Step 3: Write output
    if not dry_run:
        print("Step 3: Writing output file...")
        print("-" * 80)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get column names (add experiment_source to the end)
        if all_rows:
            # Get original column order from first row
            original_columns = [k for k in all_rows[0].keys() if k != "experiment_source"]
            all_columns = original_columns + ["experiment_source"]

            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_columns)
                writer.writeheader()
                writer.writerows(all_rows)

            print(f"✓ Output written to: {output_path}")
            print(f"  Total rows: {len(all_rows)}")
            print(f"  Columns: {len(all_columns)}")
        else:
            print("⚠ No data to write (all rows filtered out)")
            return 1
    else:
        print("Step 3: Dry run - skipping output")
        print("-" * 80)
        print("Would write:")
        print(f"  File: {output_path}")
        print(f"  Rows: {len(all_rows)}")

    print()

    # Step 4: Print summary statistics
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print()

    total_experiments = 0
    total_filtered = 0
    total_success = 0
    total_failed = 0

    for source_name, stats in all_stats.items():
        print(f"{source_name}:")
        print(f"  Total: {stats['total']}")
        if stats['filtered_mnist_ff'] > 0:
            print(f"  Filtered (mnist_ff): {stats['filtered_mnist_ff']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        kept = stats['success'] + stats['failed']
        if kept > 0:
            success_rate = stats['success'] / kept * 100
            print(f"  Success rate: {success_rate:.1f}%")
        print()

        total_experiments += stats['total']
        total_filtered += stats['filtered_mnist_ff']
        total_success += stats['success']
        total_failed += stats['failed']

    print("-" * 80)
    print(f"Overall:")
    print(f"  Total experiments: {total_experiments}")
    if total_filtered > 0:
        print(f"  Filtered (mnist_ff): {total_filtered}")
    print(f"  Kept: {len(all_rows)} ({total_success} success, {total_failed} failed)")
    if len(all_rows) > 0:
        overall_success_rate = total_success / len(all_rows) * 100
        print(f"  Overall success rate: {overall_success_rate:.1f}%")
    print()

    # Additional statistics by repository/model
    if all_rows:
        print("-" * 80)
        print("Distribution by model:")
        model_counts = {}
        for row in all_rows:
            key = f"{row['repository']}/{row['model']}"
            model_counts[key] = model_counts.get(key, 0) + 1

        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count}")

    print()
    print("=" * 80)
    print("✓ Aggregation completed successfully!")
    print("=" * 80)

    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Aggregate experiment CSV files into a single summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default usage (filter mnist_ff)
  python3 scripts/aggregate_csvs.py

  # Custom output path
  python3 scripts/aggregate_csvs.py --output results/my_summary.csv

  # Keep mnist_ff experiments
  python3 scripts/aggregate_csvs.py --keep-mnist-ff

  # Dry run with verbose output
  python3 scripts/aggregate_csvs.py --dry-run --verbose
        """
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT})"
    )

    parser.add_argument(
        "--keep-mnist-ff",
        action="store_true",
        help="Keep mnist_ff experiments (default: filter out)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing output"
    )

    args = parser.parse_args()

    # Convert output path to Path object
    output_path = Path(args.output)

    # Run aggregation
    exit_code = aggregate_csvs(
        output_path=output_path,
        filter_mnist_ff=not args.keep_mnist_ff,
        verbose=args.verbose,
        dry_run=args.dry_run
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
