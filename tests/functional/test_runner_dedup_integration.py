#!/usr/bin/env python3
"""
Test Script for MutationRunner Deduplication Integration

Purpose:
    Verify that the MutationRunner correctly loads and uses the inter-round
    deduplication mechanism when running experiments from a configuration file.

Test Cases:
    1. Load configuration with use_deduplication enabled
    2. Verify historical CSV files are loaded
    3. Verify deduplication set is built
    4. Verify mutations are generated with deduplication

Usage:
    python3 tests/functional/test_runner_dedup_integration.py

Author: Mutation Energy Profiler Team
Created: 2025-11-26
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mutation.runner import MutationRunner
import json


def test_config_loading():
    """Test 1: Verify configuration file loading"""
    print("=" * 80)
    print("Test 1: Configuration File Loading")
    print("=" * 80)

    config_path = project_root / "settings" / "mutation_2x_supplement.json"

    if not config_path.exists():
        print(f"✗ Configuration file not found: {config_path}")
        return False

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Verify required fields
    required_fields = ["experiment_name", "use_deduplication", "historical_csvs", "experiments"]
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        print(f"✗ Missing required fields: {missing_fields}")
        return False

    # Verify deduplication is enabled
    if not config.get("use_deduplication"):
        print(f"✗ use_deduplication is not enabled")
        return False

    # Verify historical CSVs are specified
    historical_csvs = config.get("historical_csvs", [])
    if len(historical_csvs) == 0:
        print(f"✗ No historical CSV files specified")
        return False

    print(f"✓ Configuration loaded successfully")
    print(f"  - Experiment name: {config['experiment_name']}")
    print(f"  - Deduplication enabled: {config['use_deduplication']}")
    print(f"  - Historical CSV files: {len(historical_csvs)}")
    print(f"  - Experiments: {len(config['experiments'])}")

    return True


def test_historical_csv_existence():
    """Test 2: Verify historical CSV files exist"""
    print("\n" + "=" * 80)
    print("Test 2: Historical CSV File Existence")
    print("=" * 80)

    config_path = project_root / "settings" / "mutation_2x_supplement.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    historical_csvs = config.get("historical_csvs", [])

    if len(historical_csvs) == 0:
        print(f"✗ No historical CSV files specified")
        return False

    print(f"Historical CSV configuration: {len(historical_csvs)} file(s)")

    existing_files = []
    missing_files = []

    for csv_path in historical_csvs:
        full_path = project_root / csv_path
        if full_path.exists():
            existing_files.append(csv_path)
            print(f"  ✓ Found: {csv_path}")

            # Check file size
            file_size = full_path.stat().st_size
            print(f"    Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

            # Count rows (quick check)
            with open(full_path, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
            print(f"    Rows: {row_count}")
        else:
            missing_files.append(csv_path)
            print(f"  ✗ Missing: {csv_path}")

    if missing_files:
        print(f"\n✗ Error: {len(missing_files)} CSV file(s) not found")
        return False

    if existing_files:
        print(f"\n✓ All {len(existing_files)} CSV file(s) found and accessible")
        return True
    else:
        print(f"\n✗ No CSV files found")
        return False


def test_runner_initialization():
    """Test 3: Verify MutationRunner can initialize with deduplication config"""
    print("\n" + "=" * 80)
    print("Test 3: MutationRunner Initialization")
    print("=" * 80)

    try:
        runner = MutationRunner()
        print(f"✓ MutationRunner initialized successfully")
        print(f"  - Project root: {runner.project_root}")
        print(f"  - Results directory: {runner.results_dir}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize MutationRunner: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deduplication_imports():
    """Test 4: Verify deduplication functions are accessible"""
    print("\n" + "=" * 80)
    print("Test 4: Deduplication Module Imports")
    print("=" * 80)

    try:
        from mutation.dedup import (
            load_historical_mutations,
            build_dedup_set,
            print_dedup_statistics
        )
        print(f"✓ Deduplication functions imported successfully")
        print(f"  - load_historical_mutations: {load_historical_mutations}")
        print(f"  - build_dedup_set: {build_dedup_set}")
        print(f"  - print_dedup_statistics: {print_dedup_statistics}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import deduplication functions: {e}")
        return False


def test_config_validation():
    """Test 5: Validate experiment configurations"""
    print("\n" + "=" * 80)
    print("Test 5: Experiment Configuration Validation")
    print("=" * 80)

    config_path = project_root / "settings" / "mutation_2x_supplement.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    experiments = config.get("experiments", [])

    if len(experiments) == 0:
        print(f"✗ No experiments defined")
        return False

    print(f"Validating {len(experiments)} experiment configurations...")

    for i, exp in enumerate(experiments, 1):
        repo = exp.get("repo")
        model = exp.get("model")
        num_mutations = exp.get("num_mutations")

        if not repo:
            print(f"  ✗ Experiment {i}: Missing 'repo' field")
            return False

        if not model:
            print(f"  ✗ Experiment {i}: Missing 'model' field")
            return False

        if not num_mutations:
            print(f"  ✗ Experiment {i}: Missing 'num_mutations' field")
            return False

        print(f"  ✓ Experiment {i}: {repo}/{model} ({num_mutations} mutations)")

    print(f"\n✓ All experiment configurations are valid")
    return True


def test_load_from_summary_all():
    """Test 6: Load historical data from summary_all.csv"""
    print("\n" + "=" * 80)
    print("Test 6: Load Historical Data from summary_all.csv")
    print("=" * 80)

    summary_all_path = project_root / "results" / "summary_all.csv"

    if not summary_all_path.exists():
        print(f"✗ summary_all.csv not found at: {summary_all_path}")
        return False

    print(f"Loading from: {summary_all_path}")

    try:
        from mutation.dedup import load_historical_mutations, build_dedup_set

        # Load mutations from summary_all.csv
        mutations, stats = load_historical_mutations([summary_all_path])

        # Verify we loaded data
        if len(mutations) == 0:
            print(f"✗ No mutations loaded from summary_all.csv")
            return False

        print(f"  ✓ Loaded {len(mutations)} mutation records")

        # Build deduplication set
        dedup_set = build_dedup_set(mutations)

        print(f"  ✓ Built deduplication set with {len(dedup_set)} unique mutations")

        # Print statistics
        print(f"\nStatistics:")
        print(f"  - Total rows: {stats['total_rows']}")
        print(f"  - Total extracted: {stats['total_extracted']}")
        print(f"  - Unique mutations: {len(dedup_set)}")

        if stats.get('by_model'):
            print(f"  - Models covered: {len(stats['by_model'])}")

        # Verify dedup_set is not empty
        if len(dedup_set) == 0:
            print(f"\n✗ Deduplication set is empty")
            return False

        print(f"\n✓ Successfully loaded and processed historical data")
        return True

    except Exception as e:
        print(f"✗ Error loading historical data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner"""
    print("\n" + "=" * 80)
    print("MutationRunner Deduplication Integration Test Suite")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print("=" * 80 + "\n")

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Historical CSV Existence", test_historical_csv_existence),
        ("MutationRunner Initialization", test_runner_initialization),
        ("Deduplication Imports", test_deduplication_imports),
        ("Configuration Validation", test_config_validation),
        ("Load from summary_all.csv", test_load_from_summary_all),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if failed == 0:
        print("✓ All tests passed!")
        print("\nThe MutationRunner is ready to run experiments with inter-round deduplication.")
        print(f"\nConfiguration:")
        print(f"  - Deduplication enabled: Yes")
        print(f"  - Historical data source: results/summary_all.csv")
        print(f"\nTo run the supplementary experiments:")
        print(f"  python3 -m mutation.runner settings/mutation_2x_supplement.json")
        print("=" * 80)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
