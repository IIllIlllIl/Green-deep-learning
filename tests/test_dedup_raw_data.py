#!/usr/bin/env python3
"""
Functional test for deduplication logic using raw_data.csv

This test verifies that the deduplication system correctly reads from
raw_data.csv instead of summary_all.csv after the v4.7.3 migration.

Test Coverage:
1. Configuration files correctly reference raw_data.csv
2. Deduplication logic reads from raw_data.csv
3. Historical mutations are loaded correctly
4. Dedup set is built correctly
5. Configuration execution uses correct CSV file
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mutation.dedup import extract_mutations_from_csv, load_historical_mutations, build_dedup_set


def test_config_files_use_raw_data():
    """Test that all configuration files reference raw_data.csv"""
    print("\n" + "=" * 80)
    print("TEST 1: Configuration files use raw_data.csv")
    print("=" * 80)

    settings_dir = project_root / "settings"
    config_files = list(settings_dir.glob("*.json"))

    # Filter out archived files
    config_files = [f for f in config_files if "archived" not in str(f)]

    print(f"Checking {len(config_files)} configuration files...")

    all_correct = True
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Check historical_csvs field
        historical_csvs = config.get("historical_csvs", [])

        for csv_path in historical_csvs:
            if "summary_all.csv" in csv_path:
                print(f"❌ FAIL: {config_file.name} still references summary_all.csv")
                all_correct = False
            elif "raw_data.csv" in csv_path:
                print(f"✅ PASS: {config_file.name} correctly references raw_data.csv")
            else:
                print(f"⚠️  WARN: {config_file.name} references unknown CSV: {csv_path}")

    if all_correct:
        print("\n✅ All configuration files correctly reference raw_data.csv")
    else:
        print("\n❌ Some configuration files still reference summary_all.csv")

    return all_correct


def test_dedup_reads_raw_data():
    """Test that dedup logic can read from raw_data.csv"""
    print("\n" + "=" * 80)
    print("TEST 2: Deduplication reads from raw_data.csv")
    print("=" * 80)

    raw_data_path = project_root / "results" / "raw_data.csv"

    if not raw_data_path.exists():
        print(f"⚠️  SKIP: raw_data.csv not found at {raw_data_path}")
        return True

    print(f"Reading from {raw_data_path}...")

    try:
        # extract_mutations_from_csv returns (mutations_list, stats_dict)
        mutations, stats = extract_mutations_from_csv(raw_data_path)
        print(f"✅ Successfully extracted {len(mutations)} mutation records")
        print(f"   Statistics: {stats}")

        # Verify statistics structure
        required_stats = ['total', 'filtered', 'extracted']
        missing_stats = [k for k in required_stats if k not in stats]

        if missing_stats:
            print(f"❌ FAIL: Missing stats keys: {missing_stats}")
            return False

        # Verify mutation structure (should have hyperparameter keys)
        if mutations:
            first = mutations[0]
            # Mutations should contain hyperparameter names (e.g., 'epochs', 'learning_rate')
            # and may contain '__mode__' key
            if not first:
                print(f"❌ FAIL: Empty mutation record")
                return False
            else:
                print(f"✅ Mutation records have correct structure (sample: {list(first.keys())[:3]}...)")

        return True

    except Exception as e:
        print(f"❌ FAIL: Error reading raw_data.csv: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_historical_mutations():
    """Test that load_historical_mutations works with raw_data.csv"""
    print("\n" + "=" * 80)
    print("TEST 3: load_historical_mutations() with raw_data.csv")
    print("=" * 80)

    raw_data_path = project_root / "results" / "raw_data.csv"

    if not raw_data_path.exists():
        print(f"⚠️  SKIP: raw_data.csv not found at {raw_data_path}")
        return True

    print(f"Loading historical mutations from {raw_data_path}...")

    try:
        mutations, stats = load_historical_mutations([raw_data_path])

        print(f"✅ Loaded {len(mutations)} mutations")
        print(f"   Statistics: {stats}")

        # Verify stats structure (use correct keys from dedup.py)
        expected_stats = ['total_files', 'successful_files', 'total_rows',
                         'total_filtered', 'total_extracted']
        missing_stats = [k for k in expected_stats if k not in stats]

        if missing_stats:
            print(f"❌ FAIL: Missing stats: {missing_stats}")
            return False

        print(f"✅ Statistics have correct structure")
        return True

    except Exception as e:
        print(f"❌ FAIL: Error loading historical mutations: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_build_dedup_set():
    """Test that build_dedup_set works correctly"""
    print("\n" + "=" * 80)
    print("TEST 4: build_dedup_set() functionality")
    print("=" * 80)

    raw_data_path = project_root / "results" / "raw_data.csv"

    if not raw_data_path.exists():
        print(f"⚠️  SKIP: raw_data.csv not found at {raw_data_path}")
        return True

    try:
        # Load mutations
        mutations, _ = load_historical_mutations([raw_data_path])

        # Build dedup set
        dedup_set = build_dedup_set(mutations)

        print(f"✅ Built dedup set with {len(dedup_set)} unique combinations")

        # Verify dedup set structure (should be set of tuples)
        if not isinstance(dedup_set, set):
            print(f"❌ FAIL: dedup_set is not a set (got {type(dedup_set)})")
            return False

        if dedup_set and not isinstance(next(iter(dedup_set)), tuple):
            print(f"❌ FAIL: dedup_set elements are not tuples")
            return False

        print(f"✅ Dedup set has correct structure")
        return True

    except Exception as e:
        print(f"❌ FAIL: Error building dedup set: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_execution_simulation():
    """Simulate config file execution with raw_data.csv"""
    print("\n" + "=" * 80)
    print("TEST 5: Simulated config execution with raw_data.csv")
    print("=" * 80)

    # Use stage2 config as test case (it's a known working config)
    config_path = project_root / "settings" / "stage2_optimized_nonparallel_and_fast_parallel.json"

    if not config_path.exists():
        print(f"⚠️  SKIP: Test config not found at {config_path}")
        return True

    print(f"Testing with config: {config_path.name}")

    try:
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Check deduplication settings
        use_dedup = config.get("use_deduplication", False)
        historical_csvs = config.get("historical_csvs", [])

        print(f"   Deduplication enabled: {use_dedup}")
        print(f"   Historical CSV files: {historical_csvs}")

        if not use_dedup:
            print(f"⚠️  WARN: Deduplication not enabled in this config")
            return True

        # Verify raw_data.csv is referenced
        raw_data_referenced = any("raw_data.csv" in csv for csv in historical_csvs)
        summary_all_referenced = any("summary_all.csv" in csv for csv in historical_csvs)

        if summary_all_referenced:
            print(f"❌ FAIL: Config still references summary_all.csv")
            return False

        if not raw_data_referenced:
            print(f"❌ FAIL: Config does not reference raw_data.csv")
            return False

        # Simulate loading historical data
        csv_paths = [project_root / csv_path for csv_path in historical_csvs]
        existing_csvs = [p for p in csv_paths if p.exists()]

        print(f"   Found {len(existing_csvs)}/{len(csv_paths)} CSV files")

        if existing_csvs:
            mutations, stats = load_historical_mutations(existing_csvs)
            dedup_set = build_dedup_set(mutations)

            print(f"✅ Successfully loaded {len(mutations)} historical mutations")
            print(f"   Built dedup set with {len(dedup_set)} unique combinations")
            print(f"   Statistics: {stats}")

        print(f"✅ Config execution simulation successful")
        return True

    except Exception as e:
        print(f"❌ FAIL: Error simulating config execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("FUNCTIONAL TEST: Deduplication with raw_data.csv")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print("=" * 80)

    tests = [
        ("Config files use raw_data.csv", test_config_files_use_raw_data),
        ("Dedup reads raw_data.csv", test_dedup_reads_raw_data),
        ("load_historical_mutations()", test_load_historical_mutations),
        ("build_dedup_set()", test_build_dedup_set),
        ("Config execution simulation", test_config_execution_simulation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, r in results if r)

    print("=" * 80)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"❌ {total - passed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
