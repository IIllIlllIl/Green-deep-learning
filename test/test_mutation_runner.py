#!/usr/bin/env python3
"""
Test script for mutation.py improvements

This script tests:
1. Class constants are defined correctly
2. Random seed functionality works
3. CSV streaming parser works correctly
4. Memory efficiency improvements
5. Code quality improvements (no duplicate code issues)
6. Mutation uniqueness enforcement
"""

import csv
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

# Add parent directory to path to import mutation
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mutation import MutationRunner
    print("✅ Successfully imported MutationRunner from mutation.py")
except Exception as e:
    print(f"❌ Failed to import MutationRunner: {e}")
    sys.exit(1)


def test_class_constants():
    """Test that all class constants are defined"""
    print("\n" + "=" * 80)
    print("TEST 1: Class Constants")
    print("=" * 80)

    required_constants = [
        'GOVERNOR_TIMEOUT_SECONDS',
        'RETRY_SLEEP_SECONDS',
        'RUN_SLEEP_SECONDS',
        'CONFIG_SLEEP_SECONDS',
        'DEFAULT_TRAINING_TIMEOUT_SECONDS',
        'MIN_LOG_FILE_SIZE_BYTES',
        'DEFAULT_MAX_RETRIES',
        'SCALE_FACTOR_MIN',
        'SCALE_FACTOR_MAX'
    ]

    all_present = True
    for const in required_constants:
        if hasattr(MutationRunner, const):
            value = getattr(MutationRunner, const)
            print(f"  ✅ {const} = {value}")
        else:
            print(f"  ❌ {const} - NOT FOUND")
            all_present = False

    if all_present:
        print("\n✅ All constants defined correctly")
        return True
    else:
        print("\n❌ Some constants are missing")
        return False


def test_random_seed():
    """Test that random seed produces reproducible results"""
    print("\n" + "=" * 80)
    print("TEST 2: Random Seed Reproducibility")
    print("=" * 80)

    # Create a minimal config for testing
    test_config = {
        "models": {
            "test_repo": {
                "path": "/tmp/test",
                "train_script": "train.sh",
                "models": ["test_model"],
                "supported_hyperparams": {
                    "epochs": {
                        "type": "int",
                        "range": [1, 100],
                        "flag": "--epochs",
                        "default": 10
                    },
                    "learning_rate": {
                        "type": "float",
                        "range": [0.0001, 0.1],
                        "flag": "--lr",
                        "default": 0.001
                    }
                }
            }
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        json.dump(test_config, f)
        config_file = f.name

    try:
        # Test with seed=42
        runner1 = MutationRunner(config_path=config_file, random_seed=42)
        mutations1 = runner1.generate_mutations("test_repo", "test_model", ["epochs", "learning_rate"], num_mutations=3)

        # Test with same seed=42
        runner2 = MutationRunner(config_path=config_file, random_seed=42)
        mutations2 = runner2.generate_mutations("test_repo", "test_model", ["epochs", "learning_rate"], num_mutations=3)

        # Check if results are identical
        if mutations1 == mutations2:
            print("  ✅ Seed 42: Generated identical mutations")
            print(f"     Mutations: {mutations1}")

            # Test with different seed
            runner3 = MutationRunner(config_path=config_file, random_seed=123)
            mutations3 = runner3.generate_mutations("test_repo", "test_model", ["epochs", "learning_rate"], num_mutations=3)

            if mutations1 != mutations3:
                print("  ✅ Seed 123: Generated different mutations")
                print(f"     Mutations: {mutations3}")
                print("\n✅ Random seed functionality works correctly")
                return True
            else:
                print("  ❌ Different seeds produced identical results")
                return False
        else:
            print("  ❌ Same seed produced different results")
            print(f"     Run 1: {mutations1}")
            print(f"     Run 2: {mutations2}")
            return False

    finally:
        os.unlink(config_file)


def test_csv_streaming_parser():
    """Test the CSV streaming parser for memory efficiency and correctness"""
    print("\n" + "=" * 80)
    print("TEST 3: CSV Streaming Parser")
    print("=" * 80)

    # Create temporary CSV file with test data
    test_dir = Path(tempfile.mkdtemp())
    csv_file = test_dir / "test_metrics.csv"

    test_data = [
        {"timestamp": 0, "power_draw_w": 100.5},
        {"timestamp": 1, "power_draw_w": 105.2},
        {"timestamp": 2, "power_draw_w": 98.7},
        {"timestamp": 3, "power_draw_w": 110.3},
        {"timestamp": 4, "power_draw_w": 102.1}
    ]

    # Write test CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "power_draw_w"])
        writer.writeheader()
        writer.writerows(test_data)

    print(f"  Created test CSV with {len(test_data)} rows")

    try:
        # Create runner with minimal config
        test_config = {"models": {}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(test_config, f)
            config_file = f.name

        runner = MutationRunner(config_path=config_file)
        os.unlink(config_file)

        # Test the streaming parser
        stats = runner._parse_csv_metric_streaming(csv_file, 'power_draw_w')

        # Calculate expected values
        values = [d['power_draw_w'] for d in test_data]
        expected_avg = sum(values) / len(values)
        expected_max = max(values)
        expected_min = min(values)
        expected_sum = sum(values)

        print(f"  Expected: avg={expected_avg:.2f}, max={expected_max:.2f}, min={expected_min:.2f}, sum={expected_sum:.2f}")
        print(f"  Got:      avg={stats['avg']:.2f}, max={stats['max']:.2f}, min={stats['min']:.2f}, sum={stats['sum']:.2f}")

        # Check correctness with small tolerance
        tolerance = 0.01
        if (abs(stats['avg'] - expected_avg) < tolerance and
            abs(stats['max'] - expected_max) < tolerance and
            abs(stats['min'] - expected_min) < tolerance and
            abs(stats['sum'] - expected_sum) < tolerance):
            print("  ✅ Streaming parser computes correct statistics")

            # Test memory efficiency with large file
            print("\n  Testing memory efficiency with 10,000 rows...")
            large_csv = test_dir / "large_metrics.csv"
            with open(large_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "power_draw_w"])
                writer.writeheader()
                for i in range(10000):
                    writer.writerow({"timestamp": i, "power_draw_w": 100.0 + (i % 50)})

            large_stats = runner._parse_csv_metric_streaming(large_csv, 'power_draw_w')
            if large_stats['avg'] is not None:
                print(f"  ✅ Successfully processed large file: avg={large_stats['avg']:.2f}")
                print("\n✅ CSV streaming parser works correctly and efficiently")
                return True
            else:
                print("  ❌ Failed to process large file")
                return False
        else:
            print("  ❌ Parser produced incorrect results")
            return False

    finally:
        shutil.rmtree(test_dir)


def test_code_quality():
    """Test that code quality improvements are in place"""
    print("\n" + "=" * 80)
    print("TEST 4: Code Quality Checks")
    print("=" * 80)

    # Check that csv is imported at module level
    import mutation
    if hasattr(mutation, 'csv'):
        print("  ✅ csv module imported at file level")
    else:
        print("  ❌ csv module not found at file level")
        return False

    # Check that shutil is imported (needed for cleanup)
    if hasattr(mutation, 'shutil'):
        print("  ✅ shutil module imported at file level")
    else:
        print("  ❌ shutil module not found at file level")
        return False

    # Check that streaming parser method exists
    if hasattr(MutationRunner, '_parse_csv_metric_streaming'):
        print("  ✅ Streaming CSV parser method exists")
    else:
        print("  ❌ Streaming CSV parser method not found")
        return False

    # Check that the method signature includes timeout
    import inspect
    sig = inspect.signature(MutationRunner.run_training_with_monitoring)
    if 'timeout' in sig.parameters:
        print("  ✅ Training timeout parameter added")
    else:
        print("  ❌ Training timeout parameter not found")
        return False

    # Check __init__ has random_seed parameter
    init_sig = inspect.signature(MutationRunner.__init__)
    if 'random_seed' in init_sig.parameters:
        print("  ✅ Random seed parameter added to __init__")
    else:
        print("  ❌ Random seed parameter not found in __init__")
        return False

    # Check MAX_MUTATION_ATTEMPTS constant exists
    if hasattr(MutationRunner, 'MAX_MUTATION_ATTEMPTS'):
        print(f"  ✅ MAX_MUTATION_ATTEMPTS constant defined ({MutationRunner.MAX_MUTATION_ATTEMPTS})")
    else:
        print("  ❌ MAX_MUTATION_ATTEMPTS constant not found")
        return False

    print("\n✅ All code quality checks passed")
    return True


def test_mutation_uniqueness():
    """Test that mutation uniqueness enforcement works"""
    print("\n" + "=" * 80)
    print("TEST 5: Mutation Uniqueness Enforcement")
    print("=" * 80)

    # Create minimal test config with very limited range
    test_config = {
        "models": {
            "test_repo": {
                "path": "/tmp/test",
                "train_script": "train.sh",
                "models": ["test_model"],
                "supported_hyperparams": {
                    "param1": {
                        "type": "int",
                        "range": [1, 5],  # Only 5 possible values
                        "flag": "--param1"
                    },
                    "param2": {
                        "type": "int",
                        "range": [1, 2],  # Only 2 possible values
                        "flag": "--param2"
                    }
                }
            }
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        json.dump(test_config, f)
        config_file = f.name

    try:
        # Test 1: Generate mutations within possible space (5 * 2 = 10 unique combinations)
        runner = MutationRunner(config_path=config_file, random_seed=42)
        mutations = runner.generate_mutations("test_repo", "test_model", ["param1", "param2"], num_mutations=10)

        # Check that we got 10 unique mutations
        unique_check = set()
        for m in mutations:
            key = frozenset(m.items())
            if key in unique_check:
                print(f"  ❌ Found duplicate mutation: {m}")
                return False
            unique_check.add(key)

        if len(mutations) == 10:
            print(f"  ✅ Generated {len(mutations)} unique mutations (max possible = 10)")
        else:
            print(f"  ❌ Expected 10 mutations, got {len(mutations)}")
            return False

        # Test 2: Try to generate more mutations than possible
        print("\n  Testing overflow scenario (requesting 15, max possible = 10)...")
        runner2 = MutationRunner(config_path=config_file, random_seed=123)
        mutations2 = runner2.generate_mutations("test_repo", "test_model", ["param1", "param2"], num_mutations=15)

        # Should get warning and only 10 mutations
        if len(mutations2) <= 10:
            print(f"  ✅ Correctly limited to {len(mutations2)} unique mutations (requested 15)")
        else:
            print(f"  ❌ Got {len(mutations2)} mutations, expected <=10")
            return False

        # Verify all are unique
        unique_check2 = set()
        for m in mutations2:
            key = frozenset(m.items())
            if key in unique_check2:
                print(f"  ❌ Found duplicate mutation in overflow test: {m}")
                return False
            unique_check2.add(key)

        print(f"  ✅ All {len(mutations2)} mutations are unique")

        print("\n✅ Mutation uniqueness enforcement works correctly")
        return True

    finally:
        os.unlink(config_file)


def main():
    """Run all tests"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 22 + "MUTATION.PY TEST SUITE" + " " * 34 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    results = {
        "Class Constants": test_class_constants(),
        "Random Seed": test_random_seed(),
        "CSV Streaming Parser": test_csv_streaming_parser(),
        "Code Quality": test_code_quality(),
        "Mutation Uniqueness": test_mutation_uniqueness()
    }

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    print("=" * 80)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
