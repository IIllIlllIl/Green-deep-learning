#!/usr/bin/env python3
"""
Test deduplication mode distinction functionality

This test verifies that the deduplication mechanism correctly distinguishes
between parallel and non-parallel execution modes, ensuring that the same
hyperparameters in different modes are treated as distinct experiments.

Test Coverage:
1. _normalize_mutation_key() distinguishes modes
2. extract_mutations_from_csv() extracts mode from experiment_id
3. build_dedup_set() includes mode in deduplication keys
4. Backward compatibility (mode parameter is optional)
"""

import sys
import tempfile
import csv
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation.hyperparams import _normalize_mutation_key
from mutation.dedup import extract_mutations_from_csv, build_dedup_set


def test_normalize_mutation_key_with_mode():
    """Test that _normalize_mutation_key distinguishes parallel/non-parallel modes"""
    print("\n" + "=" * 80)
    print("TEST 1: _normalize_mutation_key() Mode Distinction")
    print("=" * 80)

    mutation = {'learning_rate': 0.01, 'batch_size': 32}

    # Test non-parallel mode
    key_nonparallel = _normalize_mutation_key(mutation, mode='nonparallel')
    print(f"Non-parallel key: {key_nonparallel}")

    # Test parallel mode
    key_parallel = _normalize_mutation_key(mutation, mode='parallel')
    print(f"Parallel key:     {key_parallel}")

    # Test without mode (backward compatibility)
    key_no_mode = _normalize_mutation_key(mutation, mode=None)
    print(f"No mode key:      {key_no_mode}")

    # Assertions
    assert key_nonparallel != key_parallel, \
        "❌ FAILED: Same hyperparameters should generate different keys for different modes"
    assert ('__mode__', 'nonparallel') in key_nonparallel, \
        "❌ FAILED: Non-parallel key should contain mode marker"
    assert ('__mode__', 'parallel') in key_parallel, \
        "❌ FAILED: Parallel key should contain mode marker"
    assert ('__mode__', 'nonparallel') not in key_no_mode and ('__mode__', 'parallel') not in key_no_mode, \
        "❌ FAILED: Key without mode should not contain mode marker"

    print("✅ PASSED: Mode distinction works correctly")
    return True


def test_extract_mutations_from_csv_mode_detection():
    """Test that extract_mutations_from_csv correctly extracts mode from experiment_id"""
    print("\n" + "=" * 80)
    print("TEST 2: extract_mutations_from_csv() Mode Detection")
    print("=" * 80)

    # Create temporary CSV file with test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        temp_csv = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=[
            'experiment_id', 'repository', 'model',
            'hyperparam_learning_rate', 'hyperparam_batch_size'
        ])
        writer.writeheader()

        # Non-parallel experiment
        writer.writerow({
            'experiment_id': 'examples_mnist_train_20251205_120000',
            'repository': 'examples',
            'model': 'mnist',
            'hyperparam_learning_rate': '0.01',
            'hyperparam_batch_size': '32'
        })

        # Parallel experiment
        writer.writerow({
            'experiment_id': 'examples_mnist_parallel_20251205_120100',
            'repository': 'examples',
            'model': 'mnist',
            'hyperparam_learning_rate': '0.01',
            'hyperparam_batch_size': '32'
        })

    try:
        # Extract mutations
        mutations, stats = extract_mutations_from_csv(temp_csv)

        print(f"Extracted {len(mutations)} mutations")
        for i, m in enumerate(mutations, 1):
            mode = m.get('__mode__')
            params = {k: v for k, v in m.items() if k != '__mode__'}
            print(f"  {i}. mode={mode}, params={params}")

        # Assertions
        assert len(mutations) == 2, \
            f"❌ FAILED: Expected 2 mutations, got {len(mutations)}"

        # Check first mutation (non-parallel)
        assert mutations[0].get('__mode__') == 'nonparallel', \
            f"❌ FAILED: First mutation should be 'nonparallel', got '{mutations[0].get('__mode__')}'"

        # Check second mutation (parallel)
        assert mutations[1].get('__mode__') == 'parallel', \
            f"❌ FAILED: Second mutation should be 'parallel', got '{mutations[1].get('__mode__')}'"

        print("✅ PASSED: Mode detection from experiment_id works correctly")
        return True

    finally:
        # Clean up temporary file
        temp_csv.unlink()


def test_build_dedup_set_mode_distinction():
    """Test that build_dedup_set includes mode in deduplication keys"""
    print("\n" + "=" * 80)
    print("TEST 3: build_dedup_set() Mode Distinction")
    print("=" * 80)

    # Same hyperparameters in different modes
    mutations = [
        {'learning_rate': 0.01, 'batch_size': 32, '__mode__': 'nonparallel'},
        {'learning_rate': 0.01, 'batch_size': 32, '__mode__': 'parallel'},
        {'learning_rate': 0.001, 'batch_size': 64, '__mode__': 'nonparallel'},
    ]

    # Build dedup set
    dedup_set = build_dedup_set(mutations)

    print(f"Deduplication set size: {len(dedup_set)}")
    for i, key in enumerate(sorted(dedup_set), 1):
        print(f"  {i}. {key}")

    # Assertions
    assert len(dedup_set) == 3, \
        f"❌ FAILED: Expected 3 unique keys (same params in different modes should be distinct), got {len(dedup_set)}"

    print("✅ PASSED: Deduplication set correctly distinguishes modes")
    return True


def test_backward_compatibility():
    """Test that the mode parameter is optional (backward compatibility)"""
    print("\n" + "=" * 80)
    print("TEST 4: Backward Compatibility")
    print("=" * 80)

    mutation = {'learning_rate': 0.01, 'batch_size': 32}

    # Should work without mode parameter
    try:
        key1 = _normalize_mutation_key(mutation)
        print(f"Key without mode parameter: {key1}")

        key2 = _normalize_mutation_key(mutation, mode=None)
        print(f"Key with mode=None:         {key2}")

        assert key1 == key2, \
            "❌ FAILED: Keys should be identical when mode is not provided or None"

        print("✅ PASSED: Backward compatibility maintained")
        return True

    except Exception as e:
        print(f"❌ FAILED: Backward compatibility broken: {e}")
        return False


def test_real_world_scenario():
    """Test a real-world scenario with multiple experiments"""
    print("\n" + "=" * 80)
    print("TEST 5: Real-World Scenario")
    print("=" * 80)

    # Create temporary CSV with realistic experiment data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        temp_csv = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=[
            'experiment_id', 'repository', 'model',
            'hyperparam_learning_rate', 'hyperparam_epochs', 'hyperparam_batch_size'
        ])
        writer.writeheader()

        # 5 non-parallel experiments with learning_rate=0.01
        for i in range(5):
            writer.writerow({
                'experiment_id': f'examples_mnist_train_20251205_1200{i:02d}',
                'repository': 'examples',
                'model': 'mnist',
                'hyperparam_learning_rate': '0.01',
                'hyperparam_epochs': str(10 + i),
                'hyperparam_batch_size': '32'
            })

        # 5 parallel experiments with learning_rate=0.01 (same value!)
        for i in range(5):
            writer.writerow({
                'experiment_id': f'examples_mnist_parallel_20251205_1300{i:02d}',
                'repository': 'examples',
                'model': 'mnist',
                'hyperparam_learning_rate': '0.01',
                'hyperparam_epochs': str(10 + i),
                'hyperparam_batch_size': '32'
            })

    try:
        # Extract mutations
        mutations, stats = extract_mutations_from_csv(temp_csv)

        print(f"Total mutations extracted: {len(mutations)}")

        # Count by mode BEFORE calling build_dedup_set (which pops __mode__)
        nonparallel_count = sum(1 for m in mutations if m.get('__mode__') == 'nonparallel')
        parallel_count = sum(1 for m in mutations if m.get('__mode__') == 'parallel')

        print(f"Non-parallel experiments: {nonparallel_count}")
        print(f"Parallel experiments: {parallel_count}")

        # Build dedup set (this will pop __mode__ from mutations)
        dedup_set = build_dedup_set(mutations)

        print(f"Unique mutation keys: {len(dedup_set)}")

        # Assertions
        assert len(mutations) == 10, \
            f"❌ FAILED: Expected 10 mutations, got {len(mutations)}"

        assert nonparallel_count == 5, \
            f"❌ FAILED: Expected 5 non-parallel mutations, got {nonparallel_count}"

        assert parallel_count == 5, \
            f"❌ FAILED: Expected 5 parallel mutations, got {parallel_count}"

        # Even though hyperparameters are the same, modes are different, so all should be unique
        assert len(dedup_set) == 10, \
            f"❌ FAILED: Expected 10 unique keys (same params in different modes), got {len(dedup_set)}"

        print("✅ PASSED: Real-world scenario handled correctly")
        print("   Same hyperparameters in parallel and non-parallel modes are treated as distinct")
        return True

    finally:
        # Clean up temporary file
        temp_csv.unlink()


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "█" * 80)
    print("█ DEDUPLICATION MODE DISTINCTION TEST SUITE")
    print("█" * 80)

    tests = [
        ("Mode Distinction in _normalize_mutation_key", test_normalize_mutation_key_with_mode),
        ("Mode Detection from experiment_id", test_extract_mutations_from_csv_mode_detection),
        ("Mode Distinction in build_dedup_set", test_build_dedup_set_mode_distinction),
        ("Backward Compatibility", test_backward_compatibility),
        ("Real-World Scenario", test_real_world_scenario),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print("=" * 80)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
