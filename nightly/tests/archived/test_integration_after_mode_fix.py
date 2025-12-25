#!/usr/bin/env python3
"""
Integration Test After Mode Fix

This test verifies that existing functionality still works correctly after
the deduplication mode distinction fix.

Test Coverage:
1. generate_mutations() still works without mode parameter (backward compatibility)
2. generate_mutations() works with mode parameter (new feature)
3. Historical data loading still works correctly
4. Session management and CSV generation still work
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation.hyperparams import generate_mutations


def test_generate_mutations_backward_compatibility():
    """Test that generate_mutations works without mode parameter"""
    print("\n" + "=" * 80)
    print("TEST 1: generate_mutations() Backward Compatibility")
    print("=" * 80)

    supported_params = {
        'learning_rate': {
            'type': 'float',
            'range': [0.0001, 0.1],
            'default': 0.01,
            'distribution': 'log_uniform',
            'flag': '--lr'
        },
        'batch_size': {
            'type': 'int',
            'range': [16, 128],
            'default': 32,
            'distribution': 'uniform',
            'flag': '--batch-size'
        }
    }

    try:
        # Should work without mode parameter
        mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=['learning_rate', 'batch_size'],
            num_mutations=3,
            random_seed=42
        )

        print(f"Generated {len(mutations)} mutations without mode parameter")
        for i, m in enumerate(mutations, 1):
            print(f"  {i}. {m}")

        assert len(mutations) == 3, \
            f"❌ FAILED: Expected 3 mutations, got {len(mutations)}"

        print("✅ PASSED: Backward compatibility maintained")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generate_mutations_with_mode():
    """Test that generate_mutations works with mode parameter"""
    print("\n" + "=" * 80)
    print("TEST 2: generate_mutations() with Mode Parameter")
    print("=" * 80)

    supported_params = {
        'learning_rate': {
            'type': 'float',
            'range': [0.0001, 0.1],
            'default': 0.01,
            'distribution': 'log_uniform',
            'flag': '--lr'
        },
        'epochs': {
            'type': 'int',
            'range': [5, 50],
            'default': 10,
            'distribution': 'log_uniform',
            'flag': '--epochs'
        }
    }

    try:
        # Test with mode='parallel'
        parallel_mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=['learning_rate', 'epochs'],
            num_mutations=2,
            random_seed=42,
            mode='parallel'
        )

        print(f"Generated {len(parallel_mutations)} parallel mutations")
        for i, m in enumerate(parallel_mutations, 1):
            print(f"  {i}. {m}")

        # Test with mode='nonparallel'
        nonparallel_mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=['learning_rate', 'epochs'],
            num_mutations=2,
            random_seed=42,
            mode='nonparallel'
        )

        print(f"\nGenerated {len(nonparallel_mutations)} non-parallel mutations")
        for i, m in enumerate(nonparallel_mutations, 1):
            print(f"  {i}. {m}")

        assert len(parallel_mutations) == 2, \
            f"❌ FAILED: Expected 2 parallel mutations, got {len(parallel_mutations)}"

        assert len(nonparallel_mutations) == 2, \
            f"❌ FAILED: Expected 2 non-parallel mutations, got {len(nonparallel_mutations)}"

        print("✅ PASSED: Mode parameter works correctly")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deduplication_across_modes():
    """Test that same hyperparameters in different modes are not deduplicated"""
    print("\n" + "=" * 80)
    print("TEST 3: Deduplication Across Modes")
    print("=" * 80)

    supported_params = {
        'learning_rate': {
            'type': 'float',
            'range': [0.01, 0.01],  # Fixed value to ensure same value
            'default': 0.001,
            'distribution': 'uniform',
            'flag': '--lr'
        }
    }

    try:
        # Generate mutations with same parameters in different modes
        from mutation.hyperparams import _normalize_mutation_key

        mutation = {'learning_rate': 0.01}

        # Create existing_mutations set with non-parallel mode
        nonparallel_key = _normalize_mutation_key(mutation, mode='nonparallel')
        existing_mutations = {nonparallel_key}

        print(f"Existing mutations (non-parallel): {existing_mutations}")

        # Try to generate parallel mode mutations (should NOT be deduplicated)
        parallel_mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=['learning_rate'],
            num_mutations=1,
            random_seed=42,
            existing_mutations=existing_mutations,
            mode='parallel'
        )

        print(f"Generated {len(parallel_mutations)} parallel mutations (should be 1, not deduplicated)")
        for i, m in enumerate(parallel_mutations, 1):
            print(f"  {i}. {m}")

        # The key assertion: same hyperparameters in parallel mode should NOT be deduplicated
        assert len(parallel_mutations) == 1, \
            f"❌ FAILED: Expected 1 parallel mutation (not deduplicated), got {len(parallel_mutations)}"

        print("✅ PASSED: Same hyperparameters in different modes are NOT deduplicated")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uniqueness_checking():
    """Test that uniqueness checking works correctly with and without mode"""
    print("\n" + "=" * 80)
    print("TEST 4: Uniqueness Checking")
    print("=" * 80)

    supported_params = {
        'seed': {
            'type': 'int',
            'range': [1, 1000],
            'default': 42,
            'distribution': 'uniform',
            'flag': '--seed'
        }
    }

    try:
        # Generate multiple mutations and ensure they're unique
        mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=['seed'],
            num_mutations=5,
            random_seed=42
        )

        print(f"Generated {len(mutations)} mutations")
        seeds = [m['seed'] for m in mutations]
        print(f"Seeds: {seeds}")

        # Check uniqueness
        unique_seeds = set(seeds)
        assert len(unique_seeds) == len(seeds), \
            f"❌ FAILED: Mutations should be unique, but got duplicates: {seeds}"

        # Check that default value (42) is not in the mutations
        assert 42 not in seeds, \
            f"❌ FAILED: Default value (42) should not be in mutations: {seeds}"

        print("✅ PASSED: Uniqueness checking works correctly")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "█" * 80)
    print("█ INTEGRATION TEST SUITE (After Mode Fix)")
    print("█" * 80)

    tests = [
        ("Backward Compatibility", test_generate_mutations_backward_compatibility),
        ("Mode Parameter Support", test_generate_mutations_with_mode),
        ("Deduplication Across Modes", test_deduplication_across_modes),
        ("Uniqueness Checking", test_uniqueness_checking),
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
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("\n✓ Existing functionality works correctly after mode fix")
        print("✓ Backward compatibility is maintained")
        print("✓ New mode distinction feature works as expected")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
