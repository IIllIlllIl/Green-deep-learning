#!/usr/bin/env python3
"""
Test Script for Multi-Parameter Combination Deduplication

Purpose:
    Test the new filter_params feature in build_dedup_set() that allows
    deduplication based on specific parameter combinations, even when
    historical data contains additional parameters.

Test Cases:
    1. Test filter_params with single parameter
    2. Test filter_params with multiple parameters
    3. Test filter_params=None (original behavior)
    4. Test filter_params with "all" expansion
    5. Test integration with generate_mutations()
    6. Test real-world scenario: different mutations with overlapping params

Usage:
    python3 tests/test_multi_param_dedup.py

Author: Energy DL Team
Created: 2026-01-11
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mutation.dedup import build_dedup_set
from mutation.hyperparams import generate_mutations, _normalize_mutation_key


class MultiParamDedupTestRunner:
    """Test runner for multi-parameter deduplication"""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def assert_equal(self, actual, expected, test_name: str) -> bool:
        """Assert that actual equals expected"""
        if actual == expected:
            print(f"  ✓ {test_name}: {actual} == {expected}")
            return True
        else:
            print(f"  ✗ {test_name}: {actual} != {expected}")
            return False

    def assert_true(self, condition: bool, test_name: str) -> bool:
        """Assert that condition is True"""
        if condition:
            print(f"  ✓ {test_name}")
            return True
        else:
            print(f"  ✗ {test_name}")
            return False

    def test_filter_single_param(self) -> bool:
        """Test 1: Filter by single parameter"""
        print("\nTest 1: Filter by single parameter")
        print("-" * 80)

        # Historical data with multiple parameters
        historical_mutations = [
            {"learning_rate": 0.01, "batch_size": 32, "dropout": 0.5, "__mode__": "nonparallel"},
            {"learning_rate": 0.001, "batch_size": 64, "dropout": 0.3, "__mode__": "nonparallel"},
            {"learning_rate": 0.01, "batch_size": 16, "dropout": 0.2, "__mode__": "nonparallel"},  # lr=0.01 again
        ]

        # Filter only by learning_rate
        dedup_set = build_dedup_set(historical_mutations, filter_params=["learning_rate"])

        # Should have only 2 unique learning_rate values (0.01, 0.001)
        # even though other parameters differ
        test1 = self.assert_equal(len(dedup_set), 2, "Dedup set size (lr only)")

        # Verify the keys contain only learning_rate
        test2 = True
        for key in dedup_set:
            # Key should be like: (('__mode__', 'nonparallel'), ('learning_rate', '...'))
            param_names = [item[0] for item in key]
            if "learning_rate" not in param_names:
                test2 = False
                print(f"  ✗ Missing learning_rate in key: {key}")
            if "batch_size" in param_names or "dropout" in param_names:
                test2 = False
                print(f"  ✗ Unexpected params in filtered key: {key}")

        if test2:
            print(f"  ✓ Keys contain only learning_rate (+ mode)")

        return all([test1, test2])

    def test_filter_multiple_params(self) -> bool:
        """Test 2: Filter by multiple parameters"""
        print("\nTest 2: Filter by multiple parameters")
        print("-" * 80)

        # Historical data
        historical_mutations = [
            {"a": 1, "b": 2, "c": 3, "__mode__": "nonparallel"},
            {"a": 1, "b": 2, "c": 1, "__mode__": "nonparallel"},  # a,b same, c different
            {"a": 2, "b": 2, "c": 2, "__mode__": "nonparallel"},
            {"a": 2, "b": 2, "c": 3, "__mode__": "nonparallel"},  # a,b same as above
            {"a": 1, "b": 2, "c": 1, "__mode__": "nonparallel"},  # duplicate of #2
        ]

        # Filter only by a and b
        dedup_set = build_dedup_set(historical_mutations, filter_params=["a", "b"])

        # Should have 2 unique (a,b) combinations: (1,2) and (2,2)
        test1 = self.assert_equal(len(dedup_set), 2, "Dedup set size (a,b combination)")

        return test1

    def test_filter_params_none(self) -> bool:
        """Test 3: filter_params=None should use all parameters (original behavior)"""
        print("\nTest 3: filter_params=None (original behavior)")
        print("-" * 80)

        historical_mutations = [
            {"lr": 0.01, "bs": 32, "__mode__": "nonparallel"},
            {"lr": 0.01, "bs": 64, "__mode__": "nonparallel"},  # Different bs
            {"lr": 0.01, "bs": 32, "__mode__": "nonparallel"},  # Duplicate
        ]

        # Don't filter - use all params
        dedup_set = build_dedup_set(historical_mutations, filter_params=None)

        # Should have 2 unique combinations: (0.01,32) and (0.01,64)
        test1 = self.assert_equal(len(dedup_set), 2, "Dedup set size (all params)")

        return test1

    def test_integration_with_generate_mutations(self) -> bool:
        """Test 4: Integration with generate_mutations()"""
        print("\nTest 4: Integration with generate_mutations()")
        print("-" * 80)

        # Define supported params
        supported_params = {
            "learning_rate": {
                "type": "float",
                "flag": "--lr",
                "range": [0.001, 0.1],
                "default": 0.01,
                "distribution": "log_uniform"
            },
            "batch_size": {
                "type": "int",
                "flag": "--bs",
                "range": [16, 128],
                "default": 32,
                "distribution": "uniform"
            },
            "dropout": {
                "type": "float",
                "flag": "--dropout",
                "range": [0.0, 0.5],
                "default": 0.2,
                "distribution": "uniform"
            }
        }

        # Historical mutations with all three params
        historical_mutations = [
            {"learning_rate": 0.05, "batch_size": 64, "dropout": 0.3, "__mode__": "nonparallel"},
            {"learning_rate": 0.02, "batch_size": 32, "dropout": 0.1, "__mode__": "nonparallel"},
        ]

        # Build dedup set filtering only learning_rate and batch_size
        dedup_set = build_dedup_set(
            historical_mutations,
            filter_params=["learning_rate", "batch_size"]
        )

        print(f"  Historical dedup set size: {len(dedup_set)}")

        # Generate new mutations for learning_rate and batch_size only
        new_mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=["learning_rate", "batch_size"],
            num_mutations=3,
            existing_mutations=dedup_set,
            mode="nonparallel"
        )

        # Check no new mutations match historical (lr, bs) combinations
        test1 = True
        for mutation in new_mutations:
            key = _normalize_mutation_key(mutation, mode="nonparallel")
            if key in dedup_set:
                print(f"  ✗ Generated duplicate: {mutation}")
                test1 = False

        if test1:
            print(f"  ✓ No new mutations match historical (lr, bs) combinations")

        # All mutations should be unique
        mutation_keys = [_normalize_mutation_key(m, mode="nonparallel") for m in new_mutations]
        test2 = self.assert_equal(
            len(mutation_keys),
            len(set(mutation_keys)),
            "All new mutations unique"
        )

        return all([test1, test2])

    def test_real_world_scenario(self) -> bool:
        """Test 5: Real-world scenario - historical data has extra params"""
        print("\nTest 5: Real-world scenario")
        print("-" * 80)

        print("  Scenario: Historical experiments varied all params")
        print("           Current experiment varies only lr and bs")

        # Historical data has lr, bs, dropout
        historical_mutations = [
            {"learning_rate": 0.01, "batch_size": 32, "dropout": 0.5, "__mode__": "nonparallel"},
            {"learning_rate": 0.001, "batch_size": 64, "dropout": 0.3, "__mode__": "nonparallel"},
            {"learning_rate": 0.01, "batch_size": 64, "dropout": 0.2, "__mode__": "nonparallel"},
        ]

        # Current experiment only cares about lr and bs
        current_mutate_params = ["learning_rate", "batch_size"]

        # Build dedup set with filter
        dedup_set_filtered = build_dedup_set(
            historical_mutations,
            filter_params=current_mutate_params
        )

        print(f"  Historical mutations (all params): {len(historical_mutations)}")
        print(f"  Unique (lr, bs) combinations: {len(dedup_set_filtered)}")

        # Should have 3 unique (lr, bs) combinations
        test1 = self.assert_equal(len(dedup_set_filtered), 3, "Unique (lr, bs) combinations")

        # Now test that (0.01, 32) would be detected as duplicate
        test_mutation = {"learning_rate": 0.01, "batch_size": 32}
        test_key = _normalize_mutation_key(test_mutation, mode="nonparallel")

        test2 = self.assert_true(
            test_key in dedup_set_filtered,
            "(0.01, 32) detected as duplicate"
        )

        # But (0.01, 16) should NOT be duplicate
        test_mutation2 = {"learning_rate": 0.01, "batch_size": 16}
        test_key2 = _normalize_mutation_key(test_mutation2, mode="nonparallel")

        test3 = self.assert_true(
            test_key2 not in dedup_set_filtered,
            "(0.01, 16) NOT detected as duplicate"
        )

        return all([test1, test2, test3])

    def test_user_example_scenario(self) -> bool:
        """Test 6: User's example from requirement"""
        print("\nTest 6: User's example scenario")
        print("-" * 80)

        print("  Example: Model with params a, b, c (all default to 1)")

        # Simulate the mutation sequence from user's example
        mutations = [
            {"a": 1, "b": 2, "c": 3, "__mode__": "nonparallel"},  # 1st mutation
            {"a": 1, "b": 2, "c": 1, "__mode__": "nonparallel"},  # 2nd mutation
            {"a": 2, "b": 2, "c": 2, "__mode__": "nonparallel"},  # 3rd mutation
            {"a": 2, "b": 2, "c": 3, "__mode__": "nonparallel"},  # 4th mutation
            {"a": 1, "b": 2, "c": 1, "__mode__": "nonparallel"},  # 5th mutation (should be duplicate)
        ]

        # Build dedup set with all params
        dedup_set = build_dedup_set(mutations, filter_params=["a", "b", "c"])

        # Should have 4 unique combinations (5th is duplicate of 2nd)
        test1 = self.assert_equal(len(dedup_set), 4, "Unique combinations (5th is duplicate)")

        # Verify specific combinations
        key_1_2_3 = _normalize_mutation_key({"a": 1, "b": 2, "c": 3}, mode="nonparallel")
        key_1_2_1 = _normalize_mutation_key({"a": 1, "b": 2, "c": 1}, mode="nonparallel")
        key_2_2_2 = _normalize_mutation_key({"a": 2, "b": 2, "c": 2}, mode="nonparallel")
        key_2_2_3 = _normalize_mutation_key({"a": 2, "b": 2, "c": 3}, mode="nonparallel")

        test2 = self.assert_true(key_1_2_3 in dedup_set, "(1,2,3) in dedup set")
        test3 = self.assert_true(key_1_2_1 in dedup_set, "(1,2,1) in dedup set")
        test4 = self.assert_true(key_2_2_2 in dedup_set, "(2,2,2) in dedup set")
        test5 = self.assert_true(key_2_2_3 in dedup_set, "(2,2,3) in dedup set")

        return all([test1, test2, test3, test4, test5])

    def run_all_tests(self) -> int:
        """Run all tests and return exit code"""
        print("=" * 80)
        print("Multi-Parameter Combination Deduplication Test Suite")
        print("=" * 80)

        tests = [
            ("Filter single parameter", self.test_filter_single_param),
            ("Filter multiple parameters", self.test_filter_multiple_params),
            ("filter_params=None", self.test_filter_params_none),
            ("Integration with generate_mutations", self.test_integration_with_generate_mutations),
            ("Real-world scenario", self.test_real_world_scenario),
            ("User's example scenario", self.test_user_example_scenario),
        ]

        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    self.passed += 1
                    print(f"\n✓ {test_name} PASSED")
                else:
                    self.failed += 1
                    print(f"\n✗ {test_name} FAILED")
            except Exception as e:
                self.failed += 1
                print(f"\n✗ {test_name} FAILED with exception: {e}")
                import traceback
                traceback.print_exc()

        # Print summary
        print()
        print("=" * 80)
        print("Test Summary")
        print("=" * 80)
        print(f"Total tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print()

        if self.failed == 0:
            print("✓ All tests passed!")
            print("=" * 80)
            return 0
        else:
            print("✗ Some tests failed")
            print("=" * 80)
            return 1


def main():
    """Main entry point"""
    runner = MultiParamDedupTestRunner()
    exit_code = runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
