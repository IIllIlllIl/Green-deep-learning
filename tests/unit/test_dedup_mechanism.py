#!/usr/bin/env python3
"""
Test Script for Inter-Round Deduplication Mechanism

Purpose:
    Validate the inter-round hyperparameter deduplication system by:
    1. Testing CSV extraction of historical hyperparameters
    2. Testing deduplication set building
    3. Testing that generate_mutations() respects historical data
    4. Testing end-to-end integration with real CSV files

Test Cases:
    1. Extract mutations from single CSV file
    2. Extract mutations from multiple CSV files
    3. Filter by repository/model
    4. Build deduplication set
    5. Generate mutations with historical deduplication
    6. Verify no duplicates across rounds (integration test)

Usage:
    python3 tests/unit/test_dedup_mechanism.py

Author: Mutation Energy Profiler Team
Created: 2025-11-26
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mutation.dedup import (
    extract_mutations_from_csv,
    load_historical_mutations,
    build_dedup_set,
    print_dedup_statistics,
)
from mutation.hyperparams import generate_mutations, _normalize_mutation_key


class DedupTestRunner:
    """Test runner for deduplication mechanism"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.passed = 0
        self.failed = 0

        # Test CSV files
        self.csv_files = [
            self.project_root / "results" / "defualt" / "summary.csv",
            self.project_root / "results" / "mutation_1x" / "summary.csv",
            self.project_root / "results" / "mutation_2x_20251122_175401" / "summary_safe.csv",
        ]

    def assert_true(self, condition: bool, test_name: str) -> bool:
        """Assert that condition is True"""
        if condition:
            print(f"  ✓ {test_name}")
            return True
        else:
            print(f"  ✗ {test_name}")
            return False

    def assert_equal(self, actual, expected, test_name: str) -> bool:
        """Assert that actual equals expected"""
        if actual == expected:
            print(f"  ✓ {test_name}: {actual} == {expected}")
            return True
        else:
            print(f"  ✗ {test_name}: {actual} != {expected}")
            return False

    def assert_greater(self, actual, minimum, test_name: str) -> bool:
        """Assert that actual is greater than minimum"""
        if actual > minimum:
            print(f"  ✓ {test_name}: {actual} > {minimum}")
            return True
        else:
            print(f"  ✗ {test_name}: {actual} <= {minimum}")
            return False

    def test_extract_single_csv(self) -> bool:
        """Test 1: Extract mutations from a single CSV file"""
        print("\nTest 1: Extract mutations from single CSV")
        print("-" * 80)

        csv_path = self.csv_files[0]  # default experiments

        if not csv_path.exists():
            print(f"  ⚠ CSV file not found: {csv_path}")
            return False

        mutations, stats = extract_mutations_from_csv(csv_path)

        # Check we extracted some mutations
        test1 = self.assert_greater(len(mutations), 0, "Extracted mutations")

        # Check statistics
        test2 = self.assert_greater(stats["total"], 0, "Total rows counted")
        test3 = self.assert_equal(stats["extracted"], len(mutations), "Extracted count matches")

        # Check mutation structure
        test4 = True
        if mutations:
            first_mutation = mutations[0]
            # Should have at least one hyperparameter
            test4 = self.assert_greater(len(first_mutation), 0, "Mutation has hyperparameters")

        return all([test1, test2, test3, test4])

    def test_extract_multiple_csvs(self) -> bool:
        """Test 2: Extract mutations from multiple CSV files"""
        print("\nTest 2: Extract mutations from multiple CSVs")
        print("-" * 80)

        existing_files = [f for f in self.csv_files if f.exists()]

        if len(existing_files) == 0:
            print("  ⚠ No CSV files found")
            return False

        print(f"  Loading from {len(existing_files)} files")

        mutations, stats = load_historical_mutations(existing_files)

        # Check we loaded from multiple files
        test1 = self.assert_equal(stats["successful_files"], len(existing_files), "All files loaded")

        # Check we have mutations
        test2 = self.assert_greater(len(mutations), 0, "Total mutations loaded")
        test3 = self.assert_equal(stats["total_extracted"], len(mutations), "Count matches")

        # Check by_model statistics exist
        test4 = self.assert_greater(len(stats.get("by_model", {})), 0, "Model breakdown exists")

        return all([test1, test2, test3, test4])

    def test_filter_by_model(self) -> bool:
        """Test 3: Filter mutations by repository/model"""
        print("\nTest 3: Filter by repository/model")
        print("-" * 80)

        csv_path = self.csv_files[0]

        if not csv_path.exists():
            print(f"  ⚠ CSV file not found: {csv_path}")
            return False

        # Test filtering by repository
        repo = "examples"
        mutations_filtered, stats_filtered = extract_mutations_from_csv(
            csv_path,
            filter_by_repo=repo
        )

        # Should have fewer mutations than unfiltered
        mutations_all, stats_all = extract_mutations_from_csv(csv_path)

        test1 = self.assert_true(
            len(mutations_filtered) <= len(mutations_all),
            f"Filtered by repo ({len(mutations_filtered)}) <= all ({len(mutations_all)})"
        )

        # Test filtering by model
        model = "mnist"
        mutations_model, stats_model = extract_mutations_from_csv(
            csv_path,
            filter_by_model=model
        )

        test2 = self.assert_true(
            len(mutations_model) <= len(mutations_all),
            f"Filtered by model ({len(mutations_model)}) <= all ({len(mutations_all)})"
        )

        return all([test1, test2])

    def test_build_dedup_set(self) -> bool:
        """Test 4: Build deduplication set"""
        print("\nTest 4: Build deduplication set")
        print("-" * 80)

        # Create some test mutations
        test_mutations = [
            {"epochs": 10.0, "learning_rate": 0.001},
            {"epochs": 20.0, "learning_rate": 0.01},
            {"epochs": 10.0, "learning_rate": 0.001},  # Duplicate
        ]

        dedup_set = build_dedup_set(test_mutations)

        # Should have 2 unique mutations (third is duplicate)
        test1 = self.assert_equal(len(dedup_set), 2, "Duplicate removed")

        # Check set contains normalized tuples
        test2 = self.assert_true(
            all(isinstance(key, tuple) for key in dedup_set),
            "All keys are tuples"
        )

        return all([test1, test2])

    def test_generate_with_dedup(self) -> bool:
        """Test 5: Generate mutations with historical deduplication"""
        print("\nTest 5: Generate mutations with historical deduplication")
        print("-" * 80)

        # Create a simple config for testing
        supported_params = {
            "epochs": {
                "type": "int",
                "flag": "--epochs",
                "range": [5, 15],
                "default": 10,
                "distribution": "log_uniform"
            },
            "learning_rate": {
                "type": "float",
                "flag": "--lr",
                "range": [0.001, 0.1],
                "default": 0.01,
                "distribution": "log_uniform"
            }
        }

        # Create historical mutations set
        historical_mutations = [
            {"epochs": 8.0, "learning_rate": 0.005},
            {"epochs": 12.0, "learning_rate": 0.02},
        ]
        dedup_set = build_dedup_set(historical_mutations)

        print(f"  Historical mutations: {len(dedup_set)}")

        # Generate new mutations with deduplication
        new_mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=["epochs", "learning_rate"],
            num_mutations=3,
            existing_mutations=dedup_set
        )

        # Check we generated mutations
        test1 = self.assert_greater(len(new_mutations), 0, "Generated mutations")

        # Check none match historical mutations
        test2 = True
        for mutation in new_mutations:
            key = _normalize_mutation_key(mutation)
            if key in dedup_set:
                print(f"  ✗ Found duplicate in historical: {mutation}")
                test2 = False
                break

        if test2:
            print(f"  ✓ No duplicates with historical mutations")

        # Check all generated mutations are unique
        generated_keys = [_normalize_mutation_key(m) for m in new_mutations]
        test3 = self.assert_equal(
            len(generated_keys),
            len(set(generated_keys)),
            "All generated mutations are unique"
        )

        return all([test1, test2, test3])

    def test_integration_with_real_data(self) -> bool:
        """Test 6: Integration test with real CSV files"""
        print("\nTest 6: Integration test with real CSV data")
        print("-" * 80)

        existing_files = [f for f in self.csv_files if f.exists()]

        if len(existing_files) == 0:
            print("  ⚠ No CSV files found")
            return False

        # Load all historical mutations
        print(f"  Loading from {len(existing_files)} CSV files...")
        all_mutations, stats = load_historical_mutations(existing_files)

        print(f"  Loaded {len(all_mutations)} total mutations")

        # Build deduplication set
        dedup_set = build_dedup_set(all_mutations)
        print(f"  Unique mutations: {len(dedup_set)}")

        # Try to generate new mutations for a real model (examples/mnist)
        supported_params = {
            "epochs": {
                "type": "int",
                "flag": "--epochs",
                "range": [5, 20],
                "default": 10,
                "distribution": "log_uniform"
            },
            "learning_rate": {
                "type": "float",
                "flag": "--lr",
                "range": [0.001, 0.1],
                "default": 0.01,
                "distribution": "log_uniform"
            }
        }

        # Generate 5 new mutations
        print("\n  Generating 5 new mutations with historical deduplication...")
        new_mutations = generate_mutations(
            supported_params=supported_params,
            mutate_params=["epochs", "learning_rate"],
            num_mutations=5,
            existing_mutations=dedup_set
        )

        # Verify no overlaps with historical data
        test1 = True
        for mutation in new_mutations:
            key = _normalize_mutation_key(mutation)
            if key in dedup_set:
                print(f"  ✗ Generated duplicate mutation: {mutation}")
                test1 = False

        if test1:
            print(f"\n  ✓ All {len(new_mutations)} generated mutations are unique from historical data")

        return test1

    def run_all_tests(self) -> int:
        """Run all tests and return exit code"""
        print("=" * 80)
        print("Inter-Round Deduplication Test Suite")
        print("=" * 80)
        print(f"Project root: {self.project_root}")
        print(f"CSV files to test: {len(self.csv_files)}")
        for csv_file in self.csv_files:
            exists = "✓" if csv_file.exists() else "✗"
            print(f"  {exists} {csv_file.relative_to(self.project_root)}")
        print("=" * 80)

        tests = [
            ("Extract single CSV", self.test_extract_single_csv),
            ("Extract multiple CSVs", self.test_extract_multiple_csvs),
            ("Filter by model", self.test_filter_by_model),
            ("Build dedup set", self.test_build_dedup_set),
            ("Generate with dedup", self.test_generate_with_dedup),
            ("Integration test", self.test_integration_with_real_data),
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
    runner = DedupTestRunner()
    exit_code = runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
