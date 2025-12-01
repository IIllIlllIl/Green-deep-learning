#!/usr/bin/env python3
"""
Test Script for CSV Aggregation

Purpose:
    Validate the functionality of aggregate_csvs.py by running various test cases
    and checking the output for correctness.

Test Cases:
    1. Dry run test (no output file created)
    2. Basic aggregation test (with mnist_ff filtering)
    3. Aggregation without filtering (keep mnist_ff)
    4. Data integrity validation
    5. Statistics validation

Usage:
    python3 scripts/test_aggregate_csvs.py

Author: Mutation Energy Profiler Team
Created: 2025-11-26
"""

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class TestRunner:
    """Test runner for CSV aggregation script"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.script_path = self.project_root / "scripts" / "aggregate_csvs.py"
        self.test_output = self.project_root / "results" / "test_summary_all.csv"
        self.passed = 0
        self.failed = 0

    def run_command(self, args: List[str]) -> Tuple[int, str, str]:
        """
        Run a command and return exit code, stdout, stderr.

        Args:
            args: Command arguments (including script path)

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        result = subprocess.run(
            args,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        return result.returncode, result.stdout, result.stderr

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

    def test_dry_run(self) -> bool:
        """Test 1: Dry run should not create output file"""
        print("\nTest 1: Dry run (no output file)")
        print("-" * 80)

        # Remove test output if exists
        if self.test_output.exists():
            self.test_output.unlink()

        # Run with --dry-run
        exit_code, stdout, stderr = self.run_command([
            "python3",
            str(self.script_path),
            "--output", str(self.test_output),
            "--dry-run"
        ])

        # Check exit code
        test1 = self.assert_equal(exit_code, 0, "Exit code is 0")

        # Check output file was not created
        test2 = self.assert_true(not self.test_output.exists(), "No output file created")

        # Check stdout contains "Dry run"
        test3 = self.assert_true("Dry run" in stdout, "Stdout contains 'Dry run'")

        return all([test1, test2, test3])

    def test_basic_aggregation(self) -> bool:
        """Test 2: Basic aggregation with mnist_ff filtering"""
        print("\nTest 2: Basic aggregation (filter mnist_ff)")
        print("-" * 80)

        # Remove test output if exists
        if self.test_output.exists():
            self.test_output.unlink()

        # Run aggregation
        exit_code, stdout, stderr = self.run_command([
            "python3",
            str(self.script_path),
            "--output", str(self.test_output)
        ])

        # Check exit code
        test1 = self.assert_equal(exit_code, 0, "Exit code is 0")

        # Check output file was created
        test2 = self.assert_true(self.test_output.exists(), "Output file created")

        if not self.test_output.exists():
            return False

        # Read and validate output
        with open(self.test_output, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Check we have data
            test3 = self.assert_true(len(rows) > 0, f"Output has data ({len(rows)} rows)")

            # Check experiment_source column exists
            test4 = self.assert_true(
                "experiment_source" in rows[0],
                "experiment_source column exists"
            )

            # Check no mnist_ff experiments
            mnist_ff_count = sum(1 for row in rows if row.get("model") == "mnist_ff")
            test5 = self.assert_equal(mnist_ff_count, 0, "No mnist_ff experiments")

            # Check all three sources are present
            sources = set(row.get("experiment_source") for row in rows)
            expected_sources = {"default", "mutation_1x", "mutation_2x_safe"}
            test6 = self.assert_equal(sources, expected_sources, "All sources present")

        return all([test1, test2, test3, test4, test5, test6])

    def test_keep_mnist_ff(self) -> bool:
        """Test 3: Aggregation without filtering (keep mnist_ff)"""
        print("\nTest 3: Aggregation without filtering (keep mnist_ff)")
        print("-" * 80)

        # Remove test output if exists
        if self.test_output.exists():
            self.test_output.unlink()

        # Run aggregation with --keep-mnist-ff
        exit_code, stdout, stderr = self.run_command([
            "python3",
            str(self.script_path),
            "--output", str(self.test_output),
            "--keep-mnist-ff"
        ])

        # Check exit code
        test1 = self.assert_equal(exit_code, 0, "Exit code is 0")

        # Check output file was created
        test2 = self.assert_true(self.test_output.exists(), "Output file created")

        if not self.test_output.exists():
            return False

        # Read and validate output
        with open(self.test_output, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Check we have mnist_ff experiments
            mnist_ff_count = sum(1 for row in rows if row.get("model") == "mnist_ff")
            test3 = self.assert_true(mnist_ff_count > 0, f"Has mnist_ff experiments ({mnist_ff_count})")

        return all([test1, test2, test3])

    def test_data_integrity(self) -> bool:
        """Test 4: Data integrity validation"""
        print("\nTest 4: Data integrity validation")
        print("-" * 80)

        # Use output from test 2
        if not self.test_output.exists():
            print("  ⚠ Test output not found, running aggregation first...")
            exit_code, _, _ = self.run_command([
                "python3",
                str(self.script_path),
                "--output", str(self.test_output)
            ])
            if exit_code != 0:
                return False

        # Read output
        with open(self.test_output, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Check all rows have required columns
            required_columns = {
                "experiment_id", "timestamp", "repository", "model",
                "training_success", "experiment_source"
            }

            test1 = True
            for row in rows:
                if not required_columns.issubset(row.keys()):
                    test1 = False
                    break

            test1 = self.assert_true(test1, "All rows have required columns")

            # Check no duplicate experiment_ids
            exp_ids = [row["experiment_id"] for row in rows]
            unique_ids = set(exp_ids)
            test2 = self.assert_equal(len(exp_ids), len(unique_ids), "No duplicate experiment IDs")

            # Check training_success values are valid
            valid_success_values = {"True", "False"}
            test3 = True
            for row in rows:
                if row.get("training_success") not in valid_success_values:
                    test3 = False
                    break

            test3 = self.assert_true(test3, "All training_success values are valid")

        return all([test1, test2, test3])

    def test_statistics(self) -> bool:
        """Test 5: Statistics validation"""
        print("\nTest 5: Statistics validation")
        print("-" * 80)

        # Count rows in source files
        source_counts = {}

        # Count default
        default_path = self.project_root / "results" / "defualt" / "summary.csv"
        with open(default_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("model") != "mnist_ff"]
            source_counts["default"] = len(rows)

        # Count mutation_1x
        mut1x_path = self.project_root / "results" / "mutation_1x" / "summary.csv"
        with open(mut1x_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("model") != "mnist_ff"]
            source_counts["mutation_1x"] = len(rows)

        # Count mutation_2x_safe
        mut2x_path = self.project_root / "results" / "mutation_2x_20251122_175401" / "summary_safe.csv"
        with open(mut2x_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("model") != "mnist_ff"]
            source_counts["mutation_2x_safe"] = len(rows)

        expected_total = sum(source_counts.values())
        print(f"  Expected total: {expected_total}")
        print(f"    default: {source_counts['default']}")
        print(f"    mutation_1x: {source_counts['mutation_1x']}")
        print(f"    mutation_2x_safe: {source_counts['mutation_2x_safe']}")

        # Count aggregated output
        with open(self.test_output, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            actual_total = len(rows)

            # Count by source
            actual_counts = {}
            for row in rows:
                source = row.get("experiment_source")
                actual_counts[source] = actual_counts.get(source, 0) + 1

        print(f"  Actual total: {actual_total}")
        print(f"    default: {actual_counts.get('default', 0)}")
        print(f"    mutation_1x: {actual_counts.get('mutation_1x', 0)}")
        print(f"    mutation_2x_safe: {actual_counts.get('mutation_2x_safe', 0)}")

        # Check totals match
        test1 = self.assert_equal(actual_total, expected_total, "Total row count matches")

        # Check per-source counts match
        test2 = self.assert_equal(
            actual_counts.get("default", 0),
            source_counts["default"],
            "Default count matches"
        )

        test3 = self.assert_equal(
            actual_counts.get("mutation_1x", 0),
            source_counts["mutation_1x"],
            "Mutation 1x count matches"
        )

        test4 = self.assert_equal(
            actual_counts.get("mutation_2x_safe", 0),
            source_counts["mutation_2x_safe"],
            "Mutation 2x safe count matches"
        )

        return all([test1, test2, test3, test4])

    def run_all_tests(self) -> int:
        """Run all tests and return exit code"""
        print("=" * 80)
        print("CSV Aggregation Test Suite")
        print("=" * 80)
        print(f"Project root: {self.project_root}")
        print(f"Script: {self.script_path}")
        print(f"Test output: {self.test_output}")
        print("=" * 80)

        tests = [
            ("Dry run", self.test_dry_run),
            ("Basic aggregation", self.test_basic_aggregation),
            ("Keep mnist_ff", self.test_keep_mnist_ff),
            ("Data integrity", self.test_data_integrity),
            ("Statistics", self.test_statistics)
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
    runner = TestRunner()
    exit_code = runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
