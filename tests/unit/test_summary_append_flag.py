#!/usr/bin/env python3
"""
Test Script for --skip-summary-append Flag

Purpose:
    Validate that the --skip-summary-append / -S CLI flag correctly controls
    whether experiment results are appended to results/summary_all.csv

Test Cases:
    1. Default behavior: append_to_summary=True
    2. With --skip-summary-append: append_to_summary=False
    3. With -S (shorthand): append_to_summary=False
    4. Runner initialization with append_to_summary parameter
    5. _append_to_summary_all method behavior

Usage:
    python3 tests/unit/test_summary_append_flag.py

Author: Mutation Energy Profiler Team
Created: 2025-11-26
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, call

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mutation.runner import MutationRunner


class SummaryAppendTestRunner:
    """Test runner for --skip-summary-append flag"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
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

    def test_default_append_to_summary(self) -> bool:
        """Test 1: Default behavior should be append_to_summary=True"""
        print("\nTest 1: Default append_to_summary behavior")
        print("-" * 80)

        try:
            runner = MutationRunner()
            test1 = self.assert_equal(runner.append_to_summary, True, "Default append_to_summary is True")
            return test1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def test_explicit_append_false(self) -> bool:
        """Test 2: Explicit append_to_summary=False"""
        print("\nTest 2: Explicit append_to_summary=False")
        print("-" * 80)

        try:
            runner = MutationRunner(append_to_summary=False)
            test1 = self.assert_equal(runner.append_to_summary, False, "append_to_summary is False")
            return test1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def test_explicit_append_true(self) -> bool:
        """Test 3: Explicit append_to_summary=True"""
        print("\nTest 3: Explicit append_to_summary=True")
        print("-" * 80)

        try:
            runner = MutationRunner(append_to_summary=True)
            test1 = self.assert_equal(runner.append_to_summary, True, "append_to_summary is True")
            return test1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def test_append_method_when_false(self) -> bool:
        """Test 4: _append_to_summary_all returns early when append_to_summary=False"""
        print("\nTest 4: _append_to_summary_all with append_to_summary=False")
        print("-" * 80)

        try:
            runner = MutationRunner(append_to_summary=False)

            # Mock subprocess to ensure it's never called
            with patch('mutation.runner.subprocess.run') as mock_run:
                runner._append_to_summary_all()

                # subprocess.run should NOT be called when append_to_summary=False
                test1 = self.assert_true(
                    not mock_run.called,
                    "subprocess.run not called when append_to_summary=False"
                )
                return test1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_append_method_when_true(self) -> bool:
        """Test 5: _append_to_summary_all calls aggregate_csvs when append_to_summary=True"""
        print("\nTest 5: _append_to_summary_all with append_to_summary=True")
        print("-" * 80)

        try:
            runner = MutationRunner(append_to_summary=True)

            # Mock subprocess to verify it's called with correct arguments
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Total experiments: 100\nUnique hyperparameters: 50"
            mock_result.stderr = ""

            with patch('mutation.runner.subprocess.run', return_value=mock_result) as mock_run:
                runner._append_to_summary_all()

                # subprocess.run SHOULD be called when append_to_summary=True
                test1 = self.assert_true(
                    mock_run.called,
                    "subprocess.run called when append_to_summary=True"
                )

                # Check if called with correct script path
                if mock_run.called:
                    call_args = mock_run.call_args
                    called_cmd = call_args[0][0]  # First positional argument
                    aggregate_script = self.project_root / "scripts" / "aggregate_csvs.py"

                    test2 = self.assert_true(
                        str(aggregate_script) in str(called_cmd),
                        "Called with aggregate_csvs.py script"
                    )
                else:
                    test2 = False

                return test1 and test2
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_append_method_script_missing(self) -> bool:
        """Test 6: _append_to_summary_all handles missing aggregate_csvs.py gracefully"""
        print("\nTest 6: _append_to_summary_all with missing script")
        print("-" * 80)

        try:
            runner = MutationRunner(append_to_summary=True)

            # Mock the script existence check to return False
            original_exists = Path.exists
            def mock_exists(self):
                if 'aggregate_csvs.py' in str(self):
                    return False
                return original_exists(self)

            with patch.object(Path, 'exists', mock_exists):
                with patch('mutation.runner.subprocess.run') as mock_run:
                    runner._append_to_summary_all()

                    # subprocess.run should NOT be called when script is missing
                    test1 = self.assert_true(
                        not mock_run.called,
                        "subprocess.run not called when script is missing"
                    )
                    return test1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self) -> int:
        """Run all tests and return exit code"""
        print("=" * 80)
        print("Summary Append Flag Test Suite")
        print("=" * 80)
        print(f"Project root: {self.project_root}")
        print("=" * 80)

        tests = [
            ("Default append_to_summary", self.test_default_append_to_summary),
            ("Explicit append_to_summary=False", self.test_explicit_append_false),
            ("Explicit append_to_summary=True", self.test_explicit_append_true),
            ("Append method when False", self.test_append_method_when_false),
            ("Append method when True", self.test_append_method_when_true),
            ("Append method with missing script", self.test_append_method_script_missing),
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
    runner = SummaryAppendTestRunner()
    exit_code = runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
