#!/usr/bin/env python3
"""
Manual verification script for _append_to_summary_all() fix

This script performs integration-style testing without requiring pytest.
"""

import csv
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock


def test_append_functionality():
    """Manual test of the append functionality"""

    print("=" * 80)
    print("Testing _append_to_summary_all() Fix")
    print("=" * 80)

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\nTemporary directory: {temp_dir}")

    try:
        # Import after path setup
        project_root = Path(__file__).parent.parent.absolute()
        sys.path.insert(0, str(project_root))
        from mutation.runner import MutationRunner

        # Create mock runner
        runner = Mock(spec=MutationRunner)
        runner.results_dir = temp_dir
        runner.append_to_summary = True
        runner.logger = Mock()

        # Create mock session
        runner.session = Mock()
        runner.session.session_dir = temp_dir / "run_test_001"
        runner.session.session_dir.mkdir(parents=True, exist_ok=True)

        # Bind real method
        runner._append_to_summary_all = MutationRunner._append_to_summary_all.__get__(runner)

        # Test 1: Append to new file
        print("\n" + "-" * 80)
        print("TEST 1: Append to new (non-existent) file")
        print("-" * 80)

        # Create session summary
        session_data = [
            {
                "experiment_id": "test_001",
                "timestamp": "2025-12-02T10:00:00",
                "repository": "examples",
                "model": "mnist",
                "training_success": "True",
                "duration_seconds": "100.5"
            },
            {
                "experiment_id": "test_002",
                "timestamp": "2025-12-02T10:05:00",
                "repository": "examples",
                "model": "mnist_rnn",
                "training_success": "True",
                "duration_seconds": "150.2"
            }
        ]

        session_summary = runner.session.session_dir / "summary.csv"
        with open(session_summary, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=session_data[0].keys())
            writer.writeheader()
            writer.writerows(session_data)

        # Run append
        runner._append_to_summary_all()

        # Verify
        summary_all = temp_dir / "summary_all.csv"
        with open(summary_all, 'r') as f:
            data = list(csv.DictReader(f))

        assert len(data) == 2, f"Expected 2 rows, got {len(data)}"
        assert data[0]["experiment_id"] == "test_001"
        assert data[1]["experiment_id"] == "test_002"
        print("✓ TEST 1 PASSED: Created new file with 2 rows")

        # Test 2: Append preserves existing data
        print("\n" + "-" * 80)
        print("TEST 2: Append preserves existing data")
        print("-" * 80)

        # Record current row count
        initial_count = len(data)

        # Create new session
        runner.session.session_dir = temp_dir / "run_test_002"
        runner.session.session_dir.mkdir(parents=True, exist_ok=True)

        new_data = [
            {
                "experiment_id": "test_003",
                "timestamp": "2025-12-02T11:00:00",
                "repository": "examples",
                "model": "siamese",
                "training_success": "True",
                "duration_seconds": "200.0"
            }
        ]

        session_summary = runner.session.session_dir / "summary.csv"
        with open(session_summary, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_data[0].keys())
            writer.writeheader()
            writer.writerows(new_data)

        # Run append
        runner._append_to_summary_all()

        # Verify
        with open(summary_all, 'r') as f:
            data = list(csv.DictReader(f))

        assert len(data) == 3, f"Expected 3 rows, got {len(data)}"
        assert data[0]["experiment_id"] == "test_001", "First row should be preserved"
        assert data[1]["experiment_id"] == "test_002", "Second row should be preserved"
        assert data[2]["experiment_id"] == "test_003", "Third row should be new"
        print(f"✓ TEST 2 PASSED: Preserved {initial_count} rows, added 1 new row, total {len(data)} rows")

        # Test 3: Skip when append_to_summary is False
        print("\n" + "-" * 80)
        print("TEST 3: Skip when append_to_summary=False")
        print("-" * 80)

        runner.append_to_summary = False
        runner.session.session_dir = temp_dir / "run_test_003"
        runner.session.session_dir.mkdir(parents=True, exist_ok=True)

        # Create session data
        session_summary = runner.session.session_dir / "summary.csv"
        with open(session_summary, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_data[0].keys())
            writer.writeheader()
            writer.writerows(new_data)

        # Record current count
        with open(summary_all, 'r') as f:
            before_count = len(list(csv.DictReader(f)))

        # Run append (should skip)
        runner._append_to_summary_all()

        # Verify no change
        with open(summary_all, 'r') as f:
            after_count = len(list(csv.DictReader(f)))

        assert after_count == before_count, "Row count should not change when append_to_summary=False"
        print(f"✓ TEST 3 PASSED: Correctly skipped append, count unchanged ({after_count} rows)")

        # Test 4: Multiple appends accumulate
        print("\n" + "-" * 80)
        print("TEST 4: Multiple appends accumulate correctly")
        print("-" * 80)

        # Delete old file to start fresh
        summary_all.unlink()
        runner.append_to_summary = True

        # Append 3 batches
        for i in range(3):
            runner.session.session_dir = temp_dir / f"run_batch_{i}"
            runner.session.session_dir.mkdir(parents=True, exist_ok=True)

            batch_data = [
                {
                    "experiment_id": f"batch{i}_exp",
                    "timestamp": f"2025-12-02T{12+i}:00:00",
                    "repository": "examples",
                    "model": "mnist",
                    "training_success": "True",
                    "duration_seconds": "100.0"
                }
            ]

            session_summary = runner.session.session_dir / "summary.csv"
            with open(session_summary, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=batch_data[0].keys())
                writer.writeheader()
                writer.writerows(batch_data)

            runner._append_to_summary_all()

        # Verify
        with open(summary_all, 'r') as f:
            data = list(csv.DictReader(f))

        assert len(data) == 3, f"Expected 3 rows from 3 batches, got {len(data)}"
        for i in range(3):
            assert data[i]["experiment_id"] == f"batch{i}_exp"
        print(f"✓ TEST 4 PASSED: Correctly accumulated 3 batches, total {len(data)} rows")

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ Appends to new file correctly")
        print("  ✓ Preserves existing data on append")
        print("  ✓ Respects append_to_summary flag")
        print("  ✓ Multiple appends accumulate correctly")
        print("\nThe fix successfully prevents data loss!")

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    success = test_append_functionality()
    sys.exit(0 if success else 1)
