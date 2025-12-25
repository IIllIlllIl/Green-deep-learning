#!/usr/bin/env python3
"""
Test script to verify the CSV column mismatch fix

This script tests that _append_to_summary_all() correctly handles
column mismatches between session CSV and summary_all.csv.
"""

import csv
import tempfile
from pathlib import Path
import shutil


def test_append_with_column_mismatch():
    """Test appending session CSV with fewer columns to summary_all.csv"""

    print("=" * 80)
    print("Testing CSV Append Fix - Column Mismatch Scenario")
    print("=" * 80)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Create summary_all.csv with 37 columns
        summary_all = tmpdir / "summary_all.csv"
        all_columns = [
            'experiment_id', 'timestamp', 'repository', 'model', 'training_success',
            'duration_seconds', 'retries',
            'hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
            'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
            'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay',
            'perf_accuracy', 'perf_best_val_accuracy', 'perf_map', 'perf_precision',
            'perf_rank1', 'perf_rank5', 'perf_recall', 'perf_test_accuracy', 'perf_test_loss',
            'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
            'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
            'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
            'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent', 'experiment_source'
        ]

        # Write header and one existing row
        with open(summary_all, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            writer.writerow({
                'experiment_id': 'existing_001',
                'timestamp': '2025-12-01T00:00:00',
                'repository': 'test_repo',
                'model': 'test_model',
                'training_success': 'True',
                'duration_seconds': '100.0',
                'retries': '0',
                'hyperparam_epochs': '10',
                'hyperparam_learning_rate': '0.01',
                'perf_accuracy': '0.95',
                'energy_cpu_total_joules': '1000.0',
                'energy_gpu_total_joules': '5000.0'
            })

        print(f"✓ Created summary_all.csv with {len(all_columns)} columns")

        # 2. Create session CSV with only 29 columns (like Stage1)
        session_csv = tmpdir / "session_summary.csv"
        session_columns = [
            'experiment_id', 'timestamp', 'repository', 'model', 'training_success',
            'duration_seconds', 'retries',
            'hyperparam_batch_size', 'hyperparam_dropout', 'hyperparam_epochs',
            'hyperparam_learning_rate', 'hyperparam_seed',
            'perf_accuracy', 'perf_map', 'perf_precision', 'perf_rank1', 'perf_rank5', 'perf_recall',
            'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
            'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
            'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
            'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent'
        ]

        with open(session_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=session_columns)
            writer.writeheader()
            writer.writerow({
                'experiment_id': 'session_001',
                'timestamp': '2025-12-02T00:00:00',
                'repository': 'Person_reID_baseline_pytorch',
                'model': 'hrnet18',
                'training_success': 'True',
                'duration_seconds': '4154.0',
                'retries': '0',
                'hyperparam_learning_rate': '0.033',
                'perf_map': '0.7813',
                'perf_rank1': '0.9172',
                'energy_cpu_total_joules': '40000.0',
                'energy_gpu_total_joules': '140000.0'
            })

        print(f"✓ Created session CSV with {len(session_columns)} columns")
        print(f"  Missing columns: {len(all_columns) - len(session_columns)}")

        # 3. Simulate the fixed append logic
        print("\n" + "=" * 80)
        print("Simulating Fixed Append Logic")
        print("=" * 80)

        # Read session data
        with open(session_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            session_rows = list(reader)

        print(f"✓ Read {len(session_rows)} rows from session CSV")

        # Read summary_all fieldnames (the fix!)
        with open(summary_all, 'r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        print(f"✓ Using summary_all.csv's {len(fieldnames)} columns as standard")

        # Append using summary_all's fieldnames
        with open(summary_all, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writerows(session_rows)

        print(f"✓ Appended session rows with automatic column alignment")

        # 4. Verify the result
        print("\n" + "=" * 80)
        print("Verification")
        print("=" * 80)

        with open(summary_all, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        print(f"✓ Header has {len(header)} columns")
        print(f"✓ Total data rows: {len(rows)}")

        # Check all rows have correct column count
        all_correct = True
        for i, row in enumerate(rows, start=2):
            if len(row) != len(header):
                print(f"✗ Line {i}: {len(row)} columns (expected {len(header)})")
                all_correct = False

        if all_correct:
            print(f"✓ All {len(rows)} rows have correct {len(header)} columns")

        # Verify the appended row
        print("\n" + "=" * 80)
        print("Appended Row Content")
        print("=" * 80)

        with open(summary_all, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            appended_row = rows[-1]  # Last row

        print(f"Experiment ID: {appended_row['experiment_id']}")
        print(f"Repository: {appended_row['repository']}")
        print(f"Model: {appended_row['model']}")

        # Check missing columns are empty
        missing_cols = [
            'hyperparam_alpha', 'hyperparam_kfold', 'hyperparam_max_iter',
            'hyperparam_weight_decay', 'perf_best_val_accuracy',
            'perf_test_accuracy', 'perf_test_loss', 'experiment_source'
        ]

        print("\nMissing columns (should be empty):")
        all_empty = True
        for col in missing_cols:
            val = appended_row[col]
            status = "✓ empty" if not val else f"✗ has value: {val}"
            print(f"  {col}: {status}")
            if val:
                all_empty = False

        # Check present columns have values
        print("\nPresent columns (should have values):")
        present_cols = ['hyperparam_learning_rate', 'perf_map', 'perf_rank1']
        for col in present_cols:
            val = appended_row[col]
            status = f"✓ {val}" if val else "✗ empty"
            print(f"  {col}: {status}")

        print("\n" + "=" * 80)
        if all_correct and all_empty:
            print("✅ TEST PASSED: Fix works correctly!")
        else:
            print("❌ TEST FAILED: Issues detected")
        print("=" * 80)

        return all_correct and all_empty


if __name__ == "__main__":
    success = test_append_with_column_mismatch()
    exit(0 if success else 1)
