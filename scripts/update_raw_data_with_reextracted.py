#!/usr/bin/env python3
"""
Update raw_data.csv with Re-extracted Performance Metrics

This script updates raw_data.csv by re-extracting performance metrics from
terminal_output.txt files using the updated regex patterns from models_config.json.

执行方式:
    python3 scripts/update_raw_data_with_reextracted.py

功能:
1. 读取当前raw_data.csv
2. 对每一行，查找对应的terminal_output.txt文件
3. 使用更新后的正则表达式重新提取性能指标
4. 更新raw_data.csv中的perf_*或fg_perf_*列
5. 备份原文件，生成统计报告

输出:
- raw_data.csv.backup_YYYYMMDD_HHMMSS - 原文件备份
- raw_data.csv - 更新后的文件
- 统计报告（控制台输出）
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from collections import defaultdict


def load_models_config(config_path: Path) -> Dict:
    """Load models configuration with log patterns"""
    with open(config_path, 'r') as f:
        return json.load(f)


def extract_performance_from_log_file(
    log_file_path: Path,
    log_patterns: Dict[str, str]
) -> Dict[str, float]:
    """Extract performance metrics from log file (terminal_output.txt or training.log) using regex patterns"""
    if not log_file_path.exists():
        return {}

    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    metrics = {}
    for metric_name, pattern in log_patterns.items():
        try:
            match = re.search(pattern, content)
            if match:
                value = float(match.group(1))
                metrics[metric_name] = value
        except (ValueError, IndexError):
            continue

    return metrics


def find_experiment_dir(results_dir: Path, experiment_id: str) -> Optional[Path]:
    """Find experiment directory by experiment_id"""
    for run_dir in results_dir.glob("run_*"):
        if not run_dir.is_dir():
            continue

        for exp_dir in run_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            if exp_dir.name == experiment_id:
                return exp_dir

    return None


def get_perf_columns(fieldnames: list, is_parallel: bool) -> list:
    """Get list of performance metric column names"""
    prefix = "fg_perf_" if is_parallel else "perf_"
    return [f for f in fieldnames if f.startswith(prefix)]


def update_row_with_metrics(
    row: Dict,
    new_metrics: Dict[str, float],
    is_parallel: bool
) -> Dict:
    """Update row with newly extracted metrics"""
    prefix = "fg_perf_" if is_parallel else "perf_"

    for metric_name, value in new_metrics.items():
        col_name = f"{prefix}{metric_name}"
        row[col_name] = str(value)

    return row


def main():
    """Main execution function"""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    config_path = project_root / "mutation" / "models_config.json"
    raw_data_path = results_dir / "raw_data.csv"

    print("=" * 80)
    print("Update raw_data.csv with Re-extracted Performance Metrics")
    print("=" * 80)
    print()

    # Check if raw_data.csv exists
    if not raw_data_path.exists():
        print(f"❌ Error: {raw_data_path} not found")
        return 1

    # Load models config
    print("Loading models configuration...")
    models_config = load_models_config(config_path)
    print(f"✓ Loaded config from: {config_path}")
    print()

    # Backup raw_data.csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = raw_data_path.with_suffix(f'.csv.backup_{timestamp}')
    print(f"Creating backup: {backup_path.name}")

    with open(raw_data_path, 'r') as f_in:
        with open(backup_path, 'w') as f_out:
            f_out.write(f_in.read())

    print(f"✓ Backup created")
    print()

    # Load raw_data.csv
    print(f"Loading {raw_data_path.name}...")
    with open(raw_data_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"✓ Loaded {len(rows)} rows")
    print()

    # Process each row
    print("=" * 80)
    print("Re-extracting performance metrics...")
    print("=" * 80)
    print()

    stats = defaultdict(lambda: {
        'total': 0,
        'had_perf': 0,
        'found_terminal': 0,
        'extracted': 0,
        'still_missing': 0
    })

    updated_rows = []
    updated_count = 0

    for i, row in enumerate(rows, 1):
        experiment_id = row.get('experiment_id', '')
        repo = row.get('repository', '')
        model = row.get('model', '')
        mode = row.get('mode', '')

        # Determine if parallel mode
        is_parallel = (mode == 'parallel')

        # For parallel mode, use fg_repository if repository is empty
        if is_parallel and not repo:
            repo = row.get('fg_repository', '')
            model = row.get('fg_model', '')

        key = f"{repo}/{model}"
        stats[key]['total'] += 1

        # Check if already has performance metrics
        perf_cols = get_perf_columns(fieldnames, is_parallel)
        had_perf = any(row.get(col) and row.get(col).strip() for col in perf_cols)

        if had_perf:
            stats[key]['had_perf'] += 1

        # Find experiment directory
        exp_dir = find_experiment_dir(results_dir, experiment_id)
        if not exp_dir:
            updated_rows.append(row)
            continue

        # Check for log files (prefer terminal_output.txt, fallback to training.log)
        terminal_output = exp_dir / "terminal_output.txt"
        training_log = exp_dir / "training.log"

        log_file = None
        log_type = None
        if terminal_output.exists():
            log_file = terminal_output
            log_type = "terminal_output"
        elif training_log.exists():
            log_file = training_log
            log_type = "training_log"
        else:
            updated_rows.append(row)
            continue

        stats[key]['found_terminal'] += 1

        # Get log patterns for this model
        if repo not in models_config['models']:
            updated_rows.append(row)
            continue

        repo_config = models_config['models'][repo]
        perf_config = repo_config.get('performance_metrics', {})
        log_patterns = perf_config.get('log_patterns', {})

        if not log_patterns:
            updated_rows.append(row)
            continue

        # Extract metrics from log file
        new_metrics = extract_performance_from_log_file(log_file, log_patterns)

        if new_metrics:
            # Update row with new metrics
            row = update_row_with_metrics(row, new_metrics, is_parallel)
            updated_count += 1
            stats[key]['extracted'] += 1

            if i <= 10:  # Show first 10 updates
                print(f"✓ Updated row {i}: {experiment_id}")
                print(f"  Model: {key}")
                print(f"  Extracted: {list(new_metrics.keys())}")
        else:
            stats[key]['still_missing'] += 1

        updated_rows.append(row)

    print()
    print(f"Processed {len(rows)} rows, updated {updated_count} rows")
    print()

    # Write updated raw_data.csv
    print(f"Writing updated {raw_data_path.name}...")
    with open(raw_data_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"✓ Updated file saved")
    print()

    # Print statistics
    print("=" * 80)
    print("Update Summary")
    print("=" * 80)
    print(f"Total rows: {len(rows)}")
    print(f"Updated rows: {updated_count}")
    print()

    print("Per-model statistics:")
    print()
    for model_key in sorted(stats.keys()):
        s = stats[model_key]
        if s['total'] == 0:
            continue

        print(f"{model_key}:")
        print(f"  Total rows: {s['total']}")
        print(f"  Had performance data: {s['had_perf']}")
        print(f"  Found terminal_output.txt: {s['found_terminal']}")
        print(f"  Successfully extracted: {s['extracted']}")
        print(f"  Still missing: {s['still_missing']}")

        if s['total'] > 0:
            recovery_rate = s['extracted'] / s['total'] * 100
            final_coverage = (s['had_perf'] + s['extracted']) / s['total'] * 100
            print(f"  Recovery rate: {recovery_rate:.1f}%")
            print(f"  Final coverage: {final_coverage:.1f}%")
        print()

    print("=" * 80)
    print("Backup file:", backup_path.name)
    print("Updated file:", raw_data_path.name)
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Verify the updated raw_data.csv")
    print("  2. Run validation: python3 scripts/validate_raw_data.py")
    print("  3. Analyze updated statistics")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
