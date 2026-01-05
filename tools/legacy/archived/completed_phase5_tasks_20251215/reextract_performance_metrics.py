#!/usr/bin/env python3
"""
Phase 4: Re-extract Performance Metrics from Historical Data

This script re-extracts performance metrics from existing terminal_output.txt files
using the updated log_patterns from models_config.json. This allows us to recover
performance data from the 161 experiments that are missing metrics without re-running
the training.

执行方式:
    python3 scripts/reextract_performance_metrics.py

功能:
1. 扫描所有results/run_*目录找到有terminal_output.txt的实验
2. 从experiment.json读取repo/model信息
3. 使用更新后的models_config.json中的正则表达式重新提取性能指标
4. 更新experiment.json中的performance_metrics字段
5. 生成更新后的CSV数据

输出:
- 更新后的experiment.json文件（备份原文件）
- 性能指标提取统计报告
- 可选：更新raw_data.csv
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_models_config(config_path: Path) -> Dict:
    """Load models configuration with log patterns"""
    with open(config_path, 'r') as f:
        return json.load(f)


def extract_performance_from_terminal_output(
    terminal_output_path: Path,
    log_patterns: Dict[str, str]
) -> Dict[str, float]:
    """Extract performance metrics from terminal_output.txt using regex patterns

    Args:
        terminal_output_path: Path to terminal_output.txt
        log_patterns: Dictionary of metric_name -> regex_pattern

    Returns:
        Dictionary of metric_name -> value
    """
    if not terminal_output_path.exists():
        return {}

    # Read terminal output
    with open(terminal_output_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract metrics using patterns
    metrics = {}
    for metric_name, pattern in log_patterns.items():
        try:
            match = re.search(pattern, content)
            if match:
                value = float(match.group(1))
                metrics[metric_name] = value
        except (ValueError, IndexError) as e:
            # Skip invalid matches
            continue

    return metrics


def find_all_experiments(results_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """Find all experiments with terminal_output.txt

    Returns:
        List of (exp_dir, experiment_json, terminal_output) tuples
    """
    experiments = []

    for run_dir in results_dir.glob("run_*"):
        if not run_dir.is_dir():
            continue

        for exp_dir in run_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            experiment_json = exp_dir / "experiment.json"
            terminal_output = exp_dir / "terminal_output.txt"

            if experiment_json.exists() and terminal_output.exists():
                experiments.append((exp_dir, experiment_json, terminal_output))

    return experiments


def reextract_single_experiment(
    exp_dir: Path,
    experiment_json_path: Path,
    terminal_output_path: Path,
    models_config: Dict,
    dry_run: bool = False
) -> Tuple[bool, Dict, Dict]:
    """Re-extract performance metrics for a single experiment

    Args:
        exp_dir: Experiment directory
        experiment_json_path: Path to experiment.json
        terminal_output_path: Path to terminal_output.txt
        models_config: Models configuration with log patterns
        dry_run: If True, don't update files

    Returns:
        (success, old_metrics, new_metrics)
    """
    # Load experiment data
    with open(experiment_json_path, 'r') as f:
        exp_data = json.load(f)

    # Get repo and model
    repo = exp_data.get('repository') or exp_data.get('foreground', {}).get('repository')
    model = exp_data.get('model') or exp_data.get('foreground', {}).get('model')

    if not repo or not model:
        return False, {}, {}

    # Get log patterns for this model
    if repo not in models_config['models']:
        return False, {}, {}

    repo_config = models_config['models'][repo]
    perf_config = repo_config.get('performance_metrics', {})
    log_patterns = perf_config.get('log_patterns', {})

    if not log_patterns:
        return False, {}, {}

    # Extract performance metrics
    old_metrics = exp_data.get('performance_metrics', {})
    if isinstance(old_metrics, dict):
        old_metrics = old_metrics
    else:
        old_metrics = {}

    new_metrics = extract_performance_from_terminal_output(
        terminal_output_path,
        log_patterns
    )

    # Update experiment.json if new metrics found and different from old
    if new_metrics and new_metrics != old_metrics:
        if not dry_run:
            # Backup original file
            backup_path = experiment_json_path.with_suffix('.json.backup')
            if not backup_path.exists():
                with open(experiment_json_path, 'r') as f_in:
                    with open(backup_path, 'w') as f_out:
                        f_out.write(f_in.read())

            # Update performance_metrics
            exp_data['performance_metrics'] = new_metrics

            # Save updated file
            with open(experiment_json_path, 'w') as f:
                json.dump(exp_data, f, indent=2)

        return True, old_metrics, new_metrics

    return False, old_metrics, new_metrics


def main():
    """Main execution function"""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    config_path = project_root / "mutation" / "models_config.json"

    print("=" * 80)
    print("Phase 4: Historical Performance Metrics Re-extraction")
    print("=" * 80)
    print()

    # Load models config
    print("Loading models configuration...")
    models_config = load_models_config(config_path)
    print(f"✓ Loaded config from: {config_path}")
    print()

    # Find all experiments
    print("Scanning for experiments with terminal_output.txt...")
    experiments = find_all_experiments(results_dir)
    print(f"✓ Found {len(experiments)} experiments with terminal output")
    print()

    # Ask for confirmation
    print("This script will:")
    print("  1. Re-extract performance metrics from terminal_output.txt")
    print("  2. Update experiment.json files (backup will be created)")
    print("  3. Generate statistics report")
    print()
    response = input("Proceed? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return 0
    print()

    # Process all experiments
    print("=" * 80)
    print("Re-extracting performance metrics...")
    print("=" * 80)
    print()

    stats = defaultdict(lambda: {
        'total': 0,
        'had_metrics': 0,
        'extracted': 0,
        'still_missing': 0
    })

    updated_count = 0
    skipped_count = 0

    for exp_dir, exp_json, terminal_output in experiments:
        # Load experiment data to get repo/model
        with open(exp_json, 'r') as f:
            exp_data = json.load(f)

        repo = exp_data.get('repository') or exp_data.get('foreground', {}).get('repository', 'unknown')
        model = exp_data.get('model') or exp_data.get('foreground', {}).get('model', 'unknown')
        key = f"{repo}/{model}"

        stats[key]['total'] += 1

        old_metrics = exp_data.get('performance_metrics', {})
        had_metrics = bool(old_metrics and len(old_metrics) > 0)

        if had_metrics:
            stats[key]['had_metrics'] += 1

        # Re-extract
        success, old_metrics, new_metrics = reextract_single_experiment(
            exp_dir, exp_json, terminal_output, models_config, dry_run=False
        )

        if success:
            updated_count += 1
            stats[key]['extracted'] += 1
            print(f"✓ Updated: {exp_data.get('experiment_id', exp_dir.name)}")
            print(f"  Old metrics: {len(old_metrics)} fields")
            print(f"  New metrics: {len(new_metrics)} fields - {list(new_metrics.keys())}")
        else:
            skipped_count += 1
            if not new_metrics:
                stats[key]['still_missing'] += 1

    print()
    print("=" * 80)
    print("Re-extraction Summary")
    print("=" * 80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Updated: {updated_count}")
    print(f"Skipped: {skipped_count}")
    print()

    print("Per-model statistics:")
    print()
    for model_key in sorted(stats.keys()):
        s = stats[model_key]
        print(f"{model_key}:")
        print(f"  Total experiments: {s['total']}")
        print(f"  Had metrics before: {s['had_metrics']}")
        print(f"  Newly extracted: {s['extracted']}")
        print(f"  Still missing: {s['still_missing']}")
        if s['total'] > 0:
            recovery_rate = s['extracted'] / s['total'] * 100
            print(f"  Recovery rate: {recovery_rate:.1f}%")
        print()

    print("=" * 80)
    print("Next steps:")
    print("  1. Verify the updated experiment.json files")
    print("  2. Regenerate raw_data.csv from updated experiment.json files")
    print("  3. Run validation to check data completeness")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
