#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据属性映射完整性

功能:
- 比较 experiment.json 和 raw_data.csv 的属性映射
- 检查是否有属性缺失
- 生成详细的属性对照表
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

def flatten_json_keys(data, prefix=''):
    """递归展开JSON键为扁平结构"""
    keys = set()

    for key, value in data.items():
        full_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            # 递归展开嵌套字典
            nested_keys = flatten_json_keys(value, f"{full_key}_")
            keys.update(nested_keys)
        else:
            keys.add(full_key)

    return keys

def main():
    base_dir = Path(__file__).parent.parent
    raw_data_csv = base_dir / "results" / "raw_data.csv"

    # 查找最新的运行目录
    results_dir = base_dir / "results"
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                     key=lambda x: x.stat().st_mtime, reverse=True)

    if not run_dirs:
        print("❌ 未找到运行目录")
        return

    latest_run_dir = run_dirs[0]

    print("=" * 80)
    print("🔍 检查数据属性映射完整性")
    print("=" * 80)
    print(f"\n最新运行目录: {latest_run_dir.name}")

    # 1. 读取 experiment.json 的属性
    print("\n[1/3] 分析 experiment.json 的属性结构...")

    # 找一个非并行实验的JSON文件
    sample_json = None
    for exp_dir in latest_run_dir.iterdir():
        if exp_dir.is_dir():
            exp_json = exp_dir / "experiment.json"
            if exp_json.exists():
                sample_json = exp_json
                break

    if not sample_json:
        print("❌ 未找到 experiment.json 文件")
        return

    with open(sample_json, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    # 展开JSON的所有键
    json_keys = flatten_json_keys(sample_data)
    print(f"   experiment.json 中的属性数: {len(json_keys)}")

    # 2. 读取 raw_data.csv 的列
    print("\n[2/3] 分析 raw_data.csv 的列结构...")

    with open(raw_data_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        csv_columns = set(reader.fieldnames) if reader.fieldnames else set()

    # 移除前缀为 fg_ 和 bg_ 的列（这些是并行模式专用的）
    non_parallel_csv_columns = {col for col in csv_columns
                                if not col.startswith('fg_') and not col.startswith('bg_')}

    print(f"   raw_data.csv 中的列数: {len(csv_columns)}")
    print(f"   非并行模式相关列数: {len(non_parallel_csv_columns)}")

    # 3. 比较属性映射
    print("\n[3/3] 比较属性映射...")

    # 创建映射关系
    # experiment.json -> raw_data.csv 的映射规则
    json_to_csv_mapping = {
        # 基本信息
        'experiment_id': 'experiment_id',
        'timestamp': 'timestamp',
        'repository': 'repository',
        'model': 'model',
        'training_success': 'training_success',
        'duration_seconds': 'duration_seconds',
        'retries': 'retries',
        'error_message': 'error_message',

        # 超参数 (hyperparameters_*)
        'hyperparameters_alpha': 'hyperparam_alpha',
        'hyperparameters_batch_size': 'hyperparam_batch_size',
        'hyperparameters_dropout': 'hyperparam_dropout',
        'hyperparameters_epochs': 'hyperparam_epochs',
        'hyperparameters_kfold': 'hyperparam_kfold',
        'hyperparameters_learning_rate': 'hyperparam_learning_rate',
        'hyperparameters_max_iter': 'hyperparam_max_iter',
        'hyperparameters_seed': 'hyperparam_seed',
        'hyperparameters_weight_decay': 'hyperparam_weight_decay',

        # 能耗指标 (energy_metrics_*)
        'energy_metrics_cpu_energy_pkg_joules': 'energy_cpu_pkg_joules',
        'energy_metrics_cpu_energy_ram_joules': 'energy_cpu_ram_joules',
        'energy_metrics_cpu_energy_total_joules': 'energy_cpu_total_joules',
        'energy_metrics_gpu_power_avg_watts': 'energy_gpu_avg_watts',
        'energy_metrics_gpu_power_max_watts': 'energy_gpu_max_watts',
        'energy_metrics_gpu_power_min_watts': 'energy_gpu_min_watts',
        'energy_metrics_gpu_energy_total_joules': 'energy_gpu_total_joules',
        'energy_metrics_gpu_temp_avg_celsius': 'energy_gpu_temp_avg_celsius',
        'energy_metrics_gpu_temp_max_celsius': 'energy_gpu_temp_max_celsius',
        'energy_metrics_gpu_util_avg_percent': 'energy_gpu_util_avg_percent',
        'energy_metrics_gpu_util_max_percent': 'energy_gpu_util_max_percent',

        # 性能指标 (performance_metrics_*)
        'performance_metrics_accuracy': 'perf_accuracy',
        'performance_metrics_best_val_accuracy': 'perf_best_val_accuracy',
        'performance_metrics_map': 'perf_map',
        'performance_metrics_precision': 'perf_precision',
        'performance_metrics_rank1': 'perf_rank1',
        'performance_metrics_rank5': 'perf_rank5',
        'performance_metrics_recall': 'perf_recall',
        'performance_metrics_test_accuracy': 'perf_test_accuracy',
        'performance_metrics_test_loss': 'perf_test_loss',
        'performance_metrics_eval_loss': 'perf_eval_loss',
        'performance_metrics_final_training_loss': 'perf_final_training_loss',
        'performance_metrics_eval_samples_per_second': 'perf_eval_samples_per_second',
        'performance_metrics_top1_accuracy': 'perf_top1_accuracy',
        'performance_metrics_top5_accuracy': 'perf_top5_accuracy',
        'performance_metrics_top10_accuracy': 'perf_top10_accuracy',
        'performance_metrics_top20_accuracy': 'perf_top20_accuracy',
        'performance_metrics_f1': 'perf_f1',
    }

    print("\n" + "=" * 80)
    print("📊 属性映射检查")
    print("=" * 80)

    # 检查每个JSON属性是否有对应的CSV列
    missing_in_csv = []
    mapped_correctly = []

    for json_key in sorted(json_keys):
        if json_key in json_to_csv_mapping:
            csv_col = json_to_csv_mapping[json_key]
            if csv_col in csv_columns:
                mapped_correctly.append((json_key, csv_col))
            else:
                missing_in_csv.append((json_key, csv_col, '映射列不存在'))
        else:
            missing_in_csv.append((json_key, None, '无映射规则'))

    print(f"\n✅ 正确映射的属性: {len(mapped_correctly)}/{len(json_keys)}")
    print(f"❌ 缺失或未映射的属性: {len(missing_in_csv)}/{len(json_keys)}")

    if missing_in_csv:
        print("\n" + "=" * 80)
        print("⚠️  以下 experiment.json 属性缺失或未正确映射:")
        print("=" * 80)

        for json_key, csv_col, reason in missing_in_csv:
            if csv_col:
                print(f"  {json_key:<50} -> {csv_col:<40} [{reason}]")
            else:
                print(f"  {json_key:<50} -> {reason}")

    # 检查CSV中有哪些列不来自experiment.json
    print("\n" + "=" * 80)
    print("📋 raw_data.csv 中的额外列 (不直接来自 experiment.json)")
    print("=" * 80)

    extra_columns = []
    reverse_mapping = {v: k for k, v in json_to_csv_mapping.items()}

    for col in sorted(non_parallel_csv_columns):
        if col not in reverse_mapping.values():
            extra_columns.append(col)

    if extra_columns:
        print(f"\n找到 {len(extra_columns)} 个额外列:")
        for col in extra_columns:
            print(f"  - {col}")
    else:
        print("\n没有额外列")

    # 显示详细的映射表
    print("\n" + "=" * 80)
    print("📖 完整属性映射表")
    print("=" * 80)

    print(f"\n{'experiment.json':<60} {'raw_data.csv':<40} {'状态':<10}")
    print("-" * 115)

    for json_key in sorted(json_keys):
        if json_key in json_to_csv_mapping:
            csv_col = json_to_csv_mapping[json_key]
            status = "✅" if csv_col in csv_columns else "❌"
            print(f"{json_key:<60} {csv_col:<40} {status:<10}")
        else:
            print(f"{json_key:<60} {'[无映射]':<40} {'⚠️':<10}")

    print("\n✅ 检查完成!")

if __name__ == "__main__":
    main()
