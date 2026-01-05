#!/usr/bin/env python3
"""
分析summary_all.csv (37列) vs summary_old.csv (93列)
确定缺失的列以及能否从JSON数据中提取

作者: Claude Code
日期: 2025-12-12
"""

import csv
from pathlib import Path

# 93列标准格式
STANDARD_93_COLUMNS = [
    'experiment_id', 'timestamp', 'repository', 'model', 'training_success',
    'duration_seconds', 'retries',
    # 顶层超参数 (9)
    'hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
    'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
    'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay',
    # 顶层性能指标 (9)
    'perf_accuracy', 'perf_best_val_accuracy', 'perf_map', 'perf_precision',
    'perf_rank1', 'perf_rank5', 'perf_recall', 'perf_test_accuracy', 'perf_test_loss',
    # 顶层能耗 (11)
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
    'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius',
    'energy_gpu_temp_max_celsius', 'energy_gpu_util_avg_percent',
    'energy_gpu_util_max_percent',
    # 实验元数据 (5)
    'experiment_source', 'num_mutated_params', 'mutated_param', 'mode', 'error_message',
    # 前景字段 (42)
    'fg_repository', 'fg_model', 'fg_duration_seconds', 'fg_training_success', 'fg_retries',
    'fg_error_message',
    'fg_hyperparam_alpha', 'fg_hyperparam_batch_size', 'fg_hyperparam_dropout',
    'fg_hyperparam_epochs', 'fg_hyperparam_kfold', 'fg_hyperparam_learning_rate',
    'fg_hyperparam_max_iter', 'fg_hyperparam_seed', 'fg_hyperparam_weight_decay',
    'fg_perf_accuracy', 'fg_perf_best_val_accuracy', 'fg_perf_map', 'fg_perf_precision',
    'fg_perf_rank1', 'fg_perf_rank5', 'fg_perf_recall', 'fg_perf_test_accuracy',
    'fg_perf_test_loss',
    'fg_energy_cpu_pkg_joules', 'fg_energy_cpu_ram_joules', 'fg_energy_cpu_total_joules',
    'fg_energy_gpu_avg_watts', 'fg_energy_gpu_max_watts', 'fg_energy_gpu_min_watts',
    'fg_energy_gpu_total_joules', 'fg_energy_gpu_temp_avg_celsius',
    'fg_energy_gpu_temp_max_celsius', 'fg_energy_gpu_util_avg_percent',
    'fg_energy_gpu_util_max_percent',
    # 背景字段 (10)
    'bg_repository', 'bg_model', 'bg_note', 'bg_log_directory',
    'bg_hyperparam_batch_size', 'bg_hyperparam_dropout', 'bg_hyperparam_epochs',
    'bg_hyperparam_learning_rate', 'bg_hyperparam_seed', 'bg_hyperparam_weight_decay',
    'bg_energy_cpu_pkg_joules', 'bg_energy_cpu_ram_joules', 'bg_energy_cpu_total_joules',
    'bg_energy_gpu_avg_watts', 'bg_energy_gpu_max_watts', 'bg_energy_gpu_min_watts',
    'bg_energy_gpu_total_joules'
]

def analyze_columns():
    """分析列差异"""
    print("=" * 80)
    print("summary_all.csv (37列) vs 标准93列格式 - 列对比分析")
    print("=" * 80)

    # 读取summary_all.csv的列
    csv_file = Path('results/summary_all.csv')
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        current_columns = reader.fieldnames
        row_count = sum(1 for _ in reader)

    print(f"\n📊 当前状态:")
    print(f"  summary_all.csv: {len(current_columns)} 列, {row_count} 行")
    print(f"  标准格式: {len(STANDARD_93_COLUMNS)} 列")

    # 查找缺失的列
    current_set = set(current_columns)
    standard_set = set(STANDARD_93_COLUMNS)

    missing_columns = standard_set - current_set
    extra_columns = current_set - standard_set

    print(f"\n❌ 缺失的列: {len(missing_columns)} 个")
    if missing_columns:
        # 按类别组织
        categories = {
            'metadata': [],
            'fg_basic': [],
            'fg_hyperparam': [],
            'fg_perf': [],
            'fg_energy': [],
            'bg_basic': [],
            'bg_hyperparam': [],
            'bg_energy': []
        }

        for col in sorted(missing_columns):
            if col in ['num_mutated_params', 'mutated_param', 'mode', 'error_message']:
                categories['metadata'].append(col)
            elif col.startswith('fg_hyperparam_'):
                categories['fg_hyperparam'].append(col)
            elif col.startswith('fg_perf_'):
                categories['fg_perf'].append(col)
            elif col.startswith('fg_energy_'):
                categories['fg_energy'].append(col)
            elif col.startswith('fg_'):
                categories['fg_basic'].append(col)
            elif col.startswith('bg_hyperparam_'):
                categories['bg_hyperparam'].append(col)
            elif col.startswith('bg_energy_'):
                categories['bg_energy'].append(col)
            elif col.startswith('bg_'):
                categories['bg_basic'].append(col)

        for category, cols in categories.items():
            if cols:
                print(f"\n  {category}:")
                for col in cols:
                    print(f"    - {col}")

    if extra_columns:
        print(f"\n⚠️ 多余的列: {len(extra_columns)} 个")
        for col in sorted(extra_columns):
            print(f"    - {col}")

    # 分析数据来源
    print(f"\n\n" + "=" * 80)
    print("📋 缺失列的数据来源分析")
    print("=" * 80)

    print("\n✅ 可以从JSON提取的字段:")
    print("  1. 元数据字段 (4个):")
    print("     - mode: experiment.json中的'mode'字段")
    print("     - error_message: experiment.json中的'error_message'字段")
    print("     - num_mutated_params: 可从experiment_source推断")
    print("     - mutated_param: 可从experiment_source推断")

    print("\n  2. 前景字段 (36个):")
    print("     - fg_repository, fg_model: foreground.repository, foreground.model")
    print("     - fg_duration_seconds: foreground.duration_seconds")
    print("     - fg_training_success, fg_retries: foreground.training_success/retries")
    print("     - fg_error_message: foreground.error_message")
    print("     - fg_hyperparam_*: foreground.hyperparameters.*")
    print("     - fg_perf_*: foreground.performance_metrics.*")
    print("     - fg_energy_*: foreground.energy_metrics.*")

    print("\n  3. 背景字段 (10个):")
    print("     - bg_repository, bg_model: background.repository, background.model")
    print("     - bg_note, bg_log_directory: background.note, background.log_directory")
    print("     - bg_hyperparam_*: background.hyperparameters.*")
    print("     - bg_energy_*: ⚠️ 不存在（设计决定：背景训练不监控能耗）")

    print("\n❌ 不能从JSON提取的字段:")
    print("  - bg_energy_* (7个): 背景训练不监控能耗，JSON中无此数据")

    # 统计分析
    print(f"\n\n" + "=" * 80)
    print("📈 数据可恢复性统计")
    print("=" * 80)

    total_missing = len(missing_columns)
    recoverable = total_missing - 7  # 减去7个bg_energy字段

    print(f"\n  缺失列总数: {total_missing}")
    print(f"  可从JSON恢复: {recoverable} ({recoverable/total_missing*100:.1f}%)")
    print(f"  不可恢复 (bg_energy_*): 7 ({7/total_missing*100:.1f}%)")

    print("\n💡 建议:")
    print("  1. 创建重建脚本,从experiment.json提取缺失字段")
    print("  2. bg_energy_*字段保持为空（符合项目设计）")
    print("  3. 生成新的93列summary_all.csv")
    print("  4. 备份原37列版本")

if __name__ == '__main__':
    analyze_columns()
