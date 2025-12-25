#!/usr/bin/env python3
"""
验证80列格式CSV文件的正确性

功能：
1. 验证列数为80
2. 验证表头与标准一致
3. 验证数据完整性
4. 统计数据质量

作者: Green
日期: 2025-12-12
"""

import csv
import sys


# 80列标准表头
HEADER_80COL = [
    # 基础信息 (7列)
    'experiment_id', 'timestamp', 'repository', 'model',
    'training_success', 'duration_seconds', 'retries',

    # 超参数 (9列)
    'hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
    'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
    'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay',

    # 性能指标 (9列)
    'perf_accuracy', 'perf_best_val_accuracy', 'perf_map',
    'perf_precision', 'perf_rank1', 'perf_rank5',
    'perf_recall', 'perf_test_accuracy', 'perf_test_loss',

    # 能耗指标 (11列)
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
    'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
    'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent',

    # 元数据 (5列)
    'experiment_source', 'num_mutated_params', 'mutated_param', 'mode', 'error_message',

    # 前景实验详细信息 (36列)
    'fg_repository', 'fg_model', 'fg_duration_seconds', 'fg_training_success',
    'fg_retries', 'fg_error_message',
    'fg_hyperparam_alpha', 'fg_hyperparam_batch_size', 'fg_hyperparam_dropout',
    'fg_hyperparam_epochs', 'fg_hyperparam_kfold', 'fg_hyperparam_learning_rate',
    'fg_hyperparam_max_iter', 'fg_hyperparam_seed', 'fg_hyperparam_weight_decay',
    'fg_perf_accuracy', 'fg_perf_best_val_accuracy', 'fg_perf_map',
    'fg_perf_precision', 'fg_perf_rank1', 'fg_perf_rank5',
    'fg_perf_recall', 'fg_perf_test_accuracy', 'fg_perf_test_loss',
    'fg_energy_cpu_pkg_joules', 'fg_energy_cpu_ram_joules', 'fg_energy_cpu_total_joules',
    'fg_energy_gpu_avg_watts', 'fg_energy_gpu_max_watts', 'fg_energy_gpu_min_watts',
    'fg_energy_gpu_total_joules', 'fg_energy_gpu_temp_avg_celsius', 'fg_energy_gpu_temp_max_celsius',
    'fg_energy_gpu_util_avg_percent', 'fg_energy_gpu_util_max_percent',

    # 背景实验信息 (4列)
    'bg_repository', 'bg_model', 'bg_note', 'bg_log_directory'
]


def validate_header(csv_file):
    """验证表头"""
    print(f"\n测试1: 验证表头格式")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames

    print(f"  实际列数: {len(header)}")
    print(f"  预期列数: {len(HEADER_80COL)}")

    if len(header) != len(HEADER_80COL):
        print(f"  ✗ 列数不匹配")
        return False

    # 检查每一列
    mismatches = []
    for i, (actual, expected) in enumerate(zip(header, HEADER_80COL), 1):
        if actual != expected:
            mismatches.append((i, actual, expected))

    if mismatches:
        print(f"  ✗ 发现{len(mismatches)}个列名不匹配:")
        for i, actual, expected in mismatches[:10]:  # 只显示前10个
            print(f"    列{i}: '{actual}' != '{expected}'")
        return False

    print(f"  ✓ 表头完全匹配")
    return True


def validate_data_integrity(csv_file):
    """验证数据完整性"""
    print(f"\n测试2: 验证数据完整性")

    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"  总行数: {len(rows)}")

    # 统计
    stats = {
        'training_success': 0,
        'has_cpu_energy': 0,
        'has_gpu_energy': 0,
        'has_performance': 0,
        'parallel_mode': 0,
        'has_fg_data': 0,
        'has_bg_data': 0,
    }

    for row in rows:
        if row['training_success'] == 'True':
            stats['training_success'] += 1

        if row['energy_cpu_total_joules']:
            stats['has_cpu_energy'] += 1

        if row['energy_gpu_total_joules']:
            stats['has_gpu_energy'] += 1

        perf_cols = ['perf_accuracy', 'perf_test_accuracy', 'perf_precision',
                     'perf_recall', 'perf_rank1', 'perf_rank5', 'perf_map']
        if any(row[col] for col in perf_cols):
            stats['has_performance'] += 1

        if row['mode'] == 'parallel':
            stats['parallel_mode'] += 1

            if row['fg_repository']:
                stats['has_fg_data'] += 1

            if row['bg_repository']:
                stats['has_bg_data'] += 1

    print(f"  训练成功: {stats['training_success']}/{len(rows)} ({stats['training_success']/len(rows)*100:.1f}%)")
    print(f"  有CPU能耗: {stats['has_cpu_energy']}/{len(rows)} ({stats['has_cpu_energy']/len(rows)*100:.1f}%)")
    print(f"  有GPU能耗: {stats['has_gpu_energy']}/{len(rows)*100:.1f}%)")
    print(f"  有性能数据: {stats['has_performance']}/{len(rows)} ({stats['has_performance']/len(rows)*100:.1f}%)")
    print(f"  并行模式: {stats['parallel_mode']}/{len(rows)} ({stats['parallel_mode']/len(rows)*100:.1f}%)")

    if stats['parallel_mode'] > 0:
        print(f"  并行实验中有fg数据: {stats['has_fg_data']}/{stats['parallel_mode']}")
        print(f"  并行实验中有bg数据: {stats['has_bg_data']}/{stats['parallel_mode']}")

    print(f"  ✓ 数据完整性检查通过")
    return True


def validate_required_fields(csv_file):
    """验证必填字段"""
    print(f"\n测试3: 验证必填字段")

    required_fields = ['experiment_id', 'timestamp', 'repository', 'model', 'training_success']

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    missing_count = {field: 0 for field in required_fields}

    for row in rows:
        for field in required_fields:
            if not row.get(field, '').strip():
                missing_count[field] += 1

    all_good = True
    for field, count in missing_count.items():
        if count > 0:
            print(f"  ✗ {field}: {count}行缺失")
            all_good = False

    if all_good:
        print(f"  ✓ 所有必填字段完整")

    return all_good


def compare_with_summary_new(old_80col_file):
    """与summary_new.csv格式对比"""
    print(f"\n测试4: 与summary_new.csv格式对比")

    try:
        with open('results/summary_new.csv', 'r') as f:
            reader = csv.DictReader(f)
            new_header = reader.fieldnames

        with open(old_80col_file, 'r') as f:
            reader = csv.DictReader(f)
            old_header = reader.fieldnames

        if new_header == old_header:
            print(f"  ✓ 与summary_new.csv表头完全一致")
            return True
        else:
            print(f"  ✗ 与summary_new.csv表头不一致")
            return False

    except FileNotFoundError:
        print(f"  ⚠ summary_new.csv不存在，跳过对比")
        return True


def main():
    if len(sys.argv) < 2:
        csv_file = 'results/summary_old_80col.csv'
    else:
        csv_file = sys.argv[1]

    print("=" * 80)
    print(f"验证80列格式CSV: {csv_file}")
    print("=" * 80)

    tests = [
        validate_header(csv_file),
        validate_data_integrity(csv_file),
        validate_required_fields(csv_file),
        compare_with_summary_new(csv_file),
    ]

    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)
    print(f"  通过: {sum(tests)}/{len(tests)}")

    if all(tests):
        print("\n✓ 所有测试通过！80列格式正确。")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查上述错误。")
        return 1


if __name__ == '__main__':
    exit(main())
