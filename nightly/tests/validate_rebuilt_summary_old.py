#!/usr/bin/env python3
"""
验证重建的summary_old.csv数据正确性

功能：
1. 检查CSV行数是否与白名单一致
2. 随机抽样对比CSV数据与experiment.json源数据
3. 检查关键字段的完整性
4. 统计数据质量指标

作者: Green
日期: 2025-12-12
"""

import csv
import json
import random
import os
from pathlib import Path


def load_whitelist():
    """加载白名单"""
    with open('results/old_experiment_whitelist.json', 'r') as f:
        return json.load(f)


def load_csv():
    """加载CSV文件"""
    rows = []
    with open('results/summary_old.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_experiment_json(experiment_source, experiment_id):
    """查找experiment.json文件"""
    # 从experiment_id中提取信息
    parts = experiment_id.split('_')

    # 查找所有可能的路径
    results_dir = Path('results')

    for run_dir in results_dir.glob('run_*'):
        for exp_dir in run_dir.glob(f'{experiment_source}__*'):
            json_file = exp_dir / 'experiment.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get('experiment_id') == experiment_id:
                        return json_file, data

    return None, None


def compare_row_with_json(row, json_data):
    """比较CSV行与JSON数据"""
    errors = []

    # 基础字段检查
    if row['experiment_id'] != json_data.get('experiment_id'):
        errors.append(f"experiment_id不匹配: {row['experiment_id']} vs {json_data.get('experiment_id')}")

    # 检查是否是并行实验
    is_parallel = json_data.get('mode') == 'parallel'

    if is_parallel:
        fg = json_data.get('foreground', {})
        if row['repository'] != fg.get('repository'):
            errors.append(f"repository不匹配: {row['repository']} vs {fg.get('repository')}")
        if row['model'] != fg.get('model'):
            errors.append(f"model不匹配: {row['model']} vs {fg.get('model')}")
    else:
        if row['repository'] != json_data.get('repository'):
            errors.append(f"repository不匹配: {row['repository']} vs {json_data.get('repository')}")
        if row['model'] != json_data.get('model'):
            errors.append(f"model不匹配: {row['model']} vs {json_data.get('model')}")

    # 检查training_success
    json_success = str(json_data.get('training_success', '')).strip()
    csv_success = row['training_success'].strip()
    if csv_success != json_success:
        errors.append(f"training_success不匹配: {csv_success} vs {json_success}")

    return errors


def validate_data_quality(rows):
    """验证数据质量"""
    stats = {
        'total_rows': len(rows),
        'training_success_count': 0,
        'has_cpu_energy': 0,
        'has_gpu_energy': 0,
        'has_performance': 0,
        'missing_fields': {}
    }

    for row in rows:
        # 统计训练成功率
        if row.get('training_success') == 'True':
            stats['training_success_count'] += 1

        # 统计能耗数据
        if row.get('energy_cpu_total_joules'):
            stats['has_cpu_energy'] += 1
        if row.get('energy_gpu_total_joules'):
            stats['has_gpu_energy'] += 1

        # 统计性能数据（任意一个性能指标存在）
        perf_cols = ['perf_accuracy', 'perf_test_accuracy', 'perf_precision',
                     'perf_recall', 'perf_rank1', 'perf_rank5', 'perf_map']
        if any(row.get(col) for col in perf_cols):
            stats['has_performance'] += 1

        # 统计缺失字段
        for key, value in row.items():
            if not value or value.strip() == '':
                if key not in stats['missing_fields']:
                    stats['missing_fields'][key] = 0
                stats['missing_fields'][key] += 1

    return stats


def main():
    print("=" * 80)
    print("验证重建的summary_old.csv数据")
    print("=" * 80)

    # 1. 加载数据
    print("\n步骤1: 加载数据...")
    whitelist = load_whitelist()
    csv_rows = load_csv()

    print(f"✓ 白名单实验数: {len(whitelist)}")
    print(f"✓ CSV行数: {len(csv_rows)}")

    # 检查行数一致性
    if len(csv_rows) == len(whitelist):
        print(f"✓ 行数一致")
    else:
        print(f"✗ 行数不一致: CSV有{len(csv_rows)}行，白名单有{len(whitelist)}个实验")

    # 2. 随机抽样验证
    print("\n步骤2: 随机抽样验证（抽样10个实验）...")
    sample_size = min(10, len(csv_rows))
    sample_rows = random.sample(csv_rows, sample_size)

    valid_count = 0
    error_count = 0
    not_found_count = 0

    for i, row in enumerate(sample_rows, 1):
        exp_id = row['experiment_id']
        exp_source = row.get('experiment_source', '')

        print(f"\n  [{i}/{sample_size}] 验证: {exp_id}")

        json_file, json_data = find_experiment_json(exp_source, exp_id)

        if json_data is None:
            print(f"    ✗ 找不到experiment.json文件")
            not_found_count += 1
            continue

        print(f"    ✓ 找到JSON: {json_file}")

        errors = compare_row_with_json(row, json_data)

        if errors:
            print(f"    ✗ 发现{len(errors)}个不一致:")
            for error in errors:
                print(f"      - {error}")
            error_count += 1
        else:
            print(f"    ✓ 数据一致")
            valid_count += 1

    print(f"\n抽样验证结果:")
    print(f"  - 一致: {valid_count}/{sample_size}")
    print(f"  - 不一致: {error_count}/{sample_size}")
    print(f"  - 找不到JSON: {not_found_count}/{sample_size}")

    # 3. 数据质量统计
    print("\n步骤3: 数据质量统计...")
    stats = validate_data_quality(csv_rows)

    print(f"  总行数: {stats['total_rows']}")
    print(f"  训练成功: {stats['training_success_count']} ({stats['training_success_count']/stats['total_rows']*100:.1f}%)")
    print(f"  有CPU能耗数据: {stats['has_cpu_energy']} ({stats['has_cpu_energy']/stats['total_rows']*100:.1f}%)")
    print(f"  有GPU能耗数据: {stats['has_gpu_energy']} ({stats['has_gpu_energy']/stats['total_rows']*100:.1f}%)")
    print(f"  有性能数据: {stats['has_performance']} ({stats['has_performance']/stats['total_rows']*100:.1f}%)")

    # 显示缺失最多的前10个字段
    print(f"\n  缺失数据最多的字段（前10个）:")
    sorted_missing = sorted(stats['missing_fields'].items(), key=lambda x: x[1], reverse=True)
    for i, (field, count) in enumerate(sorted_missing[:10], 1):
        print(f"    {i}. {field}: {count}/{stats['total_rows']} ({count/stats['total_rows']*100:.1f}%)")

    # 4. 总结
    print("\n" + "=" * 80)
    print("验证总结:")
    print("=" * 80)

    if len(csv_rows) == len(whitelist) and error_count == 0:
        print("✓ 所有检查通过！重建的数据正确。")
        return 0
    else:
        print("✗ 发现问题，请检查上述错误。")
        return 1


if __name__ == '__main__':
    exit(main())
