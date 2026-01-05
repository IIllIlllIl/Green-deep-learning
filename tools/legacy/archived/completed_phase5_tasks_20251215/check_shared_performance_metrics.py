#!/usr/bin/env python3
"""
检查raw_data.csv中是否存在多个数据行共用性能指标的情况

问题场景：如果使用experiment_id作为唯一标识，可能会导致不同批次的实验
被错误地关联到相同的性能指标数据

用法: python3 scripts/check_shared_performance_metrics.py

版本: 1.0
创建日期: 2025-12-15
"""

import csv
from datetime import datetime
from collections import defaultdict
from pathlib import Path

def check_shared_performance_metrics(csv_path='data/raw_data.csv'):
    """检查是否存在多个数据行共用性能指标的情况"""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f'❌ 文件不存在: {csv_path}')
        return

    # 读取数据
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f'✅ 读取数据: {len(rows)}行')
    print('')

    # 所有性能指标列
    perf_columns = [col for col in rows[0].keys() if col.startswith('perf_')]

    print(f'性能指标列数: {len(perf_columns)}')
    print('')

    # 分析1: 按experiment_id分组（错误方法）
    print('=' * 80)
    print('分析1: 按experiment_id分组（可能导致数据混淆）')
    print('=' * 80)

    by_exp_id = defaultdict(list)
    for row in rows:
        exp_id = row.get('experiment_id', '')
        by_exp_id[exp_id].append(row)

    # 找出使用相同experiment_id的情况
    duplicated_exp_ids = {exp_id: rows_list for exp_id, rows_list in by_exp_id.items() if len(rows_list) > 1}

    print(f'发现 {len(duplicated_exp_ids)} 个experiment_id有多行数据')
    print('')

    # 检查这些重复的experiment_id是否有相同的性能指标
    shared_perf_cases = []

    for exp_id, exp_rows in list(duplicated_exp_ids.items())[:5]:  # 只检查前5个
        print(f'experiment_id: {exp_id}')
        print(f'  数据行数: {len(exp_rows)}')

        # 提取每行的时间戳
        timestamps = [row.get('timestamp', '') for row in exp_rows]
        print(f'  时间戳范围: {min(timestamps)} ~ {max(timestamps)}')

        # 检查性能指标是否相同
        perf_values_sets = []
        for row in exp_rows:
            perf_values = tuple(row.get(col, '') for col in perf_columns if row.get(col, ''))
            if perf_values:
                perf_values_sets.append(perf_values)

        if perf_values_sets:
            unique_perf_values = set(perf_values_sets)
            if len(unique_perf_values) == 1 and len(perf_values_sets) > 1:
                print(f'  ⚠️  发现共用性能指标: {len(perf_values_sets)}行数据使用相同的性能指标值')
                shared_perf_cases.append({
                    'exp_id': exp_id,
                    'row_count': len(exp_rows),
                    'timestamps': timestamps,
                    'perf_values': perf_values_sets[0]
                })
            else:
                print(f'  ✅ 每行有独立的性能指标值')
        else:
            print(f'  ⚠️  没有性能指标数据')

        print('')

    # 分析2: 按复合键分组（正确方法）
    print('=' * 80)
    print('分析2: 按复合键(experiment_id + timestamp)分组（正确方法）')
    print('=' * 80)

    by_composite_key = defaultdict(list)
    for row in rows:
        exp_id = row.get('experiment_id', '')
        timestamp = row.get('timestamp', '')
        composite_key = f"{exp_id}|{timestamp}"
        by_composite_key[composite_key].append(row)

    duplicated_composite_keys = {key: rows_list for key, rows_list in by_composite_key.items() if len(rows_list) > 1}

    print(f'使用复合键后，重复数据行数: {len(duplicated_composite_keys)}')

    if duplicated_composite_keys:
        print('⚠️  发现真正的重复数据（相同experiment_id + 相同timestamp）:')
        for key, dup_rows in list(duplicated_composite_keys.items())[:3]:
            print(f'  复合键: {key}')
            print(f'  重复行数: {len(dup_rows)}')
    else:
        print('✅ 使用复合键后，没有重复数据行')

    print('')

    # 总结
    print('=' * 80)
    print('总结')
    print('=' * 80)

    if shared_perf_cases:
        print(f'❌ 发现 {len(shared_perf_cases)} 个共用性能指标的案例')
        print('')
        print('示例:')
        for i, case in enumerate(shared_perf_cases[:3], 1):
            print(f'{i}. experiment_id: {case["exp_id"]}')
            print(f'   行数: {case["row_count"]}')
            print(f'   时间范围: {min(case["timestamps"])} ~ {max(case["timestamps"])}')
            print(f'   共用的性能指标值: {case["perf_values"][:3]}...')
            print('')

        print('修复方案:')
        print('1. 不应该存在多行数据共用相同性能指标的情况')
        print('2. 每个实验（唯一的experiment_id + timestamp）应该有独立的性能指标')
        print('3. 如果发现共用，需要重新从experiment.json提取正确的性能数据')
    else:
        print('✅ 未发现共用性能指标的情况')
        print('   每个实验都有独立的性能指标值')

    print('')

    # 详细统计
    print('=' * 80)
    print('详细统计')
    print('=' * 80)
    print(f'总数据行数: {len(rows)}')
    print(f'唯一experiment_id数: {len(by_exp_id)}')
    print(f'唯一复合键数: {len(by_composite_key)}')
    print(f'重复experiment_id数: {len(duplicated_exp_ids)}')
    print(f'重复复合键数: {len(duplicated_composite_keys)}')

    return shared_perf_cases

if __name__ == '__main__':
    print('=' * 80)
    print('检查raw_data.csv中是否存在多个数据行共用性能指标')
    print('=' * 80)
    print('')

    shared_cases = check_shared_performance_metrics()

    print('')
    print('=' * 80)
    print('完成')
    print('=' * 80)
