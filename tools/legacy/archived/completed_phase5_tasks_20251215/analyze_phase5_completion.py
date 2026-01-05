#!/usr/bin/env python3
"""
分析Phase 5后的实验目标距离

基于raw_data.csv分析各个模型的唯一参数值数量，
重点关注Phase 5目标模型（VulBERTa/mlp, bug-localization, MRT-OAST, mnist, mnist_ff）
"""

import csv
from collections import defaultdict

# 读取raw_data.csv
with open('data/raw_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 统计各个模型的唯一参数值
stats = defaultdict(lambda: defaultdict(lambda: {'parallel': set(), 'nonparallel': set()}))

for row in rows:
    repo = row['repository']
    model = row['model']
    mode = row.get('mode', '')

    # 跳过训练失败的实验
    if row['training_success'] != 'True':
        continue

    # 确定模式
    mode_key = 'parallel' if mode == 'parallel' else 'nonparallel'

    # 获取该模型的超参数配置
    model_key = f"{repo}/{model}"

    # 检查所有超参数字段
    for col in row.keys():
        if col.startswith('hyperparam_'):
            param_name = col.replace('hyperparam_', '')
            value = row[col]
            if value:  # 只统计非空值
                stats[model_key][param_name][mode_key].add(value)

# Phase 5目标模型
phase5_models = [
    'VulBERTa/mlp',
    'bug-localization-by-dnn-and-rvsm/default',
    'MRT-OAST/default',
    'examples/mnist',
    'examples/mnist_ff'
]

print('=' * 100)
print('Phase 5 后实验目标距离分析')
print('=' * 100)
print()

print('Phase 5 目标模型（并行模式）')
print('=' * 100)
for model in phase5_models:
    if model not in stats:
        print(f"\n{model}: 无数据")
        continue

    print(f"\n{model}:")
    param_stats = stats[model]

    # 并行模式统计
    parallel_params = []
    for param, mode_data in sorted(param_stats.items()):
        count = len(mode_data['parallel'])
        if count > 0:
            parallel_params.append((param, count))

    if parallel_params:
        print(f"  并行模式:")
        for param, count in parallel_params:
            status = "✅ 达标" if count >= 5 else f"❌ 缺{5-count}个"
            print(f"    - {param}: {count}个唯一值 {status}")
    else:
        print(f"  并行模式: 无数据")

    # 非并行模式统计
    nonparallel_params = []
    for param, mode_data in sorted(param_stats.items()):
        count = len(mode_data['nonparallel'])
        if count > 0:
            nonparallel_params.append((param, count))

    if nonparallel_params:
        print(f"  非并行模式:")
        for param, count in nonparallel_params:
            status = "✅ 达标" if count >= 5 else f"❌ 缺{5-count}个"
            print(f"    - {param}: {count}个唯一值 {status}")

print()
print('=' * 100)
print('所有模型完成度汇总（并行模式）')
print('=' * 100)

# 统计所有模型的并行模式完成度
all_models = sorted(stats.keys())
for model in all_models:
    param_stats = stats[model]
    parallel_params = {param: len(mode_data['parallel'])
                       for param, mode_data in param_stats.items()
                       if len(mode_data['parallel']) > 0}

    if not parallel_params:
        continue

    min_count = min(parallel_params.values())
    max_count = max(parallel_params.values())
    avg_count = sum(parallel_params.values()) / len(parallel_params)

    status = "✅ 达标" if min_count >= 5 else f"❌ 最少{min_count}个"
    print(f"{model}: {len(parallel_params)}参数, 最少{min_count}个, 最多{max_count}个, 平均{avg_count:.1f}个 {status}")

print()
print('=' * 100)
print('所有模型完成度汇总（非并行模式）')
print('=' * 100)

for model in all_models:
    param_stats = stats[model]
    nonparallel_params = {param: len(mode_data['nonparallel'])
                          for param, mode_data in param_stats.items()
                          if len(mode_data['nonparallel']) > 0}

    if not nonparallel_params:
        continue

    min_count = min(nonparallel_params.values())
    max_count = max(nonparallel_params.values())
    avg_count = sum(nonparallel_params.values()) / len(nonparallel_params)

    status = "✅ 达标" if min_count >= 5 else f"❌ 最少{min_count}个"
    print(f"{model}: {len(nonparallel_params)}参数, 最少{min_count}个, 最多{max_count}个, 平均{avg_count:.1f}个 {status}")
