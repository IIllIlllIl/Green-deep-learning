#!/usr/bin/env python3
"""
重新分析实验目标完成情况（清理后）

考虑:
1. 删除了不合理的 MRT-OAST accuracy 数据（样本数）
2. 不算无法修复的 0.0 值

版本: v1.0
日期: 2025-12-19
"""

import csv
from collections import defaultdict

def analyze_experiment_goals():
    """分析实验目标完成情况"""

    print("=" * 100)
    print("实验目标完成情况分析（清理后）")
    print("=" * 100)
    print()

    # 实验目标: 每个参数在两种模式下需要5个唯一值
    # 45参数 × 2模式 = 90个参数-模式组合

    # 统计每个模型的参数-模式-唯一值
    model_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # 统计性能指标空值
    perf_empty_stats = defaultdict(lambda: {'total': 0, 'empty': 0})

    with open('results/data.csv', 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            repo = row['repository']
            model = row['model']
            is_parallel = (row['is_parallel'] == 'True')
            mode = 'parallel' if is_parallel else 'nonparallel'

            model_key = f"{repo}/{model}"

            # 获取所有超参数
            hyperparams = {}
            for col in reader.fieldnames:
                if col.startswith('hyperparam_'):
                    value = row.get(col, '').strip()
                    if value:
                        param_name = col.replace('hyperparam_', '')
                        hyperparams[param_name] = value

            # 获取性能指标
            perf_values = {}
            for col in reader.fieldnames:
                if col.startswith('perf_'):
                    value = row.get(col, '').strip()
                    metric_name = col.replace('perf_', '')
                    perf_empty_stats[model_key]['total'] += 1
                    if value:
                        # 检查是否是有效值（不是0.0或空）
                        try:
                            float_val = float(value)
                            # 0.0 也算有效值（如 siamese 的 test_loss）
                            perf_values[metric_name] = value
                        except ValueError:
                            pass
                    else:
                        perf_empty_stats[model_key]['empty'] += 1

            # 检查该实验是否有性能指标
            has_perf = len(perf_values) > 0

            # 统计每个超参数的唯一值（仅当有性能指标时）
            if has_perf:
                for param_name, param_value in hyperparams.items():
                    model_stats[model_key][mode][param_name].add(param_value)

    # 分析结果
    print("每个模型的参数-模式组合达标情况:")
    print("-" * 100)
    print(f"{'模型':<50} {'模式':<12} {'达标参数':<15} {'状态'}")
    print("-" * 100)

    total_combinations = 0
    total_met = 0
    model_summary = {}

    for model_key in sorted(model_stats.keys()):
        model_summary[model_key] = {'nonparallel': 0, 'parallel': 0}

        for mode in ['nonparallel', 'parallel']:
            params = model_stats[model_key][mode]
            met_params = sum(1 for unique_vals in params.values() if len(unique_vals) >= 5)
            total_params = len(params)

            total_combinations += total_params
            total_met += met_params
            model_summary[model_key][mode] = met_params

            status = "✅ 完全达标" if met_params == total_params and total_params > 0 else f"{met_params}/{total_params}"

            print(f"{model_key:<50} {mode:<12} {met_params}/{total_params:<12} {status}")

    # 总结
    print()
    print("=" * 100)
    print("总结")
    print("=" * 100)
    print(f"总参数-模式组合: {total_combinations}")
    print(f"达标组合（≥5个唯一值）: {total_met}")
    print(f"达标率: {total_met}/{total_combinations} ({total_met/total_combinations*100:.1f}%)")
    print()

    # 分析每个模型的达标情况
    print("各模型达标情况汇总:")
    print("-" * 100)
    print(f"{'模型':<50} {'非并行':<10} {'并行':<10} {'状态'}")
    print("-" * 100)

    fully_met_models = 0
    total_models = len(model_summary)

    for model_key in sorted(model_summary.keys()):
        nonpar = model_summary[model_key]['nonparallel']
        par = model_summary[model_key]['parallel']

        # 获取该模型的总参数数
        total_params_nonpar = len(model_stats[model_key]['nonparallel'])
        total_params_par = len(model_stats[model_key]['parallel'])

        if nonpar == total_params_nonpar and par == total_params_par and total_params_nonpar > 0 and total_params_par > 0:
            status = "✅ 100%达标"
            fully_met_models += 1
        else:
            status = f"部分达标"

        print(f"{model_key:<50} {nonpar}/{total_params_nonpar:<9} {par}/{total_params_par:<9} {status}")

    print()
    print("=" * 100)
    print(f"完全达标模型: {fully_met_models}/{total_models} ({fully_met_models/total_models*100:.1f}%)")
    print("=" * 100)
    print()

    # 分析性能指标缺失情况
    print("性能指标缺失分析:")
    print("-" * 100)
    print(f"{'模型':<50} {'总指标槽位':<15} {'空值槽位':<15} {'填充率'}")
    print("-" * 100)

    for model_key in sorted(perf_empty_stats.keys()):
        stats = perf_empty_stats[model_key]
        total = stats['total']
        empty = stats['empty']
        filled = total - empty
        fill_rate = filled / total * 100 if total > 0 else 0

        print(f"{model_key:<50} {total:<15} {empty:<15} {fill_rate:.1f}%")

    print()
    print("=" * 100)
    print("说明:")
    print("  - 0.0 值被视为有效数据（如 siamese 的 test_loss）")
    print("  - 已删除不合理的 MRT-OAST accuracy 数据（样本数）")
    print("  - 仅统计有性能指标的实验")
    print("=" * 100)

if __name__ == '__main__':
    analyze_experiment_goals()
