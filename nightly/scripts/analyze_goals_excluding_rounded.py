#!/usr/bin/env python3
"""
分析实验目标完成情况（排除可能被四舍五入的数据）

考虑:
1. 仅统计高精度数据（≥3位小数）
2. 0.0 值被视为有效数据（如 siamese 的 test_loss）
3. 整数值被视为可能四舍五入（如 1, 10 等）

版本: v1.0
日期: 2025-12-19
"""

import csv
from collections import defaultdict

def is_high_precision(value_str):
    """
    判断是否为高精度值（≥3位小数）

    特殊情况:
    - 0.0 被视为高精度（有效数据）
    - 整数被视为低精度（可能四舍五入）
    - ≤2位小数被视为低精度
    """
    if not value_str:
        return False

    try:
        float_val = float(value_str)

        # 特殊处理: 0.0 视为高精度（如 siamese 的 test_loss）
        if float_val == 0.0:
            return True

        # 检查小数位数
        if '.' in value_str:
            decimal_places = len(value_str.split('.')[1].rstrip('0'))  # 去除尾随0
            return decimal_places >= 3
        else:
            # 整数被视为低精度
            return False
    except ValueError:
        return False

def analyze_experiment_goals():
    """分析实验目标完成情况（仅统计高精度数据）"""

    print("=" * 100)
    print("实验目标完成情况分析（排除可能四舍五入的数据）")
    print("=" * 100)
    print()

    # 统计每个模型的参数-模式-唯一值
    model_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # 统计数据精度
    precision_stats = {
        'total_experiments': 0,
        'high_precision_experiments': 0,
        'low_precision_experiments': 0,
        'no_perf_experiments': 0
    }

    with open('results/data.csv', 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            precision_stats['total_experiments'] += 1

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

            # 获取性能指标，检查是否有高精度值
            perf_values = {}
            has_high_precision = False

            for col in reader.fieldnames:
                if col.startswith('perf_'):
                    value = row.get(col, '').strip()
                    if value:
                        if is_high_precision(value):
                            perf_values[col] = value
                            has_high_precision = True

            # 分类统计
            if len(perf_values) == 0:
                precision_stats['no_perf_experiments'] += 1
            elif has_high_precision:
                precision_stats['high_precision_experiments'] += 1
            else:
                precision_stats['low_precision_experiments'] += 1

            # 仅统计有高精度性能指标的实验
            if has_high_precision:
                for param_name, param_value in hyperparams.items():
                    model_stats[model_key][mode][param_name].add(param_value)

    # 输出精度统计
    print("数据精度统计:")
    print("-" * 100)
    print(f"总实验数: {precision_stats['total_experiments']}")
    print(f"  有高精度性能指标: {precision_stats['high_precision_experiments']} ({precision_stats['high_precision_experiments']/precision_stats['total_experiments']*100:.1f}%)")
    print(f"  仅有低精度性能指标: {precision_stats['low_precision_experiments']} ({precision_stats['low_precision_experiments']/precision_stats['total_experiments']*100:.1f}%)")
    print(f"  无性能指标: {precision_stats['no_perf_experiments']} ({precision_stats['no_perf_experiments']/precision_stats['total_experiments']*100:.1f}%)")
    print()

    # 分析结果
    print("每个模型的参数-模式组合达标情况（仅统计高精度数据）:")
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
    if total_combinations > 0:
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
    if total_models > 0:
        print(f"完全达标模型: {fully_met_models}/{total_models} ({fully_met_models/total_models*100:.1f}%)")
    print("=" * 100)
    print()

    print("说明:")
    print("  - 高精度数据定义: ≥3位小数 或 0.0（如 siamese 的 test_loss）")
    print("  - 低精度数据定义: 整数 或 ≤2位小数")
    print("  - 仅统计有高精度性能指标的实验")
    print("=" * 100)

if __name__ == '__main__':
    analyze_experiment_goals()
