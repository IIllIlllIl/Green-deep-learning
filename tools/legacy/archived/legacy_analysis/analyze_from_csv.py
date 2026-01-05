#!/usr/bin/env python3
"""
从summary_all.csv分析实验完成情况（区分并行/非并行模式）
"""

import csv
from pathlib import Path
from collections import defaultdict

def analyze_from_csv():
    """从summary_all.csv分析实验完成情况"""

    # 数据结构：{(repo, model, param, mode): set of unique values}
    unique_values = defaultdict(set)

    csv_path = Path("results/summary_all.csv")

    print(f"读取 {csv_path}...")

    total_experiments = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_experiments += 1

            exp_id = row.get('experiment_id', '')
            repo = row.get('repository', '')
            model = row.get('model', '')

            # 确定模式
            if '_parallel' in exp_id:
                mode = 'parallel'
            else:
                mode = 'nonparallel'

            # 提取所有超参数
            hyperparam_columns = [
                'hyperparam_alpha',
                'hyperparam_batch_size',
                'hyperparam_dropout',
                'hyperparam_epochs',
                'hyperparam_kfold',
                'hyperparam_learning_rate',
                'hyperparam_max_iter',
                'hyperparam_seed',
                'hyperparam_weight_decay'
            ]

            for col in hyperparam_columns:
                value = row.get(col, '').strip()

                # 跳过空值
                if not value:
                    continue

                # 提取参数名（去掉hyperparam_前缀）
                param = col.replace('hyperparam_', '')

                # 标准化数值
                try:
                    normalized_value = f"{float(value):.6f}"
                except ValueError:
                    normalized_value = value

                unique_values[(repo, model, param, mode)].add(normalized_value)

    print(f"总共读取 {total_experiments} 个实验\n")

    return unique_values

def print_analysis(unique_values):
    """打印分析结果"""

    # 按repo/model组织
    by_model = defaultdict(lambda: defaultdict(dict))

    for (repo, model, param, mode), values in unique_values.items():
        model_key = f"{repo}/{model}"
        if mode not in by_model[model_key][param]:
            by_model[model_key][param] = {}
        by_model[model_key][param][mode] = len(values)

    # 统计
    total_combinations = 0
    complete_combinations = 0
    missing_list = []

    print("=" * 100)
    print("实验完成情况分析（基于summary_all.csv，区分并行/非并行模式）")
    print("=" * 100)
    print(f"{'模型':<40} {'参数':<20} {'非并行':<10} {'并行':<10} {'状态':<10}")
    print("-" * 100)

    for model_key in sorted(by_model.keys()):
        params = by_model[model_key]

        for param in sorted(params.keys()):
            modes = params[param]
            nonparallel_count = modes.get('nonparallel', 0)
            parallel_count = modes.get('parallel', 0)

            # 统计两种模式
            for mode in ['nonparallel', 'parallel']:
                total_combinations += 1
                count = modes.get(mode, 0)

                if count >= 5:
                    complete_combinations += 1
                else:
                    needed = 5 - count
                    missing_list.append({
                        'model': model_key,
                        'param': param,
                        'mode': mode,
                        'current': count,
                        'needed': needed
                    })

            # 打印行
            nonparallel_status = "✓" if nonparallel_count >= 5 else f"{nonparallel_count}/5"
            parallel_status = "✓" if parallel_count >= 5 else f"{parallel_count}/5"

            overall_status = "✓" if nonparallel_count >= 5 and parallel_count >= 5 else "待补充"

            print(f"{model_key:<40} {param:<20} {nonparallel_status:<10} {parallel_status:<10} {overall_status:<10}")

    print("=" * 100)
    print(f"\n总计参数-模式组合: {total_combinations}")
    print(f"已完成组合: {complete_combinations} ({complete_combinations/total_combinations*100:.1f}%)")
    print(f"待补充组合: {len(missing_list)}")

    # 详细缺失列表
    print("\n" + "=" * 100)
    print("待补充的参数-模式组合详细列表")
    print("=" * 100)

    # 按模式分组
    nonparallel_missing = [x for x in missing_list if x['mode'] == 'nonparallel']
    parallel_missing = [x for x in missing_list if x['mode'] == 'parallel']

    print(f"\n非并行模式缺失: {len(nonparallel_missing)}个")
    print("-" * 100)
    for item in nonparallel_missing:
        print(f"  {item['model']:<40} {item['param']:<20} 当前:{item['current']} 需补充:{item['needed']}")

    print(f"\n并行模式缺失: {len(parallel_missing)}个")
    print("-" * 100)
    for item in parallel_missing:
        print(f"  {item['model']:<40} {item['param']:<20} 当前:{item['current']} 需补充:{item['needed']}")

    # 计算需要的实验数
    total_needed_values = sum(x['needed'] for x in missing_list)
    print(f"\n总共需补充唯一值: {total_needed_values}个")

    print("\n注意：由于去重修复前存在错误跳过的情况，我们不考虑历史去重率")
    print("实际执行时，配置runs_per_config来生成足够的变异实验")

    return missing_list, total_needed_values

if __name__ == "__main__":
    unique_values = analyze_from_csv()
    missing_list, total_needed = print_analysis(unique_values)

    print("\n" + "=" * 100)
    print("分析完成")
    print("=" * 100)
