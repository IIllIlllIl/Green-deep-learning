#!/usr/bin/env python3
"""
分析当前实验完成情况（区分并行/非并行模式）

输出：
1. 每个参数-模式组合的唯一值数量
2. 缺失的参数-模式组合列表
3. 需要补充的实验数量估算
"""

import csv
from collections import defaultdict
from pathlib import Path

# 超参数列映射
HYPERPARAM_COLUMNS = {
    "hyperparam_alpha": "alpha",
    "hyperparam_batch_size": "batch_size",
    "hyperparam_dropout": "dropout",
    "hyperparam_epochs": "epochs",
    "hyperparam_kfold": "kfold",
    "hyperparam_learning_rate": "learning_rate",
    "hyperparam_max_iter": "max_iter",
    "hyperparam_seed": "seed",
    "hyperparam_weight_decay": "weight_decay",
}

def analyze_experiments(csv_path):
    """分析CSV文件中的实验完成情况"""

    # 数据结构：{(repo, model, param, mode): set of unique values}
    unique_values = defaultdict(set)

    # 读取CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            repo = row.get('repository', '')
            model = row.get('model', '')
            exp_id = row.get('experiment_id', '')

            # 确定模式
            if 'parallel' in exp_id:
                mode = 'parallel'
            else:
                mode = 'nonparallel'

            # 提取超参数
            for csv_col, param_name in HYPERPARAM_COLUMNS.items():
                value_str = row.get(csv_col, '').strip()
                if value_str and value_str != '':
                    try:
                        value = float(value_str)
                        # 规范化浮点数（6位小数）
                        normalized_value = f"{value:.6f}"
                        unique_values[(repo, model, param_name, mode)].add(normalized_value)
                    except ValueError:
                        pass

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
    print("实验完成情况分析（区分并行/非并行模式）")
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
                    status = "✓"
                else:
                    needed = 5 - count
                    status = f"需补充{needed}"
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

    # 考虑去重率，估算实际需要的实验数
    # 基于历史数据，去重率约50-60%
    estimated_experiments_50 = int(total_needed_values / 0.5)
    estimated_experiments_40 = int(total_needed_values / 0.6)

    print(f"\n预估需要的实验数（考虑去重）:")
    print(f"  - 保守估计（50%成功率）: {estimated_experiments_50}个")
    print(f"  - 乐观估计（60%成功率）: {estimated_experiments_40}个")

    return missing_list, total_needed_values

if __name__ == "__main__":
    csv_path = Path("results/summary_all.csv")

    if not csv_path.exists():
        print(f"错误: 找不到文件 {csv_path}")
        exit(1)

    unique_values = analyze_experiments(csv_path)
    missing_list, total_needed = print_analysis(unique_values)

    print("\n" + "=" * 100)
    print("分析完成")
    print("=" * 100)
