#!/usr/bin/env python3
"""
从实验JSON文件中重新分析完成情况（更准确）

遍历所有experiment.json文件，提取超参数和模式信息
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def analyze_from_json_files():
    """从JSON文件分析实验完成情况"""

    # 数据结构：{(repo, model, param, mode): set of unique values}
    unique_values = defaultdict(set)

    project_root = Path(".")
    results_dir = project_root / "results"

    # 遍历所有session目录
    session_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

    print(f"扫描 {len(session_dirs)} 个session目录...")

    total_experiments = 0

    for session_dir in sorted(session_dirs):
        # 遍历session中的所有实验目录
        for exp_dir in session_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            json_file = exp_dir / "experiment.json"
            if not json_file.exists():
                continue

            total_experiments += 1

            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                exp_id = data.get("experiment_id", "")

                # 确定模式
                if "parallel" in exp_id or data.get("mode") == "parallel":
                    mode = "parallel"
                else:
                    mode = "nonparallel"

                # 提取超参数（处理并行和非并行两种结构）
                if "foreground" in data:
                    # 并行实验
                    repo = data["foreground"].get("repository", "")
                    model = data["foreground"].get("model", "")
                    hyperparams = data["foreground"].get("hyperparameters", {})
                else:
                    # 非并行实验
                    repo = data.get("repository", "")
                    model = data.get("model", "")
                    hyperparams = data.get("hyperparameters", {})

                # 记录超参数
                for param, value in hyperparams.items():
                    normalized_value = f"{float(value):.6f}" if isinstance(value, (int, float)) else str(value)
                    unique_values[(repo, model, param, mode)].add(normalized_value)

            except Exception as e:
                print(f"  警告: 读取 {json_file} 失败: {e}")
                continue

    print(f"总共扫描 {total_experiments} 个实验\n")

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
    print("实验完成情况分析（基于JSON文件，区分并行/非并行模式）")
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

    # 考虑去重率，估算实际需要的实验数
    estimated_experiments_50 = int(total_needed_values / 0.5)
    estimated_experiments_40 = int(total_needed_values / 0.6)

    print(f"\n预估需要的实验数（考虑去重）:")
    print(f"  - 保守估计（50%成功率）: {estimated_experiments_50}个")
    print(f"  - 乐观估计（60%成功率）: {estimated_experiments_40}个")

    return missing_list, total_needed_values

if __name__ == "__main__":
    unique_values = analyze_from_json_files()
    missing_list, total_needed = print_analysis(unique_values)

    print("\n" + "=" * 100)
    print("分析完成")
    print("=" * 100)
