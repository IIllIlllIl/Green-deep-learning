#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析实验状况统计脚本

功能:
- 统计每个模型在并行/非并行模式下的实验数量
- 统计每个模型-参数组合的覆盖情况
- 生成详细的实验状况报告
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict, Counter

def is_empty(val):
    """检查值是否为空"""
    return val == '' or val is None

def main():
    # 读取数据 - 从tools/data_management/到项目根目录需要向上两级
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "raw_data.csv"

    print(f"📊 开始分析实验状况...")
    print(f"数据文件: {data_file}\n")

    # 读取CSV文件
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"总实验数: {total_rows}\n")

    # ===== 1. 基本统计 =====
    print("=" * 80)
    print("📋 基本统计")
    print("=" * 80)

    # 统计实验类型
    exp_source_counts = Counter(row['experiment_source'] for row in rows)
    print("\n实验类型分布:")
    for source, count in sorted(exp_source_counts.items()):
        print(f"  {source}: {count}个")

    # 统计训练成功率
    non_parallel_rows = [r for r in rows if r['mode'] != 'parallel']
    training_success_count = sum(1 for r in non_parallel_rows if r.get('training_success') == 'True')
    print(f"\n非并行模式训练成功: {training_success_count}/{len(non_parallel_rows)}个")

    parallel_rows = [r for r in rows if r['mode'] == 'parallel']
    fg_success_count = sum(1 for r in parallel_rows if r.get('fg_training_success') == 'True')
    print(f"并行模式前台训练成功: {fg_success_count}/{len(parallel_rows)}个")

    # 统计数据完整性 (非空能耗数据)
    non_parallel_with_energy = sum(1 for r in non_parallel_rows
                                   if not is_empty(r.get('energy_cpu_total_joules')))
    parallel_with_energy = sum(1 for r in parallel_rows
                              if not is_empty(r.get('fg_energy_cpu_total_joules')))

    print(f"\n数据完整性 (含能耗数据):")
    print(f"  非并行模式: {non_parallel_with_energy}/{len(non_parallel_rows)}个")
    print(f"  并行模式: {parallel_with_energy}/{len(parallel_rows)}个")
    print(f"  总计: {non_parallel_with_energy + parallel_with_energy}/{total_rows}个 ({(non_parallel_with_energy + parallel_with_energy)*100/total_rows:.1f}%)")

    # ===== 2. 按模型统计 =====
    print("\n" + "=" * 80)
    print("🔬 按模型统计")
    print("=" * 80)

    # 准备模型信息
    model_stats = defaultdict(lambda: {
        'non_parallel': {'total': 0, 'default': 0, 'mutation': 0},
        'parallel': {'total': 0, 'default': 0, 'mutation': 0}
    })

    for row in rows:
        mode = row['mode']
        exp_source = row['experiment_source']

        if mode == 'parallel':
            # 并行模式: 使用fg_前缀字段
            repo = row.get('fg_repository', '')
            model = row.get('fg_model', '')
        else:
            # 非并行模式
            repo = row.get('repository', '')
            model = row.get('model', '')

        if is_empty(repo) or is_empty(model):
            continue

        model_key = f"{repo}/{model}"

        if mode == 'parallel':
            model_stats[model_key]['parallel']['total'] += 1
            if exp_source == 'default':
                model_stats[model_key]['parallel']['default'] += 1
            else:
                model_stats[model_key]['parallel']['mutation'] += 1
        else:
            model_stats[model_key]['non_parallel']['total'] += 1
            if exp_source == 'default':
                model_stats[model_key]['non_parallel']['default'] += 1
            else:
                model_stats[model_key]['non_parallel']['mutation'] += 1

    # 打印模型统计
    print("\n每个模型的实验数量:")
    print(f"{'模型':<50} {'非并行':<30} {'并行':<30} {'总计':<10}")
    print("-" * 125)

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        non_parallel_total = stats['non_parallel']['total']
        parallel_total = stats['parallel']['total']
        total = non_parallel_total + parallel_total

        non_parallel_str = f"{non_parallel_total} (默认:{stats['non_parallel']['default']}, 变异:{stats['non_parallel']['mutation']})"
        parallel_str = f"{parallel_total} (默认:{stats['parallel']['default']}, 变异:{stats['parallel']['mutation']})"

        print(f"{model:<50} {non_parallel_str:<30} {parallel_str:<30} {total:<10}")

    # ===== 3. 按参数变异统计 =====
    print("\n" + "=" * 80)
    print("🧬 按变异参数统计")
    print("=" * 80)

    # 统计变异参数
    mutated_params = Counter(row['mutated_param'] for row in rows
                            if not is_empty(row.get('mutated_param')))
    print("\n变异参数分布:")
    for param, count in sorted(mutated_params.items(), key=lambda x: -x[1]):
        print(f"  {param}: {count}个实验")

    # ===== 4. 模型-参数组合覆盖情况 =====
    print("\n" + "=" * 80)
    print("📊 模型-参数组合覆盖情况")
    print("=" * 80)

    # 统计每个模型变异了哪些参数
    model_param_coverage = defaultdict(lambda: {
        'non_parallel': set(),
        'parallel': set()
    })

    for row in rows:
        param = row.get('mutated_param')
        if is_empty(param):
            continue

        mode = row['mode']

        if mode == 'parallel':
            repo = row.get('fg_repository', '')
            model = row.get('fg_model', '')
        else:
            repo = row.get('repository', '')
            model = row.get('model', '')

        if is_empty(repo) or is_empty(model):
            continue

        model_key = f"{repo}/{model}"

        if mode == 'parallel':
            model_param_coverage[model_key]['parallel'].add(param)
        else:
            model_param_coverage[model_key]['non_parallel'].add(param)

    print("\n每个模型已测试的参数:")
    for model in sorted(model_param_coverage.keys()):
        coverage = model_param_coverage[model]
        non_parallel_params = sorted(coverage['non_parallel'])
        parallel_params = sorted(coverage['parallel'])

        print(f"\n{model}:")
        if non_parallel_params:
            print(f"  非并行: {', '.join(non_parallel_params)}")
        else:
            print(f"  非并行: (无)")

        if parallel_params:
            print(f"  并行: {', '.join(parallel_params)}")
        else:
            print(f"  并行: (无)")

    # ===== 5. 详细的模型-参数-模式矩阵 =====
    print("\n" + "=" * 80)
    print("🗂️  模型-参数-模式实验计数矩阵")
    print("=" * 80)

    # 创建三维统计: 模型 -> 参数 -> 模式 -> 计数
    matrix = defaultdict(lambda: defaultdict(lambda: {'non_parallel': 0, 'parallel': 0}))

    for row in rows:
        param = row.get('mutated_param')
        if is_empty(param):
            param = 'default'

        mode = row['mode']

        if mode == 'parallel':
            repo = row.get('fg_repository', '')
            model = row.get('fg_model', '')
        else:
            repo = row.get('repository', '')
            model = row.get('model', '')

        if is_empty(repo) or is_empty(model):
            continue

        model_key = f"{repo}/{model}"

        if mode == 'parallel':
            matrix[model_key][param]['parallel'] += 1
        else:
            matrix[model_key][param]['non_parallel'] += 1

    # 打印矩阵
    all_params = set()
    for model_data in matrix.values():
        all_params.update(model_data.keys())
    all_params = sorted(all_params)

    for model in sorted(matrix.keys()):
        print(f"\n{model}:")
        print(f"  {'参数':<25} {'非并行':<10} {'并行':<10}")
        print(f"  {'-' * 45}")

        for param in all_params:
            if param in matrix[model]:
                non_par = matrix[model][param]['non_parallel']
                par = matrix[model][param]['parallel']
                if non_par > 0 or par > 0:
                    print(f"  {param:<25} {non_par:<10} {par:<10}")

    # ===== 6. 汇总统计 =====
    print("\n" + "=" * 80)
    print("📈 汇总统计")
    print("=" * 80)

    total_models = len(model_stats)
    total_params = len(mutated_params)

    # 统计有多少个模型-参数-模式组合
    total_combinations = 0
    for model_data in matrix.values():
        for param_data in model_data.values():
            if param_data['non_parallel'] > 0:
                total_combinations += 1
            if param_data['parallel'] > 0:
                total_combinations += 1

    print(f"\n总模型数: {total_models}")
    print(f"总变异参数数: {total_params}")
    print(f"模型-参数-模式组合数: {total_combinations}")
    print(f"总实验数: {total_rows}")

    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
