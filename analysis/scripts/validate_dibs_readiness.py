#!/usr/bin/env python3
"""
验证DiBS运行准备状态

检查:
1. 全局标准化数据是否存在
2. DiBS就绪数据是否存在（或可生成）
3. 数据质量是否符合DiBS要求
4. 提供修复建议
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys


def check_global_std_data():
    """检查全局标准化数据"""

    print("\n1. 检查全局标准化数据")
    print("-" * 40)

    data_dir = Path("data/energy_research/6groups_global_std")
    groups = ['group1_examples', 'group2_vulberta', 'group3_person_reid',
              'group4_bug_localization', 'group5_mrt_oast', 'group6_resnet']

    results = {}

    for group in groups:
        file = data_dir / f"{group}_global_std.csv"
        exists = file.exists()

        if exists:
            df = pd.read_csv(file)
            shape = df.shape
            missing_total = df.isnull().sum().sum()
            missing_percent = missing_total / (shape[0] * shape[1]) * 100

            results[group] = {
                'exists': True,
                'shape': shape,
                'missing_total': int(missing_total),
                'missing_percent': float(missing_percent),
                'columns': list(df.columns),
                'timestamp_column': 'timestamp' in df.columns
            }

            print(f"  {group}:")
            print(f"    文件: ✅ {file}")
            print(f"    形状: {shape[0]}行 × {shape[1]}列")
            print(f"    缺失值: {missing_total} ({missing_percent:.1f}%)")
            print(f"    有timestamp列: {'✅' if 'timestamp' in df.columns else '❌'}")

        else:
            results[group] = {'exists': False}
            print(f"  {group}: ❌ 文件不存在")

    return results


def check_dibs_ready_data():
    """检查DiBS就绪数据"""

    print("\n2. 检查DiBS就绪数据")
    print("-" * 40)

    data_dir = Path("data/energy_research/6groups_dibs_ready")
    groups = ['group1_examples', 'group2_vulberta', 'group3_person_reid',
              'group4_bug_localization', 'group5_mrt_oast', 'group6_resnet']

    results = {}

    for group in groups:
        file = data_dir / f"{group}_dibs_ready.csv"
        exists = file.exists()

        if exists:
            df = pd.read_csv(file)
            shape = df.shape
            missing_total = df.isnull().sum().sum()

            results[group] = {
                'exists': True,
                'shape': shape,
                'missing_total': int(missing_total),
                'all_numeric': all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
            }

            print(f"  {group}:")
            print(f"    文件: ✅ {file}")
            print(f"    形状: {shape[0]}行 × {shape[1]}列")
            print(f"    缺失值: {missing_total} {'✅' if missing_total == 0 else '❌'}")
            print(f"    全数值列: {'✅' if results[group]['all_numeric'] else '❌'}")

        else:
            results[group] = {'exists': False}
            print(f"  {group}: ❌ 文件不存在 (需要运行预处理)")

    return results


def analyze_missing_patterns(global_std_results):
    """分析缺失模式，为预处理提供建议"""

    print("\n3. 缺失模式分析")
    print("-" * 40)

    for group, info in global_std_results.items():
        if not info.get('exists', False):
            continue

        print(f"\n  {group}:")

        # 加载数据
        file = Path("data/energy_research/6groups_global_std") / f"{group}_global_std.csv"
        df = pd.read_csv(file)

        # 按列类型分析
        energy_cols = [col for col in df.columns if 'energy' in col.lower()]
        hyperparam_cols = [col for col in df.columns if col.startswith('hyperparam_')]
        perf_cols = [col for col in df.columns if col.startswith('perf_')]
        model_cols = [col for col in df.columns if col.startswith('model_')]

        print(f"    能耗列 ({len(energy_cols)}): ", end="")
        energy_missing = df[energy_cols].isnull().sum().sum()
        print(f"{energy_missing}缺失")

        print(f"    超参数列 ({len(hyperparam_cols)}): ", end="")
        hyper_missing = df[hyperparam_cols].isnull().sum().sum()
        print(f"{hyper_missing}缺失")

        print(f"    性能列 ({len(perf_cols)}): ", end="")
        perf_missing = df[perf_cols].isnull().sum().sum()
        print(f"{perf_missing}缺失")

        print(f"    模型列 ({len(model_cols)}): ", end="")
        model_missing = df[model_cols].isnull().sum().sum()
        print(f"{model_missing}缺失")

        # 识别全NaN的列
        all_nan_cols = []
        for col in df.columns:
            if df[col].isnull().all():
                all_nan_cols.append(col)

        if all_nan_cols:
            print(f"    ⚠️  全NaN列 ({len(all_nan_cols)}): {', '.join(all_nan_cols[:3])}{'...' if len(all_nan_cols) > 3 else ''}")


def generate_recommendations(global_std_results, dibs_ready_results):
    """生成修复建议"""

    print("\n4. 修复建议")
    print("-" * 40)

    groups_missing_dibs_ready = []
    groups_with_missing_values = []

    for group in ['group1_examples', 'group2_vulberta', 'group3_person_reid',
                  'group4_bug_localization', 'group5_mrt_oast', 'group6_resnet']:

        if not global_std_results.get(group, {}).get('exists', False):
            print(f"  {group}: ❌ 缺少全局标准化数据")
            continue

        if not dibs_ready_results.get(group, {}).get('exists', False):
            groups_missing_dibs_ready.append(group)
            continue

        info = dibs_ready_results[group]
        if info.get('missing_total', 0) > 0:
            groups_with_missing_values.append(group)

    if groups_missing_dibs_ready:
        print(f"  需要预处理的组 ({len(groups_missing_dibs_ready)}):")
        for group in groups_missing_dibs_ready:
            print(f"    - {group}: python scripts/preprocess_for_dibs_global_std.py --group {group}")

    if groups_with_missing_values:
        print(f"  需要重新预处理的组 (仍有缺失值):")
        for group in groups_with_missing_values:
            print(f"    - {group}: 检查预处理脚本，可能需要手动修复")

    if not groups_missing_dibs_ready and not groups_with_missing_values:
        print("  ✅ 所有组都已准备好DiBS运行！")
        print("    运行命令: python scripts/run_dibs_6groups_global_std.py")

    # DiBS运行建议
    print(f"\n5. DiBS运行建议")
    print("-" * 40)

    print("  推荐策略:")
    print("  A. 试点运行 (先验证group1):")
    print("      python scripts/run_dibs_6groups_global_std.py --group group1_examples --n-steps 2000")
    print("  B. 批量运行 (所有组):")
    print("      python scripts/run_dibs_6groups_global_std.py --n-steps 5000")
    print("  C. 并行运行 (使用nohup):")
    print("      for group in group1_examples group2_vulberta group3_person_reid group4_bug_localization group5_mrt_oast group6_resnet; do")
    print("        nohup python scripts/run_dibs_6groups_global_std.py --group $group --n-steps 5000 &")
    print("      done")


def main():
    """主函数"""

    print("=" * 80)
    print("DiBS运行准备状态验证")
    print("=" * 80)

    # 检查数据
    global_std_results = check_global_std_data()
    dibs_ready_results = check_dibs_ready_data()

    # 分析
    analyze_missing_patterns(global_std_results)

    # 生成建议
    generate_recommendations(global_std_results, dibs_ready_results)

    # 保存验证报告
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'global_std_data': global_std_results,
        'dibs_ready_data': dibs_ready_results,
        'summary': {
            'total_groups': 6,
            'global_std_available': sum(1 for info in global_std_results.values() if info.get('exists', False)),
            'dibs_ready_available': sum(1 for info in dibs_ready_results.values() if info.get('exists', False)),
            'ready_for_dibs': all(
                dibs_ready_results.get(group, {}).get('exists', False) and
                dibs_ready_results.get(group, {}).get('missing_total', 1) == 0
                for group in global_std_results.keys()
            )
        }
    }

    report_dir = Path("results/energy_research/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "dibs_readiness_report.json"

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ 验证报告保存至: {report_file}")
    print(f"\n{'='*80}")
    print("验证完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
