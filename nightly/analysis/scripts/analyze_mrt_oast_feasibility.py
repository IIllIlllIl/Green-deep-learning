#!/usr/bin/env python3
"""分析 MRT-OAST 作为第6组的可行性

目的：
1. 检查数据量和完整性
2. 分析性能指标情况
3. 分析超参数情况
4. 评估是否满足DiBS分析要求
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=" * 80)
    print("MRT-OAST 第6组可行性分析")
    print("=" * 80)

    # 读取 stage2_mediators.csv
    stage2_path = Path("../data/energy_research/processed.backup_4groups_20251224/stage2_mediators.csv")
    df = pd.read_csv(stage2_path)

    # 筛选 MRT-OAST 数据
    mrt_df = df[df['repository'] == 'MRT-OAST'].copy()

    print(f"\n📊 基础统计")
    print("=" * 80)
    print(f"总行数: {len(mrt_df)} 行")
    print(f"模型: {mrt_df['model'].unique()}")
    print(f"DiBS最低要求: 10 行")
    print(f"DiBS推荐: 20-50 行")
    print(f"状态: {'✅ 充足' if len(mrt_df) >= 20 else '⚠️ 可用但偏少'} ({len(mrt_df)} 行)")

    # 分析性能指标
    print(f"\n📈 性能指标分析")
    print("=" * 80)

    perf_cols = [col for col in mrt_df.columns if col.startswith('perf_')]

    for col in perf_cols:
        non_null = mrt_df[col].notna().sum()
        missing_rate = (len(mrt_df) - non_null) / len(mrt_df) * 100

        if non_null > 0:
            print(f"{col}:")
            print(f"  有效数据: {non_null}/{len(mrt_df)} ({100-missing_rate:.1f}%)")
            print(f"  缺失率: {missing_rate:.1f}%")
            print(f"  范围: [{mrt_df[col].min():.4f}, {mrt_df[col].max():.4f}]")
            print(f"  均值: {mrt_df[col].mean():.4f}")
            print()

    # 确定可用的性能指标
    available_perf = []
    for col in perf_cols:
        if mrt_df[col].notna().sum() > 0:
            available_perf.append(col)

    print(f"可用性能指标 ({len(available_perf)}个): {available_perf}")

    # 删除性能全缺失的行
    if available_perf:
        mrt_clean = mrt_df.dropna(subset=available_perf, how='all')
        deleted = len(mrt_df) - len(mrt_clean)
        print(f"\n删除性能全缺失行: {deleted} 行")
        print(f"保留有效数据: {len(mrt_clean)} 行")
    else:
        mrt_clean = mrt_df
        print(f"\n⚠️ 警告: 没有性能指标数据！")

    # 分析超参数
    print(f"\n🔧 超参数分析")
    print("=" * 80)

    hyperparam_cols = [col for col in mrt_clean.columns if col.startswith('hyperparam_')]

    for col in hyperparam_cols:
        non_null = mrt_clean[col].notna().sum()
        missing_rate = (len(mrt_clean) - non_null) / len(mrt_clean) * 100

        if non_null > 0:
            unique_vals = mrt_clean[col].nunique()
            print(f"{col}:")
            print(f"  有效数据: {non_null}/{len(mrt_clean)} ({100-missing_rate:.1f}%)")
            print(f"  唯一值数: {unique_vals}")

            # 显示值分布
            if unique_vals <= 10:
                value_counts = mrt_clean[col].value_counts().sort_index()
                print(f"  值分布: {dict(value_counts)}")
            else:
                print(f"  范围: [{mrt_clean[col].min()}, {mrt_clean[col].max()}]")
            print()

    # 分析能耗指标
    print(f"\n⚡ 能耗指标分析")
    print("=" * 80)

    energy_cols = [col for col in mrt_clean.columns if col.startswith('energy_')]

    for col in energy_cols:
        non_null = mrt_clean[col].notna().sum()
        missing_rate = (len(mrt_clean) - non_null) / len(mrt_clean) * 100

        if non_null > 0:
            print(f"{col}:")
            print(f"  有效数据: {non_null}/{len(mrt_clean)} ({100-missing_rate:.1f}%)")
            if missing_rate < 10:
                print(f"  范围: [{mrt_clean[col].min():.2f}, {mrt_clean[col].max():.2f}]")
            print()

    # 检查是否有 is_parallel 列
    print(f"\n🔀 并行模式分析")
    print("=" * 80)
    if 'is_parallel' in mrt_clean.columns:
        parallel_counts = mrt_clean['is_parallel'].value_counts()
        print(f"并行模式分布:")
        for mode, count in parallel_counts.items():
            mode_name = "并行" if mode == 1 or mode == True else "非并行"
            print(f"  {mode_name}: {count} 行")
    else:
        print("⚠️ 无 is_parallel 列")

    # DiBS 可行性评估
    print(f"\n✅ DiBS 因果分析可行性评估")
    print("=" * 80)

    criteria = {
        "数据量 ≥ 20": len(mrt_clean) >= 20,
        "性能指标 ≥ 1": len(available_perf) >= 1,
        "能耗数据完整": all(mrt_clean[col].notna().sum() / len(mrt_clean) > 0.8
                            for col in ['energy_cpu_total_joules', 'energy_gpu_total_joules']
                            if col in mrt_clean.columns),
        "超参数 ≥ 3": len([col for col in hyperparam_cols
                           if mrt_clean[col].notna().sum() / len(mrt_clean) > 0.5]) >= 3
    }

    all_passed = all(criteria.values())

    for criterion, passed in criteria.items():
        status = "✅" if passed else "❌"
        print(f"{status} {criterion}")

    print(f"\n{'✅ 结论: 可以作为第6组！' if all_passed else '⚠️ 结论: 需要进一步处理'}")

    # 推荐配置
    if all_passed or len(mrt_clean) >= 10:
        print(f"\n📋 推荐第6组配置")
        print("=" * 80)

        # 推荐超参数（完整性>50%）
        recommended_hyperparams = [
            col for col in hyperparam_cols
            if mrt_clean[col].notna().sum() / len(mrt_clean) > 0.5
        ]

        # 推荐性能指标（完整性>50%）
        recommended_perf = [
            col for col in available_perf
            if mrt_clean[col].notna().sum() / len(mrt_clean) > 0.5
        ]

        print("任务组名称: mrt_oast")
        print(f"仓库: ['MRT-OAST']")
        print(f"模型: ['default']")
        print(f"性能指标 ({len(recommended_perf)}个): {recommended_perf}")
        print(f"超参数 ({len(recommended_hyperparams)}个):")
        print(f"  - training_duration")
        for hp in recommended_hyperparams:
            print(f"  - {hp}")
        print("能耗指标: energy_cpu_total_joules, energy_gpu_total_joules, energy_gpu_avg_watts")
        print("中介变量: gpu_util_avg, gpu_temp_max, cpu_pkg_ratio, gpu_power_fluctuation, gpu_temp_fluctuation")
        print(f"\n预期数据量: {len(mrt_clean)} 行")

    # 保存分析结果
    output_file = Path("../docs/reports/MRT_OAST_FEASIBILITY_ANALYSIS.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("# MRT-OAST 作为第6组的可行性分析\n\n")
        f.write(f"**分析日期**: 2025-12-24\n")
        f.write(f"**数据来源**: stage2_mediators.csv\n\n")
        f.write("---\n\n")
        f.write("## 基础统计\n\n")
        f.write(f"- 总行数: {len(mrt_df)} 行\n")
        f.write(f"- 有效行数（删除性能全缺失）: {len(mrt_clean)} 行\n")
        f.write(f"- 删除行数: {len(mrt_df) - len(mrt_clean)} 行\n")
        f.write(f"- DiBS要求: ✅ 充足（{len(mrt_clean)} ≥ 20）\n\n")

        f.write("## 可行性评估\n\n")
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            f.write(f"{status} {criterion}\n")

        f.write(f"\n**结论**: {'✅ 可以作为第6组' if all_passed else '⚠️ 需要进一步处理'}\n\n")

        if all_passed or len(mrt_clean) >= 10:
            f.write("## 推荐配置\n\n")
            f.write("```python\n")
            f.write("'mrt_oast': {\n")
            f.write("    'repos': ['MRT-OAST'],\n")
            f.write("    'models': {'MRT-OAST': ['default']},\n")
            f.write(f"    'performance_cols': {recommended_perf},\n")
            f.write(f"    'hyperparams': ['training_duration'] + {recommended_hyperparams},\n")
            f.write("    'has_onehot': False,\n")
            f.write("    'onehot_cols': []\n")
            f.write("}\n")
            f.write("```\n\n")
            f.write(f"**预期数据量**: {len(mrt_clean)} 行\n")

    print(f"\n📄 分析报告已保存: {output_file}")


if __name__ == '__main__':
    main()
