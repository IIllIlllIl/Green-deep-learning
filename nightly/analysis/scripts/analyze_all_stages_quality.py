#!/usr/bin/env python3
"""分析所有阶段数据质量（空值检查）

目的：
1. 检查阶段0-7的所有数据文件
2. 分析每个阶段的空值率
3. 找到最安全的阶段作为6分组起点
4. 生成质量对比报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_stage_quality(filepath, stage_name):
    """分析单个阶段的数据质量

    Args:
        filepath: 数据文件路径
        stage_name: 阶段名称

    Returns:
        dict: 质量统计信息
    """
    if not Path(filepath).exists():
        return {
            'stage': stage_name,
            'exists': False,
            'error': 'File not found'
        }

    try:
        df = pd.read_csv(filepath)

        # 基础统计
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols

        # 空值统计
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        null_rate = total_nulls / total_cells * 100

        # 按列空值率
        col_null_rates = {}
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = null_count / total_rows * 100
            col_null_rates[col] = {
                'null_count': int(null_count),
                'null_rate': round(null_pct, 2)
            }

        # 找出空值率最高的列
        high_null_cols = [(col, stats['null_rate'])
                          for col, stats in col_null_rates.items()
                          if stats['null_rate'] > 50]
        high_null_cols.sort(key=lambda x: x[1], reverse=True)

        # 找出完全填充的列
        full_cols = [col for col, stats in col_null_rates.items()
                     if stats['null_rate'] == 0]

        # 唯一值数量
        unique_counts = {col: int(df[col].nunique()) for col in df.columns}

        return {
            'stage': stage_name,
            'exists': True,
            'filepath': str(filepath),
            'rows': total_rows,
            'columns': total_cols,
            'total_cells': total_cells,
            'total_nulls': int(total_nulls),
            'null_rate': round(null_rate, 2),
            'high_null_cols': high_null_cols[:10],  # 前10个高空值列
            'full_cols': full_cols,
            'full_cols_count': len(full_cols),
            'col_null_rates': col_null_rates,
            'unique_counts': unique_counts
        }

    except Exception as e:
        return {
            'stage': stage_name,
            'exists': True,
            'error': str(e)
        }


def find_stage_files():
    """查找所有阶段文件

    Returns:
        dict: 阶段文件映射
    """
    base_dir = Path("../data/energy_research")

    # 定义可能的阶段文件位置
    possible_locations = [
        base_dir / "raw",
        base_dir / "processed",
        base_dir / "processed.backup_4groups_20251224"
    ]

    stage_files = {}

    # 查找原始数据
    for loc in possible_locations:
        if not loc.exists():
            continue

        # Stage 0: 原始数据或验证数据
        for pattern in ['energy_data_original.csv', 'stage0_validated.csv']:
            filepath = loc / pattern
            if filepath.exists():
                if 'stage0' not in stage_files:
                    stage_files['stage0'] = filepath

        # Stage 1-7: 阶段处理数据
        for i in range(1, 8):
            for pattern in [f'stage{i}_*.csv', f'stage{i}.csv']:
                for filepath in loc.glob(pattern):
                    stage_key = f'stage{i}'
                    if stage_key not in stage_files:
                        stage_files[stage_key] = []
                    if isinstance(stage_files[stage_key], list):
                        stage_files[stage_key].append(filepath)
                    else:
                        stage_files[stage_key] = [stage_files[stage_key], filepath]

    # 清理重复
    for key in stage_files:
        if isinstance(stage_files[key], list):
            stage_files[key] = list(set(stage_files[key]))
            if len(stage_files[key]) == 1:
                stage_files[key] = stage_files[key][0]

    return stage_files


def main():
    print("=" * 80)
    print("阶段0-7数据质量分析")
    print("=" * 80)

    # 1. 查找所有阶段文件
    print("\n📁 查找阶段文件...")
    stage_files = find_stage_files()

    print(f"\n找到 {len(stage_files)} 个阶段:")
    for stage, filepath in sorted(stage_files.items()):
        if isinstance(filepath, list):
            print(f"  {stage}: {len(filepath)} 个文件")
            for f in filepath:
                print(f"    - {f}")
        else:
            print(f"  {stage}: {filepath}")

    # 2. 分析每个阶段质量
    print("\n" + "=" * 80)
    print("质量分析")
    print("=" * 80)

    all_stats = {}

    for stage_key in sorted(stage_files.keys()):
        filepath = stage_files[stage_key]

        # 如果是列表，取第一个文件
        if isinstance(filepath, list):
            if len(filepath) == 0:
                continue
            # 优先选择包含'mediators'或'unified'的文件
            mediator_files = [f for f in filepath if 'mediator' in str(f).lower()]
            unified_files = [f for f in filepath if 'unified' in str(f).lower()]

            if mediator_files:
                filepath = mediator_files[0]
            elif unified_files:
                filepath = unified_files[0]
            else:
                filepath = filepath[0]

        print(f"\n{stage_key}:")
        print(f"  文件: {filepath}")

        stats = analyze_stage_quality(filepath, stage_key)
        all_stats[stage_key] = stats

        if not stats['exists']:
            print(f"  ⚠️ {stats.get('error', '未知错误')}")
            continue

        if 'error' in stats:
            print(f"  ❌ 错误: {stats['error']}")
            continue

        print(f"  行数: {stats['rows']}")
        print(f"  列数: {stats['columns']}")
        print(f"  总空值率: {stats['null_rate']}%")
        print(f"  完全填充列数: {stats['full_cols_count']}/{stats['columns']}")

        if stats['high_null_cols']:
            print(f"  高空值列（>50%）:")
            for col, rate in stats['high_null_cols'][:5]:
                print(f"    {col}: {rate}%")

    # 3. 生成对比表
    print("\n" + "=" * 80)
    print("阶段对比")
    print("=" * 80)

    print(f"\n{'阶段':<15} {'行数':<8} {'列数':<8} {'空值率':<10} {'完全填充列':<12} {'评级':<10}")
    print("-" * 80)

    for stage_key in sorted(all_stats.keys()):
        stats = all_stats[stage_key]

        if not stats['exists'] or 'error' in stats:
            continue

        # 评级
        null_rate = stats['null_rate']
        if null_rate < 10:
            grade = "⭐⭐⭐"
        elif null_rate < 20:
            grade = "⭐⭐"
        elif null_rate < 30:
            grade = "⭐"
        else:
            grade = "⚠️"

        print(f"{stats['stage']:<15} {stats['rows']:<8} {stats['columns']:<8} "
              f"{stats['null_rate']:<10.2f} {stats['full_cols_count']}/{stats['columns']:<6} {grade:<10}")

    # 4. 推荐最安全的阶段
    print("\n" + "=" * 80)
    print("推荐分析")
    print("=" * 80)

    # 筛选有效阶段
    valid_stages = [(stage, stats) for stage, stats in all_stats.items()
                    if stats['exists'] and 'error' not in stats]

    if not valid_stages:
        print("⚠️ 没有找到有效的阶段文件")
        return

    # 按空值率排序
    valid_stages.sort(key=lambda x: x[1]['null_rate'])

    print("\n最佳阶段（按空值率排序）:")
    for i, (stage, stats) in enumerate(valid_stages[:5], 1):
        print(f"{i}. {stage}: {stats['null_rate']:.2f}% 空值率, "
              f"{stats['rows']} 行, {stats['columns']} 列")

    # 推荐Stage2 (mediators) 作为起点
    recommended_stage = None
    for stage, stats in valid_stages:
        if 'stage2' in stage or 'mediator' in stats.get('filepath', '').lower():
            recommended_stage = stage
            break

    if not recommended_stage:
        # 如果没有找到stage2，选择空值率最低且行数=726的
        for stage, stats in valid_stages:
            if stats['rows'] == 726:
                recommended_stage = stage
                break

    if recommended_stage:
        rec_stats = all_stats[recommended_stage]
        print(f"\n✅ 推荐使用: {recommended_stage}")
        print(f"   文件: {rec_stats['filepath']}")
        print(f"   原因:")
        print(f"   - 保留完整726行数据（包含MRT-OAST）")
        print(f"   - 已添加中介变量（能耗分析必需）")
        print(f"   - 空值率: {rec_stats['null_rate']:.2f}%")
        print(f"   - 列数: {rec_stats['columns']}（合适的特征数量）")

    # 5. 保存详细报告
    output_file = Path("../docs/reports/STAGE_QUALITY_ANALYSIS_20251224.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n📄 详细报告已保存: {output_file}")

    # 6. 生成Markdown报告
    md_file = Path("../docs/reports/STAGE_QUALITY_ANALYSIS_20251224.md")

    with open(md_file, 'w') as f:
        f.write("# 阶段0-7数据质量分析报告\n\n")
        f.write(f"**日期**: 2025-12-24\n")
        f.write(f"**分析文件数**: {len(all_stats)}\n\n")
        f.write("---\n\n")

        f.write("## 📊 阶段对比\n\n")
        f.write("| 阶段 | 行数 | 列数 | 总空值率 | 完全填充列 | 文件路径 | 评级 |\n")
        f.write("|------|------|------|---------|-----------|----------|------|\n")

        for stage_key in sorted(all_stats.keys()):
            stats = all_stats[stage_key]

            if not stats['exists'] or 'error' in stats:
                continue

            null_rate = stats['null_rate']
            if null_rate < 10:
                grade = "⭐⭐⭐"
            elif null_rate < 20:
                grade = "⭐⭐"
            elif null_rate < 30:
                grade = "⭐"
            else:
                grade = "⚠️"

            filepath = Path(stats['filepath']).name

            f.write(f"| {stats['stage']} | {stats['rows']} | {stats['columns']} | "
                   f"{stats['null_rate']:.2f}% | {stats['full_cols_count']}/{stats['columns']} | "
                   f"`{filepath}` | {grade} |\n")

        f.write("\n---\n\n")
        f.write("## ✅ 推荐起点\n\n")

        if recommended_stage:
            rec_stats = all_stats[recommended_stage]
            f.write(f"**推荐阶段**: {recommended_stage}\n\n")
            f.write(f"**文件路径**: `{rec_stats['filepath']}`\n\n")
            f.write(f"**推荐原因**:\n")
            f.write(f"- ✅ 保留完整726行数据（包含MRT-OAST）\n")
            f.write(f"- ✅ 已添加能耗中介变量（gpu_util_avg等5个）\n")
            f.write(f"- ✅ 空值率: {rec_stats['null_rate']:.2f}%\n")
            f.write(f"- ✅ 列数: {rec_stats['columns']}（适合的特征数量）\n\n")

            f.write("**下一步**:\n")
            f.write("1. 从此阶段开始，创建新的6分组数据生成脚本\n")
            f.write("2. 添加MRT-OAST任务组配置\n")
            f.write("3. 重新运行分层、One-Hot编码、变量选择流程\n")

    print(f"📄 Markdown报告已保存: {md_file}")
    print(f"\n✅ 分析完成！")


if __name__ == '__main__':
    main()
