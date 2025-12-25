#!/usr/bin/env python3
"""全面检查新老数据的空值情况和对比

用途: 检查新数据空值，与老数据对比，找出差异
作者: Claude
日期: 2025-12-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def check_missing_values(df, dataset_name):
    """检查数据集的空值情况

    Args:
        df: DataFrame
        dataset_name: 数据集名称

    Returns:
        dict: 空值统计信息
    """
    print(f"\n{'=' * 80}")
    print(f"{dataset_name} 空值检查")
    print(f"{'=' * 80}")
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")

    # 计算每列的空值
    missing_info = {}
    missing_cols = []

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_rate = missing_count / len(df) * 100

        if missing_count > 0:
            missing_cols.append({
                'column': col,
                'missing_count': int(missing_count),
                'missing_rate': float(missing_rate),
                'total_rows': len(df)
            })
            print(f"  ❌ {col}: {missing_count}/{len(df)} ({missing_rate:.2f}%)")

    if not missing_cols:
        print(f"  ✅ 无任何空值！所有列100%填充")

    # 完全无缺失行
    complete_rows = df.dropna()
    complete_rate = len(complete_rows) / len(df) * 100

    print(f"\n完全无缺失行: {len(complete_rows)}/{len(df)} ({complete_rate:.2f}%)")

    missing_info = {
        'dataset': dataset_name,
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'missing_cols': missing_cols,
        'missing_cols_count': len(missing_cols),
        'complete_rows': int(len(complete_rows)),
        'complete_rate': float(complete_rate)
    }

    return missing_info


def compare_columns(new_df, old_df, task_name):
    """对比新老数据的列差异

    Args:
        new_df: 新数据
        old_df: 老数据
        task_name: 任务名称

    Returns:
        dict: 列对比信息
    """
    print(f"\n{'=' * 80}")
    print(f"{task_name} 新老数据列对比")
    print(f"{'=' * 80}")

    new_cols = set(new_df.columns)
    old_cols = set(old_df.columns)

    # 新增列
    added_cols = new_cols - old_cols
    # 删除列
    removed_cols = old_cols - new_cols
    # 共同列
    common_cols = new_cols & old_cols

    print(f"\n新数据列数: {len(new_cols)}")
    print(f"老数据列数: {len(old_cols)}")
    print(f"共同列数: {len(common_cols)}")

    if added_cols:
        print(f"\n✅ 新增列 ({len(added_cols)}):")
        for col in sorted(added_cols):
            print(f"  + {col}")

    if removed_cols:
        print(f"\n❌ 删除列 ({len(removed_cols)}):")
        for col in sorted(removed_cols):
            print(f"  - {col}")

    # 对比共同列的空值情况
    print(f"\n共同列的空值对比:")
    comparison = []

    for col in sorted(common_cols):
        new_missing = new_df[col].isna().sum()
        old_missing = old_df[col].isna().sum()
        new_rate = new_missing / len(new_df) * 100
        old_rate = old_missing / len(old_df) * 100

        if new_missing > 0 or old_missing > 0:
            improvement = old_rate - new_rate
            comparison.append({
                'column': col,
                'new_missing': int(new_missing),
                'old_missing': int(old_missing),
                'new_rate': float(new_rate),
                'old_rate': float(old_rate),
                'improvement': float(improvement)
            })

            if improvement > 0:
                print(f"  ✅ {col}: {old_rate:.2f}% → {new_rate:.2f}% (改进{improvement:.2f}%)")
            elif improvement < 0:
                print(f"  ⚠️ {col}: {old_rate:.2f}% → {new_rate:.2f}% (增加{abs(improvement):.2f}%)")
            else:
                print(f"  ➡️ {col}: {new_rate:.2f}% (无变化)")

    if not comparison:
        print(f"  ✅ 共同列均无空值！")

    return {
        'task': task_name,
        'new_cols_count': len(new_cols),
        'old_cols_count': len(old_cols),
        'common_cols_count': len(common_cols),
        'added_cols': sorted(list(added_cols)),
        'removed_cols': sorted(list(removed_cols)),
        'missing_comparison': comparison
    }


def main():
    """主函数"""
    print("=" * 80)
    print("新老数据空值全面检查和对比")
    print("=" * 80)

    # 数据文件路径
    new_data_dir = Path('data/energy_research/processed')
    old_data_dir = Path('../data/energy_research/processed')

    # 任务配置
    tasks = {
        'image_classification': {
            'new_file': 'training_data_image_classification.csv',
            'old_file': 'training_data_image_classification.csv'
        },
        'person_reid': {
            'new_file': 'training_data_person_reid.csv',
            'old_file': 'training_data_person_reid.csv'
        },
        'vulberta': {
            'new_file': 'training_data_vulberta.csv',
            'old_file': 'training_data_vulberta.csv'
        },
        'bug_localization': {
            'new_file': 'training_data_bug_localization.csv',
            'old_file': 'training_data_bug_localization.csv'
        }
    }

    all_new_missing = []
    all_comparisons = []

    # 1. 检查新数据空值
    print("\n" + "=" * 80)
    print("第一部分：新数据（v2.0修正后）空值检查")
    print("=" * 80)

    for task_name, files in tasks.items():
        new_file = new_data_dir / files['new_file']

        if new_file.exists():
            df_new = pd.read_csv(new_file)
            missing_info = check_missing_values(df_new, f"{task_name} (新数据)")
            all_new_missing.append(missing_info)
        else:
            print(f"\n⚠️ {task_name} 新数据文件不存在: {new_file}")

    # 2. 检查老数据空值（如果存在）
    print("\n\n" + "=" * 80)
    print("第二部分：老数据（stage6历史数据）空值检查")
    print("=" * 80)

    # 检查老数据目录
    if old_data_dir.exists():
        print(f"✅ 老数据目录存在: {old_data_dir}")

        for task_name, files in tasks.items():
            old_file = old_data_dir / files['old_file']

            if old_file.exists():
                df_old = pd.read_csv(old_file)
                missing_info = check_missing_values(df_old, f"{task_name} (老数据)")

                # 新老数据对比
                new_file = new_data_dir / files['new_file']
                if new_file.exists():
                    df_new = pd.read_csv(new_file)
                    comparison = compare_columns(df_new, df_old, task_name)
                    all_comparisons.append(comparison)
            else:
                print(f"\n⚠️ {task_name} 老数据文件不存在: {old_file}")
    else:
        print(f"⚠️ 老数据目录不存在: {old_data_dir}")
        print(f"⏭️ 跳过老数据检查")

    # 3. 生成汇总报告
    print("\n\n" + "=" * 80)
    print("第三部分：汇总报告")
    print("=" * 80)

    print("\n新数据（v2.0修正后）空值汇总:")
    print("-" * 80)

    total_new_rows = sum(m['total_rows'] for m in all_new_missing)
    total_new_cols = sum(m['total_cols'] for m in all_new_missing)
    total_missing_cols = sum(m['missing_cols_count'] for m in all_new_missing)
    avg_complete_rate = sum(m['complete_rate'] for m in all_new_missing) / len(all_new_missing) if all_new_missing else 0

    print(f"总样本量: {total_new_rows} 行")
    print(f"总列数: {total_new_cols} 列（跨任务累加）")
    print(f"有空值的列: {total_missing_cols} 列")
    print(f"平均完全无缺失行比例: {avg_complete_rate:.2f}%")

    if total_missing_cols == 0:
        print(f"\n🎉 新数据质量完美！所有列100%填充，无任何空值！")

    # 保存详细报告
    report = {
        'summary': {
            'total_new_rows': total_new_rows,
            'total_new_cols': total_new_cols,
            'total_missing_cols': total_missing_cols,
            'avg_complete_rate': avg_complete_rate
        },
        'new_data_missing': all_new_missing,
        'column_comparisons': all_comparisons
    }

    report_file = Path('../docs/reports/MISSING_VALUES_COMPREHENSIVE_CHECK_20251224.json')
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ 详细报告已保存: {report_file}")

    # 生成Markdown报告
    generate_markdown_report(report, all_new_missing, all_comparisons)


def generate_markdown_report(report, all_new_missing, all_comparisons):
    """生成Markdown格式的报告

    Args:
        report: JSON报告数据
        all_new_missing: 新数据空值信息
        all_comparisons: 新老数据对比
    """
    md_file = Path('../docs/reports/MISSING_VALUES_COMPREHENSIVE_CHECK_20251224.md')

    md = []
    md.append("# 新老数据空值全面检查报告\n")
    md.append("**日期**: 2025-12-24\n")
    md.append("**对比**: 新数据（v2.0修正后）vs 老数据（stage6历史数据）\n\n")
    md.append("---\n\n")

    # 汇总
    md.append("## 一、新数据（v2.0）空值汇总\n\n")

    summary = report['summary']
    md.append(f"- **总样本量**: {summary['total_new_rows']} 行\n")
    md.append(f"- **总列数**: {summary['total_new_cols']} 列（跨任务累加）\n")
    md.append(f"- **有空值的列**: {summary['total_missing_cols']} 列\n")
    md.append(f"- **平均完全无缺失行比例**: {summary['avg_complete_rate']:.2f}%\n\n")

    if summary['total_missing_cols'] == 0:
        md.append("🎉 **新数据质量完美！所有列100%填充，无任何空值！**\n\n")

    # 详细空值检查
    md.append("## 二、各任务组空值详情\n\n")

    for missing_info in all_new_missing:
        md.append(f"### {missing_info['dataset']}\n\n")
        md.append(f"- 总行数: {missing_info['total_rows']}\n")
        md.append(f"- 总列数: {missing_info['total_cols']}\n")
        md.append(f"- 有空值列数: {missing_info['missing_cols_count']}\n")
        md.append(f"- 完全无缺失行: {missing_info['complete_rows']}/{missing_info['total_rows']} ({missing_info['complete_rate']:.2f}%)\n\n")

        if missing_info['missing_cols']:
            md.append("**空值列**:\n\n")
            md.append("| 列名 | 缺失数量 | 缺失率 |\n")
            md.append("|------|---------|-------|\n")
            for col_info in missing_info['missing_cols']:
                md.append(f"| {col_info['column']} | {col_info['missing_count']}/{col_info['total_rows']} | {col_info['missing_rate']:.2f}% |\n")
            md.append("\n")
        else:
            md.append("✅ **无任何空值！所有列100%填充**\n\n")

    # 新老数据对比
    if all_comparisons:
        md.append("## 三、新老数据对比\n\n")

        for comp in all_comparisons:
            md.append(f"### {comp['task']}\n\n")
            md.append(f"- 新数据列数: {comp['new_cols_count']}\n")
            md.append(f"- 老数据列数: {comp['old_cols_count']}\n")
            md.append(f"- 共同列数: {comp['common_cols_count']}\n\n")

            if comp['added_cols']:
                md.append(f"**新增列** ({len(comp['added_cols'])}):\n")
                for col in comp['added_cols']:
                    md.append(f"- ✅ {col}\n")
                md.append("\n")

            if comp['removed_cols']:
                md.append(f"**删除列** ({len(comp['removed_cols'])}):\n")
                for col in comp['removed_cols']:
                    md.append(f"- ❌ {col}\n")
                md.append("\n")

            if comp['missing_comparison']:
                md.append("**空值改进对比**:\n\n")
                md.append("| 列名 | 老数据缺失率 | 新数据缺失率 | 改进 |\n")
                md.append("|------|------------|------------|------|\n")
                for m in comp['missing_comparison']:
                    improvement = m['improvement']
                    if improvement > 0:
                        icon = "✅"
                    elif improvement < 0:
                        icon = "⚠️"
                    else:
                        icon = "➡️"
                    md.append(f"| {m['column']} | {m['old_rate']:.2f}% | {m['new_rate']:.2f}% | {icon} {improvement:+.2f}% |\n")
                md.append("\n")

    md.append("---\n\n")
    md.append("**报告生成时间**: 2025-12-24\n")
    md.append("**生成脚本**: `check_missing_values_comprehensive.py`\n")

    # 保存Markdown
    with open(md_file, 'w', encoding='utf-8') as f:
        f.writelines(md)

    print(f"✅ Markdown报告已保存: {md_file}")


if __name__ == '__main__':
    main()
