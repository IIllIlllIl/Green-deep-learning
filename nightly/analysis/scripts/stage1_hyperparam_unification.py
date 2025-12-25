#!/usr/bin/env python3
"""
阶段1: 超参数统一 (Hyperparameter Unification)

功能:
1. 加载阶段0验证数据
2. 创建统一超参数:
   - training_duration = epochs (如果有) 或 max_iter (如果有)
   - l2_regularization = weight_decay (如果有) 或 alpha (如果有)
3. 验证互斥性
4. 输出: stage1_unified.csv

作者: Analysis Module Team
日期: 2025-12-23
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据路径
DATA_DIR = PROJECT_ROOT / "data" / "energy_research"
PROCESSED_DIR = DATA_DIR / "processed"
INPUT_FILE = PROCESSED_DIR / "stage0_validated.csv"
OUTPUT_FILE = PROCESSED_DIR / "stage1_unified.csv"
REPORT_FILE = PROCESSED_DIR / "stage1_unification_report.txt"


def load_data(filepath):
    """加载CSV数据"""
    print(f"\n📂 加载数据: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    return df


def check_mutual_exclusivity(df):
    """检查epochs和max_iter的互斥性"""
    print("\n🔍 检查超参数互斥性...")

    # 检查epochs和max_iter
    has_epochs = df['hyperparam_epochs'].notna()
    has_max_iter = df['hyperparam_max_iter'].notna()

    both = (has_epochs & has_max_iter).sum()
    epochs_only = (has_epochs & ~has_max_iter).sum()
    max_iter_only = (~has_epochs & has_max_iter).sum()
    neither = (~has_epochs & ~has_max_iter).sum()

    print(f"\n  training_duration 源列分布:")
    print(f"    epochs only: {epochs_only} ({epochs_only/len(df)*100:.1f}%)")
    print(f"    max_iter only: {max_iter_only} ({max_iter_only/len(df)*100:.1f}%)")
    print(f"    both (冲突): {both} ({both/len(df)*100:.1f}%)")
    print(f"    neither: {neither} ({neither/len(df)*100:.1f}%)")

    # 检查weight_decay和alpha
    has_weight_decay = df['hyperparam_weight_decay'].notna()
    has_alpha = df['hyperparam_alpha'].notna()

    both_reg = (has_weight_decay & has_alpha).sum()
    wd_only = (has_weight_decay & ~has_alpha).sum()
    alpha_only = (~has_weight_decay & has_alpha).sum()
    neither_reg = (~has_weight_decay & ~has_alpha).sum()

    print(f"\n  l2_regularization 源列分布:")
    print(f"    weight_decay only: {wd_only} ({wd_only/len(df)*100:.1f}%)")
    print(f"    alpha only: {alpha_only} ({alpha_only/len(df)*100:.1f}%)")
    print(f"    both (冲突): {both_reg} ({both_reg/len(df)*100:.1f}%)")
    print(f"    neither: {neither_reg} ({neither_reg/len(df)*100:.1f}%)")

    issues = []
    if both > 0:
        issues.append(f"⚠️  {both} 行同时有epochs和max_iter")
    if both_reg > 0:
        issues.append(f"⚠️  {both_reg} 行同时有weight_decay和alpha")

    if not issues:
        print(f"\n✅ 互斥性验证通过（无冲突）")

    return issues, {
        'epochs_only': epochs_only,
        'max_iter_only': max_iter_only,
        'both_duration': both,
        'wd_only': wd_only,
        'alpha_only': alpha_only,
        'both_reg': both_reg
    }


def create_training_duration(df):
    """创建training_duration统一列"""
    print("\n🔧 创建 training_duration 列...")

    # 优先使用epochs，如果没有则使用max_iter
    df['training_duration'] = df['hyperparam_epochs'].fillna(df['hyperparam_max_iter'])

    filled = df['training_duration'].notna().sum()
    fill_rate = (filled / len(df)) * 100

    print(f"✅ training_duration 创建成功")
    print(f"  填充行数: {filled}/{len(df)} ({fill_rate:.1f}%)")

    # 统计来源
    from_epochs = (df['hyperparam_epochs'].notna() & df['training_duration'].notna()).sum()
    from_max_iter = (df['hyperparam_epochs'].isna() & df['hyperparam_max_iter'].notna()).sum()

    print(f"  来源分布:")
    print(f"    从 epochs: {from_epochs} ({from_epochs/filled*100:.1f}%)")
    print(f"    从 max_iter: {from_max_iter} ({from_max_iter/filled*100:.1f}%)")

    # 统计范围
    if filled > 0:
        print(f"  数值范围: {df['training_duration'].min():.0f} - {df['training_duration'].max():.0f}")

    return filled, fill_rate


def create_l2_regularization(df):
    """创建l2_regularization统一列"""
    print("\n🔧 创建 l2_regularization 列...")

    # 优先使用weight_decay，如果没有则使用alpha
    df['l2_regularization'] = df['hyperparam_weight_decay'].fillna(df['hyperparam_alpha'])

    filled = df['l2_regularization'].notna().sum()
    fill_rate = (filled / len(df)) * 100

    print(f"✅ l2_regularization 创建成功")
    print(f"  填充行数: {filled}/{len(df)} ({fill_rate:.1f}%)")

    # 统计来源
    from_wd = (df['hyperparam_weight_decay'].notna() & df['l2_regularization'].notna()).sum()
    from_alpha = (df['hyperparam_weight_decay'].isna() & df['hyperparam_alpha'].notna()).sum()

    print(f"  来源分布:")
    print(f"    从 weight_decay: {from_wd} ({from_wd/filled*100 if filled > 0 else 0:.1f}%)")
    print(f"    从 alpha: {from_alpha} ({from_alpha/filled*100 if filled > 0 else 0:.1f}%)")

    # 统计范围
    if filled > 0:
        print(f"  数值范围: {df['l2_regularization'].min():.6f} - {df['l2_regularization'].max():.6f}")

    return filled, fill_rate


def verify_unification(df):
    """验证统一结果"""
    print("\n🔍 验证统一结果...")

    issues = []

    # 验证training_duration
    td_notna = df['training_duration'].notna()
    epochs_notna = df['hyperparam_epochs'].notna()
    max_iter_notna = df['hyperparam_max_iter'].notna()

    # 如果有epochs或max_iter，应该有training_duration
    should_have_td = epochs_notna | max_iter_notna
    missing_td = should_have_td & df['training_duration'].isna()

    if missing_td.sum() > 0:
        issues.append(f"❌ {missing_td.sum()} 行应该有training_duration但缺失")
        print(f"  ❌ training_duration 缺失: {missing_td.sum()} 行")
    else:
        print(f"  ✅ training_duration 完整性正确")

    # 验证l2_regularization
    l2_notna = df['l2_regularization'].notna()
    wd_notna = df['hyperparam_weight_decay'].notna()
    alpha_notna = df['hyperparam_alpha'].notna()

    # 如果有weight_decay或alpha，应该有l2_regularization
    should_have_l2 = wd_notna | alpha_notna
    missing_l2 = should_have_l2 & df['l2_regularization'].isna()

    if missing_l2.sum() > 0:
        issues.append(f"❌ {missing_l2.sum()} 行应该有l2_regularization但缺失")
        print(f"  ❌ l2_regularization 缺失: {missing_l2.sum()} 行")
    else:
        print(f"  ✅ l2_regularization 完整性正确")

    return issues


def generate_unification_report(df, stats, issues):
    """生成统一报告"""
    print(f"\n📊 生成统一报告...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("阶段1: 超参数统一报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"输入文件: {INPUT_FILE}")
    report_lines.append(f"输出文件: {OUTPUT_FILE}")
    report_lines.append("")

    # 数据概览
    report_lines.append("=" * 80)
    report_lines.append("1. 数据概览")
    report_lines.append("=" * 80)
    report_lines.append(f"总行数: {len(df):,}")
    report_lines.append(f"原始列数: {len(df.columns) - 2}")  # 减去新增的2列
    report_lines.append(f"新增列数: 2 (training_duration, l2_regularization)")
    report_lines.append(f"最终列数: {len(df.columns)}")
    report_lines.append("")

    # 统一结果
    report_lines.append("=" * 80)
    report_lines.append("2. 超参数统一结果")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("2.1 training_duration:")
    report_lines.append(f"  填充率: {df['training_duration'].notna().sum()/len(df)*100:.1f}%")
    report_lines.append(f"  来源: epochs ({stats['epochs_only']}), max_iter ({stats['max_iter_only']})")
    if stats['both_duration'] > 0:
        report_lines.append(f"  ⚠️  冲突: {stats['both_duration']} 行同时有epochs和max_iter")
    report_lines.append("")
    report_lines.append("2.2 l2_regularization:")
    report_lines.append(f"  填充率: {df['l2_regularization'].notna().sum()/len(df)*100:.1f}%")
    report_lines.append(f"  来源: weight_decay ({stats['wd_only']}), alpha ({stats['alpha_only']})")
    if stats['both_reg'] > 0:
        report_lines.append(f"  ⚠️  冲突: {stats['both_reg']} 行同时有weight_decay和alpha")
    report_lines.append("")

    # 问题汇总
    report_lines.append("=" * 80)
    report_lines.append("3. 验证问题汇总")
    report_lines.append("=" * 80)

    if issues:
        report_lines.append(f"发现 {len(issues)} 个问题:")
        report_lines.append("")
        for i, issue in enumerate(issues, 1):
            report_lines.append(f"{i}. {issue}")
    else:
        report_lines.append("✅ 未发现问题，超参数统一成功！")

    report_lines.append("")
    report_lines.append("=" * 80)

    # 写入报告
    report_content = "\n".join(report_lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ 统一报告已保存: {REPORT_FILE}")

    # 打印到控制台
    print("\n" + report_content)

    return len(issues) == 0


def save_unified_data(df):
    """保存统一后的数据"""
    print(f"\n💾 保存统一数据...")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ 统一数据已保存: {OUTPUT_FILE}")
    print(f"  行数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    print(f"  新增列: training_duration, l2_regularization")
    print(f"  文件大小: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


def main():
    """主函数"""
    print("=" * 80)
    print("阶段1: 超参数统一 (Hyperparameter Unification)")
    print("=" * 80)

    try:
        # 1. 加载数据
        df = load_data(INPUT_FILE)

        # 2. 检查互斥性
        exclusivity_issues, stats = check_mutual_exclusivity(df)
        all_issues = list(exclusivity_issues)

        # 3. 创建统一列
        td_filled, td_rate = create_training_duration(df)
        l2_filled, l2_rate = create_l2_regularization(df)

        # 4. 验证统一结果
        verification_issues = verify_unification(df)
        all_issues.extend(verification_issues)

        # 5. 生成报告
        unification_passed = generate_unification_report(df, stats, all_issues)

        # 6. 保存数据
        save_unified_data(df)

        if unification_passed:
            print("\n" + "=" * 80)
            print("✅ 阶段1完成: 超参数统一成功")
            print("=" * 80)
            print(f"\n新增变量填充率:")
            print(f"  training_duration: {td_rate:.1f}%")
            print(f"  l2_regularization: {l2_rate:.1f}%")
            return 0
        else:
            print("\n" + "=" * 80)
            print(f"⚠️  阶段1完成: 发现 {len(all_issues)} 个问题")
            print("=" * 80)
            return 1

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
