#!/usr/bin/env python3
"""
阶段2: 能耗中介变量生成 (Energy Mediator Variables)

功能:
1. 加载阶段1统一数据
2. 创建5个能耗中介变量:
   - gpu_util_avg: GPU平均利用率（直接复制energy_gpu_util_avg_percent）
   - gpu_temp_max: GPU最高温度（直接复制energy_gpu_temp_max_celsius）
   - cpu_pkg_ratio: CPU计算能耗比 = cpu_pkg_joules / cpu_total_joules
   - gpu_power_fluctuation: GPU功率波动 = max_watts - min_watts
   - gpu_temp_fluctuation: GPU温度波动 = temp_max - temp_avg
3. 验证计算结果
4. 输出: stage2_mediators.csv

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
INPUT_FILE = PROCESSED_DIR / "stage1_unified.csv"
OUTPUT_FILE = PROCESSED_DIR / "stage2_mediators.csv"
REPORT_FILE = PROCESSED_DIR / "stage2_mediators_report.txt"


def load_data(filepath):
    """加载CSV数据"""
    print(f"\n📂 加载数据: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    return df


def check_source_columns(df):
    """检查源列是否存在"""
    print("\n🔍 检查源列...")

    required_cols = {
        'gpu_util_avg': 'energy_gpu_util_avg_percent',
        'gpu_temp_max': 'energy_gpu_temp_max_celsius',
        'cpu_pkg_ratio': ['energy_cpu_pkg_joules', 'energy_cpu_total_joules'],
        'gpu_power_fluctuation': ['energy_gpu_max_watts', 'energy_gpu_min_watts'],
        'gpu_temp_fluctuation': ['energy_gpu_temp_max_celsius', 'energy_gpu_temp_avg_celsius']
    }

    issues = []

    for var, source in required_cols.items():
        if isinstance(source, list):
            missing = [col for col in source if col not in df.columns]
            if missing:
                issues.append(f"❌ {var} 缺少源列: {missing}")
                print(f"  ❌ {var}: 缺少 {missing}")
            else:
                print(f"  ✅ {var}: 源列完整")
        else:
            if source not in df.columns:
                issues.append(f"❌ {var} 缺少源列: {source}")
                print(f"  ❌ {var}: 缺少 {source}")
            else:
                print(f"  ✅ {var}: 源列存在")

    if not issues:
        print(f"\n✅ 所有源列都存在")

    return issues


def create_gpu_util_avg(df):
    """创建gpu_util_avg（直接复制）"""
    print("\n🔧 创建 gpu_util_avg 列...")

    df['gpu_util_avg'] = df['energy_gpu_util_avg_percent'].copy()

    filled = df['gpu_util_avg'].notna().sum()
    fill_rate = (filled / len(df)) * 100

    print(f"✅ gpu_util_avg 创建成功")
    print(f"  填充行数: {filled}/{len(df)} ({fill_rate:.1f}%)")

    if filled > 0:
        print(f"  数值范围: {df['gpu_util_avg'].min():.1f}% - {df['gpu_util_avg'].max():.1f}%")
        print(f"  平均值: {df['gpu_util_avg'].mean():.1f}%")

    return filled, fill_rate


def create_gpu_temp_max(df):
    """创建gpu_temp_max（直接复制）"""
    print("\n🔧 创建 gpu_temp_max 列...")

    df['gpu_temp_max'] = df['energy_gpu_temp_max_celsius'].copy()

    filled = df['gpu_temp_max'].notna().sum()
    fill_rate = (filled / len(df)) * 100

    print(f"✅ gpu_temp_max 创建成功")
    print(f"  填充行数: {filled}/{len(df)} ({fill_rate:.1f}%)")

    if filled > 0:
        print(f"  数值范围: {df['gpu_temp_max'].min():.0f}°C - {df['gpu_temp_max'].max():.0f}°C")
        print(f"  平均值: {df['gpu_temp_max'].mean():.1f}°C")

    return filled, fill_rate


def create_cpu_pkg_ratio(df):
    """创建cpu_pkg_ratio = cpu_pkg_joules / cpu_total_joules"""
    print("\n🔧 创建 cpu_pkg_ratio 列...")

    # 只在两者都有值时计算
    mask = (df['energy_cpu_pkg_joules'].notna() &
            df['energy_cpu_total_joules'].notna() &
            (df['energy_cpu_total_joules'] > 0))  # 避免除以0

    df['cpu_pkg_ratio'] = np.nan
    df.loc[mask, 'cpu_pkg_ratio'] = (df.loc[mask, 'energy_cpu_pkg_joules'] /
                                       df.loc[mask, 'energy_cpu_total_joules'])

    filled = df['cpu_pkg_ratio'].notna().sum()
    fill_rate = (filled / len(df)) * 100

    print(f"✅ cpu_pkg_ratio 创建成功")
    print(f"  填充行数: {filled}/{len(df)} ({fill_rate:.1f}%)")

    if filled > 0:
        print(f"  数值范围: {df['cpu_pkg_ratio'].min():.4f} - {df['cpu_pkg_ratio'].max():.4f}")
        print(f"  平均值: {df['cpu_pkg_ratio'].mean():.4f}")

        # 检查异常值（比例应该在0-1之间）
        out_of_range = ((df['cpu_pkg_ratio'] < 0) | (df['cpu_pkg_ratio'] > 1)).sum()
        if out_of_range > 0:
            print(f"  ⚠️  超出[0,1]范围: {out_of_range} 行")

    return filled, fill_rate


def create_gpu_power_fluctuation(df):
    """创建gpu_power_fluctuation = max_watts - min_watts"""
    print("\n🔧 创建 gpu_power_fluctuation 列...")

    # 只在两者都有值时计算
    mask = (df['energy_gpu_max_watts'].notna() &
            df['energy_gpu_min_watts'].notna())

    df['gpu_power_fluctuation'] = np.nan
    df.loc[mask, 'gpu_power_fluctuation'] = (df.loc[mask, 'energy_gpu_max_watts'] -
                                               df.loc[mask, 'energy_gpu_min_watts'])

    filled = df['gpu_power_fluctuation'].notna().sum()
    fill_rate = (filled / len(df)) * 100

    print(f"✅ gpu_power_fluctuation 创建成功")
    print(f"  填充行数: {filled}/{len(df)} ({fill_rate:.1f}%)")

    if filled > 0:
        print(f"  数值范围: {df['gpu_power_fluctuation'].min():.1f}W - {df['gpu_power_fluctuation'].max():.1f}W")
        print(f"  平均值: {df['gpu_power_fluctuation'].mean():.1f}W")

        # 检查负值（波动不应为负）
        negative = (df['gpu_power_fluctuation'] < 0).sum()
        if negative > 0:
            print(f"  ⚠️  负值: {negative} 行（max_watts < min_watts）")

    return filled, fill_rate


def create_gpu_temp_fluctuation(df):
    """创建gpu_temp_fluctuation = temp_max - temp_avg"""
    print("\n🔧 创建 gpu_temp_fluctuation 列...")

    # 只在两者都有值时计算
    mask = (df['energy_gpu_temp_max_celsius'].notna() &
            df['energy_gpu_temp_avg_celsius'].notna())

    df['gpu_temp_fluctuation'] = np.nan
    df.loc[mask, 'gpu_temp_fluctuation'] = (df.loc[mask, 'energy_gpu_temp_max_celsius'] -
                                              df.loc[mask, 'energy_gpu_temp_avg_celsius'])

    filled = df['gpu_temp_fluctuation'].notna().sum()
    fill_rate = (filled / len(df)) * 100

    print(f"✅ gpu_temp_fluctuation 创建成功")
    print(f"  填充行数: {filled}/{len(df)} ({fill_rate:.1f}%)")

    if filled > 0:
        print(f"  数值范围: {df['gpu_temp_fluctuation'].min():.1f}°C - {df['gpu_temp_fluctuation'].max():.1f}°C")
        print(f"  平均值: {df['gpu_temp_fluctuation'].mean():.1f}°C")

        # 检查负值（波动不应为负）
        negative = (df['gpu_temp_fluctuation'] < 0).sum()
        if negative > 0:
            print(f"  ⚠️  负值: {negative} 行（temp_max < temp_avg）")

    return filled, fill_rate


def verify_mediators(df):
    """验证中介变量"""
    print("\n🔍 验证中介变量...")

    issues = []

    # 1. 检查cpu_pkg_ratio范围
    if 'cpu_pkg_ratio' in df.columns:
        out_of_range = ((df['cpu_pkg_ratio'] < 0) | (df['cpu_pkg_ratio'] > 1)).sum()
        if out_of_range > 0:
            issues.append(f"⚠️  cpu_pkg_ratio 有 {out_of_range} 个值超出[0,1]范围")
            print(f"  ⚠️  cpu_pkg_ratio: {out_of_range} 个异常值")

    # 2. 检查波动值不应为负
    if 'gpu_power_fluctuation' in df.columns:
        negative = (df['gpu_power_fluctuation'] < 0).sum()
        if negative > 0:
            issues.append(f"⚠️  gpu_power_fluctuation 有 {negative} 个负值")
            print(f"  ⚠️  gpu_power_fluctuation: {negative} 个负值")

    if 'gpu_temp_fluctuation' in df.columns:
        negative = (df['gpu_temp_fluctuation'] < 0).sum()
        if negative > 0:
            issues.append(f"⚠️  gpu_temp_fluctuation 有 {negative} 个负值")
            print(f"  ⚠️  gpu_temp_fluctuation: {negative} 个负值")

    # 3. 检查整体填充率
    mediator_cols = ['gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
                     'gpu_power_fluctuation', 'gpu_temp_fluctuation']

    at_least_one = df[mediator_cols].notna().any(axis=1).sum()
    coverage = (at_least_one / len(df)) * 100

    print(f"\n  整体覆盖:")
    print(f"    至少有1个中介变量: {at_least_one}/{len(df)} ({coverage:.1f}%)")

    if coverage < 70:
        issues.append(f"⚠️  中介变量覆盖率较低: {coverage:.1f}%")

    if not issues:
        print(f"\n✅ 中介变量验证通过")

    return issues


def generate_mediators_report(df, stats, issues):
    """生成中介变量报告"""
    print(f"\n📊 生成中介变量报告...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("阶段2: 能耗中介变量报告")
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
    report_lines.append(f"原始列数: {len(df.columns) - 5}")  # 减去新增的5列
    report_lines.append(f"新增列数: 5 (中介变量)")
    report_lines.append(f"最终列数: {len(df.columns)}")
    report_lines.append("")

    # 中介变量统计
    report_lines.append("=" * 80)
    report_lines.append("2. 中介变量填充率")
    report_lines.append("=" * 80)

    mediator_cols = ['gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
                     'gpu_power_fluctuation', 'gpu_temp_fluctuation']

    for col in mediator_cols:
        if col in df.columns:
            filled = df[col].notna().sum()
            fill_rate = (filled / len(df)) * 100
            report_lines.append(f"  {col}: {filled}/{len(df)} ({fill_rate:.1f}%)")

    at_least_one = df[mediator_cols].notna().any(axis=1).sum()
    coverage = (at_least_one / len(df)) * 100
    report_lines.append(f"\n  至少有1个中介变量: {at_least_one}/{len(df)} ({coverage:.1f}%)")
    report_lines.append("")

    # 数值范围
    report_lines.append("=" * 80)
    report_lines.append("3. 中介变量数值范围")
    report_lines.append("=" * 80)

    for col in mediator_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            report_lines.append(f"  {col}:")
            report_lines.append(f"    范围: {df[col].min():.4f} - {df[col].max():.4f}")
            report_lines.append(f"    平均: {df[col].mean():.4f}")

    report_lines.append("")

    # 问题汇总
    report_lines.append("=" * 80)
    report_lines.append("4. 验证问题汇总")
    report_lines.append("=" * 80)

    if issues:
        report_lines.append(f"发现 {len(issues)} 个问题:")
        report_lines.append("")
        for i, issue in enumerate(issues, 1):
            report_lines.append(f"{i}. {issue}")
    else:
        report_lines.append("✅ 未发现问题，中介变量生成成功！")

    report_lines.append("")
    report_lines.append("=" * 80)

    # 写入报告
    report_content = "\n".join(report_lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ 中介变量报告已保存: {REPORT_FILE}")

    # 打印到控制台
    print("\n" + report_content)

    return len(issues) == 0


def save_mediators_data(df):
    """保存中介变量数据"""
    print(f"\n💾 保存中介变量数据...")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ 中介变量数据已保存: {OUTPUT_FILE}")
    print(f"  行数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    print(f"  新增列: gpu_util_avg, gpu_temp_max, cpu_pkg_ratio,")
    print(f"          gpu_power_fluctuation, gpu_temp_fluctuation")
    print(f"  文件大小: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


def main():
    """主函数"""
    print("=" * 80)
    print("阶段2: 能耗中介变量生成 (Energy Mediator Variables)")
    print("=" * 80)

    try:
        # 1. 加载数据
        df = load_data(INPUT_FILE)

        # 2. 检查源列
        source_issues = check_source_columns(df)
        all_issues = list(source_issues)

        if source_issues:
            print("\n❌ 缺少必需的源列，无法继续")
            return 1

        # 3. 创建中介变量
        stats = {}
        stats['gpu_util_avg'] = create_gpu_util_avg(df)
        stats['gpu_temp_max'] = create_gpu_temp_max(df)
        stats['cpu_pkg_ratio'] = create_cpu_pkg_ratio(df)
        stats['gpu_power_fluctuation'] = create_gpu_power_fluctuation(df)
        stats['gpu_temp_fluctuation'] = create_gpu_temp_fluctuation(df)

        # 4. 验证中介变量
        verification_issues = verify_mediators(df)
        all_issues.extend(verification_issues)

        # 5. 生成报告
        mediators_passed = generate_mediators_report(df, stats, all_issues)

        # 6. 保存数据
        save_mediators_data(df)

        if mediators_passed or len(all_issues) == 0:
            print("\n" + "=" * 80)
            print("✅ 阶段2完成: 能耗中介变量生成成功")
            print("=" * 80)
            print(f"\n新增变量数: 5")
            print(f"平均填充率: {sum(s[1] for s in stats.values()) / len(stats):.1f}%")
            return 0
        else:
            print("\n" + "=" * 80)
            print(f"⚠️  阶段2完成: 发现 {len(all_issues)} 个问题")
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
