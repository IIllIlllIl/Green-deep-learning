#!/usr/bin/env python3
"""
数据质量分析脚本

功能:
1. 加载stage2_mediators.csv
2. 全面分析数据质量:
   - 完整性分析（缺失值统计）
   - 分布分析（数值变量）
   - 异常值检测
   - 变量间相关性
   - 分层数据质量（按repository、mode）
   - 因果分析适用性评估
3. 生成详细的质量分析报告

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
INPUT_FILE = PROCESSED_DIR / "stage2_mediators.csv"
REPORT_FILE = PROCESSED_DIR / "data_quality_report.txt"


def load_data(filepath):
    """加载CSV数据"""
    print(f"\n📂 加载数据: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    return df


def analyze_completeness(df):
    """分析数据完整性"""
    print("\n" + "="*80)
    print("1. 数据完整性分析")
    print("="*80)

    # 按列类型分组
    column_groups = {
        '元信息': [c for c in df.columns if c in ['experiment_id', 'timestamp', 'repository', 'model', 'mode', 'is_parallel']],
        '超参数': [c for c in df.columns if c.startswith('hyperparam_')],
        '超参数统一': ['training_duration', 'l2_regularization'],
        '能耗原始': [c for c in df.columns if c.startswith('energy_')],
        '能耗中介': ['gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio', 'gpu_power_fluctuation', 'gpu_temp_fluctuation'],
        '性能': [c for c in df.columns if c.startswith('perf_')],
        '后台任务': [c for c in df.columns if c.startswith('bg_')]
    }

    results = {}

    for group_name, cols in column_groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue

        print(f"\n{group_name} ({len(cols)} 列):")

        group_results = []
        for col in cols:
            filled = df[col].notna().sum()
            fill_rate = (filled / len(df)) * 100
            group_results.append({
                'column': col,
                'filled': filled,
                'fill_rate': fill_rate
            })

            if fill_rate < 50:
                status = "❌"
            elif fill_rate < 80:
                status = "⚠️ "
            else:
                status = "✅"

            print(f"  {status} {col:40s}: {filled:4d}/{len(df)} ({fill_rate:5.1f}%)")

        results[group_name] = group_results

    return results


def analyze_distributions(df):
    """分析数值变量分布"""
    print("\n" + "="*80)
    print("2. 数值变量分布分析")
    print("="*80)

    # 重点分析的变量
    key_vars = {
        '超参数统一': ['training_duration', 'l2_regularization'],
        '能耗中介': ['gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
                     'gpu_power_fluctuation', 'gpu_temp_fluctuation'],
        '超参数': ['hyperparam_learning_rate', 'hyperparam_batch_size',
                   'hyperparam_epochs', 'hyperparam_dropout'],
        '能耗': ['energy_cpu_total_joules', 'energy_gpu_total_joules']
    }

    results = {}

    for group_name, cols in key_vars.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue

        print(f"\n{group_name}:")

        group_results = []
        for col in cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue

            stats = {
                'column': col,
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'q25': data.quantile(0.25),
                'median': data.median(),
                'q75': data.quantile(0.75),
                'max': data.max(),
                'unique': data.nunique()
            }

            group_results.append(stats)

            print(f"\n  {col}:")
            print(f"    N={stats['count']}, 唯一值={stats['unique']}")
            print(f"    均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
            print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"    四分位: Q25={stats['q25']:.4f}, 中位数={stats['median']:.4f}, Q75={stats['q75']:.4f}")

        results[group_name] = group_results

    return results


def detect_outliers(df):
    """检测异常值"""
    print("\n" + "="*80)
    print("3. 异常值检测")
    print("="*80)

    # 使用IQR方法检测异常值
    key_vars = ['training_duration', 'l2_regularization', 'gpu_util_avg',
                'gpu_temp_max', 'cpu_pkg_ratio', 'gpu_power_fluctuation',
                'gpu_temp_fluctuation', 'energy_cpu_total_joules', 'energy_gpu_total_joules']

    key_vars = [c for c in key_vars if c in df.columns]

    results = {}

    for col in key_vars:
        data = df[col].dropna()
        if len(data) < 10:  # 需要足够的数据
            continue

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_rate = (outliers / len(data)) * 100

        results[col] = {
            'outliers': outliers,
            'outlier_rate': outlier_rate,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        if outlier_rate > 10:
            status = "⚠️ "
        elif outlier_rate > 5:
            status = "ℹ️ "
        else:
            status = "✅"

        print(f"{status} {col:30s}: {outliers:3d} 异常值 ({outlier_rate:4.1f}%)")
        if outliers > 0:
            print(f"   正常范围: [{lower_bound:.4f}, {upper_bound:.4f}]")

    return results


def analyze_correlations(df):
    """分析变量间相关性"""
    print("\n" + "="*80)
    print("4. 变量间相关性分析")
    print("="*80)

    # 选择关键变量
    key_vars = [
        'training_duration', 'l2_regularization',
        'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
        'gpu_power_fluctuation', 'gpu_temp_fluctuation',
        'energy_cpu_total_joules', 'energy_gpu_total_joules'
    ]

    key_vars = [c for c in key_vars if c in df.columns]

    # 计算相关性矩阵
    corr_df = df[key_vars].corr()

    print("\n高相关性变量对 (|r| > 0.7):")

    high_corr = []
    for i in range(len(corr_df.columns)):
        for j in range(i+1, len(corr_df.columns)):
            corr_val = corr_df.iloc[i, j]
            if abs(corr_val) > 0.7:
                var1 = corr_df.columns[i]
                var2 = corr_df.columns[j]
                high_corr.append({
                    'var1': var1,
                    'var2': var2,
                    'correlation': corr_val
                })
                print(f"  {var1:30s} <-> {var2:30s}: r={corr_val:6.3f}")

    if not high_corr:
        print("  ✅ 无高度相关的变量对（良好，避免多重共线性）")

    return high_corr, corr_df


def analyze_by_repository(df):
    """按repository分析数据质量"""
    print("\n" + "="*80)
    print("5. 分repository数据质量")
    print("="*80)

    if 'repository' not in df.columns:
        print("⚠️  repository列不存在")
        return None

    repos = df['repository'].value_counts()

    print(f"\nRepository分布:")
    for repo, count in repos.items():
        print(f"  {repo:40s}: {count:4d} ({count/len(df)*100:5.1f}%)")

    # 检查每个repository的数据完整性
    print(f"\n各repository数据完整性:")

    key_vars = ['energy_cpu_total_joules', 'energy_gpu_total_joules',
                'training_duration', 'gpu_util_avg']
    key_vars = [c for c in key_vars if c in df.columns]

    results = {}

    for repo in repos.index:
        repo_df = df[df['repository'] == repo]
        print(f"\n  {repo}:")

        repo_results = {}
        for var in key_vars:
            filled = repo_df[var].notna().sum()
            fill_rate = (filled / len(repo_df)) * 100
            repo_results[var] = fill_rate

            if fill_rate < 70:
                status = "⚠️ "
            else:
                status = "✅"

            print(f"    {status} {var:30s}: {fill_rate:5.1f}%")

        results[repo] = repo_results

    return results


def analyze_by_mode(df):
    """按mode分析数据质量"""
    print("\n" + "="*80)
    print("6. 分mode数据质量")
    print("="*80)

    if 'is_parallel' not in df.columns:
        print("⚠️  is_parallel列不存在")
        return None

    parallel_count = (df['is_parallel'] == True).sum()
    nonparallel_count = (df['is_parallel'] == False).sum()

    print(f"\nMode分布:")
    print(f"  并行模式: {parallel_count:4d} ({parallel_count/len(df)*100:5.1f}%)")
    print(f"  非并行模式: {nonparallel_count:4d} ({nonparallel_count/len(df)*100:5.1f}%)")

    # 检查两种模式的数据完整性
    key_vars = ['energy_cpu_total_joules', 'energy_gpu_total_joules',
                'training_duration', 'gpu_util_avg']
    key_vars = [c for c in key_vars if c in df.columns]

    print(f"\n并行 vs 非并行数据完整性对比:")

    results = {}

    for var in key_vars:
        parallel_fill = df[df['is_parallel'] == True][var].notna().sum()
        parallel_rate = (parallel_fill / parallel_count) * 100 if parallel_count > 0 else 0

        nonparallel_fill = df[df['is_parallel'] == False][var].notna().sum()
        nonparallel_rate = (nonparallel_fill / nonparallel_count) * 100 if nonparallel_count > 0 else 0

        results[var] = {
            'parallel_rate': parallel_rate,
            'nonparallel_rate': nonparallel_rate
        }

        print(f"\n  {var}:")
        print(f"    并行:   {parallel_rate:5.1f}%")
        print(f"    非并行: {nonparallel_rate:5.1f}%")

    return results


def assess_causal_readiness(df):
    """评估因果分析适用性"""
    print("\n" + "="*80)
    print("7. 因果分析适用性评估")
    print("="*80)

    issues = []

    # 1. 样本量检查
    print("\n1. 样本量检查:")
    total_samples = len(df)
    print(f"   总样本: {total_samples}")

    if total_samples < 50:
        issues.append("❌ 总样本量 < 50（DiBS最低要求）")
        print("   ❌ 样本量不足（DiBS需要至少50个样本）")
    elif total_samples < 100:
        issues.append("⚠️  总样本量 < 100（建议至少100个）")
        print("   ⚠️  样本量偏少（建议至少100个样本）")
    else:
        print("   ✅ 样本量充足")

    # 2. 变量完整性检查
    print("\n2. 关键变量完整性:")

    key_vars_groups = {
        '输入变量（超参数）': ['training_duration', 'l2_regularization',
                            'hyperparam_learning_rate', 'hyperparam_batch_size'],
        '中介变量': ['gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio'],
        '输出变量（能耗）': ['energy_cpu_total_joules', 'energy_gpu_total_joules']
    }

    for group_name, vars_list in key_vars_groups.items():
        print(f"\n   {group_name}:")
        vars_list = [v for v in vars_list if v in df.columns]

        for var in vars_list:
            fill_rate = (df[var].notna().sum() / len(df)) * 100

            if fill_rate < 50:
                status = "❌"
                issues.append(f"❌ {var} 填充率 < 50% ({fill_rate:.1f}%)")
            elif fill_rate < 70:
                status = "⚠️ "
                issues.append(f"⚠️  {var} 填充率 < 70% ({fill_rate:.1f}%)")
            else:
                status = "✅"

            print(f"     {status} {var:30s}: {fill_rate:5.1f}%")

    # 3. 数据变异性检查
    print("\n3. 数据变异性检查:")

    key_numeric_vars = ['training_duration', 'gpu_util_avg', 'energy_gpu_total_joules']
    key_numeric_vars = [v for v in key_numeric_vars if v in df.columns]

    for var in key_numeric_vars:
        unique_count = df[var].nunique()

        if unique_count < 5:
            status = "❌"
            issues.append(f"❌ {var} 唯一值 < 5 ({unique_count})")
        elif unique_count < 10:
            status = "⚠️ "
            issues.append(f"⚠️  {var} 唯一值 < 10 ({unique_count})")
        else:
            status = "✅"

        print(f"   {status} {var:30s}: {unique_count} 个唯一值")

    # 4. 分层样本量检查（如果需要分层分析）
    print("\n4. 分层样本量检查:")

    if 'repository' in df.columns:
        repos = df['repository'].value_counts()
        min_repo_samples = repos.min()

        print(f"   最小repository样本量: {min_repo_samples}")

        if min_repo_samples < 20:
            issues.append(f"⚠️  某些repository样本量 < 20")
            print("   ⚠️  某些repository样本量偏少（建议至少20个）")
        else:
            print("   ✅ 所有repository样本量充足")

    # 总结
    print("\n" + "="*80)
    print("因果分析适用性总结:")
    print("="*80)

    if not issues:
        print("✅ 数据完全满足因果分析要求")
        readiness = "excellent"
    elif len([i for i in issues if i.startswith("❌")]) > 0:
        print(f"❌ 数据存在严重问题，不建议直接进行因果分析")
        print("\n主要问题:")
        for issue in issues:
            if issue.startswith("❌"):
                print(f"  {issue}")
        readiness = "poor"
    else:
        print(f"⚠️  数据基本满足要求，但存在以下警告:")
        for issue in issues:
            print(f"  {issue}")
        readiness = "good"

    return readiness, issues


def generate_quality_report(df, all_results):
    """生成完整的数据质量报告"""
    print(f"\n📊 生成数据质量报告...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("数据质量分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"数据文件: {INPUT_FILE}")
    report_lines.append("")

    # 基本信息
    report_lines.append("=" * 80)
    report_lines.append("数据基本信息")
    report_lines.append("=" * 80)
    report_lines.append(f"总行数: {len(df):,}")
    report_lines.append(f"总列数: {len(df.columns)}")
    report_lines.append(f"数据大小: {INPUT_FILE.stat().st_size / 1024:.1f} KB")
    report_lines.append("")

    # 完整性摘要
    report_lines.append("=" * 80)
    report_lines.append("数据完整性摘要")
    report_lines.append("=" * 80)

    if 'completeness' in all_results:
        for group_name, group_data in all_results['completeness'].items():
            avg_fill = np.mean([r['fill_rate'] for r in group_data])
            report_lines.append(f"{group_name}: 平均填充率 {avg_fill:.1f}%")

    report_lines.append("")

    # 异常值摘要
    if 'outliers' in all_results:
        report_lines.append("=" * 80)
        report_lines.append("异常值检测摘要")
        report_lines.append("=" * 80)

        for var, stats in all_results['outliers'].items():
            if stats['outliers'] > 0:
                report_lines.append(f"{var}: {stats['outliers']} 个异常值 ({stats['outlier_rate']:.1f}%)")

        report_lines.append("")

    # 因果分析适用性
    if 'causal_readiness' in all_results:
        report_lines.append("=" * 80)
        report_lines.append("因果分析适用性")
        report_lines.append("=" * 80)

        readiness, issues = all_results['causal_readiness']

        if readiness == "excellent":
            report_lines.append("✅ 数据完全满足因果分析要求")
        elif readiness == "good":
            report_lines.append("⚠️  数据基本满足要求，存在以下警告:")
            for issue in issues:
                report_lines.append(f"  {issue}")
        else:
            report_lines.append("❌ 数据存在严重问题:")
            for issue in issues:
                report_lines.append(f"  {issue}")

        report_lines.append("")

    report_lines.append("=" * 80)

    # 写入报告
    report_content = "\n".join(report_lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ 数据质量报告已保存: {REPORT_FILE}")


def main():
    """主函数"""
    print("=" * 80)
    print("数据质量分析")
    print("=" * 80)

    try:
        # 加载数据
        df = load_data(INPUT_FILE)

        all_results = {}

        # 1. 完整性分析
        all_results['completeness'] = analyze_completeness(df)

        # 2. 分布分析
        all_results['distributions'] = analyze_distributions(df)

        # 3. 异常值检测
        all_results['outliers'] = detect_outliers(df)

        # 4. 相关性分析
        high_corr, corr_matrix = analyze_correlations(df)
        all_results['correlations'] = (high_corr, corr_matrix)

        # 5. 分repository分析
        all_results['by_repository'] = analyze_by_repository(df)

        # 6. 分mode分析
        all_results['by_mode'] = analyze_by_mode(df)

        # 7. 因果分析适用性
        all_results['causal_readiness'] = assess_causal_readiness(df)

        # 生成报告
        generate_quality_report(df, all_results)

        print("\n" + "=" * 80)
        print("✅ 数据质量分析完成")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
