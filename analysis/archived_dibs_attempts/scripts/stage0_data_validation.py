#!/usr/bin/env python3
"""
阶段0: 数据验证 (Data Validation)

功能:
1. 加载原始数据 (data.csv)
2. 验证数据完整性和质量
3. 生成验证报告
4. 输出: stage0_validated.csv (如果验证通过)

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
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INPUT_FILE = RAW_DIR / "energy_data_original.csv"
OUTPUT_FILE = PROCESSED_DIR / "stage0_validated.csv"

# 验证报告路径
REPORT_FILE = PROCESSED_DIR / "stage0_validation_report.txt"

# 确保输出目录存在
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def validate_file_exists(filepath):
    """验证文件是否存在"""
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    print(f"✅ 文件存在: {filepath}")
    return True


def load_data(filepath):
    """加载CSV数据"""
    print(f"\n📂 加载数据: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    return df


def validate_structure(df):
    """验证数据结构"""
    print("\n🔍 验证数据结构...")

    issues = []

    # 预期列数 (data.csv应该是56列)
    expected_columns = 56
    actual_columns = len(df.columns)

    if actual_columns != expected_columns:
        issues.append(f"⚠️  列数不匹配: 预期{expected_columns}列, 实际{actual_columns}列")
    else:
        print(f"✅ 列数正确: {actual_columns}列")

    # 预期行数 (至少应该有数据)
    if len(df) < 100:
        issues.append(f"⚠️  数据量过少: 只有{len(df)}行")
    else:
        print(f"✅ 数据量充足: {len(df)}行")

    return issues


def validate_required_columns(df):
    """验证必需列是否存在"""
    print("\n🔍 验证必需列...")

    required_columns = {
        '元信息': ['experiment_id', 'timestamp', 'repository', 'model', 'mode', 'is_parallel'],
        '超参数': ['hyperparam_learning_rate', 'hyperparam_batch_size', 'hyperparam_epochs'],
        '能耗': ['energy_cpu_total_joules', 'energy_gpu_total_joules'],
        '性能': ['perf_test_accuracy', 'perf_map', 'perf_eval_loss', 'perf_top1_accuracy']
    }

    issues = []

    for category, columns in required_columns.items():
        print(f"\n  检查 {category} 列:")
        for col in columns:
            if col not in df.columns:
                issues.append(f"❌ 缺失必需列: {col} ({category})")
                print(f"    ❌ {col}")
            else:
                print(f"    ✅ {col}")

    if not issues:
        print(f"\n✅ 所有必需列都存在")

    return issues


def validate_data_types(df):
    """验证数据类型"""
    print("\n🔍 验证数据类型...")

    issues = []

    # 检查is_parallel列应该是布尔型或0/1
    if 'is_parallel' in df.columns:
        unique_values = df['is_parallel'].dropna().unique()
        if not all(v in [True, False, 0, 1, 'True', 'False'] for v in unique_values):
            issues.append(f"⚠️  is_parallel列包含非布尔值: {unique_values}")
        else:
            print(f"✅ is_parallel列类型正确")

    # 检查超参数列应该是数值型
    hyperparam_cols = [c for c in df.columns if 'hyperparam_' in c]
    for col in hyperparam_cols:
        non_numeric = df[col].dropna().apply(lambda x: not isinstance(x, (int, float, np.int64, np.float64)))
        if non_numeric.any():
            count = non_numeric.sum()
            issues.append(f"⚠️  {col} 包含 {count} 个非数值项")

    if not issues:
        print(f"✅ 数据类型验证通过")

    return issues


def check_missing_values(df):
    """检查缺失值情况"""
    print("\n🔍 检查缺失值...")

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_rate = (missing_cells / total_cells) * 100

    print(f"  总单元格数: {total_cells:,}")
    print(f"  缺失单元格数: {missing_cells:,}")
    print(f"  总体缺失率: {missing_rate:.2f}%")

    # 检查关键列的缺失率
    critical_columns = ['experiment_id', 'timestamp', 'repository', 'model', 'mode']

    print(f"\n  关键列缺失率:")
    issues = []

    for col in critical_columns:
        if col in df.columns:
            col_missing = df[col].isna().sum()
            col_missing_rate = (col_missing / len(df)) * 100

            if col_missing_rate > 0:
                issues.append(f"❌ {col}: 缺失 {col_missing} 行 ({col_missing_rate:.2f}%)")
                print(f"    ❌ {col}: {col_missing_rate:.2f}%")
            else:
                print(f"    ✅ {col}: 0%")

    # 检查能耗和性能列的缺失率
    energy_cols = [c for c in df.columns if 'energy_' in c]
    perf_cols = [c for c in df.columns if 'perf_' in c]

    energy_missing = df[energy_cols].isna().all(axis=1).sum()
    perf_missing = df[perf_cols].isna().all(axis=1).sum()

    print(f"\n  数据完整性:")
    print(f"    能耗数据全缺失: {energy_missing} 行 ({energy_missing/len(df)*100:.2f}%)")
    print(f"    性能数据全缺失: {perf_missing} 行 ({perf_missing/len(df)*100:.2f}%)")

    if energy_missing > len(df) * 0.2:  # 超过20%
        issues.append(f"⚠️  能耗数据缺失严重: {energy_missing} 行")
    if perf_missing > len(df) * 0.2:  # 超过20%
        issues.append(f"⚠️  性能数据缺失严重: {perf_missing} 行")

    return issues, {
        'total_missing_rate': missing_rate,
        'energy_missing_rows': energy_missing,
        'perf_missing_rows': perf_missing
    }


def check_data_ranges(df):
    """检查数据范围合理性"""
    print("\n🔍 检查数据范围...")

    issues = []

    # 检查能耗数据不应为负数
    energy_cols = [c for c in df.columns if 'energy_' in c and 'joules' in c]
    for col in energy_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"❌ {col} 包含 {negative_count} 个负值")
                print(f"  ❌ {col}: {negative_count} 个负值")

    # 检查准确率应该在0-1之间
    accuracy_cols = [c for c in df.columns if 'accuracy' in c.lower()]
    for col in accuracy_cols:
        if col in df.columns:
            out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
            if out_of_range > 0:
                # 检查是否是百分比形式 (0-100)
                if df[col].max() > 1:
                    print(f"  ℹ️  {col}: 可能是百分比形式 (范围: {df[col].min():.2f}-{df[col].max():.2f})")
                else:
                    issues.append(f"❌ {col} 包含 {out_of_range} 个超出范围[0,1]的值")
                    print(f"  ❌ {col}: {out_of_range} 个超出范围的值")

    if not issues:
        print(f"✅ 数据范围验证通过")

    return issues


def check_duplicates(df):
    """检查重复记录"""
    print("\n🔍 检查重复记录...")

    issues = []

    # 检查experiment_id + timestamp的唯一性
    if 'experiment_id' in df.columns and 'timestamp' in df.columns:
        df['_composite_key'] = df['experiment_id'].astype(str) + '|' + df['timestamp'].astype(str)
        duplicates = df['_composite_key'].duplicated().sum()

        if duplicates > 0:
            issues.append(f"⚠️  发现 {duplicates} 个重复记录 (experiment_id + timestamp)")
            print(f"  ⚠️  重复记录: {duplicates} 个")

            # 显示重复的记录
            dup_keys = df[df['_composite_key'].duplicated(keep=False)]['_composite_key'].unique()
            print(f"  重复的键 (前5个): {list(dup_keys[:5])}")
        else:
            print(f"✅ 无重复记录")

        df.drop('_composite_key', axis=1, inplace=True)

    return issues


def generate_validation_report(df, all_issues, stats):
    """生成验证报告"""
    print(f"\n📊 生成验证报告...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("阶段0: 数据验证报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"输入文件: {INPUT_FILE}")
    report_lines.append("")

    # 数据概览
    report_lines.append("=" * 80)
    report_lines.append("1. 数据概览")
    report_lines.append("=" * 80)
    report_lines.append(f"总行数: {len(df):,}")
    report_lines.append(f"总列数: {len(df.columns)}")
    report_lines.append(f"总体缺失率: {stats['total_missing_rate']:.2f}%")
    report_lines.append(f"能耗数据全缺失: {stats['energy_missing_rows']} 行")
    report_lines.append(f"性能数据全缺失: {stats['perf_missing_rows']} 行")
    report_lines.append("")

    # Repository分布
    if 'repository' in df.columns:
        report_lines.append("Repository分布:")
        repo_dist = df['repository'].value_counts()
        for repo, count in repo_dist.items():
            report_lines.append(f"  {repo}: {count} ({count/len(df)*100:.1f}%)")
        report_lines.append("")

    # Mode分布
    if 'mode' in df.columns:
        report_lines.append("Mode分布:")
        mode_dist = df['mode'].value_counts()
        for mode, count in mode_dist.items():
            report_lines.append(f"  {mode}: {count} ({count/len(df)*100:.1f}%)")
        report_lines.append("")

    # 问题汇总
    report_lines.append("=" * 80)
    report_lines.append("2. 验证问题汇总")
    report_lines.append("=" * 80)

    if all_issues:
        report_lines.append(f"发现 {len(all_issues)} 个问题:")
        report_lines.append("")
        for i, issue in enumerate(all_issues, 1):
            report_lines.append(f"{i}. {issue}")
    else:
        report_lines.append("✅ 未发现问题，数据验证通过！")

    report_lines.append("")
    report_lines.append("=" * 80)

    # 写入报告文件
    report_content = "\n".join(report_lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ 验证报告已保存: {REPORT_FILE}")

    # 同时打印到控制台
    print("\n" + report_content)

    return len(all_issues) == 0


def save_validated_data(df):
    """保存验证通过的数据"""
    print(f"\n💾 保存验证数据...")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ 验证数据已保存: {OUTPUT_FILE}")
    print(f"  行数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    print(f"  文件大小: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


def main():
    """主函数"""
    print("=" * 80)
    print("阶段0: 数据验证 (Data Validation)")
    print("=" * 80)

    try:
        # 1. 验证文件存在
        validate_file_exists(INPUT_FILE)

        # 2. 加载数据
        df = load_data(INPUT_FILE)

        # 3. 执行各项验证
        all_issues = []

        # 验证数据结构
        all_issues.extend(validate_structure(df))

        # 验证必需列
        all_issues.extend(validate_required_columns(df))

        # 验证数据类型
        all_issues.extend(validate_data_types(df))

        # 检查缺失值
        missing_issues, stats = check_missing_values(df)
        all_issues.extend(missing_issues)

        # 检查数据范围
        all_issues.extend(check_data_ranges(df))

        # 检查重复记录
        all_issues.extend(check_duplicates(df))

        # 4. 生成验证报告
        validation_passed = generate_validation_report(df, all_issues, stats)

        # 5. 保存验证数据
        if validation_passed:
            save_validated_data(df)
            print("\n" + "=" * 80)
            print("✅ 阶段0完成: 数据验证通过")
            print("=" * 80)
            return 0
        else:
            print("\n" + "=" * 80)
            print(f"⚠️  阶段0完成: 发现 {len(all_issues)} 个问题")
            print("=" * 80)
            print("\n建议:")
            print("1. 查看验证报告了解详情")
            print("2. 根据问题类型决定是否继续处理")
            print("3. 如果是警告级别问题，可以继续；如果是错误级别问题，需要修复数据源")

            # 即使有警告也保存数据（供检查）
            save_validated_data(df)
            return 1

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
