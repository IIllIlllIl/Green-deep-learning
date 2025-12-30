#!/usr/bin/env python3
"""
阶段7: 最终验证 (Final Validation)

功能:
1. 验证归一化后的数据质量
2. 检查DiBS因果分析适用性
3. 生成DiBS就绪的训练数据文件（去除元信息列）
4. 生成完整的数据预处理报告

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
DATA_DIR = PROJECT_ROOT / "data" / "energy_research" / "processed"
OUTPUT_DIR = DATA_DIR
TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "energy_research" / "training"
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 任务组定义
TASK_GROUPS = {
    'image_classification': '图像分类',
    'person_reid': 'Person_reID',
    'vulberta': 'VulBERTa',
    'bug_localization': 'Bug定位'
}


def validate_normalized_data(df, task_name):
    """
    验证归一化后的数据质量

    Returns:
        dict: 验证结果统计
    """
    stats = {
        'num_samples': len(df),
        'num_features': len(df.columns),
        'total_cells': len(df) * len(df.columns),
        'missing_cells': df.isna().sum().sum(),
        'missing_rate': 0.0,
        'numeric_cols': [],
        'meta_cols': [],
        'onehot_cols': [],
        'issues': []
    }

    stats['missing_rate'] = (stats['missing_cells'] / stats['total_cells']) * 100

    # 分类列
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['experiment_id', 'timestamp', 'repository', 'model', 'mode']):
            stats['meta_cols'].append(col)
        elif col.startswith('is_'):
            stats['onehot_cols'].append(col)
        elif df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            stats['numeric_cols'].append(col)

    # 验证数值列的标准化
    print(f"\n🔍 验证数值列的标准化...")
    for col in stats['numeric_cols']:
        non_null_mask = df[col].notna()
        if non_null_mask.sum() > 0:
            col_mean = df.loc[non_null_mask, col].mean()
            col_std = df.loc[non_null_mask, col].std()

            # 检查均值接近0，标准差接近1
            if abs(col_mean) > 0.2:
                issue = f"⚠️  {col}: 均值={col_mean:.3f} (应接近0)"
                stats['issues'].append(issue)
                print(f"  {issue}")

            if abs(col_std - 1.0) > 0.2:
                issue = f"⚠️  {col}: 标准差={col_std:.3f} (应接近1)"
                stats['issues'].append(issue)
                print(f"  {issue}")

    if not stats['issues']:
        print(f"  ✅ 所有数值列标准化正确")

    # 验证One-Hot列
    if stats['onehot_cols']:
        print(f"\n🔍 验证One-Hot编码...")
        for col in stats['onehot_cols']:
            unique_vals = df[col].dropna().unique()
            if not all(v in [0, 1] for v in unique_vals):
                issue = f"❌ {col}: 包含非0/1值: {unique_vals}"
                stats['issues'].append(issue)
                print(f"  {issue}")
            else:
                print(f"  ✅ {col}: 二值化正确")

    return stats


def check_dibs_readiness(df, stats, task_name):
    """
    检查DiBS因果分析适用性

    Returns:
        dict: DiBS适用性评估
    """
    readiness = {
        'sample_size_ok': False,
        'sample_size': stats['num_samples'],
        'min_samples': 10,
        'recommended_samples': 20,
        'fill_rate_ok': False,
        'fill_rate': 100 - stats['missing_rate'],
        'min_fill_rate': 70.0,
        'variance_ok': True,
        'low_variance_cols': [],
        'overall_ready': False,
        'warnings': [],
        'recommendations': []
    }

    # 1. 样本量检查
    if stats['num_samples'] >= readiness['recommended_samples']:
        readiness['sample_size_ok'] = True
        print(f"  ✅ 样本量: {stats['num_samples']} (≥ {readiness['recommended_samples']} 推荐值)")
    elif stats['num_samples'] >= readiness['min_samples']:
        readiness['sample_size_ok'] = True
        readiness['warnings'].append(f"样本量偏少 ({stats['num_samples']}个)，建议≥{readiness['recommended_samples']}个")
        print(f"  ⚠️  样本量: {stats['num_samples']} (≥ {readiness['min_samples']} 最低要求，但 < {readiness['recommended_samples']} 推荐值)")
    else:
        readiness['sample_size_ok'] = False
        readiness['warnings'].append(f"样本量不足 ({stats['num_samples']}个)，最低要求{readiness['min_samples']}个")
        print(f"  ❌ 样本量: {stats['num_samples']} (< {readiness['min_samples']} 最低要求)")

    # 2. 填充率检查
    if readiness['fill_rate'] >= readiness['min_fill_rate']:
        readiness['fill_rate_ok'] = True
        print(f"  ✅ 填充率: {readiness['fill_rate']:.1f}% (≥ {readiness['min_fill_rate']}%)")
    else:
        readiness['fill_rate_ok'] = False
        readiness['warnings'].append(f"填充率过低 ({readiness['fill_rate']:.1f}%)，最低要求{readiness['min_fill_rate']}%")
        print(f"  ❌ 填充率: {readiness['fill_rate']:.1f}% (< {readiness['min_fill_rate']}%)")

    # 3. 变异性检查
    print(f"\n  检查变量变异性...")
    for col in stats['numeric_cols']:
        non_null_mask = df[col].notna()
        if non_null_mask.sum() > 1:
            unique_count = df.loc[non_null_mask, col].nunique()
            if unique_count < 5:
                readiness['low_variance_cols'].append((col, unique_count))
                readiness['warnings'].append(f"{col} 唯一值数过少 ({unique_count}个)")
                print(f"    ⚠️  {col}: 只有 {unique_count} 个唯一值")

    if not readiness['low_variance_cols']:
        print(f"    ✅ 所有变量变异性充足")
    else:
        readiness['variance_ok'] = False

    # 综合判断
    readiness['overall_ready'] = (
        readiness['sample_size_ok'] and
        readiness['fill_rate_ok'] and
        readiness['variance_ok']
    )

    if readiness['overall_ready']:
        print(f"\n✅ DiBS适用性: 优秀（所有检查通过）")
    elif readiness['sample_size_ok'] and readiness['fill_rate_ok']:
        print(f"\n⚠️  DiBS适用性: 良好（有警告，但可以运行）")
    else:
        print(f"\n❌ DiBS适用性: 不足（需要改进）")

    return readiness


def generate_dibs_training_data(df, stats, task_name):
    """
    生成DiBS就绪的训练数据（去除元信息列）

    Returns:
        pd.DataFrame: DiBS训练数据
    """
    print(f"\n🔧 生成DiBS训练数据...")

    # 只保留数值列和One-Hot列
    feature_cols = stats['numeric_cols'] + stats['onehot_cols']

    # 检查是否所有特征列都存在
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"  ❌ 错误: 缺失列 {missing_cols}")
        return None

    df_training = df[feature_cols].copy()

    print(f"  ✅ DiBS训练数据准备完成")
    print(f"    - 原始列数: {len(df.columns)}")
    print(f"    - 移除元信息列: {len(stats['meta_cols'])} 个")
    print(f"    - 保留特征列: {len(feature_cols)} 个")
    print(f"    - 样本数: {len(df_training)}")

    # 检查是否有全NaN列
    all_nan_cols = [col for col in df_training.columns if df_training[col].isna().all()]
    if all_nan_cols:
        print(f"\n  ⚠️  警告: 以下列全为NaN，DiBS可能无法处理:")
        for col in all_nan_cols:
            print(f"    - {col}")

    return df_training


def validate_task_group(task_name, task_display_name):
    """
    验证单个任务组

    Returns:
        dict: 验证结果
    """
    print(f"\n{'='*80}")
    print(f"任务组: {task_display_name} ({task_name})")
    print(f"{'='*80}\n")

    result = {
        'task_name': task_name,
        'task_display': task_display_name,
        'success': False,
        'stats': None,
        'readiness': None
    }

    # 输入输出文件
    input_file = OUTPUT_DIR / f"stage6_{task_name}.csv"
    output_file = TRAINING_DATA_DIR / f"training_data_{task_name}.csv"
    metadata_file = TRAINING_DATA_DIR / f"metadata_{task_name}.txt"

    # 1. 加载数据
    if not input_file.exists():
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        result['error'] = "输入文件不存在"
        return result

    print(f"📂 加载数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")

    # 2. 验证数据质量
    print(f"\n🔍 验证数据质量...")
    stats = validate_normalized_data(df, task_name)

    print(f"\n数据质量统计:")
    print(f"  - 样本数: {stats['num_samples']}")
    print(f"  - 总列数: {stats['num_features']}")
    print(f"  - 元信息列: {len(stats['meta_cols'])} 个")
    print(f"  - One-Hot列: {len(stats['onehot_cols'])} 个")
    print(f"  - 数值列: {len(stats['numeric_cols'])} 个")
    print(f"  - 总体缺失率: {stats['missing_rate']:.2f}%")
    print(f"  - 数据问题数: {len(stats['issues'])} 个")

    # 3. 检查DiBS适用性
    print(f"\n🔍 检查DiBS适用性...")
    readiness = check_dibs_readiness(df, stats, task_name)

    # 4. 生成DiBS训练数据
    df_training = generate_dibs_training_data(df, stats, task_name)

    if df_training is None:
        result['error'] = "生成DiBS训练数据失败"
        return result

    # 5. 保存DiBS训练数据
    print(f"\n💾 保存DiBS训练数据...")
    df_training.to_csv(output_file, index=False)

    file_size_kb = output_file.stat().st_size / 1024
    print(f"✅ DiBS训练数据已保存: {output_file}")
    print(f"  - 文件大小: {file_size_kb:.1f} KB")

    # 6. 保存元数据文件
    print(f"\n💾 保存元数据...")
    metadata_lines = []
    metadata_lines.append(f"任务组: {task_display_name} ({task_name})")
    metadata_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    metadata_lines.append("")
    metadata_lines.append("数据统计:")
    metadata_lines.append(f"  样本数: {stats['num_samples']}")
    metadata_lines.append(f"  特征数: {len(df_training.columns)}")
    metadata_lines.append(f"  缺失率: {stats['missing_rate']:.2f}%")
    metadata_lines.append("")
    metadata_lines.append("DiBS适用性:")
    metadata_lines.append(f"  样本量: {readiness['sample_size']} (最低{readiness['min_samples']}, 推荐{readiness['recommended_samples']})")
    metadata_lines.append(f"  填充率: {readiness['fill_rate']:.1f}% (最低{readiness['min_fill_rate']}%)")
    metadata_lines.append(f"  整体就绪: {'✅ 是' if readiness['overall_ready'] else '⚠️  有警告' if (readiness['sample_size_ok'] and readiness['fill_rate_ok']) else '❌ 否'}")
    metadata_lines.append("")
    metadata_lines.append("特征列表:")
    for i, col in enumerate(df_training.columns, 1):
        fill_rate = df_training[col].notna().sum() / len(df_training) * 100
        metadata_lines.append(f"  {i}. {col} ({fill_rate:.1f}%)")

    if readiness['warnings']:
        metadata_lines.append("")
        metadata_lines.append("警告:")
        for warning in readiness['warnings']:
            metadata_lines.append(f"  - {warning}")

    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(metadata_lines))

    print(f"✅ 元数据已保存: {metadata_file}")

    result['success'] = True
    result['stats'] = stats
    result['readiness'] = readiness

    return result


def generate_final_report(results):
    """生成最终验证报告"""
    print(f"\n{'='*80}")
    print(f"阶段7: 最终验证报告")
    print(f"{'='*80}\n")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("阶段7: 最终验证 (Final Validation) 报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 汇总统计
    report_lines.append("=" * 80)
    report_lines.append("1. 验证汇总")
    report_lines.append("=" * 80)

    total_tasks = len(results)
    success_tasks = sum(1 for r in results.values() if r['success'])
    ready_tasks = sum(
        1 for r in results.values()
        if r['success'] and r['readiness']['overall_ready']
    )

    report_lines.append(f"总任务组数: {total_tasks}")
    report_lines.append(f"验证成功: {success_tasks}")
    report_lines.append(f"DiBS就绪: {ready_tasks}")
    report_lines.append(f"有警告: {success_tasks - ready_tasks}")
    report_lines.append(f"验证失败: {total_tasks - success_tasks}")
    report_lines.append("")

    # 各任务组详情
    report_lines.append("=" * 80)
    report_lines.append("2. 各任务组详情")
    report_lines.append("=" * 80)

    for task_name, result in results.items():
        task_display = TASK_GROUPS[task_name]

        report_lines.append(f"\n{task_display} ({task_name}):")

        if result['success']:
            stats = result['stats']
            readiness = result['readiness']

            report_lines.append(f"  ✅ 验证成功")
            report_lines.append(f"  样本数: {stats['num_samples']}")
            report_lines.append(f"  特征数: {len(stats['numeric_cols']) + len(stats['onehot_cols'])}")
            report_lines.append(f"  缺失率: {stats['missing_rate']:.2f}%")

            if readiness['overall_ready']:
                report_lines.append(f"  DiBS适用性: ✅ 优秀")
            elif readiness['sample_size_ok'] and readiness['fill_rate_ok']:
                report_lines.append(f"  DiBS适用性: ⚠️  良好（有警告）")
                for warning in readiness['warnings']:
                    report_lines.append(f"    - {warning}")
            else:
                report_lines.append(f"  DiBS适用性: ❌ 不足")
                for warning in readiness['warnings']:
                    report_lines.append(f"    - {warning}")

            report_lines.append(f"  输出文件: training_data_{task_name}.csv")
            report_lines.append(f"  元数据: metadata_{task_name}.txt")
        else:
            report_lines.append(f"  ❌ 验证失败: {result.get('error', '未知错误')}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("3. DiBS就绪文件清单")
    report_lines.append("=" * 80)

    for task_name, result in results.items():
        if result['success']:
            report_lines.append(f"  - training_data_{task_name}.csv")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("✅ 阶段7完成 - 数据预处理管道全部完成")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("下一步:")
    report_lines.append("  1. 运行DiBS因果图学习: python scripts/experiments/run_dibs_task_specific.py")
    report_lines.append("  2. 查看完整报告: docs/DATA_QUALITY_REPORT_DETAILED_20251223.md")

    # 写入报告文件
    report_file = OUTPUT_DIR / "stage7_final_validation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    # 打印到控制台
    print("\n".join(report_lines))
    print(f"\n📄 报告已保存: {report_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("阶段7: 最终验证 (Final Validation)")
    print("=" * 80)

    results = {}

    # 对每个任务组进行验证
    for task_name, task_display_name in TASK_GROUPS.items():
        try:
            result = validate_task_group(task_name, task_display_name)
            results[task_name] = result
        except Exception as e:
            print(f"\n❌ 错误: 处理任务组 {task_name} 时发生异常")
            print(f"  异常信息: {str(e)}")
            import traceback
            traceback.print_exc()

            results[task_name] = {
                'task_name': task_name,
                'task_display': task_display_name,
                'success': False,
                'error': str(e)
            }

    # 生成最终报告
    generate_final_report(results)

    # 返回状态
    all_success = all(r['success'] for r in results.values())

    if all_success:
        print("\n" + "=" * 80)
        print("✅ 阶段7完成: 所有任务组验证成功")
        print("=" * 80)
        print("\n🎉 数据预处理管道（阶段0-7）全部完成！")
        print("\n下一步: 运行DiBS因果分析")
        print("  cd /home/green/energy_dl/nightly/analysis")
        print("  python scripts/experiments/run_dibs_task_specific.py")
        return 0
    else:
        print("\n" + "=" * 80)
        print("⚠️  阶段7完成: 部分任务组验证失败")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
