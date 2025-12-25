#!/usr/bin/env python3
"""数据质量验证脚本

用途: 验证分层数据满足DiBS因果分析的前提条件
作者: Claude
日期: 2025-12-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys


# 任务配置
TASK_CONFIGS = {
    'image_classification': {
        'file': 'training_data_image_classification.csv',
        'expected_rows': (100, 120),  # (min, max)
        'expected_cols': 16,
        'onehot_cols': ['is_mnist', 'is_cifar10'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate', 'seed'],
        'energy_cols': ['energy_cpu_total_joules', 'energy_gpu_total_joules',
                        'gpu_power_avg_watts'],
        'performance_cols': ['perf_test_accuracy'],
        'complete_row_target': 0.90  # >90%完全无缺失行
    },
    'person_reid': {
        'file': 'training_data_person_reid.csv',
        'expected_rows': (60, 75),
        'expected_cols': 20,
        'onehot_cols': ['is_densenet121', 'is_hrnet18', 'is_pcb'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'hyperparam_dropout', 'seed'],
        'energy_cols': ['energy_cpu_total_joules', 'energy_gpu_total_joules',
                        'gpu_power_avg_watts'],
        'performance_cols': ['perf_map', 'perf_rank1', 'perf_rank5'],
        'complete_row_target': 0.90
    },
    'vulberta': {
        'file': 'training_data_vulberta.csv',
        'expected_rows': (85, 105),
        'expected_cols': 15,
        'onehot_cols': [],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'l2_regularization', 'seed'],
        'energy_cols': ['energy_cpu_total_joules', 'energy_gpu_total_joules',
                        'gpu_power_avg_watts'],
        'performance_cols': ['perf_eval_loss'],
        'complete_row_target': 0.80  # >80%
    },
    'bug_localization': {
        'file': 'training_data_bug_localization.csv',
        'expected_rows': (80, 100),
        'expected_cols': 16,
        'onehot_cols': [],
        'hyperparams': ['training_duration', 'l2_regularization',
                        'hyperparam_kfold', 'seed'],
        'energy_cols': ['energy_cpu_total_joules', 'energy_gpu_total_joules',
                        'gpu_power_avg_watts'],
        'performance_cols': ['perf_top1_accuracy', 'perf_top5_accuracy'],
        'complete_row_target': 0.80
    }
}


def check_missing_rates(df, task_name, config):
    """检查缺失率

    目标: 超参数/能耗/性能应0%缺失
    """
    print(f"\n1. 缺失率检查:")

    critical_cols = (
        config['hyperparams'] +
        config['energy_cols'] +
        config['performance_cols']
    )

    all_pass = True
    for col in critical_cols:
        if col not in df.columns:
            print(f"  ❌ {col}: 列不存在")
            all_pass = False
            continue

        missing_count = df[col].isna().sum()
        missing_rate = missing_count / len(df) * 100

        if col in config['hyperparams']:
            # 超参数应0%缺失
            if missing_rate > 0:
                print(f"  ❌ {col}: {missing_rate:.2f}%缺失（超参数应0%缺失）")
                all_pass = False
            else:
                print(f"  ✅ {col}: 0%缺失")

        elif col in config['energy_cols']:
            # 能耗应0%缺失
            if missing_rate > 0:
                print(f"  ❌ {col}: {missing_rate:.2f}%缺失（能耗应0%缺失）")
                all_pass = False
            else:
                print(f"  ✅ {col}: 0%缺失")

        elif col in config['performance_cols']:
            # 性能应0%缺失
            if missing_rate > 0:
                print(f"  ❌ {col}: {missing_rate:.2f}%缺失（性能应0%缺失）")
                all_pass = False
            else:
                print(f"  ✅ {col}: 0%缺失")

    return all_pass


def check_complete_rows(df, task_name, config):
    """检查完全无缺失行比例"""
    print(f"\n2. 完全无缺失行检查:")

    complete_rows = df.dropna()
    complete_rate = len(complete_rows) / len(df)

    print(f"  完全无缺失行: {len(complete_rows)}/{len(df)} ({complete_rate*100:.1f}%)")
    print(f"  目标: >{config['complete_row_target']*100:.0f}%")

    if complete_rate >= config['complete_row_target']:
        print(f"  ✅ 达到目标")
        return True
    else:
        print(f"  ❌ 未达目标（差{(config['complete_row_target']-complete_rate)*100:.1f}%）")
        return False


def check_correlation_matrix(df, task_name, config):
    """检查相关矩阵可计算性"""
    print(f"\n3. 相关矩阵检查:")

    # 移除非数值列
    numeric_df = df.select_dtypes(include=[np.number])

    # 计算相关矩阵
    try:
        corr = numeric_df.corr()
    except Exception as e:
        print(f"  ❌ 相关矩阵计算失败: {e}")
        return False

    # 验证无nan值
    if corr.isna().any().any():
        nan_count = corr.isna().sum().sum()
        print(f"  ❌ 相关矩阵包含 {nan_count} 个nan值")
        return False

    # 验证对角线为1
    diag = np.diag(corr)
    if not np.allclose(diag, 1.0):
        print(f"  ❌ 对角线不全为1")
        return False

    # 验证范围[-1, 1]
    if (corr < -1).any().any() or (corr > 1).any().any():
        print(f"  ❌ 相关系数超出[-1, 1]范围")
        return False

    print(f"  ✅ 相关矩阵可计算: {corr.shape}")
    print(f"  ✅ 无nan值")
    print(f"  ✅ 对角线全为1")
    print(f"  ✅ 范围在[-1, 1]内")

    return True


def check_numeric_ranges(df, task_name, config):
    """检查数值范围合理性"""
    print(f"\n4. 数值范围检查:")

    ranges = {
        'energy_cpu_total_joules': (0, 1e9, 'J'),
        'energy_gpu_total_joules': (0, 1e9, 'J'),
        'gpu_power_avg_watts': (0, 600, 'W'),
        'gpu_util_avg': (0, 100, '%'),
        'gpu_temp_max': (20, 110, '°C'),
    }

    all_pass = True
    for col, (min_val, max_val, unit) in ranges.items():
        if col not in df.columns:
            continue

        out_of_range = df[(df[col] < min_val) | (df[col] > max_val)][col].dropna()

        if len(out_of_range) > 0:
            print(f"  ❌ {col}: {len(out_of_range)} 个值超出范围 [{min_val}, {max_val}]{unit}")
            print(f"     异常值: {list(out_of_range.values[:3])}")
            all_pass = False
        else:
            print(f"  ✅ {col}: 范围合理 [{min_val}, {max_val}]{unit}")

    return all_pass


def check_onehot_encoding(df, task_name, config):
    """检查One-Hot编码正确性"""
    if not config['onehot_cols']:
        print(f"\n5. One-Hot编码检查:")
        print(f"  ⏭️  该任务组无One-Hot编码")
        return True

    print(f"\n5. One-Hot编码检查:")
    print(f"  One-Hot列: {', '.join(config['onehot_cols'])}")

    # 检查列存在
    missing_cols = set(config['onehot_cols']) - set(df.columns)
    if missing_cols:
        print(f"  ❌ 缺少列: {missing_cols}")
        return False

    # 检查互斥性（每行和=1）
    onehot_sum = df[config['onehot_cols']].sum(axis=1)
    if not (onehot_sum == 1).all():
        invalid_count = (onehot_sum != 1).sum()
        print(f"  ❌ {invalid_count} 行违反互斥性（和≠1）")
        return False

    print(f"  ✅ 互斥性验证通过（所有行和=1）")

    # 检查覆盖率
    for col in config['onehot_cols']:
        col_sum = df[col].sum()
        print(f"  ✅ {col}: {col_sum} 个样本")

    return True


def verify_task_data(data_dir, task_name):
    """验证单个任务组的数据质量

    Args:
        data_dir: 数据目录
        task_name: 任务名称

    Returns:
        dict: 验证结果
    """
    config = TASK_CONFIGS[task_name]
    data_dir = Path(data_dir)

    print(f"\n{'=' * 80}")
    print(f"验证任务组: {task_name}")
    print(f"{'=' * 80}")

    # 加载数据
    file_path = data_dir / config['file']
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return {'task': task_name, 'passed': False, 'error': 'File not found'}

    df = pd.read_csv(file_path)

    print(f"\n基本信息:")
    print(f"  文件: {config['file']}")
    print(f"  行数: {len(df)}")
    print(f"  列数: {len(df.columns)}")
    print(f"  预期行数: {config['expected_rows'][0]}-{config['expected_rows'][1]}")
    print(f"  预期列数: {config['expected_cols']}")

    # 检查行数和列数
    row_in_range = config['expected_rows'][0] <= len(df) <= config['expected_rows'][1]
    col_match = len(df.columns) == config['expected_cols']

    if not row_in_range:
        print(f"  ⚠️ 行数不在预期范围")
    if not col_match:
        print(f"  ❌ 列数不匹配（预期{config['expected_cols']}，实际{len(df.columns)}）")

    # 执行各项检查
    checks = {
        'missing_rates': check_missing_rates(df, task_name, config),
        'complete_rows': check_complete_rows(df, task_name, config),
        'correlation_matrix': check_correlation_matrix(df, task_name, config),
        'numeric_ranges': check_numeric_ranges(df, task_name, config),
        'onehot_encoding': check_onehot_encoding(df, task_name, config)
    }

    # 汇总结果
    all_passed = all(checks.values()) and row_in_range and col_match

    result = {
        'task': task_name,
        'file': str(file_path),
        'rows': len(df),
        'columns': len(df.columns),
        'row_in_range': row_in_range,
        'col_match': col_match,
        'checks': checks,
        'passed': all_passed
    }

    return result


def generate_quality_report(results, output_file):
    """生成质量验证报告"""
    output_file = Path(output_file)

    # 生成Markdown报告
    report = []
    report.append("# 分层数据质量验证报告\n")
    report.append(f"**日期**: 2025-12-24\n")
    report.append(f"**验证任务**: {len(results)} 个任务组\n\n")
    report.append("---\n\n")

    # 验证摘要
    report.append("## 验证摘要\n\n")
    report.append("| 任务组 | 样本数 | 列数 | 缺失率检查 | 完全行检查 | 相关矩阵 | 数值范围 | One-Hot | 总体 |\n")
    report.append("|--------|--------|------|-----------|-----------|----------|----------|---------|------|\n")

    for result in results:
        task = result['task']
        rows = result['rows']
        cols = result['columns']
        checks = result['checks']

        status_icons = {
            'missing_rates': '✅' if checks['missing_rates'] else '❌',
            'complete_rows': '✅' if checks['complete_rows'] else '❌',
            'correlation_matrix': '✅' if checks['correlation_matrix'] else '❌',
            'numeric_ranges': '✅' if checks['numeric_ranges'] else '❌',
            'onehot_encoding': '✅' if checks['onehot_encoding'] else '❌',
            'overall': '✅' if result['passed'] else '❌'
        }

        report.append(f"| {task} | {rows} | {cols} | "
                      f"{status_icons['missing_rates']} | "
                      f"{status_icons['complete_rows']} | "
                      f"{status_icons['correlation_matrix']} | "
                      f"{status_icons['numeric_ranges']} | "
                      f"{status_icons['onehot_encoding']} | "
                      f"{status_icons['overall']} |\n")

    report.append("\n---\n\n")

    # 详细检查结果
    report.append("## 详细检查结果\n\n")

    all_passed = all(r['passed'] for r in results)

    if all_passed:
        report.append("### ✅ 所有检查通过！\n\n")
        report.append("- 超参数列：0%缺失 ✅\n")
        report.append("- 能耗列：0%缺失 ✅\n")
        report.append("- 性能列：0%缺失 ✅\n")
        report.append("- 相关矩阵可计算 ✅\n")
        report.append("- One-Hot编码正确 ✅\n")
        report.append("- 数值范围合理 ✅\n\n")
    else:
        report.append("### ⚠️ 部分检查未通过\n\n")
        for result in results:
            if not result['passed']:
                report.append(f"**{result['task']}**:\n")
                for check_name, passed in result['checks'].items():
                    if not passed:
                        report.append(f"- ❌ {check_name} 未通过\n")
                report.append("\n")

    # 结论
    report.append("---\n\n")
    report.append("## 结论\n\n")

    if all_passed:
        report.append("✅ **数据质量验证通过，可以进入阶段5（DiBS因果分析）**\n\n")
        report.append("所有任务组的数据满足DiBS因果分析的前提条件：\n")
        report.append("- 关键列无缺失值\n")
        report.append("- 相关矩阵可正常计算\n")
        report.append("- 数值范围合理\n")
        report.append("- One-Hot编码正确\n")
    else:
        report.append("⚠️ **数据质量验证未完全通过，建议修复后再进入DiBS分析**\n\n")
        report.append("请检查上述未通过的项目，并进行相应修复。\n")

    # 保存报告
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)

    return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='数据质量验证脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/energy_research/processed',
        help='数据目录（默认: ../data/energy_research/processed）'
    )

    parser.add_argument(
        '--output-report',
        type=str,
        default='../docs/reports/STRATIFIED_DATA_QUALITY_REPORT.md',
        help='输出报告路径（默认: ../docs/reports/STRATIFIED_DATA_QUALITY_REPORT.md）'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        choices=list(TASK_CONFIGS.keys()),
        default=list(TASK_CONFIGS.keys()),
        help='要验证的任务组（默认: 全部）'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("数据质量验证脚本")
    print("=" * 80)
    print(f"数据目录: {args.data_dir}")
    print(f"输出报告: {args.output_report}")
    print(f"验证任务: {', '.join(args.tasks)}")
    print("=" * 80)

    # 验证每个任务组
    results = []
    for task_name in args.tasks:
        try:
            result = verify_task_data(args.data_dir, task_name)
            results.append(result)
        except Exception as e:
            print(f"\n❌ 验证任务组 {task_name} 失败:")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'task': task_name,
                'passed': False,
                'error': str(e)
            })

    # 生成报告
    if results:
        report_path = generate_quality_report(results, args.output_report)

        print(f"\n{'=' * 80}")
        print("验证总结")
        print(f"{'=' * 80}")

        passed_count = sum(1 for r in results if r['passed'])
        total_count = len(results)

        print(f"通过: {passed_count}/{total_count}")

        if passed_count == total_count:
            print("\n✅ 所有任务组验证通过！")
            print(f"✅ 报告已生成: {report_path}")
            print("\n可以进入阶段5（DiBS因果分析）")
        else:
            print(f"\n⚠️ {total_count - passed_count} 个任务组未通过验证")
            print(f"⚠️ 报告已生成: {report_path}")
            print("\n建议修复未通过项后再进入DiBS分析")

        print(f"{'=' * 80}")

        sys.exit(0 if passed_count == total_count else 1)
    else:
        print("\n❌ 没有成功验证任何任务组")
        sys.exit(1)


if __name__ == '__main__':
    main()
