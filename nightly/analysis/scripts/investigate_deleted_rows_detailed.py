#!/usr/bin/env python3
"""详细调查被删除行的脚本（修正版）

通过experiment_id追踪被删除的行，分析删除原因和恢复可能性

作者: Claude
日期: 2025-12-24
版本: v2.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_all_data():
    """加载所有数据文件"""
    print("加载数据文件...")

    # 原始数据（726行，来自主项目的data.csv）
    original_path = Path('../data/energy_research/raw/energy_data_original.csv')
    df_original = pd.read_csv(original_path)

    # 分层数据
    processed_dir = Path('../data/energy_research/processed')
    task_files = {
        'image_classification': processed_dir / 'training_data_image_classification.csv',
        'person_reid': processed_dir / 'training_data_person_reid.csv',
        'vulberta': processed_dir / 'training_data_vulberta.csv',
        'bug_localization': processed_dir / 'training_data_bug_localization.csv'
    }

    df_stratified = {}
    for task_name, file_path in task_files.items():
        df_stratified[task_name] = pd.read_csv(file_path)

    print(f"  原始数据: {len(df_original)} 行")
    for task_name, df in df_stratified.items():
        print(f"  {task_name}: {len(df)} 行")

    return df_original, df_stratified


def analyze_deletion_by_experiment_id(df_original, df_stratified):
    """
    通过experiment_id追踪被删除的行

    注意：由于timestamp可能不同，我们需要组合使用
    """
    print("\n" + "=" * 80)
    print("【详细分析】通过实验ID追踪被删除的行")
    print("=" * 80)

    # 合并所有分层数据的experiment_id
    all_stratified_ids = set()
    for task_name, df in df_stratified.items():
        # 使用experiment_id + timestamp作为唯一标识
        task_ids = set(df['experiment_id'] + '|' + df['timestamp'])
        all_stratified_ids.update(task_ids)
        print(f"\n{task_name}: {len(task_ids)} 个唯一实验")

    print(f"\n分层后总唯一实验数: {len(all_stratified_ids)}")

    # 创建原始数据的唯一标识
    original_ids = set(df_original['experiment_id'] + '|' + df_original['timestamp'])
    print(f"原始数据唯一实验数: {len(original_ids)}")

    # 找出被删除的实验
    deleted_ids = original_ids - all_stratified_ids
    print(f"\n被删除的实验数: {len(deleted_ids)} ({len(deleted_ids)/len(original_ids)*100:.1f}%)")

    # 提取被删除行的详细信息
    df_original['composite_id'] = df_original['experiment_id'] + '|' + df_original['timestamp']
    df_deleted = df_original[df_original['composite_id'].isin(deleted_ids)].copy()

    return df_deleted, all_stratified_ids


def analyze_deletion_reasons(df_deleted):
    """分析删除原因的详细分类"""
    print("\n" + "=" * 80)
    print("【删除原因分类】")
    print("=" * 80)

    # 提取性能和能耗列
    perf_cols = [c for c in df_deleted.columns if c.startswith('perf_')]
    energy_cols = [c for c in df_deleted.columns if 'energy_' in c and 'joules' in c]

    print(f"\n检查的性能列数: {len(perf_cols)}")
    print(f"检查的能耗列数: {len(energy_cols)}")

    # 分类统计
    reasons = {
        'both_missing': 0,
        'perf_missing_only': 0,
        'energy_missing_only': 0,
        'task_specific_perf_missing': 0,
        'unknown': 0
    }

    detailed_records = []

    for idx, row in df_deleted.iterrows():
        repo = row.get('repository', 'unknown')
        model = row.get('model', 'unknown')

        # 检查性能和能耗缺失情况
        perf_all_null = row[perf_cols].isnull().all()
        energy_all_null = row[energy_cols].isnull().all()

        reason = ''
        if perf_all_null and energy_all_null:
            reason = 'both_missing'
            reasons['both_missing'] += 1
        elif perf_all_null:
            reason = 'perf_missing_only'
            reasons['perf_missing_only'] += 1
        elif energy_all_null:
            reason = 'energy_missing_only'
            reasons['energy_missing_only'] += 1
        else:
            # 可能是任务特定性能缺失
            reason = 'task_specific_perf_missing'
            reasons['task_specific_perf_missing'] += 1

        detailed_records.append({
            'repository': repo,
            'model': model,
            'experiment_id': row['experiment_id'],
            'reason': reason,
            'perf_null_count': row[perf_cols].isnull().sum(),
            'energy_null_count': row[energy_cols].isnull().sum()
        })

    # 打印统计结果
    print("\n删除原因统计:")
    total = len(df_deleted)
    for reason, count in reasons.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {reason}: {count} 行 ({pct:.1f}%)")

    return reasons, detailed_records


def analyze_task_specific_deletions(df_deleted, df_original):
    """
    分析任务特定的删除情况

    模拟分层预处理的逻辑，看哪些行在分组后被删除
    """
    print("\n" + "=" * 80)
    print("【任务特定删除分析】")
    print("=" * 80)

    # 定义任务组的性能指标（与preprocess_stratified_data.py一致）
    task_perf_cols = {
        'image_classification': {
            'repos': ['examples', 'pytorch_resnet_cifar10'],
            'perf_cols': ['perf_test_accuracy']
        },
        'person_reid': {
            'repos': ['Person_reID_baseline_pytorch'],
            'perf_cols': ['perf_map', 'perf_rank1', 'perf_rank5']
        },
        'vulberta': {
            'repos': ['VulBERTa'],
            'perf_cols': ['perf_eval_loss']
        },
        'bug_localization': {
            'repos': ['bug-localization-by-dnn-and-rvsm'],
            'perf_cols': ['perf_top1_accuracy', 'perf_top5_accuracy']
        }
    }

    task_deletion_stats = {}

    for task_name, config in task_perf_cols.items():
        print(f"\n{task_name}:")

        # 筛选属于该任务的行
        task_mask = df_deleted['repository'].isin(config['repos'])
        df_task_deleted = df_deleted[task_mask]

        if len(df_task_deleted) == 0:
            print(f"  无被删除的行")
            continue

        # 检查性能指标缺失情况
        perf_cols = config['perf_cols']
        perf_all_missing = df_task_deleted[perf_cols].isnull().all(axis=1).sum()
        perf_partial_missing = (df_task_deleted[perf_cols].isnull().any(axis=1) &
                                ~df_task_deleted[perf_cols].isnull().all(axis=1)).sum()

        print(f"  总删除行数: {len(df_task_deleted)}")
        print(f"  性能全缺失: {perf_all_missing} 行")
        print(f"  性能部分缺失: {perf_partial_missing} 行")
        print(f"  性能列: {', '.join(perf_cols)}")

        # 如果有部分缺失的行，可能可以通过放宽策略恢复
        recoverable = perf_partial_missing

        task_deletion_stats[task_name] = {
            'total_deleted': len(df_task_deleted),
            'perf_all_missing': perf_all_missing,
            'perf_partial_missing': perf_partial_missing,
            'potentially_recoverable': recoverable
        }

        if recoverable > 0:
            print(f"  ⚠️ 潜在可恢复: {recoverable} 行（部分性能缺失）")
        else:
            print(f"  ❌ 无可恢复行")

    return task_deletion_stats


def evaluate_recovery_options(task_deletion_stats):
    """评估数据恢复选项"""
    print("\n" + "=" * 80)
    print("【数据恢复选项评估】")
    print("=" * 80)

    total_recoverable = sum(stats['potentially_recoverable'] for stats in task_deletion_stats.values())

    print(f"\n潜在可恢复行数: {total_recoverable}")

    if total_recoverable > 0:
        print("\n恢复方案：")
        print("  方案1: 放宽性能指标要求（只要求主要指标不缺失）")
        print("         适用任务: person_reid, bug_localization（有多个性能指标）")
        print("         预期增加样本: 可能有限（<20行）")
        print("\n  方案2: 保持当前严格标准")
        print("         理由: 确保数据质量，避免引入不完整数据")
        print("         当前372行已足够DiBS分析（>10个/任务组）")
    else:
        print("\n❌ 无潜在可恢复行")
        print("   所有被删除的行都是性能或能耗完全缺失")

    return total_recoverable


def check_deletion_timing():
    """
    检查删除发生在分组前还是分组后

    通过读取extract_from_json_with_defaults.py的输出（如果存在）
    """
    print("\n" + "=" * 80)
    print("【删除时机检查】")
    print("=" * 80)

    # 检查是否有中间文件
    intermediate_files = [
        '../data/energy_research/raw/energy_data_extracted_v2.csv',
        '../data/energy_research/processed/stage0_validated.csv',
        '../data/energy_research/processed/stage2_mediators.csv'
    ]

    found_files = []
    for file_path in intermediate_files:
        p = Path(file_path)
        if p.exists():
            df = pd.read_csv(p)
            found_files.append({
                'file': p.name,
                'rows': len(df),
                'cols': len(df.columns)
            })

    if found_files:
        print("\n发现中间文件:")
        for info in found_files:
            print(f"  {info['file']}: {info['rows']} 行 × {info['cols']} 列")

        print("\n删除时机:")
        print("  阶段1（分组前）: extract_from_json_with_defaults.py")
        print("    - 删除性能全缺失行")
        print("    - 删除能耗全缺失行")
        print("    - 输出: 418行（从726行删除308行）")
        print("\n  阶段2（分组后）: preprocess_stratified_data.py")
        print("    - 按任务分组")
        print("    - 删除任务特定性能全缺失行")
        print("    - 输出: 372行（从418行删除46行）")
    else:
        print("\n未找到中间文件，无法确定删除时机")
        print("但根据代码逻辑，删除发生在:")
        print("  1. 分组前: 删除性能/能耗全缺失行")
        print("  2. 分组后: 删除任务特定性能全缺失行")

    return found_files


def generate_comprehensive_report(df_deleted, reasons, task_deletion_stats, total_recoverable):
    """生成综合报告"""
    print("\n\n" + "=" * 80)
    print("【综合报告】数据删除与恢复可能性")
    print("=" * 80)

    # 转换numpy int64为Python int
    reasons_converted = {k: int(v) for k, v in reasons.items()}
    task_stats_converted = {}
    for task_name, stats in task_deletion_stats.items():
        task_stats_converted[task_name] = {k: int(v) for k, v in stats.items()}

    report = {
        'summary': {
            'total_deleted_rows': int(len(df_deleted)),
            'total_original_rows': 726,
            'total_stratified_rows': 372,
            'deletion_rate': f"{len(df_deleted)/726*100:.1f}%"
        },
        'deletion_reasons': reasons_converted,
        'task_specific_stats': task_stats_converted,
        'recovery_assessment': {
            'potentially_recoverable_rows': int(total_recoverable),
            'recovery_feasibility': 'Low' if total_recoverable < 20 else 'Moderate',
            'recommendation': 'Keep current strict standards' if total_recoverable < 20 else 'Consider relaxing for multi-metric tasks'
        }
    }

    print("\n【1】删除概况:")
    print(f"  原始数据: {report['summary']['total_original_rows']} 行")
    print(f"  分层后数据: {report['summary']['total_stratified_rows']} 行")
    print(f"  被删除: {report['summary']['total_deleted_rows']} 行 ({report['summary']['deletion_rate']})")

    print("\n【2】删除原因分布:")
    total = report['summary']['total_deleted_rows']
    for reason, count in report['deletion_reasons'].items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {reason}: {count} 行 ({pct:.1f}%)")

    print("\n【3】按任务统计:")
    for task_name, stats in report['task_specific_stats'].items():
        print(f"  {task_name}:")
        print(f"    删除: {stats['total_deleted']} 行")
        print(f"    性能全缺失: {stats['perf_all_missing']} 行")
        print(f"    可能可恢复: {stats['potentially_recoverable']} 行")

    print("\n【4】恢复可能性:")
    print(f"  潜在可恢复行数: {report['recovery_assessment']['potentially_recoverable_rows']}")
    print(f"  恢复可行性: {report['recovery_assessment']['recovery_feasibility']}")
    print(f"  建议: {report['recovery_assessment']['recommendation']}")

    print("\n【5】最终建议:")
    if total_recoverable < 20:
        print("  ✅ 保持当前严格标准（推荐）")
        print("     理由:")
        print("     - 可恢复行数极少（<20行）")
        print("     - 当前372行已足够DiBS分析")
        print("     - 保证数据质量优于增加样本量")
    else:
        print("  ⚠️ 可考虑放宽多指标任务的要求")
        print("     理由:")
        print("     - 可恢复行数较多（≥20行）")
        print("     - person_reid和bug_localization有多个性能指标")
        print("     - 只要求主要指标不缺失可能合理")

    # 保存报告
    output_dir = Path('../docs/reports')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'DATA_DELETION_DETAILED_ANALYSIS_20251224.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 详细报告已保存: {output_file}")

    return report


def main():
    """主函数"""
    print("=" * 80)
    print("数据删除详细调查（修正版）")
    print("=" * 80)
    print("目标:")
    print("  1. 通过experiment_id追踪被删除的行")
    print("  2. 分析删除原因（性能缺失、能耗缺失、任务特定）")
    print("  3. 评估恢复可能性")
    print("  4. 确认删除发生在分组前还是分组后")
    print("=" * 80)

    # 加载数据
    df_original, df_stratified = load_all_data()

    # 追踪删除的行
    df_deleted, all_stratified_ids = analyze_deletion_by_experiment_id(df_original, df_stratified)

    # 分析删除原因
    reasons, detailed_records = analyze_deletion_reasons(df_deleted)

    # 任务特定删除分析
    task_deletion_stats = analyze_task_specific_deletions(df_deleted, df_original)

    # 评估恢复选项
    total_recoverable = evaluate_recovery_options(task_deletion_stats)

    # 检查删除时机
    found_files = check_deletion_timing()

    # 生成综合报告
    report = generate_comprehensive_report(df_deleted, reasons, task_deletion_stats, total_recoverable)

    # 保存被删除行的详细记录
    output_dir = Path('../docs/reports')
    df_deleted.to_csv(output_dir / 'deleted_rows_details.csv', index=False)
    print(f"✅ 被删除行详情已保存: {output_dir / 'deleted_rows_details.csv'}")


if __name__ == '__main__':
    main()
