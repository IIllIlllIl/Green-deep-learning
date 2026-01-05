#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析缺少能耗数据的实验

功能:
- 识别所有缺少能耗数据的实验
- 分析这些实验的特征（模型、模式、时间等）
- 检查可能的原因
- 生成详细分析报告
"""

import csv
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

def is_empty(val):
    """检查值是否为空"""
    return val == '' or val is None

def has_energy_data(row, mode):
    """检查实验是否有能耗数据"""
    if mode == 'parallel':
        # 并行模式：检查前台能耗数据
        return not is_empty(row.get('fg_energy_cpu_total_joules'))
    else:
        # 非并行模式：检查能耗数据
        return not is_empty(row.get('energy_cpu_total_joules'))

def main():
    base_dir = Path(__file__).parent.parent
    raw_data_csv = base_dir / "results" / "raw_data.csv"

    print("=" * 80)
    print("🔍 分析缺少能耗数据的实验")
    print("=" * 80)
    print(f"\n数据文件: {raw_data_csv}\n")

    # 读取数据
    with open(raw_data_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"总实验数: {total_rows}\n")

    # ===== 1. 分类实验 =====
    print("=" * 80)
    print("📊 数据完整性分类")
    print("=" * 80)

    experiments_with_energy = []
    experiments_without_energy = []

    for row in rows:
        mode = row.get('mode', '')
        has_energy = has_energy_data(row, mode)

        if has_energy:
            experiments_with_energy.append(row)
        else:
            experiments_without_energy.append(row)

    print(f"\n含能耗数据的实验: {len(experiments_with_energy)} ({len(experiments_with_energy)*100/total_rows:.1f}%)")
    print(f"缺少能耗数据的实验: {len(experiments_without_energy)} ({len(experiments_without_energy)*100/total_rows:.1f}%)")

    # ===== 2. 按模式分析缺失情况 =====
    print("\n" + "=" * 80)
    print("🔬 按训练模式分析")
    print("=" * 80)

    mode_stats = defaultdict(lambda: {'total': 0, 'with_energy': 0, 'without_energy': 0})

    for row in rows:
        mode = row.get('mode', 'unknown')
        has_energy = has_energy_data(row, mode)

        mode_stats[mode]['total'] += 1
        if has_energy:
            mode_stats[mode]['with_energy'] += 1
        else:
            mode_stats[mode]['without_energy'] += 1

    print(f"\n{'模式':<20} {'总数':<10} {'有能耗':<10} {'缺失':<10} {'缺失率':<10}")
    print("-" * 65)
    for mode in sorted(mode_stats.keys()):
        stats = mode_stats[mode]
        total = stats['total']
        with_energy = stats['with_energy']
        without = stats['without_energy']
        missing_rate = (without * 100 / total) if total > 0 else 0

        print(f"{mode:<20} {total:<10} {with_energy:<10} {without:<10} {missing_rate:.1f}%")

    # ===== 3. 按模型分析缺失情况 =====
    print("\n" + "=" * 80)
    print("🧬 按模型分析")
    print("=" * 80)

    model_stats = defaultdict(lambda: {'total': 0, 'with_energy': 0, 'without_energy': 0})

    for row in experiments_without_energy:
        mode = row.get('mode', '')

        if mode == 'parallel':
            repo = row.get('fg_repository', 'unknown')
            model = row.get('fg_model', 'unknown')
        else:
            repo = row.get('repository', 'unknown')
            model = row.get('model', 'unknown')

        if is_empty(repo) or is_empty(model):
            model_key = 'unknown'
        else:
            model_key = f"{repo}/{model}"

        model_stats[model_key]['without_energy'] += 1

    # 统计所有模型的总数
    for row in rows:
        mode = row.get('mode', '')

        if mode == 'parallel':
            repo = row.get('fg_repository', '')
            model = row.get('fg_model', '')
        else:
            repo = row.get('repository', '')
            model = row.get('model', '')

        if is_empty(repo) or is_empty(model):
            model_key = 'unknown'
        else:
            model_key = f"{repo}/{model}"

        model_stats[model_key]['total'] += 1
        if has_energy_data(row, mode):
            model_stats[model_key]['with_energy'] += 1

    print(f"\n{'模型':<50} {'总数':<8} {'有能耗':<8} {'缺失':<8} {'缺失率':<10}")
    print("-" * 90)

    for model in sorted(model_stats.keys(), key=lambda x: model_stats[x]['without_energy'], reverse=True):
        stats = model_stats[model]
        total = stats['total']
        with_energy = stats['with_energy']
        without = stats['without_energy']
        missing_rate = (without * 100 / total) if total > 0 else 0

        if without > 0:  # 只显示有缺失的模型
            print(f"{model:<50} {total:<8} {with_energy:<8} {without:<8} {missing_rate:.1f}%")

    # ===== 4. 按时间分析缺失情况 =====
    print("\n" + "=" * 80)
    print("📅 按时间段分析")
    print("=" * 80)

    time_stats = defaultdict(lambda: {'total': 0, 'with_energy': 0, 'without_energy': 0})

    for row in rows:
        timestamp_str = row.get('timestamp', '')
        if timestamp_str:
            try:
                # 解析时间戳，提取日期
                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                date_key = ts.strftime('%Y-%m-%d')
            except:
                date_key = 'unknown'
        else:
            date_key = 'unknown'

        mode = row.get('mode', '')
        has_energy = has_energy_data(row, mode)

        time_stats[date_key]['total'] += 1
        if has_energy:
            time_stats[date_key]['with_energy'] += 1
        else:
            time_stats[date_key]['without_energy'] += 1

    print(f"\n{'日期':<15} {'总数':<10} {'有能耗':<10} {'缺失':<10} {'缺失率':<10}")
    print("-" * 60)

    for date in sorted(time_stats.keys()):
        stats = time_stats[date]
        total = stats['total']
        with_energy = stats['with_energy']
        without = stats['without_energy']
        missing_rate = (without * 100 / total) if total > 0 else 0

        print(f"{date:<15} {total:<10} {with_energy:<10} {without:<10} {missing_rate:.1f}%")

    # ===== 5. 训练成功率与能耗数据的关系 =====
    print("\n" + "=" * 80)
    print("✅ 训练成功率与能耗数据关系")
    print("=" * 80)

    success_energy_stats = {
        'success_with_energy': 0,
        'success_without_energy': 0,
        'failed_with_energy': 0,
        'failed_without_energy': 0
    }

    for row in rows:
        mode = row.get('mode', '')
        has_energy = has_energy_data(row, mode)

        # 判断训练是否成功
        if mode == 'parallel':
            training_success = row.get('fg_training_success', '') == 'True'
        else:
            training_success = row.get('training_success', '') == 'True'

        if training_success and has_energy:
            success_energy_stats['success_with_energy'] += 1
        elif training_success and not has_energy:
            success_energy_stats['success_without_energy'] += 1
        elif not training_success and has_energy:
            success_energy_stats['failed_with_energy'] += 1
        else:
            success_energy_stats['failed_without_energy'] += 1

    print(f"\n训练成功 + 有能耗数据: {success_energy_stats['success_with_energy']}")
    print(f"训练成功 + 无能耗数据: {success_energy_stats['success_without_energy']}")
    print(f"训练失败 + 有能耗数据: {success_energy_stats['failed_with_energy']}")
    print(f"训练失败 + 无能耗数据: {success_energy_stats['failed_without_energy']}")

    # ===== 6. 详细列出缺失能耗数据的实验 =====
    print("\n" + "=" * 80)
    print("📋 缺失能耗数据的实验详情（前20个）")
    print("=" * 80)

    for i, row in enumerate(experiments_without_energy[:20], 1):
        exp_id = row.get('experiment_id', 'N/A')
        mode = row.get('mode', 'N/A')
        timestamp = row.get('timestamp', 'N/A')

        if mode == 'parallel':
            repo = row.get('fg_repository', 'N/A')
            model = row.get('fg_model', 'N/A')
            training_success = row.get('fg_training_success', 'N/A')
        else:
            repo = row.get('repository', 'N/A')
            model = row.get('model', 'N/A')
            training_success = row.get('training_success', 'N/A')

        print(f"\n{i}. {exp_id}")
        print(f"   模型: {repo}/{model}")
        print(f"   模式: {mode}")
        print(f"   时间: {timestamp}")
        print(f"   训练成功: {training_success}")

    if len(experiments_without_energy) > 20:
        print(f"\n   ... 还有 {len(experiments_without_energy) - 20} 个实验未显示")

    # ===== 7. 分析可能的原因 =====
    print("\n" + "=" * 80)
    print("🔎 可能的原因分析")
    print("=" * 80)

    # 统计不同情况
    parallel_no_energy = sum(1 for row in experiments_without_energy if row.get('mode') == 'parallel')
    non_parallel_no_energy = len(experiments_without_energy) - parallel_no_energy

    # 检查是否有训练失败的
    failed_experiments = []
    for row in experiments_without_energy:
        mode = row.get('mode', '')
        if mode == 'parallel':
            training_success = row.get('fg_training_success', '') == 'True'
        else:
            training_success = row.get('training_success', '') == 'True'

        if not training_success:
            failed_experiments.append(row)

    print(f"\n1. 并行模式实验缺失: {parallel_no_energy} 个")
    print(f"   原因: 并行模式的前台任务可能失败，导致能耗数据未记录")

    print(f"\n2. 非并行模式实验缺失: {non_parallel_no_energy} 个")
    print(f"   可能原因:")
    print(f"   - perf 权限问题导致CPU能耗无法监控")
    print(f"   - nvidia-smi 不可用导致GPU能耗无法监控")
    print(f"   - 能耗监控脚本执行失败")

    print(f"\n3. 训练失败的实验: {len(failed_experiments)} 个")
    print(f"   这些实验训练失败，可能没有记录完整的能耗数据")

    # ===== 8. 总结与建议 =====
    print("\n" + "=" * 80)
    print("📈 总结与建议")
    print("=" * 80)

    print(f"\n数据完整性现状:")
    print(f"  - 总实验数: {total_rows}")
    print(f"  - 有能耗数据: {len(experiments_with_energy)} ({len(experiments_with_energy)*100/total_rows:.1f}%)")
    print(f"  - 缺少能耗数据: {len(experiments_without_energy)} ({len(experiments_without_energy)*100/total_rows:.1f}%)")

    print(f"\n主要缺失来源:")
    print(f"  - 并行模式: {parallel_no_energy} 个 ({parallel_no_energy*100/len(experiments_without_energy):.1f}% of missing)")
    print(f"  - 非并行模式: {non_parallel_no_energy} 个 ({non_parallel_no_energy*100/len(experiments_without_energy):.1f}% of missing)")

    print(f"\n建议:")
    print(f"  1. 检查并行模式实验的前台任务日志，找出失败原因")
    print(f"  2. 验证能耗监控工具的权限（sudo perf, nvidia-smi）")
    print(f"  3. 检查能耗监控脚本是否正常执行")
    print(f"  4. 考虑重新运行缺失能耗数据的重要实验")

    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
