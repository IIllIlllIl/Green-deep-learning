#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据可用性分析

分析哪些数据记录是可用的，哪些是不可用的，以及不可用的原因
"""

import csv
from collections import defaultdict, Counter

def is_empty(val):
    """检查值是否为空"""
    return val == '' or val is None

def has_basic_info(row):
    """检查是否有基本信息"""
    required_fields = ['experiment_id', 'timestamp']
    for field in required_fields:
        if is_empty(row.get(field)):
            return False
    return True

def get_mode(row):
    """获取实验模式"""
    mode = row.get('mode', '')
    if is_empty(mode):
        # 如果mode字段为空，通过fg_repository判断
        if not is_empty(row.get('fg_repository')):
            return 'parallel'
        else:
            return 'non-parallel'
    return mode

def is_training_success(row, mode):
    """检查训练是否成功"""
    if mode == 'parallel':
        success = row.get('fg_training_success', '')
    else:
        success = row.get('training_success', '')
    return success == 'True'

def has_energy_data(row, mode):
    """检查是否有能耗数据"""
    if mode == 'parallel':
        # 并行模式：检查前台能耗数据
        key_field = 'fg_energy_cpu_total_joules'
    else:
        # 非并行模式：检查能耗数据
        key_field = 'energy_cpu_total_joules'

    return not is_empty(row.get(key_field))

def has_performance_data(row, mode):
    """检查是否有性能指标数据"""
    if mode == 'parallel':
        perf_fields = [
            'fg_perf_accuracy', 'fg_perf_test_accuracy', 'fg_perf_map',
            'fg_perf_precision', 'fg_perf_recall', 'fg_perf_best_val_accuracy'
        ]
    else:
        perf_fields = [
            'perf_accuracy', 'perf_test_accuracy', 'perf_map',
            'perf_precision', 'perf_recall', 'perf_best_val_accuracy',
            'perf_top1_accuracy', 'perf_top5_accuracy'
        ]

    # 至少有一个性能指标
    for field in perf_fields:
        if not is_empty(row.get(field)):
            return True
    return False

def get_model_info(row, mode):
    """获取模型信息"""
    if mode == 'parallel':
        repo = row.get('fg_repository', 'N/A')
        model = row.get('fg_model', 'N/A')
    else:
        repo = row.get('repository', 'N/A')
        model = row.get('model', 'N/A')
    return repo, model

def analyze_usability(row):
    """
    分析单条记录的可用性

    返回: (is_usable, reasons)
    """
    reasons = []

    # 1. 检查基本信息
    if not has_basic_info(row):
        reasons.append('基本信息缺失')
        return False, reasons

    # 2. 获取模式
    mode = get_mode(row)

    # 3. 检查训练是否成功
    if not is_training_success(row, mode):
        reasons.append('训练失败')

    # 4. 检查能耗数据
    if not has_energy_data(row, mode):
        reasons.append('能耗数据缺失')

    # 5. 检查性能指标
    if not has_performance_data(row, mode):
        reasons.append('性能指标缺失')

    # 如果有任何问题，则不可用
    is_usable = len(reasons) == 0

    return is_usable, reasons

def main():
    data_file = "data/raw_data.csv"

    print("=" * 100)
    print("🔍 数据可用性分析")
    print("=" * 100)
    print(f"\n数据文件: {data_file}\n")

    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"总记录数: {total_rows}\n")

    # ===== 1. 分析每条记录的可用性 =====
    print("=" * 100)
    print("📊 可用性统计")
    print("=" * 100)

    usable_records = []
    unusable_records = []

    for idx, row in enumerate(rows):
        is_usable, reasons = analyze_usability(row)

        record_info = {
            'index': idx,
            'experiment_id': row.get('experiment_id', 'N/A'),
            'mode': get_mode(row),
            'reasons': reasons
        }

        if is_usable:
            usable_records.append(record_info)
        else:
            unusable_records.append(record_info)

    print(f"\n✅ 可用记录: {len(usable_records)} ({len(usable_records)*100/total_rows:.1f}%)")
    print(f"❌ 不可用记录: {len(unusable_records)} ({len(unusable_records)*100/total_rows:.1f}%)")

    # ===== 2. 分析不可用的原因 =====
    print("\n" + "=" * 100)
    print("🔬 不可用原因统计")
    print("=" * 100)

    # 统计每种原因的出现次数
    reason_counter = Counter()
    reason_combinations = Counter()

    for record in unusable_records:
        for reason in record['reasons']:
            reason_counter[reason] += 1

        # 统计原因组合
        reason_combo = tuple(sorted(record['reasons']))
        reason_combinations[reason_combo] += 1

    print(f"\n{'不可用原因':<30} {'记录数':<10} {'占总数比例':<15} {'占不可用比例':<15}")
    print("-" * 80)

    for reason, count in reason_counter.most_common():
        pct_total = count * 100 / total_rows
        pct_unusable = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
        print(f"{reason:<30} {count:<10} {pct_total:>12.1f}% {pct_unusable:>14.1f}%")

    # ===== 3. 原因组合分析 =====
    print("\n" + "=" * 100)
    print("🧩 不可用原因组合（Top 10）")
    print("=" * 100)

    print(f"\n共发现 {len(reason_combinations)} 种不同的原因组合\n")

    for i, (combo, count) in enumerate(reason_combinations.most_common(10), 1):
        pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
        print(f"\n组合 {i}: {count} 条记录 ({pct:.1f}% of unusable)")
        print(f"原因:")
        for reason in combo:
            print(f"  - {reason}")

    # ===== 4. 按模式分析可用性 =====
    print("\n" + "=" * 100)
    print("📂 按模式分析可用性")
    print("=" * 100)

    mode_stats = defaultdict(lambda: {'total': 0, 'usable': 0, 'unusable': 0})

    for row in rows:
        mode = get_mode(row)
        is_usable, _ = analyze_usability(row)

        mode_stats[mode]['total'] += 1
        if is_usable:
            mode_stats[mode]['usable'] += 1
        else:
            mode_stats[mode]['unusable'] += 1

    print(f"\n{'模式':<20} {'总数':<10} {'可用':<10} {'不可用':<10} {'可用率':<10}")
    print("-" * 65)

    for mode in sorted(mode_stats.keys()):
        stats = mode_stats[mode]
        total = stats['total']
        usable = stats['usable']
        unusable = stats['unusable']
        usable_rate = (usable * 100 / total) if total > 0 else 0

        print(f"{mode:<20} {total:<10} {usable:<10} {unusable:<10} {usable_rate:.1f}%")

    # ===== 5. 按模型分析可用性 =====
    print("\n" + "=" * 100)
    print("🧬 按模型分析可用性")
    print("=" * 100)

    model_stats = defaultdict(lambda: {'total': 0, 'usable': 0, 'unusable': 0})

    for row in rows:
        mode = get_mode(row)
        repo, model = get_model_info(row, mode)

        if is_empty(repo) or is_empty(model):
            model_key = 'unknown'
        else:
            model_key = f"{repo}/{model}"

        is_usable, _ = analyze_usability(row)

        model_stats[model_key]['total'] += 1
        if is_usable:
            model_stats[model_key]['usable'] += 1
        else:
            model_stats[model_key]['unusable'] += 1

    print(f"\n{'模型':<50} {'总数':<8} {'可用':<8} {'不可用':<8} {'可用率':<10}")
    print("-" * 90)

    # 按可用率排序
    sorted_models = sorted(model_stats.items(),
                          key=lambda x: x[1]['usable'] / x[1]['total'] if x[1]['total'] > 0 else 0,
                          reverse=True)

    for model_key, stats in sorted_models:
        total = stats['total']
        usable = stats['usable']
        unusable = stats['unusable']
        usable_rate = (usable * 100 / total) if total > 0 else 0

        print(f"{model_key:<50} {total:<8} {usable:<8} {unusable:<8} {usable_rate:.1f}%")

    # ===== 6. 详细列出不可用记录（示例） =====
    print("\n" + "=" * 100)
    print("📋 不可用记录示例（前20个）")
    print("=" * 100)

    for i, record in enumerate(unusable_records[:20], 1):
        row = rows[record['index']]
        mode = record['mode']
        repo, model = get_model_info(row, mode)

        print(f"\n{i}. {record['experiment_id']}")
        print(f"   模型: {repo}/{model}")
        print(f"   模式: {mode}")
        print(f"   时间: {row.get('timestamp', 'N/A')}")
        print(f"   不可用原因: {', '.join(record['reasons'])}")

    if len(unusable_records) > 20:
        print(f"\n   ... 还有 {len(unusable_records) - 20} 个不可用记录未显示")

    # ===== 7. 分析训练失败的记录 =====
    print("\n" + "=" * 100)
    print("⚠️  训练失败记录详细分析")
    print("=" * 100)

    training_failed = [r for r in unusable_records if '训练失败' in r['reasons']]

    print(f"\n训练失败的记录数: {len(training_failed)}")

    if len(training_failed) > 0:
        # 按模型统计训练失败
        failed_by_model = defaultdict(int)
        for record in training_failed:
            row = rows[record['index']]
            mode = record['mode']
            repo, model = get_model_info(row, mode)
            model_key = f"{repo}/{model}"
            failed_by_model[model_key] += 1

        print(f"\n{'模型':<50} {'失败次数':<10}")
        print("-" * 65)

        for model_key, count in sorted(failed_by_model.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_key:<50} {count:<10}")

    # ===== 8. 总结 =====
    print("\n" + "=" * 100)
    print("📊 总结")
    print("=" * 100)

    print(f"\n数据可用性:")
    print(f"  - 总记录数: {total_rows}")
    print(f"  - ✅ 可用记录: {len(usable_records)} ({len(usable_records)*100/total_rows:.1f}%)")
    print(f"  - ❌ 不可用记录: {len(unusable_records)} ({len(unusable_records)*100/total_rows:.1f}%)")

    print(f"\n主要不可用原因:")
    for reason, count in reason_counter.most_common(3):
        pct = count * 100 / total_rows
        print(f"  - {reason}: {count} 条 ({pct:.1f}%)")

    print(f"\n最常见的不可用原因组合:")
    if len(reason_combinations) > 0:
        top_combo, top_count = reason_combinations.most_common(1)[0]
        print(f"  - 原因: {', '.join(top_combo)}")
        print(f"  - 记录数: {top_count}")

    print(f"\n建议:")
    print(f"  1. 优先修复能耗数据缺失问题（如果是主要原因）")
    print(f"  2. 分析训练失败的原因，提高训练成功率")
    print(f"  3. 检查性能指标缺失的模式，确保关键指标被记录")
    print(f"  4. 可用记录已足够进行数据分析（如果可用率>80%）")

    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
