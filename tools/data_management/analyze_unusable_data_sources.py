#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不可用数据来源分布分析

分析不可用数据的来源分布，包括：
1. 时间分布：哪些时间段产生的不可用数据最多
2. 实验批次分布：哪些实验批次产生的不可用数据最多
3. 模型-时间交叉分析：特定模型在特定时间的不可用数据分布
"""

import csv
from collections import defaultdict, Counter
from datetime import datetime
import re

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

def extract_batch_prefix(experiment_id):
    """
    从experiment_id提取批次前缀

    例如:
    - default__VulBERTa_mlp_004 -> default__
    - mutation_1x__VulBERTa_mlp_043 -> mutation_1x__
    - mutation_2x_safe__MRT-OAST_default_065 -> mutation_2x_safe__
    """
    match = re.match(r'^([a-zA-Z0-9_]+)__', experiment_id)
    if match:
        return match.group(1) + '__'
    else:
        return 'unknown__'

def parse_timestamp(timestamp_str):
    """
    解析时间戳字符串

    返回: datetime对象
    """
    try:
        # 尝试解析ISO格式: 2025-11-18T20:53:53.350873
        return datetime.fromisoformat(timestamp_str)
    except:
        return None

def get_date(dt):
    """获取日期字符串 (YYYY-MM-DD)"""
    if dt:
        return dt.strftime('%Y-%m-%d')
    return 'unknown'

def get_week(dt):
    """获取周字符串 (YYYY-Www)"""
    if dt:
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"
    return 'unknown'

def get_month(dt):
    """获取月份字符串 (YYYY-MM)"""
    if dt:
        return dt.strftime('%Y-%m')
    return 'unknown'

def main():
    data_file = "data/raw_data.csv"

    print("=" * 100)
    print("🔍 不可用数据来源分布分析")
    print("=" * 100)
    print(f"\n数据文件: {data_file}\n")

    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"总记录数: {total_rows}\n")

    # ===== 1. 识别不可用记录 =====
    print("=" * 100)
    print("📊 识别不可用记录")
    print("=" * 100)

    unusable_records = []

    for idx, row in enumerate(rows):
        is_usable, reasons = analyze_usability(row)

        if not is_usable:
            mode = get_mode(row)
            repo, model = get_model_info(row, mode)
            timestamp_str = row.get('timestamp', '')
            dt = parse_timestamp(timestamp_str)

            unusable_records.append({
                'index': idx,
                'experiment_id': row.get('experiment_id', 'N/A'),
                'timestamp': timestamp_str,
                'datetime': dt,
                'date': get_date(dt),
                'week': get_week(dt),
                'month': get_month(dt),
                'batch': extract_batch_prefix(row.get('experiment_id', '')),
                'model': f"{repo}/{model}",
                'mode': mode,
                'reasons': reasons
            })

    print(f"\n不可用记录数: {len(unusable_records)} ({len(unusable_records)*100/total_rows:.1f}%)\n")

    # ===== 2. 时间分布分析 =====
    print("=" * 100)
    print("📅 时间分布分析")
    print("=" * 100)

    # 2.1 按日期统计
    print("\n【按日期统计】")
    date_counter = Counter([r['date'] for r in unusable_records])

    print(f"\n{'日期':<15} {'不可用记录数':<15} {'占不可用总数':<15}")
    print("-" * 50)

    for date, count in sorted(date_counter.items()):
        pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
        print(f"{date:<15} {count:<15} {pct:>12.1f}%")

    # 2.2 按周统计
    print("\n【按周统计】")
    week_counter = Counter([r['week'] for r in unusable_records])

    print(f"\n{'周':<15} {'不可用记录数':<15} {'占不可用总数':<15}")
    print("-" * 50)

    for week, count in sorted(week_counter.items()):
        pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
        print(f"{week:<15} {count:<15} {pct:>12.1f}%")

    # 2.3 按月统计
    print("\n【按月统计】")
    month_counter = Counter([r['month'] for r in unusable_records])

    print(f"\n{'月份':<15} {'不可用记录数':<15} {'占不可用总数':<15}")
    print("-" * 50)

    for month, count in sorted(month_counter.items()):
        pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
        print(f"{month:<15} {count:<15} {pct:>12.1f}%")

    # ===== 3. 实验批次分布分析 =====
    print("\n" + "=" * 100)
    print("🧪 实验批次分布分析")
    print("=" * 100)

    batch_counter = Counter([r['batch'] for r in unusable_records])

    print(f"\n{'实验批次':<25} {'不可用记录数':<15} {'占不可用总数':<15}")
    print("-" * 60)

    for batch, count in sorted(batch_counter.items(), key=lambda x: x[1], reverse=True):
        pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
        print(f"{batch:<25} {count:<15} {pct:>12.1f}%")

    # ===== 4. 模型分布分析 =====
    print("\n" + "=" * 100)
    print("🧬 模型分布分析")
    print("=" * 100)

    model_counter = Counter([r['model'] for r in unusable_records])

    print(f"\n{'模型':<50} {'不可用记录数':<15} {'占不可用总数':<15}")
    print("-" * 80)

    for model, count in sorted(model_counter.items(), key=lambda x: x[1], reverse=True):
        pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
        print(f"{model:<50} {count:<15} {pct:>12.1f}%")

    # ===== 5. 模式分布分析 =====
    print("\n" + "=" * 100)
    print("📂 模式分布分析")
    print("=" * 100)

    mode_counter = Counter([r['mode'] for r in unusable_records])

    print(f"\n{'模式':<20} {'不可用记录数':<15} {'占不可用总数':<15}")
    print("-" * 55)

    for mode, count in sorted(mode_counter.items()):
        pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
        print(f"{mode:<20} {count:<15} {pct:>12.1f}%")

    # ===== 6. 批次-模型交叉分析 =====
    print("\n" + "=" * 100)
    print("🔬 批次-模型交叉分析")
    print("=" * 100)

    batch_model_counter = defaultdict(lambda: defaultdict(int))

    for record in unusable_records:
        batch_model_counter[record['batch']][record['model']] += 1

    for batch in sorted(batch_model_counter.keys()):
        print(f"\n【批次: {batch}】")
        print(f"{'模型':<50} {'不可用记录数':<15}")
        print("-" * 70)

        for model, count in sorted(batch_model_counter[batch].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"{model:<50} {count:<15}")

    # ===== 7. 时间-模型交叉分析 (按月) =====
    print("\n" + "=" * 100)
    print("📊 时间-模型交叉分析 (按月)")
    print("=" * 100)

    month_model_counter = defaultdict(lambda: defaultdict(int))

    for record in unusable_records:
        month_model_counter[record['month']][record['model']] += 1

    for month in sorted(month_model_counter.keys()):
        print(f"\n【月份: {month}】")
        print(f"{'模型':<50} {'不可用记录数':<15}")
        print("-" * 70)

        for model, count in sorted(month_model_counter[month].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"{model:<50} {count:<15}")

    # ===== 8. 关键发现和总结 =====
    print("\n" + "=" * 100)
    print("💡 关键发现和总结")
    print("=" * 100)

    # 找出最多不可用数据的批次
    top_batch = batch_counter.most_common(1)[0] if len(batch_counter) > 0 else ('N/A', 0)

    # 找出最多不可用数据的模型
    top_model = model_counter.most_common(1)[0] if len(model_counter) > 0 else ('N/A', 0)

    # 找出最多不可用数据的日期
    top_date = date_counter.most_common(1)[0] if len(date_counter) > 0 else ('N/A', 0)

    # 找出最多不可用数据的月份
    top_month = month_counter.most_common(1)[0] if len(month_counter) > 0 else ('N/A', 0)

    print(f"\n1. 时间分布特征:")
    print(f"   - 时间跨度: {min([r['date'] for r in unusable_records if r['date'] != 'unknown'])} 至 {max([r['date'] for r in unusable_records if r['date'] != 'unknown'])}")
    print(f"   - 不可用数据最多的日期: {top_date[0]} ({top_date[1]}条, {top_date[1]*100/len(unusable_records):.1f}%)")
    print(f"   - 不可用数据最多的月份: {top_month[0]} ({top_month[1]}条, {top_month[1]*100/len(unusable_records):.1f}%)")
    print(f"   - 涉及的日期总数: {len(date_counter)} 天")
    print(f"   - 涉及的周数: {len(week_counter)} 周")

    print(f"\n2. 实验批次特征:")
    print(f"   - 不可用数据最多的批次: {top_batch[0]} ({top_batch[1]}条, {top_batch[1]*100/len(unusable_records):.1f}%)")
    print(f"   - 涉及的批次总数: {len(batch_counter)} 个")

    print(f"\n3. 模型分布特征:")
    print(f"   - 不可用数据最多的模型: {top_model[0]} ({top_model[1]}条, {top_model[1]*100/len(unusable_records):.1f}%)")
    print(f"   - 涉及的模型总数: {len(model_counter)} 个")

    print(f"\n4. 模式分布特征:")
    for mode, count in mode_counter.items():
        pct = count * 100 / len(unusable_records)
        print(f"   - {mode}: {count}条 ({pct:.1f}%)")

    print(f"\n5. 主要不可用原因:")
    reason_counter = Counter()
    for record in unusable_records:
        for reason in record['reasons']:
            reason_counter[reason] += 1

    for reason, count in reason_counter.most_common():
        pct = count * 100 / len(unusable_records)
        print(f"   - {reason}: {count}条 ({pct:.1f}%)")

    print("\n✅ 分析完成!")

    # 输出报告文件
    report_file = "unusable_data_sources_report.txt"
    print(f"\n📄 生成报告文件: {report_file}")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("🔍 不可用数据来源分布分析报告\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"数据文件: {data_file}\n")
        f.write(f"总记录数: {total_rows}\n")
        f.write(f"不可用记录数: {len(unusable_records)} ({len(unusable_records)*100/total_rows:.1f}%)\n\n")

        # 时间分布
        f.write("=" * 100 + "\n")
        f.write("📅 时间分布分析\n")
        f.write("=" * 100 + "\n\n")

        f.write("【按日期统计】\n")
        f.write(f"{'日期':<15} {'不可用记录数':<15} {'占不可用总数':<15}\n")
        f.write("-" * 50 + "\n")
        for date, count in sorted(date_counter.items()):
            pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
            f.write(f"{date:<15} {count:<15} {pct:>12.1f}%\n")

        f.write("\n【按周统计】\n")
        f.write(f"{'周':<15} {'不可用记录数':<15} {'占不可用总数':<15}\n")
        f.write("-" * 50 + "\n")
        for week, count in sorted(week_counter.items()):
            pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
            f.write(f"{week:<15} {count:<15} {pct:>12.1f}%\n")

        f.write("\n【按月统计】\n")
        f.write(f"{'月份':<15} {'不可用记录数':<15} {'占不可用总数':<15}\n")
        f.write("-" * 50 + "\n")
        for month, count in sorted(month_counter.items()):
            pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
            f.write(f"{month:<15} {count:<15} {pct:>12.1f}%\n")

        # 实验批次分布
        f.write("\n" + "=" * 100 + "\n")
        f.write("🧪 实验批次分布分析\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"{'实验批次':<25} {'不可用记录数':<15} {'占不可用总数':<15}\n")
        f.write("-" * 60 + "\n")
        for batch, count in sorted(batch_counter.items(), key=lambda x: x[1], reverse=True):
            pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
            f.write(f"{batch:<25} {count:<15} {pct:>12.1f}%\n")

        # 模型分布
        f.write("\n" + "=" * 100 + "\n")
        f.write("🧬 模型分布分析\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"{'模型':<50} {'不可用记录数':<15} {'占不可用总数':<15}\n")
        f.write("-" * 80 + "\n")
        for model, count in sorted(model_counter.items(), key=lambda x: x[1], reverse=True):
            pct = count * 100 / len(unusable_records) if len(unusable_records) > 0 else 0
            f.write(f"{model:<50} {count:<15} {pct:>12.1f}%\n")

        # 批次-模型交叉分析
        f.write("\n" + "=" * 100 + "\n")
        f.write("🔬 批次-模型交叉分析\n")
        f.write("=" * 100 + "\n\n")

        for batch in sorted(batch_model_counter.keys()):
            f.write(f"\n【批次: {batch}】\n")
            f.write(f"{'模型':<50} {'不可用记录数':<15}\n")
            f.write("-" * 70 + "\n")
            for model, count in sorted(batch_model_counter[batch].items(),
                                       key=lambda x: x[1], reverse=True):
                f.write(f"{model:<50} {count:<15}\n")

        # 关键发现
        f.write("\n" + "=" * 100 + "\n")
        f.write("💡 关键发现和总结\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. 时间分布特征:\n")
        f.write(f"   - 时间跨度: {min([r['date'] for r in unusable_records if r['date'] != 'unknown'])} 至 {max([r['date'] for r in unusable_records if r['date'] != 'unknown'])}\n")
        f.write(f"   - 不可用数据最多的日期: {top_date[0]} ({top_date[1]}条, {top_date[1]*100/len(unusable_records):.1f}%)\n")
        f.write(f"   - 不可用数据最多的月份: {top_month[0]} ({top_month[1]}条, {top_month[1]*100/len(unusable_records):.1f}%)\n")
        f.write(f"   - 涉及的日期总数: {len(date_counter)} 天\n")
        f.write(f"   - 涉及的周数: {len(week_counter)} 周\n")

        f.write("\n2. 实验批次特征:\n")
        f.write(f"   - 不可用数据最多的批次: {top_batch[0]} ({top_batch[1]}条, {top_batch[1]*100/len(unusable_records):.1f}%)\n")
        f.write(f"   - 涉及的批次总数: {len(batch_counter)} 个\n")

        f.write("\n3. 模型分布特征:\n")
        f.write(f"   - 不可用数据最多的模型: {top_model[0]} ({top_model[1]}条, {top_model[1]*100/len(unusable_records):.1f}%)\n")
        f.write(f"   - 涉及的模型总数: {len(model_counter)} 个\n")

        f.write("\n4. 模式分布特征:\n")
        for mode, count in mode_counter.items():
            pct = count * 100 / len(unusable_records)
            f.write(f"   - {mode}: {count}条 ({pct:.1f}%)\n")

        f.write("\n5. 主要不可用原因:\n")
        for reason, count in reason_counter.most_common():
            pct = count * 100 / len(unusable_records)
            f.write(f"   - {reason}: {count}条 ({pct:.1f}%)\n")

        f.write("\n✅ 分析完成!\n")

if __name__ == "__main__":
    main()
