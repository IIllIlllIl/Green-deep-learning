#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面分析所有缺失数据

分析raw_data.csv中所有字段的缺失情况，提供详细的缺失模式报告
"""

import csv
from collections import defaultdict, Counter

def is_empty(val):
    """检查值是否为空"""
    return val == '' or val is None

def main():
    data_file = "data/raw_data.csv"

    print("=" * 100)
    print("🔍 全面缺失数据分析")
    print("=" * 100)
    print(f"\n数据文件: {data_file}\n")

    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    total_rows = len(rows)

    print(f"总记录数: {total_rows}")
    print(f"总字段数: {len(fieldnames)}\n")

    # ===== 1. 统计每个字段的缺失情况 =====
    print("=" * 100)
    print("📊 各字段缺失统计")
    print("=" * 100)

    missing_stats = []
    for col in fieldnames:
        missing_count = sum(1 for row in rows if is_empty(row.get(col)))
        if missing_count > 0:
            missing_pct = missing_count * 100 / total_rows
            missing_stats.append({
                'field': col,
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'non_missing_count': total_rows - missing_count
            })

    # 按缺失数量排序
    missing_stats.sort(key=lambda x: x['missing_count'], reverse=True)

    print(f"\n{'字段名':<50} {'缺失数':<10} {'缺失率':<10} {'有效数':<10}")
    print("-" * 100)

    for stat in missing_stats:
        print(f"{stat['field']:<50} {stat['missing_count']:<10} {stat['missing_pct']:>8.1f}% {stat['non_missing_count']:<10}")

    print(f"\n总计: {len(missing_stats)} 个字段有缺失数据")

    # ===== 2. 按缺失字段数量分组记录 =====
    print("\n" + "=" * 100)
    print("📈 记录的缺失字段数量分布")
    print("=" * 100)

    rows_by_missing_count = defaultdict(list)

    for idx, row in enumerate(rows):
        missing_count = 0
        missing_fields = []

        for col in fieldnames:
            val = row.get(col, '')
            if is_empty(val):
                missing_count += 1
                missing_fields.append(col)

        rows_by_missing_count[missing_count].append({
            'index': idx,
            'experiment_id': row.get('experiment_id', 'N/A'),
            'missing_fields': missing_fields
        })

    print(f"\n{'缺失字段数':<15} {'记录数':<10} {'占比':<10}")
    print("-" * 40)

    for missing_count in sorted(rows_by_missing_count.keys()):
        count = len(rows_by_missing_count[missing_count])
        pct = count * 100 / total_rows
        print(f"{missing_count:<15} {count:<10} {pct:>8.1f}%")

    # ===== 3. 分析完全无缺失的记录 =====
    complete_rows = rows_by_missing_count.get(0, [])
    print(f"\n✅ 完全无缺失数据的记录: {len(complete_rows)} 条 ({len(complete_rows)*100/total_rows:.1f}%)")

    # ===== 4. 分析缺失数据的记录 =====
    incomplete_rows_count = total_rows - len(complete_rows)
    print(f"⚠️  有缺失数据的记录: {incomplete_rows_count} 条 ({incomplete_rows_count*100/total_rows:.1f}%)")

    # ===== 5. 缺失模式分析 =====
    print("\n" + "=" * 100)
    print("🔬 缺失模式分析（Top 10）")
    print("=" * 100)

    # 统计缺失字段的组合
    missing_patterns = Counter()

    for missing_count, records in rows_by_missing_count.items():
        if missing_count > 0:
            for record in records:
                # 将缺失字段列表转为元组（可哈希）
                pattern = tuple(sorted(record['missing_fields']))
                missing_patterns[pattern] += 1

    print(f"\n共发现 {len(missing_patterns)} 种不同的缺失模式\n")

    print("Top 10 最常见的缺失模式:")
    print("-" * 100)

    for i, (pattern, count) in enumerate(missing_patterns.most_common(10), 1):
        pct = count * 100 / total_rows
        print(f"\n模式 {i}: {count} 条记录 ({pct:.1f}%)")
        print(f"缺失字段数: {len(pattern)}")
        print(f"缺失字段:")
        for field in pattern[:10]:  # 只显示前10个字段
            print(f"  - {field}")
        if len(pattern) > 10:
            print(f"  ... 还有 {len(pattern) - 10} 个字段")

    # ===== 6. 按字段类别分组分析 =====
    print("\n" + "=" * 100)
    print("📂 按字段类别分析缺失情况")
    print("=" * 100)

    field_categories = {
        'hyperparam': [col for col in fieldnames if col.startswith('hyperparam_')],
        'perf': [col for col in fieldnames if col.startswith('perf_')],
        'energy': [col for col in fieldnames if col.startswith('energy_')],
        'fg_hyperparam': [col for col in fieldnames if col.startswith('fg_hyperparam_')],
        'fg_perf': [col for col in fieldnames if col.startswith('fg_perf_')],
        'fg_energy': [col for col in fieldnames if col.startswith('fg_energy_')],
        'bg': [col for col in fieldnames if col.startswith('bg_')],
        'basic': ['experiment_id', 'timestamp', 'repository', 'model', 'training_success',
                  'duration_seconds', 'retries', 'mode', 'error_message']
    }

    for category, fields in field_categories.items():
        if not fields:
            continue

        total_cells = len(fields) * total_rows
        missing_cells = sum(sum(1 for row in rows if is_empty(row.get(col))) for col in fields if col in fieldnames)
        missing_pct = missing_cells * 100 / total_cells if total_cells > 0 else 0

        print(f"\n{category}:")
        print(f"  字段数: {len(fields)}")
        print(f"  总单元格数: {total_cells}")
        print(f"  缺失单元格数: {missing_cells}")
        print(f"  缺失率: {missing_pct:.1f}%")

    # ===== 7. 详细查看缺失最多的记录 =====
    print("\n" + "=" * 100)
    print("📋 缺失字段最多的10条记录")
    print("=" * 100)

    # 找出缺失字段最多的记录
    all_incomplete = []
    for missing_count, records in rows_by_missing_count.items():
        if missing_count > 0:
            for record in records:
                all_incomplete.append({
                    'missing_count': missing_count,
                    'experiment_id': record['experiment_id'],
                    'missing_fields': record['missing_fields']
                })

    all_incomplete.sort(key=lambda x: x['missing_count'], reverse=True)

    for i, record in enumerate(all_incomplete[:10], 1):
        print(f"\n{i}. Experiment ID: {record['experiment_id']}")
        print(f"   缺失字段数: {record['missing_count']}")
        print(f"   缺失字段（前15个）:")
        for field in record['missing_fields'][:15]:
            print(f"     - {field}")
        if len(record['missing_fields']) > 15:
            print(f"     ... 还有 {len(record['missing_fields']) - 15} 个字段")

    # ===== 8. 总结 =====
    print("\n" + "=" * 100)
    print("📊 总结")
    print("=" * 100)

    print(f"\n数据完整性:")
    print(f"  - 总记录数: {total_rows}")
    print(f"  - 完全无缺失: {len(complete_rows)} ({len(complete_rows)*100/total_rows:.1f}%)")
    print(f"  - 有缺失数据: {incomplete_rows_count} ({incomplete_rows_count*100/total_rows:.1f}%)")

    print(f"\n字段缺失:")
    print(f"  - 有缺失的字段数: {len(missing_stats)} / {len(fieldnames)}")
    print(f"  - 缺失最严重的字段: {missing_stats[0]['field']} ({missing_stats[0]['missing_pct']:.1f}%)")

    print(f"\n缺失模式:")
    print(f"  - 不同缺失模式数: {len(missing_patterns)}")
    print(f"  - 最常见模式的记录数: {missing_patterns.most_common(1)[0][1] if missing_patterns else 0}")

    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
