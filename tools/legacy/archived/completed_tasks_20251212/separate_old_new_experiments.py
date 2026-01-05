#!/usr/bin/env python3
"""
分离新老实验到不同的CSV文件

老实验: experiment_source为 'default', 'mutation_1x', 'mutation_2x_safe'
新实验: experiment_source为空或其他值

输出:
- results/summary_old.csv: 211个老实验
- results/summary_new.csv: 265个新实验
"""

import csv
import sys

def separate_experiments():
    """分离新老实验"""

    input_file = 'results/summary_all.csv'
    old_file = 'results/summary_old.csv'
    new_file = 'results/summary_new.csv'

    # 定义老实验的source值
    old_sources = {'default', 'mutation_1x', 'mutation_2x_safe'}

    # 读取所有实验
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    # 分离新老实验
    old_experiments = []
    new_experiments = []

    for row in all_rows:
        source = row.get('experiment_source', '').strip()
        if source in old_sources:
            old_experiments.append(row)
        else:
            new_experiments.append(row)

    # 写入老实验CSV
    with open(old_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_experiments)

    # 写入新实验CSV
    with open(new_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_experiments)

    # 打印统计信息
    print(f"分离新老实验完成")
    print("=" * 70)
    print(f"总实验数: {len(all_rows)}")
    print(f"老实验数: {len(old_experiments)} → {old_file}")
    print(f"新实验数: {len(new_experiments)} → {new_file}")
    print()

    print("老实验source分布:")
    old_source_dist = {}
    for row in old_experiments:
        source = row.get('experiment_source', '').strip()
        old_source_dist[source] = old_source_dist.get(source, 0) + 1

    for source, count in sorted(old_source_dist.items()):
        print(f"  {source:20s}: {count:3d} 行")

    print()
    print("新实验source分布:")
    new_source_dist = {}
    for row in new_experiments:
        source = row.get('experiment_source', '').strip()
        source_key = source if source else '(empty)'
        new_source_dist[source_key] = new_source_dist.get(source_key, 0) + 1

    for source, count in sorted(new_source_dist.items()):
        print(f"  {source:20s}: {count:3d} 行")

    print("=" * 70)

    # 验证数据完整性
    if len(old_experiments) + len(new_experiments) != len(all_rows):
        print("❌ 错误：分离后的行数不匹配！")
        return False

    print("✓ 数据完整性验证通过")
    return True

if __name__ == '__main__':
    success = separate_experiments()
    sys.exit(0 if success else 1)
