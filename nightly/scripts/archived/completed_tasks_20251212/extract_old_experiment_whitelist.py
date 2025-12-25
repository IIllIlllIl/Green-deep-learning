#!/usr/bin/env python3
"""
从summary_old.csv提取安全实验白名单

目的: 从现有的summary_old.csv中提取实验ID和experiment_source，
     生成白名单用于后续从experiment.json重建CSV

输出: results/old_experiment_whitelist.json
格式:
{
    "experiment_id1": "default",
    "experiment_id2": "mutation_1x",
    ...
}
"""

import csv
import json
import sys

def extract_whitelist():
    """从summary_old.csv提取白名单"""

    input_file = 'results/summary_old.csv'
    output_file = 'results/old_experiment_whitelist.json'

    print("=" * 70)
    print("从summary_old.csv提取安全实验白名单")
    print("=" * 70)

    whitelist = {}

    # 读取summary_old.csv
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 提取experiment_id和experiment_source
    for row in rows:
        exp_id = row.get('experiment_id', '').strip()
        exp_source = row.get('experiment_source', '').strip()

        if exp_id and exp_source:
            whitelist[exp_id] = exp_source

    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(whitelist, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print(f"✓ 从 {input_file} 读取 {len(rows)} 行")
    print(f"✓ 提取 {len(whitelist)} 个安全实验")
    print()

    # 按experiment_source分类统计
    source_dist = {}
    for exp_id, source in whitelist.items():
        source_dist[source] = source_dist.get(source, 0) + 1

    print("实验源分布:")
    for source, count in sorted(source_dist.items()):
        print(f"  {source:20s}: {count:3d} 个实验")

    print()
    print(f"✓ 白名单已保存至: {output_file}")
    print("=" * 70)

    return True

if __name__ == '__main__':
    success = extract_whitelist()
    sys.exit(0 if success else 1)
