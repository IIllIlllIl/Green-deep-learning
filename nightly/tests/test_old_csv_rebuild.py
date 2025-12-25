#!/usr/bin/env python3
"""
验证重建的老实验CSV的正确性

比较summary_old.csv和summary_old_rebuilt.csv:
1. 行数是否一致
2. 列数是否一致
3. experiment_id是否完全匹配
4. 关键字段是否一致（repository, model, experiment_source）
"""

import csv
import sys

def load_csv(filename):
    """加载CSV文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def validate_rebuild():
    """验证重建的CSV"""

    print("=" * 70)
    print("验证重建的老实验CSV")
    print("=" * 70)
    print()

    # 加载两个CSV文件
    print("步骤1: 加载CSV文件...")
    original = load_csv('results/summary_old.csv')
    rebuilt = load_csv('results/summary_old_rebuilt.csv')
    print(f"  原始CSV: {len(original)} 行")
    print(f"  重建CSV: {len(rebuilt)} 行")
    print()

    # 验证1: 行数
    print("验证1: 行数是否一致")
    if len(original) == len(rebuilt):
        print(f"  ✓ 行数一致: {len(original)} 行")
    else:
        print(f"  ✗ 行数不一致: 原始{len(original)}行 vs 重建{len(rebuilt)}行")
        return False
    print()

    # 验证2: 列数
    print("验证2: 列数是否一致")
    orig_cols = set(original[0].keys())
    rebuilt_cols = set(rebuilt[0].keys())
    if orig_cols == rebuilt_cols:
        print(f"  ✓ 列数一致: {len(orig_cols)} 列")
    else:
        print(f"  ✗ 列不一致")
        missing = orig_cols - rebuilt_cols
        extra = rebuilt_cols - orig_cols
        if missing:
            print(f"    缺失列: {missing}")
        if extra:
            print(f"    额外列: {extra}")
        return False
    print()

    # 创建字典以便快速查找
    orig_dict = {row['experiment_id']: row for row in original}
    rebuilt_dict = {row['experiment_id']: row for row in rebuilt}

    # 验证3: experiment_id完全匹配
    print("验证3: experiment_id是否完全匹配")
    orig_ids = set(orig_dict.keys())
    rebuilt_ids = set(rebuilt_dict.keys())
    if orig_ids == rebuilt_ids:
        print(f"  ✓ experiment_id完全匹配: {len(orig_ids)} 个")
    else:
        print(f"  ✗ experiment_id不匹配")
        missing = orig_ids - rebuilt_ids
        extra = rebuilt_ids - orig_ids
        if missing:
            print(f"    原始有但重建缺失: {len(missing)} 个")
            for eid in list(missing)[:5]:
                print(f"      - {eid}")
            if len(missing) > 5:
                print(f"      ... 还有 {len(missing)-5} 个")
        if extra:
            print(f"    重建有但原始没有: {len(extra)} 个")
        return False
    print()

    # 验证4: 关键字段一致性
    print("验证4: 关键字段一致性")
    key_fields = ['repository', 'model', 'experiment_source', 'timestamp']
    mismatch_count = 0
    mismatch_details = []

    for exp_id in orig_ids:
        orig_row = orig_dict[exp_id]
        rebuilt_row = rebuilt_dict[exp_id]

        for field in key_fields:
            orig_val = orig_row.get(field, '').strip()
            rebuilt_val = rebuilt_row.get(field, '').strip()

            if orig_val != rebuilt_val:
                mismatch_count += 1
                mismatch_details.append({
                    'experiment_id': exp_id,
                    'field': field,
                    'original': orig_val,
                    'rebuilt': rebuilt_val
                })

    if mismatch_count == 0:
        print(f"  ✓ 所有关键字段一致")
    else:
        print(f"  ⚠️  发现 {mismatch_count} 个不匹配")
        for detail in mismatch_details[:10]:
            print(f"    {detail['experiment_id']} - {detail['field']}:")
            print(f"      原始: {detail['original']}")
            print(f"      重建: {detail['rebuilt']}")
        if len(mismatch_details) > 10:
            print(f"    ... 还有 {len(mismatch_details)-10} 个不匹配")
    print()

    # 验证5: 数据完整性（非空字段检查）
    print("验证5: 数据完整性检查")
    empty_count_orig = 0
    empty_count_rebuilt = 0

    important_fields = ['repository', 'model', 'training_success', 'duration_seconds']

    for row in original:
        for field in important_fields:
            if not row.get(field, '').strip():
                empty_count_orig += 1

    for row in rebuilt:
        for field in important_fields:
            if not row.get(field, '').strip():
                empty_count_rebuilt += 1

    print(f"  原始CSV空字段数: {empty_count_orig}")
    print(f"  重建CSV空字段数: {empty_count_rebuilt}")

    if empty_count_rebuilt <= empty_count_orig:
        print(f"  ✓ 重建CSV数据完整性正常或优于原始")
    else:
        print(f"  ⚠️  重建CSV空字段数增加了 {empty_count_rebuilt - empty_count_orig} 个")
    print()

    # 总结
    print("=" * 70)
    if mismatch_count == 0:
        print("✓ 验证成功！重建的CSV与原始CSV一致")
        print("=" * 70)
        return True
    else:
        print("⚠️  验证发现差异，请检查详细信息")
        print("=" * 70)
        return False

if __name__ == '__main__':
    success = validate_rebuild()
    sys.exit(0 if success else 1)
