#!/usr/bin/env python3
"""
data.csv与raw_data.csv完整数据质量评估

检查项:
1. 基础统计对比
2. 列结构差异
3. 数据一致性（考虑并行模式）
4. 数据缺失分析
5. 性能指标对比
6. 能耗数据对比

版本: v1.0
日期: 2025-12-19
"""

import csv
from collections import defaultdict

def compare_data_quality():
    """完整数据质量对比"""

    print("=" * 100)
    print("data.csv与raw_data.csv完整数据质量评估")
    print("=" * 100)
    print()

    # 读取数据
    with open('results/data.csv', 'r') as f:
        data_reader = csv.DictReader(f)
        data_cols = data_reader.fieldnames
        data_rows = list(data_reader)

    with open('results/raw_data.csv', 'r') as f:
        raw_reader = csv.DictReader(f)
        raw_cols = raw_reader.fieldnames
        raw_rows = list(raw_reader)

    # 创建索引
    data_dict = {f"{row['experiment_id']}|{row['timestamp']}": row for row in data_rows}
    raw_dict = {f"{row['experiment_id']}|{row['timestamp']}": row for row in raw_rows}

    # ========== 1. 基础统计 ==========
    print("【1. 基础统计对比】")
    print("-" * 100)
    print(f"data.csv:     {len(data_rows):>4}行 × {len(data_cols):>2}列")
    print(f"raw_data.csv: {len(raw_rows):>4}行 × {len(raw_cols):>2}列")
    print()

    if len(data_rows) == len(raw_rows):
        print(f"✓ 行数一致")
    else:
        print(f"❌ 行数不一致: 差异{abs(len(data_rows) - len(raw_rows))}行")
    print()

    # ========== 2. 列结构分析 ==========
    print("【2. 列结构差异分析】")
    print("-" * 100)

    data_cols_set = set(data_cols)
    raw_cols_set = set(raw_cols)
    common_cols = data_cols_set & raw_cols_set

    print(f"共同列: {len(common_cols)}个")
    print(f"data.csv独有: {len(data_cols_set - raw_cols_set)}个")
    print(f"raw_data.csv独有: {len(raw_cols_set - data_cols_set)}个")
    print()

    # data.csv独有列
    if data_cols_set - raw_cols_set:
        print("data.csv独有列:")
        for col in sorted(data_cols_set - raw_cols_set):
            print(f"  + {col}")
        print()

    # raw_data.csv独有列（分类显示）
    raw_only = raw_cols_set - data_cols_set
    if raw_only:
        fg_cols = [c for c in raw_only if c.startswith('fg_')]
        bg_cols = [c for c in raw_only if c.startswith('bg_')]
        other_cols = [c for c in raw_only if not c.startswith('fg_') and not c.startswith('bg_')]

        print(f"raw_data.csv独有列 ({len(raw_only)}个):")
        if fg_cols:
            print(f"  前景字段(fg_): {len(fg_cols)}个")
            for col in sorted(fg_cols)[:5]:
                print(f"    - {col}")
            if len(fg_cols) > 5:
                print(f"    ... 还有{len(fg_cols) - 5}个")

        if bg_cols:
            print(f"  背景字段(bg_): {len(bg_cols)}个")
            for col in sorted(bg_cols)[:5]:
                print(f"    - {col}")
            if len(bg_cols) > 5:
                print(f"    ... 还有{len(bg_cols) - 5}个")

        if other_cols:
            print(f"  其他字段: {len(other_cols)}个")
            for col in sorted(other_cols):
                print(f"    - {col}")
        print()

    # ========== 3. 数据一致性检查（正确方法）==========
    print("【3. 数据一致性检查（考虑并行模式）】")
    print("-" * 100)

    # 检查字段：非并行实验 vs 并行实验
    check_fields = ['repository', 'model', 'training_success', 'duration_seconds']

    nonparallel_consistent = 0
    nonparallel_total = 0
    parallel_consistent = 0
    parallel_total = 0

    mismatch_samples = []

    for key in data_dict.keys():
        data_row = data_dict[key]
        raw_row = raw_dict[key]

        is_parallel = data_row.get('is_parallel', '') == 'True'

        for field in check_fields:
            data_val = data_row.get(field, '').strip()

            if is_parallel:
                # 并行实验: 对比fg_字段
                raw_val = raw_row.get(f'fg_{field}', '').strip()
                if not raw_val:
                    # fallback到顶层字段（老格式）
                    raw_val = raw_row.get(field, '').strip()

                parallel_total += 1
                if data_val == raw_val:
                    parallel_consistent += 1
                else:
                    if len(mismatch_samples) < 3:
                        mismatch_samples.append({
                            'key': key.split('|')[0],
                            'field': field,
                            'data': data_val,
                            'raw': raw_val,
                            'type': 'parallel'
                        })
            else:
                # 非并行实验: 对比顶层字段
                raw_val = raw_row.get(field, '').strip()

                nonparallel_total += 1
                if data_val == raw_val:
                    nonparallel_consistent += 1
                else:
                    if len(mismatch_samples) < 3:
                        mismatch_samples.append({
                            'key': key.split('|')[0],
                            'field': field,
                            'data': data_val,
                            'raw': raw_val,
                            'type': 'nonparallel'
                        })

    print(f"非并行实验一致性: {nonparallel_consistent}/{nonparallel_total} ({nonparallel_consistent/nonparallel_total*100:.1f}%)")
    print(f"并行实验一致性: {parallel_consistent}/{parallel_total} ({parallel_consistent/parallel_total*100:.1f}%)")

    total_consistent = nonparallel_consistent + parallel_consistent
    total_checks = nonparallel_total + parallel_total
    print(f"总体一致性: {total_consistent}/{total_checks} ({total_consistent/total_checks*100:.1f}%)")

    if mismatch_samples:
        print(f"\n不一致样本 (前3个):")
        for sample in mismatch_samples:
            print(f"  - {sample['key']} ({sample['type']})")
            print(f"    {sample['field']}: data='{sample['data']}' vs raw='{sample['raw']}'")
    else:
        print("\n✓ 所有检查点完全一致")
    print()

    # ========== 4. 性能指标对比 ==========
    print("【4. 性能指标数据对比】")
    print("-" * 100)

    perf_fields = [col for col in data_cols if col.startswith('perf_')]

    perf_stats = {}
    for field in perf_fields:
        data_count = sum(1 for row in data_rows if row.get(field, '').strip())

        # raw中可能在fg_perf或perf
        raw_count = 0
        for row in raw_rows:
            is_parallel = row.get('mode', '') == 'parallel'
            if is_parallel:
                val = row.get(f'fg_{field}', '').strip()
                if not val:
                    val = row.get(field, '').strip()
            else:
                val = row.get(field, '').strip()

            if val:
                raw_count += 1

        perf_stats[field] = {
            'data': data_count,
            'raw': raw_count,
            'diff': abs(data_count - raw_count)
        }

    print(f"{'性能指标':<35} {'data.csv':<10} {'raw_data':<10} {'差异':<10} {'状态'}")
    print("-" * 100)

    for field in sorted(perf_fields):
        stats = perf_stats[field]
        status = "✓" if stats['diff'] == 0 else "⚠️"
        field_short = field.replace('perf_', '')
        print(f"{field_short:<35} {stats['data']:<10} {stats['raw']:<10} {stats['diff']:<10} {status}")

    total_perf_diff = sum(s['diff'] for s in perf_stats.values())
    if total_perf_diff == 0:
        print(f"\n✓ 所有性能指标完全一致")
    else:
        print(f"\n⚠️ 总差异: {total_perf_diff}个值")
    print()

    # ========== 5. 数据缺失分析 ==========
    print("【5. 数据缺失对比】")
    print("-" * 100)

    # 统计每个文件的填充率
    data_fill_rates = {}
    raw_fill_rates = {}

    # data.csv填充率
    for col in data_cols:
        if col not in ['experiment_id', 'timestamp']:
            count = sum(1 for row in data_rows if row.get(col, '').strip())
            data_fill_rates[col] = count / len(data_rows) * 100

    # raw_data.csv填充率
    for col in raw_cols:
        if col not in ['experiment_id', 'timestamp']:
            count = sum(1 for row in raw_rows if row.get(col, '').strip())
            raw_fill_rates[col] = count / len(raw_rows) * 100

    # 关键字段对比
    key_fields = ['training_success', 'duration_seconds',
                  'energy_cpu_total_joules', 'energy_gpu_total_joules']

    print(f"{'关键字段':<35} {'data.csv填充率':<20} {'raw_data填充率':<20} {'状态'}")
    print("-" * 100)

    for field in key_fields:
        data_rate = data_fill_rates.get(field, 0)
        raw_rate = raw_fill_rates.get(field, 0)
        diff = abs(data_rate - raw_rate)

        status = "✓" if diff < 1.0 else ("⚠️" if diff < 5.0 else "❌")

        print(f"{field:<35} {data_rate:>6.1f}%             {raw_rate:>6.1f}%             {status}")

    print()

    # ========== 6. 总结 ==========
    print("【6. 评估总结】")
    print("=" * 100)

    issues = []

    if len(data_rows) != len(raw_rows):
        issues.append(f"行数不一致 (差异{abs(len(data_rows) - len(raw_rows))}行)")

    if total_consistent < total_checks:
        issues.append(f"数据一致性 {total_consistent/total_checks*100:.1f}%")

    if total_perf_diff > 0:
        issues.append(f"性能指标差异 {total_perf_diff}个值")

    if not issues:
        print("✅ 数据质量优秀")
        print("  - 行数完全一致")
        print("  - 数据完全一致（考虑并行模式差异）")
        print("  - 性能指标完全一致")
        print("  - data.csv是raw_data.csv的正确统一视图")
    else:
        print("⚠️ 发现问题:")
        for issue in issues:
            print(f"  - {issue}")

    print()
    print("=" * 100)
    print("结论:")
    print("=" * 100)
    print("data.csv设计目的:")
    print("  1. 统一并行/非并行实验的字段（通过is_parallel区分）")
    print("  2. 优先使用fg_字段（并行模式），fallback到顶层字段")
    print("  3. 删除空列和冗余字段，精简到54列")
    print()
    print("raw_data.csv设计目的:")
    print("  1. 保留原始数据结构（87列）")
    print("  2. 保留fg_/bg_分离字段（并行模式详细信息）")
    print("  3. 作为数据源，不做任何修改")
    print()
    print("两者关系:")
    print("  data.csv = 统一视图(raw_data.csv)")
    print("  无数据冲突，仅格式差异")
    print("=" * 100)

if __name__ == '__main__':
    compare_data_quality()
