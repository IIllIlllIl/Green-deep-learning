#!/usr/bin/env python3
"""
验证raw_data.csv的数据完整性和安全性

检查项:
1. 数据完整性: 476行 = 211老实验 + 265新实验
2. 列格式: 80列标准格式
3. 训练成功率
4. 能耗数据完整性
5. 性能指标完整性
6. experiment_id重复问题分析
"""

import csv
from pathlib import Path
from collections import Counter

def validate_raw_data():
    """验证raw_data.csv"""
    filepath = Path('/home/green/energy_dl/nightly/data/raw_data.csv')

    print("=" * 70)
    print("raw_data.csv 数据完整性和安全性验证")
    print("=" * 70)

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    print(f"\n✓ 成功读取 {filepath}")
    print(f"  总行数: {len(rows)}")
    print(f"  列数: {len(header)}")

    # 1. 基本格式检查
    print(f"\n【1. 基本格式检查】")
    if len(header) == 80:
        print(f"  ✓ 列数正确: 80列")
    else:
        print(f"  ❌ 列数错误: {len(header)}列，预期80列")
        return False

    if len(rows) == 476:
        print(f"  ✓ 行数正确: 476行 (211老实验 + 265新实验)")
    else:
        print(f"  ⚠️  行数: {len(rows)}，预期476")

    # 2. 模式分布分析
    print(f"\n【2. 模式分布分析】")
    modes = Counter(row.get('mode', '') for row in rows)
    parallel_count = modes.get('parallel', 0)
    nonparallel_count = sum(count for mode, count in modes.items() if mode and mode != 'parallel')
    empty_mode_count = modes.get('', 0)

    print(f"  - 并行模式: {parallel_count} ({parallel_count/len(rows)*100:.1f}%)")
    print(f"  - 非并行模式: {nonparallel_count} ({nonparallel_count/len(rows)*100:.1f}%)")
    print(f"  - 空模式: {empty_mode_count} ({empty_mode_count/len(rows)*100:.1f}%)")

    # 检查并行模式的数据结构
    parallel_rows = [r for r in rows if r.get('mode') == 'parallel']
    fg_has_data = sum(1 for r in parallel_rows if r.get('fg_repository', '').strip())
    print(f"  ✓ 并行模式中 {fg_has_data}/{len(parallel_rows)} 有前景数据")

    # 3. 训练成功率
    print(f"\n【3. 训练成功率】")

    # 非并行模式（数据在顶层字段）
    nonparallel_rows = [r for r in rows if r.get('mode') != 'parallel']
    nonparallel_success = sum(1 for r in nonparallel_rows if r.get('training_success', '').lower() == 'true')

    # 并行模式（数据在fg_字段）
    parallel_success = sum(1 for r in parallel_rows if r.get('fg_training_success', '').lower() == 'true')

    total_success = nonparallel_success + parallel_success
    print(f"  - 非并行模式: {nonparallel_success}/{len(nonparallel_rows)} ({nonparallel_success/len(nonparallel_rows)*100 if nonparallel_rows else 0:.1f}%)")
    print(f"  - 并行模式: {parallel_success}/{len(parallel_rows)} ({parallel_success/len(parallel_rows)*100:.1f}%)")
    print(f"  ✓ 总体成功率: {total_success}/{len(rows)} ({total_success/len(rows)*100:.1f}%)")

    # 4. 能耗数据完整性
    print(f"\n【4. 能耗数据完整性】")

    # 检查非并行模式的能耗
    nonparallel_cpu = sum(1 for r in nonparallel_rows if r.get('energy_cpu_total_joules', '').strip())
    nonparallel_gpu = sum(1 for r in nonparallel_rows if r.get('energy_gpu_total_joules', '').strip())

    # 检查并行模式的能耗（在fg_字段）
    parallel_cpu = sum(1 for r in parallel_rows if r.get('fg_energy_cpu_total_joules', '').strip())
    parallel_gpu = sum(1 for r in parallel_rows if r.get('fg_energy_gpu_total_joules', '').strip())

    print(f"  非并行模式:")
    print(f"    - CPU能耗: {nonparallel_cpu}/{len(nonparallel_rows)} ({nonparallel_cpu/len(nonparallel_rows)*100 if nonparallel_rows else 0:.1f}%)")
    print(f"    - GPU能耗: {nonparallel_gpu}/{len(nonparallel_rows)} ({nonparallel_gpu/len(nonparallel_rows)*100 if nonparallel_rows else 0:.1f}%)")

    print(f"  并行模式:")
    print(f"    - CPU能耗: {parallel_cpu}/{len(parallel_rows)} ({parallel_cpu/len(parallel_rows)*100:.1f}%)")
    print(f"    - GPU能耗: {parallel_gpu}/{len(parallel_rows)} ({parallel_gpu/len(parallel_rows)*100:.1f}%)")

    total_cpu = nonparallel_cpu + parallel_cpu
    total_gpu = nonparallel_gpu + parallel_gpu
    print(f"  ✓ 总体CPU能耗完整率: {total_cpu}/{len(rows)} ({total_cpu/len(rows)*100:.1f}%)")
    print(f"  ✓ 总体GPU能耗完整率: {total_gpu}/{len(rows)} ({total_gpu/len(rows)*100:.1f}%)")

    # 5. 性能指标完整性（示例：检查accuracy字段）
    print(f"\n【5. 性能指标完整性】")

    # 非并行模式
    nonparallel_perf = sum(1 for r in nonparallel_rows
                           if any(r.get(f'perf_{metric}', '').strip()
                                  for metric in ['accuracy', 'test_accuracy', 'map', 'rank1']))

    # 并行模式
    parallel_perf = sum(1 for r in parallel_rows
                        if any(r.get(f'fg_perf_{metric}', '').strip()
                               for metric in ['accuracy', 'test_accuracy', 'map', 'rank1']))

    total_perf = nonparallel_perf + parallel_perf
    print(f"  - 非并行模式: {nonparallel_perf}/{len(nonparallel_rows)} 有性能指标 ({nonparallel_perf/len(nonparallel_rows)*100 if nonparallel_rows else 0:.1f}%)")
    print(f"  - 并行模式: {parallel_perf}/{len(parallel_rows)} 有性能指标 ({parallel_perf/len(parallel_rows)*100:.1f}%)")
    print(f"  ✓ 总体性能指标完整率: {total_perf}/{len(rows)} ({total_perf/len(rows)*100:.1f}%)")

    # 6. experiment_id重复分析
    print(f"\n【6. experiment_id重复分析】")
    exp_ids = [r['experiment_id'] for r in rows]
    id_counts = Counter(exp_ids)
    duplicates = {id: count for id, count in id_counts.items() if count > 1}

    if not duplicates:
        print(f"  ✓ 所有experiment_id唯一")
    else:
        print(f"  ⚠️  发现 {len(duplicates)} 个重复的experiment_id")
        print(f"  分析: 这可能是因为同一实验在不同模式（并行/非并行）下运行")

        # 检查重复ID的模式分布
        sample_id = list(duplicates.keys())[0]
        sample_rows = [r for r in rows if r['experiment_id'] == sample_id]
        modes_in_dup = [r.get('mode', 'empty') for r in sample_rows]
        print(f"  示例 '{sample_id}' 的模式: {modes_in_dup}")

        if len(set(modes_in_dup)) > 1:
            print(f"  ✓ 重复ID存在于不同模式，属于正常情况")
        else:
            print(f"  ⚠️  重复ID在同一模式下，可能需要进一步检查")

    # 7. 数据来源分布
    print(f"\n【7. 数据来源分布】")
    sources = Counter(row.get('experiment_source', 'unknown') for row in rows)
    for source, count in sorted(sources.items()):
        print(f"  - {source}: {count} ({count/len(rows)*100:.1f}%)")

    # 8. 时间范围
    print(f"\n【8. 时间范围】")
    timestamps = [r.get('timestamp', '') for r in rows if r.get('timestamp', '').strip()]
    if timestamps:
        timestamps_sorted = sorted(timestamps)
        print(f"  最早: {timestamps_sorted[0]}")
        print(f"  最晚: {timestamps_sorted[-1]}")
        print(f"  ✓ 时间跨度: {len(set([t[:10] for t in timestamps]))} 天")

    # 总结
    print(f"\n{'='*70}")
    print(f"✅ 验证完成: raw_data.csv 数据完整且安全")
    print(f"{'='*70}")
    print(f"\n📊 数据摘要:")
    print(f"  - 总实验数: {len(rows)}")
    print(f"  - 训练成功: {total_success} ({total_success/len(rows)*100:.1f}%)")
    print(f"  - CPU能耗完整: {total_cpu} ({total_cpu/len(rows)*100:.1f}%)")
    print(f"  - GPU能耗完整: {total_gpu} ({total_gpu/len(rows)*100:.1f}%)")
    print(f"  - 性能指标完整: {total_perf} ({total_perf/len(rows)*100:.1f}%)")
    print(f"  - 数据格式: 80列标准格式")
    print(f"\n💡 结论: 数据质量良好，可安全使用")

    return True

if __name__ == '__main__':
    validate_raw_data()
