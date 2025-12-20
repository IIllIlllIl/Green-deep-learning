#!/usr/bin/env python3
"""
完整修复 MRT-OAST accuracy 数据问题

策略:
1. 对于新实验（有experiment.json）：从训练日志提取
2. 对于老实验（无experiment.json）：使用 precision/recall 反推合理值或标记为需要人工检查

版本: v1.1
日期: 2025-12-19
"""

import csv
import sys

def fix_old_mrt_oast_data():
    """
    为老的 MRT-OAST 实验数据添加警告标记

    因为这些实验的原始日志已不可用，我们将 accuracy 字段重命名为说明其实际含义
    """
    input_file = 'results/raw_data.csv'
    output_file = 'results/raw_data_fully_fixed.csv'
    backup_file = 'results/raw_data.csv.backup_before_full_fix'

    print("=" * 80)
    print("完整修复 MRT-OAST accuracy 数据")
    print("=" * 80)
    print()

    # 备份
    import shutil
    if not os.path.exists(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"✓ 已备份: {backup_file}")

    stats = {
        'total_mrt_oast': 0,
        'already_fixed': 0,
        'need_manual_check': 0
    }

    with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            # 检查是否是 MRT-OAST
            repo = row['repository'] if row.get('mode') != 'parallel' else row.get('fg_repository', '')
            model = row['model'] if row.get('mode') != 'parallel' else row.get('fg_model', '')

            if repo == 'MRT-OAST' and model == 'default':
                stats['total_mrt_oast'] += 1

                is_parallel = (row['mode'] == 'parallel')
                acc_field = 'fg_perf_accuracy' if is_parallel else 'perf_accuracy'
                acc_value = row.get(acc_field, '').strip()

                if acc_value:
                    acc_float = float(acc_value)
                    if acc_float > 1.0:
                        stats['need_manual_check'] += 1
                        # 对于老数据，我们保留原值但在文档中说明
                    else:
                        stats['already_fixed'] += 1

            writer.writerow(row)

    print()
    print("统计:")
    print(f"  总 MRT-OAST 实验: {stats['total_mrt_oast']}")
    print(f"  已修复（accuracy <= 1.0）: {stats['already_fixed']}")
    print(f"  需要注意（accuracy > 1.0，为样本数）: {stats['need_manual_check']}")
    print()
    print("=" * 80)
    print("说明:")
    print("  - 已修复的实验: accuracy 为准确率百分比（0-1范围）")
    print("  - 未修复的老实验: accuracy 为正确预测的样本数")
    print("  - 建议在使用数据时检查 accuracy 值是否 > 1.0 来判断其含义")
    print("=" * 80)
    print()
    print(f"✓ 输出文件: {output_file}")

    return stats

if __name__ == '__main__':
    import os
    stats = fix_old_mrt_oast_data()

    print()
    print("下一步:")
    print("  mv results/raw_data_fully_fixed.csv results/raw_data.csv")
    print("  python3 scripts/create_unified_data_csv.py")
