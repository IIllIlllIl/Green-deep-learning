#!/usr/bin/env python3
"""
删除不合理的 MRT-OAST accuracy 数据（样本数 > 1.0）

版本: v1.0
日期: 2025-12-19
"""

import csv
import shutil

def clean_mrt_oast_accuracy():
    """删除 MRT-OAST 中 accuracy > 1.0 的值（样本数）"""
    input_file = 'results/raw_data.csv'
    output_file = 'results/raw_data_cleaned.csv'
    backup_file = 'results/raw_data.csv.backup_before_clean'

    print("=" * 80)
    print("删除不合理的 MRT-OAST accuracy 数据")
    print("=" * 80)
    print()

    # 备份
    if not os.path.exists(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"✓ 已备份: {backup_file}")

    stats = {
        'total_rows': 0,
        'mrt_oast_rows': 0,
        'cleaned_nonparallel': 0,
        'cleaned_parallel': 0,
        'kept_valid': 0
    }

    with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            stats['total_rows'] += 1

            # 检查是否是 MRT-OAST
            is_parallel = (row['mode'] == 'parallel')
            repo = row['fg_repository'] if is_parallel else row['repository']
            model = row['fg_model'] if is_parallel else row['model']

            if repo == 'MRT-OAST' and model == 'default':
                stats['mrt_oast_rows'] += 1

                # 检查并清理 accuracy
                if is_parallel:
                    acc_field = 'fg_perf_accuracy'
                else:
                    acc_field = 'perf_accuracy'

                acc_value = row.get(acc_field, '').strip()

                if acc_value:
                    try:
                        acc_float = float(acc_value)
                        if acc_float > 1.0:
                            # 删除不合理的 accuracy 值
                            row[acc_field] = ''
                            if is_parallel:
                                stats['cleaned_parallel'] += 1
                            else:
                                stats['cleaned_nonparallel'] += 1
                            print(f"  删除: {row['experiment_id'][:50]} - accuracy={acc_value} (样本数)")
                        else:
                            # 保留合理的值（百分比）
                            stats['kept_valid'] += 1
                    except ValueError:
                        pass

            writer.writerow(row)

    # 输出报告
    print()
    print("=" * 80)
    print("清理统计")
    print("=" * 80)
    print(f"总行数: {stats['total_rows']}")
    print(f"MRT-OAST 实验: {stats['mrt_oast_rows']}")
    print(f"  删除非并行 accuracy: {stats['cleaned_nonparallel']}")
    print(f"  删除并行 accuracy: {stats['cleaned_parallel']}")
    print(f"  保留有效 accuracy: {stats['kept_valid']}")
    print(f"  总删除: {stats['cleaned_nonparallel'] + stats['cleaned_parallel']}")
    print()
    print("=" * 80)
    print(f"✓ 输出文件: {output_file}")
    print("=" * 80)

    return stats

if __name__ == '__main__':
    import os
    stats = clean_mrt_oast_accuracy()

    print()
    print("下一步:")
    print("  mv results/raw_data_cleaned.csv results/raw_data.csv")
    print("  python3 scripts/create_unified_data_csv.py")
