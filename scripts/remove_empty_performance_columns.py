#!/usr/bin/env python3
"""
删除data.csv中的空列

目标: 删除perf_accuracy和perf_eval_loss两个空列

版本: v1.0
日期: 2025-12-19
"""

import csv
import shutil
from datetime import datetime

def remove_empty_columns():
    """删除空列"""

    input_file = 'results/data.csv'
    output_file = 'results/data_cleaned.csv'
    backup_file = f'results/data.csv.backup_before_column_removal_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    print("=" * 100)
    print("删除data.csv中的空列")
    print("=" * 100)
    print()

    # 备份
    shutil.copy(input_file, backup_file)
    print(f"✓ 已备份: {backup_file}")
    print()

    # 要删除的列
    columns_to_remove = ['perf_accuracy', 'perf_eval_loss']

    # 读取并处理
    with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
        reader = csv.DictReader(fin)

        # 新的字段列表（排除要删除的列）
        new_fieldnames = [col for col in reader.fieldnames if col not in columns_to_remove]

        writer = csv.DictWriter(fout, fieldnames=new_fieldnames)
        writer.writeheader()

        row_count = 0
        for row in reader:
            # 删除指定列
            for col in columns_to_remove:
                row.pop(col, None)

            writer.writerow(row)
            row_count += 1

    # 输出报告
    print("删除操作完成:")
    print("-" * 100)
    print(f"原始列数: {len(reader.fieldnames)}")
    print(f"新列数: {len(new_fieldnames)}")
    print(f"删除列数: {len(columns_to_remove)}")
    print()

    print("已删除的列:")
    for col in columns_to_remove:
        print(f"  - {col}")
    print()

    print(f"处理行数: {row_count}")
    print()

    print("=" * 100)
    print(f"✓ 输出文件: {output_file}")
    print(f"✓ 备份文件: {backup_file}")
    print("=" * 100)
    print()

    print("下一步:")
    print("  1. 验证: python3 scripts/validate_cleaned_csv.py")
    print("  2. 替换: mv results/data_cleaned.csv results/data.csv")
    print()

    return {
        'original_columns': len(reader.fieldnames),
        'new_columns': len(new_fieldnames),
        'removed_columns': len(columns_to_remove),
        'rows': row_count
    }

if __name__ == '__main__':
    stats = remove_empty_columns()

    print(f"总结: 从{stats['original_columns']}列减少到{stats['new_columns']}列")
    print(f"精简度: {stats['removed_columns']}/{stats['original_columns']} ({stats['removed_columns']/stats['original_columns']*100:.1f}%)")
