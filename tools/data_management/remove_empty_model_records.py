#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除空模型记录脚本

删除raw_data.csv中repository和model都为空的记录
这些记录将在后续重新加载时正确提取foreground数据
"""

import csv
import sys

def get_mode(row):
    """获取实验模式"""
    mode = row.get('mode', '')
    if not mode:
        # 如果mode字段为空，通过fg_repository判断
        if row.get('fg_repository'):
            return 'parallel'
        else:
            return 'non-parallel'
    return mode

def is_empty_model_record(row):
    """检查是否是空模型记录"""
    mode = get_mode(row)

    if mode == 'parallel':
        repo = row.get('fg_repository', '')
        model = row.get('fg_model', '')
    else:
        repo = row.get('repository', '')
        model = row.get('model', '')

    # 如果repository和model都为空，则是空模型记录
    return repo == '' and model == ''

def main():
    input_file = "data/raw_data.csv"
    output_file = "data/raw_data.csv.tmp"

    print("=" * 80)
    print("删除空模型记录")
    print("=" * 80)
    print(f"\n读取: {input_file}")

    # 读取所有记录
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total_rows = len(rows)
    print(f"总记录数: {total_rows}")

    # 筛选出非空模型记录
    non_empty_rows = []
    empty_records = []

    for row in rows:
        if is_empty_model_record(row):
            empty_records.append(row)
        else:
            non_empty_rows.append(row)

    print(f"\n空模型记录数: {len(empty_records)}")
    print(f"保留记录数: {len(non_empty_rows)}")

    # 显示前10个空模型记录
    if empty_records:
        print(f"\n空模型记录样本（前10个）:")
        for i, record in enumerate(empty_records[:10], 1):
            exp_id = record.get('experiment_id', 'N/A')
            timestamp = record.get('timestamp', 'N/A')
            mode = get_mode(record)
            print(f"  {i}. {exp_id} ({timestamp[:10] if timestamp != 'N/A' else 'N/A'}, {mode})")

    # 写入新文件
    print(f"\n写入: {output_file}")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(non_empty_rows)

    print(f"\n✅ 删除完成!")
    print(f"   删除: {len(empty_records)}条")
    print(f"   保留: {len(non_empty_rows)}条")

    # 替换原文件
    import os
    os.rename(output_file, input_file)
    print(f"\n✅ 已更新 {input_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
