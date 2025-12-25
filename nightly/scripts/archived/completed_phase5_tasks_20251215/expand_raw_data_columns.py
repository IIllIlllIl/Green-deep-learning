#!/usr/bin/env python3
"""
扩展raw_data.csv，添加缺失的性能指标列

用法: python3 scripts/expand_raw_data_columns.py

缺失的7个性能指标列:
- perf_eval_loss (VulBERTa/mlp需要)
- perf_final_training_loss (VulBERTa/mlp需要)
- perf_eval_samples_per_second (VulBERTa/mlp需要)
- perf_top1_accuracy (bug-localization需要)
- perf_top5_accuracy (bug-localization需要)
- perf_top10_accuracy (bug-localization需要)
- perf_top20_accuracy (bug-localization需要)

版本: 1.0
创建日期: 2025-12-15
"""

import csv
from datetime import datetime
from pathlib import Path
import shutil

# 新增列
NEW_COLUMNS = [
    'perf_eval_loss',
    'perf_final_training_loss',
    'perf_eval_samples_per_second',
    'perf_top1_accuracy',
    'perf_top5_accuracy',
    'perf_top10_accuracy',
    'perf_top20_accuracy'
]

def expand_raw_data_columns(csv_path='results/raw_data.csv'):
    """扩展raw_data.csv的列定义"""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f'❌ 文件不存在: {csv_path}')
        return False

    # 备份
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = csv_path.parent / f'raw_data.csv.backup_80col_{timestamp}'

    shutil.copy(csv_path, backup_path)
    print(f'✅ 备份: {backup_path}')

    # 读取现有数据
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames
        rows = list(reader)

    print(f'✅ 读取数据: {len(rows)}行, {len(old_fieldnames)}列')

    # 检查新列是否已存在
    existing_new_cols = [col for col in NEW_COLUMNS if col in old_fieldnames]
    if existing_new_cols:
        print(f'⚠️  以下列已存在，跳过: {existing_new_cols}')
        new_cols_to_add = [col for col in NEW_COLUMNS if col not in old_fieldnames]
    else:
        new_cols_to_add = NEW_COLUMNS

    if not new_cols_to_add:
        print('✅ 所有新列都已存在，无需扩展')
        return True

    # 创建新的fieldnames
    new_fieldnames = list(old_fieldnames)

    # 在perf_test_loss后插入新列
    if 'perf_test_loss' in new_fieldnames:
        insert_index = new_fieldnames.index('perf_test_loss') + 1
    else:
        # 如果perf_test_loss不存在，在最后插入
        insert_index = len(new_fieldnames)

    for col in new_cols_to_add:
        new_fieldnames.insert(insert_index, col)
        insert_index += 1

    print(f'列数: {len(old_fieldnames)} → {len(new_fieldnames)}')
    print(f'新增列: {new_cols_to_add}')

    # 写回文件
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f'✅ 已更新: {csv_path}')
    print(f'   列数: {len(new_fieldnames)}')

    # 验证
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        final_fieldnames = reader.fieldnames
        final_rows = list(reader)

    if len(final_rows) == len(rows) and len(final_fieldnames) == len(new_fieldnames):
        print(f'✅ 验证通过: {len(final_rows)}行, {len(final_fieldnames)}列')
        return True
    else:
        print(f'❌ 验证失败:')
        print(f'   预期: {len(rows)}行, {len(new_fieldnames)}列')
        print(f'   实际: {len(final_rows)}行, {len(final_fieldnames)}列')
        return False

if __name__ == '__main__':
    print('=' * 80)
    print('扩展raw_data.csv - 添加缺失的性能指标列')
    print('=' * 80)
    print('')

    success = expand_raw_data_columns()

    print('')
    print('=' * 80)
    if success:
        print('✅ 完成')
    else:
        print('❌ 失败')
    print('=' * 80)
