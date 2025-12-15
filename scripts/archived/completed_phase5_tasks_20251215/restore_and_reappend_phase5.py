#!/usr/bin/env python3
"""
恢复raw_data.csv到Phase 5之前的状态（512行），并重新追加Phase 5数据

步骤：
1. 从备份恢复到512行（Phase 4之后）
2. 扩展列到87列（添加7个新的性能指标列）
3. 重新运行append_session_to_raw_data.py追加Phase 5数据

版本: 1.0
创建日期: 2025-12-15
"""

import csv
from pathlib import Path
import shutil
from datetime import datetime

# 新增的7列
NEW_COLUMNS = [
    'perf_eval_loss',
    'perf_final_training_loss',
    'perf_eval_samples_per_second',
    'perf_top1_accuracy',
    'perf_top5_accuracy',
    'perf_top10_accuracy',
    'perf_top20_accuracy'
]

def restore_and_expand():
    """恢复并扩展raw_data.csv"""

    backup_file = Path('results/raw_data.csv.backup_20251215_171809')
    output_file = Path('results/raw_data.csv')

    if not backup_file.exists():
        print(f'❌ 备份文件不存在: {backup_file}')
        return False

    # 创建当前文件的备份
    if output_file.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        current_backup = output_file.parent / f'raw_data.csv.backup_before_restore_{timestamp}'
        shutil.copy(output_file, current_backup)
        print(f'✅ 当前文件已备份: {current_backup}')

    # 读取512行备份数据
    with open(backup_file, 'r') as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames
        rows = list(reader)

    print(f'✅ 读取备份数据: {len(rows)}行, {len(old_fieldnames)}列')

    # 创建扩展后的fieldnames
    new_fieldnames = list(old_fieldnames)

    # 在perf_test_loss后插入新列
    if 'perf_test_loss' in new_fieldnames:
        insert_index = new_fieldnames.index('perf_test_loss') + 1
    else:
        insert_index = len(new_fieldnames)

    for col in NEW_COLUMNS:
        if col not in new_fieldnames:
            new_fieldnames.insert(insert_index, col)
            insert_index += 1

    print(f'✅ 扩展列: {len(old_fieldnames)} → {len(new_fieldnames)}')
    print(f'   新增: {[col for col in NEW_COLUMNS if col not in old_fieldnames]}')

    # 写入扩展后的CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f'✅ 已恢复并扩展: {output_file}')
    print(f'   数据行: {len(rows)}')
    print(f'   列数: {len(new_fieldnames)}')

    # 验证
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        final_fieldnames = reader.fieldnames
        final_rows = list(reader)

    if len(final_rows) == len(rows) and len(final_fieldnames) == len(new_fieldnames):
        print(f'✅ 验证通过')
        return True
    else:
        print(f'❌ 验证失败')
        return False

if __name__ == '__main__':
    print('=' * 80)
    print('恢复raw_data.csv到Phase 5之前的状态（512行，87列）')
    print('=' * 80)
    print('')

    success = restore_and_expand()

    print('')
    print('=' * 80)
    if success:
        print('✅ 恢复成功')
        print('')
        print('下一步: 运行以下命令重新追加Phase 5数据')
        print('  python3 scripts/append_session_to_raw_data.py results/run_20251214_160925')
    else:
        print('❌ 恢复失败')
    print('=' * 80)
