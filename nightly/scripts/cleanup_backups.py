#!/usr/bin/env python3
"""
备份文件清理脚本

目标:
1. 识别所有备份文件
2. 评估保留必要性
3. 归档或删除不必要的备份

版本: v1.0
日期: 2025-12-19
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def analyze_backups():
    """分析所有备份文件"""

    print("=" * 100)
    print("备份文件分析")
    print("=" * 100)
    print()

    # 查找所有备份文件
    results_dir = Path('results')
    backup_files = []

    for pattern in ['*.backup*', '*.bak*']:
        backup_files.extend(results_dir.rglob(pattern))

    # 按位置分类
    categories = {
        'results根目录': [],
        'archived目录': [],
        'backup_archive目录': [],
        'session目录': []
    }

    for file in backup_files:
        rel_path = file.relative_to(results_dir)
        parts = rel_path.parts

        if len(parts) == 1:
            categories['results根目录'].append(file)
        elif 'archived' in parts[0] or 'archive' in parts[0]:
            if 'backup_archive' in str(file):
                categories['backup_archive目录'].append(file)
            else:
                categories['archived目录'].append(file)
        elif parts[0].startswith('run_'):
            categories['session目录'].append(file)
        else:
            categories['results根目录'].append(file)

    # 输出分析结果
    total_size = 0
    total_files = 0

    for category, files in categories.items():
        if files:
            print(f"\n【{category}】: {len(files)}个文件")
            print("-" * 100)

            category_size = 0
            for file in sorted(files):
                size = file.stat().st_size
                size_mb = size / 1024 / 1024
                mtime = datetime.fromtimestamp(file.stat().st_mtime)

                category_size += size
                total_size += size
                total_files += 1

                rel_path = file.relative_to(results_dir)
                print(f"  {str(rel_path):<70} {size_mb:>6.2f}MB  {mtime.strftime('%Y-%m-%d %H:%M')}")

            print(f"  小计: {len(files)}个文件, {category_size/1024/1024:.2f}MB")

    print()
    print("=" * 100)
    print(f"总计: {total_files}个备份文件, {total_size/1024/1024:.2f}MB")
    print("=" * 100)
    print()

    return categories, total_size

def recommend_cleanup(categories):
    """推荐清理方案"""

    print("\n" + "=" * 100)
    print("清理建议")
    print("=" * 100)
    print()

    # 1. results根目录的备份
    root_backups = categories['results根目录']
    if root_backups:
        print("【1. results根目录备份】:")
        print()

        # 按文件名分组
        data_csv_backups = [f for f in root_backups if 'data.csv' in f.name]
        raw_data_backups = [f for f in root_backups if 'raw_data.csv' in f.name]

        if data_csv_backups:
            print("  data.csv备份:")
            latest = max(data_csv_backups, key=lambda f: f.stat().st_mtime)
            for f in sorted(data_csv_backups, key=lambda x: x.stat().st_mtime, reverse=True):
                age_days = (datetime.now().timestamp() - f.stat().st_mtime) / 86400
                if f == latest:
                    print(f"    ✅ 保留: {f.name} (最新, {age_days:.1f}天前)")
                else:
                    print(f"    🗑️ 可删除: {f.name} ({age_days:.1f}天前)")

        if raw_data_backups:
            print(f"\n  raw_data.csv备份:")
            latest = max(raw_data_backups, key=lambda f: f.stat().st_mtime)
            for f in sorted(raw_data_backups, key=lambda x: x.stat().st_mtime, reverse=True):
                age_days = (datetime.now().timestamp() - f.stat().st_mtime) / 86400
                if f == latest:
                    print(f"    ✅ 保留: {f.name} (最新, {age_days:.1f}天前)")
                else:
                    print(f"    🗑️ 可删除: {f.name} ({age_days:.1f}天前)")

    # 2. backup_archive目录
    archive_backups = categories['backup_archive目录']
    if archive_backups:
        print(f"\n【2. backup_archive目录】:")
        print(f"  📦 已归档: {len(archive_backups)}个文件")
        print(f"  💡 建议: 保留归档（已整理）")

    # 3. archived目录的备份
    archived_backups = categories['archived目录']
    if archived_backups:
        print(f"\n【3. archived/summary_archive目录】:")
        print(f"  📦 已归档: {len(archived_backups)}个文件")
        print(f"  💡 建议: 保留归档（历史参考）")

    # 4. session目录的备份
    session_backups = categories['session目录']
    if session_backups:
        print(f"\n【4. session目录】:")
        print(f"  📂 Session备份: {len(session_backups)}个文件")
        print(f"  💡 建议: 评估session价值后决定")

    print()
    print("=" * 100)
    print("清理策略:")
    print("=" * 100)
    print("  ✅ 保留: 每个文件的最新备份（1个）")
    print("  🗑️ 删除: 旧的重复备份（如data.csv.backup_before_refix）")
    print("  📦 保留: 所有归档目录的备份（已整理）")
    print("  💾 保留: backup_archive_20251219目录（批量归档）")
    print()

def execute_cleanup(dry_run=True):
    """执行清理操作"""

    print("\n" + "=" * 100)
    if dry_run:
        print("清理预览（dry-run模式）")
    else:
        print("执行清理")
    print("=" * 100)
    print()

    results_dir = Path('results')

    # 定义删除规则
    to_delete = []
    to_keep = []

    # 1. data.csv备份 - 只保留最新的
    data_csv_backups = list(results_dir.glob('data.csv.backup*'))
    if data_csv_backups:
        latest = max(data_csv_backups, key=lambda f: f.stat().st_mtime)
        for f in data_csv_backups:
            if f == latest:
                to_keep.append(('data.csv最新备份', f))
            else:
                to_delete.append(('data.csv旧备份', f))

    # 2. raw_data.csv备份 - 只保留最新的
    raw_data_backups = list(results_dir.glob('raw_data.csv.backup*'))
    if raw_data_backups:
        latest = max(raw_data_backups, key=lambda f: f.stat().st_mtime)
        for f in raw_data_backups:
            if f == latest:
                to_keep.append(('raw_data.csv最新备份', f))
            else:
                to_delete.append(('raw_data.csv旧备份', f))

    # 输出清理计划
    if to_delete:
        print("🗑️ 计划删除:")
        total_delete_size = 0
        for reason, file in to_delete:
            size = file.stat().st_size / 1024 / 1024
            total_delete_size += size
            print(f"  - {file.name:<60} {size:>6.2f}MB  ({reason})")
        print(f"  小计: {len(to_delete)}个文件, {total_delete_size:.2f}MB")

    if to_keep:
        print(f"\n✅ 保留:")
        total_keep_size = 0
        for reason, file in to_keep:
            size = file.stat().st_size / 1024 / 1024
            total_keep_size += size
            print(f"  - {file.name:<60} {size:>6.2f}MB  ({reason})")
        print(f"  小计: {len(to_keep)}个文件, {total_keep_size:.2f}MB")

    print()
    print("📦 归档目录（不清理）:")
    print("  - archived/summary_archive/ - 10个备份文件")
    print("  - backup_archive_20251219/ - 26个备份文件")

    # 执行删除
    if not dry_run and to_delete:
        print()
        confirm = input("确认删除以上文件? (yes/no): ")
        if confirm.lower() == 'yes':
            for reason, file in to_delete:
                file.unlink()
                print(f"  ✓ 已删除: {file.name}")
            print(f"\n✓ 清理完成: 删除{len(to_delete)}个文件")
        else:
            print("\n取消清理")

    return to_delete, to_keep

if __name__ == '__main__':
    # 分析备份文件
    categories, total_size = analyze_backups()

    # 推荐清理方案
    recommend_cleanup(categories)

    # 预览清理（dry-run）
    to_delete, to_keep = execute_cleanup(dry_run=True)

    print()
    print("下一步:")
    print("  python3 scripts/cleanup_backups.py  # 预览清理")
    print("  # 检查清理计划后，修改脚本设置dry_run=False执行实际清理")
