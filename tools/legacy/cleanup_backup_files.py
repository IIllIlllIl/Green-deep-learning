#!/usr/bin/env python3
"""
清理results/目录中的过时备份文件

清理策略：
1. backup_archive_20251219/: 删除13个时间戳备份，保留关键备份（phase/fix等）
2. results/根目录: 删除data.csv相关备份（data.csv是raw_data.csv的子集，仍在使用）
3. results/根目录: 保留raw_data.csv.backup_before_clean（重要快照）
4. session备份: 删除run_20251212_224937/summary.csv.backup_before_reextraction（已过时）

不删除的文件：
- raw_data.csv.backup_before_clean (重要清理前快照)
- backup_archive中的关键备份（phase/fix/dedup/80col/restore）
- data.csv相关备份 (data.csv仍是主分析文件，保留备份)
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# 定义路径
RESULTS_DIR = Path("results")
BACKUP_ARCHIVE_DIR = RESULTS_DIR / "backup_archive_20251219"

# 归档目录名称
CLEANUP_ARCHIVE_DIR = RESULTS_DIR / "backup_cleanup_archive_20251219"

def create_cleanup_archive():
    """创建清理归档目录"""
    if not CLEANUP_ARCHIVE_DIR.exists():
        CLEANUP_ARCHIVE_DIR.mkdir(parents=True)
        print(f"✓ 创建清理归档目录: {CLEANUP_ARCHIVE_DIR}")
    return CLEANUP_ARCHIVE_DIR

def cleanup_timestamped_backups():
    """清理backup_archive中的时间戳备份"""
    print("\n" + "=" * 80)
    print("清理 backup_archive_20251219/ 中的时间戳备份")
    print("=" * 80)

    if not BACKUP_ARCHIVE_DIR.exists():
        print("目录不存在，跳过")
        return

    # 排除的关键备份类型
    excluded_keywords = ["before_phase", "before_fix", "before_dedup", "before_restore", "80col"]

    backups = sorted(BACKUP_ARCHIVE_DIR.glob("*.backup*"))
    timestamped = [
        b for b in backups
        if not any(keyword in b.name for keyword in excluded_keywords)
    ]

    print(f"\n发现时间戳备份: {len(timestamped)}个")

    if not timestamped:
        print("✓ 无需清理")
        return

    # 保留最新的2个
    to_keep = sorted(timestamped, key=lambda x: x.stat().st_mtime, reverse=True)[:2]
    to_remove = [b for b in timestamped if b not in to_keep]

    print(f"\n保留最新的 {len(to_keep)} 个备份:")
    for backup in to_keep:
        mtime = datetime.fromtimestamp(backup.stat().st_mtime)
        print(f"  ✓ {backup.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")

    print(f"\n删除 {len(to_remove)} 个过时备份:")
    for backup in to_remove:
        print(f"  × {backup.name}")
        backup.unlink()

    print(f"\n✓ 清理完成，删除了 {len(to_remove)} 个文件")

def cleanup_session_backup():
    """清理session备份"""
    print("\n" + "=" * 80)
    print("清理 Session 备份")
    print("=" * 80)

    session_backup = RESULTS_DIR / "run_20251212_224937" / "summary.csv.backup_before_reextraction"

    if not session_backup.exists():
        print(f"✓ 文件不存在: {session_backup.name}")
        return

    print(f"\n删除过时session备份:")
    print(f"  × {session_backup}")
    session_backup.unlink()
    print(f"✓ 清理完成")

def generate_cleanup_summary():
    """生成清理总结"""
    print("\n" + "=" * 80)
    print("清理总结")
    print("=" * 80)

    # 统计backup_archive剩余文件
    if BACKUP_ARCHIVE_DIR.exists():
        remaining = list(BACKUP_ARCHIVE_DIR.glob("*.backup*"))
        print(f"\nbackup_archive_20251219/ 剩余备份: {len(remaining)}个")

        # 按类型分组
        phase = [b for b in remaining if "before_phase" in b.name]
        fix = [b for b in remaining if "before_fix" in b.name]
        dedup = [b for b in remaining if "before_dedup" in b.name]
        restore = [b for b in remaining if "before_restore" in b.name]
        col80 = [b for b in remaining if "80col" in b.name]
        timestamped = [b for b in remaining if not any(keyword in b.name for keyword in
                      ["before_phase", "before_fix", "before_dedup", "before_restore", "80col"])]

        print(f"  - Phase备份: {len(phase)}个")
        print(f"  - Fix备份: {len(fix)}个")
        print(f"  - Dedup备份: {len(dedup)}个")
        print(f"  - Restore备份: {len(restore)}个")
        print(f"  - 80col备份: {len(col80)}个")
        print(f"  - 时间戳备份: {len(timestamped)}个 (保留最新2个)")

    # 统计results根目录备份
    root_backups = list(RESULTS_DIR.glob("*.backup*"))
    print(f"\nresults/ 根目录备份: {len(root_backups)}个")
    for backup in sorted(root_backups):
        size_kb = backup.stat().st_size / 1024
        print(f"  ✓ {backup.name} ({size_kb:.1f} KB)")

def main():
    print("备份文件清理工具")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 切换到项目目录
    os.chdir("/home/green/energy_dl/nightly")

    # 确认当前主数据文件状态
    raw_data = RESULTS_DIR / "raw_data.csv"
    if not raw_data.exists():
        print("错误: raw_data.csv不存在，中止清理")
        return

    # 执行清理
    cleanup_timestamped_backups()
    cleanup_session_backup()
    generate_cleanup_summary()

    print("\n" + "=" * 80)
    print("清理完成")
    print("=" * 80)

    print("\n保留的重要备份:")
    print("  1. data/raw_data.csv.backup_before_clean - 重要清理前快照")
    print("  2. data/data.csv.backup_* - data.csv相关备份（data.csv仍在使用）")
    print("  3. backup_archive中的关键备份（phase/fix/dedup等）")
    print("  4. backup_archive中最新的2个时间戳备份")

if __name__ == "__main__":
    main()
