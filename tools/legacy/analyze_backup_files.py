#!/usr/bin/env python3
"""
分析results/目录中的备份文件，评估是否可以清理

根据以下规则评估备份文件：
1. 如果当前主文件完整且验证通过，中间备份可以清理
2. 保留最近的关键备份（before_phase*, before_fix*等）
3. 清理重复的时间戳备份
"""

import os
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 定义路径
RESULTS_DIR = Path("results")
BACKUP_ARCHIVE_DIR = RESULTS_DIR / "backup_archive_20251219"
SUMMARY_ARCHIVE_DIR = RESULTS_DIR / "archived" / "summary_archive"

# 当前主数据文件
CURRENT_RAW_DATA = RESULTS_DIR / "raw_data.csv"
CURRENT_DATA = RESULTS_DIR / "data.csv"

def get_file_info(filepath):
    """获取文件信息"""
    if not filepath.exists():
        return None

    stat = filepath.stat()
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    return {
        'size': stat.st_size,
        'mtime': datetime.fromtimestamp(stat.st_mtime),
        'columns': len(header) if header else 0,
        'rows': len(rows),
        'header': header
    }

def analyze_backup_archive():
    """分析backup_archive_20251219目录中的备份"""
    print("=" * 80)
    print("分析 backup_archive_20251219/ 目录")
    print("=" * 80)

    if not BACKUP_ARCHIVE_DIR.exists():
        print("目录不存在")
        return

    backups = sorted(BACKUP_ARCHIVE_DIR.glob("*.backup*"))

    # 按类型分组
    backup_types = defaultdict(list)
    for backup in backups:
        name = backup.name
        if "before_phase" in name:
            backup_types["phase"].append(backup)
        elif "before_fix" in name:
            backup_types["fix"].append(backup)
        elif "before_dedup" in name:
            backup_types["dedup"].append(backup)
        elif "before_restore" in name:
            backup_types["restore"].append(backup)
        elif "80col" in name:
            backup_types["80col"].append(backup)
        else:
            # 按日期的备份
            backup_types["timestamped"].append(backup)

    print(f"\n总备份数: {len(backups)}")
    print(f"分类:")
    for btype, files in sorted(backup_types.items()):
        print(f"  - {btype}: {len(files)}个")

    # 分析时间戳备份
    if backup_types["timestamped"]:
        print(f"\n时间戳备份详情 ({len(backup_types['timestamped'])}个):")
        timestamped = sorted(backup_types["timestamped"], key=lambda x: x.stat().st_mtime)

        for backup in timestamped:
            info = get_file_info(backup)
            print(f"  {backup.name}")
            print(f"    - 时间: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    - 大小: {info['size']:,} bytes ({info['size']/1024:.1f} KB)")
            print(f"    - 行数: {info['rows']}")

    # 分析关键备份
    print(f"\n关键备份:")
    for btype in ["phase", "fix", "dedup", "80col"]:
        if backup_types[btype]:
            print(f"\n  {btype.upper()}类型:")
            for backup in sorted(backup_types[btype]):
                info = get_file_info(backup)
                print(f"    {backup.name}")
                print(f"      - 时间: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"      - 行数: {info['rows']}")

def analyze_summary_archive():
    """分析summary_archive目录中的备份"""
    print("\n" + "=" * 80)
    print("分析 archived/summary_archive/ 目录")
    print("=" * 80)

    if not SUMMARY_ARCHIVE_DIR.exists():
        print("目录不存在")
        return

    backups = sorted(SUMMARY_ARCHIVE_DIR.glob("*.backup*"))
    print(f"\n总备份数: {len(backups)}")

    # 归档说明已存在
    readme = SUMMARY_ARCHIVE_DIR / "README_ARCHIVE.md"
    if readme.exists():
        print(f"✓ 归档说明文件已存在: {readme.name}")

    print(f"\n这些备份已在v4.7.3归档，属于历史数据")

def analyze_current_files():
    """分析当前主数据文件"""
    print("\n" + "=" * 80)
    print("当前主数据文件状态")
    print("=" * 80)

    # raw_data.csv
    if CURRENT_RAW_DATA.exists():
        info = get_file_info(CURRENT_RAW_DATA)
        print(f"\nraw_data.csv:")
        print(f"  - 最后修改: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - 行数: {info['rows']}")
        print(f"  - 列数: {info['columns']}")
        print(f"  - 大小: {info['size']:,} bytes ({info['size']/1024:.1f} KB)")

    # data.csv
    if CURRENT_DATA.exists():
        info = get_file_info(CURRENT_DATA)
        print(f"\ndata.csv:")
        print(f"  - 最后修改: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - 行数: {info['rows']}")
        print(f"  - 列数: {info['columns']}")
        print(f"  - 大小: {info['size']:,} bytes ({info['size']/1024:.1f} KB)")

    # 比较两个文件
    if CURRENT_RAW_DATA.exists() and CURRENT_DATA.exists():
        raw_info = get_file_info(CURRENT_RAW_DATA)
        data_info = get_file_info(CURRENT_DATA)

        print(f"\n差异分析:")
        print(f"  - 行数差异: {data_info['rows'] - raw_info['rows']}")
        print(f"  - 列数差异: {data_info['columns'] - raw_info['columns']}")

        if data_info['rows'] == raw_info['rows'] and data_info['columns'] != raw_info['columns']:
            print(f"  ⚠️ 列数不同但行数相同，可能是data.csv是raw_data.csv的子集")

def analyze_results_root_backups():
    """分析results/根目录的备份"""
    print("\n" + "=" * 80)
    print("分析 results/ 根目录的备份")
    print("=" * 80)

    root_backups = list(RESULTS_DIR.glob("*.backup*"))

    if not root_backups:
        print("\n✓ 根目录无备份文件")
        return

    print(f"\n根目录备份数: {len(root_backups)}")
    for backup in sorted(root_backups):
        info = get_file_info(backup)
        print(f"\n  {backup.name}")
        print(f"    - 时间: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    - 行数: {info['rows']}")
        print(f"    - 大小: {info['size']:,} bytes ({info['size']/1024:.1f} KB)")

def analyze_data_csv_backups():
    """分析data.csv相关备份"""
    print("\n" + "=" * 80)
    print("分析 data.csv 相关备份")
    print("=" * 80)

    data_backups = list(RESULTS_DIR.glob("data.csv.backup*"))

    if not data_backups:
        print("\n无data.csv备份")
        return

    print(f"\ndata.csv备份数: {len(data_backups)}")
    for backup in sorted(data_backups):
        info = get_file_info(backup)
        print(f"\n  {backup.name}")
        print(f"    - 时间: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    - 行数: {info['rows']}")
        print(f"    - 列数: {info['columns']}")

def generate_recommendations():
    """生成清理建议"""
    print("\n" + "=" * 80)
    print("清理建议")
    print("=" * 80)

    current_raw_info = get_file_info(CURRENT_RAW_DATA)

    print(f"\n基于当前状态 (raw_data.csv: {current_raw_info['rows']}行):")

    # 检查backup_archive中的备份
    if BACKUP_ARCHIVE_DIR.exists():
        backups = list(BACKUP_ARCHIVE_DIR.glob("*.backup*"))

        # 时间戳备份
        timestamped = [b for b in backups if not any(keyword in b.name for keyword in
                      ["before_phase", "before_fix", "before_dedup", "before_restore", "80col"])]

        print(f"\n1. backup_archive_20251219/ 中的时间戳备份 ({len(timestamped)}个):")
        print(f"   - 这些是Phase 4-7执行期间的中间备份")
        print(f"   - 当前raw_data.csv已包含所有Phase数据")
        print(f"   - 建议: 可以清理 (保留最新的1-2个即可)")

        # 关键备份
        phase_backups = [b for b in backups if "before_phase" in b.name]
        fix_backups = [b for b in backups if "before_fix" in b.name]

        if phase_backups:
            print(f"\n2. Phase相关备份 ({len(phase_backups)}个):")
            for b in sorted(phase_backups):
                print(f"   - {b.name}")
            print(f"   - 建议: 保留before_phase6 (Phase 6前的最后快照)")

        if fix_backups:
            print(f"\n3. Fix相关备份 ({len(fix_backups)}个):")
            for b in sorted(fix_backups):
                print(f"   - {b.name}")
            print(f"   - 建议: 保留最新的before_fix (修复前快照)")

    # data.csv备份
    data_backups = list(RESULTS_DIR.glob("data.csv.backup*"))
    if data_backups:
        print(f"\n4. data.csv相关备份 ({len(data_backups)}个):")
        for b in sorted(data_backups):
            print(f"   - {b.name}")
        print(f"   - data.csv与raw_data.csv的关系需要确认")
        print(f"   - 建议: 如果data.csv已废弃，可删除所有data.csv备份")

    # raw_data.csv.backup_before_clean
    before_clean = RESULTS_DIR / "raw_data.csv.backup_before_clean"
    if before_clean.exists():
        info = get_file_info(before_clean)
        print(f"\n5. raw_data.csv.backup_before_clean:")
        print(f"   - 行数: {info['rows']} (当前: {current_raw_info['rows']})")
        print(f"   - 时间: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   - 建议: {'保留 (重要清理前快照)' if info['rows'] == current_raw_info['rows'] else '检查差异后决定'}")

    # Session备份
    session_backup = RESULTS_DIR / "run_20251212_224937" / "summary.csv.backup_before_reextraction"
    if session_backup.exists():
        print(f"\n6. Session备份:")
        print(f"   - run_20251212_224937/summary.csv.backup_before_reextraction")
        print(f"   - 建议: 如果session已完成且数据已合并到raw_data.csv，可删除")

def main():
    print("备份文件分析工具")
    print("=" * 80)

    # 切换到项目目录
    os.chdir("/home/green/energy_dl/nightly")

    # 执行各项分析
    analyze_current_files()
    analyze_backup_archive()
    analyze_summary_archive()
    analyze_results_root_backups()
    analyze_data_csv_backups()
    generate_recommendations()

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
