#!/usr/bin/env python3
"""
归档summary文件并清理过时备份

功能:
1. 创建 results/summary_archive/ 目录
2. 移动所有过时的summary文件到归档目录
3. 保留必要的备份文件
4. 生成归档清单
"""

import shutil
from pathlib import Path
from datetime import datetime

def archive_summary_files():
    """归档summary文件"""
    results_dir = Path('/home/green/energy_dl/nightly/results')
    archive_dir = results_dir / 'summary_archive'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建归档目录
    archive_dir.mkdir(exist_ok=True)
    print(f"✓ 创建归档目录: {archive_dir}")

    # 定义要归档的文件
    files_to_archive = {
        # 过时的summary文件
        'summary_all.csv': '历史汇总文件（已被raw_data.csv替代）',
        'summary_all_enhanced.csv': '增强版汇总文件（已废弃）',
        'summary_all_reorganized.csv': '重组版汇总文件（已废弃）',
        'summary_new_old_separation.csv': '临时分离文件（已废弃）',
        'summary_old_93col.csv': '93列格式文件（已转换为80列）',

        # 保留的文件（记录但不移动）
        # 'summary_old.csv': '老实验数据（93列） - 源数据，保留',
        # 'summary_new.csv': '新实验数据（80列） - 源数据，保留',
        # 'raw_data.csv': '合并后的原始数据（80列） - 主文件，保留',
    }

    # 要清理的备份文件模式
    backups_to_clean = {
        'summary_all.csv.backup_20251211_144013': '旧版汇总备份',
        'summary_all.csv.backup_before_reorganization_20251211_153625': '重组前备份',
        'summary_all_enhanced.csv.backup_before_add_3cols': '增强版备份',
        'summary_new.csv.backup_step5': '新数据临时备份',
        'summary_old_80col.csv.backup_step5': '80列转换备份',
        'summary_old.csv.backup_20251212_163203': '旧版备份1',
        'summary_old.csv.backup_20251212_174304': '旧版备份2',
        'summary_old.csv.backup_20251212_194255': '旧版备份3',

        # 保留的重要备份
        # 'summary_old.csv.backup_80col': '80列原始备份 - 重要，保留',
        # 'summary_old.csv.backup_before_93col_replacement': '93列替换前备份 - 重要，保留',
    }

    # 执行归档
    archived_files = []
    kept_files = []

    print(f"\n📦 归档过时的summary文件...")

    for filename, description in files_to_archive.items():
        filepath = results_dir / filename
        if filepath.exists():
            dest = archive_dir / filename
            shutil.move(str(filepath), str(dest))
            archived_files.append((filename, description))
            print(f"  ✓ {filename} -> summary_archive/")
        else:
            print(f"  ⊘ {filename} (不存在)")

    print(f"\n🗑️  清理过时的备份文件...")

    for filename, description in backups_to_clean.items():
        filepath = results_dir / filename
        if filepath.exists():
            dest = archive_dir / filename
            shutil.move(str(filepath), str(dest))
            archived_files.append((filename, description))
            print(f"  ✓ {filename} -> summary_archive/")
        else:
            print(f"  ⊘ {filename} (不存在)")

    # 检查保留的文件
    print(f"\n✅ 保留的重要文件:")
    keep_files = [
        ('raw_data.csv', '合并后的原始数据（80列） - 主数据文件'),
        ('summary_old.csv', '老实验数据（93列） - 源数据'),
        ('summary_new.csv', '新实验数据（80列） - 源数据'),
        ('summary_old.csv.backup_80col', '80列原始备份 - 重要备份'),
        ('summary_old.csv.backup_before_93col_replacement', '93列替换前备份 - 重要备份'),
    ]

    for filename, description in keep_files:
        filepath = results_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            print(f"  ✓ {filename} ({size:.1f} KB) - {description}")
            kept_files.append((filename, description))
        else:
            print(f"  ⚠️  {filename} - {description} (不存在)")

    # 生成归档清单
    readme_path = archive_dir / 'README_ARCHIVE.md'
    with open(readme_path, 'w') as f:
        f.write(f"# Summary Files Archive\n\n")
        f.write(f"**归档日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 归档原因\n\n")
        f.write(f"随着v4.7.3版本的发布，我们完成了以下工作：\n\n")
        f.write(f"1. ✅ 合并 summary_old.csv (93列) 和 summary_new.csv (80列) 为 raw_data.csv (80列)\n")
        f.write(f"2. ✅ 验证 raw_data.csv 数据完整性和安全性 (476行, 100%完整)\n")
        f.write(f"3. ✅ 归档所有过时的summary文件和备份\n\n")
        f.write(f"## 归档文件清单\n\n")
        f.write(f"### Summary文件 ({len([f for f, d in archived_files if not 'backup' in f])}个)\n\n")

        for filename, description in archived_files:
            if 'backup' not in filename.lower():
                f.write(f"- **{filename}**: {description}\n")

        f.write(f"\n### 备份文件 ({len([f for f, d in archived_files if 'backup' in f])}个)\n\n")

        for filename, description in archived_files:
            if 'backup' in filename.lower():
                f.write(f"- **{filename}**: {description}\n")

        f.write(f"\n## 保留的文件\n\n")
        f.write(f"以下文件保留在 `results/` 目录中：\n\n")

        for filename, description in kept_files:
            f.write(f"- **{filename}**: {description}\n")

        f.write(f"\n## 数据访问\n\n")
        f.write(f"如需访问原始数据，请使用：\n\n")
        f.write(f"- **主数据文件**: `data/raw_data.csv` (476行, 80列, 100%完整)\n")
        f.write(f"- **老实验数据**: `results/summary_old.csv` (211行, 93列)\n")
        f.write(f"- **新实验数据**: `results/summary_new.csv` (265行, 80列)\n\n")
        f.write(f"## 归档文件使用\n\n")
        f.write(f"归档文件仅供历史参考，不推荐用于分析。如需恢复归档文件，请联系项目维护者。\n\n")
        f.write(f"---\n\n")
        f.write(f"**归档人**: Claude (AI助手)\n")
        f.write(f"**项目版本**: v4.7.3\n")

    print(f"\n✓ 生成归档清单: {readme_path}")

    # 统计
    print(f"\n{'='*70}")
    print(f"📊 归档统计")
    print(f"{'='*70}")
    print(f"  归档文件: {len(archived_files)}")
    print(f"  保留文件: {len(kept_files)}")
    print(f"  归档位置: {archive_dir}")
    print(f"\n✅ 归档完成")

    return len(archived_files), len(kept_files)

if __name__ == '__main__':
    archive_summary_files()
