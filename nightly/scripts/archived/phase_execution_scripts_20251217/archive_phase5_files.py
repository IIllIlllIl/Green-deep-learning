#!/usr/bin/env python3
"""
归档已完成任务的脚本和文档

功能：
1. 归档Phase 5相关分析脚本（已完成）
2. 归档一次性任务脚本（expand_raw_data_columns等）
3. 整理重复功能的脚本
4. 清理过期备份文件

版本: 1.0
创建日期: 2025-12-15
"""

import shutil
from pathlib import Path
from datetime import datetime

def create_archive_dir(base_dir: Path, archive_name: str) -> Path:
    """创建归档目录"""
    archive_dir = base_dir / "archived" / archive_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir

def archive_file(file_path: Path, archive_dir: Path):
    """归档文件"""
    if file_path.exists():
        dest = archive_dir / file_path.name
        shutil.move(str(file_path), str(dest))
        print(f"  ✓ 归档: {file_path.name} → {archive_dir.relative_to(Path.cwd())}")
        return True
    return False

def main():
    print("=" * 80)
    print("项目整理与归档 - v4.7.7")
    print("=" * 80)
    print()

    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"

    # 创建归档目录
    archive_date = datetime.now().strftime("%Y%m%d")
    archive_dir = create_archive_dir(scripts_dir, f"completed_phase5_tasks_{archive_date}")

    print("1. 归档Phase 5相关分析脚本")
    print("-" * 80)

    phase5_scripts = [
        "analyze_shared_performance_issue.py",
        "check_shared_performance_metrics.py",
        "analyze_phase5_completion.py",
        "estimate_phase5_time.py",
    ]

    archived_count = 0
    for script_name in phase5_scripts:
        script_path = scripts_dir / script_name
        if archive_file(script_path, archive_dir):
            archived_count += 1

    print(f"\n  归档 {archived_count}/{len(phase5_scripts)} 个Phase 5脚本")
    print()

    print("2. 归档一次性任务脚本")
    print("-" * 80)

    onetime_scripts = [
        "expand_raw_data_columns.py",
        "restore_and_reappend_phase5.py",
        "recalculate_num_mutated_params.py",
        "reextract_performance_metrics.py",
    ]

    archived_count = 0
    for script_name in onetime_scripts:
        script_path = scripts_dir / script_name
        if archive_file(script_path, archive_dir):
            archived_count += 1

    print(f"\n  归档 {archived_count}/{len(onetime_scripts)} 个一次性任务脚本")
    print()

    print("3. 标记重复功能脚本（手动整合）")
    print("-" * 80)

    duplicate_note = """
重复功能脚本（建议整合）：

数据追加功能：
- add_new_experiments_to_raw_data.py (189行)
- append_session_to_raw_data.py (476行) ← 保留（功能最完整）

建议：将add_new_experiments_to_raw_data.py的功能整合到append_session_to_raw_data.py
    """

    print(duplicate_note)

    # 创建说明文件
    readme_path = archive_dir / "README_ARCHIVE.md"
    with open(readme_path, 'w') as f:
        f.write(f"# 归档文件说明\n\n")
        f.write(f"**归档日期**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**版本**: v4.7.7\n\n")
        f.write(f"## Phase 5相关脚本\n\n")
        for script in phase5_scripts:
            f.write(f"- `{script}` - Phase 5数据分析任务已完成\n")
        f.write(f"\n## 一次性任务脚本\n\n")
        for script in onetime_scripts:
            f.write(f"- `{script}` - 一次性数据处理任务已完成\n")
        f.write(f"\n## 说明\n\n")
        f.write(f"这些脚本已完成其设计目的，归档保存以备参考。\n")
        f.write(f"如需重新使用，可从归档目录复制回scripts/目录。\n")

    print(f"\n  ✓ 创建归档说明: {readme_path.relative_to(project_root)}")
    print()

    print("4. 检查results/目录中的备份文件")
    print("-" * 80)

    results_dir = project_root / "results"

    # 统计备份文件
    backup_files = list(results_dir.glob("*.backup_*"))
    bak_files = list(results_dir.glob("*.bak"))

    print(f"  发现 {len(backup_files)} 个 .backup_* 文件")
    print(f"  发现 {len(bak_files)} 个 .bak 文件")

    if backup_files or bak_files:
        print(f"\n  建议手动清理7天前的备份文件:")
        print(f"    cd results && find . -name '*.backup_*' -mtime +7 -ls")
        print(f"    cd results && find . -name '*.bak' -mtime +7 -ls")
    print()

    print("=" * 80)
    print("归档完成")
    print("=" * 80)
    print()

    print(f"归档目录: {archive_dir.relative_to(project_root)}")
    print(f"归档文件数: {len(list(archive_dir.glob('*.py')))} 个")
    print()

    print("下一步:")
    print("  1. 手动整合重复功能脚本")
    print("  2. 清理过期备份文件（谨慎操作）")
    print("  3. 更新文档（README.md, CLAUDE.md）")
    print()

if __name__ == "__main__":
    main()
