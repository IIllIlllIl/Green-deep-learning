#!/usr/bin/env python3

"""
路径更新脚本
日期: 2026-01-05
用途: 自动更新所有文件中的路径引用
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 项目根目录
PROJECT_ROOT = Path("/home/green/energy_dl/nightly")

# 路径映射表
PATH_MAPPINGS = {
    # 数据文件路径
    'results/raw_data.csv': 'data/raw_data.csv',
    'results/data.csv': 'data/data.csv',
    'results/recoverable_energy_data.json': 'data/recoverable_energy_data.json',

    # 相对路径（从不同层级）
    '../results/raw_data.csv': '../data/raw_data.csv',
    '../../results/raw_data.csv': '../../data/raw_data.csv',
    '../../../results/raw_data.csv': '../../../data/raw_data.csv',

    '../results/data.csv': '../data/data.csv',
    '../../results/data.csv': '../../data/data.csv',

    # 脚本路径
    'scripts/validate_raw_data.py': 'tools/data_management/validate_raw_data.py',
    'scripts/analyze_experiment_status.py': 'tools/data_management/analyze_experiment_status.py',
    'scripts/analyze_missing_energy_data.py': 'tools/data_management/analyze_missing_energy_data.py',
    'scripts/repair_missing_energy_data.py': 'tools/data_management/repair_missing_energy_data.py',
    'scripts/verify_recoverable_data.py': 'tools/data_management/verify_recoverable_data.py',
    'scripts/append_session_to_raw_data.py': 'tools/data_management/append_session_to_raw_data.py',
    'scripts/compare_data_vs_raw_data.py': 'tools/data_management/compare_data_vs_raw_data.py',
    'scripts/create_unified_data_csv.py': 'tools/data_management/create_unified_data_csv.py',
    'scripts/generate_mutation_config.py': 'tools/config_management/generate_mutation_config.py',
    'scripts/validate_mutation_config.py': 'tools/config_management/validate_mutation_config.py',
}

# 需要扫描的文件类型
SCAN_EXTENSIONS = ['.py', '.md', '.sh', '.json', '.txt']

# 需要扫描的目录
SCAN_DIRS = [
    'tools',
    'analysis',
    'docs',
    'tests',
    'mutation',
    'settings',
]

# 排除的目录
EXCLUDE_DIRS = [
    '__pycache__',
    '.git',
    'repos',
    'archives',
    'environment',
]


class PathUpdater:
    """路径更新器"""

    def __init__(self, root: Path, dry_run: bool = True):
        self.root = root
        self.dry_run = dry_run
        self.updated_files = []
        self.errors = []

    def should_scan_file(self, file_path: Path) -> bool:
        """检查文件是否应该扫描"""
        # 检查扩展名
        if file_path.suffix not in SCAN_EXTENSIONS:
            return False

        # 检查是否在排除目录中
        for exclude_dir in EXCLUDE_DIRS:
            if exclude_dir in file_path.parts:
                return False

        return True

    def update_file_paths(self, file_path: Path) -> Tuple[int, List[str]]:
        """更新文件中的路径"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"读取失败 {file_path}: {e}")
            return 0, []

        original_content = content
        changes = []

        # 对每个路径映射进行替换
        for old_path, new_path in PATH_MAPPINGS.items():
            # 直接字符串替换
            if old_path in content:
                content = content.replace(old_path, new_path)
                changes.append(f"{old_path} → {new_path}")

        # 如果有变更
        if content != original_content:
            if not self.dry_run:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                except Exception as e:
                    self.errors.append(f"写入失败 {file_path}: {e}")
                    return 0, []

            self.updated_files.append(file_path)
            return len(changes), changes

        return 0, []

    def scan_and_update(self) -> None:
        """扫描并更新所有文件"""
        print(f"{'[DRY RUN] ' if self.dry_run else ''}开始扫描项目文件...\n")

        total_files = 0
        updated_count = 0

        for scan_dir in SCAN_DIRS:
            dir_path = self.root / scan_dir
            if not dir_path.exists():
                print(f"⚠️  目录不存在: {scan_dir}")
                continue

            print(f"📁 扫描目录: {scan_dir}/")

            for file_path in dir_path.rglob('*'):
                if not file_path.is_file():
                    continue

                if not self.should_scan_file(file_path):
                    continue

                total_files += 1

                num_changes, changes = self.update_file_paths(file_path)

                if num_changes > 0:
                    updated_count += 1
                    rel_path = file_path.relative_to(self.root)
                    print(f"\n✅ {rel_path}")
                    for change in changes:
                        print(f"   - {change}")

        # 显示总结
        print("\n" + "="*60)
        print("总结")
        print("="*60)
        print(f"扫描文件数: {total_files}")
        print(f"更新文件数: {updated_count}")

        if self.errors:
            print(f"\n❌ 错误数: {len(self.errors)}")
            for error in self.errors:
                print(f"   - {error}")

        if self.dry_run:
            print("\n⚠️  这是DRY RUN模式，未实际修改文件")
            print("   使用 --execute 参数来实际执行修改")
        else:
            print("\n✅ 所有文件已更新！")

    def generate_report(self) -> str:
        """生成更新报告"""
        report = []
        report.append("# 路径更新报告\n")
        report.append(f"**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**模式**: {'DRY RUN' if self.dry_run else 'EXECUTE'}\n")
        report.append("\n## 更新的文件列表\n")

        for file_path in self.updated_files:
            rel_path = file_path.relative_to(self.root)
            report.append(f"- {rel_path}\n")

        if self.errors:
            report.append("\n## 错误列表\n")
            for error in self.errors:
                report.append(f"- {error}\n")

        return ''.join(report)


def main():
    """主函数"""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description='更新项目文件中的路径引用')
    parser.add_argument('--execute', action='store_true',
                       help='实际执行修改（默认为dry-run）')
    parser.add_argument('--report', type=str,
                       help='保存报告到指定文件')

    args = parser.parse_args()

    # 创建更新器
    updater = PathUpdater(PROJECT_ROOT, dry_run=not args.execute)

    # 执行扫描和更新
    updater.scan_and_update()

    # 生成报告
    if args.report:
        report = updater.generate_report()
        report_path = PROJECT_ROOT / args.report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 报告已保存: {report_path}")

    # 返回状态码
    return 0 if not updater.errors else 1


if __name__ == '__main__':
    sys.exit(main())
