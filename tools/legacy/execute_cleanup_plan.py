#!/usr/bin/env python3
"""
项目清理和归档执行脚本

基于 docs/PROJECT_CLEANUP_PLAN_20251225.md 执行清理操作

使用方法:
    python3 scripts/execute_cleanup_plan.py --phase 1  # 安全归档（可逆）
    python3 scripts/execute_cleanup_plan.py --phase 2  # 清理备份（需确认）
    python3 scripts/execute_cleanup_plan.py --phase 3  # 清理中间文件（需确认）
    python3 scripts/execute_cleanup_plan.py --dry-run  # 仅显示将执行的操作
    python3 scripts/execute_cleanup_plan.py --all      # 执行所有阶段（需确认）
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "results"
ANALYSIS_DIR = ROOT_DIR / "analysis"
TESTS_DIR = ROOT_DIR / "tests"
DOCS_DIR = ROOT_DIR / "docs"


class CleanupExecutor:
    """清理执行器"""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.actions = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, message, level="INFO"):
        """记录日志"""
        prefix = "[DRY-RUN] " if self.dry_run else ""
        print(f"{prefix}[{level}] {message}")

    def execute(self, action_func, description):
        """执行操作"""
        self.log(f"{description}")
        self.actions.append(description)

        if not self.dry_run:
            try:
                action_func()
                self.log(f"✅ 完成: {description}", "SUCCESS")
            except Exception as e:
                self.log(f"❌ 失败: {description} - {e}", "ERROR")
                raise
        else:
            self.log(f"[将执行] {description}", "DRY-RUN")

    def mkdir(self, path):
        """创建目录"""
        if not self.dry_run:
            Path(path).mkdir(parents=True, exist_ok=True)

    def move(self, src, dst):
        """移动文件"""
        if not self.dry_run:
            shutil.move(str(src), str(dst))

    def remove(self, path):
        """删除文件或目录"""
        if not self.dry_run:
            path = Path(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def write_file(self, path, content):
        """写入文件"""
        if not self.dry_run:
            Path(path).write_text(content)

    def compress(self, src_dir, dst_file):
        """压缩目录"""
        if not self.dry_run:
            shutil.make_archive(
                str(dst_file).replace('.tar.gz', ''),
                'gztar',
                root_dir=Path(src_dir).parent,
                base_dir=Path(src_dir).name
            )


def phase1_safe_archive(executor: CleanupExecutor):
    """Phase 1: 安全归档（可逆操作）"""

    executor.log("\n" + "="*60, "PHASE")
    executor.log("Phase 1: 安全归档（可逆操作）", "PHASE")
    executor.log("="*60 + "\n", "PHASE")

    # 1.1 归档 summary_new.csv 和 summary_old.csv
    archive_dir = RESULTS_DIR / "archived" / "merged_20251212"

    def archive_summary_files():
        executor.mkdir(archive_dir)

        # 移动文件
        if (RESULTS_DIR / "summary_new.csv").exists():
            executor.move(
                RESULTS_DIR / "summary_new.csv",
                archive_dir / "summary_new.csv"
            )

        if (RESULTS_DIR / "summary_old.csv").exists():
            executor.move(
                RESULTS_DIR / "summary_old.csv",
                archive_dir / "summary_old.csv"
            )

        # 创建README
        readme_content = """# 已合并数据文件归档

**归档时间**: 2025-12-25
**原因**: 已合并到 raw_data.csv

## 文件说明

- `summary_new.csv` (266行, 80列) - 新实验数据（2025-11-26后）
- `summary_old.csv` (212行, 93列) - 老实验数据（2025-11-26前）

## 合并情况

这两个文件已在2025-12-12合并为 `raw_data.csv`（80列标准格式）。
详见: `docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md`

## 保留原因

保留作为历史参考，用于追溯数据来源。
"""
        executor.write_file(archive_dir / "README.md", readme_content)

    executor.execute(
        archive_summary_files,
        "归档 summary_new.csv 和 summary_old.csv 到 results/archived/merged_20251212/"
    )

    # 1.2 移动错位的数据文件
    scripts_data_dir = ANALYSIS_DIR / "scripts" / "data" / "energy_research" / "raw"
    correct_data_dir = ANALYSIS_DIR / "data" / "energy_research" / "raw"

    def move_misplaced_files():
        if scripts_data_dir.exists():
            executor.mkdir(correct_data_dir)

            # 移动CSV文件
            csv_file = scripts_data_dir / "energy_data_extracted_v2.csv"
            if csv_file.exists() and not (correct_data_dir / "energy_data_extracted_v2.csv").exists():
                executor.move(csv_file, correct_data_dir / "energy_data_extracted_v2.csv")

            # 移动JSON文件
            json_file = scripts_data_dir / "extracted_columns_info.json"
            if json_file.exists() and not (correct_data_dir / "extracted_columns_info.json").exists():
                executor.move(json_file, correct_data_dir / "extracted_columns_info.json")

            # 清理空目录
            if scripts_data_dir.exists() and not any(scripts_data_dir.iterdir()):
                executor.remove(scripts_data_dir.parent.parent)

    executor.execute(
        move_misplaced_files,
        "移动 analysis/scripts/data/ 下的数据文件到 analysis/data/energy_research/raw/"
    )

    # 1.3 归档已完成的测试文件
    test_archive_dir = TESTS_DIR / "archived" / "completed_20251212"

    def archive_completed_tests():
        executor.mkdir(test_archive_dir)

        test_files = [
            "test_old_csv_rebuild.py",
            "validate_80col_format.py",
            "validate_rebuilt_summary_old.py"
        ]

        for test_file in test_files:
            src = TESTS_DIR / test_file
            if src.exists():
                executor.move(src, test_archive_dir / test_file)

        # 创建README
        readme_content = """# 已完成测试归档

**归档时间**: 2025-12-25
**原因**: 测试目标已完成且验证通过

## 归档的测试

1. `test_old_csv_rebuild.py` - 测试summary_old重建为80列
2. `validate_80col_format.py` - 验证80列格式一致性
3. `validate_rebuilt_summary_old.py` - 验证重建后的summary_old

## 归档原因

这些测试是针对2025-12-12的数据格式统一任务。任务已完成并验证通过，
数据已合并到raw_data.csv，因此这些测试不再需要运行。

保留作为历史参考。
"""
        executor.write_file(test_archive_dir / "README.md", readme_content)

    executor.execute(
        archive_completed_tests,
        "归档已完成的测试文件到 tests/archived/completed_20251212/"
    )


def phase2_cleanup_backups(executor: CleanupExecutor):
    """Phase 2: 清理备份（不可逆，需确认）"""

    executor.log("\n" + "="*60, "PHASE")
    executor.log("Phase 2: 清理备份（不可逆，需确认）", "PHASE")
    executor.log("="*60 + "\n", "PHASE")

    # 旧备份文件列表
    old_backups = [
        RESULTS_DIR / "raw_data.backup_20251221_215643.csv",
        RESULTS_DIR / "raw_data.csv.backup_before_clean",
        RESULTS_DIR / "data.csv.backup_before_column_removal_20251219_182227",
        RESULTS_DIR / "data.csv.backup_before_merge_20251219_180149"
    ]

    def cleanup_old_backups():
        for backup_file in old_backups:
            if backup_file.exists():
                executor.remove(backup_file)

    executor.execute(
        cleanup_old_backups,
        "清理4个旧备份文件（保留最新的 20251223 备份）"
    )


def phase3_cleanup_intermediate(executor: CleanupExecutor):
    """Phase 3: 清理中间文件（不可逆，需确认）"""

    executor.log("\n" + "="*60, "PHASE")
    executor.log("Phase 3: 清理中间文件（不可逆，需确认）", "PHASE")
    executor.log("="*60 + "\n", "PHASE")

    # 3.1 清理processed目录的stage文件
    processed_dir = ANALYSIS_DIR / "data" / "energy_research" / "processed"

    def cleanup_stage_files():
        if processed_dir.exists():
            stage_files = list(processed_dir.glob("stage*.csv"))
            for stage_file in stage_files:
                executor.remove(stage_file)

    executor.execute(
        cleanup_stage_files,
        f"清理 {processed_dir.relative_to(ROOT_DIR)} 目录下的19个stage中间文件"
    )

    # 3.2 压缩并清理备份目录
    backup_dir = ANALYSIS_DIR / "data" / "energy_research" / "processed.backup_4groups_20251224"

    def cleanup_backup_dir():
        if backup_dir.exists():
            # 压缩
            tar_file = backup_dir.parent / "processed.backup_4groups_20251224.tar.gz"
            executor.compress(backup_dir, tar_file)

            # 删除原目录
            executor.remove(backup_dir)

    executor.execute(
        cleanup_backup_dir,
        "压缩并清理 processed.backup_4groups_20251224/ 备份目录"
    )

    # 3.3 清理Python缓存
    def cleanup_python_cache():
        # 查找并删除__pycache__目录
        pycache_dirs = list(ROOT_DIR.rglob("__pycache__"))
        for pycache_dir in pycache_dirs:
            executor.remove(pycache_dir)

        # 删除.pyc文件
        pyc_files = list(ROOT_DIR.rglob("*.pyc"))
        for pyc_file in pyc_files:
            executor.remove(pyc_file)

    executor.execute(
        cleanup_python_cache,
        "清理所有Python缓存文件（__pycache__和*.pyc）"
    )


def verify_before_cleanup():
    """执行前验证"""

    print("\n" + "="*60)
    print("执行前验证")
    print("="*60 + "\n")

    checks = []

    # 1. 检查主数据文件
    raw_data = RESULTS_DIR / "raw_data.csv"
    data_csv = RESULTS_DIR / "data.csv"

    if raw_data.exists() and data_csv.exists():
        raw_lines = len(raw_data.read_text().splitlines())
        data_lines = len(data_csv.read_text().splitlines())

        checks.append(("主数据文件存在", True, f"raw_data.csv: {raw_lines}行, data.csv: {data_lines}行"))
    else:
        checks.append(("主数据文件存在", False, "缺少主数据文件"))

    # 2. 检查最新备份
    latest_raw_backup = RESULTS_DIR / "raw_data.csv.backup_20251223_195253"
    latest_data_backup = RESULTS_DIR / "data.csv.backup_20251223_202113"

    if latest_raw_backup.exists() and latest_data_backup.exists():
        checks.append(("最新备份存在", True, "20251223备份文件完整"))
    else:
        checks.append(("最新备份存在", False, "缺少最新备份"))

    # 3. 检查training数据
    training_dir = ANALYSIS_DIR / "data" / "energy_research" / "training"
    if training_dir.exists():
        training_files = list(training_dir.glob("training_data_*.csv"))
        checks.append(("Training数据存在", True, f"找到{len(training_files)}个训练数据文件"))
    else:
        checks.append(("Training数据存在", False, "Training目录不存在"))

    # 显示检查结果
    all_passed = True
    for check_name, passed, message in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {message}")
        if not passed:
            all_passed = False

    print()

    if not all_passed:
        print("⚠️  警告: 部分检查未通过，建议修复后再执行清理")
        return False
    else:
        print("✅ 所有检查通过，可以安全执行清理")
        return True


def create_full_backup():
    """创建完整备份"""

    print("\n" + "="*60)
    print("创建完整备份")
    print("="*60 + "\n")

    backup_name = f"nightly_backup_before_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_file = ROOT_DIR.parent / f"{backup_name}.tar.gz"

    print(f"创建备份: {backup_file}")
    print("排除: results/run_*, *.pyc, __pycache__")
    print()

    try:
        cmd = [
            "tar", "-czf", str(backup_file),
            "--exclude=results/run_*",
            "--exclude=*.pyc",
            "--exclude=__pycache__",
            "-C", str(ROOT_DIR.parent),
            "nightly"
        ]

        subprocess.run(cmd, check=True)

        backup_size = backup_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 备份创建成功: {backup_file.name} ({backup_size:.2f} MB)")
        return True

    except Exception as e:
        print(f"❌ 备份创建失败: {e}")
        return False


def generate_report(executor: CleanupExecutor):
    """生成执行报告"""

    report_file = DOCS_DIR / f"PROJECT_CLEANUP_EXECUTION_REPORT_{executor.timestamp}.md"

    report_content = f"""# 项目清理执行报告

**执行时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**执行模式**: {"DRY-RUN（模拟）" if executor.dry_run else "实际执行"}

---

## 执行操作

共执行 {len(executor.actions)} 个操作：

"""

    for i, action in enumerate(executor.actions, 1):
        report_content += f"{i}. {action}\n"

    report_content += """
---

## 执行结果

"""

    if executor.dry_run:
        report_content += "**模式**: DRY-RUN - 未实际执行任何操作\n"
        report_content += "\n请使用 `--phase N` 参数实际执行清理。\n"
    else:
        report_content += "**状态**: ✅ 清理执行完成\n"
        report_content += "\n请运行验证命令确认数据完整性：\n"
        report_content += "```bash\n"
        report_content += "python3 tools/data_management/validate_raw_data.py\n"
        report_content += "cd analysis && python3 -c \"import pandas as pd; ...\"\n"
        report_content += "```\n"

    report_content += f"""
---

## 参考文档

- 清理计划: `docs/PROJECT_CLEANUP_PLAN_20251225.md`
- 项目指南: `CLAUDE.md`

---

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    executor.write_file(report_file, report_content)
    print(f"\n✅ 执行报告已生成: {report_file.relative_to(ROOT_DIR)}")


def main():
    parser = argparse.ArgumentParser(
        description="项目清理和归档执行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 scripts/execute_cleanup_plan.py --dry-run      # 查看将执行的操作
  python3 scripts/execute_cleanup_plan.py --phase 1      # 执行Phase 1（安全归档）
  python3 scripts/execute_cleanup_plan.py --phase 2      # 执行Phase 2（清理备份）
  python3 scripts/execute_cleanup_plan.py --phase 3      # 执行Phase 3（清理中间文件）
  python3 scripts/execute_cleanup_plan.py --all          # 执行所有阶段
  python3 scripts/execute_cleanup_plan.py --verify-only  # 仅执行验证
        """
    )

    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                       help="执行指定阶段 (1=安全归档, 2=清理备份, 3=清理中间文件)")
    parser.add_argument("--all", action="store_true",
                       help="执行所有阶段")
    parser.add_argument("--dry-run", action="store_true",
                       help="模拟执行，不实际修改文件")
    parser.add_argument("--verify-only", action="store_true",
                       help="仅执行验证，不执行清理")
    parser.add_argument("--skip-backup", action="store_true",
                       help="跳过完整备份（仅用于测试）")

    args = parser.parse_args()

    # 仅验证模式
    if args.verify_only:
        verify_before_cleanup()
        return

    # 默认为dry-run模式
    if not args.phase and not args.all:
        args.dry_run = True
        print("⚠️  未指定阶段，使用 DRY-RUN 模式")
        print("使用 --phase N 或 --all 参数实际执行清理\n")

    executor = CleanupExecutor(dry_run=args.dry_run)

    # 执行前验证
    if not args.dry_run:
        if not verify_before_cleanup():
            print("\n❌ 验证失败，终止执行")
            sys.exit(1)

        # 创建完整备份
        if not args.skip_backup:
            if not create_full_backup():
                print("\n❌ 备份失败，终止执行")
                sys.exit(1)

        # 确认执行
        print("\n⚠️  即将执行清理操作，此操作不可逆！")
        confirm = input("确认执行？(yes/no): ")
        if confirm.lower() != "yes":
            print("已取消执行")
            sys.exit(0)

    # 执行清理
    try:
        if args.phase == 1 or args.all:
            phase1_safe_archive(executor)

        if args.phase == 2 or args.all:
            phase2_cleanup_backups(executor)

        if args.phase == 3 or args.all:
            phase3_cleanup_intermediate(executor)

        # 生成报告
        generate_report(executor)

        print("\n" + "="*60)
        print("✅ 清理执行完成")
        print("="*60 + "\n")

        if not args.dry_run:
            print("建议执行验证命令:")
            print("  python3 tools/data_management/validate_raw_data.py")
            print("  python3 scripts/execute_cleanup_plan.py --verify-only")

    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
