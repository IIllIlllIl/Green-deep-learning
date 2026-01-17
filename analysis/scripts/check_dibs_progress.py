#!/usr/bin/env python3
"""
DiBS交互项分析进度监控脚本

用途: 检查后台运行的DiBS分析状态
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def check_dibs_status():
    """检查DiBS分析状态"""

    print("="*60)
    print("DiBS交互项分析状态检查")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. 检查进程
    print("\n[1] 进程状态:")
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        dibs_processes = [line for line in result.stdout.split('\n')
                         if 'run_dibs_6groups_interaction.py' in line
                         and 'grep' not in line]

        if dibs_processes:
            print(f"  ✅ DiBS进程正在运行 ({len(dibs_processes)}个)")
            for proc in dibs_processes:
                parts = proc.split()
                if len(parts) >= 2:
                    print(f"    PID: {parts[1]}, CPU: {parts[2]}%, 运行时间: {parts[9]}")
        else:
            print(f"  ⚠️  未找到DiBS进程（可能已完成或未启动）")
    except Exception as e:
        print(f"  ❌ 检查进程失败: {e}")

    # 2. 检查日志文件
    print("\n[2] 日志文件:")
    log_file = Path("dibs_interaction_run.log")
    if log_file.exists():
        size = log_file.stat().st_size
        print(f"  ✅ 日志文件存在: {log_file}")
        print(f"  文件大小: {size:,} bytes ({size/1024:.1f} KB)")

        # 读取最后30行
        with open(log_file) as f:
            lines = f.readlines()
            total_lines = len(lines)
            print(f"  总行数: {total_lines}")

            if total_lines > 0:
                print(f"\n  最新输出 (最后10行):")
                for line in lines[-10:]:
                    print(f"    {line.rstrip()}")
    else:
        print(f"  ❌ 日志文件不存在: {log_file}")

    # 3. 检查输出目录
    print("\n[3] 输出文件:")
    results_base = Path("results/energy_research/dibs_interaction")
    if results_base.exists():
        # 找到最新的时间戳目录
        run_dirs = sorted(results_base.glob("2*"), reverse=True)
        if run_dirs:
            latest_dir = run_dirs[0]
            print(f"  ✅ 最新运行目录: {latest_dir.name}")

            # 统计文件
            files = list(latest_dir.glob("*"))
            print(f"  已生成文件: {len(files)}个")

            # 按类型分组
            npy_files = [f for f in files if f.suffix == '.npy']
            json_files = [f for f in files if f.suffix == '.json']
            md_files = [f for f in files if f.suffix == '.md']

            print(f"    - 因果图 (.npy): {len(npy_files)}个")
            print(f"    - 结果 (.json): {len(json_files)}个")
            print(f"    - 报告 (.md): {len(md_files)}个")

            # 期望文件数
            expected_per_group = 3  # .npy, .json, feature_names.json
            expected_total = 6 * expected_per_group + 1  # 6组 + 1个总报告

            if len(files) >= expected_total:
                print(f"  ✅ 分析可能已完成 ({len(files)}/{expected_total}文件)")
            else:
                print(f"  ⏳ 分析进行中 ({len(files)}/{expected_total}文件)")

            # 检查是否有总报告
            report_file = latest_dir / "DIBS_INTERACTION_ANALYSIS_REPORT.md"
            if report_file.exists():
                print(f"  ✅ 总结报告已生成: {report_file.name}")
            else:
                print(f"  ⏳ 总结报告未生成（可能还在运行）")
        else:
            print(f"  ⚠️  未找到运行目录")
    else:
        print(f"  ⚠️  输出目录不存在: {results_base}")

    # 4. 估算进度
    print("\n[4] 进度估算:")
    if results_base.exists() and run_dirs:
        latest_dir = run_dirs[0]
        result_files = list(latest_dir.glob("*_result.json"))
        completed_groups = len(result_files)
        total_groups = 6

        progress_pct = (completed_groups / total_groups) * 100
        print(f"  已完成任务组: {completed_groups}/{total_groups} ({progress_pct:.1f}%)")

        if completed_groups > 0:
            print(f"  已完成的组:")
            for result_file in sorted(result_files):
                print(f"    ✅ {result_file.stem.replace('_result', '')}")

        if completed_groups < total_groups:
            remaining = total_groups - completed_groups
            print(f"  剩余: {remaining}组")

    print("\n" + "="*60)
    print("检查完成")
    print("="*60)


if __name__ == "__main__":
    check_dibs_status()
