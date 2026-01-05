#!/usr/bin/env python3
"""
默认值基线测试结果分析脚本

功能:
1. 读取 summary.csv 并生成统计报告
2. 对比顺序vs并行训练的性能和能耗
3. 识别高能耗和低效率模型
4. 生成排序表格（按能耗、时长、准确率等）

使用方法:
    python3 analyze_baseline.py [results_dir]

默认分析目录: results/default_baseline_11models/
"""

import sys
import os
import csv
from pathlib import Path
from collections import defaultdict

def read_summary_csv(csv_path):
    """读取 summary.csv 文件"""
    experiments = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append(row)
    return experiments

def parse_experiment_name(exp_id):
    """解析实验名称，提取模型和模式"""
    parts = exp_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].endswith('_parallel'):
        mode = 'parallel'
        model_parts = parts[0]
    elif len(parts) == 2 and parts[1].isdigit():
        mode = 'sequential'
        model_parts = parts[0]
    else:
        mode = 'sequential'
        model_parts = exp_id

    return {
        'model': model_parts,
        'mode': mode,
        'exp_id': exp_id
    }

def format_duration(seconds):
    """格式化时长为 小时:分钟:秒"""
    try:
        s = float(seconds)
        hours = int(s // 3600)
        minutes = int((s % 3600) // 60)
        secs = int(s % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    except:
        return "N/A"

def format_energy(joules):
    """格式化能耗为 Wh"""
    try:
        j = float(joules)
        wh = j / 3600.0
        return f"{wh:.2f} Wh"
    except:
        return "N/A"

def safe_float(value, default=0.0):
    """安全地将字符串转换为浮点数"""
    try:
        return float(value) if value else default
    except:
        return default

def analyze_baseline(results_dir):
    """分析基线测试结果"""
    csv_path = Path(results_dir) / "summary.csv"

    if not csv_path.exists():
        print(f"❌ 错误: 找不到文件 {csv_path}")
        return

    print(f"📊 分析基线测试结果: {results_dir}\n")
    print("=" * 80)

    # 读取数据
    experiments = read_summary_csv(csv_path)
    total = len(experiments)

    # 基本统计
    print(f"\n📈 基本统计")
    print(f"{'总实验数:':<20} {total}")

    success_count = sum(1 for exp in experiments if exp['training_success'] == 'True')
    print(f"{'成功:':<20} {success_count}/{total} ({success_count/total*100:.1f}%)")

    total_duration = sum(safe_float(exp['duration_seconds']) for exp in experiments)
    print(f"{'总时长:':<20} {format_duration(total_duration)}")

    # 按模式分组
    sequential = [exp for exp in experiments if '_parallel' not in exp['experiment_id']]
    parallel = [exp for exp in experiments if '_parallel' in exp['experiment_id']]

    print(f"\n{'顺序训练实验:':<20} {len(sequential)}")
    print(f"{'并行训练实验:':<20} {len(parallel)}")

    # 能耗统计
    print(f"\n⚡ 能耗统计")
    print("-" * 80)

    total_gpu_energy = sum(safe_float(exp['energy_gpu_total_joules']) for exp in experiments)
    total_cpu_energy = sum(safe_float(exp['energy_cpu_total_joules']) for exp in experiments)

    seq_gpu_energy = sum(safe_float(exp['energy_gpu_total_joules']) for exp in sequential)
    seq_cpu_energy = sum(safe_float(exp['energy_cpu_total_joules']) for exp in sequential)

    par_gpu_energy = sum(safe_float(exp['energy_gpu_total_joules']) for exp in parallel)
    par_cpu_energy = sum(safe_float(exp['energy_cpu_total_joules']) for exp in parallel)

    print(f"{'总GPU能耗:':<20} {format_energy(total_gpu_energy)}")
    print(f"{'总CPU能耗:':<20} {format_energy(total_cpu_energy)}")
    print(f"{'总能耗:':<20} {format_energy(total_gpu_energy + total_cpu_energy)}")
    print()
    print(f"{'顺序GPU能耗:':<20} {format_energy(seq_gpu_energy)}")
    print(f"{'顺序CPU能耗:':<20} {format_energy(seq_cpu_energy)}")
    print(f"{'顺序总能耗:':<20} {format_energy(seq_gpu_energy + seq_cpu_energy)}")
    print()
    print(f"{'并行GPU能耗:':<20} {format_energy(par_gpu_energy)}")
    print(f"{'并行CPU能耗:':<20} {format_energy(par_cpu_energy)}")
    print(f"{'并行总能耗:':<20} {format_energy(par_gpu_energy + par_cpu_energy)}")

    if seq_gpu_energy > 0:
        gpu_increase = (par_gpu_energy - seq_gpu_energy) / seq_gpu_energy * 100
        total_increase = ((par_gpu_energy + par_cpu_energy) - (seq_gpu_energy + seq_cpu_energy)) / (seq_gpu_energy + seq_cpu_energy) * 100
        print(f"\n{'并行GPU能耗增加:':<20} {gpu_increase:+.1f}%")
        print(f"{'并行总能耗增加:':<20} {total_increase:+.1f}%")

    # GPU能耗排名 Top 10
    print(f"\n🔥 GPU能耗排名 (Top 10)")
    print("-" * 80)
    sorted_by_gpu = sorted(experiments,
                          key=lambda x: safe_float(x['energy_gpu_total_joules']),
                          reverse=True)

    print(f"{'排名':<5} {'实验ID':<50} {'GPU能耗':<15} {'时长':<12}")
    print("-" * 80)
    for i, exp in enumerate(sorted_by_gpu[:10], 1):
        exp_id = exp['experiment_id']
        gpu_energy = format_energy(exp['energy_gpu_total_joules'])
        duration = format_duration(exp['duration_seconds'])
        print(f"{i:<5} {exp_id:<50} {gpu_energy:<15} {duration:<12}")

    # 时长排名 Top 10
    print(f"\n⏱️  运行时长排名 (Top 10)")
    print("-" * 80)
    sorted_by_duration = sorted(experiments,
                                key=lambda x: safe_float(x['duration_seconds']),
                                reverse=True)

    print(f"{'排名':<5} {'实验ID':<50} {'时长':<15} {'GPU能耗':<12}")
    print("-" * 80)
    for i, exp in enumerate(sorted_by_duration[:10], 1):
        exp_id = exp['experiment_id']
        duration = format_duration(exp['duration_seconds'])
        gpu_energy = format_energy(exp['energy_gpu_total_joules'])
        print(f"{i:<5} {exp_id:<50} {duration:<15} {gpu_energy:<12}")

    # GPU利用率统计
    print(f"\n📊 GPU利用率统计")
    print("-" * 80)
    sorted_by_util = sorted(experiments,
                           key=lambda x: safe_float(x['energy_gpu_util_avg_percent']),
                           reverse=True)

    print(f"{'排名':<5} {'实验ID':<50} {'平均利用率':<12} {'最大利用率':<12}")
    print("-" * 80)
    for i, exp in enumerate(sorted_by_util[:10], 1):
        exp_id = exp['experiment_id']
        avg_util = safe_float(exp['energy_gpu_util_avg_percent'])
        max_util = safe_float(exp['energy_gpu_util_max_percent'])
        print(f"{i:<5} {exp_id:<50} {avg_util:>6.1f}%     {max_util:>6.0f}%")

    # 性能指标（如果有）
    print(f"\n🎯 性能指标")
    print("-" * 80)

    # 检查有准确率的实验（排除MRT-OAST的特殊指标）
    with_accuracy = [exp for exp in experiments
                     if exp.get('perf_accuracy')
                     and safe_float(exp['perf_accuracy']) > 0
                     and safe_float(exp['perf_accuracy']) < 100]  # 排除大于100的异常值
    if with_accuracy:
        print("\n分类任务准确率:")
        print(f"{'实验ID':<50} {'准确率':<10}")
        print("-" * 60)
        for exp in sorted(with_accuracy, key=lambda x: safe_float(x['perf_accuracy']), reverse=True):
            acc = safe_float(exp['perf_accuracy'])
            print(f"{exp['experiment_id']:<50} {acc:>7.2f}%")

    # Person Re-ID mAP (数值已经是百分比，0-1范围需要×100)
    with_map = [exp for exp in experiments if exp.get('perf_map') and safe_float(exp['perf_map']) > 0]
    if with_map:
        print("\nPerson Re-ID:")
        print(f"{'实验ID':<50} {'mAP':<10} {'Rank-1':<10} {'Rank-5':<10}")
        print("-" * 80)
        for exp in sorted(with_map, key=lambda x: safe_float(x['perf_map']), reverse=True):
            # 这些值已经是小数形式(0-1)，需要乘以100
            map_val = safe_float(exp['perf_map']) * 100
            rank1 = safe_float(exp['perf_rank1']) * 100
            rank5 = safe_float(exp['perf_rank5']) * 100
            print(f"{exp['experiment_id']:<50} {map_val:>6.2f}%   {rank1:>6.2f}%   {rank5:>6.2f}%")

    # 温度统计
    print(f"\n🌡️  GPU温度统计")
    print("-" * 80)
    avg_temps = [safe_float(exp['energy_gpu_temp_avg_celsius']) for exp in experiments if exp.get('energy_gpu_temp_avg_celsius')]
    max_temps = [safe_float(exp['energy_gpu_temp_max_celsius']) for exp in experiments if exp.get('energy_gpu_temp_max_celsius')]

    if avg_temps and max_temps:
        print(f"{'平均温度范围:':<20} {min(avg_temps):.1f}°C - {max(avg_temps):.1f}°C")
        print(f"{'最高温度范围:':<20} {min(max_temps):.1f}°C - {max(max_temps):.1f}°C")
        print(f"{'总体平均温度:':<20} {sum(avg_temps)/len(avg_temps):.1f}°C")

    print("\n" + "=" * 80)
    print("✅ 分析完成\n")

if __name__ == "__main__":
    # 默认目录
    default_dir = "results/default_baseline_11models"

    # 从命令行参数获取目录，或使用默认值
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = default_dir

    # 检查是否在正确的工作目录
    if not Path(results_dir).exists() and Path("../"+results_dir).exists():
        results_dir = "../" + results_dir

    analyze_baseline(results_dir)
