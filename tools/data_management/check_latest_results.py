#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查最新实验结果是否完全加入数据文件

功能:
- 检查运行目录中的实验(experiment.json)是否在 raw_data.csv 中
- 验证数据属性完整性
- 生成详细检查报告
"""

import csv
import os
import json
from pathlib import Path
from collections import defaultdict

def load_experiment_ids_from_csv(csv_file):
    """从CSV文件加载所有experiment_id"""
    exp_ids = set()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = row.get('experiment_id', '').strip()
            if exp_id:
                exp_ids.add(exp_id)
    return exp_ids

def get_experiment_json_files(run_dir):
    """获取运行目录中所有experiment.json文件"""
    experiment_files = []
    run_path = Path(run_dir)

    if not run_path.exists():
        return experiment_files

    for exp_dir in run_path.iterdir():
        if exp_dir.is_dir():
            exp_json = exp_dir / "experiment.json"
            if exp_json.exists():
                experiment_files.append((exp_dir.name, exp_json))

    return experiment_files

def load_experiment_json(json_file):
    """加载experiment.json文件"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"   ⚠️  读取失败: {e}")
        return None

def check_parallel_experiment(exp_dir):
    """检查是否是并行实验，并获取前后台数据"""
    fg_json = exp_dir / "foreground" / "experiment.json"
    bg_json = exp_dir / "background" / "experiment.json"

    is_parallel = fg_json.exists() or bg_json.exists()

    fg_data = None
    bg_data = None

    if fg_json.exists():
        fg_data = load_experiment_json(fg_json)

    if bg_json.exists():
        bg_data = load_experiment_json(bg_json)

    return is_parallel, fg_data, bg_data

def main():
    base_dir = Path(__file__).parent.parent
    raw_data_csv = base_dir / "results" / "raw_data.csv"

    # 查找最新的运行目录
    results_dir = base_dir / "results"
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                     key=lambda x: x.stat().st_mtime, reverse=True)

    if not run_dirs:
        print("❌ 未找到运行目录")
        return

    latest_run_dir = run_dirs[0]

    print("=" * 80)
    print("🔍 检查最新实验结果是否完全加入数据文件")
    print("=" * 80)
    print(f"\n最新运行目录: {latest_run_dir.name}")
    print(f"运行目录最后修改时间: {latest_run_dir.stat().st_mtime}")
    print(f"Raw data文件: {raw_data_csv}")

    # 1. 加载CSV中的experiment_id
    print("\n[1/5] 加载 raw_data.csv 中的实验ID...")
    csv_exp_ids = load_experiment_ids_from_csv(raw_data_csv)
    print(f"   raw_data.csv 中共有 {len(csv_exp_ids)} 个实验ID")

    # 2. 获取运行目录中的所有实验目录
    print("\n[2/5] 扫描运行目录中的实验...")

    all_exp_dirs = sorted([d for d in latest_run_dir.iterdir() if d.is_dir()])
    print(f"   运行目录中共有 {len(all_exp_dirs)} 个实验目录")

    # 3. 检查每个实验目录
    print("\n[3/5] 检查实验类型和数据...")

    non_parallel_exps = []
    parallel_exps = []
    unknown_exps = []

    for exp_dir in all_exp_dirs:
        exp_json = exp_dir / "experiment.json"

        # 检查是否是并行实验
        is_parallel, fg_data, bg_data = check_parallel_experiment(exp_dir)

        if is_parallel:
            parallel_exps.append({
                'dir_name': exp_dir.name,
                'fg_data': fg_data,
                'bg_data': bg_data
            })
        elif exp_json.exists():
            data = load_experiment_json(exp_json)
            if data:
                non_parallel_exps.append({
                    'dir_name': exp_dir.name,
                    'data': data
                })
            else:
                unknown_exps.append(exp_dir.name)
        else:
            unknown_exps.append(exp_dir.name)

    print(f"   非并行实验: {len(non_parallel_exps)}")
    print(f"   并行实验: {len(parallel_exps)}")
    print(f"   未知类型: {len(unknown_exps)}")

    # 4. 检查是否在CSV中
    print("\n[4/5] 检查实验是否已加入CSV...")

    missing_non_parallel = []
    missing_parallel = []

    for exp in non_parallel_exps:
        exp_id = exp['data'].get('experiment_id', '')
        if exp_id not in csv_exp_ids:
            missing_non_parallel.append(exp)

    for exp in parallel_exps:
        # 并行实验的ID通常是目录名
        exp_id = exp['dir_name']
        if exp_id not in csv_exp_ids:
            # 也检查前台数据的ID
            if exp['fg_data']:
                fg_exp_id = exp['fg_data'].get('experiment_id', '')
                if fg_exp_id not in csv_exp_ids:
                    missing_parallel.append(exp)
            else:
                missing_parallel.append(exp)

    # 5. 生成报告
    print("\n[5/5] 生成检查报告...")
    print("\n" + "=" * 80)
    print("📊 检查报告")
    print("=" * 80)

    print(f"\n运行目录中的实验总数: {len(all_exp_dirs)}")
    print(f"  - 非并行实验: {len(non_parallel_exps)}")
    print(f"  - 并行实验: {len(parallel_exps)}")
    print(f"  - 未知类型: {len(unknown_exps)}")

    print(f"\nCSV中已有的实验数: {len(csv_exp_ids)}")

    print(f"\n缺失的实验数:")
    print(f"  - 非并行: {len(missing_non_parallel)}")
    print(f"  - 并行: {len(missing_parallel)}")
    print(f"  - 总计: {len(missing_non_parallel) + len(missing_parallel)}")

    if missing_non_parallel:
        print("\n" + "=" * 80)
        print("❌ 以下非并行实验未加入 raw_data.csv:")
        print("=" * 80)

        for i, exp in enumerate(missing_non_parallel[:20], 1):
            data = exp['data']
            exp_id = data.get('experiment_id', 'N/A')
            repo = data.get('repository', 'N/A')
            model = data.get('model', 'N/A')
            success = data.get('training_success', False)
            has_energy = bool(data.get('energy_metrics', {}).get('cpu_energy_total_joules'))

            print(f"\n{i}. {exp_id}")
            print(f"   模型: {repo}/{model}")
            print(f"   训练成功: {success}")
            print(f"   有能耗数据: {has_energy}")

        if len(missing_non_parallel) > 20:
            print(f"\n   ... 还有 {len(missing_non_parallel) - 20} 个实验未显示")

    if missing_parallel:
        print("\n" + "=" * 80)
        print("❌ 以下并行实验未加入 raw_data.csv:")
        print("=" * 80)

        for i, exp in enumerate(missing_parallel[:20], 1):
            dir_name = exp['dir_name']
            fg_data = exp['fg_data']
            bg_data = exp['bg_data']

            print(f"\n{i}. {dir_name}")
            if fg_data:
                print(f"   前台: {fg_data.get('repository', 'N/A')}/{fg_data.get('model', 'N/A')}")
                print(f"   前台成功: {fg_data.get('training_success', False)}")
            if bg_data:
                print(f"   后台: {bg_data.get('repository', 'N/A')}/{bg_data.get('model', 'N/A')}")

        if len(missing_parallel) > 20:
            print(f"\n   ... 还有 {len(missing_parallel) - 20} 个实验未显示")

    # 检查数据属性
    print("\n" + "=" * 80)
    print("📋 数据属性完整性检查")
    print("=" * 80)

    if non_parallel_exps:
        # 检查一个非并行实验的属性
        sample_data = non_parallel_exps[0]['data']
        print(f"\n非并行实验数据结构 (以 {non_parallel_exps[0]['dir_name']} 为例):")
        print(f"  根属性: {list(sample_data.keys())}")
        if 'hyperparameters' in sample_data:
            print(f"  超参数: {list(sample_data['hyperparameters'].keys())}")
        if 'energy_metrics' in sample_data:
            print(f"  能耗指标: {list(sample_data['energy_metrics'].keys())}")
        if 'performance_metrics' in sample_data:
            print(f"  性能指标: {list(sample_data['performance_metrics'].keys())}")

    if parallel_exps:
        sample_exp = parallel_exps[0]
        print(f"\n并行实验数据结构 (以 {sample_exp['dir_name']} 为例):")
        if sample_exp['fg_data']:
            print(f"  前台数据属性: {list(sample_exp['fg_data'].keys())}")
        if sample_exp['bg_data']:
            print(f"  后台数据属性: {list(sample_exp['bg_data'].keys())}")

    # 总结
    print("\n" + "=" * 80)
    print("📈 总结")
    print("=" * 80)

    total_missing = len(missing_non_parallel) + len(missing_parallel)

    if total_missing > 0:
        print(f"\n⚠️  发现 {total_missing} 个实验未加入 raw_data.csv")
        print(f"   运行目录: {latest_run_dir.name}")
        print(f"\n建议操作:")
        print(f"   1. 检查这些实验的 experiment.json 是否有效")
        print(f"   2. 使用数据提取脚本将 JSON 转换为 session_data.csv")
        print(f"   3. 使用 tools/data_management/append_session_to_raw_data.py 添加到 raw_data.csv")
    else:
        print("\n✅ 所有实验都已加入 raw_data.csv")

    print("\n✅ 检查完成!")

if __name__ == "__main__":
    main()
