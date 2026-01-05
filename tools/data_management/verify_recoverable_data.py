#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证缺失能耗数据的文件来源

功能:
- 检查缺失能耗数据的实验的原始文件
- 验证 experiment.json 或其他文件中是否包含能耗数据
- 生成数据修复的可追溯报告
- 确保所有数据都有明确的文件来源
"""

import csv
import json
import os
from pathlib import Path
from collections import defaultdict

def is_empty(val):
    """检查值是否为空"""
    return val == '' or val is None

def has_energy_data(row, mode):
    """检查实验是否有能耗数据"""
    if mode == 'parallel':
        return not is_empty(row.get('fg_energy_cpu_total_joules'))
    else:
        return not is_empty(row.get('energy_cpu_total_joules'))

def find_experiment_directory(exp_id, results_dir):
    """在所有 run_* 目录中查找实验目录"""
    for run_dir in results_dir.glob('run_*'):
        exp_dir = run_dir / exp_id
        if exp_dir.exists():
            return exp_dir
    return None

def load_experiment_json(json_file):
    """加载并返回 experiment.json 文件内容"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return None

def extract_energy_from_json(data, mode):
    """从 JSON 数据中提取能耗信息"""
    energy_data = {}

    if mode == 'parallel':
        # 并行模式：从 foreground 中提取
        if 'foreground' in data and 'energy_metrics' in data['foreground']:
            fg_energy = data['foreground']['energy_metrics']
            energy_data = {
                'fg_energy_cpu_pkg_joules': fg_energy.get('cpu_energy_pkg_joules'),
                'fg_energy_cpu_ram_joules': fg_energy.get('cpu_energy_ram_joules'),
                'fg_energy_cpu_total_joules': fg_energy.get('cpu_energy_total_joules'),
                'fg_energy_gpu_avg_watts': fg_energy.get('gpu_power_avg_watts'),
                'fg_energy_gpu_max_watts': fg_energy.get('gpu_power_max_watts'),
                'fg_energy_gpu_min_watts': fg_energy.get('gpu_power_min_watts'),
                'fg_energy_gpu_total_joules': fg_energy.get('gpu_energy_total_joules'),
                'fg_energy_gpu_temp_avg_celsius': fg_energy.get('gpu_temp_avg_celsius'),
                'fg_energy_gpu_temp_max_celsius': fg_energy.get('gpu_temp_max_celsius'),
                'fg_energy_gpu_util_avg_percent': fg_energy.get('gpu_util_avg_percent'),
                'fg_energy_gpu_util_max_percent': fg_energy.get('gpu_util_max_percent'),
            }

            # 同时提取其他前台数据
            if 'repository' in data['foreground']:
                energy_data['fg_repository'] = data['foreground'].get('repository')
            if 'model' in data['foreground']:
                energy_data['fg_model'] = data['foreground'].get('model')
            if 'training_success' in data['foreground']:
                energy_data['fg_training_success'] = data['foreground'].get('training_success')
            if 'duration_seconds' in data['foreground']:
                energy_data['fg_duration_seconds'] = data['foreground'].get('duration_seconds')

            # 提取前台超参数
            if 'hyperparameters' in data['foreground']:
                fg_hyper = data['foreground']['hyperparameters']
                for key, value in fg_hyper.items():
                    energy_data[f'fg_hyperparam_{key}'] = value

            # 提取前台性能指标
            if 'performance_metrics' in data['foreground']:
                fg_perf = data['foreground']['performance_metrics']
                for key, value in fg_perf.items():
                    energy_data[f'fg_perf_{key}'] = value

    else:
        # 非并行模式：直接从根级提取
        if 'energy_metrics' in data:
            energy = data['energy_metrics']
            energy_data = {
                'energy_cpu_pkg_joules': energy.get('cpu_energy_pkg_joules'),
                'energy_cpu_ram_joules': energy.get('cpu_energy_ram_joules'),
                'energy_cpu_total_joules': energy.get('cpu_energy_total_joules'),
                'energy_gpu_avg_watts': energy.get('gpu_power_avg_watts'),
                'energy_gpu_max_watts': energy.get('gpu_power_max_watts'),
                'energy_gpu_min_watts': energy.get('gpu_power_min_watts'),
                'energy_gpu_total_joules': energy.get('gpu_energy_total_joules'),
                'energy_gpu_temp_avg_celsius': energy.get('gpu_temp_avg_celsius'),
                'energy_gpu_temp_max_celsius': energy.get('gpu_temp_max_celsius'),
                'energy_gpu_util_avg_percent': energy.get('gpu_util_avg_percent'),
                'energy_gpu_util_max_percent': energy.get('gpu_util_max_percent'),
            }

    # 过滤掉 None 值
    energy_data = {k: v for k, v in energy_data.items() if v is not None}

    return energy_data

def main():
    base_dir = Path(__file__).parent.parent
    raw_data_csv = base_dir / "results" / "raw_data.csv"
    results_dir = base_dir / "results"

    print("=" * 80)
    print("🔍 验证缺失能耗数据的文件来源")
    print("=" * 80)
    print(f"\n数据文件: {raw_data_csv}")
    print(f"实验目录: {results_dir}\n")

    # 1. 读取 CSV 数据
    print("[1/5] 读取 raw_data.csv...")
    with open(raw_data_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"   总实验数: {len(rows)}")

    # 2. 识别缺失能耗数据的实验
    print("\n[2/5] 识别缺失能耗数据的实验...")
    experiments_without_energy = []

    for row in rows:
        mode = row.get('mode', '')
        has_energy = has_energy_data(row, mode)

        if not has_energy:
            experiments_without_energy.append({
                'csv_row': row,
                'exp_id': row.get('experiment_id', ''),
                'mode': mode
            })

    print(f"   缺失能耗数据的实验: {len(experiments_without_energy)}")

    # 3. 检查每个实验的原始文件
    print("\n[3/5] 检查实验目录中的原始文件...")

    recoverable_experiments = []
    unrecoverable_experiments = []

    for i, exp in enumerate(experiments_without_energy, 1):
        exp_id = exp['exp_id']
        mode = exp['mode']

        if i <= 5 or i % 50 == 0:  # 显示前5个和每50个
            print(f"   检查 {i}/{len(experiments_without_energy)}: {exp_id}")

        # 查找实验目录
        exp_dir = find_experiment_directory(exp_id, results_dir)

        if not exp_dir:
            unrecoverable_experiments.append({
                **exp,
                'reason': '实验目录不存在',
                'source_file': None,
                'recoverable_data': {}
            })
            continue

        # 检查 experiment.json
        exp_json = exp_dir / "experiment.json"

        if not exp_json.exists():
            unrecoverable_experiments.append({
                **exp,
                'reason': 'experiment.json 不存在',
                'source_file': None,
                'exp_dir': str(exp_dir),
                'recoverable_data': {}
            })
            continue

        # 加载 JSON 数据
        json_data = load_experiment_json(exp_json)

        if not json_data:
            unrecoverable_experiments.append({
                **exp,
                'reason': 'experiment.json 无法读取',
                'source_file': str(exp_json),
                'exp_dir': str(exp_dir),
                'recoverable_data': {}
            })
            continue

        # 提取能耗数据
        energy_data = extract_energy_from_json(json_data, mode)

        if not energy_data:
            unrecoverable_experiments.append({
                **exp,
                'reason': 'experiment.json 中无能耗数据',
                'source_file': str(exp_json),
                'exp_dir': str(exp_dir),
                'json_data': json_data,
                'recoverable_data': {}
            })
            continue

        # 数据可恢复
        recoverable_experiments.append({
            **exp,
            'source_file': str(exp_json),
            'exp_dir': str(exp_dir),
            'recoverable_data': energy_data,
            'json_data': json_data
        })

    print(f"\n   ✅ 可恢复的实验: {len(recoverable_experiments)}")
    print(f"   ❌ 不可恢复的实验: {len(unrecoverable_experiments)}")

    # 4. 生成详细报告
    print("\n[4/5] 生成详细报告...")

    print("\n" + "=" * 80)
    print("📊 数据可恢复性报告")
    print("=" * 80)

    print(f"\n总缺失数: {len(experiments_without_energy)}")
    print(f"可从文件恢复: {len(recoverable_experiments)} ({len(recoverable_experiments)*100/len(experiments_without_energy):.1f}%)")
    print(f"无法恢复: {len(unrecoverable_experiments)} ({len(unrecoverable_experiments)*100/len(experiments_without_energy):.1f}%)")

    # 按模式分类
    print("\n按训练模式分类:")
    mode_stats = defaultdict(lambda: {'recoverable': 0, 'unrecoverable': 0})

    for exp in recoverable_experiments:
        mode_stats[exp['mode']]['recoverable'] += 1

    for exp in unrecoverable_experiments:
        mode_stats[exp['mode']]['unrecoverable'] += 1

    for mode in sorted(mode_stats.keys()):
        stats = mode_stats[mode]
        total = stats['recoverable'] + stats['unrecoverable']
        print(f"  {mode or '(非并行)': <15}: 可恢复 {stats['recoverable']}, 不可恢复 {stats['unrecoverable']}, 总计 {total}")

    # 5. 显示可恢复实验的示例
    print("\n" + "=" * 80)
    print("📋 可恢复实验示例（前10个）")
    print("=" * 80)

    for i, exp in enumerate(recoverable_experiments[:10], 1):
        print(f"\n{i}. {exp['exp_id']}")
        print(f"   模式: {exp['mode']}")
        print(f"   源文件: {exp['source_file']}")
        print(f"   可恢复的数据字段: {list(exp['recoverable_data'].keys())}")

        # 显示关键能耗值
        data = exp['recoverable_data']
        if exp['mode'] == 'parallel':
            cpu_energy = data.get('fg_energy_cpu_total_joules')
            gpu_energy = data.get('fg_energy_gpu_total_joules')
            print(f"   前台CPU总能耗: {cpu_energy} J")
            print(f"   前台GPU总能耗: {gpu_energy} J")
        else:
            cpu_energy = data.get('energy_cpu_total_joules')
            gpu_energy = data.get('energy_gpu_total_joules')
            print(f"   CPU总能耗: {cpu_energy} J")
            print(f"   GPU总能耗: {gpu_energy} J")

    # 显示不可恢复实验的原因
    if unrecoverable_experiments:
        print("\n" + "=" * 80)
        print("❌ 不可恢复实验的原因分析")
        print("=" * 80)

        reason_counts = defaultdict(int)
        for exp in unrecoverable_experiments:
            reason_counts[exp['reason']] += 1

        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} 个")

        # 显示几个示例
        print("\n示例（前5个）:")
        for i, exp in enumerate(unrecoverable_experiments[:5], 1):
            print(f"\n{i}. {exp['exp_id']}")
            print(f"   原因: {exp['reason']}")
            if exp.get('source_file'):
                print(f"   文件: {exp['source_file']}")

    # 6. 保存详细数据到JSON文件
    print("\n[5/5] 保存详细数据...")

    output_file = base_dir / "results" / "recoverable_energy_data.json"

    output_data = {
        'summary': {
            'total_missing': len(experiments_without_energy),
            'recoverable': len(recoverable_experiments),
            'unrecoverable': len(unrecoverable_experiments),
            'recovery_rate': f"{len(recoverable_experiments)*100/len(experiments_without_energy):.1f}%"
        },
        'recoverable_experiments': [
            {
                'experiment_id': exp['exp_id'],
                'mode': exp['mode'],
                'source_file': exp['source_file'],
                'exp_dir': exp['exp_dir'],
                'data': exp['recoverable_data']
            }
            for exp in recoverable_experiments
        ],
        'unrecoverable_experiments': [
            {
                'experiment_id': exp['exp_id'],
                'mode': exp['mode'],
                'reason': exp['reason'],
                'exp_dir': exp.get('exp_dir'),
                'source_file': exp.get('source_file')
            }
            for exp in unrecoverable_experiments
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"   详细数据已保存到: {output_file}")

    # 7. 总结与建议
    print("\n" + "=" * 80)
    print("📈 总结与建议")
    print("=" * 80)

    if recoverable_experiments:
        print(f"\n✅ 好消息！发现 {len(recoverable_experiments)} 个实验的能耗数据可以从原始文件恢复")
        print(f"   恢复后数据完整性将提升至: {(583 + len(recoverable_experiments))*100/len(rows):.1f}%")
        print(f"\n建议:")
        print(f"   1. 查看 {output_file} 了解详细信息")
        print(f"   2. 使用安全的数据修复脚本恢复这些数据")
        print(f"   3. 所有数据都有明确的文件来源，可追溯")

    if unrecoverable_experiments:
        print(f"\n⚠️  有 {len(unrecoverable_experiments)} 个实验的数据无法恢复")
        print(f"   主要原因: {list(reason_counts.keys())[0]} ({reason_counts[list(reason_counts.keys())[0]]} 个)")
        print(f"   建议: 根据具体情况决定是否重新运行这些实验")

    print("\n✅ 验证完成!")

if __name__ == "__main__":
    main()
