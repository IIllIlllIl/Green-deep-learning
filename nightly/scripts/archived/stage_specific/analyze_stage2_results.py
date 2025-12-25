#!/usr/bin/env python3
"""
分析Stage2实验结果
检查实验完成情况、数据完整性、参数唯一值数量
"""

import csv
import json
from collections import defaultdict
from datetime import datetime

def analyze_stage2_results():
    """分析Stage2实验运行结果"""

    # 读取summary_all.csv
    summary_all_file = "results/summary_all.csv"
    stage2_session_file = "results/run_20251203_225507/summary.csv"

    print("=" * 80)
    print("Stage2 实验结果分析")
    print("=" * 80)
    print()

    # 1. 统计Stage2新增实验
    print("【1. Stage2 Session统计】")
    with open(stage2_session_file, 'r') as f:
        reader = csv.DictReader(f)
        stage2_experiments = list(reader)

    print(f"  - 新增实验数量: {len(stage2_experiments)}")

    # 统计运行时长
    if stage2_experiments:
        first_exp = stage2_experiments[0]
        last_exp = stage2_experiments[-1]

        start_time = datetime.fromisoformat(first_exp['timestamp'])
        end_time = datetime.fromisoformat(last_exp['timestamp'])
        duration = end_time - start_time

        print(f"  - 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - 总运行时长: {duration.total_seconds() / 3600:.2f} 小时")
        print(f"  - 平均单次实验时长: {duration.total_seconds() / len(stage2_experiments) / 60:.1f} 分钟")

    # 统计模型分布
    model_counts = defaultdict(int)
    mode_counts = defaultdict(int)

    for exp in stage2_experiments:
        model_key = f"{exp['repository']}_{exp['model']}"
        model_counts[model_key] += 1

        # 判断是否并行
        exp_id = exp['experiment_id']
        if '_parallel' in exp_id:
            mode_counts['parallel'] += 1
        else:
            mode_counts['non-parallel'] += 1

    print(f"\n  模型分布:")
    for model, count in sorted(model_counts.items()):
        print(f"    {model}: {count}个实验")

    print(f"\n  训练模式:")
    for mode, count in sorted(mode_counts.items()):
        print(f"    {mode}: {count}个实验")

    # 2. 分析参数唯一值变化
    print("\n" + "=" * 80)
    print("【2. 参数唯一值数量分析】")
    print("=" * 80)

    with open(summary_all_file, 'r') as f:
        reader = csv.DictReader(f)
        all_experiments = list(reader)

    print(f"\n  总实验数量: {len(all_experiments)} (包括Stage2前: {len(all_experiments) - len(stage2_experiments)})")

    # 按模型统计参数唯一值
    model_params = defaultdict(lambda: defaultdict(set))

    for exp in all_experiments:
        model_key = f"{exp['repository']}_{exp['model']}"

        # 统计各超参数的唯一值
        for key, value in exp.items():
            if key.startswith('hyperparam_') and value:
                param_name = key.replace('hyperparam_', '')
                model_params[model_key][param_name].add(value)

    print("\n  各模型参数唯一值数量:")

    # 特别关注Stage2中的模型
    stage2_models = [
        'MRT-OAST_default',
        'Person_reID_baseline_pytorch_hrnet18',
        'Person_reID_baseline_pytorch_pcb',
        'examples_mnist_ff',
        'examples_mnist',
        'examples_mnist_rnn',
        'examples_siamese'
    ]

    target_achieved = 0
    target_total = 0

    for model in sorted(model_params.keys()):
        print(f"\n  {model}:")
        for param, values in sorted(model_params[model].items()):
            unique_count = len(values)
            status = "✓" if unique_count >= 5 else " "
            print(f"    {status} {param}: {unique_count}个唯一值")

            if model in stage2_models:
                target_total += 1
                if unique_count >= 5:
                    target_achieved += 1

    print(f"\n  Stage2相关模型参数达标情况: {target_achieved}/{target_total} ({target_achieved*100/target_total:.1f}%)")

    # 3. 检查数据完整性
    print("\n" + "=" * 80)
    print("【3. 数据完整性检查】")
    print("=" * 80)

    with open(summary_all_file, 'r') as f:
        first_line = f.readline()
        columns = first_line.strip().split(',')

    print(f"\n  CSV列数: {len(columns)}")
    print(f"  标准列数: 37")
    print(f"  格式检查: {'✓ 通过' if len(columns) == 37 else '✗ 错误'}")

    # 检查必要列
    required_columns = [
        'experiment_id', 'timestamp', 'repository', 'model',
        'training_success', 'duration_seconds',
        'energy_cpu_total_joules', 'energy_gpu_total_joules'
    ]

    missing_columns = [col for col in required_columns if col not in columns]
    if missing_columns:
        print(f"  ✗ 缺失必要列: {missing_columns}")
    else:
        print(f"  ✓ 所有必要列存在")

    # 4. 检查训练成功率
    print("\n" + "=" * 80)
    print("【4. 训练成功率统计】")
    print("=" * 80)

    success_count = sum(1 for exp in stage2_experiments if exp['training_success'] == 'True')
    print(f"\n  Stage2成功率: {success_count}/{len(stage2_experiments)} ({success_count*100/len(stage2_experiments):.1f}%)")

    all_success_count = sum(1 for exp in all_experiments if exp['training_success'] == 'True')
    print(f"  总体成功率: {all_success_count}/{len(all_experiments)} ({all_success_count*100/len(all_experiments):.1f}%)")

    # 5. Stage2预期 vs 实际
    print("\n" + "=" * 80)
    print("【5. Stage2预期 vs 实际对比】")
    print("=" * 80)

    expected_experiments = 44  # 从配置文件得知
    actual_experiments = len(stage2_experiments)

    print(f"\n  预期实验数: {expected_experiments}")
    print(f"  实际完成数: {actual_experiments}")
    print(f"  完成比例: {actual_experiments*100/expected_experiments:.1f}%")
    print(f"  差异: {expected_experiments - actual_experiments}个实验")

    if actual_experiments < expected_experiments:
        print(f"\n  ⚠️  可能原因:")
        print(f"     - 去重机制跳过了重复的超参数组合")
        print(f"     - 某些runs_per_config生成的随机值与历史数据重复")
        print(f"     - 需要检查去重日志确认具体原因")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

if __name__ == '__main__':
    analyze_stage2_results()
