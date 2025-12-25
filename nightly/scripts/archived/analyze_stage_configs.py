#!/usr/bin/env python3
"""
分析Stage3和Stage4配置文件的实验数量和时间预估
"""

import json

def analyze_config(file_path):
    """分析配置文件"""
    print(f"\n分析配置文件: {file_path}")

    with open(file_path, 'r') as f:
        config = json.load(f)

    print(f"配置名称: {config.get('experiment_name')}")
    print(f"描述: {config.get('description')}")

    # 统计实验
    experiments = config.get('experiments', [])
    print(f"实验项数量: {len(experiments)}")

    # 计算runs_per_config总和
    total_expected_runs = 0
    model_stats = {}

    for exp in experiments:
        runs = exp.get('runs_per_config', 1)
        total_expected_runs += runs

        # 统计模型
        repo = exp.get('repo', 'unknown')
        model = exp.get('model', 'unknown')
        model_key = f"{repo}/{model}"

        if model_key not in model_stats:
            model_stats[model_key] = {'runs': 0, 'configs': 0}

        model_stats[model_key]['runs'] += runs
        model_stats[model_key]['configs'] += 1

    print(f"预期实验数 (runs_per_config总和): {total_expected_runs}")

    # 显示模型分布
    print(f"\n模型分布:")
    for model, stats in sorted(model_stats.items()):
        print(f"  {model}: {stats['configs']}个配置项, {stats['runs']}个预期实验")

    return total_expected_runs

def main():
    """主函数"""
    print("=" * 80)
    print("Stage3和Stage4配置文件分析")
    print("=" * 80)

    # 分析Stage3
    stage3_path = "settings/stage3_optimized_mnist_ff_and_medium_parallel.json"
    stage3_runs = analyze_config(stage3_path)

    # 分析Stage4
    stage4_path = "settings/stage4_optimized_vulberta_densenet121_parallel.json"
    stage4_runs = analyze_config(stage4_path)

    print("\n" + "=" * 80)
    print("合并分析")
    print("=" * 80)

    total_configs = 29 + 8  # 实验项数量
    total_expected_runs = stage3_runs + stage4_runs

    print(f"总实验项数量: {total_configs}")
    print(f"总预期实验数: {total_expected_runs}")

    # 基于Stage2经验预估
    stage2_completion_rate = 0.385  # 38.5%完成率
    estimated_actual_runs = int(total_expected_runs * stage2_completion_rate)

    print(f"\n基于Stage2经验预估 (完成率{stage2_completion_rate:.1%}):")
    print(f"  预计实际完成实验数: {estimated_actual_runs} (基于{total_expected_runs}个预期实验)")

    # 时间预估
    original_time_estimate = (36 + 32)  # Stage3 36小时 + Stage4 32小时 = 68小时
    original_per_experiment_hours = original_time_estimate / total_configs

    # 基于实际实验数量重新预估
    estimated_hours = estimated_actual_runs * original_per_experiment_hours
    time_reduction = 1 - (estimated_hours / original_time_estimate)

    print(f"\n时间预估分析:")
    print(f"  原始预估时间: {original_time_estimate}小时 (36 + 32)")
    print(f"  基于Stage2经验重新预估: {estimated_hours:.1f}小时")
    print(f"  时间减少: {time_reduction:.1%}")
    print(f"  平均每个实验项时间: {original_per_experiment_hours:.2f}小时")

    # 建议合并后的配置
    print(f"\n合并建议:")
    print(f"  配置文件名称: stage3_4_merged_optimized_parallel.json")
    print(f"  描述: 阶段3-4合并优化版: mnist_ff剩余 + 中速模型 + VulBERTa + densenet121")
    print(f"  实验项数量: {total_configs}")
    print(f"  预期实验数: {total_expected_runs}")
    print(f"  预计实际完成数: {estimated_actual_runs}")
    print(f"  原始时间预估: {original_time_estimate}小时")
    print(f"  重新预估时间: {estimated_hours:.1f}小时")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
