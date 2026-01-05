#!/usr/bin/env python3
"""
合并Stage3和Stage4配置文件
"""

import json
from datetime import datetime

def merge_configs():
    """合并Stage3和Stage4配置文件"""

    # 读取Stage3配置
    with open('settings/stage3_optimized_mnist_ff_and_medium_parallel.json', 'r') as f:
        stage3 = json.load(f)

    # 读取Stage4配置
    with open('settings/stage4_optimized_vulberta_densenet121_parallel.json', 'r') as f:
        stage4 = json.load(f)

    # 计算统计数据
    stage3_experiments = stage3['experiments']
    stage4_experiments = stage4['experiments']

    total_config_items = len(stage3_experiments) + len(stage4_experiments)

    # 计算runs_per_config总和
    def calculate_total_runs(experiments):
        total = 0
        for exp in experiments:
            total += exp.get('runs_per_config', 1)
        return total

    stage3_runs = calculate_total_runs(stage3_experiments)
    stage4_runs = calculate_total_runs(stage4_experiments)
    total_expected_runs = stage3_runs + stage4_runs

    # 基于Stage2经验预估实际完成数
    stage2_completion_rate = 0.385  # 38.5%
    estimated_actual_runs = int(total_expected_runs * stage2_completion_rate)

    # 时间预估
    original_time_estimate = 36 + 32  # Stage3 36小时 + Stage4 32小时
    estimated_time_hours = estimated_actual_runs * (original_time_estimate / total_config_items)

    # 创建合并配置
    merged_config = {
        "experiment_name": "stage3_4_merged_optimized_parallel",
        "description": f"阶段3-4合并优化版: mnist_ff剩余(8个) + 中速模型并行(21个) + VulBERTa(4个) + densenet121(4个) = 37个实验项 (预计{estimated_time_hours:.1f}小时，基于Stage2 {stage2_completion_rate:.1%}完成率重新预估)",
        "mode": "mutation",
        "max_retries": 2,
        "governor": "performance",
        "use_deduplication": True,
        "historical_csvs": [
            "results/summary_all.csv"
        ],
        "experiments": stage3_experiments + stage4_experiments
    }

    # 添加合并注释
    merged_config["experiments"].insert(0, {
        "comment": "=== 合并配置说明 ===",
        "note": f"本配置合并了原Stage3和Stage4，共{total_config_items}个实验项，{total_expected_runs}个预期实验",
        "estimated_actual_runs": estimated_actual_runs,
        "estimated_time_hours": f"{estimated_time_hours:.1f}",
        "original_stage3_runs": stage3_runs,
        "original_stage4_runs": stage4_runs,
        "merge_date": datetime.now().strftime("%Y-%m-%d")
    })

    # 输出文件
    output_file = "settings/stage3_4_merged_optimized_parallel.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_config, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Stage3和Stage4配置文件合并完成")
    print("=" * 80)
    print()

    print(f"输出文件: {output_file}")
    print()

    print("合并统计:")
    print(f"  Stage3实验项: {len(stage3_experiments)}个")
    print(f"  Stage4实验项: {len(stage4_experiments)}个")
    print(f"  合并后实验项: {total_config_items}个")
    print()

    print("实验数量:")
    print(f"  Stage3预期实验: {stage3_runs}个 (runs_per_config总和)")
    print(f"  Stage4预期实验: {stage4_runs}个 (runs_per_config总和)")
    print(f"  合并后期望实验: {total_expected_runs}个")
    print()

    print("基于Stage2经验预估:")
    print(f"  Stage2完成率: {stage2_completion_rate:.1%}")
    print(f"  预计实际完成实验: {estimated_actual_runs}个")
    print(f"  原始时间预估: {original_time_estimate}小时")
    print(f"  重新预估时间: {estimated_time_hours:.1f}小时")
    print(f"  时间减少: {(1 - estimated_time_hours/original_time_estimate):.1%}")
    print()

    print("配置结构:")
    print(f"  实验名称: {merged_config['experiment_name']}")
    print(f"  描述: {merged_config['description']}")
    print(f"  去重机制: {'启用' if merged_config['use_deduplication'] else '禁用'}")
    print(f"  历史CSV文件: {len(merged_config['historical_csvs'])}个")
    print()

    print("归档建议:")
    print("  1. 移动 stage3_optimized_mnist_ff_and_medium_parallel.json 到 settings/archived/")
    print("  2. 移动 stage4_optimized_vulberta_densenet121_parallel.json 到 settings/archived/")
    print("  3. 更新 README.md 和 CLAUDE.md 中的配置引用")
    print()

    print("=" * 80)

    return output_file

if __name__ == '__main__':
    merge_configs()
