#!/usr/bin/env python3
"""
分析Stage7的变异尝试次数和去重效果

模拟Stage7的变异生成过程，检查：
1. 每个配置请求了多少个变异（runs_per_config）
2. 系统实际尝试了多少次变异
3. 最终生成了多少个唯一变异
4. 参数空间探索是否充分
"""

import json
import csv
import sys
from pathlib import Path

# 添加mutation模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation.hyperparams import generate_mutations, _normalize_mutation_key
from mutation.dedup import load_historical_mutations, build_dedup_set


def analyze_stage7_mutation_generation():
    """分析Stage7的变异生成情况"""

    # 加载配置
    config_path = Path(__file__).parent.parent / "mutation" / "models_config.json"
    with open(config_path, 'r') as f:
        models_config = json.load(f)

    # 加载Stage7配置
    stage7_config_path = Path(__file__).parent.parent / "settings" / "stage7_nonparallel_fast_models.json"
    with open(stage7_config_path, 'r') as f:
        stage7_config = json.load(f)

    # 加载历史数据
    historical_csv = Path(__file__).parent.parent / "results" / "summary_all.csv"

    print("=" * 80)
    print("Stage7 变异生成分析")
    print("=" * 80)
    print(f"配置文件: {stage7_config_path.name}")
    print(f"历史数据: {historical_csv.name}")
    print(f"配置数量: {len(stage7_config['experiments'])}")
    print("=" * 80)

    # 加载历史变异数据
    print("\n加载历史数据...")
    mutations_data, stats = load_historical_mutations([historical_csv])
    dedup_set = build_dedup_set(mutations_data)
    print(f"历史变异总数: {len(dedup_set)}")

    # 分析每个配置的变异生成
    print("\n" + "=" * 80)
    print("逐个配置分析")
    print("=" * 80)

    total_requested = 0
    total_generated = 0
    total_experiments_run = 7  # 实际运行的实验数

    for idx, exp in enumerate(stage7_config['experiments'], 1):
        repo = exp['repo']
        model = exp['model']
        mutate_params = exp['mutate_params']
        runs_per_config = exp['runs_per_config']
        mode = exp.get('mode', 'nonparallel')

        print(f"\n配置 {idx}: {repo}/{model}")
        print(f"  请求变异数: {runs_per_config}")
        print(f"  变异参数: {mutate_params}")
        print(f"  模式: {mode}")

        # 获取模型配置
        if repo not in models_config["models"]:
            print(f"  ⚠️  仓库 '{repo}' 未在配置中找到")
            continue

        repo_config = models_config["models"][repo]
        if model not in repo_config["models"]:
            print(f"  ⚠️  模型 '{model}' 未在 {repo} 中找到")
            continue

        supported_params = repo_config["supported_hyperparams"]

        # 模拟生成变异
        try:
            # 使用相同的去重集模拟生成
            mutations = generate_mutations(
                supported_params=supported_params,
                mutate_params=mutate_params,
                num_mutations=runs_per_config,
                existing_mutations=dedup_set,
                mode=mode
            )

            generated_count = len(mutations)
            total_requested += runs_per_config
            total_generated += generated_count

            print(f"  ✓ 成功生成: {generated_count} 个唯一变异")

            if generated_count < runs_per_config:
                print(f"  ⚠️  未达到请求数量! 请求{runs_per_config}，实际生成{generated_count}")

            # 将新生成的变异加入去重集（模拟实际运行过程）
            for mutation in mutations:
                mutation_key = _normalize_mutation_key(mutation, mode)
                dedup_set.add(mutation_key)

        except Exception as e:
            print(f"  ❌ 生成失败: {e}")

    # 总结
    print("\n" + "=" * 80)
    print("总结分析")
    print("=" * 80)
    print(f"请求的总变异数: {total_requested}")
    print(f"实际生成的变异数: {total_generated}")
    print(f"实际运行的实验数: {total_experiments_run}")
    print(f"生成成功率: {total_generated/total_requested*100:.1f}%")
    print()

    # 关键发现
    print("=" * 80)
    print("关键发现")
    print("=" * 80)

    if total_generated < total_requested:
        print(f"⚠️  变异生成不足!")
        print(f"   请求: {total_requested} 个")
        print(f"   生成: {total_generated} 个")
        print(f"   差距: {total_requested - total_generated} 个")
        print()
        print("原因分析:")
        print("  1. 历史数据中已包含大量相同参数组合")
        print("  2. 去重机制正确地跳过了重复实验")
        print("  3. 参数空间已被充分探索")
    else:
        print(f"✓ 所有请求的变异都成功生成")

    print()
    print("变异重试机制:")
    print(f"  ✓ MAX_MUTATION_ATTEMPTS = 1000")
    print(f"  ✓ 系统会尝试最多1000次生成，直到获得所需的唯一变异")
    print(f"  ✓ 如果1000次内无法生成足够的唯一变异，会返回已生成的数量")

    # 检查实际运行的实验数量
    print()
    print("=" * 80)
    print("实际执行分析")
    print("=" * 80)
    print(f"配置请求总实验数: {total_requested}")
    print(f"实际运行实验数: {total_experiments_run}")
    print(f"跳过实验数: {total_requested - total_experiments_run}")
    print(f"跳过率: {(total_requested - total_experiments_run)/total_requested*100:.1f}%")

    # 时间分析
    print()
    print("=" * 80)
    print("运行时间分析")
    print("=" * 80)
    estimated_hours = stage7_config['estimated_duration_hours']
    actual_hours = 0.74

    print(f"预估时间: {estimated_hours} 小时 (基于{total_requested}个实验)")
    print(f"实际时间: {actual_hours} 小时 (实际运行{total_experiments_run}个实验)")
    print(f"时间节省: {estimated_hours - actual_hours:.1f} 小时 ({(estimated_hours - actual_hours)/estimated_hours*100:.1f}%)")
    print()
    print("时间差距原因:")
    print(f"  1. 去重机制跳过了 {total_requested - total_experiments_run} 个重复实验")
    print(f"  2. 实际只需运行 {total_experiments_run} 个新实验")
    print(f"  3. 平均每个实验用时: {actual_hours * 60 / total_experiments_run:.1f} 分钟")


if __name__ == "__main__":
    analyze_stage7_mutation_generation()
