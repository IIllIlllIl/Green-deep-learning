#!/usr/bin/env python3
"""
深入分析Stage7变异重试机制

关键问题：
1. 当变异与历史数据重复时，mutate_hyperparameter()被调用了多少次？
2. 成功率（命中率）是多少？
3. MAX_MUTATION_ATTEMPTS=1000是否足够？
4. 参数空间是否已接近饱和？
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
import math

# 添加mutation模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation.hyperparams import mutate_hyperparameter, _normalize_mutation_key
from mutation.dedup import load_historical_mutations, build_dedup_set


def simulate_mutation_generation_with_tracking(
    supported_params: Dict,
    mutate_params: List[str],
    num_mutations: int,
    existing_mutations: Set,
    mode: str = "nonparallel"
) -> Dict[str, Any]:
    """
    模拟变异生成过程，跟踪实际的调用次数

    Returns:
        dict: {
            'requested': 请求的变异数量,
            'generated': 成功生成的变异数量,
            'attempts': 实际尝试次数（mutate_hyperparameter调用次数）,
            'duplicates': 重复次数,
            'hit_rate': 命中率（成功/尝试）
        }
    """
    MAX_MUTATION_ATTEMPTS = 1000

    # 确定要变异的参数
    if "all" in mutate_params:
        params_to_mutate = list(supported_params.keys())
    else:
        params_to_mutate = [p for p in mutate_params if p in supported_params]

    # 构建默认值key
    default_mutation = {}
    for param in params_to_mutate:
        default_value = supported_params[param].get("default")
        if default_value is not None:
            default_mutation[param] = default_value

    default_key = _normalize_mutation_key(default_mutation, mode) if default_mutation else None

    # 初始化seen_mutations（包含历史数据）
    seen_mutations = existing_mutations.copy()
    if default_key:
        seen_mutations.add(default_key)

    mutations = []
    attempts = 0
    duplicates = 0

    while len(mutations) < num_mutations and attempts < MAX_MUTATION_ATTEMPTS:
        attempts += 1

        # 生成新变异（这是实际调用mutate_hyperparameter的地方）
        mutation = {}
        for param in params_to_mutate:
            param_config = supported_params[param]
            mutation[param] = mutate_hyperparameter(param_config, param)

        # 检查唯一性
        mutation_key = _normalize_mutation_key(mutation, mode)

        if mutation_key not in seen_mutations:
            seen_mutations.add(mutation_key)
            mutations.append(mutation)
        else:
            duplicates += 1

    hit_rate = len(mutations) / attempts if attempts > 0 else 0

    return {
        'requested': num_mutations,
        'generated': len(mutations),
        'attempts': attempts,
        'duplicates': duplicates,
        'hit_rate': hit_rate,
        'mutations': mutations
    }


def analyze_mutation_retry_mechanism():
    """分析Stage7的变异重试机制"""

    # 加载配置
    config_path = Path(__file__).parent.parent / "mutation" / "models_config.json"
    with open(config_path, 'r') as f:
        models_config = json.load(f)

    stage7_config_path = Path(__file__).parent.parent / "settings" / "stage7_nonparallel_fast_models.json"
    with open(stage7_config_path, 'r') as f:
        stage7_config = json.load(f)

    # 加载历史数据
    historical_csv = Path(__file__).parent.parent / "results" / "summary_all.csv"

    print("=" * 80)
    print("Stage7 变异重试机制深度分析")
    print("=" * 80)
    print(f"关键参数: MAX_MUTATION_ATTEMPTS = 1000")
    print(f"分析目标: 实际mutate_hyperparameter()调用次数")
    print("=" * 80)

    # 加载历史变异
    print("\n正在加载历史数据...")
    mutations_data, stats = load_historical_mutations([historical_csv])
    dedup_set = build_dedup_set(mutations_data)
    print(f"✓ 已加载 {len(dedup_set)} 个历史变异")

    # 分析每个配置
    print("\n" + "=" * 80)
    print("逐个配置分析 - 变异调用统计")
    print("=" * 80)

    total_stats = {
        'requested': 0,
        'generated': 0,
        'attempts': 0,
        'duplicates': 0
    }

    results = []

    for idx, exp in enumerate(stage7_config['experiments'], 1):
        repo = exp['repo']
        model = exp['model']
        mutate_params = exp['mutate_params']
        runs_per_config = exp['runs_per_config']
        mode = exp.get('mode', 'nonparallel')

        print(f"\n{'─' * 80}")
        print(f"配置 {idx}: {repo}/{model}")
        print(f"{'─' * 80}")
        print(f"请求变异数: {runs_per_config}")
        print(f"变异参数数量: {len(mutate_params)}")
        print(f"模式: {mode}")

        # 获取模型配置
        if repo not in models_config["models"]:
            print(f"  ⚠️  跳过（仓库未找到）")
            continue

        repo_config = models_config["models"][repo]
        if model not in repo_config["models"]:
            print(f"  ⚠️  跳过（模型未找到）")
            continue

        supported_params = repo_config["supported_hyperparams"]

        # 模拟生成并跟踪
        stats = simulate_mutation_generation_with_tracking(
            supported_params=supported_params,
            mutate_params=mutate_params,
            num_mutations=runs_per_config,
            existing_mutations=dedup_set,
            mode=mode
        )

        # 更新去重集
        for mutation in stats['mutations']:
            mutation_key = _normalize_mutation_key(mutation, mode)
            dedup_set.add(mutation_key)

        # 记录结果
        results.append({
            'config': f"{repo}/{model}",
            **stats
        })

        # 更新总计
        total_stats['requested'] += stats['requested']
        total_stats['generated'] += stats['generated']
        total_stats['attempts'] += stats['attempts']
        total_stats['duplicates'] += stats['duplicates']

        # 打印统计
        print(f"\n结果:")
        print(f"  请求变异数:   {stats['requested']}")
        print(f"  成功生成数:   {stats['generated']}")
        print(f"  实际尝试次数: {stats['attempts']} 次mutate_hyperparameter()调用")
        print(f"  重复次数:     {stats['duplicates']}")
        print(f"  命中率:       {stats['hit_rate']*100:.1f}%")

        if stats['generated'] < stats['requested']:
            print(f"  ⚠️  未达到目标！差距: {stats['requested'] - stats['generated']}")

    # 总结
    print("\n" + "=" * 80)
    print("总体统计")
    print("=" * 80)

    total_hit_rate = total_stats['generated'] / total_stats['attempts'] if total_stats['attempts'] > 0 else 0

    print(f"\n请求总变异数:     {total_stats['requested']}")
    print(f"成功生成数:       {total_stats['generated']}")
    print(f"实际调用次数:     {total_stats['attempts']} 次")
    print(f"重复跳过次数:     {total_stats['duplicates']}")
    print(f"总体命中率:       {total_hit_rate*100:.1f}%")
    print(f"平均重试倍数:     {total_stats['attempts']/total_stats['generated']:.2f}x")

    # 分析MAX_MUTATION_ATTEMPTS的充分性
    print("\n" + "=" * 80)
    print("MAX_MUTATION_ATTEMPTS 充分性分析")
    print("=" * 80)

    max_attempts_per_config = max(r['attempts'] for r in results)
    max_config = [r for r in results if r['attempts'] == max_attempts_per_config][0]

    print(f"\nMAX_MUTATION_ATTEMPTS 上限: 1000")
    print(f"最大实际尝试次数: {max_attempts_per_config} ({max_config['config']})")
    print(f"安全余量: {1000 - max_attempts_per_config} 次 ({(1000 - max_attempts_per_config)/1000*100:.1f}%)")

    if max_attempts_per_config < 100:
        print(f"\n✅ 结论: 1000次上限非常充分")
        print(f"   实际最多只用了{max_attempts_per_config}次，仅占上限的{max_attempts_per_config/10:.1f}%")
    elif max_attempts_per_config < 500:
        print(f"\n✅ 结论: 1000次上限充分")
        print(f"   最多使用{max_attempts_per_config}次，仍有{1000-max_attempts_per_config}次余量")
    elif max_attempts_per_config < 900:
        print(f"\n⚠️  结论: 1000次上限基本充分")
        print(f"   接近上限的{max_attempts_per_config/10:.1f}%，建议监控")
    else:
        print(f"\n❌ 结论: 1000次上限可能不足")
        print(f"   已使用{max_attempts_per_config/10:.1f}%，建议增加上限")

    # 参数空间饱和度分析
    print("\n" + "=" * 80)
    print("参数空间饱和度分析")
    print("=" * 80)

    print(f"\n命中率趋势:")
    print(f"  总体命中率: {total_hit_rate*100:.1f}%")
    print(f"  重复率:     {total_stats['duplicates']/total_stats['attempts']*100:.1f}%")

    if total_hit_rate > 0.5:
        print(f"\n✅ 参数空间仍有充足探索空间")
        print(f"   命中率{total_hit_rate*100:.1f}%表明超过一半的尝试都能找到新变异")
    elif total_hit_rate > 0.2:
        print(f"\n⚠️  参数空间开始饱和")
        print(f"   命中率{total_hit_rate*100:.1f}%，需要更多尝试才能找到新变异")
    else:
        print(f"\n❌ 参数空间高度饱和")
        print(f"   命中率仅{total_hit_rate*100:.1f}%，大部分尝试都是重复")

    # 详细表格
    print("\n" + "=" * 80)
    print("详细配置对比表")
    print("=" * 80)
    print(f"\n{'配置':<35} {'请求':>6} {'生成':>6} {'尝试':>6} {'重复':>6} {'命中率':>8}")
    print("─" * 80)

    for r in results:
        hit_rate_pct = r['hit_rate'] * 100
        print(f"{r['config']:<35} {r['requested']:>6} {r['generated']:>6} {r['attempts']:>6} {r['duplicates']:>6} {hit_rate_pct:>7.1f}%")


if __name__ == "__main__":
    analyze_mutation_retry_mechanism()
