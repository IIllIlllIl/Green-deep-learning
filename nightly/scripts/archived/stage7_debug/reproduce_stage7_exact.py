#!/usr/bin/env python3
"""
精确复现Stage7运行逻辑，追踪变异生成和执行流程

目的：找出为什么配置要求48个实验，但只运行了7个
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from mutation.hyperparams import generate_mutations
from mutation.dedup import load_historical_mutations, build_dedup_set

# 读取Stage7运行前的历史数据（381行）
print("=" * 80)
print("精确复现Stage7运行逻辑")
print("=" * 80)

with open('results/summary_all.csv', 'r') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

historical_rows_381 = all_rows[:-7]

# 创建临时CSV
temp_csv = Path('/tmp/summary_381.csv')
with open(temp_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(historical_rows_381)

# 加载历史变异
print(f"\n加载历史数据: {len(historical_rows_381)} 行")
mutations_data, stats = load_historical_mutations([temp_csv])
dedup_set = build_dedup_set(mutations_data)
print(f"历史变异去重集: {len(dedup_set)} 个")

# 加载配置
with open('settings/stage7_nonparallel_fast_models.json', 'r') as f:
    stage7_config = json.load(f)

with open('mutation/models_config.json', 'r') as f:
    models_config = json.load(f)

# 模拟runner.py的执行流程
print("\n" + "=" * 80)
print("模拟执行流程")
print("=" * 80)

total_experiments_to_run = 0
total_generated = 0

for exp_idx, exp in enumerate(stage7_config['experiments'], 1):
    repo = exp['repo']
    model = exp['model']
    runs_per_config = exp['runs_per_config']
    mutate_params = exp['mutate_params']
    mode = exp.get('mode', 'nonparallel')

    print(f"\n{'─' * 80}")
    print(f"配置 {exp_idx}/{len(stage7_config['experiments'])}: {repo}/{model}")
    print(f"{'─' * 80}")
    print(f"runs_per_config: {runs_per_config}")
    print(f"mutate_params: {mutate_params}")
    print(f"mode: {mode}")

    # 获取模型配置
    if repo not in models_config["models"]:
        print(f"⚠️  跳过（仓库未找到）")
        continue

    repo_config = models_config["models"][repo]
    if model not in repo_config["models"]:
        print(f"⚠️  跳过（模型未找到）")
        continue

    supported_params = repo_config["supported_hyperparams"]

    # 调用generate_mutations（这是runner.py实际调用的）
    print(f"\n调用 generate_mutations(num_mutations={runs_per_config}, mode='{mode}')...")

    mutations = generate_mutations(
        supported_params=supported_params,
        mutate_params=mutate_params,
        num_mutations=runs_per_config,
        existing_mutations=dedup_set,
        mode=mode
    )

    generated_count = len(mutations)
    total_generated += generated_count

    print(f"\n结果：")
    print(f"  请求: {runs_per_config} 个变异")
    print(f"  生成: {generated_count} 个变异")

    if generated_count < runs_per_config:
        print(f"  ❌ 警告：未达到目标！差距: {runs_per_config - generated_count}")
        print(f"     这意味着在1000次尝试内无法生成足够的唯一变异")
    else:
        print(f"  ✅ 成功生成所有请求的变异")

    # 根据runner.py逻辑，应该运行所有生成的变异
    print(f"\n  根据runner.py逻辑，应该运行 {generated_count} 个实验")
    total_experiments_to_run += generated_count

    # 更新去重集（模拟实际运行后的状态）
    for mutation in mutations:
        from mutation.hyperparams import _normalize_mutation_key
        mutation_key = _normalize_mutation_key(mutation, mode)
        dedup_set.add(mutation_key)

# 总结
print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"\n配置要求的总runs_per_config: 48")
print(f"成功生成的变异数: {total_generated}")
print(f"应该运行的实验数: {total_experiments_to_run}")
print(f"\n实际运行的实验数: 7")
print(f"\n❓ 问题：为什么应该运行{total_experiments_to_run}个，实际只运行了7个？")

if total_generated < 48:
    print(f"\n可能原因：generate_mutations无法生成足够的唯一变异")
    print(f"差距: {48 - total_generated} 个")
else:
    print(f"\n❌ 矛盾：generate_mutations生成了{total_generated}个变异")
    print(f"   但runner.py只运行了7个！")
    print(f"\n   这说明在generate_mutations和run_experiment之间")
    print(f"   存在某种未知的过滤或去重逻辑！")
