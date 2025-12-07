#!/usr/bin/env python3
"""
精确追踪generate_mutations的内部行为

目标：确认实际调用mutate_hyperparameter的次数
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

# 修改hyperparams模块，添加调用计数
import mutation.hyperparams as hp

# 保存原始函数
original_mutate = hp.mutate_hyperparameter
mutation_call_count = []

def tracked_mutate(*args, **kwargs):
    """包装mutate_hyperparameter以追踪调用次数"""
    mutation_call_count.append(1)
    return original_mutate(*args, **kwargs)

# 替换函数
hp.mutate_hyperparameter = tracked_mutate

from mutation.dedup import load_historical_mutations, build_dedup_set

# 读取Stage7运行前的历史数据
with open('results/summary_all.csv', 'r') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

historical_rows_381 = all_rows[:-7]

temp_csv = Path('/tmp/summary_381.csv')
with open(temp_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(historical_rows_381)

mutations_data, stats = load_historical_mutations([temp_csv])
dedup_set = build_dedup_set(mutations_data)

# 加载配置
with open('settings/stage7_nonparallel_fast_models.json', 'r') as f:
    stage7_config = json.load(f)

with open('mutation/models_config.json', 'r') as f:
    models_config = json.load(f)

print("=" * 80)
print("追踪mutate_hyperparameter调用次数")
print("=" * 80)

total_calls = 0
total_generated = 0

for exp_idx, exp in enumerate(stage7_config['experiments'], 1):
    repo = exp['repo']
    model = exp['model']
    runs_per_config = exp['runs_per_config']

    # 重置计数器
    mutation_call_count.clear()

    print(f"\n配置{exp_idx}: {repo}/{model} (请求{runs_per_config}个)")

    repo_config = models_config["models"][repo]
    supported_params = repo_config["supported_hyperparams"]

    # 生成变异
    mutations = hp.generate_mutations(
        supported_params=supported_params,
        mutate_params=exp['mutate_params'],
        num_mutations=runs_per_config,
        existing_mutations=dedup_set,
        mode="nonparallel"
    )

    calls_this_config = len(mutation_call_count)
    generated_this_config = len(mutations)

    total_calls += calls_this_config
    total_generated += generated_this_config

    print(f"  mutate_hyperparameter调用次数: {calls_this_config}")
    print(f"  成功生成变异数: {generated_this_config}")
    print(f"  重试倍数: {calls_this_config/generated_this_config:.2f}x")

    # 更新dedup_set
    for mutation in mutations:
        from mutation.hyperparams import _normalize_mutation_key
        key = _normalize_mutation_key(mutation, "nonparallel")
        dedup_set.add(key)

print("\n" + "=" * 80)
print("总计")
print("=" * 80)
print(f"mutate_hyperparameter总调用次数: {total_calls}")
print(f"成功生成变异总数: {total_generated}")
print(f"平均重试倍数: {total_calls/total_generated:.2f}x")
print(f"请求总数: 48")

if total_generated < 48:
    print(f"\n❌ generate_mutations只生成了{total_generated}个变异（而非48个）")
    print(f"   差距: {48 - total_generated}个")
    print(f"   这说明遇到了MAX_MUTATION_ATTEMPTS上限或其他限制")
