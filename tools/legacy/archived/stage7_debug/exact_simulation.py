#!/usr/bin/env python3
"""
使用与runner.py完全相同的方式加载数据并生成变异

关键：使用真实的summary_all.csv（包含Stage7运行前的所有数据）
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from mutation.hyperparams import generate_mutations
from mutation.dedup import load_historical_mutations, build_dedup_set

# 关键：使用真实的summary_all.csv路径（就像runner.py那样）
historical_csvs = [Path.cwd() / "results" / "summary_all.csv"]

print("=" * 80)
print("使用真实runner.py逻辑重现")
print("=" * 80)

# 读取当前summary_all.csv的行数
with open(historical_csvs[0], 'r') as f:
    all_lines = f.readlines()
    current_rows = len(all_lines) - 1  # 减去header

print(f"\n当前summary_all.csv总行数: {current_rows}")
print(f"Stage7添加的行数: 7")
print(f"Stage7运行前应该有: {current_rows - 7} 行")

# 模拟runner.py的加载方式
print("\n加载历史数据（使用runner.py的exact逻辑）...")
mutations_data, stats = load_historical_mutations(historical_csvs)
dedup_set = build_dedup_set(mutations_data)

print(f"历史变异数: {len(dedup_set)}")

# 加载配置
with open('settings/stage7_nonparallel_fast_models.json', 'r') as f:
    config = json.load(f)

with open('mutation/models_config.json', 'r') as f:
    models_config = json.load(f)

# 测试第一个配置
exp = config['experiments'][0]
repo = exp['repo']
model = exp['model']

print(f"\n测试配置: {repo}/{model}")
print(f"请求变异数: {exp['runs_per_config']}")

repo_config = models_config["models"][repo]
supported_params = repo_config["supported_hyperparams"]

# 完全模拟runner.py的调用
mutations = generate_mutations(
    supported_params=supported_params,
    mutate_params=exp['mutate_params'],
    num_mutations=exp['runs_per_config'],
    existing_mutations=dedup_set,
    mode="nonparallel"
)

print(f"\ngenerate_mutations返回: {len(mutations)} 个变异")

if len(mutations) < exp['runs_per_config']:
    print(f"\n❌ 只生成了{len(mutations)}个变异（而非{exp['runs_per_config']}个）")
    print(f"   这意味着在MAX_MUTATION_ATTEMPTS(1000)次尝试内")
    print(f"   只能找到{len(mutations)}个与历史数据不重复的变异！")
    print(f"\n💡 这就是缺陷：")
    print(f"   historical_csvs包含了Stage7运行前的所有实验")
    print(f"   这些实验已经充分覆盖了参数空间")
    print(f"   导致generate_mutations很难找到新的唯一变异")
else:
    print(f"\n✅ 成功生成了所有{len(mutations)}个请求的变异")
    print(f"   但为什么实际只运行了1个？")
