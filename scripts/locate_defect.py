#!/usr/bin/env python3
"""
定位缺陷：为什么48个请求的变异只运行了7个

关键假设验证：
1. dedup_set是否在配置间被动态更新？
2. generate_mutations实际返回的数量是否正确？
3. 是否存在运行时的额外去重逻辑？
"""

import csv
import json
from pathlib import Path

# 读取summary_all.csv
with open('results/summary_all.csv', 'r') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

print("=" * 80)
print("缺陷定位分析")
print("=" * 80)

# 分析Stage7前后的数据
stage7_before = all_rows[:-7]
stage7_added = all_rows[-7:]

print(f"\nStage7运行前: {len(stage7_before)} 行")
print(f"Stage7运行后: {len(all_rows)} 行")
print(f"Stage7新增: {len(stage7_added)} 行")

# 检查Stage7新增的实验分布
print("\n" + "=" * 80)
print("Stage7新增的7个实验分布")
print("=" * 80)

models_count = {}
for row in stage7_added:
    key = f"{row['repository']}/{row['model']}"
    models_count[key] = models_count.get(key, 0) + 1

for model, count in models_count.items():
    print(f"{model}: {count} 个实验")

# 读取Stage7配置
with open('settings/stage7_nonparallel_fast_models.json', 'r') as f:
    config = json.load(f)

print("\n" + "=" * 80)
print("Stage7配置要求")
print("=" * 80)

for i, exp in enumerate(config['experiments'], 1):
    key = f"{exp['repo']}/{exp['model']}"
    actual = models_count.get(key, 0)
    expected = exp['runs_per_config']
    print(f"{key}:")
    print(f"  配置要求: {expected} 个")
    print(f"  实际运行: {actual} 个")
    print(f"  差距: {expected - actual} 个")

# 关键发现
print("\n" + "=" * 80)
print("缺陷定位")
print("=" * 80)

total_expected = sum(exp['runs_per_config'] for exp in config['experiments'])
total_actual = len(stage7_added)

print(f"\n总计：")
print(f"  配置要求: {total_expected} 个实验")
print(f"  实际运行: {total_actual} 个实验")
print(f"  差距: {total_expected - total_actual} 个")

print("\n❌ 缺陷确认：")
print(f"   每个配置只运行了1个实验，而非配置要求的7个！")
print(f"   这意味着generate_mutations返回的列表中")
print(f"   有41个变异被丢弃或跳过！")

# 检查可能的原因
print("\n" + "=" * 80)
print("可能原因分析")
print("=" * 80)

print("\n假设1: dedup_set使用了动态更新的summary_all.csv")
print("  验证：dedup_set在runner.py:408-439只加载一次")
print("  结论：❌ 不太可能，dedup_set不会动态更新")

print("\n假设2: generate_mutations只返回了1个变异")
print("  验证：我的模拟显示每个配置都能生成7个")
print("  结论：❓ 需要检查实际运行时的返回值")

print("\n假设3: 存在运行时的额外去重逻辑")
print("  验证：需要检查run_experiment或相关代码")
print("  结论：❓ 需要深入检查")

print("\n假设4: historical_csvs指向的文件在运行时被实时读取")
print("  验证：如果每次生成时都重新加载CSV...")
print("  结论：❓ 需要检查load_historical_mutations的调用时机")

# 检查关键代码路径
print("\n" + "=" * 80)
print("关键代码路径检查")
print("=" * 80)

print("\nrunner.py:408-439 (配置开始前)")
print("  → load_historical_mutations(historical_csvs)")
print("  → build_dedup_set(mutations)")
print("  → dedup_set = {...}")

print("\nrunner.py:1108-1117 (每个配置)")
print("  → mutations = generate_mutations(")
print("       existing_mutations=dedup_set  # 使用预加载的dedup_set")
print("     )")

print("\nrunner.py:1119-1134 (运行循环)")
print("  → for run, mutation in enumerate(mutations, 1):")
print("       result = self.run_experiment(...)")

print("\n❓ 关键问题：")
print("   mutations列表的长度在实际运行时是多少？")
print("   如果是7，为什么只运行了1次循环？")
print("   如果是1，为什么generate_mutations只返回1个？")
