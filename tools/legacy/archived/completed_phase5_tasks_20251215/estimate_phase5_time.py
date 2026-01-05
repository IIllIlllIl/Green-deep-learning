#!/usr/bin/env python3
"""
Phase 5配置时间估算

基于Phase 4的实际运行时间数据：
- VulBERTa/mlp (并行): 平均 3,755秒 (~63分钟)
- bug-localization (并行): 平均 981秒 (~16分钟)
- MRT-OAST (并行): 平均 1,568秒 (~26分钟)
- examples/mnist (并行): 估计 ~5-10分钟（轻量级模型）
- examples/mnist_ff (并行): 估计 ~5-10分钟（轻量级模型）
"""

# 实验时间估算（秒）
experiment_times = {
    "VulBERTa/mlp": 3755,          # 实际数据
    "bug-localization": 981,       # 实际数据
    "MRT-OAST": 1568,              # 实际数据
    "examples/mnist": 450,         # 估计（轻量级，5-10分钟）
    "examples/mnist_ff": 450,      # 估计（轻量级，5-10分钟）
}

# Phase 5配置实验数量
experiments = {
    "VulBERTa/mlp": 12,            # 3个参数 × 4次
    "bug-localization": 8,         # 2个参数 × 4次
    "MRT-OAST": 12,                # 3个参数 × 4次
    "examples/mnist": 20,          # 4个参数 × 5次
    "examples/mnist_ff": 20,       # 4个参数 × 5次
}

# 计算总时间
total_seconds = 0
details = []

for model, count in experiments.items():
    time_per_exp = experiment_times[model]
    model_total = time_per_exp * count
    total_seconds += model_total
    hours = model_total / 3600
    details.append(f"{model}: {count}实验 × {time_per_exp}秒 = {model_total}秒 (~{hours:.2f}小时)")

# 打印详细信息
print("=" * 80)
print("Phase 5 配置时间估算（无去重情况）")
print("=" * 80)
print()
for detail in details:
    print(detail)
print()
print(f"总计时间: {total_seconds}秒 = {total_seconds/3600:.2f}小时")
print()

# 考虑去重情况
print("=" * 80)
print("考虑去重情况（假设50%去重率）")
print("=" * 80)
dedup_seconds = total_seconds * 0.5
print(f"预计实际时间: {dedup_seconds}秒 = {dedup_seconds/3600:.2f}小时")
print()

# 考虑更保守的去重率
print("=" * 80)
print("考虑去重情况（假设30%去重率）")
print("=" * 80)
dedup_seconds_30 = total_seconds * 0.7
print(f"预计实际时间: {dedup_seconds_30}秒 = {dedup_seconds_30/3600:.2f}小时")
print()

# 总结
print("=" * 80)
print("总结")
print("=" * 80)
print(f"总实验数: {sum(experiments.values())}个")
print(f"无去重预计: {total_seconds/3600:.2f}小时")
print(f"50%去重预计: {dedup_seconds/3600:.2f}小时")
print(f"30%去重预计: {dedup_seconds_30/3600:.2f}小时")
print()
