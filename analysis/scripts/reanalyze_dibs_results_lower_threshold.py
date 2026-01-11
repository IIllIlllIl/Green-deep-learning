#!/usr/bin/env python3
"""
重新分析DiBS结果 - 使用更低的阈值
目的: 检查为什么没有检测到超参数相关的因果路径
"""

import numpy as np
import json
from pathlib import Path

# 结果目录
result_dir = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/questions_2_3_dibs/20260105_212940")

# 加载group1的结果
group_id = "group1_examples"

# 加载因果图
graph = np.load(result_dir / f"{group_id}_causal_graph.npy")

# 加载特征名称
with open(result_dir / f"{group_id}_feature_names.json") as f:
    feature_names = json.load(f)

print("="*80)
print(f"DiBS因果图详细分析 - {group_id}")
print("="*80)

# 变量索引
hyperparams = [i for i, name in enumerate(feature_names) if name.startswith("hyperparam_")]
performance = [i for i, name in enumerate(feature_names) if name.startswith("perf_")]
energy = [i for i, name in enumerate(feature_names) if "energy" in name and not ("util" in name or "temp" in name)]
mediators = [i for i, name in enumerate(feature_names) if "util" in name or "temp" in name]

print(f"\n变量分类:")
print(f"  超参数索引: {hyperparams}")
for idx in hyperparams:
    print(f"    [{idx}] {feature_names[idx]}")

print(f"\n  性能指标索引: {performance}")
for idx in performance:
    print(f"    [{idx}] {feature_names[idx]}")

print(f"\n  能耗指标索引: {energy}")
for idx in energy:
    print(f"    [{idx}] {feature_names[idx]}")

print(f"\n  中介变量索引: {mediators}")
for idx in mediators:
    print(f"    [{idx}] {feature_names[idx]}")

# 分析超参数的出边（超参数影响其他变量）
print("\n" + "="*80)
print("超参数的出边分析（超参数 → 其他变量）")
print("="*80)

for hp_idx in hyperparams:
    hp_name = feature_names[hp_idx]
    print(f"\n{hp_name} (索引{hp_idx}) 的出边:")

    # 找到所有出边
    outgoing_edges = []
    for j in range(len(feature_names)):
        if j == hp_idx:
            continue
        strength = graph[hp_idx, j]
        if strength > 0.01:  # 降低阈值到0.01
            outgoing_edges.append((j, strength))

    # 按强度排序
    outgoing_edges.sort(key=lambda x: x[1], reverse=True)

    if not outgoing_edges:
        print(f"  ❌ 没有检测到任何出边（>0.01）")
    else:
        print(f"  ✅ 检测到 {len(outgoing_edges)} 条出边:")
        for target_idx, strength in outgoing_edges[:10]:  # 显示前10条
            marker = "⭐" if strength > 0.3 else "🔸" if strength > 0.1 else "・"
            print(f"    {marker} → {feature_names[target_idx]:<40} 强度={strength:.4f}")

# 分析性能和能耗的入边
print("\n" + "="*80)
print("性能指标的入边分析（其他变量 → 性能）")
print("="*80)

for perf_idx in performance:
    perf_name = feature_names[perf_idx]
    print(f"\n{perf_name} (索引{perf_idx}) 的入边:")

    incoming_edges = []
    for i in range(len(feature_names)):
        if i == perf_idx:
            continue
        strength = graph[i, perf_idx]
        if strength > 0.01:
            incoming_edges.append((i, strength))

    incoming_edges.sort(key=lambda x: x[1], reverse=True)

    if not incoming_edges:
        print(f"  ❌ 没有检测到任何入边（>0.01）")
    else:
        print(f"  ✅ 检测到 {len(incoming_edges)} 条入边:")
        for source_idx, strength in incoming_edges[:10]:
            marker = "⭐" if strength > 0.3 else "🔸" if strength > 0.1 else "・"
            print(f"    {marker} {feature_names[source_idx]:<40} → 强度={strength:.4f}")

# 重新提取证据（降低阈值到0.1）
print("\n" + "="*80)
print("重新提取证据（阈值=0.1）")
print("="*80)

threshold = 0.1

# 问题2: 共同超参数
common_hyperparams = []
for hp_idx in hyperparams:
    hp_name = feature_names[hp_idx]

    # 找到影响的性能指标
    perf_targets = []
    for p_idx in performance:
        if graph[hp_idx, p_idx] > threshold:
            perf_targets.append((feature_names[p_idx], graph[hp_idx, p_idx]))

    # 找到影响的能耗指标
    energy_targets = []
    for e_idx in energy:
        if graph[hp_idx, e_idx] > threshold:
            energy_targets.append((feature_names[e_idx], graph[hp_idx, e_idx]))

    if perf_targets and energy_targets:
        common_hyperparams.append({
            "hyperparam": hp_name,
            "affects_performance": perf_targets,
            "affects_energy": energy_targets
        })

print(f"\n共同超参数（同时影响性能和能耗，阈值>0.1）: {len(common_hyperparams)}个")
for hp_info in common_hyperparams:
    print(f"\n  {hp_info['hyperparam']}:")
    print(f"    影响性能: {hp_info['affects_performance']}")
    print(f"    影响能耗: {hp_info['affects_energy']}")

# 问题3: 中介路径
mediation_paths = []
for hp_idx in hyperparams:
    for m_idx in mediators:
        for e_idx in energy:
            strength_hm = graph[hp_idx, m_idx]
            strength_me = graph[m_idx, e_idx]

            if strength_hm > threshold and strength_me > threshold:
                mediation_paths.append({
                    "path": f"{feature_names[hp_idx]} → {feature_names[m_idx]} → {feature_names[e_idx]}",
                    "strength_step1": strength_hm,
                    "strength_step2": strength_me,
                    "indirect_strength": strength_hm * strength_me
                })

print(f"\n中介路径（超参数→中介→能耗，阈值>0.1）: {len(mediation_paths)}条")
for path_info in mediation_paths[:10]:
    print(f"  {path_info['path']}")
    print(f"    步骤1强度={path_info['strength_step1']:.4f}, 步骤2强度={path_info['strength_step2']:.4f}, 间接强度={path_info['indirect_strength']:.4f}")

print("\n" + "="*80)
print("分析完成")
print("="*80)
