#!/usr/bin/env python3
"""
分析所有6个任务组的DiBS结果
重点：超参数的因果影响
"""

import numpy as np
import json
from pathlib import Path

result_dir = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/questions_2_3_dibs/20260105_212940")

groups = [
    "group1_examples",
    "group2_vulberta",
    "group3_person_reid",
    "group4_bug_localization",
    "group5_mrt_oast",
    "group6_resnet"
]

threshold = 0.1  # 使用0.1阈值

print("="*100)
print("所有任务组的超参数因果影响分析（阈值=0.1）")
print("="*100)

summary = []

for group_id in groups:
    # 加载数据
    graph = np.load(result_dir / f"{group_id}_causal_graph.npy")
    with open(result_dir / f"{group_id}_feature_names.json") as f:
        feature_names = json.load(f)

    # 变量分类
    hyperparams = [i for i, name in enumerate(feature_names) if name.startswith("hyperparam_")]
    performance = [i for i, name in enumerate(feature_names) if name.startswith("perf_")]
    energy = [i for i, name in enumerate(feature_names) if "energy" in name and not ("util" in name or "temp" in name)]
    mediators = [i for i, name in enumerate(feature_names) if "util" in name or "temp" in name]

    print(f"\n{'='*100}")
    print(f"任务组: {group_id}")
    print(f"{'='*100}")
    print(f"超参数数量: {len(hyperparams)}, 性能指标: {len(performance)}, 能耗指标: {len(energy)}, 中介变量: {len(mediators)}")

    # 统计超参数的出边
    hyperparam_effects = {}

    for hp_idx in hyperparams:
        hp_name = feature_names[hp_idx]

        # 影响性能
        perf_edges = []
        for p_idx in performance:
            if graph[hp_idx, p_idx] > threshold:
                perf_edges.append((feature_names[p_idx], graph[hp_idx, p_idx]))

        # 影响能耗
        energy_edges = []
        for e_idx in energy:
            if graph[hp_idx, e_idx] > threshold:
                energy_edges.append((feature_names[e_idx], graph[hp_idx, e_idx]))

        # 影响中介变量
        mediator_edges = []
        for m_idx in mediators:
            if graph[hp_idx, m_idx] > threshold:
                mediator_edges.append((feature_names[m_idx], graph[hp_idx, m_idx]))

        hyperparam_effects[hp_name] = {
            "performance": perf_edges,
            "energy": energy_edges,
            "mediators": mediator_edges,
            "total_edges": len(perf_edges) + len(energy_edges) + len(mediator_edges)
        }

    # 打印结果
    if hyperparams:
        print(f"\n超参数因果影响:")
        for hp_name, effects in hyperparam_effects.items():
            print(f"\n  {hp_name}:")
            print(f"    总出边: {effects['total_edges']}条")
            if effects['performance']:
                print(f"    → 性能: {effects['performance']}")
            if effects['energy']:
                print(f"    → 能耗: {effects['energy']}")
            if effects['mediators']:
                print(f"    → 中介: {effects['mediators']}")
            if effects['total_edges'] == 0:
                print(f"    ❌ 没有显著影响（阈值>{threshold}）")
    else:
        print(f"\n  ⚠️ 该任务组没有超参数特征")

    # 统计共同超参数
    common_hp = [hp for hp, eff in hyperparam_effects.items() if eff['performance'] and eff['energy']]

    # 统计中介路径
    mediation_count = 0
    for hp_idx in hyperparams:
        for m_idx in mediators:
            for e_idx in energy:
                if graph[hp_idx, m_idx] > threshold and graph[m_idx, e_idx] > threshold:
                    mediation_count += 1

    summary.append({
        "group": group_id,
        "n_hyperparams": len(hyperparams),
        "hyperparams_with_effects": sum(1 for eff in hyperparam_effects.values() if eff['total_edges'] > 0),
        "common_hyperparams": len(common_hp),
        "mediation_paths": mediation_count
    })

# 汇总统计
print(f"\n{'='*100}")
print("汇总统计")
print(f"{'='*100}")

print(f"\n| 任务组 | 超参数数 | 有影响的超参数 | 共同超参数 | 中介路径 |")
print(f"|--------|---------|--------------|-----------|----------|")
for s in summary:
    print(f"| {s['group']:<30} | {s['n_hyperparams']:^9} | {s['hyperparams_with_effects']:^14} | {s['common_hyperparams']:^10} | {s['mediation_paths']:^9} |")

total_hyperparams = sum(s['n_hyperparams'] for s in summary)
total_with_effects = sum(s['hyperparams_with_effects'] for s in summary)
total_common = sum(s['common_hyperparams'] for s in summary)
total_mediation = sum(s['mediation_paths'] for s in summary)

print(f"|{'总计':<30} | {total_hyperparams:^9} | {total_with_effects:^14} | {total_common:^10} | {total_mediation:^9} |")

print(f"\n关键发现:")
print(f"  1. 有效超参数比例: {total_with_effects}/{total_hyperparams} = {total_with_effects/total_hyperparams*100 if total_hyperparams > 0 else 0:.1f}%")
print(f"  2. 共同超参数总数: {total_common}个（同时影响能耗和性能）")
print(f"  3. 中介路径总数: {total_mediation}条（超参数→中介→能耗）")

print(f"\n{'='*100}")
