#!/usr/bin/env python3
"""
验证交互项权衡检测

目标：明确统计交互项因果边，检查是否能产生权衡关系
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# 添加utils路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from tradeoff_detection import TradeoffDetector, ENERGY_PERF_RULES


def load_and_analyze_interaction_ate(group_id):
    """
    加载交互项ATE数据并分析交互项

    参数:
        group_id: 任务组ID（如'group1_examples'）

    返回:
        causal_effects: 所有因果效应（包括交互项）
        interaction_edges: 仅交互项因果边
        stats: 统计信息
    """
    interaction_dir = "results/energy_research/data/interaction/whitelist_with_ate/"
    file_path = os.path.join(interaction_dir, f"{group_id}_causal_edges_whitelist_with_ate.csv")

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return {}, {}, {}

    df = pd.read_csv(file_path)

    causal_effects = {}
    interaction_edges = {}

    # 统计
    total_edges = len(df)
    interaction_count = 0
    pos_sig = 0  # 正向显著
    neg_sig = 0  # 负向显著
    pos_nonsig = 0
    neg_nonsig = 0

    for _, row in df.iterrows():
        edge = f"{row['source']}->{row['target']}"
        is_interaction = '_x_is_parallel' in row['source']

        # 只保留有效ATE且已计算的边
        if pd.notna(row['ate']) and row.get('ate_computed', False):
            effect = {
                'ate': row['ate'],
                'ci_lower': row.get('ci_lower', None),
                'ci_upper': row.get('ci_upper', None),
                'is_significant': row.get('is_significant', False),
                'is_interaction': is_interaction
            }

            causal_effects[edge] = effect

            if is_interaction:
                interaction_edges[edge] = effect
                interaction_count += 1

                # 统计方向和显著性
                if row['ate'] > 0:
                    if row.get('is_significant', False):
                        pos_sig += 1
                    else:
                        pos_nonsig += 1
                else:
                    if row.get('is_significant', False):
                        neg_sig += 1
                    else:
                        neg_nonsig += 1

    stats = {
        'total_edges': total_edges,
        'causal_effects_count': len(causal_effects),
        'interaction_count': interaction_count,
        'pos_sig': pos_sig,
        'neg_sig': neg_sig,
        'pos_nonsig': pos_nonsig,
        'neg_nonsig': neg_nonsig,
        'can_form_tradeoff': (pos_sig > 0 and neg_sig > 0)
    }

    print(f"  加载了 {len(causal_effects)}/{total_edges} 条因果边")
    print(f"  其中交互项: {interaction_count} 条")
    print(f"  交互项方向分布: +显著={pos_sig}, -显著={neg_sig}, +不显著={pos_nonsig}, -不显著={neg_nonsig}")
    if pos_sig > 0 and neg_sig > 0:
        print(f"  ✅ 可以形成权衡 (有{pos_sig}个正向 + {neg_sig}个负向)")
    else:
        print(f"  ❌ 无法形成权衡 (方向单一)")

    return causal_effects, interaction_edges, stats


def detect_interaction_tradeoffs(group_id, interaction_edges):
    """
    专门检测交互项的权衡关系

    参数:
        group_id: 任务组ID
        interaction_edges: 交互项因果边字典

    返回:
        interaction_tradeoffs: 交互项权衡列表
    """
    if not interaction_edges:
        return []

    # 按源节点分组
    source_to_targets = {}
    for edge, effect in interaction_edges.items():
        source = edge.split('->')[0]
        if source not in source_to_targets:
            source_to_targets[source] = []
        target = edge.split('->')[1]
        source_to_targets[source].append({
            'target': target,
            'ate': effect['ate'],
            'is_significant': effect['is_significant']
        })

    # 检测权衡
    interaction_tradeoffs = []

    for source, targets in source_to_targets.items():
        # 只考虑有多个目标的源节点
        if len(targets) < 2:
            continue

        # 遍历所有目标对
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                target1 = targets[i]
                target2 = targets[j]

                # 权衡条件：方向相反且都显著
                if (target1['is_significant'] and target2['is_significant'] and
                    ((target1['ate'] > 0 and target2['ate'] < 0) or
                     (target1['ate'] < 0 and target2['ate'] > 0))):

                    interaction_tradeoffs.append({
                        'group_id': group_id,
                        'intervention': source,
                        'metric1': target1['target'],
                        'metric2': target2['target'],
                        'ate1': target1['ate'],
                        'ate2': target2['ate'],
                        'sign1': '+' if target1['ate'] > 0 else '-',
                        'sign2': '+' if target2['ate'] > 0 else '-',
                        'is_significant': True
                    })

    return interaction_tradeoffs


def main():
    """主函数：执行交互项权衡验证"""

    print("=" * 70)
    print("交互项权衡验证分析")
    print("=" * 70)

    # 任务组映射
    groups = {
        1: "group1_examples",
        2: "group2_vulberta",
        3: "group3_person_reid",
        4: "group4_bug_localization",
        5: "group5_mrt_oast",
        6: "group6_resnet"
    }

    # 存储结果
    all_interaction_stats = []
    all_interaction_tradeoffs = []

    # 对每个任务组进行分析
    for group_num, group_id in groups.items():
        print(f"\n{'='*70}")
        print(f"任务组 {group_num}: {group_id}")
        print(f"{'='*70}")

        # 加载数据
        causal_effects, interaction_edges, stats = load_and_analyze_interaction_ate(group_id)

        if not interaction_edges:
            print(f"  ⚠️  无交互项因果边，跳过权衡检测")
            continue

        stats['group_id'] = group_id
        stats['group_num'] = group_num
        all_interaction_stats.append(stats)

        # 检测交互项权衡
        print(f"\n  检测交互项权衡...")
        tradeoffs = detect_interaction_tradeoffs(group_id, interaction_edges)

        if tradeoffs:
            print(f"  ✅ 检测到 {len(tradeoffs)} 个交互项权衡:")
            for t in tradeoffs:
                print(f"     - {t['intervention'][:40]}... → {t['metric1']} vs {t['metric2']}")
        else:
            print(f"  ❌ 未检测到交互项权衡")
            print(f"     原因: 交互项效应方向单一 ({stats['pos_sig']}正向 vs {stats['neg_sig']}负向)")

        all_interaction_tradeoffs.extend(tradeoffs)

    # 保存结果
    print(f"\n{'='*70}")
    print("保存验证结果")
    print(f"{'='*70}")

    output_dir = "results/energy_research/interaction_tradeoff_verification/"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存统计摘要
    if all_interaction_stats:
        stats_df = pd.DataFrame(all_interaction_stats)
        stats_csv = os.path.join(output_dir, "interaction_stats_summary.csv")
        stats_df.to_csv(stats_csv, index=False)
        print(f"  ✅ 统计摘要: {stats_csv}")

    # 2. 保存交互项权衡
    if all_interaction_tradeoffs:
        tradeoffs_df = pd.DataFrame(all_interaction_tradeoffs)
        tradeoffs_csv = os.path.join(output_dir, "interaction_tradeoffs.csv")
        tradeoffs_df.to_csv(tradeoffs_csv, index=False)
        print(f"  ✅ 交互项权衡: {tradeoffs_csv}")

        # 保存JSON格式
        json_path = os.path.join(output_dir, "interaction_tradeoffs.json")
        with open(json_path, 'w') as f:
            json.dump(all_interaction_tradeoffs, f, indent=2)
        print(f"  ✅ JSON格式: {json_path}")
    else:
        print(f"  ⚠️  无交互项权衡需要保存")

    # 生成验证报告
    print(f"\n{'='*70}")
    print("验证报告")
    print(f"{'='*70}")

    total_interaction_edges = sum(s['interaction_count'] for s in all_interaction_stats)
    total_pos_sig = sum(s['pos_sig'] for s in all_interaction_stats)
    total_neg_sig = sum(s['neg_sig'] for s in all_interaction_stats)
    total_tradeoffs = len(all_interaction_tradeoffs)

    print(f"\n总交互项因果边: {total_interaction_edges}")
    print(f"  正向显著: {total_pos_sig}")
    print(f"  负向显著: {total_neg_sig}")
    print(f"\n检测到的交互项权衡: {total_tradeoffs}")

    if total_tradeoffs == 0:
        print(f"\n❌ 验证结论: 交互项无法形成权衡关系")
        print(f"   原因: 所有显著的交互项效应都是正向的(+), 没有显著的负向效应(-)")
        print(f"\n💡 建议: 转向研究交互项的调节效应")
    else:
        print(f"\n✅ 验证结论: 发现{total_tradeoffs}个交互项权衡!")
        print(f"   详细结果见: {output_dir}")

    print(f"\n✅ 验证分析完成!")


if __name__ == "__main__":
    main()
