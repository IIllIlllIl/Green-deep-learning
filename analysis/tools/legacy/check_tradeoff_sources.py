#!/usr/bin/env python3
"""
双数据源权衡检测验证脚本
检查全局标准化ATE和交互项ATE的权衡检测结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from utils.tradeoff_detection import TradeoffDetector, ENERGY_PERF_RULES

def load_ate_data(csv_path, ate_col='ate'):
    """加载ATE数据并转换为因果效应字典"""
    df = pd.read_csv(csv_path)
    causal_effects = {}

    for _, row in df.iterrows():
        edge = f"{row['source']}->{row['target']}"
        ate_val = row[ate_col]

        # 跳过空值
        if pd.isna(ate_val):
            continue

        causal_effects[edge] = {
            'ate': ate_val,
            'ci_lower': row.get(f'{ate_col}_ci_lower', row.get('ci_lower', ate_val)),
            'ci_upper': row.get(f'{ate_col}_ci_upper', row.get('ci_upper', ate_val)),
            'is_significant': row.get(f'{ate_col}_is_significant', row.get('is_significant', True))
        }

    return causal_effects

def analyze_group(group_name, global_std_path, interaction_path):
    """分析单个任务组的权衡关系"""
    print("=" * 70)
    print(f"任务组: {group_name}")
    print("=" * 70)

    # 加载数据
    global_std_df = pd.read_csv(global_std_path)
    interaction_df = pd.read_csv(interaction_path)

    print(f"\n数据统计:")
    print(f"  全局标准化ATE: {len(global_std_df)} 条边, {global_std_df['ate_global_std'].notna().sum()} 条有ATE值")
    print(f"  交互项ATE: {len(interaction_df)} 条边, {interaction_df['ate'].notna().sum()} 条有ATE值")

    # 转换为因果效应格式
    global_std_effects = load_ate_data(global_std_path, 'ate_global_std')
    interaction_effects = load_ate_data(interaction_path, 'ate')

    print(f"\n转换后的因果边:")
    print(f"  全局标准化: {len(global_std_effects)} 条")
    print(f"  交互项: {len(interaction_effects)} 条")

    # 运行权衡检测
    print(f"\n权衡检测结果:")

    # 创建检测器（先不要求显著性，看看有多少候选权衡）
    try:
        detector = TradeoffDetector(rules=ENERGY_PERF_RULES, verbose=False)

        # 全局标准化
        tradeoffs_global_std = detector.detect_tradeoffs(global_std_effects, require_significance=False)
        tradeoffs_global_std_sig = detector.detect_tradeoffs(global_std_effects, require_significance=True)

        print(f"  全局标准化ATE:")
        print(f"    全部权衡: {len(tradeoffs_global_std)} 个")
        print(f"    显著权衡: {len(tradeoffs_global_std_sig)} 个")

        if tradeoffs_global_std_sig:
            print(f"    示例（显著）:")
            for i, t in enumerate(tradeoffs_global_std_sig[:3], 1):
                print(f"      {i}. {t['intervention']} → {t['metric1']} ({t['sign1']}) vs {t['metric2']} ({t['sign2']})")

        # 交互项
        detector2 = TradeoffDetector(rules=ENERGY_PERF_RULES, verbose=False)
        tradeoffs_interaction = detector2.detect_tradeoffs(interaction_effects, require_significance=False)
        tradeoffs_interaction_sig = detector2.detect_tradeoffs(interaction_effects, require_significance=True)

        print(f"  交互项ATE:")
        print(f"    全部权衡: {len(tradeoffs_interaction)} 个")
        print(f"    显著权衡: {len(tradeoffs_interaction_sig)} 个")

        if tradeoffs_interaction_sig:
            print(f"    示例（显著）:")
            for i, t in enumerate(tradeoffs_interaction_sig[:3], 1):
                print(f"      {i}. {t['intervention']} → {t['metric1']} ({t['sign1']}) vs {t['metric2']} ({t['sign2']})")

        # 对比结果
        print(f"\n对比分析（显著权衡）:")
        set1 = {(t['intervention'], t['metric1'], t['metric2']) for t in tradeoffs_global_std_sig}
        set2 = {(t['intervention'], t['metric1'], t['metric2']) for t in tradeoffs_interaction_sig}
        consistent = set1 & set2
        only_global = set1 - set2
        only_interaction = set2 - set1

        print(f"  一致的权衡: {len(consistent)} 个")
        if consistent:
            for item in list(consistent)[:3]:
                print(f"    {item[0]} → {item[1]} vs {item[2]}")

        print(f"  仅全局标准化: {len(only_global)} 个")
        if only_global:
            for item in list(only_global)[:3]:
                print(f"    {item[0]} → {item[1]} vs {item[2]}")

        print(f"  仅交互项: {len(only_interaction)} 个")
        if only_interaction:
            for item in list(only_interaction)[:3]:
                print(f"    {item[0]} → {item[1]} vs {item[2]}")

        return {
            'group': group_name,
            'global_std_all': len(tradeoffs_global_std),
            'global_std_sig': len(tradeoffs_global_std_sig),
            'interaction_all': len(tradeoffs_interaction),
            'interaction_sig': len(tradeoffs_interaction_sig),
            'consistent': len(consistent),
            'only_global': len(only_global),
            'only_interaction': len(only_interaction)
        }

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("双数据源权衡检测验证")
    print("=" * 70)
    print("对比全局标准化ATE vs 交互项ATE")
    print("=" * 70)

    # 定义数据源路径
    base_dir = "/home/green/energy_dl/nightly/analysis/results/energy_research/data"
    global_std_dir = f"{base_dir}/global_std_dibs_ate"
    interaction_dir = f"{base_dir}/interaction/whitelist_with_ate"

    groups = [
        ("Group1 (examples)", "group1_examples"),
        ("Group2 (VulBERTa)", "group2_vulberta"),
        ("Group3 (Person_reID)", "group3_person_reid"),
        ("Group4 (bug_localization)", "group4_bug_localization"),
        ("Group5 (MRT-OAST)", "group5_mrt_oast"),
        ("Group6 (ResNet)", "group6_resnet"),
    ]

    results = []

    for group_name, group_id in groups:
        global_std_path = f"{global_std_dir}/{group_id}_dibs_global_std_ate.csv"
        interaction_path = f"{interaction_dir}/{group_id}_causal_edges_whitelist_with_ate.csv"

        if not os.path.exists(global_std_path) or not os.path.exists(interaction_path):
            print(f"\n⚠️  跳过 {group_name}: 文件不存在")
            continue

        result = analyze_group(group_name, global_std_path, interaction_path)
        if result:
            results.append(result)

    # 汇总结果
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    if results:
        df = pd.DataFrame(results)
        print(f"\n{df.to_string(index=False)}")

        print(f"\n总计:")
        print(f"  全局标准化显著权衡: {df['global_std_sig'].sum()} 个")
        print(f"  交互项显著权衡: {df['interaction_sig'].sum()} 个")
        print(f"  一致权衡: {df['consistent'].sum()} 个")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
