#!/usr/bin/env python3
"""
执行算法1：基于全局标准化ATE的权衡检测

使用修复后的全局标准化ATE数据，检测能耗vs性能等权衡关系
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加utils路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from tradeoff_detection import TradeoffDetector, ENERGY_PERF_RULES

def load_global_std_ate(group_id):
    """
    加载全局标准化ATE数据

    参数:
        group_id: 任务组ID（如'group1_examples'）

    返回:
        causal_effects: 因果效应字典
            {
                'source->target': {
                    'ate': float,
                    'ci_lower': float,
                    'ci_upper': float,
                    'is_significant': bool
                },
                ...
            }
    """
    base_dir = Path(__file__).parent.parent
    ate_dir = base_dir / "results/energy_research/data/global_std_dibs_ate/"
    file_path = ate_dir / f"{group_id}_dibs_global_std_ate.csv"

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return {}

    df = pd.read_csv(file_path)

    causal_effects = {}
    missing_ate = 0
    total_edges = len(df)

    for _, row in df.iterrows():
        edge = f"{row['source']}->{row['target']}"

        # 只保留有效ATE且已计算的边
        if pd.notna(row['ate_global_std']) and row.get('ate_global_std_is_significant', False):
            causal_effects[edge] = {
                'ate': row['ate_global_std'],
                'ci_lower': row.get('ate_global_std_ci_lower', None),
                'ci_upper': row.get('ate_global_std_ci_upper', None),
                'is_significant': row.get('ate_global_std_is_significant', False)
            }
        else:
            missing_ate += 1

    print(f"  加载了 {len(causal_effects)}/{total_edges} 条边的ATE数据")
    if missing_ate > 0:
        print(f"  ⚠️  {missing_ate} 条边缺少有效ATE数据")

    return causal_effects


def main():
    """主函数：执行权衡检测"""

    print("=" * 70)
    print("算法1执行：基于全局标准化ATE的权衡检测")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 任务组映射
    group_mapping = {
        1: "group1_examples",
        2: "group2_vulberta",
        3: "group3_person_reid",
        4: "group4_bug_localization",
        5: "group5_mrt_oast",
        6: "group6_resnet"
    }

    # 初始化权衡检测器
    print("\n初始化权衡检测器...")
    detector = TradeoffDetector(
        rules=ENERGY_PERF_RULES,  # 能耗/性能专用规则
        current_values={},        # 使用默认（全局标准化数据使用0）
        verbose=True
    )

    # 存储所有权衡结果
    all_tradeoffs = {}
    all_stats = []

    # 对每个任务组执行权衡检测
    for group_num, group_id in group_mapping.items():
        print(f"\n{'='*70}")
        print(f"处理任务组 {group_num}: {group_id}")
        print(f"{'='*70}")

        # 加载ATE数据
        causal_effects = load_global_std_ate(group_id)

        if not causal_effects:
            print(f"  ⚠️  跳过（无有效ATE数据）")
            continue

        # 执行权衡检测
        print("\n  执行权衡检测...")
        tradeoffs = detector.detect_tradeoffs(
            causal_effects=causal_effects,
            require_significance=True  # 只保留统计显著的权衡
        )

        all_tradeoffs[group_id] = tradeoffs

        # 统计信息
        stats = {
            'group_id': group_id,
            'group_num': group_num,
            'total_edges': len(causal_effects),
            'tradeoffs_detected': len(tradeoffs),
            'tradeoffs_ratio': f"{len(tradeoffs)/len(causal_effects)*100:.1f}%"
        }

        # 分析权衡类型
        energy_vs_perf = 0
        hyperparam_intervention = 0

        for t in tradeoffs:
            # 检查是否为能耗vs性能权衡
            has_energy = ('energy' in t['metric1'].lower() or 'energy' in t['metric2'].lower())
            has_perf = ('perf' in t['metric1'].lower() or 'perf' in t['metric2'].lower())
            if has_energy and has_perf:
                energy_vs_perf += 1

            # 检查是否为超参数干预
            if 'hyperparam' in t['intervention'].lower():
                hyperparam_intervention += 1

        stats['energy_vs_perf'] = energy_vs_perf
        stats['hyperparam_intervention'] = hyperparam_intervention

        all_stats.append(stats)

        print(f"\n  ✅ 检测完成:")
        print(f"     总权衡数: {len(tradeoffs)}")
        print(f"     能耗vs性能: {energy_vs_perf}")
        print(f"     超参数干预: {hyperparam_intervention}")

    # 保存结果
    print(f"\n{'='*70}")
    print("保存结果")
    print(f"{'='*70}")

    output_dir = "results/energy_research/tradeoff_detection_global_std/"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存完整JSON
    json_path = os.path.join(output_dir, "all_tradeoffs_global_std.json")
    with open(json_path, 'w') as f:
        json.dump(all_tradeoffs, f, indent=2)
    print(f"  ✅ 完整结果: {json_path}")

    # 2. 生成摘要表
    summary_df = pd.DataFrame(all_stats)
    summary_csv = os.path.join(output_dir, "tradeoff_summary_global_std.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"  ✅ 摘要表: {summary_csv}")

    # 3. 生成详细权衡表
    all_tradeoff_records = []
    for group_id, tradeoffs in all_tradeoffs.items():
        for t in tradeoffs:
            record = {
                'group_id': group_id,
                'intervention': t['intervention'],
                'metric1': t['metric1'],
                'metric2': t['metric2'],
                'ate1': t['ate1'],
                'ate2': t['ate2'],
                'sign1': t['sign1'],
                'sign2': t['sign2'],
                'is_significant': t['is_significant']
            }
            all_tradeoff_records.append(record)

    if all_tradeoff_records:
        detailed_df = pd.DataFrame(all_tradeoff_records)
        detailed_csv = os.path.join(output_dir, "tradeoff_detailed_global_std.csv")
        detailed_df.to_csv(detailed_csv, index=False)
        print(f"  ✅ 详细权衡表: {detailed_csv}")

    # 4. 保存配置信息
    config_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ate_source': 'global_std_dibs_ate',
        'rules_used': 'ENERGY_PERF_RULES',
        'require_significance': True,
        'total_groups_processed': len(all_stats),
        'total_tradeoffs': sum(s['tradeoffs_detected'] for s in all_stats),
        'total_energy_vs_perf': sum(s['energy_vs_perf'] for s in all_stats)
    }

    config_path = os.path.join(output_dir, "config_info.json")
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    print(f"  ✅ 配置信息: {config_path}")

    # 打印总体统计
    print(f"\n{'='*70}")
    print("总体统计")
    print(f"{'='*70}")

    total_tradeoffs = sum(s['tradeoffs_detected'] for s in all_stats)
    total_energy_perf = sum(s['energy_vs_perf'] for s in all_stats)
    total_hyperparam = sum(s['hyperparam_intervention'] for s in all_stats)

    print(f"\n总权衡数: {total_tradeoffs}")
    print(f"  能耗vs性能: {total_energy_perf}")
    print(f"  超参数干预: {total_hyperparam}")

    print(f"\n任务组分布:")
    for s in all_stats:
        print(f"  {s['group_id']}: {s['tradeoffs_detected']} 个权衡 "
              f"({s['tradeoffs_ratio']})")

    # 输出一些示例权衡
    if all_tradeoff_records:
        print(f"\n示例权衡（前5个）:")
        for i, record in enumerate(all_tradeoff_records[:5], 1):
            print(f"  {i}. [{record['group_id']}] {record['intervention']} → "
                  f"{record['metric1']} vs {record['metric2']}")

    print(f"\n✅ 算法1执行完成！")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()