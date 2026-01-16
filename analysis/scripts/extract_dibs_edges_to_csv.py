#!/usr/bin/env python3
"""
DiBS因果边提取脚本

功能: 从DiBS分析结果JSON文件中提取所有因果边，生成CSV文件方便后续分析

输出CSV文件:
  1. dibs_direct_edges.csv - 所有直接因果边
  2. dibs_indirect_paths.csv - 所有间接因果路径
  3. dibs_all_edges_summary.csv - 按任务组汇总的边统计

作者: Claude
日期: 2026-01-16
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_dibs_results(result_dir):
    """
    加载所有DiBS结果JSON文件

    参数:
        result_dir: 结果目录路径

    返回:
        results: 结果字典列表
    """
    result_dir = Path(result_dir)
    results = []

    for json_file in sorted(result_dir.glob("*_result.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            results.append(result)

    return results


def extract_direct_edges(results, threshold=0.3):
    """
    提取所有直接因果边

    包括:
      - 超参数 → 能耗
      - 超参数 → 性能
      - 性能 → 能耗
      - 能耗 → 性能
      - 中介 → 能耗
      - 中介 → 性能

    参数:
        results: DiBS结果列表
        threshold: 边强度阈值

    返回:
        edges_df: 直接因果边DataFrame
    """
    all_edges = []

    for result in results:
        task_id = result['task_id']
        task_name = result['task_name']

        # 1. 超参数 → 能耗（问题1的直接边）
        for edge in result['question1_evidence']['direct_hyperparam_to_energy']:
            if edge['strength'] >= threshold:
                all_edges.append({
                    'task_group': task_id,
                    'task_name': task_name,
                    'edge_type': 'hyperparam_to_energy',
                    'source': edge['hyperparam'],
                    'target': edge['energy_var'],
                    'strength': edge['strength'],
                    'research_question': 'Q1_direct',
                    'source_type': 'hyperparam',
                    'target_type': 'energy'
                })

        # 2. 性能 → 能耗（问题2的直接边）
        for edge in result['question2_evidence']['direct_edges_perf_to_energy']:
            if edge['strength'] >= threshold:
                all_edges.append({
                    'task_group': task_id,
                    'task_name': task_name,
                    'edge_type': 'performance_to_energy',
                    'source': edge['from'],
                    'target': edge['to'],
                    'strength': edge['strength'],
                    'research_question': 'Q2_tradeoff',
                    'source_type': 'performance',
                    'target_type': 'energy'
                })

        # 3. 能耗 → 性能（问题2的直接边）
        for edge in result['question2_evidence']['direct_edges_energy_to_perf']:
            if edge['strength'] >= threshold:
                all_edges.append({
                    'task_group': task_id,
                    'task_name': task_name,
                    'edge_type': 'energy_to_performance',
                    'source': edge['from'],
                    'target': edge['to'],
                    'strength': edge['strength'],
                    'research_question': 'Q2_tradeoff',
                    'source_type': 'energy',
                    'target_type': 'performance'
                })

        # 4. 提取所有超参数→性能的直接边（从问题2的共同超参数）
        for common_hp in result['question2_evidence']['common_hyperparams']:
            hp_name = common_hp['hyperparam']

            # 超参数 → 性能
            for perf_target in common_hp['affects_performance']:
                if perf_target['strength'] >= threshold:
                    all_edges.append({
                        'task_group': task_id,
                        'task_name': task_name,
                        'edge_type': 'hyperparam_to_performance',
                        'source': hp_name,
                        'target': perf_target['var'],
                        'strength': perf_target['strength'],
                        'research_question': 'Q2_common_hyperparam',
                        'source_type': 'hyperparam',
                        'target_type': 'performance'
                    })

    # 转换为DataFrame并排序
    edges_df = pd.DataFrame(all_edges)

    if len(edges_df) > 0:
        edges_df = edges_df.sort_values(
            by=['task_group', 'strength', 'source', 'target'],
            ascending=[True, False, True, True]
        )
        edges_df = edges_df.reset_index(drop=True)

    return edges_df


def extract_indirect_paths(results, threshold=0.3):
    """
    提取所有间接因果路径

    包括:
      - 超参数 → 中介 → 能耗
      - 超参数 → 中介 → 性能
      - 性能 → 中介 → 能耗
      - 多步路径（≥4节点）

    参数:
        results: DiBS结果列表
        threshold: 边强度阈值

    返回:
        paths_df: 间接因果路径DataFrame
    """
    all_paths = []

    for result in results:
        task_id = result['task_id']
        task_name = result['task_name']

        # 1. 超参数 → 中介 → 能耗（问题1的中介路径）
        for path in result['question1_evidence']['mediated_hyperparam_to_energy']:
            # 检查两段边的强度是否都达到阈值
            if path['strength_step1'] >= threshold and path['strength_step2'] >= threshold:
                all_paths.append({
                    'task_group': task_id,
                    'task_name': task_name,
                    'path_type': 'hyperparam_mediator_energy',
                    'source': path['hyperparam'],
                    'mediator1': path['mediator'],
                    'mediator2': None,
                    'target': path['energy_var'],
                    'strength_step1': path['strength_step1'],
                    'strength_step2': path['strength_step2'],
                    'strength_step3': None,
                    'indirect_strength': path['indirect_strength'],
                    'num_steps': 2,
                    'research_question': 'Q1_mediated',
                    'path_description': f"{path['hyperparam']} → {path['mediator']} → {path['energy_var']}"
                })

        # 2. 超参数 → 中介 → 能耗（问题3的中介路径）
        for path in result['question3_evidence']['mediation_paths_to_energy']:
            if path['strength_X_to_M'] >= threshold and path['strength_M_to_Y'] >= threshold:
                all_paths.append({
                    'task_group': task_id,
                    'task_name': task_name,
                    'path_type': 'hyperparam_mediator_energy',
                    'source': path['hyperparam'],
                    'mediator1': path['mediator'],
                    'mediator2': None,
                    'target': path['outcome'],
                    'strength_step1': path['strength_X_to_M'],
                    'strength_step2': path['strength_M_to_Y'],
                    'strength_step3': None,
                    'indirect_strength': path['indirect_strength'],
                    'num_steps': 2,
                    'research_question': 'Q3_mediation_energy',
                    'path_description': f"{path['hyperparam']} → {path['mediator']} → {path['outcome']}",
                    'mediation_type': path['mediation_type'],
                    'direct_strength': path['direct_strength']
                })

        # 3. 超参数 → 中介 → 性能（问题3的中介路径）
        for path in result['question3_evidence']['mediation_paths_to_performance']:
            if path['strength_X_to_M'] >= threshold and path['strength_M_to_Y'] >= threshold:
                all_paths.append({
                    'task_group': task_id,
                    'task_name': task_name,
                    'path_type': 'hyperparam_mediator_performance',
                    'source': path['hyperparam'],
                    'mediator1': path['mediator'],
                    'mediator2': None,
                    'target': path['outcome'],
                    'strength_step1': path['strength_X_to_M'],
                    'strength_step2': path['strength_M_to_Y'],
                    'strength_step3': None,
                    'indirect_strength': path['indirect_strength'],
                    'num_steps': 2,
                    'research_question': 'Q3_mediation_performance',
                    'path_description': f"{path['hyperparam']} → {path['mediator']} → {path['outcome']}",
                    'mediation_type': path['mediation_type'],
                    'direct_strength': path['direct_strength']
                })

        # 4. 性能 → 中介 → 能耗（问题2的中介权衡路径）
        for path in result['question2_evidence']['mediated_tradeoffs']:
            # 从路径描述中提取节点
            path_str = path['path']
            nodes = [node.strip() for node in path_str.split('→')]

            if path['strength_step1'] >= threshold and path['strength_step2'] >= threshold:
                all_paths.append({
                    'task_group': task_id,
                    'task_name': task_name,
                    'path_type': 'performance_mediator_energy',
                    'source': nodes[0],
                    'mediator1': nodes[1],
                    'mediator2': None,
                    'target': nodes[2],
                    'strength_step1': path['strength_step1'],
                    'strength_step2': path['strength_step2'],
                    'strength_step3': None,
                    'indirect_strength': path['path_strength'],
                    'num_steps': 2,
                    'research_question': 'Q2_mediated_tradeoff',
                    'path_description': path_str
                })

        # 5. 多步路径（≥4节点）
        for path in result['question3_evidence']['multi_step_paths']:
            # 从路径描述中提取节点
            path_str = path['path']
            nodes = [node.strip() for node in path_str.split('→')]

            if (path['strength_step1'] >= threshold and
                path['strength_step2'] >= threshold and
                path['strength_step3'] >= threshold):

                all_paths.append({
                    'task_group': task_id,
                    'task_name': task_name,
                    'path_type': 'multi_step',
                    'source': nodes[0],
                    'mediator1': nodes[1],
                    'mediator2': nodes[2],
                    'target': nodes[3],
                    'strength_step1': path['strength_step1'],
                    'strength_step2': path['strength_step2'],
                    'strength_step3': path['strength_step3'],
                    'indirect_strength': path['path_strength'],
                    'num_steps': 3,
                    'research_question': 'Q3_multi_step',
                    'path_description': path_str
                })

    # 转换为DataFrame并排序
    paths_df = pd.DataFrame(all_paths)

    if len(paths_df) > 0:
        paths_df = paths_df.sort_values(
            by=['task_group', 'num_steps', 'indirect_strength'],
            ascending=[True, True, False]
        )
        paths_df = paths_df.reset_index(drop=True)

    return paths_df


def generate_summary_statistics(results, edges_df, paths_df):
    """
    生成按任务组汇总的边统计

    参数:
        results: DiBS结果列表
        edges_df: 直接因果边DataFrame
        paths_df: 间接因果路径DataFrame

    返回:
        summary_df: 汇总统计DataFrame
    """
    summary_data = []

    for result in results:
        task_id = result['task_id']
        task_name = result['task_name']

        # 从edges_df和paths_df中统计该任务组的边数
        task_direct_edges = len(edges_df[edges_df['task_group'] == task_id]) if len(edges_df) > 0 else 0
        task_indirect_paths = len(paths_df[paths_df['task_group'] == task_id]) if len(paths_df) > 0 else 0

        # 细分统计
        q1_direct = len(result['question1_evidence']['direct_hyperparam_to_energy'])
        q1_mediated = len(result['question1_evidence']['mediated_hyperparam_to_energy'])

        q2_perf_to_energy = len(result['question2_evidence']['direct_edges_perf_to_energy'])
        q2_energy_to_perf = len(result['question2_evidence']['direct_edges_energy_to_perf'])
        q2_common_hp = len(result['question2_evidence']['common_hyperparams'])
        q2_mediated = len(result['question2_evidence']['mediated_tradeoffs'])

        q3_mediation_energy = len(result['question3_evidence']['mediation_paths_to_energy'])
        q3_mediation_perf = len(result['question3_evidence']['mediation_paths_to_performance'])
        q3_multi_step = len(result['question3_evidence']['multi_step_paths'])

        summary_data.append({
            'task_group': task_id,
            'task_name': task_name,
            'n_samples': result['n_samples'],
            'n_features': result['n_features'],
            'n_hyperparams': result['variable_classification']['n_hyperparams'],
            'n_performance': result['variable_classification']['n_performance'],
            'n_energy': result['variable_classification']['n_energy'],
            'n_mediators': result['variable_classification']['n_mediators'],
            'direct_edges_extracted': task_direct_edges,
            'indirect_paths_extracted': task_indirect_paths,
            'q1_direct_hp_to_energy': q1_direct,
            'q1_mediated_hp_to_energy': q1_mediated,
            'q2_perf_to_energy': q2_perf_to_energy,
            'q2_energy_to_perf': q2_energy_to_perf,
            'q2_common_hyperparams': q2_common_hp,
            'q2_mediated_tradeoffs': q2_mediated,
            'q3_mediation_to_energy': q3_mediation_energy,
            'q3_mediation_to_perf': q3_mediation_perf,
            'q3_multi_step_paths': q3_multi_step,
            'total_edges': task_direct_edges + task_indirect_paths,
            'elapsed_time_minutes': result['elapsed_time_minutes'],
            'graph_max': result['graph_stats']['max'],
            'graph_mean': result['graph_stats']['mean'],
            'strong_edges_gt_0.3': result['edges']['threshold_0.3'],
            'total_edges_gt_0.01': result['edges']['threshold_0.01']
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='从DiBS结果提取因果边到CSV文件')
    parser.add_argument('--result-dir', type=str,
                       default='results/energy_research/dibs_6groups_final',
                       help='DiBS结果目录（默认为最新）')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='边强度阈值（默认0.3）')
    parser.add_argument('--output-dir', type=str, default='results/energy_research/dibs_edges_csv',
                       help='输出CSV文件目录')

    args = parser.parse_args()

    # 查找最新结果目录
    result_base = Path(args.result_dir)
    if not result_base.exists():
        print(f"❌ 结果目录不存在: {result_base}")
        return

    # 找到最新的时间戳目录
    timestamp_dirs = sorted([d for d in result_base.iterdir() if d.is_dir()])
    if not timestamp_dirs:
        print(f"❌ 没有找到结果子目录")
        return

    latest_dir = timestamp_dirs[-1]
    print(f"使用结果目录: {latest_dir}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载DiBS结果
    print("\n加载DiBS结果...")
    results = load_dibs_results(latest_dir)
    print(f"  加载了 {len(results)} 个任务组的结果")

    # 提取直接因果边
    print(f"\n提取直接因果边（阈值 > {args.threshold}）...")
    edges_df = extract_direct_edges(results, threshold=args.threshold)
    print(f"  提取了 {len(edges_df)} 条直接因果边")

    if len(edges_df) > 0:
        edges_file = output_dir / 'dibs_direct_edges.csv'
        edges_df.to_csv(edges_file, index=False, encoding='utf-8')
        print(f"  ✅ 保存到: {edges_file}")

        # 打印按类型统计
        print("\n  按边类型统计:")
        edge_type_counts = edges_df['edge_type'].value_counts()
        for edge_type, count in edge_type_counts.items():
            print(f"    - {edge_type}: {count}条")
    else:
        print(f"  ⚠️ 没有找到强度 > {args.threshold} 的直接因果边")

    # 提取间接因果路径
    print(f"\n提取间接因果路径（阈值 > {args.threshold}）...")
    paths_df = extract_indirect_paths(results, threshold=args.threshold)
    print(f"  提取了 {len(paths_df)} 条间接因果路径")

    if len(paths_df) > 0:
        paths_file = output_dir / 'dibs_indirect_paths.csv'
        paths_df.to_csv(paths_file, index=False, encoding='utf-8')
        print(f"  ✅ 保存到: {paths_file}")

        # 打印按路径类型统计
        print("\n  按路径类型统计:")
        path_type_counts = paths_df['path_type'].value_counts()
        for path_type, count in path_type_counts.items():
            print(f"    - {path_type}: {count}条")

        # 打印按步数统计
        print("\n  按路径步数统计:")
        step_counts = paths_df['num_steps'].value_counts().sort_index()
        for steps, count in step_counts.items():
            print(f"    - {steps}步路径: {count}条")
    else:
        print(f"  ⚠️ 没有找到强度 > {args.threshold} 的间接因果路径")

    # 生成汇总统计
    print("\n生成汇总统计...")
    summary_df = generate_summary_statistics(results, edges_df, paths_df)
    summary_file = output_dir / 'dibs_all_edges_summary.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"  ✅ 保存到: {summary_file}")

    # 打印总体统计
    print("\n" + "="*70)
    print("总体统计")
    print("="*70)
    print(f"  任务组数: {len(results)}")
    print(f"  直接因果边总数: {len(edges_df)}")
    print(f"  间接因果路径总数: {len(paths_df)}")
    print(f"  总因果关系数: {len(edges_df) + len(paths_df)}")
    print(f"  边强度阈值: {args.threshold}")
    print("="*70)

    # 打印Top 10强边
    if len(edges_df) > 0:
        print("\nTop 10 最强直接因果边:")
        print("-"*70)
        top_edges = edges_df.nlargest(10, 'strength')[['task_name', 'source', 'target', 'strength', 'edge_type']]
        for idx, row in top_edges.iterrows():
            print(f"  {row['task_name'][:20]:20s} | {row['source']:30s} → {row['target']:30s} | {row['strength']:.4f}")

    print(f"\n✅ 所有CSV文件已保存到: {output_dir}")


if __name__ == "__main__":
    main()
