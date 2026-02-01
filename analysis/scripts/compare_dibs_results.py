#!/usr/bin/env python3
"""
对比两种标准化方法的DiBS因果图差异

目的：验证是否需要重新运行DiBS
方法：对比group1_examples在两种标准化数据上的DiBS结果
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set


def load_dibs_results(group_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载两种标准化方法的DiBS结果"""

    # 路径配置
    base_dir = Path("/home/green/energy_dl/nightly/analysis")

    # 1. 组内标准化的DiBS结果
    within_dir = base_dir / "results" / "energy_research" / "data" / "interaction"
    within_files = [
        within_dir / "dibs_particle_edges" / f"{group_name}_dibs_particle_edges.csv",
        within_dir / "dibs_mean_adjacency" / f"{group_name}_dibs_mean_adjacency.csv",
        within_dir / "dibs_edge_probabilities" / f"{group_name}_dibs_edge_probabilities.csv"
    ]

    # 2. 全局标准化的DiBS结果（如果存在）
    global_dir = base_dir / "results" / "energy_research" / "data" / "global_std"

    # 先检查全局标准化是否已运行DiBS
    global_results_exist = False
    for file in global_dir.rglob(f"*{group_name}*"):
        if file.is_file():
            global_results_exist = True
            break

    print(f"加载 {group_name} 的DiBS结果:")
    print(f"- 组内标准化结果: {'✅' if within_files[0].exists() else '❌'}")
    print(f"- 全局标准化结果: {'✅' if global_results_exist else '❌'}")

    # 加载组内标准化结果
    within_edges = None
    if within_files[0].exists():
        within_edges = pd.read_csv(within_files[0])
        print(f"  组内标准化: {len(within_edges)} 条边")

    # 加载全局标准化结果（如果存在）
    global_edges = None
    if global_results_exist:
        # 查找最新的全局标准化结果
        global_edge_files = list(global_dir.rglob(f"*{group_name}*edges*.csv"))
        if global_edge_files:
            global_edges = pd.read_csv(global_edge_files[0])
            print(f"  全局标准化: {len(global_edges)} 条边")

    return within_edges, global_edges


def compare_edge_sets(within_edges: pd.DataFrame, global_edges: pd.DataFrame) -> Dict:
    """对比两个边集合的差异"""

    # 创建边标识符集合
    def create_edge_set(edges_df: pd.DataFrame) -> Set[str]:
        if edges_df is None or len(edges_df) == 0:
            return set()
        return set(f"{row['source']}→{row['target']}" for _, row in edges_df.iterrows())

    within_set = create_edge_set(within_edges)
    global_set = create_edge_set(global_edges)

    # 计算重叠指标
    intersection = within_set.intersection(global_set)
    union = within_set.union(global_set)

    results = {
        "within_edges": len(within_set),
        "global_edges": len(global_set),
        "common_edges": len(intersection),
        "jaccard_similarity": len(intersection) / len(union) if len(union) > 0 else 0,
        "within_only": len(within_set - global_set),
        "global_only": len(global_set - within_set),
        "common_edge_list": list(intersection)[:20]  # 只显示前20个
    }

    return results


def compare_edge_strengths(within_edges: pd.DataFrame, global_edges: pd.DataFrame) -> pd.DataFrame:
    """对比共同边的强度差异"""

    if within_edges is None or global_edges is None:
        return pd.DataFrame()

    # 创建边到强度的映射
    within_strengths = {}
    for _, row in within_edges.iterrows():
        edge_key = f"{row['source']}→{row['target']}"
        within_strengths[edge_key] = row.get('strength', row.get('probability', 0.5))

    global_strengths = {}
    for _, row in global_edges.iterrows():
        edge_key = f"{row['source']}→{row['target']}"
        global_strengths[edge_key] = row.get('strength', row.get('probability', 0.5))

    # 找出共同边
    common_edges = set(within_strengths.keys()).intersection(set(global_strengths.keys()))

    # 创建对比DataFrame
    comparison_data = []
    for edge in common_edges:
        comparison_data.append({
            'edge': edge,
            'within_strength': within_strengths[edge],
            'global_strength': global_strengths[edge],
            'strength_diff': global_strengths[edge] - within_strengths[edge],
            'abs_strength_diff': abs(global_strengths[edge] - within_strengths[edge])
        })

    return pd.DataFrame(comparison_data)


def analyze_hyperparameter_edges(edges_df: pd.DataFrame) -> Dict:
    """分析超参数相关边"""

    if edges_df is None or len(edges_df) == 0:
        return {}

    # 定义超参数相关的边
    hyperparam_keywords = ['hyperparam_', 'learning_rate', 'epochs', 'batch_size',
                          'dropout', 'l2_regularization', 'seed', 'alpha', 'kfold']

    hyperparam_edges = []
    energy_edges = []
    perf_edges = []

    for _, row in edges_df.iterrows():
        source = str(row['source'])
        target = str(row['target'])

        # 检查是否涉及超参数
        is_hyperparam_edge = any(keyword in source or keyword in target
                                 for keyword in hyperparam_keywords)

        if is_hyperparam_edge:
            edge_type = 'unknown'
            if any('energy' in col for col in [source, target]):
                edge_type = 'hyperparam→energy'
            elif any('perf' in col for col in [source, target]):
                edge_type = 'hyperparam→perf'
            elif any('hyperparam' in col for col in [source, target]):
                edge_type = 'hyperparam→hyperparam'

            hyperparam_edges.append({
                'edge': f"{source}→{target}",
                'type': edge_type,
                'strength': row.get('strength', row.get('probability', 0.5))
            })

        # 分类其他边
        if 'energy' in source and 'perf' in target:
            energy_edges.append(f"{source}→{target}")
        elif 'perf' in source and 'energy' in target:
            energy_edges.append(f"{source}→{target}")

    # 按强度排序
    hyperparam_edges_sorted = sorted(hyperparam_edges, key=lambda x: x['strength'], reverse=True)

    return {
        'total_hyperparam_edges': len(hyperparam_edges),
        'hyperparam_energy_edges': len([e for e in hyperparam_edges if e['type'] == 'hyperparam→energy']),
        'hyperparam_perf_edges': len([e for e in hyperparam_edges if e['type'] == 'hyperparam→perf']),
        'top_hyperparam_edges': hyperparam_edges_sorted[:10],
        'energy_perf_edges': len(energy_edges)
    }


def visualize_comparison(within_edges: pd.DataFrame, global_edges: pd.DataFrame,
                        group_name: str, output_dir: Path):
    """可视化对比结果"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 边集合对比（韦恩图）
    plt.figure(figsize=(10, 8))

    from matplotlib_venn import venn2

    within_set = set(f"{row['source']}→{row['target']}" for _, row in within_edges.iterrows()) if within_edges is not None else set()
    global_set = set(f"{row['source']}→{row['target']}" for _, row in global_edges.iterrows()) if global_edges is not None else set()

    venn = venn2([within_set, global_set],
                set_labels=('Within-Group\nStandardization', 'Global\nStandardization'),
                set_colors=('#FF6B6B', '#4ECDC4'))

    plt.title(f'DiBS Edge Sets Comparison: {group_name}', fontsize=16, fontweight='bold')
    plt.savefig(output_dir / f'{group_name}_venn_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 边强度对比散点图（如果有共同边）
    strength_comparison = compare_edge_strengths(within_edges, global_edges)
    if len(strength_comparison) > 0:
        plt.figure(figsize=(10, 8))

        plt.scatter(strength_comparison['within_strength'],
                   strength_comparison['global_strength'],
                   alpha=0.6, s=50)

        # 添加对角线（y=x）
        min_val = min(strength_comparison[['within_strength', 'global_strength']].min().min(), 0)
        max_val = max(strength_comparison[['within_strength', 'global_strength']].max().max(), 1)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')

        plt.xlabel('Within-Group Standardization Edge Strength', fontsize=12)
        plt.ylabel('Global Standardization Edge Strength', fontsize=12)
        plt.title(f'Edge Strength Correlation: {group_name}', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 计算相关系数
        correlation = strength_comparison['within_strength'].corr(strength_comparison['global_strength'])
        plt.text(0.05, 0.95, f'Pearson r = {correlation:.3f}',
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.savefig(output_dir / f'{group_name}_strength_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"可视化保存至: {output_dir}")


def main():
    """主函数"""

    print("=" * 80)
    print("DiBS结果对比分析：组内标准化 vs 全局标准化")
    print("=" * 80)

    # 配置
    group_name = "group1_examples"
    output_dir = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/reports/dibs_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print(f"\n1. 加载 {group_name} 的DiBS结果")
    print("-" * 40)

    within_edges, global_edges = load_dibs_results(group_name)

    if within_edges is None:
        print("❌ 未找到组内标准化DiBS结果")
        return

    # 2. 对比边集合
    print(f"\n2. 边集合对比")
    print("-" * 40)

    edge_comparison = compare_edge_sets(within_edges, global_edges)

    print(f"组内标准化边数: {edge_comparison['within_edges']}")
    print(f"全局标准化边数: {edge_comparison['global_edges']}")
    print(f"共同边数: {edge_comparison['common_edges']}")
    print(f"Jaccard相似度: {edge_comparison['jaccard_similarity']:.3f}")
    print(f"仅组内标准化有的边: {edge_comparison['within_only']}")
    print(f"仅全局标准化有的边: {edge_comparison['global_only']}")

    # 3. 分析超参数相关边
    print(f"\n3. 超参数相关边分析")
    print("-" * 40)

    within_hyperparam = analyze_hyperparameter_edges(within_edges)
    global_hyperparam = analyze_hyperparameter_edges(global_edges)

    print("组内标准化:")
    print(f"  超参数边总数: {within_hyperparam.get('total_hyperparam_edges', 0)}")
    print(f"  超参数→能耗边: {within_hyperparam.get('hyperparam_energy_edges', 0)}")
    print(f"  超参数→性能边: {within_hyperparam.get('hyperparam_perf_edges', 0)}")

    if global_edges is not None:
        print("\n全局标准化:")
        print(f"  超参数边总数: {global_hyperparam.get('total_hyperparam_edges', 0)}")
        print(f"  超参数→能耗边: {global_hyperparam.get('hyperparam_energy_edges', 0)}")
        print(f"  超参数→性能边: {global_hyperparam.get('hyperparam_perf_edges', 0)}")

    # 4. 详细对比（如果有全局标准化结果）
    if global_edges is not None:
        print(f"\n4. 详细边强度对比")
        print("-" * 40)

        strength_df = compare_edge_strengths(within_edges, global_edges)
        if len(strength_df) > 0:
            print(f"共同边数: {len(strength_df)}")
            print(f"平均强度差异: {strength_df['abs_strength_diff'].mean():.4f}")
            print(f"最大强度差异: {strength_df['abs_strength_diff'].max():.4f}")

            # 显示差异最大的边
            print("\n差异最大的5条边:")
            top_diff = strength_df.nlargest(5, 'abs_strength_diff')
            for _, row in top_diff.iterrows():
                print(f"  {row['edge']}: {row['within_strength']:.3f} → {row['global_strength']:.3f} (Δ={row['strength_diff']:.3f})")

    # 5. 可视化
    print(f"\n5. 生成可视化")
    print("-" * 40)

    if global_edges is not None:
        visualize_comparison(within_edges, global_edges, group_name, output_dir)
    else:
        print("⚠️ 未找到全局标准化DiBS结果，跳过可视化")
        print("建议先运行全局标准化DiBS:")
        print("  python scripts/run_dibs_6groups_global_std.py --group group1_examples")

    # 6. 结论与建议
    print(f"\n6. 结论与建议")
    print("-" * 40)

    if global_edges is None:
        print("❌ 未找到全局标准化DiBS结果")
        print("\n建议执行以下步骤:")
        print("1. 预处理全局标准化数据供DiBS使用")
        print("2. 运行DiBS（至少对group1_examples）")
        print("3. 重新运行本对比分析")
        print("\n快速命令:")
        print("  python scripts/preprocess_for_dibs.py --method global_std --group group1_examples")
        print("  python scripts/run_dibs_6groups_global_std.py --group group1_examples")
    else:
        jaccard = edge_comparison['jaccard_similarity']

        if jaccard > 0.8:
            print("✅ 高重叠率 (>80%)：标准化方法对因果结构影响较小")
            print("建议：可以直接使用现有白名单重新计算ATE")
        elif jaccard > 0.5:
            print("⚠️ 中等重叠率 (50-80%)：标准化方法对因果结构有中等影响")
            print("建议：重新运行DiBS以获得更准确的结果")
        else:
            print("❌ 低重叠率 (<50%)：标准化方法显著改变因果结构")
            print("建议：必须重新运行DiBS，现有白名单可能不适用")

        print(f"\n具体指标: Jaccard相似度 = {jaccard:.3f}")

    # 7. 保存详细报告
    report_data = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'group_name': group_name,
        'edge_comparison': edge_comparison,
        'within_hyperparam_analysis': within_hyperparam,
        'global_hyperparam_analysis': global_hyperparam if global_edges is not None else None,
        'recommendation': ''
    }

    if global_edges is not None:
        if jaccard > 0.8:
            report_data['recommendation'] = 'use_existing_whitelist'
        elif jaccard > 0.5:
            report_data['recommendation'] = 'rerun_dibs_partial'
        else:
            report_data['recommendation'] = 'rerun_dibs_full'
    else:
        report_data['recommendation'] = 'run_dibs_first'

    report_file = output_dir / f'{group_name}_dibs_comparison_report.json'
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\n详细报告保存至: {report_file}")
    print("\n" + "=" * 80)
    print("对比分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
