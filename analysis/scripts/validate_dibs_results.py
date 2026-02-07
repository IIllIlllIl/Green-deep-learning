#!/usr/bin/env python3
"""
验证DiBS因果学习结果的脚本

验收标准:
1. 因果图大小与数据特征数完全匹配
2. 每组成功生成因果图文件（正确维度）
3. 检查因果图的质量指标（边强度分布、收敛性）
4. 验证样本数正确（特别是group5为60样本）

用法:
python validate_dibs_results.py --group group5_mrt_oast
python validate_dibs_results.py --all
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

def validate_group_results(group_id, results_dir):
    """验证单个组的DiBS结果"""
    print(f"\n验证组: {group_id}")
    print("-" * 40)

    group_dir = results_dir / group_id
    if not group_dir.exists():
        print(f"❌ 结果目录不存在: {group_dir}")
        return False

    # 检查必需文件
    required_files = [
        f"{group_id}_dibs_causal_graph.csv",
        f"{group_id}_dibs_summary.json",
        f"{group_id}_feature_names.json",
        f"{group_id}_dibs_config.json"
    ]

    missing_files = []
    for file in required_files:
        if not (group_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ 缺失文件: {missing_files}")
        return False

    print("✅ 所有必需文件存在")

    # 1. 验证因果图大小
    try:
        causal_graph_file = group_dir / f"{group_id}_dibs_causal_graph.csv"
        causal_graph_df = pd.read_csv(causal_graph_file, index_col=0)

        # 检查是否为方阵
        n_rows, n_cols = causal_graph_df.shape
        if n_rows != n_cols:
            print(f"❌ 因果图不是方阵: {n_rows}行 × {n_cols}列")
            return False

        print(f"✅ 因果图大小: {n_rows}×{n_cols} (方阵)")

        # 检查特征名称一致性
        features_file = group_dir / f"{group_id}_feature_names.json"
        with open(features_file, 'r') as f:
            feature_names = json.load(f)

        if len(feature_names) != n_rows:
            print(f"❌ 特征数不匹配: 特征名称{len(feature_names)}个, 因果图{n_rows}×{n_cols}")
            return False

        # 检查列名匹配
        if list(causal_graph_df.columns) != feature_names:
            print("❌ 因果图列名与特征名称不匹配")
            return False

        print(f"✅ 特征名称一致: {len(feature_names)}个特征")

    except Exception as e:
        print(f"❌ 验证因果图时出错: {e}")
        return False

    # 2. 验证摘要信息
    try:
        summary_file = group_dir / f"{group_id}_dibs_summary.json"
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        # 检查关键字段
        required_fields = ['samples', 'features', 'edges_gt_0.3', 'strong_edge_percentage']
        for field in required_fields:
            if field not in summary:
                print(f"❌ 摘要中缺失字段: {field}")
                return False

        print(f"✅ 摘要信息完整:")
        print(f"   样本数: {summary['samples']}")
        print(f"   特征数: {summary['features']}")
        print(f"   强边数(>0.3): {summary['edges_gt_0.3']}")
        print(f"   强边比例: {summary['strong_edge_percentage']:.1f}%")

        # 验证样本数（特别检查group5）
        if group_id == "group5_mrt_oast":
            if summary['samples'] != 60:
                print(f"❌ group5样本数不正确: {summary['samples']} (应为60)")
                return False
            else:
                print(f"✅ group5样本数正确: {summary['samples']}样本")

        # 验证特征数与因果图一致
        if summary['features'] != n_rows:
            print(f"❌ 特征数不匹配: 摘要中{summary['features']}, 因果图中{n_rows}")
            return False

        print(f"✅ 特征数一致: {summary['features']}")

    except Exception as e:
        print(f"❌ 验证摘要时出错: {e}")
        return False

    # 3. 验证因果图质量指标
    try:
        # 检查因果图数值范围
        causal_matrix = causal_graph_df.values
        min_val = np.min(causal_matrix)
        max_val = np.max(causal_matrix)
        mean_val = np.mean(causal_matrix)
        std_val = np.std(causal_matrix)

        print(f"✅ 因果图数值统计:")
        print(f"   最小值: {min_val:.6f}")
        print(f"   最大值: {max_val:.6f}")
        print(f"   平均值: {mean_val:.6f}")
        print(f"   标准差: {std_val:.6f}")

        # 检查是否有NaN或Inf
        if np.any(np.isnan(causal_matrix)):
            print("❌ 因果图中包含NaN值")
            return False

        if np.any(np.isinf(causal_matrix)):
            print("❌ 因果图中包含无穷大值")
            return False

        print("✅ 因果图无NaN/Inf值")

        # 检查边强度分布
        edges_001 = np.sum(causal_matrix > 0.01)
        edges_01 = np.sum(causal_matrix > 0.1)
        edges_03 = np.sum(causal_matrix > 0.3)
        edges_05 = np.sum(causal_matrix > 0.5)

        total_possible_edges = n_rows * (n_rows - 1)

        print(f"✅ 边强度分布:")
        print(f"   >0.01: {edges_001}条 ({edges_001/total_possible_edges*100:.1f}%)")
        print(f"   >0.1:  {edges_01}条 ({edges_01/total_possible_edges*100:.1f}%)")
        print(f"   >0.3:  {edges_03}条 ({edges_03/total_possible_edges*100:.1f}%)")
        print(f"   >0.5:  {edges_05}条 ({edges_05/total_possible_edges*100:.1f}%)")

        # 检查强边比例是否合理（通常在1-10%之间）
        strong_edge_pct = edges_03 / total_possible_edges * 100
        if strong_edge_pct < 0.1 or strong_edge_pct > 30:
            print(f"⚠️  强边比例异常: {strong_edge_pct:.1f}% (通常1-10%)")
            # 不视为失败，只是警告

    except Exception as e:
        print(f"❌ 验证因果图质量时出错: {e}")
        return False

    print(f"\n✅ 组 {group_id} 验证通过!")
    return True

def main():
    parser = argparse.ArgumentParser(description="验证DiBS因果学习结果")
    parser.add_argument("--group", type=str, help="验证特定组（如: group5_mrt_oast）")
    parser.add_argument("--all", action="store_true", help="验证所有组")
    parser.add_argument("--results-dir", type=str,
                       default="results/energy_research/data/global_std",
                       help="结果目录")

    args = parser.parse_args()

    if not args.group and not args.all:
        print("请指定 --group 或 --all")
        parser.print_help()
        return

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return

    print("=" * 80)
    print("DiBS结果验证")
    print("=" * 80)

    if args.all:
        # 验证所有组
        groups = [
            "group1_examples", "group2_vulberta", "group3_person_reid",
            "group4_bug_localization", "group5_mrt_oast", "group6_resnet"
        ]

        validation_results = []
        for group_id in groups:
            success = validate_group_results(group_id, results_dir)
            validation_results.append((group_id, success))

        # 汇总结果
        print(f"\n{'='*80}")
        print("验证汇总")
        print(f"{'='*80}")

        total_groups = len(validation_results)
        passed_groups = sum(1 for _, success in validation_results if success)

        for group_id, success in validation_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{group_id}: {status}")

        print(f"\n总组数: {total_groups}")
        print(f"通过组数: {passed_groups}")
        print(f"失败组数: {total_groups - passed_groups}")

        if passed_groups == total_groups:
            print(f"\n🎉 所有组验证通过！")
        else:
            print(f"\n⚠️  {total_groups - passed_groups}个组验证失败")
            sys.exit(1)

    else:
        # 验证特定组
        success = validate_group_results(args.group, results_dir)
        if success:
            print(f"\n🎉 组 {args.group} 验证通过！")
        else:
            print(f"\n❌ 组 {args.group} 验证失败")
            sys.exit(1)

if __name__ == "__main__":
    main()