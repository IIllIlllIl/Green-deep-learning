#!/usr/bin/env python3
"""
全局标准化数据敏感性分析脚本

分析全局标准化ATE对以下因素的敏感性:
1. 边强度阈值 (0.1, 0.3, 0.5)
2. 数据清洗方法 (不同NaN填充策略)
3. 混淆因素选择 (不同阈值)

使用方法:
    python sensitivity_analysis_global_std.py --group 1
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt

# 添加父目录到路径，以便导入utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.causal_inference import CausalInferenceEngine
    ECONML_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入CausalInferenceEngine: {e}")
    print("请确保在causal-research环境中运行: conda activate causal-research")
    ECONML_AVAILABLE = False


def load_data_and_whitelist(group_id: str, global_std_dir: str, whitelist_dir: str):
    """加载数据和白名单"""
    global_std_file = os.path.join(global_std_dir, f"{group_id}_global_std.csv")
    whitelist_file = os.path.join(whitelist_dir, f"{group_id}_causal_edges_whitelist.csv")

    if not os.path.exists(global_std_file):
        raise FileNotFoundError(f"全局标准化数据文件不存在: {global_std_file}")
    if not os.path.exists(whitelist_file):
        raise FileNotFoundError(f"白名单文件不存在: {whitelist_file}")

    data_df = pd.read_csv(global_std_file)
    whitelist_df = pd.read_csv(whitelist_file)

    return data_df, whitelist_df


def clean_data_with_strategy(data_df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """使用不同策略清洗数据"""
    cleaned_df = data_df.copy()

    if strategy == "median":
        # 数值列：用中位数填充
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isna().sum() > 0:
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)

    elif strategy == "mean":
        # 数值列：用均值填充
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isna().sum() > 0:
                mean_val = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(mean_val)

    elif strategy == "drop":
        # 删除包含NaN的行
        cleaned_df = cleaned_df.dropna()

    elif strategy == "zero":
        # 用0填充
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isna().sum() > 0:
                cleaned_df[col] = cleaned_df[col].fillna(0)

    # 布尔列：用False填充
    bool_cols = cleaned_df.select_dtypes(include=[bool]).columns
    for col in bool_cols:
        if cleaned_df[col].isna().sum() > 0:
            cleaned_df[col] = cleaned_df[col].fillna(False)

    return cleaned_df


def build_causal_graph_from_whitelist(whitelist_df: pd.DataFrame, var_names: List[str], threshold: float = 0.3):
    """从白名单构建因果图"""
    n_vars = len(var_names)
    causal_graph = np.zeros((n_vars, n_vars))

    # 创建变量名到索引的映射
    var_to_idx = {var: idx for idx, var in enumerate(var_names)}

    # 填充邻接矩阵
    for _, row in whitelist_df.iterrows():
        source = row['source']
        target = row['target']
        strength = row.get('strength', 0.5)

        if source in var_to_idx and target in var_to_idx and strength > threshold:
            source_idx = var_to_idx[source]
            target_idx = var_to_idx[target]
            causal_graph[source_idx, target_idx] = strength

    return causal_graph


def run_sensitivity_analysis_for_group(
    group_num: int,
    global_std_dir: str,
    whitelist_dir: str,
    output_dir: str
) -> Dict:
    """为单个组运行敏感性分析"""
    # 组名映射
    group_mapping = {
        1: "group1_examples",
        2: "group2_vulberta",
        3: "group3_person_reid",
        4: "group4_bug_localization",
        5: "group5_mrt_oast",
        6: "group6_resnet"
    }

    group_id = group_mapping.get(group_num, f"group{group_num}")

    print(f"\n{'='*80}")
    print(f"敏感性分析: {group_id}")
    print(f"{'='*80}")

    # 1. 加载数据
    print(f"1. 加载数据...")
    try:
        data_df, whitelist_df = load_data_and_whitelist(group_id, global_std_dir, whitelist_dir)
        print(f"   ✅ 数据加载成功: {len(data_df)} 行, {len(whitelist_df)} 条边")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return {"success": False, "error": str(e)}

    # 2. 敏感性分析1: 不同边强度阈值
    print(f"\n2. 敏感性分析1: 不同边强度阈值")
    threshold_results = {}

    thresholds = [0.1, 0.3, 0.5]
    for threshold in thresholds:
        print(f"   阈值: {threshold}")
        try:
            # 获取特征列
            exclude_cols = ['timestamp', 'experiment_id', 'session_id']
            feature_cols = [col for col in data_df.columns if col not in exclude_cols]

            # 构建因果图
            causal_graph = build_causal_graph_from_whitelist(whitelist_df, feature_cols, threshold)

            # 计算边数
            edge_count = np.sum(causal_graph > 0)

            threshold_results[threshold] = {
                "edge_count": int(edge_count),
                "edge_percentage": float(edge_count / (len(feature_cols) * (len(feature_cols) - 1)) * 100)
            }

            print(f"     边数: {edge_count}条 ({threshold_results[threshold]['edge_percentage']:.1f}%)")

        except Exception as e:
            print(f"     ❌ 阈值 {threshold} 分析失败: {e}")
            threshold_results[threshold] = {"error": str(e)}

    # 3. 敏感性分析2: 不同数据清洗方法
    print(f"\n3. 敏感性分析2: 不同数据清洗方法")
    cleaning_results = {}

    cleaning_strategies = ["median", "mean", "drop", "zero"]
    original_nan = data_df.isna().sum().sum()

    for strategy in cleaning_strategies:
        print(f"   清洗策略: {strategy}")
        try:
            # 清洗数据
            cleaned_df = clean_data_with_strategy(data_df, strategy)

            # 统计信息
            remaining_nan = cleaned_df.isna().sum().sum()
            rows_removed = len(data_df) - len(cleaned_df) if strategy == "drop" else 0

            cleaning_results[strategy] = {
                "original_nan": int(original_nan),
                "remaining_nan": int(remaining_nan),
                "rows_removed": int(rows_removed),
                "final_rows": len(cleaned_df),
                "nan_reduction_percent": float((original_nan - remaining_nan) / original_nan * 100) if original_nan > 0 else 100.0
            }

            print(f"     原始NaN: {original_nan}, 剩余NaN: {remaining_nan}")
            print(f"     行数变化: {len(data_df)} → {len(cleaned_df)} ({rows_removed}行被删除)")
            print(f"     NaN减少: {cleaning_results[strategy]['nan_reduction_percent']:.1f}%")

        except Exception as e:
            print(f"     ❌ 清洗策略 {strategy} 失败: {e}")
            cleaning_results[strategy] = {"error": str(e)}

    # 4. 敏感性分析3: ATE对阈值的敏感性（抽样分析）
    print(f"\n4. 敏感性分析3: ATE对阈值的敏感性（抽样分析）")
    ate_sensitivity_results = {}

    if ECONML_AVAILABLE:
        # 选择几条重要的边进行分析
        important_edges = whitelist_df.nlargest(5, 'strength')[['source', 'target', 'strength']].to_dict('records')

        for edge in important_edges:
            source = edge['source']
            target = edge['target']
            strength = edge['strength']

            print(f"   分析边: {source} → {target} (强度: {strength:.3f})")

            edge_results = {}
            for threshold in [0.1, 0.3, 0.5]:
                try:
                    # 使用中位数清洗策略
                    cleaned_df = clean_data_with_strategy(data_df, "median")

                    # 获取特征列
                    exclude_cols = ['timestamp', 'experiment_id', 'session_id']
                    feature_cols = [col for col in cleaned_df.columns if col not in exclude_cols]

                    # 构建因果图
                    causal_graph = build_causal_graph_from_whitelist(whitelist_df, feature_cols, threshold)

                    # 检查边是否存在
                    if source in feature_cols and target in feature_cols:
                        source_idx = feature_cols.index(source)
                        target_idx = feature_cols.index(target)

                        if causal_graph[source_idx, target_idx] > 0:
                            # 获取混淆因素
                            engine = CausalInferenceEngine(verbose=False)
                            confounders = engine._get_confounders_from_graph(
                                source, target, causal_graph, feature_cols, threshold
                            )

                            # 计算ATE
                            ate, ci = engine.estimate_ate(
                                cleaned_df, source, target, confounders
                            )

                            edge_results[threshold] = {
                                "ate": float(ate),
                                "ci_lower": float(ci[0]),
                                "ci_upper": float(ci[1]),
                                "is_significant": not (ci[0] <= 0 <= ci[1]),
                                "confounders_count": len(confounders)
                            }

                            print(f"     阈值 {threshold}: ATE={ate:.4f}, CI=[{ci[0]:.4f}, {ci[1]:.4f}], 混淆因素={len(confounders)}")
                        else:
                            edge_results[threshold] = {"error": "边被阈值过滤"}
                            print(f"     阈值 {threshold}: 边被过滤")
                    else:
                        edge_results[threshold] = {"error": "变量不存在"}
                        print(f"     阈值 {threshold}: 变量不存在")

                except Exception as e:
                    edge_results[threshold] = {"error": str(e)}
                    print(f"     阈值 {threshold}: 计算失败 - {e}")

            ate_sensitivity_results[f"{source}->{target}"] = edge_results

    # 5. 保存结果
    print(f"\n5. 保存敏感性分析结果...")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "group_id": group_id,
        "group_num": group_num,
        "threshold_sensitivity": threshold_results,
        "cleaning_sensitivity": cleaning_results,
        "ate_sensitivity": ate_sensitivity_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    output_file = os.path.join(output_dir, f"{group_id}_sensitivity_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   ✅ 结果已保存: {output_file}")

    # 6. 生成摘要
    print(f"\n6. 生成摘要...")

    # 阈值敏感性摘要
    print(f"   阈值敏感性摘要:")
    for threshold, result in threshold_results.items():
        if "edge_count" in result:
            print(f"     阈值 {threshold}: {result['edge_count']}条边 ({result['edge_percentage']:.1f}%)")

    # 清洗敏感性摘要
    print(f"\n   清洗敏感性摘要:")
    for strategy, result in cleaning_results.items():
        if "final_rows" in result:
            print(f"     {strategy}: {result['final_rows']}行, NaN减少{result['nan_reduction_percent']:.1f}%")

    # ATE敏感性摘要
    if ate_sensitivity_results:
        print(f"\n   ATE敏感性摘要:")
        for edge, edge_results in ate_sensitivity_results.items():
            print(f"     {edge}:")
            for threshold, result in edge_results.items():
                if "ate" in result:
                    print(f"       阈值 {threshold}: ATE={result['ate']:.4f}, 显著={result['is_significant']}")

    return {
        "success": True,
        "results_file": output_file,
        "summary": {
            "threshold_sensitivity": len(threshold_results),
            "cleaning_sensitivity": len(cleaning_results),
            "ate_sensitivity": len(ate_sensitivity_results)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="全局标准化数据敏感性分析")
    parser.add_argument("--group", type=int, choices=range(1, 7),
                       help="只处理指定组（1-6），不指定则处理所有组")
    parser.add_argument("--global-std-dir", type=str,
                       default="data/energy_research/6groups_global_std",
                       help="全局标准化数据目录")
    parser.add_argument("--whitelist-dir", type=str,
                       default="results/energy_research/data/interaction/whitelist",
                       help="白名单目录")
    parser.add_argument("--output-dir", type=str,
                       default="results/energy_research/data/sensitivity_analysis",
                       help="输出目录")

    args = parser.parse_args()

    print("=" * 80)
    print("全局标准化数据敏感性分析")
    print("=" * 80)

    print(f"\n配置:")
    print(f"  全局标准化数据目录: {args.global_std_dir}")
    print(f"  白名单目录: {args.whitelist_dir}")
    print(f"  输出目录: {args.output_dir}")

    # 检查EconML是否可用
    if not ECONML_AVAILABLE:
        print("\n❌ 无法导入CausalInferenceEngine")
        print("请确保在causal-research环境中运行: conda activate causal-research")
        return 1

    # 确定要处理的组
    if args.group:
        groups_to_process = [args.group]
        print(f"\n处理指定组: {args.group}")
    else:
        groups_to_process = list(range(1, 7))
        print(f"\n处理所有组: 1-6")

    # 处理每个组
    all_results = []

    for group_num in groups_to_process:
        result = run_sensitivity_analysis_for_group(
            group_num=group_num,
            global_std_dir=args.global_std_dir,
            whitelist_dir=args.whitelist_dir,
            output_dir=args.output_dir
        )

        all_results.append({
            "group": group_num,
            **result
        })

    # 生成总报告
    print(f"\n{'='*80}")
    print("敏感性分析总报告")
    print(f"{'='*80}")

    successful_groups = [r for r in all_results if r.get('success', False)]
    failed_groups = [r for r in all_results if not r.get('success', False)]

    print(f"\n处理完成:")
    print(f"  成功组数: {len(successful_groups)}")
    print(f"  失败组数: {len(failed_groups)}")

    # 保存总报告
    if successful_groups:
        total_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "global_std_dir": args.global_std_dir,
                "whitelist_dir": args.whitelist_dir,
                "output_dir": args.output_dir
            },
            "results": all_results,
            "summary": {
                "total_groups": len(all_results),
                "successful_groups": len(successful_groups),
                "failed_groups": len(failed_groups)
            }
        }

        total_report_file = os.path.join(args.output_dir, "sensitivity_analysis_total_report.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(total_report_file, 'w') as f:
            json.dump(total_report, f, indent=2)

        print(f"\n✅ 总报告已保存: {total_report_file}")

    print(f"\n{'='*80}")
    print("所有任务完成")
    print(f"{'='*80}")

    return 0 if len(failed_groups) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())