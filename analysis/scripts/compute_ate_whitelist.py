#!/usr/bin/env python3
"""
为白名单因果边计算ATE (Average Treatment Effect)
使用已验收的CausalInferenceEngine（CTF风格DML方法）

读取白名单CSV文件，为每条因果边计算ATE和置信区间，
并将结果添加到CSV中。

使用方法:
    # Dry run (测试模式)
    python compute_ate_whitelist.py --dry-run

    # 实际运行（更新所有白名单文件）
    python compute_ate_whitelist.py

    # 只处理特定group
    python compute_ate_whitelist.py --group 1

依赖:
    - analysis/utils/causal_inference.py (已验收的CTF风格ATE)
    - EconML 0.14.1 (已安装到causal-research环境)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
import time

# 添加父目录到路径，以便导入utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.causal_inference import CausalInferenceEngine
    ECONML_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入CausalInferenceEngine: {e}")
    print("请确保在causal-research环境中运行: conda activate causal-research")
    ECONML_AVAILABLE = False


def build_causal_graph_from_whitelist(
    whitelist_df: pd.DataFrame,
    var_names: List[str],
    default_strength: float = 0.5
) -> np.ndarray:
    """
    从白名单DataFrame构建因果图邻接矩阵

    参数:
        whitelist_df: 白名单DataFrame，包含source, target, strength列
        var_names: 所有变量名列表
        default_strength: 默认边权重（用于不在白名单中的边，设为0）

    返回:
        causal_graph: (n_vars, n_vars)邻接矩阵
    """
    n_vars = len(var_names)
    causal_graph = np.zeros((n_vars, n_vars))

    # 创建变量名到索引的映射
    var_to_idx = {var: i for i, var in enumerate(var_names)}

    # 填充白名单中的边
    for _, row in whitelist_df.iterrows():
        source = row['source']
        target = row['target']
        strength = row.get('strength', default_strength)

        if source in var_to_idx and target in var_to_idx:
            source_idx = var_to_idx[source]
            target_idx = var_to_idx[target]
            causal_graph[source_idx, target_idx] = strength

    return causal_graph


def process_whitelist_file(
    whitelist_path: str,
    data_path: str,
    dry_run: bool = False,
    threshold: float = 0.3
) -> pd.DataFrame:
    """
    处理单个白名单文件

    参数:
        whitelist_path: 白名单CSV路径
        data_path: 数据CSV路径
        dry_run: 是否只测试不保存
        threshold: 边权重阈值

    返回:
        updated_df: 更新后的DataFrame（包含ATE结果）
    """
    print(f"\n📂 处理白名单: {os.path.basename(whitelist_path)}")
    print(f"   数据文件: {os.path.basename(data_path)}")

    # 读取数据
    try:
        whitelist_df = pd.read_csv(whitelist_path)
        data_df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

    print(f"   ✅ 读取成功: {len(whitelist_df)} 条边, {len(data_df)} 条数据")

    # 数据清洗：处理NaN值
    print(f"   数据清洗...")
    original_rows = len(data_df)
    original_nan = data_df.isna().sum().sum()

    # 数值列：用中位数填充
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data_df[col].isna().sum() > 0:
            median_val = data_df[col].median()
            data_df[col] = data_df[col].fillna(median_val)
            na_count = data_df[col].isna().sum()
            if na_count == 0:
                print(f"      {col}: 用中位数 {median_val:.4f} 填充")

    # 布尔列：用False填充
    bool_cols = data_df.select_dtypes(include=[bool]).columns
    for col in bool_cols:
        if data_df[col].isna().sum() > 0:
            data_df[col] = data_df[col].fillna(False)
            print(f"      {col}: 用False填充")

    # 其他列：用众数填充
    other_cols = [col for col in data_df.columns
                  if col not in numeric_cols and col not in bool_cols]
    for col in other_cols:
        if data_df[col].isna().sum() > 0:
            mode_val = data_df[col].mode()
            if len(mode_val) > 0:
                data_df[col] = data_df[col].fillna(mode_val.iloc[0])
                print(f"      {col}: 用众数 '{mode_val.iloc[0]}' 填充")
            else:
                # 如果众数不存在（全NaN），用第一个非NaN值或默认值
                non_na_vals = data_df[col].dropna()
                if len(non_na_vals) > 0:
                    data_df[col] = data_df[col].fillna(non_na_vals.iloc[0])
                    print(f"      {col}: 用第一个非NaN值填充")
                else:
                    # 如果全NaN，删除该列
                    print(f"      {col}: 全为NaN，删除该列")
                    data_df = data_df.drop(columns=[col])

    cleaned_nan = data_df.isna().sum().sum()
    print(f"   ✅ 清洗完成: 原始NaN {original_nan} → 剩余NaN {cleaned_nan}")
    if cleaned_nan > 0:
        print(f"   ⚠ 警告: 仍有 {cleaned_nan} 个NaN，可能会影响ATE计算")

    # 检查必要的列
    required_whitelist_cols = ['source', 'target']
    missing_cols = [col for col in required_whitelist_cols if col not in whitelist_df.columns]
    if missing_cols:
        print(f"❌ 白名单缺少必要列: {missing_cols}")
        return None

    # 添加ATE列（如果不存在）
    ate_cols = ['ate', 'ate_ci_lower', 'ate_ci_upper', 'ate_is_significant', 'ate_confounders_count']
    for col in ate_cols:
        if col not in whitelist_df.columns:
            if col == 'ate_confounders_count':
                whitelist_df[col] = 0  # 整数
            elif col == 'ate_is_significant':
                whitelist_df[col] = False  # 布尔值，避免dtype警告
            else:
                whitelist_df[col] = np.nan  # 浮点数

    # 获取数据中的所有变量名（排除非特征列）
    exclude_cols = ['timestamp', 'experiment_id', 'session_id']  # 常见非特征列
    feature_cols = [col for col in data_df.columns if col not in exclude_cols]

    # 构建因果图
    print(f"   构建因果图...")
    causal_graph = build_causal_graph_from_whitelist(whitelist_df, feature_cols)

    # 初始化因果推断引擎
    engine = CausalInferenceEngine(verbose=True)

    # 计算每条边的ATE
    print(f"   开始计算ATE...")
    start_time = time.time()

    results = engine.analyze_all_edges_ctf_style(
        data=data_df,
        causal_graph=causal_graph,
        var_names=feature_cols,
        threshold=threshold,
        ref_df=None,  # 暂时不使用ref_df
        t_strategy=None  # 暂时使用默认T0/T1
    )

    elapsed_time = time.time() - start_time
    print(f"   ✅ ATE计算完成，耗时: {elapsed_time:.1f}秒")
    print(f"   成功分析: {len(results)} 条边")

    # 更新白名单DataFrame
    updated_count = 0
    for i, row in whitelist_df.iterrows():
        source = row['source']
        target = row['target']
        edge_key = f"{source}->{target}"

        if edge_key in results:
            result = results[edge_key]
            whitelist_df.at[i, 'ate'] = result['ate']
            whitelist_df.at[i, 'ate_ci_lower'] = result['ci_lower']
            whitelist_df.at[i, 'ate_ci_upper'] = result['ci_upper']
            whitelist_df.at[i, 'ate_is_significant'] = result['is_significant']
            whitelist_df.at[i, 'ate_confounders_count'] = len(result['confounders'])
            updated_count += 1

    print(f"   更新了 {updated_count}/{len(whitelist_df)} 条边的ATE结果")

    # 如果非dry run，保存结果
    if not dry_run:
        output_path = whitelist_path  # 覆盖原文件
        whitelist_df.to_csv(output_path, index=False)
        print(f"   💾 结果已保存到: {output_path}")
    else:
        print(f"   🧪 Dry run模式，未保存文件")

        # 显示前几条结果作为示例
        if updated_count > 0:
            print(f"\n   示例结果 (前3条):")
            sample_df = whitelist_df[['source', 'target', 'ate', 'ate_is_significant']].head(3)
            for _, row in sample_df.iterrows():
                if not pd.isna(row['ate']):
                    print(f"     {row['source']} → {row['target']}: ATE={row['ate']:.3f}, 显著={row['ate_is_significant']}")

    return whitelist_df


def main():
    parser = argparse.ArgumentParser(description='为白名单因果边计算ATE')
    parser.add_argument('--dry-run', action='store_true',
                       help='测试模式，不保存文件')
    parser.add_argument('--group', type=int, choices=range(1, 7),
                       help='只处理特定group (1-6)')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='边权重阈值 (默认: 0.3)')

    args = parser.parse_args()

    if not ECONML_AVAILABLE:
        print("❌ 无法导入CausalInferenceEngine，请检查环境配置")
        print("   确保在causal-research环境中运行: conda activate causal-research")
        print("   并已安装EconML: pip install econml==0.14.1")
        sys.exit(1)

    # 定义白名单和数据文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    whitelist_dir = os.path.join(
        base_dir, 'results', 'energy_research', 'data', 'interaction', 'whitelist'
    )

    data_dir = os.path.join(
        base_dir, 'data', 'energy_research', '6groups_interaction'
    )

    # 定义group映射（使用交互项数据集）
    groups = {
        1: ('group1_examples', 'group1_examples_interaction.csv'),
        2: ('group2_vulberta', 'group2_vulberta_interaction.csv'),
        3: ('group3_person_reid', 'group3_person_reid_interaction.csv'),
        4: ('group4_bug_localization', 'group4_bug_localization_interaction.csv'),
        5: ('group5_mrt_oast', 'group5_mrt_oast_interaction.csv'),
        6: ('group6_resnet', 'group6_resnet_interaction.csv')
    }

    # 确定要处理的groups
    if args.group:
        groups_to_process = {args.group: groups[args.group]}
    else:
        groups_to_process = groups

    print(f"🚀 开始处理白名单ATE计算")
    print(f"   模式: {'🧪 Dry Run' if args.dry_run else '🚀 实际运行'}")
    print(f"   阈值: {args.threshold}")
    print(f"   处理 {len(groups_to_process)} 个group")

    total_start = time.time()
    results = {}

    for group_num, (whitelist_prefix, data_file) in groups_to_process.items():
        whitelist_path = os.path.join(whitelist_dir, f"{whitelist_prefix}_causal_edges_whitelist.csv")
        data_path = os.path.join(data_dir, data_file)

        if not os.path.exists(whitelist_path):
            print(f"❌ 白名单文件不存在: {whitelist_path}")
            continue

        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            continue

        result_df = process_whitelist_file(
            whitelist_path, data_path,
            dry_run=args.dry_run,
            threshold=args.threshold
        )

        if result_df is not None:
            results[group_num] = result_df

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"🎉 处理完成!")
    print(f"   总耗时: {total_time:.1f}秒")
    print(f"   成功处理: {len(results)}/{len(groups_to_process)} 个group")

    if args.dry_run:
        print(f"\n💡 建议:")
        print(f"   1. 检查上述输出是否正常")
        print(f"   2. 确认ATE计算结果符合预期")
        print(f"   3. 移除--dry-run参数实际运行")
    else:
        print(f"\n✅ 所有白名单文件已更新ATE结果")

    return 0


if __name__ == '__main__':
    sys.exit(main())