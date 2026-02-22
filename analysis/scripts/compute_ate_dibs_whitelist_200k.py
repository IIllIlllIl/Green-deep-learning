#!/usr/bin/env python3
"""
基于DiBS白名单因果图的全局标准化数据ATE计算脚本（方案C，200k步）

基于白名单过滤后的DiBS因果图计算全局标准化数据的ATE
使用已验收的CausalInferenceEngine（CTF风格DML方法）

数据源:
  - 全局标准化数据: data/energy_research/6groups_global_std/
  - DiBS白名单边: results/energy_research/data/global_std_whitelist_200k/

输出:
  - ATE结果文件（每条边的ATE估计、置信区间、显著性）
  - 保存到: results/energy_research/data/global_std_dibs_ate_200k/

使用方法:
    # Dry run (测试模式)
    python compute_ate_dibs_whitelist_200k.py --dry-run

    # 实际运行（更新所有组）
    python compute_ate_dibs_whitelist_200k.py

    # 只处理特定group
    python compute_ate_dibs_whitelist_200k.py --group 1

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
import json
from pathlib import Path
import shutil

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
    从白名单构建因果图邻接矩阵

    参数:
        whitelist_df: 白名单DataFrame，包含source和target列
        var_names: 所有变量名列表
        default_strength: 默认边强度

    返回:
        causal_graph: 邻接矩阵，shape (n_vars, n_vars)
    """
    n_vars = len(var_names)
    causal_graph = np.zeros((n_vars, n_vars))

    # 创建变量名到索引的映射
    var_to_idx = {var: idx for idx, var in enumerate(var_names)}

    # 填充邻接矩阵
    for _, row in whitelist_df.iterrows():
        source = row['source']
        target = row['target']

        if source in var_to_idx and target in var_to_idx:
            source_idx = var_to_idx[source]
            target_idx = var_to_idx[target]

            # 使用strength列的值，如果不存在则使用默认值
            strength = row.get('strength', default_strength)
            if pd.isna(strength):
                strength = default_strength
            causal_graph[source_idx, target_idx] = strength

    return causal_graph


def load_whitelist_edges(whitelist_file: str) -> pd.DataFrame:
    """
    加载白名单边CSV文件

    参数:
        whitelist_file: 白名单边CSV文件路径

    返回:
        whitelist_df: DataFrame包含所有边信息
    """
    if not os.path.exists(whitelist_file):
        raise FileNotFoundError(f"白名单文件不存在: {whitelist_file}")

    # 读取CSV文件
    whitelist_df = pd.read_csv(whitelist_file)

    # 验证必要的列是否存在
    required_cols = ['source', 'target']
    for col in required_cols:
        if col not in whitelist_df.columns:
            raise ValueError(f"白名单文件缺少必要列 '{col}'")

    # 检查是否有重复边
    edge_pairs = whitelist_df[['source', 'target']].drop_duplicates()
    if len(edge_pairs) < len(whitelist_df):
        print(f"⚠ 警告: 白名单文件中有重复边，已自动去重")
        whitelist_df = whitelist_df.drop_duplicates(subset=['source', 'target'])

    return whitelist_df


def compute_ate_for_group(
    group_num: int,
    global_std_dir: str,
    whitelist_dir: str,
    output_dir: str,
    dry_run: bool = False,
    threshold: float = 0.3
) -> Dict:
    """
    为单个组计算基于白名单边的全局标准化ATE

    返回:
        results: 包含处理结果的字典
    """
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
    print(f"处理组 {group_num}: {group_id}")
    print(f"{'='*80}")

    # 1. 构建文件路径
    global_std_file = os.path.join(global_std_dir, f"{group_id}_global_std.csv")
    whitelist_file = os.path.join(whitelist_dir, group_id, f"{group_id}_dibs_edges_whitelist.csv")
    output_file = os.path.join(output_dir, f"{group_id}_dibs_whitelist_200k_ate.csv")

    print(f"   全局标准化数据: {global_std_file}")
    print(f"   白名单边文件: {whitelist_file}")
    print(f"   输出文件: {output_file}")

    # 检查文件是否存在
    if not os.path.exists(global_std_file):
        print(f"❌ 全局标准化数据文件不存在: {global_std_file}")
        return {"success": False, "error": f"数据文件不存在: {global_std_file}"}

    if not os.path.exists(whitelist_file):
        print(f"❌ 白名单边文件不存在: {whitelist_file}")
        return {"success": False, "error": f"白名单边文件不存在: {whitelist_file}"}

    # 2. 读取数据
    print(f"   读取数据...")
    try:
        data_df = pd.read_csv(global_std_file)
        # 加载白名单边
        whitelist_df = load_whitelist_edges(whitelist_file)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return {"success": False, "error": f"读取文件失败: {e}"}

    print(f"   ✅ 读取成功: {len(data_df)} 条数据, {len(whitelist_df)} 条白名单边")

    if dry_run:
        print(f"   🧪 Dry run模式 - 只检查数据，不计算ATE")
        # 统计边信息
        unique_sources = whitelist_df['source'].nunique()
        unique_targets = whitelist_df['target'].nunique()
        all_vars = set(whitelist_df['source']).union(set(whitelist_df['target']))

        # 检查数据中是否存在白名单变量
        missing_vars = []
        for var in all_vars:
            if var not in data_df.columns:
                missing_vars.append(var)

        if missing_vars:
            print(f"   ⚠ 警告: {len(missing_vars)} 个变量在数据中不存在:")
            for var in missing_vars[:5]:
                print(f"      {var}")
            if len(missing_vars) > 5:
                print(f"      ... 共{len(missing_vars)}个变量")

        return {"success": True, "dry_run": True, "data_rows": len(data_df),
                "whitelist_edges": len(whitelist_df), "unique_sources": unique_sources,
                "unique_targets": unique_targets, "missing_vars": len(missing_vars)}

    # 3. 数据清洗：处理NaN值
    print(f"   数据清洗...")
    original_rows = len(data_df)
    original_nan = data_df.isna().sum().sum()

    # 首先删除全为NaN的数值列（防御性编程）
    all_nan_numeric_cols = []
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data_df[col].isna().all():
            all_nan_numeric_cols.append(col)

    if all_nan_numeric_cols:
        print(f"   删除全NaN数值列: {len(all_nan_numeric_cols)} 个")
        for col in all_nan_numeric_cols[:5]:  # 只显示前5个
            print(f"      {col}: 全为NaN，删除")
        if len(all_nan_numeric_cols) > 5:
            print(f"      ... 共{len(all_nan_numeric_cols)}个列")
        data_df = data_df.drop(columns=all_nan_numeric_cols)
        numeric_cols = data_df.select_dtypes(include=[np.number]).columns

    # 数值列：用中位数填充
    for col in numeric_cols:
        if data_df[col].isna().sum() > 0:
            median_val = data_df[col].median()
            # 检查median_val是否为NaN（全NaN列应该已被删除，但以防万一）
            if pd.isna(median_val):
                print(f"      ⚠ {col}: 中位数为NaN，用0填充")
                data_df[col] = data_df[col].fillna(0)
            else:
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

    # 4. 准备变量列表
    # 从白名单边中获取所有变量
    all_vars_in_edges = set(whitelist_df['source']).union(set(whitelist_df['target']))

    # 从数据中获取所有特征列（排除非特征列）
    exclude_cols = ['timestamp', 'experiment_id', 'session_id']
    data_feature_cols = [col for col in data_df.columns if col not in exclude_cols]

    # 找出白名单边和数据共有的变量
    common_vars = [var for var in all_vars_in_edges if var in data_feature_cols]
    edge_only_vars = [var for var in all_vars_in_edges if var not in data_feature_cols]
    data_only_vars = [var for var in data_feature_cols if var not in all_vars_in_edges]

    print(f"   变量匹配分析:")
    print(f"     - 白名单边中变量: {len(all_vars_in_edges)} 个")
    print(f"     - 数据中特征变量: {len(data_feature_cols)} 个")
    print(f"     - 共有变量: {len(common_vars)} 个")
    if edge_only_vars:
        print(f"     - 仅白名单中有: {len(edge_only_vars)} 个 (例如: {', '.join(edge_only_vars[:3])}{'...' if len(edge_only_vars) > 3 else ''})")
    if data_only_vars:
        print(f"     - 仅数据中有: {len(data_only_vars)} 个 (仅统计前3个...)")

    # 如果共有的变量太少，无法进行有意义的ATE计算
    if len(common_vars) < 5:
        print(f"❌ 共有变量太少: 仅{len(common_vars)}个，无法进行有意义的ATE计算")
        return {"success": False, "error": "共有变量太少"}

    # 5. 过滤白名单边，只保留共有变量的边
    # 首先复制原始白名单数据，不修改原始数据
    filtered_whitelist = whitelist_df.copy()

    # 过滤掉包含非共有变量的边
    original_edge_count = len(filtered_whitelist)
    filtered_whitelist = filtered_whitelist[
        filtered_whitelist['source'].isin(common_vars) &
        filtered_whitelist['target'].isin(common_vars)
    ]
    filtered_edge_count = len(filtered_whitelist)

    removed_edge_count = original_edge_count - filtered_edge_count
    if removed_edge_count > 0:
        print(f"   ⚠ 过滤掉 {removed_edge_count} 条包含非共有变量的边")
        print(f"   ✅ 过滤后保留 {filtered_edge_count} 条边")

    if filtered_edge_count == 0:
        print(f"❌ 过滤后无有效边")
        return {"success": False, "error": "过滤后无有效边"}

    # 6. 构建因果图邻接矩阵
    print(f"   构建因果图邻接矩阵...")
    # 使用sorted确保变量顺序一致
    sorted_common_vars = sorted(common_vars)
    causal_graph = build_causal_graph_from_whitelist(
        filtered_whitelist,
        sorted_common_vars,
        default_strength=0.5
    )

    # 验证邻接矩阵的非零边数应与白名单边数一致
    non_zero_edges = np.sum(causal_graph > 0)
    print(f"   ✅ 构建完成: {causal_graph.shape[0]}×{causal_graph.shape[1]} 矩阵, {non_zero_edges} 条非零边")

    if non_zero_edges != filtered_edge_count:
        print(f"   ⚠ 警告: 邻接矩阵非零边数({non_zero_edges})与白名单边数({filtered_edge_count})不一致")
        # 这可能是由于重复边或同一对变量有多条边导致的，继续执行

    # 7. 添加ATE列到白名单DataFrame
    ate_cols = [
        'ate_whitelist_200k',
        'ate_whitelist_200k_ci_lower',
        'ate_whitelist_200k_ci_upper',
        'ate_whitelist_200k_is_significant',
        'ate_whitelist_200k_confounders_count'
    ]

    for col in ate_cols:
        if col not in filtered_whitelist.columns:
            if col == 'ate_whitelist_200k_confounders_count':
                filtered_whitelist[col] = 0  # 整数
            elif col == 'ate_whitelist_200k_is_significant':
                filtered_whitelist[col] = False  # 布尔值
            else:
                filtered_whitelist[col] = np.nan  # 浮点数

    # 8. 初始化因果推断引擎
    engine = CausalInferenceEngine(verbose=True)

    # 9. 计算每条边的ATE（全局标准化数据）
    print(f"   开始计算白名单ATE...")
    start_time = time.time()

    try:
        results = engine.analyze_all_edges_ctf_style(
            data=data_df,
            causal_graph=causal_graph,
            var_names=sorted_common_vars,
            threshold=0,  # 使用0，因为白名单边已经经过阈值过滤
            ref_df=None,  # 使用CTF风格：自动创建数据均值向量
            t_strategy='quantile'  # 使用CTF风格：25/75分位数T0/T1
        )

        elapsed_time = time.time() - start_time
        print(f"   ✅ ATE计算完成！耗时: {elapsed_time:.1f}秒")

        # 10. 更新白名单DataFrame
        print(f"   更新白名单数据...")

        # 检查results的类型
        if isinstance(results, dict):
            # results是字典的字典，键是"source->target"格式
            print(f"   results类型: 字典 (包含 {len(results)} 个键值对)")

            # 创建结果映射字典
            results_dict = {}
            for edge_key, result in results.items():
                if isinstance(result, dict) and 'ate' in result:
                    # 解析边键 "source->target"
                    if '->' in edge_key:
                        source, target = edge_key.split('->', 1)
                        key = (source, target)
                        results_dict[key] = result
                    else:
                        print(f"   ⚠ 警告: 跳过无效边键格式: {edge_key}")
                else:
                    print(f"   ⚠ 警告: 跳过无效结果: {type(result)}")

            valid_results = len(results_dict)
            print(f"   有效结果数: {valid_results}/{len(results)}")

        else:
            print(f"   ❌ 错误: results类型未知: {type(results)}")
            results_dict = {}
            valid_results = 0

        # 更新每条边的ATE信息
        updated_count = 0
        for idx, row in filtered_whitelist.iterrows():
            source = row['source']
            target = row['target']
            key = (source, target)

            if key in results_dict:
                result = results_dict[key]

                # 更新白名单ATE列
                filtered_whitelist.at[idx, 'ate_whitelist_200k'] = result.get('ate', np.nan)
                filtered_whitelist.at[idx, 'ate_whitelist_200k_ci_lower'] = result.get('ci_lower', np.nan)
                filtered_whitelist.at[idx, 'ate_whitelist_200k_ci_upper'] = result.get('ci_upper', np.nan)
                filtered_whitelist.at[idx, 'ate_whitelist_200k_is_significant'] = result.get('is_significant', False)

                # 处理混淆因素计数
                confounders = result.get('confounders', [])
                if isinstance(confounders, list):
                    filtered_whitelist.at[idx, 'ate_whitelist_200k_confounders_count'] = len(confounders)
                else:
                    filtered_whitelist.at[idx, 'ate_whitelist_200k_confounders_count'] = 0

                updated_count += 1

        print(f"   ✅ 更新完成: {updated_count}/{len(filtered_whitelist)} 条边已更新")

        # 11. 保存结果
        print(f"   保存结果...")
        os.makedirs(output_dir, exist_ok=True)
        filtered_whitelist.to_csv(output_file, index=False)
        print(f"   ✅ 结果已保存: {output_file}")

        # 12. 生成摘要统计
        ate_values = filtered_whitelist['ate_whitelist_200k'].dropna()
        significant_count = filtered_whitelist['ate_whitelist_200k_is_significant'].sum()

        summary = {
            "group_id": group_id,
            "group_num": group_num,
            "data_rows": len(data_df),
            "whitelist_edges_original": len(whitelist_df),
            "whitelist_edges_filtered": len(filtered_whitelist),
            "edges_removed": removed_edge_count,
            "common_vars": len(common_vars),
            "ate_computed": len(ate_values),
            "ate_significant": int(significant_count),
            "ate_mean": float(ate_values.mean()) if len(ate_values) > 0 else np.nan,
            "ate_std": float(ate_values.std()) if len(ate_values) > 0 else np.nan,
            "ate_min": float(ate_values.min()) if len(ate_values) > 0 else np.nan,
            "ate_max": float(ate_values.max()) if len(ate_values) > 0 else np.nan,
            "elapsed_seconds": elapsed_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存摘要
        summary_file = os.path.join(output_dir, f"{group_id}_ate_whitelist_200k_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✅ 摘要已保存: {summary_file}")

        return {
            "success": True,
            "summary": summary,
            "output_file": output_file,
            "summary_file": summary_file
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ ATE计算失败: {e}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "error": str(e),
            "elapsed_seconds": elapsed_time
        }


def main():
    parser = argparse.ArgumentParser(description="基于DiBS白名单因果图计算全局标准化数据的ATE（方案C，200k步）")
    parser.add_argument("--dry-run", action="store_true",
                       help="只测试不写入文件")
    parser.add_argument("--group", type=int, choices=range(1, 7),
                       help="只处理指定组（1-6）")
    parser.add_argument("--global-std-dir", type=str,
                       default="data/energy_research/6groups_global_std",
                       help="全局标准化数据目录")
    parser.add_argument("--whitelist-dir", type=str,
                       default="results/energy_research/data/global_std_whitelist_200k",
                       help="白名单边目录")
    parser.add_argument("--output-dir", type=str,
                       default="results/energy_research/data/global_std_dibs_ate_200k",
                       help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.0,
                       help="边强度阈值（默认: 0.0，白名单已过滤）")

    args = parser.parse_args()

    print("=" * 80)
    print("DiBS白名单因果图ATE计算（方案C，200k步）")
    print("=" * 80)

    print(f"\n配置:")
    print(f"  全局标准化数据目录: {args.global_std_dir}")
    print(f"  白名单边目录: {args.whitelist_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  边强度阈值: {args.threshold}")
    print(f"  Dry run模式: {args.dry_run}")

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
        result = compute_ate_for_group(
            group_num=group_num,
            global_std_dir=args.global_std_dir,
            whitelist_dir=args.whitelist_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            threshold=args.threshold
        )

        all_results.append({
            "group": group_num,
            **result
        })

    # 生成总报告
    print(f"\n{'='*80}")
    print("白名单ATE计算总报告")
    print(f"{'='*80}")

    successful_groups = [r for r in all_results if r.get('success', False)]
    failed_groups = [r for r in all_results if not r.get('success', False)]

    print(f"\n处理完成:")
    print(f"  成功组数: {len(successful_groups)}")
    print(f"  失败组数: {len(failed_groups)}")

    if successful_groups:
        print(f"\n成功组详情:")
        for result in successful_groups:
            if result.get('dry_run', False):
                print(f"  组 {result['group']}: Dry run完成 - {result.get('data_rows', 0)}行数据, {result.get('whitelist_edges', 0)}条边")
            elif 'summary' in result:
                summary = result['summary']
                print(f"  组 {result['group']}: {summary['ate_computed']}/{summary['whitelist_edges_filtered']}条边计算ATE")
                print(f"      ATE显著: {summary['ate_significant']}条")
                print(f"      ATE均值: {summary['ate_mean']:.4f}, 标准差: {summary['ate_std']:.4f}")

    if failed_groups:
        print(f"\n失败组详情:")
        for result in failed_groups:
            print(f"  组 {result['group']}: {result.get('error', '未知错误')}")

    # 保存总报告
    if not args.dry_run and successful_groups:
        total_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "global_std_dir": args.global_std_dir,
                "whitelist_dir": args.whitelist_dir,
                "output_dir": args.output_dir,
                "threshold": args.threshold,
                "dry_run": args.dry_run
            },
            "results": all_results,
            "summary": {
                "total_groups": len(all_results),
                "successful_groups": len(successful_groups),
                "failed_groups": len(failed_groups)
            }
        }

        total_report_file = os.path.join(args.output_dir, "ate_whitelist_200k_total_report.json")
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