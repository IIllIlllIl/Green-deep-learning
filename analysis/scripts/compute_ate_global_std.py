#!/usr/bin/env python3
"""
全局标准化数据ATE计算脚本

基于全局标准化数据重新计算白名单因果边的ATE
使用已验收的CausalInferenceEngine（CTF风格DML方法）

数据源:
  - 全局标准化数据: data/energy_research/6groups_global_std/
  - 白名单文件: results/energy_research/data/interaction/whitelist/

输出:
  - 更新后的白名单文件（添加全局标准化ATE）
  - 保存到: results/energy_research/data/global_std_ate/

使用方法:
    # Dry run (测试模式)
    python compute_ate_global_std.py --dry-run

    # 实际运行（更新所有组）
    python compute_ate_global_std.py

    # 只处理特定group
    python compute_ate_global_std.py --group 1

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
            causal_graph[source_idx, target_idx] = strength

    return causal_graph


def compute_ate_for_group(
    group_num: int,
    global_std_dir: str,
    whitelist_dir: str,
    output_dir: str,
    dry_run: bool = False,
    threshold: float = 0.3
) -> Dict:
    """
    为单个组计算全局标准化ATE

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
    whitelist_file = os.path.join(whitelist_dir, f"{group_id}_causal_edges_whitelist.csv")
    output_file = os.path.join(output_dir, f"{group_id}_causal_edges_whitelist_global_std_ate.csv")

    print(f"   全局标准化数据: {global_std_file}")
    print(f"   白名单文件: {whitelist_file}")
    print(f"   输出文件: {output_file}")

    # 检查文件是否存在
    if not os.path.exists(global_std_file):
        print(f"❌ 全局标准化数据文件不存在: {global_std_file}")
        return {"success": False, "error": f"数据文件不存在: {global_std_file}"}

    if not os.path.exists(whitelist_file):
        print(f"❌ 白名单文件不存在: {whitelist_file}")
        return {"success": False, "error": f"白名单文件不存在: {whitelist_file}"}

    # 2. 读取数据
    print(f"   读取数据...")
    try:
        data_df = pd.read_csv(global_std_file)
        whitelist_df = pd.read_csv(whitelist_file)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return {"success": False, "error": f"读取文件失败: {e}"}

    print(f"   ✅ 读取成功: {len(data_df)} 条数据, {len(whitelist_df)} 条边")

    if dry_run:
        print(f"   🧪 Dry run模式 - 只检查数据，不计算ATE")
        return {"success": True, "dry_run": True, "data_rows": len(data_df), "edges": len(whitelist_df)}

    # 3. 数据清洗：处理NaN值
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

    # 4. 检查必要的列
    required_whitelist_cols = ['source', 'target']
    missing_cols = [col for col in required_whitelist_cols if col not in whitelist_df.columns]
    if missing_cols:
        print(f"❌ 白名单缺少必要列: {missing_cols}")
        return {"success": False, "error": f"白名单缺少列: {missing_cols}"}

    # 5. 添加全局标准化ATE列（如果不存在）
    global_std_ate_cols = [
        'ate_global_std',
        'ate_global_std_ci_lower',
        'ate_global_std_ci_upper',
        'ate_global_std_is_significant',
        'ate_global_std_confounders_count'
    ]

    for col in global_std_ate_cols:
        if col not in whitelist_df.columns:
            if col == 'ate_global_std_confounders_count':
                whitelist_df[col] = 0  # 整数
            elif col == 'ate_global_std_is_significant':
                whitelist_df[col] = False  # 布尔值
            else:
                whitelist_df[col] = np.nan  # 浮点数

    # 6. 获取数据中的所有变量名（排除非特征列）
    exclude_cols = ['timestamp', 'experiment_id', 'session_id']  # 常见非特征列
    feature_cols = [col for col in data_df.columns if col not in exclude_cols]

    # 7. 构建因果图
    print(f"   构建因果图...")
    causal_graph = build_causal_graph_from_whitelist(whitelist_df, feature_cols)

    # 8. 初始化因果推断引擎
    engine = CausalInferenceEngine(verbose=True)

    # 9. 计算每条边的ATE（全局标准化数据）
    print(f"   开始计算全局标准化ATE...")
    start_time = time.time()

    try:
        results = engine.analyze_all_edges_ctf_style(
            data=data_df,
            causal_graph=causal_graph,
            var_names=feature_cols,
            threshold=threshold,
            ref_df=None,  # 暂时不使用ref_df
            t_strategy=None  # 暂时使用默认T0/T1
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

        elif isinstance(results, list):
            # results是列表
            print(f"   results类型: 列表 (包含 {len(results)} 个元素)")

            # 创建结果映射字典
            results_dict = {}
            valid_results = 0

            for result in results:
                # 检查result是否是字典（有效结果）
                if isinstance(result, dict) and 'source_idx' in result:
                    source_idx = result['source_idx']
                    target_idx = result['target_idx']

                    # 确保索引是整数
                    if isinstance(source_idx, int) and isinstance(target_idx, int):
                        source_name = feature_cols[source_idx]
                        target_name = feature_cols[target_idx]

                        key = (source_name, target_name)
                        results_dict[key] = result
                        valid_results += 1
                    else:
                        print(f"   ⚠ 警告: 跳过无效结果 - 索引不是整数: source_idx={source_idx}, target_idx={target_idx}")
                else:
                    # 可能是字符串消息或无效结果
                    if isinstance(result, str):
                        print(f"   ⚠ 警告: 跳过字符串结果: {result[:100]}...")
                    else:
                        print(f"   ⚠ 警告: 跳过非字典结果: {type(result)}")

            print(f"   有效结果数: {valid_results}/{len(results)}")
        else:
            print(f"   ❌ 错误: results类型未知: {type(results)}")
            results_dict = {}
            valid_results = 0

        # 更新每条边的ATE信息
        updated_count = 0
        for idx, row in whitelist_df.iterrows():
            source = row['source']
            target = row['target']
            key = (source, target)

            if key in results_dict:
                result = results_dict[key]

                # 更新全局标准化ATE列
                whitelist_df.at[idx, 'ate_global_std'] = result.get('ate', np.nan)
                whitelist_df.at[idx, 'ate_global_std_ci_lower'] = result.get('ci_lower', np.nan)
                whitelist_df.at[idx, 'ate_global_std_ci_upper'] = result.get('ci_upper', np.nan)
                whitelist_df.at[idx, 'ate_global_std_is_significant'] = result.get('is_significant', False)

                # 处理混淆因素计数
                confounders = result.get('confounders', [])
                if isinstance(confounders, list):
                    whitelist_df.at[idx, 'ate_global_std_confounders_count'] = len(confounders)
                else:
                    whitelist_df.at[idx, 'ate_global_std_confounders_count'] = 0

                updated_count += 1

        print(f"   ✅ 更新完成: {updated_count}/{len(whitelist_df)} 条边已更新")

        # 11. 保存结果
        print(f"   保存结果...")
        os.makedirs(output_dir, exist_ok=True)
        whitelist_df.to_csv(output_file, index=False)
        print(f"   ✅ 结果已保存: {output_file}")

        # 12. 生成摘要统计
        ate_values = whitelist_df['ate_global_std'].dropna()
        significant_count = whitelist_df['ate_global_std_is_significant'].sum()

        summary = {
            "group_id": group_id,
            "group_num": group_num,
            "data_rows": len(data_df),
            "edges_total": len(whitelist_df),
            "edges_updated": updated_count,
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
        summary_file = os.path.join(output_dir, f"{group_id}_ate_global_std_summary.json")
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
    parser = argparse.ArgumentParser(description="计算全局标准化数据的ATE")
    parser.add_argument("--dry-run", action="store_true",
                       help="只测试不写入文件")
    parser.add_argument("--group", type=int, choices=range(1, 7),
                       help="只处理指定组（1-6）")
    parser.add_argument("--global-std-dir", type=str,
                       default="data/energy_research/6groups_global_std",
                       help="全局标准化数据目录")
    parser.add_argument("--whitelist-dir", type=str,
                       default="results/energy_research/data/interaction/whitelist",
                       help="白名单目录")
    parser.add_argument("--output-dir", type=str,
                       default="results/energy_research/data/global_std_ate",
                       help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="边强度阈值（默认: 0.3）")

    args = parser.parse_args()

    print("=" * 80)
    print("全局标准化数据ATE计算")
    print("=" * 80)

    print(f"\n配置:")
    print(f"  全局标准化数据目录: {args.global_std_dir}")
    print(f"  白名单目录: {args.whitelist_dir}")
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
    print("全局标准化ATE计算总报告")
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
                print(f"  组 {result['group']}: Dry run完成 - {result.get('data_rows', 0)}行数据, {result.get('edges', 0)}条边")
            elif 'summary' in result:
                summary = result['summary']
                print(f"  组 {result['group']}: {summary['edges_updated']}/{summary['edges_total']}条边更新")
                print(f"      ATE计算: {summary['ate_computed']}条, 显著: {summary['ate_significant']}条")
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

        total_report_file = os.path.join(args.output_dir, "ate_global_std_total_report.json")
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