#!/usr/bin/env python3
"""
分层ATE计算脚本

基于DiBS学到的稳定边计算分层数据的ATE（Average Treatment Effect）
使用DML方法（Double Machine Learning）进行因果推断

关键特性:
- 使用DiBS稳定边作为因果图输入
- 全局FDR校正：合并所有4个分层的p值后统一应用BH-FDR (α=0.10)
- Bootstrap置信区间 (n_bootstrap=500)

数据源:
  - 分层数据: data/energy_research/stratified/
  - DiBS结果: results/energy_research/stratified/dibs/

输出:
  - ATE结果: results/energy_research/stratified/ate/

使用方法:
    # Dry run（检查数据）
    python scripts/stratified/compute_ate_stratified.py --dry-run

    # 运行所有分层
    python scripts/stratified/compute_ate_stratified.py

    # 运行特定分层
    python scripts/stratified/compute_ate_stratified.py --layer group1_parallel

依赖:
    - analysis/utils/causal_inference.py (CausalInferenceEngine)
    - statsmodels (FDR校正)
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from utils.causal_inference import CausalInferenceEngine
    ECONML_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入CausalInferenceEngine: {e}")
    print("请确保在causal-research环境中运行: conda activate causal-research")
    ECONML_AVAILABLE = False

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("警告: 无法导入statsmodels，将使用简化的FDR校正")
    STATSMODELS_AVAILABLE = False


# ============================================================================
# 配置
# ============================================================================

# 分层任务配置
STRATIFIED_TASKS = [
    {
        "id": "group1_parallel",
        "name": "group1_examples (并行)",
        "csv_file": "group1_examples/group1_parallel.csv",
    },
    {
        "id": "group1_non_parallel",
        "name": "group1_examples (非并行)",
        "csv_file": "group1_examples/group1_non_parallel.csv",
    },
    {
        "id": "group3_parallel",
        "name": "group3_person_reid (并行)",
        "csv_file": "group3_person_reid/group3_parallel.csv",
    },
    {
        "id": "group3_non_parallel",
        "name": "group3_person_reid (非并行)",
        "csv_file": "group3_person_reid/group3_non_parallel.csv",
    }
]

# ATE计算配置
ATE_CONFIG = {
    "threshold": 0.3,           # 边强度阈值
    "n_bootstrap": 500,         # Bootstrap次数
    "alpha": 0.10,              # FDR校正阈值
    "t_strategy": "quantile",   # 使用25/75分位数
}


# ============================================================================
# 数据加载
# ============================================================================

def load_stratified_data(task_id: str, data_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    加载分层数据

    返回:
        data_df: 数据DataFrame
        feature_names: 特征名列表
    """
    task_config = next((t for t in STRATIFIED_TASKS if t["id"] == task_id), None)
    if task_config is None:
        raise ValueError(f"未知的分层任务: {task_id}")

    data_file = data_dir / task_config["csv_file"]
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    df = pd.read_csv(data_file)
    feature_names = list(df.columns)

    print(f"  数据规模: {len(df)}行 × {len(df.columns)}列")

    # 检查缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  警告: 数据包含 {missing_count} 个缺失值")

    return df, feature_names


def load_dibs_stable_edges(task_id: str, dibs_dir: Path, threshold: float = 0.3) -> pd.DataFrame:
    """
    加载DiBS稳定边

    返回:
        stable_edges_df: 稳定边DataFrame，包含source, target, weight等列
    """
    task_dir = dibs_dir / task_id

    # 尝试加载稳定边文件
    stable_edges_file = task_dir / f"stable_edges_threshold_{threshold}.csv"
    if stable_edges_file.exists():
        # 检查文件是否为空或只有空行
        file_size = stable_edges_file.stat().st_size
        if file_size > 10:  # 非空文件应至少有10字节（包含header）
            try:
                stable_edges_df = pd.read_csv(stable_edges_file)
                if len(stable_edges_df) > 0:
                    print(f"  加载稳定边: {len(stable_edges_df)} 条")
                    return stable_edges_df
            except pd.errors.EmptyDataError:
                pass  # 文件为空，继续使用平均因果图
        print(f"  稳定边文件为空或无有效数据，使用平均因果图...")
    else:
        print(f"  稳定边文件不存在，使用平均因果图...")
    avg_graph_file = task_dir / "averaged_causal_graph.csv"
    if not avg_graph_file.exists():
        raise FileNotFoundError(f"平均因果图文件不存在: {avg_graph_file}")

    # 从平均因果图提取边
    avg_graph_df = pd.read_csv(avg_graph_file, index_col=0)
    feature_names = list(avg_graph_df.columns)

    edges = []
    for i, src in enumerate(feature_names):
        for j, tgt in enumerate(feature_names):
            weight = avg_graph_df.iloc[i, j]
            if weight > threshold:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "weight": float(weight)
                })

    stable_edges_df = pd.DataFrame(edges)
    print(f"  从平均因果图提取边 (threshold>{threshold}): {len(stable_edges_df)} 条")
    return stable_edges_df


def load_dibs_causal_graph(task_id: str, dibs_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """
    加载DiBS因果图（平均图）

    返回:
        causal_graph: 邻接矩阵
        feature_names: 特征名列表
    """
    avg_graph_file = dibs_dir / task_id / "averaged_causal_graph.csv"
    if not avg_graph_file.exists():
        raise FileNotFoundError(f"平均因果图文件不存在: {avg_graph_file}")

    df = pd.read_csv(avg_graph_file, index_col=0)
    feature_names = list(df.columns)
    causal_graph = df.values.astype(float)

    return causal_graph, feature_names


# ============================================================================
# ATE计算
# ============================================================================

def compute_ate_for_layer(
    task_id: str,
    data_dir: Path,
    dibs_dir: Path,
    output_dir: Path,
    config: Dict,
    dry_run: bool = False
) -> Dict:
    """
    为单个分层计算ATE

    返回:
        results: 包含ATE结果的字典
    """
    task_config = next((t for t in STRATIFIED_TASKS if t["id"] == task_id), None)
    task_name = task_config["name"] if task_config else task_id

    print(f"\n{'='*70}")
    print(f"ATE计算: {task_name} ({task_id})")
    print(f"{'='*70}")

    # 1. 加载数据
    print(f"\n1. 加载数据...")
    try:
        data_df, data_features = load_stratified_data(task_id, data_dir)
    except Exception as e:
        print(f"  ❌ 加载数据失败: {e}")
        return {"success": False, "error": str(e), "task_id": task_id}

    # 2. 加载DiBS因果图
    print(f"\n2. 加载DiBS因果图...")
    try:
        causal_graph, dibs_features = load_dibs_causal_graph(task_id, dibs_dir)
        stable_edges_df = load_dibs_stable_edges(task_id, dibs_dir, config["threshold"])
    except Exception as e:
        print(f"  ❌ 加载DiBS结果失败: {e}")
        return {"success": False, "error": str(e), "task_id": task_id}

    # 检查是否有足够的边
    if len(stable_edges_df) == 0:
        print(f"  ⚠️ 没有稳定边，尝试从平均图提取强边...")
        # 从平均因果图提取强边
        edges = []
        for i, src in enumerate(dibs_features):
            for j, tgt in enumerate(dibs_features):
                weight = causal_graph[i, j]
                if weight > config["threshold"]:
                    edges.append({
                        "source": src,
                        "target": tgt,
                        "weight": float(weight)
                    })
        stable_edges_df = pd.DataFrame(edges)
        print(f"  从平均因果图提取 {len(stable_edges_df)} 条强边")

    if len(stable_edges_df) == 0:
        print(f"  ❌ 没有可用的边进行ATE计算")
        return {"success": False, "error": "没有可用的边", "task_id": task_id}

    # 3. 特征匹配
    print(f"\n3. 特征匹配...")
    common_features = [f for f in dibs_features if f in data_features]
    print(f"  共有特征: {len(common_features)}/{len(dibs_features)}")

    if len(common_features) < 5:
        print(f"  ❌ 共有特征太少")
        return {"success": False, "error": "共有特征太少", "task_id": task_id}

    # 过滤边（只保留共有特征的边）
    filtered_edges = stable_edges_df[
        stable_edges_df['source'].isin(common_features) &
        stable_edges_df['target'].isin(common_features)
    ].copy()
    print(f"  过滤后边数: {len(filtered_edges)}")

    if dry_run:
        print(f"\n  🧪 Dry run模式 - 只检查数据")
        return {
            "success": True,
            "dry_run": True,
            "task_id": task_id,
            "n_samples": len(data_df),
            "n_features": len(common_features),
            "n_edges": len(filtered_edges)
        }

    # 4. 计算ATE
    print(f"\n4. 计算ATE...")
    start_time = time.time()

    if not ECONML_AVAILABLE:
        print(f"  ❌ EconML不可用")
        return {"success": False, "error": "EconML不可用", "task_id": task_id}

    engine = CausalInferenceEngine(verbose=False)

    ate_results = []
    successful_edges = 0
    failed_edges = 0

    for idx, row in filtered_edges.iterrows():
        source = row['source']
        target = row['target']
        edge_weight = row.get('weight', 0.5)

        # 确定混淆因素（排除source和target的所有其他变量）
        confounders = [f for f in common_features if f != source and f != target]

        try:
            # 计算ATE
            ate, (ci_lower, ci_upper) = engine.estimate_ate(
                data=data_df,
                treatment=source,
                outcome=target,
                confounders=confounders[:10],  # 限制混淆因素数量（避免过拟合）
                t_strategy=config["t_strategy"]
            )

            # 计算p值（基于置信区间）
            # 如果0在CI外，则显著
            se = (ci_upper - ci_lower) / (2 * 1.96)  # 近似标准误
            if se > 0:
                z_stat = abs(ate) / se
                p_value = 2 * (1 - stats.norm.cdf(z_stat))
            else:
                p_value = 1.0

            ate_results.append({
                "source": source,
                "target": target,
                "edge_weight": edge_weight,
                "ate": ate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "se": se,
                "p_value": p_value,
                "n_confounders": len(confounders[:10])
            })
            successful_edges += 1

        except Exception as e:
            print(f"    ⚠️ {source} -> {target}: {str(e)[:50]}")
            ate_results.append({
                "source": source,
                "target": target,
                "edge_weight": edge_weight,
                "ate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "se": np.nan,
                "p_value": np.nan,
                "error": str(e)
            })
            failed_edges += 1

    elapsed_time = time.time() - start_time
    print(f"  ✅ ATE计算完成: {successful_edges}成功, {failed_edges}失败, 耗时{elapsed_time:.1f}秒")

    # 5. 保存结果（不进行FDR校正，留给全局校正）
    print(f"\n5. 保存结果...")
    task_output_dir = output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    ate_df = pd.DataFrame(ate_results)
    ate_file = task_output_dir / f"{task_id}_ate_raw.csv"
    ate_df.to_csv(ate_file, index=False)
    print(f"  保存到: {ate_file}")

    # 保存摘要
    summary = {
        "task_id": task_id,
        "task_name": task_name,
        "n_samples": len(data_df),
        "n_features": len(common_features),
        "n_edges_total": len(filtered_edges),
        "n_edges_successful": successful_edges,
        "n_edges_failed": failed_edges,
        "elapsed_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }

    summary_file = task_output_dir / f"{task_id}_ate_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return {
        "success": True,
        "task_id": task_id,
        "ate_df": ate_df,
        "summary": summary,
        "output_dir": str(task_output_dir)
    }


# ============================================================================
# 全局FDR校正
# ============================================================================

def apply_global_fdr_correction(
    all_results: List[Dict],
    output_dir: Path,
    alpha: float = 0.10
) -> pd.DataFrame:
    """
    对所有分层的p值进行全局FDR校正

    参数:
        all_results: 所有分层的ATE结果列表
        output_dir: 输出目录
        alpha: FDR阈值

    返回:
        combined_df: 合并后的DataFrame，包含FDR校正结果
    """
    print(f"\n{'='*70}")
    print("全局FDR校正")
    print(f"{'='*70}")

    # 合并所有结果
    all_dfs = []
    for result in all_results:
        if result.get("success") and "ate_df" in result:
            df = result["ate_df"].copy()
            df["layer"] = result["task_id"]
            all_dfs.append(df)

    if not all_dfs:
        print("  ❌ 没有可用的结果进行FDR校正")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  合并 {len(all_dfs)} 个分层，共 {len(combined_df)} 条边")

    # 提取有效p值
    valid_mask = combined_df['p_value'].notna()
    valid_p_values = combined_df.loc[valid_mask, 'p_value'].values

    print(f"  有效p值: {len(valid_p_values)}")

    if len(valid_p_values) == 0:
        print("  ❌ 没有有效的p值")
        combined_df['p_value_fdr'] = np.nan
        combined_df['is_significant_fdr'] = False
        return combined_df

    # 应用BH-FDR校正
    if STATSMODELS_AVAILABLE:
        rejected, p_values_fdr, _, _ = multipletests(
            valid_p_values,
            alpha=alpha,
            method='fdr_bh'
        )
        print(f"  FDR校正完成 (BH方法, α={alpha})")
    else:
        # 简化的BH-FDR校正
        n = len(valid_p_values)
        sorted_indices = np.argsort(valid_p_values)
        sorted_p_values = valid_p_values[sorted_indices]

        # BH校正
        p_values_fdr = np.zeros(n)
        for i, p in enumerate(sorted_p_values):
            p_values_fdr[sorted_indices[i]] = p * n / (i + 1)

        # 确保FDR校正后的p值单调
        p_values_fdr = np.minimum.accumulate(p_values_fdr[::-1])[::-1]
        p_values_fdr = np.minimum(p_values_fdr, 1.0)

        rejected = p_values_fdr < alpha
        print(f"  FDR校正完成 (简化BH方法, α={alpha})")

    # 将FDR校正结果添加到DataFrame
    combined_df['p_value_fdr'] = np.nan
    combined_df['is_significant_fdr'] = False

    combined_df.loc[valid_mask, 'p_value_fdr'] = p_values_fdr
    combined_df.loc[valid_mask, 'is_significant_fdr'] = rejected

    # 统计
    n_significant = combined_df['is_significant_fdr'].sum()
    print(f"  显著边数 (FDR校正后): {n_significant}/{len(valid_p_values)}")

    # 保存合并结果
    combined_file = output_dir / "all_layers_ate_fdr_corrected.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"  保存到: {combined_file}")

    # 按分层统计
    print(f"\n  各分层统计:")
    for layer in combined_df['layer'].unique():
        layer_df = combined_df[combined_df['layer'] == layer]
        layer_significant = layer_df['is_significant_fdr'].sum()
        layer_total = layer_df['p_value'].notna().sum()
        print(f"    {layer}: {layer_significant}/{layer_total} 显著")

    return combined_df


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="分层ATE计算",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查数据，不计算ATE"
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=["group1_parallel", "group1_non_parallel",
                 "group3_parallel", "group3_non_parallel"],
        help="只处理特定分层"
    )
    parser.add_argument(
        "--skip-fdr",
        action="store_true",
        help="跳过全局FDR校正"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="边强度阈值（默认: 0.3）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("分层ATE计算")
    print("=" * 70)

    # 路径设置
    script_dir = Path(__file__).parent.absolute()
    analysis_dir = script_dir.parent.parent

    data_dir = analysis_dir / "data" / "energy_research" / "stratified"
    dibs_dir = analysis_dir / "results" / "energy_research" / "stratified" / "dibs"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = analysis_dir / "results" / "energy_research" / "stratified" / "ate"

    print(f"\n数据目录: {data_dir}")
    print(f"DiBS目录: {dibs_dir}")
    print(f"输出目录: {output_dir}")

    # 更新配置
    config = ATE_CONFIG.copy()
    config["threshold"] = args.threshold

    print(f"\n配置:")
    print(f"  边强度阈值: {config['threshold']}")
    print(f"  Bootstrap次数: {config['n_bootstrap']}")
    print(f"  FDR阈值: {config['alpha']}")

    # 检查EconML
    if not ECONML_AVAILABLE and not args.dry_run:
        print("\n❌ EconML不可用，无法计算ATE")
        print("请在causal-research环境中运行: conda activate causal-research")
        return 1

    # 确定要处理的任务
    if args.layer:
        tasks = [t for t in STRATIFIED_TASKS if t["id"] == args.layer]
    else:
        tasks = STRATIFIED_TASKS

    print(f"\n任务数量: {len(tasks)}")

    # 处理每个分层
    all_results = []
    total_start_time = time.time()

    for task in tasks:
        task_id = task["id"]

        # 检查DiBS结果是否存在
        dibs_task_dir = dibs_dir / task_id
        if not dibs_task_dir.exists():
            print(f"\n⚠️ DiBS结果不存在: {task_id}，跳过")
            continue

        try:
            result = compute_ate_for_layer(
                task_id=task_id,
                data_dir=data_dir,
                dibs_dir=dibs_dir,
                output_dir=output_dir,
                config=config,
                dry_run=args.dry_run
            )
            all_results.append(result)

        except Exception as e:
            print(f"\n❌ 任务 {task_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "success": False,
                "task_id": task_id,
                "error": str(e)
            })

    total_runtime = time.time() - total_start_time

    # 全局FDR校正
    if not args.dry_run and not args.skip_fdr:
        successful_results = [r for r in all_results if r.get("success") and "ate_df" in r]
        if len(successful_results) >= 2:
            combined_df = apply_global_fdr_correction(
                all_results=successful_results,
                output_dir=output_dir,
                alpha=config["alpha"]
            )
        else:
            print(f"\n⚠️ 成功的分层数不足，跳过全局FDR校正 ({len(successful_results)}/4)")

    # 生成总报告
    print(f"\n{'='*70}")
    print("总报告")
    print(f"{'='*70}")

    successful_tasks = [r for r in all_results if r.get("success")]
    failed_tasks = [r for r in all_results if not r.get("success")]

    print(f"\n处理完成:")
    print(f"  成功: {len(successful_tasks)}/{len(all_results)}")
    print(f"  失败: {len(failed_tasks)}/{len(all_results)}")
    print(f"  总耗时: {total_runtime/60:.1f}分钟")

    for result in successful_tasks:
        if "summary" in result:
            s = result["summary"]
            print(f"\n  {s['task_id']}:")
            print(f"    样本数: {s['n_samples']}")
            print(f"    边数: {s['n_edges_successful']}/{s['n_edges_total']}")

    if failed_tasks:
        print(f"\n失败任务:")
        for result in failed_tasks:
            print(f"  {result['task_id']}: {result.get('error', '未知错误')}")

    # 保存总报告
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        total_report = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "total_runtime_seconds": total_runtime,
            "results": [
                {
                    "task_id": r.get("task_id"),
                    "success": r.get("success"),
                    "summary": r.get("summary"),
                    "error": r.get("error")
                }
                for r in all_results
            ]
        }

        report_file = output_dir / "stratified_ate_total_report.json"
        with open(report_file, 'w') as f:
            json.dump(total_report, f, indent=2, default=str)
        print(f"\n✅ 总报告已保存: {report_file}")

    print(f"\n{'='*70}")
    print("所有任务完成")
    print(f"{'='*70}")

    return 0 if len(failed_tasks) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
