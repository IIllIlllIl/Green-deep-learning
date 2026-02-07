#!/usr/bin/env python3
"""
分层DiBS因果图学习脚本

在分层数据上执行DiBS因果发现，包含敏感性分析。

数据源: analysis/data/energy_research/stratified/
输出: analysis/results/energy_research/stratified/dibs/

使用方法:
    # Dry run（检查数据）
    python scripts/stratified/run_dibs_stratified.py --dry-run

    # 运行所有分层
    python scripts/stratified/run_dibs_stratified.py

    # 运行特定分层
    python scripts/stratified/run_dibs_stratified.py --layer group1_parallel

    # 快速测试（单次运行）
    python scripts/stratified/run_dibs_stratified.py --n-seeds 1

依赖:
    - analysis/utils/causal_discovery.py (CausalGraphLearner)
    - scipy (二项式检验)
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
from scipy.stats import binomtest

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.causal_discovery import CausalGraphLearner


# ============================================================================
# 配置
# ============================================================================

# 基础配置（与全局分析一致）
DIBS_CONFIG = {
    "alpha_linear": 0.05,
    "beta_linear": 0.1,
    "n_particles": 20,
    "tau": 1.0,
    "n_steps": 5000,
    "n_grad_mc_samples": 128,
    "n_acyclicity_mc_samples": 32
}

# 小样本层适配（group3_non_parallel: 93样本）
SMALL_SAMPLE_CONFIG = {
    "alpha_linear": 0.08,
    "beta_linear": 0.15,
    "n_particles": 15,
    "tau": 1.2,
    "n_steps": 10000,
    "n_grad_mc_samples": 128,
    "n_acyclicity_mc_samples": 32
}

# 分层任务配置
STRATIFIED_TASKS = [
    {
        "id": "group1_parallel",
        "name": "group1_examples (并行)",
        "csv_file": "group1_examples/group1_parallel.csv",
        "expected_samples": 178,
        "expected_features": 26,
        "use_small_sample_config": False,
        "n_sensitivity_runs": 3
    },
    {
        "id": "group1_non_parallel",
        "name": "group1_examples (非并行)",
        "csv_file": "group1_examples/group1_non_parallel.csv",
        "expected_samples": 126,
        "expected_features": 26,
        "use_small_sample_config": False,
        "n_sensitivity_runs": 5
    },
    {
        "id": "group3_parallel",
        "name": "group3_person_reid (并行)",
        "csv_file": "group3_person_reid/group3_parallel.csv",
        "expected_samples": 113,
        "expected_features": 28,
        "use_small_sample_config": False,
        "n_sensitivity_runs": 5
    },
    {
        "id": "group3_non_parallel",
        "name": "group3_person_reid (非并行)",
        "csv_file": "group3_person_reid/group3_non_parallel.csv",
        "expected_samples": 93,
        "expected_features": 28,
        "use_small_sample_config": True,
        "n_sensitivity_runs": 7
    }
]

# 随机种子配置
RANDOM_SEEDS = [42, 123, 456, 789, 1011, 2022, 3033]


# ============================================================================
# 数据加载
# ============================================================================

def load_stratified_data(task_config: Dict, data_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """
    加载分层数据

    返回:
        X: 数据矩阵 (n_samples, n_features)
        feature_names: 特征名列表
    """
    data_file = data_dir / task_config["csv_file"]

    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    df = pd.read_csv(data_file)

    print(f"  数据规模: {len(df)}行 × {len(df.columns)}列")
    print(f"  预期规模: {task_config['expected_samples']}行 × {task_config['expected_features']}列")

    # 检查样本数
    if abs(len(df) - task_config["expected_samples"]) > 5:
        print(f"  ⚠️ 警告: 样本数差异较大")

    # 检查缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        raise ValueError(f"数据包含 {missing_count} 个缺失值，DiBS无法运行")

    print(f"  ✅ 缺失值检查: 0个缺失值")

    # 检查常量列
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"  ⚠️ 警告: 发现 {len(constant_cols)} 个常量列（DiBS会自动处理）")

    feature_names = list(df.columns)
    X = df.values.astype(float)

    return X, feature_names


# ============================================================================
# DiBS运行
# ============================================================================

def run_single_dibs(
    X: np.ndarray,
    feature_names: List[str],
    config: Dict,
    random_seed: int,
    verbose: bool = False
) -> Tuple[np.ndarray, float]:
    """
    运行单次DiBS

    返回:
        causal_graph: 因果图邻接矩阵
        runtime: 运行时间（秒）
    """
    n_vars = X.shape[1]

    learner = CausalGraphLearner(
        n_vars=n_vars,
        alpha=config["alpha_linear"],
        n_steps=config["n_steps"],
        n_particles=config["n_particles"],
        beta=config["beta_linear"],
        tau=config["tau"],
        n_grad_mc_samples=config["n_grad_mc_samples"],
        n_acyclicity_mc_samples=config["n_acyclicity_mc_samples"],
        random_seed=random_seed
    )

    start_time = time.time()

    # 转换为DataFrame
    data_df = pd.DataFrame(X, columns=feature_names)
    causal_graph = learner.fit(data=data_df, verbose=verbose)

    runtime = time.time() - start_time

    return causal_graph, runtime


def save_single_run_results(
    causal_graph: np.ndarray,
    feature_names: List[str],
    runtime: float,
    seed: int,
    output_dir: Path,
    threshold: float = 0.3
):
    """保存单次运行结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存因果图
    graph_df = pd.DataFrame(causal_graph, index=feature_names, columns=feature_names)
    graph_df.to_csv(output_dir / "causal_graph.csv")

    # 提取强边
    edges = []
    for i, src in enumerate(feature_names):
        for j, tgt in enumerate(feature_names):
            weight = causal_graph[i, j]
            if weight > threshold:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "weight": float(weight)
                })

    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(output_dir / f"edges_threshold_{threshold}.csv", index=False)

    # 保存摘要
    summary = {
        "seed": seed,
        "runtime_seconds": runtime,
        "n_features": len(feature_names),
        "graph_stats": {
            "min": float(causal_graph.min()),
            "max": float(causal_graph.max()),
            "mean": float(causal_graph.mean()),
            "std": float(causal_graph.std())
        },
        "n_edges_gt_0.1": int(np.sum(causal_graph > 0.1)),
        "n_edges_gt_0.3": int(np.sum(causal_graph > 0.3)),
        "n_edges_gt_0.5": int(np.sum(causal_graph > 0.5)),
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return len(edges)


# ============================================================================
# 敏感性分析
# ============================================================================

def compute_edge_consistency(
    graphs: List[np.ndarray],
    feature_names: List[str],
    threshold: float = 0.3,
    alpha: float = 0.05
) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    """
    计算边在多次运行中的检测一致性（使用统计显著性）

    返回:
        consistency_matrix: 每条边被检测到的次数
        stable_edges: 统计显著的稳定边列表
        stability_report: 详细稳定性报告
    """
    n_runs = len(graphs)
    n_vars = graphs[0].shape[0]

    # 统计每条边被检测到的次数
    consistency_matrix = np.zeros((n_vars, n_vars))
    for graph in graphs:
        consistency_matrix += (graph > threshold).astype(int)

    # 使用二项式检验确定稳定边
    stable_edges = []
    stability_report = []

    for i in range(n_vars):
        for j in range(n_vars):
            detections = int(consistency_matrix[i, j])
            if detections > 0:
                # 单尾二项式检验
                result = binomtest(detections, n_runs, 0.5, alternative='greater')
                p_value = result.pvalue
                is_stable = p_value < alpha

                entry = {
                    "source": feature_names[i],
                    "target": feature_names[j],
                    "source_idx": i,
                    "target_idx": j,
                    "detections": detections,
                    "n_runs": n_runs,
                    "detection_rate": detections / n_runs,
                    "p_value": float(p_value),
                    "is_stable": is_stable
                }
                stability_report.append(entry)

                if is_stable:
                    stable_edges.append(entry)

    return consistency_matrix, stable_edges, stability_report


def evaluate_convergence_via_consistency(
    consistency_matrix: np.ndarray,
    n_runs: int,
    convergence_threshold: float = 0.7
) -> Tuple[bool, float, Dict]:
    """
    通过敏感性分析的一致性评估收敛性
    """
    detected_edges = consistency_matrix > 0
    if np.sum(detected_edges) == 0:
        return False, 0.0, {"error": "no edges detected"}

    consistency_scores = consistency_matrix[detected_edges] / n_runs
    avg_consistency = float(np.mean(consistency_scores))
    high_consistency_ratio = float(np.mean(consistency_scores >= convergence_threshold))

    # 判断收敛：如果≥50%的边有≥70%的一致性
    converged = high_consistency_ratio >= 0.5

    report = {
        "avg_consistency": avg_consistency,
        "high_consistency_ratio": high_consistency_ratio,
        "n_detected_edges": int(np.sum(detected_edges)),
        "convergence_threshold": convergence_threshold,
        "converged": converged
    }

    return converged, avg_consistency, report


def run_sensitivity_analysis(
    task_config: Dict,
    data_dir: Path,
    output_dir: Path,
    n_seeds: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    运行敏感性分析

    返回:
        results: 包含所有结果的字典
    """
    task_id = task_config["id"]
    task_name = task_config["name"]

    print(f"\n{'='*70}")
    print(f"敏感性分析: {task_name} ({task_id})")
    print(f"{'='*70}")

    # 确定运行次数和配置
    if n_seeds is not None:
        n_runs = n_seeds
    else:
        n_runs = task_config["n_sensitivity_runs"]

    seeds = RANDOM_SEEDS[:n_runs]

    if task_config["use_small_sample_config"]:
        config = SMALL_SAMPLE_CONFIG
        print(f"  使用小样本配置 (n_steps={config['n_steps']})")
    else:
        config = DIBS_CONFIG
        print(f"  使用标准配置 (n_steps={config['n_steps']})")

    print(f"  运行次数: {n_runs}")
    print(f"  随机种子: {seeds}")

    # 加载数据
    print(f"\n1. 加载数据...")
    X, feature_names = load_stratified_data(task_config, data_dir)

    # 创建输出目录
    task_output_dir = output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # 运行多次DiBS
    print(f"\n2. 运行DiBS ({n_runs}次)...")
    graphs = []
    runtimes = []
    all_edges_counts = []

    for i, seed in enumerate(seeds):
        print(f"\n   运行 {i+1}/{n_runs} (seed={seed})...")
        run_output_dir = task_output_dir / f"run_seed_{seed}"

        causal_graph, runtime = run_single_dibs(
            X, feature_names, config, seed, verbose=verbose
        )
        graphs.append(causal_graph)
        runtimes.append(runtime)

        n_edges = save_single_run_results(
            causal_graph, feature_names, runtime, seed, run_output_dir
        )
        all_edges_counts.append(n_edges)

        print(f"      完成！耗时: {runtime:.1f}秒, 强边数: {n_edges}")

    # 计算平均因果图
    print(f"\n3. 计算平均因果图...")
    averaged_graph = np.mean(graphs, axis=0)

    # 保存平均因果图
    avg_graph_df = pd.DataFrame(averaged_graph, index=feature_names, columns=feature_names)
    avg_graph_df.to_csv(task_output_dir / "averaged_causal_graph.csv")

    # 敏感性分析
    print(f"\n4. 敏感性分析...")
    consistency_matrix, stable_edges, stability_report = compute_edge_consistency(
        graphs, feature_names, threshold=0.3, alpha=0.05
    )

    # 保存稳定边
    stable_edges_df = pd.DataFrame(stable_edges)
    stable_edges_df.to_csv(task_output_dir / "stable_edges_threshold_0.3.csv", index=False)
    print(f"   稳定边数量（统计显著）: {len(stable_edges)}")

    # 保存敏感性分析报告
    sensitivity_report = {
        "task_id": task_id,
        "task_name": task_name,
        "n_runs": n_runs,
        "seeds": seeds,
        "config": config,
        "n_stable_edges": len(stable_edges),
        "stability_report": stability_report,
        "timestamp": datetime.now().isoformat()
    }
    with open(task_output_dir / "sensitivity_analysis.json", "w") as f:
        json.dump(sensitivity_report, f, indent=2)

    # 收敛性评估
    print(f"\n5. 收敛性评估...")
    converged, avg_consistency, convergence_report = evaluate_convergence_via_consistency(
        consistency_matrix, n_runs
    )

    convergence_report["runtimes"] = runtimes
    convergence_report["total_runtime"] = sum(runtimes)
    convergence_report["edges_counts"] = all_edges_counts

    with open(task_output_dir / "convergence_report.json", "w") as f:
        json.dump(convergence_report, f, indent=2)

    print(f"   平均一致性: {avg_consistency:.2f}")
    print(f"   收敛状态: {'✅ 已收敛' if converged else '⚠️ 可能未收敛'}")

    # 汇总
    print(f"\n{'='*70}")
    print(f"完成: {task_name}")
    print(f"  总运行时间: {sum(runtimes):.1f}秒 ({sum(runtimes)/60:.1f}分钟)")
    print(f"  稳定边数: {len(stable_edges)}")
    print(f"  平均一致性: {avg_consistency:.2f}")
    print(f"{'='*70}")

    return {
        "task_id": task_id,
        "task_name": task_name,
        "n_runs": n_runs,
        "n_stable_edges": len(stable_edges),
        "avg_consistency": avg_consistency,
        "converged": converged,
        "total_runtime": sum(runtimes),
        "output_dir": str(task_output_dir)
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="分层DiBS因果图学习",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查数据，不运行DiBS"
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=["group1_parallel", "group1_non_parallel",
                 "group3_parallel", "group3_non_parallel"],
        help="只运行特定分层"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        help="指定运行次数（覆盖默认配置）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("分层DiBS因果图学习")
    print("=" * 70)

    # 路径设置
    script_dir = Path(__file__).parent.absolute()
    analysis_dir = script_dir.parent.parent

    data_dir = analysis_dir / "data" / "energy_research" / "stratified"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = analysis_dir / "results" / "energy_research" / "stratified" / "dibs"

    print(f"\n数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    # 确定要运行的任务
    if args.layer:
        tasks = [t for t in STRATIFIED_TASKS if t["id"] == args.layer]
    else:
        tasks = STRATIFIED_TASKS

    print(f"任务数量: {len(tasks)}")

    # Dry run模式
    if args.dry_run:
        print(f"\n{'='*70}")
        print("Dry Run模式 - 检查数据")
        print(f"{'='*70}")

        for task in tasks:
            print(f"\n检查: {task['name']} ({task['id']})")
            try:
                X, feature_names = load_stratified_data(task, data_dir)
                print(f"  ✅ 数据检查通过")
            except Exception as e:
                print(f"  ❌ 数据检查失败: {e}")

        return

    # 运行敏感性分析
    all_results = []
    total_start_time = time.time()

    for task in tasks:
        try:
            results = run_sensitivity_analysis(
                task_config=task,
                data_dir=data_dir,
                output_dir=output_dir,
                n_seeds=args.n_seeds,
                verbose=args.verbose
            )
            all_results.append(results)

        except Exception as e:
            print(f"\n❌ 任务 {task['id']} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_runtime = time.time() - total_start_time

    # 生成总报告
    if all_results:
        print(f"\n{'='*70}")
        print("总报告")
        print(f"{'='*70}")

        summary = {
            "total_tasks": len(all_results),
            "total_runtime_seconds": total_runtime,
            "total_runtime_minutes": total_runtime / 60,
            "tasks": all_results,
            "timestamp": datetime.now().isoformat()
        }

        # 保存总报告
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "stratified_dibs_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # 打印汇总
        for r in all_results:
            status = "✅" if r["converged"] else "⚠️"
            print(f"\n{r['task_name']}:")
            print(f"  稳定边数: {r['n_stable_edges']}")
            print(f"  平均一致性: {r['avg_consistency']:.2f}")
            print(f"  收敛状态: {status}")
            print(f"  运行时间: {r['total_runtime']/60:.1f}分钟")

        print(f"\n总运行时间: {total_runtime/60:.1f}分钟 ({total_runtime/3600:.2f}小时)")
        print(f"报告已保存: {output_dir / 'stratified_dibs_summary.json'}")

    print(f"\n{'='*70}")
    print("所有任务完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
