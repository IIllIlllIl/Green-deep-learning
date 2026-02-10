#!/usr/bin/env python3
"""
6分组全局标准化数据DiBS因果分析脚本

目的: 在全局标准化数据上执行DiBS因果发现
数据源: analysis/data/energy_research/6groups_dibs_ready/
研究问题:
  - 问题1: 超参数对能耗的影响（跨组可比版本）
  - 问题2: 能耗和性能之间的权衡关系
  - 问题3: 中间变量的中介效应

关键改进:
1. 使用全局标准化数据（所有组相同尺度）
2. hyperparam_seed用-1填充（区分有/无seed）
3. 性能指标结构性缺失用-999标记
4. 跨组可比因果发现

创建日期: 2026-01-30
基于: run_dibs_6groups_interaction.py
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.causal_discovery import CausalGraphLearner

# ========== 最优配置（v2.0 - 基于CTF论文修改） ==========
# CTF源码参考: analysis/CTF_original/src/discovery.py
# L142: 使用MarginalDiBS变体
# L148: n_particles=50, steps=13000
# 默认值: alpha_linear=1.0, beta_linear=1.0
OPTIMAL_CONFIG = {
    "variant": "MarginalDiBS",    # v2.0新增：从JointDiBS切换到MarginalDiBS（匹配CTF论文）
    "alpha_linear": 0.5,          # v3.0修改：从1.0改为0.5（降低DAG惩罚，产生更多边）
    "beta_linear": 1.0,           # v2.0修改：从0.1改为1.0（CTF使用默认值）
    "n_particles": 100,           # v3.0修改：从50改为100（增加粒子数，更广探索）
    "tau": 1.0,                   # Gumbel-softmax温度
    "n_steps": 60000,             # v3.0修改：从13000→20000→60000（预计4.5h，6h预算内）
    "n_grad_mc_samples": 128,     # MC梯度样本数
    "n_acyclicity_mc_samples": 32  # 无环性MC样本数
}

# ========== 6个任务组配置（v2.0 - 数据清理后预期特征数）==========
# v2.0更新：
# 1. 删除全0列（不适用于该组的超参数和模型）
# 2. Group4超参数语义合并：max_iter→epochs, alpha→l2_regularization
# 预期特征数已更新为清理后的值
TASK_GROUPS = [
    {
        "id": "group1_examples",
        "name": "examples（图像分类-小型）",
        "csv_file": "group1_examples_dibs_ready.csv",
        "expected_samples": 304,
        "expected_features": 23,  # v2.0更新：删除全0列后约23列
        "description": "小型图像分类模型，包含MNIST相关模型"
    },
    {
        "id": "group2_vulberta",
        "name": "VulBERTa（代码漏洞检测）",
        "csv_file": "group2_vulberta_dibs_ready.csv",
        "expected_samples": 72,
        "expected_features": 24,  # v2.0更新：删除全0列后约24列
        "description": "代码漏洞检测模型，BERT架构"
    },
    {
        "id": "group3_person_reid",
        "name": "Person_reID（行人重识别）",
        "csv_file": "group3_person_reid_dibs_ready.csv",
        "expected_samples": 206,
        "expected_features": 25,  # v2.0更新：删除全0列后约25列
        "description": "行人重识别模型，关注识别准确性"
    },
    {
        "id": "group4_bug_localization",
        "name": "bug-localization（缺陷定位）",
        "csv_file": "group4_bug_localization_dibs_ready.csv",
        "expected_samples": 90,
        "expected_features": 24,  # v2.0更新：超参数合并后约24列
        "description": "代码缺陷定位模型，多标签分类"
    },
    {
        "id": "group5_mrt_oast",
        "name": "MRT-OAST（缺陷定位）",
        "csv_file": "group5_mrt_oast_dibs_ready.csv",
        "expected_samples": 60,  # 删除了12个缺失perf_accuracy的行
        "expected_features": 26,  # v2.0更新：删除全0列后约26列
        "description": "另一缺陷定位模型，不同架构"
    },
    {
        "id": "group6_resnet",
        "name": "pytorch_resnet（图像分类-ResNet）",
        "csv_file": "group6_resnet_dibs_ready.csv",
        "expected_samples": 74,
        "expected_features": 24,  # v2.0更新：删除全0列后约24列
        "description": "标准ResNet图像分类模型"
    }
]


def load_task_group_data(task_config):
    """
    加载单个任务组的DiBS就绪数据

    参数:
        task_config: 任务组配置字典

    返回:
        data: 处理后的DataFrame
        feature_names: 特征名称列表
    """
    # 使用DiBS就绪数据路径
    data_dir = Path(__file__).parent.parent / "data" / "energy_research" / "6groups_dibs_ready"
    data_file = data_dir / task_config["csv_file"]

    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    # 加载数据
    df = pd.read_csv(data_file)

    print(f"  数据规模: {len(df)}行 × {len(df.columns)}列")
    print(f"  预期规模: {task_config['expected_samples']}行 × {task_config['expected_features']}列")
    print(f"  描述: {task_config['description']}")

    if len(df) != task_config["expected_samples"]:
        print(f"  ⚠️ 警告: 样本数不符合预期（预期{task_config['expected_samples']}，实际{len(df)}）")

    if len(df.columns) != task_config["expected_features"]:
        print(f"  ⚠️ 警告: 特征数不符合预期（预期{task_config['expected_features']}，实际{len(df.columns)}）")

    # 检查缺失值 - DiBS要求零缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  ❌ 错误: 发现 {missing_count} 个缺失值！DiBS无法运行。")
        print(f"  请重新运行预处理脚本: python scripts/preprocess_for_dibs_global_std.py")
        raise ValueError(f"数据包含 {missing_count} 个缺失值")

    print(f"  ✅ 缺失值检查: 0个缺失值")

    # 检查数据类型
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  ⚠️ 警告: 列 '{col}' 不是数值类型 ({df[col].dtype})")
            # 尝试转换为数值
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"    已转换为数值类型")
            except:
                print(f"    ❌ 无法转换为数值类型，可能影响DiBS运行")

    feature_names = list(df.columns)

    return df.values, feature_names


def run_dibs_for_group(task_config, output_dir, config=OPTIMAL_CONFIG, verbose=True):
    """
    为单个组运行DiBS

    返回:
        results: DiBS结果字典
    """
    group_id = task_config["id"]
    group_name = task_config["name"]

    print(f"\n{'='*80}")
    print(f"运行DiBS: {group_name} ({group_id})")
    print(f"{'='*80}")

    # 1. 加载数据
    print(f"\n1. 加载数据...")
    X, feature_names = load_task_group_data(task_config)

    # 2. 创建输出目录
    group_output_dir = output_dir / group_id
    group_output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 初始化DiBS学习器
    print(f"\n2. 初始化DiBS学习器...")
    print(f"   配置参数:")
    for key, value in config.items():
        print(f"     {key}: {value}")

    learner = CausalGraphLearner(
        n_vars=len(feature_names),
        alpha=config["alpha_linear"],
        n_steps=config["n_steps"],  # v2.0修复：传递n_steps参数（匹配CTF论文的13000步）
        n_particles=config["n_particles"],
        beta=config["beta_linear"],
        tau=config["tau"],
        n_grad_mc_samples=config["n_grad_mc_samples"],
        n_acyclicity_mc_samples=config["n_acyclicity_mc_samples"],
        variant=config.get("variant", "JointDiBS")  # v2.0新增：支持MarginalDiBS变体
    )

    # 4. 运行DiBS
    print(f"\n3. 运行DiBS ({config['n_steps']}步)...")
    start_time = time.time()

    try:
        # 将numpy数组转换为DataFrame
        data_df = pd.DataFrame(X, columns=feature_names)
        causal_graph = learner.fit(
            data=data_df,
            verbose=verbose
        )
    except Exception as e:
        print(f"\n❌ DiBS运行失败: {e}")
        raise

    elapsed_time = time.time() - start_time
    print(f"   DiBS运行完成！耗时: {elapsed_time:.1f}秒")

    # 5. 分析因果图
    print(f"\n4. 分析因果图...")

    # 因果图统计
    graph_min = float(causal_graph.min())
    graph_max = float(causal_graph.max())
    graph_mean = float(causal_graph.mean())
    graph_std = float(causal_graph.std())

    # 不同阈值下的边数
    edges_001 = int(np.sum(causal_graph > 0.01))
    edges_01 = int(np.sum(causal_graph > 0.1))
    edges_03 = int(np.sum(causal_graph > 0.3))
    edges_05 = int(np.sum(causal_graph > 0.5))

    print(f"   最小值: {graph_min:.6f}")
    print(f"   最大值: {graph_max:.6f}")
    print(f"   平均值: {graph_mean:.6f}")
    print(f"   标准差: {graph_std:.6f}")
    print(f"\n   边数统计:")
    print(f"     >0.01: {edges_001}条")
    print(f"     >0.1:  {edges_01}条")
    print(f"     >0.3:  {edges_03}条 ⭐ 强边")
    print(f"     >0.5:  {edges_05}条")

    # 6. 保存结果
    print(f"\n5. 保存结果...")

    # 保存因果图（邻接矩阵）
    causal_graph_df = pd.DataFrame(causal_graph, index=feature_names, columns=feature_names)
    causal_graph_file = group_output_dir / f"{group_id}_dibs_causal_graph.csv"
    causal_graph_df.to_csv(causal_graph_file)
    print(f"   因果图（邻接矩阵）: {causal_graph_file}")

    # 保存边列表（阈值>0.3）
    edges = []
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            weight = causal_graph[i, j]
            if weight > 0.3:  # 强边阈值
                edges.append({
                    "source": feature_names[i],
                    "target": feature_names[j],
                    "weight": float(weight)
                })

    edges_df = pd.DataFrame(edges)
    edges_file = group_output_dir / f"{group_id}_dibs_edges_threshold_0.3.csv"
    edges_df.to_csv(edges_file, index=False)
    print(f"   强边列表（阈值>0.3）: {edges_file} ({len(edges)}条边)")

    # 保存特征名称
    features_file = group_output_dir / f"{group_id}_feature_names.json"
    with open(features_file, 'w') as f:
        json.dump(feature_names, f)
    print(f"   特征名称: {features_file}")

    # 保存配置
    config_file = group_output_dir / f"{group_id}_dibs_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "group_config": task_config,
            "dibs_config": config,
            "runtime_seconds": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    print(f"   配置: {config_file}")

    # 7. 生成摘要
    print(f"\n6. 生成摘要...")

    # 计算强边数（权重 > 0.3）
    strong_edges = edges_03
    total_possible_edges = len(feature_names) * (len(feature_names) - 1)

    summary = {
        "group_id": group_id,
        "group_name": group_name,
        "samples": X.shape[0],
        "features": X.shape[1],
        "graph_min": graph_min,
        "graph_max": graph_max,
        "graph_mean": graph_mean,
        "graph_std": graph_std,
        "edges_gt_0.01": edges_001,
        "edges_gt_0.1": edges_01,
        "edges_gt_0.3": edges_03,
        "edges_gt_0.5": edges_05,
        "strong_edges_weight_gt_0.3": int(strong_edges),
        "total_possible_edges": int(total_possible_edges),
        "strong_edge_percentage": float(strong_edges / total_possible_edges * 100),
        "runtime_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }

    summary_file = group_output_dir / f"{group_id}_dibs_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   摘要: {summary_file}")

    print(f"\n{'='*80}")
    print(f"DiBS完成: {group_name}")
    print(f"强边数（>0.3）: {strong_edges}/{total_possible_edges} ({strong_edges/total_possible_edges*100:.1f}%)")
    print(f"{'='*80}")

    return {
        "summary": summary,
        "causal_graph": causal_graph,
        "edges": edges,
        "feature_names": feature_names,
        "files": {
            "causal_graph": causal_graph_file,
            "edges": edges_file,
            "features": features_file,
            "config": config_file,
            "summary": summary_file
        }
    }


def main():
    parser = argparse.ArgumentParser(description="运行DiBS因果发现（全局标准化数据）")
    parser.add_argument("--group", type=str,
                       help="运行特定组（如: group1_examples），不指定则运行所有组")
    parser.add_argument("--output-dir", type=str,
                       default="results/energy_research/data/global_std",
                       help="输出目录")
    parser.add_argument("--n-steps", type=int, default=OPTIMAL_CONFIG["n_steps"],
                       help="DiBS步数")
    parser.add_argument("--n-particles", type=int, default=OPTIMAL_CONFIG["n_particles"],
                       help="粒子数")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    parser.add_argument("--dry-run", action="store_true",
                       help="只检查不运行")

    args = parser.parse_args()

    print("=" * 80)
    print("DiBS因果发现（全局标准化数据）")
    print("=" * 80)

    # 配置
    config = OPTIMAL_CONFIG.copy()
    config["n_steps"] = args.n_steps
    config["n_particles"] = args.n_particles

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定要运行的组
    if args.group:
        # 运行指定组
        task_groups = [tg for tg in TASK_GROUPS if tg["id"] == args.group]
        if not task_groups:
            print(f"❌ 未找到组: {args.group}")
            return
    else:
        # 运行所有组
        task_groups = TASK_GROUPS

    print(f"\n运行配置:")
    print(f"  输出目录: {output_dir}")
    print(f"  运行组数: {len(task_groups)}")
    print(f"  DiBS步数: {config['n_steps']}")
    print(f"  粒子数: {config['n_particles']}")
    print(f"  详细模式: {args.verbose}")
    print(f"  试运行: {args.dry_run}")

    if args.dry_run:
        print(f"\n试运行模式 - 只检查数据:")
        for task_config in task_groups:
            print(f"\n检查: {task_config['name']} ({task_config['id']})")
            try:
                X, feature_names = load_task_group_data(task_config)
                print(f"  ✅ 数据检查通过: {X.shape[0]}样本, {X.shape[1]}特征")
            except Exception as e:
                print(f"  ❌ 数据检查失败: {e}")
        return

    # 运行DiBS
    all_results = []

    for task_config in task_groups:
        try:
            results = run_dibs_for_group(
                task_config=task_config,
                output_dir=output_dir,
                config=config,
                verbose=args.verbose
            )
            all_results.append(results)

        except Exception as e:
            print(f"\n❌ 组 {task_config['id']} 运行失败: {e}")
            print(f"   跳过该组，继续运行其他组...")
            continue

    # 生成总报告
    if all_results:
        print(f"\n{'='*80}")
        print("DiBS运行总报告")
        print(f"{'='*80}")

        total_summary = {
            "total_groups": len(all_results),
            "successful_groups": len(all_results),
            "groups": [],
            "timestamp": datetime.now().isoformat()
        }

        for results in all_results:
            summary = results["summary"]
            total_summary["groups"].append(summary)

            print(f"\n{summary['group_name']}:")
            print(f"  样本数: {summary['samples']}")
            print(f"  特征数: {summary['features']}")
            print(f"  强边数: {summary['edges_gt_0.3']}")
            print(f"  强边比例: {summary['strong_edge_percentage']:.1f}%")
            print(f"  运行时间: {summary['runtime_seconds']:.1f}秒")

        # 保存总报告
        total_report_file = output_dir / "dibs_global_std_total_report.json"
        with open(total_report_file, 'w') as f:
            json.dump(total_summary, f, indent=2)

        print(f"\n✅ DiBS运行完成！")
        print(f"   总报告: {total_report_file}")

    else:
        print(f"\n❌ 没有成功运行的组")

    print(f"\n{'='*80}")
    print("所有任务完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
