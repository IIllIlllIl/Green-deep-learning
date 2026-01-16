#!/usr/bin/env python3
"""
新6分组数据的DiBS因果分析脚本

目的: 在新生成的6组数据（基于data.csv）上执行DiBS因果发现
数据源: analysis/data/energy_research/dibs_training/ (2026-01-15生成)
研究问题:
  - 问题1: 超参数对能耗的影响
  - 问题2: 能耗和性能之间的权衡关系
  - 问题3: 中间变量的中介效应

创建日期: 2026-01-15
基于: run_dibs_for_questions_2_3.py (成功配置: alpha=0.05, beta=0.1, particles=20)
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.causal_discovery import CausalGraphLearner

# ========== 最优配置（基于2026-01-05参数调优结果） ==========
OPTIMAL_CONFIG = {
    "alpha_linear": 0.05,        # DiBS默认值，效果良好
    "beta_linear": 0.1,          # 低无环约束，允许更多边探索 ⭐
    "n_particles": 20,           # 最佳性价比
    "tau": 1.0,                  # Gumbel-softmax温度
    "n_steps": 5000,             # 足够收敛
    "n_grad_mc_samples": 128,    # MC梯度样本数
    "n_acyclicity_mc_samples": 32  # 无环性MC样本数
}

# ========== 新6个任务组配置（2026-01-15更新） ==========
TASK_GROUPS = [
    {
        "id": "group1_examples",
        "name": "examples（图像分类-小型）",
        "csv_file": "group1_examples.csv",
        "expected_samples": 126,
        "expected_features": 18
    },
    {
        "id": "group2_vulberta",
        "name": "VulBERTa（代码漏洞检测）",
        "csv_file": "group2_vulberta.csv",
        "expected_samples": 52,
        "expected_features": 16
    },
    {
        "id": "group3_person_reid",
        "name": "Person_reID（行人重识别）",
        "csv_file": "group3_person_reid.csv",
        "expected_samples": 118,
        "expected_features": 19
    },
    {
        "id": "group4_bug_localization",
        "name": "bug-localization（缺陷定位）",
        "csv_file": "group4_bug_localization.csv",
        "expected_samples": 40,
        "expected_features": 17
    },
    {
        "id": "group5_mrt_oast",
        "name": "MRT-OAST（缺陷定位）",
        "csv_file": "group5_mrt_oast.csv",
        "expected_samples": 46,
        "expected_features": 16
    },
    {
        "id": "group6_resnet",
        "name": "pytorch_resnet（图像分类-ResNet）",
        "csv_file": "group6_resnet.csv",
        "expected_samples": 41,
        "expected_features": 18
    }
]


def load_task_group_data(task_config):
    """
    加载单个任务组的数据

    参数:
        task_config: 任务组配置字典

    返回:
        data: 标准化后的DataFrame
        feature_names: 特征名称列表
    """
    data_dir = Path(__file__).parent.parent / "data" / "energy_research" / "dibs_training"
    data_file = data_dir / task_config["csv_file"]

    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    # 加载数据
    df = pd.read_csv(data_file)

    print(f"  数据规模: {len(df)}行 × {len(df.columns)}列")
    print(f"  预期规模: {task_config['expected_samples']}行 × {task_config['expected_features']}列")

    if len(df) != task_config["expected_samples"]:
        print(f"  ⚠️ 警告: 样本数不符合预期（预期{task_config['expected_samples']}，实际{len(df)}）")

    if len(df.columns) != task_config["expected_features"]:
        print(f"  ⚠️ 警告: 特征数不符合预期（预期{task_config['expected_features']}，实际{len(df.columns)}）")

    # 检查缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  ⚠️ 警告: 发现 {missing_count} 个缺失值！DiBS要求零缺失值。")
        raise ValueError(f"数据包含缺失值，DiBS无法运行")

    # 检查常量特征（DiBS会崩溃）
    const_features = []
    for col in df.columns:
        if df[col].nunique() == 1:
            const_features.append(col)

    if const_features:
        print(f"  ⚠️ 警告: 发现 {len(const_features)} 个常量特征（将被移除）:")
        for col in const_features:
            print(f"    - {col} = {df[col].iloc[0]}")

        # 移除常量特征
        df = df.drop(columns=const_features)
        print(f"  ✅ 移除后: {len(df.columns)}列")

    # 保存特征名称
    feature_names = df.columns.tolist()

    # 返回DataFrame（CausalGraphLearner期望DataFrame输入）
    return df, feature_names


def classify_variables(feature_names):
    """
    将变量分类为：超参数、性能、能耗、中介变量、其他

    参数:
        feature_names: 特征名称列表

    返回:
        classification: 字典，包含各类变量的索引
    """
    classification = {
        "hyperparams": [],      # 超参数（X）
        "performance": [],      # 性能指标（Y_perf）
        "energy": [],           # 能耗指标（Y_energy）
        "mediators": [],        # 中介变量（M）
        "others": []            # 其他控制变量
    }

    for idx, name in enumerate(feature_names):
        if name.startswith("hyperparam_"):
            classification["hyperparams"].append(idx)
        elif name.startswith("perf_"):
            classification["performance"].append(idx)
        elif name.startswith("energy_gpu") or name.startswith("energy_cpu"):
            # 特殊处理：GPU利用率、温度、功率是中介变量
            if "util" in name or "temp" in name or "watts" in name:
                classification["mediators"].append(idx)
            else:
                classification["energy"].append(idx)
        elif name in ["duration_seconds", "retries", "num_mutated_params"]:
            classification["others"].append(idx)
        else:
            classification["others"].append(idx)

    return classification


def extract_research_question_1_evidence(causal_graph, feature_names, var_classification, threshold=0.3):
    """
    提取研究问题1的证据：超参数对能耗的影响

    参数:
        causal_graph: 因果图矩阵 (n_vars × n_vars)
        feature_names: 特征名称列表
        var_classification: 变量分类字典
        threshold: 边强度阈值

    返回:
        evidence: 包含超参数效应的字典
    """
    evidence = {
        "direct_hyperparam_to_energy": [],
        "mediated_hyperparam_to_energy": []
    }

    hyperparam_vars = var_classification["hyperparams"]
    energy_vars = var_classification["energy"]
    mediator_vars = var_classification["mediators"]

    # 1. 检测超参数→能耗的直接因果边
    for hp_idx in hyperparam_vars:
        for e_idx in energy_vars:
            strength = causal_graph[hp_idx, e_idx]
            if strength > threshold:
                evidence["direct_hyperparam_to_energy"].append({
                    "hyperparam": feature_names[hp_idx],
                    "energy_var": feature_names[e_idx],
                    "strength": float(strength)
                })

    # 2. 检测超参数→中介→能耗的间接路径
    for hp_idx in hyperparam_vars:
        for m_idx in mediator_vars:
            for e_idx in energy_vars:
                strength_hm = causal_graph[hp_idx, m_idx]
                strength_me = causal_graph[m_idx, e_idx]

                if strength_hm > threshold and strength_me > threshold:
                    evidence["mediated_hyperparam_to_energy"].append({
                        "hyperparam": feature_names[hp_idx],
                        "mediator": feature_names[m_idx],
                        "energy_var": feature_names[e_idx],
                        "strength_step1": float(strength_hm),
                        "strength_step2": float(strength_me),
                        "indirect_strength": float(strength_hm * strength_me)
                    })

    return evidence


def extract_research_question_2_evidence(causal_graph, feature_names, var_classification, threshold=0.3):
    """
    提取研究问题2的证据：能耗-性能权衡关系

    参数:
        causal_graph: 因果图矩阵 (n_vars × n_vars)
        feature_names: 特征名称列表
        var_classification: 变量分类字典
        threshold: 边强度阈值

    返回:
        evidence: 包含权衡关系证据的字典
    """
    evidence = {
        "direct_edges_perf_to_energy": [],
        "direct_edges_energy_to_perf": [],
        "common_hyperparams": [],
        "mediated_tradeoffs": []
    }

    perf_vars = var_classification["performance"]
    energy_vars = var_classification["energy"]
    hyperparam_vars = var_classification["hyperparams"]
    mediator_vars = var_classification["mediators"]

    # 1. 检测性能→能耗的直接因果边
    for i in perf_vars:
        for j in energy_vars:
            strength = causal_graph[i, j]
            if strength > threshold:
                evidence["direct_edges_perf_to_energy"].append({
                    "from": feature_names[i],
                    "to": feature_names[j],
                    "strength": float(strength)
                })

    # 2. 检测能耗→性能的直接因果边
    for i in energy_vars:
        for j in perf_vars:
            strength = causal_graph[i, j]
            if strength > threshold:
                evidence["direct_edges_energy_to_perf"].append({
                    "from": feature_names[i],
                    "to": feature_names[j],
                    "strength": float(strength)
                })

    # 3. 检测同时影响性能和能耗的超参数（权衡候选）
    for hp_idx in hyperparam_vars:
        hp_name = feature_names[hp_idx]

        # 找到该超参数影响的性能指标
        perf_targets = []
        for p_idx in perf_vars:
            if causal_graph[hp_idx, p_idx] > threshold:
                perf_targets.append({
                    "var": feature_names[p_idx],
                    "strength": float(causal_graph[hp_idx, p_idx])
                })

        # 找到该超参数影响的能耗指标
        energy_targets = []
        for e_idx in energy_vars:
            if causal_graph[hp_idx, e_idx] > threshold:
                energy_targets.append({
                    "var": feature_names[e_idx],
                    "strength": float(causal_graph[hp_idx, e_idx])
                })

        # 如果同时影响性能和能耗，记录为共同超参数
        if perf_targets and energy_targets:
            evidence["common_hyperparams"].append({
                "hyperparam": hp_name,
                "affects_performance": perf_targets,
                "affects_energy": energy_targets
            })

    # 4. 检测通过中介变量的间接权衡关系
    for p_idx in perf_vars:
        for m_idx in mediator_vars:
            for e_idx in energy_vars:
                strength_pm = causal_graph[p_idx, m_idx]
                strength_me = causal_graph[m_idx, e_idx]

                if strength_pm > threshold and strength_me > threshold:
                    evidence["mediated_tradeoffs"].append({
                        "path": f"{feature_names[p_idx]} → {feature_names[m_idx]} → {feature_names[e_idx]}",
                        "strength_step1": float(strength_pm),
                        "strength_step2": float(strength_me),
                        "path_strength": float(strength_pm * strength_me)
                    })

    return evidence


def extract_research_question_3_evidence(causal_graph, feature_names, var_classification, threshold=0.3):
    """
    提取研究问题3的证据：中介效应路径

    参数:
        causal_graph: 因果图矩阵
        feature_names: 特征名称列表
        var_classification: 变量分类字典
        threshold: 边强度阈值

    返回:
        evidence: 包含中介路径的字典
    """
    evidence = {
        "mediation_paths_to_energy": [],
        "mediation_paths_to_performance": [],
        "multi_step_paths": []
    }

    hyperparam_vars = var_classification["hyperparams"]
    perf_vars = var_classification["performance"]
    energy_vars = var_classification["energy"]
    mediator_vars = var_classification["mediators"]

    # 1. 三节点路径：超参数 → 中介变量 → 能耗
    for hp_idx in hyperparam_vars:
        for m_idx in mediator_vars:
            for e_idx in energy_vars:
                strength_hm = causal_graph[hp_idx, m_idx]
                strength_me = causal_graph[m_idx, e_idx]

                if strength_hm > threshold and strength_me > threshold:
                    direct_strength = causal_graph[hp_idx, e_idx]

                    evidence["mediation_paths_to_energy"].append({
                        "path_id": f"HP{hp_idx}_M{m_idx}_E{e_idx}",
                        "hyperparam": feature_names[hp_idx],
                        "mediator": feature_names[m_idx],
                        "outcome": feature_names[e_idx],
                        "strength_X_to_M": float(strength_hm),
                        "strength_M_to_Y": float(strength_me),
                        "indirect_strength": float(strength_hm * strength_me),
                        "direct_strength": float(direct_strength),
                        "mediation_type": "partial" if direct_strength > 0.01 else "full"
                    })

    # 2. 三节点路径：超参数 → 中介变量 → 性能
    for hp_idx in hyperparam_vars:
        for m_idx in mediator_vars:
            for p_idx in perf_vars:
                strength_hm = causal_graph[hp_idx, m_idx]
                strength_mp = causal_graph[m_idx, p_idx]

                if strength_hm > threshold and strength_mp > threshold:
                    direct_strength = causal_graph[hp_idx, p_idx]

                    evidence["mediation_paths_to_performance"].append({
                        "path_id": f"HP{hp_idx}_M{m_idx}_P{p_idx}",
                        "hyperparam": feature_names[hp_idx],
                        "mediator": feature_names[m_idx],
                        "outcome": feature_names[p_idx],
                        "strength_X_to_M": float(strength_hm),
                        "strength_M_to_Y": float(strength_mp),
                        "indirect_strength": float(strength_hm * strength_mp),
                        "direct_strength": float(direct_strength),
                        "mediation_type": "partial" if direct_strength > 0.01 else "full"
                    })

    # 3. 四节点路径（可选）：超参数 → 中介1 → 中介2 → 能耗/性能
    for hp_idx in hyperparam_vars:
        for m1_idx in mediator_vars:
            for m2_idx in mediator_vars:
                if m1_idx == m2_idx:
                    continue

                strength_hm1 = causal_graph[hp_idx, m1_idx]
                strength_m1m2 = causal_graph[m1_idx, m2_idx]

                if strength_hm1 > threshold and strength_m1m2 > threshold:
                    for e_idx in energy_vars + perf_vars:
                        strength_m2y = causal_graph[m2_idx, e_idx]

                        if strength_m2y > threshold:
                            evidence["multi_step_paths"].append({
                                "path": f"{feature_names[hp_idx]} → {feature_names[m1_idx]} → {feature_names[m2_idx]} → {feature_names[e_idx]}",
                                "strength_step1": float(strength_hm1),
                                "strength_step2": float(strength_m1m2),
                                "strength_step3": float(strength_m2y),
                                "path_strength": float(strength_hm1 * strength_m1m2 * strength_m2y)
                            })

    return evidence


def run_dibs_analysis(task_config, data, feature_names, config, output_dir):
    """
    运行单个任务组的DiBS分析

    参数:
        task_config: 任务组配置
        data: 数据DataFrame
        feature_names: 特征名称列表
        config: DiBS配置
        output_dir: 输出目录

    返回:
        result: 分析结果字典
    """
    task_id = task_config["id"]
    task_name = task_config["name"]

    print("\n" + "="*80)
    print(f"任务组: {task_id} - {task_name}")
    print("="*80)

    # 分类变量
    var_classification = classify_variables(feature_names)

    print(f"\n变量分类:")
    print(f"  超参数: {len(var_classification['hyperparams'])}个")
    for idx in var_classification['hyperparams']:
        print(f"    - {feature_names[idx]}")
    print(f"  性能指标: {len(var_classification['performance'])}个")
    for idx in var_classification['performance']:
        print(f"    - {feature_names[idx]}")
    print(f"  能耗指标: {len(var_classification['energy'])}个")
    for idx in var_classification['energy']:
        print(f"    - {feature_names[idx]}")
    print(f"  中介变量: {len(var_classification['mediators'])}个")
    for idx in var_classification['mediators']:
        print(f"    - {feature_names[idx]}")
    print(f"  其他变量: {len(var_classification['others'])}个")

    # 创建DiBS learner
    learner = CausalGraphLearner(
        n_vars=len(feature_names),
        alpha=config["alpha_linear"],
        n_particles=config["n_particles"],
        beta=config["beta_linear"],
        tau=config["tau"],
        n_steps=config["n_steps"],
        n_grad_mc_samples=config["n_grad_mc_samples"],
        n_acyclicity_mc_samples=config["n_acyclicity_mc_samples"],
        random_seed=42
    )

    # 执行DiBS
    print(f"\n执行DiBS因果发现...")
    print(f"  alpha_linear: {config['alpha_linear']}")
    print(f"  beta_linear: {config['beta_linear']}")
    print(f"  n_particles: {config['n_particles']}")
    print(f"  n_steps: {config['n_steps']}")

    start_time = time.time()

    try:
        causal_graph = learner.fit(data, verbose=True)
        elapsed_time = time.time() - start_time
        success = True
        error_msg = None

        print(f"\n✅ DiBS执行成功！耗时: {elapsed_time/60:.2f}分钟")

    except Exception as e:
        elapsed_time = time.time() - start_time
        causal_graph = None
        success = False
        error_msg = str(e)
        print(f"\n❌ DiBS执行失败: {error_msg}")

        # 打印完整的堆栈跟踪以调试
        import traceback
        print("\n完整错误信息:")
        traceback.print_exc()

        return {
            "task_id": task_id,
            "task_name": task_name,
            "success": False,
            "elapsed_time_minutes": elapsed_time / 60,
            "error_message": error_msg
        }

    # 分析因果图
    graph_min = float(causal_graph.min())
    graph_max = float(causal_graph.max())
    graph_mean = float(causal_graph.mean())
    graph_std = float(causal_graph.std())

    # 不同阈值下的边数
    edges_001 = int(np.sum(causal_graph > 0.01))
    edges_01 = int(np.sum(causal_graph > 0.1))
    edges_03 = int(np.sum(causal_graph > 0.3))
    edges_05 = int(np.sum(causal_graph > 0.5))

    print(f"\n因果图统计:")
    print(f"  最小值: {graph_min:.6f}")
    print(f"  最大值: {graph_max:.6f}")
    print(f"  平均值: {graph_mean:.6f}")
    print(f"  标准差: {graph_std:.6f}")
    print(f"\n边数统计:")
    print(f"  >0.01: {edges_001}条")
    print(f"  >0.1:  {edges_01}条")
    print(f"  >0.3:  {edges_03}条 ⭐ 强边")
    print(f"  >0.5:  {edges_05}条")

    # 提取研究问题证据
    print(f"\n提取研究问题1证据（超参数→能耗）...")
    q1_evidence = extract_research_question_1_evidence(
        causal_graph, feature_names, var_classification, threshold=0.3
    )
    print(f"  直接边（超参数→能耗）: {len(q1_evidence['direct_hyperparam_to_energy'])}条")
    print(f"  间接路径（超参数→中介→能耗）: {len(q1_evidence['mediated_hyperparam_to_energy'])}条")

    print(f"\n提取研究问题2证据（能耗-性能权衡）...")
    q2_evidence = extract_research_question_2_evidence(
        causal_graph, feature_names, var_classification, threshold=0.3
    )
    print(f"  直接边（性能→能耗）: {len(q2_evidence['direct_edges_perf_to_energy'])}条")
    print(f"  直接边（能耗→性能）: {len(q2_evidence['direct_edges_energy_to_perf'])}条")
    print(f"  共同超参数: {len(q2_evidence['common_hyperparams'])}个")
    print(f"  中介权衡路径: {len(q2_evidence['mediated_tradeoffs'])}条")

    print(f"\n提取研究问题3证据（中介效应）...")
    q3_evidence = extract_research_question_3_evidence(
        causal_graph, feature_names, var_classification, threshold=0.3
    )
    print(f"  超参数→中介→能耗: {len(q3_evidence['mediation_paths_to_energy'])}条")
    print(f"  超参数→中介→性能: {len(q3_evidence['mediation_paths_to_performance'])}条")
    print(f"  多步路径: {len(q3_evidence['multi_step_paths'])}条")

    # 保存因果图矩阵
    graph_file = output_dir / f"{task_id}_causal_graph.npy"
    np.save(graph_file, causal_graph)
    print(f"\n✅ 因果图矩阵已保存: {graph_file}")

    # 保存特征名称
    names_file = output_dir / f"{task_id}_feature_names.json"
    with open(names_file, 'w') as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)

    # 构建结果字典
    result = {
        "task_id": task_id,
        "task_name": task_name,
        "success": True,
        "elapsed_time_minutes": elapsed_time / 60,
        "n_samples": len(data),
        "n_features": len(feature_names),
        "variable_classification": {
            "n_hyperparams": len(var_classification["hyperparams"]),
            "n_performance": len(var_classification["performance"]),
            "n_energy": len(var_classification["energy"]),
            "n_mediators": len(var_classification["mediators"]),
            "hyperparam_names": [feature_names[i] for i in var_classification["hyperparams"]],
            "performance_names": [feature_names[i] for i in var_classification["performance"]],
            "energy_names": [feature_names[i] for i in var_classification["energy"]],
            "mediator_names": [feature_names[i] for i in var_classification["mediators"]]
        },
        "graph_stats": {
            "min": graph_min,
            "max": graph_max,
            "mean": graph_mean,
            "std": graph_std
        },
        "edges": {
            "threshold_0.01": edges_001,
            "threshold_0.1": edges_01,
            "threshold_0.3": edges_03,
            "threshold_0.5": edges_05
        },
        "question1_evidence": q1_evidence,
        "question2_evidence": q2_evidence,
        "question3_evidence": q3_evidence,
        "config": config,
        "feature_names": feature_names
    }

    # 保存单个任务结果
    result_file = output_dir / f"{task_id}_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ 分析结果已保存: {result_file}")

    return result


def generate_summary_report(all_results, output_dir):
    """生成总结报告"""

    report_file = output_dir / "NEW_6GROUPS_DIBS_ANALYSIS_REPORT.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 新6分组数据DiBS因果分析报告\n\n")
        f.write(f"**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**数据源**: data.csv (970行) → 6组清洗数据 (423样本)\n")
        f.write(f"**任务组数**: {len(all_results)}个\n")
        f.write(f"**DiBS配置**: alpha=0.05, beta=0.1, particles=20, steps=5000\n\n")

        f.write("---\n\n")

        # 总体统计
        f.write("## 📊 总体统计\n\n")

        successful = sum(1 for r in all_results if r['success'])
        total_time = sum(r['elapsed_time_minutes'] for r in all_results)

        f.write(f"- **成功任务组**: {successful}/{len(all_results)}\n")
        f.write(f"- **总耗时**: {total_time:.1f}分钟 ({total_time/60:.2f}小时)\n")
        f.write(f"- **平均耗时**: {total_time/len(all_results):.1f}分钟/组\n\n")

        # 详细结果表格
        f.write("## 📋 任务组详细结果\n\n")
        f.write("| 任务组 | 状态 | 耗时(分) | 样本数 | 特征数 | 超参数 | 性能 | 能耗 | 中介 | 强边(>0.3) | 总边(>0.01) |\n")
        f.write("|--------|------|---------|-------|-------|--------|------|------|------|-----------|------------|\n")

        for r in all_results:
            status = "✅ 成功" if r['success'] else "❌ 失败"

            if r['success']:
                var_class = r['variable_classification']
                f.write(f"| {r['task_name'][:25]} | {status} | "
                       f"{r['elapsed_time_minutes']:.1f} | {r['n_samples']} | "
                       f"{r['n_features']} | {var_class['n_hyperparams']} | "
                       f"{var_class['n_performance']} | {var_class['n_energy']} | "
                       f"{var_class['n_mediators']} | {r['edges']['threshold_0.3']} | "
                       f"{r['edges']['threshold_0.01']} |\n")
            else:
                f.write(f"| {r['task_name'][:25]} | {status} | "
                       f"{r['elapsed_time_minutes']:.1f} | - | - | - | - | - | - | - | - |\n")

        f.write("\n")

        successful_results = [r for r in all_results if r['success']]

        # 问题1证据汇总
        f.write("## 🎯 研究问题1：超参数对能耗的影响\n\n")

        if not successful_results:
            f.write("❌ 所有任务组DiBS分析失败，无法提取证据。\n\n")
        else:
            total_direct_hp_to_energy = sum(len(r['question1_evidence']['direct_hyperparam_to_energy']) for r in successful_results)
            total_mediated_hp_to_energy = sum(len(r['question1_evidence']['mediated_hyperparam_to_energy']) for r in successful_results)

            f.write(f"### 总体发现\n\n")
            f.write(f"- **直接因果边（超参数→能耗）**: {total_direct_hp_to_energy}条\n")
            f.write(f"- **间接路径（超参数→中介→能耗）**: {total_mediated_hp_to_energy}条\n")
            f.write(f"- **总因果路径**: {total_direct_hp_to_energy + total_mediated_hp_to_energy}条\n\n")

            # 详细列出直接效应
            if total_direct_hp_to_energy > 0:
                f.write(f"### 超参数→能耗直接效应 (Top 10)\n\n")
                f.write("| 任务组 | 超参数 | 能耗指标 | 强度 |\n")
                f.write("|--------|--------|----------|------|\n")

                all_direct = []
                for r in successful_results:
                    for edge in r['question1_evidence']['direct_hyperparam_to_energy']:
                        all_direct.append({
                            "task": r['task_name'][:20],
                            "hp": edge['hyperparam'],
                            "energy": edge['energy_var'],
                            "strength": edge['strength']
                        })

                all_direct.sort(key=lambda x: x['strength'], reverse=True)

                for edge in all_direct[:10]:
                    f.write(f"| {edge['task']} | {edge['hp']} | {edge['energy']} | {edge['strength']:.4f} |\n")

                f.write("\n")

        # 问题2证据汇总
        f.write("## 🔄 研究问题2：能耗-性能权衡关系\n\n")

        if not successful_results:
            f.write("❌ 所有任务组DiBS分析失败，无法提取证据。\n\n")
        else:
            total_direct_perf_to_energy = sum(len(r['question2_evidence']['direct_edges_perf_to_energy']) for r in successful_results)
            total_direct_energy_to_perf = sum(len(r['question2_evidence']['direct_edges_energy_to_perf']) for r in successful_results)
            total_common_hyperparams = sum(len(r['question2_evidence']['common_hyperparams']) for r in successful_results)
            total_mediated_tradeoffs = sum(len(r['question2_evidence']['mediated_tradeoffs']) for r in successful_results)

            f.write(f"### 总体发现\n\n")
            f.write(f"- **直接因果边（性能→能耗）**: {total_direct_perf_to_energy}条\n")
            f.write(f"- **直接因果边（能耗→性能）**: {total_direct_energy_to_perf}条\n")
            f.write(f"- **共同超参数**: {total_common_hyperparams}个（同时影响能耗和性能）\n")
            f.write(f"- **中介权衡路径**: {total_mediated_tradeoffs}条\n\n")

        # 问题3证据汇总
        f.write("## 🔬 研究问题3：中介效应路径\n\n")

        if not successful_results:
            f.write("❌ 所有任务组DiBS分析失败，无法提取证据。\n\n")
        else:
            total_mediation_to_energy = sum(len(r['question3_evidence']['mediation_paths_to_energy']) for r in successful_results)
            total_mediation_to_perf = sum(len(r['question3_evidence']['mediation_paths_to_performance']) for r in successful_results)
            total_multi_step = sum(len(r['question3_evidence']['multi_step_paths']) for r in successful_results)

            f.write(f"### 总体发现\n\n")
            f.write(f"- **中介路径（超参数→中介→能耗）**: {total_mediation_to_energy}条\n")
            f.write(f"- **中介路径（超参数→中介→性能）**: {total_mediation_to_perf}条\n")
            f.write(f"- **多步路径（≥4节点）**: {total_multi_step}条\n")
            f.write(f"- **总中介路径**: {total_mediation_to_energy + total_mediation_to_perf + total_multi_step}条\n\n")

        # 结论
        f.write("## 💡 结论与下一步\n\n")

        if successful:
            f.write(f"✅ DiBS成功在{successful}/{len(all_results)}个任务组上完成因果发现。\n\n")
            f.write("### 下一步建议\n\n")
            f.write("1. 使用回归分析量化DiBS发现的因果边强度\n")
            f.write("2. 对中介路径进行Sobel检验验证\n")
            f.write("3. 生成因果图可视化\n")
            f.write("4. 撰写研究发现报告\n")
        else:
            f.write("❌ 所有任务组DiBS分析失败。\n\n")

        f.write("\n---\n\n")
        f.write(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n✅ 总结报告已保存: {report_file}")

    return report_file


def main():
    """主函数"""
    print("="*80)
    print("新6分组数据DiBS因果分析")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据源: data.csv (2026-01-15生成)")
    print(f"任务组数: {len(TASK_GROUPS)}")
    print(f"DiBS配置: alpha={OPTIMAL_CONFIG['alpha_linear']}, beta={OPTIMAL_CONFIG['beta_linear']}, particles={OPTIMAL_CONFIG['n_particles']}")
    print("="*80)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "results" / "energy_research" / "new_6groups_dibs" / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 运行所有任务组
    all_results = []

    for i, task_config in enumerate(TASK_GROUPS, 1):
        print(f"\n{'='*80}")
        print(f"进度: {i}/{len(TASK_GROUPS)}")
        print(f"{'='*80}")

        try:
            # 加载数据
            print(f"\n加载数据: {task_config['name']}")
            data, feature_names = load_task_group_data(task_config)

            # 运行DiBS分析
            result = run_dibs_analysis(
                task_config,
                data,
                feature_names,
                OPTIMAL_CONFIG,
                output_dir
            )

            all_results.append(result)

        except KeyboardInterrupt:
            print("\n\n用户中断分析")
            break

        except Exception as e:
            print(f"\n❌ 任务组执行异常: {e}")
            import traceback
            traceback.print_exc()

            all_results.append({
                "task_id": task_config["id"],
                "task_name": task_config["name"],
                "success": False,
                "elapsed_time_minutes": 0,
                "error_message": str(e)
            })

    # 生成总结报告
    print("\n" + "="*80)
    print("生成总结报告...")
    print("="*80)

    if all_results:
        report_file = generate_summary_report(all_results, output_dir)

        print(f"\n{'='*80}")
        print("✅ DiBS分析完成！")
        print(f"{'='*80}")
        print(f"  成功任务组: {sum(1 for r in all_results if r['success'])}/{len(all_results)}")
        print(f"  结果目录: {output_dir}")
        print(f"  总结报告: {report_file}")
        print(f"{'='*80}\n")
    else:
        print("\n❌ 没有完成任何任务组")

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
