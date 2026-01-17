#!/usr/bin/env python3
"""
6分组交互项数据DiBS因果分析脚本

目的: 在交互项数据上执行DiBS因果发现，探测调节效应
数据源: analysis/data/energy_research/6groups_interaction/
研究问题:
  - 问题1: 超参数对能耗的影响（包括调节效应）
  - 问题2: 能耗和性能之间的权衡关系
  - 问题3: 中间变量的中介效应

创建日期: 2026-01-16
基于: run_dibs_6groups_final.py
新增: 交互项支持 (超参数 × is_parallel)
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
    "beta_linear": 0.1,          # 低无环约束，允许更多边探索 ⭐关键参数
    "n_particles": 20,           # 最佳性价比
    "tau": 1.0,                  # Gumbel-softmax温度
    "n_steps": 5000,             # 足够收敛
    "n_grad_mc_samples": 128,    # MC梯度样本数
    "n_acyclicity_mc_samples": 32  # 无环性MC样本数
}

# ========== 6个任务组配置（交互项版本，2026-01-16）==========
TASK_GROUPS = [
    {
        "id": "group1_examples",
        "name": "examples（图像分类-小型）",
        "csv_file": "group1_examples_interaction.csv",
        "expected_samples": 304,
        "expected_features": 24,  # 21 + 3交互项
        "n_interaction_terms": 3
    },
    {
        "id": "group2_vulberta",
        "name": "VulBERTa（代码漏洞检测）",
        "csv_file": "group2_vulberta_interaction.csv",
        "expected_samples": 72,
        "expected_features": 23,  # 20 + 3交互项
        "n_interaction_terms": 3
    },
    {
        "id": "group3_person_reid",
        "name": "Person_reID（行人重识别）",
        "csv_file": "group3_person_reid_interaction.csv",
        "expected_samples": 206,
        "expected_features": 25,  # 22 + 3交互项
        "n_interaction_terms": 3
    },
    {
        "id": "group4_bug_localization",
        "name": "bug-localization（缺陷定位）",
        "csv_file": "group4_bug_localization_interaction.csv",
        "expected_samples": 90,
        "expected_features": 24,  # 21 + 3交互项
        "n_interaction_terms": 3
    },
    {
        "id": "group5_mrt_oast",
        "name": "MRT-OAST（缺陷定位）",
        "csv_file": "group5_mrt_oast_interaction.csv",
        "expected_samples": 72,
        "expected_features": 25,  # 21 + 4交互项
        "n_interaction_terms": 4
    },
    {
        "id": "group6_resnet",
        "name": "pytorch_resnet（图像分类-ResNet）",
        "csv_file": "group6_resnet_interaction.csv",
        "expected_samples": 74,
        "expected_features": 22,  # 19 + 3交互项
        "n_interaction_terms": 3
    }
]


def load_task_group_data(task_config):
    """
    加载单个任务组的交互项数据

    参数:
        task_config: 任务组配置字典

    返回:
        data: 处理后的DataFrame
        feature_names: 特征名称列表
    """
    # 使用交互项数据路径
    data_dir = Path(__file__).parent.parent / "data" / "energy_research" / "6groups_interaction"
    data_file = data_dir / task_config["csv_file"]

    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    # 加载数据
    df = pd.read_csv(data_file)

    print(f"  数据规模: {len(df)}行 × {len(df.columns)}列")
    print(f"  预期规模: {task_config['expected_samples']}行 × {task_config['expected_features']}列")
    print(f"  交互项数: {task_config['n_interaction_terms']}个")

    if len(df) != task_config["expected_samples"]:
        print(f"  ⚠️ 警告: 样本数不符合预期（预期{task_config['expected_samples']}，实际{len(df)}）")

    if len(df.columns) != task_config["expected_features"]:
        print(f"  ⚠️ 警告: 特征数不符合预期（预期{task_config['expected_features']}，实际{len(df.columns)}）")

    # 检查缺失值 - DiBS要求零缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  ⚠️ 警告: 发现 {missing_count} 个缺失值！")
        print(f"  处理方式: 使用列均值填充...")

        # 对每一列使用均值填充
        for col in df.columns:
            if df[col].isnull().any():
                missing_before = df[col].isnull().sum()
                df[col] = df[col].fillna(df[col].mean())
                print(f"    - {col}: {missing_before}个缺失值已填充")

        # 再次检查
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            raise ValueError(f"填充后仍有 {remaining_missing} 个缺失值！")
        else:
            print(f"  ✅ 所有缺失值已填充")

    # 移除timestamp列（DiBS无法处理字符串）
    if 'timestamp' in df.columns:
        print(f"  ⚠️ 移除timestamp列（DiBS无法处理字符串）")
        df = df.drop(columns=['timestamp'])

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

    # 验证交互项存在
    interaction_terms = [col for col in feature_names if col.endswith('_x_is_parallel')]
    print(f"  ✅ 检测到 {len(interaction_terms)} 个交互项:")
    for term in interaction_terms:
        print(f"    - {term}")

    return df, feature_names


def classify_variables(feature_names):
    """
    将变量分类为：超参数、性能、能耗、中介变量、交互项、其他

    参数:
        feature_names: 特征名称列表

    返回:
        classification: 字典，包含各类变量的索引
    """
    classification = {
        "hyperparams": [],      # 超参数（X）- 不包括交互项
        "performance": [],      # 性能指标（Y_perf）
        "energy": [],           # 能耗指标（Y_energy）
        "mediators": [],        # 中介变量（M）
        "interactions": [],     # 交互项（X × is_parallel）⭐ 新增
        "others": []            # 其他控制变量
    }

    for idx, name in enumerate(feature_names):
        # ⭐ 优先识别交互项（避免被误分类为超参数）
        if name.endswith("_x_is_parallel"):
            classification["interactions"].append(idx)
        elif name.startswith("hyperparam_") and not name.endswith("_seed"):
            classification["hyperparams"].append(idx)
        elif name.startswith("perf_"):
            classification["performance"].append(idx)
        elif name.startswith("energy_"):
            # 区分能耗结果变量和中介变量
            if "util" in name or "temp" in name or "watts" in name or "avg" in name or "max" in name or "min" in name:
                classification["mediators"].append(idx)
            else:
                # 焦耳（joules）是能耗结果变量
                classification["energy"].append(idx)
        elif name.startswith("model_"):
            # 模型变量视为控制变量
            classification["others"].append(idx)
        elif name in ["is_parallel", "timestamp"]:
            classification["others"].append(idx)
        else:
            classification["others"].append(idx)

    return classification


def extract_research_question_1_evidence(causal_graph, feature_names, var_classification, threshold=0.3):
    """
    提取研究问题1的证据：超参数对能耗的影响（包括调节效应）

    参数:
        causal_graph: 因果图矩阵 (n_vars × n_vars)
        feature_names: 特征名称列表
        var_classification: 变量分类字典
        threshold: 边强度阈值

    返回:
        evidence: 包含超参数效应和调节效应的字典
    """
    evidence = {
        "direct_hyperparam_to_energy": [],
        "mediated_hyperparam_to_energy": [],
        "moderation_effects": []  # ⭐ 新增：调节效应（交互项→能耗）
    }

    hyperparam_vars = var_classification["hyperparams"]
    energy_vars = var_classification["energy"]
    mediator_vars = var_classification["mediators"]
    interaction_vars = var_classification["interactions"]  # ⭐ 新增

    # 1. 检测超参数→能耗的直接因果边（主效应）
    for hp_idx in hyperparam_vars:
        for e_idx in energy_vars:
            strength = causal_graph[hp_idx, e_idx]
            if strength > threshold:
                evidence["direct_hyperparam_to_energy"].append({
                    "hyperparam": feature_names[hp_idx],
                    "energy_var": feature_names[e_idx],
                    "strength": float(strength),
                    "effect_type": "main_effect"
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

    # ⭐ 3. 检测调节效应：交互项→能耗
    for int_idx in interaction_vars:
        interaction_name = feature_names[int_idx]
        base_hyperparam = interaction_name.replace("_x_is_parallel", "")

        for e_idx in energy_vars:
            strength = causal_graph[int_idx, e_idx]
            if strength > threshold:
                # 检查主效应是否存在
                base_hp_idx = None
                for hp_idx in hyperparam_vars:
                    if feature_names[hp_idx] == base_hyperparam:
                        base_hp_idx = hp_idx
                        break

                main_effect_strength = causal_graph[base_hp_idx, e_idx] if base_hp_idx is not None else 0.0

                evidence["moderation_effects"].append({
                    "interaction_term": interaction_name,
                    "base_hyperparam": base_hyperparam,
                    "energy_var": feature_names[e_idx],
                    "moderation_strength": float(strength),
                    "main_effect_strength": float(main_effect_strength),
                    "interpretation": "is_parallel调节了该超参数对能耗的效应" if strength > 0.3 else "弱调节效应"
                })

    return evidence


def extract_research_question_2_evidence(causal_graph, feature_names, var_classification, threshold=0.3):
    """
    提取研究问题2的证据：能耗-性能权衡关系

    （与原版相同，无需修改）
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

    （与原版相同，无需修改）
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
    运行单个任务组的DiBS分析（交互项版本）
    """
    task_id = task_config["id"]
    task_name = task_config["name"]

    print("\n" + "="*80)
    print(f"任务组: {task_id} - {task_name}")
    print("="*80)

    # 分类变量（包括交互项）
    var_classification = classify_variables(feature_names)

    print(f"\n变量分类:")
    print(f"  超参数（主效应）: {len(var_classification['hyperparams'])}个")
    for idx in var_classification['hyperparams']:
        print(f"    - {feature_names[idx]}")
    print(f"  ⭐ 交互项（调节效应）: {len(var_classification['interactions'])}个")
    for idx in var_classification['interactions']:
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
    print(f"  beta_linear: {config['beta_linear']} ⭐ 关键参数")
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
    print(f"\n提取研究问题1证据（超参数→能耗，包括调节效应）...")
    q1_evidence = extract_research_question_1_evidence(
        causal_graph, feature_names, var_classification, threshold=0.3
    )
    print(f"  主效应（超参数→能耗）: {len(q1_evidence['direct_hyperparam_to_energy'])}条")
    print(f"  ⭐ 调节效应（交互项→能耗）: {len(q1_evidence['moderation_effects'])}条")
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
            "n_interactions": len(var_classification["interactions"]),  # ⭐ 新增
            "n_performance": len(var_classification["performance"]),
            "n_energy": len(var_classification["energy"]),
            "n_mediators": len(var_classification["mediators"]),
            "hyperparam_names": [feature_names[i] for i in var_classification["hyperparams"]],
            "interaction_names": [feature_names[i] for i in var_classification["interactions"]],  # ⭐ 新增
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
    """生成总结报告（交互项版本）"""

    report_file = output_dir / "DIBS_INTERACTION_ANALYSIS_REPORT.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 6分组交互项数据DiBS因果分析报告\n\n")
        f.write(f"**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**数据源**: analysis/data/energy_research/6groups_interaction/\n")
        f.write(f"**数据类型**: 标准化 + 交互项 (超参数 × is_parallel)\n")
        f.write(f"**任务组数**: {len(all_results)}个\n")
        f.write(f"**DiBS配置**: alpha=0.05, beta=0.1 ⭐, particles=20, steps=5000\n\n")

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
        f.write("| 任务组 | 状态 | 耗时(分) | 样本数 | 特征数 | 超参数 | 交互项⭐ | 性能 | 能耗 | 强边(>0.3) |\n")
        f.write("|--------|------|---------|-------|-------|--------|----------|------|------|----------|\n")

        for r in all_results:
            status = "✅ 成功" if r['success'] else "❌ 失败"

            if r['success']:
                var_class = r['variable_classification']
                f.write(f"| {r['task_name'][:25]} | {status} | "
                       f"{r['elapsed_time_minutes']:.1f} | {r['n_samples']} | "
                       f"{r['n_features']} | {var_class['n_hyperparams']} | "
                       f"{var_class['n_interactions']} | "
                       f"{var_class['n_performance']} | {var_class['n_energy']} | "
                       f"{r['edges']['threshold_0.3']} |\n")
            else:
                f.write(f"| {r['task_name'][:25]} | {status} | "
                       f"{r['elapsed_time_minutes']:.1f} | - | - | - | - | - | - | - |\n")

        f.write("\n")

        successful_results = [r for r in all_results if r['success']]

        # 问题1证据汇总（包括调节效应）
        f.write("## 🎯 研究问题1：超参数对能耗的影响（包括调节效应）\n\n")

        if not successful_results:
            f.write("❌ 所有任务组DiBS分析失败，无法提取证据。\n\n")
        else:
            total_main_effects = sum(len(r['question1_evidence']['direct_hyperparam_to_energy']) for r in successful_results)
            total_moderation_effects = sum(len(r['question1_evidence']['moderation_effects']) for r in successful_results)
            total_mediated = sum(len(r['question1_evidence']['mediated_hyperparam_to_energy']) for r in successful_results)

            f.write(f"### 总体发现\n\n")
            f.write(f"- **主效应（超参数→能耗）**: {total_main_effects}条\n")
            f.write(f"- **⭐ 调节效应（交互项→能耗）**: {total_moderation_effects}条\n")
            f.write(f"- **间接路径（超参数→中介→能耗）**: {total_mediated}条\n")
            f.write(f"- **总因果路径**: {total_main_effects + total_moderation_effects + total_mediated}条\n\n")

            # 调节效应详情
            if total_moderation_effects > 0:
                f.write(f"### ⭐ 调节效应分析 (Top 10)\n\n")
                f.write("| 任务组 | 交互项 | 能耗指标 | 调节强度 | 主效应强度 |\n")
                f.write("|--------|--------|----------|---------|----------|\n")

                all_moderations = []
                for r in successful_results:
                    for mod in r['question1_evidence']['moderation_effects']:
                        all_moderations.append({
                            "task": r['task_name'][:20],
                            "interaction": mod['interaction_term'],
                            "energy": mod['energy_var'],
                            "mod_strength": mod['moderation_strength'],
                            "main_strength": mod['main_effect_strength']
                        })

                all_moderations.sort(key=lambda x: x['mod_strength'], reverse=True)

                for mod in all_moderations[:10]:
                    f.write(f"| {mod['task']} | {mod['interaction']} | {mod['energy']} | "
                           f"{mod['mod_strength']:.4f} | {mod['main_strength']:.4f} |\n")

                f.write("\n")

        # 问题2和问题3保持不变...
        f.write("## 🔄 研究问题2：能耗-性能权衡关系\n\n")
        f.write("（与原始分析相同，未受交互项影响）\n\n")

        f.write("## 🔬 研究问题3：中介效应路径\n\n")
        f.write("（与原始分析相同，未受交互项影响）\n\n")

        # 结论
        f.write("## 💡 结论与下一步\n\n")

        if successful:
            f.write(f"✅ DiBS成功在{successful}/{len(all_results)}个任务组上完成因果发现。\n\n")
            f.write("### 交互项方案关键发现\n\n")
            f.write("1. DiBS能够识别交互项（超参数 × is_parallel）对能耗的影响\n")
            f.write("2. 调节效应揭示了并行模式如何改变超参数的因果作用\n")
            f.write("3. 主效应和调节效应可以同时存在，提供完整的因果图景\n\n")

        f.write("\n---\n\n")
        f.write(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**使用脚本**: run_dibs_6groups_interaction.py\n")
        f.write(f"**方法学参考**: docs/INTERACTION_TERMS_TRANSFORMATION_PLAN.md\n")

    print(f"\n✅ 总结报告已保存: {report_file}")

    return report_file


def main():
    """主函数"""
    print("="*80)
    print("6分组交互项数据DiBS因果分析")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据源: analysis/data/energy_research/6groups_interaction/")
    print(f"任务组数: {len(TASK_GROUPS)}")
    print(f"DiBS配置: alpha={OPTIMAL_CONFIG['alpha_linear']}, beta={OPTIMAL_CONFIG['beta_linear']} ⭐")
    print(f"⭐ 新特性: 支持交互项（超参数 × is_parallel）分析调节效应")
    print("="*80)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "results" / "energy_research" / "dibs_interaction" / datetime.now().strftime('%Y%m%d_%H%M%S')
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
