#!/usr/bin/env python3
"""
DiBS参数扫描测试脚本

目的: 系统性地测试不同参数组合，寻找能产生边的最佳配置
创建日期: 2026-01-04
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

# 测试参数配置
EXPERIMENTS = [
    # 阶段1: 极小alpha值测试（关键！之前从未测试过）
    {
        "id": "A1",
        "name": "极小alpha (0.001)",
        "alpha": 0.001,
        "n_particles": 20,
        "beta": 1.0,
        "tau": 1.0,
        "n_steps": 5000,
        "n_grad_mc_samples": 128,
        "priority": 1
    },
    {
        "id": "A2",
        "name": "小alpha (0.01)",
        "alpha": 0.01,
        "n_particles": 20,
        "beta": 1.0,
        "tau": 1.0,
        "n_steps": 5000,
        "n_grad_mc_samples": 128,
        "priority": 1
    },
    {
        "id": "A3",
        "name": "DiBS默认alpha (0.05)",
        "alpha": 0.05,
        "n_particles": 20,
        "beta": 1.0,
        "tau": 1.0,
        "n_steps": 5000,
        "n_grad_mc_samples": 128,
        "priority": 1
    },

    # 阶段2: 增加粒子数
    {
        "id": "A4",
        "name": "中等粒子数 (50)",
        "alpha": 0.05,
        "n_particles": 50,
        "beta": 1.0,
        "tau": 1.0,
        "n_steps": 5000,
        "n_grad_mc_samples": 128,
        "priority": 2
    },
    {
        "id": "A5",
        "name": "大粒子数 (100)",
        "alpha": 0.05,
        "n_particles": 100,
        "beta": 1.0,
        "tau": 1.0,
        "n_steps": 5000,
        "n_grad_mc_samples": 128,
        "priority": 2
    },

    # 阶段3: 降低约束
    {
        "id": "A6",
        "name": "低beta约束 (0.1)",
        "alpha": 0.05,
        "n_particles": 20,
        "beta": 0.1,
        "tau": 1.0,
        "n_steps": 5000,
        "n_grad_mc_samples": 128,
        "priority": 3
    },
    {
        "id": "A7",
        "name": "中beta约束 (0.5)",
        "alpha": 0.05,
        "n_particles": 20,
        "beta": 0.5,
        "tau": 1.0,
        "n_steps": 5000,
        "n_grad_mc_samples": 128,
        "priority": 3
    },

    # 阶段4: 调整tau
    {
        "id": "A8",
        "name": "低tau温度 (0.5)",
        "alpha": 0.05,
        "n_particles": 20,
        "beta": 1.0,
        "tau": 0.5,
        "n_steps": 5000,
        "n_grad_mc_samples": 128,
        "priority": 4
    },

    # 阶段5: 增加MC样本数
    {
        "id": "A9",
        "name": "高MC样本数 (256)",
        "alpha": 0.05,
        "n_particles": 20,
        "beta": 1.0,
        "tau": 1.0,
        "n_steps": 5000,
        "n_grad_mc_samples": 256,
        "priority": 4
    },

    # 阶段6: 组合优化（基于阶段1-5的最佳结果）
    {
        "id": "B1",
        "name": "组合优化: 极小alpha + 大粒子数",
        "alpha": 0.001,
        "n_particles": 100,
        "beta": 1.0,
        "tau": 1.0,
        "n_steps": 10000,
        "n_grad_mc_samples": 128,
        "priority": 5
    },
    {
        "id": "B2",
        "name": "组合优化: 小alpha + 低约束",
        "alpha": 0.01,
        "n_particles": 50,
        "beta": 0.1,
        "tau": 1.0,
        "n_steps": 10000,
        "n_grad_mc_samples": 256,
        "priority": 5
    },
]

def load_test_data():
    """
    加载测试数据（使用examples任务组，219样本，15变量）

    注意: 这里需要准备好处理过的数据文件
    """
    data_file = Path(__file__).parent.parent / "data" / "energy_research" / "processed" / "examples_for_dibs.csv"

    if not data_file.exists():
        # 如果处理后的文件不存在，使用原始数据并进行快速处理
        print("警告: 处理后的数据文件不存在，使用原始数据")
        raw_file = Path(__file__).parent.parent / "data" / "energy_research" / "raw" / "energy_data_original.csv"

        if not raw_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {raw_file}")

        # 加载并过滤examples数据
        df = pd.read_csv(raw_file)

        # 过滤examples任务
        examples_repos = ['examples']
        df = df[df['repository'].isin(examples_repos)].copy()

        # 选择数值型列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols]

        # 移除全NaN列
        df = df.dropna(axis=1, how='all')

        # 移除缺失率>30%的列
        missing_per_col = df.isna().sum() / len(df)
        cols_to_keep = missing_per_col[missing_per_col <= 0.30].index.tolist()
        df = df[cols_to_keep]

        # 填充剩余缺失值
        df = df.fillna(df.mean())

        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns
        )

        return df_scaled

    return pd.read_csv(data_file)

def run_experiment(exp_config, data, output_dir):
    """
    运行单个实验

    参数:
        exp_config: 实验配置字典
        data: 测试数据
        output_dir: 输出目录

    返回:
        result_dict: 实验结果字典
    """
    exp_id = exp_config["id"]
    exp_name = exp_config["name"]

    print("\n" + "=" * 80)
    print(f"实验 {exp_id}: {exp_name}")
    print("=" * 80)

    # 打印参数
    print(f"参数配置:")
    print(f"  alpha_linear: {exp_config['alpha']}")
    print(f"  n_particles: {exp_config['n_particles']}")
    print(f"  beta_linear: {exp_config['beta']}")
    print(f"  tau: {exp_config['tau']}")
    print(f"  n_steps: {exp_config['n_steps']}")
    print(f"  n_grad_mc_samples: {exp_config['n_grad_mc_samples']}")

    # 创建learner
    learner = CausalGraphLearner(
        n_vars=len(data.columns),
        alpha=exp_config['alpha'],
        n_particles=exp_config['n_particles'],
        beta=exp_config['beta'],
        tau=exp_config['tau'],
        n_steps=exp_config['n_steps'],
        n_grad_mc_samples=exp_config['n_grad_mc_samples'],
        n_acyclicity_mc_samples=32,
        random_seed=42
    )

    # 运行DiBS
    start_time = time.time()
    try:
        causal_graph = learner.fit(data, verbose=True)
        elapsed_time = time.time() - start_time
        success = True
        error_msg = None
    except Exception as e:
        elapsed_time = time.time() - start_time
        causal_graph = None
        success = False
        error_msg = str(e)
        print(f"\n❌ 实验失败: {error_msg}")

    # 分析结果
    if success and causal_graph is not None:
        graph_min = float(causal_graph.min())
        graph_max = float(causal_graph.max())
        graph_mean = float(causal_graph.mean())
        graph_std = float(causal_graph.std())

        # 不同阈值下的边数
        edges_001 = int(np.sum(causal_graph > 0.01))
        edges_01 = int(np.sum(causal_graph > 0.1))
        edges_02 = int(np.sum(causal_graph > 0.2))
        edges_03 = int(np.sum(causal_graph > 0.3))
        edges_05 = int(np.sum(causal_graph > 0.5))

        # 判断是否成功
        if edges_001 > 0:
            status = "成功" if edges_03 > 0 else "部分成功"
        else:
            status = "失败"

        print(f"\n{'='*80}")
        print(f"实验结果:")
        print(f"  状态: {status}")
        print(f"  耗时: {elapsed_time/60:.2f}分钟")
        print(f"  图矩阵统计:")
        print(f"    Min: {graph_min:.6f}")
        print(f"    Max: {graph_max:.6f}")
        print(f"    Mean: {graph_mean:.6f}")
        print(f"    Std: {graph_std:.6f}")
        print(f"  边数统计:")
        print(f"    >0.01: {edges_001}")
        print(f"    >0.1: {edges_01}")
        print(f"    >0.2: {edges_02}")
        print(f"    >0.3: {edges_03}")
        print(f"    >0.5: {edges_05}")

        if edges_03 > 0:
            print(f"\n🎉🎉🎉 成功检测到{edges_03}条边！")
        elif edges_001 > 0:
            print(f"\n⚠️  检测到{edges_001}条弱边（需降低阈值）")
        else:
            print(f"\n❌ 图矩阵仍然全为0")

        # 保存图矩阵
        graph_file = output_dir / f"{exp_id}_graph.npy"
        np.save(graph_file, causal_graph)

    else:
        graph_min = graph_max = graph_mean = graph_std = None
        edges_001 = edges_01 = edges_02 = edges_03 = edges_05 = 0
        status = "错误"

    # 构建结果字典
    result = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "status": status,
        "success": success,
        "elapsed_time_minutes": elapsed_time / 60,
        "parameters": {
            "alpha": exp_config['alpha'],
            "n_particles": exp_config['n_particles'],
            "beta": exp_config['beta'],
            "tau": exp_config['tau'],
            "n_steps": exp_config['n_steps'],
            "n_grad_mc_samples": exp_config['n_grad_mc_samples']
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
            "threshold_0.2": edges_02,
            "threshold_0.3": edges_03,
            "threshold_0.5": edges_05
        },
        "error_message": error_msg
    }

    # 保存单个实验结果
    result_file = output_dir / f"{exp_id}_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    return result

def generate_summary_report(all_results, output_dir):
    """生成总结报告"""

    # 保存完整结果JSON
    full_results_file = output_dir / "all_results.json"
    with open(full_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # 生成Markdown报告
    report_file = output_dir / "PARAMETER_SWEEP_REPORT.md"

    with open(report_file, 'w') as f:
        f.write("# DiBS参数扫描测试报告\n\n")
        f.write(f"**测试日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**测试数量**: {len(all_results)}个实验\n\n")

        # 总体统计
        f.write("## 总体统计\n\n")
        successful = sum(1 for r in all_results if r['status'] in ['成功', '部分成功'])
        failed = sum(1 for r in all_results if r['status'] == '失败')
        errors = sum(1 for r in all_results if r['status'] == '错误')

        f.write(f"- 成功/部分成功: {successful}/{len(all_results)}\n")
        f.write(f"- 完全失败（0边）: {failed}/{len(all_results)}\n")
        f.write(f"- 运行错误: {errors}/{len(all_results)}\n")
        f.write(f"- 总耗时: {sum(r['elapsed_time_minutes'] for r in all_results):.1f}分钟\n\n")

        # 详细结果表格
        f.write("## 实验结果详情\n\n")
        f.write("| ID | 名称 | 状态 | 耗时(分) | 图Max | 边数(>0.3) | 边数(>0.01) | Alpha | Particles |\n")
        f.write("|-------|------|------|---------|--------|-----------|------------|-------|--------|\n")

        for r in all_results:
            graph_max = r['graph_stats']['max']
            graph_max_str = f"{graph_max:.6f}" if graph_max is not None else "N/A"
            f.write(f"| {r['exp_id']} | {r['exp_name'][:30]} | {r['status']} | "
                   f"{r['elapsed_time_minutes']:.1f} | {graph_max_str} | "
                   f"{r['edges']['threshold_0.3']} | {r['edges']['threshold_0.01']} | "
                   f"{r['parameters']['alpha']} | {r['parameters']['n_particles']} |\n")

        # 关键发现
        f.write("\n## 关键发现\n\n")

        # 找出最佳结果
        best_result = max(all_results, key=lambda r: r['edges']['threshold_0.01'] if r['success'] else 0)

        if best_result['edges']['threshold_0.01'] > 0:
            f.write("### ✅ 成功实验\n\n")
            f.write(f"**最佳实验**: {best_result['exp_id']} - {best_result['exp_name']}\n\n")
            f.write(f"- 检测到边数(>0.01): {best_result['edges']['threshold_0.01']}\n")
            f.write(f"- 检测到边数(>0.3): {best_result['edges']['threshold_0.3']}\n")
            f.write(f"- 图矩阵最大值: {best_result['graph_stats']['max']:.6f}\n")
            f.write(f"- 参数配置:\n")
            for k, v in best_result['parameters'].items():
                f.write(f"  - {k}: {v}\n")
        else:
            f.write("### ❌ 所有实验均失败\n\n")
            f.write("所有实验的图矩阵均为0或接近0，DiBS无法在能耗数据上产生因果边。\n\n")
            f.write("**结论**: 确认DiBS不适用于能耗数据集。建议采用回归分析等替代方法。\n")

        # 参数影响分析
        f.write("\n## 参数影响分析\n\n")
        f.write("### Alpha参数影响\n\n")
        alpha_results = {}
        for r in all_results:
            alpha = r['parameters']['alpha']
            if alpha not in alpha_results:
                alpha_results[alpha] = []
            alpha_results[alpha].append(r['edges']['threshold_0.01'])

        f.write("| Alpha | 平均边数(>0.01) | 最大边数 |\n")
        f.write("|-------|----------------|----------|\n")
        for alpha in sorted(alpha_results.keys()):
            avg_edges = np.mean(alpha_results[alpha])
            max_edges = max(alpha_results[alpha])
            f.write(f"| {alpha} | {avg_edges:.1f} | {max_edges} |\n")

        f.write("\n### N_particles参数影响\n\n")
        particles_results = {}
        for r in all_results:
            n_p = r['parameters']['n_particles']
            if n_p not in particles_results:
                particles_results[n_p] = []
            particles_results[n_p].append(r['edges']['threshold_0.01'])

        f.write("| N_particles | 平均边数(>0.01) | 最大边数 |\n")
        f.write("|-------------|----------------|----------|\n")
        for n_p in sorted(particles_results.keys()):
            avg_edges = np.mean(particles_results[n_p])
            max_edges = max(particles_results[n_p])
            f.write(f"| {n_p} | {avg_edges:.1f} | {max_edges} |\n")

        f.write("\n---\n\n")
        f.write("**报告生成时间**: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")

    print(f"\n✅ 总结报告已保存: {report_file}")
    return report_file

def main():
    """主函数"""
    print("="*80)
    print("DiBS参数扫描测试")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试实验数: {len(EXPERIMENTS)}")
    print("="*80)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "results" / "dibs_parameter_sweep" / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 加载数据
    print("\n加载测试数据...")
    try:
        data = load_test_data()
        print(f"✅ 数据加载成功: {len(data)}行 × {len(data.columns)}列")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 运行所有实验
    all_results = []

    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n进度: {i}/{len(EXPERIMENTS)}")

        try:
            result = run_experiment(exp_config, data, output_dir)
            all_results.append(result)

            # 实时决策：如果前3个实验都失败，可以提前终止
            if i == 3 and all(r['edges']['threshold_0.01'] == 0 for r in all_results):
                print("\n⚠️ 前3个实验均产生0边，继续测试可能意义不大")
                user_input = input("是否继续测试剩余实验? (y/n): ")
                if user_input.lower() != 'y':
                    print("用户选择提前终止测试")
                    break

        except KeyboardInterrupt:
            print("\n\n用户中断测试")
            break
        except Exception as e:
            print(f"\n❌ 实验运行异常: {e}")
            # 记录错误并继续下一个
            result = {
                "exp_id": exp_config["id"],
                "exp_name": exp_config["name"],
                "status": "错误",
                "success": False,
                "elapsed_time_minutes": 0,
                "parameters": exp_config,
                "graph_stats": {"min": None, "max": None, "mean": None, "std": None},
                "edges": {"threshold_0.01": 0, "threshold_0.1": 0, "threshold_0.2": 0,
                         "threshold_0.3": 0, "threshold_0.5": 0},
                "error_message": str(e)
            }
            all_results.append(result)

    # 生成总结报告
    print("\n" + "="*80)
    print("生成总结报告...")
    print("="*80)

    if all_results:
        report_file = generate_summary_report(all_results, output_dir)
        print(f"\n✅ 参数扫描测试完成！")
        print(f"结果保存在: {output_dir}")
        print(f"总结报告: {report_file}")
    else:
        print(f"\n❌ 没有完成任何实验")

    print("\n" + "="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
