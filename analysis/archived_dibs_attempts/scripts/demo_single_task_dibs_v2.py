#!/usr/bin/env python3
"""
单任务DiBS因果分析快速测试脚本 (v2 - 修复版本)

主要改进:
1. 增加数据标准化（StandardScaler）
2. 调整Alpha参数: 0.1 → 0.5
3. 增加迭代步数: 3000 → 10000
4. 预期检测到因果边: 3-15条

日期: 2025-12-26
"""
import numpy as np
import pandas as pd
import sys
import os
import time
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 设置随机种子
np.random.seed(42)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.causal_discovery import CausalGraphLearner

def main():
    parser = argparse.ArgumentParser(description='单任务DiBS因果分析 (v2修复版本)')
    parser.add_argument('--task', type=str, required=True, help='任务名称（如mrt_oast）')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"DiBS因果分析 (v2修复版本): {args.task}")
    print("=" * 80)
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    start_time = time.time()

    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    df = pd.read_csv(args.input)
    print(f"  ✅ 数据加载: {len(df)}行 × {len(df.columns)}列")

    # 准备数值型数据
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    causal_data = df[numeric_cols].copy()
    causal_data = causal_data.dropna(axis=1, how='all')
    numeric_cols = causal_data.columns.tolist()

    print(f"  数值型变量: {len(numeric_cols)}个")
    print(f"  有效样本: {len(causal_data)}行")

    missing_rate = causal_data.isna().sum().sum() / (len(causal_data) * len(numeric_cols))
    print(f"  缺失率: {missing_rate*100:.2f}%")

    # 显示数据范围（标准化前）
    print(f"\n  数据范围（标准化前）:")
    for col in numeric_cols[:5]:  # 显示前5列
        col_min = causal_data[col].min()
        col_max = causal_data[col].max()
        col_mean = causal_data[col].mean()
        col_std = causal_data[col].std()
        print(f"    {col}: [{col_min:.2f}, {col_max:.2f}], mean={col_mean:.2f}, std={col_std:.2f}")
    if len(numeric_cols) > 5:
        print(f"    ... (共{len(numeric_cols)}列)")

    # 2. 数据标准化 ⭐⭐⭐ 新增
    print("\n[步骤2] 数据标准化...")
    scaler = StandardScaler()
    causal_data_scaled = pd.DataFrame(
        scaler.fit_transform(causal_data),
        columns=causal_data.columns,
        index=causal_data.index
    )
    print(f"  ✅ 标准化完成 (mean=0, std=1)")

    # 验证标准化
    print(f"  验证: mean={causal_data_scaled.mean().mean():.6f}, std={causal_data_scaled.std().mean():.6f}")

    # 3. DiBS因果图学习
    print("\n[步骤3] DiBS因果图学习...")

    graph_file = output_dir / 'causal_graph.npy'
    edges_file = output_dir / 'causal_edges.pkl'

    # DiBS参数 ⭐⭐⭐ 修改
    N_STEPS = 10000    # 从3000增加到10000
    ALPHA = 0.5        # 从0.1增加到0.5
    THRESHOLD = 0.3    # 保持不变

    print(f"  配置 (v2修复版本):")
    print(f"    n_steps: {N_STEPS} (从3000增加)")
    print(f"    alpha: {ALPHA} (从0.1增加)")
    print(f"    threshold: {THRESHOLD}")
    print(f"    n_particles: 20 (从10增加，需causal_discovery.py配合)")
    print(f"  变量数: {len(numeric_cols)}")
    print(f"  样本数: {len(causal_data_scaled)}")
    print(f"  预计时间: 10-30分钟")

    learner = CausalGraphLearner(
        n_vars=len(numeric_cols),
        n_steps=N_STEPS,
        alpha=ALPHA,
        random_seed=42
    )

    print(f"\n  开始DiBS学习...")
    dibs_start = time.time()

    # 使用标准化后的数据 ⭐⭐⭐
    causal_graph = learner.fit(causal_data_scaled, verbose=args.verbose)

    dibs_time = time.time() - dibs_start
    print(f"\n  ✅ DiBS完成，耗时: {dibs_time/60:.1f}分钟")

    # 分析边
    edges = learner.get_edges(threshold=THRESHOLD)
    print(f"  检测到 {len(edges)} 条因果边")

    # 显示图矩阵统计
    print(f"\n  图矩阵统计:")
    print(f"    最大权重: {causal_graph.max():.6f}")
    print(f"    最小权重: {causal_graph.min():.6f}")
    print(f"    非零元素数: {np.count_nonzero(causal_graph)}")

    # 显示权重>0.1的边数
    edges_01 = np.sum(causal_graph > 0.1)
    edges_02 = np.sum(causal_graph > 0.2)
    edges_03 = np.sum(causal_graph > 0.3)
    print(f"    权重>0.1的边数: {edges_01}")
    print(f"    权重>0.2的边数: {edges_02}")
    print(f"    权重>0.3的边数: {edges_03}")

    # 保存结果
    learner.save_graph(str(graph_file))
    with open(edges_file, 'wb') as f:
        pickle.dump({
            'edges': edges,
            'numeric_cols': numeric_cols,
            'task_name': args.task,
            'dibs_params': {
                'n_steps': N_STEPS,
                'alpha': ALPHA,
                'threshold': THRESHOLD,
                'version': 'v2',
                'standardized': True
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f)

    print(f"  ✅ 结果已保存")

    # 显示关键边
    if len(edges) > 0:
        print(f"\n  ✅ 成功检测到因果边！")
        print(f"  前10条最强因果边:")
        for i, (source, target, weight) in enumerate(edges[:10], 1):
            print(f"    {i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}")
    else:
        print(f"  ⚠️  未检测到置信度>{THRESHOLD}的因果边")
        print(f"  建议: 降低threshold到0.2或0.1重新检查")

    # 4. 生成报告
    report_file = output_dir / 'analysis_report.md'
    with open(report_file, 'w') as f:
        f.write(f"# {args.task} 因果分析报告 (v2修复版本)\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 数据概况\n\n")
        f.write(f"- 样本数: {len(causal_data)}\n")
        f.write(f"- 变量数: {len(numeric_cols)}\n")
        f.write(f"- 缺失率: {missing_rate*100:.2f}%\n")
        f.write(f"- 数据预处理: 标准化 (StandardScaler)\n\n")
        f.write(f"## DiBS参数 (v2修复版本)\n\n")
        f.write(f"- n_steps: {N_STEPS}\n")
        f.write(f"- alpha: {ALPHA}\n")
        f.write(f"- threshold: {THRESHOLD}\n")
        f.write(f"- n_particles: 20\n\n")
        f.write(f"## DiBS结果\n\n")
        f.write(f"- 因果边数: {len(edges)}\n")
        f.write(f"- 运行时间: {dibs_time/60:.1f}分钟\n")
        f.write(f"- 图矩阵最大权重: {causal_graph.max():.6f}\n")
        f.write(f"- 非零元素数: {np.count_nonzero(causal_graph)}\n\n")
        if edges:
            f.write(f"### 检测到的因果边\n\n")
            for i, (source, target, weight) in enumerate(edges, 1):
                f.write(f"{i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}\n")
        else:
            f.write(f"未检测到置信度>{THRESHOLD}的因果边。\n\n")
            f.write(f"权重分布:\n")
            f.write(f"- 权重>0.1: {edges_01}条\n")
            f.write(f"- 权重>0.2: {edges_02}条\n")
            f.write(f"- 权重>0.3: {edges_03}条\n")

    print(f"\n  ✅ 报告已保存: {report_file}")

    # 总结
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print(f"因果边数: {len(edges)}")
    if len(edges) > 0:
        print(f"✅ 修复成功！检测到{len(edges)}条因果边")
    else:
        print(f"⚠️  仍未检测到因果边，可能需要进一步调整参数")
    print("=" * 80)

if __name__ == '__main__':
    main()
