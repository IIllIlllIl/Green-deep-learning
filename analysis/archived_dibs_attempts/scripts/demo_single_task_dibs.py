#!/usr/bin/env python3
"""
单任务DiBS因果分析快速测试脚本
用途: 快速测试单个任务组的DiBS分析
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

# 设置随机种子
np.random.seed(42)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.causal_discovery import CausalGraphLearner

def main():
    parser = argparse.ArgumentParser(description='单任务DiBS因果分析')
    parser.add_argument('--task', type=str, required=True, help='任务名称（如mrt_oast）')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"DiBS因果分析: {args.task}")
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

    # 2. DiBS因果图学习
    print("\n[步骤2] DiBS因果图学习...")

    graph_file = output_dir / 'causal_graph.npy'
    edges_file = output_dir / 'causal_edges.pkl'

    # DiBS参数
    N_STEPS = 3000
    ALPHA = 0.1
    THRESHOLD = 0.3

    print(f"  配置: n_steps={N_STEPS}, alpha={ALPHA}, threshold={THRESHOLD}")
    print(f"  变量数: {len(numeric_cols)}")
    print(f"  样本数: {len(causal_data)}")
    print(f"  预计时间: 15-60分钟")

    learner = CausalGraphLearner(
        n_vars=len(numeric_cols),
        n_steps=N_STEPS,
        alpha=ALPHA,
        random_seed=42
    )

    print(f"\n  开始DiBS学习...")
    dibs_start = time.time()

    causal_graph = learner.fit(causal_data, verbose=args.verbose)

    dibs_time = time.time() - dibs_start
    print(f"\n  ✅ DiBS完成，耗时: {dibs_time/60:.1f}分钟")

    # 分析边
    edges = learner.get_edges(threshold=THRESHOLD)
    print(f"  检测到 {len(edges)} 条因果边")

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
                'threshold': THRESHOLD
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f)

    print(f"  ✅ 结果已保存")

    # 显示关键边
    if len(edges) > 0:
        print(f"\n  前10条最强因果边:")
        for i, (source, target, weight) in enumerate(edges[:10], 1):
            print(f"    {i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}")
    else:
        print(f"  ⚠️  未检测到置信度>{THRESHOLD}的因果边")

    # 3. 生成报告
    report_file = output_dir / 'analysis_report.md'
    with open(report_file, 'w') as f:
        f.write(f"# {args.task} 因果分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 数据概况\n\n")
        f.write(f"- 样本数: {len(causal_data)}\n")
        f.write(f"- 变量数: {len(numeric_cols)}\n")
        f.write(f"- 缺失率: {missing_rate*100:.2f}%\n\n")
        f.write(f"## DiBS结果\n\n")
        f.write(f"- 因果边数: {len(edges)}\n")
        f.write(f"- 运行时间: {dibs_time/60:.1f}分钟\n\n")
        if edges:
            f.write(f"### 前10条因果边\n\n")
            for i, (source, target, weight) in enumerate(edges[:10], 1):
                f.write(f"{i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}\n")

    print(f"\n  ✅ 报告已保存: {report_file}")

    # 总结
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print(f"因果边数: {len(edges)}")
    print("=" * 80)

if __name__ == '__main__':
    main()
