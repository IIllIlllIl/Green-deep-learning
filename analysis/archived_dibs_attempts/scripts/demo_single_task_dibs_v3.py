#!/usr/bin/env python3
"""
单任务DiBS因果分析快速测试脚本 (v3 - 最终修复版本)

主要改进:
1. 数据标准化（StandardScaler）
2. Alpha参数: 0.1 → 0.9 (论文建议值)
3. 迭代步数: 3000 → 10000
4. 粒子数: 10 → 20
5. ⭐⭐⭐ 新增：移除缺失率>30%的列

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
    parser = argparse.ArgumentParser(description='单任务DiBS因果分析 (v3最终修复版本)')
    parser.add_argument('--task', type=str, required=True, help='任务名称（如mrt_oast）')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"DiBS因果分析 (v3最终修复版本): {args.task}")
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

    print(f"  原始数值型变量: {len(causal_data.columns)}个")
    print(f"  有效样本: {len(causal_data)}行")

    missing_rate_before = causal_data.isna().sum().sum() / (len(causal_data) * len(causal_data.columns))
    print(f"  原始缺失率: {missing_rate_before*100:.2f}%")

    # 2. 移除高缺失列 ⭐⭐⭐ 新增
    print("\n[步骤2] 移除高缺失列...")

    # 计算每列的缺失率
    missing_per_col = causal_data.isna().sum() / len(causal_data)

    # 显示缺失率统计
    print(f"  列缺失率分布:")
    for col, miss_rate in missing_per_col.items():
        if miss_rate > 0:
            print(f"    {col}: {miss_rate*100:.1f}%")

    # 保留缺失率<=30%的列
    MISSING_THRESHOLD = 0.30
    cols_to_keep = missing_per_col[missing_per_col <= MISSING_THRESHOLD].index.tolist()
    cols_removed = [col for col in causal_data.columns if col not in cols_to_keep]

    print(f"\n  阈值: 缺失率<={MISSING_THRESHOLD*100:.0f}%")
    print(f"  保留: {len(cols_to_keep)}列")
    print(f"  移除: {len(cols_removed)}列")
    if cols_removed:
        print(f"  移除的列: {cols_removed}")

    causal_data = causal_data[cols_to_keep]
    numeric_cols = cols_to_keep

    missing_rate_after = causal_data.isna().sum().sum() / (len(causal_data) * len(numeric_cols))
    print(f"  过滤后缺失率: {missing_rate_after*100:.2f}%")

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

    # 3. 数据标准化
    print("\n[步骤3] 数据标准化...")
    scaler = StandardScaler()
    causal_data_scaled = pd.DataFrame(
        scaler.fit_transform(causal_data),
        columns=causal_data.columns,
        index=causal_data.index
    )
    print(f"  ✅ 标准化完成 (mean=0, std=1)")

    # 验证标准化
    print(f"  验证: mean={causal_data_scaled.mean().mean():.6f}, std={causal_data_scaled.std().mean():.6f}")

    # 4. DiBS因果图学习
    print("\n[步骤4] DiBS因果图学习...")

    graph_file = output_dir / 'causal_graph.npy'
    edges_file = output_dir / 'causal_edges.pkl'

    # DiBS参数 ⭐⭐⭐ v3修改
    N_STEPS = 10000    # 从3000增加到10000
    ALPHA = 0.9        # 从0.5增加到0.9 (论文建议值)
    THRESHOLD = 0.3    # 保持不变

    print(f"  配置 (v3最终修复版本):")
    print(f"    n_steps: {N_STEPS} (从3000增加)")
    print(f"    alpha: {ALPHA} ⭐⭐⭐ (从0.5增加到0.9，论文建议值)")
    print(f"    threshold: {THRESHOLD}")
    print(f"    n_particles: 20 (从10增加)")
    print(f"  变量数: {len(numeric_cols)}")
    print(f"  样本数: {len(causal_data_scaled)}")
    print(f"  预计时间: 15-30分钟")

    learner = CausalGraphLearner(
        n_vars=len(numeric_cols),
        n_steps=N_STEPS,
        alpha=ALPHA,
        random_seed=42
    )

    print(f"\n  开始DiBS学习...")
    print(f"  注意: alpha=0.9可能产生更多边（更稠密的图）")
    dibs_start = time.time()

    # 使用标准化后的数据
    causal_graph = learner.fit(causal_data_scaled, verbose=args.verbose)

    dibs_time = time.time() - dibs_start
    print(f"\n  ✅ DiBS完成，耗时: {dibs_time/60:.1f}分钟")

    # 分析边
    edges = learner.get_edges(threshold=THRESHOLD)
    print(f"  检测到 {len(edges)} 条因果边 (threshold={THRESHOLD})")

    # 显示图矩阵统计
    print(f"\n  图矩阵统计:")
    print(f"    最大权重: {causal_graph.max():.6f}")
    print(f"    最小权重: {causal_graph.min():.6f}")
    print(f"    平均权重: {causal_graph.mean():.6f}")
    print(f"    非零元素数: {np.count_nonzero(causal_graph)}")

    # 显示不同阈值下的边数
    edges_01 = np.sum(causal_graph > 0.1)
    edges_02 = np.sum(causal_graph > 0.2)
    edges_03 = np.sum(causal_graph > 0.3)
    edges_04 = np.sum(causal_graph > 0.4)
    edges_05 = np.sum(causal_graph > 0.5)
    print(f"    权重>0.1的边数: {edges_01}")
    print(f"    权重>0.2的边数: {edges_02}")
    print(f"    权重>0.3的边数: {edges_03}")
    print(f"    权重>0.4的边数: {edges_04}")
    print(f"    权重>0.5的边数: {edges_05}")

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
                'version': 'v3',
                'standardized': True,
                'missing_filter': MISSING_THRESHOLD
            },
            'data_stats': {
                'n_samples': len(causal_data_scaled),
                'n_vars': len(numeric_cols),
                'missing_rate': missing_rate_after,
                'cols_removed': cols_removed
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f)

    print(f"  ✅ 结果已保存")

    # 显示关键边
    if len(edges) > 0:
        print(f"\n  🎉 成功检测到因果边！")
        print(f"  前10条最强因果边:")
        for i, (source, target, weight) in enumerate(edges[:10], 1):
            print(f"    {i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}")
    else:
        print(f"\n  ⚠️  未检测到置信度>{THRESHOLD}的因果边")
        if edges_01 > 0:
            print(f"  提示: 有{edges_01}条边权重>0.1，可以降低threshold查看")
        else:
            print(f"  严重警告: 图矩阵仍然全为0！")
            print(f"  可能原因: 数据中确实没有明显的因果关系")

    # 5. 生成报告
    report_file = output_dir / 'analysis_report.md'
    with open(report_file, 'w') as f:
        f.write(f"# {args.task} 因果分析报告 (v3最终修复版本)\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 数据概况\n\n")
        f.write(f"- 样本数: {len(causal_data)}\n")
        f.write(f"- 原始变量数: {len(df.select_dtypes(include=[np.number]).columns)}\n")
        f.write(f"- 过滤后变量数: {len(numeric_cols)}\n")
        f.write(f"- 移除列数: {len(cols_removed)}\n")
        if cols_removed:
            f.write(f"- 移除的列: {', '.join(cols_removed)}\n")
        f.write(f"- 原始缺失率: {missing_rate_before*100:.2f}%\n")
        f.write(f"- 过滤后缺失率: {missing_rate_after*100:.2f}%\n")
        f.write(f"- 数据预处理: 标准化 (StandardScaler)\n\n")
        f.write(f"## DiBS参数 (v3最终修复版本)\n\n")
        f.write(f"- n_steps: {N_STEPS}\n")
        f.write(f"- alpha: {ALPHA} (论文建议值)\n")
        f.write(f"- threshold: {THRESHOLD}\n")
        f.write(f"- n_particles: 20\n")
        f.write(f"- 缺失值过滤阈值: {MISSING_THRESHOLD*100:.0f}%\n\n")
        f.write(f"## DiBS结果\n\n")
        f.write(f"- 因果边数: {len(edges)}\n")
        f.write(f"- 运行时间: {dibs_time/60:.1f}分钟\n")
        f.write(f"- 图矩阵最大权重: {causal_graph.max():.6f}\n")
        f.write(f"- 图矩阵平均权重: {causal_graph.mean():.6f}\n")
        f.write(f"- 非零元素数: {np.count_nonzero(causal_graph)}\n\n")

        if edges:
            f.write(f"### 检测到的因果边\n\n")
            for i, (source, target, weight) in enumerate(edges, 1):
                f.write(f"{i}. {numeric_cols[source]} → {numeric_cols[target]}: {weight:.3f}\n")
        else:
            f.write(f"未检测到置信度>{THRESHOLD}的因果边。\n\n")
            f.write(f"### 权重分布\n\n")
            f.write(f"- 权重>0.1: {edges_01}条\n")
            f.write(f"- 权重>0.2: {edges_02}条\n")
            f.write(f"- 权重>0.3: {edges_03}条\n")
            f.write(f"- 权重>0.4: {edges_04}条\n")
            f.write(f"- 权重>0.5: {edges_05}条\n")

    print(f"\n  ✅ 报告已保存: {report_file}")

    # 总结
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print(f"因果边数: {len(edges)}")
    print(f"图矩阵最大权重: {causal_graph.max():.6f}")

    if len(edges) > 0:
        print(f"✅✅✅ v3修复成功！检测到{len(edges)}条因果边")
    elif edges_01 > 0:
        print(f"⚠️  阈值0.3未检测到边，但有{edges_01}条边权重>0.1")
        print(f"建议: 降低threshold重新分析")
    else:
        print(f"❌❌❌ v3修复失败！图矩阵仍然全为0")
        print(f"结论: DiBS可能不适用于此数据集")
    print("=" * 80)

if __name__ == '__main__':
    main()
