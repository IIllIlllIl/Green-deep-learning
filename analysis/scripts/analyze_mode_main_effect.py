#!/usr/bin/env python3
"""
步骤1: 分析并行/非并行模式的主效应

目的：
快速判断模式对能耗的影响是否显著，决定是否需要深入的分层DiBS分析。

创建日期: 2026-01-06
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 设置
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# 路径
DATA_FILE = Path("/home/green/energy_dl/nightly/data/raw_data.csv")
OUTPUT_DIR = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/mode_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """加载数据"""
    print("加载数据...")
    df = pd.read_csv(DATA_FILE)

    # 创建is_parallel列
    df['is_parallel'] = (df['mode'] == 'parallel').astype(int)

    print(f"  总样本数: {len(df)}")
    print(f"  并行模式: {df['is_parallel'].sum()} ({df['is_parallel'].sum()/len(df)*100:.1f}%)")
    print(f"  非并行模式: {(~df['is_parallel'].astype(bool)).sum()} ({(~df['is_parallel'].astype(bool)).sum()/len(df)*100:.1f}%)")

    return df


def analyze_mode_main_effect(df):
    """分析模式的主效应"""

    print("\n" + "="*80)
    print("模式主效应分析（t检验）")
    print("="*80)

    # 能耗指标
    energy_metrics = [
        'energy_gpu_total_joules',
        'energy_gpu_avg_watts',
        'energy_gpu_max_watts',
        'energy_gpu_min_watts',
        'energy_cpu_total_joules',
    ]

    results = []

    for metric in energy_metrics:
        # 分组数据
        parallel = df[df['is_parallel'] == 1][metric].dropna()
        non_parallel = df[df['is_parallel'] == 0][metric].dropna()

        if len(parallel) < 10 or len(non_parallel) < 10:
            print(f"\n❌ {metric}: 样本量不足")
            continue

        # t检验
        t_stat, p_value = stats.ttest_ind(parallel, non_parallel)

        # Cohen's d (效应量)
        pooled_std = np.sqrt(((len(parallel)-1)*parallel.std()**2 + (len(non_parallel)-1)*non_parallel.std()**2) / (len(parallel)+len(non_parallel)-2))
        cohens_d = (parallel.mean() - non_parallel.mean()) / pooled_std

        # 结果
        result = {
            'metric': metric,
            'parallel_mean': float(parallel.mean()),
            'parallel_std': float(parallel.std()),
            'parallel_n': int(len(parallel)),
            'non_parallel_mean': float(non_parallel.mean()),
            'non_parallel_std': float(non_parallel.std()),
            'non_parallel_n': int(len(non_parallel)),
            'mean_diff': float(parallel.mean() - non_parallel.mean()),
            'mean_diff_pct': float((parallel.mean() - non_parallel.mean()) / non_parallel.mean() * 100),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'is_significant': p_value < 0.05
        }

        results.append(result)

        # 打印结果
        print(f"\n{metric}:")
        print(f"  并行模式: {parallel.mean():.2f} ± {parallel.std():.2f} (n={len(parallel)})")
        print(f"  非并行模式: {non_parallel.mean():.2f} ± {non_parallel.std():.2f} (n={len(non_parallel)})")
        print(f"  平均差异: {result['mean_diff']:.2f} ({result['mean_diff_pct']:.1f}%)")
        print(f"  t检验: t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        print(f"  效应量: Cohen's d={cohens_d:.3f} ({'大' if abs(cohens_d) > 0.8 else '中' if abs(cohens_d) > 0.5 else '小' if abs(cohens_d) > 0.2 else '极小'})")
        print(f"  显著性: {'✅ 显著' if p_value < 0.05 else '❌ 不显著'}")

    return results


def analyze_mode_by_group(df):
    """按任务组分析模式效应"""

    print("\n" + "="*80)
    print("按仓库分析模式效应")
    print("="*80)

    # 按repository分组
    repos = ['examples', 'VulBERTa', 'Person_reID_baseline_pytorch',
             'pytorch_resnet_cifar10', 'bug-localization-by-dnn-and-rvsm', 'MRT-OAST']
    energy_metric = 'energy_gpu_total_joules'

    results = []

    for repo in repos:
        repo_df = df[df['repository'] == repo]

        if len(repo_df) == 0:
            continue

        parallel = repo_df[repo_df['is_parallel'] == 1][energy_metric].dropna()
        non_parallel = repo_df[repo_df['is_parallel'] == 0][energy_metric].dropna()

        if len(parallel) < 10 or len(non_parallel) < 10:
            print(f"\n{repo}: 样本量不足 (并行={len(parallel)}, 非并行={len(non_parallel)})")
            continue

        t_stat, p_value = stats.ttest_ind(parallel, non_parallel)

        result = {
            'repo': repo,
            'parallel_n': int(len(parallel)),
            'non_parallel_n': int(len(non_parallel)),
            'parallel_mean': float(parallel.mean()),
            'non_parallel_mean': float(non_parallel.mean()),
            'mean_diff_pct': float((parallel.mean() - non_parallel.mean()) / non_parallel.mean() * 100),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05
        }

        results.append(result)

        print(f"\n{repo} ({energy_metric}):")
        print(f"  并行: {parallel.mean():.2f} (n={len(parallel)})")
        print(f"  非并行: {non_parallel.mean():.2f} (n={len(non_parallel)})")
        print(f"  差异: {result['mean_diff_pct']:.1f}%")
        print(f"  p值: {p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")

    return results


def visualize_mode_effect(df):
    """可视化模式效应"""

    print("\n生成可视化...")

    energy_metrics = [
        ('energy_gpu_total_joules', 'GPU Total Energy (J)'),
        ('energy_gpu_avg_watts', 'GPU Avg Power (W)'),
        ('energy_gpu_max_watts', 'GPU Max Power (W)'),
    ]

    # 图1: 箱线图对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (metric, label) in enumerate(energy_metrics):
        ax = axes[i]

        # 准备数据
        plot_df = df[['is_parallel', metric]].dropna()
        plot_df['Mode'] = plot_df['is_parallel'].map({1: 'Parallel', 0: 'Non-Parallel'})

        # 箱线图
        sns.boxplot(x='Mode', y=metric, data=plot_df, ax=ax)

        # 添加显著性标记
        parallel = plot_df[plot_df['is_parallel'] == 1][metric]
        non_parallel = plot_df[plot_df['is_parallel'] == 0][metric]
        _, p_value = stats.ttest_ind(parallel, non_parallel)

        if p_value < 0.05:
            # 添加显著性星号
            y_max = plot_df[metric].max()
            ax.text(0.5, y_max * 1.05, '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*',
                   ha='center', fontsize=16)

        ax.set_title(f'{label}\n(p={p_value:.4f})')
        ax.set_xlabel('Training Mode')
        ax.set_ylabel(label)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mode_effect_boxplots.png', dpi=300)
    print(f"  ✅ 保存: mode_effect_boxplots.png")
    plt.close()

    # 图2: 分组对比
    repos = ['examples', 'VulBERTa', 'Person_reID_baseline_pytorch']
    metric = 'energy_gpu_total_joules'

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, repo in enumerate(repos):
        repo_df = df[df['repository'] == repo]

        if len(repo_df) == 0:
            continue

        ax = axes[i]

        plot_df = repo_df[['is_parallel', metric]].dropna()
        plot_df['Mode'] = plot_df['is_parallel'].map({1: 'Parallel', 0: 'Non-Parallel'})

        sns.boxplot(x='Mode', y=metric, data=plot_df, ax=ax)

        # 显著性
        parallel = plot_df[plot_df['is_parallel'] == 1][metric]
        non_parallel = plot_df[plot_df['is_parallel'] == 0][metric]

        if len(parallel) >= 10 and len(non_parallel) >= 10:
            _, p_value = stats.ttest_ind(parallel, non_parallel)

            if p_value < 0.05:
                y_max = plot_df[metric].max()
                ax.text(0.5, y_max * 1.05, '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*',
                       ha='center', fontsize=16)

            ax.set_title(f'{repo}\n(p={p_value:.4f})')
        else:
            ax.set_title(f'{repo}\n(样本量不足)')

        ax.set_xlabel('Training Mode')
        ax.set_ylabel('GPU Total Energy (J)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mode_effect_by_repo.png', dpi=300)
    print(f"  ✅ 保存: mode_effect_by_repo.png")
    plt.close()


def generate_report(overall_results, group_results):
    """生成报告"""

    report_file = OUTPUT_DIR / "MODE_MAIN_EFFECT_REPORT.md"

    with open(report_file, 'w') as f:
        f.write("# 并行/非并行模式主效应分析报告\n\n")
        f.write(f"**分析日期**: 2026-01-06\n")
        f.write(f"**目的**: 判断是否需要进行分层DiBS分析\n\n")

        f.write("---\n\n")

        # 总体效应
        f.write("## 📊 总体效应分析\n\n")

        significant_count = sum(1 for r in overall_results if r['is_significant'])

        f.write(f"**显著能耗指标**: {significant_count}/{len(overall_results)}\n\n")

        # 表格
        f.write("| 能耗指标 | 并行模式 | 非并行模式 | 差异 | p值 | Cohen's d | 显著性 |\n")
        f.write("|---------|---------|-----------|------|-----|-----------|--------|\n")

        for r in overall_results:
            f.write(f"| {r['metric']} | {r['parallel_mean']:.2f}±{r['parallel_std']:.2f} | "
                   f"{r['non_parallel_mean']:.2f}±{r['non_parallel_std']:.2f} | "
                   f"{r['mean_diff_pct']:+.1f}% | {r['p_value']:.4f} | {r['cohens_d']:.3f} | "
                   f"{'✅' if r['is_significant'] else '❌'} |\n")

        f.write("\n")

        # 按组分析
        if group_results:
            f.write("## 📋 按仓库分析\n\n")

            f.write("| 仓库 | 并行(n) | 非并行(n) | 并行均值 | 非并行均值 | 差异% | p值 | 显著 |\n")
            f.write("|--------|---------|----------|---------|-----------|-------|-----|------|\n")

            for r in group_results:
                f.write(f"| {r['repo']} | {r['parallel_n']} | {r['non_parallel_n']} | "
                       f"{r['parallel_mean']:.2f} | {r['non_parallel_mean']:.2f} | "
                       f"{r['mean_diff_pct']:+.1f}% | {r['p_value']:.4f} | "
                       f"{'✅' if r['is_significant'] else '❌'} |\n")

            f.write("\n")

        # 决策建议
        f.write("## 💡 决策建议\n\n")

        if significant_count >= len(overall_results) / 2:
            f.write("### ✅ 建议进行完整的分层DiBS分析\n\n")
            f.write(f"**理由**: {significant_count}/{len(overall_results)}个能耗指标显示模式有显著影响。\n\n")
            f.write("**后续步骤**:\n")
            f.write("1. 按模式准备DiBS训练数据\n")
            f.write("2. 分别运行并行和非并行模式的DiBS\n")
            f.write("3. 比较两个因果图的差异\n")
            f.write("4. 运行交互效应回归分析\n")
            f.write("5. 运行分层中介分析\n\n")
        else:
            f.write("### ⚠️ 建议进行简化分析\n\n")
            f.write(f"**理由**: 仅{significant_count}/{len(overall_results)}个能耗指标显示模式有显著影响。\n\n")
            f.write("**后续步骤**:\n")
            f.write("1. 跳过分层DiBS（样本量减半，可能不稳定）\n")
            f.write("2. 只运行交互效应回归分析（验证当前发现的边）\n")
            f.write("3. 在报告中说明模式影响较小\n\n")

        # 关键发现
        f.write("## 🔍 关键发现\n\n")

        # 找出效应量最大的指标
        max_effect = max(overall_results, key=lambda x: abs(x['cohens_d']))

        f.write(f"**效应量最大的指标**: {max_effect['metric']}\n")
        f.write(f"- Cohen's d = {max_effect['cohens_d']:.3f}\n")
        f.write(f"- 平均差异 = {max_effect['mean_diff_pct']:+.1f}%\n")
        f.write(f"- p值 = {max_effect['p_value']:.4f}\n\n")

        # 方向
        positive_effects = [r for r in overall_results if r['mean_diff'] > 0 and r['is_significant']]
        negative_effects = [r for r in overall_results if r['mean_diff'] < 0 and r['is_significant']]

        if positive_effects:
            f.write(f"**并行模式能耗更高的指标**: {len(positive_effects)}个\n")
            for r in positive_effects:
                f.write(f"- {r['metric']}: +{r['mean_diff_pct']:.1f}% (p={r['p_value']:.4f})\n")
            f.write("\n")

        if negative_effects:
            f.write(f"**并行模式能耗更低的指标**: {len(negative_effects)}个\n")
            for r in negative_effects:
                f.write(f"- {r['metric']}: {r['mean_diff_pct']:.1f}% (p={r['p_value']:.4f})\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("**报告生成时间**: 2026-01-06\n")
        f.write("**下一步**: 根据决策建议选择分析策略\n")

    print(f"\n✅ 报告已保存: {report_file}")

    return report_file


def save_results_json(overall_results, group_results):
    """保存JSON结果"""

    json_file = OUTPUT_DIR / "mode_main_effect_results.json"

    # 转换numpy类型为Python原生类型
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    results = {
        'overall': convert_to_python_types(overall_results),
        'by_group': convert_to_python_types(group_results)
    }

    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ JSON结果已保存: {json_file}")


def main():
    """主函数"""

    print("="*80)
    print("步骤1: 并行/非并行模式主效应分析")
    print("="*80)

    # 加载数据
    df = load_data()

    # 总体效应分析
    overall_results = analyze_mode_main_effect(df)

    # 按组分析
    group_results = analyze_mode_by_group(df)

    # 可视化
    visualize_mode_effect(df)

    # 保存结果
    save_results_json(overall_results, group_results)

    # 生成报告
    report_file = generate_report(overall_results, group_results)

    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  报告文件: {report_file}")
    print("="*80)


if __name__ == "__main__":
    main()
