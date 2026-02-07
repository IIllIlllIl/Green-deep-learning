#!/usr/bin/env python3
"""
RQ1 数据分析脚本 - 超参数对能耗的影响
执行RQ1a（主效应）和RQ1b（调节效应）分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 色盲友好配色
COLOR_POSITIVE = '#E64B35'  # 红色 - 增加能耗
COLOR_NEGATIVE = '#4DBBD5'  # 蓝色 - 减少能耗
COLOR_NS = '#999999'        # 灰色 - 不显著

# 路径配置
BASE_DIR = Path('/home/green/energy_dl/nightly/analysis')
DATA_DIR = BASE_DIR / 'results/energy_research/data/global_std_dibs_ate'
OUTPUT_DIR = BASE_DIR / 'results/energy_research/rq_analysis'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# 能耗变量列表
ENERGY_VARIABLES = [
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules',
    'energy_cpu_total_joules', 'energy_gpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_min_watts', 'energy_gpu_max_watts'
]

# 组别名称映射
GROUP_NAMES = {
    'group1': 'Examples',
    'group2': 'VulBERTa',
    'group3': 'Person ReID',
    'group4': 'Bug Localization',
    'group5': 'MRT OAST',
    'group6': 'ResNet'
}


def load_all_data():
    """加载所有6组DiBS+ATE数据"""
    all_data = []

    for csv_file in sorted(DATA_DIR.glob('*.csv')):
        df = pd.read_csv(csv_file)
        # 从文件名提取组别
        group_id = csv_file.stem.split('_')[0]
        df['group'] = group_id
        df['group_name'] = GROUP_NAMES.get(group_id, group_id)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    print(f"加载数据: {len(combined)} 条边，来自 {len(all_data)} 组")
    return combined


def filter_rq1a_main_effects(df):
    """
    RQ1a: 过滤主效应边
    条件: source以hyperparam_开头且不含_x_is_parallel，target为energy变量，strength >= 0.3
    """
    mask = (
        df['source'].str.startswith('hyperparam_') &
        ~df['source'].str.contains('_x_is_parallel') &
        df['target'].isin(ENERGY_VARIABLES) &
        (df['strength'] >= 0.3)
    )
    result = df[mask].copy()
    print(f"RQ1a主效应: 筛选出 {len(result)} 条边")
    return result


def filter_rq1b_moderation_effects(df):
    """
    RQ1b: 过滤调节效应边
    条件: source包含_x_is_parallel，target为energy变量，strength >= 0.3
    """
    mask = (
        df['source'].str.contains('_x_is_parallel') &
        df['target'].isin(ENERGY_VARIABLES) &
        (df['strength'] >= 0.3)
    )
    result = df[mask].copy()
    print(f"RQ1b调节效应: 筛选出 {len(result)} 条边")
    return result


def create_table(df, table_name, filename):
    """生成并保存分析表格"""
    # 选择关键列并格式化
    table = df[['group_name', 'source', 'target', 'strength',
                'ate_global_std', 'ate_global_std_ci_lower',
                'ate_global_std_ci_upper', 'ate_global_std_is_significant']].copy()

    # 重命名列
    table.columns = ['Group', 'Source', 'Target', 'Edge Strength',
                     'ATE', 'CI Lower', 'CI Upper', 'Significant']

    # 按组别和效应大小排序
    table = table.sort_values(['Group', 'ATE'], ascending=[True, False])

    # 保存CSV
    output_path = TABLES_DIR / filename
    table.to_csv(output_path, index=False)
    print(f"保存表格: {output_path}")

    return table


def create_forest_plot(df, title, filename):
    """创建森林图"""
    if len(df) == 0:
        print(f"警告: {title} 无数据，跳过图表生成")
        return

    # 过滤有效ATE数据
    plot_df = df.dropna(subset=['ate_global_std']).copy()
    if len(plot_df) == 0:
        print(f"警告: {title} 无有效ATE数据")
        return

    # 创建标签
    plot_df['label'] = plot_df.apply(
        lambda x: f"{x['group_name']}: {x['source']} → {x['target']}", axis=1
    )

    # 按ATE排序
    plot_df = plot_df.sort_values('ate_global_std', ascending=True)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.4)))

    y_positions = range(len(plot_df))

    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ate = row['ate_global_std']
        ci_lower = row['ate_global_std_ci_lower']
        ci_upper = row['ate_global_std_ci_upper']
        is_sig = row['ate_global_std_is_significant']

        # 选择颜色
        if not is_sig:
            color = COLOR_NS
        elif ate > 0:
            color = COLOR_POSITIVE
        else:
            color = COLOR_NEGATIVE

        # 绘制置信区间
        if pd.notna(ci_lower) and pd.notna(ci_upper):
            ax.hlines(y=i, xmin=ci_lower, xmax=ci_upper, color=color, linewidth=2, alpha=0.7)

        # 绘制点估计
        ax.scatter(ate, i, color=color, s=100, zorder=5,
                   marker='o' if is_sig else 'x')

    # 添加零线
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # 设置标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df['label'])
    ax.set_xlabel('Average Treatment Effect (ATE)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLOR_POSITIVE, label='Increases Energy (sig.)'),
        Patch(facecolor=COLOR_NEGATIVE, label='Decreases Energy (sig.)'),
        Patch(facecolor=COLOR_NS, label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    # 保存图表
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"保存森林图: {output_path}")


def create_bar_chart(df, title, filename, figsize=(10, 6)):
    """创建柱状图（按需求文档规格）"""
    if len(df) == 0:
        print(f"警告: {title} 无数据，跳过图表生成")
        return

    # 过滤有效ATE数据
    plot_df = df.dropna(subset=['ate_global_std']).copy()
    if len(plot_df) == 0:
        print(f"警告: {title} 无有效ATE数据")
        return

    # 创建标签（简化版）
    plot_df['label'] = plot_df.apply(
        lambda x: f"{x['source'].replace('hyperparam_', '').replace('_x_is_parallel', '×P')}\n→{x['target'].replace('energy_', '')}", axis=1
    )

    # 按ATE排序
    plot_df = plot_df.sort_values('ate_global_std', ascending=False)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    x_positions = range(len(plot_df))
    bar_colors = []

    for _, row in plot_df.iterrows():
        ate = row['ate_global_std']
        is_sig = row['ate_global_std_is_significant']
        if not is_sig:
            bar_colors.append(COLOR_NS)
        elif ate > 0:
            bar_colors.append(COLOR_POSITIVE)
        else:
            bar_colors.append(COLOR_NEGATIVE)

    # 绘制柱状图
    bars = ax.bar(x_positions, plot_df['ate_global_std'], color=bar_colors, edgecolor='black', linewidth=0.5)

    # 添加误差线
    yerr_lower = plot_df['ate_global_std'] - plot_df['ate_global_std_ci_lower']
    yerr_upper = plot_df['ate_global_std_ci_upper'] - plot_df['ate_global_std']
    ax.errorbar(x_positions, plot_df['ate_global_std'],
                yerr=[yerr_lower.fillna(0), yerr_upper.fillna(0)],
                fmt='none', color='black', capsize=3, capthick=1, linewidth=1)

    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # 设置标签
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df['label'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Average Treatment Effect (ATE)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 添加组别标注
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        y_offset = 0.02 if row['ate_global_std'] >= 0 else -0.02
        va = 'bottom' if row['ate_global_std'] >= 0 else 'top'
        ax.annotate(row['group_name'], (i, row['ate_global_std'] + y_offset),
                    ha='center', va=va, fontsize=8, color='gray')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_POSITIVE, edgecolor='black', label='Increases Energy (sig.)'),
        Patch(facecolor=COLOR_NEGATIVE, edgecolor='black', label='Decreases Energy (sig.)'),
        Patch(facecolor=COLOR_NS, edgecolor='black', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)

    plt.tight_layout()

    # 保存图表
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"保存柱状图: {output_path}")


def print_summary(df_main, df_mod):
    """输出分析摘要"""
    print("\n" + "=" * 60)
    print("RQ1 分析摘要")
    print("=" * 60)

    # RQ1a摘要
    print("\n【RQ1a: 主效应分析】")
    print(f"  总边数: {len(df_main)}")
    if len(df_main) > 0:
        sig_main = df_main[df_main['ate_global_std_is_significant'] == True]
        print(f"  显著边数: {len(sig_main)}")

        if len(sig_main) > 0:
            pos = sig_main[sig_main['ate_global_std'] > 0]
            neg = sig_main[sig_main['ate_global_std'] < 0]
            print(f"  - 增加能耗: {len(pos)} 条")
            print(f"  - 减少能耗: {len(neg)} 条")

            # 列出具体边
            print("\n  主要发现:")
            for _, row in sig_main.iterrows():
                direction = "↑" if row['ate_global_std'] > 0 else "↓"
                print(f"    {row['group_name']}: {row['source']} → {row['target']} "
                      f"(ATE={row['ate_global_std']:.4f}) {direction}")

    # RQ1b摘要
    print("\n【RQ1b: 调节效应分析】")
    print(f"  总边数: {len(df_mod)}")
    if len(df_mod) > 0:
        sig_mod = df_mod[df_mod['ate_global_std_is_significant'] == True]
        print(f"  显著边数: {len(sig_mod)}")

        if len(sig_mod) > 0:
            pos = sig_mod[sig_mod['ate_global_std'] > 0]
            neg = sig_mod[sig_mod['ate_global_std'] < 0]
            print(f"  - 增加能耗: {len(pos)} 条")
            print(f"  - 减少能耗: {len(neg)} 条")

            # 列出具体边
            print("\n  主要发现:")
            for _, row in sig_mod.iterrows():
                direction = "↑" if row['ate_global_std'] > 0 else "↓"
                print(f"    {row['group_name']}: {row['source']} → {row['target']} "
                      f"(ATE={row['ate_global_std']:.4f}) {direction}")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("RQ1 数据分析 - 超参数对能耗的影响")
    print("=" * 60)

    # 确保输出目录存在
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("\n[1/4] 加载DiBS+ATE数据...")
    all_data = load_all_data()

    # 2. RQ1a: 主效应分析
    print("\n[2/4] RQ1a主效应分析...")
    df_main = filter_rq1a_main_effects(all_data)
    if len(df_main) > 0:
        table1 = create_table(df_main, "表1: 超参数对能耗的主效应",
                              "table1_rq1a_main_effects.csv")
        # 生成柱状图（需求文档规格：10×6 inch）
        create_bar_chart(df_main,
                         "RQ1a: Hyperparameter Main Effects on Energy Consumption",
                         "figure1_rq1a_main_effects_bar.pdf",
                         figsize=(10, 6))
        # 生成森林图（补充图表）
        create_forest_plot(df_main,
                          "RQ1a: Hyperparameter Main Effects on Energy Consumption",
                          "figure1_rq1a_main_effects_forest.pdf")

    # 3. RQ1b: 调节效应分析
    print("\n[3/4] RQ1b调节效应分析...")
    df_mod = filter_rq1b_moderation_effects(all_data)
    if len(df_mod) > 0:
        table2 = create_table(df_mod, "表2: 并行化对超参数-能耗关系的调节效应",
                              "table2_rq1b_moderation_effects.csv")
        # 生成柱状图（需求文档规格：12×6 inch）
        create_bar_chart(df_mod,
                         "RQ1b: Parallelization Moderation Effects on Energy",
                         "figure2_rq1b_moderation_effects_bar.pdf",
                         figsize=(12, 6))
        # 生成森林图（补充图表）
        create_forest_plot(df_mod,
                          "RQ1b: Parallelization Moderation Effects on Energy",
                          "figure2_rq1b_moderation_effects_forest.pdf")

    # 4. 输出摘要
    print("\n[4/4] 生成分析摘要...")
    print_summary(df_main, df_mod)

    print("\n分析完成!")
    print(f"表格保存至: {TABLES_DIR}")
    print(f"图表保存至: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
