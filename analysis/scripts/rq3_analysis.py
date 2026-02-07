#!/usr/bin/env python3
"""
RQ3 数据分析脚本 - 能耗与性能权衡
检测并分析超参数对能耗和性能产生相反影响的权衡关系

数据来源: tradeoff_detection_global_std/tradeoff_detailed_global_std.csv
输出:
- 表5: 权衡关系汇总
- 表6: 能耗vs性能权衡详情
- 图5: 权衡散点图 (8×6 inch)
- 图6: 6组权衡数量分布 (10×6 inch)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 色盲友好配色
COLORS = {
    'hyperparam': '#E64B35',    # 红色
    'interaction': '#4DBBD5',   # 蓝色
    'other': '#999999'          # 灰色
}

# 路径配置
BASE_DIR = Path('/home/green/energy_dl/nightly/analysis')
TRADEOFF_FILE = BASE_DIR / 'results/energy_research/tradeoff_detection_global_std/tradeoff_detailed_global_std.csv'
OUTPUT_DIR = BASE_DIR / 'results/energy_research/rq_analysis'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# 变量分类
ENERGY_VARIABLES = [
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules',
    'energy_cpu_total_joules', 'energy_gpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_min_watts', 'energy_gpu_max_watts'
]

PERFORMANCE_VARIABLES = [
    'perf_test_accuracy', 'perf_map', 'perf_rank1', 'perf_rank5',
    'perf_precision', 'perf_recall', 'perf_eval_samples_per_second',
    'perf_top1_accuracy', 'perf_top5_accuracy', 'perf_top10_accuracy',
    'perf_top20_accuracy', 'perf_final_training_loss', 'perf_eval_loss'
]

# 组别名称映射
GROUP_NAMES = {
    'group1_examples': 'Examples',
    'group2_vulberta': 'VulBERTa',
    'group3_person_reid': 'Person ReID',
    'group4_bug_localization': 'Bug Localization',
    'group5_mrt_oast': 'MRT OAST',
    'group6_resnet': 'ResNet'
}


def get_intervention_type(intervention):
    """获取干预变量类型"""
    if '_x_is_parallel' in intervention:
        return 'interaction'
    elif intervention.startswith('hyperparam_'):
        return 'hyperparam'
    else:
        return 'other'


def get_tradeoff_type(metric1, metric2):
    """获取权衡类型"""
    is_energy1 = metric1 in ENERGY_VARIABLES
    is_energy2 = metric2 in ENERGY_VARIABLES
    is_perf1 = metric1 in PERFORMANCE_VARIABLES or metric1.startswith('perf_')
    is_perf2 = metric2 in PERFORMANCE_VARIABLES or metric2.startswith('perf_')

    if (is_energy1 and is_perf2) or (is_perf1 and is_energy2):
        return 'energy_vs_performance'
    elif is_energy1 and is_energy2:
        return 'energy_vs_energy'
    elif is_perf1 and is_perf2:
        return 'performance_vs_performance'
    else:
        return 'other'


def load_tradeoff_data():
    """加载权衡检测数据"""
    df = pd.read_csv(TRADEOFF_FILE)
    print(f"加载权衡数据: {len(df)} 条记录")

    # 添加分类列
    df['intervention_type'] = df['intervention'].apply(get_intervention_type)
    df['tradeoff_type'] = df.apply(lambda x: get_tradeoff_type(x['metric1'], x['metric2']), axis=1)
    df['group_name'] = df['group_id'].map(GROUP_NAMES)

    return df


def generate_table5_summary(df):
    """生成表5: 权衡关系汇总"""
    summary_data = []

    for int_type in ['hyperparam', 'interaction', 'other']:
        subset = df[df['intervention_type'] == int_type]

        total = len(subset)
        energy_vs_perf = len(subset[subset['tradeoff_type'] == 'energy_vs_performance'])
        energy_vs_energy = len(subset[subset['tradeoff_type'] == 'energy_vs_energy'])
        other = total - energy_vs_perf - energy_vs_energy

        type_label = {
            'hyperparam': 'Hyperparameter',
            'interaction': 'Interaction (×Parallel)',
            'other': 'Other'
        }[int_type]

        summary_data.append({
            'Intervention Type': type_label,
            'Total Tradeoffs': total,
            'Energy vs Performance': energy_vs_perf,
            'Energy vs Energy': energy_vs_energy,
            'Other': other
        })

    # 添加总计行
    summary_data.append({
        'Intervention Type': 'Total',
        'Total Tradeoffs': len(df),
        'Energy vs Performance': len(df[df['tradeoff_type'] == 'energy_vs_performance']),
        'Energy vs Energy': len(df[df['tradeoff_type'] == 'energy_vs_energy']),
        'Other': len(df) - len(df[df['tradeoff_type'] == 'energy_vs_performance']) - len(df[df['tradeoff_type'] == 'energy_vs_energy'])
    })

    return pd.DataFrame(summary_data)


def generate_table6_energy_perf_tradeoffs(df):
    """生成表6: 能耗vs性能权衡详情"""
    # 筛选能耗vs性能权衡
    energy_perf = df[df['tradeoff_type'] == 'energy_vs_performance'].copy()

    if len(energy_perf) == 0:
        return pd.DataFrame()

    # 确定哪个是能耗变量，哪个是性能变量
    table_data = []
    for _, row in energy_perf.iterrows():
        if row['metric1'] in ENERGY_VARIABLES:
            energy_var = row['metric1']
            perf_var = row['metric2']
            ate_energy = row['ate1']
            ate_perf = row['ate2']
        else:
            energy_var = row['metric2']
            perf_var = row['metric1']
            ate_energy = row['ate2']
            ate_perf = row['ate1']

        # 确定权衡方向
        if ate_energy > 0 and ate_perf < 0:
            direction = 'Energy↑ Performance↓'
        elif ate_energy < 0 and ate_perf > 0:
            direction = 'Energy↓ Performance↑'
        elif ate_energy > 0 and ate_perf > 0:
            direction = 'Both↑ (Same direction)'
        else:
            direction = 'Both↓ (Same direction)'

        table_data.append({
            'Group': row['group_name'],
            'Intervention': row['intervention'].replace('hyperparam_', '').replace('_x_is_parallel', '×P'),
            'Energy Variable': energy_var.replace('energy_', ''),
            'Performance Variable': perf_var.replace('perf_', ''),
            'ATE Energy': ate_energy,
            'ATE Performance': ate_perf,
            'Direction': direction
        })

    return pd.DataFrame(table_data)


def create_tradeoff_scatter(df, filename):
    """创建图5: 权衡散点图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for int_type, color in COLORS.items():
        subset = df[df['intervention_type'] == int_type]
        if len(subset) == 0:
            continue

        label = {
            'hyperparam': 'Hyperparameter',
            'interaction': 'Interaction (×Parallel)',
            'other': 'Other'
        }[int_type]

        ax.scatter(subset['ate1'], subset['ate2'],
                  c=color, label=label, s=60, alpha=0.7, edgecolors='black', linewidths=0.5)

    # 添加参考线
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # 标注象限
    ax.text(0.95, 0.95, 'Q1: Both +', transform=ax.transAxes, ha='right', va='top', fontsize=9, color='gray')
    ax.text(0.05, 0.95, 'Q2: Trade-off', transform=ax.transAxes, ha='left', va='top', fontsize=9, color='gray')
    ax.text(0.05, 0.05, 'Q3: Both -', transform=ax.transAxes, ha='left', va='bottom', fontsize=9, color='gray')
    ax.text(0.95, 0.05, 'Q4: Trade-off', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='gray')

    ax.set_xlabel('ATE (Metric 1)', fontsize=12)
    ax.set_ylabel('ATE (Metric 2)', fontsize=12)
    ax.set_title('RQ3: Tradeoff Distribution by Intervention Type', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"保存散点图: {output_path}")


def create_group_tradeoff_bar(df, filename):
    """创建图6: 6组权衡数量分布（堆叠柱状图）"""
    # 按组和干预类型统计
    groups = list(GROUP_NAMES.values())
    int_types = ['hyperparam', 'interaction', 'other']

    data = {int_type: [] for int_type in int_types}

    for group in groups:
        group_data = df[df['group_name'] == group]
        for int_type in int_types:
            count = len(group_data[group_data['intervention_type'] == int_type])
            data[int_type].append(count)

    # 创建堆叠柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(groups))
    width = 0.6

    bottom = np.zeros(len(groups))

    labels = {
        'hyperparam': 'Hyperparameter',
        'interaction': 'Interaction (×Parallel)',
        'other': 'Other'
    }

    for int_type in int_types:
        values = data[int_type]
        ax.bar(x, values, width, label=labels[int_type], bottom=bottom, color=COLORS[int_type], edgecolor='black', linewidth=0.5)
        bottom += np.array(values)

    # 添加总数标签
    for i, total in enumerate(bottom):
        if total > 0:
            ax.text(i, total + 0.5, str(int(total)), ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Task Group', fontsize=12)
    ax.set_ylabel('Number of Tradeoffs', fontsize=12)
    ax.set_title('RQ3: Tradeoff Distribution Across Groups', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"保存柱状图: {output_path}")


def print_summary(df, table5, table6):
    """输出分析摘要"""
    print("\n" + "=" * 60)
    print("RQ3 分析摘要")
    print("=" * 60)

    print(f"\n【总体统计】")
    print(f"  总权衡数: {len(df)}")

    print(f"\n【按干预变量类型】")
    for int_type in ['hyperparam', 'interaction', 'other']:
        count = len(df[df['intervention_type'] == int_type])
        print(f"  - {int_type}: {count}条")

    print(f"\n【按权衡类型】")
    for tt in ['energy_vs_performance', 'energy_vs_energy', 'other']:
        count = len(df[df['tradeoff_type'] == tt])
        print(f"  - {tt}: {count}条")

    print(f"\n【能耗vs性能权衡详情】")
    energy_perf = df[df['tradeoff_type'] == 'energy_vs_performance']
    if len(energy_perf) == 0:
        print("  ⚠️ 无能耗vs性能权衡")
    else:
        print(f"  共 {len(energy_perf)} 条:")
        for _, row in energy_perf.iterrows():
            print(f"    {row['group_name']}: {row['intervention']} → ({row['metric1']} vs {row['metric2']})")

    print(f"\n【按组分布】")
    for group_name in GROUP_NAMES.values():
        count = len(df[df['group_name'] == group_name])
        print(f"  - {group_name}: {count}条")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("RQ3 数据分析 - 能耗与性能权衡")
    print("=" * 60)

    # 确保输出目录存在
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("\n[1/5] 加载权衡检测数据...")
    df = load_tradeoff_data()

    # 2. 生成表5: 权衡关系汇总
    print("\n[2/5] 生成表5: 权衡关系汇总...")
    table5 = generate_table5_summary(df)
    output_path = TABLES_DIR / 'table5_rq3_tradeoff_summary.csv'
    table5.to_csv(output_path, index=False)
    print(f"  保存: {output_path}")
    print(table5.to_string(index=False))

    # 3. 生成表6: 能耗vs性能权衡详情
    print("\n[3/5] 生成表6: 能耗vs性能权衡详情...")
    table6 = generate_table6_energy_perf_tradeoffs(df)
    if len(table6) > 0:
        output_path = TABLES_DIR / 'table6_rq3_energy_perf_tradeoffs.csv'
        table6.to_csv(output_path, index=False)
        print(f"  保存: {output_path}")
        print(table6.to_string(index=False))
    else:
        print("  ⚠️ 无能耗vs性能权衡，跳过表6生成")

    # 4. 生成图5: 权衡散点图
    print("\n[4/5] 生成图5: 权衡散点图...")
    create_tradeoff_scatter(df, 'figure5_rq3_tradeoff_scatter.pdf')

    # 5. 生成图6: 6组权衡数量分布
    print("\n[5/5] 生成图6: 6组权衡数量分布...")
    create_group_tradeoff_bar(df, 'figure6_rq3_group_tradeoff_bar.pdf')

    # 输出摘要
    print_summary(df, table5, table6)

    print("\n分析完成!")
    print(f"表格保存至: {TABLES_DIR}")
    print(f"图表保存至: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
