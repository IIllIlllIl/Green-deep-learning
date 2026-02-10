#!/usr/bin/env python3
"""
RQ2 数据分析脚本 - 中间变量解释
利用中间变量（GPU温度、利用率）解释超参数对能耗的因果机制

数据来源: 全局标准化因果图
输出:
- 表3: 中介变量频率统计
- 表4: 主要因果路径
- 图3: 因果路径Sankey图
- 图4: 6组中介模式热力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict
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
    'interaction': '#F39B7F',   # 浅红
    'mediator': '#4DBBD5',      # 蓝色
    'energy': '#00A087',        # 绿色
    'performance': '#3C5488',   # 深蓝
    'other': '#999999'          # 灰色
}

# 路径配置
BASE_DIR = Path('/home/green/energy_dl/nightly/analysis')
CAUSAL_GRAPH_DIR = BASE_DIR / 'results/energy_research/data/global_std'
OUTPUT_DIR = BASE_DIR / 'results/energy_research/rq_analysis'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# 变量分类 (v2.0)
MEDIATOR_VARIABLES = [
    'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
    'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent'
]

ENERGY_VARIABLES = [
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules',
    'energy_cpu_total_joules', 'energy_gpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_min_watts', 'energy_gpu_max_watts'
]

PERFORMANCE_VARIABLES = [
    'perf_test_accuracy', 'perf_map', 'perf_rank1', 'perf_rank5',
    'perf_precision', 'perf_recall', 'perf_eval_samples_per_second',
    'perf_top1_accuracy', 'perf_top5_accuracy', 'perf_top10_accuracy', 'perf_top20_accuracy'
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


def get_variable_category(var_name):
    """获取变量类别"""
    if '_x_is_parallel' in var_name:
        return 'interaction'
    elif var_name.startswith('hyperparam_'):
        return 'hyperparam'
    elif var_name in ENERGY_VARIABLES:
        return 'energy'
    elif var_name in MEDIATOR_VARIABLES:
        return 'mediator'
    elif var_name.startswith('perf_') or var_name in PERFORMANCE_VARIABLES:
        return 'performance'
    elif var_name.startswith('model_'):
        return 'control'
    elif var_name == 'is_parallel':
        return 'mode'
    else:
        return 'other'


def load_causal_graph(group_dir):
    """加载因果图和特征名"""
    group_name = group_dir.name

    # 查找因果图CSV文件
    edge_files = list(group_dir.glob('*_dibs_edges_threshold_0.3.csv'))
    if not edge_files:
        print(f"  警告: {group_name} 无强边文件")
        return None, None, None

    # 加载特征名
    feature_file = list(group_dir.glob('*_feature_names.json'))
    if not feature_file:
        print(f"  警告: {group_name} 无特征名文件")
        return None, None, None

    with open(feature_file[0]) as f:
        feature_names = json.load(f)

    # 加载边数据
    edges_df = pd.read_csv(edge_files[0])

    return edges_df, feature_names, group_name


def extract_indirect_paths(edges_df, feature_names, group_name, min_strength=0.3):
    """
    从因果图提取间接路径

    RQ2要求的路径类型:
    - 主效应路径: hyperparam → mediator → energy
    - 调节效应路径: interaction → mediator → energy
    """
    paths = []

    # 确定边强度列名（兼容不同格式）
    strength_col = 'weight' if 'weight' in edges_df.columns else 'strength'

    # 构建邻接字典
    adj = defaultdict(list)
    for _, row in edges_df.iterrows():
        if row[strength_col] >= min_strength:
            adj[row['source']].append((row['target'], row[strength_col]))

    # 搜索2-hop路径: source → mediator → target
    for source in feature_names:
        source_cat = get_variable_category(source)

        # 只关注超参数和交互项作为起点
        if source_cat not in ['hyperparam', 'interaction']:
            continue

        # 排除seed
        if 'seed' in source:
            continue

        for mediator, s1 in adj.get(source, []):
            mediator_cat = get_variable_category(mediator)

            # 只关注物理状态变量作为中介
            if mediator_cat != 'mediator':
                continue

            for target, s2 in adj.get(mediator, []):
                target_cat = get_variable_category(target)

                # 只关注能耗变量作为终点
                if target_cat != 'energy':
                    continue

                path_strength = min(s1, s2)  # 使用最小边强度

                # 确定路径类型
                if source_cat == 'hyperparam':
                    path_type = 'main_effect'
                else:
                    path_type = 'moderation_effect'

                paths.append({
                    'group': group_name,
                    'group_name': GROUP_NAMES.get(group_name, group_name),
                    'source': source,
                    'mediator': mediator,
                    'target': target,
                    'path': f"{source} → {mediator} → {target}",
                    'path_type': path_type,
                    'step1_strength': s1,
                    'step2_strength': s2,
                    'path_strength': path_strength,
                    'source_category': source_cat,
                    'mediator_category': mediator_cat,
                    'target_category': target_cat
                })

    return paths


def generate_mediator_frequency_table(all_paths):
    """生成表3: 中介变量频率统计"""
    if len(all_paths) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(all_paths)

    # 按中介变量统计
    stats = []
    for mediator in MEDIATOR_VARIABLES:
        med_paths = df[df['mediator'] == mediator]

        if len(med_paths) == 0:
            continue

        # 主效应路径数
        main_count = len(med_paths[med_paths['path_type'] == 'main_effect'])
        # 调节效应路径数
        mod_count = len(med_paths[med_paths['path_type'] == 'moderation_effect'])

        # 涉及的超参数
        hyperparams = med_paths[med_paths['path_type'] == 'main_effect']['source'].unique()
        hyperparams_str = ', '.join([h.replace('hyperparam_', '') for h in hyperparams]) if len(hyperparams) > 0 else '-'

        # 涉及的交互项
        interactions = med_paths[med_paths['path_type'] == 'moderation_effect']['source'].unique()
        interactions_str = ', '.join([i.replace('hyperparam_', '').replace('_x_is_parallel', '×P') for i in interactions]) if len(interactions) > 0 else '-'

        # 涉及的能耗变量
        energy_vars = med_paths['target'].unique()
        energy_str = ', '.join([e.replace('energy_', '') for e in energy_vars])

        # 组覆盖
        groups = med_paths['group_name'].unique()
        group_coverage = f"{len(groups)}/6"

        stats.append({
            'Mediator': mediator.replace('energy_', ''),
            'Main Effect Paths': main_count,
            'Moderation Paths': mod_count,
            'Total Paths': main_count + mod_count,
            'Hyperparameters': hyperparams_str,
            'Interactions': interactions_str,
            'Energy Variables': energy_str,
            'Group Coverage': group_coverage
        })

    return pd.DataFrame(stats)


def generate_main_paths_table(all_paths):
    """生成表4: 主要因果路径"""
    if len(all_paths) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(all_paths)

    # 按路径类型和路径强度排序
    df = df.sort_values(['path_type', 'path_strength'], ascending=[True, False])

    # 简化显示
    table = df[['group_name', 'path_type', 'source', 'mediator', 'target',
                'step1_strength', 'step2_strength', 'path_strength']].copy()

    # 简化变量名
    table['source'] = table['source'].str.replace('hyperparam_', '').str.replace('_x_is_parallel', '×P')
    table['mediator'] = table['mediator'].str.replace('energy_', '')
    table['target'] = table['target'].str.replace('energy_', '')

    table.columns = ['Group', 'Path Type', 'Source', 'Mediator', 'Target',
                     'Step1 Strength', 'Step2 Strength', 'Path Strength']

    return table


def create_sankey_diagram(all_paths, filename):
    """创建图3: 因果路径Sankey图"""
    if len(all_paths) == 0:
        print("警告: 无路径数据，跳过Sankey图")
        return

    try:
        import plotly.graph_objects as go
        has_plotly = True
    except ImportError:
        has_plotly = False
        print("警告: plotly未安装，使用matplotlib替代")

    df = pd.DataFrame(all_paths)

    if has_plotly:
        # 使用plotly创建Sankey图
        # 收集所有节点
        sources = df['source'].unique().tolist()
        mediators = df['mediator'].unique().tolist()
        targets = df['target'].unique().tolist()

        all_nodes = sources + mediators + targets
        node_indices = {node: i for i, node in enumerate(all_nodes)}

        # 简化节点标签
        labels = []
        for node in all_nodes:
            label = node.replace('hyperparam_', '').replace('_x_is_parallel', '×P').replace('energy_', '')
            labels.append(label)

        # 节点颜色
        node_colors = []
        for node in all_nodes:
            cat = get_variable_category(node)
            node_colors.append(COLORS.get(cat, COLORS['other']))

        # 构建链接
        source_indices = []
        target_indices = []
        values = []
        link_colors = []

        # source → mediator
        for _, row in df.iterrows():
            source_indices.append(node_indices[row['source']])
            target_indices.append(node_indices[row['mediator']])
            values.append(row['step1_strength'])
            cat = get_variable_category(row['source'])
            link_colors.append(COLORS.get(cat, COLORS['other']).replace(')', ', 0.5)').replace('rgb', 'rgba'))

        # mediator → target
        for _, row in df.iterrows():
            source_indices.append(node_indices[row['mediator']])
            target_indices.append(node_indices[row['target']])
            values.append(row['step2_strength'])
            link_colors.append(COLORS.get('mediator', COLORS['other']).replace(')', ', 0.5)').replace('rgb', 'rgba'))

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=['rgba(150,150,150,0.3)'] * len(source_indices)
            )
        )])

        fig.update_layout(
            title_text="RQ2: Causal Paths through Mediators (Sankey Diagram)",
            font=dict(family="Times New Roman", size=12),
            width=1200,
            height=800
        )

        # 保存
        output_path = FIGURES_DIR / filename
        fig.write_image(str(output_path), format='pdf')
        fig.write_image(str(output_path.with_suffix('.png')), format='png', scale=2)
        print(f"保存Sankey图: {output_path}")

    else:
        # 使用matplotlib创建简化的流程图
        create_flow_diagram_matplotlib(df, filename)


def create_flow_diagram_matplotlib(df, filename):
    """使用matplotlib创建简化的流程图（Sankey替代）"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 收集节点
    sources = df['source'].unique()
    mediators = df['mediator'].unique()
    targets = df['target'].unique()

    # 位置设置
    y_positions = {}

    # 左侧：sources
    for i, s in enumerate(sources):
        y_positions[s] = ('left', i)

    # 中间：mediators
    for i, m in enumerate(mediators):
        y_positions[m] = ('middle', i)

    # 右侧：targets
    for i, t in enumerate(targets):
        y_positions[t] = ('right', i)

    x_coords = {'left': 0.1, 'middle': 0.5, 'right': 0.9}

    # 计算y坐标
    def get_y(nodes, idx, total_height=0.8):
        n = len(nodes)
        if n == 1:
            return 0.5
        spacing = total_height / (n - 1) if n > 1 else 0
        return 0.1 + idx * spacing

    # 绘制节点
    for node, (pos, idx) in y_positions.items():
        x = x_coords[pos]
        if pos == 'left':
            y = get_y(sources, idx)
        elif pos == 'middle':
            y = get_y(mediators, idx)
        else:
            y = get_y(targets, idx)

        cat = get_variable_category(node)
        color = COLORS.get(cat, COLORS['other'])

        label = node.replace('hyperparam_', '').replace('_x_is_parallel', '×P').replace('energy_', '')

        ax.scatter(x, y, s=500, c=color, zorder=5, edgecolors='black', linewidths=1)

        # 标签位置
        if pos == 'left':
            ax.text(x - 0.05, y, label, ha='right', va='center', fontsize=9)
        elif pos == 'right':
            ax.text(x + 0.05, y, label, ha='left', va='center', fontsize=9)
        else:
            ax.text(x, y + 0.05, label, ha='center', va='bottom', fontsize=9)

        y_positions[node] = (x, y)

    # 更新y_positions为实际坐标
    for node in list(y_positions.keys()):
        pos, idx = y_positions[node] if isinstance(y_positions[node], tuple) and len(y_positions[node]) == 2 and isinstance(y_positions[node][0], str) else (None, None)
        if pos is not None:
            x = x_coords[pos]
            if pos == 'left':
                y = get_y(sources, idx)
            elif pos == 'middle':
                y = get_y(mediators, idx)
            else:
                y = get_y(targets, idx)
            y_positions[node] = (x, y)

    # 绘制连线
    for _, row in df.iterrows():
        # source → mediator
        if row['source'] in y_positions and row['mediator'] in y_positions:
            x1, y1 = y_positions[row['source']]
            x2, y2 = y_positions[row['mediator']]
            alpha = min(row['step1_strength'], 1.0)
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=alpha*0.8, lw=1.5))

        # mediator → target
        if row['mediator'] in y_positions and row['target'] in y_positions:
            x1, y1 = y_positions[row['mediator']]
            x2, y2 = y_positions[row['target']]
            alpha = min(row['step2_strength'], 1.0)
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=alpha*0.8, lw=1.5))

    # 添加列标题
    ax.text(0.1, 0.95, 'Hyperparameters\n& Interactions', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.95, 'Mediators\n(GPU State)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(0.9, 0.95, 'Energy\nOutcomes', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('RQ2: Causal Paths through Mediators', fontsize=14, fontweight='bold', pad=20)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['hyperparam'], edgecolor='black', label='Hyperparameter'),
        Patch(facecolor=COLORS['interaction'], edgecolor='black', label='Interaction'),
        Patch(facecolor=COLORS['mediator'], edgecolor='black', label='Mediator'),
        Patch(facecolor=COLORS['energy'], edgecolor='black', label='Energy'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"保存流程图: {output_path}")


def create_mediator_heatmap(all_paths, filename):
    """创建图4: 6组中介模式热力图"""
    if len(all_paths) == 0:
        print("警告: 无路径数据，跳过热力图")
        return

    df = pd.DataFrame(all_paths)

    # 构建热力图矩阵：组 × 中介变量
    groups = list(GROUP_NAMES.values())
    mediators = [m.replace('energy_', '') for m in MEDIATOR_VARIABLES]

    # 初始化矩阵
    matrix = np.zeros((len(groups), len(mediators)))

    for i, group in enumerate(groups):
        group_paths = df[df['group_name'] == group]
        for j, med in enumerate(MEDIATOR_VARIABLES):
            count = len(group_paths[group_paths['mediator'] == med])
            matrix[i, j] = count

    # 创建热力图
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix, cmap='Blues', aspect='auto')

    # 设置标签
    ax.set_xticks(range(len(mediators)))
    ax.set_xticklabels([m.replace('gpu_', '') for m in mediators], rotation=45, ha='right')
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)

    # 添加数值标注
    for i in range(len(groups)):
        for j in range(len(mediators)):
            value = int(matrix[i, j])
            color = 'white' if value > matrix.max() / 2 else 'black'
            ax.text(j, i, str(value), ha='center', va='center', color=color, fontsize=10)

    ax.set_xlabel('Mediator Variable', fontsize=12)
    ax.set_ylabel('Task Group', fontsize=12)
    ax.set_title('RQ2: Mediator Usage Pattern Across Groups', fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Path Count', fontsize=11)

    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"保存热力图: {output_path}")


def print_summary(all_paths):
    """输出分析摘要"""
    print("\n" + "=" * 60)
    print("RQ2 分析摘要")
    print("=" * 60)

    if len(all_paths) == 0:
        print("\n⚠️ 未发现符合条件的间接路径")
        print("可能原因:")
        print("  1. 边强度阈值(0.3)过高")
        print("  2. 超参数→中介变量或中介变量→能耗的边不存在")
        return

    df = pd.DataFrame(all_paths)

    print(f"\n【路径统计】")
    print(f"  总路径数: {len(df)}")

    main_effect = df[df['path_type'] == 'main_effect']
    mod_effect = df[df['path_type'] == 'moderation_effect']

    print(f"  - 主效应路径: {len(main_effect)}")
    print(f"  - 调节效应路径: {len(mod_effect)}")

    print(f"\n【中介变量使用频率】")
    for med in MEDIATOR_VARIABLES:
        count = len(df[df['mediator'] == med])
        print(f"  {med.replace('energy_', '')}: {count}条路径")

    print(f"\n【组覆盖情况】")
    for group_name in GROUP_NAMES.values():
        count = len(df[df['group_name'] == group_name])
        print(f"  {group_name}: {count}条路径")

    if len(main_effect) > 0:
        print(f"\n【主效应路径示例】")
        for _, row in main_effect.head(3).iterrows():
            print(f"  {row['group_name']}: {row['source'].replace('hyperparam_', '')} → "
                  f"{row['mediator'].replace('energy_', '')} → "
                  f"{row['target'].replace('energy_', '')} "
                  f"(强度={row['path_strength']:.3f})")

    if len(mod_effect) > 0:
        print(f"\n【调节效应路径示例】")
        for _, row in mod_effect.head(3).iterrows():
            print(f"  {row['group_name']}: {row['source'].replace('hyperparam_', '').replace('_x_is_parallel', '×P')} → "
                  f"{row['mediator'].replace('energy_', '')} → "
                  f"{row['target'].replace('energy_', '')} "
                  f"(强度={row['path_strength']:.3f})")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("RQ2 数据分析 - 中间变量解释")
    print("=" * 60)

    # 确保输出目录存在
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载所有组的因果图并提取间接路径
    print("\n[1/5] 加载因果图并提取间接路径...")
    all_paths = []

    for group_dir in sorted(CAUSAL_GRAPH_DIR.iterdir()):
        if not group_dir.is_dir():
            continue

        print(f"  处理: {group_dir.name}")
        edges_df, feature_names, group_name = load_causal_graph(group_dir)

        if edges_df is None:
            continue

        paths = extract_indirect_paths(edges_df, feature_names, group_name)
        print(f"    发现 {len(paths)} 条间接路径")
        all_paths.extend(paths)

    print(f"\n  总计: {len(all_paths)} 条间接路径")

    # 2. 生成表3: 中介变量频率统计
    print("\n[2/5] 生成表3: 中介变量频率统计...")
    table3 = generate_mediator_frequency_table(all_paths)
    if len(table3) > 0:
        output_path = TABLES_DIR / 'table3_rq2_mediator_frequency.csv'
        table3.to_csv(output_path, index=False)
        print(f"  保存: {output_path}")
    else:
        print("  ⚠️ 无数据生成")

    # 3. 生成表4: 主要因果路径
    print("\n[3/5] 生成表4: 主要因果路径...")
    table4 = generate_main_paths_table(all_paths)
    if len(table4) > 0:
        output_path = TABLES_DIR / 'table4_rq2_main_paths.csv'
        table4.to_csv(output_path, index=False)
        print(f"  保存: {output_path}")
    else:
        print("  ⚠️ 无数据生成")

    # 4. 生成图3: Sankey图
    print("\n[4/5] 生成图3: 因果路径流程图...")
    create_flow_diagram_matplotlib(pd.DataFrame(all_paths) if all_paths else pd.DataFrame(),
                                   'figure3_rq2_causal_paths.pdf')

    # 5. 生成图4: 热力图
    print("\n[5/5] 生成图4: 6组中介模式热力图...")
    create_mediator_heatmap(all_paths, 'figure4_rq2_mediator_heatmap.pdf')

    # 输出摘要
    print_summary(all_paths)

    print("\n分析完成!")
    print(f"表格保存至: {TABLES_DIR}")
    print(f"图表保存至: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
