#!/usr/bin/env python3
"""
DiBS因果图可视化

目的：
为6个任务组生成DiBS因果图可视化，突出显示：
1. 超参数→能耗的因果边（问题1）
2. 超参数→性能的因果边
3. 中介路径（问题3）

创建日期: 2026-01-06
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# 数据目录
DIBS_RESULT_DIR = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/questions_2_3_dibs/20260105_212940")
OUTPUT_DIR = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/causal_graph_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 任务组
TASK_GROUPS = [
    "group1_examples",
    "group2_vulberta",
    "group3_person_reid",
    "group4_bug_localization",
    "group5_mrt_oast",
    "group6_resnet"
]

# 绘图设置
sns.set_style("white")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def load_dibs_results(group_id):
    """加载DiBS结果"""

    # 加载因果图矩阵
    graph_file = DIBS_RESULT_DIR / f"{group_id}_causal_graph.npy"
    if not graph_file.exists():
        raise FileNotFoundError(f"因果图文件不存在: {graph_file}")

    graph = np.load(graph_file)

    # 加载特征名称
    features_file = DIBS_RESULT_DIR / f"{group_id}_feature_names.json"
    if not features_file.exists():
        raise FileNotFoundError(f"特征名称文件不存在: {features_file}")

    with open(features_file, 'r') as f:
        feature_names = json.load(f)

    print(f"\n加载DiBS结果: {group_id}")
    print(f"  因果图维度: {graph.shape}")
    print(f"  特征数: {len(feature_names)}")

    return graph, feature_names


def classify_variables(feature_names):
    """变量分类"""

    hyperparams = []
    performance = []
    energy = []
    mediators = []

    for name in feature_names:
        if name.startswith('hyperparam_'):
            hyperparams.append(name)
        elif name.startswith('perf_'):
            performance.append(name)
        elif name.startswith('energy_'):
            if name.startswith('energy_gpu_util') or name.startswith('energy_gpu_temp') or name.startswith('energy_gpu_memory'):
                mediators.append(name)
            else:
                energy.append(name)
        elif name.startswith('duration_'):
            pass  # 跳过duration
        else:
            mediators.append(name)

    return {
        'hyperparams': hyperparams,
        'performance': performance,
        'energy': energy,
        'mediators': mediators
    }


def extract_significant_edges(graph, feature_names, threshold=0.1):
    """提取显著因果边"""

    edges = []

    for i, source in enumerate(feature_names):
        for j, target in enumerate(feature_names):
            if i != j and graph[i, j] > threshold:
                edges.append({
                    'source': source,
                    'target': target,
                    'strength': float(graph[i, j]),
                    'source_idx': i,
                    'target_idx': j
                })

    # 按强度排序
    edges.sort(key=lambda x: x['strength'], reverse=True)

    return edges


def create_graph_layout(var_classes):
    """创建图布局（按变量类型分层）"""

    layout = {}

    # 层次布局
    # 第1层: 超参数（左上）
    # 第2层: 中介变量（中上）
    # 第3层: 性能指标（右上）
    # 第4层: 能耗指标（下方）

    y_positions = {
        'hyperparams': 0.8,
        'mediators': 0.5,
        'performance': 0.8,
        'energy': 0.2
    }

    x_positions = {
        'hyperparams': 0.1,
        'mediators': 0.5,
        'performance': 0.9,
        'energy': 0.5
    }

    # 计算每个变量的位置
    for var_type, variables in var_classes.items():
        if not variables:
            continue

        base_y = y_positions[var_type]
        base_x = x_positions[var_type]

        n = len(variables)

        if var_type == 'hyperparams':
            # 超参数垂直排列在左侧
            for i, var in enumerate(variables):
                y = base_y - (i * 0.15)
                layout[var] = (base_x, y)

        elif var_type == 'performance':
            # 性能指标垂直排列在右侧
            for i, var in enumerate(variables):
                y = base_y - (i * 0.15)
                layout[var] = (base_x, y)

        elif var_type == 'mediators':
            # 中介变量水平排列在中间
            for i, var in enumerate(variables):
                x = base_x - 0.2 + (i * 0.1)
                layout[var] = (x, base_y)

        elif var_type == 'energy':
            # 能耗指标水平排列在底部
            for i, var in enumerate(variables):
                x = base_x - 0.3 + (i * 0.1)
                layout[var] = (x, base_y)

    return layout


def simplify_variable_name(name):
    """简化变量名"""

    replacements = {
        'hyperparam_': 'HP: ',
        'perf_': 'Perf: ',
        'energy_gpu_': 'GPU: ',
        'energy_cpu_': 'CPU: ',
        'energy_': 'E: ',
        '_avg_watts': ' (avg W)',
        '_max_watts': ' (max W)',
        '_min_watts': ' (min W)',
        '_total_joules': ' (total J)',
        '_joules': ' (J)',
        '_avg_percent': ' (avg %)',
        '_max_percent': ' (max %)',
        '_avg_celsius': ' (avg °C)',
        '_max_celsius': ' (max °C)',
        'learning_rate': 'LR',
        'weight_decay': 'WD',
        'batch_size': 'BS',
        'epochs': 'EP',
        'dropout': 'Drop',
        'best_val_accuracy': 'Val Acc',
        'test_accuracy': 'Test Acc',
    }

    simplified = name
    for old, new in replacements.items():
        simplified = simplified.replace(old, new)

    return simplified


def visualize_causal_graph(group_id, graph, feature_names, threshold=0.1):
    """可视化因果图"""

    print(f"\n开始可视化: {group_id}")

    # 变量分类
    var_classes = classify_variables(feature_names)

    print(f"  超参数: {len(var_classes['hyperparams'])}")
    print(f"  性能指标: {len(var_classes['performance'])}")
    print(f"  能耗指标: {len(var_classes['energy'])}")
    print(f"  中介变量: {len(var_classes['mediators'])}")

    # 提取显著边
    edges = extract_significant_edges(graph, feature_names, threshold)

    print(f"  显著边数（threshold={threshold}): {len(edges)}")

    # 创建图布局
    layout = create_graph_layout(var_classes)

    # 创建图形
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 标题
    ax.text(0.5, 0.95, f'DiBS因果图: {group_id}',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # 定义颜色
    colors = {
        'hyperparams': '#3498db',  # 蓝色
        'performance': '#2ecc71',  # 绿色
        'energy': '#e74c3c',       # 红色
        'mediators': '#f39c12'     # 橙色
    }

    # 绘制节点
    node_boxes = {}

    for var_type, variables in var_classes.items():
        for var in variables:
            if var not in layout:
                continue

            x, y = layout[var]

            # 简化变量名
            label = simplify_variable_name(var)

            # 创建节点框
            box_width = 0.12
            box_height = 0.04

            box = FancyBboxPatch(
                (x - box_width/2, y - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.005",
                edgecolor=colors[var_type],
                facecolor=colors[var_type],
                alpha=0.3,
                linewidth=2
            )
            ax.add_patch(box)

            # 添加文本
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=9, fontweight='bold', color='black')

            node_boxes[var] = (x, y)

    # 绘制边
    for edge in edges:
        source = edge['source']
        target = edge['target']
        strength = edge['strength']

        if source not in node_boxes or target not in node_boxes:
            continue

        x1, y1 = node_boxes[source]
        x2, y2 = node_boxes[target]

        # 确定边的类型和颜色
        source_type = None
        target_type = None

        for var_type, variables in var_classes.items():
            if source in variables:
                source_type = var_type
            if target in variables:
                target_type = var_type

        # 边颜色（根据类型）
        if source_type == 'hyperparams' and target_type == 'energy':
            edge_color = '#e74c3c'  # 红色 - 问题1的关键边
            linewidth = 2 + strength * 3
            alpha = 0.8
        elif source_type == 'hyperparams' and target_type == 'mediators':
            edge_color = '#f39c12'  # 橙色 - 中介路径第一段
            linewidth = 1.5 + strength * 2
            alpha = 0.6
        elif source_type == 'mediators' and target_type == 'energy':
            edge_color = '#9b59b6'  # 紫色 - 中介路径第二段
            linewidth = 1.5 + strength * 2
            alpha = 0.6
        elif source_type == 'hyperparams' and target_type == 'performance':
            edge_color = '#2ecc71'  # 绿色 - 问题2相关
            linewidth = 1.5 + strength * 2
            alpha = 0.6
        else:
            edge_color = '#95a5a6'  # 灰色 - 其他边
            linewidth = 1 + strength * 1.5
            alpha = 0.4

        # 绘制箭头
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color=edge_color,
            linewidth=linewidth,
            alpha=alpha,
            connectionstyle="arc3,rad=0.1",
            zorder=1
        )
        ax.add_patch(arrow)

    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=colors['hyperparams'], alpha=0.3,
                     edgecolor=colors['hyperparams'], linewidth=2, label='超参数'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['mediators'], alpha=0.3,
                     edgecolor=colors['mediators'], linewidth=2, label='中介变量'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['performance'], alpha=0.3,
                     edgecolor=colors['performance'], linewidth=2, label='性能指标'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['energy'], alpha=0.3,
                     edgecolor=colors['energy'], linewidth=2, label='能耗指标'),
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
             framealpha=0.9, edgecolor='black')

    # 添加说明
    ax.text(0.02, 0.02,
           f'显著边数: {len(edges)} (threshold={threshold})\n'
           f'边宽度 = 强度 × 系数',
           fontsize=9, va='bottom', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 保存图形
    output_file = OUTPUT_DIR / f"{group_id}_causal_graph.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ 图形已保存: {output_file}")

    return output_file


def generate_all_visualizations():
    """为所有任务组生成可视化"""

    print("="*80)
    print("DiBS因果图可视化")
    print("="*80)

    results = []

    for group_id in TASK_GROUPS:
        print(f"\n{'='*80}")
        print(f"处理: {group_id}")
        print(f"{'='*80}")

        try:
            # 加载DiBS结果
            graph, feature_names = load_dibs_results(group_id)

            # 生成可视化
            output_file = visualize_causal_graph(group_id, graph, feature_names, threshold=0.1)

            results.append({
                'group': group_id,
                'status': 'success',
                'output_file': str(output_file)
            })

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            results.append({
                'group': group_id,
                'status': 'failed',
                'error': str(e)
            })

    # 保存结果摘要
    summary_file = OUTPUT_DIR / "visualization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("✅ 可视化完成！")
    print("="*80)
    print(f"  成功: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("="*80)

    return results


def main():
    """主函数"""
    generate_all_visualizations()


if __name__ == "__main__":
    main()
