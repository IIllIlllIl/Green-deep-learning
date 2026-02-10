#!/usr/bin/env python3
"""
DiBS因果边白名单过滤脚本

根据因果逻辑白名单规则过滤DiBS发现的因果边，排除反因果、反直觉的边。

使用方法:
    python filter_causal_edges_by_whitelist.py --input group1_dibs_edges_threshold_0.3.csv
    python filter_causal_edges_by_whitelist.py --input-dir global_std/ --output-dir whitelist/

输入: CSV文件，支持两种格式：
  1. 简单格式（自动分类）: source, target, weight
  2. 完整格式（已有分类）: source, target, source_category, target_category, strength
输出: 过滤后的CSV文件，仅保留白名单中的因果边

版本: v2.1 (2026-02-10)
更新: 添加自动变量分类功能，支持DiBS直接输出格式
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 白名单规则定义（基于因果逻辑和研究问题）
# 版本: v2.0 (2026-02-07)
# 修订: 功率变量(energy_gpu_*_watts)从mediator改为energy
# ============================================================================

WHITELIST_RULES = {
    # 规则组1: 超参数主效应 (RQ1)
    ('hyperparam', 'energy'): True,       # 超参数 → 能耗（含功率）
    ('hyperparam', 'mediator'): True,     # 超参数 → 中间变量（温度、利用率）
    ('hyperparam', 'performance'): True,  # 超参数 → 性能

    # 规则组2: 交互项调节效应 (RQ1b)
    ('interaction', 'energy'): True,      # 交互项 → 能耗
    ('interaction', 'mediator'): True,    # 交互项 → 中间变量
    ('interaction', 'performance'): True, # 交互项 → 性能

    # 规则组3: 中介效应 (RQ2)
    ('mediator', 'energy'): True,         # 中间变量 → 能耗（含功率）
    ('mediator', 'mediator'): True,       # 中间变量 → 中间变量
    ('mediator', 'performance'): True,    # 中间变量 → 性能（用于RQ3）
    ('energy', 'energy'): True,           # 能耗 → 能耗 (能耗指标间关系)

    # 规则组4: 控制变量影响
    ('control', 'energy'): True,          # 模型控制变量 → 能耗
    ('control', 'mediator'): True,        # 模型控制变量 → 中间变量
    ('control', 'performance'): True,     # 模型控制变量 → 性能
    ('mode', 'energy'): True,             # 并行模式 → 能耗
    ('mode', 'mediator'): True,           # 并行模式 → 中间变量
    ('mode', 'performance'): True,        # 并行模式 → 性能

    # 禁止规则（显式列出以便理解）:
    # ('energy', 'performance'): False,   # 能耗不直接影响性能（权衡通过超参数产生）
    # ('performance', '*'): False,        # 性能不应作为因
    # ('energy', 'hyperparam'): False,    # 反因果
    # ('energy', 'mediator'): False,      # 能耗不应影响物理状态（反因果）
}


# ============================================================================
# 变量分类函数（与convert_dibs_to_csv.py保持一致）
# 版本: v2.0 (2026-02-07)
# ============================================================================

# 能耗指标列表（焦耳 + 功率）
ENERGY_VARS = [
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
    'energy_gpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_min_watts', 'energy_gpu_max_watts'
]

# 中间变量列表（仅温度和利用率，物理状态变量）
MEDIATOR_VARS = [
    'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
    'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent'
]


def get_variable_category(var_name: str) -> str:
    """
    获取变量类别

    变量分类规则 (v2.0 - 2026-02-07更新):
    - energy: 能耗结果变量（包括焦耳和功率指标）
    - mediator: 机制中介变量（仅温度和利用率，物理状态变量）
    - hyperparam: 超参数处理变量
    - interaction: 交互项调节变量
    - performance: 性能结果变量
    - control: 模型控制变量
    - mode: 并行模式变量

    修订说明:
    - 功率变量(energy_gpu_*_watts)从mediator改为energy
    - 中间变量仅保留物理状态变量(温度、利用率)
    """
    if '_x_is_parallel' in var_name:
        return 'interaction'
    elif var_name.startswith('hyperparam_'):
        return 'hyperparam'
    elif var_name in ENERGY_VARS:
        return 'energy'
    elif var_name in MEDIATOR_VARS:
        return 'mediator'
    elif var_name.startswith('perf_'):
        return 'performance'
    elif var_name.startswith('model_'):
        return 'control'
    elif var_name == 'is_parallel':
        return 'mode'
    else:
        return 'other'


def add_categories_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    如果输入数据缺少分类列，自动添加source_category和target_category

    参数:
        df: 输入DataFrame，必须包含source和target列

    返回:
        添加了分类列的DataFrame
    """
    df = df.copy()

    # 检查是否已有分类列
    if 'source_category' not in df.columns:
        logger.info("  检测到缺少source_category列，自动添加...")
        df['source_category'] = df['source'].apply(get_variable_category)

    if 'target_category' not in df.columns:
        logger.info("  检测到缺少target_category列，自动添加...")
        df['target_category'] = df['target'].apply(get_variable_category)

    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化列名（处理weight/strength混用）

    参数:
        df: 输入DataFrame

    返回:
        列名标准化的DataFrame
    """
    df = df.copy()

    # 如果只有weight列，创建strength列
    if 'strength' not in df.columns and 'weight' in df.columns:
        logger.info("  检测到weight列，创建strength列...")
        df['strength'] = df['weight']

    return df

def is_edge_allowed(source_cat: str, target_cat: str) -> bool:
    """
    检查因果边是否在白名单中

    参数:
        source_cat: 源变量类别
        target_cat: 目标变量类别

    返回:
        True if 允许, False if 禁止
    """
    return WHITELIST_RULES.get((source_cat, target_cat), False)


def filter_causal_edges_by_whitelist(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据白名单规则过滤因果边

    参数:
        edges_df: 包含 source, target 列的DataFrame
                  可选: source_category, target_category (如缺失会自动添加)

    返回:
        过滤后的DataFrame（仅包含合理的因果边）
    """
    # 验证必需列
    required_cols = ['source', 'target']
    missing_cols = [col for col in required_cols if col not in edges_df.columns]
    if missing_cols:
        raise ValueError(f"输入数据缺少必需列: {missing_cols}")

    # 标准化列名
    df = normalize_column_names(edges_df)

    # 自动添加分类列（如果缺失）
    df = add_categories_if_missing(df)

    # 应用白名单过滤
    mask = df.apply(
        lambda row: is_edge_allowed(row['source_category'], row['target_category']),
        axis=1
    )

    filtered_df = df[mask].copy()

    return filtered_df


def get_filter_statistics(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> dict:
    """
    计算过滤统计信息

    返回:
        包含统计信息的字典
    """
    n_original = len(original_df)
    n_filtered = len(filtered_df)
    n_removed = n_original - n_filtered

    stats = {
        'n_original': n_original,
        'n_filtered': n_filtered,
        'n_removed': n_removed,
        'retention_rate': f"{n_filtered / n_original * 100:.1f}%" if n_original > 0 else "0%",
        'removal_rate': f"{n_removed / n_original * 100:.1f}%" if n_original > 0 else "0%"
    }

    # 按规则组统计
    if 'source_category' in filtered_df.columns and 'target_category' in filtered_df.columns:
        rule_groups = {
            'Q1_hyperparam_main': [
                ('hyperparam', 'energy'),
                ('hyperparam', 'mediator')
            ],
            'Q1_interaction_moderation': [
                ('interaction', 'energy'),
                ('interaction', 'mediator')
            ],
            'Q2_performance': [
                ('hyperparam', 'performance'),
                ('interaction', 'performance'),
                ('mediator', 'performance')  # v1.1��增：间接路径
            ],
            'Q3_mediation': [
                ('mediator', 'energy'),
                ('mediator', 'mediator'),
                ('energy', 'energy')
            ],
            'control_effects': [
                ('control', 'energy'),
                ('control', 'mediator'),
                ('control', 'performance'),
                ('mode', 'energy'),
                ('mode', 'mediator'),
                ('mode', 'performance')
            ]
        }

        for group_name, edge_types in rule_groups.items():
            count = 0
            for source_cat, target_cat in edge_types:
                count += len(filtered_df[
                    (filtered_df['source_category'] == source_cat) &
                    (filtered_df['target_category'] == target_cat)
                ])
            stats[f'n_{group_name}'] = count

    return stats


def process_single_file(input_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    """
    处理单个CSV文件

    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
        dry_run: 是否仅预览不写入

    返回:
        统计信息字典
    """
    logger.info(f"读取文件: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        raise

    logger.info(f"  原始边数: {len(df)}")

    # 过滤
    filtered_df = filter_causal_edges_by_whitelist(df)

    # 统计（使用原始df计算，但需要确保有category列）
    df_with_cats = add_categories_if_missing(normalize_column_names(df))
    stats = get_filter_statistics(df_with_cats, filtered_df)

    logger.info(f"  过滤后边数: {stats['n_filtered']}")
    logger.info(f"  保留率: {stats['retention_rate']}")
    logger.info(f"  移除率: {stats['removal_rate']}")

    # 分规则组统计
    if 'n_Q1_hyperparam_main' in stats:
        logger.info(f"  Q1超参数主效应: {stats['n_Q1_hyperparam_main']}条")
        logger.info(f"  Q1交互项调节: {stats['n_Q1_interaction_moderation']}条")
        logger.info(f"  Q2性能效应: {stats['n_Q2_performance']}条")
        logger.info(f"  Q3中介效应: {stats['n_Q3_mediation']}条")
        logger.info(f"  控制变量效应: {stats['n_control_effects']}条")

    # 写入
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_path, index=False)
        logger.info(f"输出文件: {output_path}")
    else:
        logger.info("  (dry-run模式，未写入文件)")
        # 显示前5条保留的边
        if len(filtered_df) > 0:
            logger.info("\n  示例保留的边（前5条）:")
            for i, row in filtered_df.head(5).iterrows():
                strength_col = 'strength' if 'strength' in filtered_df.columns else 'weight'
                logger.info(f"    {row['source']} → {row['target']} "
                          f"({row['source_category']} → {row['target_category']}, "
                          f"强度={row[strength_col]:.2f})")

    return stats


def process_directory(input_dir: Path, output_dir: Path, pattern: str, dry_run: bool = False):
    """
    批量处理目录中的CSV文件

    参数:
        input_dir: 输入目录（可以是包含多个group子目录的父目录，也可以是直接包含边文件的目录）
        output_dir: 输出目录
        pattern: 文件名模式（如 "*_dibs_edges_threshold_0.3.csv"）
        dry_run: 是否仅预览不写入
    """
    # 查找匹配文件
    input_files = sorted(input_dir.glob(pattern))

    # 如果没有找到，尝试在子目录中查找
    if not input_files:
        input_files = sorted(input_dir.glob(f'*/{pattern}'))

    if not input_files:
        logger.warning(f"未找到匹配文件: {input_dir}/{pattern}")
        return

    logger.info(f"找到 {len(input_files)} 个文件")
    logger.info("")

    # 处理每个文件
    all_stats = []
    for input_path in input_files:
        # 生成输出文件名：替换threshold_0.3为whitelist
        output_filename = input_path.name.replace('_threshold_0.3.csv', '_whitelist.csv')
        output_filename = output_filename.replace('_0.3.csv', '_whitelist.csv')

        # 保持子目录结构
        if input_path.parent != input_dir:
            relative_path = input_path.parent.relative_to(input_dir)
            output_path = output_dir / relative_path / output_filename
        else:
            output_path = output_dir / output_filename

        try:
            stats = process_single_file(input_path, output_path, dry_run)
            stats['filename'] = input_path.name
            all_stats.append(stats)
            logger.info("")
        except Exception as e:
            logger.error(f"处理文件 {input_path} 失败: {e}")
            continue

    # 汇总统计
    if all_stats:
        logger.info("=" * 70)
        logger.info("汇总统计")
        logger.info("=" * 70)

        summary_df = pd.DataFrame(all_stats)

        # 总体统计
        total_original = summary_df['n_original'].sum()
        total_filtered = summary_df['n_filtered'].sum()
        total_removed = summary_df['n_removed'].sum()

        logger.info(f"总计原始边数: {total_original}")
        logger.info(f"总计过滤后: {total_filtered}")
        logger.info(f"总计移除: {total_removed}")
        logger.info(f"总体保留率: {total_filtered / total_original * 100:.1f}%")

        # 分组统计
        if 'n_Q1_hyperparam_main' in summary_df.columns:
            logger.info("")
            logger.info("分研究问题统计:")
            logger.info(f"  Q1超参数主效应: {summary_df['n_Q1_hyperparam_main'].sum()}条")
            logger.info(f"  Q1交互项调节: {summary_df['n_Q1_interaction_moderation'].sum()}条")
            logger.info(f"  Q2性能效应: {summary_df['n_Q2_performance'].sum()}条")
            logger.info(f"  Q3中介效应: {summary_df['n_Q3_mediation'].sum()}条")
            logger.info(f"  控制变量效应: {summary_df['n_control_effects'].sum()}条")

        # 详细表格
        logger.info("")
        logger.info("各组详细统计:")
        logger.info(f"{'文件名':<40} {'原始':<8} {'保留':<8} {'移除':<8} {'保留率':<10}")
        logger.info("-" * 80)
        for _, row in summary_df.iterrows():
            logger.info(f"{row['filename']:<40} {row['n_original']:<8} "
                       f"{row['n_filtered']:<8} {row['n_removed']:<8} {row['retention_rate']:<10}")


def main():
    parser = argparse.ArgumentParser(
        description='DiBS因果边白名单过滤脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单个DiBS边文件
  python filter_causal_edges_by_whitelist.py \\
      --input results/energy_research/data/global_std/group1_examples/group1_examples_dibs_edges_threshold_0.3.csv \\
      --output results/energy_research/data/global_std_whitelist/group1_examples_dibs_edges_whitelist.csv

  # 批量处理目录（递归子目录）
  python filter_causal_edges_by_whitelist.py \\
      --input-dir results/energy_research/data/global_std \\
      --output-dir results/energy_research/data/global_std_whitelist \\
      --pattern "*_dibs_edges_threshold_0.3.csv"

  # 预览模式（不写入文件）
  python filter_causal_edges_by_whitelist.py \\
      --input-dir results/energy_research/data/global_std \\
      --pattern "*_dibs_edges_threshold_0.3.csv" \\
      --dry-run
        """
    )

    # 输入参数
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--input', type=str, help='输入CSV文件路径')
    group1.add_argument('--input-dir', type=str, help='输入目录路径')

    # 输出参数
    parser.add_argument('--output', type=str, help='输出CSV文件路径（单文件模式）')
    parser.add_argument('--output-dir', type=str, help='输出目录路径（批量模式）')

    # 批量处理参数
    parser.add_argument('--pattern', type=str, default='*_dibs_edges_threshold_0.3.csv',
                       help='文件名模式（批量模式，默认: *_dibs_edges_threshold_0.3.csv）')

    # 其他参数
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，不写入文件')

    args = parser.parse_args()

    # 单文件模式
    if args.input:
        input_path = Path(args.input)

        if not input_path.exists():
            logger.error(f"输入文件不存在: {input_path}")
            sys.exit(1)

        # 确定输出路径
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / input_path.name.replace('_0.3.csv', '_whitelist.csv')

        # 处理
        try:
            stats = process_single_file(input_path, output_path, args.dry_run)
        except Exception as e:
            logger.error(f"处理失败: {e}")
            sys.exit(1)

    # 批量模式
    elif args.input_dir:
        input_dir = Path(args.input_dir)

        if not input_dir.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            sys.exit(1)

        # 确定输出目录
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = input_dir.parent / 'whitelist'

        # 批量处理
        try:
            process_directory(input_dir, output_dir, args.pattern, args.dry_run)
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            sys.exit(1)

    logger.info("")
    logger.info("✅ 完成！")


if __name__ == '__main__':
    main()
