#!/usr/bin/env python3
"""
DiBS因果边白名单过滤脚本

根据因果逻辑白名单规则过滤DiBS发现的因果边，排除反因果、反直觉的边。

使用方法:
    python filter_causal_edges_by_whitelist.py --input group1_causal_edges_0.3.csv
    python filter_causal_edges_by_whitelist.py --input-dir threshold/ --output-dir whitelist/

输入: CSV文件，包含source_category和target_category列
输出: 过滤后的CSV文件，仅保留白名单中的因果边
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
# ============================================================================

WHITELIST_RULES = {
    # 规则组1: 超参数主效应 (Q1, Q2)
    ('hyperparam', 'energy'): True,       # 超参数 → 能耗
    ('hyperparam', 'mediator'): True,     # 超参数 → 中间变量
    ('hyperparam', 'performance'): True,  # 超参数 → 性能

    # 规则组2: 交互项调节效应 (Q1, Q2)
    ('interaction', 'energy'): True,      # 交互项 → 能耗
    ('interaction', 'mediator'): True,    # 交互项 → 中间变量
    ('interaction', 'performance'): True, # 交互项 → 性能

    # 规则组3: 中间变量中介效应 (Q2/Q3)
    ('mediator', 'energy'): True,         # 中间变量 → 能耗
    ('mediator', 'mediator'): True,       # 中间变量 → 中间变量
    ('mediator', 'performance'): True,    # 中间变量 → 性能
    ('energy', 'energy'): True,           # 能耗 → 能耗 (能耗分解)

    # 规则组4: 控制变量影响
    ('control', 'energy'): True,          # 模型控制变量 → 能耗
    ('control', 'mediator'): True,        # 模型控制变量 → 中间变量
    ('control', 'performance'): True,     # 模型控制变量 → 性能
    ('mode', 'energy'): True,             # 并行模式 → 能耗
    ('mode', 'mediator'): True,           # 并行模式 → 中间变量
    ('mode', 'performance'): True,        # 并行模式 → 性能

    # 注意: 白名单矩阵中的⚠️在实现中都视为False（禁止）
    # 理由参见 docs/CAUSAL_EDGE_WHITELIST_DESIGN.md 第5.2节"特殊情况说明"
    # 其他所有组合默认为 False（黑名单）
}

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
        edges_df: 包含 source_category, target_category 列的DataFrame

    返回:
        过滤后的DataFrame（仅包含合理的因果边）
    """
    # 验证必需列
    required_cols = ['source_category', 'target_category']
    missing_cols = [col for col in required_cols if col not in edges_df.columns]
    if missing_cols:
        raise ValueError(f"输入数据缺少必需列: {missing_cols}")

    # 应用白名单过滤
    mask = edges_df.apply(
        lambda row: is_edge_allowed(row['source_category'], row['target_category']),
        axis=1
    )

    filtered_df = edges_df[mask].copy()

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

    # 统计
    stats = get_filter_statistics(df, filtered_df)

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
                logger.info(f"    {row['source']} → {row['target']} "
                          f"({row['source_category']} → {row['target_category']}, "
                          f"强度={row['strength']:.2f})")

    return stats


def process_directory(input_dir: Path, output_dir: Path, pattern: str, dry_run: bool = False):
    """
    批量处理目录中的CSV文件

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件名模式（如 "*_causal_edges_0.3.csv"）
        dry_run: 是否仅预览不写入
    """
    # 查找匹配文件
    input_files = sorted(input_dir.glob(pattern))

    if not input_files:
        logger.warning(f"未找到匹配文件: {input_dir}/{pattern}")
        return

    logger.info(f"找到 {len(input_files)} 个文件")
    logger.info("")

    # 处理每个文件
    all_stats = []
    for input_path in input_files:
        output_filename = input_path.name.replace('_0.3.csv', '_whitelist.csv')
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
  # 处理单个文件
  python filter_causal_edges_by_whitelist.py \\
      --input threshold/group1_examples_causal_edges_0.3.csv \\
      --output whitelist/group1_examples_causal_edges_whitelist.csv

  # 批量处理目录
  python filter_causal_edges_by_whitelist.py \\
      --input-dir results/energy_research/data/interaction/threshold \\
      --output-dir results/energy_research/data/interaction/whitelist \\
      --pattern "*_causal_edges_0.3.csv"

  # 预览模式（不写入文件）
  python filter_causal_edges_by_whitelist.py \\
      --input-dir threshold/ \\
      --output-dir whitelist/ \\
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
    parser.add_argument('--pattern', type=str, default='*_causal_edges_0.3.csv',
                       help='文件名模式（批量模式，默认: *_causal_edges_0.3.csv）')

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
