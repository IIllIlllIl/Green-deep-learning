#!/usr/bin/env python3
"""数据分层预处理脚本

用途: 将418行统一数据按任务分层，生成4个训练数据文件
作者: Claude
日期: 2025-12-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import json


# 任务组配置（5组方案 - 2025-12-24优化）
# 说明: 将image_classification拆分为examples和resnet两个子组，消除超参数冲突
TASK_GROUPS = {
    'image_classification_examples': {
        'repos': ['examples'],
        'models': {
            'examples': ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese']
        },
        'performance_cols': ['perf_test_accuracy'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'hyperparam_batch_size', 'hyperparam_seed'],  # 保留batch_size
        'has_onehot': True,
        'onehot_cols': ['is_mnist', 'is_mnist_ff', 'is_mnist_rnn', 'is_siamese']
    },
    'image_classification_resnet': {
        'repos': ['pytorch_resnet_cifar10'],
        'models': {
            'pytorch_resnet_cifar10': ['resnet20']
        },
        'performance_cols': ['perf_test_accuracy'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'l2_regularization', 'hyperparam_seed'],  # 保留l2_regularization
        'has_onehot': False,
        'onehot_cols': []
    },
    'person_reid': {
        'repos': ['Person_reID_baseline_pytorch'],
        'models': {
            'Person_reID_baseline_pytorch': ['densenet121', 'hrnet18', 'pcb']
        },
        'performance_cols': ['perf_map', 'perf_rank1', 'perf_rank5'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'hyperparam_dropout', 'hyperparam_seed'],
        'has_onehot': True,
        'onehot_cols': ['is_densenet121', 'is_hrnet18', 'is_pcb']
    },
    'vulberta': {
        'repos': ['VulBERTa'],
        'models': {
            'VulBERTa': ['mlp']
        },
        'performance_cols': ['perf_eval_loss'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'l2_regularization', 'hyperparam_seed'],
        'has_onehot': False,
        'onehot_cols': []
    },
    'bug_localization': {
        'repos': ['bug-localization-by-dnn-and-rvsm'],
        'models': {
            'bug-localization-by-dnn-and-rvsm': ['default']
        },
        'performance_cols': ['perf_top1_accuracy', 'perf_top5_accuracy'],
        'hyperparams': ['training_duration', 'l2_regularization',
                        'hyperparam_kfold', 'hyperparam_seed'],
        'has_onehot': False,
        'onehot_cols': []
    }
}

# 共享列（所有任务组都有）
COMMON_COLS = {
    'identifier': ['experiment_id', 'timestamp'],
    'energy': ['energy_cpu_total_joules', 'energy_gpu_total_joules',
               'energy_gpu_avg_watts'],
    'mediator': ['gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
                 'gpu_power_fluctuation', 'gpu_temp_fluctuation']
}


def load_extracted_data(input_path, limit=None):
    """加载提取的数据

    Args:
        input_path: 输入CSV路径
        limit: 限制读取行数（用于dry run）

    Returns:
        DataFrame: 加载的数据
    """
    print(f"\n加载数据: {input_path}")

    if limit:
        df = pd.read_csv(input_path, nrows=limit)
        print(f"  ⚠️ Dry Run模式: 只加载前 {limit} 行")
    else:
        df = pd.read_csv(input_path)

    print(f"  ✅ 加载完成: {len(df)} 行 × {len(df.columns)} 列")

    return df


def create_onehot_encoding(df, task_name):
    """为任务组创建One-Hot编码

    Args:
        df: DataFrame
        task_name: 任务名称

    Returns:
        DataFrame: 添加了One-Hot列的数据
    """
    task_config = TASK_GROUPS[task_name]

    if not task_config['has_onehot']:
        return df

    print(f"\n  生成One-Hot编码:")

    df_copy = df.copy()

    if task_name == 'image_classification_examples':
        # 图像分类examples子组: 4个模型的One-Hot编码
        df_copy['is_mnist'] = (df_copy['model'] == 'mnist').astype(int)
        df_copy['is_mnist_ff'] = (df_copy['model'] == 'mnist_ff').astype(int)
        df_copy['is_mnist_rnn'] = (df_copy['model'] == 'mnist_rnn').astype(int)
        df_copy['is_siamese'] = (df_copy['model'] == 'siamese').astype(int)

        # 验证互斥性
        onehot_sum = df_copy[['is_mnist', 'is_mnist_ff', 'is_mnist_rnn', 'is_siamese']].sum(axis=1)
        assert (onehot_sum == 1).all(), "图像分类examples组One-Hot编码互斥性验证失败"

        print(f"    is_mnist: {df_copy['is_mnist'].sum()} 个")
        print(f"    is_mnist_ff: {df_copy['is_mnist_ff'].sum()} 个")
        print(f"    is_mnist_rnn: {df_copy['is_mnist_rnn'].sum()} 个")
        print(f"    is_siamese: {df_copy['is_siamese'].sum()} 个")

    elif task_name == 'person_reid':
        # Person_reID: is_densenet121, is_hrnet18, is_pcb
        df_copy['is_densenet121'] = (df_copy['model'] == 'densenet121').astype(int)
        df_copy['is_hrnet18'] = (df_copy['model'] == 'hrnet18').astype(int)
        df_copy['is_pcb'] = (df_copy['model'] == 'pcb').astype(int)

        # 验证互斥性
        onehot_sum = df_copy[['is_densenet121', 'is_hrnet18', 'is_pcb']].sum(axis=1)
        assert (onehot_sum == 1).all(), "Person_reID One-Hot编码互斥性验证失败"

        print(f"    is_densenet121: {df_copy['is_densenet121'].sum()} 个")
        print(f"    is_hrnet18: {df_copy['is_hrnet18'].sum()} 个")
        print(f"    is_pcb: {df_copy['is_pcb'].sum()} 个")

    print(f"  ✅ One-Hot编码生成完成")

    return df_copy


def select_task_columns(df, task_name):
    """选择任务相关列

    Args:
        df: DataFrame
        task_name: 任务名称

    Returns:
        DataFrame: 只包含任务相关列的数据
    """
    task_config = TASK_GROUPS[task_name]

    # 构建列选择列表
    selected_cols = (
        COMMON_COLS['identifier'] +
        task_config['onehot_cols'] +
        task_config['hyperparams'] +
        COMMON_COLS['energy'] +
        COMMON_COLS['mediator'] +
        task_config['performance_cols']
    )

    # 验证所有列存在
    missing_cols = set(selected_cols) - set(df.columns)
    if missing_cols:
        print(f"  ⚠️ 警告: 以下列不存在: {missing_cols}")
        # 只选择存在的列
        selected_cols = [col for col in selected_cols if col in df.columns]

    df_selected = df[selected_cols].copy()

    print(f"  ✅ 列选择完成: {len(selected_cols)} 列")

    return df_selected


def remove_missing_performance(df, task_name):
    """删除性能全缺失的行

    Args:
        df: DataFrame
        task_name: 任务名称

    Returns:
        DataFrame: 删除缺失行后的数据
    """
    task_config = TASK_GROUPS[task_name]
    perf_cols = task_config['performance_cols']

    original_len = len(df)

    # 删除所有性能指标都缺失的行
    df_clean = df.dropna(subset=perf_cols, how='all').copy()

    removed = original_len - len(df_clean)

    print(f"  ✅ 删除性能全缺失行: {removed} 行（保留 {len(df_clean)} 行）")

    return df_clean


def save_stratified_data(df, task_name, output_dir, dry_run=False):
    """保存分层数据

    Args:
        df: DataFrame
        task_name: 任务名称
        output_dir: 输出目录
        dry_run: 是否为dry run模式

    Returns:
        Path: 输出文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        filename = f'training_data_{task_name}_dryrun.csv'
    else:
        filename = f'training_data_{task_name}.csv'

    output_path = output_dir / filename

    df.to_csv(output_path, index=False)

    print(f"  ✅ 保存成功: {output_path}")
    print(f"     行数: {len(df)}")
    print(f"     列数: {len(df.columns)}")

    return output_path


def process_task_group(df, task_name, output_dir, dry_run=False):
    """处理单个任务组

    Args:
        df: 原始DataFrame
        task_name: 任务名称
        output_dir: 输出目录
        dry_run: 是否为dry run模式

    Returns:
        dict: 处理结果统计
    """
    task_config = TASK_GROUPS[task_name]

    print(f"\n{'=' * 80}")
    print(f"处理任务组: {task_name}")
    print(f"{'=' * 80}")

    # 1. 筛选仓库
    print(f"\n1. 筛选仓库: {task_config['repos']}")
    mask = df['repository'].isin(task_config['repos'])
    task_df = df[mask].copy()
    print(f"  ✅ 筛选完成: {len(task_df)} 行")

    if len(task_df) == 0:
        print(f"  ⚠️ 警告: 没有数据，跳过该任务组")
        return None

    # 2. 添加One-Hot编码
    print(f"\n2. 添加One-Hot编码:")
    task_df = create_onehot_encoding(task_df, task_name)

    # 3. 选择任务相关列
    print(f"\n3. 选择任务相关列:")
    task_df = select_task_columns(task_df, task_name)

    # 4. 删除性能缺失行
    print(f"\n4. 删除性能缺失行:")
    task_df = remove_missing_performance(task_df, task_name)

    # 5. 保存数据
    print(f"\n5. 保存分层数据:")
    output_path = save_stratified_data(task_df, task_name, output_dir, dry_run)

    # 返回统计信息
    stats = {
        'task_name': task_name,
        'rows': len(task_df),
        'columns': len(task_df.columns),
        'output_path': str(output_path),
        'performance_cols': task_config['performance_cols'],
        'missing_rate': {
            col: task_df[col].isna().sum() / len(task_df) * 100
            for col in task_config['performance_cols']
            if col in task_df.columns
        }
    }

    return stats


def generate_summary_report(all_stats, output_dir):
    """生成汇总报告

    Args:
        all_stats: 所有任务组的统计信息
        output_dir: 输出目录
    """
    print(f"\n{'=' * 80}")
    print("处理汇总")
    print(f"{'=' * 80}")

    total_rows = 0
    for stats in all_stats:
        if stats:
            print(f"\n{stats['task_name']}:")
            print(f"  行数: {stats['rows']}")
            print(f"  列数: {stats['columns']}")
            print(f"  输出: {Path(stats['output_path']).name}")
            print(f"  性能指标缺失率:")
            for col, missing_rate in stats['missing_rate'].items():
                print(f"    {col}: {missing_rate:.2f}%")

            total_rows += stats['rows']

    print(f"\n总计: {total_rows} 行有效数据")

    # 保存统计信息到JSON
    stats_file = Path(output_dir) / 'stratified_data_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n✅ 统计信息已保存: {stats_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='数据分层预处理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # Dry run（处理前20行）
  python3 preprocess_stratified_data.py --dry-run --limit 20

  # 全量执行
  python3 preprocess_stratified_data.py

  # 指定输出目录
  python3 preprocess_stratified_data.py --output-dir /path/to/output
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default='../data/energy_research/raw/energy_data_extracted_v2.csv',
        help='输入CSV文件路径（默认: ../data/energy_research/raw/energy_data_extracted_v2.csv）'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/energy_research/processed',
        help='输出目录（默认: ../data/energy_research/processed）'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run模式：只处理少量数据'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Dry run模式下的最大行数（默认: 20）'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        choices=list(TASK_GROUPS.keys()),
        default=list(TASK_GROUPS.keys()),
        help='要处理的任务组（默认: 全部）'
    )

    args = parser.parse_args()

    # 打印配置
    print("=" * 80)
    print("数据分层预处理脚本")
    print("=" * 80)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"Dry Run: {'是' if args.dry_run else '否'}")
    if args.dry_run:
        print(f"限制行数: {args.limit}")
    print(f"任务组: {', '.join(args.tasks)}")
    print("=" * 80)

    # 1. 加载数据
    limit = args.limit if args.dry_run else None
    df = load_extracted_data(args.input, limit=limit)

    # 2. 处理每个任务组
    all_stats = []
    for task_name in args.tasks:
        try:
            stats = process_task_group(df, task_name, args.output_dir, args.dry_run)
            if stats:
                all_stats.append(stats)
        except Exception as e:
            print(f"\n❌ 处理任务组 {task_name} 失败:")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 3. 生成汇总报告
    if all_stats:
        generate_summary_report(all_stats, args.output_dir)

        print(f"\n{'=' * 80}")
        print("✅ 数据分层预处理完成！")
        print(f"{'=' * 80}")
    else:
        print(f"\n{'=' * 80}")
        print("⚠️ 没有生成任何输出文件")
        print(f"{'=' * 80}")
        sys.exit(1)


if __name__ == '__main__':
    main()
