#!/usr/bin/env python3
"""
步骤2: 准备分层DiBS数据（按并行/非并行模式）

目的：
为并行和非并行模式分别生成DiBS训练数据（6个任务组 × 2种模式 = 12个数据集）

创建日期: 2026-01-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# 路径
DATA_FILE = Path("/home/green/energy_dl/nightly/data/raw_data.csv")
OUTPUT_DIR_PARALLEL = Path("/home/green/energy_dl/nightly/analysis/data/energy_research/dibs_training_parallel")
OUTPUT_DIR_NON_PARALLEL = Path("/home/green/energy_dl/nightly/analysis/data/energy_research/dibs_training_non_parallel")

OUTPUT_DIR_PARALLEL.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_NON_PARALLEL.mkdir(parents=True, exist_ok=True)

# 任务组定义（修正为实际数据中的模型名称）
TASK_GROUPS = {
    'group1_examples': {
        'repos': ['examples'],
        'models': ['mnist_ff', 'mnist_rnn', 'siamese', 'mnist']  # 修正：实际模型名称
    },
    'group2_vulberta': {
        'repos': ['VulBERTa'],
        'models': ['mlp', 'lstm', 'cnn']  # 保持不变：实际只有mlp，但保留其他以备将来
    },
    'group3_person_reid': {
        'repos': ['Person_reID_baseline_pytorch'],
        'models': ['densenet121', 'hrnet18', 'pcb', 'resnet50', 'resnet50_ibn_a']  # 修正：添加实际模型
    },
    'group4_bug_localization': {
        'repos': ['bug-localization-by-dnn-and-rvsm'],
        'models': ['default', 'dnn', 'dnn_attention', 'dnn_bilstm', 'dnn_cnn']  # 修正：添加default
    },
    'group5_mrt_oast': {
        'repos': ['MRT-OAST'],
        'models': ['default', 'resnet', 'googlenet', 'densenet']  # 修正：添加default
    },
    'group6_resnet': {
        'repos': ['pytorch_resnet_cifar10'],
        'models': ['resnet20', 'resnet32', 'resnet44']  # 保持不变：匹配正确
    }
}

# 缺失率阈值（提高到90%，因为我们会先回填默认值）
MISSING_THRESHOLD = 0.90

# 超参数默认值配置
DEFAULT_VALUES = {
    'hyperparam_learning_rate': 0.001,
    'hyperparam_batch_size': 32,
    'hyperparam_epochs': 10,
    'hyperparam_dropout': 0.0,
    'hyperparam_weight_decay': 0.0,
    'hyperparam_seed': 42,
    'hyperparam_alpha': 0.1,
    'hyperparam_kfold': 5,
    'hyperparam_max_iter': 100
}

# 模型特定默认值（根据实际配置）
MODEL_SPECIFIC_DEFAULTS = {
    'VulBERTa': {
        'hyperparam_learning_rate': 2e-5,
        'hyperparam_batch_size': 16,
        'hyperparam_epochs': 3
    },
    'Person_reID_baseline_pytorch': {
        'hyperparam_learning_rate': 0.1,
        'hyperparam_batch_size': 64,
        'hyperparam_epochs': 60,
        'hyperparam_weight_decay': 5e-4
    },
    'pytorch_resnet_cifar10': {
        'hyperparam_learning_rate': 0.1,
        'hyperparam_batch_size': 128,
        'hyperparam_epochs': 200,
        'hyperparam_weight_decay': 5e-4
    },
    'examples': {
        'hyperparam_learning_rate': 0.001,
        'hyperparam_batch_size': 32,
        'hyperparam_epochs': 10
    },
    'bug-localization-by-dnn-and-rvsm': {
        'hyperparam_learning_rate': 0.001,
        'hyperparam_batch_size': 32,
        'hyperparam_epochs': 50
    },
    'MRT-OAST': {
        'hyperparam_learning_rate': 0.01,
        'hyperparam_batch_size': 32,
        'hyperparam_epochs': 100
    }
}


def backfill_hyperparam_defaults(df, mode='parallel'):
    """
    回填超参数默认值

    原始数据中，默认配置的超参数记录为NaN，需要根据repository回填默认值

    参数:
        df: 数据框
        mode: 'parallel' 或 'non-parallel'
    """
    df = df.copy()

    print("  回填超参数默认值...")

    # 根据模式选择超参数列前缀
    if mode == 'parallel':
        hyperparam_cols = [col for col in df.columns if col.startswith('fg_hyperparam_')]
        repo_cols = ['fg_repository', 'repository']  # 同时检查fg_repository和repository
    else:
        hyperparam_cols = [col for col in df.columns if col.startswith('hyperparam_')]
        repo_cols = ['repository']

    # 统计回填前的缺失情况
    missing_before = {col: df[col].isna().sum() for col in hyperparam_cols if col in df.columns}

    # 对每个repository应用特定默认值
    for repo_col in repo_cols:
        if repo_col not in df.columns:
            continue

        for repo in df[repo_col].unique():
            if pd.isna(repo):
                continue

            for col in hyperparam_cols:
                if col not in df.columns:
                    continue

                # 找到该repository下该列为空的行
                mask = (df[repo_col] == repo) & (df[col].isna())

                if mask.sum() == 0:
                    continue

                # 从列名中提取参数名（去掉fg_前缀）
                param_name = col.replace('fg_', '')

                # 使用模型特定默认值或通用默认值
                if repo in MODEL_SPECIFIC_DEFAULTS and param_name in MODEL_SPECIFIC_DEFAULTS[repo]:
                    default_val = MODEL_SPECIFIC_DEFAULTS[repo][param_name]
                elif param_name in DEFAULT_VALUES:
                    default_val = DEFAULT_VALUES[param_name]
                else:
                    continue

                df.loc[mask, col] = default_val

    # 统计回填后的缺失情况
    missing_after = {col: df[col].isna().sum() for col in hyperparam_cols if col in df.columns}

    # 输出回填统计
    for col in hyperparam_cols:
        if col in missing_before and col in missing_after:
            filled = missing_before[col] - missing_after[col]
            if filled > 0:
                print(f"    {col}: 回填 {filled} 个默认值 ({missing_before[col]} → {missing_after[col]})")

    return df


def load_data():
    """加载数据"""
    print("加载数据...")
    df = pd.read_csv(DATA_FILE)
    print(f"  总样本数: {len(df)}")

    # 创建is_parallel列
    df['is_parallel'] = (df['mode'] == 'parallel').astype(int)

    # 分组统计
    print(f"\n模式分布:")
    print(f"  并行模式: {df['is_parallel'].sum()} ({df['is_parallel'].sum()/len(df)*100:.1f}%)")
    print(f"  非并行模式: {(~df['is_parallel'].astype(bool)).sum()} ({(~df['is_parallel'].astype(bool)).sum()/len(df)*100:.1f}%)")

    return df


def prepare_task_group_data(df, group_id, group_config, mode='parallel', missing_threshold=0.40):
    """
    准备单个任务组的DiBS数据（指定模式）

    mode: 'parallel' 或 'non-parallel'
    """

    # 过滤模式
    if mode == 'parallel':
        mode_df = df[df['is_parallel'] == 1].copy()
    else:
        mode_df = df[df['is_parallel'] == 0].copy()

    # 过滤仓库和模型
    repos = group_config['repos']
    models = group_config['models']

    # 对于并行模式，需要检查fg_repository和bg_repository
    if mode == 'parallel':
        # 并行模式：检查fg_repository或bg_repository
        repo_mask = (
            (mode_df['fg_repository'].isin(repos)) |
            (mode_df['bg_repository'].isin(repos)) |
            (mode_df['repository'].isin(repos))
        )
        model_mask = (
            (mode_df['fg_model'].isin(models)) |
            (mode_df['bg_model'].isin(models)) |
            (mode_df['model'].isin(models))
        )
        group_df = mode_df[repo_mask & model_mask].copy()
    else:
        # 非并行模式：只检查repository和model
        group_df = mode_df[
            (mode_df['repository'].isin(repos)) &
            (mode_df['model'].isin(models))
        ].copy()

    if len(group_df) == 0:
        return None, f"没有{mode}模式数据"

    print(f"  原始样本数: {len(group_df)}")

    # **关键步骤：回填超参数默认值**（传入mode参数）
    group_df = backfill_hyperparam_defaults(group_df, mode=mode)

    # 选择特征列（根据模式选择正确的列前缀）
    if mode == 'parallel':
        # 并行模式：
        # - 超参数和能耗使用fg_前缀
        # - 性能数据仍使用perf_前缀（不是fg_perf_，因为fg_perf_*全为空）
        hyperparam_cols = [col for col in group_df.columns if col.startswith('fg_hyperparam_')]
        perf_cols = [col for col in group_df.columns if col.startswith('perf_') and not col.startswith('fg_perf_') and not col.startswith('bg_perf_')]
        energy_cols = [col for col in group_df.columns if col.startswith('fg_energy_')]
        control_cols = ['fg_duration_seconds']
    else:
        # 非并行模式：使用非fg_前缀的列
        hyperparam_cols = [col for col in group_df.columns if col.startswith('hyperparam_')]
        perf_cols = [col for col in group_df.columns if col.startswith('perf_')]
        energy_cols = [col for col in group_df.columns if col.startswith('energy_')]
        control_cols = ['duration_seconds']

    all_cols = hyperparam_cols + perf_cols + energy_cols + control_cols

    # 确保列存在
    available_cols = [col for col in all_cols if col in group_df.columns]

    data = group_df[available_cols].copy()

    # 移除缺失率过高的列
    missing_rates = data.isnull().sum() / len(data)
    cols_to_keep = missing_rates[missing_rates <= missing_threshold].index.tolist()

    removed_cols = set(available_cols) - set(cols_to_keep)

    data = data[cols_to_keep]

    # 移除仍有缺失值的行
    data_clean = data.dropna()

    # 如果是并行模式，重命名列（去掉fg_前缀）以便统一后续分析
    if mode == 'parallel':
        rename_dict = {col: col.replace('fg_', '') for col in data_clean.columns if col.startswith('fg_')}
        data_clean = data_clean.rename(columns=rename_dict)

    # 重新提取列名（重命名后的）
    final_cols = data_clean.columns.tolist()
    final_hyperparams = [col for col in final_cols if col.startswith('hyperparam_')]
    final_perf = [col for col in final_cols if col.startswith('perf_')]
    final_energy = [col for col in final_cols if col.startswith('energy_')]
    final_controls = [col for col in final_cols if col == 'duration_seconds']

    stats = {
        'group': group_id,
        'mode': mode,
        'total_samples': len(group_df),
        'samples_after_cleaning': len(data_clean),
        'total_features': len(available_cols),
        'features_after_filtering': len(cols_to_keep),
        'removed_features': len(removed_cols),
        'removed_feature_names': list(removed_cols),
        'missing_threshold': missing_threshold,
        'hyperparams': final_hyperparams,
        'performance': final_perf,
        'energy': final_energy,
        'controls': final_controls
    }

    return data_clean, stats


def main():
    """主函数"""

    print("="*80)
    print("准备分层DiBS训练数据（按模式）")
    print("="*80)

    # 加载数据
    df = load_data()

    # 为每个模式和任务组准备数据
    all_stats = {
        'parallel': {},
        'non_parallel': {}
    }

    for mode, output_dir in [('parallel', OUTPUT_DIR_PARALLEL),
                              ('non_parallel', OUTPUT_DIR_NON_PARALLEL)]:

        print(f"\n{'='*80}")
        print(f"处理模式: {mode.upper()}")
        print(f"{'='*80}")

        for group_id, group_config in TASK_GROUPS.items():
            print(f"\n处理: {group_id}")

            data, stats = prepare_task_group_data(
                df, group_id, group_config,
                mode=mode,
                missing_threshold=MISSING_THRESHOLD
            )

            if data is None:
                print(f"  ❌ {stats}")
                all_stats[mode][group_id] = {'status': 'failed', 'reason': stats}
                continue

            # 保存数据
            output_file = output_dir / f"{group_id}.csv"
            data.to_csv(output_file, index=False)

            # 保存统计
            all_stats[mode][group_id] = stats

            print(f"  ✅ 样本数: {stats['total_samples']} → {stats['samples_after_cleaning']} (清洗后)")
            print(f"     特征数: {stats['total_features']} → {stats['features_after_filtering']} (过滤后)")
            print(f"     - 超参数: {len(stats['hyperparams'])}")
            print(f"     - 性能: {len(stats['performance'])}")
            print(f"     - 能耗: {len(stats['energy'])}")
            print(f"     保存至: {output_file}")

            # 检查样本量警告
            if stats['samples_after_cleaning'] < 30:
                print(f"  ⚠️  警告: 样本量过少 (n={stats['samples_after_cleaning']}), DiBS结果可能不稳定")
            elif stats['samples_after_cleaning'] < 50:
                print(f"  ⚠️  警告: 样本量偏少 (n={stats['samples_after_cleaning']}), 建议增加实验")

    # 保存汇总统计
    summary_file_parallel = OUTPUT_DIR_PARALLEL / "data_preparation_summary.json"
    summary_file_non_parallel = OUTPUT_DIR_NON_PARALLEL / "data_preparation_summary.json"

    with open(summary_file_parallel, 'w') as f:
        json.dump(all_stats['parallel'], f, indent=2)

    with open(summary_file_non_parallel, 'w') as f:
        json.dump(all_stats['non_parallel'], f, indent=2)

    print("\n" + "="*80)
    print("✅ 数据准备完成！")
    print("="*80)

    # 对比统计
    print("\n并行 vs 非并行样本量对比:\n")
    print(f"{'任务组':<25} {'并行':<10} {'非并行':<10} {'总计':<10}")
    print("-" * 60)

    for group_id in TASK_GROUPS.keys():
        parallel_samples = all_stats['parallel'].get(group_id, {}).get('samples_after_cleaning', 0)
        non_parallel_samples = all_stats['non_parallel'].get(group_id, {}).get('samples_after_cleaning', 0)
        total = parallel_samples + non_parallel_samples

        print(f"{group_id:<25} {parallel_samples:<10} {non_parallel_samples:<10} {total:<10}")

    print("\n输出目录:")
    print(f"  并行模式: {OUTPUT_DIR_PARALLEL}")
    print(f"  非并行模式: {OUTPUT_DIR_NON_PARALLEL}")
    print("="*80)


if __name__ == "__main__":
    main()
