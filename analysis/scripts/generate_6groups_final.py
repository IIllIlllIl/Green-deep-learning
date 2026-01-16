#!/usr/bin/env python3
"""
生成6分组数据 - 最终版本

功能:
1. 语义超参数合并 (alpha ≡ weight_decay → l2_regularization)
2. 按共用超参数和性能指标分组
3. 保留所有818条可用数据 (无缺失率阈值)
4. 添加模型变量 (One-hot n-1编码)

作者: Energy DL Project
日期: 2026-01-15
版本: 1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def unify_semantic_hyperparams(df):
    """
    统一语义相同但名称不同的超参数

    当前合并:
    - alpha ≡ weight_decay → l2_regularization (L2正则化)

    参数:
        df: 原始数据框

    返回:
        df: 合并后的数据框
    """
    df = df.copy()

    # L2正则化合并
    if 'hyperparam_alpha' in df.columns and 'hyperparam_weight_decay' in df.columns:
        df['hyperparam_l2_regularization'] = df['hyperparam_weight_decay'].fillna(
            df['hyperparam_alpha']
        )
        print(f"✅ L2正则化合并完成:")
        print(f"   - hyperparam_alpha 缺失率: {df['hyperparam_alpha'].isnull().mean()*100:.1f}%")
        print(f"   - hyperparam_weight_decay 缺失率: {df['hyperparam_weight_decay'].isnull().mean()*100:.1f}%")
        print(f"   - hyperparam_l2_regularization 缺失率: {df['hyperparam_l2_regularization'].isnull().mean()*100:.1f}%")

    return df


def add_model_variables(df, drop_first=True):
    """
    添加模型变量 (One-hot编码)

    参数:
        df: 数据框
        drop_first: 是否使用n-1编码 (去掉第一个模型作为基准)

    返回:
        df: 添加模型变量后的数据框
        model_vars: 模型变量名列表
    """
    df = df.copy()

    if 'model' not in df.columns:
        print("⚠️  警告: 数据中没有'model'列,跳过模型变量添加")
        return df, []

    # One-hot编码
    model_dummies = pd.get_dummies(df['model'], prefix='model', drop_first=drop_first)
    model_vars = model_dummies.columns.tolist()

    # 合并到原数据
    df = pd.concat([df, model_dummies], axis=1)

    if drop_first:
        baseline_model = df['model'].iloc[0] if len(df) > 0 else 'unknown'
        print(f"✅ 模型变量添加完成 (n-1编码, 基准: {baseline_model}):")
    else:
        print(f"✅ 模型变量添加完成 (完整编码):")

    print(f"   - 模型变量: {model_vars}")

    return df, model_vars


def filter_usable_data(df):
    """
    筛选可用数据: 训练成功 + 有能耗 + 有性能指标

    参数:
        df: 原始数据框

    返回:
        df_usable: 可用数据框
    """
    print("\n" + "="*60)
    print("筛选可用数据")
    print("="*60)

    print(f"原始数据: {len(df)} 条")

    # 1. 训练成功 (data.csv使用training_success列)
    if 'training_success' in df.columns:
        df_success = df[df['training_success'] == True].copy()
    elif 'status' in df.columns:
        df_success = df[df['status'] == 'success'].copy()
    else:
        print("⚠️  警告: 找不到训练状态列,使用所有数据")
        df_success = df.copy()

    print(f"训练成功: {len(df_success)} 条 ({len(df_success)/len(df)*100:.1f}%)")

    # 2. 有能耗数据
    energy_cols = [col for col in df.columns if col.startswith('energy_')]
    has_energy = ~df_success[energy_cols].isnull().all(axis=1)
    df_with_energy = df_success[has_energy].copy()
    print(f"有能耗数据: {len(df_with_energy)} 条 ({len(df_with_energy)/len(df)*100:.1f}%)")

    # 3. 有性能指标
    perf_cols = [col for col in df.columns if col.startswith('perf_')]
    has_perf = ~df_with_energy[perf_cols].isnull().all(axis=1)
    df_usable = df_with_energy[has_perf].copy()
    print(f"有性能指标: {len(df_usable)} 条 ({len(df_usable)/len(df)*100:.1f}%)")

    print(f"\n✅ 最终可用数据: {len(df_usable)} 条")

    return df_usable


def define_group_configs():
    """
    定义6个分组的配置

    每组包含:
    - repository: 仓库名称
    - models: 模型列表
    - hyperparams: 该组使用的超参数列表
    - perf_metrics: 该组使用的性能指标列表

    返回:
        group_configs: 分组配置字典
    """
    group_configs = {
        'group1_examples': {
            'repository': 'examples',
            'models': ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese'],
            'hyperparams': [
                'hyperparam_batch_size',
                'hyperparam_learning_rate',
                'hyperparam_epochs',
                'hyperparam_seed'
            ],
            'perf_metrics': ['perf_test_accuracy']
        },

        'group2_vulberta': {
            'repository': 'VulBERTa',
            'models': ['mlp'],
            'hyperparams': [
                'hyperparam_learning_rate',
                'hyperparam_epochs',
                'hyperparam_seed',
                'hyperparam_l2_regularization'  # 使用合并后的L2正则化
            ],
            'perf_metrics': [
                'perf_eval_loss',
                'perf_final_training_loss',
                'perf_eval_samples_per_second'
            ]
        },

        'group3_person_reid': {
            'repository': 'Person_reID_baseline_pytorch',
            'models': ['densenet121', 'hrnet18', 'pcb'],
            'hyperparams': [
                'hyperparam_dropout',
                'hyperparam_learning_rate',
                'hyperparam_epochs',
                'hyperparam_seed'
            ],
            'perf_metrics': ['perf_map', 'perf_rank1', 'perf_rank5']
        },

        'group4_bug_localization': {
            'repository': 'bug-localization-by-dnn-and-rvsm',
            'models': ['default'],
            'hyperparams': [
                'hyperparam_alpha',
                'hyperparam_kfold',
                'hyperparam_max_iter',
                'hyperparam_seed'
            ],
            'perf_metrics': [
                'perf_top1_accuracy',
                'perf_top5_accuracy',
                'perf_top10_accuracy',
                'perf_top20_accuracy'
            ]
        },

        'group5_mrt_oast': {
            'repository': 'MRT-OAST',
            'models': ['default'],
            'hyperparams': [
                'hyperparam_dropout',
                'hyperparam_learning_rate',
                'hyperparam_epochs',
                'hyperparam_seed',
                'hyperparam_l2_regularization'  # 使用合并后的L2正则化
            ],
            'perf_metrics': [
                'perf_accuracy',
                'perf_precision',
                'perf_recall'
            ]
        },

        'group6_resnet': {
            'repository': 'pytorch_resnet_cifar10',
            'models': ['resnet20'],
            'hyperparams': [
                'hyperparam_learning_rate',
                'hyperparam_epochs',
                'hyperparam_seed',
                'hyperparam_l2_regularization'  # 使用合并后的L2正则化
            ],
            'perf_metrics': [
                'perf_best_val_accuracy',
                'perf_test_accuracy'
            ]
        }
    }

    return group_configs


def generate_group_data(df_usable, group_name, group_config, output_dir):
    """
    生成单个分组的数据

    参数:
        df_usable: 可用数据框
        group_name: 分组名称
        group_config: 分组配置
        output_dir: 输出目录

    返回:
        group_df: 分组数据框
        stats: 统计信息字典
    """
    print("\n" + "="*60)
    print(f"生成 {group_name}")
    print("="*60)

    # 1. 筛选该组的模型 (使用repository+model)
    repository = group_config['repository']
    models = group_config['models']
    group_df = df_usable[
        (df_usable['repository'] == repository) &
        (df_usable['model'].isin(models))
    ].copy()
    print(f"仓库: {repository}")
    print(f"模型: {models}")
    print(f"数据行数: {len(group_df)}")

    if len(group_df) == 0:
        print(f"⚠️  警告: {group_name} 没有数据!")
        return None, None

    # 2. 获取能耗列
    energy_cols = [col for col in group_df.columns if col.startswith('energy_')]

    # 3. 获取控制变量
    control_cols = ['is_parallel', 'timestamp']

    # 4. 获取该组使用的超参数
    hyperparams = group_config['hyperparams']
    available_hyperparams = [hp for hp in hyperparams if hp in group_df.columns]
    if len(available_hyperparams) < len(hyperparams):
        missing = set(hyperparams) - set(available_hyperparams)
        print(f"⚠️  警告: 缺少超参数列: {missing}")

    # 5. 获取该组使用的性能指标
    perf_metrics = group_config['perf_metrics']
    available_perf = [pm for pm in perf_metrics if pm in group_df.columns]
    if len(available_perf) < len(perf_metrics):
        missing = set(perf_metrics) - set(available_perf)
        print(f"⚠️  警告: 缺少性能指标列: {missing}")

    # 6. 添加模型变量 (One-hot n-1编码)
    group_df, model_vars = add_model_variables(group_df, drop_first=True)

    # 7. 选择最终列
    selected_cols = (
        energy_cols +
        control_cols +
        model_vars +
        available_hyperparams +
        available_perf
    )

    group_df_final = group_df[selected_cols].copy()

    # 8. 统计信息
    stats = {
        'group_name': group_name,
        'n_models': len(models),
        'n_rows': len(group_df_final),
        'n_cols': len(selected_cols),
        'models': models,
        'model_vars': model_vars,
        'hyperparams': available_hyperparams,
        'perf_metrics': available_perf
    }

    # 9. 保存数据
    output_path = output_dir / f"{group_name}.csv"
    group_df_final.to_csv(output_path, index=False)
    print(f"✅ 保存到: {output_path}")
    print(f"   - 行数: {len(group_df_final)}")
    print(f"   - 列数: {len(selected_cols)}")

    return group_df_final, stats


def main():
    """主函数"""
    print("="*60)
    print("生成6分组数据 - 最终版本")
    print("="*60)

    # 1. 设置路径
    project_root = Path(__file__).resolve().parents[2]
    data_file = project_root / 'data' / 'data.csv'
    output_dir = project_root / 'analysis' / 'data' / 'energy_research' / '6groups_final'

    print(f"\n输入文件: {data_file}")
    print(f"输出目录: {output_dir}")

    # 2. 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 加载数据
    print("\n" + "="*60)
    print("加载数据")
    print("="*60)
    df = pd.read_csv(data_file)
    print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")

    # 4. 语义超参数合并
    print("\n" + "="*60)
    print("语义超参数合并")
    print("="*60)
    df = unify_semantic_hyperparams(df)

    # 5. 筛选可用数据
    df_usable = filter_usable_data(df)

    # 6. 获取分组配置
    print("\n" + "="*60)
    print("生成6个分组")
    print("="*60)
    group_configs = define_group_configs()

    # 7. 生成每个分组的数据
    all_stats = []
    for group_name, group_config in group_configs.items():
        group_df, stats = generate_group_data(df_usable, group_name, group_config, output_dir)
        if stats:
            all_stats.append(stats)

    # 8. 生成总结报告
    print("\n" + "="*60)
    print("总结报告")
    print("="*60)

    total_rows = sum(s['n_rows'] for s in all_stats)
    print(f"\n总数据行数: {total_rows}/{len(df_usable)} ({total_rows/len(df_usable)*100:.1f}%)")
    print(f"生成分组数: {len(all_stats)}")

    for stats in all_stats:
        print(f"\n{stats['group_name']}:")
        print(f"  - 模型数: {stats['n_models']}")
        print(f"  - 数据行数: {stats['n_rows']}")
        print(f"  - 列数: {stats['n_cols']}")
        print(f"  - 模型: {stats['models']}")
        print(f"  - 模型变量: {stats['model_vars']}")

    # 9. 保存统计信息
    stats_file = output_dir / 'generation_stats.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("6分组数据生成统计\n")
        f.write("="*60 + "\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"输入文件: {data_file}\n")
        f.write(f"可用数据: {len(df_usable)} 行\n")
        f.write(f"总数据行数: {total_rows} ({total_rows/len(df_usable)*100:.1f}%)\n\n")

        for stats in all_stats:
            f.write(f"\n{stats['group_name']}:\n")
            f.write(f"  模型数: {stats['n_models']}\n")
            f.write(f"  数据行数: {stats['n_rows']}\n")
            f.write(f"  列数: {stats['n_cols']}\n")
            f.write(f"  模型: {stats['models']}\n")
            f.write(f"  模型变量: {stats['model_vars']}\n")
            f.write(f"  超参数: {stats['hyperparams']}\n")
            f.write(f"  性能指标: {stats['perf_metrics']}\n")

    print(f"\n✅ 统计信息已保存到: {stats_file}")
    print("\n" + "="*60)
    print("完成!")
    print("="*60)


if __name__ == '__main__':
    main()
