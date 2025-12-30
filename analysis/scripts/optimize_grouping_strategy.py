#!/usr/bin/env python3
"""探究最优分组策略以保留最多数据

问题：不同模型支持不同的超参数集合，当前任务分组产生超参数冲突，导致数据删除
目标：找出能保留最多数据的分组方案

作者: Claude
日期: 2025-12-24
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations


def load_models_config():
    """加载models_config.json"""
    config_path = Path('../../mutation/models_config.json')
    with open(config_path) as f:
        return json.load(f)


def extract_hyperparams_per_model(models_config):
    """
    提取每个模型支持的超参数集合

    Returns:
        dict: {(repository, model): set(hyperparams)}
    """
    print("=" * 80)
    print("【步骤1】提取每个模型支持的超参数集合")
    print("=" * 80)

    model_hyperparams = {}

    for repo_name, repo_config in models_config['models'].items():
        supported_params = repo_config.get('supported_hyperparams', {})

        # 将原始参数名映射到统一名称
        unified_params = set()
        for param in supported_params.keys():
            if param in ['epochs', 'max_iter']:
                unified_params.add('training_duration')
            elif param in ['learning_rate', 'lr']:
                unified_params.add('hyperparam_learning_rate')
            elif param in ['weight_decay', 'alpha']:
                unified_params.add('l2_regularization')
            elif param == 'batch_size':
                unified_params.add('hyperparam_batch_size')
            elif param == 'dropout':
                unified_params.add('hyperparam_dropout')
            elif param == 'seed':
                unified_params.add('seed')
            elif param == 'kfold':
                unified_params.add('hyperparam_kfold')
            else:
                unified_params.add(f'hyperparam_{param}')

        # 对于每个模型
        models = repo_config.get('models', [])
        for model_name in models:
            model_hyperparams[(repo_name, model_name)] = unified_params

        print(f"\n{repo_name}:")
        print(f"  模型数: {len(models)}")
        print(f"  支持超参数: {sorted(unified_params)}")

    return model_hyperparams


def load_original_data():
    """加载原始数据"""
    print("\n" + "=" * 80)
    print("【步骤2】加载原始数据并分析每个模型的实验数量")
    print("=" * 80)

    data_path = Path('../data/energy_research/raw/energy_data_original.csv')
    df = pd.read_csv(data_path)

    print(f"\n总行数: {len(df)}")

    # 统计每个repository+model的数据量
    model_counts = df.groupby(['repository', 'model']).size().to_dict()

    print("\n每个模型的实验数量:")
    for (repo, model), count in sorted(model_counts.items()):
        print(f"  {repo}/{model}: {count} 个实验")

    return df, model_counts


def analyze_current_grouping_conflicts(df, model_hyperparams):
    """
    分析当前4个任务组的超参数冲突情况
    """
    print("\n" + "=" * 80)
    print("【步骤3】分析当前分组的超参数冲突")
    print("=" * 80)

    # 当前4个任务组定义
    current_groups = {
        'image_classification': {
            'repos_models': [
                ('examples', 'mnist'),
                ('examples', 'mnist_ff'),
                ('examples', 'mnist_rnn'),
                ('examples', 'siamese'),
                ('pytorch_resnet_cifar10', 'resnet20')
            ]
        },
        'person_reid': {
            'repos_models': [
                ('Person_reID_baseline_pytorch', 'densenet121'),
                ('Person_reID_baseline_pytorch', 'hrnet18'),
                ('Person_reID_baseline_pytorch', 'pcb')
            ]
        },
        'vulberta': {
            'repos_models': [
                ('VulBERTa', 'mlp')
            ]
        },
        'bug_localization': {
            'repos_models': [
                ('bug-localization-by-dnn-and-rvsm', 'default')
            ]
        }
    }

    conflict_analysis = {}

    for group_name, group_config in current_groups.items():
        print(f"\n{group_name}:")

        repos_models = group_config['repos_models']

        # 收集该组所有模型的超参数集合
        all_hyperparams = []
        for repo, model in repos_models:
            params = model_hyperparams.get((repo, model), set())
            all_hyperparams.append(params)
            print(f"  {repo}/{model}: {sorted(params)}")

        # 找出交集（所有模型都支持的参数）
        if all_hyperparams:
            common_params = set.intersection(*all_hyperparams)
            all_params_union = set.union(*all_hyperparams)
            conflict_params = all_params_union - common_params

            print(f"\n  共同支持的超参数 ({len(common_params)}个): {sorted(common_params)}")
            if conflict_params:
                print(f"  ⚠️ 冲突的超参数 ({len(conflict_params)}个): {sorted(conflict_params)}")

                # 详细分析每个冲突参数
                for param in sorted(conflict_params):
                    supporting_models = [f"{r}/{m}" for r, m in repos_models
                                        if param in model_hyperparams.get((r, m), set())]
                    not_supporting = [f"{r}/{m}" for r, m in repos_models
                                     if param not in model_hyperparams.get((r, m), set())]
                    print(f"    {param}:")
                    print(f"      支持: {supporting_models}")
                    print(f"      不支持: {not_supporting}")

            conflict_analysis[group_name] = {
                'common_params': common_params,
                'conflict_params': conflict_params,
                'total_models': len(repos_models)
            }

    return conflict_analysis, current_groups


def calculate_data_retention(df, grouping_config, model_hyperparams):
    """
    计算给定分组配置的数据保留率

    Args:
        df: 原始数据
        grouping_config: 分组配置 {group_name: {'repos_models': [(repo, model), ...]}}
        model_hyperparams: 每个模型支持的超参数

    Returns:
        dict: 每个组的数据保留统计
    """
    retention_stats = {}

    for group_name, group_config in grouping_config.items():
        repos_models = group_config['repos_models']

        # 筛选属于该组的数据
        mask = df.apply(
            lambda row: (row['repository'], row['model']) in repos_models,
            axis=1
        )
        df_group = df[mask].copy()

        # 确定该组的共同超参数
        hyperparam_sets = [model_hyperparams.get((repo, model), set())
                          for repo, model in repos_models]
        if hyperparam_sets:
            common_params = set.intersection(*hyperparam_sets)
        else:
            common_params = set()

        # 统计数据
        total_rows = len(df_group)

        # 检查性能指标缺失（需要根据任务定义）
        # 简化：假设每个组都有特定的性能指标
        perf_cols = [c for c in df_group.columns if c.startswith('perf_')]

        # 删除性能全缺失的行
        if perf_cols:
            perf_valid_mask = df_group[perf_cols].notna().any(axis=1)
            retained_rows = perf_valid_mask.sum()
        else:
            retained_rows = total_rows

        retention_stats[group_name] = {
            'total_rows': total_rows,
            'retained_rows': retained_rows,
            'retention_rate': retained_rows / total_rows * 100 if total_rows > 0 else 0,
            'common_params': common_params,
            'num_models': len(repos_models)
        }

    return retention_stats


def propose_alternative_grouping_strategies(df, model_hyperparams, model_counts):
    """
    提出替代分组策略

    策略1: 按超参数集合聚类（完全消除冲突）
    策略2: 在任务内按超参数集合子分组
    策略3: 按模型单独分组（无冲突但样本量小）
    """
    print("\n" + "=" * 80)
    print("【步骤4】提出替代分组策略")
    print("=" * 80)

    # 策略1: 按超参数集合聚类
    print("\n策略1: 按超参数集合聚类（完全消除超参数冲突）")
    print("-" * 80)

    # 将模型按超参数集合分组
    hyperparam_signature_groups = defaultdict(list)
    for (repo, model), params in model_hyperparams.items():
        # 使用冻结集合作为key
        signature = frozenset(params)
        hyperparam_signature_groups[signature].append((repo, model))

    strategy1_groups = {}
    for idx, (signature, models) in enumerate(hyperparam_signature_groups.items(), 1):
        group_name = f"hyperparam_group_{idx}"

        # 计算该组的数据量
        total_experiments = sum(model_counts.get(model, 0) for model in models)

        print(f"\n{group_name}:")
        print(f"  超参数集合: {sorted(signature)}")
        print(f"  模型数: {len(models)}")
        print(f"  总实验数: {total_experiments}")
        print(f"  包含模型:")
        for repo, model in models:
            count = model_counts.get((repo, model), 0)
            print(f"    - {repo}/{model}: {count} 个实验")

        strategy1_groups[group_name] = {
            'repos_models': models,
            'strategy': 'by_hyperparam_signature'
        }

    # 策略2: 在任务内按超参数集合子分组
    print("\n\n策略2: 在任务内按超参数集合子分组（保持任务同质性）")
    print("-" * 80)

    strategy2_groups = {}

    # 图像分类任务内分组
    print("\n图像分类任务子分组:")

    # examples模型（不支持weight_decay）
    examples_models = [('examples', m) for m in ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese']]
    examples_total = sum(model_counts.get(m, 0) for m in examples_models)

    strategy2_groups['image_classification_examples'] = {
        'repos_models': examples_models,
        'strategy': 'task_then_hyperparam'
    }

    print(f"  image_classification_examples:")
    print(f"    模型: {[f'{r}/{m}' for r, m in examples_models]}")
    print(f"    实验数: {examples_total}")
    print(f"    超参数: {sorted(model_hyperparams.get(examples_models[0], set()))}")

    # pytorch_resnet_cifar10模型（支持weight_decay）
    cifar_models = [('pytorch_resnet_cifar10', 'resnet20')]
    cifar_total = sum(model_counts.get(m, 0) for m in cifar_models)

    strategy2_groups['image_classification_resnet'] = {
        'repos_models': cifar_models,
        'strategy': 'task_then_hyperparam'
    }

    print(f"\n  image_classification_resnet:")
    print(f"    模型: {[f'{r}/{m}' for r, m in cifar_models]}")
    print(f"    实验数: {cifar_total}")
    print(f"    超参数: {sorted(model_hyperparams.get(cifar_models[0], set()))}")

    # Person_reID任务内分组（检查是否有超参数差异）
    print("\n\nPerson_reID任务子分组:")
    person_reid_models = [
        ('Person_reID_baseline_pytorch', 'densenet121'),
        ('Person_reID_baseline_pytorch', 'hrnet18'),
        ('Person_reID_baseline_pytorch', 'pcb')
    ]

    # 检查这3个模型是否有超参数差异
    person_reid_params = [model_hyperparams.get(m, set()) for m in person_reid_models]
    person_reid_common = set.intersection(*person_reid_params) if person_reid_params else set()
    person_reid_all = set.union(*person_reid_params) if person_reid_params else set()
    person_reid_conflict = person_reid_all - person_reid_common

    if person_reid_conflict:
        print(f"  ⚠️ Person_reID模型间有超参数冲突: {sorted(person_reid_conflict)}")
        # 需要进一步子分组
    else:
        print(f"  ✅ Person_reID模型间无超参数冲突，保持一组")
        strategy2_groups['person_reid'] = {
            'repos_models': person_reid_models,
            'strategy': 'task_then_hyperparam'
        }

    # VulBERTa和Bug定位（单模型，无需子分组）
    strategy2_groups['vulberta'] = {
        'repos_models': [('VulBERTa', 'mlp')],
        'strategy': 'task_then_hyperparam'
    }

    strategy2_groups['bug_localization'] = {
        'repos_models': [('bug-localization-by-dnn-and-rvsm', 'default')],
        'strategy': 'task_then_hyperparam'
    }

    # 策略3: 按模型单独分组（保证无冲突但样本量小）
    print("\n\n策略3: 按模型单独分组（无冲突但样本量可能较小）")
    print("-" * 80)

    strategy3_groups = {}
    for (repo, model), count in sorted(model_counts.items()):
        group_name = f"{repo}_{model}".replace('-', '_').replace('/', '_')
        strategy3_groups[group_name] = {
            'repos_models': [(repo, model)],
            'strategy': 'by_model'
        }
        print(f"  {group_name}: {count} 个实验, 超参数: {sorted(model_hyperparams.get((repo, model), set()))}")

    return {
        'strategy1_by_hyperparam': strategy1_groups,
        'strategy2_task_then_hyperparam': strategy2_groups,
        'strategy3_by_model': strategy3_groups
    }


def compare_strategies(df, strategies, model_hyperparams):
    """
    比较所有策略的数据保留率
    """
    print("\n" + "=" * 80)
    print("【步骤5】比较各策略的数据保留率")
    print("=" * 80)

    comparison_results = {}

    for strategy_name, grouping_config in strategies.items():
        print(f"\n{strategy_name}:")
        print("=" * 80)

        retention_stats = calculate_data_retention(df, grouping_config, model_hyperparams)

        total_retained = sum(stats['retained_rows'] for stats in retention_stats.values())
        total_original = sum(stats['total_rows'] for stats in retention_stats.values())
        overall_retention = total_retained / total_original * 100 if total_original > 0 else 0

        comparison_results[strategy_name] = {
            'total_retained': total_retained,
            'total_original': total_original,
            'overall_retention_rate': overall_retention,
            'num_groups': len(grouping_config),
            'group_details': retention_stats
        }

        print(f"\n总体统计:")
        print(f"  原始总数: {total_original} 行")
        print(f"  保留总数: {total_retained} 行")
        print(f"  总体保留率: {overall_retention:.1f}%")
        print(f"  分组数: {len(grouping_config)}")

        print(f"\n各组详情:")
        for group_name, stats in retention_stats.items():
            print(f"  {group_name}:")
            print(f"    原始: {stats['total_rows']} 行")
            print(f"    保留: {stats['retained_rows']} 行")
            print(f"    保留率: {stats['retention_rate']:.1f}%")
            print(f"    模型数: {stats['num_models']}")
            print(f"    共同超参数数: {len(stats['common_params'])}")

    return comparison_results


def generate_recommendation(comparison_results, df):
    """
    生成最优分组建议
    """
    print("\n" + "=" * 80)
    print("【步骤6】生成最优分组建议")
    print("=" * 80)

    # 按保留率排序
    sorted_strategies = sorted(
        comparison_results.items(),
        key=lambda x: x[1]['overall_retention_rate'],
        reverse=True
    )

    print("\n策略排名（按数据保留率）:")
    for rank, (strategy_name, results) in enumerate(sorted_strategies, 1):
        print(f"{rank}. {strategy_name}:")
        print(f"   保留率: {results['overall_retention_rate']:.1f}%")
        print(f"   保留行数: {results['total_retained']}/{results['total_original']}")
        print(f"   分组数: {results['num_groups']}")

    best_strategy_name, best_results = sorted_strategies[0]

    print("\n" + "=" * 80)
    print("【推荐方案】")
    print("=" * 80)
    print(f"\n最优策略: {best_strategy_name}")
    print(f"数据保留率: {best_results['overall_retention_rate']:.1f}%")
    print(f"相比当前方案（372行，51.2%）的改进: +{best_results['total_retained'] - 372} 行")

    # 详细分组建议
    print("\n建议分组配置:")
    for group_name, stats in best_results['group_details'].items():
        print(f"\n  {group_name}:")
        print(f"    样本数: {stats['retained_rows']}")
        print(f"    模型数: {stats['num_models']}")
        print(f"    超参数数: {len(stats['common_params'])}")
        print(f"    超参数列表: {sorted(stats['common_params'])}")

    # DiBS适用性评估
    print("\n" + "=" * 80)
    print("【DiBS适用性评估】")
    print("=" * 80)

    groups_below_threshold = [
        (name, stats['retained_rows'])
        for name, stats in best_results['group_details'].items()
        if stats['retained_rows'] < 30
    ]

    if groups_below_threshold:
        print("\n⚠️ 以下分组样本量较小（<30）:")
        for group_name, count in groups_below_threshold:
            print(f"  {group_name}: {count} 个样本")
        print("\n建议:")
        print("  - 样本量 < 10: 不适合DiBS（无法学习因果图）")
        print("  - 样本量 10-30: 可以运行但统计功效较低")
        print("  - 样本量 > 30: 适合DiBS因果分析")
    else:
        print("\n✅ 所有分组样本量充足（≥30），适合DiBS因果分析")

    return best_strategy_name, best_results


def save_results(comparison_results, best_strategy_name, strategies):
    """保存结果"""
    output_dir = Path('../docs/reports')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细统计
    results_to_save = {}
    for strategy_name, results in comparison_results.items():
        results_copy = {
            'total_retained': int(results['total_retained']),
            'total_original': int(results['total_original']),
            'overall_retention_rate': float(results['overall_retention_rate']),
            'num_groups': int(results['num_groups']),
            'group_details': {}
        }
        # 转换set为list，int64为int以便JSON序列化
        for group_name, stats in results['group_details'].items():
            results_copy['group_details'][group_name] = {
                'total_rows': int(stats['total_rows']),
                'retained_rows': int(stats['retained_rows']),
                'retention_rate': float(stats['retention_rate']),
                'num_models': int(stats['num_models']),
                'common_params': sorted(stats['common_params'])
            }
        results_to_save[strategy_name] = results_copy

    output_file = output_dir / 'GROUPING_OPTIMIZATION_RESULTS_20251224.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'comparison_results': results_to_save,
            'best_strategy': best_strategy_name,
            'recommendation': f"Use {best_strategy_name} for maximum data retention"
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_file}")

    # 保存推荐配置（Python格式，可直接用于脚本）
    best_config = strategies[best_strategy_name]

    config_output = output_dir / 'RECOMMENDED_GROUPING_CONFIG.py'
    with open(config_output, 'w', encoding='utf-8') as f:
        f.write(f"""# 推荐的分组配置
# 生成时间: 2025-12-24
# 策略: {best_strategy_name}
# 数据保留率: {comparison_results[best_strategy_name]['overall_retention_rate']:.1f}%

TASK_GROUPS = {{
""")
        for group_name, group_config in best_config.items():
            repos_models = group_config['repos_models']

            # 提取仓库和模型
            repos_dict = defaultdict(list)
            for repo, model in repos_models:
                repos_dict[repo].append(model)

            f.write(f"    '{group_name}': {{\n")
            f.write(f"        'repos': {list(repos_dict.keys())},\n")
            f.write(f"        'models': {{\n")
            for repo, models in repos_dict.items():
                f.write(f"            '{repo}': {models},\n")
            f.write(f"        }},\n")
            f.write(f"    }},\n")

        f.write("}\n")

    print(f"✅ 推荐配置已保存: {config_output}")


def main():
    """主函数"""
    print("=" * 80)
    print("探究最优分组策略以保留最多数据")
    print("=" * 80)
    print("目标: 找出能最大化数据保留率的分组方案")
    print("=" * 80)

    # 加载配置和数据
    models_config = load_models_config()
    model_hyperparams = extract_hyperparams_per_model(models_config)
    df, model_counts = load_original_data()

    # 分析当前分组冲突
    conflict_analysis, current_groups = analyze_current_grouping_conflicts(df, model_hyperparams)

    # 计算当前方案的保留率
    print("\n" + "=" * 80)
    print("【当前方案（4个任务组）】")
    print("=" * 80)
    current_retention = calculate_data_retention(df, current_groups, model_hyperparams)
    current_total = sum(stats['retained_rows'] for stats in current_retention.values())
    current_original = sum(stats['total_rows'] for stats in current_retention.values())
    current_rate = current_total / current_original * 100 if current_original > 0 else 0

    print(f"\n当前方案数据保留:")
    print(f"  原始: {current_original} 行")
    print(f"  保留: {current_total} 行")
    print(f"  保留率: {current_rate:.1f}%")

    # 提出替代策略
    strategies = propose_alternative_grouping_strategies(df, model_hyperparams, model_counts)

    # 比较策略
    comparison_results = compare_strategies(df, strategies, model_hyperparams)

    # 生成建议
    best_strategy_name, best_results = generate_recommendation(comparison_results, df)

    # 保存结果
    save_results(comparison_results, best_strategy_name, strategies)

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
