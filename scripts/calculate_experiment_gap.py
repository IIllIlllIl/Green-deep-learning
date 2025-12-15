#!/usr/bin/env python3
"""
计算距离实验目标的差距和预计运行时间

实验目标：每个超参数在两种模式（并行/非并行）下都需要：
1. 1个默认值实验 - 建立基线
2. 5个唯一单参数变异实验 - 研究单参数影响
3. 完整数据：能耗 + 任意性能指标

总目标：45参数 × 2模式 × 6实验 = 540个有效实验

用法: python3 scripts/calculate_experiment_gap.py

版本: 1.0
创建日期: 2025-12-15
"""

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 每个模型的平均运行时间（秒）- 基于Phase 4和Phase 5的实际数据
MODEL_AVG_RUNTIME = {
    'VulBERTa/mlp': {
        'nonparallel': 3755,  # ~1小时
        'parallel': 3755
    },
    'bug-localization-by-dnn-and-rvsm/default': {
        'nonparallel': 981,  # ~16分钟
        'parallel': 981
    },
    'MRT-OAST/default': {
        'nonparallel': 1568,  # ~26分钟
        'parallel': 1568
    },
    'examples/mnist': {
        'nonparallel': 450,  # ~7.5分钟（估计）
        'parallel': 450
    },
    'examples/mnist_ff': {
        'nonparallel': 450,  # ~7.5分钟（估计）
        'parallel': 450
    },
    'examples/mnist_rnn': {
        'nonparallel': 450,  # ~7.5分钟（估计）
        'parallel': 450
    },
    'examples/siamese': {
        'nonparallel': 450,  # ~7.5分钟（估计）
        'parallel': 450
    },
    'Person_reID_baseline_pytorch/hrnet18': {
        'nonparallel': 3600,  # ~1小时（估计）
        'parallel': 3600
    },
    'Person_reID_baseline_pytorch/pcb': {
        'nonparallel': 3600,  # ~1小时（估计）
        'parallel': 3600
    },
    'Person_reID_baseline_pytorch/densenet121': {
        'nonparallel': 7200,  # ~2小时（估计）
        'parallel': 7200
    },
    'pytorch_resnet_cifar10/resnet20': {
        'nonparallel': 3600,  # ~1小时（估计）
        'parallel': 3600
    }
}

# 每个模型的参数列表
MODEL_PARAMS = {
    'VulBERTa/mlp': ['epochs', 'learning_rate', 'seed', 'weight_decay'],
    'bug-localization-by-dnn-and-rvsm/default': ['alpha', 'kfold', 'max_iter', 'seed'],
    'MRT-OAST/default': ['dropout', 'epochs', 'learning_rate', 'seed', 'weight_decay'],
    'examples/mnist': ['batch_size', 'epochs', 'learning_rate', 'seed'],
    'examples/mnist_ff': ['batch_size', 'epochs', 'learning_rate', 'seed'],
    'examples/mnist_rnn': ['batch_size', 'epochs', 'learning_rate', 'seed'],
    'examples/siamese': ['batch_size', 'epochs', 'learning_rate', 'seed'],
    'Person_reID_baseline_pytorch/hrnet18': ['dropout', 'epochs', 'learning_rate', 'seed'],
    'Person_reID_baseline_pytorch/pcb': ['dropout', 'epochs', 'learning_rate', 'seed'],
    'Person_reID_baseline_pytorch/densenet121': ['dropout', 'epochs', 'learning_rate', 'seed'],
    'pytorch_resnet_cifar10/resnet20': ['epochs', 'learning_rate', 'seed', 'weight_decay']
}

def analyze_current_status():
    """分析当前实验完成情况"""

    csv_path = Path('results/raw_data.csv')

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f'总数据行数: {len(rows)}')
    print('')

    # 按模型-模式-参数分组统计唯一值数量
    by_model_mode_param = defaultdict(lambda: defaultdict(set))

    for row in rows:
        mode = row.get('mode', 'nonparallel')

        # 处理mode字段为空的情况
        if not mode or mode == '':
            mode = 'nonparallel'
        elif mode == 'parallel':
            mode = 'parallel'
        else:
            mode = 'nonparallel'

        # 根据模式获取正确的字段
        if mode == 'parallel':
            repo = row.get('fg_repository', '')
            model = row.get('fg_model', '')
            training_success = row.get('fg_training_success', '').lower() == 'true'
            has_perf = any(v for k, v in row.items() if k.startswith('fg_perf_') and v)
            has_energy = (row.get('fg_energy_cpu_total_joules', '') and
                         row.get('fg_energy_gpu_total_joules', ''))
            hyperparam_prefix = 'fg_hyperparam_'
        else:
            repo = row.get('repository', '')
            model = row.get('model', '')
            training_success = row.get('training_success', '').lower() == 'true'
            has_perf = any(v for k, v in row.items() if k.startswith('perf_') and v)
            has_energy = (row.get('energy_cpu_total_joules', '') and
                         row.get('energy_gpu_total_joules', ''))
            hyperparam_prefix = 'hyperparam_'

        model_key = f'{repo}/{model}' if repo else model

        # 只统计训练成功且有性能数据和能耗数据的实验
        if not training_success or not has_perf or not has_energy:
            continue

        # 获取超参数值
        for key, value in row.items():
            if key.startswith(hyperparam_prefix) and value:
                param_name = key.replace(hyperparam_prefix, '')
                param_key = f'{model_key}|{mode}|{param_name}'
                by_model_mode_param[param_key][param_name].add(value)

    return by_model_mode_param, len(rows)

def calculate_gap():
    """计算实验差距"""

    print('=' * 80)
    print('实验目标差距计算')
    print('=' * 80)
    print('')

    by_model_mode_param, total_rows = analyze_current_status()

    # 目标：每个参数在每种模式下需要5个唯一值
    target_per_param = 5

    # 统计缺口
    gaps_by_model = defaultdict(lambda: {'nonparallel': {}, 'parallel': {}})
    total_missing = 0
    total_needed_experiments = 0

    for model, params in MODEL_PARAMS.items():
        for mode in ['nonparallel', 'parallel']:
            for param in params:
                param_key = f'{model}|{mode}|{param}'
                current_count = len(by_model_mode_param[param_key][param])

                gap = max(0, target_per_param - current_count)

                if gap > 0:
                    gaps_by_model[model][mode][param] = {
                        'current': current_count,
                        'target': target_per_param,
                        'gap': gap
                    }
                    total_missing += gap
                    # 每个缺口需要1个实验来填补
                    total_needed_experiments += gap

    # 按模型打印详细缺口
    print('各模型缺口详情:')
    print('')

    model_experiment_needs = defaultdict(lambda: {'nonparallel': 0, 'parallel': 0})

    for model in sorted(MODEL_PARAMS.keys()):
        gaps = gaps_by_model[model]

        if not gaps['nonparallel'] and not gaps['parallel']:
            print(f'{model}: ✅ 完全达标')
            continue

        print(f'{model}:')

        for mode in ['nonparallel', 'parallel']:
            if gaps[mode]:
                print(f'  {mode}模式:')
                for param, gap_info in sorted(gaps[mode].items()):
                    print(f'    {param}: {gap_info["current"]}/{gap_info["target"]} (缺{gap_info["gap"]}个)')
                    model_experiment_needs[model][mode] += gap_info['gap']

        print('')

    # 计算总运行时间
    print('=' * 80)
    print('预计运行时间')
    print('=' * 80)
    print('')

    total_time_seconds = 0

    for model in sorted(MODEL_PARAMS.keys()):
        needs = model_experiment_needs[model]

        if needs['nonparallel'] == 0 and needs['parallel'] == 0:
            continue

        model_time_seconds = 0

        print(f'{model}:')

        for mode in ['nonparallel', 'parallel']:
            if needs[mode] > 0:
                avg_runtime = MODEL_AVG_RUNTIME.get(model, {}).get(mode, 1800)
                mode_time_seconds = needs[mode] * avg_runtime
                model_time_seconds += mode_time_seconds

                hours = mode_time_seconds / 3600
                print(f'  {mode}: {needs[mode]}个实验 × {avg_runtime/60:.1f}分钟 = {hours:.2f}小时')

        total_time_seconds += model_time_seconds
        total_hours = model_time_seconds / 3600
        print(f'  小计: {total_hours:.2f}小时')
        print('')

    # 总结
    print('=' * 80)
    print('总结')
    print('=' * 80)
    print('')

    print(f'当前实验总数: {total_rows}')
    print(f'有效实验数: {sum(len(v[list(v.keys())[0]]) for v in by_model_mode_param.values())}')
    print('')
    print(f'缺失唯一值总数: {total_missing}个')
    print(f'需要补充实验数: {total_needed_experiments}个')
    print('')

    total_hours = total_time_seconds / 3600
    total_days = total_hours / 24

    print(f'预计总运行时间: {total_hours:.2f}小时 ({total_days:.2f}天)')
    print('')

    # 按模式汇总
    nonparallel_experiments = sum(model_experiment_needs[m]['nonparallel'] for m in MODEL_PARAMS)
    parallel_experiments = sum(model_experiment_needs[m]['parallel'] for m in MODEL_PARAMS)

    print('按模式汇总:')
    print(f'  非并行模式: {nonparallel_experiments}个实验')
    print(f'  并行模式: {parallel_experiments}个实验')
    print('')

    # 达标情况
    total_param_mode_combos = sum(len(params) * 2 for params in MODEL_PARAMS.values())
    complete_combos = sum(
        1 for model, params in MODEL_PARAMS.items()
        for mode in ['nonparallel', 'parallel']
        if all(
            len(by_model_mode_param[f'{model}|{mode}|{param}'][param]) >= target_per_param
            for param in params
        )
    )

    print(f'达标情况: {complete_combos}/{total_param_mode_combos} ({complete_combos*100//total_param_mode_combos}%)')
    print('')

    # 完全达标的模型
    complete_models_nonparallel = [
        model for model in MODEL_PARAMS
        if all(
            len(by_model_mode_param[f'{model}|nonparallel|{param}'][param]) >= target_per_param
            for param in MODEL_PARAMS[model]
        )
    ]

    complete_models_parallel = [
        model for model in MODEL_PARAMS
        if all(
            len(by_model_mode_param[f'{model}|parallel|{param}'][param]) >= target_per_param
            for param in MODEL_PARAMS[model]
        )
    ]

    print(f'非并行模式达标: {len(complete_models_nonparallel)}/11 模型')
    print(f'并行模式达标: {len(complete_models_parallel)}/11 模型')

if __name__ == '__main__':
    calculate_gap()
