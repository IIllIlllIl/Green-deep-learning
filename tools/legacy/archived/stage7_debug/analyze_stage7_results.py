#!/usr/bin/env python3
"""
分析Stage7实验结果
- 检查去重效果
- 统计每个模型的实验数量和唯一值
- 验证数据完整性
"""

import csv
from collections import defaultdict
import json

def analyze_stage7_results():
    # 读取summary_all.csv
    with open('results/summary_all.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_experiments = len(rows)
    print(f"总实验数: {total_experiments}")
    print(f"CSV列数: {len(reader.fieldnames)}")
    print()

    # Stage7涉及的模型
    stage7_models = [
        ("examples", "mnist"),
        ("examples", "mnist_ff"),
        ("examples", "mnist_rnn"),
        ("examples", "siamese"),
        ("MRT-OAST", "default"),
        ("bug-localization-by-dnn-and-rvsm", "default"),
        ("pytorch_resnet_cifar10", "resnet20")
    ]

    # 统计每个模型的非并行实验
    model_stats = defaultdict(lambda: {"total": 0, "nonparallel": 0, "parallel": 0, "params": defaultdict(set)})

    for row in rows:
        repo = row.get('repository', '')
        model = row.get('model', '')
        exp_id = row.get('experiment_id', '')

        # 判断是否为并行模式（experiment_id包含_parallel后缀）
        is_parallel = '_parallel' in exp_id
        mode = 'parallel' if is_parallel else 'nonparallel'

        key = (repo, model)
        model_stats[key]["total"] += 1
        model_stats[key][mode] += 1

        # 统计参数唯一值
        for param_key in ['hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
                          'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
                          'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay']:
            param_value = row.get(param_key, '')
            if param_value:  # 如果参数不为空
                model_stats[key]["params"][param_key].add(param_value)

    # 打印Stage7相关模型的统计
    print("=" * 80)
    print("Stage7模型统计（非并行模式）")
    print("=" * 80)

    for repo, model in stage7_models:
        key = (repo, model)
        stats = model_stats[key]
        print(f"\n{repo}/{model}:")
        print(f"  总实验数: {stats['total']}")
        print(f"  非并行模式: {stats['nonparallel']}")
        print(f"  并行模式: {stats['parallel']}")
        print(f"  参数唯一值:")

        for param_key in sorted(stats["params"].keys()):
            unique_count = len(stats["params"][param_key])
            print(f"    {param_key}: {unique_count}个唯一值")

    # 读取Stage7配置
    print("\n" + "=" * 80)
    print("Stage7配置期望 vs 实际")
    print("=" * 80)

    with open('settings/stage7_nonparallel_fast_models.json', 'r') as f:
        config = json.load(f)

    print(f"\n配置信息:")
    print(f"  预计实验数: {config['estimated_experiments']}")
    print(f"  预计时长: {config['estimated_duration_hours']}小时")
    print(f"  实际新增实验: 7个")
    print(f"  实际运行时长: 0.74小时")
    print(f"  去重跳过率: {(config['estimated_experiments'] - 7) / config['estimated_experiments'] * 100:.1f}%")

    # 分析最新7个实验
    print("\n" + "=" * 80)
    print("最新7个实验详情")
    print("=" * 80)

    latest_7 = rows[-7:]
    for i, row in enumerate(latest_7, 1):
        exp_id = row.get('experiment_id', '')
        repo = row.get('repository', '')
        model = row.get('model', '')
        duration = float(row.get('duration_seconds', 0)) / 60  # 转换为分钟
        success = row.get('training_success', '')

        # 提取关键参数
        params = []
        for key in ['hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_batch_size',
                    'hyperparam_seed', 'hyperparam_dropout', 'hyperparam_weight_decay',
                    'hyperparam_max_iter', 'hyperparam_kfold', 'hyperparam_alpha']:
            value = row.get(key, '')
            if value:
                param_name = key.replace('hyperparam_', '')
                params.append(f"{param_name}={value}")

        print(f"\n实验{i}: {exp_id}")
        print(f"  模型: {repo}/{model}")
        print(f"  时长: {duration:.1f}分钟")
        print(f"  成功: {success}")
        print(f"  参数: {', '.join(params[:3])}")  # 只显示前3个参数避免太长

if __name__ == "__main__":
    analyze_stage7_results()
