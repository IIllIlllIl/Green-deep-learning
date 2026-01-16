#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析数据在6分组回归分析下的可用性

根据 analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md 中定义的6分组方案，
分析当前数据集中哪些数据可用、哪些不可用。

6分组定义：
- 组1a: examples (mnist, mnist_ff, mnist_rnn, siamese)
- 组1b: pytorch_resnet (resnet20)
- 组2: Person_reID (densenet121, hrnet18, pcb)
- 组3: VulBERTa (mlp)
- 组4: bug_localization (default)
- 组5: MRT-OAST (default)

可用性标准：
1. 训练成功 (training_success = True)
2. 有能耗数据（至少一个能耗字段非空）
3. 有超参数数据（组内需要的超参数字段非空）
注意：不需要性能指标数据
"""

import csv
from collections import defaultdict
from typing import Dict, List, Tuple

# 6分组定义
GROUP_DEFINITIONS = {
    'group1a_examples': {
        'models': [
            'examples/mnist',
            'examples/mnist_ff',
            'examples/mnist_rnn',
            'examples/siamese'
        ],
        'hyperparams': ['hyperparam_batch_size', 'hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_seed']
    },
    'group1b_resnet': {
        'models': ['pytorch_resnet_cifar10/resnet20'],
        'hyperparams': ['hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_weight_decay', 'hyperparam_seed']
    },
    'group2_person_reid': {
        'models': [
            'Person_reID_baseline_pytorch/densenet121',
            'Person_reID_baseline_pytorch/hrnet18',
            'Person_reID_baseline_pytorch/pcb'
        ],
        'hyperparams': ['hyperparam_dropout', 'hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_seed']
    },
    'group3_vulberta': {
        'models': ['VulBERTa/mlp'],
        'hyperparams': ['hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_weight_decay', 'hyperparam_seed']
    },
    'group4_bug_localization': {
        'models': ['bug-localization-by-dnn-and-rvsm/default'],
        'hyperparams': ['hyperparam_alpha', 'hyperparam_kfold', 'hyperparam_max_iter', 'hyperparam_seed']
    },
    'group5_mrt_oast': {
        'models': ['MRT-OAST/default'],
        'hyperparams': ['hyperparam_dropout', 'hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_weight_decay']
    }
}

# 能耗字段（至少一个有值即可）
ENERGY_FIELDS = [
    'energy_cpu_pkg_joules',
    'energy_cpu_ram_joules',
    'energy_cpu_total_joules',
    'energy_gpu_total_joules',
    'energy_gpu_avg_watts'
]


def get_model_identifier(row: Dict[str, str]) -> str:
    """获取模型标识符"""
    repo = row.get('repository', '').strip()
    model = row.get('model', '').strip()
    if not repo or repo == '/':
        return 'unknown'
    return f"{repo}/{model}" if model else repo


def has_energy_data(row: Dict[str, str]) -> bool:
    """检查是否有能耗数据"""
    for field in ENERGY_FIELDS:
        value = row.get(field, '').strip()
        if value and value != 'N/A' and value != '0' and value != '0.0':
            return True
    return False


def check_hyperparams(row: Dict[str, str], required_params: List[str]) -> Tuple[bool, List[str]]:
    """
    检查超参数是否完整

    返回: (是否全部有值, 缺失的参数列表)
    """
    missing = []
    for param in required_params:
        value = row.get(param, '').strip()
        if not value or value == 'N/A':
            missing.append(param)

    return len(missing) == 0, missing


def analyze_group_usability(csv_file: str = 'data/raw_data.csv'):
    """分析6分组下的数据可用性"""

    print("=" * 100)
    print("6分组回归分析数据可用性分析")
    print("=" * 100)
    print()
    print(f"数据文件: {csv_file}")
    print()

    # 读取数据
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_records = len(rows)
    print(f"总记录数: {total_records}")
    print()

    # 按组分析
    group_stats = {}
    usable_records = defaultdict(list)
    unusable_records = defaultdict(list)

    # 记录每个模型的数据
    for row in rows:
        model_id = get_model_identifier(row)
        training_success = row.get('training_success', '').strip() == 'True'
        has_energy = has_energy_data(row)

        # 找到所属的组
        group_name = None
        required_params = None
        for gname, gdef in GROUP_DEFINITIONS.items():
            if model_id in gdef['models']:
                group_name = gname
                required_params = gdef['hyperparams']
                break

        if not group_name:
            continue  # 不在6分组中的模型，跳过

        # 检查超参数
        has_all_params, missing_params = check_hyperparams(row, required_params)

        # 判断是否可用
        is_usable = training_success and has_energy and has_all_params

        record_info = {
            'experiment_id': row.get('experiment_id', ''),
            'model': model_id,
            'training_success': training_success,
            'has_energy': has_energy,
            'has_all_params': has_all_params,
            'missing_params': missing_params
        }

        if is_usable:
            usable_records[group_name].append(record_info)
        else:
            unusable_records[group_name].append(record_info)

    # 统计每组的可用性
    print("=" * 100)
    print("📊 各组数据可用性统计")
    print("=" * 100)
    print()

    total_usable = 0
    total_in_groups = 0

    for group_name, group_def in GROUP_DEFINITIONS.items():
        usable_count = len(usable_records[group_name])
        unusable_count = len(unusable_records[group_name])
        total_count = usable_count + unusable_count
        usable_rate = (usable_count / total_count * 100) if total_count > 0 else 0

        total_usable += usable_count
        total_in_groups += total_count

        group_stats[group_name] = {
            'total': total_count,
            'usable': usable_count,
            'unusable': unusable_count,
            'usable_rate': usable_rate
        }

        print(f"{group_name}:")
        print(f"  模型: {', '.join(group_def['models'])}")
        print(f"  需要的超参数: {', '.join(group_def['hyperparams'])}")
        print(f"  总记录数: {total_count}")
        print(f"  ✅ 可用记录: {usable_count} ({usable_rate:.1f}%)")
        print(f"  ❌ 不可用记录: {unusable_count} ({100-usable_rate:.1f}%)")
        print()

    # 总体统计
    print("=" * 100)
    print("📈 总体统计")
    print("=" * 100)
    print()
    print(f"6分组覆盖的记录数: {total_in_groups}")
    print(f"✅ 可用记录总数: {total_usable} ({total_usable/total_in_groups*100:.1f}%)")
    print(f"❌ 不可用记录总数: {total_in_groups - total_usable} ({(total_in_groups - total_usable)/total_in_groups*100:.1f}%)")
    print()

    # 不可用原因分析
    print("=" * 100)
    print("🔍 不可用原因详细分析")
    print("=" * 100)
    print()

    for group_name in GROUP_DEFINITIONS.keys():
        unusable = unusable_records[group_name]
        if not unusable:
            continue

        print(f"\n{group_name} - 不可用记录: {len(unusable)}条")
        print("-" * 80)

        # 统计不可用原因
        reasons = defaultdict(int)
        for rec in unusable:
            if not rec['training_success']:
                reasons['训练失败'] += 1
            if not rec['has_energy']:
                reasons['能耗数据缺失'] += 1
            if not rec['has_all_params']:
                reasons['超参数缺失'] += 1

        print(f"  不可用原因统计:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}条")

        # 超参数缺失详情
        param_missing_stats = defaultdict(int)
        for rec in unusable:
            if not rec['has_all_params']:
                for param in rec['missing_params']:
                    param_missing_stats[param] += 1

        if param_missing_stats:
            print(f"\n  超参数缺失详情:")
            for param, count in sorted(param_missing_stats.items(), key=lambda x: -x[1]):
                print(f"    - {param}: {count}条记录缺失")

    # 保存结果
    print("\n" + "=" * 100)
    print("💾 保存分析结果")
    print("=" * 100)
    print()

    # 保存可用记录统计
    with open('data_usability_for_regression_summary.txt', 'w') as f:
        f.write("6分组回归分析数据可用性摘要\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总记录数: {total_records}\n")
        f.write(f"6分组覆盖的记录数: {total_in_groups}\n")
        f.write(f"可用记录总数: {total_usable} ({total_usable/total_in_groups*100:.1f}%)\n\n")

        f.write("各组统计:\n")
        f.write("-" * 80 + "\n")
        for group_name, stats in group_stats.items():
            f.write(f"\n{group_name}:\n")
            f.write(f"  总数: {stats['total']}\n")
            f.write(f"  可用: {stats['usable']} ({stats['usable_rate']:.1f}%)\n")
            f.write(f"  不可用: {stats['unusable']} ({100-stats['usable_rate']:.1f}%)\n")

    print("✅ 摘要已保存: data_usability_for_regression_summary.txt")

    # 保存详细的不可用记录
    with open('unusable_records_for_regression_detail.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Group', 'Experiment_ID', 'Model', 'Training_Success', 'Has_Energy', 'Has_All_Params', 'Missing_Params'])

        for group_name in GROUP_DEFINITIONS.keys():
            for rec in unusable_records[group_name]:
                writer.writerow([
                    group_name,
                    rec['experiment_id'],
                    rec['model'],
                    rec['training_success'],
                    rec['has_energy'],
                    rec['has_all_params'],
                    ', '.join(rec['missing_params']) if rec['missing_params'] else ''
                ])

    print("✅ 详细不可用记录已保存: unusable_records_for_regression_detail.csv")
    print()


if __name__ == '__main__':
    analyze_group_usability()
