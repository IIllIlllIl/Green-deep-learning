#!/usr/bin/env python3
"""
实验完成度分析脚本

根据重新澄清的实验目标分析当前进度：
1. 每个超参数在两种模式下都需要：
   - 1个默认值实验
   - 5个唯一的单参数变异实验
2. 所有实验需要完整的能耗和性能记录

输出：
- 当前完成情况统计
- 缺失实验详细列表
- 需要补齐的实验数量
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 支持的超参数列表（从models_config.json读取）
SUPPORTED_PARAMS = [
    'alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
    'learning_rate', 'max_iter', 'seed', 'weight_decay'
]

# 模型-参数映射（45个组合）
MODEL_PARAMS = {
    'examples/mnist': ['batch_size', 'epochs', 'learning_rate', 'seed'],
    'examples/mnist_rnn': ['batch_size', 'epochs', 'learning_rate', 'seed'],
    'examples/mnist_ff': ['batch_size', 'epochs', 'learning_rate', 'seed'],
    'examples/siamese': ['batch_size', 'epochs', 'learning_rate', 'seed'],
    'Person_reID_baseline_pytorch/densenet121': ['dropout', 'epochs', 'learning_rate', 'seed'],
    'Person_reID_baseline_pytorch/hrnet18': ['dropout', 'epochs', 'learning_rate', 'seed'],
    'Person_reID_baseline_pytorch/pcb': ['dropout', 'epochs', 'learning_rate', 'seed'],
    'VulBERTa/mlp': ['epochs', 'learning_rate', 'seed', 'weight_decay'],
    'pytorch_resnet_cifar10/resnet20': ['epochs', 'learning_rate', 'seed', 'weight_decay'],
    'MRT-OAST/default': ['dropout', 'epochs', 'learning_rate', 'seed', 'weight_decay'],
    'bug-localization-by-dnn-and-rvsm/default': ['alpha', 'kfold', 'max_iter', 'seed'],
}

def count_model_param_combinations():
    """统计总的模型-参数组合数"""
    total = sum(len(params) for params in MODEL_PARAMS.values())
    return total

def extract_mutated_params(row: Dict) -> Set[str]:
    """提取实验中变异的参数

    通过比较超参数值与默认值来判断哪些参数被变异了
    """
    mutated = set()

    # 获取仓库和模型
    repo = row.get('repository', '')
    model = row.get('model', '')
    model_key = f"{repo}/{model}"

    # 检查experiment_id中的mode
    exp_id = row.get('experiment_id', '')
    is_parallel = 'parallel' in exp_id

    # 根据模式选择字段前缀
    prefix = 'fg_' if is_parallel else ''

    # 检查每个超参数
    for param in SUPPORTED_PARAMS:
        col_name = f'{prefix}hyperparam_{param}'
        value = row.get(col_name, '')

        if value and value.strip():
            # 如果有值，说明这个参数在配置中出现了
            # 但我们需要判断是否是变异值
            # 简单方法：如果num_mutated_params > 0，则根据mutated_param字段判断
            num_mutated = row.get('num_mutated_params', '0')
            try:
                num_mutated = int(num_mutated)
            except:
                num_mutated = 0

            if num_mutated > 0:
                # 检查mutated_param字段
                mutated_param = row.get('mutated_param', '')
                if param in mutated_param:
                    mutated.add(param)

    return mutated

def is_valid_experiment(row: Dict) -> Tuple[bool, str]:
    """检查实验是否有效

    返回: (是否有效, 失败原因)
    """
    # 1. 训练必须成功
    if row.get('training_success') != 'True':
        return False, "训练失败"

    # 2. 必须有能耗数据
    exp_id = row.get('experiment_id', '')
    is_parallel = 'parallel' in exp_id
    prefix = 'fg_' if is_parallel else ''

    cpu_energy = row.get(f'{prefix}energy_cpu_total_joules', '')
    gpu_energy = row.get(f'{prefix}energy_gpu_total_joules', '')

    if not cpu_energy or not gpu_energy:
        return False, "能耗数据缺失"

    # 3. 必须有性能数据（任意一个性能指标）
    perf_columns = [
        'perf_accuracy', 'perf_best_val_accuracy', 'perf_map', 'perf_precision',
        'perf_rank1', 'perf_rank5', 'perf_recall', 'perf_test_accuracy', 'perf_test_loss',
        'fg_perf_accuracy', 'fg_perf_best_val_accuracy', 'fg_perf_map', 'fg_perf_precision',
        'fg_perf_rank1', 'fg_perf_rank5', 'fg_perf_recall', 'fg_perf_test_accuracy', 'fg_perf_test_loss'
    ]

    has_any_perf = any(row.get(col, '').strip() for col in perf_columns)
    if not has_any_perf:
        return False, "性能数据缺失"

    return True, ""

def analyze_completion():
    """分析实验完成度"""

    # 读取raw_data.csv
    csv_path = PROJECT_ROOT / 'results' / 'raw_data.csv'

    if not csv_path.exists():
        print(f"❌ 错误: {csv_path} 不存在")
        return

    print("=" * 80)
    print("实验完成度分析")
    print("=" * 80)
    print(f"数据源: {csv_path}")
    print()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"总实验数: {len(rows)}")
    print()

    # 统计每个模型-参数-模式组合的实验
    # Key: (model_key, param, mode)
    # Value: {'default': [...], 'mutation': [...]}
    experiments = defaultdict(lambda: {'default': [], 'mutation': []})

    # 统计无效实验
    invalid_experiments = []

    for row in rows:
        repo = row.get('repository', '')
        model = row.get('model', '')
        model_key = f"{repo}/{model}"

        # 跳过未知模型
        if model_key not in MODEL_PARAMS:
            continue

        # 确定模式
        exp_id = row.get('experiment_id', '')
        mode = 'parallel' if 'parallel' in exp_id else 'nonparallel'

        # 检查实验有效性
        is_valid, reason = is_valid_experiment(row)
        if not is_valid:
            invalid_experiments.append({
                'experiment_id': exp_id,
                'model': model_key,
                'mode': mode,
                'reason': reason
            })
            continue

        # 确定实验类型（默认值还是变异）
        num_mutated = row.get('num_mutated_params', '0')
        try:
            num_mutated = int(num_mutated)
        except:
            num_mutated = 0

        if num_mutated == 0:
            # 默认值实验
            # 为所有支持的参数记录
            for param in MODEL_PARAMS[model_key]:
                key = (model_key, param, mode)
                experiments[key]['default'].append(row)
        elif num_mutated == 1:
            # 单参数变异实验
            mutated_param = row.get('mutated_param', '')
            # 提取参数名
            for param in MODEL_PARAMS[model_key]:
                if param in mutated_param:
                    key = (model_key, param, mode)

                    # 提取参数值用于唯一性检查
                    is_parallel = 'parallel' in exp_id
                    prefix = 'fg_' if is_parallel else ''
                    col_name = f'{prefix}hyperparam_{param}'
                    param_value = row.get(col_name, '')

                    if param_value:
                        row['_param_value'] = param_value
                        experiments[key]['mutation'].append(row)
                    break
        else:
            # 多参数变异实验 - 标记为无效
            invalid_experiments.append({
                'experiment_id': exp_id,
                'model': model_key,
                'mode': mode,
                'reason': f"多参数变异({num_mutated}个参数)"
            })

    # 分析完成度
    print("=" * 80)
    print("实验目标要求")
    print("=" * 80)

    total_combinations = count_model_param_combinations()
    print(f"总模型-参数组合: {total_combinations}")
    print(f"总模式: 2 (并行/非并行)")
    print(f"总目标组合: {total_combinations} × 2 = {total_combinations * 2}")
    print()
    print("每个组合需要:")
    print("  - 1个默认值实验")
    print("  - 5个唯一的单参数变异实验")
    print()

    # 统计完成情况
    completed = 0
    partial = 0
    missing = 0

    missing_details = []

    for model_key, params in sorted(MODEL_PARAMS.items()):
        for param in params:
            for mode in ['nonparallel', 'parallel']:
                key = (model_key, param, mode)
                exps = experiments[key]

                # 检查默认值实验
                default_count = len(exps['default'])

                # 检查变异实验唯一值
                mutation_values = set()
                for exp in exps['mutation']:
                    value = exp.get('_param_value', '')
                    if value:
                        mutation_values.add(value)

                unique_mutation_count = len(mutation_values)

                # 判断完成状态
                if default_count >= 1 and unique_mutation_count >= 5:
                    completed += 1
                elif default_count >= 1 or unique_mutation_count > 0:
                    partial += 1
                    missing_details.append({
                        'model': model_key,
                        'param': param,
                        'mode': mode,
                        'default_count': default_count,
                        'unique_mutation_count': unique_mutation_count,
                        'need_default': max(0, 1 - default_count),
                        'need_mutation': max(0, 5 - unique_mutation_count)
                    })
                else:
                    missing += 1
                    missing_details.append({
                        'model': model_key,
                        'param': param,
                        'mode': mode,
                        'default_count': 0,
                        'unique_mutation_count': 0,
                        'need_default': 1,
                        'need_mutation': 5
                    })

    total = total_combinations * 2

    print("=" * 80)
    print("完成度统计")
    print("=" * 80)
    print(f"完全完成: {completed}/{total} ({completed/total*100:.1f}%)")
    print(f"部分完成: {partial}/{total} ({partial/total*100:.1f}%)")
    print(f"完全缺失: {missing}/{total} ({missing/total*100:.1f}%)")
    print()

    # 统计需要补齐的实验数
    need_default = sum(d['need_default'] for d in missing_details)
    need_mutation = sum(d['need_mutation'] for d in missing_details)

    print("=" * 80)
    print("需要补齐的实验")
    print("=" * 80)
    print(f"默认值实验: {need_default}")
    print(f"变异实验: {need_mutation}")
    print(f"总计: {need_default + need_mutation}")
    print()

    # 无效实验统计
    print("=" * 80)
    print("无效实验统计")
    print("=" * 80)
    print(f"总无效实验: {len(invalid_experiments)}")
    print()

    # 按原因分组
    by_reason = defaultdict(int)
    for exp in invalid_experiments:
        by_reason[exp['reason']] += 1

    print("无效原因分布:")
    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print()

    # 输出缺失详情
    if missing_details:
        print("=" * 80)
        print("缺失实验详情（按模型和模式分组）")
        print("=" * 80)

        # 按模型和模式分组
        by_model_mode = defaultdict(list)
        for detail in missing_details:
            key = (detail['model'], detail['mode'])
            by_model_mode[key].append(detail)

        for (model_key, mode), details in sorted(by_model_mode.items()):
            print(f"\n{model_key} ({mode}):")

            for d in sorted(details, key=lambda x: x['param']):
                status = []
                if d['default_count'] == 0:
                    status.append(f"缺默认值")
                if d['unique_mutation_count'] < 5:
                    status.append(f"缺{5-d['unique_mutation_count']}个变异")

                print(f"  {d['param']:15s}: "
                      f"默认={d['default_count']}/1, "
                      f"变异={d['unique_mutation_count']}/5  "
                      f"[{', '.join(status)}]")

    # 输出JSON格式的详细报告
    report = {
        'summary': {
            'total_combinations': total,
            'completed': completed,
            'partial': partial,
            'missing': missing,
            'completion_rate': f"{completed/total*100:.1f}%"
        },
        'need_experiments': {
            'default': need_default,
            'mutation': need_mutation,
            'total': need_default + need_mutation
        },
        'invalid_experiments': {
            'total': len(invalid_experiments),
            'by_reason': dict(by_reason)
        },
        'missing_details': missing_details,
        'invalid_details': invalid_experiments
    }

    # 保存报告
    report_path = PROJECT_ROOT / 'results' / 'experiment_completion_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"详细报告已保存至: {report_path}")
    print("=" * 80)

if __name__ == '__main__':
    analyze_completion()
