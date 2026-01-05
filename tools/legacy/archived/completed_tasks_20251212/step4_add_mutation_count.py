#!/usr/bin/env python3
"""
步骤4: 添加变异超参数计数列

通过与models_config.json中的默认值对比，计算每个实验变异的超参数数量：
- 0: 全部采用默认值（baseline实验）
- 1: 变异1个超参数
- N: 变异N个超参数

逻辑：
1. experiment.json中没有提到的超参数 = 采用默认值或模型不支持该参数（不算变异）
2. experiment.json中提到的超参数 = 与默认值对比，不同则算变异
"""

import json
import csv

def load_models_config():
    """加载模型配置"""
    with open('mutation/models_config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_default_hyperparams(models_config, repository, model):
    """获取指定模型的默认超参数

    Args:
        models_config: 配置字典
        repository: 仓库名
        model: 模型名

    Returns:
        dict: {param_name: default_value}
    """
    if repository not in models_config['models']:
        return {}

    repo_config = models_config['models'][repository]
    supported = repo_config.get('supported_hyperparams', {})

    defaults = {}
    for param, config in supported.items():
        default_value = config.get('default')
        if default_value is not None:  # null也是有效的默认值
            defaults[param] = default_value

    return defaults

def compare_values(actual, default, param_type, param_name=''):
    """比较实际值与默认值

    Args:
        actual: 实际值（字符串）
        default: 默认值（可以是None）
        param_type: 参数类型（int/float）
        param_name: 参数名称（用于特殊处理）

    Returns:
        bool: True表示不同（变异），False表示相同（默认）
    """
    if actual == '' or actual is None:
        return False  # 空值表示未设置，使用默认值

    # 特殊处理：如果default是None（如seed参数），只要有值就算变异
    if default is None:
        return True

    try:
        if param_type == 'int':
            return int(float(actual)) != default
        elif param_type == 'float':
            # 浮点数比较，考虑精度
            return abs(float(actual) - default) > 1e-9
        else:
            return str(actual) != str(default)
    except (ValueError, TypeError):
        return False

def count_mutated_params(row, models_config, is_parallel=False):
    """计算变异的超参数数量

    Args:
        row: CSV行数据
        models_config: 模型配置
        is_parallel: 是否为并行模式

    Returns:
        int: 变异的超参数数量
    """
    # 获取repository和model
    if is_parallel:
        repository = row.get('fg_repository', '').strip()
        model = row.get('fg_model', '').strip()
        prefix = 'fg_hyperparam_'
    else:
        repository = row.get('repository', '').strip()
        model = row.get('model', '').strip()
        prefix = 'hyperparam_'

    if not repository:
        return 0

    # 获取默认超参数
    defaults = get_default_hyperparams(models_config, repository, model)
    if not defaults:
        return 0

    # 获取参数类型信息
    repo_config = models_config['models'].get(repository, {})
    param_types = {}
    for param, config in repo_config.get('supported_hyperparams', {}).items():
        param_types[param] = config.get('type', 'str')

    # 比较每个超参数
    mutated_count = 0

    for param, default_value in defaults.items():
        col_name = f'{prefix}{param}'
        actual_value = row.get(col_name, '').strip()

        # 如果CSV中没有这个列或值为空，说明使用了默认值
        if not actual_value:
            continue

        # 比较实际值与默认值
        param_type = param_types.get(param, 'str')
        if compare_values(actual_value, default_value, param_type):
            mutated_count += 1

    return mutated_count

def add_mutation_count_column():
    """添加变异超参数计数列到summary_new.csv"""

    models_config = load_models_config()

    # 读取summary_new.csv
    with open('results/summary_new.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames)
        rows = list(reader)

    print("添加变异超参数计数列")
    print("=" * 70)
    print(f"总实验数: {len(rows)}")
    print()

    # 添加新列
    new_column = 'num_mutated_params'
    if new_column in header:
        print(f"⚠️  列 '{new_column}' 已存在，将覆盖")
        header.remove(new_column)

    # 插入到experiment_source之后
    insert_pos = header.index('experiment_source') + 1 if 'experiment_source' in header else len(header)
    header.insert(insert_pos, new_column)

    # 计算每行的变异超参数数量
    print("计算变异超参数数量...")
    mutation_counts = []

    for row in rows:
        mode = row.get('mode', '').strip()
        is_parallel = (mode == 'parallel')

        count = count_mutated_params(row, models_config, is_parallel)
        row[new_column] = count
        mutation_counts.append(count)

    print("完成！")
    print()

    # 统计分布
    count_dist = {}
    for count in mutation_counts:
        count_dist[count] = count_dist.get(count, 0) + 1

    print("变异超参数数量分布：")
    print("-" * 70)
    for count in sorted(count_dist.keys()):
        num_exps = count_dist[count]
        percentage = num_exps / len(rows) * 100
        bar = '#' * int(percentage / 2)
        print(f"  {count:2d} 个参数变异: {num_exps:3d} 个实验 ({percentage:5.1f}%) {bar}")

    print()

    # 写入更新后的CSV
    output_file = 'results/summary_new.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ 已更新 {output_file}")
    print(f"  新增列: {new_column}")
    print(f"  总列数: {len(header)} (原{len(header)-1} + 1)")
    print()

    # 显示一些示例
    print("示例（前5个）：")
    print("-" * 70)
    for i, row in enumerate(rows[:5], 1):
        exp_id = row['experiment_id']
        repo = row.get('fg_repository' if row.get('mode') == 'parallel' else 'repository', '')
        model = row.get('fg_model' if row.get('mode') == 'parallel' else 'model', '')
        count = row[new_column]
        mode = row.get('mode', '') or '(non-parallel)'

        print(f"{i}. {exp_id[:50]:50s}")
        print(f"   {repo}/{model:20s} mode={mode:15s} 变异数={count}")

    return header, rows

if __name__ == '__main__':
    header, rows = add_mutation_count_column()
    print()
    print("下一步: 验证变异计数的正确性")
