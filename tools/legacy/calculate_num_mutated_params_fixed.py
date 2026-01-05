#!/usr/bin/env python3
"""
修复后的num_mutated_params计算逻辑

根本问题：
原逻辑只检查参数是否有值（非空），不比较值是否等于默认值。
导致所有设置了参数的实验都被标记为"变异"，即使使用的是默认值。

修复方案：
1. 加载models_config.json获取每个模型的默认值
2. 对比实验中的参数值与默认值
3. 只有当值不同时，才计为变异参数

作者: Claude Code
日期: 2025-12-12
"""

import json
from pathlib import Path
from typing import Dict, Tuple


def load_models_config():
    """加载models_config.json"""
    config_path = Path('mutation/models_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def get_model_defaults(models_config: dict, repo: str, model: str) -> Dict[str, any]:
    """
    获取指定模型的默认超参数值

    返回: {param_name: default_value} 字典
    """
    defaults = {}

    # 在models配置中查找对应的仓库
    models_data = models_config.get('models', {})

    if repo in models_data:
        repo_config = models_data[repo]
        supported_params = repo_config.get('supported_hyperparams', {})

        for param_name, param_config in supported_params.items():
            default_value = param_config.get('default')
            if default_value is not None:
                defaults[param_name] = default_value

    return defaults


def normalize_value(value, expected_type: str):
    """
    标准化参数值以便比较

    处理类型转换和精度问题：
    - int类型：转换为int
    - float类型：转换为float（处理精度差异）
    - 空值/None：返回None
    """
    if value in ['', None]:
        return None

    try:
        if expected_type == 'int':
            return int(float(value))
        elif expected_type == 'float':
            # float类型比较时考虑精度
            return float(value)
        else:
            return value
    except (ValueError, TypeError):
        return value


def is_value_mutated(exp_value, default_value, param_type: str) -> bool:
    """
    判断实验值是否与默认值不同

    参数:
        exp_value: 实验中的参数值
        default_value: 默认值
        param_type: 参数类型 ('int' 或 'float')

    返回:
        True: 值不同（变异）
        False: 值相同（默认值）
    """
    # 标准化两个值
    norm_exp = normalize_value(exp_value, param_type)
    norm_def = normalize_value(default_value, param_type)

    # 如果实验值为空，视为使用默认值
    if norm_exp is None:
        return False

    # 如果默认值为None（models_config中未定义默认值），保守处理
    if norm_def is None:
        # 如果实验配置了值，但models_config没有定义默认值，
        # 保守地认为这是变异（虽然可能不准确）
        return True

    # 比较值
    if param_type == 'float':
        # float类型使用相对误差比较（处理浮点精度问题）
        return abs(norm_exp - norm_def) > abs(norm_def * 1e-6)
    else:
        return norm_exp != norm_def


def calculate_num_mutated_params_fixed(row: Dict, models_config: dict) -> Tuple[int, str]:
    """
    修复后的变异参数计算函数

    逻辑：
    1. 获取该模型的默认值
    2. 遍历所有超参数列
    3. 对比实验值与默认值
    4. 只有当值不同时，才计为变异参数

    参数:
        row: CSV行数据（字典）
        models_config: models_config.json数据

    返回:
        (num_mutated_params, mutated_param)
        - num_mutated_params: 变异参数数量
        - mutated_param: 如果只有1个变异参数，返回参数名；否则返回空字符串
    """
    # 确定模式和参数前缀
    mode = row.get('mode', '')
    is_parallel = (mode == 'parallel')

    if is_parallel:
        param_prefix = 'fg_hyperparam_'
        repo = row.get('fg_repository', '')
        model = row.get('fg_model', '')
    else:
        param_prefix = 'hyperparam_'
        repo = row.get('repository', '')
        model = row.get('model', '')

    # 获取该模型的默认值
    defaults = get_model_defaults(models_config, repo, model)

    if not defaults:
        # 如果无法获取默认值，fallback到旧逻辑（计数非空参数）
        # 并在stderr输出警告
        import sys
        print(f"⚠️ 警告: 无法获取模型 {repo}/{model} 的默认值，使用旧逻辑", file=sys.stderr)
        return calculate_num_mutated_params_old(row)

    # 标准参数列表
    standard_params = ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                       'learning_rate', 'max_iter', 'seed', 'weight_decay']

    # 获取参数类型映射
    param_types = {}
    models_data = models_config.get('models', {})
    if repo in models_data:
        supported_params = models_data[repo].get('supported_hyperparams', {})
        for param_name, param_config in supported_params.items():
            param_types[param_name] = param_config.get('type', 'str')

    # 计数变异参数
    mutated_count = 0
    mutated_param_name = None

    for param in standard_params:
        col_name = f'{param_prefix}{param}'

        # 获取实验中的参数值
        exp_value = row.get(col_name, '')

        # 如果实验中没有这个参数，跳过（视为使用默认值）
        if exp_value in ['', None]:
            continue

        # 获取默认值
        default_value = defaults.get(param)

        # 获取参数类型
        param_type = param_types.get(param, 'str')

        # 判断是否变异
        if is_value_mutated(exp_value, default_value, param_type):
            mutated_count += 1
            if mutated_count == 1:
                mutated_param_name = param

    # 返回结果
    if mutated_count == 1:
        return mutated_count, mutated_param_name
    else:
        return mutated_count, ''


def calculate_num_mutated_params_old(row: Dict) -> Tuple[int, str]:
    """
    旧的计算逻辑（作为fallback）
    只计数非空参数，不比较默认值
    """
    if row.get('mode') == 'parallel':
        param_prefix = 'fg_hyperparam_'
    else:
        param_prefix = 'hyperparam_'

    standard_params = ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                       'learning_rate', 'max_iter', 'seed', 'weight_decay']

    count = 0
    mutated_param = None

    for param in standard_params:
        col_name = f'{param_prefix}{param}'
        if col_name in row and row[col_name] not in ['', None]:
            count += 1
            if count == 1:
                mutated_param = param

    return count, mutated_param if count == 1 else ''


# 测试函数
def test_fixed_logic():
    """测试修复后的逻辑"""
    print("=" * 80)
    print("测试修复后的num_mutated_params计算逻辑")
    print("=" * 80)

    # 加载配置
    models_config = load_models_config()

    # 测试案例1：default__examples_mnist_008（所有值都是默认值）
    test_row_1 = {
        'mode': '',
        'repository': 'examples',
        'model': 'mnist',
        'hyperparam_epochs': '10',
        'hyperparam_learning_rate': '0.01',
        'hyperparam_batch_size': '32',
        'hyperparam_seed': '1',
    }

    num, param = calculate_num_mutated_params_fixed(test_row_1, models_config)
    print("\n测试案例1: default__examples_mnist_008")
    print(f"  输入: epochs=10, lr=0.01, batch=32, seed=1")
    print(f"  默认值: epochs=10, lr=0.01, batch=32, seed=1")
    print(f"  结果: num_mutated_params={num}, mutated_param={param}")
    print(f"  预期: num_mutated_params=0, mutated_param=''")
    print(f"  {'✓ 通过' if num == 0 and param == '' else '✗ 失败'}")

    # 测试案例2：单参数变异
    test_row_2 = {
        'mode': '',
        'repository': 'examples',
        'model': 'mnist',
        'hyperparam_epochs': '12',  # 变异
        'hyperparam_learning_rate': '0.01',
        'hyperparam_batch_size': '32',
        'hyperparam_seed': '1',
    }

    num, param = calculate_num_mutated_params_fixed(test_row_2, models_config)
    print("\n测试案例2: 单参数变异（epochs）")
    print(f"  输入: epochs=12, lr=0.01, batch=32, seed=1")
    print(f"  默认值: epochs=10, lr=0.01, batch=32, seed=1")
    print(f"  结果: num_mutated_params={num}, mutated_param={param}")
    print(f"  预期: num_mutated_params=1, mutated_param='epochs'")
    print(f"  {'✓ 通过' if num == 1 and param == 'epochs' else '✗ 失败'}")

    # 测试案例3：多参数变异
    test_row_3 = {
        'mode': '',
        'repository': 'examples',
        'model': 'mnist',
        'hyperparam_epochs': '12',  # 变异
        'hyperparam_learning_rate': '0.02',  # 变异
        'hyperparam_batch_size': '32',
        'hyperparam_seed': '1',
    }

    num, param = calculate_num_mutated_params_fixed(test_row_3, models_config)
    print("\n测试案例3: 多参数变异（epochs, lr）")
    print(f"  输入: epochs=12, lr=0.02, batch=32, seed=1")
    print(f"  默认值: epochs=10, lr=0.01, batch=32, seed=1")
    print(f"  结果: num_mutated_params={num}, mutated_param={param}")
    print(f"  预期: num_mutated_params=2, mutated_param=''")
    print(f"  {'✓ 通过' if num == 2 and param == '' else '✗ 失败'}")

    # 测试案例4：浮点精度测试
    test_row_4 = {
        'mode': '',
        'repository': 'examples',
        'model': 'mnist',
        'hyperparam_learning_rate': '0.010000001',  # 接近默认值（浮点精度误差）
    }

    num, param = calculate_num_mutated_params_fixed(test_row_4, models_config)
    print("\n测试案例4: 浮点精度测试")
    print(f"  输入: lr=0.010000001")
    print(f"  默认值: lr=0.01")
    print(f"  结果: num_mutated_params={num}, mutated_param={param}")
    print(f"  预期: num_mutated_params=0 (浮点精度容差范围内)")
    print(f"  {'✓ 通过' if num == 0 else '✗ 失败'}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == '__main__':
    test_fixed_logic()
