#!/usr/bin/env python3
"""
重新计算 raw_data.csv 中所有实验的 num_mutated_params 和 mutated_param

用途：
1. 修复空值问题（201个实验的 num_mutated_params 为空）
2. 修复seed误判问题（44个实验被错误标记为seed变异）
3. 统一使用修复后的计算逻辑

修复范围：
- 所有676个实验
- 使用最新的 models_config.json（已修复seed默认值）
- 基于 calculate_num_mutated_params_fixed.py 的逻辑

作者：Claude Code
日期：2025-12-21
"""

import csv
import json
import sys
from pathlib import Path
from datetime import datetime
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

    # 如果默认值为None（models_config中未定义默认值）
    if norm_def is None:
        return False  # 修复：不再保守地认为是变异

    # 比较值
    if param_type == 'float':
        # float类型使用相对误差比较（处理浮点精度问题）
        return abs(norm_exp - norm_def) > abs(norm_def * 1e-6)
    else:
        return norm_exp != norm_def


def calculate_num_mutated_params(row: Dict, models_config: dict) -> Tuple[int, str]:
    """
    计算变异参数数量

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
        # 如果无法获取默认值，返回0（保守处理）
        return 0, ''

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


def recalculate_all_rows(input_csv: Path, output_csv: Path, models_config: dict,
                         verbose: bool = True):
    """
    重新计算所有行的 num_mutated_params 和 mutated_param

    参数:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        models_config: 模型配置
        verbose: 是否显示详细信息
    """
    # 读取CSV
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total_rows = len(rows)
    changed_count = 0
    empty_fixed = 0
    seed_fixed = 0

    if verbose:
        print('=' * 100)
        print(f'重新计算 num_mutated_params')
        print('=' * 100)
        print(f'总行数: {total_rows}')
        print(f'输入: {input_csv}')
        print(f'输出: {output_csv}')
        print('')

    # 重新计算每一行
    updated_rows = []
    for i, row in enumerate(rows, 1):
        old_num = row.get('num_mutated_params', '')
        old_param = row.get('mutated_param', '')

        # 计算新值
        new_num, new_param = calculate_num_mutated_params(row, models_config)

        # 更新
        row['num_mutated_params'] = str(new_num)
        row['mutated_param'] = new_param

        # 统计变化
        if str(new_num) != old_num or new_param != old_param:
            changed_count += 1

            # 统计修复类型
            if old_num == '':
                empty_fixed += 1
            elif old_num == '1' and old_param == 'seed' and new_num == 0:
                seed_fixed += 1

            if verbose and changed_count <= 10:  # 只显示前10个变化
                exp_id = row.get('experiment_id', '')[:60]
                print(f'{i:4d}. {exp_id:60s}')
                print(f'      旧值: num={old_num:2s}, param={old_param}')
                print(f'      新值: num={new_num:2d}, param={new_param}')

        updated_rows.append(row)

        # 进度显示
        if verbose and i % 100 == 0:
            print(f'进度: {i}/{total_rows} ({i/total_rows*100:.1f}%)')

    # 写入新CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    # 显示统计
    if verbose:
        print('')
        print('=' * 100)
        print('✅ 完成')
        print('=' * 100)
        print(f'总行数: {total_rows}')
        print(f'修改数: {changed_count} ({changed_count/total_rows*100:.1f}%)')
        print(f'  - 空值修复: {empty_fixed}个')
        print(f'  - seed误判修复: {seed_fixed}个')
        print(f'  - 其他修复: {changed_count - empty_fixed - seed_fixed}个')
        print(f'未修改: {total_rows - changed_count} ({(total_rows-changed_count)/total_rows*100:.1f}%)')
        print('')
        print(f'输出文件: {output_csv}')

    return {
        'total': total_rows,
        'changed': changed_count,
        'empty_fixed': empty_fixed,
        'seed_fixed': seed_fixed,
        'unchanged': total_rows - changed_count
    }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='重新计算 raw_data.csv 中所有实验的 num_mutated_params',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 重新计算（会自动备份）
  python3 scripts/recalculate_num_mutated_params_all.py

  # 指定输入输出文件
  python3 scripts/recalculate_num_mutated_params_all.py -i results/raw_data.csv -o results/raw_data_fixed.csv

  # 静默模式
  python3 scripts/recalculate_num_mutated_params_all.py --quiet
"""
    )

    parser.add_argument('-i', '--input', type=str, default='results/raw_data.csv',
                        help='输入CSV文件路径（默认：results/raw_data.csv）')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出CSV文件路径（默认：覆盖输入文件）')
    parser.add_argument('-c', '--config', type=str, default='mutation/models_config.json',
                        help='models_config.json路径（默认：mutation/models_config.json）')
    parser.add_argument('--no-backup', action='store_true',
                        help='不创建备份文件')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式')

    args = parser.parse_args()

    input_csv = Path(args.input)

    # 确定输出文件
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = input_csv

    # 验证输入文件存在
    if not input_csv.exists():
        print(f'❌ 错误: 输入文件不存在: {input_csv}', file=sys.stderr)
        sys.exit(1)

    # 创建备份
    if not args.no_backup and output_csv == input_csv:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = input_csv.parent / f'{input_csv.stem}.backup_{timestamp}{input_csv.suffix}'

        import shutil
        shutil.copy(input_csv, backup_path)

        if not args.quiet:
            print(f'✅ 已备份: {backup_path}')
            print('')

    # 加载配置
    try:
        models_config = load_models_config()
    except Exception as e:
        print(f'❌ 错误: 加载模型配置失败: {e}', file=sys.stderr)
        sys.exit(1)

    # 重新计算
    try:
        stats = recalculate_all_rows(input_csv, output_csv, models_config,
                                     verbose=not args.quiet)

        if not args.quiet:
            print('✅ 重新计算完成')

        sys.exit(0)

    except Exception as e:
        print(f'❌ 错误: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
