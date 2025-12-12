#!/usr/bin/env python3
"""
步骤5: 增强变异分析（改进版step4）

功能增强：
1. 更新num_mutated_params计数逻辑，将seed也视为可变异参数（即使default=null）
2. 识别单一变异的参数名称，填充mutated_param列
3. 从models_config.json读取默认值，填充空的hyperparam列

修复问题：
- step4中seed参数的处理逻辑：当default=null时被视为变异，但空值时被跳过
- 需要统一：只要实验中设置了seed值（非空），就算作变异参数
"""

import json
import csv
from typing import Dict, List, Tuple, Any, Optional

def load_models_config() -> Dict:
    """加载模型配置"""
    with open('mutation/models_config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_default_hyperparams(models_config: Dict, repository: str, model: str) -> Dict:
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
        # 注意：None也是有效的默认值（如seed参数）
        defaults[param] = default_value

    return defaults

def get_param_types(models_config: Dict, repository: str) -> Dict[str, str]:
    """获取参数类型信息

    Args:
        models_config: 配置字典
        repository: 仓库名

    Returns:
        dict: {param_name: type}
    """
    if repository not in models_config['models']:
        return {}

    repo_config = models_config['models'][repository]
    param_types = {}
    for param, config in repo_config.get('supported_hyperparams', {}).items():
        param_types[param] = config.get('type', 'str')

    return param_types

def compare_values(actual: Any, default: Any, param_type: str, param_name: str = '') -> bool:
    """比较实际值与默认值

    重要逻辑：
    - 如果actual为空或None，返回False（使用默认值，非变异）
    - 如果default为None（如seed），只要actual有值就算变异
    - 如果default有值，比较actual与default是否不同

    Args:
        actual: 实际值（字符串或空）
        default: 默认值（可以是None）
        param_type: 参数类型（int/float/str）
        param_name: 参数名称（用于特殊处理）

    Returns:
        bool: True表示变异，False表示使用默认值
    """
    # 空值表示未设置，使用默认值
    if actual == '' or actual is None:
        return False

    # 特殊处理：default为None时（如seed），只要有值就算变异
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

def analyze_mutations(row: Dict, models_config: Dict, is_parallel: bool = False) -> Tuple[int, List[str], Dict[str, Any]]:
    """分析变异的超参数

    Args:
        row: CSV行数据
        models_config: 模型配置
        is_parallel: 是否为并行模式

    Returns:
        tuple: (变异数量, 变异参数列表, 默认值字典)
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
        return 0, [], {}

    # 获取默认超参数和类型信息
    defaults = get_default_hyperparams(models_config, repository, model)
    if not defaults:
        return 0, [], {}

    param_types = get_param_types(models_config, repository)

    # 比较每个超参数
    mutated_params = []

    for param, default_value in defaults.items():
        col_name = f'{prefix}{param}'
        actual_value = row.get(col_name, '').strip()

        # 如果CSV中没有这个列或值为空，说明使用了默认值
        if not actual_value:
            continue

        # 比较实际值与默认值
        param_type = param_types.get(param, 'str')
        if compare_values(actual_value, default_value, param_type, param):
            mutated_params.append(param)

    return len(mutated_params), mutated_params, defaults

def format_value(value: Any, param_type: str) -> str:
    """格式化参数值为字符串

    Args:
        value: 参数值
        param_type: 参数类型

    Returns:
        str: 格式化后的字符串
    """
    if value is None:
        return ''

    if param_type == 'int':
        return str(int(value))
    elif param_type == 'float':
        return str(float(value))
    else:
        return str(value)

def enhance_mutation_analysis(csv_file='results/summary_new.csv'):
    """增强变异分析：更新计数、识别参数名、填充默认值

    Args:
        csv_file: CSV文件路径（默认为summary_new.csv）
    """

    models_config = load_models_config()

    # 读取CSV文件
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames)
        rows = list(reader)

    print("增强变异分析")
    print("=" * 70)
    print(f"总实验数: {len(rows)}")
    print()

    # 添加新列：mutated_param（单一变异时的参数名）
    new_column = 'mutated_param'
    if new_column in header:
        print(f"⚠️  列 '{new_column}' 已存在，将覆盖")
        header.remove(new_column)

    # 插入到num_mutated_params之后
    if 'num_mutated_params' in header:
        insert_pos = header.index('num_mutated_params') + 1
    else:
        insert_pos = header.index('experiment_source') + 1 if 'experiment_source' in header else len(header)

    header.insert(insert_pos, new_column)

    # 处理每一行
    print("分析变异超参数...")
    mutation_counts = []
    single_param_count = 0
    multi_param_count = 0
    baseline_count = 0
    filled_defaults = 0

    for idx, row in enumerate(rows, 1):
        mode = row.get('mode', '').strip()
        is_parallel = (mode == 'parallel')

        # 分析变异
        count, mutated_params, defaults = analyze_mutations(row, models_config, is_parallel)

        # 更新num_mutated_params
        row['num_mutated_params'] = count
        mutation_counts.append(count)

        # 填充mutated_param（仅当变异1个参数时）
        if count == 1:
            row[new_column] = mutated_params[0]
            single_param_count += 1
        elif count == 0:
            row[new_column] = ''
            baseline_count += 1
        else:
            row[new_column] = ''  # 多参数变异时为空
            multi_param_count += 1

        # 填充空的hyperparam列（使用默认值）
        if is_parallel:
            repository = row.get('fg_repository', '').strip()
            prefix = 'fg_hyperparam_'
        else:
            repository = row.get('repository', '').strip()
            prefix = 'hyperparam_'

        if repository and defaults:
            param_types = get_param_types(models_config, repository)
            for param, default_value in defaults.items():
                col_name = f'{prefix}{param}'
                if col_name in row and not row[col_name].strip():
                    # 填充默认值
                    formatted = format_value(default_value, param_types.get(param, 'str'))
                    row[col_name] = formatted
                    if formatted:
                        filled_defaults += 1

        # 进度显示
        if idx % 50 == 0 or idx == len(rows):
            print(f"  处理进度: {idx}/{len(rows)}", end='\r')

    print()
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
        label = "baseline" if count == 0 else f"{count}个参数"
        print(f"  {label:15s}: {num_exps:3d} 个实验 ({percentage:5.1f}%) {bar}")

    print()
    print(f"统计汇总：")
    print(f"  - Baseline实验（0个变异）: {baseline_count}")
    print(f"  - 单参数变异实验: {single_param_count}")
    print(f"  - 多参数变异实验: {multi_param_count}")
    print(f"  - 填充默认值次数: {filled_defaults}")
    print()

    # 写入更新后的CSV
    output_file = csv_file  # 使用传入的文件路径
    backup_file = f'{csv_file}.backup_step5'

    # 备份原文件
    import shutil
    shutil.copy(output_file, backup_file)
    print(f"✓ 已备份原文件: {backup_file}")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ 已更新 {output_file}")
    print(f"  新增列: {new_column}")
    print(f"  总列数: {len(header)}")
    print()

    # 显示单参数变异示例
    print("单参数变异示例（前10个）：")
    print("-" * 70)
    single_param_examples = [r for r in rows if r.get('num_mutated_params') == '1'][:10]
    for i, row in enumerate(single_param_examples, 1):
        exp_id = row['experiment_id'][:50]
        repo = row.get('fg_repository' if row.get('mode') == 'parallel' else 'repository', '')
        model = row.get('fg_model' if row.get('mode') == 'parallel' else 'model', '')
        param = row.get('mutated_param', '')
        mode = row.get('mode', '') or 'non-parallel'

        # 获取参数值
        prefix = 'fg_hyperparam_' if row.get('mode') == 'parallel' else 'hyperparam_'
        param_value = row.get(f'{prefix}{param}', '')

        print(f"{i:2d}. {repo}/{model:15s} | mode={mode:12s} | param={param:15s} value={param_value}")

    print()

    # 显示多参数变异示例
    print("多参数变异示例（前5个）：")
    print("-" * 70)
    multi_param_examples = [r for r in rows if int(r.get('num_mutated_params', 0)) > 1][:5]
    for i, row in enumerate(multi_param_examples, 1):
        exp_id = row['experiment_id'][:50]
        repo = row.get('fg_repository' if row.get('mode') == 'parallel' else 'repository', '')
        model = row.get('fg_model' if row.get('mode') == 'parallel' else 'model', '')
        count = row.get('num_mutated_params', '')
        mode = row.get('mode', '') or 'non-parallel'

        print(f"{i}. {repo}/{model:20s} | mode={mode:12s} | 变异数={count}")

    return header, rows

if __name__ == '__main__':
    import sys

    print(__doc__)
    print()

    # 支持命令行参数指定CSV文件
    csv_file = 'results/summary_new.csv'
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"处理文件: {csv_file}")
        print()

    header, rows = enhance_mutation_analysis(csv_file)

    print()
    print("=" * 70)
    print("✓ 步骤5完成！")
    print()
    print("改进内容：")
    print("  1. ✓ 更新num_mutated_params计数逻辑（seed视为可变异参数）")
    print("  2. ✓ 识别单一变异的参数名称（mutated_param列）")
    print("  3. ✓ 填充空的hyperparam列（从models_config.json）")
    print()
    print("下一步: 验证增强后的数据质量")
