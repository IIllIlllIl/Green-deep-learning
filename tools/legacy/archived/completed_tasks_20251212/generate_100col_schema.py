#!/usr/bin/env python3
"""
生成100列CSV格式定义（完整覆盖所有experiment.json字段）

修复的关键点:
1. 能耗指标字段名映射错误（重复添加后缀）
2. 缺少背景实验的超参数字段（6个）

作者: Claude Code
日期: 2025-12-12
"""

import json
from pathlib import Path
from collections import defaultdict


def generate_100col_header():
    """生成100列标准表头"""

    header = [
        # ============ 基础信息 (7列) ============
        'experiment_id', 'timestamp', 'repository', 'model',
        'training_success', 'duration_seconds', 'retries',

        # ============ 超参数 (9列) ============
        'hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
        'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
        'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay',

        # ============ 性能指标 (9列) ============
        'perf_accuracy', 'perf_best_val_accuracy', 'perf_map',
        'perf_precision', 'perf_rank1', 'perf_rank5',
        'perf_recall', 'perf_test_accuracy', 'perf_test_loss',

        # ============ 能耗指标 (11列) ============
        # 修复: JSON字段已包含单位后缀，无需重复添加
        'energy_cpu_pkg_joules',           # <- energy_metrics.cpu_energy_pkg_joules
        'energy_cpu_ram_joules',           # <- energy_metrics.cpu_energy_ram_joules
        'energy_cpu_total_joules',         # <- energy_metrics.cpu_energy_total_joules
        'energy_gpu_avg_watts',            # <- energy_metrics.gpu_power_avg_watts
        'energy_gpu_max_watts',            # <- energy_metrics.gpu_power_max_watts
        'energy_gpu_min_watts',            # <- energy_metrics.gpu_power_min_watts
        'energy_gpu_total_joules',         # <- energy_metrics.gpu_energy_total_joules
        'energy_gpu_temp_avg_celsius',     # <- energy_metrics.gpu_temp_avg_celsius
        'energy_gpu_temp_max_celsius',     # <- energy_metrics.gpu_temp_max_celsius
        'energy_gpu_util_avg_percent',     # <- energy_metrics.gpu_util_avg_percent
        'energy_gpu_util_max_percent',     # <- energy_metrics.gpu_util_max_percent

        # ============ 元数据 (5列) ============
        'experiment_source', 'num_mutated_params', 'mutated_param', 'mode', 'error_message',

        # ============ 前景实验详细信息 (42列) ============
        # 前景基础信息 (6列)
        'fg_repository', 'fg_model', 'fg_duration_seconds', 'fg_training_success',
        'fg_retries', 'fg_error_message',

        # 前景超参数 (9列)
        'fg_hyperparam_alpha', 'fg_hyperparam_batch_size', 'fg_hyperparam_dropout',
        'fg_hyperparam_epochs', 'fg_hyperparam_kfold', 'fg_hyperparam_learning_rate',
        'fg_hyperparam_max_iter', 'fg_hyperparam_seed', 'fg_hyperparam_weight_decay',

        # 前景性能指标 (9列)
        'fg_perf_accuracy', 'fg_perf_best_val_accuracy', 'fg_perf_map',
        'fg_perf_precision', 'fg_perf_rank1', 'fg_perf_rank5',
        'fg_perf_recall', 'fg_perf_test_accuracy', 'fg_perf_test_loss',

        # 前景能耗指标 (11列) - 修复: 不重复添加单位后缀
        'fg_energy_cpu_pkg_joules',
        'fg_energy_cpu_ram_joules',
        'fg_energy_cpu_total_joules',
        'fg_energy_gpu_avg_watts',
        'fg_energy_gpu_max_watts',
        'fg_energy_gpu_min_watts',
        'fg_energy_gpu_total_joules',
        'fg_energy_gpu_temp_avg_celsius',
        'fg_energy_gpu_temp_max_celsius',
        'fg_energy_gpu_util_avg_percent',
        'fg_energy_gpu_util_max_percent',

        # ============ 背景实验信息 (17列) - 新增6个超参数列 ============
        # 背景基础信息 (4列)
        'bg_repository', 'bg_model', 'bg_note', 'bg_log_directory',

        # 背景超参数 (6列) - **新增**: 之前80列格式缺失这些
        'bg_hyperparam_batch_size', 'bg_hyperparam_dropout',
        'bg_hyperparam_epochs', 'bg_hyperparam_learning_rate',
        'bg_hyperparam_seed', 'bg_hyperparam_weight_decay',

        # 背景能耗指标 (7列) - **新增**: 80列格式缺失这些
        'bg_energy_cpu_pkg_joules',
        'bg_energy_cpu_ram_joules',
        'bg_energy_cpu_total_joules',
        'bg_energy_gpu_avg_watts',
        'bg_energy_gpu_max_watts',
        'bg_energy_gpu_min_watts',
        'bg_energy_gpu_total_joules',
    ]

    return header


def map_json_to_csv_fixed(json_field):
    """
    修复后的JSON字段到CSV列名映射函数

    关键修复:
    1. 能耗字段不重复添加单位后缀
    2. 支持背景实验的超参数和能耗字段
    """

    # 基础字段映射
    mappings = {
        'experiment_id': 'experiment_id',
        'timestamp': 'timestamp',
        'repository': 'repository',
        'model': 'model',
        'training_success': 'training_success',
        'duration_seconds': 'duration_seconds',
        'retries': 'retries',
        'error_message': 'error_message',
        'mode': 'mode',
    }

    # 超参数映射
    if json_field.startswith('hyperparameters.'):
        param = json_field.replace('hyperparameters.', '')
        return f'hyperparam_{param}'

    # 性能指标映射
    if json_field.startswith('performance_metrics.'):
        metric = json_field.replace('performance_metrics.', '')
        return f'perf_{metric}'

    # 能耗指标映射（修复: 不重复添加单位后缀）
    if json_field.startswith('energy_metrics.'):
        metric = json_field.replace('energy_metrics.', '')

        # 映射规则:
        # cpu_energy_pkg_joules -> energy_cpu_pkg_joules (移除energy前缀，添加energy_前缀)
        # gpu_power_avg_watts -> energy_gpu_avg_watts (power改为直接添加energy_前缀)
        # gpu_energy_total_joules -> energy_gpu_total_joules
        # gpu_temp_avg_celsius -> energy_gpu_temp_avg_celsius

        if metric.startswith('cpu_energy_'):
            # cpu_energy_pkg_joules -> energy_cpu_pkg_joules
            return f"energy_{metric.replace('cpu_energy_', 'cpu_')}"
        elif metric.startswith('gpu_power_'):
            # gpu_power_avg_watts -> energy_gpu_avg_watts
            return f"energy_{metric.replace('gpu_power_', 'gpu_')}"
        elif metric.startswith('gpu_energy_'):
            # gpu_energy_total_joules -> energy_gpu_total_joules
            return f"energy_{metric.replace('gpu_energy_', 'gpu_')}"
        else:
            # gpu_temp_avg_celsius, gpu_util_avg_percent
            return f"energy_{metric}"

    # 前景实验映射
    if json_field.startswith('foreground.'):
        sub_field = json_field.replace('foreground.', '')

        if sub_field in ['repository', 'model', 'duration_seconds', 'training_success', 'retries', 'error_message']:
            return f'fg_{sub_field}'
        elif sub_field.startswith('hyperparameters.'):
            param = sub_field.replace('hyperparameters.', '')
            return f'fg_hyperparam_{param}'
        elif sub_field.startswith('performance_metrics.'):
            metric = sub_field.replace('performance_metrics.', '')
            return f'fg_perf_{metric}'
        elif sub_field.startswith('energy_metrics.'):
            metric = sub_field.replace('energy_metrics.', '')
            # 使用相同的修复逻辑
            if metric.startswith('cpu_energy_'):
                return f"fg_energy_{metric.replace('cpu_energy_', 'cpu_')}"
            elif metric.startswith('gpu_power_'):
                return f"fg_energy_{metric.replace('gpu_power_', 'gpu_')}"
            elif metric.startswith('gpu_energy_'):
                return f"fg_energy_{metric.replace('gpu_energy_', 'gpu_')}"
            else:
                return f"fg_energy_{metric}"

    # 背景实验映射（新增超参数和能耗支持）
    if json_field.startswith('background.'):
        sub_field = json_field.replace('background.', '')

        if sub_field in ['repository', 'model', 'note', 'log_directory']:
            return f'bg_{sub_field}'
        elif sub_field.startswith('hyperparameters.'):
            param = sub_field.replace('hyperparameters.', '')
            return f'bg_hyperparam_{param}'
        elif sub_field.startswith('energy_metrics.'):
            metric = sub_field.replace('energy_metrics.', '')
            # 使用相同的修复逻辑
            if metric.startswith('cpu_energy_'):
                return f"bg_energy_{metric.replace('cpu_energy_', 'cpu_')}"
            elif metric.startswith('gpu_power_'):
                return f"bg_energy_{metric.replace('gpu_power_', 'gpu_')}"
            elif metric.startswith('gpu_energy_'):
                return f"bg_energy_{metric.replace('gpu_energy_', 'gpu_')}"
            else:
                return f"bg_energy_{metric}"

    return mappings.get(json_field, None)


def extract_all_fields(json_data, prefix=''):
    """递归提取JSON中的所有字段"""
    fields = set()

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            current_key = f"{prefix}{key}" if prefix else key
            fields.add(current_key)

            if isinstance(value, dict):
                sub_fields = extract_all_fields(value, f"{current_key}.")
                fields.update(sub_fields)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                sub_fields = extract_all_fields(value[0], f"{current_key}[].")
                fields.update(sub_fields)

    return fields


def main():
    print("=" * 80)
    print("100列CSV格式定义生成")
    print("=" * 80)

    # 生成100列表头
    header_100col = generate_100col_header()

    print(f"\n生成的100列表头:")
    print(f"  总列数: {len(header_100col)}")

    # 按类别统计
    categories = {
        '基础信息': [c for c in header_100col if not any(c.startswith(p) for p in ['hyperparam_', 'perf_', 'energy_', 'fg_', 'bg_', 'experiment_source', 'num_mutated_params', 'mutated_param', 'mode'])],
        '超参数': [c for c in header_100col if c.startswith('hyperparam_')],
        '性能指标': [c for c in header_100col if c.startswith('perf_')],
        '能耗指标': [c for c in header_100col if c.startswith('energy_') and not c.startswith(('fg_', 'bg_'))],
        '元数据': ['experiment_source', 'num_mutated_params', 'mutated_param', 'mode', 'error_message'],
        '前景实验': [c for c in header_100col if c.startswith('fg_')],
        '背景实验': [c for c in header_100col if c.startswith('bg_')],
    }

    for category, cols in categories.items():
        print(f"\n{category}: {len(cols)}列")
        for i, col in enumerate(cols, 1):
            print(f"  {i:2d}. {col}")

    # 验证JSON字段覆盖
    print("\n" + "=" * 80)
    print("验证JSON字段覆盖")
    print("=" * 80)

    # 查找所有JSON文件
    results_dir = Path('results')
    json_files = []

    for old_dir in ['mutation_2x_20251122_175401', 'default', 'mutation_1x', 'archived']:
        old_path = results_dir / old_dir
        if old_path.exists():
            json_files.extend(old_path.glob('*/experiment.json'))

    for run_dir in results_dir.glob('run_*'):
        json_files.extend(run_dir.glob('*/experiment.json'))

    print(f"\n找到 {len(json_files)} 个experiment.json文件")

    # 提取所有字段
    all_json_fields = set()
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                fields = extract_all_fields(data)
                all_json_fields.update(fields)
        except Exception as e:
            print(f"  ✗ 读取失败: {json_file}: {e}")

    print(f"提取到 {len(all_json_fields)} 个唯一JSON字段")

    # 映射字段
    json_to_csv_mapping = {}
    unmapped_fields = []

    for json_field in sorted(all_json_fields):
        csv_col = map_json_to_csv_fixed(json_field)
        if csv_col:
            json_to_csv_mapping[json_field] = csv_col
        else:
            unmapped_fields.append(json_field)

    # 检查覆盖情况
    csv_cols_from_json = set(json_to_csv_mapping.values())
    csv_cols_in_header = set(header_100col)

    metadata_cols = csv_cols_in_header - csv_cols_from_json
    missing_in_csv = csv_cols_from_json - csv_cols_in_header

    print(f"\n覆盖情况:")
    print(f"  100列表头中的列数: {len(header_100col)}")
    print(f"  从JSON映射得到的列数: {len(csv_cols_from_json)}")
    print(f"  元数据列（CSV特有）: {len(metadata_cols)}")
    print(f"  JSON中有但CSV缺失: {len(missing_in_csv)}")

    if missing_in_csv:
        print(f"\n⚠️ 仍有缺失字段:")
        for col in sorted(missing_in_csv):
            json_fields = [jf for jf, cc in json_to_csv_mapping.items() if cc == col]
            print(f"  - {col} <- {', '.join(json_fields)}")
    else:
        print(f"\n✓ 100列格式完整覆盖所有JSON字段!")

    if unmapped_fields:
        print(f"\n⚠️ 有 {len(unmapped_fields)} 个JSON字段未映射:")
        for field in sorted(unmapped_fields):
            print(f"  - {field}")

    # 输出100列定义到文件
    output_file = 'results/100col_schema_definition.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("100列CSV格式完整定义\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总列数: {len(header_100col)}\n")
        f.write(f"JSON文件数: {len(json_files)}\n")
        f.write(f"JSON字段数: {len(all_json_fields)}\n")
        f.write(f"已映射字段: {len(json_to_csv_mapping)}\n")
        f.write(f"缺失字段: {len(missing_in_csv)}\n\n")

        f.write("100列表头:\n")
        f.write("-" * 80 + "\n")
        for i, col in enumerate(header_100col, 1):
            f.write(f"{i:3d}. {col}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("JSON字段映射详情\n")
        f.write("=" * 80 + "\n\n")

        for field in sorted(all_json_fields):
            csv_col = json_to_csv_mapping.get(field, '未映射')
            in_csv = '✓' if csv_col in csv_cols_in_header else '✗'
            f.write(f"{in_csv} {field:60s} -> {csv_col}\n")

    print(f"\n✓ 100列定义已保存到: {output_file}")

    # 生成Python代码
    code_output = 'results/100col_header_code.py'
    with open(code_output, 'w') as f:
        f.write("# 100列CSV标准表头定义\n")
        f.write("# 自动生成于: 2025-12-12\n\n")
        f.write("HEADER_100COL = [\n")
        for col in header_100col:
            f.write(f"    '{col}',\n")
        f.write("]\n\n")
        f.write(f"# 总列数: {len(header_100col)}\n")

    print(f"✓ Python代码已保存到: {code_output}")

    return len(missing_in_csv)


if __name__ == '__main__':
    missing_count = main()

    if missing_count > 0:
        print(f"\n⚠️ 仍需添加 {missing_count} 列")
        exit(1)
    else:
        print(f"\n✅ 100列格式完整覆盖所有JSON字段")
        exit(0)
