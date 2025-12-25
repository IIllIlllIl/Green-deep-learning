#!/usr/bin/env python3
"""
分析experiment.json文件的所有字段，检查80列CSV是否完整覆盖

功能：
1. 扫描所有experiment.json文件
2. 收集所有顶层字段和嵌套字段
3. 对比80列CSV格式
4. 报告缺失的字段
5. 建议需要添加的列

作者: Green
日期: 2025-12-12
"""

import json
import os
from pathlib import Path
from collections import defaultdict


# 80列标准表头
HEADER_80COL = [
    # 基础信息 (7列)
    'experiment_id', 'timestamp', 'repository', 'model',
    'training_success', 'duration_seconds', 'retries',

    # 超参数 (9列)
    'hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
    'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
    'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay',

    # 性能指标 (9列)
    'perf_accuracy', 'perf_best_val_accuracy', 'perf_map',
    'perf_precision', 'perf_rank1', 'perf_rank5',
    'perf_recall', 'perf_test_accuracy', 'perf_test_loss',

    # 能耗指标 (11列)
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
    'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
    'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent',

    # 元数据 (5列)
    'experiment_source', 'num_mutated_params', 'mutated_param', 'mode', 'error_message',

    # 前景实验详细信息 (36列)
    'fg_repository', 'fg_model', 'fg_duration_seconds', 'fg_training_success',
    'fg_retries', 'fg_error_message',
    'fg_hyperparam_alpha', 'fg_hyperparam_batch_size', 'fg_hyperparam_dropout',
    'fg_hyperparam_epochs', 'fg_hyperparam_kfold', 'fg_hyperparam_learning_rate',
    'fg_hyperparam_max_iter', 'fg_hyperparam_seed', 'fg_hyperparam_weight_decay',
    'fg_perf_accuracy', 'fg_perf_best_val_accuracy', 'fg_perf_map',
    'fg_perf_precision', 'fg_perf_rank1', 'fg_perf_rank5',
    'fg_perf_recall', 'fg_perf_test_accuracy', 'fg_perf_test_loss',
    'fg_energy_cpu_pkg_joules', 'fg_energy_cpu_ram_joules', 'fg_energy_cpu_total_joules',
    'fg_energy_gpu_avg_watts', 'fg_energy_gpu_max_watts', 'fg_energy_gpu_min_watts',
    'fg_energy_gpu_total_joules', 'fg_energy_gpu_temp_avg_celsius', 'fg_energy_gpu_temp_max_celsius',
    'fg_energy_gpu_util_avg_percent', 'fg_energy_gpu_util_max_percent',

    # 背景实验信息 (4列)
    'bg_repository', 'bg_model', 'bg_note', 'bg_log_directory'
]


def find_all_json_files():
    """查找所有experiment.json文件"""
    results_dir = Path('results')
    json_files = []

    # 老实验目录
    old_dirs = ['mutation_2x_20251122_175401', 'default', 'mutation_1x', 'archived']
    for old_dir in old_dirs:
        old_path = results_dir / old_dir
        if old_path.exists():
            json_files.extend(old_path.glob('*/experiment.json'))

    # 新实验目录
    for run_dir in results_dir.glob('run_*'):
        json_files.extend(run_dir.glob('*/experiment.json'))

    return json_files


def extract_all_fields(json_data, prefix=''):
    """递归提取JSON中的所有字段"""
    fields = set()

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            current_key = f"{prefix}{key}" if prefix else key
            fields.add(current_key)

            if isinstance(value, dict):
                # 嵌套字典
                sub_fields = extract_all_fields(value, f"{current_key}.")
                fields.update(sub_fields)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # 列表中的字典
                sub_fields = extract_all_fields(value[0], f"{current_key}[].")
                fields.update(sub_fields)

    return fields


def map_json_to_csv(json_field):
    """将JSON字段映射到CSV列名"""
    # 非并行模式映射
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

    # 能耗指标映射（需要处理命名差异）
    if json_field.startswith('energy_metrics.'):
        metric = json_field.replace('energy_metrics.', '')
        # gpu_power_* -> gpu_*_watts
        if metric.startswith('gpu_power_'):
            return f"energy_gpu_{metric.replace('gpu_power_', '')}_watts"
        # cpu_energy_* -> cpu_*_joules
        elif metric.startswith('cpu_energy_'):
            return f"energy_{metric}"
        # gpu_energy_* -> gpu_*_joules
        elif metric.startswith('gpu_energy_'):
            return f"energy_{metric}"
        # gpu_temp_*, gpu_util_*
        elif metric.startswith('gpu_'):
            return f"energy_{metric}"

    # 前景实验映射
    if json_field.startswith('foreground.'):
        sub_field = json_field.replace('foreground.', '')
        if sub_field == 'repository':
            return 'fg_repository'
        elif sub_field == 'model':
            return 'fg_model'
        elif sub_field == 'duration_seconds':
            return 'fg_duration_seconds'
        elif sub_field == 'training_success':
            return 'fg_training_success'
        elif sub_field == 'retries':
            return 'fg_retries'
        elif sub_field == 'error_message':
            return 'fg_error_message'
        elif sub_field.startswith('hyperparameters.'):
            param = sub_field.replace('hyperparameters.', '')
            return f'fg_hyperparam_{param}'
        elif sub_field.startswith('performance_metrics.'):
            metric = sub_field.replace('performance_metrics.', '')
            return f'fg_perf_{metric}'
        elif sub_field.startswith('energy_metrics.'):
            metric = sub_field.replace('energy_metrics.', '')
            if metric.startswith('gpu_power_'):
                return f"fg_energy_gpu_{metric.replace('gpu_power_', '')}_watts"
            elif metric.startswith('cpu_energy_'):
                return f"fg_energy_{metric}"
            elif metric.startswith('gpu_energy_'):
                return f"fg_energy_{metric}"
            elif metric.startswith('gpu_'):
                return f"fg_energy_{metric}"

    # 背景实验映射
    if json_field.startswith('background.'):
        sub_field = json_field.replace('background.', '')
        if sub_field == 'repository':
            return 'bg_repository'
        elif sub_field == 'model':
            return 'bg_model'
        elif sub_field == 'note':
            return 'bg_note'
        elif sub_field == 'log_directory':
            return 'bg_log_directory'
        elif sub_field.startswith('hyperparameters.'):
            param = sub_field.replace('hyperparameters.', '')
            return f'bg_hyperparam_{param}'

    return mappings.get(json_field, None)


def main():
    print("=" * 80)
    print("分析experiment.json字段覆盖情况")
    print("=" * 80)

    # 1. 查找所有JSON文件
    print("\n步骤1: 查找所有experiment.json文件...")
    json_files = find_all_json_files()
    print(f"✓ 找到 {len(json_files)} 个文件")

    # 2. 提取所有字段
    print("\n步骤2: 提取所有JSON字段...")
    all_json_fields = set()
    field_examples = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                fields = extract_all_fields(data)
                all_json_fields.update(fields)

                # 记录示例值
                for field in fields:
                    if field not in field_examples:
                        field_examples[field] = str(json_file)
        except Exception as e:
            print(f"  ✗ 读取失败: {json_file}: {e}")

    print(f"✓ 提取到 {len(all_json_fields)} 个唯一字段")

    # 3. 映射到CSV列
    print("\n步骤3: 映射JSON字段到CSV列...")
    json_to_csv_mapping = {}
    unmapped_fields = []

    for json_field in sorted(all_json_fields):
        csv_col = map_json_to_csv(json_field)
        if csv_col:
            json_to_csv_mapping[json_field] = csv_col
        else:
            unmapped_fields.append(json_field)

    print(f"✓ 已映射 {len(json_to_csv_mapping)} 个字段")
    print(f"  未映射 {len(unmapped_fields)} 个字段")

    # 4. 检查覆盖情况
    print("\n步骤4: 检查80列CSV覆盖情况...")

    csv_cols_from_json = set(json_to_csv_mapping.values())
    csv_cols_in_header = set(HEADER_80COL)

    # CSV中有但JSON中没有的列（元数据列）
    metadata_cols = csv_cols_in_header - csv_cols_from_json

    # JSON中有但CSV中没有的列
    missing_in_csv = csv_cols_from_json - csv_cols_in_header

    print(f"  80列CSV表头中的列数: {len(HEADER_80COL)}")
    print(f"  从JSON映射得到的列数: {len(csv_cols_from_json)}")
    print(f"  元数据列（CSV特有）: {len(metadata_cols)}")
    print(f"  JSON中有但CSV缺失: {len(missing_in_csv)}")

    # 5. 详细报告
    print("\n" + "=" * 80)
    print("详细分析报告")
    print("=" * 80)

    print("\n【1】JSON中所有字段分类:")
    print("-" * 80)

    # 按类别分组
    categories = defaultdict(list)
    for field in sorted(all_json_fields):
        if field.startswith('foreground.'):
            categories['前景实验'].append(field)
        elif field.startswith('background.'):
            categories['背景实验'].append(field)
        elif field.startswith('hyperparameters.'):
            categories['超参数'].append(field)
        elif field.startswith('performance_metrics.'):
            categories['性能指标'].append(field)
        elif field.startswith('energy_metrics.'):
            categories['能耗指标'].append(field)
        else:
            categories['基础字段'].append(field)

    for category, fields in sorted(categories.items()):
        print(f"\n{category} ({len(fields)}个):")
        for field in fields:
            csv_col = json_to_csv_mapping.get(field, '❌ 未映射')
            in_csv = '✓' if csv_col in csv_cols_in_header else '✗'
            print(f"  {in_csv} {field:50s} -> {csv_col}")

    print("\n【2】CSV中缺失的JSON字段:")
    print("-" * 80)
    if missing_in_csv:
        for csv_col in sorted(missing_in_csv):
            # 找到对应的JSON字段
            json_fields = [jf for jf, cc in json_to_csv_mapping.items() if cc == csv_col]
            print(f"  ✗ {csv_col:50s} <- {', '.join(json_fields)}")
    else:
        print("  ✓ 无缺失")

    print("\n【3】未映射的JSON字段:")
    print("-" * 80)
    if unmapped_fields:
        for field in sorted(unmapped_fields):
            print(f"  ⚠ {field}")
    else:
        print("  ✓ 所有字段已映射")

    print("\n【4】元数据列（CSV特有，不来自JSON）:")
    print("-" * 80)
    for col in sorted(metadata_cols):
        print(f"  • {col}")

    # 6. 建议
    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)

    if missing_in_csv:
        new_total = len(HEADER_80COL) + len(missing_in_csv)
        print(f"\n需要添加 {len(missing_in_csv)} 列到CSV格式")
        print(f"新的总列数: {new_total} 列")
        print(f"\n建议新增的列:")
        for i, csv_col in enumerate(sorted(missing_in_csv), 1):
            print(f"  {i}. {csv_col}")
    else:
        print("\n✓ 当前80列CSV格式已完整覆盖所有JSON字段！")

    if unmapped_fields:
        print(f"\n⚠ 有 {len(unmapped_fields)} 个JSON字段未映射，可能需要添加映射规则")

    # 7. 输出完整字段列表
    print("\n保存完整字段分析到文件...")
    with open('results/json_field_analysis.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Experiment.json字段完整分析\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总JSON文件数: {len(json_files)}\n")
        f.write(f"唯一JSON字段数: {len(all_json_fields)}\n")
        f.write(f"已映射字段数: {len(json_to_csv_mapping)}\n")
        f.write(f"80列CSV列数: {len(HEADER_80COL)}\n")
        f.write(f"缺失CSV列数: {len(missing_in_csv)}\n\n")

        f.write("所有JSON字段详细列表:\n")
        f.write("-" * 80 + "\n")
        for field in sorted(all_json_fields):
            csv_col = json_to_csv_mapping.get(field, '未映射')
            in_csv = '✓' if csv_col in csv_cols_in_header else '✗'
            f.write(f"{in_csv} {field:60s} -> {csv_col}\n")

    print(f"✓ 已保存到 results/json_field_analysis.txt")

    return len(missing_in_csv)


if __name__ == '__main__':
    missing_count = main()

    if missing_count > 0:
        print(f"\n结论: 需要扩展CSV格式，添加 {missing_count} 列")
        exit(1)
    else:
        print(f"\n结论: 80列CSV格式完整覆盖所有JSON字段")
        exit(0)
