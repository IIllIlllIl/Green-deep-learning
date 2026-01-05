#!/usr/bin/env python3
"""
从experiment.json直接重建summary_old.csv（93列格式）

关键特性：
1. 不使用旧CSV数据，完全从JSON重建
2. 使用修复后的93列格式（包含所有背景实验信息）
3. 正确的字段映射（不重复添加单位后缀）
4. 改进的JSON文件查找逻辑（支持老实验路径）

作者: Claude Code
日期: 2025-12-12
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# 93列标准表头
HEADER_93COL = [
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

    # 前景实验详细信息 (42列)
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

    # 背景实验信息 (17列) - 完整版本
    'bg_repository', 'bg_model', 'bg_note', 'bg_log_directory',
    'bg_hyperparam_batch_size', 'bg_hyperparam_dropout',
    'bg_hyperparam_epochs', 'bg_hyperparam_learning_rate',
    'bg_hyperparam_seed', 'bg_hyperparam_weight_decay',
    'bg_energy_cpu_pkg_joules', 'bg_energy_cpu_ram_joules', 'bg_energy_cpu_total_joules',
    'bg_energy_gpu_avg_watts', 'bg_energy_gpu_max_watts', 'bg_energy_gpu_min_watts',
    'bg_energy_gpu_total_joules',
]


def load_whitelist():
    """加载summary_old.csv中的实验ID白名单"""
    whitelist = []
    try:
        with open('results/summary_old.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                whitelist.append(row['experiment_id'])
    except Exception as e:
        print(f"⚠️ 无法加载白名单: {e}")
        return []

    print(f"✓ 加载白名单: {len(whitelist)} 个实验ID")
    return whitelist


def find_experiment_json(experiment_id):
    """
    改进的JSON文件查找逻辑
    支持老实验和新实验的不同路径结构

    experiment_id格式: {source}__{directory_name}
    如: default__MRT-OAST_default_001, mutation_2x_safe__examples_mnist_013

    实际目录名: {directory_name}
    如: MRT-OAST_default_001, examples_mnist_013
    """
    results_dir = Path('results')

    # 提取实际目录名（移除source前缀）
    if '__' in experiment_id:
        source_prefix, dir_name = experiment_id.split('__', 1)
    else:
        dir_name = experiment_id
        source_prefix = ''

    # 1. 尝试老实验目录
    old_dirs = [
        'mutation_2x_20251122_175401',
        'default',
        'mutation_1x',
        'archived'
    ]
    for old_dir in old_dirs:
        json_path = results_dir / old_dir / dir_name / 'experiment.json'
        if json_path.exists():
            return json_path

    # 2. 尝试新实验目录（run_YYYYMMDD_HHMMSS）
    for run_dir in sorted(results_dir.glob('run_*')):
        json_path = run_dir / dir_name / 'experiment.json'
        if json_path.exists():
            return json_path

    return None


def safe_get_nested(data, path, default=''):
    """
    安全获取嵌套字典值
    path: 点分隔的路径字符串，如 "foreground.hyperparameters.batch_size"
    """
    keys = path.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current if current is not None else default


def map_energy_field(metric):
    """
    修复后的能耗字段映射
    不重复添加单位后缀
    """
    if metric.startswith('cpu_energy_'):
        # cpu_energy_pkg_joules -> cpu_pkg_joules
        return metric.replace('cpu_energy_', 'cpu_')
    elif metric.startswith('gpu_power_'):
        # gpu_power_avg_watts -> gpu_avg_watts
        return metric.replace('gpu_power_', 'gpu_')
    elif metric.startswith('gpu_energy_'):
        # gpu_energy_total_joules -> gpu_total_joules
        return metric.replace('gpu_energy_', 'gpu_')
    else:
        # gpu_temp_avg_celsius, gpu_util_avg_percent
        return metric


def extract_row_from_json(json_data, experiment_id):
    """
    从experiment.json提取一行93列CSV数据
    """
    row = {col: '' for col in HEADER_93COL}

    # 确定模式
    mode = json_data.get('mode', '')
    is_parallel = (mode == 'parallel')

    # 基础信息
    row['experiment_id'] = experiment_id
    row['timestamp'] = json_data.get('timestamp', '')
    row['mode'] = mode
    row['training_success'] = json_data.get('training_success', '')
    row['duration_seconds'] = json_data.get('duration_seconds', '')
    row['retries'] = json_data.get('retries', '')
    row['error_message'] = json_data.get('error_message', '')

    if is_parallel:
        # 并行模式：前景+背景
        foreground = json_data.get('foreground', {})
        background = json_data.get('background', {})

        # 前景基础信息
        row['fg_repository'] = foreground.get('repository', '')
        row['fg_model'] = foreground.get('model', '')
        row['fg_duration_seconds'] = foreground.get('duration_seconds', '')
        row['fg_training_success'] = foreground.get('training_success', '')
        row['fg_retries'] = foreground.get('retries', '')
        row['fg_error_message'] = foreground.get('error_message', '')

        # 前景超参数
        fg_hyperparams = foreground.get('hyperparameters', {})
        for param in ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                      'learning_rate', 'max_iter', 'seed', 'weight_decay']:
            if param in fg_hyperparams:
                row[f'fg_hyperparam_{param}'] = fg_hyperparams[param]

        # 前景性能指标
        fg_perf = foreground.get('performance_metrics', {})
        for metric in ['accuracy', 'best_val_accuracy', 'map', 'precision',
                       'rank1', 'rank5', 'recall', 'test_accuracy', 'test_loss']:
            if metric in fg_perf:
                row[f'fg_perf_{metric}'] = fg_perf[metric]

        # 前景能耗指标
        fg_energy = foreground.get('energy_metrics', {})
        for json_field, value in fg_energy.items():
            csv_field = f"fg_energy_{map_energy_field(json_field)}"
            if csv_field in row:
                row[csv_field] = value

        # 背景基础信息
        row['bg_repository'] = background.get('repository', '')
        row['bg_model'] = background.get('model', '')
        row['bg_note'] = background.get('note', '')
        row['bg_log_directory'] = background.get('log_directory', '')

        # 背景超参数（新增）
        bg_hyperparams = background.get('hyperparameters', {})
        for param in ['batch_size', 'dropout', 'epochs', 'learning_rate', 'seed', 'weight_decay']:
            if param in bg_hyperparams:
                row[f'bg_hyperparam_{param}'] = bg_hyperparams[param]

        # 背景能耗指标（新增）
        bg_energy = background.get('energy_metrics', {})
        for json_field, value in bg_energy.items():
            csv_field = f"bg_energy_{map_energy_field(json_field)}"
            if csv_field in row:
                row[csv_field] = value

    else:
        # 非并行模式：顶层数据
        row['repository'] = json_data.get('repository', '')
        row['model'] = json_data.get('model', '')

        # 超参数
        hyperparams = json_data.get('hyperparameters', {})
        for param in ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                      'learning_rate', 'max_iter', 'seed', 'weight_decay']:
            if param in hyperparams:
                row[f'hyperparam_{param}'] = hyperparams[param]

        # 性能指标
        perf = json_data.get('performance_metrics', {})
        for metric in ['accuracy', 'best_val_accuracy', 'map', 'precision',
                       'rank1', 'rank5', 'recall', 'test_accuracy', 'test_loss']:
            if metric in perf:
                row[f'perf_{metric}'] = perf[metric]

        # 能耗指标
        energy = json_data.get('energy_metrics', {})
        for json_field, value in energy.items():
            csv_field = f"energy_{map_energy_field(json_field)}"
            if csv_field in row:
                row[csv_field] = value

    return row


def infer_experiment_source(experiment_id):
    """从experiment_id推断实验来源"""
    if 'default' in experiment_id:
        return 'default'
    elif 'mutation_1x' in experiment_id:
        return 'mutation_1x'
    elif 'mutation_2x' in experiment_id:
        return 'mutation_2x_safe'
    else:
        return 'unknown'


def calculate_num_mutated_params(row):
    """
    计算变异参数数量
    基于非空的hyperparam_*列（或fg_hyperparam_*列）
    """
    # 确定使用哪组超参数列
    if row.get('mode') == 'parallel':
        param_prefix = 'fg_hyperparam_'
    else:
        param_prefix = 'hyperparam_'

    # 标准参数列表
    standard_params = ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                       'learning_rate', 'max_iter', 'seed', 'weight_decay']

    # 计数非空参数
    count = 0
    mutated_param = None

    for param in standard_params:
        col_name = f'{param_prefix}{param}'
        if col_name in row and row[col_name] not in ['', None]:
            count += 1
            if count == 1:
                mutated_param = param

    return count, mutated_param if count == 1 else ''


def rebuild_summary_old_from_json():
    """
    主函数：从experiment.json重建summary_old.csv（93列）
    """
    print("=" * 80)
    print("从experiment.json重建summary_old.csv（93列格式）")
    print("=" * 80)

    # 1. 加载白名单
    print("\n步骤1: 加载实验ID白名单...")
    whitelist = load_whitelist()
    if not whitelist:
        print("❌ 无法加载白名单，终止")
        return False

    # 2. 查找JSON文件
    print("\n步骤2: 查找experiment.json文件...")
    found_jsons = {}
    not_found = []

    for exp_id in whitelist:
        json_path = find_experiment_json(exp_id)
        if json_path:
            found_jsons[exp_id] = json_path
        else:
            not_found.append(exp_id)

    print(f"  ✓ 找到JSON文件: {len(found_jsons)}/{len(whitelist)}")
    if not_found:
        print(f"  ⚠️ 未找到JSON文件: {len(not_found)}个")
        print(f"    前5个: {not_found[:5]}")

    if len(found_jsons) == 0:
        print("❌ 未找到任何JSON文件，终止")
        return False

    # 3. 提取数据
    print("\n步骤3: 从JSON提取数据...")
    rows = []
    success_count = 0
    error_count = 0

    for exp_id, json_path in found_jsons.items():
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # 提取行数据
            row = extract_row_from_json(json_data, exp_id)

            # 推断experiment_source
            row['experiment_source'] = infer_experiment_source(exp_id)

            # 计算num_mutated_params和mutated_param
            num_mutated, mutated_param = calculate_num_mutated_params(row)
            row['num_mutated_params'] = num_mutated
            row['mutated_param'] = mutated_param

            rows.append(row)
            success_count += 1

        except Exception as e:
            print(f"  ✗ 解析失败 {exp_id}: {e}")
            error_count += 1

    print(f"  ✓ 成功提取: {success_count}个")
    if error_count > 0:
        print(f"  ✗ 提取失败: {error_count}个")

    # 4. 按timestamp排序
    print("\n步骤4: 排序数据...")
    rows.sort(key=lambda x: x.get('timestamp', ''))

    # 5. 写入CSV
    print("\n步骤5: 写入93列CSV文件...")

    # 备份现有文件
    backup_path = f"results/summary_old.csv.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if Path('results/summary_old.csv').exists():
        import shutil
        shutil.copy('results/summary_old.csv', backup_path)
        print(f"  ✓ 备份现有文件: {backup_path}")

    # 写入新文件
    output_file = 'results/summary_old_93col.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER_93COL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ 写入完成: {output_file}")
    print(f"    行数: {len(rows)}")
    print(f"    列数: {len(HEADER_93COL)}")

    # 6. 数据质量统计
    print("\n步骤6: 数据质量统计...")

    stats = {
        'training_success': 0,
        'cpu_energy': 0,
        'gpu_energy': 0,
        'performance': 0,
        'parallel_mode': 0,
        'bg_hyperparams': 0,
        'bg_energy': 0,
    }

    for row in rows:
        if row.get('training_success') == 'True':
            stats['training_success'] += 1
        if row.get('energy_cpu_total_joules'):
            stats['cpu_energy'] += 1
        if row.get('energy_gpu_total_joules') or row.get('fg_energy_gpu_total_joules'):
            stats['gpu_energy'] += 1
        if row.get('perf_accuracy') or row.get('fg_perf_accuracy'):
            stats['performance'] += 1
        if row.get('mode') == 'parallel':
            stats['parallel_mode'] += 1
        if row.get('bg_hyperparam_batch_size'):
            stats['bg_hyperparams'] += 1
        if row.get('bg_energy_cpu_total_joules'):
            stats['bg_energy'] += 1

    total = len(rows)
    print(f"  训练成功: {stats['training_success']}/{total} ({stats['training_success']/total*100:.1f}%)")
    print(f"  CPU能耗完整: {stats['cpu_energy']}/{total} ({stats['cpu_energy']/total*100:.1f}%)")
    print(f"  GPU能耗完整: {stats['gpu_energy']}/{total} ({stats['gpu_energy']/total*100:.1f}%)")
    print(f"  性能数据完整: {stats['performance']}/{total} ({stats['performance']/total*100:.1f}%)")
    print(f"  并行模式实验: {stats['parallel_mode']}/{total} ({stats['parallel_mode']/total*100:.1f}%)")
    print(f"  背景超参数填充: {stats['bg_hyperparams']}/{total} ({stats['bg_hyperparams']/total*100:.1f}%)")
    print(f"  背景能耗填充: {stats['bg_energy']}/{total} ({stats['bg_energy']/total*100:.1f}%)")

    print("\n" + "=" * 80)
    print("✅ 重建完成！")
    print("=" * 80)
    print(f"\n输出文件: {output_file}")
    print(f"备份文件: {backup_path}")
    print(f"\n下一步:")
    print("  1. 验证数据完整性")
    print("  2. 如果验证通过，替换原文件:")
    print(f"     mv {output_file} results/summary_old.csv")

    return True


if __name__ == '__main__':
    success = rebuild_summary_old_from_json()
    exit(0 if success else 1)
