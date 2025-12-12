#!/usr/bin/env python3
"""
将summary_old.csv从37列格式转换为80列格式

功能：
1. 读取37列格式的summary_old.csv
2. 从对应的experiment.json读取完整数据
3. 输出80列格式的summary_old_80col.csv
4. 保持与summary_new.csv格式一致

作者: Green
日期: 2025-12-12
版本: 1.0
"""

import csv
import json
import os
from pathlib import Path
from datetime import datetime


# 80列标准表头（与summary_new.csv一致）
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


def find_experiment_json(experiment_source, experiment_id):
    """查找experiment.json文件"""
    # 老实验的可能位置
    old_dirs = [
        'results/mutation_2x_20251122_175401',
        'results/default',
        'results/mutation_1x',
        'results/archived'
    ]

    # 新实验的位置
    new_dirs = list(Path('results').glob('run_*'))

    all_dirs = old_dirs + [str(d) for d in new_dirs]

    for base_dir in all_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue

        # 尝试直接匹配实验目录
        for exp_dir in base_path.glob('*'):
            if not exp_dir.is_dir():
                continue

            json_file = exp_dir / 'experiment.json'
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if data.get('experiment_id') == experiment_id:
                            return json_file, data
                except:
                    continue

    return None, None


def extract_from_json(json_data, old_row):
    """从experiment.json提取数据，填充80列格式"""
    row = {col: '' for col in HEADER_80COL}

    # 检查是否是并行实验
    is_parallel = json_data.get('mode') == 'parallel'

    # 基础信息
    row['experiment_id'] = json_data.get('experiment_id', '')
    row['timestamp'] = json_data.get('timestamp', '')

    if is_parallel:
        # 并行模式：主数据在foreground中
        fg = json_data.get('foreground', {})
        bg = json_data.get('background', {})

        row['repository'] = fg.get('repository', '')
        row['model'] = fg.get('model', '')
        row['training_success'] = str(fg.get('training_success', ''))
        row['duration_seconds'] = str(fg.get('duration_seconds', ''))
        row['retries'] = str(fg.get('retries', ''))
        row['error_message'] = fg.get('error_message', '')

        # 超参数
        hyperparams = fg.get('hyperparameters', {})
        for param in ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                      'learning_rate', 'max_iter', 'seed', 'weight_decay']:
            row[f'hyperparam_{param}'] = str(hyperparams.get(param, ''))

        # 性能指标
        perf = fg.get('performance_metrics', {})
        for metric in ['accuracy', 'best_val_accuracy', 'map', 'precision',
                       'rank1', 'rank5', 'recall', 'test_accuracy', 'test_loss']:
            row[f'perf_{metric}'] = str(perf.get(metric, ''))

        # 能耗指标
        energy = fg.get('energy_metrics', {})
        for metric in ['cpu_pkg_joules', 'cpu_ram_joules', 'cpu_total_joules',
                       'gpu_avg_watts', 'gpu_max_watts', 'gpu_min_watts',
                       'gpu_total_joules', 'gpu_temp_avg_celsius', 'gpu_temp_max_celsius',
                       'gpu_util_avg_percent', 'gpu_util_max_percent']:
            row[f'energy_{metric}'] = str(energy.get(metric, ''))

        # 填充fg_*列（并行模式特有）
        row['fg_repository'] = fg.get('repository', '')
        row['fg_model'] = fg.get('model', '')
        row['fg_training_success'] = str(fg.get('training_success', ''))
        row['fg_duration_seconds'] = str(fg.get('duration_seconds', ''))
        row['fg_retries'] = str(fg.get('retries', ''))
        row['fg_error_message'] = fg.get('error_message', '')

        for param in ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                      'learning_rate', 'max_iter', 'seed', 'weight_decay']:
            row[f'fg_hyperparam_{param}'] = str(hyperparams.get(param, ''))

        for metric in ['accuracy', 'best_val_accuracy', 'map', 'precision',
                       'rank1', 'rank5', 'recall', 'test_accuracy', 'test_loss']:
            row[f'fg_perf_{metric}'] = str(perf.get(metric, ''))

        for metric in ['cpu_pkg_joules', 'cpu_ram_joules', 'cpu_total_joules',
                       'gpu_avg_watts', 'gpu_max_watts', 'gpu_min_watts',
                       'gpu_total_joules', 'gpu_temp_avg_celsius', 'gpu_temp_max_celsius',
                       'gpu_util_avg_percent', 'gpu_util_max_percent']:
            row[f'fg_energy_{metric}'] = str(energy.get(metric, ''))

        # 背景实验信息
        row['bg_repository'] = bg.get('repository', '')
        row['bg_model'] = bg.get('model', '')
        row['bg_note'] = bg.get('note', '')
        row['bg_log_directory'] = bg.get('log_directory', '')

        row['mode'] = 'parallel'

    else:
        # 非并行模式：数据在顶层
        row['repository'] = json_data.get('repository', '')
        row['model'] = json_data.get('model', '')
        row['training_success'] = str(json_data.get('training_success', ''))
        row['duration_seconds'] = str(json_data.get('duration_seconds', ''))
        row['retries'] = str(json_data.get('retries', ''))
        row['error_message'] = json_data.get('error_message', '')

        # 超参数
        hyperparams = json_data.get('hyperparameters', {})
        for param in ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                      'learning_rate', 'max_iter', 'seed', 'weight_decay']:
            row[f'hyperparam_{param}'] = str(hyperparams.get(param, ''))

        # 性能指标
        perf = json_data.get('performance_metrics', {})
        for metric in ['accuracy', 'best_val_accuracy', 'map', 'precision',
                       'rank1', 'rank5', 'recall', 'test_accuracy', 'test_loss']:
            row[f'perf_{metric}'] = str(perf.get(metric, ''))

        # 能耗指标
        energy = json_data.get('energy_metrics', {})
        for metric in ['cpu_pkg_joules', 'cpu_ram_joules', 'cpu_total_joules',
                       'gpu_avg_watts', 'gpu_max_watts', 'gpu_min_watts',
                       'gpu_total_joules', 'gpu_temp_avg_celsius', 'gpu_temp_max_celsius',
                       'gpu_util_avg_percent', 'gpu_util_max_percent']:
            row[f'energy_{metric}'] = str(energy.get(metric, ''))

        row['mode'] = json_data.get('mode', '')

    # 元数据
    row['experiment_source'] = json_data.get('experiment_source', old_row.get('experiment_source', ''))
    row['num_mutated_params'] = str(json_data.get('num_mutated_params', ''))
    row['mutated_param'] = json_data.get('mutated_param', '')

    return row


def convert_old_row_to_80col(old_row):
    """将37列格式的行转换为80列格式（使用老数据填充）"""
    row = {col: '' for col in HEADER_80COL}

    # 复制37列中已有的数据
    for col in old_row.keys():
        if col in HEADER_80COL:
            row[col] = old_row[col]

    # 设置mode（从experiment_id判断是否并行）
    if '_parallel' in old_row.get('experiment_id', ''):
        row['mode'] = 'parallel'
    else:
        row['mode'] = ''

    # num_mutated_params和mutated_param留空，后续通过enhance脚本填充
    row['num_mutated_params'] = ''
    row['mutated_param'] = ''

    return row


def main():
    print("=" * 80)
    print("将summary_old.csv转换为80列格式")
    print("=" * 80)

    # 备份
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'results/summary_old.csv.backup_{timestamp}'
    os.system(f'cp results/summary_old.csv {backup_file}')
    print(f"\n✓ 已备份: {backup_file}")

    # 读取37列CSV
    print("\n步骤1: 读取summary_old.csv...")
    rows_37col = []
    with open('results/summary_old.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_37col.append(row)
    print(f"✓ 读取 {len(rows_37col)} 行")

    # 转换为80列
    print("\n步骤2: 转换为80列格式...")
    rows_80col = []
    found_json_count = 0
    not_found_count = 0

    for i, old_row in enumerate(rows_37col, 1):
        exp_id = old_row['experiment_id']
        exp_source = old_row.get('experiment_source', '')

        if i % 50 == 0:
            print(f"  处理进度: {i}/{len(rows_37col)}")

        # 尝试从JSON读取完整数据
        json_file, json_data = find_experiment_json(exp_source, exp_id)

        if json_data:
            # 从JSON提取数据
            row_80col = extract_from_json(json_data, old_row)
            found_json_count += 1
        else:
            # JSON不存在，使用37列数据填充
            row_80col = convert_old_row_to_80col(old_row)
            not_found_count += 1

        rows_80col.append(row_80col)

    print(f"✓ 转换完成")
    print(f"  - 从JSON读取: {found_json_count} ({found_json_count/len(rows_37col)*100:.1f}%)")
    print(f"  - 使用CSV数据: {not_found_count} ({not_found_count/len(rows_37col)*100:.1f}%)")

    # 写入80列CSV
    print("\n步骤3: 写入summary_old_80col.csv...")
    output_file = 'results/summary_old_80col.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER_80COL)
        writer.writeheader()
        writer.writerows(rows_80col)

    print(f"✓ 已写入: {output_file}")
    print(f"  - 行数: {len(rows_80col)}")
    print(f"  - 列数: {len(HEADER_80COL)}")

    # 验证
    print("\n步骤4: 验证输出文件...")
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        data_rows = list(reader)

    print(f"✓ 验证通过")
    print(f"  - 表头列数: {len(header)} (预期80)")
    print(f"  - 数据行数: {len(data_rows)} (预期{len(rows_37col)})")

    if len(header) == 80 and len(data_rows) == len(rows_37col):
        print("\n" + "=" * 80)
        print("✓ 转换成功！")
        print("=" * 80)
        print(f"\n输出文件: {output_file}")
        print(f"备份文件: {backup_file}")
        print("\n下一步:")
        print("  1. 运行增强脚本填充变异分析列:")
        print("     python3 scripts/step5_enhance_mutation_analysis.py results/summary_old_80col.csv")
        print("  2. 验证80列格式数据:")
        print("     python3 tests/validate_80col_format.py")
        return 0
    else:
        print("\n✗ 验证失败！请检查输出文件。")
        return 1


if __name__ == '__main__':
    exit(main())
