#!/usr/bin/env python3
"""
从experiment.json重建summary_all.csv为93列格式

问题: summary_all.csv仅有37列,缺少56个字段
解决: 从476个experiment.json文件直接提取完整数据生成93列CSV

作者: Claude Code
日期: 2025-12-12
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

# 93列标准格式
FIELDNAMES_93COL = [
    'experiment_id', 'timestamp', 'repository', 'model', 'training_success',
    'duration_seconds', 'retries',
    # 顶层超参数 (9)
    'hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
    'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
    'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay',
    # 顶层性能指标 (9)
    'perf_accuracy', 'perf_best_val_accuracy', 'perf_map', 'perf_precision',
    'perf_rank1', 'perf_rank5', 'perf_recall', 'perf_test_accuracy', 'perf_test_loss',
    # 顶层能耗 (11)
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
    'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius',
    'energy_gpu_temp_max_celsius', 'energy_gpu_util_avg_percent',
    'energy_gpu_util_max_percent',
    # 实验元数据 (5)
    'experiment_source', 'num_mutated_params', 'mutated_param', 'mode', 'error_message',
    # 前景字段 (42)
    'fg_repository', 'fg_model', 'fg_duration_seconds', 'fg_training_success', 'fg_retries',
    'fg_error_message',
    'fg_hyperparam_alpha', 'fg_hyperparam_batch_size', 'fg_hyperparam_dropout',
    'fg_hyperparam_epochs', 'fg_hyperparam_kfold', 'fg_hyperparam_learning_rate',
    'fg_hyperparam_max_iter', 'fg_hyperparam_seed', 'fg_hyperparam_weight_decay',
    'fg_perf_accuracy', 'fg_perf_best_val_accuracy', 'fg_perf_map', 'fg_perf_precision',
    'fg_perf_rank1', 'fg_perf_rank5', 'fg_perf_recall', 'fg_perf_test_accuracy',
    'fg_perf_test_loss',
    'fg_energy_cpu_pkg_joules', 'fg_energy_cpu_ram_joules', 'fg_energy_cpu_total_joules',
    'fg_energy_gpu_avg_watts', 'fg_energy_gpu_max_watts', 'fg_energy_gpu_min_watts',
    'fg_energy_gpu_total_joules', 'fg_energy_gpu_temp_avg_celsius',
    'fg_energy_gpu_temp_max_celsius', 'fg_energy_gpu_util_avg_percent',
    'fg_energy_gpu_util_max_percent',
    # 背景字段 (10)
    'bg_repository', 'bg_model', 'bg_note', 'bg_log_directory',
    'bg_hyperparam_batch_size', 'bg_hyperparam_dropout', 'bg_hyperparam_epochs',
    'bg_hyperparam_learning_rate', 'bg_hyperparam_seed', 'bg_hyperparam_weight_decay',
    'bg_energy_cpu_pkg_joules', 'bg_energy_cpu_ram_joules', 'bg_energy_cpu_total_joules',
    'bg_energy_gpu_avg_watts', 'bg_energy_gpu_max_watts', 'bg_energy_gpu_min_watts',
    'bg_energy_gpu_total_joules'
]


def find_experiment_json(experiment_id):
    """查找experiment.json文件"""
    results_dir = Path('results')

    # 从experiment_id提取目录名
    # 格式: default__MRT-OAST_default_001 或 MRT-OAST_default_001
    if '__' in experiment_id:
        source_prefix, dir_name = experiment_id.split('__', 1)
    else:
        dir_name = experiment_id

    # 1. 尝试新实验目录 (run_YYYYMMDD_HHMMSS/)
    for run_dir in sorted(results_dir.glob('run_*'), reverse=True):
        json_path = run_dir / dir_name / 'experiment.json'
        if json_path.exists():
            return json_path

    # 2. 尝试老实验目录
    old_dirs = ['mutation_2x_20251122_175401', 'default', 'mutation_1x', 'archived']
    for old_dir in old_dirs:
        json_path = results_dir / old_dir / dir_name / 'experiment.json'
        if json_path.exists():
            return json_path

    return None


def extract_experiment_source_info(experiment_source):
    """从experiment_source提取mutated_param和num_mutated_params"""
    if not experiment_source or experiment_source in ['default', 'baseline']:
        return '', 0

    # 格式: mutation_1x, mutation_2x_safe等
    if 'mutation' in experiment_source.lower():
        # 从experiment_source推断,通常是单参数变异
        return '', 1
    return '', 0


def json_to_csv_row(json_data, experiment_id):
    """将experiment.json转换为CSV行"""
    row = defaultdict(str)

    # 基础字段
    row['experiment_id'] = experiment_id
    row['timestamp'] = json_data.get('timestamp', '')

    # 判断是否为并行模式
    mode = json_data.get('mode', '')
    row['mode'] = mode

    if mode == 'parallel':
        # 并行模式: 数据在foreground和background中
        fg = json_data.get('foreground', {})
        bg = json_data.get('background', {})

        # 顶层字段为空
        row['repository'] = ''
        row['model'] = ''
        row['training_success'] = ''
        row['duration_seconds'] = ''
        row['retries'] = ''

        # 前景字段
        row['fg_repository'] = fg.get('repository', '')
        row['fg_model'] = fg.get('model', '')
        row['fg_duration_seconds'] = fg.get('duration_seconds', '')
        row['fg_training_success'] = fg.get('training_success', '')
        row['fg_retries'] = fg.get('retries', '')
        row['fg_error_message'] = fg.get('error_message', '')

        # 前景超参数
        fg_hyperparams = fg.get('hyperparameters', {})
        for param in ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                      'learning_rate', 'max_iter', 'seed', 'weight_decay']:
            key = f'fg_hyperparam_{param}'
            row[key] = fg_hyperparams.get(param, '')

        # 前景性能指标
        fg_perf = fg.get('performance_metrics', {})
        for metric in ['accuracy', 'best_val_accuracy', 'map', 'precision',
                       'rank1', 'rank5', 'recall', 'test_accuracy', 'test_loss']:
            key = f'fg_perf_{metric}'
            row[key] = fg_perf.get(metric, '')

        # 前景能耗
        fg_energy = fg.get('energy_metrics', {})
        row['fg_energy_cpu_pkg_joules'] = fg_energy.get('cpu_energy_pkg_joules', '')
        row['fg_energy_cpu_ram_joules'] = fg_energy.get('cpu_energy_ram_joules', '')
        row['fg_energy_cpu_total_joules'] = fg_energy.get('cpu_energy_total_joules', '')
        row['fg_energy_gpu_avg_watts'] = fg_energy.get('gpu_power_avg_watts', '')
        row['fg_energy_gpu_max_watts'] = fg_energy.get('gpu_power_max_watts', '')
        row['fg_energy_gpu_min_watts'] = fg_energy.get('gpu_power_min_watts', '')
        row['fg_energy_gpu_total_joules'] = fg_energy.get('gpu_energy_total_joules', '')
        row['fg_energy_gpu_temp_avg_celsius'] = fg_energy.get('gpu_temp_avg_celsius', '')
        row['fg_energy_gpu_temp_max_celsius'] = fg_energy.get('gpu_temp_max_celsius', '')
        row['fg_energy_gpu_util_avg_percent'] = fg_energy.get('gpu_util_avg_percent', '')
        row['fg_energy_gpu_util_max_percent'] = fg_energy.get('gpu_util_max_percent', '')

        # 背景字段
        row['bg_repository'] = bg.get('repository', '')
        row['bg_model'] = bg.get('model', '')
        row['bg_note'] = bg.get('note', '')
        row['bg_log_directory'] = bg.get('log_directory', '')

        # 背景超参数
        bg_hyperparams = bg.get('hyperparameters', {})
        for param in ['batch_size', 'dropout', 'epochs', 'learning_rate', 'seed', 'weight_decay']:
            key = f'bg_hyperparam_{param}'
            row[key] = bg_hyperparams.get(param, '')

        # 背景能耗: 全部为空(设计决定)
        # bg_energy_* 字段保持为空

    else:
        # 非并行模式: 数据在顶层
        row['repository'] = json_data.get('repository', '')
        row['model'] = json_data.get('model', '')
        row['training_success'] = json_data.get('training_success', '')
        row['duration_seconds'] = json_data.get('duration_seconds', '')
        row['retries'] = json_data.get('retries', '')
        row['error_message'] = json_data.get('error_message', '')

        # 顶层超参数
        hyperparams = json_data.get('hyperparameters', {})
        for param in ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                      'learning_rate', 'max_iter', 'seed', 'weight_decay']:
            key = f'hyperparam_{param}'
            row[key] = hyperparams.get(param, '')

        # 顶层性能指标
        perf = json_data.get('performance_metrics', {})
        for metric in ['accuracy', 'best_val_accuracy', 'map', 'precision',
                       'rank1', 'rank5', 'recall', 'test_accuracy', 'test_loss']:
            key = f'perf_{metric}'
            row[key] = perf.get(metric, '')

        # 顶层能耗
        energy = json_data.get('energy_metrics', {})
        row['energy_cpu_pkg_joules'] = energy.get('cpu_energy_pkg_joules', '')
        row['energy_cpu_ram_joules'] = energy.get('cpu_energy_ram_joules', '')
        row['energy_cpu_total_joules'] = energy.get('cpu_energy_total_joules', '')
        row['energy_gpu_avg_watts'] = energy.get('gpu_power_avg_watts', '')
        row['energy_gpu_max_watts'] = energy.get('gpu_power_max_watts', '')
        row['energy_gpu_min_watts'] = energy.get('gpu_power_min_watts', '')
        row['energy_gpu_total_joules'] = energy.get('gpu_energy_total_joules', '')
        row['energy_gpu_temp_avg_celsius'] = energy.get('gpu_temp_avg_celsius', '')
        row['energy_gpu_temp_max_celsius'] = energy.get('gpu_temp_max_celsius', '')
        row['energy_gpu_util_avg_percent'] = energy.get('gpu_util_avg_percent', '')
        row['energy_gpu_util_max_percent'] = energy.get('gpu_util_max_percent', '')

        # 前景和背景字段为空

    # 实验来源
    row['experiment_source'] = json_data.get('experiment_source', '')

    # 从experiment_source推断mutated_param和num_mutated_params
    mutated_param, num_mutated = extract_experiment_source_info(row['experiment_source'])
    row['mutated_param'] = mutated_param
    row['num_mutated_params'] = num_mutated

    return row


def rebuild_summary_all():
    """重建summary_all.csv为93列格式"""
    print("=" * 80)
    print("从experiment.json重建summary_all.csv (93列格式)")
    print("=" * 80)

    # 1. 读取原37列CSV,获取experiment_id列表
    print("\n步骤1: 读取原summary_all.csv...")
    csv_file = Path('results/summary_all.csv')

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        experiment_ids = [row['experiment_id'] for row in reader]

    print(f"  ✓ 找到 {len(experiment_ids)} 个实验ID")

    # 2. 从experiment.json提取数据
    print("\n步骤2: 从experiment.json提取数据...")
    csv_rows = []
    found_count = 0
    missing_count = 0

    for exp_id in experiment_ids:
        json_path = find_experiment_json(exp_id)

        if not json_path:
            print(f"  ✗ 未找到: {exp_id}")
            missing_count += 1
            continue

        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            row = json_to_csv_row(json_data, exp_id)
            csv_rows.append(row)
            found_count += 1

            if found_count % 50 == 0:
                print(f"  处理进度: {found_count}/{len(experiment_ids)} ({found_count/len(experiment_ids)*100:.1f}%)")

        except Exception as e:
            print(f"  ✗ 解析失败: {exp_id} - {e}")
            missing_count += 1

    print(f"\n  ✓ 成功提取: {found_count} 个实验")
    if missing_count > 0:
        print(f"  ✗ 失败/缺失: {missing_count} 个实验")

    # 3. 写入新CSV
    print("\n步骤3: 写入93列CSV...")
    output_file = Path('results/summary_all_93col.csv')

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES_93COL, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"  ✓ 写入完成: {output_file}")
    print(f"  ✓ 行数: {len(csv_rows)}")
    print(f"  ✓ 列数: {len(FIELDNAMES_93COL)}")

    # 4. 数据质量统计
    print("\n步骤4: 数据质量统计...")
    stats = {
        'training_success': 0,
        'cpu_energy': 0,
        'gpu_energy': 0,
        'parallel_mode': 0,
        'non_parallel_mode': 0,
    }

    for row in csv_rows:
        if row.get('training_success') or row.get('fg_training_success'):
            stats['training_success'] += 1
        if row.get('energy_cpu_total_joules') or row.get('fg_energy_cpu_total_joules'):
            stats['cpu_energy'] += 1
        if row.get('energy_gpu_total_joules') or row.get('fg_energy_gpu_total_joules'):
            stats['gpu_energy'] += 1
        if row.get('mode') == 'parallel':
            stats['parallel_mode'] += 1
        else:
            stats['non_parallel_mode'] += 1

    total = len(csv_rows)
    print(f"  训练成功: {stats['training_success']}/{total} ({stats['training_success']/total*100:.1f}%)")
    print(f"  CPU能耗完整: {stats['cpu_energy']}/{total} ({stats['cpu_energy']/total*100:.1f}%)")
    print(f"  GPU能耗完整: {stats['gpu_energy']}/{total} ({stats['gpu_energy']/total*100:.1f}%)")
    print(f"  并行模式: {stats['parallel_mode']}/{total} ({stats['parallel_mode']/total*100:.1f}%)")
    print(f"  非并行模式: {stats['non_parallel_mode']}/{total} ({stats['non_parallel_mode']/total*100:.1f}%)")

    # 5. 完成
    print("\n" + "=" * 80)
    print("✅ 重建完成!")
    print("=" * 80)
    print(f"\n输出文件: {output_file}")
    print(f"下一步: 运行 python3 scripts/validate_summary_all_93col.py 验证数据")

    return output_file


if __name__ == '__main__':
    rebuild_summary_all()
