#!/usr/bin/env python3
"""
步骤3: 从experiment.json重建新实验CSV

遍历所有run_*/目录，读取experiment.json，构建完整的新实验CSV
"""

import json
import os
import glob
import csv
from datetime import datetime

def load_header_design():
    """加载表头设计"""
    with open('results/new_csv_header_design.json', 'r', encoding='utf-8') as f:
        return json.load(f)['full_header']

def extract_value(data, path):
    """从嵌套字典中提取值

    Args:
        data: JSON数据
        path: 路径，如 'energy_metrics.cpu_energy_total_joules'

    Returns:
        值或空字符串
    """
    keys = path.split('.')
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return ''

    return current if current is not None else ''

def process_experiment_json(json_path, header):
    """处理单个experiment.json文件

    Args:
        json_path: experiment.json文件路径
        header: CSV表头列表

    Returns:
        dict: CSV行数据
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    row = {col: '' for col in header}  # 初始化所有列为空

    # 基础信息
    row['experiment_id'] = data.get('experiment_id', '')
    row['timestamp'] = data.get('timestamp', '')
    row['duration_seconds'] = data.get('duration_seconds', '')
    row['training_success'] = data.get('training_success', '')
    row['retries'] = data.get('retries', 0)
    row['mode'] = data.get('mode', '')  # parallel 或空
    row['error_message'] = data.get('error_message', '')

    # 判断模式
    is_parallel = data.get('mode') == 'parallel'

    if is_parallel:
        # 并行模式：数据在foreground中
        fg = data.get('foreground', {})
        bg = data.get('background', {})

        # 基础字段（来自foreground）
        row['repository'] = fg.get('repository', '')
        row['model'] = fg.get('model', '')
        row['training_success'] = fg.get('training_success', '')

        # Hyperparameters
        fg_hyper = fg.get('hyperparameters', {})
        row['hyperparam_alpha'] = fg_hyper.get('alpha', '')
        row['hyperparam_batch_size'] = fg_hyper.get('batch_size', '')
        row['hyperparam_dropout'] = fg_hyper.get('dropout', '')
        row['hyperparam_epochs'] = fg_hyper.get('epochs', '')
        row['hyperparam_kfold'] = fg_hyper.get('kfold', '')
        row['hyperparam_learning_rate'] = fg_hyper.get('learning_rate', '')
        row['hyperparam_max_iter'] = fg_hyper.get('max_iter', '')
        row['hyperparam_seed'] = fg_hyper.get('seed', '')
        row['hyperparam_weight_decay'] = fg_hyper.get('weight_decay', '')

        # Performance metrics
        fg_perf = fg.get('performance_metrics', {})
        row['perf_accuracy'] = fg_perf.get('accuracy', '')
        row['perf_best_val_accuracy'] = fg_perf.get('best_val_accuracy', '')
        row['perf_map'] = fg_perf.get('map', '')
        row['perf_precision'] = fg_perf.get('precision', '')
        row['perf_rank1'] = fg_perf.get('rank1', '')
        row['perf_rank5'] = fg_perf.get('rank5', '')
        row['perf_recall'] = fg_perf.get('recall', '')
        row['perf_test_accuracy'] = fg_perf.get('test_accuracy', '')
        row['perf_test_loss'] = fg_perf.get('test_loss', '')

        # Energy metrics
        fg_energy = fg.get('energy_metrics', {})
        row['energy_cpu_pkg_joules'] = fg_energy.get('cpu_energy_pkg_joules', '')
        row['energy_cpu_ram_joules'] = fg_energy.get('cpu_energy_ram_joules', '')
        row['energy_cpu_total_joules'] = fg_energy.get('cpu_energy_total_joules', '')
        row['energy_gpu_avg_watts'] = fg_energy.get('gpu_power_avg_watts', '')
        row['energy_gpu_max_watts'] = fg_energy.get('gpu_power_max_watts', '')
        row['energy_gpu_min_watts'] = fg_energy.get('gpu_power_min_watts', '')
        row['energy_gpu_total_joules'] = fg_energy.get('gpu_energy_total_joules', '')
        row['energy_gpu_temp_avg_celsius'] = fg_energy.get('gpu_temp_avg_celsius', '')
        row['energy_gpu_temp_max_celsius'] = fg_energy.get('gpu_temp_max_celsius', '')
        row['energy_gpu_util_avg_percent'] = fg_energy.get('gpu_util_avg_percent', '')
        row['energy_gpu_util_max_percent'] = fg_energy.get('gpu_util_max_percent', '')

        # Foreground详细数据（fg_*字段）
        row['fg_repository'] = fg.get('repository', '')
        row['fg_model'] = fg.get('model', '')
        row['fg_duration_seconds'] = fg.get('duration_seconds', '')
        row['fg_training_success'] = fg.get('training_success', '')
        row['fg_retries'] = fg.get('retries', '')
        row['fg_error_message'] = fg.get('error_message', '')

        # Foreground hyperparameters (fg_hyperparam_*)
        for param in ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                      'learning_rate', 'max_iter', 'seed', 'weight_decay']:
            row[f'fg_hyperparam_{param}'] = fg_hyper.get(param, '')

        # Foreground performance (fg_perf_*)
        for metric in ['accuracy', 'best_val_accuracy', 'map', 'precision',
                       'rank1', 'rank5', 'recall', 'test_accuracy', 'test_loss']:
            row[f'fg_perf_{metric}'] = fg_perf.get(metric, '')

        # Foreground energy (fg_energy_*)
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

        # Background info
        row['bg_repository'] = bg.get('repository', '')
        row['bg_model'] = bg.get('model', '')
        row['bg_note'] = bg.get('note', '')
        row['bg_log_directory'] = bg.get('log_directory', '')

    else:
        # 非并行模式：数据在顶层
        row['repository'] = data.get('repository', '')
        row['model'] = data.get('model', '')

        # Hyperparameters
        hyper = data.get('hyperparameters', {})
        row['hyperparam_alpha'] = hyper.get('alpha', '')
        row['hyperparam_batch_size'] = hyper.get('batch_size', '')
        row['hyperparam_dropout'] = hyper.get('dropout', '')
        row['hyperparam_epochs'] = hyper.get('epochs', '')
        row['hyperparam_kfold'] = hyper.get('kfold', '')
        row['hyperparam_learning_rate'] = hyper.get('learning_rate', '')
        row['hyperparam_max_iter'] = hyper.get('max_iter', '')
        row['hyperparam_seed'] = hyper.get('seed', '')
        row['hyperparam_weight_decay'] = hyper.get('weight_decay', '')

        # Performance metrics
        perf = data.get('performance_metrics', {})
        row['perf_accuracy'] = perf.get('accuracy', '')
        row['perf_best_val_accuracy'] = perf.get('best_val_accuracy', '')
        row['perf_map'] = perf.get('map', '')
        row['perf_precision'] = perf.get('precision', '')
        row['perf_rank1'] = perf.get('rank1', '')
        row['perf_rank5'] = perf.get('rank5', '')
        row['perf_recall'] = perf.get('recall', '')
        row['perf_test_accuracy'] = perf.get('test_accuracy', '')
        row['perf_test_loss'] = perf.get('test_loss', '')

        # Energy metrics
        energy = data.get('energy_metrics', {})
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

    # experiment_source 暂时为空（后续推断）
    row['experiment_source'] = ''

    return row

def rebuild_new_csv():
    """重建新实验CSV"""

    header = load_header_design()
    all_rows = []
    processed_count = 0
    skipped_count = 0

    print("从experiment.json重建新实验CSV")
    print("=" * 70)

    # 遍历所有run目录
    run_dirs = sorted(glob.glob("results/run_*"))
    print(f"找到 {len(run_dirs)} 个run目录\n")

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        exp_json_files = sorted(glob.glob(f"{run_dir}/*/experiment.json"))

        if not exp_json_files:
            continue

        print(f"处理 {run_name}: {len(exp_json_files)} 个实验...", end=' ')

        run_processed = 0
        for json_path in exp_json_files:
            try:
                row = process_experiment_json(json_path, header)
                all_rows.append(row)
                run_processed += 1
                processed_count += 1
            except Exception as e:
                print(f"\n  ⚠️  失败: {json_path}")
                print(f"      错误: {e}")
                skipped_count += 1

        print(f"完成 {run_processed}/{len(exp_json_files)}")

    print()
    print("=" * 70)
    print(f"扫描完成！")
    print(f"  成功处理: {processed_count} 个实验")
    print(f"  失败跳过: {skipped_count} 个实验")
    print()

    # 过滤掉VulBERTa/cnn实验（训练代码未实现，数据无效）
    filtered_rows = []
    vulberta_cnn_count = 0

    for row in all_rows:
        if row.get('repository') == 'VulBERTa' and row.get('model') == 'cnn':
            vulberta_cnn_count += 1
            continue  # 跳过VulBERTa/cnn实验
        filtered_rows.append(row)

    print(f"过滤掉 {vulberta_cnn_count} 个VulBERTa/cnn实验（训练代码未实现）")
    print(f"有效实验数: {len(filtered_rows)}")
    print()

    # 写入CSV
    output_file = 'results/summary_new.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"✓ 新实验CSV已生成: {output_file}")
    print(f"  总行数: {len(filtered_rows)}")
    print(f"  总列数: {len(header)}")
    print()

    # 统计信息
    mode_dist = {}
    repo_dist = {}
    for row in filtered_rows:
        mode = row.get('mode', '') or '(non-parallel)'
        mode_dist[mode] = mode_dist.get(mode, 0) + 1

        repo = row.get('repository', '')
        repo_dist[repo] = repo_dist.get(repo, 0) + 1

    print("模式分布:")
    for mode, count in sorted(mode_dist.items()):
        print(f"  {mode:20s}: {count:3d} 行")

    print()
    print("Repository分布:")
    for repo, count in sorted(repo_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {repo:40s}: {count:3d} 行")

    return filtered_rows, header

if __name__ == '__main__':
    rows, header = rebuild_new_csv()
    print()
    print("下一步: 验证CSV数据完整性")
