#!/usr/bin/env python3
"""
创建统一并行数据版本的 data.csv

目标：
1. 统一并行/非并行字段（fg_ vs 顶层）
2. 添加 is_parallel 列区分模式
3. 保留所有性能指标列，暂不进行标准化

版本: v1.0
日期: 2025-12-19
"""

import csv
import sys
from datetime import datetime

def get_field_value(row, field, is_parallel):
    """
    获取字段值，处理并行/非并行模式

    并行模式: 优先使用fg_字段，fallback到顶层字段
    非并行模式: 直接使用顶层字段
    """
    if is_parallel:
        # 优先使用 fg_ 前缀字段
        fg_value = row.get(f'fg_{field}', '').strip()
        if fg_value:
            return fg_value
    # fallback 到顶层字段
    return row.get(field, '').strip()

def transform_row(row):
    """
    转换单行数据，统一并行/非并行字段
    """
    is_parallel = (row['mode'] == 'parallel')

    # 构建新行
    new_row = {
        # === 基础信息 (9列) ===
        'experiment_id': row['experiment_id'],
        'timestamp': row['timestamp'],
        'repository': get_field_value(row, 'repository', is_parallel),
        'model': get_field_value(row, 'model', is_parallel),
        'is_parallel': str(is_parallel),
        'training_success': get_field_value(row, 'training_success', is_parallel),
        'duration_seconds': get_field_value(row, 'duration_seconds', is_parallel),
        'retries': get_field_value(row, 'retries', is_parallel),
        'error_message': get_field_value(row, 'error_message', is_parallel),

        # === 超参数 (9列) ===
        'hyperparam_alpha': get_field_value(row, 'hyperparam_alpha', is_parallel),
        'hyperparam_batch_size': get_field_value(row, 'hyperparam_batch_size', is_parallel),
        'hyperparam_dropout': get_field_value(row, 'hyperparam_dropout', is_parallel),
        'hyperparam_epochs': get_field_value(row, 'hyperparam_epochs', is_parallel),
        'hyperparam_kfold': get_field_value(row, 'hyperparam_kfold', is_parallel),
        'hyperparam_learning_rate': get_field_value(row, 'hyperparam_learning_rate', is_parallel),
        'hyperparam_max_iter': get_field_value(row, 'hyperparam_max_iter', is_parallel),
        'hyperparam_seed': get_field_value(row, 'hyperparam_seed', is_parallel),
        'hyperparam_weight_decay': get_field_value(row, 'hyperparam_weight_decay', is_parallel),

        # === 性能指标 (16列) - 保留所有列 ===
        'perf_accuracy': get_field_value(row, 'perf_accuracy', is_parallel),
        'perf_best_val_accuracy': get_field_value(row, 'perf_best_val_accuracy', is_parallel),
        'perf_map': get_field_value(row, 'perf_map', is_parallel),
        'perf_precision': get_field_value(row, 'perf_precision', is_parallel),
        'perf_rank1': get_field_value(row, 'perf_rank1', is_parallel),
        'perf_rank5': get_field_value(row, 'perf_rank5', is_parallel),
        'perf_recall': get_field_value(row, 'perf_recall', is_parallel),
        'perf_test_accuracy': get_field_value(row, 'perf_test_accuracy', is_parallel),
        'perf_test_loss': get_field_value(row, 'perf_test_loss', is_parallel),
        'perf_eval_loss': get_field_value(row, 'perf_eval_loss', is_parallel),
        'perf_final_training_loss': get_field_value(row, 'perf_final_training_loss', is_parallel),
        'perf_eval_samples_per_second': get_field_value(row, 'perf_eval_samples_per_second', is_parallel),
        'perf_top1_accuracy': get_field_value(row, 'perf_top1_accuracy', is_parallel),
        'perf_top5_accuracy': get_field_value(row, 'perf_top5_accuracy', is_parallel),
        'perf_top10_accuracy': get_field_value(row, 'perf_top10_accuracy', is_parallel),
        'perf_top20_accuracy': get_field_value(row, 'perf_top20_accuracy', is_parallel),

        # === 能耗指标 (11列) ===
        'energy_cpu_pkg_joules': get_field_value(row, 'energy_cpu_pkg_joules', is_parallel),
        'energy_cpu_ram_joules': get_field_value(row, 'energy_cpu_ram_joules', is_parallel),
        'energy_cpu_total_joules': get_field_value(row, 'energy_cpu_total_joules', is_parallel),
        'energy_gpu_avg_watts': get_field_value(row, 'energy_gpu_avg_watts', is_parallel),
        'energy_gpu_max_watts': get_field_value(row, 'energy_gpu_max_watts', is_parallel),
        'energy_gpu_min_watts': get_field_value(row, 'energy_gpu_min_watts', is_parallel),
        'energy_gpu_total_joules': get_field_value(row, 'energy_gpu_total_joules', is_parallel),
        'energy_gpu_temp_avg_celsius': get_field_value(row, 'energy_gpu_temp_avg_celsius', is_parallel),
        'energy_gpu_temp_max_celsius': get_field_value(row, 'energy_gpu_temp_max_celsius', is_parallel),
        'energy_gpu_util_avg_percent': get_field_value(row, 'energy_gpu_util_avg_percent', is_parallel),
        'energy_gpu_util_max_percent': get_field_value(row, 'energy_gpu_util_max_percent', is_parallel),

        # === 实验元数据 (4列) ===
        'experiment_source': row['experiment_source'],
        'num_mutated_params': row['num_mutated_params'],
        'mutated_param': row['mutated_param'],
        'mode': row['mode'],

        # === 并行模式额外信息 (7列，仅并行模式填充) ===
        'bg_repository': row['bg_repository'] if is_parallel else '',
        'bg_model': row['bg_model'] if is_parallel else '',
        'bg_note': row['bg_note'] if is_parallel else '',
        'bg_log_directory': row['bg_log_directory'] if is_parallel else '',
        'fg_duration_seconds': row.get('fg_duration_seconds', '') if is_parallel else '',
        'fg_retries': row.get('fg_retries', '') if is_parallel else '',
        'fg_error_message': row.get('fg_error_message', '') if is_parallel else '',
    }

    return new_row

def main():
    """主函数"""
    input_file = 'data/raw_data.csv'
    output_file = 'data/data.csv'

    print("=" * 80)
    print("创建统一并行数据版本的 data.csv")
    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print()

    # 定义输出列顺序 (56列)
    output_fieldnames = [
        # 基础信息 (9)
        'experiment_id', 'timestamp', 'repository', 'model', 'is_parallel',
        'training_success', 'duration_seconds', 'retries', 'error_message',

        # 超参数 (9)
        'hyperparam_alpha', 'hyperparam_batch_size', 'hyperparam_dropout',
        'hyperparam_epochs', 'hyperparam_kfold', 'hyperparam_learning_rate',
        'hyperparam_max_iter', 'hyperparam_seed', 'hyperparam_weight_decay',

        # 性能指标 (16)
        'perf_accuracy', 'perf_best_val_accuracy', 'perf_map', 'perf_precision',
        'perf_rank1', 'perf_rank5', 'perf_recall', 'perf_test_accuracy',
        'perf_test_loss', 'perf_eval_loss', 'perf_final_training_loss',
        'perf_eval_samples_per_second', 'perf_top1_accuracy', 'perf_top5_accuracy',
        'perf_top10_accuracy', 'perf_top20_accuracy',

        # 能耗指标 (11)
        'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
        'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
        'energy_gpu_total_joules', 'energy_gpu_temp_avg_celsius',
        'energy_gpu_temp_max_celsius', 'energy_gpu_util_avg_percent',
        'energy_gpu_util_max_percent',

        # 实验元数据 (4)
        'experiment_source', 'num_mutated_params', 'mutated_param', 'mode',

        # 并行模式额外信息 (7)
        'bg_repository', 'bg_model', 'bg_note', 'bg_log_directory',
        'fg_duration_seconds', 'fg_retries', 'fg_error_message'
    ]

    print(f"目标列数: {len(output_fieldnames)}")
    print()

    # 统计信息
    stats = {
        'total': 0,
        'parallel': 0,
        'nonparallel': 0,
        'parallel_old_format': 0,  # 仅顶层字段
        'parallel_new_format': 0,  # 仅fg_字段
        'parallel_mixed': 0,       # 两者都有
    }

    try:
        with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=output_fieldnames)
            writer.writeheader()

            for row in reader:
                new_row = transform_row(row)
                writer.writerow(new_row)

                stats['total'] += 1

                if row['mode'] == 'parallel':
                    stats['parallel'] += 1

                    # 检查数据格式
                    has_toplevel = bool(row.get('training_success', '').strip())
                    has_fg = bool(row.get('fg_training_success', '').strip())

                    if has_toplevel and has_fg:
                        stats['parallel_mixed'] += 1
                    elif has_toplevel:
                        stats['parallel_old_format'] += 1
                    elif has_fg:
                        stats['parallel_new_format'] += 1
                else:
                    stats['nonparallel'] += 1

        print("✓ 转换完成")
        print()
        print("=" * 80)
        print("统计报告")
        print("=" * 80)
        print(f"总行数: {stats['total']}")
        print(f"列数: 87 → {len(output_fieldnames)}")
        print()
        print("模式分布:")
        print(f"  非并行: {stats['nonparallel']} ({stats['nonparallel']/stats['total']*100:.1f}%)")
        print(f"  并行: {stats['parallel']} ({stats['parallel']/stats['total']*100:.1f}%)")
        print()
        print("并行数据格式分布:")
        print(f"  仅顶层字段 (老格式): {stats['parallel_old_format']} ({stats['parallel_old_format']/stats['parallel']*100:.1f}%)")
        print(f"  仅fg_字段 (新格式): {stats['parallel_new_format']} ({stats['parallel_new_format']/stats['parallel']*100:.1f}%)")
        print(f"  两者都有 (混合): {stats['parallel_mixed']} ({stats['parallel_mixed']/stats['parallel']*100:.1f}%)")
        print()
        print("=" * 80)
        print(f"✓ 输出文件: {output_file}")
        print("=" * 80)

        return 0

    except FileNotFoundError:
        print(f"✗ 错误: 找不到输入文件 {input_file}")
        return 1
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
