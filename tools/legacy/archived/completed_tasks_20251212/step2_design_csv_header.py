#!/usr/bin/env python3
"""
步骤2: 设计新CSV表头结构

基于扫描的experiment.json字段，设计新的CSV表头：
1. 保持与老CSV兼容的基础字段
2. 添加mode字段区分并行/非并行
3. 区分foreground/background数据（并行模式）
"""

def design_csv_header():
    """设计新CSV表头"""

    # 老CSV的37列（保持兼容）
    old_header = [
        'experiment_id',
        'timestamp',
        'repository',
        'model',
        'training_success',
        'duration_seconds',
        'retries',
        'hyperparam_alpha',
        'hyperparam_batch_size',
        'hyperparam_dropout',
        'hyperparam_epochs',
        'hyperparam_kfold',
        'hyperparam_learning_rate',
        'hyperparam_max_iter',
        'hyperparam_seed',
        'hyperparam_weight_decay',
        'perf_accuracy',
        'perf_best_val_accuracy',
        'perf_map',
        'perf_precision',
        'perf_rank1',
        'perf_rank5',
        'perf_recall',
        'perf_test_accuracy',
        'perf_test_loss',
        'energy_cpu_pkg_joules',
        'energy_cpu_ram_joules',
        'energy_cpu_total_joules',
        'energy_gpu_avg_watts',
        'energy_gpu_max_watts',
        'energy_gpu_min_watts',
        'energy_gpu_total_joules',
        'energy_gpu_temp_avg_celsius',
        'energy_gpu_temp_max_celsius',
        'energy_gpu_util_avg_percent',
        'energy_gpu_util_max_percent',
        'experiment_source'
    ]

    # 新增字段
    new_fields = [
        'mode',  # parallel 或空（非并行）
        'error_message',  # 错误信息

        # 并行模式专用字段 - Foreground
        'fg_repository',
        'fg_model',
        'fg_duration_seconds',
        'fg_training_success',
        'fg_retries',
        'fg_error_message',

        # Foreground hyperparameters
        'fg_hyperparam_alpha',
        'fg_hyperparam_batch_size',
        'fg_hyperparam_dropout',
        'fg_hyperparam_epochs',
        'fg_hyperparam_kfold',
        'fg_hyperparam_learning_rate',
        'fg_hyperparam_max_iter',
        'fg_hyperparam_seed',
        'fg_hyperparam_weight_decay',

        # Foreground performance
        'fg_perf_accuracy',
        'fg_perf_best_val_accuracy',
        'fg_perf_map',
        'fg_perf_precision',
        'fg_perf_rank1',
        'fg_perf_rank5',
        'fg_perf_recall',
        'fg_perf_test_accuracy',
        'fg_perf_test_loss',

        # Foreground energy
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

        # Background info (简化，只记录基本信息)
        'bg_repository',
        'bg_model',
        'bg_note',
        'bg_log_directory'
    ]

    # 完整表头
    full_header = old_header + new_fields

    print("新CSV表头设计")
    print("=" * 70)
    print(f"老表头列数: {len(old_header)}")
    print(f"新增列数: {len(new_fields)}")
    print(f"总列数: {len(full_header)}")
    print()

    print("表头结构：")
    print("-" * 70)
    print(f"基础字段 (37列): 与老CSV完全兼容")
    print(f"  - experiment_id ~ experiment_source")
    print()
    print(f"新增字段 ({len(new_fields)}列):")
    print(f"  - mode: 并行模式标记")
    print(f"  - error_message: 错误信息")
    print(f"  - fg_*: 并行模式foreground数据 (38列)")
    print(f"  - bg_*: 并行模式background基本信息 (4列)")
    print()

    # 保存表头到文件
    import json
    output_file = "results/new_csv_header_design.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'old_header': old_header,
            'new_fields': new_fields,
            'full_header': full_header,
            'total_columns': len(full_header),
            'old_columns': len(old_header),
            'new_columns': len(new_fields)
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ 表头设计已保存到: {output_file}")
    print()

    # 打印完整表头列表
    print("完整表头列表：")
    print("-" * 70)
    for i, col in enumerate(full_header, 1):
        marker = "[NEW]" if col in new_fields else ""
        print(f"{i:3d}. {col:40s} {marker}")

    return full_header

if __name__ == '__main__':
    header = design_csv_header()
    print()
    print(f"✓ 新CSV将包含 {len(header)} 列")
    print("下一步: 从experiment.json填充数据")
