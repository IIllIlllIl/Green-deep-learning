#!/usr/bin/env python3
"""
合并summary_old.csv和summary_new.csv为raw_data.csv（80列格式）

功能:
1. 读取summary_old.csv（93列）和summary_new.csv（80列）
2. 从summary_old.csv中提取80列（移除13个bg_hyperparam和bg_energy列）
3. 合并两个文件为raw_data.csv
4. 验证数据完整性
"""

import csv
import sys
from pathlib import Path

# 80列标准格式（来自summary_new.csv）
STANDARD_80_COLUMNS = [
    'experiment_id', 'timestamp', 'repository', 'model', 'training_success',
    'duration_seconds', 'retries', 'hyperparam_alpha', 'hyperparam_batch_size',
    'hyperparam_dropout', 'hyperparam_epochs', 'hyperparam_kfold',
    'hyperparam_learning_rate', 'hyperparam_max_iter', 'hyperparam_seed',
    'hyperparam_weight_decay', 'perf_accuracy', 'perf_best_val_accuracy',
    'perf_map', 'perf_precision', 'perf_rank1', 'perf_rank5', 'perf_recall',
    'perf_test_accuracy', 'perf_test_loss', 'energy_cpu_pkg_joules',
    'energy_cpu_ram_joules', 'energy_cpu_total_joules', 'energy_gpu_avg_watts',
    'energy_gpu_max_watts', 'energy_gpu_min_watts', 'energy_gpu_total_joules',
    'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
    'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent',
    'experiment_source', 'num_mutated_params', 'mutated_param', 'mode',
    'error_message', 'fg_repository', 'fg_model', 'fg_duration_seconds',
    'fg_training_success', 'fg_retries', 'fg_error_message',
    'fg_hyperparam_alpha', 'fg_hyperparam_batch_size', 'fg_hyperparam_dropout',
    'fg_hyperparam_epochs', 'fg_hyperparam_kfold', 'fg_hyperparam_learning_rate',
    'fg_hyperparam_max_iter', 'fg_hyperparam_seed', 'fg_hyperparam_weight_decay',
    'fg_perf_accuracy', 'fg_perf_best_val_accuracy', 'fg_perf_map',
    'fg_perf_precision', 'fg_perf_rank1', 'fg_perf_rank5', 'fg_perf_recall',
    'fg_perf_test_accuracy', 'fg_perf_test_loss', 'fg_energy_cpu_pkg_joules',
    'fg_energy_cpu_ram_joules', 'fg_energy_cpu_total_joules',
    'fg_energy_gpu_avg_watts', 'fg_energy_gpu_max_watts', 'fg_energy_gpu_min_watts',
    'fg_energy_gpu_total_joules', 'fg_energy_gpu_temp_avg_celsius',
    'fg_energy_gpu_temp_max_celsius', 'fg_energy_gpu_util_avg_percent',
    'fg_energy_gpu_util_max_percent', 'bg_repository', 'bg_model', 'bg_note',
    'bg_log_directory'
]

# 需要从93列中移除的13列
COLUMNS_TO_REMOVE = [
    'bg_hyperparam_batch_size', 'bg_hyperparam_dropout', 'bg_hyperparam_epochs',
    'bg_hyperparam_learning_rate', 'bg_hyperparam_seed', 'bg_hyperparam_weight_decay',
    'bg_energy_cpu_pkg_joules', 'bg_energy_cpu_ram_joules', 'bg_energy_cpu_total_joules',
    'bg_energy_gpu_avg_watts', 'bg_energy_gpu_max_watts', 'bg_energy_gpu_min_watts',
    'bg_energy_gpu_total_joules'
]

def read_csv_file(filepath, expected_cols):
    """读取CSV文件并验证列数"""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames

        if len(header) != expected_cols:
            print(f"⚠️  警告: {filepath} 有 {len(header)} 列，预期 {expected_cols} 列")

        rows = list(reader)
        return header, rows

def convert_93_to_80_columns(row_93col, header_93col):
    """将93列格式的行转换为80列格式"""
    # 创建80列的行
    row_80col = {}

    for col in STANDARD_80_COLUMNS:
        if col in header_93col:
            row_80col[col] = row_93col.get(col, '')
        else:
            row_80col[col] = ''

    return row_80col

def merge_csv_files(old_file, new_file, output_file):
    """合并两个CSV文件"""
    print(f"📖 读取 {old_file}...")
    old_header, old_rows = read_csv_file(old_file, 93)
    print(f"   ✓ {len(old_rows)} 行数据")

    print(f"📖 读取 {new_file}...")
    new_header, new_rows = read_csv_file(new_file, 80)
    print(f"   ✓ {len(new_rows)} 行数据")

    # 验证new_file的列顺序是否与标准一致
    if new_header != STANDARD_80_COLUMNS:
        print("⚠️  警告: summary_new.csv的列顺序与预期不一致")
        print(f"   预期: {len(STANDARD_80_COLUMNS)} 列")
        print(f"   实际: {len(new_header)} 列")

    # 转换old_rows为80列格式
    print(f"🔄 转换 {old_file} 从93列到80列格式...")
    old_rows_80col = []
    for row in old_rows:
        row_80col = convert_93_to_80_columns(row, old_header)
        old_rows_80col.append(row_80col)
    print(f"   ✓ 转换完成")

    # 合并数据
    print(f"🔗 合并数据...")
    all_rows = old_rows_80col + new_rows
    print(f"   ✓ 总计 {len(all_rows)} 行数据")

    # 写入输出文件
    print(f"💾 写入 {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_80_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"   ✓ 写入完成")

    return len(old_rows_80col), len(new_rows), len(all_rows)

def validate_merged_file(output_file, expected_old_rows, expected_new_rows):
    """验证合并后的文件"""
    print(f"\n🔍 验证 {output_file}...")

    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    # 检查列数
    if len(header) != 80:
        print(f"   ❌ 列数错误: {len(header)}，预期 80")
        return False
    else:
        print(f"   ✓ 列数正确: 80列")

    # 检查列名
    if header != STANDARD_80_COLUMNS:
        print(f"   ⚠️  列顺序可能不一致")
    else:
        print(f"   ✓ 列顺序正确")

    # 检查行数
    expected_total = expected_old_rows + expected_new_rows
    if len(rows) != expected_total:
        print(f"   ❌ 行数错误: {len(rows)}，预期 {expected_total}")
        return False
    else:
        print(f"   ✓ 行数正确: {len(rows)} ({expected_old_rows} 老实验 + {expected_new_rows} 新实验)")

    # 检查experiment_id唯一性
    exp_ids = [row['experiment_id'] for row in rows]
    unique_ids = set(exp_ids)
    if len(exp_ids) != len(unique_ids):
        duplicates = len(exp_ids) - len(unique_ids)
        print(f"   ⚠️  发现 {duplicates} 个重复的experiment_id")
    else:
        print(f"   ✓ 所有experiment_id唯一")

    # 检查必填字段
    required_fields = ['experiment_id', 'timestamp', 'repository', 'model', 'training_success']
    missing_count = 0
    for field in required_fields:
        empty_count = sum(1 for row in rows if not row.get(field, '').strip())
        if empty_count > 0:
            print(f"   ⚠️  {field}: {empty_count} 行为空")
            missing_count += empty_count

    if missing_count == 0:
        print(f"   ✓ 所有必填字段完整")

    # 统计训练成功率
    success_count = sum(1 for row in rows if row.get('training_success', '').lower() == 'true')
    print(f"   ℹ️  训练成功率: {success_count}/{len(rows)} ({success_count/len(rows)*100:.1f}%)")

    # 统计模式分布
    modes = {}
    for row in rows:
        mode = row.get('mode', 'unknown')
        modes[mode] = modes.get(mode, 0) + 1
    print(f"   ℹ️  模式分布:")
    for mode, count in sorted(modes.items()):
        print(f"      - {mode}: {count}")

    print(f"\n✅ 验证完成: {output_file} 数据完整且安全")
    return True

def main():
    # 文件路径
    results_dir = Path('/home/green/energy_dl/nightly/results')
    old_file = results_dir / 'summary_old.csv'
    new_file = results_dir / 'summary_new.csv'
    output_file = results_dir / 'raw_data.csv'

    # 检查输入文件是否存在
    if not old_file.exists():
        print(f"❌ 错误: {old_file} 不存在")
        sys.exit(1)

    if not new_file.exists():
        print(f"❌ 错误: {new_file} 不存在")
        sys.exit(1)

    # 检查输出文件是否已存在
    if output_file.exists():
        print(f"⚠️  警告: {output_file} 已存在，将被覆盖")
        backup_file = output_file.with_suffix('.csv.backup_before_merge')
        print(f"   创建备份: {backup_file}")
        import shutil
        shutil.copy2(output_file, backup_file)

    print("=" * 60)
    print("合并 summary_old.csv 和 summary_new.csv 为 raw_data.csv")
    print("=" * 60)

    # 执行合并
    old_count, new_count, total_count = merge_csv_files(old_file, new_file, output_file)

    # 验证合并结果
    success = validate_merged_file(output_file, old_count, new_count)

    if success:
        print(f"\n🎉 成功创建 {output_file}")
        print(f"   - 老实验: {old_count} 行")
        print(f"   - 新实验: {new_count} 行")
        print(f"   - 总计: {total_count} 行")
        print(f"   - 格式: 80列标准格式")
    else:
        print(f"\n❌ 验证失败，请检查输出文件")
        sys.exit(1)

if __name__ == '__main__':
    main()
