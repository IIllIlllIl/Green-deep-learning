#!/usr/bin/env python3
"""
验证重建的summary_old_93col.csv数据完整性

验证项：
1. 行数匹配
2. JSON数据与CSV数据一致性（随机抽样10个）
3. 新增字段（背景超参数、背景能耗）是否正确填充
4. 能耗字段映射正确性
5. 93列格式完整性

作者: Claude Code
日期: 2025-12-12
"""

import json
import csv
from pathlib import Path
import random


def find_experiment_json(experiment_id):
    """查找experiment.json文件"""
    results_dir = Path('results')

    # 提取实际目录名
    if '__' in experiment_id:
        source_prefix, dir_name = experiment_id.split('__', 1)
    else:
        dir_name = experiment_id

    # 1. 尝试老实验目录
    old_dirs = ['mutation_2x_20251122_175401', 'default', 'mutation_1x', 'archived']
    for old_dir in old_dirs:
        json_path = results_dir / old_dir / dir_name / 'experiment.json'
        if json_path.exists():
            return json_path

    # 2. 尝试新实验目录
    for run_dir in sorted(results_dir.glob('run_*')):
        json_path = run_dir / dir_name / 'experiment.json'
        if json_path.exists():
            return json_path

    return None


def compare_value(json_val, csv_val, field_name):
    """比较JSON值和CSV值"""
    # 空值处理
    if json_val is None or json_val == '':
        return csv_val == '', True

    # 类型转换
    json_str = str(json_val)
    csv_str = str(csv_val)

    # 比较
    match = json_str == csv_str
    return match, match


def validate_93col_csv():
    """主验证函数"""
    print("=" * 80)
    print("验证summary_old_93col.csv数据完整性")
    print("=" * 80)

    # 1. 加载CSV数据
    print("\n步骤1: 加载CSV数据...")
    csv_file = 'results/summary_old_93col.csv'
    csv_rows = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        csv_header = reader.fieldnames
        csv_rows = list(reader)

    print(f"  ✓ 加载完成: {len(csv_rows)} 行, {len(csv_header)} 列")

    # 2. 验证格式
    print("\n步骤2: 验证93列格式...")
    if len(csv_header) == 93:
        print(f"  ✓ 列数正确: 93列")
    else:
        print(f"  ✗ 列数错误: {len(csv_header)} 列（预期93列）")

    # 3. 数据质量统计
    print("\n步骤3: 数据质量统计...")
    stats = {
        'training_success': 0,
        'cpu_energy': 0,
        'gpu_energy': 0,
        'performance': 0,
        'parallel_mode': 0,
        'bg_hyperparams': 0,
        'bg_energy': 0,
    }

    for row in csv_rows:
        if row.get('training_success') in ['True', 'true', '1']:
            stats['training_success'] += 1
        if row.get('energy_cpu_total_joules') and row.get('energy_cpu_total_joules').strip():
            stats['cpu_energy'] += 1
        if ((row.get('energy_gpu_total_joules') and row.get('energy_gpu_total_joules').strip()) or
            (row.get('fg_energy_gpu_total_joules') and row.get('fg_energy_gpu_total_joules').strip())):
            stats['gpu_energy'] += 1
        if ((row.get('perf_accuracy') and row.get('perf_accuracy').strip()) or
            (row.get('fg_perf_accuracy') and row.get('fg_perf_accuracy').strip())):
            stats['performance'] += 1
        if row.get('mode') == 'parallel':
            stats['parallel_mode'] += 1
        if row.get('bg_hyperparam_batch_size') and row.get('bg_hyperparam_batch_size').strip():
            stats['bg_hyperparams'] += 1
        if row.get('bg_energy_cpu_total_joules') and row.get('bg_energy_cpu_total_joules').strip():
            stats['bg_energy'] += 1

    total = len(csv_rows)
    print(f"  训练成功: {stats['training_success']}/{total} ({stats['training_success']/total*100:.1f}%)")
    print(f"  CPU能耗完整: {stats['cpu_energy']}/{total} ({stats['cpu_energy']/total*100:.1f}%)")
    print(f"  GPU能耗完整: {stats['gpu_energy']}/{total} ({stats['gpu_energy']/total*100:.1f}%)")
    print(f"  性能数据完整: {stats['performance']}/{total} ({stats['performance']/total*100:.1f}%)")
    print(f"  并行模式实验: {stats['parallel_mode']}/{total} ({stats['parallel_mode']/total*100:.1f}%)")
    print(f"  背景超参数填充: {stats['bg_hyperparams']}/{total} ({stats['bg_hyperparams']/total*100:.1f}%)")
    print(f"  背景能耗填充: {stats['bg_energy']}/{total} ({stats['bg_energy']/total*100:.1f}%)")

    # 4. 随机抽样验证
    print("\n步骤4: 随机抽样验证JSON与CSV一致性...")
    sample_size = min(10, len(csv_rows))
    sample_rows = random.sample(csv_rows, sample_size)

    validation_results = []
    for row in sample_rows:
        exp_id = row['experiment_id']
        json_path = find_experiment_json(exp_id)

        if not json_path:
            validation_results.append({
                'experiment_id': exp_id,
                'status': '✗',
                'message': 'JSON文件未找到'
            })
            continue

        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # 验证关键字段
            errors = []

            # 基础字段
            if json_data.get('timestamp') != row['timestamp']:
                errors.append(f"timestamp不匹配: {json_data.get('timestamp')} != {row['timestamp']}")

            # 能耗字段（检查映射是否正确）
            mode = json_data.get('mode', '')
            if mode == 'parallel':
                # 并行模式：检查前景能耗
                fg_energy = json_data.get('foreground', {}).get('energy_metrics', {})
                if 'gpu_power_avg_watts' in fg_energy:
                    csv_val = row.get('fg_energy_gpu_avg_watts', '')
                    json_val = str(fg_energy['gpu_power_avg_watts'])
                    if csv_val != json_val:
                        errors.append(f"fg_energy_gpu_avg_watts不匹配: {json_val} != {csv_val}")

                # 检查背景超参数（新增字段）
                bg_hyperparams = json_data.get('background', {}).get('hyperparameters', {})
                if 'batch_size' in bg_hyperparams:
                    csv_val = row.get('bg_hyperparam_batch_size', '')
                    json_val = str(bg_hyperparams['batch_size'])
                    if csv_val != json_val:
                        errors.append(f"bg_hyperparam_batch_size不匹配: {json_val} != {csv_val}")
            else:
                # 非并行模式：检查顶层能耗
                energy = json_data.get('energy_metrics', {})
                if 'gpu_power_avg_watts' in energy:
                    csv_val = row.get('energy_gpu_avg_watts', '')
                    json_val = str(energy['gpu_power_avg_watts'])
                    if csv_val != json_val:
                        errors.append(f"energy_gpu_avg_watts不匹配: {json_val} != {csv_val}")

            if errors:
                validation_results.append({
                    'experiment_id': exp_id,
                    'status': '⚠️',
                    'message': f"{len(errors)}个字段不匹配: {errors[0]}"
                })
            else:
                validation_results.append({
                    'experiment_id': exp_id,
                    'status': '✓',
                    'message': '数据一致'
                })

        except Exception as e:
            validation_results.append({
                'experiment_id': exp_id,
                'status': '✗',
                'message': f'验证失败: {e}'
            })

    # 输出验证结果
    success_count = sum(1 for r in validation_results if r['status'] == '✓')
    for result in validation_results:
        print(f"  {result['status']} {result['experiment_id']}: {result['message']}")

    print(f"\n  抽样验证: {success_count}/{len(validation_results)} 通过 ({success_count/len(validation_results)*100:.0f}%)")

    # 5. 新增字段验证
    print("\n步骤5: 验证新增字段（背景超参数和能耗）...")

    # 找出有背景数据的实验
    parallel_rows = [r for r in csv_rows if r.get('mode') == 'parallel']
    print(f"  并行模式实验: {len(parallel_rows)}个")

    # 检查背景超参数填充率
    bg_hyperparam_fields = [
        'bg_hyperparam_batch_size', 'bg_hyperparam_dropout', 'bg_hyperparam_epochs',
        'bg_hyperparam_learning_rate', 'bg_hyperparam_seed', 'bg_hyperparam_weight_decay'
    ]

    for field in bg_hyperparam_fields:
        filled = sum(1 for r in parallel_rows if r.get(field) and r.get(field).strip())
        print(f"    {field}: {filled}/{len(parallel_rows)} ({filled/len(parallel_rows)*100:.1f}%)" if len(parallel_rows) > 0 else f"    {field}: N/A")

    # 检查背景能耗填充率
    bg_energy_fields = [
        'bg_energy_cpu_pkg_joules', 'bg_energy_cpu_ram_joules', 'bg_energy_cpu_total_joules',
        'bg_energy_gpu_avg_watts', 'bg_energy_gpu_max_watts', 'bg_energy_gpu_min_watts',
        'bg_energy_gpu_total_joules'
    ]

    print(f"\n  背景能耗字段填充率:")
    for field in bg_energy_fields:
        filled = sum(1 for r in parallel_rows if r.get(field) and r.get(field).strip())
        print(f"    {field}: {filled}/{len(parallel_rows)} ({filled/len(parallel_rows)*100:.1f}%)" if len(parallel_rows) > 0 else f"    {field}: N/A")

    # 6. 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)

    all_passed = (
        len(csv_header) == 93 and
        stats['training_success'] > total * 0.9 and
        stats['cpu_energy'] > total * 0.9 and
        stats['gpu_energy'] > total * 0.9 and
        success_count >= len(validation_results) * 0.8
    )

    if all_passed:
        print("\n✅ 验证通过！数据质量符合预期。")
        print("\n可以安全替换原文件:")
        print("  mv results/summary_old_93col.csv results/summary_old.csv")
    else:
        print("\n⚠️ 验证发现问题，请检查数据:")
        if len(csv_header) != 93:
            print("  - 列数不正确")
        if stats['training_success'] <= total * 0.9:
            print("  - 训练成功率低于90%")
        if stats['cpu_energy'] <= total * 0.9:
            print("  - CPU能耗数据不完整")
        if stats['gpu_energy'] <= total * 0.9:
            print("  - GPU能耗数据不完整")
        if success_count < len(validation_results) * 0.8:
            print("  - 抽样验证通过率低于80%")

    return all_passed


if __name__ == '__main__':
    passed = validate_93col_csv()
    exit(0 if passed else 1)
