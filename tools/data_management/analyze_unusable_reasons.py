#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析不可用数据的具体原因

重点分析性能指标缺失和训练失败的具体情况
"""

import csv
from collections import defaultdict

def is_empty(val):
    """检查值是否为空"""
    return val == '' or val is None

def get_mode(row):
    """获取实验模式"""
    mode = row.get('mode', '')
    if is_empty(mode):
        if not is_empty(row.get('fg_repository')):
            return 'parallel'
        else:
            return 'non-parallel'
    return mode

def main():
    data_file = "data/raw_data.csv"

    print("=" * 100)
    print("🔍 不可用数据深入分析")
    print("=" * 100)
    print(f"\n数据文件: {data_file}\n")

    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)

    # ===== 1. 分析性能指标缺失的记录 =====
    print("=" * 100)
    print("📊 性能指标缺失详细分析")
    print("=" * 100)

    perf_missing_records = []

    for row in rows:
        mode = get_mode(row)

        # 检查性能指标
        if mode == 'parallel':
            perf_fields = [
                'fg_perf_accuracy', 'fg_perf_test_accuracy', 'fg_perf_map',
                'fg_perf_precision', 'fg_perf_recall', 'fg_perf_best_val_accuracy'
            ]
            repo = row.get('fg_repository', 'unknown')
            model = row.get('fg_model', 'unknown')
        else:
            perf_fields = [
                'perf_accuracy', 'perf_test_accuracy', 'perf_map',
                'perf_precision', 'perf_recall', 'perf_best_val_accuracy',
                'perf_top1_accuracy', 'perf_top5_accuracy'
            ]
            repo = row.get('repository', 'unknown')
            model = row.get('model', 'unknown')

        # 检查是否所有性能字段都为空
        has_perf = any(not is_empty(row.get(field)) for field in perf_fields)

        if not has_perf:
            perf_missing_records.append({
                'experiment_id': row.get('experiment_id', 'N/A'),
                'repo': repo,
                'model': model,
                'mode': mode,
                'training_success': row.get('fg_training_success' if mode == 'parallel' else 'training_success', ''),
                'has_energy': not is_empty(row.get('fg_energy_cpu_total_joules' if mode == 'parallel' else 'energy_cpu_total_joules')),
                'timestamp': row.get('timestamp', 'N/A'),
                'error_message': row.get('fg_error_message' if mode == 'parallel' else 'error_message', '')
            })

    print(f"\n性能指标缺失的记录数: {len(perf_missing_records)} ({len(perf_missing_records)*100/total_rows:.1f}%)")

    # 按模型统计
    perf_missing_by_model = defaultdict(lambda: {'count': 0, 'training_success': 0, 'training_failed': 0, 'has_energy': 0})

    for record in perf_missing_records:
        model_key = f"{record['repo']}/{record['model']}"
        perf_missing_by_model[model_key]['count'] += 1

        if record['training_success'] == 'True':
            perf_missing_by_model[model_key]['training_success'] += 1
        else:
            perf_missing_by_model[model_key]['training_failed'] += 1

        if record['has_energy']:
            perf_missing_by_model[model_key]['has_energy'] += 1

    print(f"\n{'模型':<50} {'总缺失':<10} {'训练成功':<12} {'训练失败':<12} {'有能耗':<10}")
    print("-" * 100)

    for model_key in sorted(perf_missing_by_model.keys(), key=lambda x: perf_missing_by_model[x]['count'], reverse=True):
        stats = perf_missing_by_model[model_key]
        print(f"{model_key:<50} {stats['count']:<10} {stats['training_success']:<12} "
              f"{stats['training_failed']:<12} {stats['has_energy']:<10}")

    # ===== 2. 分析训练失败的记录 =====
    print("\n" + "=" * 100)
    print("⚠️  训练失败详细分析")
    print("=" * 100)

    training_failed_records = []

    for row in rows:
        mode = get_mode(row)

        if mode == 'parallel':
            training_success = row.get('fg_training_success', '') == 'True'
            repo = row.get('fg_repository', 'unknown')
            model = row.get('fg_model', 'unknown')
            error_msg = row.get('fg_error_message', '')
        else:
            training_success = row.get('training_success', '') == 'True'
            repo = row.get('repository', 'unknown')
            model = row.get('model', 'unknown')
            error_msg = row.get('error_message', '')

        if not training_success:
            training_failed_records.append({
                'experiment_id': row.get('experiment_id', 'N/A'),
                'repo': repo,
                'model': model,
                'mode': mode,
                'timestamp': row.get('timestamp', 'N/A'),
                'error_message': error_msg
            })

    print(f"\n训练失败的记录数: {len(training_failed_records)} ({len(training_failed_records)*100/total_rows:.1f}%)")

    # 按模型统计
    failed_by_model = defaultdict(int)
    for record in training_failed_records:
        model_key = f"{record['repo']}/{record['model']}"
        failed_by_model[model_key] += 1

    print(f"\n{'模型':<50} {'失败次数':<12} {'失败率':<10}")
    print("-" * 75)

    for model_key in sorted(failed_by_model.keys(), key=lambda x: failed_by_model[x], reverse=True):
        count = failed_by_model[model_key]
        # 计算该模型的总实验数
        total_for_model = sum(1 for row in rows
                             if f"{row.get('repository', '')}/{row.get('model', '')}" == model_key
                             or f"{row.get('fg_repository', '')}/{row.get('fg_model', '')}" == model_key)
        failure_rate = count * 100 / total_for_model if total_for_model > 0 else 0
        print(f"{model_key:<50} {count:<12} {failure_rate:.1f}%")

    # 查看错误消息
    print(f"\n训练失败的错误消息示例（前10个）:")
    print("-" * 100)

    for i, record in enumerate(training_failed_records[:10], 1):
        print(f"\n{i}. {record['experiment_id']}")
        print(f"   模型: {record['repo']}/{record['model']}")
        print(f"   模式: {record['mode']}")
        error_msg = record['error_message'][:200] if record['error_message'] else "无错误消息"
        print(f"   错误: {error_msg}")

    # ===== 3. 分析能耗数据缺失的记录 =====
    print("\n" + "=" * 100)
    print("⚡ 能耗数据缺失详细分析")
    print("=" * 100)

    energy_missing_records = []

    for row in rows:
        mode = get_mode(row)

        if mode == 'parallel':
            has_energy = not is_empty(row.get('fg_energy_cpu_total_joules'))
            repo = row.get('fg_repository', 'unknown')
            model = row.get('fg_model', 'unknown')
            training_success = row.get('fg_training_success', '') == 'True'
        else:
            has_energy = not is_empty(row.get('energy_cpu_total_joules'))
            repo = row.get('repository', 'unknown')
            model = row.get('model', 'unknown')
            training_success = row.get('training_success', '') == 'True'

        if not has_energy:
            energy_missing_records.append({
                'experiment_id': row.get('experiment_id', 'N/A'),
                'repo': repo,
                'model': model,
                'mode': mode,
                'training_success': training_success,
                'timestamp': row.get('timestamp', 'N/A')
            })

    print(f"\n能耗数据缺失的记录数: {len(energy_missing_records)} ({len(energy_missing_records)*100/total_rows:.1f}%)")

    # 按模型统计
    energy_missing_by_model = defaultdict(lambda: {'count': 0, 'training_success': 0, 'training_failed': 0})

    for record in energy_missing_records:
        model_key = f"{record['repo']}/{record['model']}"
        energy_missing_by_model[model_key]['count'] += 1

        if record['training_success']:
            energy_missing_by_model[model_key]['training_success'] += 1
        else:
            energy_missing_by_model[model_key]['training_failed'] += 1

    print(f"\n{'模型':<50} {'总缺失':<10} {'训练成功':<12} {'训练失败':<12}")
    print("-" * 90)

    for model_key in sorted(energy_missing_by_model.keys(), key=lambda x: energy_missing_by_model[x]['count'], reverse=True):
        stats = energy_missing_by_model[model_key]
        print(f"{model_key:<50} {stats['count']:<10} {stats['training_success']:<12} {stats['training_failed']:<12}")

    # ===== 4. VulBERTa 特别分析 =====
    print("\n" + "=" * 100)
    print("🔬 VulBERTa 模型特别分析")
    print("=" * 100)

    vulberta_records = [row for row in rows
                       if row.get('repository') == 'VulBERTa'
                       or row.get('fg_repository') == 'VulBERTa']

    print(f"\nVulBERTa 总记录数: {len(vulberta_records)}")

    # 检查VulBERTa的性能字段
    if len(vulberta_records) > 0:
        sample_row = vulberta_records[0]
        mode = get_mode(sample_row)

        print(f"\n示例记录的所有性能字段值:")
        print(f"模式: {mode}")

        if mode == 'parallel':
            perf_fields = ['fg_perf_accuracy', 'fg_perf_test_accuracy', 'fg_perf_map',
                          'fg_perf_precision', 'fg_perf_recall', 'fg_perf_best_val_accuracy',
                          'fg_perf_test_loss']
        else:
            perf_fields = ['perf_accuracy', 'perf_test_accuracy', 'perf_map',
                          'perf_precision', 'perf_recall', 'perf_best_val_accuracy',
                          'perf_test_loss', 'perf_eval_loss', 'perf_final_training_loss']

        for field in perf_fields:
            val = sample_row.get(field, '')
            print(f"  {field}: {val if val else '(空)'}")

    # ===== 5. 总结 =====
    print("\n" + "=" * 100)
    print("📊 不可用数据原因总结")
    print("=" * 100)

    print(f"\n1. 性能指标缺失 ({len(perf_missing_records)} 条, {len(perf_missing_records)*100/total_rows:.1f}%):")
    print(f"   主要影响模型:")
    for model_key in sorted(perf_missing_by_model.keys(), key=lambda x: perf_missing_by_model[x]['count'], reverse=True)[:3]:
        stats = perf_missing_by_model[model_key]
        print(f"   - {model_key}: {stats['count']} 条")
        print(f"     * 训练成功但无性能指标: {stats['training_success']} 条")
        print(f"     * 训练失败: {stats['training_failed']} 条")

    print(f"\n2. 训练失败 ({len(training_failed_records)} 条, {len(training_failed_records)*100/total_rows:.1f}%):")
    print(f"   主要影响模型:")
    for model_key in sorted(failed_by_model.keys(), key=lambda x: failed_by_model[x], reverse=True)[:3]:
        count = failed_by_model[model_key]
        print(f"   - {model_key}: {count} 条")

    print(f"\n3. 能耗数据缺失 ({len(energy_missing_records)} 条, {len(energy_missing_records)*100/total_rows:.1f}%):")
    print(f"   主要影响模型:")
    for model_key in sorted(energy_missing_by_model.keys(), key=lambda x: energy_missing_by_model[x]['count'], reverse=True)[:3]:
        stats = energy_missing_by_model[model_key]
        print(f"   - {model_key}: {stats['count']} 条")
        print(f"     * 训练成功但无能耗: {stats['training_success']} 条")
        print(f"     * 训练失败: {stats['training_failed']} 条")

    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
