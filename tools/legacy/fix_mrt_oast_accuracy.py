#!/usr/bin/env python3
"""
修复 raw_data.csv 中的 MRT-OAST accuracy 数据问题

问题: MRT-OAST 的 accuracy 字段存储的是样本数而非百分比
解决: 从训练日志中重新提取真实的准确率百分比

版本: v1.0
日期: 2025-12-19
"""

import csv
import json
import re
import os
from pathlib import Path

def find_experiment_json(exp_id, timestamp):
    """根据 experiment_id 和 timestamp 查找对应的 experiment.json 文件"""
    # 查找所有 experiment.json 文件
    results_dir = Path('results')

    for json_file in results_dir.rglob('experiment.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if data.get('experiment_id') == exp_id and data.get('timestamp') == timestamp:
                return json_file.parent
        except:
            continue

    return None

def extract_accuracy_from_log(log_file):
    """从训练日志中提取真实的准确率百分比"""
    if not os.path.exists(log_file):
        return None

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # 匹配中文输出: 准确率 (Accuracy): 0.9186
        pattern = r'准确率\s*\(Accuracy\)\s*:\s*([0-9.]+)'
        match = re.search(pattern, log_content)

        if match:
            return float(match.group(1))

        # 也尝试英文模式（备用）
        pattern_en = r'Accuracy:\s*\d+/\d+\s*=\s*([0-9.]+)'
        match_en = re.search(pattern_en, log_content)

        if match_en:
            return float(match_en.group(1))

    except Exception as e:
        print(f"  警告: 无法从日志提取准确率: {e}")

    return None

def fix_mrt_oast_accuracy():
    """修复 MRT-OAST accuracy 数据"""
    input_file = 'data/raw_data.csv'
    output_file = 'results/raw_data_fixed.csv'
    backup_file = 'data/raw_data.csv.backup_before_fix'

    print("=" * 80)
    print("修复 MRT-OAST accuracy 数据问题")
    print("=" * 80)
    print()

    # 备份原文件
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy(input_file, backup_file)
        print(f"✓ 已备份原文件: {backup_file}")

    # 统计
    stats = {
        'total_rows': 0,
        'mrt_oast_rows': 0,
        'fixed_rows': 0,
        'failed_rows': 0,
        'sample_fixes': []
    }

    with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            stats['total_rows'] += 1

            # 检查是否是 MRT-OAST 实验
            repo = row['repository'] if row.get('mode') != 'parallel' else row.get('fg_repository', '')
            model = row['model'] if row.get('mode') != 'parallel' else row.get('fg_model', '')

            if repo == 'MRT-OAST' and model == 'default':
                stats['mrt_oast_rows'] += 1

                # 检查 accuracy 是否异常（大于1表示是样本数）
                is_parallel = (row['mode'] == 'parallel')
                acc_field = 'fg_perf_accuracy' if is_parallel else 'perf_accuracy'
                acc_value = row.get(acc_field, '').strip()

                if acc_value and float(acc_value) > 1.0:
                    # 尝试从训练日志提取真实准确率
                    exp_id = row['experiment_id']
                    timestamp = row['timestamp']

                    exp_dir = find_experiment_json(exp_id, timestamp)

                    if exp_dir:
                        log_file = exp_dir / 'training.log'
                        real_accuracy = extract_accuracy_from_log(log_file)

                        if real_accuracy is not None:
                            old_value = acc_value
                            row[acc_field] = str(real_accuracy)
                            stats['fixed_rows'] += 1

                            # 记录前5个修复示例
                            if len(stats['sample_fixes']) < 5:
                                stats['sample_fixes'].append({
                                    'exp_id': exp_id,
                                    'old': old_value,
                                    'new': real_accuracy
                                })
                        else:
                            stats['failed_rows'] += 1
                            print(f"  ✗ 无法提取准确率: {exp_id}")
                    else:
                        stats['failed_rows'] += 1
                        print(f"  ✗ 找不到实验目录: {exp_id}")

            writer.writerow(row)

    # 输出报告
    print()
    print("=" * 80)
    print("修复统计报告")
    print("=" * 80)
    print(f"总行数: {stats['total_rows']}")
    print(f"MRT-OAST 实验: {stats['mrt_oast_rows']}")
    print(f"  成功修复: {stats['fixed_rows']}")
    print(f"  修复失败: {stats['failed_rows']}")
    print()

    if stats['sample_fixes']:
        print("修复示例:")
        for i, fix in enumerate(stats['sample_fixes'], 1):
            print(f"  {i}. {fix['exp_id']}")
            print(f"     修复前: {fix['old']} (样本数)")
            print(f"     修复后: {fix['new']} (准确率百分比)")
        print()

    print("=" * 80)
    print(f"✓ 修复完成")
    print(f"✓ 输出文件: {output_file}")
    print(f"✓ 备份文件: {backup_file}")
    print("=" * 80)

    return stats

if __name__ == '__main__':
    stats = fix_mrt_oast_accuracy()

    if stats['fixed_rows'] > 0:
        print()
        print("下一步: 请检查修复后的数据，如果正确则替换原文件:")
        print("  mv results/raw_data_fixed.csv data/raw_data.csv")
    elif stats['mrt_oast_rows'] == 0:
        print()
        print("注意: 未找到需要修复的 MRT-OAST 实验数据")
    else:
        print()
        print("警告: 所有修复尝试均失败，请检查日志文件")
