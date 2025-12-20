#!/usr/bin/env python3
"""
性能指标保守合并脚本

合并操作:
1. 将 MRT-OAST 的 accuracy 重命名为 test_accuracy
2. 将 VulBERTa/mlp 的 eval_loss 重命名为 test_loss

原则: 无数据丢失，仅统一命名

版本: v1.0
日期: 2025-12-19
"""

import csv
import shutil
from datetime import datetime

def merge_performance_metrics():
    """执行保守合并操作"""

    input_file = 'results/data.csv'
    output_file = 'results/data_merged_metrics.csv'
    backup_file = f'results/data.csv.backup_before_merge_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    print("=" * 100)
    print("性能指标保守合并")
    print("=" * 100)
    print()

    # 备份原文件
    shutil.copy(input_file, backup_file)
    print(f"✓ 已备份原文件: {backup_file}")
    print()

    # 统计信息
    stats = {
        'total_rows': 0,
        'accuracy_merged': 0,
        'eval_loss_merged': 0,
        'unchanged_rows': 0
    }

    # 读取并处理数据
    with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            stats['total_rows'] += 1
            modified = False

            # 操作1: MRT-OAST 的 accuracy → test_accuracy
            if row['repository'] == 'MRT-OAST':
                accuracy_value = row.get('perf_accuracy', '').strip()
                if accuracy_value:
                    # 检查 test_accuracy 是否已有值
                    test_accuracy_value = row.get('perf_test_accuracy', '').strip()
                    if not test_accuracy_value:
                        # 将 accuracy 移动到 test_accuracy
                        row['perf_test_accuracy'] = accuracy_value
                        row['perf_accuracy'] = ''
                        stats['accuracy_merged'] += 1
                        modified = True

            # 操作2: VulBERTa/mlp 的 eval_loss → test_loss
            if row['repository'] == 'VulBERTa' and row['model'] == 'mlp':
                eval_loss_value = row.get('perf_eval_loss', '').strip()
                if eval_loss_value:
                    # 检查 test_loss 是否已有值
                    test_loss_value = row.get('perf_test_loss', '').strip()
                    if not test_loss_value:
                        # 将 eval_loss 移动到 test_loss
                        row['perf_test_loss'] = eval_loss_value
                        row['perf_eval_loss'] = ''
                        stats['eval_loss_merged'] += 1
                        modified = True

            if not modified:
                stats['unchanged_rows'] += 1

            writer.writerow(row)

    # 输出报告
    print("合并统计:")
    print("-" * 100)
    print(f"总行数: {stats['total_rows']}")
    print(f"  accuracy → test_accuracy: {stats['accuracy_merged']} 个实验")
    print(f"  eval_loss → test_loss: {stats['eval_loss_merged']} 个实验")
    print(f"  未修改: {stats['unchanged_rows']} 个实验")
    print(f"  总合并: {stats['accuracy_merged'] + stats['eval_loss_merged']} 个实验")
    print()

    print("=" * 100)
    print("合并详情:")
    print("=" * 100)
    print(f"✓ 操作1: MRT-OAST 的 perf_accuracy 字段内容移至 perf_test_accuracy")
    print(f"  影响实验: {stats['accuracy_merged']} 个")
    print(f"  原 perf_accuracy 列: 清空（不删除列）")
    print()
    print(f"✓ 操作2: VulBERTa/mlp 的 perf_eval_loss 字段内容移至 perf_test_loss")
    print(f"  影响实验: {stats['eval_loss_merged']} 个")
    print(f"  原 perf_eval_loss 列: 清空（不删除列）")
    print()

    print("=" * 100)
    print(f"✓ 输出文件: {output_file}")
    print(f"✓ 备份文件: {backup_file}")
    print("=" * 100)
    print()

    print("下一步:")
    print("  1. 验证合并后的数据: python3 scripts/validate_merged_metrics.py")
    print("  2. 如果验证通过: mv results/data_merged_metrics.csv results/data.csv")
    print()

    return stats

if __name__ == '__main__':
    stats = merge_performance_metrics()

    # 简要总结
    total_merged = stats['accuracy_merged'] + stats['eval_loss_merged']
    print(f"合并完成! 共影响 {total_merged} 个实验（{total_merged/stats['total_rows']*100:.1f}%）")
