#!/usr/bin/env python3
"""
验证合并后的性能指标数据质量

检查项:
1. 数据完整性 - 无数据丢失
2. 合并正确性 - 值已移动到新列
3. 列填充率变化
4. 实验目标达成度

版本: v1.0
日期: 2025-12-19
"""

import csv
from collections import defaultdict

def validate_merged_metrics():
    """验证合并后的数据质量"""

    original_file = 'data/data.csv'
    merged_file = 'results/data_merged_metrics.csv'

    print("=" * 100)
    print("性能指标合并验证")
    print("=" * 100)
    print()

    # 1. 读取原始数据
    print("正在读取原始数据...")
    original_data = {}
    with open(original_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['experiment_id']}|{row['timestamp']}"
            original_data[key] = row

    # 2. 读取合并后数据
    print("正在读取合并后数据...")
    merged_data = {}
    with open(merged_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['experiment_id']}|{row['timestamp']}"
            merged_data[key] = row

    print(f"✓ 原始数据: {len(original_data)} 行")
    print(f"✓ 合并数据: {len(merged_data)} 行")
    print()

    # 3. 验证数据完整性
    print("=" * 100)
    print("验证1: 数据完整性")
    print("=" * 100)

    if len(original_data) != len(merged_data):
        print("❌ 错误: 行数不匹配!")
        return False

    missing_keys = set(original_data.keys()) - set(merged_data.keys())
    if missing_keys:
        print(f"❌ 错误: 缺失 {len(missing_keys)} 个实验!")
        return False

    print("✓ 行数一致: 676行")
    print("✓ 无实验缺失")
    print()

    # 4. 验证合并正确性
    print("=" * 100)
    print("验证2: 合并正确性")
    print("=" * 100)

    validation_errors = []

    # 验证 MRT-OAST accuracy → test_accuracy
    mrt_oast_checked = 0
    mrt_oast_correct = 0

    for key, orig_row in original_data.items():
        if orig_row['repository'] == 'MRT-OAST':
            merged_row = merged_data[key]
            orig_accuracy = orig_row.get('perf_accuracy', '').strip()

            if orig_accuracy:
                mrt_oast_checked += 1
                merged_test_accuracy = merged_row.get('perf_test_accuracy', '').strip()
                merged_accuracy = merged_row.get('perf_accuracy', '').strip()

                # 检查值已移动
                if merged_test_accuracy == orig_accuracy and not merged_accuracy:
                    mrt_oast_correct += 1
                else:
                    validation_errors.append(f"MRT-OAST {key}: accuracy未正确移动")

    # 验证 VulBERTa/mlp eval_loss → test_loss
    vulberta_checked = 0
    vulberta_correct = 0

    for key, orig_row in original_data.items():
        if orig_row['repository'] == 'VulBERTa' and orig_row['model'] == 'mlp':
            merged_row = merged_data[key]
            orig_eval_loss = orig_row.get('perf_eval_loss', '').strip()

            if orig_eval_loss:
                vulberta_checked += 1
                merged_test_loss = merged_row.get('perf_test_loss', '').strip()
                merged_eval_loss = merged_row.get('perf_eval_loss', '').strip()

                # 检查值已移动
                if merged_test_loss == orig_eval_loss and not merged_eval_loss:
                    vulberta_correct += 1
                else:
                    validation_errors.append(f"VulBERTa/mlp {key}: eval_loss未正确移动")

    print(f"MRT-OAST accuracy 合并:")
    print(f"  检查实验: {mrt_oast_checked}")
    print(f"  合并正确: {mrt_oast_correct}")
    if mrt_oast_correct == mrt_oast_checked:
        print(f"  ✓ 100% 正确")
    else:
        print(f"  ❌ {mrt_oast_checked - mrt_oast_correct} 个错误")
    print()

    print(f"VulBERTa/mlp eval_loss 合并:")
    print(f"  检查实验: {vulberta_checked}")
    print(f"  合并正确: {vulberta_correct}")
    if vulberta_correct == vulberta_checked:
        print(f"  ✓ 100% 正确")
    else:
        print(f"  ❌ {vulberta_checked - vulberta_correct} 个错误")
    print()

    if validation_errors:
        print("发现错误:")
        for error in validation_errors[:10]:
            print(f"  - {error}")
        if len(validation_errors) > 10:
            print(f"  ... 还有 {len(validation_errors) - 10} 个错误")
        return False

    # 5. 统计列填充率变化
    print("=" * 100)
    print("验证3: 列填充率变化")
    print("=" * 100)

    metrics_to_check = [
        'test_accuracy', 'accuracy', 'test_loss', 'eval_loss'
    ]

    for metric in metrics_to_check:
        col_name = f'perf_{metric}'

        # 原始填充率
        orig_count = sum(1 for row in original_data.values() if row.get(col_name, '').strip())
        orig_rate = orig_count / len(original_data) * 100

        # 合并后填充率
        merged_count = sum(1 for row in merged_data.values() if row.get(col_name, '').strip())
        merged_rate = merged_count / len(merged_data) * 100

        change = merged_count - orig_count

        print(f"{metric}:")
        print(f"  原始: {orig_count} ({orig_rate:.1f}%)")
        print(f"  合并: {merged_count} ({merged_rate:.1f}%)")
        if change > 0:
            print(f"  变化: +{change} ✓")
        elif change < 0:
            print(f"  变化: {change} ⚠️")
        else:
            print(f"  变化: 0 ✓")
        print()

    # 6. 验证实验目标达成度
    print("=" * 100)
    print("验证4: 实验目标达成度")
    print("=" * 100)

    # 统计每个模型的参数-模式-唯一值
    model_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    with open(merged_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            repo = row['repository']
            model = row['model']
            is_parallel = (row['is_parallel'] == 'True')
            mode = 'parallel' if is_parallel else 'nonparallel'

            model_key = f"{repo}/{model}"

            # 获取超参数
            hyperparams = {}
            for col in reader.fieldnames:
                if col.startswith('hyperparam_'):
                    value = row.get(col, '').strip()
                    if value:
                        param_name = col.replace('hyperparam_', '')
                        hyperparams[param_name] = value

            # 获取性能指标（检查是否有任何性能数据）
            has_perf = False
            for col in reader.fieldnames:
                if col.startswith('perf_'):
                    value = row.get(col, '').strip()
                    if value:
                        has_perf = True
                        break

            # 统计唯一值
            if has_perf:
                for param_name, param_value in hyperparams.items():
                    model_stats[model_key][mode][param_name].add(param_value)

    # 计算达标情况
    total_combinations = 0
    total_met = 0

    for model_key in model_stats.keys():
        for mode in ['nonparallel', 'parallel']:
            params = model_stats[model_key][mode]
            for param_name, unique_vals in params.items():
                total_combinations += 1
                if len(unique_vals) >= 5:
                    total_met += 1

    print(f"参数-模式组合: {total_combinations}")
    print(f"达标组合（≥5个唯一值）: {total_met}")
    if total_combinations > 0:
        print(f"达标率: {total_met}/{total_combinations} ({total_met/total_combinations*100:.1f}%)")

    if total_met == total_combinations:
        print("✓ 100% 达标")
    else:
        print(f"⚠️ 部分未达标: {total_combinations - total_met} 个组合")
    print()

    # 7. 总结
    print("=" * 100)
    print("验证总结")
    print("=" * 100)

    all_passed = (
        len(original_data) == len(merged_data) and
        mrt_oast_correct == mrt_oast_checked and
        vulberta_correct == vulberta_checked and
        len(validation_errors) == 0
    )

    if all_passed:
        print("✅ 所有验证通过!")
        print()
        print("合并成功，可以安全替换原文件:")
        print("  mv results/data_merged_metrics.csv data/data.csv")
        return True
    else:
        print("❌ 验证失败，请检查错误信息")
        return False

if __name__ == '__main__':
    success = validate_merged_metrics()
    exit(0 if success else 1)
