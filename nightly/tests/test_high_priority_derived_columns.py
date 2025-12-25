#!/usr/bin/env python3
"""
测试3个高优先级派生变量的正确性

测试内容：
1. energy_gpu_power_range - GPU功率波动范围
2. energy_gpu_percent - GPU能耗占比
3. temperature_range - GPU温度波动范围

版本: v1.0
日期: 2025-12-11
"""

import csv
import sys
from pathlib import Path


def test_column_presence():
    """测试1: 检查新列是否存在"""
    print("\n=== 测试1: 列存在性检查 ===")

    csv_file = Path('/home/green/energy_dl/nightly/results/summary_all_enhanced.csv')

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    required_cols = ['energy_gpu_power_range', 'energy_gpu_percent', 'temperature_range']

    all_present = True
    for col in required_cols:
        present = col in fieldnames
        status = "✅ PASS" if present else "❌ FAIL"
        print(f"  {status} - 列 '{col}' {'存在' if present else '不存在'}")
        if not present:
            all_present = False

    return all_present


def test_data_completeness():
    """测试2: 检查数据完整性（无空值）"""
    print("\n=== 测试2: 数据完整性检查 ===")

    csv_file = Path('/home/green/energy_dl/nightly/results/summary_all_enhanced.csv')

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    cols = ['energy_gpu_power_range', 'energy_gpu_percent', 'temperature_range']

    all_complete = True
    for col in cols:
        non_empty = sum(1 for row in rows if row.get(col, '').strip())
        pct = non_empty / total * 100

        is_complete = non_empty == total
        status = "✅ PASS" if is_complete else "❌ FAIL"
        print(f"  {status} - {col}: {non_empty}/{total} ({pct:.1f}%)")

        if not is_complete:
            all_complete = False

    return all_complete


def test_formula_correctness():
    """测试3: 验证计算公式正确性"""
    print("\n=== 测试3: 公式正确性验证 ===")

    csv_file = Path('/home/green/energy_dl/nightly/results/summary_all_enhanced.csv')

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 随机选择50个样本进行验证
    import random
    sample_size = min(50, len(rows))
    samples = random.sample(rows, sample_size)

    errors = []

    for i, row in enumerate(samples):
        exp_id = row.get('experiment_id', f'row_{i}')

        try:
            # 验证 energy_gpu_power_range
            gpu_max = float(row['energy_gpu_max_watts'])
            gpu_min = float(row['energy_gpu_min_watts'])
            power_range = float(row['energy_gpu_power_range'])
            expected_power_range = gpu_max - gpu_min

            if abs(power_range - expected_power_range) > 0.01:
                errors.append(f"{exp_id}: energy_gpu_power_range 错误 (期望={expected_power_range:.2f}, 实际={power_range:.2f})")

            # 验证 energy_gpu_percent
            cpu_total = float(row['energy_cpu_total_joules'])
            gpu_total = float(row['energy_gpu_total_joules'])
            gpu_percent = float(row['energy_gpu_percent'])
            total_energy = cpu_total + gpu_total
            expected_gpu_percent = (gpu_total / total_energy * 100) if total_energy > 0 else 0

            if abs(gpu_percent - expected_gpu_percent) > 0.01:
                errors.append(f"{exp_id}: energy_gpu_percent 错误 (期望={expected_gpu_percent:.2f}, 实际={gpu_percent:.2f})")

            # 验证 temperature_range
            temp_max = float(row['energy_gpu_temp_max_celsius'])
            temp_avg = float(row['energy_gpu_temp_avg_celsius'])
            temp_range = float(row['temperature_range'])
            expected_temp_range = temp_max - temp_avg

            if abs(temp_range - expected_temp_range) > 0.01:
                errors.append(f"{exp_id}: temperature_range 错误 (期望={expected_temp_range:.2f}, 实际={temp_range:.2f})")

        except (ValueError, KeyError, ZeroDivisionError) as e:
            errors.append(f"{exp_id}: 计算异常 - {e}")

    if errors:
        print(f"  ❌ FAIL - 发现 {len(errors)} 个错误:")
        for err in errors[:5]:  # 只显示前5个
            print(f"     • {err}")
        if len(errors) > 5:
            print(f"     ... 还有 {len(errors) - 5} 个错误")
        return False
    else:
        print(f"  ✅ PASS - {sample_size} 个样本全部通过验证")
        return True


def test_value_ranges():
    """测试4: 验证数值范围合理性"""
    print("\n=== 测试4: 数值范围合理性检查 ===")

    csv_file = Path('/home/green/energy_dl/nightly/results/summary_all_enhanced.csv')

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    errors = []

    for row in rows:
        exp_id = row.get('experiment_id', 'unknown')

        # 验证 energy_gpu_power_range >= 0
        power_range = row.get('energy_gpu_power_range', '')
        if power_range:
            try:
                val = float(power_range)
                if val < 0:
                    errors.append(f"{exp_id}: energy_gpu_power_range < 0 ({val})")
            except ValueError:
                pass

        # 验证 energy_gpu_percent 在 [0, 100]
        gpu_percent = row.get('energy_gpu_percent', '')
        if gpu_percent:
            try:
                val = float(gpu_percent)
                if not (0 <= val <= 100):
                    errors.append(f"{exp_id}: energy_gpu_percent 超出范围 ({val})")
            except ValueError:
                pass

        # 验证 temperature_range >= 0
        temp_range = row.get('temperature_range', '')
        if temp_range:
            try:
                val = float(temp_range)
                if val < 0:
                    errors.append(f"{exp_id}: temperature_range < 0 ({val})")
            except ValueError:
                pass

    if errors:
        print(f"  ❌ FAIL - 发现 {len(errors)} 个范围错误:")
        for err in errors[:5]:
            print(f"     • {err}")
        if len(errors) > 5:
            print(f"     ... 还有 {len(errors) - 5} 个错误")
        return False
    else:
        print(f"  ✅ PASS - 所有 {len(rows)} 行数值范围正常")
        return True


def test_statistics():
    """测试5: 生成统计摘要"""
    print("\n=== 测试5: 统计摘要 ===")

    csv_file = Path('/home/green/energy_dl/nightly/results/summary_all_enhanced.csv')

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cols = ['energy_gpu_power_range', 'energy_gpu_percent', 'temperature_range']

    for col in cols:
        values = []
        for row in rows:
            val = row.get(col, '')
            if val:
                try:
                    values.append(float(val))
                except ValueError:
                    pass

        if values:
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)

            print(f"\n  {col}:")
            print(f"    • 样本数: {len(values)}/{len(rows)} ({len(values)/len(rows)*100:.1f}%)")
            print(f"    • 最小值: {min_val:.2f}")
            print(f"    • 平均值: {avg:.2f}")
            print(f"    • 最大值: {max_val:.2f}")

    return True


def main():
    """主函数 - 运行所有测试"""
    print("=" * 80)
    print("高优先级派生变量验证测试")
    print("=" * 80)

    csv_file = Path('/home/green/energy_dl/nightly/results/summary_all_enhanced.csv')

    # 检查文件存在性
    if not csv_file.exists():
        print(f"\n❌ 错误: 文件不存在: {csv_file}")
        sys.exit(1)

    print(f"\n测试文件: {csv_file}")

    # 运行所有测试
    results = {
        "列存在性": test_column_presence(),
        "数据完整性": test_data_completeness(),
        "公式正确性": test_formula_correctness(),
        "数值范围": test_value_ranges(),
        "统计摘要": test_statistics()
    }

    # 总结
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有测试通过！")
        print("=" * 80)
        return 0
    else:
        print("❌ 部分测试失败！")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
