#!/usr/bin/env python3
"""
添加3个高优先级派生变量到 summary_all_enhanced.csv

新增列：
1. energy_gpu_power_range - GPU功率波动范围 (瓦特)
2. energy_gpu_percent - GPU能耗占比 (%)
3. temperature_range - GPU温度波动范围 (摄氏度)

版本: v1.0
日期: 2025-12-11
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List


def validate_data_completeness(rows: List[Dict]) -> bool:
    """验证数据完整性"""
    required_fields = [
        'energy_gpu_max_watts',
        'energy_gpu_min_watts',
        'energy_cpu_total_joules',
        'energy_gpu_total_joules',
        'energy_gpu_temp_max_celsius',
        'energy_gpu_temp_avg_celsius'
    ]

    print("\n【数据完整性检查】")
    all_complete = True

    for field in required_fields:
        non_empty = sum(1 for row in rows if row.get(field, '').strip())
        pct = non_empty / len(rows) * 100
        status = "✅" if non_empty == len(rows) else "❌"
        print(f"  {status} {field:35s}: {non_empty:3d}/{len(rows)} ({pct:5.1f}%)")

        if non_empty != len(rows):
            all_complete = False

    return all_complete


def calculate_derived_columns(row: Dict) -> Dict[str, float]:
    """计算派生列的值"""
    derived = {}

    try:
        # 1. energy_gpu_power_range - GPU功率波动范围
        gpu_max = float(row.get('energy_gpu_max_watts', 0))
        gpu_min = float(row.get('energy_gpu_min_watts', 0))
        derived['energy_gpu_power_range'] = gpu_max - gpu_min

        # 2. energy_gpu_percent - GPU能耗占比
        cpu_total = float(row.get('energy_cpu_total_joules', 0))
        gpu_total = float(row.get('energy_gpu_total_joules', 0))
        total_energy = cpu_total + gpu_total

        if total_energy > 0:
            derived['energy_gpu_percent'] = (gpu_total / total_energy) * 100
        else:
            derived['energy_gpu_percent'] = 0.0

        # 3. temperature_range - GPU温度波动范围
        temp_max = float(row.get('energy_gpu_temp_max_celsius', 0))
        temp_avg = float(row.get('energy_gpu_temp_avg_celsius', 0))
        derived['temperature_range'] = temp_max - temp_avg

    except (ValueError, TypeError) as e:
        print(f"⚠️  警告: 计算失败 {row.get('experiment_id', 'unknown')}: {e}")
        derived = {
            'energy_gpu_power_range': None,
            'energy_gpu_percent': None,
            'temperature_range': None
        }

    return derived


def validate_results(rows: List[Dict]) -> bool:
    """验证计算结果的合理性"""
    print("\n【结果验证】")

    issues = []

    for row in rows:
        exp_id = row.get('experiment_id', 'unknown')

        # 验证GPU功率范围应为非负数
        power_range = row.get('energy_gpu_power_range', '')
        if power_range:
            try:
                val = float(power_range)
                if val < 0:
                    issues.append(f"{exp_id}: GPU功率范围为负数 ({val})")
            except ValueError:
                pass

        # 验证GPU能耗占比应在0-100之间
        gpu_pct = row.get('energy_gpu_percent', '')
        if gpu_pct:
            try:
                val = float(gpu_pct)
                if not (0 <= val <= 100):
                    issues.append(f"{exp_id}: GPU能耗占比超出范围 ({val}%)")
            except ValueError:
                pass

        # 验证温度范围应为非负数
        temp_range = row.get('temperature_range', '')
        if temp_range:
            try:
                val = float(temp_range)
                if val < 0:
                    issues.append(f"{exp_id}: 温度范围为负数 ({val})")
            except ValueError:
                pass

    if issues:
        print(f"  ❌ 发现 {len(issues)} 个问题:")
        for issue in issues[:10]:  # 只显示前10个
            print(f"     • {issue}")
        if len(issues) > 10:
            print(f"     ... 还有 {len(issues) - 10} 个问题")
        return False
    else:
        print("  ✅ 所有验证通过")
        return True


def generate_statistics(rows: List[Dict]) -> None:
    """生成统计摘要"""
    print("\n【统计摘要】")

    new_cols = ['energy_gpu_power_range', 'energy_gpu_percent', 'temperature_range']

    for col in new_cols:
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

            print(f"\n{col}:")
            print(f"  • 样本数: {len(values)}/{len(rows)} ({len(values)/len(rows)*100:.1f}%)")
            print(f"  • 平均值: {avg:.2f}")
            print(f"  • 最小值: {min_val:.2f}")
            print(f"  • 最大值: {max_val:.2f}")


def main():
    """主函数"""
    print("=" * 80)
    print("添加3个高优先级派生变量到 Enhanced CSV")
    print("=" * 80)

    # 文件路径
    input_file = Path('/home/green/energy_dl/nightly/results/summary_all_enhanced.csv')
    backup_file = input_file.with_suffix('.csv.backup_before_add_3cols')
    output_file = input_file

    # 检查输入文件
    if not input_file.exists():
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        sys.exit(1)

    print(f"\n输入文件: {input_file}")
    print(f"备份文件: {backup_file}")

    # 读取CSV
    print("\n【读取数据】")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"  • 原始行数: {len(rows)}")
    print(f"  • 原始列数: {len(fieldnames)}")

    # 验证数据完整性
    if not validate_data_completeness(rows):
        print("\n❌ 错误: 数据不完整，无法继续")
        sys.exit(1)

    # 计算派生列
    print("\n【计算派生列】")
    new_columns = ['energy_gpu_power_range', 'energy_gpu_percent', 'temperature_range']

    for i, row in enumerate(rows):
        derived = calculate_derived_columns(row)
        for col, val in derived.items():
            row[col] = val if val is not None else ''

        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i + 1}/{len(rows)}")

    print(f"  ✅ 完成 {len(rows)} 行数据计算")

    # 验证结果
    if not validate_results(rows):
        print("\n⚠️  警告: 验证发现问题，但继续写入文件")

    # 生成统计摘要
    generate_statistics(rows)

    # 更新列名
    new_fieldnames = list(fieldnames)
    for col in new_columns:
        if col not in new_fieldnames:
            new_fieldnames.append(col)

    print(f"\n【写入文件】")
    print(f"  • 新列数: {len(new_fieldnames)} (原始 {len(fieldnames)} + 新增 {len(new_columns)})")

    # 备份原文件
    print(f"\n  1. 创建备份...")
    import shutil
    shutil.copy2(input_file, backup_file)
    print(f"     ✅ 备份完成: {backup_file}")

    # 写入新文件
    print(f"\n  2. 写入增强数据...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"     ✅ 写入完成: {output_file}")

    # 最终验证
    print("\n【最终验证】")
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        final_fieldnames = reader.fieldnames
        final_rows = list(reader)

    print(f"  • 文件行数: {len(final_rows)} (预期 {len(rows)})")
    print(f"  • 文件列数: {len(final_fieldnames)} (预期 {len(new_fieldnames)})")

    # 检查新列是否存在
    for col in new_columns:
        if col in final_fieldnames:
            non_empty = sum(1 for row in final_rows if row.get(col, '').strip())
            print(f"  ✅ {col}: {non_empty}/{len(final_rows)} 行有数据")
        else:
            print(f"  ❌ {col}: 列不存在！")

    print("\n" + "=" * 80)
    print("✅ 处理完成！")
    print("=" * 80)
    print(f"\n新增的3个列:")
    print("  1. energy_gpu_power_range - GPU功率波动范围 (瓦特)")
    print("  2. energy_gpu_percent - GPU能耗占比 (%)")
    print("  3. temperature_range - GPU温度波动范围 (摄氏度)")
    print(f"\n输出文件: {output_file}")
    print(f"备份文件: {backup_file}")


if __name__ == '__main__':
    main()
