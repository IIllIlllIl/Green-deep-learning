#!/usr/bin/env python3
"""
重新计算raw_data.csv中的num_mutated_params和mutated_param字段

使用修复后的逻辑：比较实验值与默认值，而不是简单计数非空参数

输出：
- raw_data_fixed.csv: 修正后的数据
- 统计报告：展示修正前后的差异

作者: Claude Code
日期: 2025-12-12
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

# 导入修复后的计算函数
sys.path.insert(0, str(Path(__file__).parent))
from calculate_num_mutated_params_fixed import (
    load_models_config,
    calculate_num_mutated_params_fixed
)


def recalculate_raw_data():
    """重新计算raw_data.csv中的num_mutated_params"""
    print("=" * 80)
    print("重新计算raw_data.csv中的num_mutated_params字段")
    print("=" * 80)
    print()

    # 1. 加载models_config
    print("步骤1: 加载models_config.json...")
    try:
        models_config = load_models_config()
        print("  ✓ 加载成功")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return False

    # 2. 读取raw_data.csv
    print("\n步骤2: 读取raw_data.csv...")
    csv_path = Path('results/raw_data.csv')

    if not csv_path.exists():
        print(f"  ✗ 文件不存在: {csv_path}")
        return False

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    print(f"  ✓ 读取 {len(rows)} 行数据")

    # 3. 重新计算每行的num_mutated_params
    print("\n步骤3: 重新计算num_mutated_params...")

    changes = []
    unchanged = 0
    errors = 0

    for i, row in enumerate(rows):
        old_num = row.get('num_mutated_params', '')
        old_param = row.get('mutated_param', '')

        try:
            new_num, new_param = calculate_num_mutated_params_fixed(row, models_config)

            row['num_mutated_params'] = str(new_num)
            row['mutated_param'] = new_param

            if str(old_num) != str(new_num) or old_param != new_param:
                changes.append({
                    'row': i + 1,
                    'exp_id': row.get('experiment_id', ''),
                    'model': f"{row.get('repository', '')}/{row.get('model', '')}",
                    'old_num': old_num,
                    'new_num': new_num,
                    'old_param': old_param,
                    'new_param': new_param
                })
            else:
                unchanged += 1

        except Exception as e:
            print(f"  ✗ 处理失败 (行{i+1}): {e}")
            errors += 1

    print(f"  ✓ 处理完成")
    print(f"    变化: {len(changes)} 行")
    print(f"    未变化: {unchanged} 行")
    print(f"    错误: {errors} 行")

    # 4. 写入新文件
    print("\n步骤4: 写入修正后的数据...")
    output_path = Path('results/raw_data_fixed.csv')

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ 写入完成: {output_path}")

    # 5. 生成统计报告
    print("\n步骤5: 生成统计报告...")
    print("\n" + "=" * 80)
    print("修正统计")
    print("=" * 80)

    # 按变化类型分组
    by_change_type = defaultdict(list)
    for change in changes:
        change_key = f"{change['old_num']} → {change['new_num']}"
        by_change_type[change_key].append(change)

    print(f"\n总修正数: {len(changes)} 行")
    print("\n按变化类型分组:")
    for change_type, items in sorted(by_change_type.items(), key=lambda x: -len(x[1])):
        print(f"\n  {change_type}: {len(items)} 行")
        # 显示前3个例子
        for item in items[:3]:
            print(f"    - {item['exp_id']} ({item['model']})")
            print(f"      mutated_param: '{item['old_param']}' → '{item['new_param']}'")
        if len(items) > 3:
            print(f"    ... 还有 {len(items) - 3} 行")

    # 统计新的num_mutated_params分布
    print("\n" + "=" * 80)
    print("修正后的num_mutated_params分布")
    print("=" * 80)

    new_distribution = defaultdict(int)
    for row in rows:
        num = row.get('num_mutated_params', '')
        new_distribution[num] += 1

    print(f"\n{'num_mutated_params':<20} {'数量':<10} {'百分比':<10}")
    print("-" * 40)
    total = len(rows)
    for num in sorted(new_distribution.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        count = new_distribution[num]
        pct = count / total * 100
        print(f"{num:<20} {count:<10} {pct:>6.1f}%")

    print("\n" + "=" * 80)
    print("✅ 重新计算完成")
    print("=" * 80)
    print(f"\n输出文件: {output_path}")
    print(f"\n关键发现:")

    default_count = new_distribution.get('0', 0)
    single_mut = new_distribution.get('1', 0)
    multi_mut = sum(new_distribution.get(str(i), 0) for i in range(2, 10))

    print(f"  - 默认值实验 (num=0): {default_count} 个 ({default_count/total*100:.1f}%)")
    print(f"  - 单参数变异 (num=1): {single_mut} 个 ({single_mut/total*100:.1f}%)")
    print(f"  - 多参数变异 (num≥2): {multi_mut} 个 ({multi_mut/total*100:.1f}%)")

    print(f"\n下一步:")
    print("  1. 检查raw_data_fixed.csv的准确性")
    print("  2. 如果验证通过，替换原文件:")
    print("     cp results/raw_data.csv results/raw_data.csv.backup_before_fix")
    print("     mv results/raw_data_fixed.csv results/raw_data.csv")
    print("  3. 重新运行完成度分析:")
    print("     python3 scripts/analyze_experiment_completion.py")

    return True


if __name__ == '__main__':
    success = recalculate_raw_data()
    exit(0 if success else 1)
