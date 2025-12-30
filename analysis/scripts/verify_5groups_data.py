#!/usr/bin/env python3
"""验证新5组分层数据的质量和安全性

作者: Claude
日期: 2025-12-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def verify_5groups_data():
    """验证5组数据"""

    print("=" * 80)
    print("新5组分层数据质量验证")
    print("=" * 80)

    data_dir = Path('../data/energy_research/processed')

    # 预期值（来自OPTIMAL_GROUPING_STRATEGY_REPORT）
    expected = {
        'image_classification_examples': {'rows': 219, 'cols': 19},
        'image_classification_resnet': {'rows': 39, 'cols': 15},
        'person_reid': {'rows': 116, 'cols': 20},
        'vulberta': {'rows': 82, 'cols': 15},
        'bug_localization': {'rows': 80, 'cols': 16}
    }

    total_rows = 0
    all_passed = True

    for task_name, expect in expected.items():
        print(f"\n{'=' * 80}")
        print(f"验证任务组: {task_name}")
        print(f"{'=' * 80}")

        # 1. 读取数据
        filepath = data_dir / f'training_data_{task_name}.csv'
        df = pd.read_csv(filepath)

        print(f"\n1. 数据维度检查:")
        print(f"  实际行数: {len(df)} | 预期: {expect['rows']}")
        print(f"  实际列数: {len(df.columns)} | 预期: {expect['cols']}")

        if len(df) != expect['rows']:
            print(f"  ❌ 行数不匹配！")
            all_passed = False
        elif len(df.columns) != expect['cols']:
            print(f"  ❌ 列数不匹配！")
            all_passed = False
        else:
            print(f"  ✅ 维度正确")

        # 2. 空值检查
        print(f"\n2. 空值检查:")
        null_counts = df.isnull().sum()
        has_nulls = null_counts[null_counts > 0]

        if len(has_nulls) > 0:
            print(f"  ⚠️ 发现空值:")
            for col, count in has_nulls.items():
                print(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
            all_passed = False
        else:
            print(f"  ✅ 无空值（100%填充）")

        # 3. 相关矩阵检查
        print(f"\n3. 相关矩阵检查:")
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()

            if corr_matrix.isnull().any().any():
                print(f"  ❌ 相关矩阵包含NaN值")
                all_passed = False
            else:
                print(f"  ✅ 相关矩阵可计算（无NaN值）")
        except Exception as e:
            print(f"  ❌ 相关矩阵计算失败: {e}")
            all_passed = False

        # 4. 超参数检查
        print(f"\n4. 超参数完整性:")
        hyperparam_cols = [col for col in df.columns if 'hyperparam_' in col or col == 'training_duration']
        print(f"  超参数数量: {len(hyperparam_cols)}")
        print(f"  超参数列: {hyperparam_cols}")

        # 5. 性能指标检查
        print(f"\n5. 性能指标完整性:")
        perf_cols = [col for col in df.columns if 'perf_' in col]
        print(f"  性能指标数量: {len(perf_cols)}")
        print(f"  性能指标列: {perf_cols}")

        for col in perf_cols:
            missing = df[col].isnull().sum()
            if missing > 0:
                print(f"    ⚠️ {col}: {missing} 个缺失 ({missing/len(df)*100:.2f}%)")
                all_passed = False
            else:
                print(f"    ✅ {col}: 0 个缺失 (100%填充)")

        total_rows += len(df)

    # 总体结果
    print(f"\n{'=' * 80}")
    print("总体验证结果")
    print(f"{'=' * 80}")
    print(f"总行数: {total_rows} | 预期: 536")
    print(f"任务组数: 5")

    if total_rows == 536 and all_passed:
        print(f"\n✅ **所有验证通过！数据安全可靠！**")
        return True
    else:
        print(f"\n❌ **验证失败，请检查问题！**")
        return False


def compare_with_old_4groups():
    """对比新旧方案"""

    print(f"\n{'=' * 80}")
    print("新旧方案对比")
    print(f"{'=' * 80}")

    # 读取备份的4组数据统计
    backup_dir = Path('../data/energy_research/processed.backup_4groups_20251224')
    old_stats_file = backup_dir / 'stratified_data_stats.json'

    if old_stats_file.exists():
        with open(old_stats_file, 'r') as f:
            old_stats = json.load(f)

        old_total = sum(task['rows'] for task in old_stats)

        print(f"\n旧方案（4组）:")
        print(f"  总行数: {old_total}")
        for task in old_stats:
            print(f"  {task['task_name']}: {task['rows']} 行")

        print(f"\n新方案（5组）:")
        print(f"  总行数: 536")
        print(f"  image_classification_examples: 219 行")
        print(f"  image_classification_resnet: 39 行")
        print(f"  person_reid: 116 行")
        print(f"  vulberta: 82 行")
        print(f"  bug_localization: 80 行")

        improvement = 536 - old_total
        improvement_pct = improvement / old_total * 100

        print(f"\n改进:")
        print(f"  增加行数: {improvement} 行")
        print(f"  增加比例: {improvement_pct:.1f}%")

        if improvement == 164:
            print(f"  ✅ 增加量与预期完全一致（164行，+44%）")
    else:
        print(f"  ⚠️ 未找到旧方案统计文件")


if __name__ == '__main__':
    passed = verify_5groups_data()
    compare_with_old_4groups()

    if passed:
        print(f"\n{'=' * 80}")
        print("✅ 新5组数据已验证通过，可以安全使用！")
        print(f"{'=' * 80}")
