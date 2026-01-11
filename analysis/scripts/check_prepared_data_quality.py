#!/usr/bin/env python3
"""
检查准备好的分层数据质量
"""

import pandas as pd

# 分析每个CSV文件
groups = ['group2_vulberta', 'group3_person_reid', 'group6_resnet']

for group in groups:
    print('=' * 80)
    print(f'{group}')
    print('=' * 80)

    for mode in ['parallel', 'non_parallel']:
        mode_name = '并行' if mode == 'parallel' else '非并行'
        filepath = f'data/energy_research/dibs_training_{mode}/{group}.csv'

        try:
            df = pd.read_csv(filepath)
            print(f'\n{mode_name}模式: {len(df)} 样本, {len(df.columns)} 特征')

            # 检查NaN
            nan_counts = df.isna().sum()
            total_nan = nan_counts.sum()
            if total_nan > 0:
                print(f'  ❌ 有NaN: {total_nan} 个NaN值')
                nan_cols = nan_counts[nan_counts > 0]
                for col, count in nan_cols.items():
                    print(f'    {col}: {count} NaN')
            else:
                print('  ✅ 无NaN')

            # 检查零方差
            variance = df.var()
            zero_var = variance[variance == 0]
            if len(zero_var) > 0:
                print(f'  ❌ 零方差列: {list(zero_var.index)}')
            else:
                print('  ✅ 无零方差列')

            # 列类型
            hyperparam_cols = [col for col in df.columns if col.startswith('hyperparam_')]
            perf_cols = [col for col in df.columns if col.startswith('perf_')]
            energy_cols = [col for col in df.columns if col.startswith('energy_')]
            print(f'  列分布: {len(hyperparam_cols)}超参数 + {len(perf_cols)}性能 + {len(energy_cols)}能耗')

        except FileNotFoundError:
            print(f'\n{mode_name}模式: 文件不存在')

    print()
