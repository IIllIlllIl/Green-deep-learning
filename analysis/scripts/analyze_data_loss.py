#!/usr/bin/env python3
"""
分析分层数据损失的详细原因
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

# 加载原始数据
df = pd.read_csv('../data/raw_data.csv')

print('=' * 80)
print('原始数据统计')
print('=' * 80)
print(f'总样本数: {len(df)}')
print(f'并行模式样本数: {df["mode"].eq("parallel").sum()}')
print(f'非并行模式样本数（NaN）: {df["mode"].isna().sum()}')
print()
print('mode列取值分布:')
print(df['mode'].value_counts(dropna=False))
print()

# 检查每个组的分布
groups = {
    'group2_vulberta': (['VulBERTa'], ['mlp', 'cnn']),
    'group3_person_reid': (['Person_reID_baseline_pytorch'], ['AGW', 'PCB']),
    'group6_resnet': (['pytorch_resnet_cifar10'], ['resnet56'])
}

for group_name, (repos, models) in groups.items():
    print('=' * 80)
    print(f'{group_name}')
    print('=' * 80)

    for mode in ['parallel', 'non_parallel']:
        mode_name = '并行' if mode == 'parallel' else '非并行'

        # 筛选数据
        if mode == 'parallel':
            mode_df = df[df['mode'] == 'parallel']
        else:  # non_parallel
            mode_df = df[df['mode'].isna()]

        if mode == 'parallel':
            group_mask = (
                (mode_df['fg_repository'].isin(repos)) |
                (mode_df['bg_repository'].isin(repos)) |
                (mode_df['repository'].isin(repos))
            )
        else:
            group_mask = mode_df['repository'].isin(repos)

        group_df = mode_df[group_mask].copy()

        print(f'\n{mode_name}模式:')
        print(f'  初始样本数: {len(group_df)}')

        if len(group_df) == 0:
            continue

        # 检查能耗数据完整性
        energy_cols = [col for col in group_df.columns if col.startswith('energy_')]
        energy_complete = ~group_df[energy_cols].isna().any(axis=1)
        energy_missing = (~energy_complete).sum()
        print(f'  能耗数据不完整: {energy_missing} ({energy_missing/len(group_df)*100:.1f}%)')
        print(f'  能耗数据完整: {energy_complete.sum()} ({energy_complete.sum()/len(group_df)*100:.1f}%)')

        # 模拟清洗过程
        # 步骤1: 选择特征列
        feature_cols = []

        # 超参数
        hyperparam_cols = [col for col in group_df.columns if col.startswith('hyperparam_')]
        feature_cols.extend(hyperparam_cols)

        # 性能指标
        perf_cols = [col for col in group_df.columns if col.startswith('perf_')]
        feature_cols.extend(perf_cols)

        # 能耗指标
        feature_cols.extend(energy_cols)

        # 步骤2: 删除缺失率>90%的列（默认值回填后）
        kept_cols = []
        for col in feature_cols:
            if col in group_df.columns:
                missing_rate = group_df[col].isna().sum() / len(group_df)
                if missing_rate < 0.90:
                    kept_cols.append(col)

        print(f'  特征选择后保留列数: {len(kept_cols)}')

        # 步骤3: 删除方差为0的列
        df_subset = group_df[kept_cols].copy()
        variance_zero = (df_subset.var() == 0) | df_subset.var().isna()
        cols_after_var = [col for col, is_zero in variance_zero.items() if not is_zero]
        print(f'  删除零方差列后: {len(cols_after_var)}')

        # 步骤4: 删除有任何NaN的行
        df_cleaned = df_subset[cols_after_var].dropna()
        print(f'  删除NaN行后: {len(df_cleaned)} ({len(df_cleaned)/len(group_df)*100:.1f}%)')

        # 分析损失原因
        print(f'\n  损失分析:')
        print(f'    能耗不完整导致的损失: {energy_missing}')

        # 检查超参数NaN情况（回填后仍有NaN的）
        for col in hyperparam_cols[:5]:
            if col in cols_after_var:
                nan_count = df_subset[col].isna().sum()
                if nan_count > 0:
                    print(f'    {col} NaN: {nan_count}')

        print()
