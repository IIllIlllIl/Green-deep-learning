#!/usr/bin/env python3
"""
全局标准化数据生成脚本

目的：实施全局标准化，恢复跨组可比性

与之前（组内标准化）的关键区别：
1. 合并所有6组数据后，使用全局均值/标准差进行标准化
2. 保留结构性NaN（某些组特有的超参数/性能指标）
3. hyperparam_seed用特殊值(-1)填充，表示"未设置seed"
4. 所有组使用相同的尺度，ATE可以跨组比较

输出：
- data/energy_research/6groups_global_std/ 目录
- 每个组的全局标准化数据
- 全局标准化参数（用于反标准化和解释）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def create_global_standardized_data():
    """生成全局标准化数据集"""

    print("=" * 80)
    print("全局标准化数据生成")
    print("=" * 80)

    # 路径配置
    final_dir = Path('data/energy_research/6groups_final/')
    output_dir = Path('data/energy_research/6groups_global_std/')
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        'group1_examples',
        'group2_vulberta',
        'group3_person_reid',
        'group4_bug_localization',
        'group5_mrt_oast',
        'group6_resnet'
    ]

    # 1. 读取并合并所有数据
    print('\n步骤1: 读取并合并所有6组数据')
    print('-' * 40)

    all_data = []
    for group in groups:
        file = final_dir / f'{group}.csv'
        df = pd.read_csv(file)
        df['_group'] = group
        all_data.append(df)
        print(f'  {group}: {len(df)} 行, {len(df.columns)} 列')

    merged = pd.concat(all_data, ignore_index=True)
    print(f'\n合并后总样本: {len(merged)}')
    print(f'合并后总列数: {len(merged.columns)}')

    # 2. 处理 hyperparam_seed 缺失值
    print('\n步骤2: 处理 hyperparam_seed 缺失值')
    print('-' * 40)

    if 'hyperparam_seed' in merged.columns:
        seed_missing_before = merged['hyperparam_seed'].isna().sum()
        print(f'  缺失seed数: {seed_missing_before}/{len(merged)} ({seed_missing_before/len(merged)*100:.1f}%)')

        # 用-1填充缺失的seed，表示"未设置seed"
        merged['hyperparam_seed'].fillna(-1, inplace=True)
        print(f'  填充策略: 用-1填充缺失值（表示"未设置seed"）')
        print(f'  填充后缺失数: {merged["hyperparam_seed"].isna().sum()}')

    # 3. 确定需要标准化的列
    print('\n步骤3: 确定标准化列')
    print('-' * 40)

    # 不需要标准化的列
    no_std_cols = ['_group', 'timestamp', 'is_parallel']

    # 模型标识列（不标准化）
    model_cols = [col for col in merged.columns if col.startswith('model_')]

    # 需要标准化的列
    energy_cols = [col for col in merged.columns if 'energy' in col.lower()]
    hyperparam_cols = [col for col in merged.columns if col.startswith('hyperparam_')]
    perf_cols = [col for col in merged.columns if col.startswith('perf_')]

    cols_to_standardize = energy_cols + hyperparam_cols + perf_cols
    cols_to_standardize = [col for col in cols_to_standardize if col not in no_std_cols]

    print(f'  能耗列: {len(energy_cols)} 个')
    print(f'  超参数列: {len(hyperparam_cols)} 个')
    print(f'  性能指标列: {len(perf_cols)} 个')
    print(f'  总计标准化列: {len(cols_to_standardize)} 个')

    # 4. 全局标准化（保留结构性NaN）
    print('\n步骤4: 执行全局标准化')
    print('-' * 40)

    std_data = merged.copy()
    standardization_params = {}

    for col in cols_to_standardize:
        if col in std_data.columns:
            # 获取非NaN值
            valid_values = std_data[col].dropna()

            if len(valid_values) > 1:  # 至少需要2个值才能计算标准差
                # 计算全局统计量
                global_mean = float(valid_values.mean())
                global_std = float(valid_values.std())

                # 避免除零
                if global_std > 0:
                    # 标准化非NaN值
                    mask = std_data[col].notna()
                    std_data.loc[mask, col] = (std_data.loc[mask, col] - global_mean) / global_std

                    standardization_params[col] = {
                        'mean': global_mean,
                        'std': global_std,
                        'n_valid': int(len(valid_values)),
                        'n_nan': int(std_data[col].isna().sum()),
                        'min_original': float(valid_values.min()),
                        'max_original': float(valid_values.max())
                    }

                    print(f'  {col}: mean={global_mean:.4f}, std={global_std:.4f}, '
                          f'valid={len(valid_values)}, nan={int(std_data[col].isna().sum())}')

    # 5. 重建交互项
    print('\n步骤5: 重建交互项')
    print('-' * 40)

    # 获取超参数列（排除seed，因为seed × is_parallel 没有物理意义）
    interaction_hyperparams = [col for col in hyperparam_cols
                                if col != 'hyperparam_seed' and col in std_data.columns]

    interaction_count = 0
    for hp in interaction_hyperparams:
        interaction_col = f'{hp}_x_is_parallel'
        # 只在两个变量都存在时创建交互项
        if hp in std_data.columns and 'is_parallel' in std_data.columns:
            std_data[interaction_col] = std_data[hp] * std_data['is_parallel']
            interaction_count += 1

            # 统计交互项的有效值数
            valid_count = std_data[interaction_col].notna().sum()
            print(f'  {interaction_col}: {valid_count}/{len(std_data)} 有效值')

    print(f'\n创建交互项: {interaction_count} 个')

    # 6. 保存全局标准化参数
    print('\n步骤6: 保存全局标准化参数')
    print('-' * 40)

    params_file = output_dir / 'global_standardization_params.json'
    with open(params_file, 'w') as f:
        json.dump(standardization_params, f, indent=2)

    print(f'  保存至: {params_file}')

    # 7. 分组保存数据
    print('\n步骤7: 分组保存全局标准化数据')
    print('-' * 40)

    for group in groups:
        # 筛选该组数据
        group_data = std_data[std_data['_group'] == group].copy()
        group_data = group_data.drop(columns=['_group'])

        # 保存
        output_file = output_dir / f'{group}_global_std.csv'
        group_data.to_csv(output_file, index=False)

        print(f'  {group}: {len(group_data)} 行 → {output_file}')

    # 8. 生成汇总报告
    print('\n步骤8: 生成汇总报告')
    print('-' * 40)

    report = {
        'generation_date': datetime.now().isoformat(),
        'total_samples': int(len(std_data)),
        'groups': groups,
        'standardization_strategy': 'Global standardization with structural NaNs preserved',
        'key_differences_from_previous': [
            '全局标准化：所有组使用相同的均值和标准差',
            '保留结构性NaN：组特有的超参数/性能指标保持为NaN',
            'hyperparam_seed用-1填充，表示"未设置seed"',
            'ATE跨组可比：相同的尺度',
            '交互项自然继承NaN'
        ],
        'standardization_params': {
            'total_columns_standardized': len(standardization_params),
            'columns': list(standardization_params.keys())
        },
        'interaction_terms': {
            'total_created': interaction_count,
            'hyperparams_used': [hp.replace('hyperparam_', '') for hp in interaction_hyperparams]
        },
        'samples_per_group': {}
    }

    for group in groups:
        group_data = std_data[std_data['_group'] == group]
        report['samples_per_group'][group] = int(len(group_data))

    report_file = output_dir / 'generation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'  保存至: {report_file}')

    # 9. 对比全局vs组内标准化（energy_gpu_avg_watts为例）
    print('\n步骤9: 对比全局vs组内标准化')
    print('-' * 40)

    if 'energy_gpu_avg_watts' in standardization_params:
        global_std = standardization_params['energy_gpu_avg_watts']['std']
        print(f'\nenergy_gpu_avg_watts:')
        print(f'  全局标准差: {global_std:.2f} watts')
        print(f'  → 所有组的ATE都表示"变化{global_std:.2f} watts"的效应')
        print(f'  → 跨组可直接比较')

    # 读取旧的组内标准化参数进行对比
    old_params_file = final_dir.parent / '6groups_interaction' / 'standardization_params.json'
    if old_params_file.exists():
        print(f'\n组内标准化（之前）:')
        with open(old_params_file, 'r') as f:
            old_params = json.load(f)

        for group_key in old_params:
            if 'energy_gpu_avg_watts' in group_key:
                group_std = old_params[group_key]['std'][3]  # energy_gpu_avg_watts是第4个能耗列
                group_name = group_key.replace('_energy', '')
                print(f'  {group_name}: {group_std:.2f} watts')

        print(f'\n差异: 组内标准差范围 8.10-73.09 watts → 全局统一标准差')

    print('\n' + '=' * 80)
    print('全局标准化数据生成完成！')
    print('=' * 80)
    print(f'\n输出目录: {output_dir}')
    print(f'主要文件:')
    print(f'  - global_standardization_params.json')
    print(f'  - generation_report.json')
    print(f'  - [group]_global_std.csv (6个文件)')

    return std_data, standardization_params, output_dir


if __name__ == '__main__':
    create_global_standardized_data()
