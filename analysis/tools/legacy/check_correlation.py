#!/usr/bin/env python3
"""
检查group6中perf_test_accuracy和perf_best_val_accuracy的相关性
"""

import pandas as pd
import numpy as np
import os

# 加载group6的全局标准化数据
data_file = "data/energy_research/6groups_global_std/group6_resnet_global_std.csv"
data_df = pd.read_csv(data_file)

print(f"数据行数: {len(data_df)}")
print(f"数据列数: {len(data_df.columns)}")

# 检查两个变量是否存在
if 'perf_test_accuracy' not in data_df.columns:
    print("❌ perf_test_accuracy 不存在")
else:
    print(f"perf_test_accuracy 存在，非空值: {data_df['perf_test_accuracy'].notna().sum()}")
    print(f"perf_test_accuracy 统计: mean={data_df['perf_test_accuracy'].mean():.4f}, std={data_df['perf_test_accuracy'].std():.4f}")

if 'perf_best_val_accuracy' not in data_df.columns:
    print("❌ perf_best_val_accuracy 不存在")
else:
    print(f"perf_best_val_accuracy 存在，非空值: {data_df['perf_best_val_accuracy'].notna().sum()}")
    print(f"perf_best_val_accuracy 统计: mean={data_df['perf_best_val_accuracy'].mean():.4f}, std={data_df['perf_best_val_accuracy'].std():.4f}")

# 计算相关性
if 'perf_test_accuracy' in data_df.columns and 'perf_best_val_accuracy' in data_df.columns:
    # 移除NaN值
    valid_data = data_df[['perf_test_accuracy', 'perf_best_val_accuracy']].dropna()
    print(f"\n有效数据行数: {len(valid_data)}")

    if len(valid_data) > 0:
        correlation = valid_data['perf_test_accuracy'].corr(valid_data['perf_best_val_accuracy'])
        print(f"相关性 (Pearson): {correlation:.6f}")

        # 检查是否高度相关
        if abs(correlation) > 0.95:
            print("⚠️  警告: 变量高度相关 (|r| > 0.95)")

        # 检查数据分布
        print(f"\nperf_test_accuracy 范围: [{valid_data['perf_test_accuracy'].min():.4f}, {valid_data['perf_test_accuracy'].max():.4f}]")
        print(f"perf_best_val_accuracy 范围: [{valid_data['perf_best_val_accuracy'].min():.4f}, {valid_data['perf_best_val_accuracy'].max():.4f}]")

        # 检查是否有极端值
        test_std = valid_data['perf_test_accuracy'].std()
        best_val_std = valid_data['perf_best_val_accuracy'].std()
        print(f"\n标准差: perf_test_accuracy={test_std:.4f}, perf_best_val_accuracy={best_val_std:.4f}")

        # 检查是否有超过3σ的极端值
        test_extreme = valid_data[np.abs(valid_data['perf_test_accuracy']) > 3]
        best_val_extreme = valid_data[np.abs(valid_data['perf_best_val_accuracy']) > 3]
        print(f"超过3σ的极端值: perf_test_accuracy={len(test_extreme)}, perf_best_val_accuracy={len(best_val_extreme)}")

        # 查看前几个值
        print(f"\n前5个值:")
        for i in range(min(5, len(valid_data))):
            print(f"  {i+1}: test={valid_data.iloc[i]['perf_test_accuracy']:.4f}, best_val={valid_data.iloc[i]['perf_best_val_accuracy']:.4f}")

# 检查其他组的ATE值范围
print(f"\n{'='*80}")
print("检查其他组的ATE值范围")
print(f"{'='*80}")

output_dir = "results/energy_research/data/global_std_dibs_ate"
for group_num in range(1, 7):
    group_name = f"group{group_num}"
    if group_num == 1:
        group_name = "group1_examples"
    elif group_num == 2:
        group_name = "group2_vulberta"
    elif group_num == 3:
        group_name = "group3_person_reid"
    elif group_num == 4:
        group_name = "group4_bug_localization"
    elif group_num == 5:
        group_name = "group5_mrt_oast"
    elif group_num == 6:
        group_name = "group6_resnet"

    file_path = os.path.join(output_dir, f"{group_name}_dibs_global_std_ate.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        ate_values = df['ate_global_std'].dropna()
        if len(ate_values) > 0:
            ate_abs_max = np.abs(ate_values).max()
            print(f"组{group_num}: {len(ate_values)}个ATE值, 最大绝对值={ate_abs_max:.4f}")
            if ate_abs_max > 10:
                print(f"  ⚠️  警告: 有极端ATE值 (>10)")
                extreme_edges = df[np.abs(df['ate_global_std']) > 10]
                for _, row in extreme_edges.iterrows():
                    print(f"    极端边: {row['source']}→{row['target']}, ATE={row['ate_global_std']:.4f}")