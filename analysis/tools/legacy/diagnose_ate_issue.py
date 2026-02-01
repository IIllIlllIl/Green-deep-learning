#!/usr/bin/env python3
"""
诊断ATE异常值问题的详细分析
"""

import pandas as pd
import numpy as np
import os

# 加载group6的全局标准化数据
data_file = "data/energy_research/6groups_global_std/group6_resnet_global_std.csv"
data_df = pd.read_csv(data_file)

print(f"数据诊断: group6_resnet")
print(f"=" * 80)
print(f"数据行数: {len(data_df)}")
print(f"数据列数: {len(data_df.columns)}")

# 分析两个变量的详细统计
treatment = 'perf_test_accuracy'
outcome = 'perf_best_val_accuracy'

print(f"\n变量分析:")
print(f"处理变量: {treatment}")
print(f"结果变量: {outcome}")

# 基本统计
print(f"\n{treatment} 统计:")
print(f"  均值: {data_df[treatment].mean():.6f}")
print(f"  标准差: {data_df[treatment].std():.6f}")
print(f"  最小值: {data_df[treatment].min():.6f}")
print(f"  最大值: {data_df[treatment].max():.6f}")
print(f"  范围: {data_df[treatment].max() - data_df[treatment].min():.6f}")
print(f"  变异系数 (std/mean): {data_df[treatment].std() / abs(data_df[treatment].mean()):.6f}")

print(f"\n{outcome} 统计:")
print(f"  均值: {data_df[outcome].mean():.6f}")
print(f"  标准差: {data_df[outcome].std():.6f}")
print(f"  最小值: {data_df[outcome].min():.6f}")
print(f"  最大值: {data_df[outcome].max():.6f}")

# 相关性分析
correlation = data_df[treatment].corr(data_df[outcome])
print(f"\n相关性分析:")
print(f"  Pearson相关系数: {correlation:.6f}")
print(f"  决定系数 (R²): {correlation**2:.6f}")

# 检查线性关系
print(f"\n线性关系检查:")
# 计算回归斜率 (简单线性回归)
cov = data_df[treatment].cov(data_df[outcome])
var_treatment = data_df[treatment].var()
slope = cov / var_treatment if var_treatment > 0 else 0
print(f"  回归斜率 (cov/var): {slope:.6f}")

# 计算ATE的理论值（基于回归斜率）
# 在全局标准化数据中，ATE表示treatment增加1个标准差时outcome的变化
# 但这里treatment的标准差很小，所以ATE会很大
treatment_std = data_df[treatment].std()
outcome_std = data_df[outcome].std()
print(f"\n标准化尺度分析:")
print(f"  treatment标准差: {treatment_std:.6f}")
print(f"  outcome标准差: {outcome_std:.6f}")
print(f"  标准差比率 (outcome/treatment): {outcome_std/treatment_std:.6f}")

# 计算理论ATE（基于回归）
# ATE = slope * (treatment的1个标准差变化)
theoretical_ate = slope * treatment_std
print(f"  理论ATE (基于回归): {theoretical_ate:.6f}")

# 但DML计算的ATE是全局标准化的ATE
# 在全局标准化数据中，ATE应该是：当treatment增加1个全局标准差时，outcome变化多少个全局标准差
# 实际上，由于数据已经全局标准化，treatment和outcome的全局标准差都是1
# 但这里treatment的样本标准差很小，导致问题

print(f"\n问题诊断:")
print(f"  1. 高度共线性: r={correlation:.6f} (>0.95)")
print(f"  2. treatment变异性极低: std={treatment_std:.6f} (接近0)")
print(f"  3. 回归斜率很大: {slope:.6f}")
print(f"  4. 理论ATE: {theoretical_ate:.6f}")

# 模拟DML可能遇到的问题
print(f"\nDML估计问题模拟:")
print(f"  当treatment变异性极低时，DML的model_t（预测treatment的模型）")
print(f"  可能无法准确估计treatment，导致ATE估计不稳定")

# 检查其他极端ATE
print(f"\n{'='*80}")
print(f"检查group4的极端ATE (energy_gpu_total_joules→energy_gpu_avg_watts)")
print(f"{'='*80}")

# 加载group4数据
group4_file = "data/energy_research/6groups_global_std/group4_bug_localization_global_std.csv"
if os.path.exists(group4_file):
    group4_df = pd.read_csv(group4_file)
    treatment_g4 = 'energy_gpu_total_joules'
    outcome_g4 = 'energy_gpu_avg_watts'

    if treatment_g4 in group4_df.columns and outcome_g4 in group4_df.columns:
        corr_g4 = group4_df[treatment_g4].corr(group4_df[outcome_g4])
        std_t_g4 = group4_df[treatment_g4].std()
        std_o_g4 = group4_df[outcome_g4].std()

        print(f"  相关性: {corr_g4:.6f}")
        print(f"  treatment标准差: {std_t_g4:.6f}")
        print(f"  outcome标准差: {std_o_g4:.6f}")

        # 计算回归斜率
        cov_g4 = group4_df[treatment_g4].cov(group4_df[outcome_g4])
        var_t_g4 = group4_df[treatment_g4].var()
        slope_g4 = cov_g4 / var_t_g4 if var_t_g4 > 0 else 0
        print(f"  回归斜率: {slope_g4:.6f}")
        print(f"  理论ATE: {slope_g4 * std_t_g4:.6f}")

# 建议的修复方案
print(f"\n{'='*80}")
print(f"修复方案建议")
print(f"{'='*80}")

print(f"1. 在estimate_ate方法中添加合理性检查:")
print(f"   - 检查|ATE| > 10的极端值")
print(f"   - 对于极端值，使用_simple_ate_estimate作为后备")
print(f"   - 记录警告信息")

print(f"\n2. 数据预处理检查:")
print(f"   - 检查变量相关性 > 0.95的情况")
print(f"   - 检查treatment变异性 (std < 0.1)")
print(f"   - 对于问题变量对，使用简化估计")

print(f"\n3. 阈值选择依据:")
print(f"   - 全局标准化数据，正常ATE应在[-3, 3]范围内")
print(f"   - 设置安全阈值|ATE| > 10")
print(f"   - 允许合理变异但过滤极端值")

print(f"\n4. 简化ATE估计的适用性:")
print(f"   - _simple_ate_estimate适用于全局标准化数据")
print(f"   - 它计算分组均值差异，不依赖DML")
print(f"   - 虽然可能忽略混淆因素，但比极端ATE更可靠")

# 计算简化ATE作为对比
print(f"\n{'='*80}")
print(f"简化ATE估计结果")
print(f"{'='*80}")

# 模拟_simple_ate_estimate
median = data_df[treatment].median()
low_group = data_df[data_df[treatment] <= median]
high_group = data_df[data_df[treatment] > median]

mean_low = low_group[outcome].mean()
mean_high = high_group[outcome].mean()
simple_ate = mean_high - mean_low

print(f"  中位数分组:")
print(f"    low组: n={len(low_group)}, mean={mean_low:.6f}")
print(f"    high组: n={len(high_group)}, mean={mean_high:.6f}")
print(f"  简化ATE: {simple_ate:.6f}")

# 与DML的ATE比较
dml_ate = 50.08010233862722
print(f"\n  DML ATE: {dml_ate:.6f}")
print(f"  简化ATE: {simple_ate:.6f}")
print(f"  差异: {abs(dml_ate - simple_ate):.6f}")
print(f"  简化ATE/DML ATE: {simple_ate/dml_ate:.6%}")