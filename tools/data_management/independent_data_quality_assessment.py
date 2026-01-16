#!/usr/bin/env python3
"""
Independent Data Quality Assessment for raw_data.csv
完全独立的数据质量评估脚本
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import json

# 读取数据
csv_path = '/home/green/energy_dl/nightly/data/raw_data.csv'
df = pd.read_csv(csv_path)

print("=" * 80)
print("独立数据质量评估报告")
print("=" * 80)
print(f"\n数据文件: {csv_path}")
print(f"分析时间: 2026-01-14\n")

# ============================================================
# 1. 基本统计信息
# ============================================================
print("\n" + "=" * 80)
print("1. 基本数据统计")
print("=" * 80)

total_records = len(df)
total_columns = len(df.columns)

print(f"\n总记录数（含header）: {total_records + 1}")
print(f"实际数据记录数: {total_records}")
print(f"总列数: {total_columns}")

# 显示列名
print(f"\n列名列表（前20个）:")
for i, col in enumerate(df.columns[:20], 1):
    print(f"  {i:2d}. {col}")
if total_columns > 20:
    print(f"  ... (还有 {total_columns - 20} 列)")

# ============================================================
# 2. 关键字段缺失情况分析
# ============================================================
print("\n" + "=" * 80)
print("2. 关键字段缺失情况分析")
print("=" * 80)

# 定义关键字段
key_fields = {
    '模型标识': ['repository', 'model'],
    '训练状态': ['training_success'],
    '能耗数据': ['energy_cpu_total_joules', 'energy_gpu_total_joules'],
    '性能指标': [col for col in df.columns if col.startswith('perf_')]
}

# 分析每个类别的缺失情况
print("\n关键字段缺失统计:")
print(f"\n{'字段类别':<15} {'字段名':<35} {'非空数':<10} {'缺失数':<10} {'缺失率':<10}")
print("-" * 80)

for category, fields in key_fields.items():
    for field in fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            null_count = df[field].isna().sum()
            null_rate = (null_count / total_records) * 100
            print(f"{category:<15} {field:<35} {non_null:<10} {null_count:<10} {null_rate:>6.2f}%")
        else:
            print(f"{category:<15} {field:<35} {'字段不存在'}")

# ============================================================
# 3. 模型分布分析
# ============================================================
print("\n" + "=" * 80)
print("3. 模型分布分析")
print("=" * 80)

# 创建复合模型标识
df['model_id'] = df['repository'].fillna('/') + '/' + df['model'].fillna('')

# 统计每个模型的记录数
model_counts = df['model_id'].value_counts()

print(f"\n总模型数: {len(model_counts)}")
print(f"\n模型记录数分布（Top 15）:")
print(f"{'模型':<50} {'记录数':<10} {'占比':<10}")
print("-" * 70)
for model, count in model_counts.head(15).items():
    percentage = (count / total_records) * 100
    print(f"{model:<50} {count:<10} {percentage:>6.2f}%")

# ============================================================
# 4. 训练成功率分析
# ============================================================
print("\n" + "=" * 80)
print("4. 训练成功率分析")
print("=" * 80)

# 处理 training_success 字段
# 只查看非空记录
valid_training_records = df[df['training_success'].notna()]
fg_valid_training = df[df['fg_training_success'].notna()]

print(f"\n前台训练记录:")
print(f"  总记录数: {len(valid_training_records)}")
if len(valid_training_records) > 0:
    success_count = (valid_training_records['training_success'] == True).sum() + \
                   (valid_training_records['training_success'] == 'True').sum()
    print(f"  训练成功: {success_count} ({success_count/len(valid_training_records)*100:.2f}%)")
    print(f"  训练失败: {len(valid_training_records) - success_count} ({(len(valid_training_records) - success_count)/len(valid_training_records)*100:.2f}%)")

print(f"\n后台训练记录 (fg_*):")
print(f"  总记录数: {len(fg_valid_training)}")
if len(fg_valid_training) > 0:
    fg_success = (fg_valid_training['fg_training_success'] == True).sum() + \
                 (fg_valid_training['fg_training_success'] == 'True').sum()
    print(f"  训练成功: {fg_success} ({fg_success/len(fg_valid_training)*100:.2f}%)")

# ============================================================
# 5. 能耗数据完整性分析
# ============================================================
print("\n" + "=" * 80)
print("5. 能耗数据完整性分析")
print("=" * 80)

# 定义有能耗数据的条件
def has_energy_data(row):
    """检查是否有有效的能耗数据"""
    cpu_valid = pd.notna(row['energy_cpu_total_joules']) and \
                row['energy_cpu_total_joules'] not in ['', 'N/A', 'NA']
    gpu_valid = pd.notna(row['energy_gpu_total_joules']) and \
                row['energy_gpu_total_joules'] not in ['', 'N/A', 'NA']
    return cpu_valid or gpu_valid

# 应用到前台和后台数据
df['has_energy'] = df.apply(has_energy_data, axis=1)

# 前台能耗数据
foreground_records = df[df['repository'].notna()]
fg_with_energy = foreground_records[foreground_records['has_energy']].shape[0]

print(f"\n前台训练能耗数据:")
print(f"  总记录数: {len(foreground_records)}")
print(f"  有能耗数据: {fg_with_energy} ({fg_with_energy/len(foreground_records)*100:.2f}%)")
print(f"  缺失能耗: {len(foreground_records) - fg_with_energy} ({(len(foreground_records) - fg_with_energy)/len(foreground_records)*100:.2f}%)")

# 按模型分析能耗数据缺失
print(f"\n按模型统计能耗数据完整性（Top 10缺失最多）:")
model_energy_stats = foreground_records.groupby('model_id').agg({
    'has_energy': ['count', 'sum']
}).round(2)
model_energy_stats.columns = ['总数', '有能耗']
model_energy_stats['缺失'] = model_energy_stats['总数'] - model_energy_stats['有能耗']
model_energy_stats['缺失率%'] = (model_energy_stats['缺失'] / model_energy_stats['总数'] * 100).round(2)
model_energy_stats = model_energy_stats.sort_values('缺失', ascending=False)

print(f"\n{'模型':<50} {'总数':<8} {'有能耗':<8} {'缺失':<8} {'缺失率':<10}")
print("-" * 84)
for model, row in model_energy_stats.head(10).iterrows():
    print(f"{model:<50} {int(row['总数']):<8} {int(row['有能耗']):<8} {int(row['缺失']):<8} {row['缺失率%']:>6.2f}%")

# ============================================================
# 6. 性能指标完整性分析
# ============================================================
print("\n" + "=" * 80)
print("6. 性能指标完整性分析")
print("=" * 80)

# 获取所有性能指标列
perf_cols = [col for col in df.columns if col.startswith('perf_')]
print(f"\n性能指标字段总数: {len(perf_cols)}")
print(f"性能指标字段列表: {', '.join(perf_cols[:10])}" + (f" ... (还有{len(perf_cols)-10}个)" if len(perf_cols) > 10 else ""))

# 定义有性能指标的条件
def has_performance_metrics(row):
    """检查是否有至少一个有效的性能指标"""
    for col in perf_cols:
        if pd.notna(row[col]) and row[col] not in ['', 'N/A', 'NA']:
            return True
    return False

df['has_perf'] = df.apply(has_performance_metrics, axis=1)

# 统计性能指标完整性
fg_with_perf = foreground_records[foreground_records.apply(has_performance_metrics, axis=1)].shape[0]

print(f"\n前台训练性能指标:")
print(f"  总记录数: {len(foreground_records)}")
print(f"  有性能指标: {fg_with_perf} ({fg_with_perf/len(foreground_records)*100:.2f}%)")
print(f"  缺失性能指标: {len(foreground_records) - fg_with_perf} ({(len(foreground_records) - fg_with_perf)/len(foreground_records)*100:.2f}%)")

# 按模型分析性能指标缺失
print(f"\n按模型统计性能指标完整性（Top 10缺失最多）:")
foreground_records['has_perf'] = foreground_records.apply(has_performance_metrics, axis=1)
model_perf_stats = foreground_records.groupby('model_id').agg({
    'has_perf': ['count', 'sum']
}).round(2)
model_perf_stats.columns = ['总数', '有性能指标']
model_perf_stats['缺失'] = model_perf_stats['总数'] - model_perf_stats['有性能指标']
model_perf_stats['缺失率%'] = (model_perf_stats['缺失'] / model_perf_stats['总数'] * 100).round(2)
model_perf_stats = model_perf_stats.sort_values('缺失', ascending=False)

print(f"\n{'模型':<50} {'总数':<8} {'有指标':<8} {'缺失':<8} {'缺失率':<10}")
print("-" * 84)
for model, row in model_perf_stats.head(10).iterrows():
    print(f"{model:<50} {int(row['总数']):<8} {int(row['有性能指标']):<8} {int(row['缺失']):<8} {row['缺失率%']:>6.2f}%")

# ============================================================
# 7. 数据可用性综合分析
# ============================================================
print("\n" + "=" * 80)
print("7. 数据可用性综合分析")
print("=" * 80)

# 定义"可用记录"标准
def is_usable_record(row):
    """
    可用记录标准：
    1. training_success = True
    2. 有能耗数据（CPU或GPU至少一个）
    3. 有性能指标（至少一个）
    """
    # 检查训练成功
    training_success = row['training_success'] in [True, 'True']

    # 检查能耗数据
    has_energy = has_energy_data(row)

    # 检查性能指标
    has_perf = has_performance_metrics(row)

    return training_success and has_energy and has_perf

# 应用可用性判断
foreground_records['is_usable'] = foreground_records.apply(is_usable_record, axis=1)

usable_count = foreground_records['is_usable'].sum()
unusable_count = len(foreground_records) - usable_count

print(f"\n数据可用性总览:")
print(f"  总前台记录数: {len(foreground_records)}")
print(f"  ✅ 完全可用: {usable_count} ({usable_count/len(foreground_records)*100:.2f}%)")
print(f"  ❌ 不可用: {unusable_count} ({unusable_count/len(foreground_records)*100:.2f}%)")

# 分析不可用原因
print(f"\n不可用原因详细分析:")

# 准备分类统计
training_failed = foreground_records[~foreground_records['training_success'].isin([True, 'True'])].shape[0]
no_energy = foreground_records[
    foreground_records['training_success'].isin([True, 'True']) &
    ~foreground_records.apply(has_energy_data, axis=1)
].shape[0]
no_perf = foreground_records[
    foreground_records['training_success'].isin([True, 'True']) &
    ~foreground_records.apply(has_performance_metrics, axis=1)
].shape[0]

# 组合问题统计
training_success_records = foreground_records[foreground_records['training_success'].isin([True, 'True'])]
has_energy_col = training_success_records.apply(has_energy_data, axis=1)
has_perf_col = training_success_records.apply(has_performance_metrics, axis=1)

no_energy_only = ((~has_energy_col) & has_perf_col).sum()
no_perf_only = (has_energy_col & (~has_perf_col)).sum()
no_both = ((~has_energy_col) & (~has_perf_col)).sum()

print(f"  训练失败: {training_failed} ({training_failed/len(foreground_records)*100:.2f}%)")
print(f"  缺失能耗数据（训练成功）: {no_energy} ({no_energy/len(foreground_records)*100:.2f}%)")
print(f"  缺失性能指标（训练成功）: {no_perf} ({no_perf/len(foreground_records)*100:.2f}%)")
print(f"\n  组合问题分析（训练成功的记录）:")
print(f"    仅缺能耗: {no_energy_only} ({no_energy_only/len(training_success_records)*100:.2f}%)")
print(f"    仅缺性能指标: {no_perf_only} ({no_perf_only/len(training_success_records)*100:.2f}%)")
print(f"    能耗和性能指标都缺: {no_both} ({no_both/len(training_success_records)*100:.2f}%)")

# ============================================================
# 8. 按模型可用性分析
# ============================================================
print("\n" + "=" * 80)
print("8. 按模型可用性分析")
print("=" * 80)

model_usability = foreground_records.groupby('model_id').agg({
    'is_usable': ['count', 'sum']
}).round(2)
model_usability.columns = ['总数', '可用数']
model_usability['不可用'] = model_usability['总数'] - model_usability['可用数']
model_usability['可用率%'] = (model_usability['可用数'] / model_usability['总数'] * 100).round(2)
model_usability = model_usability.sort_values('总数', ascending=False)

print(f"\n模型可用性统计（按记录数排序）:")
print(f"\n{'模型':<50} {'总数':<8} {'可用':<8} {'不可用':<8} {'可用率':<10}")
print("-" * 84)
for model, row in model_usability.iterrows():
    print(f"{model:<50} {int(row['总数']):<8} {int(row['可用数']):<8} {int(row['不可用']):<8} {row['可用率%']:>6.2f}%")

# 高质量模型（100%可用率）
high_quality_models = model_usability[model_usability['可用率%'] == 100.0]
print(f"\n✅ 高质量模型（100%可用率）: {len(high_quality_models)}个")
if len(high_quality_models) > 0:
    print(f"   总记录数: {int(high_quality_models['总数'].sum())}")
    for model, row in high_quality_models.iterrows():
        print(f"   - {model}: {int(row['总数'])}条")

# ============================================================
# 9. 异常数据识别
# ============================================================
print("\n" + "=" * 80)
print("9. 异常数据识别")
print("=" * 80)

# 识别异常模式
print(f"\n异常模式检测:")

# 1. 空模型名
empty_model = df[(df['repository'].isna()) | (df['model'].isna()) |
                 (df['repository'] == '/') | (df['model'] == '')].shape[0]
print(f"  1. 空模型名或'/'模型: {empty_model}条")

# 2. 训练成功但无能耗数据
success_no_energy = foreground_records[
    foreground_records['training_success'].isin([True, 'True']) &
    ~foreground_records.apply(has_energy_data, axis=1)
].shape[0]
print(f"  2. 训练成功但无能耗数据: {success_no_energy}条")

# 3. 训练成功但无性能指标
success_no_perf = foreground_records[
    foreground_records['training_success'].isin([True, 'True']) &
    ~foreground_records.apply(has_performance_metrics, axis=1)
].shape[0]
print(f"  3. 训练成功但无性能指标: {success_no_perf}条")

# 4. 有能耗但训练失败
failed_with_energy = foreground_records[
    ~foreground_records['training_success'].isin([True, 'True']) &
    foreground_records.apply(has_energy_data, axis=1)
].shape[0]
print(f"  4. 训练失败但有能耗数据: {failed_with_energy}条")

# 5. 异常持续时间
duration_stats = foreground_records['duration_seconds'].describe()
print(f"\n  5. 训练持续时间统计:")
print(f"     平均: {duration_stats['mean']:.2f}秒 ({duration_stats['mean']/60:.2f}分钟)")
print(f"     中位数: {duration_stats['50%']:.2f}秒")
print(f"     最小值: {duration_stats['min']:.2f}秒")
print(f"     最大值: {duration_stats['max']:.2f}秒")

# 异常短或长的训练
very_short = foreground_records[foreground_records['duration_seconds'] < 60].shape[0]
very_long = foreground_records[foreground_records['duration_seconds'] > 10000].shape[0]
print(f"     异常短(<1分钟): {very_short}条")
print(f"     异常长(>2.78小时): {very_long}条")

# ============================================================
# 10. 数据质量问题总结
# ============================================================
print("\n" + "=" * 80)
print("10. 数据质量问题总结")
print("=" * 80)

issues = []

# P0 - 严重问题
if no_perf_only > 0:
    issues.append({
        'priority': 'P0',
        'issue': '性能指标大量缺失',
        'count': no_perf_only,
        'percentage': f"{no_perf_only/len(foreground_records)*100:.2f}%",
        'impact': '严重影响数据可用性',
        'fixability': '困难 - 需要重新运行实验或从日志恢复'
    })

if training_failed > len(foreground_records) * 0.1:
    issues.append({
        'priority': 'P0',
        'issue': '训练失败率过高',
        'count': training_failed,
        'percentage': f"{training_failed/len(foreground_records)*100:.2f}%",
        'impact': '大量实验无效',
        'fixability': '困难 - 需要调试并重新运行'
    })

# P1 - 重要问题
if no_energy_only > 0:
    issues.append({
        'priority': 'P1',
        'issue': '能耗数据缺失',
        'count': no_energy_only,
        'percentage': f"{no_energy_only/len(foreground_records)*100:.2f}%",
        'impact': '影响能耗分析',
        'fixability': '中等 - 可能可以从recoverable数据恢复'
    })

# P2 - 次要问题
if empty_model > 0:
    issues.append({
        'priority': 'P2',
        'issue': '异常模型标识',
        'count': empty_model,
        'percentage': f"{empty_model/len(df)*100:.2f}%",
        'impact': '数据质量低',
        'fixability': '容易 - 清理或标记'
    })

print(f"\n发现 {len(issues)} 个主要数据质量问题:\n")
for i, issue in enumerate(issues, 1):
    print(f"{i}. [{issue['priority']}] {issue['issue']}")
    print(f"   影响记录: {issue['count']}条 ({issue['percentage']})")
    print(f"   影响程度: {issue['impact']}")
    print(f"   修复可行性: {issue['fixability']}")
    print()

# ============================================================
# 11. 修复建议
# ============================================================
print("\n" + "=" * 80)
print("11. 修复建议（按优先级排序）")
print("=" * 80)

recommendations = [
    {
        'priority': 'P0',
        'action': '修复性能指标缺失问题',
        'steps': [
            '1. 检查实验日志，确认是否有性能指标输出',
            '2. 识别哪些模型系统性缺失性能指标（如VulBERTa/mlp, bug-localization）',
            '3. 分析代码，修复性能指标收集逻辑',
            '4. 考虑重新运行受影响的实验（如果修复可行）',
            '5. 或者在分析中排除这些无性能指标的记录'
        ],
        'expected_impact': f'可恢复 {no_perf_only} 条记录（如果可以从日志提取）'
    },
    {
        'priority': 'P1',
        'action': '恢复能耗数据',
        'steps': [
            '1. 检查是否存在 recoverable_energy_data.json',
            '2. 使用现有的 repair_missing_energy_data.py 脚本',
            '3. 验证修复后的数据完整性',
            '4. 备份修复前后的数据进行对比'
        ],
        'expected_impact': f'可能恢复部分 {no_energy_only} 条缺失能耗的记录'
    },
    {
        'priority': 'P1',
        'action': '分析训练失败原因',
        'steps': [
            '1. 收集所有训练失败记录的 error_message',
            '2. 按错误类型分类统计',
            '3. 修复可修复的错误（如配置问题、依赖问题）',
            '4. 对于无法修复的，在分析中排除'
        ],
        'expected_impact': f'理解 {training_failed} 条失败记录的原因'
    },
    {
        'priority': 'P2',
        'action': '清理异常数据',
        'steps': [
            '1. 识别并标记所有 "/" 或空模型名的记录',
            '2. 检查这些记录是否有任何价值',
            '3. 考虑创建一个清理后的数据集用于分析',
            '4. 保留原始数据作为备份'
        ],
        'expected_impact': f'清理 {empty_model} 条异常记录，提升数据质量'
    }
]

for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. [{rec['priority']}] {rec['action']}")
    print(f"\n   具体步骤:")
    for step in rec['steps']:
        print(f"   {step}")
    print(f"\n   预期效果: {rec['expected_impact']}")
    print()

# ============================================================
# 12. 数据使用建议
# ============================================================
print("\n" + "=" * 80)
print("12. 数据使用建议")
print("=" * 80)

print(f"\n根据数据质量评估，推荐以下数据使用策略:\n")

# 策略1: 高质量数据集
if len(high_quality_models) > 0:
    hq_record_count = int(high_quality_models['总数'].sum())
    print(f"📊 策略1: 高质量数据集（推荐用于精确分析）")
    print(f"   范围: 仅使用100%可用率的模型")
    print(f"   模型数: {len(high_quality_models)}个")
    print(f"   记录数: {hq_record_count}条")
    print(f"   优点: 数据完整，结果可靠")
    print(f"   缺点: 样本量较小，模型覆盖有限")
    print()

# 策略2: 平衡数据集
balanced_models = model_usability[model_usability['可用率%'] >= 80.0]
if len(balanced_models) > 0:
    balanced_count = int(balanced_models['可用数'].sum())
    print(f"📊 策略2: 平衡数据集（推荐用于一般分析）")
    print(f"   范围: 使用可用率≥80%的模型")
    print(f"   模型数: {len(balanced_models)}个")
    print(f"   可用记录数: {balanced_count}条")
    print(f"   优点: 样本量较大，质量可接受")
    print(f"   缺点: 可能有少量不完整数据")
    print()

# 策略3: 最大化数据集
print(f"📊 策略3: 最大化数据集（用于探索性分析）")
print(f"   范围: 使用所有可用记录")
print(f"   可用记录数: {usable_count}条")
print(f"   优点: 样本量最大，覆盖面广")
print(f"   缺点: 数据质量参差不齐")
print()

# 策略4: 特定分析数据集
print(f"📊 策略4: 特定分析数据集")
print(f"   能耗分析: 使用有能耗数据的记录 ({fg_with_energy}条)")
print(f"   性能分析: 使用有性能指标的记录 ({fg_with_perf}条)")
print(f"   综合分析: 使用完全可用的记录 ({usable_count}条)")
print()

# ============================================================
# 13. 输出统计摘要到JSON
# ============================================================
summary = {
    'analysis_date': '2026-01-14',
    'total_records': total_records,
    'total_columns': total_columns,
    'foreground_records': len(foreground_records),
    'usable_records': int(usable_count),
    'usability_rate': f"{usable_count/len(foreground_records)*100:.2f}%",
    'training_success_rate': f"{(len(foreground_records) - training_failed)/len(foreground_records)*100:.2f}%",
    'energy_completeness': f"{fg_with_energy/len(foreground_records)*100:.2f}%",
    'performance_completeness': f"{fg_with_perf/len(foreground_records)*100:.2f}%",
    'high_quality_models': len(high_quality_models),
    'high_quality_records': int(high_quality_models['总数'].sum()) if len(high_quality_models) > 0 else 0,
    'issues': issues,
    'top_10_models': model_usability.head(10).to_dict('index')
}

output_json = '/home/green/energy_dl/nightly/data_quality_assessment_summary.json'
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n" + "=" * 80)
print(f"统计摘要已保存到: {output_json}")
print("=" * 80)
