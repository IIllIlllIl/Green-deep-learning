#!/usr/bin/env python3
"""
为6个任务组生成DiBS训练数据

基于836行完整数据，为每个任务组生成标准化的DiBS输入数据

创建日期: 2026-01-05
数据源: data/energy_research/raw/energy_data_original.csv (836行)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

# 6个任务组定义
TASK_GROUPS = [
    {
        'id': 'group1_examples',
        'name': 'examples（图像分类-小型）',
        'repositories': ['examples'],
        'description': '4个小型图像分类模型（MNIST等）',
        'expected_samples': 259
    },
    {
        'id': 'group2_vulberta',
        'name': 'VulBERTa（代码漏洞检测）',
        'repositories': ['VulBERTa'],
        'description': '基于BERT的漏洞检测模型',
        'expected_samples': 152
    },
    {
        'id': 'group3_person_reid',
        'name': 'Person_reID（行人重识别）',
        'repositories': ['Person_reID_baseline_pytorch'],
        'description': '行人重识别基线模型',
        'expected_samples': 146
    },
    {
        'id': 'group4_bug_localization',
        'name': 'bug-localization（缺陷定位）',
        'repositories': ['bug-localization-by-dnn-and-rvsm'],
        'description': 'DNN+RVSM缺陷定位模型',
        'expected_samples': 142
    },
    {
        'id': 'group5_mrt_oast',
        'name': 'MRT-OAST（缺陷定位）',
        'repositories': ['MRT-OAST'],
        'description': 'MRT-OAST缺陷定位模型',
        'expected_samples': 88
    },
    {
        'id': 'group6_resnet',
        'name': 'pytorch_resnet（图像分类-ResNet）',
        'repositories': ['pytorch_resnet_cifar10'],
        'description': 'ResNet CIFAR-10分类',
        'expected_samples': 49
    }
]

# 数据处理参数
MAX_MISSING_RATE = 0.40  # 列最大缺失率阈值（调整为40%以保留更多性能指标）
MIN_SAMPLES = 10  # 最小样本量

def process_task_group(df_full, task_group, verbose=True):
    """
    处理单个任务组数据

    参数:
        df_full: 完整数据（836行）
        task_group: 任务组配置字典
        verbose: 是否输出详细信息

    返回:
        df_processed: 处理后的数据
        stats: 处理统计信息
    """
    group_id = task_group['id']
    group_name = task_group['name']
    repos = task_group['repositories']

    if verbose:
        print(f"\n{'='*80}")
        print(f"处理任务组: {group_name}")
        print(f"ID: {group_id}")
        print(f"Repository: {repos}")
        print(f"{'='*80}")

    # 1. 过滤对应任务组的数据
    df = df_full[df_full['repository'].isin(repos)].copy()
    n_samples_raw = len(df)

    if verbose:
        print(f"\n[步骤1] 过滤repository")
        print(f"  原始数据: {len(df_full)}行")
        print(f"  过滤后: {n_samples_raw}行")
        print(f"  预期: {task_group['expected_samples']}行")
        if n_samples_raw != task_group['expected_samples']:
            print(f"  ⚠️  警告: 实际样本数与预期不符！")

    # 检查最小样本量
    if n_samples_raw < MIN_SAMPLES:
        raise ValueError(f"样本量不足: {n_samples_raw} < {MIN_SAMPLES}")

    # 2. 选择数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].copy()

    if verbose:
        print(f"\n[步骤2] 选择数值型列")
        print(f"  总列数: {len(df.columns)}")
        print(f"  数值列数: {len(numeric_cols)}")

    # 3. 移除全NaN列
    df_no_all_nan = df_numeric.dropna(axis=1, how='all')
    n_cols_removed_all_nan = len(df_numeric.columns) - len(df_no_all_nan.columns)

    if verbose:
        print(f"\n[步骤3] 移除全NaN列")
        print(f"  移除列数: {n_cols_removed_all_nan}")
        print(f"  保留列数: {len(df_no_all_nan.columns)}")

    # 4. 计算缺失率并移除高缺失率列
    missing_rate = df_no_all_nan.isna().sum() / len(df_no_all_nan)
    cols_high_missing = missing_rate[missing_rate > MAX_MISSING_RATE].index.tolist()
    cols_to_keep = missing_rate[missing_rate <= MAX_MISSING_RATE].index.tolist()

    df_low_missing = df_no_all_nan[cols_to_keep].copy()

    if verbose:
        print(f"\n[步骤4] 移除缺失率>{MAX_MISSING_RATE*100}%的列")
        print(f"  高缺失率列数: {len(cols_high_missing)}")
        if len(cols_high_missing) > 0 and len(cols_high_missing) <= 10:
            print(f"  高缺失率列: {cols_high_missing}")
        print(f"  保留列数: {len(cols_to_keep)}")

    # 检查是否还有足够的特征
    if len(cols_to_keep) < 3:
        raise ValueError(f"特征数不足: {len(cols_to_keep)} < 3")

    # 5. 移除零方差列（常数列）
    var_per_col = df_low_missing.var()
    zero_var_cols = var_per_col[var_per_col == 0].index.tolist()
    cols_with_var = var_per_col[var_per_col > 0].index.tolist()

    df_with_var = df_low_missing[cols_with_var].copy()

    if verbose:
        print(f"\n[步骤5] 移除零方差列（常数列）")
        print(f"  零方差列数: {len(zero_var_cols)}")
        if len(zero_var_cols) > 0 and len(zero_var_cols) <= 10:
            print(f"  零方差列: {zero_var_cols}")
        print(f"  保留列数: {len(cols_with_var)}")

    # 6. 填充缺失值（用均值）
    df_filled = df_with_var.fillna(df_with_var.mean())

    # 检查是否还有NaN
    remaining_nan = df_filled.isna().sum().sum()
    if remaining_nan > 0:
        # 如果还有NaN（可能某列全NaN但没被检测到），用0填充
        df_filled = df_filled.fillna(0)
        if verbose:
            print(f"  ⚠️  警告: 填充后仍有{remaining_nan}个NaN，已用0填充")

    if verbose:
        print(f"\n[步骤6] 填充缺失值（均值填充）")
        print(f"  填充前缺失值: {df_with_var.isna().sum().sum()}")
        print(f"  填充后缺失值: {df_filled.isna().sum().sum()}")

    # 7. 标准化（Z-score）
    scaler = StandardScaler()
    df_scaled_values = scaler.fit_transform(df_filled)
    df_scaled = pd.DataFrame(
        df_scaled_values,
        columns=df_filled.columns,
        index=df_filled.index
    )

    if verbose:
        print(f"\n[步骤7] 标准化（Z-score）")
        print(f"  均值: {df_scaled.mean().mean():.6f} (应接近0)")
        print(f"  标准差: {df_scaled.std().mean():.6f} (应接近1)")

    # 8. 统计信息
    stats = {
        'group_id': group_id,
        'group_name': group_name,
        'repositories': repos,
        'n_samples_raw': n_samples_raw,
        'n_samples_final': len(df_scaled),
        'n_features_raw': len(numeric_cols),
        'n_features_final': len(df_scaled.columns),
        'n_cols_removed_all_nan': n_cols_removed_all_nan,
        'n_cols_removed_high_missing': len(cols_high_missing),
        'n_cols_removed_zero_var': len(zero_var_cols),
        'feature_names': df_scaled.columns.tolist(),
        'missing_rate_before_fill': float((df_with_var.isna().sum().sum() / (df_with_var.shape[0] * df_with_var.shape[1]))),
        'processing_success': True
    }

    if verbose:
        print(f"\n[完成] 任务组 {group_name}")
        print(f"  最终数据: {stats['n_samples_final']}行 × {stats['n_features_final']}列")
        print(f"  特征保留率: {stats['n_features_final']/stats['n_features_raw']*100:.1f}%")
        print(f"  缺失率（填充前）: {stats['missing_rate_before_fill']*100:.2f}%")

    return df_scaled, stats

def main():
    """主函数"""
    print("="*80)
    print("为6个任务组生成DiBS训练数据")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 输入/输出路径
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'energy_research' / 'raw' / 'energy_data_original.csv'
    output_dir = base_dir / 'data' / 'energy_research' / 'dibs_training'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n输入文件: {input_file}")
    print(f"输出目录: {output_dir}")

    # 加载数据
    print(f"\n{'='*80}")
    print("加载完整数据...")
    print(f"{'='*80}")

    if not input_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {input_file}")

    df_full = pd.read_csv(input_file)
    print(f"✓ 数据加载成功: {len(df_full)}行 × {len(df_full.columns)}列")

    # 验证数据行数
    if len(df_full) != 836:
        print(f"⚠️  警告: 数据行数为{len(df_full)}，预期836行")

    # 处理所有任务组
    all_stats = []
    success_count = 0

    for task_group in TASK_GROUPS:
        try:
            df_processed, stats = process_task_group(df_full, task_group, verbose=True)

            # 保存处理后的数据
            output_file = output_dir / f"{task_group['id']}.csv"
            df_processed.to_csv(output_file, index=False)
            print(f"\n✓ 数据已保存: {output_file}")

            stats['output_file'] = str(output_file)
            all_stats.append(stats)
            success_count += 1

        except Exception as e:
            print(f"\n✗ 任务组 {task_group['name']} 处理失败: {e}")
            stats = {
                'group_id': task_group['id'],
                'group_name': task_group['name'],
                'processing_success': False,
                'error_message': str(e)
            }
            all_stats.append(stats)

    # 生成总结报告
    print(f"\n{'='*80}")
    print("数据生成总结")
    print(f"{'='*80}")

    print(f"\n成功率: {success_count}/{len(TASK_GROUPS)} ({success_count/len(TASK_GROUPS)*100:.0f}%)")

    # 创建总结表格
    print(f"\n任务组数据总结:")
    print(f"{'ID':<25} {'样本数':<10} {'特征数':<10} {'状态':<10}")
    print("-"*80)

    for stats in all_stats:
        if stats['processing_success']:
            status = "✓ 成功"
            samples = f"{stats['n_samples_final']}"
            features = f"{stats['n_features_final']}"
        else:
            status = "✗ 失败"
            samples = "N/A"
            features = "N/A"

        print(f"{stats['group_id']:<25} {samples:<10} {features:<10} {status:<10}")

    # 保存统计信息JSON
    stats_file = output_dir / 'generation_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_file': str(input_file),
            'output_dir': str(output_dir),
            'total_tasks': len(TASK_GROUPS),
            'successful_tasks': success_count,
            'task_stats': all_stats
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 统计信息已保存: {stats_file}")

    # 生成Markdown报告
    report_file = output_dir / 'DATA_GENERATION_REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 6任务组DiBS训练数据生成报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**数据源**: {input_file}\n")
        f.write(f"**总样本数**: {len(df_full)}行\n")
        f.write(f"**成功率**: {success_count}/{len(TASK_GROUPS)} ({success_count/len(TASK_GROUPS)*100:.0f}%)\n\n")

        f.write("## 任务组详情\n\n")
        f.write("| 任务组 | 样本数 | 特征数 | 保留率 | 缺失率 | 状态 |\n")
        f.write("|--------|--------|--------|--------|--------|------|\n")

        for stats in all_stats:
            if stats['processing_success']:
                name = stats['group_name']
                samples = stats['n_samples_final']
                features = stats['n_features_final']
                retention = f"{stats['n_features_final']/stats['n_features_raw']*100:.1f}%"
                missing = f"{stats['missing_rate_before_fill']*100:.2f}%"
                status = "✓"
            else:
                name = stats['group_name']
                samples = "N/A"
                features = "N/A"
                retention = "N/A"
                missing = "N/A"
                status = "✗"

            f.write(f"| {name} | {samples} | {features} | {retention} | {missing} | {status} |\n")

        f.write("\n## 数据处理流程\n\n")
        f.write("1. 按repository过滤任务组数据\n")
        f.write("2. 选择数值型列\n")
        f.write("3. 移除全NaN列\n")
        f.write(f"4. 移除缺失率>{MAX_MISSING_RATE*100}%的列\n")
        f.write("5. 移除零方差列（常数列）\n")
        f.write("6. 填充缺失值（均值填充）\n")
        f.write("7. 标准化（Z-score）\n\n")

        f.write("## 输出文件\n\n")
        for stats in all_stats:
            if stats['processing_success']:
                f.write(f"- `{stats['group_id']}.csv` - {stats['n_samples_final']}行 × {stats['n_features_final']}列\n")

        f.write(f"\n---\n\n")
        f.write(f"**报告生成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"✓ Markdown报告已保存: {report_file}")

    print(f"\n{'='*80}")
    print("数据生成完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    if success_count == len(TASK_GROUPS):
        print("\n🎉 所有任务组数据生成成功！")
        print("\n下一步:")
        print("  1. 查看生成报告: cat data/energy_research/dibs_training/DATA_GENERATION_REPORT.md")
        print("  2. 运行DiBS分析: 使用生成的6个CSV文件")
        return 0
    else:
        print(f"\n⚠️  {len(TASK_GROUPS) - success_count} 个任务组失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
