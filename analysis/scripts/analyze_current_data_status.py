#!/usr/bin/env python3
"""分析当前数据现状

用途: 全面分析项目中所有数据文件的情况和可用性
作者: Claude
日期: 2026-01-14
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def analyze_file(file_path, file_name):
    """分析单个数据文件"""
    print(f"\n{'='*80}")
    print(f"文件: {file_name}")
    print(f"路径: {file_path}")
    print(f"{'='*80}")

    if not Path(file_path).exists():
        print(f"❌ 文件不存在")
        return None

    df = pd.read_csv(file_path)

    print(f"✅ 基本信息:")
    print(f"  - 总行数: {len(df)} (含header: {len(df)+1})")
    print(f"  - 列数: {len(df.columns)}")
    print(f"  - 文件大小: {Path(file_path).stat().st_size / 1024:.1f} KB")

    # 检查关键列
    key_columns = {
        '实验ID': 'experiment_id',
        '仓库': 'repository',
        '模型': 'model',
        '时间戳': 'timestamp',
        '训练状态': 'training_status',
    }

    print(f"\n  关键列:")
    for name, col in key_columns.items():
        if col in df.columns:
            unique_count = df[col].nunique()
            missing_count = df[col].isna().sum()
            print(f"    ✅ {name} ({col}): {unique_count} 个唯一值, {missing_count} 个缺失")
        else:
            print(f"    ❌ {name} ({col}): 不存在")

    return df


def analyze_energy_data(df, file_name):
    """分析能耗数据完整性"""
    print(f"\n  能耗数据:")

    energy_cols = [col for col in df.columns if col.startswith('energy_')]

    if not energy_cols:
        print(f"    ❌ 无能耗列")
        return 0

    print(f"    - 能耗列数: {len(energy_cols)}")

    # 检查有完整能耗数据的行
    has_energy = df[energy_cols].notna().any(axis=1)
    energy_count = has_energy.sum()
    energy_rate = energy_count / len(df) * 100

    print(f"    - 有能耗数据的行: {energy_count}/{len(df)} ({energy_rate:.1f}%)")

    # 关键能耗指标
    key_energy = ['energy_cpu_total_joules', 'energy_gpu_total_joules']
    for col in key_energy:
        if col in df.columns:
            count = df[col].notna().sum()
            rate = count / len(df) * 100
            print(f"    - {col}: {count}/{len(df)} ({rate:.1f}%)")

    return energy_count


def analyze_performance_data(df, file_name):
    """分析性能数据完整性"""
    print(f"\n  性能数据:")

    perf_cols = [col for col in df.columns if col.startswith('perf_')]

    if not perf_cols:
        print(f"    ❌ 无性能列")
        return 0

    print(f"    - 性能列数: {len(perf_cols)}")

    # 检查有完整性能数据的行
    has_perf = df[perf_cols].notna().any(axis=1)
    perf_count = has_perf.sum()
    perf_rate = perf_count / len(df) * 100

    print(f"    - 有性能数据的行: {perf_count}/{len(df)} ({perf_rate:.1f}%)")

    return perf_count


def analyze_hyperparams(df, file_name):
    """分析超参数完整性"""
    print(f"\n  超参数数据:")

    hyperparam_cols = [col for col in df.columns if col.startswith('hyperparam_')]

    if not hyperparam_cols:
        print(f"    ❌ 无超参数列")
        return {}

    print(f"    - 超参数列数: {len(hyperparam_cols)}")

    # 统计每个超参数的完整性
    stats = {}
    for col in hyperparam_cols:
        count = df[col].notna().sum()
        rate = count / len(df) * 100
        stats[col] = {'count': count, 'rate': rate}

    # 显示前5个
    print(f"    - 主要超参数完整性:")
    for col in ['hyperparam_epochs', 'hyperparam_learning_rate',
                'hyperparam_batch_size', 'hyperparam_seed', 'hyperparam_dropout']:
        if col in stats:
            s = stats[col]
            print(f"      • {col}: {s['count']}/{len(df)} ({s['rate']:.1f}%)")

    return stats


def analyze_source_tracking(df, file_name):
    """分析数据来源追踪（仅适用于backfilled数据）"""
    print(f"\n  数据来源追踪:")

    source_cols = [col for col in df.columns if col.endswith('_source')]

    if not source_cols:
        print(f"    ⚠️  无 source 列（非回溯数据）")
        return None

    print(f"    - Source 列数: {len(source_cols)}")

    # 统计各种来源
    all_sources = []
    for col in source_cols:
        all_sources.extend(df[col].dropna().tolist())

    from collections import Counter
    source_counts = Counter(all_sources)

    print(f"    - 数据来源分布:")
    for source, count in source_counts.most_common():
        print(f"      • {source}: {count} 个单元格")

    return source_counts


def analyze_usability(df, file_name):
    """分析数据可用性（训练成功 + 有能耗 + 有性能）"""
    print(f"\n  数据可用性分析:")

    # 训练成功
    if 'training_status' in df.columns:
        success = df['training_status'] == 'success'
        success_count = success.sum()
        print(f"    - 训练成功: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
    else:
        success = pd.Series([True] * len(df))
        success_count = len(df)
        print(f"    ⚠️  无 training_status 列，假设全部成功")

    # 有能耗数据
    energy_cols = [col for col in df.columns if col.startswith('energy_')]
    if energy_cols:
        has_energy = df[energy_cols].notna().any(axis=1)
        energy_count = has_energy.sum()
        print(f"    - 有能耗数据: {energy_count}/{len(df)} ({energy_count/len(df)*100:.1f}%)")
    else:
        has_energy = pd.Series([False] * len(df))
        energy_count = 0
        print(f"    ❌ 无能耗列")

    # 有性能数据
    perf_cols = [col for col in df.columns if col.startswith('perf_')]
    if perf_cols:
        has_perf = df[perf_cols].notna().any(axis=1)
        perf_count = has_perf.sum()
        print(f"    - 有性能数据: {perf_count}/{len(df)} ({perf_count/len(df)*100:.1f}%)")
    else:
        has_perf = pd.Series([False] * len(df))
        perf_count = 0
        print(f"    ❌ 无性能列")

    # 完全可用（三者都满足）
    fully_usable = success & has_energy & has_perf
    usable_count = fully_usable.sum()
    usable_rate = usable_count / len(df) * 100

    print(f"\n    ✅ 完全可用（训练成功 + 有能耗 + 有性能）:")
    print(f"       {usable_count}/{len(df)} ({usable_rate:.1f}%)")

    # 按仓库统计可用性
    if 'repository' in df.columns:
        print(f"\n    按仓库统计可用性:")
        for repo in df['repository'].dropna().unique():
            repo_df = df[df['repository'] == repo]
            repo_usable = fully_usable[df['repository'] == repo].sum()
            repo_total = len(repo_df)
            repo_rate = repo_usable / repo_total * 100 if repo_total > 0 else 0
            print(f"      • {repo}: {repo_usable}/{repo_total} ({repo_rate:.1f}%)")

    return {
        'total': len(df),
        'success': success_count,
        'has_energy': energy_count,
        'has_perf': perf_count,
        'fully_usable': usable_count,
        'usable_rate': usable_rate,
    }


def main():
    print("="*80)
    print("能耗DL项目 - 数据现状分析")
    print("="*80)
    print(f"分析时间: 2026-01-14")

    # 分析的文件列表
    files = {
        'raw_data.csv': 'data/raw_data.csv',
        'data.csv': 'data/data.csv',
        'raw_data_backfilled.csv': 'analysis/data/energy_research/backfilled/raw_data_backfilled.csv',
    }

    results = {}

    for name, path in files.items():
        df = analyze_file(path, name)
        if df is not None:
            energy_count = analyze_energy_data(df, name)
            perf_count = analyze_performance_data(df, name)
            hyperparam_stats = analyze_hyperparams(df, name)
            source_stats = analyze_source_tracking(df, name)
            usability_stats = analyze_usability(df, name)

            results[name] = {
                'path': path,
                'rows': len(df),
                'columns': len(df.columns),
                'energy_count': energy_count,
                'perf_count': perf_count,
                'usability': usability_stats,
            }

    # 生成总结
    print(f"\n{'='*80}")
    print("总结对比")
    print(f"{'='*80}")

    print(f"\n文件对比:")
    print(f"{'文件':<30} {'行数':>10} {'列数':>8} {'能耗':>10} {'性能':>10} {'可用':>10}")
    print(f"{'-'*80}")
    for name, stats in results.items():
        if stats:
            print(f"{name:<30} {stats['rows']:>10} {stats['columns']:>8} "
                  f"{stats['energy_count']:>10} {stats['perf_count']:>10} "
                  f"{stats['usability']['fully_usable']:>10}")

    # 保存结果
    output_path = 'analysis/data/energy_research/data_status_analysis.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 分析结果已保存: {output_path}")

    return results


if __name__ == '__main__':
    results = main()
    print("\n✅ 数据现状分析完成！")
