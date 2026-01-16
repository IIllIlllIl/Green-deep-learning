#!/usr/bin/env python3
"""去重数据文件

用途: 移除 raw_data.csv 和 data.csv 中的重复记录
策略: 使用 experiment_id + timestamp 作为唯一键
作者: Claude
日期: 2026-01-14
版本: v1.0

去重规则:
1. 优先保留最新的记录（timestamp最大）
2. 如果timestamp相同，保留数据最完整的记录（能耗+性能）
3. 生成去重报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime


def analyze_duplicates(df, file_name):
    """分析重复数据"""
    print(f"\n{'='*80}")
    print(f"分析 {file_name} 的重复情况")
    print(f"{'='*80}")

    print(f"\n基本信息:")
    print(f"  总行数: {len(df)}")
    print(f"  唯一 experiment_id: {df['experiment_id'].nunique()}")
    print(f"  唯一 timestamp: {df['timestamp'].nunique()}")

    # 检查 experiment_id 重复
    dup_ids = df[df.duplicated(subset=['experiment_id'], keep=False)]

    if len(dup_ids) > 0:
        print(f"\n⚠️  发现重复:")
        print(f"  重复的行数: {len(dup_ids)}")
        print(f"  重复的 experiment_id 数量: {dup_ids['experiment_id'].nunique()}")

        # 统计重复次数
        dup_counts = dup_ids['experiment_id'].value_counts()
        print(f"\n  重复次数分布:")
        for count in sorted(dup_counts.unique(), reverse=True):
            num_ids = (dup_counts == count).sum()
            print(f"    重复 {count} 次: {num_ids} 个 experiment_id")

        return True, dup_ids
    else:
        print(f"\n✅ 无重复数据")
        return False, None


def calculate_completeness_score(row, energy_cols, perf_cols):
    """计算数据完整性得分"""
    score = 0

    # 能耗数据得分（权重50%）
    if energy_cols:
        energy_complete = row[energy_cols].notna().sum()
        score += (energy_complete / len(energy_cols)) * 50

    # 性能数据得分（权重50%）
    if perf_cols:
        perf_complete = row[perf_cols].notna().sum()
        score += (perf_complete / len(perf_cols)) * 50

    return score


def deduplicate_dataframe(df, file_name):
    """去重DataFrame
    
    策略:
    1. 按 experiment_id 分组
    2. 对于每组重复记录，选择最优的一条:
       - 优先选择 timestamp 最新的
       - 如果 timestamp 相同，选择数据最完整的
    """
    print(f"\n{'='*80}")
    print(f"去重 {file_name}")
    print(f"{'='*80}")

    original_count = len(df)

    # 识别能耗和性能列
    energy_cols = [c for c in df.columns if c.startswith('energy_')]
    perf_cols = [c for c in df.columns if c.startswith('perf_')]

    print(f"\n去重策略:")
    print(f"  1. 按 experiment_id 分组")
    print(f"  2. 保留每组中 timestamp 最新的记录")
    print(f"  3. 如果 timestamp 相同，保留数据最完整的记录")

    # 计算每行的完整性得分
    df['_completeness_score'] = df.apply(
        lambda row: calculate_completeness_score(row, energy_cols, perf_cols),
        axis=1
    )

    # 转换 timestamp 为 datetime
    df['_timestamp_dt'] = pd.to_datetime(df['timestamp'])

    # 按 experiment_id 分组，保留最优记录
    df_dedup = df.sort_values(
        ['experiment_id', '_timestamp_dt', '_completeness_score'],
        ascending=[True, False, False]
    ).drop_duplicates(subset=['experiment_id'], keep='first')

    # 删除临时列
    df_dedup = df_dedup.drop(columns=['_completeness_score', '_timestamp_dt'])

    removed_count = original_count - len(df_dedup)

    print(f"\n去重结果:")
    print(f"  原始行数: {original_count}")
    print(f"  去重后行数: {len(df_dedup)}")
    print(f"  移除行数: {removed_count} ({removed_count/original_count*100:.1f}%)")

    return df_dedup, removed_count




def generate_report(original_df, dedup_df, removed_count, file_name, output_dir):
    """生成去重报告"""
    report_path = output_dir / f'{file_name}_deduplication_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{file_name} 去重报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"去重时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 去重结果\n\n")
        f.write(f"原始行数: {len(original_df)}\n")
        f.write(f"去重后行数: {len(dedup_df)}\n")
        f.write(f"移除行数: {removed_count} ({removed_count/len(original_df)*100:.1f}%)\n\n")

        f.write("## 去重策略\n\n")
        f.write("1. 按 experiment_id 分组\n")
        f.write("2. 保留每组中 timestamp 最新的记录\n")
        f.write("3. 如果 timestamp 相同，保留数据最完整的记录\n\n")

        f.write("## 数据质量对比\n\n")

        # 能耗完整性
        energy_cols = [c for c in dedup_df.columns if c.startswith('energy_')]
        if energy_cols:
            orig_energy = original_df[energy_cols].notna().any(axis=1).sum()
            dedup_energy = dedup_df[energy_cols].notna().any(axis=1).sum()
            f.write(f"能耗数据:\n")
            f.write(f"  原始: {orig_energy}/{len(original_df)} ({orig_energy/len(original_df)*100:.1f}%)\n")
            f.write(f"  去重后: {dedup_energy}/{len(dedup_df)} ({dedup_energy/len(dedup_df)*100:.1f}%)\n\n")

        # 性能完整性
        perf_cols = [c for c in dedup_df.columns if c.startswith('perf_')]
        if perf_cols:
            orig_perf = original_df[perf_cols].notna().any(axis=1).sum()
            dedup_perf = dedup_df[perf_cols].notna().any(axis=1).sum()
            f.write(f"性能数据:\n")
            f.write(f"  原始: {orig_perf}/{len(original_df)} ({orig_perf/len(original_df)*100:.1f}%)\n")
            f.write(f"  去重后: {dedup_perf}/{len(dedup_df)} ({dedup_perf/len(dedup_df)*100:.1f}%)\n\n")

    print(f"\n✅ 报告已生成: {report_path}")
    return report_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='去重 raw_data.csv 和 data.csv 中的重复记录'
    )
    parser.add_argument(
        '--raw-data',
        default='data/raw_data.csv',
        help='raw_data.csv 路径 (默认: data/raw_data.csv)'
    )
    parser.add_argument(
        '--data',
        default='data/data.csv',
        help='data.csv 路径 (默认: data/data.csv)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/deduplication',
        help='输出目录 (默认: data/deduplication)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，不保存文件'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        default=True,
        help='备份原始文件 (默认: True)'
    )

    args = parser.parse_args()

    print("="*80)
    print("数据去重工具 v1.0")
    print("="*80)
    print(f"raw_data.csv: {args.raw_data}")
    print(f"data.csv: {args.data}")
    print(f"输出目录: {args.output_dir}")
    if args.dry_run:
        print("⚠️  DRY RUN 模式 - 不会保存文件")
    print("="*80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 处理 raw_data.csv
    print("\n" + "="*80)
    print("处理 raw_data.csv")
    print("="*80)

    raw_df = pd.read_csv(args.raw_data)
    has_dup, _ = analyze_duplicates(raw_df, 'raw_data.csv')

    if has_dup:
        raw_dedup, raw_removed = deduplicate_dataframe(raw_df, 'raw_data.csv')
        results['raw_data'] = {
            'original': len(raw_df),
            'deduped': len(raw_dedup),
            'removed': raw_removed
        }

        if not args.dry_run:
            # 备份原文件
            if args.backup:
                backup_path = Path(args.raw_data).parent / f'raw_data.csv.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                raw_df.to_csv(backup_path, index=False)
                print(f"\n✅ 原文件已备份: {backup_path}")

            # 保存去重后的文件
            output_path = output_dir / 'raw_data_deduped.csv'
            raw_dedup.to_csv(output_path, index=False)
            print(f"✅ 去重后文件已保存: {output_path}")

            # 生成报告
            generate_report(raw_df, raw_dedup, raw_removed, 'raw_data', output_dir)
    else:
        results['raw_data'] = {'original': len(raw_df), 'deduped': len(raw_df), 'removed': 0}

    # 处理 data.csv
    print("\n" + "="*80)
    print("处理 data.csv")
    print("="*80)

    data_df = pd.read_csv(args.data)
    has_dup, _ = analyze_duplicates(data_df, 'data.csv')

    if has_dup:
        data_dedup, data_removed = deduplicate_dataframe(data_df, 'data.csv')
        results['data'] = {
            'original': len(data_df),
            'deduped': len(data_dedup),
            'removed': data_removed
        }

        if not args.dry_run:
            # 备份原文件
            if args.backup:
                backup_path = Path(args.data).parent / f'data.csv.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                data_df.to_csv(backup_path, index=False)
                print(f"\n✅ 原文件已备份: {backup_path}")

            # 保存去重后的文件
            output_path = output_dir / 'data_deduped.csv'
            data_dedup.to_csv(output_path, index=False)
            print(f"✅ 去重后文件已保存: {output_path}")

            # 生成报告
            generate_report(data_df, data_dedup, data_removed, 'data', output_dir)
    else:
        results['data'] = {'original': len(data_df), 'deduped': len(data_df), 'removed': 0}

    # 总结
    print("\n" + "="*80)
    print("去重完成总结")
    print("="*80)
    
    for file_name, stats in results.items():
        print(f"\n{file_name}.csv:")
        print(f"  原始行数: {stats['original']}")
        print(f"  去重后行数: {stats['deduped']}")
        print(f"  移除行数: {stats['removed']} ({stats['removed']/stats['original']*100:.1f}%)")

    if not args.dry_run:
        print(f"\n✅ 去重后的文件保存在: {output_dir}")
        print(f"✅ 原始文件已备份")
    else:
        print(f"\n⚠️  DRY RUN 模式 - 未保存任何文件")

    return 0


if __name__ == '__main__':
    sys.exit(main())
