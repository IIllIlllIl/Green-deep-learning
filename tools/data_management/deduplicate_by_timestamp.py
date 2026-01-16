#!/usr/bin/env python3
"""正确的数据去重脚本

用途: 移除 raw_data.csv 中的重复记录
唯一键: timestamp（每次实验运行的唯一标识）
作者: Claude
日期: 2026-01-14
版本: v2.0 - 修正版

重要发现:
- experiment_id 不是唯一键（代表实验配置，可以重复运行）
- timestamp 才是唯一键（每次运行的时间戳）
- raw_data.csv 中存在 210 对重复记录（相同 timestamp，不同 experiment_id 前缀）

去重策略:
1. 使用 timestamp 作为唯一键
2. 保留第一条记录（keep='first'）
3. 移除 experiment_id 前缀不同但实际是同一次运行的重复记录
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime


def analyze_duplicates_by_timestamp(df, file_name):
    """分析基于 timestamp 的重复数据"""
    print(f"\n{'='*80}")
    print(f"分析 {file_name} 的重复情况（基于 timestamp）")
    print(f"{'='*80}")

    print(f"\n基本信息:")
    print(f"  总行数: {len(df)}")
    print(f"  唯一 timestamp: {df['timestamp'].nunique()}")
    print(f"  唯一 experiment_id: {df['experiment_id'].nunique()}")

    # 检查 timestamp 重复
    dup_ts = df[df.duplicated(subset=['timestamp'], keep=False)]

    if len(dup_ts) > 0:
        print(f"\n⚠️  发现重复:")
        print(f"  重复的行数: {len(dup_ts)}")
        print(f"  重复的 timestamp 数量: {dup_ts['timestamp'].nunique()}")

        # 统计重复次数
        ts_counts = dup_ts['timestamp'].value_counts()
        print(f"\n  重复次数分布:")
        for count in sorted(ts_counts.unique(), reverse=True):
            num_ts = (ts_counts == count).sum()
            print(f"    重复 {count} 次: {num_ts} 个 timestamp")

        # 显示重复的例子
        print(f"\n  重复示例（前3个）:")
        for i, (ts, count) in enumerate(ts_counts.head(3).items(), 1):
            examples = dup_ts[dup_ts['timestamp'] == ts][['experiment_id', 'repository']].head(2)
            print(f"    {i}. timestamp: {ts}")
            for idx, row in examples.iterrows():
                print(f"       - {row['experiment_id']} ({row['repository']})")

        return True, dup_ts
    else:
        print(f"\n✅ 无重复数据")
        return False, None


def deduplicate_by_timestamp(df, file_name):
    """按 timestamp 去重
    
    策略:
    - 使用 timestamp 作为唯一键
    - 保留第一条记录（keep='first'）
    """
    print(f"\n{'='*80}")
    print(f"去重 {file_name}")
    print(f"{'='*80}")

    original_count = len(df)

    print(f"\n去重策略:")
    print(f"  - 唯一键: timestamp")
    print(f"  - 保留策略: 第一条记录（keep='first'）")

    # 按 timestamp 去重
    df_dedup = df.drop_duplicates(subset=['timestamp'], keep='first')

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
        f.write(f"{file_name} 去重报告（基于 timestamp）\n")
        f.write("="*80 + "\n\n")

        f.write(f"去重时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 去重结果\n\n")
        f.write(f"原始行数: {len(original_df)}\n")
        f.write(f"去重后行数: {len(dedup_df)}\n")
        f.write(f"移除行数: {removed_count} ({removed_count/len(original_df)*100:.1f}%)\n\n")

        f.write("## 去重策略\n\n")
        f.write("唯一键: timestamp\n")
        f.write("保留策略: 第一条记录（keep='first'）\n\n")

        f.write("## 数据质量对比\n\n")

        # 能耗完整性
        energy_cols = [c for c in dedup_df.columns if c.startswith('energy_')]
        if energy_cols:
            orig_energy = original_df[energy_cols].notna().any(axis=1).sum()
            dedup_energy = dedup_df[energy_cols].notna().any(axis=1).sum()
            f.write(f"能耗数据:\n")
            f.write(f"  原始: {orig_energy}/{len(original_df)} ({orig_energy/len(original_df)*100:.1f}%)\n")
            f.write(f"  去重后: {dedup_energy}/{len(dedup_df)} ({dedup_energy/len(dedup_df)*100:.1f}%)\n\n")

    print(f"\n✅ 报告已生成: {report_path}")
    return report_path



def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='按 timestamp 去重数据文件'
    )
    parser.add_argument(
        '--input',
        default='data/raw_data.csv',
        help='输入文件路径 (默认: data/raw_data.csv)'
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

    args = parser.parse_args()

    print("="*80)
    print("数据去重工具 v2.0（基于 timestamp）")
    print("="*80)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output_dir}")
    if args.dry_run:
        print("⚠️  DRY RUN 模式 - 不会保存文件")
    print("="*80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    df = pd.read_csv(args.input)
    file_name = Path(args.input).stem

    # 分析重复
    has_dup, _ = analyze_duplicates_by_timestamp(df, file_name)

    if has_dup:
        # 去重
        df_dedup, removed_count = deduplicate_by_timestamp(df, file_name)

        if not args.dry_run:
            # 保存去重后的文件
            output_path = output_dir / f'{file_name}_deduped.csv'
            df_dedup.to_csv(output_path, index=False)
            print(f"\n✅ 去重后文件已保存: {output_path}")

            # 生成报告
            generate_report(df, df_dedup, removed_count, file_name, output_dir)
        else:
            print(f"\n⚠️  DRY RUN 模式 - 未保存文件")
    else:
        print(f"\n✅ 数据无重复，无需去重")

    return 0


if __name__ == '__main__':
    sys.exit(main())
