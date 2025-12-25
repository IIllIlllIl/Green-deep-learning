#!/usr/bin/env python3
"""验证数据行数差异问题

目的：
1. 确认 data.csv 的实际行数
2. 确认 stage2_mediators.csv 的实际行数
3. 分析5组数据的删减情况
4. 找出报告中 "648行" 的来源
"""

import pandas as pd
from pathlib import Path

def main():
    print("=" * 80)
    print("数据行数差异验证")
    print("=" * 80)

    # 1. 检查主项目的 data.csv
    print("\n1️⃣ 检查主项目 data.csv:")
    data_csv_path = Path("../../results/data.csv")
    if data_csv_path.exists():
        df_data = pd.read_csv(data_csv_path)
        print(f"   文件: {data_csv_path}")
        print(f"   总行数: {len(df_data)} 行（不含header）")
        print(f"   列数: {len(df_data.columns)} 列")
    else:
        print(f"   ⚠️ 文件不存在: {data_csv_path}")
        df_data = None

    # 2. 检查 energy_data_original.csv
    print("\n2️⃣ 检查 energy_data_original.csv:")
    original_path = Path("../data/energy_research/raw/energy_data_original.csv")
    if original_path.exists():
        df_original = pd.read_csv(original_path)
        print(f"   文件: {original_path}")
        print(f"   总行数: {len(df_original)} 行（不含header）")
        print(f"   列数: {len(df_original.columns)} 列")
    else:
        print(f"   ⚠️ 文件不存在: {original_path}")
        df_original = None

    # 3. 检查 stage2_mediators.csv（备份目录）
    print("\n3️⃣ 检查 stage2_mediators.csv（备份目录）:")
    stage2_path = Path("../data/energy_research/processed.backup_4groups_20251224/stage2_mediators.csv")
    if stage2_path.exists():
        df_stage2 = pd.read_csv(stage2_path)
        print(f"   文件: {stage2_path}")
        print(f"   总行数: {len(df_stage2)} 行（不含header）")
        print(f"   列数: {len(df_stage2.columns)} 列")
    else:
        print(f"   ⚠️ 文件不存在: {stage2_path}")
        df_stage2 = None

    # 4. 分析5组数据文件
    print("\n4️⃣ 分析5组数据文件:")
    processed_dir = Path("../data/energy_research/processed")

    task_files = {
        'image_classification_examples': 'training_data_image_classification_examples.csv',
        'image_classification_resnet': 'training_data_image_classification_resnet.csv',
        'person_reid': 'training_data_person_reid.csv',
        'vulberta': 'training_data_vulberta.csv',
        'bug_localization': 'training_data_bug_localization.csv'
    }

    total_rows = 0
    for task_name, filename in task_files.items():
        filepath = processed_dir / filename
        if filepath.exists():
            df_task = pd.read_csv(filepath)
            rows = len(df_task)
            total_rows += rows
            print(f"   {task_name}: {rows} 行")
        else:
            print(f"   {task_name}: ⚠️ 文件不存在")

    print(f"\n   总计: {total_rows} 行")

    # 5. 计算删减情况（如果stage2存在）
    if df_stage2 is not None:
        print("\n5️⃣ 删减分析（基于 stage2_mediators.csv）:")

        # 定义任务组配置
        task_configs = {
            'image_classification_examples': {
                'repos': ['examples'],
                'perf_cols': ['perf_test_accuracy']
            },
            'image_classification_resnet': {
                'repos': ['pytorch_resnet_cifar10'],
                'perf_cols': ['perf_test_accuracy']
            },
            'person_reid': {
                'repos': ['Person_reID_baseline_pytorch'],
                'perf_cols': ['perf_map', 'perf_rank1', 'perf_rank5']
            },
            'vulberta': {
                'repos': ['VulBERTa'],
                'perf_cols': ['perf_eval_loss']
            },
            'bug_localization': {
                'repos': ['bug-localization-by-dnn-and-rvsm'],
                'perf_cols': ['perf_top1_accuracy', 'perf_top5_accuracy']
            }
        }

        total_original = 0
        total_kept = 0

        for task_name, config in task_configs.items():
            # 筛选仓库
            task_df = df_stage2[df_stage2['repository'].isin(config['repos'])]
            original_rows = len(task_df)

            # 删除性能全缺失的行
            available_perf_cols = [col for col in config['perf_cols'] if col in task_df.columns]
            kept_df = task_df.dropna(subset=available_perf_cols, how='all')
            kept_rows = len(kept_df)

            deleted = original_rows - kept_rows

            print(f"   {task_name}: {original_rows} -> {kept_rows} (删除 {deleted} 行)")

            total_original += original_rows
            total_kept += kept_rows

        total_deleted = total_original - total_kept
        print(f"\n   总计: {total_original} -> {total_kept} (删除 {total_deleted} 行)")
        print(f"   删除占比: {total_deleted / total_original * 100:.1f}%")

    # 6. 对比数据一致性
    print("\n6️⃣ 数据一致性检查:")
    if df_data is not None and df_original is not None:
        if len(df_data) == len(df_original):
            print(f"   ✅ data.csv 与 energy_data_original.csv 行数一致: {len(df_data)} 行")
        else:
            print(f"   ⚠️ 行数不一致:")
            print(f"      data.csv: {len(df_data)} 行")
            print(f"      energy_data_original.csv: {len(df_original)} 行")
            print(f"      差异: {abs(len(df_data) - len(df_original))} 行")

    if df_data is not None and df_stage2 is not None:
        if len(df_data) == len(df_stage2):
            print(f"   ✅ data.csv 与 stage2_mediators.csv 行数一致: {len(df_data)} 行")
        else:
            print(f"   ⚠️ 行数不一致:")
            print(f"      data.csv: {len(df_data)} 行")
            print(f"      stage2_mediators.csv: {len(df_stage2)} 行")
            print(f"      差异: {abs(len(df_data) - len(df_stage2))} 行")

    # 7. 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)

    if df_data is not None:
        print(f"✅ 主项目 data.csv: {len(df_data)} 行")

    if df_stage2 is not None:
        print(f"✅ 输入文件 stage2_mediators.csv: {len(df_stage2)} 行")

    print(f"✅ 5组输出数据: {total_rows} 行")

    if df_stage2 is not None:
        deleted = len(df_stage2) - total_rows
        print(f"❌ 数据缺失: {deleted} 行 ({deleted / len(df_stage2) * 100:.1f}%)")

    print("\n📌 关键发现:")
    print("   - 报告中提到 '648行' 可能是错误统计")
    print(f"   - 实际应该是: 从 {len(df_stage2) if df_stage2 is not None else '?'} 行删减到 {total_rows} 行")
    print(f"   - 删减了 {len(df_stage2) - total_rows if df_stage2 is not None else '?'} 行")


if __name__ == '__main__':
    main()
