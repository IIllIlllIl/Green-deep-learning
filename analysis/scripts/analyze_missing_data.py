#!/usr/bin/env python3
"""分析被筛选掉的数据

目的：找出哪78行数据（726-648）没有被包含在5组任务中
"""

import pandas as pd
from pathlib import Path

def main():
    print("=" * 80)
    print("分析被筛选掉的数据")
    print("=" * 80)

    # 读取 stage2_mediators.csv
    stage2_path = Path("../data/energy_research/processed.backup_4groups_20251224/stage2_mediators.csv")
    df = pd.read_csv(stage2_path)

    print(f"\n总数据: {len(df)} 行")

    # 定义5组任务的仓库
    task_repos = {
        'image_classification_examples': ['examples'],
        'image_classification_resnet': ['pytorch_resnet_cifar10'],
        'person_reid': ['Person_reID_baseline_pytorch'],
        'vulberta': ['VulBERTa'],
        'bug_localization': ['bug-localization-by-dnn-and-rvsm']
    }

    # 收集所有被包含的仓库
    all_included_repos = set()
    for repos in task_repos.values():
        all_included_repos.update(repos)

    print(f"\n包含的仓库: {sorted(all_included_repos)}")

    # 找出不在5组中的数据
    mask = df['repository'].isin(all_included_repos)
    included_df = df[mask]
    excluded_df = df[~mask]

    print(f"\n✅ 包含在5组中: {len(included_df)} 行")
    print(f"❌ 被筛选掉: {len(excluded_df)} 行")

    if len(excluded_df) > 0:
        print(f"\n被筛选掉的数据分布:")
        excluded_repos = excluded_df['repository'].value_counts()
        for repo, count in excluded_repos.items():
            print(f"   {repo}: {count} 行")

        # 检查这些被筛选掉的数据是否有性能指标
        print(f"\n被筛选数据的性能指标情况:")
        perf_cols = [col for col in excluded_df.columns if col.startswith('perf_')]

        for repo, count in excluded_repos.items():
            repo_df = excluded_df[excluded_df['repository'] == repo]

            # 检查哪些性能列有数据
            has_perf = []
            for col in perf_cols:
                non_null = repo_df[col].notna().sum()
                if non_null > 0:
                    has_perf.append(f"{col}({non_null})")

            print(f"   {repo}: {', '.join(has_perf) if has_perf else '无性能数据'}")

    # 统计每组的情况
    print(f"\n" + "=" * 80)
    print("5组任务的详细统计")
    print("=" * 80)

    for task_name, repos in task_repos.items():
        task_df = df[df['repository'].isin(repos)]
        print(f"\n{task_name}:")
        print(f"   仓库: {repos}")
        print(f"   总行数: {len(task_df)}")

        if len(task_df) > 0:
            # 检查模型分布
            model_counts = task_df['model'].value_counts()
            print(f"   模型分布:")
            for model, count in model_counts.items():
                print(f"      {model}: {count} 行")


if __name__ == '__main__':
    main()
