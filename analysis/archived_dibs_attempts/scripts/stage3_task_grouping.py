#!/usr/bin/env python3
"""
阶段3: 任务分组 (Task Grouping)

功能:
1. 加载stage2_mediators.csv
2. 按4个任务组分割数据:
   - 图像分类 (examples + pytorch_resnet_cifar10)
   - Person_reID (Person_reID_baseline_pytorch)
   - VulBERTa (VulBERTa)
   - Bug定位 (bug-localization-by-dnn-and-rvsm)
3. 为每个任务组生成独立CSV文件
4. 输出: 4个任务组CSV文件

作者: Analysis Module Team
日期: 2025-12-23
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据路径
DATA_DIR = PROJECT_ROOT / "data" / "energy_research"
PROCESSED_DIR = DATA_DIR / "processed"
INPUT_FILE = PROCESSED_DIR / "stage2_mediators.csv"
REPORT_FILE = PROCESSED_DIR / "stage3_task_grouping_report.txt"

# 任务组定义
TASK_GROUPS = {
    'image_classification': {
        'name': '图像分类',
        'repositories': ['examples', 'pytorch_resnet_cifar10'],
        'output_file': PROCESSED_DIR / "stage3_image_classification.csv"
    },
    'person_reid': {
        'name': 'Person_reID检索',
        'repositories': ['Person_reID_baseline_pytorch'],
        'output_file': PROCESSED_DIR / "stage3_person_reid.csv"
    },
    'vulberta': {
        'name': 'VulBERTa漏洞检测',
        'repositories': ['VulBERTa'],
        'output_file': PROCESSED_DIR / "stage3_vulberta.csv"
    },
    'bug_localization': {
        'name': 'Bug定位',
        'repositories': ['bug-localization-by-dnn-and-rvsm'],
        'output_file': PROCESSED_DIR / "stage3_bug_localization.csv"
    }
}


def load_data(filepath):
    """加载CSV数据"""
    print(f"\n📂 加载数据: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    return df


def verify_repository_column(df):
    """验证repository列"""
    print("\n🔍 验证repository列...")

    if 'repository' not in df.columns:
        raise ValueError("❌ 缺少repository列")

    repos = df['repository'].unique()
    print(f"✅ 发现 {len(repos)} 个repository:")
    for repo in sorted(repos):
        count = (df['repository'] == repo).sum()
        print(f"  - {repo}: {count} 行")

    return repos


def split_into_task_groups(df):
    """分割数据到任务组"""
    print("\n🔧 分割数据到任务组...")

    task_group_data = {}

    for group_id, group_info in TASK_GROUPS.items():
        print(f"\n  处理: {group_info['name']}...")

        # 筛选属于此任务组的数据
        mask = df['repository'].isin(group_info['repositories'])
        group_df = df[mask].copy()

        task_group_data[group_id] = {
            'data': group_df,
            'info': group_info,
            'sample_count': len(group_df)
        }

        print(f"    样本数: {len(group_df)}")
        print(f"    Repository: {', '.join(group_info['repositories'])}")

        # 显示repository分布（如果多个）
        if len(group_info['repositories']) > 1:
            for repo in group_info['repositories']:
                repo_count = (group_df['repository'] == repo).sum()
                print(f"      - {repo}: {repo_count} 行")

    return task_group_data


def verify_split_completeness(df, task_group_data):
    """验证分割完整性"""
    print("\n🔍 验证分割完整性...")

    total_samples = sum(group['sample_count'] for group in task_group_data.values())
    original_samples = len(df)

    print(f"  原始样本数: {original_samples}")
    print(f"  分组后总数: {total_samples}")

    if total_samples == original_samples:
        print(f"  ✅ 样本数一致")
    else:
        diff = original_samples - total_samples
        print(f"  ⚠️  差异: {diff} 行")

        # 检查未分配的行
        all_repos = []
        for group_info in TASK_GROUPS.values():
            all_repos.extend(group_info['repositories'])

        unassigned = df[~df['repository'].isin(all_repos)]
        if len(unassigned) > 0:
            print(f"  ⚠️  未分配的repository:")
            for repo in unassigned['repository'].unique():
                count = (unassigned['repository'] == repo).sum()
                print(f"    - {repo}: {count} 行")

    # 检查重叠
    print("\n  检查任务组间重叠:")
    has_overlap = False

    group_ids = list(task_group_data.keys())
    for i in range(len(group_ids)):
        for j in range(i+1, len(group_ids)):
            group_i = task_group_data[group_ids[i]]
            group_j = task_group_data[group_ids[j]]

            # 检查experiment_id是否有重叠
            ids_i = set(group_i['data']['experiment_id'])
            ids_j = set(group_j['data']['experiment_id'])
            overlap = ids_i & ids_j

            if overlap:
                has_overlap = True
                print(f"  ⚠️  {group_i['info']['name']} 和 {group_j['info']['name']} 重叠: {len(overlap)} 个实验")

    if not has_overlap:
        print(f"  ✅ 无重叠")

    return total_samples == original_samples and not has_overlap


def analyze_task_group_quality(task_group_data):
    """分析各任务组数据质量"""
    print("\n📊 任务组数据质量分析...")

    key_vars = ['energy_cpu_total_joules', 'energy_gpu_total_joules',
                'training_duration', 'gpu_util_avg']

    results = {}

    for group_id, group in task_group_data.items():
        group_df = group['data']
        group_name = group['info']['name']

        print(f"\n  {group_name}:")
        print(f"    样本数: {len(group_df)}")

        # 计算关键变量填充率
        group_results = {}
        for var in key_vars:
            if var in group_df.columns:
                filled = group_df[var].notna().sum()
                fill_rate = (filled / len(group_df)) * 100
                group_results[var] = fill_rate

                if fill_rate < 70:
                    status = "⚠️ "
                else:
                    status = "✅"

                print(f"    {status} {var:30s}: {fill_rate:5.1f}%")

        results[group_id] = group_results

    return results


def save_task_groups(task_group_data):
    """保存任务组数据"""
    print("\n💾 保存任务组数据...")

    saved_files = []

    for group_id, group in task_group_data.items():
        group_df = group['data']
        output_file = group['info']['output_file']
        group_name = group['info']['name']

        group_df.to_csv(output_file, index=False)

        file_size = output_file.stat().st_size / 1024

        print(f"\n  ✅ {group_name}:")
        print(f"     文件: {output_file.name}")
        print(f"     行数: {len(group_df)}")
        print(f"     列数: {len(group_df.columns)}")
        print(f"     大小: {file_size:.1f} KB")

        saved_files.append({
            'group_id': group_id,
            'group_name': group_name,
            'file_path': output_file,
            'sample_count': len(group_df),
            'column_count': len(group_df.columns),
            'file_size_kb': file_size
        })

    return saved_files


def generate_grouping_report(df, task_group_data, quality_results, saved_files):
    """生成任务分组报告"""
    print(f"\n📊 生成任务分组报告...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("阶段3: 任务分组报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"输入文件: {INPUT_FILE}")
    report_lines.append("")

    # 数据概览
    report_lines.append("=" * 80)
    report_lines.append("1. 数据概览")
    report_lines.append("=" * 80)
    report_lines.append(f"原始总行数: {len(df):,}")
    report_lines.append(f"任务组数量: {len(task_group_data)}")
    report_lines.append("")

    # 任务组统计
    report_lines.append("=" * 80)
    report_lines.append("2. 任务组统计")
    report_lines.append("=" * 80)

    for file_info in saved_files:
        report_lines.append(f"\n{file_info['group_name']}:")
        report_lines.append(f"  文件名: {file_info['file_path'].name}")
        report_lines.append(f"  样本数: {file_info['sample_count']}")
        report_lines.append(f"  列数: {file_info['column_count']}")
        report_lines.append(f"  文件大小: {file_info['file_size_kb']:.1f} KB")

    report_lines.append("")

    # 数据质量摘要
    report_lines.append("=" * 80)
    report_lines.append("3. 数据质量摘要")
    report_lines.append("=" * 80)

    for group_id, quality in quality_results.items():
        group_name = task_group_data[group_id]['info']['name']
        report_lines.append(f"\n{group_name}:")

        for var, fill_rate in quality.items():
            status = "✅" if fill_rate >= 70 else "⚠️ "
            report_lines.append(f"  {status} {var}: {fill_rate:.1f}%")

    report_lines.append("")

    # 样本量对比
    report_lines.append("=" * 80)
    report_lines.append("4. 样本量对比")
    report_lines.append("=" * 80)

    total_grouped = sum(f['sample_count'] for f in saved_files)
    report_lines.append(f"原始样本: {len(df)}")
    report_lines.append(f"分组后总数: {total_grouped}")
    report_lines.append(f"样本完整性: {'✅ 一致' if total_grouped == len(df) else '⚠️  不一致'}")

    report_lines.append("")
    report_lines.append("=" * 80)

    # 写入报告
    report_content = "\n".join(report_lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ 任务分组报告已保存: {REPORT_FILE}")

    # 打印到控制台
    print("\n" + report_content)


def main():
    """主函数"""
    print("=" * 80)
    print("阶段3: 任务分组 (Task Grouping)")
    print("=" * 80)

    try:
        # 1. 加载数据
        df = load_data(INPUT_FILE)

        # 2. 验证repository列
        repos = verify_repository_column(df)

        # 3. 分割到任务组
        task_group_data = split_into_task_groups(df)

        # 4. 验证分割完整性
        is_complete = verify_split_completeness(df, task_group_data)

        # 5. 分析任务组质量
        quality_results = analyze_task_group_quality(task_group_data)

        # 6. 保存任务组数据
        saved_files = save_task_groups(task_group_data)

        # 7. 生成报告
        generate_grouping_report(df, task_group_data, quality_results, saved_files)

        print("\n" + "=" * 80)
        print("✅ 阶段3完成: 任务分组成功")
        print("=" * 80)
        print(f"\n生成的任务组文件:")
        for file_info in saved_files:
            print(f"  - {file_info['file_path'].name} ({file_info['sample_count']} 样本)")

        if is_complete:
            return 0
        else:
            print("\n⚠️  警告: 样本完整性检查失败")
            return 1

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
