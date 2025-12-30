#!/usr/bin/env python3
"""
阶段4: One-Hot编码 (One-Hot Encoding)

功能:
1. 为每个任务组添加One-Hot编码列
2. 图像分类组: is_mnist, is_cifar10
3. Person_reID组: is_densenet121, is_hrnet18, is_pcb
4. VulBERTa和Bug定位: 单一repository/model，无需编码
5. 输出: 4个任务组的编码后CSV文件

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
REPORT_FILE = PROCESSED_DIR / "stage4_onehot_report.txt"

# 任务组文件
TASK_FILES = {
    'image_classification': {
        'input': PROCESSED_DIR / 'stage3_image_classification.csv',
        'output': PROCESSED_DIR / 'stage4_image_classification.csv',
        'name': '图像分类',
        'onehot_config': {
            'type': 'repository',
            'columns': {
                'is_mnist': 'examples',
                'is_cifar10': 'pytorch_resnet_cifar10'
            }
        }
    },
    'person_reid': {
        'input': PROCESSED_DIR / 'stage3_person_reid.csv',
        'output': PROCESSED_DIR / 'stage4_person_reid.csv',
        'name': 'Person_reID检索',
        'onehot_config': {
            'type': 'model',
            'columns': {
                'is_densenet121': 'densenet121',
                'is_hrnet18': 'hrnet18',
                'is_pcb': 'pcb'
            }
        }
    },
    'vulberta': {
        'input': PROCESSED_DIR / 'stage3_vulberta.csv',
        'output': PROCESSED_DIR / 'stage4_vulberta.csv',
        'name': 'VulBERTa漏洞检测',
        'onehot_config': None  # 单一repository/model，无需编码
    },
    'bug_localization': {
        'input': PROCESSED_DIR / 'stage3_bug_localization.csv',
        'output': PROCESSED_DIR / 'stage4_bug_localization.csv',
        'name': 'Bug定位',
        'onehot_config': None  # 单一repository/model，无需编码
    }
}


def load_task_group(filepath, task_name):
    """加载任务组数据"""
    print(f"\n📂 加载 {task_name}...")
    df = pd.read_csv(filepath)
    print(f"   行数: {len(df)}, 列数: {len(df.columns)}")
    return df


def add_onehot_encoding(df, onehot_config, task_name):
    """
    添加One-Hot编码列

    参数:
        df: DataFrame
        onehot_config: One-Hot配置 {'type': 'repository'/'model', 'columns': {...}}
        task_name: 任务组名称

    返回:
        编码后的DataFrame
    """
    if onehot_config is None:
        print(f"   ℹ️  {task_name}: 单一repository/model，无需One-Hot编码")
        return df, 0

    print(f"\n🔧 添加One-Hot编码 ({task_name})...")

    encoding_type = onehot_config['type']  # 'repository' or 'model'
    columns_map = onehot_config['columns']  # {new_col: value}

    # 根据类型选择源列
    source_column = encoding_type

    print(f"   编码类型: {encoding_type}")
    print(f"   源列: {source_column}")
    print(f"   新增列数: {len(columns_map)}")

    # 创建One-Hot列
    added_columns = []
    for new_col, target_value in columns_map.items():
        df[new_col] = (df[source_column] == target_value).astype(int)
        added_columns.append(new_col)

        # 统计
        count = df[new_col].sum()
        percentage = (count / len(df)) * 100
        print(f"   ✅ {new_col}: {count} 行 ({percentage:.1f}%)")

    return df, len(added_columns)


def verify_onehot_encoding(df, onehot_config, task_name):
    """
    验证One-Hot编码的正确性

    检查:
    1. 列是否二值化 (0或1)
    2. 每行是否恰好有一个1 (互斥性)
    3. 总和是否等于行数
    """
    if onehot_config is None:
        return True

    print(f"\n🔍 验证One-Hot编码 ({task_name})...")

    columns_map = onehot_config['columns']
    onehot_cols = list(columns_map.keys())

    all_valid = True

    # 1. 检查二值化
    for col in onehot_cols:
        unique_values = df[col].unique()
        if not set(unique_values).issubset({0, 1}):
            print(f"   ❌ {col}: 包含非二值数据 {unique_values}")
            all_valid = False
        else:
            print(f"   ✅ {col}: 二值化正确 (0或1)")

    # 2. 检查互斥性 (每行恰好有一个1)
    row_sums = df[onehot_cols].sum(axis=1)
    if (row_sums != 1).any():
        invalid_count = (row_sums != 1).sum()
        print(f"   ⚠️  互斥性违规: {invalid_count} 行不满足恰好一个1")
        all_valid = False
    else:
        print(f"   ✅ 互斥性: 所有行恰好有一个1")

    # 3. 检查总和
    total_sum = df[onehot_cols].sum().sum()
    expected_sum = len(df)
    if total_sum == expected_sum:
        print(f"   ✅ 总和验证: {total_sum} = {expected_sum}")
    else:
        print(f"   ❌ 总和验证: {total_sum} ≠ {expected_sum}")
        all_valid = False

    return all_valid


def save_encoded_data(df, output_file, task_name):
    """保存编码后的数据"""
    df.to_csv(output_file, index=False)
    file_size = output_file.stat().st_size / 1024

    print(f"\n💾 保存 {task_name}:")
    print(f"   文件: {output_file.name}")
    print(f"   行数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"   大小: {file_size:.1f} KB")

    return {
        'task_name': task_name,
        'file_path': output_file,
        'row_count': len(df),
        'column_count': len(df.columns),
        'file_size_kb': file_size
    }


def generate_onehot_report(results):
    """生成One-Hot编码报告"""
    print(f"\n📊 生成One-Hot编码报告...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("阶段4: One-Hot编码报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 任务组摘要
    report_lines.append("=" * 80)
    report_lines.append("1. 任务组编码摘要")
    report_lines.append("=" * 80)

    for result in results:
        report_lines.append(f"\n{result['task_name']}:")
        report_lines.append(f"  输出文件: {result['file_path'].name}")
        report_lines.append(f"  行数: {result['row_count']}")
        report_lines.append(f"  列数: {result['column_count']} (新增: {result.get('added_columns', 0)}列)")
        report_lines.append(f"  文件大小: {result['file_size_kb']:.1f} KB")

        if result.get('added_columns', 0) > 0:
            report_lines.append(f"  验证结果: {'✅ 通过' if result.get('validation', False) else '❌ 失败'}")

    report_lines.append("")

    # 统计摘要
    report_lines.append("=" * 80)
    report_lines.append("2. 编码统计")
    report_lines.append("=" * 80)

    total_samples = sum(r['row_count'] for r in results)
    tasks_with_encoding = sum(1 for r in results if r.get('added_columns', 0) > 0)
    total_onehot_cols = sum(r.get('added_columns', 0) for r in results)

    report_lines.append(f"任务组总数: {len(results)}")
    report_lines.append(f"需要编码的任务组: {tasks_with_encoding}")
    report_lines.append(f"新增One-Hot列总数: {total_onehot_cols}")
    report_lines.append(f"总样本数: {total_samples}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("✅ 阶段4: One-Hot编码完成")
    report_lines.append("=" * 80)

    # 写入报告
    report_content = "\n".join(report_lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ One-Hot编码报告已保存: {REPORT_FILE}")

    # 打印到控制台
    print("\n" + report_content)


def main():
    """主函数"""
    print("=" * 80)
    print("阶段4: One-Hot编码 (One-Hot Encoding)")
    print("=" * 80)

    try:
        results = []

        for task_id, task_config in TASK_FILES.items():
            # 1. 加载数据
            df = load_task_group(task_config['input'], task_config['name'])

            # 2. 添加One-Hot编码
            df_encoded, added_count = add_onehot_encoding(
                df,
                task_config['onehot_config'],
                task_config['name']
            )

            # 3. 验证编码
            validation_passed = verify_onehot_encoding(
                df_encoded,
                task_config['onehot_config'],
                task_config['name']
            )

            # 4. 保存数据
            result = save_encoded_data(
                df_encoded,
                task_config['output'],
                task_config['name']
            )

            result['added_columns'] = added_count
            result['validation'] = validation_passed
            results.append(result)

        # 5. 生成报告
        generate_onehot_report(results)

        print("\n" + "=" * 80)
        print("✅ 阶段4完成: One-Hot编码成功")
        print("=" * 80)

        # 检查是否所有验证通过
        all_validated = all(r.get('validation', True) for r in results)

        if all_validated:
            return 0
        else:
            print("\n⚠️  警告: 部分任务组验证失败")
            return 1

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
