#!/usr/bin/env python3
"""
阶段6: 归一化 (Normalization)

功能:
1. 使用StandardScaler标准化数值变量
2. 保留元信息列（不标准化）
3. 保留One-Hot编码列（不标准化）
4. 对超参数、能耗中介变量、能耗输出、性能指标进行标准化
5. 保存标准化参数（mean, std）
6. 输出: 4个任务组的标准化CSV文件

作者: Analysis Module Team
日期: 2025-12-23
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
import pickle

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据路径
DATA_DIR = PROJECT_ROOT / "data" / "energy_research" / "processed"
OUTPUT_DIR = DATA_DIR
SCALERS_DIR = DATA_DIR / "scalers"
SCALERS_DIR.mkdir(parents=True, exist_ok=True)

# 任务组定义
TASK_GROUPS = {
    'image_classification': '图像分类',
    'person_reid': 'Person_reID',
    'vulberta': 'VulBERTa',
    'bug_localization': 'Bug定位'
}


def identify_column_types(df):
    """
    识别列的类型（元信息/One-Hot/数值）

    Returns:
        dict: {
            'meta': [...],      # 元信息列（不标准化）
            'onehot': [...],    # One-Hot编码列（不标准化）
            'numeric': [...]    # 数值列（需要标准化）
        }
    """
    column_types = {
        'meta': [],
        'onehot': [],
        'numeric': []
    }

    # 元信息列（固定）
    meta_keywords = ['experiment_id', 'timestamp', 'repository', 'model', 'mode']

    for col in df.columns:
        # 元信息列
        if any(keyword in col.lower() for keyword in meta_keywords):
            column_types['meta'].append(col)
        # One-Hot编码列
        elif col.startswith('is_'):
            column_types['onehot'].append(col)
        # 数值列
        elif df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            column_types['numeric'].append(col)
        else:
            # 未知类型，暂时归为元信息
            column_types['meta'].append(col)

    return column_types


def standardize_numeric_columns(df, numeric_cols):
    """
    使用StandardScaler标准化数值列

    Args:
        df: DataFrame
        numeric_cols: 需要标准化的列名列表

    Returns:
        df_scaled: 标准化后的DataFrame
        scaler_params: 标准化参数 {col: {'mean': x, 'std': y}}
    """
    df_scaled = df.copy()
    scaler_params = {}

    for col in numeric_cols:
        # 只对非空值进行标准化
        non_null_mask = df[col].notna()

        if non_null_mask.sum() > 0:  # 至少有一个非空值
            # 计算均值和标准差
            mean = df.loc[non_null_mask, col].mean()
            std = df.loc[non_null_mask, col].std()

            # 避免除以零
            if std == 0 or pd.isna(std):
                print(f"  ⚠️  警告: {col} 标准差为0或NaN，跳过标准化（保留原值）")
                scaler_params[col] = {'mean': mean, 'std': 1.0, 'skipped': True}
            else:
                # 标准化: (x - mean) / std
                df_scaled.loc[non_null_mask, col] = (
                    df.loc[non_null_mask, col] - mean
                ) / std

                scaler_params[col] = {'mean': mean, 'std': std, 'skipped': False}

                # 验证标准化结果
                scaled_mean = df_scaled.loc[non_null_mask, col].mean()
                scaled_std = df_scaled.loc[non_null_mask, col].std()

                if not (abs(scaled_mean) < 1e-6 and abs(scaled_std - 1.0) < 1e-6):
                    print(f"  ⚠️  警告: {col} 标准化后均值={scaled_mean:.6f}, 标准差={scaled_std:.6f}")
        else:
            print(f"  ⚠️  警告: {col} 全为NaN，跳过标准化")
            scaler_params[col] = {'mean': 0.0, 'std': 1.0, 'all_nan': True}

    return df_scaled, scaler_params


def normalize_task_group(task_name, task_display_name):
    """
    归一化单个任务组的数据

    Args:
        task_name: 任务组名称（英文）
        task_display_name: 任务组显示名称（中文）

    Returns:
        bool: 成功返回True，失败返回False
    """
    print(f"\n{'='*80}")
    print(f"任务组: {task_display_name} ({task_name})")
    print(f"{'='*80}\n")

    # 输入输出文件
    input_file = OUTPUT_DIR / f"stage5_{task_name}.csv"
    output_file = OUTPUT_DIR / f"stage6_{task_name}.csv"
    scaler_file = SCALERS_DIR / f"scaler_{task_name}.pkl"

    # 1. 加载数据
    if not input_file.exists():
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return False

    print(f"📂 加载数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")

    # 2. 识别列类型
    print(f"\n🔍 识别列类型...")
    column_types = identify_column_types(df)

    print(f"\n  元信息列 ({len(column_types['meta'])} 个):")
    for col in column_types['meta']:
        print(f"    - {col}")

    print(f"\n  One-Hot编码列 ({len(column_types['onehot'])} 个):")
    for col in column_types['onehot']:
        print(f"    - {col}")

    print(f"\n  数值列 ({len(column_types['numeric'])} 个):")
    for col in column_types['numeric']:
        non_null = df[col].notna().sum()
        fill_rate = non_null / len(df) * 100
        print(f"    - {col}: {non_null}/{len(df)} ({fill_rate:.1f}%)")

    # 3. 标准化数值列
    print(f"\n📐 标准化数值列...")
    df_scaled, scaler_params = standardize_numeric_columns(df, column_types['numeric'])

    print(f"\n✅ 标准化完成:")
    print(f"  - 标准化列数: {len([p for p in scaler_params.values() if not p.get('skipped', False) and not p.get('all_nan', False)])}")
    print(f"  - 跳过列数: {len([p for p in scaler_params.values() if p.get('skipped', False)])}")
    print(f"  - 全NaN列数: {len([p for p in scaler_params.values() if p.get('all_nan', False)])}")

    # 4. 验证数据范围
    print(f"\n🔍 验证标准化后的数据范围...")
    for col in column_types['numeric']:
        if col in scaler_params and not scaler_params[col].get('all_nan', False):
            non_null_mask = df_scaled[col].notna()
            if non_null_mask.sum() > 0:
                col_min = df_scaled.loc[non_null_mask, col].min()
                col_max = df_scaled.loc[non_null_mask, col].max()
                col_mean = df_scaled.loc[non_null_mask, col].mean()
                col_std = df_scaled.loc[non_null_mask, col].std()

                # 检查是否合理
                if abs(col_mean) > 0.1:
                    print(f"  ⚠️  {col}: 均值={col_mean:.3f} (应接近0)")
                if abs(col_std - 1.0) > 0.1 and not scaler_params[col].get('skipped', False):
                    print(f"  ⚠️  {col}: 标准差={col_std:.3f} (应接近1)")

    print(f"✅ 数据范围验证完成")

    # 5. 保存标准化参数
    print(f"\n💾 保存标准化参数...")
    scaler_info = {
        'task_name': task_name,
        'task_display_name': task_display_name,
        'scaler_params': scaler_params,
        'column_types': column_types,
        'num_samples': len(df),
        'num_features': len(column_types['numeric']),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler_info, f)

    print(f"✅ 标准化参数已保存: {scaler_file}")

    # 6. 保存归一化数据
    print(f"\n💾 保存归一化数据...")
    df_scaled.to_csv(output_file, index=False)

    file_size_kb = output_file.stat().st_size / 1024
    print(f"✅ 归一化数据已保存: {output_file}")
    print(f"  - 行数: {len(df_scaled)}")
    print(f"  - 列数: {len(df_scaled.columns)}")
    print(f"  - 文件大小: {file_size_kb:.1f} KB")

    return True


def generate_normalization_report(results):
    """生成归一化报告"""
    print(f"\n{'='*80}")
    print(f"阶段6: 归一化报告")
    print(f"{'='*80}\n")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("阶段6: 归一化 (Normalization) 报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 汇总统计
    report_lines.append("=" * 80)
    report_lines.append("1. 归一化汇总")
    report_lines.append("=" * 80)

    total_tasks = len(results)
    success_tasks = sum(1 for r in results.values() if r['success'])

    report_lines.append(f"总任务组数: {total_tasks}")
    report_lines.append(f"成功任务组: {success_tasks}")
    report_lines.append(f"失败任务组: {total_tasks - success_tasks}")
    report_lines.append("")

    # 各任务组详情
    report_lines.append("=" * 80)
    report_lines.append("2. 各任务组详情")
    report_lines.append("=" * 80)

    for task_name, result in results.items():
        task_display = TASK_GROUPS[task_name]
        status = "✅ 成功" if result['success'] else "❌ 失败"

        report_lines.append(f"\n{task_display} ({task_name}): {status}")

        if result['success']:
            report_lines.append(f"  输出文件: stage6_{task_name}.csv")
            report_lines.append(f"  标准化参数: scaler_{task_name}.pkl")
        else:
            report_lines.append(f"  错误: {result.get('error', '未知错误')}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("✅ 阶段6完成")
    report_lines.append("=" * 80)

    # 写入报告文件
    report_file = OUTPUT_DIR / "stage6_normalization_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    # 打印到控制台
    print("\n".join(report_lines))
    print(f"\n📄 报告已保存: {report_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("阶段6: 归一化 (Normalization)")
    print("=" * 80)

    results = {}

    # 对每个任务组进行归一化
    for task_name, task_display_name in TASK_GROUPS.items():
        try:
            success = normalize_task_group(task_name, task_display_name)
            results[task_name] = {
                'success': success,
                'task_display': task_display_name
            }
        except Exception as e:
            print(f"\n❌ 错误: 处理任务组 {task_name} 时发生异常")
            print(f"  异常信息: {str(e)}")
            import traceback
            traceback.print_exc()

            results[task_name] = {
                'success': False,
                'task_display': task_display_name,
                'error': str(e)
            }

    # 生成归一化报告
    generate_normalization_report(results)

    # 返回状态
    all_success = all(r['success'] for r in results.values())

    if all_success:
        print("\n" + "=" * 80)
        print("✅ 阶段6完成: 所有任务组归一化成功")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("⚠️  阶段6完成: 部分任务组归一化失败")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
