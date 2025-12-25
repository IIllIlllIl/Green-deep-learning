#!/usr/bin/env python3
"""
阶段5: 变量选择 (Variable Selection)

功能:
1. 为每个任务组选择13-16个关键变量
2. 基于填充率、相关性和因果分析需求
3. 包含：元信息、超参数、中介变量、能耗输出、性能指标
4. 输出: 4个任务组的最终分析数据

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
REPORT_FILE = PROCESSED_DIR / "stage5_variable_selection_report.txt"

# 任务组配置
TASK_CONFIGS = {
    'image_classification': {
        'input': PROCESSED_DIR / 'stage4_image_classification.csv',
        'output': PROCESSED_DIR / 'stage5_image_classification.csv',
        'name': '图像分类',
        'variables': {
            # 元信息 (4)
            'metadata': [
                'experiment_id',
                'repository',
                'model',
                'timestamp'
            ],
            # 超参数 (4)
            'hyperparameters': [
                'hyperparam_learning_rate',
                'hyperparam_batch_size',
                'training_duration',
                'seed'
            ],
            # One-Hot编码 (2)
            'onehot': [
                'is_mnist',
                'is_cifar10'
            ],
            # 能耗中介变量 (5)
            'mediators': [
                'gpu_util_avg',
                'gpu_temp_max',
                'cpu_pkg_ratio',
                'gpu_power_fluctuation',
                'gpu_temp_fluctuation'
            ],
            # 能耗输出 (2)
            'energy_outputs': [
                'energy_cpu_total_joules',
                'energy_gpu_total_joules'
            ],
            # 性能指标 (1)
            'performance': [
                'perf_test_accuracy'
            ]
        }
    },
    'person_reid': {
        'input': PROCESSED_DIR / 'stage4_person_reid.csv',
        'output': PROCESSED_DIR / 'stage5_person_reid.csv',
        'name': 'Person_reID检索',
        'variables': {
            # 元信息 (4)
            'metadata': [
                'experiment_id',
                'repository',
                'model',
                'timestamp'
            ],
            # 超参数 (3)
            'hyperparameters': [
                'hyperparam_learning_rate',
                'hyperparam_dropout',
                'training_duration'
            ],
            # One-Hot编码 (3)
            'onehot': [
                'is_densenet121',
                'is_hrnet18',
                'is_pcb'
            ],
            # 能耗中介变量 (5)
            'mediators': [
                'gpu_util_avg',
                'gpu_temp_max',
                'cpu_pkg_ratio',
                'gpu_power_fluctuation',
                'gpu_temp_fluctuation'
            ],
            # 能耗输出 (2)
            'energy_outputs': [
                'energy_cpu_total_joules',
                'energy_gpu_total_joules'
            ],
            # 性能指标 (3)
            'performance': [
                'perf_map',
                'perf_rank1',
                'perf_rank5'
            ]
        }
    },
    'vulberta': {
        'input': PROCESSED_DIR / 'stage4_vulberta.csv',
        'output': PROCESSED_DIR / 'stage5_vulberta.csv',
        'name': 'VulBERTa漏洞检测',
        'variables': {
            # 元信息 (4)
            'metadata': [
                'experiment_id',
                'repository',
                'model',
                'timestamp'
            ],
            # 超参数 (2)
            'hyperparameters': [
                'hyperparam_learning_rate',
                'training_duration'
            ],
            # One-Hot编码 (0) - 单一模型
            'onehot': [],
            # 能耗中介变量 (5)
            'mediators': [
                'gpu_util_avg',
                'gpu_temp_max',
                'cpu_pkg_ratio',
                'gpu_power_fluctuation',
                'gpu_temp_fluctuation'
            ],
            # 能耗输出 (2)
            'energy_outputs': [
                'energy_cpu_total_joules',
                'energy_gpu_total_joules'
            ],
            # 性能指标 (1)
            'performance': [
                'perf_eval_loss'
            ]
        }
    },
    'bug_localization': {
        'input': PROCESSED_DIR / 'stage4_bug_localization.csv',
        'output': PROCESSED_DIR / 'stage5_bug_localization.csv',
        'name': 'Bug定位',
        'variables': {
            # 元信息 (4)
            'metadata': [
                'experiment_id',
                'repository',
                'model',
                'timestamp'
            ],
            # 超参数 (2)
            'hyperparameters': [
                'hyperparam_learning_rate',
                'training_duration'
            ],
            # One-Hot编码 (0) - 单一模型
            'onehot': [],
            # 能耗中介变量 (5)
            'mediators': [
                'gpu_util_avg',
                'gpu_temp_max',
                'cpu_pkg_ratio',
                'gpu_power_fluctuation',
                'gpu_temp_fluctuation'
            ],
            # 能耗输出 (2)
            'energy_outputs': [
                'energy_cpu_total_joules',
                'energy_gpu_total_joules'
            ],
            # 性能指标 (2)
            'performance': [
                'perf_top1_accuracy',
                'perf_top5_accuracy'
            ]
        }
    }
}


def load_task_group(filepath, task_name):
    """加载任务组数据"""
    print(f"\n📂 加载 {task_name}...")
    df = pd.read_csv(filepath)
    print(f"   原始: {len(df)}行 × {len(df.columns)}列")
    return df


def select_variables(df, var_config, task_name):
    """
    选择变量并验证

    参数:
        df: DataFrame
        var_config: 变量配置字典
        task_name: 任务组名称

    返回:
        选择后的DataFrame, 选择的列列表
    """
    print(f"\n🔧 选择变量 ({task_name})...")

    # 收集所有选择的变量
    all_vars = []
    for category, vars_list in var_config.items():
        all_vars.extend(vars_list)

    print(f"   目标变量数: {len(all_vars)}")
    print(f"   类别数: {len(var_config)}")

    # 验证变量存在性
    missing_vars = []
    existing_vars = []

    for var in all_vars:
        if var in df.columns:
            existing_vars.append(var)
        else:
            missing_vars.append(var)

    if missing_vars:
        print(f"\n   ⚠️  缺失变量 ({len(missing_vars)}):")
        for var in missing_vars:
            print(f"      - {var}")
    else:
        print(f"   ✅ 所有变量存在")

    # 选择列
    df_selected = df[existing_vars].copy()

    # 统计各类别变量数
    print(f"\n   变量类别分布:")
    for category, vars_list in var_config.items():
        existing_count = sum(1 for v in vars_list if v in existing_vars)
        total_count = len(vars_list)
        print(f"      {category:20s}: {existing_count}/{total_count}")

    return df_selected, existing_vars


def analyze_selected_variables(df, selected_vars, task_name):
    """
    分析选择的变量质量

    检查:
    1. 填充率
    2. 唯一值数量（变异性）
    3. 数据类型
    """
    print(f"\n📊 变量质量分析 ({task_name})...")

    # 填充率分析
    fill_rates = {}
    for var in selected_vars:
        if var in df.columns:
            fill_rate = df[var].notna().sum() / len(df) * 100
            fill_rates[var] = fill_rate

    # 按填充率分类
    high_fill = [v for v, r in fill_rates.items() if r >= 90]
    medium_fill = [v for v, r in fill_rates.items() if 50 <= r < 90]
    low_fill = [v for v, r in fill_rates.items() if r < 50]

    print(f"\n   填充率分布:")
    print(f"      高填充 (≥90%): {len(high_fill)} 个")
    print(f"      中填充 (50-90%): {len(medium_fill)} 个")
    print(f"      低填充 (<50%): {len(low_fill)} 个")

    if low_fill:
        print(f"\n   ⚠️  低填充率变量:")
        for var in low_fill:
            print(f"      - {var}: {fill_rates[var]:.1f}%")

    # 唯一值分析（数值型变量）
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_selected = [v for v in selected_vars if v in numeric_vars]

    low_variance = []
    for var in numeric_selected:
        unique_count = df[var].nunique()
        if unique_count < 5:
            low_variance.append((var, unique_count))

    if low_variance:
        print(f"\n   ⚠️  低变异性变量 (<5唯一值):")
        for var, count in low_variance:
            print(f"      - {var}: {count} 唯一值")

    # 计算整体质量分数
    avg_fill_rate = np.mean(list(fill_rates.values()))
    quality_score = "优秀" if avg_fill_rate >= 80 else "良好" if avg_fill_rate >= 60 else "一般"

    print(f"\n   整体评估:")
    print(f"      平均填充率: {avg_fill_rate:.1f}%")
    print(f"      质量评级: {quality_score}")

    return fill_rates


def save_selected_data(df, output_file, task_name):
    """保存选择后的数据"""
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


def generate_selection_report(results, all_fill_rates):
    """生成变量选择报告"""
    print(f"\n📊 生成变量选择报告...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("阶段5: 变量选择报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 任务组摘要
    report_lines.append("=" * 80)
    report_lines.append("1. 任务组变量选择摘要")
    report_lines.append("=" * 80)

    for result in results:
        task_id = result['task_id']
        task_config = TASK_CONFIGS[task_id]
        task_name = result['task_name']
        fill_rates = all_fill_rates[task_id]

        report_lines.append(f"\n{task_name}:")
        report_lines.append(f"  输出文件: {result['file_path'].name}")
        report_lines.append(f"  样本数: {result['row_count']}")
        report_lines.append(f"  变量数: {result['column_count']}")
        report_lines.append(f"  文件大小: {result['file_size_kb']:.1f} KB")

        # 变量类别统计
        report_lines.append(f"  变量类别:")
        for category, vars_list in task_config['variables'].items():
            report_lines.append(f"    - {category}: {len(vars_list)}个")

        # 填充率统计
        avg_fill = np.mean(list(fill_rates.values()))
        report_lines.append(f"  平均填充率: {avg_fill:.1f}%")

    report_lines.append("")

    # 统计摘要
    report_lines.append("=" * 80)
    report_lines.append("2. 整体统计")
    report_lines.append("=" * 80)

    total_samples = sum(r['row_count'] for r in results)
    avg_vars = np.mean([r['column_count'] for r in results])

    report_lines.append(f"任务组总数: {len(results)}")
    report_lines.append(f"总样本数: {total_samples}")
    report_lines.append(f"平均变量数: {avg_vars:.1f}")
    report_lines.append(f"变量数范围: {min(r['column_count'] for r in results)}-{max(r['column_count'] for r in results)}")

    # 变量选择设计说明
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("3. 变量选择设计")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("所有任务组包含:")
    report_lines.append("  1. 元信息 (4): experiment_id, repository, model, timestamp")
    report_lines.append("  2. 能耗中介 (5): gpu_util_avg, gpu_temp_max, cpu_pkg_ratio, gpu_power_fluctuation, gpu_temp_fluctuation")
    report_lines.append("  3. 能耗输出 (2): energy_cpu_total_joules, energy_gpu_total_joules")
    report_lines.append("")
    report_lines.append("任务特定变量:")
    report_lines.append("  - 图像分类: 4超参数 + 2 One-Hot + 1性能 (18变量)")
    report_lines.append("  - Person_reID: 3超参数 + 3 One-Hot + 3性能 (20变量)")
    report_lines.append("  - VulBERTa: 2超参数 + 0 One-Hot + 1性能 (14变量)")
    report_lines.append("  - Bug定位: 2超参数 + 0 One-Hot + 2性能 (15变量)")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("✅ 阶段5: 变量选择完成")
    report_lines.append("=" * 80)

    # 写入报告
    report_content = "\n".join(report_lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ 变量选择报告已保存: {REPORT_FILE}")

    # 打印到控制台
    print("\n" + report_content)


def main():
    """主函数"""
    print("=" * 80)
    print("阶段5: 变量选择 (Variable Selection)")
    print("=" * 80)

    try:
        results = []
        all_fill_rates = {}

        for task_id, task_config in TASK_CONFIGS.items():
            # 1. 加载数据
            df = load_task_group(task_config['input'], task_config['name'])

            # 2. 选择变量
            df_selected, selected_vars = select_variables(
                df,
                task_config['variables'],
                task_config['name']
            )

            # 3. 分析变量质量
            fill_rates = analyze_selected_variables(
                df_selected,
                selected_vars,
                task_config['name']
            )

            all_fill_rates[task_id] = fill_rates

            # 4. 保存数据
            result = save_selected_data(
                df_selected,
                task_config['output'],
                task_config['name']
            )

            result['task_id'] = task_id
            results.append(result)

        # 5. 生成报告
        generate_selection_report(results, all_fill_rates)

        print("\n" + "=" * 80)
        print("✅ 阶段5完成: 变量选择成功")
        print("=" * 80)
        print(f"\n生成的最终分析文件:")
        for result in results:
            print(f"  - {result['file_path'].name} ({result['column_count']}变量, {result['row_count']}样本)")

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
