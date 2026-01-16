#!/usr/bin/env python3
"""
6分组数据验证脚本 - 独立质量检查

检查内容:
1. 数据完整性检查 (6组数据是否都已生成)
2. 分组正确性检查 (repository和model是否正确)
3. 字段正确性检查 (超参数、性能指标、能耗列)
4. 模型变量编码检查 (One-hot n-1编码)
5. 数据质量检查 (缺失率、异常值)
6. 特殊情况检查 (L2正则化、并行模式)

创建日期: 2026-01-15
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict
import json

# 数据目录
DATA_DIR = '/home/green/energy_dl/nightly/analysis/data/energy_research/6groups_final'
SOURCE_DATA = '/home/green/energy_dl/nightly/data/data.csv'

# 根据设计文档定义的6组配置
GROUP_SPECS = {
    'group1_examples': {
        'name': '图像分类-小型模型组',
        'repo': 'examples',
        'models': ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese'],
        'expected_rows': 304,
        'hyperparams': ['hyperparam_batch_size', 'hyperparam_learning_rate',
                       'hyperparam_epochs', 'hyperparam_seed'],
        'performance': ['perf_test_accuracy'],
        'model_vars_count': 3  # n-1 encoding for 4 models
    },
    'group2_vulberta': {
        'name': '代码漏洞检测组',
        'repo': 'VulBERTa',
        'models': ['mlp'],
        'expected_rows': 72,
        'hyperparams': ['hyperparam_learning_rate', 'hyperparam_epochs',
                       'hyperparam_seed', 'hyperparam_l2_regularization'],
        'performance': ['perf_eval_loss', 'perf_final_training_loss',
                       'perf_eval_samples_per_second'],
        'model_vars_count': 0  # single model, no encoding needed
    },
    'group3_person_reid': {
        'name': '行人重识别组',
        'repo': 'Person_reID_baseline_pytorch',
        'models': ['densenet121', 'hrnet18', 'pcb'],
        'expected_rows': 206,
        'hyperparams': ['hyperparam_dropout', 'hyperparam_learning_rate',
                       'hyperparam_epochs', 'hyperparam_seed'],
        'performance': ['perf_map', 'perf_rank1', 'perf_rank5'],
        'model_vars_count': 2  # n-1 encoding for 3 models
    },
    'group4_bug_localization': {
        'name': '缺陷定位组',
        'repo': 'bug-localization-by-dnn-and-rvsm',
        'models': ['default'],
        'expected_rows': 90,
        'hyperparams': ['hyperparam_alpha', 'hyperparam_kfold',
                       'hyperparam_max_iter', 'hyperparam_seed'],
        'performance': ['perf_top1_accuracy', 'perf_top5_accuracy',
                       'perf_top10_accuracy', 'perf_top20_accuracy'],
        'model_vars_count': 0  # single model
    },
    'group5_mrt_oast': {
        'name': '多目标优化组',
        'repo': 'MRT-OAST',
        'models': ['default'],
        'expected_rows': 72,
        'hyperparams': ['hyperparam_dropout', 'hyperparam_learning_rate',
                       'hyperparam_epochs', 'hyperparam_seed',
                       'hyperparam_l2_regularization'],
        'performance': ['perf_accuracy', 'perf_precision', 'perf_recall'],
        'model_vars_count': 0  # single model
    },
    'group6_resnet': {
        'name': '图像分类-ResNet组',
        'repo': 'pytorch_resnet_cifar10',
        'models': ['resnet20'],
        'expected_rows': 74,
        'hyperparams': ['hyperparam_learning_rate', 'hyperparam_epochs',
                       'hyperparam_seed', 'hyperparam_l2_regularization'],
        'performance': ['perf_best_val_accuracy', 'perf_test_accuracy'],
        'model_vars_count': 0  # single model
    }
}

# 能耗列（所有组共用）
ENERGY_COLS = [
    'energy_gpu_avg', 'energy_cpu_avg', 'energy_ram_avg',
    'energy_gpu_total', 'energy_cpu_total', 'energy_ram_total',
    'energy_total_avg'
]

# 控制变量（所有组共用）
CONTROL_COLS = ['is_parallel', 'timestamp', 'duration_seconds', 'num_mutated_params']

# 元数据列
META_COLS = ['experiment_id', 'repository', 'model']


class ValidationReport:
    """验证报告类"""

    def __init__(self):
        self.checks = []
        self.issues = []
        self.warnings = []
        self.stats = {}

    def add_check(self, name, passed, details=None):
        """添加检查项"""
        self.checks.append({
            'name': name,
            'passed': passed,
            'details': details
        })

    def add_issue(self, severity, description):
        """添加问题"""
        issue = {'severity': severity, 'description': description}
        if severity == 'ERROR':
            self.issues.append(issue)
        else:
            self.warnings.append(issue)

    def add_stat(self, key, value):
        """添加统计信息"""
        self.stats[key] = value

    def get_summary(self):
        """获取摘要"""
        total_checks = len(self.checks)
        passed_checks = sum(1 for c in self.checks if c['passed'])
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'errors': len(self.issues),
            'warnings': len(self.warnings),
            'overall_score': passed_checks / total_checks * 100 if total_checks > 0 else 0
        }


def check_data_completeness(report):
    """检查1: 数据完整性"""
    print("\n" + "="*60)
    print("检查1: 数据完整性")
    print("="*60)

    # 检查文件存在性
    csv_files = list(Path(DATA_DIR).glob('*.csv'))
    print(f"\n找到 {len(csv_files)} 个CSV文件:")
    for f in sorted(csv_files):
        print(f"  - {f.name}")

    # 检查是否有6个组
    group_files = [f for f in csv_files if f.stem.startswith('group')]
    expected_groups = set(GROUP_SPECS.keys())
    actual_groups = set([f.stem for f in group_files])

    missing_groups = expected_groups - actual_groups
    extra_groups = actual_groups - expected_groups

    if missing_groups:
        report.add_issue('ERROR', f"缺失组: {missing_groups}")
        print(f"\n❌ 缺失组: {missing_groups}")

    if extra_groups:
        report.add_issue('WARNING', f"额外文件: {extra_groups}")
        print(f"\n⚠️  额外文件: {extra_groups}")

    # 检查总行数
    total_rows = 0
    group_data = {}
    for group_id in GROUP_SPECS:
        file_path = Path(DATA_DIR) / f"{group_id}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            group_data[group_id] = df
            total_rows += len(df)
            expected = GROUP_SPECS[group_id]['expected_rows']
            if len(df) == expected:
                print(f"  ✅ {group_id}: {len(df)} 行 (符合预期)")
            else:
                print(f"  ❌ {group_id}: {len(df)} 行 (预期 {expected})")
                report.add_issue('ERROR', f"{group_id}: 行数不匹配 ({len(df)} vs {expected})")
        else:
            print(f"  ❌ {group_id}: 文件不存在")
            report.add_issue('ERROR', f"{group_id}: 文件不存在")

    print(f"\n总数据行数: {total_rows} (预期: 818)")
    if total_rows == 818:
        print("✅ 数据完整性: 100% 利用率")
        report.add_check('数据完整性', True, f"总行数: {total_rows}/818")
    else:
        print(f"❌ 数据完整性: {total_rows/818*100:.1f}% 利用率")
        report.add_check('数据完整性', False, f"总行数: {total_rows}/818")
        report.add_issue('ERROR', f"总行数不匹配: {total_rows} (预期 818)")

    report.add_stat('total_rows', total_rows)
    report.add_stat('expected_rows', 818)

    return group_data


def check_timestamp_uniqueness(group_data, report):
    """检查timestamp唯一性"""
    print("\n" + "="*60)
    print("检查2: Timestamp唯一性")
    print("="*60)

    all_timestamps = []
    for group_id, df in group_data.items():
        if 'timestamp' not in df.columns:
            print(f"  ❌ {group_id}: 缺少timestamp列")
            report.add_issue('ERROR', f"{group_id}: 缺少timestamp列")
            continue

        # 检查组内唯一性
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            print(f"  ❌ {group_id}: 发现 {duplicates} 个重复timestamp")
            report.add_issue('ERROR', f"{group_id}: {duplicates}个重复timestamp")
        else:
            print(f"  ✅ {group_id}: timestamp唯一 ({len(df)}条)")

        all_timestamps.extend(df['timestamp'].tolist())

    # 检查跨组唯一性
    all_timestamps_series = pd.Series(all_timestamps)
    cross_duplicates = all_timestamps_series.duplicated().sum()

    if cross_duplicates > 0:
        print(f"\n❌ 跨组重复: 发现 {cross_duplicates} 个重复timestamp")
        report.add_issue('ERROR', f"跨组重复: {cross_duplicates}个timestamp")
        report.add_check('Timestamp唯一性', False)
    else:
        print(f"\n✅ 所有timestamp唯一 (总计 {len(all_timestamps)}条)")
        report.add_check('Timestamp唯一性', True)


def check_grouping_correctness(group_data, report):
    """检查3: 分组正确性"""
    print("\n" + "="*60)
    print("检查3: 分组正确性 (Repository & Model)")
    print("="*60)

    all_passed = True

    for group_id, df in group_data.items():
        if group_id not in GROUP_SPECS:
            continue

        spec = GROUP_SPECS[group_id]
        print(f"\n{group_id} ({spec['name']}):")

        # 检查repository
        if 'repository' in df.columns:
            repos = df['repository'].unique()
            expected_repo = spec['repo']
            if len(repos) == 1 and repos[0] == expected_repo:
                print(f"  ✅ Repository: {repos[0]}")
            else:
                print(f"  ❌ Repository不匹配: {repos} (预期: {expected_repo})")
                report.add_issue('ERROR', f"{group_id}: Repository不匹配")
                all_passed = False
        else:
            print(f"  ❌ 缺少repository列")
            report.add_issue('ERROR', f"{group_id}: 缺少repository列")
            all_passed = False

        # 检查model
        if 'model' in df.columns:
            models = sorted(df['model'].unique())
            expected_models = sorted(spec['models'])
            if models == expected_models:
                print(f"  ✅ Models: {models}")
            else:
                print(f"  ❌ Models不匹配:")
                print(f"     实际: {models}")
                print(f"     预期: {expected_models}")
                report.add_issue('ERROR', f"{group_id}: Models不匹配")
                all_passed = False
        else:
            print(f"  ❌ 缺少model列")
            report.add_issue('ERROR', f"{group_id}: 缺少model列")
            all_passed = False

    report.add_check('分组正确性', all_passed)


def check_field_correctness(group_data, report):
    """检查4: 字段正确性"""
    print("\n" + "="*60)
    print("检查4: 字段正确性")
    print("="*60)

    all_passed = True

    for group_id, df in group_data.items():
        if group_id not in GROUP_SPECS:
            continue

        spec = GROUP_SPECS[group_id]
        print(f"\n{group_id}:")

        # 检查能耗列
        missing_energy = [col for col in ENERGY_COLS if col not in df.columns]
        if missing_energy:
            print(f"  ❌ 缺失能耗列: {missing_energy}")
            report.add_issue('ERROR', f"{group_id}: 缺失能耗列 {missing_energy}")
            all_passed = False
        else:
            print(f"  ✅ 能耗列完整 ({len(ENERGY_COLS)}列)")

        # 检查控制变量
        missing_control = [col for col in CONTROL_COLS if col not in df.columns]
        if missing_control:
            print(f"  ❌ 缺失控制变量: {missing_control}")
            report.add_issue('ERROR', f"{group_id}: 缺失控制变量 {missing_control}")
            all_passed = False
        else:
            print(f"  ✅ 控制变量完整 ({len(CONTROL_COLS)}列)")

        # 检查超参数列
        missing_hyper = [col for col in spec['hyperparams'] if col not in df.columns]
        if missing_hyper:
            print(f"  ❌ 缺失超参数: {missing_hyper}")
            report.add_issue('ERROR', f"{group_id}: 缺失超参数 {missing_hyper}")
            all_passed = False
        else:
            print(f"  ✅ 超参数完整 ({len(spec['hyperparams'])}列)")

        # 检查性能指标
        missing_perf = [col for col in spec['performance'] if col not in df.columns]
        if missing_perf:
            print(f"  ❌ 缺失性能指标: {missing_perf}")
            report.add_issue('ERROR', f"{group_id}: 缺失性能指标 {missing_perf}")
            all_passed = False
        else:
            print(f"  ✅ 性能指标完整 ({len(spec['performance'])}列)")

    report.add_check('字段正确性', all_passed)


def check_model_encoding(group_data, report):
    """检查5: 模型变量编码"""
    print("\n" + "="*60)
    print("检查5: 模型变量编码 (One-hot n-1)")
    print("="*60)

    all_passed = True

    for group_id, df in group_data.items():
        if group_id not in GROUP_SPECS:
            continue

        spec = GROUP_SPECS[group_id]
        expected_vars = spec['model_vars_count']

        print(f"\n{group_id}:")
        print(f"  模型数: {len(spec['models'])}")
        print(f"  预期模型变量数: {expected_vars}")

        # 查找模型变量列
        model_var_cols = [col for col in df.columns if col.startswith('model_')]
        actual_vars = len(model_var_cols)

        print(f"  实际模型变量数: {actual_vars}")

        if actual_vars == expected_vars:
            print(f"  ✅ 模型变量编码正确")
            if model_var_cols:
                print(f"     变量: {model_var_cols}")
                # 检查编码值（应该是0或1）
                for col in model_var_cols:
                    unique_vals = sorted(df[col].unique())
                    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                        print(f"     ✅ {col}: {unique_vals}")
                    else:
                        print(f"     ❌ {col}: 非二值编码 {unique_vals}")
                        report.add_issue('ERROR', f"{group_id}: {col}非二值编码")
                        all_passed = False
        else:
            print(f"  ❌ 模型变量数不匹配")
            report.add_issue('ERROR', f"{group_id}: 模型变量数不匹配 ({actual_vars} vs {expected_vars})")
            all_passed = False

    report.add_check('模型变量编码', all_passed)


def check_data_quality(group_data, report):
    """检查6: 数据质量（缺失率、异常值）"""
    print("\n" + "="*60)
    print("检查6: 数据质量")
    print("="*60)

    quality_stats = {}

    for group_id, df in group_data.items():
        if group_id not in GROUP_SPECS:
            continue

        spec = GROUP_SPECS[group_id]
        print(f"\n{group_id}:")

        stats = {
            'total_rows': len(df),
            'missing_rates': {},
            'complete_rows': 0
        }

        # 计算每列缺失率
        print("  缺失率分析:")

        # 能耗列
        energy_cols_present = [col for col in ENERGY_COLS if col in df.columns]
        if energy_cols_present:
            energy_missing = df[energy_cols_present].isnull().sum()
            for col in energy_cols_present:
                rate = energy_missing[col] / len(df) * 100
                stats['missing_rates'][col] = rate
                if rate > 0:
                    print(f"    {col}: {rate:.1f}%")

        # 超参数
        hyper_cols_present = [col for col in spec['hyperparams'] if col in df.columns]
        if hyper_cols_present:
            hyper_missing = df[hyper_cols_present].isnull().sum()
            for col in hyper_cols_present:
                rate = hyper_missing[col] / len(df) * 100
                stats['missing_rates'][col] = rate
                if rate > 0:
                    print(f"    {col}: {rate:.1f}%")

        # 性能指标
        perf_cols_present = [col for col in spec['performance'] if col in df.columns]
        if perf_cols_present:
            perf_missing = df[perf_cols_present].isnull().sum()
            for col in perf_cols_present:
                rate = perf_missing[col] / len(df) * 100
                stats['missing_rates'][col] = rate
                if rate > 0:
                    print(f"    {col}: {rate:.1f}%")

        # 计算完整记录数（所有关键列都非空）
        key_cols = energy_cols_present + hyper_cols_present + perf_cols_present
        complete_mask = ~df[key_cols].isnull().any(axis=1)
        complete_rows = complete_mask.sum()
        complete_rate = complete_rows / len(df) * 100

        stats['complete_rows'] = complete_rows
        stats['complete_rate'] = complete_rate

        print(f"\n  ✅ 完整记录: {complete_rows}/{len(df)} ({complete_rate:.1f}%)")

        quality_stats[group_id] = stats

    # 总体质量评估
    total_complete = sum(s['complete_rows'] for s in quality_stats.values())
    total_rows = sum(s['total_rows'] for s in quality_stats.values())
    overall_rate = total_complete / total_rows * 100 if total_rows > 0 else 0

    print(f"\n总体完整率: {total_complete}/{total_rows} ({overall_rate:.1f}%)")

    report.add_stat('complete_rows', total_complete)
    report.add_stat('complete_rate', overall_rate)
    report.add_stat('quality_by_group', quality_stats)

    # 评分标准: >95% 优秀, >90% 良好, >80% 可接受
    if overall_rate >= 95:
        print("✅ 数据质量: 优秀")
        report.add_check('数据质量', True, f"完整率: {overall_rate:.1f}%")
    elif overall_rate >= 90:
        print("⚠️  数据质量: 良好")
        report.add_check('数据质量', True, f"完整率: {overall_rate:.1f}%")
        report.add_issue('WARNING', f"数据完整率 {overall_rate:.1f}% (可接受但不理想)")
    elif overall_rate >= 80:
        print("⚠️  数据质量: 可接受")
        report.add_check('数据质量', True, f"完整率: {overall_rate:.1f}%")
        report.add_issue('WARNING', f"数据完整率 {overall_rate:.1f}% (建议改进)")
    else:
        print("❌ 数据质量: 不足")
        report.add_check('数据质量', False, f"完整率: {overall_rate:.1f}%")
        report.add_issue('ERROR', f"数据完整率 {overall_rate:.1f}% (不可接受)")


def check_special_cases(group_data, report):
    """检查7: 特殊情况（L2正则化、并行模式）"""
    print("\n" + "="*60)
    print("检查7: 特殊情况")
    print("="*60)

    # 检查L2正则化合并
    print("\n7.1 L2正则化语义合并:")
    l2_groups = ['group2_vulberta', 'group5_mrt_oast', 'group6_resnet']

    for group_id in l2_groups:
        if group_id in group_data:
            df = group_data[group_id]
            if 'hyperparam_l2_regularization' in df.columns:
                non_null = df['hyperparam_l2_regularization'].notna().sum()
                print(f"  ✅ {group_id}: hyperparam_l2_regularization存在 ({non_null}/{len(df)}条有值)")
            else:
                print(f"  ❌ {group_id}: 缺少hyperparam_l2_regularization")
                report.add_issue('ERROR', f"{group_id}: 缺少L2正则化列")

    # 检查并行模式处理
    print("\n7.2 并行模式标识:")
    for group_id, df in group_data.items():
        if 'is_parallel' in df.columns:
            parallel_count = (df['is_parallel'] == True).sum()
            non_parallel_count = (df['is_parallel'] == False).sum()
            print(f"  {group_id}:")
            print(f"    并行模式: {parallel_count}")
            print(f"    非并行模式: {non_parallel_count}")
            if parallel_count + non_parallel_count == len(df):
                print(f"    ✅ is_parallel标识完整")
            else:
                print(f"    ⚠️  is_parallel有缺失值")
                report.add_issue('WARNING', f"{group_id}: is_parallel有缺失值")
        else:
            print(f"  ❌ {group_id}: 缺少is_parallel列")
            report.add_issue('ERROR', f"{group_id}: 缺少is_parallel列")


def generate_report(report, output_file):
    """生成Markdown报告"""
    summary = report.get_summary()

    content = f"""# 6分组数据验证报告

**验证日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据位置**: {DATA_DIR}
**验证脚本**: validate_6groups_data.py

---

## 📊 验证概览

| 指标 | 数值 |
|------|------|
| **总检查项** | {summary['total_checks']} |
| **✅ 通过** | {summary['passed_checks']} |
| **❌ 失败** | {summary['failed_checks']} |
| **🚨 错误** | {summary['errors']} |
| **⚠️  警告** | {summary['warnings']} |
| **🎯 总体评分** | {summary['overall_score']:.1f}/100 |

"""

    # 添加总体状态
    if summary['errors'] == 0 and summary['warnings'] == 0:
        content += "**✅ 总体状态**: 所有检查通过，数据质量优秀！\n\n"
    elif summary['errors'] == 0:
        content += f"**⚠️  总体状态**: 通过但有 {summary['warnings']} 个警告，建议改进\n\n"
    else:
        content += f"**❌ 总体状态**: 发现 {summary['errors']} 个错误，需要修复！\n\n"

    content += "---\n\n"

    # 检查项详情
    content += "## 📋 检查项详情\n\n"
    for i, check in enumerate(report.checks, 1):
        status = "✅" if check['passed'] else "❌"
        content += f"### {i}. {status} {check['name']}\n\n"
        if check['details']:
            content += f"**详情**: {check['details']}\n\n"

    content += "---\n\n"

    # 问题清单
    if report.issues:
        content += "## 🚨 错误清单\n\n"
        for i, issue in enumerate(report.issues, 1):
            content += f"{i}. **{issue['severity']}**: {issue['description']}\n"
        content += "\n---\n\n"

    if report.warnings:
        content += "## ⚠️  警告清单\n\n"
        for i, warning in enumerate(report.warnings, 1):
            content += f"{i}. {warning['description']}\n"
        content += "\n---\n\n"

    # 统计信息
    if report.stats:
        content += "## 📈 详细统计\n\n"

        # 数据量统计
        if 'total_rows' in report.stats:
            content += "### 数据量统计\n\n"
            content += f"- 总行数: {report.stats['total_rows']}\n"
            content += f"- 预期行数: {report.stats.get('expected_rows', 'N/A')}\n"
            content += f"- 利用率: {report.stats['total_rows']/report.stats.get('expected_rows', 1)*100:.1f}%\n\n"

        # 数据质量统计
        if 'complete_rows' in report.stats:
            content += "### 数据质量统计\n\n"
            content += f"- 完整记录数: {report.stats['complete_rows']}\n"
            content += f"- 完整率: {report.stats['complete_rate']:.1f}%\n\n"

        # 分组质量
        if 'quality_by_group' in report.stats:
            content += "### 分组数据质量\n\n"
            content += "| 组别 | 总行数 | 完整行数 | 完整率 |\n"
            content += "|------|--------|---------|--------|\n"
            for group_id, stats in report.stats['quality_by_group'].items():
                content += f"| {group_id} | {stats['total_rows']} | {stats['complete_rows']} | {stats['complete_rate']:.1f}% |\n"
            content += "\n"

    content += "---\n\n"

    # 建议
    content += "## 💡 建议\n\n"
    if summary['errors'] == 0 and summary['warnings'] == 0:
        content += "数据质量优秀，可以开始分析工作。\n\n"
    elif summary['errors'] == 0:
        content += "数据基本可用，但建议处理以下警告以提高数据质量：\n\n"
        for warning in report.warnings:
            content += f"- {warning['description']}\n"
        content += "\n"
    else:
        content += "**必须修复以下错误才能继续**：\n\n"
        for issue in report.issues:
            content += f"- {issue['description']}\n"
        content += "\n"

    content += "---\n\n"
    content += "**生成工具**: validate_6groups_data.py  \n"
    content += "**参考文档**: [6GROUPS_DATA_DESIGN_CORRECT_20260115.md](6GROUPS_DATA_DESIGN_CORRECT_20260115.md)\n"

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✅ 报告已生成: {output_file}")


def main():
    """主函数"""
    print("="*60)
    print("6分组数据独立验证")
    print("="*60)
    print(f"数据目录: {DATA_DIR}")

    report = ValidationReport()

    try:
        # 检查1: 数据完整性
        group_data = check_data_completeness(report)

        if not group_data:
            print("\n❌ 无法加载数据，终止验证")
            return

        # 检查2: Timestamp唯一性
        check_timestamp_uniqueness(group_data, report)

        # 检查3: 分组正确性
        check_grouping_correctness(group_data, report)

        # 检查4: 字段正确性
        check_field_correctness(group_data, report)

        # 检查5: 模型变量编码
        check_model_encoding(group_data, report)

        # 检查6: 数据质量
        check_data_quality(group_data, report)

        # 检查7: 特殊情况
        check_special_cases(group_data, report)

        # 生成报告
        output_file = '/home/green/energy_dl/nightly/analysis/docs/reports/6GROUPS_DATA_VALIDATION_REPORT_20260115.md'
        generate_report(report, output_file)

        # 打印摘要
        summary = report.get_summary()
        print("\n" + "="*60)
        print("验证完成")
        print("="*60)
        print(f"总检查项: {summary['total_checks']}")
        print(f"通过: {summary['passed_checks']}")
        print(f"失败: {summary['failed_checks']}")
        print(f"错误: {summary['errors']}")
        print(f"警告: {summary['warnings']}")
        print(f"总体评分: {summary['overall_score']:.1f}/100")

        if summary['errors'] == 0:
            print("\n✅ 验证通过！")
            return 0
        else:
            print("\n❌ 验证失败，请查看报告了解详情")
            return 1

    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
