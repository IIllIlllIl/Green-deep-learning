#!/usr/bin/env python3
"""
文档和脚本整理归档方案

分析当前状态并生成归档计划
"""

import os
from pathlib import Path
from collections import defaultdict

def analyze_scripts():
    """分析scripts目录"""
    scripts_dir = Path('/home/green/energy_dl/nightly/scripts')

    # 按功能分类
    categories = {
        '已完成任务-数据重建': [
            'rebuild_summary_old_from_json_93col.py',  # 已完成93列重建
            'rebuild_old_csv_from_whitelist.py',  # 已完成
            'rebuild_summary_all_93col.py',  # 已完成summary_all重建（已废弃）
            'convert_summary_old_to_80col.py',  # 临时转换脚本
            'step1_scan_experiment_json.py',  # 步骤脚本，已完成
            'step2_design_csv_header.py',  # 步骤脚本，已完成
            'step3_rebuild_new_csv.py',  # 步骤脚本，已完成
        ],
        '已完成任务-数据修复': [
            'step1_fix_experiment_source.py',  # 已完成修复
            'step2_add_mutated_param.py',  # 已完成添加
            'step3_fill_default_hyperparams.py',  # 已完成填充
            'step4_add_mutation_count.py',  # 已完成添加
            'step5_enhance_mutation_analysis.py',  # 已完成增强
            'fix_csv_null_values.py',  # 已完成修复
            'add_high_priority_columns.py',  # 已完成添加
        ],
        '已完成任务-配置修复': [
            'fix_stage_configs.py',  # Stage7-13配置修复，已完成
        ],
        '已完成任务-数据分离': [
            'separate_old_new_experiments.py',  # 已完成分离
            'extract_old_experiment_whitelist.py',  # 已完成提取
        ],
        '临时分析工具': [
            'analyze_summary_all_columns.py',  # 临时分析，已完成
            'analyze_json_field_coverage.py',  # 临时分析，已完成
            'generate_100col_schema.py',  # 临时生成，已废弃
        ],
        '当前有效-核心工具': [
            'merge_csv_to_raw_data.py',  # 合并CSV
            'validate_raw_data.py',  # 验证raw_data.csv
            'archive_summary_files.py',  # 归档文件
            'validate_93col_rebuild.py',  # 验证93列重建（可归档，重建已完成）
        ],
        '当前有效-配置工具': [
            'generate_mutation_config.py',  # 生成变异配置
            'validate_mutation_config.py',  # 验证变异配置
            'verify_stage_configs.py',  # 验证stage配置
        ],
        '当前有效-分析工具': [
            'analyze_baseline.py',  # 分析基线
            'analyze_experiments.py',  # 分析实验
        ],
        '当前有效-下载工具': [
            'download_pretrained_models.py',  # 下载预训练模型
        ],
        '已废弃': [
            'aggregate_csvs.py',  # 已被merge_csv_to_raw_data.py替代
        ],
    }

    print("=" * 70)
    print("Scripts 归档方案")
    print("=" * 70)

    total_scripts = 0
    archive_scripts = 0

    for category, scripts in categories.items():
        print(f"\n### {category} ({len(scripts)}个)")
        for script in scripts:
            total_scripts += 1
            if '已完成' in category or '临时' in category or '已废弃' in category:
                archive_scripts += 1
                print(f"  [归档] {script}")
            else:
                print(f"  [保留] {script}")

    print(f"\n统计:")
    print(f"  总脚本: {total_scripts}")
    print(f"  归档: {archive_scripts}")
    print(f"  保留: {total_scripts - archive_scripts}")

    return categories

def analyze_docs():
    """分析docs目录"""
    docs_dir = Path('/home/green/energy_dl/nightly/docs')

    # 重要文档（保留）
    keep_docs = {
        '核心文档': [
            'README.md',
            'QUICK_REFERENCE.md',
            'FEATURES_OVERVIEW.md',
            'SETTINGS_CONFIGURATION_GUIDE.md',
            'OUTPUT_STRUCTURE_QUICKREF.md',
            'MUTATION_RANGES_QUICK_REFERENCE.md',
            'JSON_CONFIG_BEST_PRACTICES.md',
            'SCRIPTS_DOCUMENTATION.md',
            '11_MODELS_FINAL_DEFINITION.md',
            '11_MODELS_OVERVIEW.md',
            'PARALLEL_TRAINING_USAGE.md',
        ],
        '当前有效分析报告': [
            'results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md',
            'results_reports/SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md',
            'results_reports/OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md',
            'results_reports/VULBERTA_CNN_CLEANUP_REPORT_20251211.md',
            'results_reports/FINAL_CONFIG_INTEGRATION_REPORT.md',
            'results_reports/RUNTIME_STATISTICS_20251211.md',
            'results_reports/STAGE11_BUG_FIX_REPORT.md',
        ],
    }

    # 已完成的临时报告（可归档）
    archive_docs_patterns = [
        'STAGE*_EXECUTION_REPORT.md',  # 阶段执行报告
        'STAGE*_CONFIG_*.md',  # 配置报告
        'DAILY_SUMMARY_*.md',  # 每日总结
        '*_COMPLETION_REPORT.md',  # 完成报告
        'CSV_FIX_*.md',  # CSV修复报告
        'DEDUP_*.md',  # 去重报告
        'MISSING_COLUMNS_*.md',  # 缺失列分析
        '*_OPTIMIZATION_REPORT.md',  # 优化报告
    ]

    print("\n" + "=" * 70)
    print("Docs 归档方案")
    print("=" * 70)

    print("\n### 保留的核心文档")
    for category, docs in keep_docs.items():
        print(f"\n{category}:")
        for doc in docs:
            print(f"  [保留] {doc}")

    print("\n### 建议归档的文档类型")
    for pattern in archive_docs_patterns:
        print(f"  [归档] {pattern}")

if __name__ == '__main__':
    categories = analyze_scripts()
    analyze_docs()

    print("\n" + "=" * 70)
    print("归档建议")
    print("=" * 70)
    print("""
1. 创建归档目录:
   - scripts/archived/completed_tasks_20251212/
   - docs/archived/completed_reports_20251212/

2. 脚本整合:
   - 合并CSV相关: merge_csv_to_raw_data.py, validate_raw_data.py
   - 合并配置验证: validate_mutation_config.py, verify_stage_configs.py
   - 保留analyze工具独立

3. 文档归档:
   - 阶段执行报告 → archived/stage_reports/
   - 临时分析报告 → archived/temporary_analysis/
   - 保留核心文档和最新报告
    """)
