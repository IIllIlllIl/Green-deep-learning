#!/usr/bin/env python3
"""
归档过时文档和脚本

将已完成的脚本和报告归档到archived目录
"""

import shutil
from pathlib import Path
from datetime import datetime

def archive_scripts():
    """归档已完成的脚本"""
    scripts_dir = Path('/home/green/energy_dl/nightly/scripts')
    archive_dir = scripts_dir / 'archived' / 'completed_tasks_20251212'
    archive_dir.mkdir(parents=True, exist_ok=True)

    # 需要归档的脚本
    archive_list = [
        # 数据重建（已完成）
        'rebuild_summary_old_from_json_93col.py',
        'rebuild_old_csv_from_whitelist.py',
        'rebuild_summary_all_93col.py',
        'convert_summary_old_to_80col.py',
        'step1_scan_experiment_json.py',
        'step2_design_csv_header.py',
        'step3_rebuild_new_csv.py',

        # 数据修复（已完成）
        'step1_fix_experiment_source.py',
        'step2_add_mutated_param.py',
        'step3_fill_default_hyperparams.py',
        'step4_add_mutation_count.py',
        'step5_enhance_mutation_analysis.py',
        'fix_csv_null_values.py',
        'add_high_priority_columns.py',

        # 配置修复（已完成）
        'fix_stage_configs.py',

        # 数据分离（已完成）
        'separate_old_new_experiments.py',
        'extract_old_experiment_whitelist.py',

        # 临时分析工具
        'analyze_summary_all_columns.py',
        'analyze_json_field_coverage.py',
        'generate_100col_schema.py',

        # 已废弃
        'aggregate_csvs.py',

        # 验证脚本（重建已完成，可归档）
        'validate_93col_rebuild.py',
    ]

    print("=" * 70)
    print("归档Scripts")
    print("=" * 70)

    archived_count = 0
    for script in archive_list:
        src = scripts_dir / script
        if src.exists():
            dest = archive_dir / script
            shutil.move(str(src), str(dest))
            print(f"✓ {script}")
            archived_count += 1
        else:
            print(f"⊘ {script} (不存在)")

    # 创建README
    readme = archive_dir / 'README_ARCHIVE.md'
    with open(readme, 'w') as f:
        f.write(f"# Scripts Archive - Completed Tasks\n\n")
        f.write(f"**归档日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 归档原因\n\n")
        f.write(f"这些脚本已完成其任务，归档以保持scripts目录整洁。\n\n")
        f.write(f"## 归档脚本分类\n\n")

        categories = {
            '数据重建脚本 (已完成)': [
                'rebuild_summary_old_from_json_93col.py - 93列重建',
                'rebuild_old_csv_from_whitelist.py - 从白名单重建',
                'rebuild_summary_all_93col.py - summary_all重建',
                'convert_summary_old_to_80col.py - 80列转换',
                'step1_scan_experiment_json.py - 扫描实验JSON',
                'step2_design_csv_header.py - 设计CSV表头',
                'step3_rebuild_new_csv.py - 重建新CSV',
            ],
            '数据修复脚本 (已完成)': [
                'step1_fix_experiment_source.py - 修复实验来源',
                'step2_add_mutated_param.py - 添加变异参数',
                'step3_fill_default_hyperparams.py - 填充默认超参数',
                'step4_add_mutation_count.py - 添加变异计数',
                'step5_enhance_mutation_analysis.py - 增强变异分析',
                'fix_csv_null_values.py - 修复空值',
                'add_high_priority_columns.py - 添加高优先级列',
            ],
            '配置修复脚本 (已完成)': [
                'fix_stage_configs.py - 修复Stage配置',
            ],
            '数据分离脚本 (已完成)': [
                'separate_old_new_experiments.py - 分离新老实验',
                'extract_old_experiment_whitelist.py - 提取白名单',
            ],
            '临时分析工具': [
                'analyze_summary_all_columns.py - 列分析',
                'analyze_json_field_coverage.py - 字段覆盖分析',
                'generate_100col_schema.py - 生成schema',
            ],
            '验证脚本': [
                'validate_93col_rebuild.py - 验证93列重建',
            ],
            '已废弃脚本': [
                'aggregate_csvs.py - 已被merge_csv_to_raw_data.py替代',
            ],
        }

        for category, scripts in categories.items():
            f.write(f"### {category}\n\n")
            for script in scripts:
                f.write(f"- {script}\n")
            f.write(f"\n")

        f.write(f"## 当前有效脚本\n\n")
        f.write(f"保留在 `scripts/` 目录中的脚本：\n\n")
        f.write(f"### 核心工具\n")
        f.write(f"- merge_csv_to_raw_data.py - 合并CSV为raw_data.csv\n")
        f.write(f"- validate_raw_data.py - 验证raw_data.csv\n")
        f.write(f"- archive_summary_files.py - 归档summary文件\n\n")
        f.write(f"### 配置工具\n")
        f.write(f"- generate_mutation_config.py - 生成变异配置\n")
        f.write(f"- validate_mutation_config.py - 验证变异配置\n")
        f.write(f"- verify_stage_configs.py - 验证stage配置\n\n")
        f.write(f"### 分析工具\n")
        f.write(f"- analyze_baseline.py - 分析基线\n")
        f.write(f"- analyze_experiments.py - 分析实验\n\n")
        f.write(f"### 下载工具\n")
        f.write(f"- download_pretrained_models.py - 下载预训练模型\n\n")
        f.write(f"---\n\n")
        f.write(f"**归档人**: Claude (AI助手)\n")
        f.write(f"**项目版本**: v4.7.3\n")

    print(f"\n✓ 归档 {archived_count} 个脚本")
    print(f"✓ 创建 README_ARCHIVE.md")
    return archived_count

def archive_docs():
    """归档已完成的文档"""
    docs_dir = Path('/home/green/energy_dl/nightly/docs')

    # 创建归档目录
    stage_reports_dir = docs_dir / 'archived' / 'stage_reports_20251212'
    temp_analysis_dir = docs_dir / 'archived' / 'temporary_analysis_20251212'
    stage_reports_dir.mkdir(parents=True, exist_ok=True)
    temp_analysis_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("归档Docs")
    print("=" * 70)

    # Stage相关报告
    stage_reports = [
        'results_reports/STAGE3_4_EXECUTION_REPORT.md',
        'results_reports/STAGE7_EXECUTION_REPORT.md',
        'results_reports/STAGE7_8_FIX_EXECUTION_REPORT.md',
        'results_reports/STAGE7_CONFIG_FIX_REPORT.md',
        'results_reports/STAGE7_13_CONFIG_BUG_ANALYSIS.md',
        'results_reports/STAGE7_13_DESIGN_SUMMARY.md',
        'results_reports/STAGE11_ACTUAL_STATE_CORRECTION.md',
        'settings_reports/STAGE7_13_EXECUTION_PLAN.md',
    ]

    print("\n### Stage报告")
    stage_count = 0
    for report in stage_reports:
        src = docs_dir / report
        if src.exists():
            dest = stage_reports_dir / src.name
            shutil.move(str(src), str(dest))
            print(f"✓ {report}")
            stage_count += 1
        else:
            print(f"⊘ {report} (不存在)")

    # 临时分析报告
    temp_reports = [
        'results_reports/CSV_FIX_COMPREHENSIVE_SUMMARY.md',
        'results_reports/MISSING_COLUMNS_DETAILED_ANALYSIS.md',
        'results_reports/DEDUP_MODE_FIX_REPORT.md',
        'results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md',
        'results_reports/DEDUPLICATION_RANDOM_MUTATION_ANALYSIS.md',
        'results_reports/PROJECT_CLEANUP_SUMMARY_20251208.md',
        'results_reports/DAILY_SUMMARY_20251205.md',
    ]

    print("\n### 临时分析报告")
    temp_count = 0
    for report in temp_reports:
        src = docs_dir / report
        if src.exists():
            dest = temp_analysis_dir / src.name
            shutil.move(str(src), str(dest))
            print(f"✓ {report}")
            temp_count += 1
        else:
            print(f"⊘ {report} (不存在)")

    # 创建Stage归档README
    stage_readme = stage_reports_dir / 'README_ARCHIVE.md'
    with open(stage_readme, 'w') as f:
        f.write(f"# Stage Reports Archive\n\n")
        f.write(f"**归档日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 归档原因\n\n")
        f.write(f"这些Stage报告记录了分阶段实验的执行过程。所有阶段已完成，归档以保持docs目录整洁。\n\n")
        f.write(f"## 归档报告\n\n")
        for report in stage_reports:
            f.write(f"- {Path(report).name}\n")
        f.write(f"\n---\n\n")
        f.write(f"**项目状态**: 所有Stage实验已100%完成 (90/90参数-模式组合)\n")

    # 创建临时分析归档README
    temp_readme = temp_analysis_dir / 'README_ARCHIVE.md'
    with open(temp_readme, 'w') as f:
        f.write(f"# Temporary Analysis Reports Archive\n\n")
        f.write(f"**归档日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 归档原因\n\n")
        f.write(f"这些临时分析报告完成了特定的数据分析和修复任务，归档以保持docs目录整洁。\n\n")
        f.write(f"## 归档报告\n\n")
        for report in temp_reports:
            f.write(f"- {Path(report).name}\n")
        f.write(f"\n---\n\n")
        f.write(f"**项目版本**: v4.7.3\n")

    print(f"\n✓ 归档 {stage_count} 个Stage报告")
    print(f"✓ 归档 {temp_count} 个临时分析报告")
    print(f"✓ 创建归档README文件")

    return stage_count + temp_count

def main():
    """主函数"""
    print("=" * 70)
    print("文档和脚本归档")
    print("=" * 70)

    scripts_count = archive_scripts()
    docs_count = archive_docs()

    print("\n" + "=" * 70)
    print("归档统计")
    print("=" * 70)
    print(f"  归档脚本: {scripts_count}")
    print(f"  归档文档: {docs_count}")
    print(f"  总计: {scripts_count + docs_count}")
    print(f"\n✅ 归档完成")

if __name__ == '__main__':
    main()
