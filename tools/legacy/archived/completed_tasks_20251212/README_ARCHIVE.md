# Scripts Archive - Completed Tasks

**归档日期**: 2025-12-12 20:45:39

## 归档原因

这些脚本已完成其任务，归档以保持scripts目录整洁。

## 归档脚本分类

### 数据重建脚本 (已完成)

- rebuild_summary_old_from_json_93col.py - 93列重建
- rebuild_old_csv_from_whitelist.py - 从白名单重建
- rebuild_summary_all_93col.py - summary_all重建
- convert_summary_old_to_80col.py - 80列转换
- step1_scan_experiment_json.py - 扫描实验JSON
- step2_design_csv_header.py - 设计CSV表头
- step3_rebuild_new_csv.py - 重建新CSV

### 数据修复脚本 (已完成)

- step1_fix_experiment_source.py - 修复实验来源
- step2_add_mutated_param.py - 添加变异参数
- step3_fill_default_hyperparams.py - 填充默认超参数
- step4_add_mutation_count.py - 添加变异计数
- step5_enhance_mutation_analysis.py - 增强变异分析
- fix_csv_null_values.py - 修复空值
- add_high_priority_columns.py - 添加高优先级列

### 配置修复脚本 (已完成)

- fix_stage_configs.py - 修复Stage配置

### 数据分离脚本 (已完成)

- separate_old_new_experiments.py - 分离新老实验
- extract_old_experiment_whitelist.py - 提取白名单

### 临时分析工具

- analyze_summary_all_columns.py - 列分析
- analyze_json_field_coverage.py - 字段覆盖分析
- generate_100col_schema.py - 生成schema

### 验证脚本

- validate_93col_rebuild.py - 验证93列重建

### 已废弃脚本

- aggregate_csvs.py - 已被merge_csv_to_raw_data.py替代

## 当前有效脚本

保留在 `scripts/` 目录中的脚本：

### 核心工具
- merge_csv_to_raw_data.py - 合并CSV为raw_data.csv
- validate_raw_data.py - 验证raw_data.csv
- archive_summary_files.py - 归档summary文件

### 配置工具
- generate_mutation_config.py - 生成变异配置
- validate_mutation_config.py - 验证变异配置
- verify_stage_configs.py - 验证stage配置

### 分析工具
- analyze_baseline.py - 分析基线
- analyze_experiments.py - 分析实验

### 下载工具
- download_pretrained_models.py - 下载预训练模型

---

**归档人**: Claude (AI助手)
**项目版本**: v4.7.3
