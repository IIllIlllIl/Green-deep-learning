# 文件结构更新报告 (2026-01-15)

## 概述

本次更新将分散的脚本文件整理到统一的目录结构中，提高了项目的组织性和可维护性。

## 主要变更

### 1. scripts目录合并到tools目录

**原因**: 避免scripts和tools两个功能重叠的目录，统一脚本管理。

**变更内容**:
- 将scripts目录中的8个数据分析脚本移动到 `tools/data_management/`
- 将scripts目录中的2个项目管理脚本移动到 `tools/` 根目录
- 删除空的scripts目录

### 2. 文件移动详情

#### 数据分析脚本 (→ tools/data_management/)
- analyze_all_missing_data.py
- analyze_data_usability_for_regression.py
- analyze_data_usability.py
- analyze_unusable_data_sources.py
- analyze_unusable_reasons.py
- independent_data_quality_assessment.py
- independent_quality_assessment.py
- remove_empty_model_records.py

#### 项目管理脚本 (→ tools/)
- restructure_project.sh
- update_paths.py

### 3. 其他文件整理（之前完成）

#### Python脚本 (→ scripts/ → tools/data_management/)
- 上述8个数据分析脚本

#### 文档文件 (→ docs/)
- INDEPENDENT_DATA_QUALITY_ASSESSMENT_REPORT.md

#### TXT报告 (→ docs/results_reports/)
- data_usability_for_regression_summary.txt
- data_usability_report.txt
- missing_data_full_report.txt
- missing_data_report.txt
- unusable_data_sources_report.txt
- unusable_reasons_detail_report.txt

#### 数据文件 (→ data/)
- data_quality_assessment_summary.json
- unusable_records_for_regression_detail.csv

## 更新后的目录结构

```
tools/
├── data_management/     # 数据管理工具（19个活跃脚本）
├── config_management/   # 配置管理工具（4个脚本）
├── legacy/              # 历史脚本归档
│   ├── completed_data_tasks_20260110/
│   └── archived/
├── quick_health_check.sh    # 项目健康检查脚本
├── restructure_project.sh   # 项目重构脚本
├── update_paths.py          # 路径更新工具
└── README.md               # 工具目录说明文档
```

## 文档更新需求

以下文档中包含对scripts目录的引用，需要根据实际情况更新：

### 需要更新的文档（高优先级）
1. **docs/CLAUDE_FULL_REFERENCE.md** - 包含多处scripts/引用
2. **docs/SCRIPTS_QUICKREF.md** - 脚本快速参考
3. **docs/QUICKSTART_BASELINE.md** - 基线快速开始指南

### 脚本路径变更映射

| 原路径 | 新路径 | 状态 |
|--------|--------|------|
| scripts/merge_csv_to_raw_data.py | tools/legacy/completed_data_tasks_20260110/ | 已归档 |
| scripts/calculate_experiment_gap.py | tools/legacy/ | 已归档 |
| scripts/analyze_experiments.py | tools/legacy/ | 已归档 |
| scripts/fix_stage_configs.py | tools/legacy/archived/completed_tasks_20251212/ | 已归档 |
| scripts/validate_config.py | tools/config_management/validate_mutation_config.py | 重命名 |
| scripts/validate_models_config.py | tools/config_management/validate_models_config.py | 已存在 |

## 建议

1. **更新文档引用**: 建议逐步更新文档中的脚本路径引用，特别是常用文档如CLAUDE_FULL_REFERENCE.md
2. **使用新路径**: 今后请使用tools/下的脚本路径，不再使用scripts/
3. **查找脚本**: 使用 `ls tools/*/` 或查看 tools/README.md 来找到需要的脚本

## 统计信息

- **合并前**: scripts目录10个文件 + tools目录15个文件
- **合并后**: tools目录25个文件（统一管理）
- **文件整理总数**: 17个文件（含之前整理的文档和数据文件）

---

**维护者**: Green
**更新时间**: 2026-01-15