# Tools 目录说明

本目录包含项目的所有工具脚本，按功能分类组织。

**最后更新**: 2026-01-10
**状态**: ✅ 脚本重复性分析和清理已完成

---

## 📁 目录结构

```
tools/
├── data_management/     # 数据管理工具（19个活跃脚本）
├── config_management/   # 配置管理工具（4个脚本）
├── legacy/              # 历史脚本归档
│   ├── completed_data_tasks_20260110/  # 已完成的数据任务（4个脚本）
│   └── archived/        # 其他归档脚本
├── quick_health_check.sh    # 项目健康检查脚本
├── restructure_project.sh   # 项目重构脚本
└── update_paths.py          # 路径更新工具
```

---

## 🔧 数据管理工具 (data_management/)

**活跃脚本**: 19个

### 数据追加与验证

| 脚本 | 功能 | 优先级 |
|------|------|--------|
| \`append_session_to_raw_data.py\` | 从session目录追加实验数据到raw_data.csv | ⭐⭐⭐ |
| \`validate_raw_data.py\` | 验证raw_data.csv数据完整性和安全性 | ⭐⭐⭐ |
| \`check_latest_results.py\` | 检查最新实验是否已加入数据文件 | ⭐⭐ |
| \`compare_data_vs_raw_data.py\` | 比较data.csv和raw_data.csv一致性 | ⭐⭐ |

### 数据分析

| 脚本 | 功能 | 优先级 |
|------|------|--------|
| \`analyze_experiment_status.py\` | 分析实验状况统计（模型、参数覆盖等） | ⭐⭐⭐ |
| \`analyze_missing_energy_data.py\` | 分析缺少能耗数据的实验 | ⭐⭐⭐ |
| \`verify_recoverable_data.py\` | 验证缺失能耗数据的文件来源和可恢复性 | ⭐⭐ |

### 数据修复与处理

| 脚本 | 功能 | 优先级 |
|------|------|--------|
| \`repair_missing_energy_data.py\` | 安全修复缺失的能耗数据 | ⭐⭐⭐ |
| \`create_unified_data_csv.py\` | 创建统一并行数据版本的data.csv | ⭐⭐ |
| \`validate_merged_metrics.py\` | 验证合并后的性能指标数据质量 | ⭐⭐ |
| \`check_attribute_mapping.py\` | 检查数据属性映射完整性 | ⭐ |
| \`remove_empty_model_records.py\` | 删除空的模型记录（无用数据清理） | ⭐ |

### 数据可用性评估

| 脚本 | 功能 | 优先级 |
|------|------|--------|
| \`analyze_data_usability.py\` | 分析数据可用性（能耗、性能完整性） | ⭐⭐⭐ |
| \`analyze_data_usability_for_regression.py\` | 评估回归分析可用数据 | ⭐⭐⭐ |
| \`analyze_all_missing_data.py\` | 分析所有缺失数据情况 | ⭐⭐ |
| \`analyze_unusable_data_sources.py\` | 分析不可用数据来源 | ⭐⭐ |
| \`analyze_unusable_reasons.py\` | 分析数据不可用原因 | ⭐⭐ |
| \`independent_data_quality_assessment.py\` | 独立数据质量评估 | ⭐⭐ |
| \`independent_quality_assessment.py\` | 独立质量评估（早期版本） | ⭐ |

### 使用示例

\`\`\`bash
# 追加新实验数据（最常用）
python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS

# 验证数据完整性
python3 tools/data_management/validate_raw_data.py

# 分析实验状况
python3 tools/data_management/analyze_experiment_status.py

# 分析缺失能耗数据
python3 tools/data_management/analyze_missing_energy_data.py

# 修复缺失能耗数据
python3 tools/data_management/repair_missing_energy_data.py
\`\`\`

---

## ⚙️ 配置管理工具 (config_management/)

**活跃脚本**: 4个

| 脚本 | 功能 |
|------|------|
| \`generate_mutation_config.py\` | 生成变异配置文件 |
| \`validate_models_config.py\` | 验证models_config.json完整性和有效性 |
| \`validate_mutation_config.py\` | 验证变异配置JSON格式 |
| \`verify_stage_configs.py\` | 检查stage配置文件的runs_per_config定义 |

### 使用示例

\`\`\`bash
# 验证模型配置
python3 tools/config_management/validate_models_config.py

# 验证变异配置
python3 tools/config_management/validate_mutation_config.py settings/stage2_*.json

# 检查stage配置
python3 tools/config_management/verify_stage_configs.py
\`\`\`

---

## 🗄️ 历史脚本归档 (legacy/)

### completed_data_tasks_20260110/ (4个归档脚本)

**归档日期**: 2026-01-10
**原因**: 一次性任务已完成，功能被通用脚本取代

| 归档脚本 | 原功能 | 替代方案 |
|---------|--------|----------|
| \`add_new_experiments_to_raw_data.py\` | 从特定session提取4个实验 | 使用 \`append_session_to_raw_data.py\` |
| \`merge_csv_to_raw_data.py\` | 合并summary_old/new为raw_data | 一次性任务，已完成 |
| \`update_raw_data_with_reextracted.py\` | 重新提取性能指标 | 一次性修复任务，已完成 |
| \`merge_performance_metrics.py\` | 合并性能指标列 | 一次性任务，已完成 |

⚠️ **注意**: 请勿使用归档脚本！它们仅用于历史参考。详见 \`legacy/completed_data_tasks_20260110/README.md\`

---

## 🔍 如何查找脚本

### 按功能查找

\`\`\`bash
# 列出所有数据管理脚本
ls -lh tools/data_management/*.py

# 按关键词搜索
grep -l "追加\|append" tools/data_management/*.py
grep -l "验证\|validate" tools/data_management/*.py
grep -l "分析\|analyze" tools/data_management/*.py
grep -l "修复\|repair" tools/data_management/*.py
\`\`\`

### 查看脚本文档

\`\`\`bash
# 查看脚本顶部文档
head -30 tools/data_management/script_name.py

# 查看脚本帮助（如果支持）
python3 tools/data_management/script_name.py --help
\`\`\`

---

## 📋 脚本开发最佳实践

### 创建新脚本前的检查清单

- [ ] 查阅 \`docs/SCRIPTS_QUICKREF.md\` 确认无类似脚本
- [ ] 搜索现有脚本目录
- [ ] 测试现有脚本是否能满足需求（80%即可考虑复用）
- [ ] 确认确实需要新脚本后再开发

### 新脚本开发规范

1. **添加完整文档字符串**
2. **设计为通用工具** - 使用命令行参数而非硬编码值
3. **包含测试** - 添加示例用法
4. **一次性任务脚本的处理** - 任务完成后归档到 \`tools/legacy/\`

---

## 📚 相关文档

- [CLAUDE.md § 脚本复用检查指南](../CLAUDE.md#-脚本复用检查指南-) - 使用指南
- [docs/SCRIPTS_QUICKREF.md](../docs/SCRIPTS_QUICKREF.md) - 脚本快速参考
- [docs/SCRIPT_DUPLICATION_ANALYSIS_REPORT.md](../docs/SCRIPT_DUPLICATION_ANALYSIS_REPORT.md) - 重复性分析报告 (2026-01-10)
- [legacy/completed_data_tasks_20260110/README.md](legacy/completed_data_tasks_20260110/README.md) - 归档脚本说明

---

## 📊 统计信息

**最后统计**: 2026-01-15

- **活跃脚本总数**: 25个
  - 数据管理: 19个（+8个来自scripts目录合并）
  - 配置管理: 4个
  - 项目级工具: 2个
- **归档脚本**: 4个 (completed_data_tasks_20260110)
- **结构优化**: scripts目录成功合并到tools目录

---

**维护者**: Green
**最后更新**: 2026-01-15
