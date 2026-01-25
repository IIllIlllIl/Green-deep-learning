# docs/ 目录重组方案

**创建日期**: 2026-01-25
**状态**: 📋 待执行

---

## 📊 分析摘要

### 当前状态
- **docs/ 根目录文档**: 54个 markdown 文件
- **子目录**: archived/, environment/, results_reports/, settings_reports/
- **主要问题**:
  1. ❌ 过时文档与当前设计不一致
  2. ❌ 大量重复内容的报告类文档
  3. ❌ 重要文档被淹没在54个文件中

---

## 🚨 关键发现：过时文档

### 1. JSON_CONFIG_WRITING_STANDARDS.md ⚠️⚠️⚠️ 严重过时

**问题**:
- 文档日期: 2025-12-13
- 强调"**单参数变异原则**"（核心原则）
- 但 EXPERIMENT_EXPANSION_PLAN_20260105.md (2026-01-05) 已明确改为 "**允许多参数变异**"

**冲突内容**:
```markdown
# JSON_CONFIG_WRITING_STANDARDS.md (过时)
## 🎯 核心原则
1. **单参数变异原则**: 每个实验配置只能变异一个超参数

# EXPERIMENT_EXPANSION_PLAN_20260105.md (当前)
### 关键变化
| 维度 | 原设计 | 新设计 |
| **变异约束** | 每次只变异1个超参数 | 可同时变异多个超参数 |
```

**影响**: 如果用户阅读此规范，会按照旧标准编写配置，导致与当前实验设计冲突！

**建议**: ⚠️ **紧急更新** - 添加"多参数变异"部分，或归档旧版本创建新版本

---

## 📋 重复内容分析

### 数据相关文档 (6个相似文档)

| 文档 | 主题 | 重复度 | 建议 |
|------|------|--------|------|
| DATA_USABILITY_SUMMARY_20260113.md | 数据可用性分析 | 高 | 归档 |
| DATA_USABILITY_SUMMARY_20260113_v2.md | 数据可用性分析(修复后) | 高 | 归档 |
| DATA_REPAIR_REPORT_20260113.md | 数据修复报告 | 中 | 归档 |
| DATA_REPAIR_FINAL_SUMMARY_20260113.md | 数据修复最终总结 | 中 | 归档 |
| DATA_RECOVERY_ANALYSIS_REPORT_20260113.md | 数据恢复分析 | 中 | 归档 |
| DATA_USABILITY_FOR_REGRESSION_20260114.md | 回归数据可用性 | 中 | 保留(可能当前使用) |

**建议**: 全部归档到 `docs/reports/data_quality/`

### 去重相关文档 (3个相似文档)

| 文档 | 主题 | 建议 |
|------|------|------|
| DEDUPLICATION_USER_GUIDE.md | 去重用户指南 | 合并到 DATA_MASTER_GUIDE.md |
| MULTI_PARAM_DEDUP_GUIDE.md | 多参数去重指南 | 合并 |
| MULTI_PARAM_DEDUP_IMPACT_ANALYSIS.md | 去重影响分析 | 归档到 reports/ |

### 快速参考文档 (多个)

| 文档 | 主题 | 建议 |
|------|------|------|
| QUICK_REFERENCE.md | 快速参考 | 合并 |
| SCRIPTS_QUICKREF.md | 脚本快速参考 | 合并 |
| OUTPUT_STRUCTURE_QUICKREF.md | 输出结构快速参考 | 合并或归档 |
| MUTATION_RANGES_QUICK_REFERENCE.md | 变异范围快速参考 | 合并或归档 |

---

## 🗂️ 重组方案

### 目标结构

```
docs/
├── ⭐ 核心规范文档 (保留在根目录)
│   ├── CLAUDE_FULL_REFERENCE.md
│   ├── DATA_MASTER_GUIDE.md
│   ├── DEVELOPMENT_WORKFLOW.md
│   ├── SCRIPT_DEV_STANDARDS.md
│   ├── INDEPENDENT_VALIDATION_GUIDE.md
│   ├── JSON_CONFIG_WRITING_STANDARDS.md  ⚠️ 需要更新！
│   ├── RAW_DATA_CSV_USAGE_GUIDE.md
│   ├── EXPERIMENT_EXPANSION_PLAN_20260105.md
│   └── 11_MODELS_FINAL_DEFINITION.md
│
├── reports/  📊 报告类文档归档
│   ├── data_quality/
│   │   ├── DATA_USABILITY_SUMMARY_20260113.md
│   │   ├── DATA_USABILITY_SUMMARY_20260113_v2.md
│   │   ├── DATA_REPAIR_REPORT_20260113.md
│   │   ├── DATA_REPAIR_FINAL_SUMMARY_20260113.md
│   │   ├── DATA_RECOVERY_ANALYSIS_REPORT_20260113.md
│   │   └── DATA_USABILITY_FOR_REGRESSION_20260114.md
│   ├── cleanup/
│   │   ├── CLEANUP_REPORT_20260116.md
│   │   ├── PROJECT_CLEANUP_PLAN_20251225.md
│   │   ├── PROJECT_CLEANUP_COMPLETION_REPORT_20251225.md
│   │   └── PROJECT_CLEANUP_SUMMARY_20251225.md
│   ├── configuration/
│   │   ├── TASK_PROGRESS_DATA_EXTRACTION_FIX.md
│   │   ├── CSV_REBUILD_FROM_EXPERIMENT_JSON.md
│   │   └── DOCUMENTATION_REORGANIZATION_REPORT_20260104.md
│   ├── analysis/
│   │   ├── INDEPENDENT_DATA_QUALITY_ASSESSMENT_REPORT.md
│   │   └── DATA_DOCUMENTATION_CONSOLIDATION_*.md
│   └── environment/
│       ┚ー ENVIRONMENT_ANALYSIS_20260123.md
│
├── archived/  🗄️ 过时文档归档
│   ├── plans/
│   │   ├── RESTRUCTURE_PLAN_20260105.md  (已完成)
│   │   └── DATA_DOCUMENTATION_CONSOLIDATION_PLAN.md
│   ┚ー old_standards/
│       └── JSON_CONFIG_WRITING_STANDARDS_old.md  (如果需要完全重写)
│
├── guides/  📘 专题指南 (从根目录移入)
│   ├── SETTINGS_CONFIGURATION_GUIDE.md
│   ├── JSON_CONFIG_BEST_PRACTICES.md
│   ├── PARALLEL_TRAINING_USAGE.md
│   ├── TERMINAL_OUTPUT_CAPTURE_GUIDE.md
│   ├── APPEND_SESSION_TO_RAW_DATA_GUIDE.md
│   └── energy_monitoring_improvements.md
│
├── reference/  📚 参考文档 (从根目录移入)
│   ├── QUICK_REFERENCE.md  (合并后)
│   ├── FEATURES_OVERVIEW.md
│   ├── REPOSITORIES_LINKS.md
│   ├── USAGE_EXAMPLES.md
│   └── FLOAT_NORMALIZATION_EXPLAINED.md
│
├── specs/  📋 技术规格 (从根目录移入)
│   ├── 11_MODELS_FINAL_DEFINITION.md
│   ┚ー HYPERPARAMETER_RANGES.md (如果存在)
│
└── results_reports/  🔬 实验结果报告 (保留)
    └── ... (现有内容)

environment/  🖥️ 环境配置 (保留)
settings_reports/  ⚙️ 配置报告 (保留)
archived/  🗄️ 归档 (保留现有内容)
```

---

## 📝 详细操作清单

### 第一阶段：创建新目录结构

```bash
cd /home/green/energy_dl/nightly/docs

# 创建新目录
mkdir -p reports/{data_quality,cleanup,configuration,analysis,environment}
mkdir -p archived/{plans,old_standards}
mkdir -p guides reference specs
```

### 第二阶段：移动报告类文档到 reports/

#### data_quality/
```bash
# 数据质量相关报告
mv DATA_USABILITY_SUMMARY_20260113.md reports/data_quality/
mv DATA_USABILITY_SUMMARY_20260113_v2.md reports/data_quality/
mv DATA_REPAIR_REPORT_20260113.md reports/data_quality/
mv DATA_REPAIR_FINAL_SUMMARY_20260113.md reports/data_quality/
mv DATA_RECOVERY_ANALYSIS_REPORT_20260113.md reports/data_quality/
mv DATA_USABILITY_FOR_REGRESSION_20260114.md reports/data_quality/
```

#### cleanup/
```bash
# 清理相关报告
mv CLEANUP_REPORT_20260116.md reports/cleanup/
mv PROJECT_CLEANUP_PLAN_20251225.md reports/cleanup/
mv PROJECT_CLEANUP_COMPLETION_REPORT_20251225.md reports/cleanup/
mv PROJECT_CLEANUP_SUMMARY_20251225.md reports/cleanup/
```

#### configuration/
```bash
# 配置相关报告
mv TASK_PROGRESS_DATA_EXTRACTION_FIX.md reports/configuration/
mv CSV_REBUILD_FROM_EXPERIMENT_JSON.md reports/configuration/
mv DOCUMENTATION_REORGANIZATION_REPORT_20260104.md reports/configuration/
```

#### analysis/
```bash
# 分析相关报告
mv INDEPENDENT_DATA_QUALITY_ASSESSMENT_REPORT.md reports/analysis/
mv DATA_DOCUMENTATION_CONSOLIDATION_COMPLETION_REPORT.md reports/analysis/
```

#### environment/
```bash
# 环境相关报告
mv ENVIRONMENT_ANALYSIS_20260123.md reports/environment/
```

### 第三阶段：移动指南类文档到 guides/

```bash
# 用户指南
mv SETTINGS_CONFIGURATION_GUIDE.md guides/
mv JSON_CONFIG_BEST_PRACTICES.md guides/
mv PARALLEL_TRAINING_USAGE.md guides/
mv TERMINAL_OUTPUT_CAPTURE_GUIDE.md guides/
mv APPEND_SESSION_TO_RAW_DATA_GUIDE.md guides/
mv energy_monitoring_improvements.md guides/
```

### 第四阶段：移动参考文档到 reference/

```bash
# 参考文档
mv FEATURES_OVERVIEW.md reference/
mv REPOSITORIES_LINKS.md reference/
mv USAGE_EXAMPLES.md reference/
mv FLOAT_NORMALIZATION_EXPLAINED.md reference/
```

### 第五阶段：归档过时文档

```bash
# 已完成的计划
mv RESTRUCTURE_PLAN_20260105.md archived/plans/
mv DATA_DOCUMENTATION_CONSOLIDATION_PLAN.md archived/plans/

# 其他过时文档 (根据实际情况)
# mv ... archived/
```

### 第六阶段：合并去重相关文档

```bash
# 合并去重指南到 DATA_MASTER_GUIDE.md (如适用)
# 或者创建独立的 docs/guides/DEDUPLICATION_GUIDE.md
# 然后归档旧文档
mv DEDUPLICATION_USER_GUIDE.md archived/
mv MULTI_PARAM_DEDUP_GUIDE.md archived/
mv MULTI_PARAM_DEDUP_IMPACT_ANALYSIS.md reports/configuration/
```

### 第七阶段：处理快速参考文档

**选项 A**: 合并为单个 QUICK_REFERENCE.md
**选项 B**: 分别归档到 reference/

```bash
# 如果选择合并，需要手动编辑
# 否则移动到 reference/
mv QUICK_REFERENCE.md reference/
mv SCRIPTS_QUICKREF.md reference/
mv OUTPUT_STRUCTURE_QUICKREF.md reference/
mv MUTATION_RANGES_QUICK_REFERENCE.md reference/
```

### 第八阶段：处理技术规格文档

```bash
# 模型定义等规格文档
mv 11_MODELS_FINAL_DEFINITION.md specs/
```

### 第九阶段：紧急更新 JSON_CONFIG_WRITING_STANDARDS.md ⚠️

**需要添加的内容**:
1. 多参数变异配置格式
2. 更新核心原则（不再限制单参数）
3. 添加新旧配置对比
4. 更新验证清单

**建议**:
- 选项 A: 在现有文档末尾添加"多参数变异扩展"章节
- 选项 B: 创建 JSON_CONFIG_WRITING_STANDARDS_V2.md，归档旧版本

---

## ✅ 保留在 docs/ 根目录的文档

### 核心规范 (不移动)

1. **CLAUDE_FULL_REFERENCE.md** - 完整参考文档 ⭐⭐⭐
2. **DATA_MASTER_GUIDE.md** - 数据使用主指南 ⭐⭐⭐⭐⭐
3. **RAW_DATA_CSV_USAGE_GUIDE.md** - raw_data.csv 使用指南 ⭐⭐⭐⭐⭐
4. **DEVELOPMENT_WORKFLOW.md** - 开发工作流程 ⭐⭐⭐
5. **SCRIPT_DEV_STANDARDS.md** - 脚本开发规范 ⭐⭐⭐
6. **INDEPENDENT_VALIDATION_GUIDE.md** - 独立验证规范 ⭐⭐⭐
7. **JSON_CONFIG_WRITING_STANDARDS.md** - JSON配置规范 ⭐⭐⭐ ⚠️ **需更新**
8. **EXPERIMENT_EXPANSION_PLAN_20260105.md** - 当前实验扩展方案 ⭐⭐⭐

### 其他保留

- **README.md** - docs 目录说明
- **FILE_STRUCTURE_UPDATE_20260115.md** - 最近的结构更新
- **V4_7_7_UPDATE_NOTES.md** - 版本更新说明
- **SCRIPTS_DOCUMENTATION.md** - 脚本文档
- **RESTRUCTURE_QUICKSTART.md** - 重组快速开始
- **QUICKSTART_BASELINE.md** - 快速开始基线
- **GPU_MEMORY_CLEANUP_FIX.md** - GPU修复
- **ORPHAN_PROCESS_ANALYSIS.md** - 孤立进程分析
- **SUMMARY_ALL_README.md** - README 汇总

---

## 📊 预期效果

### 重组前
- docs/ 根目录: **54个 markdown 文件**
- 文档类型混杂: 规范、报告、指南、参考
- 重要文档难以查找

### 重组后
- docs/ 根目录: **~15个核心文档** ⭐
  - 规范文档: 8个
  - 其他重要文档: ~7个
- reports/: 所有历史报告按主题分类
- guides/: 用户指南集中管理
- reference/: 参考文档集中管理
- specs/: 技术规格集中管理
- archived/: 过时文档归档

---

## 🚨 风险与注意事项

### 1. 文档链接可能失效

**影响**: 移动文档后，其他文档中的相对链接会失效

**解决方案**:
- 执行移动后，运行链接检查脚本
- 更新所有文档中的相对路径引用
- 或使用符号链接保持兼容性

### 2. JSON_CONFIG_WRITING_STANDARDS.md 过时问题

**影响**: 用户可能按照旧规范编写配置，导致与当前设计冲突

**解决方案**:
- **立即行动**: 在文档顶部添加醒目警告
- **短期方案**: 添加"多参数变异扩展"章节
- **长期方案**: 重写整个文档，归档旧版本

### 3. 用户习惯

**影响**: 用户可能习惯原有文档位置

**解决方案**:
- 在 README.md 中说明新的文档结构
- 保留重定向或符号链接
- 给予过渡期

---

## 📝 执行顺序

1. ✅ **创建目录结构**
2. ✅ **移动报告类文档** (低风险)
3. ✅ **移动指南和参考文档** (中风险)
4. ✅ **归档过时文档** (低风险)
5. ⚠️ **合并重复文档** (需要手动编辑)
6. ⚠️ **更新 JSON_CONFIG_WRITING_STANDARDS.md** (紧急)
7. ✅ **检查和修复链接** (必需)
8. ✅ **更新 README.md** (必需)
9. ✅ **验证重组效果** (必需)

---

## 🔄 后续维护

### 定期归档

- 每月将完成的报告归档到 reports/
- 每季度检查文档时效性
- 及时更新过时内容

### 文档生命周期

1. **活跃期** → 保留在根目录或 guides/
2. **稳定期** → 移到 reference/
3. **过期期** → 移到 archived/

---

**维护者**: Claude Code
**创建日期**: 2026-01-25
**状态**: 📋 待用户审核和执行
