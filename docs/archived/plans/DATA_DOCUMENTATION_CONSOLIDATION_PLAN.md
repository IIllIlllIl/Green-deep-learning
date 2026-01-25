# 数据文档整合计划

**创建日期**: 2026-01-15
**目的**: 整合78个数据相关文档，消除重复和过时信息，建立清晰的文档结构
**核心原则**:
- ✅ **优先使用 data.csv** (统一格式，易于使用)
- ✅ **raw_data.csv 作为备选** (完整性更高，但需要特殊处理)
- ✅ **timestamp 是唯一键** (experiment_id 可重复)

---

## 📊 现状分析

### 文档统计

- **总文档数**: 78个数据相关Markdown文档
- **主要分布**:
  - `docs/results_reports/`: 23个文件（历史报告）
  - `analysis/docs/reports/`: 18个文件（分析报告）
  - `docs/`: 10个核心文档
  - `analysis/docs/`: 10个核心文档
  - 其他归档目录: 17个文件

### 核心问题

1. **重复问题**:
   - 多个数据质量评估报告（至少6个）
   - 多个数据修复报告（至少4个）
   - 多个数据对比报告（至少5个）
   - 重复分析报告（DATA_QUALITY, DATA_USABILITY等）

2. **过时信息**:
   - 基于旧数据的分析（2025-12-xx的报告）
   - 已废弃的数据提取方案
   - 错误的数据理解（认为experiment_id是唯一键）
   - 过时的数据统计（726行 → 970行）

3. **信息分散**:
   - 数据格式说明分散在多个文档
   - 使用指南不统一
   - 关键警告未在主文档中突出

---

## 🎯 整合目标

### 1. 创建主数据文档

**文档名称**: `docs/DATA_MASTER_GUIDE.md`

**内容结构**:
```markdown
# 能耗DL项目 - 数据使用主指南

## 快速开始
- 推荐：使用 data.csv（统一格式）
- 备选：使用 raw_data.csv（完整性更高）

## 数据文件对比
- data.csv vs raw_data.csv
- 何时使用哪个文件

## 数据质量现状
- 最新数据统计（2026-01-15）
- 可用性分析
- 完整性评估

## 关键注意事项 ⚠️⚠️⚠️
1. timestamp 是唯一键（不是 experiment_id）
2. experiment_id 可重复（不同轮次可用相同ID）
3. 并行模式数据在 fg_ 前缀字段
4. 优先使用 data.csv（已处理好并行/非并行）

## 数据处理最佳实践
- 去重方法（使用 timestamp）
- 缺失值处理
- 数据验证流程

## 常见错误和解决方案
- 误用 experiment_id 作为唯一键
- 忽略并行模式数据
- 数据加载格式错误

## 参考文档
- 详细使用指南
- 历史修复报告
- 数据变更日志
```

### 2. 保留的核心文档

| 文档 | 状态 | 说明 |
|------|------|------|
| `docs/DATA_MASTER_GUIDE.md` | 🆕 新建 | **主数据指南**（整合所有核心信息） |
| `docs/RAW_DATA_CSV_USAGE_GUIDE.md` | ✅ 保留 | raw_data.csv详细使用指南 |
| `docs/DATA_USABILITY_FOR_REGRESSION_20260114.md` | ✅ 保留 | 最新的6分组回归分析可用性 |
| `docs/DATA_REPAIR_FINAL_SUMMARY_20260113.md` | ✅ 保留 | 数据修复最终总结 |
| `analysis/docs/DATA_UNDERSTANDING_CORRECTION_20251228.md` | ✅ 保留 | 数据理解关键更正 ⭐⭐⭐ |
| `analysis/docs/DATA_UNIQUENESS_CLARIFICATION_20251228.md` | ✅ 保留 | 唯一标识说明 ⭐⭐⭐ |
| `analysis/docs/DATA_FILES_COMPARISON.md` | ✅ 保留 | data.csv vs raw_data.csv 对比 |
| `analysis/data/energy_research/DUPLICATE_DATA_ANALYSIS_REPORT_CORRECTED.md` | ✅ 保留 | 去重分析（修订版） |

### 3. 归档的文档

将以下类型的文档移到归档目录：

**归档目录**: `docs/archived/data_reports_archive_20260115/`

**归档类别**:

#### A. 历史数据报告（2025-12之前）
- 所有 CSV_* 开头的修复报告（2025-12-03到2025-12-12）
- 旧的数据提取报告
- 过时的数据质量报告

#### B. 重复的数据分析报告
- `DATA_QUALITY_CHECK_REPORT.md` (2025-12-19) → 被新版本替代
- `DATA_CSV_QUALITY_ASSESSMENT_REPORT.md` (2025-12) → 被新版本替代
- `DATA_USABILITY_SUMMARY_20260113.md` → 被 v2 替代
- `DATA_USABILITY_SUMMARY_20260113_v2.md` → 被 REGRESSION 版本替代

#### C. 中间过程报告
- `DATA_RECOVERY_ANALYSIS_REPORT_20260113.md` → 整合到最终总结
- `DATA_REPAIR_REPORT_20260113.md` → 整合到最终总结
- 各种诊断报告（DIAGNOSIS, INVESTIGATION等）

#### D. 已完成任务的报告
- Phase 3, 4, 5 完成报告
- 各种修复完成报告
- 执行报告（EXECUTION_REPORT）

#### E. analysis 目录中的过时报告
- `analysis/archived_dibs_attempts/` 下的所有 DATA_* 文档（已归档）
- 旧的数据生成报告（被新版本替代）
- 分层数据质量报告（中间版本）

### 4. 删除的文档

以下文档可以安全删除（内容已整合或完全过时）:

- `DUPLICATE_DATA_ANALYSIS_REPORT.md` (非修订版) → 保留修订版即可
- 重复的 `DATA_GENERATION_REPORT.md`（backup目录）
- 已标记为 DEPRECATED 的文档

---

## 🔧 执行计划

### 阶段1: 创建主数据文档 ✅

**任务**:
1. 创建 `docs/DATA_MASTER_GUIDE.md`
2. 整合以下内容：
   - 最新数据统计（970行，577可用）
   - data.csv vs raw_data.csv 对比
   - 关键注意事项（timestamp是唯一键等）
   - 使用建议和最佳实践
   - 常见错误清单

**信息来源**:
- `DATA_USABILITY_FOR_REGRESSION_20260114.md` (最新统计)
- `DATA_FILES_COMPARISON.md` (文件对比)
- `DATA_UNDERSTANDING_CORRECTION_20251228.md` (关键更正)
- `DATA_UNIQUENESS_CLARIFICATION_20251228.md` (唯一键说明)
- `RAW_DATA_CSV_USAGE_GUIDE.md` (使用细节)

### 阶段2: 更新主文档索引 ✅

**任务**:
1. 在 `CLAUDE.md` 添加"数据处理关键提醒"章节
2. 在 `analysis/docs/INDEX.md` 添加"数据使用注意事项"章节

**内容要点**:
- ⚠️⚠️⚠️ 警告：timestamp 是唯一键，不是 experiment_id
- ✅ 推荐使用 data.csv（除非需要最大完整性）
- 📖 参考 DATA_MASTER_GUIDE.md 获取详细信息
- 🔗 链接到关键文档

### 阶段3: 归档历史文档 ✅

**任务**:
1. 创建归档目录结构
2. 移动47个历史文档到归档目录
3. 创建归档说明文件（README_ARCHIVE.md）

**归档目录结构**:
```
docs/archived/data_reports_archive_20260115/
├── README_ARCHIVE.md (归档说明)
├── csv_repairs_2025_12/ (CSV修复相关)
├── data_extraction_2025_12/ (数据提取相关)
├── data_quality_2025_12_2026_01/ (数据质量报告)
├── data_repair_2026_01/ (数据修复过程)
└── intermediate_reports/ (中间过程报告)
```

### 阶段4: 删除冗余文档 ✅

**任务**:
1. 删除非修订版的重复报告
2. 删除backup目录中的重复文档
3. 删除已明确标记为DEPRECATED的文档

**删除清单**:
- `analysis/data/energy_research/DUPLICATE_DATA_ANALYSIS_REPORT.md` (保留CORRECTED版本)
- `analysis/data/energy_research/dibs_training_backup_*/DATA_GENERATION_REPORT.md`
- `analysis/docs/archives/DEPRECATED_*.md`

### 阶段5: 验证和测试 ✅

**任务**:
1. 验证所有链接正确
2. 确认主文档内容完整
3. 测试数据加载代码示例
4. 确保归档文档可访问

---

## 📋 最终文档结构

### 主项目文档（docs/）

```
docs/
├── DATA_MASTER_GUIDE.md ⭐⭐⭐ [新建] 主数据指南
├── RAW_DATA_CSV_USAGE_GUIDE.md ⭐⭐ raw_data.csv详细使用
├── DATA_USABILITY_FOR_REGRESSION_20260114.md ⭐ 6分组可用性
├── DATA_REPAIR_FINAL_SUMMARY_20260113.md ⭐ 修复最终总结
├── APPEND_SESSION_TO_RAW_DATA_GUIDE.md - Session追加指南
├── CSV_REBUILD_FROM_EXPERIMENT_JSON.md - 从JSON重建CSV
└── archived/
    ├── data_reports_archive_20260115/ [新建归档]
    │   ├── README_ARCHIVE.md
    │   ├── csv_repairs_2025_12/ (10个文档)
    │   ├── data_extraction_2025_12/ (8个文档)
    │   ├── data_quality_2025_12_2026_01/ (12个文档)
    │   ├── data_repair_2026_01/ (4个文档)
    │   └── intermediate_reports/ (13个文档)
    ├── MUTATION_2X_DATA_SUMMARY.md
    └── BOUNDARY_TEST_DATA_OUTPUT.md
```

### Analysis文档（analysis/docs/）

```
analysis/docs/
├── DATA_UNDERSTANDING_CORRECTION_20251228.md ⭐⭐⭐ 关键更正
├── DATA_UNIQUENESS_CLARIFICATION_20251228.md ⭐⭐⭐ 唯一标识
├── DATA_FILES_COMPARISON.md ⭐⭐ 文件对比
├── ENERGY_DATA_PROCESSING_PROPOSAL.md - 处理方案
├── STRATIFIED_DATA_QUALITY_FINAL_ASSESSMENT.md - 最终评估
└── reports/
    ├── DATA_COMPARISON_OLD_VS_NEW_20251229.md - 新旧对比
    ├── DATA_UPDATE_40PERCENT_THRESHOLD_20260105.md - 更新报告
    └── [其他分析报告]
```

### Analysis数据文档（analysis/data/energy_research/）

```
analysis/data/energy_research/
├── DATA_STATUS_REPORT_20260114.md ⭐ 数据现状报告
├── RAW_DATA_VS_DATA_CSV_COMPARISON.md - 详细对比
├── DUPLICATE_DATA_ANALYSIS_REPORT_CORRECTED.md ⭐ 去重分析
└── dibs_training/
    └── DATA_GENERATION_REPORT.md - DiBS数据生成
```

---

## ✅ 成功标准

1. **文档减少**: 从78个减少到约30个活跃文档（~62%减少）
2. **清晰性**: 主数据指南作为单一真实来源
3. **可发现性**: 关键警告在主文档中突出显示
4. **可追溯性**: 历史文档归档但可访问
5. **一致性**: 所有文档推荐使用 data.csv，说明 timestamp 是唯一键

---

## 🔄 维护计划

### 定期审查（每月）

- 检查是否有新的数据报告需要整合
- 更新 DATA_MASTER_GUIDE.md 中的统计数据
- 归档超过3个月的临时报告

### 更新触发条件

- 数据结构变化
- 数据文件更新（新实验数据）
- 发现新的数据质量问题
- 用户报告文档错误或困惑

---

## 📞 相关资源

- **CLAUDE.md**: 项目快速指南（将添加数据处理关键提醒）
- **analysis/docs/INDEX.md**: 分析模块文档索引（将添加数据注意事项）
- **tools/data_management/**: 数据处理脚本
- **data/**: 数据文件目录（data.csv, raw_data.csv）

---

**文档维护者**: Claude Assistant
**下次审查**: 2026-02-15
