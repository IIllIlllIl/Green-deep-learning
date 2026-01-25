# Analysis模��文档组织说明

**最后更新**: 2026-01-23
**整理目的**: 将docs中的文档按重要性分级，使高价值信息更容易获取

---

## 📁 文档组织结构

```
docs/
├── README.md                           # 本文件 - 文档组织说明
├── INDEX.md                            # 主索引 - 项目文档总索引
│
├── essential_guides/                   # ⭐⭐⭐ 必读指南（不读可能引入错误）
│   ├── DATA_UNIQUENESS_CLARIFICATION_20251228.md
│   ├── DATA_UNDERSTANDING_CORRECTION_20251228.md
│   ├── UNIQUENESS_CLARIFICATION_SUMMARY_20251228.md
│   └── DATA_FILES_COMPARISON.md
│
├── current_plans/                      # ⭐⭐ 当前有效的方案
│   ├── QUESTION1_REGRESSION_ANALYSIS_PLAN.md
│   ├── QUESTIONS_2_3_DIBS_ANALYSIS_PLAN.md
│   ├── VALUE_TRANSFORMATION_COMPREHENSIVE_GUIDE.md
│   └── INTERACTION_TERMS_TRANSFORMATION_PLAN.md
│
├── technical_reference/                # ⭐ 技术参考（25个文档）
│   ├── DiBS相关文档（12个）
│   ├── 数据处理文档（5个）
│   ├── 因果边白名单文档（3个）
│   ├── 验证报告（2个）
│   └── 其他技术文档（3个）
│
└── guides/                             # 使用指南（6个文档）
    ├── DOCUMENTATION_INDEX.md
    ├── ENVIRONMENT_SETUP.md
    ├── MIGRATION_GUIDE.md
    ├── IMPROVEMENT_GUIDE.md
    ├── REPLICATION_QUICK_START.md
    └── USAGE_GUIDE_FOR_NEW_RESEARCH.md
```

---

## 🎯 文档分级说明

### 📌 essential_guides/ - 必读指南 ⭐⭐⭐

**重要性**: 极高
**不读后果**: 可能引入严重错误，导致数据处理错误或结果偏差

**包含内容**:
- **DATA_UNIQUENESS_CLARIFICATION_20251228.md** - 唯一标识说明
  - 避免：用experiment_id去重导致丢失89条记录
- **DATA_UNDERSTANDING_CORRECTION_20251228.md** - 数据理解更正
  - 纠正：训练成功率100%（不是85.5%），可用样本692个（不是587个）
- **UNIQUENESS_CLARIFICATION_SUMMARY_20251228.md** - 唯一性总结
- **DATA_FILES_COMPARISON.md** - 数据文件对比
  - 指导：data.csv vs raw_data.csv的选择

**什么时候需要读**:
- ✅ 在处理数据前（**必须**先读）
- ✅ 在遇到数据不一致问题时
- ✅ 在进行任何数据分析前

---

### 📋 current_plans/ - 当前方案 ⭐⭐

**重要性**: 高
**内容**: 当前正在执行或准备执行的分析方案

**包含内容**:
- **QUESTION1_REGRESSION_ANALYSIS_PLAN.md** - 问题1回归分析方案
  - 超参数对能耗影响的回归分析
- **QUESTIONS_2_3_DIBS_ANALYSIS_PLAN.md** - 问题2&3 DiBS方案
  - 能耗-性能权衡与中介效应分析
- **VALUE_TRANSFORMATION_COMPREHENSIVE_GUIDE.md** - 值转换综合指南
  - 消除并行/非并行模式偏差的方案
- **INTERACTION_TERMS_TRANSFORMATION_PLAN.md** - 交互项转换方案

**什么时候需要读**:
- ✅ 在开始新的分析任务前
- ✅ 在了解当前研究进展时
- ✅ 在编写分析脚本时

---

### 📚 technical_reference/ - 技术参考 ⭐

**重要性**: 中等
**内容**: 技术细节、验证报告、设计文档

**包含内容**（25个文档）:

**DiBS相关**（12个）:
- DIBS_VERIFICATION_REPORT_20260116.md - DiBS验证报告
- DIBS_PARAMETER_TUNING_ANALYSIS.md - 参数调优分析
- DIBS_EDGES_CSV_* - DiBS边CSV相关文档（3个）
- DIBS_RESULTS_* - DiBS结果相关文档（5个）
- DIBS_INTERACTION_* - DiBS交互相关文档（2个）

**数据处理**（5个）:
- DATA_INVENTORY_ANALYSIS_20260104.md - 数据清单分析
- DATA_ISOLATION_MIGRATION_REPORT.md - 数据隔离迁移报告
- DATA_LOSS_ROOT_CAUSE_ANALYSIS.md - 数据丢失根因分析
- FG_COLUMN_FIX_REPORT.md - FG列修复报告
- HYPERPARAM_DEFAULT_VALUES.md - 超参数默认值

**因果边白名单**（3个）:
- CAUSAL_EDGE_WHITELIST_DESIGN.md - 白名单设计
- CAUSAL_EDGE_WHITELIST_SUMMARY.md - 白名单总结
- CAUSAL_EDGE_WHITELIST_REVIEW_REPORT.md - 白名单审查报告

**其他**（5个）:
- COLUMN_USAGE_ANALYSIS.md - 列使用分析
- STRATIFIED_DATA_QUALITY_FINAL_ASSESSMENT.md - 分层数据质量最终评估
- VALIDATION_SUMMARY.md - 验证总结

**什么时候需要读**:
- ✅ 在深入了解技术实现时
- ✅ 在验证结果正确性时
- ✅ 在调试问题时

---

### 📖 guides/ - 使用指南

**重要性**: 中等
**内容**: 环境配置、快速开始、使用指南

**包含内容**（6个文档）:
- **DOCUMENTATION_INDEX.md** - 文档索引
- **ENVIRONMENT_SETUP.md** - 环境设置指南
- **MIGRATION_GUIDE.md** - 迁移指南
- **IMPROVEMENT_GUIDE.md** - 改进指南
- **REPLICATION_QUICK_START.md** - 论文复现快速开始
- **USAGE_GUIDE_FOR_NEW_RESEARCH.md** - 新研究使用指南

**什么时候需要读**:
- ✅ 新用户首次使用时
- ✅ 配置环境时
- ✅ 复现论文时

---

## 🚀 快速导航指南

### 如果你是新用户

1. **必读**（按顺序）:
   - 📖 本文件（README.md）- 了解文档组织
   - ⚠️ [essential_guides/DATA_UNDERSTANDING_CORRECTION_20251228.md](essential_guides/DATA_UNDERSTANDING_CORRECTION_20251228.md)
   - ⚠️ [essential_guides/DATA_UNIQUENESS_CLARIFICATION_20251228.md](essential_guides/DATA_UNIQUENESS_CLARIFICATION_20251228.md)

2. **了解当前任务**:
   - 📋 [current_plans/QUESTION1_REGRESSION_ANALYSIS_PLAN.md](current_plans/QUESTION1_REGRESSION_ANALYSIS_PLAN.md)
   - 📚 [INDEX.md](INDEX.md) - 项目总索引

3. **环境设置**:
   - 🔧 [guides/ENVIRONMENT_SETUP.md](guides/ENVIRONMENT_SETUP.md)

### 如果你要开始新分析

1. **检查数据使用**:
   - ⚠️ [essential_guides/DATA_FILES_COMPARISON.md](essential_guides/DATA_FILES_COMPARISON.md)

2. **查看相关方案**:
   - 📋 [current_plans/](current_plans/) - 当前有效的方案

3. **参考技术文档**:
   - 📚 [technical_reference/](technical_reference/) - 技术细节

### 如果你要验证结果

1. **最新验证报告**:
   - 📊 [technical_reference/DIBS_VERIFICATION_REPORT_20260116.md](technical_reference/DIBS_VERIFICATION_REPORT_20260116.md)

2. **查看历史验证**:
   - 📚 [technical_reference/VALIDATION_SUMMARY.md](technical_reference/VALIDATION_SUMMARY.md)

---

## 📝 维护指南

### 添加新文档时的规则

1. **必读指南** (`essential_guides/`):
   - 仅添加包含关键信息、不读可能引入错误的文档
   - 示例：数据使用关键提醒、API重要变更等

2. **当前方案** (`current_plans/`):
   - 仅添加当前正在执行或准备执行的方案
   - 当方案执行完毕后，考虑是否保留或移除

3. **技术参考** (`technical_reference/`):
   - 添加技术设计、验证报告等参考文档
   - 定期清理过时的技术文档

4. **使用指南** (`guides/`):
   - 添加面向用户的使用说明和教程

### 方案完成后的处理

当方案完成并验证后：
- 保留方案文档在 `current_plans/`（作为历史参考）
- 或移除旧版本，只保留最终总结文档

---

## 📊 整理统计

**整理日期**: 2026-01-23
**整理内容**:

| 类别 | 文档数 | 说明 |
|------|--------|------|
| **必读指南** | 4 | 不读可能引入错误 |
| **当前方案** | 4 | 正在执行或准备执行 |
| **技术参考** | 25 | 技术细节和验证 |
| **使用指南** | 6 | 环境配置和使用 |
| **索引文件** | 2 | README.md, INDEX.md |
| **总计** | **41** | - |

**主要改进**:
- ✅ 将关键文档从135个中提取出来，形成清晰的分级
- ✅ 必读指南集中管理，避免引入错误
- ✅ 当前方案清晰可见，任务状态明确
- ✅ 技术参考有序分类，易于查找
- ✅ 简化的目录结构，仅保留4个分类

**与之前版本的变化**:
- ❌ 移除了 `archived/` 目录（历史归档不再保留）
- ❌ 移除了 `reports/` 目录（实验报告已移出docs）
- ✅ 合并了值转换相关的3个文档为1个综合指南
- ✅ 清理了顶层目录，仅保留索引文件

---

## 📌 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|---------|
| v2.0 | 2026-01-23 | 简化结构，移除归档目录，更新文档统计 |
| v1.0 | 2026-01-23 | 初始版本，建立分级体系 |

---

**维护者**: Claude (AI Assistant)
**整理人**: Green
**最后更新**: 2026-01-23
