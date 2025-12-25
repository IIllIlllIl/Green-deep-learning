# 项目文档总索引

**最后更新**: 2025-12-24
**项目**: Causality-Aided Fairness Trade-off Analysis (ASE 2023论文复现 + 能耗研究扩展)

---

## 📂 文档结构

本项目的文档分为以下几个部分：

### 1. 核心技术文档（顶层）

位置：`docs/`

| 文档 | 用途 | 受众 |
|------|------|------|
| **CODE_WORKFLOW_EXPLAINED.md** | 代码整体流程、各阶段目的和性能特征 | 开发者、代码理解 |
| **MIGRATION_GUIDE.md** | 迁移系统到新数据集的完整指南 | 研究者、新数据集应用 |

### 2. 使用指南

位置：`docs/guides/`

| 文档 | 用途 | 受众 |
|------|------|------|
| **ENVIRONMENT_SETUP.md** | 环境配置和依赖安装 | 新用户、首次设置 |
| **REPLICATION_QUICK_START.md** | 快速开始复现实验 | 论文复现者 |
| **USAGE_GUIDE_FOR_NEW_RESEARCH.md** | 用于新研究的使用指南 | 高级研究者 |
| **IMPROVEMENT_GUIDE.md** | 系统改进和优化指南 | 贡献者、开发者 |
| **DOCUMENTATION_INDEX.md** | 旧版文档索引（可能过时） | 参考 |

### 3. 实验报告

位置：`docs/reports/`

| 文档 | 用途 | 实验日期 |
|------|------|---------|
| **DATA_FLOW_EXPLANATION_20251224.md** 📊 ⭐⭐⭐ | 数据流程完整说明（726→648→536/594） | 2025-12-24 |
| **MRT_OAST_FEASIBILITY_ANALYSIS.md** 🔬 ⭐⭐ | MRT-OAST作为第6组可行性分析 | 2025-12-24 |
| **5GROUPS_DATA_GENERATION_REPORT_20251224.md** 📋 ⭐⭐ | 5组数据生成报告（含6组扩展建议） | 2025-12-24 |
| **ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md** ⭐ | Adult数据集完整因果分析（首次成功） | 2025-12-21 |
| **ADULT_DATASET_VALIDATION_REPORT.md** | Adult数据集验证报告 | 2025-12-21 |
| **LARGE_SCALE_EXPERIMENT_REPORT.md** | 大规模实验报告 | 2025-12-21 |
| **TEST_VALIDATION_REPORT.md** | 测试验证报告 | 2025-12-20 |
| **REPLICATION_EVALUATION.md** | 复现评估报告 | 2025-12-17 |
| **DATA_ISOLATION_MIGRATION_REPORT.md** 🔧 | 数据隔离迁移报告（按用途分类） | 2025-12-22 |

⭐ **推荐阅读**: 最新、最完整的实验报告
📊 **数据流程**: 数据筛选和分组方案说明
🔬 **可行性**: 新任务组可行性分析
📋 **执行报告**: 数据生成执行记录
🔧 **项目管理**: 数据组织和结构变更

### 4. 归档文档

位置：`docs/reports/archives/`

这些是项目早期的阶段性报告，已被更完整的报告替代：

- **阶段性报告** (5个): STAGE1_*.md
- **项目状态** (3个): PROJECT_STATUS*.md, PROGRESS_UPDATE.md
- **技术评估** (4个): PAPER_COMPARISON_REPORT.md, GPU_TEST_REPORT.md, CODE_REVIEW_REPORT.md, TASK_COMPLETION_SUMMARY.md
- **计划文档** (3个): FULL_REPLICATION_PLAN.md, QUICK_SUMMARY.md, DELIVERY_CHECKLIST.md

### 5. 数据与结果组织 ⭐ **[2025-12-22 新增]**

位置：`data/`, `results/`, `logs/`

**重要变更**: 数据和结果已按用途隔离为两大类：

| 目录 | 说明文档 | 用途 |
|------|---------|------|
| **data/** | [data/README.md](../data/README.md) | 数据集组织和管理 |
| **results/** | [results/README.md](../results/README.md) | 因果分析结果 |

**隔离原则**:
- `paper_replication/` - 论文复现数据和结果（Adult, COMPAS, German）
- `energy_research/` - 能耗研究数据和结果（主项目扩展）

这种隔离确保两类研究的数据和结果不会混淆。详见各目录的README.md。

### 6. 能耗研究方案文档 🔬 **[2025-12-24 重大更新 - 6组方案]**

位置：`docs/`

**研究框架** (v5.0 - 6组方案 ⭐⭐⭐):
- **任务分组**: 6个任务组（examples、resnet、Person_reID、VulBERTa、Bug定位、**MRT-OAST**）
- **数据保留率**: 81.8%（594/726行）
- **模型覆盖**: 11/11模型（100%）
- **起始阶段**: Stage2 (mediators.csv, 726行, 46.49%空值率)
- **数据优势**: +58行相比5组方案（+10.8%）

| 文档 | 用途 | 状态 |
|------|------|------|
| **[6GROUPS_DATA_GENERATION_EXECUTION_REPORT_20251224.md](reports/6GROUPS_DATA_GENERATION_EXECUTION_REPORT_20251224.md)** 📋 ⭐⭐⭐ | **6组数据生成执行报告**（完整总结） | ✅ **最新** |
| **[6GROUPS_DIBS_EXPERIMENT_INSTRUCTIONS_20251224.md](reports/6GROUPS_DIBS_EXPERIMENT_INSTRUCTIONS_20251224.md)** 🚀 ⭐⭐⭐ | **DiBS因果分析实验指令**（完整bash脚本） | ✅ **最新** |
| **[6GROUPS_DATA_GENERATION_PLAN_20251224.md](reports/6GROUPS_DATA_GENERATION_PLAN_20251224.md)** 📋 ⭐⭐⭐ | 6组数据生成完整方案 | ✅ 已完成 |
| **[STAGE_QUALITY_ANALYSIS_20251224.md](reports/STAGE_QUALITY_ANALYSIS_20251224.md)** 🔍 ⭐⭐ | 阶段0-7数据质量分析 | ✅ 已验证 |
| **[DATA_FLOW_EXPLANATION_20251224.md](reports/DATA_FLOW_EXPLANATION_20251224.md)** 📊 ⭐⭐⭐ | 数据流程完整说明（推荐6组方案） | ✅ 参考 |
| **[MRT_OAST_FEASIBILITY_ANALYSIS.md](reports/MRT_OAST_FEASIBILITY_ANALYSIS.md)** 🔬 ⭐⭐ | MRT-OAST可行性分析 | ✅ 已验证 |
| **[VARIABLE_EXPANSION_PLAN.md](reports/VARIABLE_EXPANSION_PLAN.md)** 📋 ⭐⭐ | 变量扩展计划v3.0 | ✅ 参考 |
| **[DATA_QUALITY_REPORT_DETAILED_20251223.md](DATA_QUALITY_REPORT_DETAILED_20251223.md)** 📊 ⭐⭐ | 数据质量详细报告（4组方案） | ✅ 参考 |
| **[ENERGY_DATA_PROCESSING_PROPOSAL.md](ENERGY_DATA_PROCESSING_PROPOSAL.md)** | 能耗数据因果分析处理方案（3种方案对比） | ✅ 完成 |
| **[COLUMN_USAGE_ANALYSIS.md](COLUMN_USAGE_ANALYSIS.md)** | 54列完整使用分析（已纳入/未纳入原因） | ✅ 完成 |
| **[DATA_FILES_COMPARISON.md](DATA_FILES_COMPARISON.md)** | data.csv vs raw_data.csv文件对比说明 | ✅ 完成 |

📋 **当前方案**: v5.0 - **6任务组方案（推荐）** ⭐
📊 **数据保留率**: **81.8%**（594/726行，历史最高）
🎯 **DiBS就绪**: ✅ **已生成6个训练数据文件**（覆盖所有11个模型）
⏳ **下一步**: DiBS因果分析（预计4-7小时）

---

## 🚀 快速开始

### 新用户

1. **环境设置**: 阅读 [ENVIRONMENT_SETUP.md](guides/ENVIRONMENT_SETUP.md)
2. **快速复现**: 阅读 [REPLICATION_QUICK_START.md](guides/REPLICATION_QUICK_START.md)
3. **查看结果**: 阅读 [ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md)

### 理解代码

1. **整体流程**: 阅读 [CODE_WORKFLOW_EXPLAINED.md](CODE_WORKFLOW_EXPLAINED.md)
   - 5个执行阶段
   - 算法原理（DiBS, DML）
   - 性能特征分析

### 应用到新数据集

1. **迁移指南**: 阅读 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
   - 数据集要求
   - 3步迁移流程
   - 完整代码模板
   - 常见陷阱

### 高级研究

1. **使用指南**: 阅读 [USAGE_GUIDE_FOR_NEW_RESEARCH.md](guides/USAGE_GUIDE_FOR_NEW_RESEARCH.md)
2. **改进指南**: 阅读 [IMPROVEMENT_GUIDE.md](guides/IMPROVEMENT_GUIDE.md)

---

## 📊 项目里程碑

### 最新成就

**2025-12-23**: ✅ **能耗数据预处理管道完成（阶段0-7）** ⭐⭐⭐
- 样本量: 648个有效实验（4任务组）
- 数据质量: 图像分类93.3%填充率，Person_reID 96.0%（优秀）
- DiBS就绪: 4个训练数据文件，所有任务组适用性优秀
- 详见: [DATA_QUALITY_REPORT_DETAILED_20251223.md](DATA_QUALITY_REPORT_DETAILED_20251223.md)

**2025-12-21**: ✅ **首次完成Adult数据集完整因果分析**
- 运行时间: 61.4分钟
- 配置数: 10个
- DiBS因果边: 6条
- DML显著效应: 4条
- 复现质量: 90% (4.5/5)
- 详见: [ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md)

### 关键发现

1. **过拟合证据**: Tr_F1 → Te_Acc, ATE = -0.052 (训练F1提高导致测试准确率下降)
2. **训练-测试一致性**: Te_Acc → Tr_Acc, ATE = 0.91 (高度相关)
3. **DiBS速度优化**: 从超时(>1小时)到成功(1.6分钟)，提升>97%

---

## 📁 其他重要目录

### 脚本目录 (`scripts/`)

```
scripts/
├── demos/              # 演示脚本
│   ├── demo_quick_run.py
│   ├── demo_large_scale.py
│   ├── demo_adult_dataset.py
│   └── demo_adult_full_analysis.py
├── experiments/        # 完整实验脚本
│   └── run_adult_analysis.sh
├── utils/              # 工具脚本
│   ├── monitor_progress.sh
│   └── activate_env.sh
└── testing/            # 测试脚本
    ├── test_dibs_quick.py
    └── run_tests.py
```

### 日志目录 (`logs/`)

**已隔离**: 按用途分为论文复现和能耗研究

```
logs/
├── paper_replication/          # 论文复现日志
│   ├── experiments/
│   │   ├── adult_full_analysis_20251221_163516.log
│   │   ├── adult_dataset_run.log
│   │   └── large_scale_run.log
│   ├── demos/
│   │   └── demo_output.log
│   └── status/
│       └── adult_analysis_status.txt
├── energy_research/            # 能耗研究日志（待添加）
├── experiments/                # 旧日志（保留）
├── demos/                      # 旧日志（保留）
└── status/                     # 旧日志（保留）
```

详见：[data/README.md](../data/README.md), [results/README.md](../results/README.md)

### 核心代码 (`utils/`)

- **model.py**: FFNN神经网络和训练器
- **metrics.py**: 性能和公平性指标计算
- **fairness_methods.py**: 公平性方法（Reweighing等）
- **causal_discovery.py**: DiBS因果图学习
- **causal_inference.py**: DML因果推断
- **tradeoff_detection.py**: 权衡检测

---

## 🔗 外部资源

- **论文**: "Causality-Aided Trade-off Analysis for Machine Learning Fairness" (ASE 2023)
- **数据集**: UCI Adult Dataset
- **依赖库**: PyTorch, JAX, EconML, AIF360

---

## 📝 维护建议

### 文档更新规则

1. **新实验报告**: 添加到 `docs/reports/`，在本索引中更新
2. **过时报告**: 移动到 `docs/reports/archives/`
3. **新指南**: 添加到 `docs/guides/`
4. **技术文档**: 更新顶层文档（CODE_WORKFLOW_EXPLAINED.md等）

### 版本控制

- 重要文档应包含更新时间
- 实验报告应包含实验日期
- 归档文档保留原始时间戳

---

**文档组织者**: Claude Code
**项目维护**: 持续更新中
