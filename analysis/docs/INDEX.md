# 项目文档总索引

**最后更新**: 2025-12-22
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
| **ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md** ⭐ | Adult数据集完整因果分析（首次成功） | 2025-12-21 |
| **ADULT_DATASET_VALIDATION_REPORT.md** | Adult数据集验证报告 | 2025-12-21 |
| **LARGE_SCALE_EXPERIMENT_REPORT.md** | 大规模实验报告 | 2025-12-21 |
| **TEST_VALIDATION_REPORT.md** | 测试验证报告 | 2025-12-20 |
| **REPLICATION_EVALUATION.md** | 复现评估报告 | 2025-12-17 |
| **DATA_ISOLATION_MIGRATION_REPORT.md** 🔧 | 数据隔离迁移报告（按用途分类） | 2025-12-22 |

⭐ **推荐阅读**: 最新、最完整的实验报告
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

### 6. 能耗研究方案文档 🔬 **[2025-12-22 新增]**

位置：`docs/`

**研究框架**:
- **敏感属性**: 是否并行训练（`is_parallel`）
- **干预方法**: 超参数（learning_rate, batch_size, epochs, dropout, weight_decay, seed）
- **结果变量**: 能耗指标、性能指标

| 文档 | 用途 | 状态 |
|------|------|------|
| **ENERGY_DATA_PROCESSING_PROPOSAL.md** 📋 | 能耗数据因果分析处理方案（3种模型处理方案对比） | 方案阶段 - 采用方案A |
| **COLUMN_USAGE_ANALYSIS.md** 📊 | 54列完整使用分析（已纳入/未纳入原因说明） | 完成 |
| **DATA_FILES_COMPARISON.md** | data.csv vs raw_data.csv 文件对比说明 | 完成 |

📋 **当前方案**: 方案A（协变量方法），将11个模型One-Hot编码，使用全部616个样本
📊 **变量设计**: 18输入（6超参数+1并行模式+11模型）+ 6输出（3能耗+2性能+1时长）= 24变量

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

### 最新成就 (2025-12-21)

✅ **首次完成Adult数据集完整因果分析**
- 运行时间: 61.4分钟
- 配置数: 10个
- DiBS因果边: 6条
- DML显著效应: 4条
- 复现质量: 90% (4.5/5)

详见: [ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md)

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
