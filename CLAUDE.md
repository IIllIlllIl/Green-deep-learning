# Claude 助手指南 - Mutation-Based Training Energy Profiler

**项目版本**: v4.7.9 (2025-12-20)
**最后更新**: 2025-12-20
**状态**: 🎉 Complete - All 11 Models 100% Achieved!

---

## 📋 任务执行要求

**重要**: 每次执行任务前必须遵循以下流程：

1. **理解与规划**：首先确认从 `CLAUDE.md` 中获取的当前进度和规范，并列出本次任务的具体步骤。
2. **开发与检查**：若需修改、生成或删除代码，请先编写/更新测试验证，再运行全量测试确保兼容性。
3. **维护与归档**：任务完成后，更新 `README.md`、`CLAUDE.md` 中的进度，并归档旧文件。
4. **使用中文回答**：所有回复和文档必须使用中文。

---

## 🎯 项目概述

### 核心目标
研究深度学习训练超参数对能耗和性能的影响。通过自动化变异超参数、监控能耗、收集性能指标，支持大规模实验研究。

### 当前状态 🎉

**🎉 重大成就 (2025-12-19)**: Phase 7完成，项目目标100%达成！

| 指标 | 当前值 | 状态 |
|------|--------|------|
| **实验总数** | 676个 | ✅ 完成 |
| **有效模型** | 11个 | ✅ 100%达标 |
| **训练成功率** | 676/676 (100.0%) | ✅ 完美 |
| **能耗数据** | 616/676 (91.1%) | ✅ 优秀 |
| **性能数据** | 616/676 (91.1%) | ✅ 优秀 |
| **参数-模式组合** | 90/90 (100%) | ✅ 完成 |

**详细进度**: 查看 [项目进度完整总结](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md) ⭐⭐⭐

### 实验目标

每个超参数在两种模式（并行/非并行）下都需要：
1. **1个默认值实验** - 建立基线
2. **5个唯一单参数变异实验** - 研究单参数影响
3. **完整数据**: 能耗 + 性能指标

**总目标**: 45参数 × 2模式 × 6实验 = **540个有效实验**
**实际达成**: 616个有效实验，90/90参数-模式组合全部达标 (100%) 🎊

---

## 📁 项目结构

### 目录结构
```
energy_dl/nightly/
├── docs/                    # 所有文档文件
│   ├── results_reports/    # 实验结果报告 ⭐
│   ├── environment/        # 环境配置文档
│   ├── settings_reports/   # 配置相关报告
│   └── archived/           # 过时文档归档
├── scripts/                # 辅助脚本（Python/Bash）
├── tests/                  # 测试文件
├── settings/               # 运行配置文件
│   └── archived/          # 过时配置归档
├── results/               # 实验结果数据
├── mutation/              # 核心代码
├── config/                # 模型配置
└── analysis/              # 因果推断分析模块 🔬
    ├── docs/              # 分析模块文档
    │   ├── INDEX.md       # 文档总索引 ⭐
    │   ├── CODE_WORKFLOW_EXPLAINED.md  # 代码流程详解
    │   ├── MIGRATION_GUIDE.md          # 数据迁移指南
    │   ├── guides/        # 使用指南
    │   └── reports/       # 实验报告
    ├── scripts/           # 分析脚本
    │   ├── demos/         # 演示脚本
    │   ├── experiments/   # 实验脚本
    │   └── utils/         # 工具脚本
    ├── utils/             # 核心模块
    │   ├── causal_discovery.py   # DiBS因果图学习
    │   ├── causal_inference.py   # DML因果推断
    │   ├── model.py              # 神经网络模型
    │   ├── metrics.py            # 指标计算
    │   └── fairness_methods.py   # 公平性方法
    ├── tests/             # 测试套件
    ├── data/              # 训练数据
    ├── results/           # 因果分析结果
    ├── logs/              # 分析日志
    ├── config.py          # 配置参数
    ├── requirements.txt   # 依赖清单
    └── README.md          # 分析模块说明
```

### 核心文件位置

**代码文件**:
- `mutation/runner.py` - 主运行器
- `mutation/session.py` - Session管理
- `mutation/energy_monitor.py` - 能耗监控
- `config/models_config.json` - 模型配置

**数据文件**:
- `results/raw_data.csv` - **主数据文件** (676行，87列) ⭐⭐⭐
  - 验证脚本: `scripts/validate_raw_data.py`
  - 追加脚本: `scripts/append_session_to_raw_data.py`
  - 备份文件: `results/raw_data.csv.backup_20251219_154656`

**配置文件**:
- 已完成: `settings/stage2_optimized_*.json`, `stage3_4_merged_*.json`
- 归档: `settings/archived/` - 过时配置文件

---

## 📚 核心文档索引

### 快速参考指南
- [项目进度完整总结](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md) - **项目总体状况** ⭐⭐⭐
- [FEATURES_OVERVIEW.md](docs/FEATURES_OVERVIEW.md) - 功能特性总览
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - 快速参考
- [SCRIPTS_QUICKREF.md](docs/SCRIPTS_QUICKREF.md) - 脚本快速参考

### 配置与规范
- [JSON_CONFIG_WRITING_STANDARDS.md](docs/JSON_CONFIG_WRITING_STANDARDS.md) - **JSON配置书写规范** ⭐⭐⭐
- [JSON_CONFIG_BEST_PRACTICES.md](docs/JSON_CONFIG_BEST_PRACTICES.md) - 配置最佳实践
- [SETTINGS_CONFIGURATION_GUIDE.md](docs/SETTINGS_CONFIGURATION_GUIDE.md) - 配置指南
- [11_MODELS_FINAL_DEFINITION.md](docs/11_MODELS_FINAL_DEFINITION.md) - 11个模型最终定义

### 执行报告
- [PHASE7_EXECUTION_REPORT.md](docs/results_reports/PHASE7_EXECUTION_REPORT.md) - **Phase 7最终补齐** ⭐⭐⭐
- [PHASE6_EXECUTION_REPORT.md](docs/results_reports/PHASE6_EXECUTION_REPORT.md) - Phase 6 VulBERTa补齐
- [PHASE5_PERFORMANCE_METRICS_MISSING_ISSUE.md](docs/results_reports/PHASE5_PERFORMANCE_METRICS_MISSING_ISSUE.md) - Phase 5执行
- [PHASE4_VALIDATION_EXECUTION_REPORT.md](docs/results_reports/PHASE4_VALIDATION_EXECUTION_REPORT.md) - Phase 4验证
- [STAGE3_4_EXECUTION_REPORT.md](docs/results_reports/STAGE3_4_EXECUTION_REPORT.md) - Stage3-4合并执行
- [STAGE7_8_FIX_EXECUTION_REPORT.md](docs/results_reports/STAGE7_8_FIX_EXECUTION_REPORT.md) - Stage7-8修复执行

### Bug修复报告
- [STAGE11_BUG_FIX_REPORT.md](docs/results_reports/STAGE11_BUG_FIX_REPORT.md) - 并行runs_per_config修复 ⭐⭐⭐
- [CSV_FIX_COMPREHENSIVE_SUMMARY.md](docs/results_reports/CSV_FIX_COMPREHENSIVE_SUMMARY.md) - CSV追加bug综合报告
- [DEDUP_MODE_FIX_REPORT.md](docs/results_reports/DEDUP_MODE_FIX_REPORT.md) - 去重模式区分修复
- [STAGE7_CONFIG_FIX_REPORT.md](docs/results_reports/STAGE7_CONFIG_FIX_REPORT.md) - Stage7-13配置修复
- [VULBERTA_CNN_CLEANUP_REPORT_20251211.md](docs/results_reports/VULBERTA_CNN_CLEANUP_REPORT_20251211.md) - VulBERTa/cnn清理

### 数据格式与设计
- [DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md) - **数据格式设计决定** ⭐⭐⭐
- [SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md](docs/results_reports/SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md) - 80列vs93列对比
- [OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md](docs/results_reports/OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md) - 背景超参数分析
- [OUTPUT_STRUCTURE_QUICKREF.md](docs/OUTPUT_STRUCTURE_QUICKREF.md) - 输出结构参考

---

## 🔬 因果推断分析模块 (analysis/)

### 模块概述

`analysis/` 是一个独立的因果推断分析模块，用于 **ASE 2023论文复现 + 能耗研究扩展**：
- **论文标题**: *"Causality-Aided Trade-off Analysis for Machine Learning Fairness"*
- **研究目标**:
  - 原论文：研究ML模型中公平性、准确率、鲁棒性等指标之间的因果关系和权衡
  - 能耗扩展：研究深度学习训练超参数对能耗和性能的因果影响 ⭐
- **核心技术**: DiBS（因果图学习）+ DML（因果推断）
- **项目状态**:
  - ✅ Adult数据集完整因果分析完成（2025-12-21）
  - ⏳ 能耗数据因果分析方案确认中（2025-12-22）

### 数据与结果组织 🗂️ **[2025-12-22 重要更新]**

analysis 模块的数据和结果已按**用途隔离**为两大类，确保论文复现和能耗研究的数据不会混淆：

**隔离原则**:
```
analysis/
├── data/                       # 训练数据目录
│   ├── paper_replication/      # 论文复现数据（Adult, COMPAS, German）
│   │   └── adult_training_data.csv  (10个配置，2方法×5alpha)
│   └── energy_research/        # 能耗研究数据（主项目扩展）
│       ├── raw/                # 原始数据（从主项目复制）
│       │   └── energy_data_original.csv  (data.csv副本, 54列)
│       └── processed/          # 预处理后的因果分析数据（待生成）
│           └── training_data_*.csv  (4个任务组，分层分析)
│
├── results/                    # 因果分析结果目录
│   ├── paper_replication/      # 论文复现结果
│   │   ├── adult_causal_graph.npy  (6条因果边)
│   │   └── adult_causal_edges.pkl  (4条显著)
│   └── energy_research/        # 能耗研究结果（待生成）
│       ├── energy_causal_graph.npy  (待生成)
│       └── task_specific/      # 分层分析结果
│
└── logs/                       # 运行日志目录
    ├── paper_replication/      # 论文复现日志
    │   └── experiments/
    │       └── adult_full_analysis_20251221_163516.log
    └── energy_research/        # 能耗研究日志（待添加）
```

**关键说明**:
- `paper_replication/` - 用于ASE 2023论文方法复现（Adult, COMPAS, German数据集）
- `energy_research/` - 用于主项目能耗数据的因果分析（616个实验，11个模型）
- 两类数据和结果完全隔离，避免混淆和误用

**详细文档**: 查看 [analysis/data/README.md](analysis/data/README.md) 和 [analysis/results/README.md](analysis/results/README.md)

### 与主项目的关系

| 维度 | 主项目 (energy_dl/nightly) | 分析模块 (analysis/) |
|------|-------------------------|-------------------|
| **研究目标** | 训练超参数 → 能耗 + 性能 | 超参数/方法 → 公平性 + 性能 |
| **数据收集** | ✅ 676个实验，91.1%有效 | 使用主项目数据或独立数据 |
| **核心分析** | 描述性统计、可视化 | 因果图学习、因果效应估计 |
| **主要输出** | `raw_data.csv`（能耗+性能） | 因果图、因果效应、权衡模式 |
| **技术栈** | PyTorch, perf, nvidia-smi | DiBS (JAX), DML (EconML), AIF360 |

**集成路径**:
```
主项目收集数据 (raw_data.csv)
    ↓
转换为因果分析格式
    ↓
应用DiBS学习因果图
    ↓
应用DML估计因果效应
    ↓
检测权衡模式（如能耗 vs 性能）
```

### 核心功能

#### 1. DiBS因果图学习 (`utils/causal_discovery.py`)
- **功能**: 从观测数据中学习变量间的有向无环图(DAG)
- **算法**: Differentiable Bayesian Structure Learning
- **输入**: 实验数据矩阵 (N个配置 × P个指标)
- **输出**: 邻接矩阵 (P×P) + 因果边列表
- **应用**: 发现超参数 → 性能/公平性的因果路径

#### 2. DML因果推断 (`utils/causal_inference.py`)
- **功能**: 估计因果边的平均因果效应(ATE)和置信区间
- **算法**: Double Machine Learning（消除混淆偏差）
- **输入**: DiBS学习的因果图 + 原始数据
- **输出**: 每条边的 ATE、标准误、p值、显著性
- **应用**: 量化"训练F1提高1单位 → 测试准确率降低5.2%"

#### 3. 权衡检测 (`utils/tradeoff_detection.py`)
- **功能**: 识别指标间的权衡模式（一个变量同时对两个目标有相反影响）
- **算法**: 论文Algorithm 1
- **示例权衡**: accuracy vs fairness, performance vs robustness
- **应用**: 自动发现训练策略的副作用

### 快速开始

#### 方式1: 运行Adult数据集完整分析

```bash
# 进入analysis目录
cd analysis

# 激活环境
conda activate fairness

# 运行完整分析（约60分钟，GPU加速）
bash scripts/experiments/run_adult_analysis.sh

# 或后台运行
nohup bash scripts/experiments/run_adult_analysis.sh > adult_analysis.log 2>&1 &

# 监控进度
bash scripts/utils/monitor_progress.sh
```

**预期输出**:
- `data/adult_training_data.csv` - 10个配置的训练数据
- `results/adult_causal_graph.npy` - DiBS学习的因果图
- `results/adult_causal_edges.pkl` - 筛选后的因果边
- `logs/experiments/` - 完整运行日志

#### 方式2: 将主项目数据迁移到analysis分析

**步骤**:

1. **准备数据转换脚本** (`analysis/scripts/convert_energy_data.py`，需创建):

```python
"""
将主项目的raw_data.csv转换为analysis模块可用的格式
"""
import pandas as pd
import numpy as np

# 读取主项目数据
df_energy = pd.read_csv('../results/raw_data.csv')

# 提取关键列（根据实际需求调整）
# 示例：分析 learning_rate 对 能耗和性能的因果影响
analysis_data = df_energy[[
    'repo', 'model', 'mode',
    'learning_rate', 'batch_size', 'epochs',  # 超参数
    'energy_cpu_avg', 'energy_gpu_avg',       # 能耗指标
    'train_loss', 'test_acc', 'test_f1'       # 性能指标
]].copy()

# 处理缺失值
analysis_data = analysis_data.dropna()

# 保存
analysis_data.to_csv('data/energy_training_data.csv', index=False)
print(f"转换完成: {len(analysis_data)} 样本")
```

2. **应用因果分析**:

```bash
# 运行因果分析（示例，需根据实际数据调整）
cd analysis
python scripts/demos/demo_energy_analysis.py
```

3. **解读结果**:
   - 因果图：哪些超参数影响哪些指标
   - 因果效应：learning_rate提高0.1 → 能耗增加多少W？
   - 权衡检测：是否存在"性能 vs 能耗"权衡？

### 关键文档索引

**必读文档**:
- [analysis/README.md](analysis/README.md) - 模块总体介绍 ⭐⭐⭐
- [analysis/docs/INDEX.md](analysis/docs/INDEX.md) - **文档总索引**（2025-12-22更新）⭐⭐⭐
- [analysis/docs/CODE_WORKFLOW_EXPLAINED.md](analysis/docs/CODE_WORKFLOW_EXPLAINED.md) - 代码流程详解（61分钟完整过程）

**使用指南**:
- [analysis/docs/MIGRATION_GUIDE.md](analysis/docs/MIGRATION_GUIDE.md) - **数据迁移指南**（应用到新数据集）⭐⭐⭐
- [analysis/docs/guides/ENVIRONMENT_SETUP.md](analysis/docs/guides/ENVIRONMENT_SETUP.md) - 环境配置
- [analysis/docs/guides/REPLICATION_QUICK_START.md](analysis/docs/guides/REPLICATION_QUICK_START.md) - 快速复现

**实验报告**:
- [analysis/docs/reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](analysis/docs/reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md) - **Adult数据集完整分析**（首次成功）⭐⭐⭐
- [analysis/docs/reports/REPLICATION_EVALUATION.md](analysis/docs/reports/REPLICATION_EVALUATION.md) - 复现评估报告
- [analysis/docs/reports/VARIABLE_EXPANSION_PLAN.md](analysis/docs/reports/VARIABLE_EXPANSION_PLAN.md) - **变量扩展计划 v3.0**（最终版）⭐⭐⭐

**能耗研究方案** 🔬 **[2025-12-22 新增]**:
- [analysis/docs/ENERGY_DATA_PROCESSING_PROPOSAL.md](analysis/docs/ENERGY_DATA_PROCESSING_PROPOSAL.md) - 能耗数据因果分析处理方案（3种模型处理方案对比）⭐⭐
- [analysis/docs/COLUMN_USAGE_ANALYSIS.md](analysis/docs/COLUMN_USAGE_ANALYSIS.md) - 54列完整使用分析（已纳入/未纳入原因说明）⭐
- [analysis/docs/DATA_FILES_COMPARISON.md](analysis/docs/DATA_FILES_COMPARISON.md) - data.csv vs raw_data.csv 文件对比说明
- [analysis/docs/DATA_ISOLATION_MIGRATION_REPORT.md](analysis/docs/DATA_ISOLATION_MIGRATION_REPORT.md) - 数据隔离迁移报告（按用途分类）

### 技术栈与依赖

**核心库**:
```python
# 因果推断
jax>=0.4.1              # DiBS优化引擎
dibs                    # 因果图学习
econml                  # DML因果推断

# 机器学习
torch>=1.12.0           # 神经网络训练
scikit-learn            # 指标计算

# 公平性
aif360                  # 公平性方法和指标

# 数据处理
pandas
numpy
```

**完整依赖**: 查看 [analysis/requirements.txt](analysis/requirements.txt)

### 主要成就

✅ **首次完成Adult数据集端到端因果分析**（2025-12-21）:
- **运行时间**: 61.4分钟（GPU加速）
- **配置数**: 10个（2方法 × 5 alpha值）
- **因果边检测**: 6条高置信度因果边
- **统计显著**: 4条边的因果效应统计显著（p < 0.05）
- **复现质量**: 90% (4.5/5)

**关键发现**:
1. **过拟合证据**: Tr_F1 → Te_Acc, ATE = -0.052（训练F1提高1单位，测试准确率降低5.2%）
2. **DiBS性能突破**: 从超时(>1小时) → 成功(1.6分钟)，速度提升>97%
3. **DML因果推断**: 4/6边统计显著，提供置信区间的可靠因果效应估计

### 使用场景

#### 场景1: 分析主项目的"能耗 vs 性能"权衡

**步骤**:
1. 提取 `raw_data.csv` 中的能耗和性能列
2. 转换为 analysis 模块格式
3. 运行DiBS学习因果图
4. 使用DML估计因果效应
5. 检测是否存在"learning_rate → energy_gpu"和"learning_rate → test_acc"的相反因果效应

**预期输出**: 发现某些超参数（如batch_size）可能同时降低能耗但也降低准确率

#### 场景2: 研究公平性方法的副作用

**步骤**:
1. 使用 analysis 模块已有的Adult数据集分析
2. 观察Reweighing方法如何影响准确率和公平性
3. 量化"公平性提高 → 准确率下降"的因果效应

**已有结果**: 查看 [ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](analysis/docs/reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md)

#### 场景3: 论文复现与方法验证

**目的**: 验证DiBS和DML在真实数据上的有效性
**数据集**: Adult, COMPAS, German（论文使用的数据集）
**状态**: Adult已完成，COMPAS和German待复现

### 常见问题

#### 1. 如何将主项目数据应用到analysis？

**答**: 参考 [analysis/docs/MIGRATION_GUIDE.md](analysis/docs/MIGRATION_GUIDE.md)，核心步骤：
1. 确定研究问题（如"learning_rate对能耗的因果影响"）
2. 提取相关列（超参数 + 能耗 + 性能）
3. 转换为数值型（DiBS要求）
4. 创建数据加载脚本（参考 `analysis/scripts/demos/demo_adult_dataset.py`）
5. 运行因果分析

#### 2. DiBS学习需要多少样本？

**答**:
- **最少**: 10个配置（Adult实验使用）
- **推荐**: 20-50个配置（更高统计功效）
- **理想**: 100+个配置（论文级别）

主项目有 **676个实验**，完全足够进行因果分析！

#### 3. 如何解读因果效应(ATE)？

**答**: ATE = 平均因果效应（Average Treatment Effect）
- **示例**: ATE(Tr_F1 → Te_Acc) = -0.052
- **解释**: 训练F1提高1个单位（如从0.60到1.60），测试准确率降低0.052（5.2%）
- **统计显著**: 如果p < 0.05且置信区间不包含0，则因果关系可靠

#### 4. analysis模块可以分析其他数据集吗？

**答**: 可以！参考迁移指南，核心要求：
- 数据是表格型（CSV/DataFrame）
- 有明确的输入变量（超参数、方法等）
- 有明确的输出变量（性能、能耗、公平性等）
- 样本量 > 10（越多越好）

### 当前任务：变量扩展计划 (2025-12-22) ✅ **[方案确认完成 - v3.0最终版]**

**背景**: 原始因果分析只使用了15个变量（27.8%列使用率），存在大量未利用的有价值列。

**目标**: 扩展到更全面的变量集，提升因果发现能力。

**核心改进**:
- 样本量从10个提升到370个（**37倍提升**）
- 图像分类组从26个提升至185个（**7倍提升**，合并MNIST+CIFAR-10）
- 新增One-Hot编码控制异质性（避免DiBS混淆基线差异和因果效应）
- 变量数从15个优化至14-17个/任务组（动态选择）

**进度**:

✅ **方案1：超参数统一** (已确认)
- 统一 `training_duration` = epochs + max_iter
- 统一 `l2_regularization` = weight_decay + alpha
- 新增 `seed` = 随机种子（填充率34.5%）
- 数据验证完成：互斥性100%，无冲突

✅ **方案2：能耗中介变量** (已确认)
- 新增5个中介变量（填充率79.4%）：
  - `gpu_util_avg` - GPU利用率（主中介）
  - `gpu_temp_max` - 最高温度（散热压力）
  - `cpu_pkg_ratio` - CPU计算能耗比
  - `gpu_power_fluctuation` = max_watts - min_watts（负载波动性）
  - `gpu_temp_fluctuation` = temp_max - temp_avg（**温度波动性**，范围0-11°C）
- **因果意义**: 探索"超参数 → 中介变量 → 能耗"的因果路径

✅ **方案3：分层因果分析 + One-Hot编码** (已确认 - v3.0最终版 - 2025-12-22)
- **决策**: 采用**4个任务组**（合并MNIST+CIFAR-10为"图像分类"）
- **任务分组**:
  | 任务组 | 样本量 | 性能指标 | One-Hot变量 |
  |--------|--------|----------|------------|
  | 图像分类 (MNIST+CIFAR-10) | **185个** | test_accuracy | `is_mnist`, `is_cifar10` (2个) |
  | Person_reID检索 | 93个 | mAP | `is_densenet121`, `is_hrnet18`, `is_pcb` (3个) |
  | VulBERTa漏洞检测 | 52个 | eval_loss | 无（单模型） |
  | Bug定位 | 40个 | top1_accuracy | 无（单模型） |
- **关键突破**: One-Hot编码避免DiBS将数据集/模型基线差异误判为因果关系
- **优势**: 保留语义完整性，样本量充足（全部满足DiBS要求），支持任务特定优化建议

**文档**:
- [VARIABLE_EXPANSION_PLAN.md](analysis/docs/reports/VARIABLE_EXPANSION_PLAN.md) **v3.0** - 完整扩展方案（最终版） ⭐⭐⭐
- [ENERGY_DATA_PROCESSING_PROPOSAL.md](analysis/docs/ENERGY_DATA_PROCESSING_PROPOSAL.md) - 能耗数据因果分析处理方案（3种模型处理方案对比）⭐⭐
- [COLUMN_USAGE_ANALYSIS.md](analysis/docs/COLUMN_USAGE_ANALYSIS.md) - 54列完整使用分析（已纳入/未纳入原因说明）⭐
- [DATA_FILES_COMPARISON.md](analysis/docs/DATA_FILES_COMPARISON.md) - data.csv vs raw_data.csv 文件对比说明

**下一步**:
1. ✅ 完成方案确认（方案1-3全部确认，v3.0最终版）
2. ⏳ 实现预处理脚本 `analysis/scripts/preprocess_stratified_data.py`（支持One-Hot编码）
3. ⏳ 运行**4个任务组**的分层DiBS分析（预估60分钟，可并行）
4. ⏳ 生成任务特定和综合报告（对比跨任务共性）
5. ⏳ 对比v1.0（Adult, 10样本）vs v3.0（能耗分层, 370样本）的因果发现改进

---

### 长期计划

1. **集成到主项目分析流程**:
   - 创建 `scripts/run_causal_analysis.py`
   - 自动从 `raw_data.csv` 提取数据并运行因果分析
   - 生成因果分析报告

2. **扩展到更多数据集**:
   - 运行COMPAS数据集分析
   - 运行German数据集分析
   - 验证方法的普遍性

3. **可视化改进**:
   - 交互式因果图展示
   - 因果效应热力图
   - 权衡模式可视化

---

## 🔧 开发工作流程

### 1. 添加新功能
1. 在`mutation/`相应模块中实现功能
2. 在`tests/`中添加测试用例
3. 在`docs/`中更新相关文档
4. 验证通过后提交

### 2. 修复Bug
1. 在`tests/`中创建复现测试
2. 在`mutation/`中修复代码
3. 运行所有测试确保无回归
4. 在`docs/results_reports/`中记录修复报告

### 3. 更新配置
1. 在`settings/`中创建新配置文件
2. 在`docs/settings_reports/`中说明配置变更
3. 测试配置有效性
4. 归档旧配置文件到`settings/archived/`

### 4. 生成报告
1. 分析脚本放在`scripts/`
2. 报告文档放在`docs/results_reports/`
3. 使用标准命名：`{主题}_{描述}.md`
4. 包含时间戳和版本信息

---

## ⚠️ 关键注意事项

### 1. 实验ID唯一性 ⭐⭐⭐ **[关键经验]**

**问题**: 不同批次的实验会产生相同的experiment_id

**正确做法**: 使用 **experiment_id + timestamp** 作为唯一标识符

```python
# ✅ 正确 - 使用复合键
composite_key = f"{row['experiment_id']}|{row['timestamp']}"
if composite_key in existing_keys:
    ...
```

**详细**: [Phase 5性能指标缺失问题分析](docs/results_reports/PHASE5_PERFORMANCE_METRICS_MISSING_ISSUE.md) ⭐⭐⭐

### 2. JSON配置书写规范 ⭐⭐⭐

**核心原则**: 每个实验配置只能变异一个超参数

**必须遵循**:
- ✅ 使用`"mutate": ["参数名"]`数组格式（**单参数变异**）
- ✅ 使用`"repo"`而非`"repository"`
- ✅ 使用`"mode"`而非`"mutation_type"`
- ❌ **禁止**使用`"mutate_params"`对象格式

**配置类型**:
1. **默认值实验**: `"mode": "default"` - 建立基线
2. **单参数变异**: `"mode": "mutation"`, `"mutate": ["参数名"]` - 研究单参数影响
3. **并行模式**: 使用`foreground/background`嵌套结构

**示例**:
```json
{
  "comment": "VulBERTa/mlp - learning_rate变异",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "mutate": ["learning_rate"],
  "runs_per_config": 5
}
```

**详细文档**: [JSON_CONFIG_WRITING_STANDARDS.md](docs/JSON_CONFIG_WRITING_STANDARDS.md) ⭐⭐⭐

### 3. CSV数据完整性
- `raw_data.csv`为主数据文件，使用复合键去重
- 追加数据时使用`append_session_to_raw_data.py`
- 每次修改前备份

### 4. 测试要求
- 所有代码变更必须通过现有测试
- 新功能必须添加测试用例
- CSV格式变更必须运行`verify_csv_append_fix.py`

### 5. 文档同步
- 代码变更时更新相关文档
- 使用标准Markdown格式
- 在文档头部包含版本和日期信息

---

## 🚀 快速开始命令

### 查看项目状态
```bash
# 验证数据完整性
python3 scripts/validate_raw_data.py

# 查看最新版本
grep '当前版本' README.md
```

### 运行测试
```bash
# 运行CSV修复验证测试
python3 tests/verify_csv_append_fix.py

# 运行所有单元测试
python3 -m pytest tests/unit/
```

### 数据分析
```bash
# 分析唯一值数量
python3 scripts/analyze_unique_values.py

# 计算实验缺口
python3 scripts/calculate_experiment_gap.py
```

### 调试命令
```bash
# 检查CSV格式
python3 tests/verify_csv_append_fix.py

# 验证JSON配置
python3 -m json.tool settings/stage2_optimized_*.json
```

---

## 📊 最新进展

### v4.7.9 (2025-12-19) - 当前版本 🎉 **[项目完成]**

**Phase 7执行完成**: 最终补齐所有剩余实验（52实验，26.79小时）
- **实验数**: 52个 (VulBERTa/mlp非并行:20 + bug-localization非并行:20 + bug-localization并行:12)
- **执行时间**: 2025-12-17 21:13 - 2025-12-19 01:14
- **数据完整性**: 100% (训练成功、能耗、性能指标)
- **详细报告**: [PHASE7_EXECUTION_REPORT.md](docs/results_reports/PHASE7_EXECUTION_REPORT.md)

**关键Bug修复**: 非并行模式数据提取Bug ⭐⭐⭐
- **问题**: `append_session_to_raw_data.py` 使用错误字段名 `energy_consumption`
- **影响**: 所有非并行实验的能耗和性能数据为空
- **修复**: 统一使用 `energy_metrics` 和 `performance_metrics`，添加字段映射
- **修改文件**: `scripts/append_session_to_raw_data.py:215-268`
- **结果**: 重新合并后100%数据完整

**项目目标100%达成** 🎊:
- 11个模型在两种模式（非并行+并行）下全部达标
- 90/90 参数-模式组合 (100%)
- 有效实验: 616个
- 剩余缺口: 0个实验 ✅

**完整版本历史**: 查看 [项目进度完整总结](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md)

---

## 📞 常见问题

### 1. CSV格式错误
```
症状: GitHub报告"row X should actually have Y columns"
原因: _append_to_summary_all()列不匹配
解决: 运行验证测试，检查runner.py第167-200行
```

### 2. 实验重复运行
```
症状: 相同超参数重复实验
原因: 去重机制未启用或配置错误
解决: 检查配置文件中的use_deduplication和historical_csvs设置
```

### 3. 能耗数据缺失
```
症状: energy_*列为空
原因: perf权限问题或nvidia-smi不可用
解决: 使用sudo运行，检查GPU驱动
```

### 4. 配置执行失败
```
症状: JSON配置文件解析错误或KeyError
解决:
  - 使用python -m json.tool验证JSON格式
  - 确保使用"repo"而非"repository"
  - 并行模式使用foreground/background结构
  - 参考Stage2配置作为模板
```

### 5. KeyError: 'repo'
```
症状: 运行配置文件时报KeyError: 'repo'错误
原因: 配置文件使用了"repository"键而非"repo"
解决: 将配置中所有"repository"改为"repo"
参考: docs/results_reports/STAGE7_CONFIG_FIX_REPORT.md
```

---

## ✅ 质量保证清单

### 每次提交前检查
- [ ] 所有测试通过：`python3 -m pytest tests/`
- [ ] CSV格式正确：`python3 tests/verify_csv_append_fix.py`
- [ ] 文档已更新：相关Markdown文件
- [ ] 配置已归档：过时文件移至`*/archived/`
- [ ] 版本号已更新：README.md和相关文档

### 每次发布前检查
- [ ] 完整功能测试：所有11个模型运行正常
- [ ] 数据完整性验证：raw_data.csv格式正确
- [ ] 性能指标提取：所有模型性能数据完整
- [ ] 能耗监控：CPU/GPU数据准确
- [ ] 文档一致性：所有文档反映最新状态

---

## 📈 数据文件说明

### raw_data.csv (主数据文件) ⭐⭐⭐

**文件信息**:
- 总行数: 676行
- 列数: 87列
- 数据来源: 合并所有实验数据（历史211 + 新265 + Phase4-7共200）

**数据完整性**:
- ✅ 训练成功: 676/676 (100.0%)
- ✅ 能耗数据: 616/676 (91.1%)
- ✅ 性能数据: 616/676 (91.1%)

**相关脚本**:
- `scripts/validate_raw_data.py` - 验证数据完整性
- `scripts/append_session_to_raw_data.py` - 追加session数据
- `scripts/merge_csv_to_raw_data.py` - 合并数据文件

**数据格式设计**: 查看 [DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md) ⭐⭐⭐

---

## 🗂️ 11个有效模型

### 快速模型（训练时间 < 5分钟）
1. **examples/mnist** - 基础MNIST CNN (~2分钟)
2. **examples/mnist_ff** - MNIST前馈网络 (~1.5分钟)
3. **examples/mnist_rnn** - MNIST RNN (~3分钟)
4. **examples/siamese** - Siamese网络 (~4分钟)

### 中速模型（训练时间 5-30分钟）
5. **pytorch_resnet_cifar10/resnet20** - ResNet20 CIFAR10 (~15分钟)
6. **MRT-OAST/default** - 多目标优化 (~20分钟)
7. **VulBERTa/mlp** - 漏洞检测MLP (~25分钟)
8. **bug-localization-by-dnn-and-rvsm/default** - Bug定位 (~18分钟)

### 慢速模型（训练时间 > 30分钟）
9. **Person_reID/densenet121** - 行人重识别 (~45分钟)
10. **Person_reID/hrnet18** - 行人重识别HRNet (~50分钟)
11. **Person_reID/pcb** - 行人重识别PCB (~60分钟)

**详细定义**: [11_MODELS_FINAL_DEFINITION.md](docs/11_MODELS_FINAL_DEFINITION.md)

---

## 🛠️ 关键脚本工具

### 数据处理
- `scripts/merge_csv_to_raw_data.py` - 合并所有实验数据
- `scripts/append_session_to_raw_data.py` - 追加session数据
- `scripts/validate_raw_data.py` - 验证数据完整性

### 分析工具
- `scripts/calculate_experiment_gap.py` - 计算实验缺口
- `scripts/analyze_experiments.py` - 实验数据分析
- `scripts/analyze_unique_values.py` - 唯一值分析

### 配置工具
- `scripts/generate_config.py` - 生成实验配置
- `scripts/validate_config.py` - 验证配置文件
- `scripts/fix_stage_configs.py` - 修复配置文件

### 测试脚本
- `tests/verify_csv_append_fix.py` - CSV追加修复验证
- `tests/test_dedup_mode_distinction.py` - 去重模式区分测试
- `tests/test_runs_per_config_fix.py` - runs_per_config修复测试
- `tests/test_parallel_runs_per_config_fix.py` - 并行模式修复测试

**完整文档**: [SCRIPTS_QUICKREF.md](docs/SCRIPTS_QUICKREF.md)

---

**维护者**: Green
**Claude助手指南版本**: 2.0
**最后更新**: 2025-12-20
**项目状态**: 🎉 完成 - v4.7.9 项目目标100%达成，11个模型全部完全达标！

> **提示**：
> - 本文件为Claude助手的主要参考指南，保持简洁易读
> - 详细内容查看对应的文档链接
> - 每次任务前请先阅读"任务执行要求"章节
> - 当项目结构或规范变更时，及时更新此文档
