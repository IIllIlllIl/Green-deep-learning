# 6分组回归分析数据可用性报告

**报告日期**: 2026-01-14
**分析对象**: `data/raw_data.csv` (1225条记录)
**分析目的**: 评估数据在6分组回归分析方案下的可用性
**参考文档**: `analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md`

---

## 📋 执行摘要

### 总体数据可用性

| 指标 | 数值 | 评级 |
|------|------|------|
| **总记录数** | 1,225 | - |
| **6分组覆盖记录数** | 1,120 | - |
| **✅ 可用记录总数** | **405** | ⚠️ **36.2%** |
| **❌ 不可用记录总数** | 715 | 63.8% |

**关键发现** ⚠️:
- **当前可用率仅36.2%，远低于预期的87.1%（633行）**
- **主要原因**: **超参数数据严重缺失**（所有不可用记录都缺失超参数）
- **可修复性**: 高（通过默认值回溯可大幅提升）

---

## 📊 6分组定义与可用性详情

### 组1a: examples 组

**包含模型**:
- examples/mnist
- examples/mnist_ff
- examples/mnist_rnn
- examples/siamese

**需要的超参数**:
- `hyperparam_batch_size`
- `hyperparam_epochs`
- `hyperparam_learning_rate`
- `hyperparam_seed`

**数据可用性**:
| 指标 | 数值 |
|------|------|
| 总记录数 | 354 |
| ✅ 可用记录 | **158 (44.6%)** |
| ❌ 不可用记录 | 196 (55.4%) |

**不可用原因**:
- 超参数缺失: 196条 (100%)
- 训练失败: 12条 (6.1%)

**超参数缺失详情**:
| 超参数 | 缺失记录数 | 缺失率 |
|-------|-----------|--------|
| `hyperparam_batch_size` | 161 | 45.5% |
| `hyperparam_learning_rate` | 144 | 40.7% |
| `hyperparam_seed` | 143 | 40.4% |
| `hyperparam_epochs` | 140 | 39.5% |

---

### 组1b: pytorch_resnet 组

**包含模型**:
- pytorch_resnet_cifar10/resnet20

**需要的超参数**:
- `hyperparam_epochs`
- `hyperparam_learning_rate`
- `hyperparam_weight_decay`
- `hyperparam_seed`

**数据可用性**:
| 指标 | 数值 |
|------|------|
| 总记录数 | 87 |
| ✅ 可用记录 | **41 (47.1%)** |
| ❌ 不可用记录 | 46 (52.9%) |

**不可用原因**:
- 超参数缺失: 46条 (100%)

**超参数缺失详情**:
| 超参数 | 缺失记录数 | 缺失率 |
|-------|-----------|--------|
| `hyperparam_seed` | 36 | 41.4% |
| `hyperparam_learning_rate` | 30 | 34.5% |
| `hyperparam_weight_decay` | 30 | 34.5% |
| `hyperparam_epochs` | 30 | 34.5% |

---

### 组2: Person_reID 组

**包含模型**:
- Person_reID_baseline_pytorch/densenet121
- Person_reID_baseline_pytorch/hrnet18
- Person_reID_baseline_pytorch/pcb

**需要的超参数**:
- `hyperparam_dropout`
- `hyperparam_epochs`
- `hyperparam_learning_rate`
- `hyperparam_seed`

**数据可用性**:
| 指标 | 数值 |
|------|------|
| 总记录数 | 261 |
| ✅ 可用记录 | **114 (43.7%)** |
| ❌ 不可用记录 | 147 (56.3%) |

**不可用原因**:
- 超参数缺失: 147条 (100%)
- 训练失败: 16条 (10.9%)

**超参数缺失详情**:
| 超参数 | 缺失记录数 | 缺失率 |
|-------|-----------|--------|
| `hyperparam_seed` | 117 | 44.8% |
| `hyperparam_dropout` | 91 | 34.9% |
| `hyperparam_learning_rate` | 91 | 34.9% |
| `hyperparam_epochs` | 90 | 34.5% |

---

### 组3: VulBERTa 组 ❌

**包含模型**:
- VulBERTa/mlp

**需要的超参数**:
- `hyperparam_epochs`
- `hyperparam_learning_rate`
- `hyperparam_weight_decay`
- `hyperparam_seed`

**数据可用性**:
| 指标 | 数值 |
|------|------|
| 总记录数 | 164 |
| ✅ 可用记录 | **29 (17.7%)** ⚠️ |
| ❌ 不可用记录 | 135 (82.3%) |

**不可用原因**:
- 超参数缺失: 135条 (100%)
- 能耗数据缺失: 24条 (17.8%)

**超参数缺失详情**:
| 超参数 | 缺失记录数 | 缺失率 |
|-------|-----------|--------|
| `hyperparam_learning_rate` | 103 | 62.8% |
| `hyperparam_weight_decay` | 102 | 62.2% |
| `hyperparam_seed` | 102 | 62.2% |
| `hyperparam_epochs` | 101 | 61.6% |

**⚠️ 严重问题**: 该组可用率仅17.7%，远低于其他组

---

### 组4: bug_localization 组 ❌

**包含模型**:
- bug-localization-by-dnn-and-rvsm/default

**需要的超参数**:
- `hyperparam_alpha`
- `hyperparam_kfold`
- `hyperparam_max_iter`
- `hyperparam_seed`

**数据可用性**:
| 指标 | 数值 |
|------|------|
| 总记录数 | 149 |
| ✅ 可用记录 | **25 (16.8%)** ⚠️ |
| ❌ 不可用记录 | 124 (83.2%) |

**不可用原因**:
- 超参数缺失: 124条 (100%)
- 能耗数据缺失: 2条 (1.6%)

**超参数缺失详情**:
| 超参数 | 缺失记录数 | 缺失率 |
|-------|-----------|--------|
| `hyperparam_kfold` | 100 | 67.1% |
| `hyperparam_alpha` | 95 | 63.8% |
| `hyperparam_seed` | 95 | 63.8% |
| `hyperparam_max_iter` | 94 | 63.1% |

**⚠️ 严重问题**: 该组可用率仅16.8%，最低的组

---

### 组5: MRT-OAST 组

**包含模型**:
- MRT-OAST/default

**需要的超参数**:
- `hyperparam_dropout`
- `hyperparam_epochs`
- `hyperparam_learning_rate`
- `hyperparam_weight_decay`

**数据可用性**:
| 指标 | 数值 |
|------|------|
| 总记录数 | 105 |
| ✅ 可用记录 | **38 (36.2%)** |
| ❌ 不可用记录 | 67 (63.8%) |

**不可用原因**:
- 超参数缺失: 67条 (100%)

**超参数缺失详情**:
| 超参数 | 缺失记录数 | 缺失率 |
|-------|-----------|--------|
| `hyperparam_learning_rate` | 55 | 52.4% |
| `hyperparam_dropout` | 54 | 51.4% |
| `hyperparam_weight_decay` | 53 | 50.5% |
| `hyperparam_epochs` | 53 | 50.5% |

---

## 🔍 问题分析

### 主要问题：超参数数据严重缺失

**问题规模**: 715条不可用记录中，**100%都是因为超参数缺失**

**缺失原因**:
根据 `analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md` §阶段1的说明：

> **默认值回溯**: 从默认值实验和models_config.json回溯超参数
> - 预期: 超参数填充率 47% → 95%

**当前状态**: 超参数尚未进行默认值回溯，导致大量单参数变异实验的超参数缺失

### 各组问题严重程度排序

| 排名 | 组别 | 可用率 | 问题严重性 | 主要缺失超参数 |
|-----|------|--------|-----------|--------------|
| 1 | **组4: bug_localization** | 16.8% | ❌ 非常严重 | kfold (67.1%), alpha (63.8%) |
| 2 | **组3: VulBERTa** | 17.7% | ❌ 非常严重 | learning_rate (62.8%), weight_decay (62.2%) |
| 3 | **组5: MRT-OAST** | 36.2% | ⚠️ 严重 | learning_rate (52.4%), dropout (51.4%) |
| 4 | **组2: Person_reID** | 43.7% | ⚠️ 中等 | seed (44.8%), dropout (34.9%) |
| 5 | **组1a: examples** | 44.6% | ⚠️ 中等 | batch_size (45.5%), learning_rate (40.7%) |
| 6 | **组1b: resnet** | 47.1% | ⚠️ 中等 | seed (41.4%), learning_rate (34.5%) |

---

## 📈 与预期的对比

### 预期 vs 实际

根据 `analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md` §3.1：

| 组别 | 预期样本量（回溯后） | 实际可用 | 差异 | 差异率 |
|------|-------------------|---------|------|--------|
| **组1a: examples** | 139 | 158 | +19 | +13.7% ✅ |
| **组1b: resnet** | 59 | 41 | -18 | -30.5% ❌ |
| **组2: Person_reID** | 116 | 114 | -2 | -1.7% ✅ |
| **组3: VulBERTa** | 118 | 29 | -89 | -75.4% ❌ |
| **组4: bug_localization** | 127 | 25 | -102 | -80.3% ❌ |
| **组5: MRT-OAST** | 74 | 38 | -36 | -48.6% ❌ |
| **总计** | **633** | **405** | **-228** | **-36.0%** ❌ |

**关键发现**:
- ✅ **组1a和组2接近预期**（差异<15%）
- ❌ **组3和组4严重偏离预期**（差异>75%）
- ❌ **总体缺口228条记录**，主要集中在VulBERTa和bug_localization

### 预期vs实际的原因分析

**预期数据基于**:
- 假设已完成默认值回溯（超参数填充率95%）
- 数据文件: `energy_data_original.csv` (726行)

**实际数据状态**:
- ❌ 尚未进行默认值回溯（超参数填充率约40-50%）
- 数据文件: `data/raw_data.csv` (1225行，更新后）

---

## 💡 修复方案与预期效果

### P0 - 默认值回溯（立即执行）⭐⭐⭐

**操作**: 实现默认值回溯脚本，从默认值实验和models_config.json提取超参数

**预期效果**:

| 组别 | 当前可用 | 预期回溯后 | 提升 |
|------|---------|-----------|------|
| **组1a: examples** | 158 | 139-158 | 持平或略降 |
| **组1b: resnet** | 41 | 59 | +18 (+43.9%) |
| **组2: Person_reID** | 114 | 116 | +2 (+1.8%) |
| **组3: VulBERTa** | 29 | 118 | +89 (+306.9%) ✅ |
| **组4: bug_localization** | 25 | 127 | +102 (+408.0%) ✅ |
| **组5: MRT-OAST** | 38 | 74 | +36 (+94.7%) ✅ |
| **总计** | **405** | **633** | **+228 (+56.3%)** ✅ |

**最终可用率**: 36.2% → **56.5%** (基于1120条有效记录)

**实施步骤**:
1. 创建脚本: `analysis/scripts/backfill_hyperparameters.py`
2. 从以下来源提取默认值:
   - 默认值实验记录
   - `models_config.json`
   - 训练脚本中的默认配置
3. 填充单参数变异实验的其他超参数
4. 验证填充值的合理性

**参考文档**: `analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md` §阶段1

---

## 📋 详细不可用记录

**详细记录已保存至**: `unusable_records_for_regression_detail.csv`

该文件包含715条不可用记录的详细信息：
- Group: 所属组别
- Experiment_ID: 实验ID
- Model: 模型标识
- Training_Success: 训练是否成功
- Has_Energy: 是否有能耗数据
- Has_All_Params: 是否有全部超参数
- Missing_Params: 缺失的超参数列表

---

## 🎯 下一步行动

### 立即执行（P0）⭐⭐⭐

1. **实现默认值回溯脚本**
   - 文件: `analysis/scripts/backfill_hyperparameters.py`
   - 预期提升: +228条可用记录（+56.3%）
   - 预估时间: 2-3小时

2. **验证回溯数据质量**
   - 文件: `analysis/scripts/validate_backfilled_data.py`
   - 预估时间: 30分钟

### 后续任务（P1）

3. **生成6组数据文件**
   - 按方案A'分组保存6个CSV
   - 预估时间: 30分钟

4. **开始回归分析**
   - 对6组分别运行回归分析
   - 预估时间: 2-3小时

---

## 📚 生成的文件

1. **data_usability_for_regression_summary.txt** - 可用性摘要
2. **unusable_records_for_regression_detail.csv** - 不可用记录详情（715条）
3. **analyze_data_usability_for_regression.py** - 分析脚本
4. **docs/DATA_USABILITY_FOR_REGRESSION_20260114.md** - 本报告

---

## 📌 总结

### 当前状态 ⚠️

- **可用率**: 36.2% (405/1120)
- **主要问题**: 超参数数据缺失（100%不可用记录都因此）
- **严重问题组**: VulBERTa (17.7%), bug_localization (16.8%)

### 修复潜力 ✅

- **默认值回溯后**: 可用率可提升至**56.5%** (633/1120)
- **新增可用记录**: +228条 (+56.3%)
- **最受益组别**: VulBERTa (+306.9%), bug_localization (+408.0%)

### 建议行动 🎯

**优先级P0 - 立即执行**:
1. 实现默认值回溯脚本（预计2-3小时）
2. 验证回溯数据质量（预计30分钟）
3. 开始回归分析（回溯完成后）

---

**报告生成时间**: 2026-01-14
**分析工具**: `analyze_data_usability_for_regression.py`
**数据源**: `data/raw_data.csv` (1225条记录)
**参考文档**: `analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md`
