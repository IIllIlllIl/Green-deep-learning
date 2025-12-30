# ⚠️ 已废弃 - 数据列增加说明

**⚠️ 废弃原因**: 此报告基于错误的理解——错误加载了raw_data.csv（87列）而非data.csv（56列）

**✅ 正确理解**:
- analysis模块应使用主项目的**data.csv**（56列，精简版）而非raw_data.csv（87列，完整版）
- data.csv已删除32个fg_*详细列，更适合因果分析

**✅ 正确报告**: 请查看 [数据加载纠正报告](DATA_LOADING_CORRECTION_REPORT_20251223.md)

**废弃日期**: 2025-12-23
**原因**: 错误使用了未处理的raw_data.csv而非经过处理的data.csv

---

# 原报告内容（已废弃）

# 数据列增加说明 - data.csv (54列) vs raw_data.csv (87列)

**日期**: 2025-12-23
**对比**: data.csv (54列) → raw_data.csv (87列)  **❌ 错误对比**
**增量**: +33列

---

## 📊 列增加原因总结

### 核心原因：**并行模式前景任务数据完整性**

旧数据文件`data.csv`（54列）是**精简版**，缺少并行模式下前景任务（foreground task）的详细数据。

新数据文件`raw_data.csv`（87列）是**完整版**，包含：
1. ✅ 所有基础列（超参数、能耗、性能）
2. ✅ **并行模式前景任务详细数据** (`fg_*`列) ⭐⭐⭐
3. ✅ 并行模式后台任务信息 (`bg_*`列)

---

## 🔍 新增列详细分类（34个新增列）

### 1. 前景任务能耗指标 (11个)

**前缀**: `fg_energy_*`

| 列名 | 说明 | 单位 |
|------|------|------|
| `fg_energy_cpu_pkg_joules` | 前景任务CPU Package能耗 | 焦耳 |
| `fg_energy_cpu_ram_joules` | 前景任务CPU RAM能耗 | 焦耳 |
| `fg_energy_cpu_total_joules` | 前景任务CPU总能耗 | 焦耳 |
| `fg_energy_gpu_total_joules` | 前景任务GPU总能耗 | 焦耳 |
| `fg_energy_gpu_avg_watts` | 前景任务GPU平均功率 | 瓦特 |
| `fg_energy_gpu_max_watts` | 前景任务GPU最大功率 | 瓦特 |
| `fg_energy_gpu_min_watts` | 前景任务GPU最小功率 | 瓦特 |
| `fg_energy_gpu_util_avg_percent` | 前景任务GPU平均利用率 | 百分比 |
| `fg_energy_gpu_util_max_percent` | 前景任务GPU最大利用率 | 百分比 |
| `fg_energy_gpu_temp_avg_celsius` | 前景任务GPU平均温度 | 摄氏度 |
| `fg_energy_gpu_temp_max_celsius` | 前景任务GPU最高温度 | 摄氏度 |

**重要性**: ⭐⭐⭐
**用途**: 研究并行模式下前景任务的独立能耗特征

---

### 2. 前景任务超参数 (9个)

**前缀**: `fg_hyperparam_*`

| 列名 | 说明 |
|------|------|
| `fg_hyperparam_learning_rate` | 前景任务学习率 |
| `fg_hyperparam_batch_size` | 前景任务批次大小 |
| `fg_hyperparam_epochs` | 前景任务训练轮数 |
| `fg_hyperparam_dropout` | 前景任务Dropout比例 |
| `fg_hyperparam_weight_decay` | 前景任务权重衰减 |
| `fg_hyperparam_alpha` | 前景任务Alpha参数 |
| `fg_hyperparam_seed` | 前景任务随机种子 |
| `fg_hyperparam_max_iter` | 前景任务最大迭代次数 |
| `fg_hyperparam_kfold` | 前景任务K折交叉验证 |

**重要性**: ⭐⭐⭐
**用途**: 区分并行模式下前景任务和后台任务的超参数配置

---

### 3. 前景任务性能指标 (9个)

**前缀**: `fg_perf_*`

| 列名 | 说明 |
|------|------|
| `fg_perf_test_accuracy` | 前景任务测试准确率 |
| `fg_perf_accuracy` | 前景任务准确率（通用） |
| `fg_perf_best_val_accuracy` | 前景任务最佳验证准确率 |
| `fg_perf_test_loss` | 前景任务测试损失 |
| `fg_perf_map` | 前景任务mAP（Person_reID） |
| `fg_perf_rank1` | 前景任务Rank-1准确率 |
| `fg_perf_rank5` | 前景任务Rank-5准确率 |
| `fg_perf_precision` | 前景任务精确率 |
| `fg_perf_recall` | 前景任务召回率 |

**重要性**: ⭐⭐⭐
**用途**: 评估并行模式下前景任务的训练效果

---

### 4. 前景任务元信息 (3个)

**前缀**: `fg_*`

| 列名 | 说明 |
|------|------|
| `fg_repository` | 前景任务仓库名称 |
| `fg_model` | 前景任务模型名称 |
| `fg_training_success` | 前景任务训练是否成功 |

**重要性**: ⭐⭐
**用途**: 标识并行模式下前景任务的基本信息

---

### 5. 其他新增列 (2个)

| 列名 | 说明 | 重要性 |
|------|------|--------|
| `perf_accuracy` | 通用准确率指标 | ⭐⭐ |
| `perf_eval_loss` | 评估损失（VulBERTa） | ⭐⭐ |

**用途**: 补充性能指标字段

---

### 6. 删除的列 (1个)

| 列名 | 说明 | 替代方案 |
|------|------|----------|
| `is_parallel` | 布尔值，标识是否并行 | 使用`mode`列（值为"parallel"或"default"） |

**原因**: `mode`列提供更丰富的语义信息

---

## 🧩 并行模式数据结构说明

### 并行模式实验的数据组织

在**并行模式**下，一个实验记录包含：

```
基础列 (53列)
├── 实验元信息: experiment_id, timestamp, mode="parallel"
├── 后台任务信息:
│   ├── bg_repository - 后台任务仓库
│   ├── bg_model - 后台任务模型
│   ├── bg_note - 后台任务备注
│   └── bg_log_directory - 后台任务日志目录
│
├── 前景任务超参数: hyperparam_* (9个)
├── 前景任务能耗: energy_* (12个)
└── 前景任务性能: perf_* (14个)

新增前景任务详细列 (32列)
├── fg_hyperparam_* (9个) - 前景任务超参数详情
├── fg_energy_* (11个) - 前景任务能耗详情
├── fg_perf_* (9个) - 前景任务性能详情
└── fg_* (3个) - 前景任务元信息

其他新增列 (2列)
└── perf_accuracy, perf_eval_loss
```

**关键理解**:
- **基础列**中的`hyperparam_*`, `energy_*`, `perf_*`可能是**合并值**或**前景任务值**
- **`fg_*`列**明确标识**前景任务独立数据**
- **`bg_*`列**提供**后台任务元信息**（不包含详细能耗和性能）

---

## 📈 为什么需要这些列？

### 1. 并行模式因果分析需求

**问题**: 并行模式下，前景任务和后台任务同时运行，如何区分它们的影响？

**解决**:
- `fg_*`列：前景任务的**独立**能耗、性能数据
- 基础列：**总体**或**前景任务**数据（取决于数据提取策略）
- `bg_*`列：后台任务基本信息

**因果分析价值**:
```
研究问题: 并行模式下，后台任务对前景任务的影响
数据支持:
  - 自变量: bg_repository, bg_model (后台任务类型)
  - 因变量: fg_energy_*, fg_perf_* (前景任务表现)
  - 控制变量: fg_hyperparam_* (前景任务配置)
```

### 2. 非并行模式数据完整性

**非并行模式**: `fg_*`列为空或NaN

**示例**:
```
mode="default":
  - hyperparam_learning_rate: 0.001 ✅
  - fg_hyperparam_learning_rate: NaN ⚠️ (无前景任务概念)

mode="parallel":
  - hyperparam_learning_rate: 0.001 ✅ (前景任务)
  - fg_hyperparam_learning_rate: 0.001 ✅ (前景任务明确标识)
  - bg_repository: "MRT-OAST" ✅ (后台任务)
```

---

## 📊 数据文件对比总结

| 特性 | data.csv (54列) | raw_data.csv (87列) |
|------|----------------|-------------------|
| **版本** | 精简版 | 完整版 |
| **行数** | 676 | 726 (+50) |
| **列数** | 54 | 87 (+33) |
| **并行模式支持** | ⚠️ 部分（缺少fg_*） | ✅ 完整 |
| **前景任务详情** | ❌ 无 | ✅ 有（32列） |
| **后台任务信息** | ✅ 有（4列） | ✅ 有（4列） |
| **适用场景** | 简单统计分析 | 因果分析、并行模式研究 |
| **数据完整性** | 基础完整 | 全面完整 ⭐⭐⭐ |

---

## 🎯 对因果分析的影响

### 积极影响

1. ✅ **并行模式因果分析成为可能**
   - 可以研究"后台任务类型 → 前景任务能耗/性能"的因果关系
   - 可以控制前景任务配置（fg_hyperparam_*）

2. ✅ **变量选择更精确**
   - 明确区分前景任务和总体数据
   - 避免混淆变量（confounding）

3. ✅ **数据样本量增加**
   - 726个实验（vs 676个）
   - VulBERTa: 52→129个（+148%）
   - Bug定位: 40→122个（+205%）

### 潜在挑战

1. ⚠️ **缺失值增加**
   - 非并行模式下，所有`fg_*`列为NaN
   - 需要**分层分析**：并行组 vs 非并行组

2. ⚠️ **变量数量增加**
   - 从54个候选变量增加到87个
   - 需要**更严格的变量筛选**（填充率阈值）

3. ⚠️ **列名规范问题**
   - 主项目使用`repository`而非`repo`
   - 需要更新数据处理脚本

---

## ✅ 建议的处理策略

### 策略1: 分层分析（推荐）⭐⭐⭐

**方案**: 将数据分为两组，分别分析

```python
# 非并行组（使用基础列）
df_nonparallel = df[df['mode'] == 'default']
# 使用列: hyperparam_*, energy_*, perf_*

# 并行组（使用fg_*列）
df_parallel = df[df['mode'] == 'parallel']
# 使用列: fg_hyperparam_*, fg_energy_*, fg_perf_*, bg_*
```

**优势**:
- ✅ 避免缺失值问题
- ✅ 研究问题更明确
- ✅ 可对比并行 vs 非并行的因果差异

### 策略2: 合并列（备选）

**方案**: 合并基础列和fg_列

```python
# 示例：合并learning_rate
df['learning_rate_unified'] = df['fg_hyperparam_learning_rate'].fillna(df['hyperparam_learning_rate'])
```

**优势**:
- ✅ 样本量最大（726个）
- ✅ 单一模型分析

**劣势**:
- ⚠️ 语义混淆（并行和非并行数据混合）
- ⚠️ 无法研究并行模式特定问题

### 策略3: 仅使用基础列（最简单）

**方案**: 忽略所有`fg_*`列，只使用54个基础列

**优势**:
- ✅ 兼容旧分析流程
- ✅ 无需处理缺失值

**劣势**:
- ⚠️ 损失并行模式详细信息
- ⚠️ 无法深入研究前景/后台任务交互

---

## 📖 相关文档

- [能耗数据更新报告](ENERGY_DATA_UPDATE_REPORT_20251223.md) - 数据整合详情
- [数据文件对比说明](../DATA_FILES_COMPARISON.md) - data.csv vs raw_data.csv
- [变量扩展计划 v3.0](VARIABLE_EXPANSION_PLAN.md) - 因果分析变量设计

---

## 🔑 关键结论

### 列增加的3个核心原因：

1. **并行模式数据完整性** - 新增32个`fg_*`列，记录前景任务详细数据
2. **性能指标补充** - 新增2个`perf_*`列（perf_accuracy, perf_eval_loss）
3. **数据规范优化** - 删除`is_parallel`列，使用`mode`列替代

### 对因果分析的影响：

| 维度 | 影响 | 应对策略 |
|------|------|----------|
| **样本量** | ✅ +50行（+7.4%） | 直接受益 |
| **变量数** | ⚠️ +33列 | 分层分析，严格筛选 |
| **数据质量** | ✅ 完整性提升 | 使用raw_data.csv |
| **研究能力** | ✅ 支持并行模式分析 | 新研究方向 |

### 推荐方案：

**采用分层分析策略**（策略1），分别研究：
1. **非并行组**: 超参数 → 能耗/性能（基础因果关系）
2. **并行组**: 后台任务 + 前景超参数 → 前景能耗/性能（并行模式因果关系）
3. **对比分析**: 并行 vs 非并行的因果效应差异

---

**报告生成**: 2025-12-23 20:20
**文件位置**: `/home/green/energy_dl/nightly/analysis/docs/reports/COLUMN_INCREASE_EXPLANATION.md`
