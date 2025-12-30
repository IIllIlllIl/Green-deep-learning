# 问题1：超参数对能耗影响的回归分析方案

**创建日期**: 2025-12-30
**文档版本**: v1.0
**状态**: ✅ 方案确认

---

## 📋 执行摘要

本文档记录了针对研究问题1（超参数对能耗的影响）的回归分析方案设计。在DiBS因果图学习失败后，我们转向使用回归分析方法来回答核心研究问题。

### 核心研究问题

**问题1**: 超参数对能耗的影响（方向和大小）🔬

- **研究目标**:
  - 识别哪些超参数显著影响GPU/CPU能耗
  - 量化每个超参数变化1单位时，能耗变化多少焦耳
  - 区分不同任务类型的超参数效应差异

- **分析方法**:
  - 任务组分层回归（方案A'优化版）
  - 随机森林特征重要性
  - 因果森林（heterogeneous treatment effects）

---

## 🎯 方案设计

### 1. 能耗指标选择 ⭐⭐⭐

**选定的4个能耗指标**:

| 能耗指标 | 列名 | 单位 | 含义 |
|---------|------|------|------|
| **CPU Package能耗** | `energy_cpu_pkg_joules` | 焦耳 (J) | CPU核心计算能耗 |
| **CPU RAM能耗** | `energy_cpu_ram_joules` | 焦耳 (J) | 内存能耗 |
| **GPU总能耗** | `energy_gpu_total_joules` | 焦耳 (J) | GPU总能耗 |
| **系统总能耗** | `energy_cpu_total_joules + energy_gpu_total_joules` | 焦耳 (J) | 整个训练过程的总能耗 |

**选择理由**:

1. **CPU Package能耗** (`energy_cpu_pkg_joules`):
   - 反映CPU核心的计算能耗
   - 与CPU密集型操作（如数据加载、预处理）相关
   - 可用于分析"CPU vs GPU计算分配"的影响

2. **CPU RAM能耗** (`energy_cpu_ram_joules`):
   - 反映内存访问和数据传输的能耗
   - 与batch_size、数据集大小相关
   - 可用于分析内存访问模式的能耗影响

3. **GPU总能耗** (`energy_gpu_total_joules`):
   - 反映GPU计算的总能耗
   - **主要关注指标**（深度学习训练的主要能耗来源）
   - 与learning_rate、batch_size、模型复杂度直接相关

4. **系统总能耗** (派生指标):
   - 反映整个训练过程的总能耗
   - 用于评估总体能耗优化效果
   - 计算方式: `total_energy = energy_cpu_total_joules + energy_gpu_total_joules`

**排除的能耗指标**:

| 指标 | 列名 | 排除理由 |
|------|------|---------|
| GPU平均功率 | `energy_gpu_avg_watts` | 功率指标，不是能量，且与total_joules高度相关 |
| GPU功率波动 | `gpu_power_fluctuation` | 派生指标，已在中介效应分析中考虑 |
| CPU总能耗 | `energy_cpu_total_joules` | 单独使用意义不大，已包含在系统总能耗中 |

---

### 2. 超参数选择 ⭐⭐⭐

**使用统一后的超参数（合并的超参数）**

#### 2.1 统一超参数（2个）✅ **新增**

| 统一超参数 | 来源列 | 单位 | 含义 |
|-----------|-------|------|------|
| **training_duration** | `hyperparam_epochs` 或 `hyperparam_max_iter` | 轮数/迭代次数 | 训练时长（统一不同框架的命名） |
| **l2_regularization** | `hyperparam_weight_decay` 或 `hyperparam_alpha` | 系数 | L2正则化强度（统一PyTorch和scikit-learn） |

**统一逻辑**:

```python
# 训练时长统一
df['training_duration'] = df['hyperparam_epochs'].fillna(df['hyperparam_max_iter'])

# L2正则化统一
df['l2_regularization'] = df['hyperparam_weight_decay'].fillna(df['hyperparam_alpha'])
```

**统一理由**:
- `epochs` 和 `max_iter` 语义相同：控制训练迭代次数
- `weight_decay` 和 `alpha` 语义相同：都是L2正则化系数（源码已验证）
- 统一后可以跨任务进行超参数效应对比

#### 2.2 常规超参数（4个）

| 超参数 | 列名 | 单位 | 填充率（预期） |
|-------|------|------|---------------|
| **学习率** | `hyperparam_learning_rate` | - | ~95% |
| **批量大小** | `hyperparam_batch_size` | 样本数 | ~95% |
| **Dropout** | `hyperparam_dropout` | 比例 (0-1) | ~30% （仅部分模型） |
| **随机种子** | `hyperparam_seed` | 整数 | ~95% |

**注意事项**:
- `dropout` 填充率低，仅在Person_reID和MRT-OAST任务组中使用
- 其他任务组（如examples、resnet）不使用dropout，将被排除

#### 2.3 超参数总结

**每个任务组的超参数集合**（根据任务特点动态选择）:

| 任务组 | 超参数列表 | 数量 |
|-------|-----------|------|
| **组1a: examples** | `training_duration`, `learning_rate`, `batch_size`, `seed` | 4个 |
| **组1b: pytorch_resnet** | `training_duration`, `learning_rate`, `l2_regularization`, `seed` | 4个 |
| **组2: Person_reID** | `training_duration`, `learning_rate`, `dropout`, `seed` | 4个 |
| **组3: VulBERTa** | `training_duration`, `learning_rate`, `l2_regularization`, `seed` | 4个 |
| **组4: bug_localization** | `training_duration`, `l2_regularization`, `kfold`, `seed` | 4个 |
| **组5: MRT-OAST** | `training_duration`, `learning_rate`, `dropout`, `l2_regularization`, `seed` | 5个 |

**说明**:
- 每个任务组使用其最相关的4-5个超参数
- 避免使用填充率<10%的超参数（降低噪声）

---

### 3. 分组策略：方案A'（6组任务分层分析）⭐⭐⭐

**核心思路**: 按任务类型分组，每组使用其特定的超参数集合，最大化数据保留率

#### 3.1 任务组定义

| 任务组 | 包含模型 | 超参数 (4-5个) | 样本量（回溯后） |
|-------|---------|---------------|----------------|
| **组1a: examples** | mnist, mnist_ff, mnist_rnn, siamese | `batch_size`, `epochs`, `learning_rate`, `seed` | 139 |
| **组1b: pytorch_resnet** | resnet20 | `epochs`, `learning_rate`, `weight_decay`, `seed` | 59 |
| **组2: Person_reID** | densenet121, hrnet18, pcb | `dropout`, `epochs`, `learning_rate`, `seed` | 116 |
| **组3: VulBERTa** | mlp | `epochs`, `learning_rate`, `weight_decay`, `seed` | 118 |
| **组4: bug_localization** | default | `alpha`, `kfold`, `max_iter`, `seed` | 127 |
| **组5: MRT-OAST** | default | `dropout`, `epochs`, `learning_rate`, `weight_decay` | 74 |
| **总计** | 11个模型 | - | **633行 (87.1%)** ✅ |

#### 3.2 分组优势

1. **样本量最大化**: 633行（vs 旧方案418行，+51.4%）
2. **超参数同质性**: 每组内超参数完全一致，避免混淆
3. **任务特定分析**: 可以发现任务特定的超参数效应模式
4. **数据保留率高**: 87.1%（vs 方案B的51.2%）

#### 3.3 与DiBS 6分组（594行）的区别

| 维度 | 回归分析方案A' (633行) | DiBS 6分组 (594行) | 差异 |
|------|----------------------|-------------------|------|
| **样本量** | 633行 | 594行 | **+39行 (+6.6%)** |
| **数据保留率** | 87.1% | 81.8% | +5.3% |
| **筛选条件** | 能耗完整 + 超参数完整 | 能耗完整 + **性能完整** + 超参数完整 |
| **是否需要性能数据** | ❌ 不需要 | ✅ 必须有 |
| **分析方法** | 回归分析 | DiBS因果图学习 |

**关键差异**: 回归分析只研究能耗，不需要性能数据，因此可以保留更多样本（+39行）。

---

### 4. 控制变量 ⭐

除了超参数（自变量）和能耗（因变量），还需要控制以下变量：

| 控制变量 | 列名 | 类型 | 含义 |
|---------|------|------|------|
| **并行模式** | `is_parallel` | 二值 (0/1) | 是否为并行训练 |
| **训练时长** | `duration_seconds` | 连续 | 训练耗时（秒） |
| **模型类型** | `model` | 类别 | 模型名称（仅在多模型任务组） |

**控制逻辑**:
- 在回归模型中加入控制变量，消除混淆偏差
- 例如: `energy ~ learning_rate + batch_size + is_parallel + duration_seconds`

---

## 🔬 回归分析方法

### 方法1: 多元线性回归（基线方法）

**模型形式**:
```
energy_gpu_total_joules = β0 + β1*training_duration + β2*learning_rate
                         + β3*batch_size + β4*seed
                         + β5*is_parallel + β6*duration_seconds + ε
```

**优势**:
- ✅ 解释性强：系数β直接表示超参数对能耗的线性影响
- ✅ 统计显著性：可以进行假设检验（t检验、p值）
- ✅ 可视化简单：系数森林图（forest plot）

**劣势**:
- ⚠️ 假设线性关系（可能不成立）
- ⚠️ 无法捕捉交互效应（如learning_rate × batch_size）

**适用场景**: 快速探索性分析，识别主要影响因素

---

### 方法2: 随机森林回归（非线性方法）

**模型形式**:
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf.fit(X, y)  # X: 超参数, y: 能耗
```

**优势**:
- ✅ 捕捉非线性关系
- ✅ 自动处理交互效应
- ✅ 特征重要性排序

**劣势**:
- ⚠️ 黑盒模型，解释性差
- ⚠️ 无法给出超参数的边际效应（只有重要性排序）

**适用场景**: 验证线性回归的假设，识别最重要的超参数

---

### 方法3: 因果森林（Causal Forest）⭐ **推荐**

**模型形式**:
```python
from econml.dml import CausalForestDML

cf = CausalForestDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestRegressor(),
    random_state=42
)
cf.fit(Y=energy, T=hyperparam, X=controls)
ate = cf.ate(X_test)  # 平均因果效应
```

**优势**:
- ✅ 估计异质性因果效应（不同样本的效应不同）
- ✅ 控制混淆变量（类似DML）
- ✅ 提供置信区间

**劣势**:
- ⚠️ 计算复杂度高
- ⚠️ 需要大样本量（每个任务组>100为佳）

**适用场景**: 深入因果推断，回答"超参数对不同任务的效应是否不同？"

---

## 📊 分析流程

### 阶段1: 数据准备 ⏳ **当前任务**

**步骤**:
1. **加载新数据**: `energy_data_original.csv` (726行, 56列)
2. **筛选能耗完整**: 保留692行（删除34行能耗缺失）
3. **默认值回溯**: 从默认值实验和models_config.json回溯超参数
   - 预期: 超参数填充率 47% → 95%
4. **超参数统一**: 创建`training_duration`和`l2_regularization`
5. **任务分组**: 按方案A'分为6组
6. **数据验证**: 检查回溯后的数据质量
7. **保存处理后数据**: 633行，6个任务组CSV

**输出文件**:
```
analysis/data/energy_research/processed/
├── group1a_examples.csv (139行)
├── group1b_resnet.csv (59行)
├── group2_person_reid.csv (116行)
├── group3_vulberta.csv (118行)
├── group4_bug_localization.csv (127行)
└── group5_mrt_oast.csv (74行)
```

**预估时间**: 1-2小时（包含脚本编写和数据验证）

---

### 阶段2: 回归分析 ⏳ **后续任务**

**步骤**:
1. **分组回归**: 对6个任务组分别运行线性回归
2. **系数森林图**: 可视化超参数系数及置信区间
3. **显著性检验**: 识别统计显著的超参数（p < 0.05）
4. **随机森林验证**: 使用RF验证线性假设，计算特征重要性
5. **交互效应检验**: 检查是否存在显著交互（如lr × batch_size）

**输出文件**:
```
analysis/results/energy_research/question1/
├── regression_coefficients_by_group.csv
├── significance_tests.csv
├── forest_plot_gpu_energy.png
└── feature_importance_rf.png
```

**预估时间**: 2-3小时（包含脚本编写、运行、可视化）

---

### 阶段3: 因果森林分析 ⏳ **可选任务**

**步骤**:
1. **运行因果森林**: 对每个任务组估计异质性因果效应
2. **效应分布**: 可视化不同样本的因果效应分布
3. **对比分析**: 对比6组任务的因果效应差异

**输出文件**:
```
analysis/results/energy_research/question1/
├── causal_forest_ate_by_group.csv
└── heterogeneous_effects_distribution.png
```

**预估时间**: 3-4小时

---

### 阶段4: 结果汇总与报告 ⏳ **最终任务**

**步骤**:
1. **跨任务趋势**: 总结6组任务的共性发现
2. **任务特定发现**: 强调任务特定的超参数效应
3. **可操作建议**: 给出能耗优化建议

**输出文件**:
```
analysis/docs/reports/QUESTION1_REGRESSION_ANALYSIS_REPORT.md
```

**预估时间**: 2-3小时

---

## 🎯 预期成果

### 1. 全局趋势（横跨所有任务）

**示例发现**:
- `learning_rate` 提高1个标准差 → GPU能耗增加 **X%** (p < 0.001) ✅ **显著**
- `batch_size` 提高1个标准差 → GPU能耗增加 **Y%** (p < 0.05) ✅ **显著**
- `training_duration` 提高1个标准差 → GPU能耗增加 **Z%** (p < 0.001) ✅ **显著**
- `is_parallel` (并行模式) → GPU能耗增加 **W%** (p < 0.01) ✅ **显著**

### 2. 任务特定发现

**示例**:
- **examples组**: `batch_size` 对能耗影响最大（β = X）
- **resnet组**: `learning_rate` 对能耗影响最大（β = Y）
- **Person_reID组**: `dropout` 显著降低能耗（β = -Z）

### 3. 可操作性建议

**能耗优化策略**:
1. **降低学习率**: 在不显著影响性能的前提下，降低learning_rate可减少能耗
2. **调整batch_size**: 根据任务类型选择合适的batch_size（权衡能耗和性能）
3. **并行模式权衡**: 并行训练虽提高速度，但增加能耗X%

---

## 📝 关键决策记录

### 决策1: 能耗指标选择 ✅

**决定**: 使用4个能耗指标
- `energy_cpu_pkg_joules` - CPU Package能耗
- `energy_cpu_ram_joules` - CPU RAM能耗
- `energy_gpu_total_joules` - GPU总能耗 ⭐ **主要指标**
- `total_energy` - 系统总能耗（派生）

**理由**: 覆盖CPU、GPU、内存的完整能耗画像，GPU总能耗是深度学习训练的主要关注点

**决策时间**: 2025-12-30
**决策者**: Green + Claude

---

### 决策2: 超参数选择 ✅

**决定**: 使用统一后的超参数（合并的超参数）
- `training_duration` ← 统一 `epochs` + `max_iter`
- `l2_regularization` ← 统一 `weight_decay` + `alpha`
- 其他常规超参数: `learning_rate`, `batch_size`, `dropout`, `seed`

**理由**:
- 统一超参数可以跨任务比较效应
- 消除框架差异（PyTorch vs scikit-learn）
- 提高数据利用率

**决策时间**: 2025-12-30
**决策者**: Green + Claude

---

### 决策3: 分组策略 ✅

**决定**: 采用方案A'（6组任务分层分析）

**理由**:
- 数据保留率最高（87.1%，633行）
- 每组超参数同质，避免混淆
- 可以发现任务特定的超参数效应
- 比DiBS方案多保留39行样本（不需要性能数据）

**决策时间**: 2025-12-29（方案设计），2025-12-30（能耗指标和超参数确认）
**决策者**: Green + Claude

---

### 决策4: 分析方法 ✅

**决定**: 使用三种回归方法
1. **多元线性回归**（基线，快速识别主效应）
2. **随机森林回归**（验证非线性，特征重要性）
3. **因果森林**（可选，深入因果推断）

**理由**:
- 线性回归提供解释性
- 随机森林验证假设
- 因果森林提供异质性因果效应

**决策时间**: 2025-12-30
**决策者**: Green + Claude

---

## 🚀 下一步行动

### 立即任务（优先级：高）⭐⭐⭐

1. **实现默认值回溯脚本**
   - 文件: `analysis/scripts/backfill_hyperparameters.py`
   - 功能: 从默认值实验和models_config.json提取参数，填充单参数变异实验
   - 预期: 超参数填充率 47% → 95%
   - 预估时间: 1-2小时

2. **数据质量验证**
   - 文件: `analysis/scripts/validate_backfilled_data.py`
   - 功能: 检查回溯值的合理性，生成数据质量报告
   - 预估时间: 30分钟

3. **生成6组任务数据**
   - 文件: `analysis/scripts/generate_regression_groups.py`
   - 功能: 按方案A'分组，保存6个CSV文件
   - 预估时间: 30分钟

### 后续任务（优先级：中）

4. **运行分组回归分析**
   - 文件: `analysis/scripts/run_group_regression.py`
   - 功能: 对6组分别运行线性回归，生成系数和显著性检验
   - 预估时间: 1-2小时

5. **生成可视化**
   - 文件: `analysis/scripts/visualize_regression_results.py`
   - 功能: 系数森林图、特征重要性图
   - 预估时间: 1小时

6. **生成分析报告**
   - 文件: `analysis/docs/reports/QUESTION1_REGRESSION_ANALYSIS_REPORT.md`
   - 功能: 汇总发现，提出建议
   - 预估时间: 2-3小时

---

## 📚 参考文档

- [DATA_COMPARISON_OLD_VS_NEW_20251229.md](reports/DATA_COMPARISON_OLD_VS_NEW_20251229.md) - 新旧数据集对比
- [VARIABLE_EXPANSION_PLAN.md](reports/VARIABLE_EXPANSION_PLAN.md) - v3.0变量扩展方案（超参数统一的来源）
- [ENERGY_DATA_PROCESSING_PROPOSAL.md](ENERGY_DATA_PROCESSING_PROPOSAL.md) - DiBS方案（已失败）
- [DIBS_FINAL_FAILURE_REPORT_20251226.md](reports/DIBS_FINAL_FAILURE_REPORT_20251226.md) - DiBS失败原因分析

---

## 📌 版本历史

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2025-12-30 | 初始版本：确认能耗指标（4个）、超参数（统一后的合并超参数）、分组策略（方案A' 6组）、分析方法（回归分析） | Green + Claude |

---

**文档状态**: ✅ 方案确认
**维护者**: Green + Claude
**下次更新**: 完成默认值回溯后更新数据统计
