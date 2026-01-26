# 标准化问题与DiBS实现差距总结

**创建日期**: 2026-01-26
**最后更新**: 2026-01-26 (添加CTF缺失值处理策略分析)
**创建背景**: 基于与Claude的深度讨论分析
**文档状态**: 已更新
**相关文档**:
- [INTERACTION_TERMS_TRANSFORMATION_PLAN.md](../current_plans/INTERACTION_TERMS_TRANSFORMATION_PLAN.md)
- [DIBS_PARAMETER_TUNING_ANALYSIS.md](DIBS_PARAMETER_TUNING_ANALYSIS.md)
- [STAGE2_CODE_COMPARISON_20260125.md](../current_plans/STAGE2_CODE_COMPARISON_20260125.md)

---

## 📋 执行摘要

### 核心发现问题
1. **标准化方法错误**: 采用组内独立标准化，破坏跨组可比性
2. **DiBS实现与参考论文存在差距**: 算法变体、参数设置、先验知识不同
3. **数据填充策略需优化**: 中位数填充可能引入偏差

### 关键影响
- ❌ **跨组比较失效**: ATE表示"1个组内标准差变化"，无法回答"哪个模型的超参数效应最强"
- ⚠️ **与CTF结果不可直接对比**: 算法差异导致因果图学习结果不同
- 🔴 **交互项解释混乱**: 因标准化问题放大尺度差异

### 推荐解决方案
1. **立即修复**: 改为全局标准化，恢复跨组可比性
2. **中期改进**: 对齐CTF实现（MarginalDiBS + BGe模型 + 先验知识）
3. **长期优化**: 智能数据填充策略

---

## 🔍 核心问题：标准化方法的选择

### 当前实现问题
**方法**: 组内独立标准化（每组使用不同的均值/标准差）
**位置**: `analysis/scripts/generate_interaction_terms_data.py:120-174`

```python
# 分组标准化实现
scaler_energy = StandardScaler()
df_transformed[energy_cols] = scaler_energy.fit_transform(df_transformed[energy_cols])
# 每组独立执行标准化
```

### 导致的严重后果
1. **ATE尺度不可比**:
   - group1: "1个单位" = 72.97 watts (energy_gpu_avg_watts标准差)
   - group2: "1个单位" = 8.10 watts (9.0倍差异)
   - 无法跨组比较效应大小

2. **交互项尺度不一致**:
   - 交互项标准差 = σ_hyperparam × √[p(1-p)]
   - σ_hyperparam每组不同 → 交互项尺度不同
   - group2 batch_size标准差=0.0 → 交互项方差=0（严重bug！）

3. **研究问题受阻**:
   - ❌ 无法回答：哪个模型的超参数对能耗影响最大？
   - ❌ 无法判断：并行模式的调节效应是否跨模型一致？

### CTF原始方法对比
**方法**: 全局标准化（所有数据统一标准化）
**位置**: `CTF_original/src/discovery.py:136-137`

```python
# CTF全局标准化
scaler = StandardScaler()
collected_data = scaler.fit_transform(collected_df)  # 全局标准化！
```

---

## ⚙️ DiBS实现深度检查

### 1. DiBS输入参数及值

**当前最优配置** (`run_dibs_6groups_final.py:31-39`):

| 参数 | 当前值 | 默认值 | 作用 | 参考论文值 |
|------|--------|--------|------|-----------|
| `alpha_linear` | 0.05 | 0.9 | DAG惩罚参数，控制稀疏性 | 0.05 (DiBS默认) |
| `beta_linear` | 0.1 | 1.0 | 无环约束强度，**关键参数** | 1.0 |
| `n_particles` | 20 | 20 | SVGD粒子数，影响探索能力 | 50 |
| `n_steps` | 5000 | 10000 | 迭代步数 | 13000 |
| `tau` | 1.0 | 1.0 | Gumbel-softmax温度 | 1.0 |
| `n_grad_mc_samples` | 128 | 128 | 梯度MC样本数 | - |
| `n_acyclicity_mc_samples` | 32 | 32 | 无环性MC样本数 | - |

**参数调优来源**: 基于`DIBS_PARAMETER_TUNING_ANALYSIS.md`的系统测试，发现`alpha=0.05, beta=0.1`最优。

### 2. DiBS先验知识输入

**当前实现** (`causal_discovery.py:122`):
- ✅ **图结构先验**: `ErdosReniDAGDistribution(n_edges_per_node=2)`
- ❌ **边概率先验**: 未实现`edge_prior_prob`参数
- ❌ **外部因果图**: 无外部先验图文件
- ❌ **干预掩码**: `interv_mask=None` (观测数据)

**CTF原始代码** (`CTF_original/src/discovery.py`):
- ✅ **干预掩码**: 对`METHOD_COLUMNS`特殊处理
- ✅ **外部因果图**: `causal-graph/`目录提供先验结构
- ✅ **可配置先验**: `ScaleFreeDAGDistribution`选项

**关键差异**: 当前实现依赖数据驱动学习，CTF注入更强的领域先验知识。

---

## ⚖️ DiBS实现与参考论文代码差距对比

| 维度 | CTF原始代码 | 当前实现 | 差距分析 | 影响程度 |
|------|-------------|----------|----------|----------|
| **DiBS类型** | `MarginalDiBS` | `JointDiBS` | 边缘vs联合后验估计 | ⭐⭐⭐ |
| **似然模型** | `BGe` (Bayesian Gaussian) | `LinearGaussian` | BGe适合混合类型数据 | ⭐⭐⭐⭐ |
| **标准化** | 全局`StandardScaler()` | 分组标准化 | **核心可比性问题** | ⭐⭐⭐⭐⭐ |
| **参数设置** | n_particles=50, steps=13000 | n_particles=20, steps=5000 | CTF使用更多资源 | ⭐⭐ |
| **先验知识** | 干预掩码+外部因果图 | 仅Erdős-Rényi先验 | CTF有更强先验注入 | ⭐⭐⭐⭐ |
| **数据清洗** | `dropna()`删除缺失值 | 均值/中位数填充 | 策略不同，影响样本量 | ⭐⭐ |
| **干预处理** | `interv_mask`显式编码 | 无干预掩码 | CTF针对公平性方法优化 | ⭐⭐⭐ |

### 算法层面关键差异
1. **DiBS类型不同**: CTF使用`MarginalDiBS`（边缘后验），当前使用`JointDiBS`（联合后验）
2. **似然模型不同**: CTF使用`BGe`适合混合类型数据，当前使用`LinearGaussian`假设线性关系
3. **收敛标准不同**: CTF使用更多粒子和迭代步数，可能更精确但更耗时

---

## 📈 导致实验结果差异的因素分析

### A. 标准化问题（P0优先级）
**影响**: 破坏所有跨组比较结论
**表现**:
- ATE表示"处理变量变化1个组内标准差"
- 不同组的"1个单位"代表不同原始变化量
- 交互项尺度因组而异

### B. 算法差异（P1优先级）
**影响**: 因果图学习结果不可直接比较
**表现**:
- 不同的后验估计方法（边缘vs联合）
- 不同的数据分布假设（BGe vs LinearGaussian）
- 不同的收敛特性

### C. 先验知识差异（P2优先级）
**影响**: 图结构搜索方向和效率不同
**表现**:
- CTF: 领域知识引导，更快找到有意义结构
- 当前: 纯数据驱动，可能发现噪声边

### D. 数据预处理差异（P3优先级）
**影响**: 数据质量和完整性不同
**表现**:
- 缺失值处理: 删除vs填充
- 异常值处理: 方法不同
- 变量选择: CTF过滤特定列

---

## 🧪 数据填充策略检查

### ATE计算阶段 (`compute_ate_whitelist.py:112-155`)
1. **数值列**: 中位数填充（鲁棒性强，适合能耗/超参数数据）
2. **布尔列**: `False`填充（模型标识等二值变量）
3. **分类列**: 众数填充 → 第一个非NaN值 → 删除全NaN列

### DiBS阶段 (`causal_discovery.py`)
- 无显式缺失值处理
- 依赖输入数据已预处理
- 假设零缺失值（使用均值填充）

### CTF原始策略 (`CTF_original/src/load_data.py`)
- `dropna()`直接删除缺失值
- `replace([np.inf, -np.inf], np.nan)`处理无穷值

### 策略对比
| 策略 | 当前实现 | CTF原始 | 优劣分析 |
|------|----------|---------|----------|
| **数据保留** | 保守填充，最大化利用 | 严格删除，保证质量 | 当前可能引入偏差，CTF可能丢失信息 |
| **适用场景** | 样本量有限时 | 数据充足时 | 需根据缺失机制选择 |
| **对ATE影响** | 可能引入填充偏差 | 减少偏差但方差增大 | 需权衡偏差-方差 |

---

## 🔍 CTF缺失值处理策略深度分析

### 📊 1. CTF原始代码的缺失值处理逻辑

**关键发现**：CTF根据**分析阶段**采用不同的策略：

| 分析阶段 | 代码文件 | 缺失值处理策略 | 代码位置 | 标准化时机 |
|----------|----------|---------------|----------|-----------|
| **DiBS因果发现** | `discovery.py` | `dropna()` 删除所有缺失行 | L118-119 | **删除后**进行全局标准化 |
| **ATE计算（推理）** | `inf.py` | `fillna(0)` 用0填充缺失值 | L115-116 | 填充后进行标准化（可选） |

#### 详细代码分析

**DiBS因果发现阶段** (`CTF_original/src/discovery.py:118-137`)：
```python
# 1. 替换无穷值为NaN
collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2. 删除所有包含NaN的行 ⭐ 关键策略
collected_df.dropna(inplace=True)

# 3. 数据打乱
collected_df = collected_df.sample(frac=1).reset_index(drop=True)

# 4. 全局标准化（在删除缺失值后执行）
scaler = StandardScaler()
collected_data = scaler.fit_transform(collected_df)  # 全局标准化！
```

**ATE计算阶段** (`CTF_original/src/inf.py:114-116`)：
```python
# 1. 替换无穷值为NaN
collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2. 注释掉了dropna()，改用fillna(0)
# collected_df.dropna(inplace=True)
collected_df.fillna(0, inplace=True)  # ⭐ 关键策略：用0填充

# 3. 数据打乱
collected_df = collected_df.sample(frac=1).reset_index(drop=True)
```

### 🎯 2. CTF的全局标准化流程

**标准化执行时机**：**在缺失值处理后**进行

```
原始数据 → 替换无穷值→NaN → DiBS阶段：dropna()删除 / ATE阶段：fillna(0)填充 → 数据打乱 → 全局StandardScaler标准化 → 标准化后数据
```

**关键特征**：
1. **先处理缺失值，后标准化**：确保标准化时不包含NaN
2. **全局而非分组**：所有数据使用相同的均值/标准差
3. **阶段差异化**：DiBS阶段严格删除，ATE阶段保守填充

### 📈 3. 全局标准化面临的数据缺失问题

#### 问题本质
如果直接合并6组数据进行全局标准化：
1. **合并后NaN位置**：某组在某列有缺失值 → 合并后对应位置为NaN
2. **标准化失败**：`StandardScaler.fit_transform()` 无法处理NaN
3. **样本量减少**：如果使用`dropna()`，可能大幅减少样本

#### CTF的解决方案评估
**优点**：
- ✅ **简单直接**：`dropna()`保证数据质量
- ✅ **标准化稳定**：无NaN干扰标准化参数
- ✅ **与算法兼容**：DiBS要求零缺失值

**缺点（对我们的数据）**：
- ⚠️ **样本量风险**：能耗数据缺失较多（~40%），删除可能导致样本不足
- ⚠️ **信息损失**：删除整行可能丢失其他可用信息
- ⚠️ **偏差风险**：缺失可能非随机，删除引入选择偏差

---

## 🎯 问题总结与修复建议

### 核心问题优先级排序

#### P0: 标准化可比性问题 ⭐⭐⭐⭐⭐
**问题**: 组内独立标准化破坏跨组ATE可比性
**影响**: 无法回答"哪个模型的超参数效应最强"
**解决**: 全局标准化（合并6组后统一标准化）

#### P1: DiBS与CTF算法差异 ⭐⭐⭐⭐
**问题**: 算法变体、似然模型、先验知识不同
**影响**: 因果图学习结果不可直接比较
**解决**: 考虑采用`MarginalDiBS + BGe`组合

#### P2: 先验知识缺失 ⭐⭐⭐
**问题**: 缺乏领域知识注入（如超参数→能耗方向）
**影响**: 学习效率低，可能发现错误因果方向
**解决**: 实现`edge_prior_prob`参数

#### P3: 数据填充策略 ⭐⭐
**问题**: 中位数填充可能引入偏差
**影响**: 特别是对超参数（batch_size等）
**解决**: 基于模型类型的智能填充

### 推荐修复路线图

#### 阶段1: 标准化修复与缺失值处理 (2-3天)
```python
# 1. 诊断各组缺失情况
python scripts/diagnose_missing_patterns.py

# 2. 实现全局标准化数据生成（考虑缺失值处理）
python scripts/create_global_standardized_data.py --strategy conservative

# 3. 修复group2 batch_size标准差为0的问题
python scripts/fix_group2_batchsize_issue.py

# 4. 重新计算全局标准化ATE
python scripts/compute_ate_whitelist_global.py

# 5. 敏感性分析：对比不同缺失值处理策略
python scripts/sensitivity_analysis_missing_strategies.py
```

**关键实施细节**：
1. **缺失值诊断**：分析各组的缺失比例、模式和机制
2. **策略选择**：基于全局均值的保守填充（主分析）vs CTF风格删除（敏感性分析）
3. **标准化时机**：先处理缺失值，后执行全局标准化
4. **交互项重建**：基于全局标准化后的超参数重建交互项

#### 阶段2: 算法对齐 (3-4天)
1. 测试`MarginalDiBS + BGe`组合
2. 实现先验知识注入机制
3. 对比当前与CTF风格的结果差异
4. 统一缺失值处理策略：DiBS阶段严格删除，ATE阶段保守填充（与CTF一致）

#### 阶段3: 数据质量提升 (1-2天)
1. 改进缺失值填充策略：基于缺失机制的智能填充
2. 增加数据质量验证检查
3. 实施变量重要性选择
4. 建立数据预处理pipeline：缺失值处理 → 全局标准化 → 交互项创建

### 关键决策点

| 决策 | 选项A | 选项B | 推荐 | 理由 |
|------|-------|-------|------|------|
| **标准化** | 保持分组标准化 | 改为全局标准化 | **B** | 支持跨组比较，向参考论文靠拢 |
| **DiBS类型** | 保持`JointDiBS` | 改为`MarginalDiBS` | B | CTF使用MarginalDiBS，更成熟 |
| **似然模型** | 保持`LinearGaussian` | 改为`BGe` | B | BGe适合混合类型数据，更通用 |
| **先验知识** | 保持隐式分类 | 实现显式干预掩码 | B | 提升学习效率，注入领域知识 |
| **缺失值处理** | 统一中位数填充 | 阶段差异化（DiBS删除，ATE填充） | B | 与CTF一致，DiBS要求零缺失值 |

---

## 📊 现状评估与下一步行动

### ✅ 已取得的成果
1. ATE计算框架稳定: 84.1%边成功计算，100%统计显著
2. 参数调优完成: 找到最优DiBS配置
3. 交互项问题修复: 8条交互项边全部可计算

### ⚠️ 待解决的关键问题
1. **标准化可比性**: 阻碍跨模型比较结论
2. **先验知识缺失**: 因果图搜索无领域约束
3. **数据异常**: group2 batch_size标准差为0
4. **缺失值处理策略**: 需要与CTF对齐（DiBS阶段删除，ATE阶段填充）

### 🎯 立即行动建议
1. **首先修复标准化问题**: 这是影响所有分析结论的核心问题
2. **诊断缺失值模式**: 分析能耗数据的缺失机制和影响
3. **实施保守填充的全局标准化**: 基于全局均值的填充，保证可比性
4. **进行敏感性分析**: 对比不同缺失值处理策略的结果
5. **保持ATE计算逻辑不变**: 仅修改数据预处理层
6. **优先实现跨组可比性**: 支持"哪些超参数效应是跨模型稳健的"研究问题

### 📝 最终目标
建立与参考论文思路一致、可跨模型比较的因果推断框架，支持可靠的超参数能耗效应分析。

---

## 📞 相关资源

### 代码文件
1. `analysis/scripts/generate_interaction_terms_data.py` - 标准化实现
2. `analysis/utils/causal_discovery.py` - DiBS封装类
3. `analysis/scripts/compute_ate_whitelist.py` - ATE计算和数据填充
4. `CTF_original/src/discovery.py` - CTF原始DiBS实现

### 参考文档
1. [DIBS_PARAMETER_TUNING_ANALYSIS.md](DIBS_PARAMETER_TUNING_ANALYSIS.md) - 参数调优分析
2. [STAGE2_CODE_COMPARISON_20260125.md](../current_plans/STAGE2_CODE_COMPARISON_20260125.md) - 代码差异对比
3. [INTERACTION_TERMS_TRANSFORMATION_PLAN.md](../current_plans/INTERACTION_TERMS_TRANSFORMATION_PLAN.md) - 交互项转换方案

**维护者**: Green
**最后更新**: 2026-01-26 (添加CTF缺失值处理策略分析)
**文档版本**: 1.1