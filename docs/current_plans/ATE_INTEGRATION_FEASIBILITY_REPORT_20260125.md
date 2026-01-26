# ATE集成方案可行性评估报告

**日期**: 2026-01-25
**评估人**: Claude (Sonnet 4.5)
**版本**: v1.0
**状态**: ✅ 建议实施（有条件）

---

## 📊 执行摘要

### 总体评分: 3.5/5 🟡

**结论**: **建议实施，但需要重大修改**

### 关键发现

1. ✅ **技术路径清晰** - Phase 0-3的技术方案基本合理
2. ⚠️ **领域适配挑战** - CTF的fairness逻辑不能直接应用到能耗研究
3. ❌ **ref_df策略缺失** - 这是最关键的问题，需要创新设计
4. ⚠️ **时间估算偏乐观** - 15-22天可能需要30-40天
5. ✅ **P0风险识别准确** - 但有2个新风险需要添加

### 核心建议

1. **必须重新设计ref_df构建策略**（P0+风险）
2. **简化原因寻找功能**（从P1降为P2）
3. **采用渐进式实施**（最小可行方案优先）
4. **考虑混合方案**（部分CTF逻辑+部分现有逻辑）

---

## 🔍 详细评估

## 1. 技术可行性 (21/30分)

### 1.1 Phase 0-3技术路径合理性 (6/10分)

**✅ 优点**:
- Phase顺序合理：理解→重构→扩展→验证
- 代码重构范围可控（2个核心文件）
- CTF源码已获取并分析

**❌ 缺点**:
- Phase 0与Phase 1有重叠（理解CTF应该在重构之前）
- 缺少"原型验证"阶段
- 测试验证阶段（Phase 3）过于简单

**建议**: 在Phase 0和Phase 1之间插入"Phase 0.5: 原型验证"

### 1.2 代码重构复杂度 (7/10分)

**需要修改的代码**:
```
causal_inference.py: ~400行
├── estimate_ate()           # 需要重构签名和逻辑
├── _identify_confounders()  # 需要替换为predecessors()
└── analyze_all_edges()      # 需要适配新的接口

tradeoff_detection.py: ~400行
├── detect_tradeoffs()       # 基本可用
└── 需要新增: find_causes() # 全新功能，~150行

新增文件:
└── add_ate_to_whitelist.py  # 新工具，~200行
```

**复杂度评估**:
- 中等复杂度：需要理解因果图、DML、网络分析
- 风险点：混淆因素识别逻辑可能引入bug
- 测试挑战：需要真实的因果图数据

**建议**: 先在group1上验证，再扩展到其他5组

### 1.3 技术陷阱和盲点 (4/10分)

**❌ 已识别的陷阱**:

1. **NetworkX图操作陷阱**
   ```python
   # CTF代码: dg.predecessors(parent)
   # 问题: 如果parent没有predecessors，返回空列表
   # 影响: X_cols可能为空，导致DML失败
   ```
   **缓解**: 添加空列表检查和后备方案

2. **ref_df索引对齐陷阱**
   ```python
   # CTF代码: ate = est.ate(X=ref_df[X_cols], T0=T0, T1=T1)
   # 问题: ref_df的索引可能与data_df不一致
   # 影响: ATE计算可能使用错误的参考值
   ```
   **缓解**: 显式对齐索引，或使用均值而非ref_df

3. **T0/T1数值范围陷阱**
   ```python
   # 问题: T0和T1的差值过小时，ATE不稳定
   # 影响: 置信区间过宽，统计不显著
   ```
   **缓解**: 添加T1-T0最小阈值检查

**⚠️ 潜在盲点**:
- CTF代码可能有领域特定的假设（未在源码中体现）
- 能耗数据的分布可能与fairness数据差异很大
- DiBS因果图可能有噪声边，影响predecessors()结果

### 1.4 时间估算合理性 (4/10分)

**原估算**: 15-22天
**重新评估**: 30-40天（保守估计）

**分解**:
| 阶段 | 原估算 | 修正后 | 理由 |
|------|--------|--------|------|
| Phase 0 | 2-3天 | 3-4天 | 需要理解能耗领域特性 |
| Phase 0.5 | - | 5-7天 | **新增**：原型验证 |
| Phase 1 | 5-7天 | 10-12天 | ref_df策略设计复杂 |
| Phase 2 | 5-7天 | 8-10天 | 原因寻找可简化 |
| Phase 3 | 3-5天 | 4-7天 | 需要全面验证 |

**理由**:
1. ref_df策略需要创新设计（不是简单复制CTF）
2. 原因寻找功能复杂度高，容易出bug
3. 测试验证需要覆盖6组数据

**建议**: 采用迭代开发，每3天一个里程碑

---

## 2. 领域适配性 (18/30分)

### 2.1 CTF逻辑适配到能耗研究 (5/15分)

**❌ 核心差异分析**:

| 维度 | CTF (Fairness) | 能耗研究 | 适配难度 |
|------|----------------|----------|----------|
| **干预变量** | Fairness方法 (categorical) | 超参数 (continuous) | 🔴 高 |
| **Baseline概念** | 明确 (无fairness干预) | **不存在** | 🔴 致命 |
| **数据结构** | 多方法 × Model_Width | 超参数组合 | 🟡 中 |
| **因果图** | 手工定义 | DiBS学习 | 🟢 低 |
| **评估指标** | Accuracy, Fairness | Energy, Performance | 🟢 低 |

**关键问题**:

1. **Baseline缺失**
   - CTF: `normal_df` = Baseline数据（无fairness方法）
   - 能耗: 没有"无超参数"的实验（所有实验都有超参数）

2. **干预变量类型差异**
   - CTF: Fairness方法是binary (0/1或on/off)
   - 能耗: 超参数是continuous (learning_rate: 0.001-0.1)

3. **T0/T1含义差异**
   - CTF: T0=无干预, T1=有干预（清晰的因果对比）
   - 能耗: T0=低值?, T1=高值?（任意选择）

**结论**: ❌ **不能直接适配**，需要重新设计

### 2.2 ref_df构建策略 (3/10分)

**CTF的ref_df**:
```python
ref_df = normal_df.groupby(["Model_Width"]).mean().reset_index()
# normal_df = Baseline数据（无fairness方法）
```

**能耗研究的选项**:

| 选项 | 描述 | 优点 | 缺点 | 可行性 |
|------|------|------|------|--------|
| **A. 非并行模式** | `data[data['is_parallel']==0]` | 有明确对照 | 数据量减少50% | 🟡 中 |
| **B. 最小能耗组** | 能耗最低的10%实验 | 能耗基准 | 样本量小，偏倚 | 🔴 低 |
| **C. 中位数分组** | 每个超参数的中位数 | 平衡 | 无因果含义 | 🔴 低 |
| **D. 虚拟ref_df** | 所有超参数设为0.5分位 | 完全对照 | 不真实 | 🟡 中 |
| **E. 不使用ref_df** | 直接用data_df的均值 | 简单 | 偏离CTF | 🟢 高 |

**推荐**: **选项E + 选项A混合**
- 主要：不使用ref_df（简化版）
- 备选：非并行模式作为ref_df（完整版）

**理由**:
1. 能耗研究没有明确的"Baseline"，ref_df概念不适用
2. CTF使用ref_df是为了标准化不同Model_Width，能耗研究不需要
3. EconML的`ate()`方法可以不传T0/T1（默认使用均值对比）

### 2.3 T0/T1策略设计 (5/10分)

**CTF的T0/T1**:
```python
T0 = ref_df[pc].mean()  # Baseline状态
T1 = data_df[data_df[fair_method] == 1][pc].mean()  # 干预状态
```

**能耗研究的选项**:

| 选项 | T0 | T1 | 优点 | 缺点 | 可行性 |
|------|-----|-----|------|------|--------|
| **A. Min/Max** | min(treatment) | max(treatment) | 覆盖全范围 | 极端值，不稳定 | 🟡 中 |
| **B. 25/75分位** | 25th percentile | 75th percentile | 鲁棒性 | 任意选择 | 🟢 高 |
| **C. Mean±SD** | mean - 1sd | mean + 1sd | 统计意义 | 可能越界 | 🟢 高 |
| **D. 不使用T0/T1** | - | - | 简化 | 偏离CTF | 🟢 高 |

**推荐**: **选项B (25/75分位) 或 选项D (不使用)**

**实施建议**:
```python
# 选项B: 25/75分位
T0 = data_df[treatment].quantile(0.25)
T1 = data_df[treatment].quantile(0.75)

# 选项D: 不使用T0/T1（推荐）
ate = est.ate(X=X)  # EconML会自动计算边际效应
```

### 2.4 领域特定挑战 (5/5分)

**识别到的挑战**:

1. **⚠️ 能耗数据的异质性**
   - 不同模型的能耗差异巨大（MRT-OAST vs MNIST-FF）
   - 解决：必须按模型分组分析（已在6groups_final中实现）

2. **⚠️ 超参数的相关性**
   - batch_size和learning_rate可能相关
   - 解决：使用DML的混淆因素控制

3. **⚠️ 交互项的处理**
   - CTF没有交互项（Method × Model_Width）
   - 能耗有交互项（batch_size × is_parallel）
   - 解决：将交互项视为独立的干预变量

---

## 3. 风险评估 (14/20分)

### 3.1 P0风险评估

**原P0风险**:
1. ✅ ref_df构建不明确 - **已评估，建议不使用ref_df**
2. ✅ T0/T1策略缺失 - **已评估，建议使用25/75分位或不使用**
3. ✅ 混淆因素识别错误 - **已评估，建议使用predecessors()**

**新增P0+风险**:

| 风险 | 影响 | 概率 | 缓解方案 |
|------|------|------|----------|
| **P0+1: ref_df缺失导致ATE不可比** | 致命 | 高 | 使用选项E（不使用ref_df） |
| **P0+2: T0/T1选择导致ATE不稳定** | 高 | 中 | 使用25/75分位，添加阈值检查 |
| **P0+3: 能耗数据分布偏态导致DML失败** | 高 | 中 | 对数变换能耗数据，使用RobustScaler |

### 3.2 缓解方案有效性 (6/10分)

**原缓解方案评估**:
| 风险 | 原缓解方案 | 有效性 | 改进建议 |
|------|-----------|--------|----------|
| ref_df构建 | 先读CTF源码确认 | ❌ 无效 | 改用不使用ref_df |
| T0/T1策略 | 实现多种选择策略 | ⚠️ 部分 | 推荐25/75分位 |
| 混淆因素 | 使用predecessors() | ✅ 有效 | 添加空列表检查 |

**新缓解方案**:
```python
# 1. ref_df: 不使用（简化版）
def compute_ate_simple(parent, child, data_df, causal_graph):
    # 使用predecessors识别混淆因素
    X_cols = _get_confounders_from_graph(parent, child, causal_graph)

    # 不使用ref_df和T0/T1
    est = LinearDML(model_y=RandomForestRegressor(),
                    model_t=RandomForestRegressor())
    est.fit(Y, T, X=data_df[X_cols])

    # EconML自动计算边际效应
    ate = est.ate(X=data_df[X_cols])
    return ate

# 2. T0/T1: 25/75分位（完整版）
def compute_ate_full(parent, child, data_df, causal_graph, T0_q=0.25, T1_q=0.75):
    X_cols = _get_confounders_from_graph(parent, child, causal_graph)

    est = LinearDML(model_y=RandomForestRegressor(),
                    model_t=RandomForestRegressor())
    est.fit(Y, T, X=data_df[X_cols])

    T0 = data_df[parent].quantile(T0_q)
    T1 = data_df[parent].quantile(T1_q)

    # 添加阈值检查
    if abs(T1 - T0) < 0.1 * data_df[parent].std():
        warnings.warn(f"T1-T1差值过小: {T1-T0}")

    ate = est.ate(X=data_df[X_cols], T0=T0, T1=T1)
    return ate
```

### 3.3 遗漏风险 (2/5分)

**新识别的风险**:

1. **⚠️ DiBS因果图的噪声边**
   - 风险: DiBS可能发现虚假因果，影响predecessors()
   - 影响: 混淆因素包含噪声，导致ATE有偏
   - 缓解: 只使用白名单边构建因果图

2. **⚠️ 能耗数据的零值和异常值**
   - 风险: gpu_total_joules可能有0或极端值
   - 影响: DML拟合失败
   - 缓解: 预处理过滤异常值

3. **⚠️ 小样本问题**
   - 风险: group2和group5只有72个样本
   - 影响: DML的RandomForest可能过拟合
   - 缓解: 使用更简单的模型（LinearRegression）

### 3.4 最坏情况 (1/5分)

**最坏场景**:
1. 实施后发现ATE全部不显著
2. ref_df策略完全错误，需要重来
3. 原因寻找功能发现0个trade-off
4. 项目延期1个月，预算超支

**概率**: 20%
**应对**:
- 第3天进行原型验证（Phase 0.5）
- 设置明确的停止标准（如：ATE显著率<10%则停止）
- 准备降级方案（只计算ATE，不做原因寻找）

---

## 4. 替代方案 (7/10分)

### 4.1 更简单的方法 (推荐) ⭐⭐⭐⭐⭐

**方案: 混合方案（CTF核心逻辑 + 简化实现）**

```python
# 核心改动：只替换混淆因素识别，保持其他逻辑
def compute_ate_hybrid(parent, child, data_df, causal_graph):
    # 1. ✅ 使用predecessors()（CTF核心）
    X_cols = _get_confounders_from_graph(parent, child, causal_graph)

    # 2. ❌ 不使用ref_df（简化）
    # 3. ❌ 不使用T0/T1（简化）

    # 4. 保持现有的DML拟合逻辑
    est = LinearDML(model_y=RandomForestRegressor(),
                    model_t=RandomForestRegressor())
    est.fit(Y, T, X=data_df[X_cols])
    ate = est.ate(X=data_df[X_cols])

    return ate
```

**优点**:
- 实施时间: 7-10天（vs 原方案30-40天）
- 风险低：只改核心，保持简单
- 可验证：容易对比现有结果

**缺点**:
- 偏离CTF的ref_df和T0/T1逻辑
- 可能无法完全复现CTF结果

**适用场景**: 如果目标是"尽可能像CTF"，而非"完全复现CTF"

### 4.2 完全复现CTF（不推荐） ⭐⭐

**方案**: 完全按照CTF实现

**优点**:
- 理论上最接近CTF
- 完整性最高

**缺点**:
- 时间: 40-50天
- 风险: ref_df策略可能不适用于能耗数据
- 复杂度: 原因寻找功能非常复杂

**适用场景**: 如果有充足时间和预算，且必须完全复现CTF

### 4.3 最小可行方案（推荐） ⭐⭐⭐⭐

**方案**: 只实现ATE计算，不做原因寻找

**范围**:
1. ✅ 重构混淆因素识别（使用predecessors）
2. ✅ 实现ref_df策略（或选择不使用）
3. ✅ 实现T0/T1策略（或选择不使用）
4. ❌ 不实现原因寻找功能
5. ❌ 不扩展tradeoff检测

**优点**:
- 实施时间: 5-7天
- 快速验证可行性
- 可以先用于问题1的回归分析

**缺点**:
- 功能不完整
- 无法支持问题2的trade-off检测

**适用场景**: 快速原型验证，或作为Phase 0.5

### 4.4 渐进式方案（最推荐） ⭐⭐⭐⭐⭐

**方案**: 分3个迭代，每个迭代都有可用的产出

| 迭代 | 时间 | 范围 | 产出 |
|------|------|------|------|
| **MVP** | 5-7天 | 只重构ATE计算（混合方案） | 可用的ATE估计 |
| **V1.0** | +5天 | 添加T0/T1策略 | 改进的ATE估计 |
| **V2.0** | +10天 | 添加原因寻找 | 完整的trade-off检测 |

**优点**:
- 风险可控：每个迭代都有产出
- 灵活性：可以随时调整方向
- 快速反馈：MVP可以立即验证

**适用场景**: 推荐所有情况使用此方案

---

## 5. 优先级建议 (7/10分)

### 5.1 必须功能 (P0)

1. **混淆因素识别重构** ⭐⭐⭐⭐⭐
   ```python
   def _get_confounders_from_graph(parent, child, causal_graph):
       # 使用NetworkX的predecessors()
       parent_parents = list(causal_graph.predecessors(parent))
       child_parents = list(causal_graph.predecessors(child))
       X_cols = list(set(parent_parents + child_parents))

       # 移除parent自身
       if parent in X_cols:
           X_cols.remove(parent)

       return X_cols
   ```

   **理由**: CTF的核心逻辑，必须一致

2. **ATE接口重构** ⭐⭐⭐⭐⭐
   ```python
   def compute_ate(parent, child, data_df, causal_graph, ref_df=None, T0=None, T1=None):
       # 支持可选的ref_df和T0/T1
       if ref_df is None:
           ref_df = data_df

       if T0 is None:
           T0 = data_df[parent].quantile(0.25)
       if T1 is None:
           T1 = data_df[parent].quantile(0.75)

       # ... 计算ATE
   ```

   **理由**: 支持灵活配置，兼容CTF和现有逻辑

### 5.2 推荐功能 (P1)

3. **ref_df策略实现** ⭐⭐⭐⭐
   ```python
   def build_ref_df(data_df, groupby_col='is_parallel'):
       # 选项A: 非并行模式
       if groupby_col == 'is_parallel':
           return data_df[data_df['is_parallel'] == 0].groupby('model').mean()

       # 选项E: 直接使用data_df（简化版）
       return data_df.groupby('model').mean()
   ```

   **理由**: 支持CTF完整逻辑，但可以简化

4. **T0/T1策略实现** ⭐⭐⭐⭐
   ```python
   def compute_T0_T1(data_df, treatment, strategy='quantile'):
       if strategy == 'quantile':
           return data_df[treatment].quantile(0.25), data_df[treatment].quantile(0.75)
       elif strategy == 'std':
           mean = data_df[treatment].mean()
           std = data_df[treatment].std()
           return mean - std, mean + std
       elif strategy == 'minmax':
           return data_df[treatment].min(), data_df[treatment].max()
   ```

   **理由**: 支持多种策略，便于对比

### 5.3 可选功能 (P2)

5. **原因寻找功能** ⭐⭐
   - **理由**: CTF的复杂逻辑（280-330行），实施风险高
   - **建议**: 先用简单的ATE sign检测，后续再扩展

6. **白名单ATE集成工具** ⭐⭐⭐
   - **理由**: 实用工具，但不是核心
   - **建议**: 可以手动计算，不一定要自动化

### 5.4 最小可行方案 (MVP)

**时间**: 5-7天
**范围**: 只做P0-1和P0-2

```python
# 最小实现
def compute_ate_mvp(parent, child, data_df, causal_graph):
    # 1. 使用predecessors()识别混淆因素
    X_cols = _get_confounders_from_graph(parent, child, causal_graph)

    # 2. 不使用ref_df和T0/T1
    est = LinearDML(model_y=RandomForestRegressor(),
                    model_t=RandomForestRegressor())
    est.fit(Y, T, X=data_df[X_cols])
    ate = est.ate(X=data_df[X_cols])

    return ate
```

**产出**:
- ✅ 可用的ATE估计
- ✅ 与CTF的核心逻辑一致（混淆因素识别）
- ❌ 不包含ref_df和T0/T1
- ❌ 不包含原因寻找

---

## 📋 决策矩阵

### 继续原方案的条件

| 条件 | 检查方式 | 当前状态 |
|------|---------|---------|
| ref_df策略已明确 | 有明确的构建方法 | ❌ 未满足 |
| T0/T1策略已明确 | 有明确的选择逻辑 | ⚠️ 部分满足 |
| 时间预算充足 | ≥30天 | ❌ 未确认 |
| 团队理解CTF | 能完整解释CTF逻辑 | ⚠️ 待确认 |
| 风险可接受 | P0+风险有缓解方案 | ✅ 已评估 |

**结论**: ❌ **不建议完全按原方案继续**

### 需要修改的地方

| 原方案 | 修改建议 | 优先级 |
|--------|---------|--------|
| 使用ref_df | 改为可选参数（默认不使用） | P0 |
| T0/T1使用min/max | 改为25/75分位 | P0 |
| Phase 0-3顺序 | 插入Phase 0.5原型验证 | P0 |
| 实施原因寻找 | 降级为P2可选功能 | P1 |
| 15-22天估算 | 修改为30-40天 | P1 |
| 完全复现CTF | 改为混合方案 | P0 |

### 建议停止的点

如果出现以下情况，建议停止或重新评估：

1. **Phase 0.5原型验证失败**
   - MVP实现的ATE显著率<10%
   - 混淆因素识别导致错误（如包含target本身）

2. **ref_df策略无法确定**
   - 尝试3种策略都无法合理构建ref_df
   - ref_df导致ATE比现有方法更差

3. **时间或预算超支**
   - 实际进度比计划慢50%以上
   - 预算不足以支持完整的30-40天实施

4. **质量问题**
   - DiBS因果图质量差（白名单边<100条）
   - 能耗数据质量问题（缺失率>30%）

---

## 🎯 最终建议

### 推荐方案: 渐进式混合方案 ⭐⭐⭐⭐⭐

**核心理念**: 快速验证，迭代改进，风险可控

**实施路线图**:

#### 迭代1: MVP (5-7天)
**目标**: 验证核心可行性

**范围**:
1. ✅ 重构混淆因素识别（使用predecessors）
2. ✅ 实现ATE计算（不使用ref_df和T0/T1）
3. ✅ 在group1上验证
4. ❌ 不实现原因寻找
5. ❌ 不做全面测试

**成功标准**:
- ATE计算成功，无错误
- 至少50%的边ATE显著
- 与现有方法可对比

**决策点**: 如果成功 → 进入迭代2；如果失败 → 重新评估

#### 迭代2: V1.0 (+5天)
**目标**: 添加ref_df和T0/T1支持

**范围**:
1. ✅ 添加ref_df可选参数
2. ✅ 实现T0/T1策略（25/75分位）
3. ✅ 扩展到所有6组数据
4. ❌ 仍然不实现原因寻找

**成功标准**:
- ref_df版本可用（虽然可能不是必需的）
- T0/T1版本比MVP有改进（显著率提升）
- 6组数据全部成功

**决策点**: 如果成功 → 进入迭代3；如果改进不明显 → 跳过迭代3

#### 迭代3: V2.0 (+10天)
**目标**: 实现原因寻找（可选）

**范围**:
1. ✅ 实现CTF的原因寻找逻辑（280-330行）
2. ✅ 扩展tradeoff检测
3. ✅ 全面测试和文档

**成功标准**:
- 原因寻找功能可用
- 检测到≥5个trade-off
- 技术文档完整

**决策点**: 如果成功 → 项目完成；如果太复杂 → 放弃原因寻找，使用V1.0

### 时间和预算

**总时间**: 10-22天（取决于迭代3是否执行）
- 迭代1 (MVP): 5-7天
- 迭代2 (V1.0): +5天
- 迭代3 (V2.0): +10天（可选）

**预算风险**: 🟡 中等
- MVP: 低风险（5-7天）
- V1.0: 中风险（+5天）
- V2.0: 高风险（+10天，可能跳过）

### 关键成功因素

1. **快速原型**: MVP必须在7天内完成并验证
2. **频繁反馈**: 每3天检查进度，及时调整
3. **灵活降级**: 准备好跳过迭代3，使用V1.0
4. **明确停止点**: 设置清晰的停止标准

---

## 📊 评分卡

| 维度 | 分数 | 权重 | 加权分 | 等级 |
|------|------|------|--------|------|
| 技术可行性 | 21/30 | 30% | 0.21 | 🟡 中 |
| 领域适配性 | 18/30 | 30% | 0.18 | 🔴 低 |
| 风险评估 | 14/20 | 20% | 0.14 | 🟡 中 |
| 替代方案 | 7/10 | 10% | 0.07 | 🟢 高 |
| 优先级建议 | 7/10 | 10% | 0.07 | 🟢 高 |
| **总分** | **67/100** | **100%** | **0.67** | **🟡 中** |

**等级定义**:
- 🟢 优秀 (≥80分): 强烈推荐实施
- 🟡 良好 (60-79分): 建议实施（有条件）
- 🔴 一般 (<60分): 不建议实施

**最终建议**: 🟡 **建议实施（采用渐进式混合方案）**

---

## 📝 附录

### A. 参考文件

1. **CTF源码**:
   - `/home/green/energy_dl/nightly/analysis/CTF_original/src/inf.py`
   - 重点行: 78-100 (compute_ate), 280-330 (原因寻找)

2. **当前实现**:
   - `/home/green/energy_dl/nightly/analysis/utils/causal_inference.py`
   - `/home/green/energy_dl/nightly/analysis/utils/tradeoff_detection.py`

3. **白名单数据**:
   - `/home/green/energy_dl/nightly/analysis/results/energy_research/data/interaction/whitelist/*.csv`

### B. 关键代码片段

**CTF的compute_ate** (78-100行):
```python
def compute_ate(parent, child, data_df, ref_df, dg, T0, T1):
    parent_parents = list(dg.predecessors(parent))
    child_parents = list(dg.predecessors(child))
    X_cols = list(set(parent_parents + child_parents))

    if "EGR" in X_cols:
        X_cols.remove("EGR")
    if parent in X_cols:
        X_cols.remove(parent)

    X = data_df[X_cols]
    T = data_df[parent]
    Y = data_df[child]

    est = LinearDML(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor(),
        random_state=0)
    est.fit(Y, T, X=X)
    ate = est.ate(X=ref_df[X_cols], T0=T0, T1=T1)

    return ate
```

**CTF的原因寻找** (280-330行):
```python
# 1. 找到共同祖先
ancestor_A = set(list(nx.ancestors(dg, metric_A)))
ancestor_B = set(list(nx.ancestors(dg, metric_B)))
common_ancestor = list(ancestor_A.intersection(ancestor_B))

# 2. 分析路径
for ca in common_ancestor:
    toX_paths = nx.all_simple_paths(dg, source=ca, target=metric_A)
    toX_last_step = set([x[-2] for x in toX_paths])
    toY_paths = nx.all_simple_paths(dg, source=ca, target=metric_B)
    toY_last_step = set([x[-2] for x in toY_paths])
    ca_last_step[ca] = (toX_last_step, toY_last_step)

# 3. 过滤潜在原因
explored_step = set()
potential_causes = common_ancestor.copy()
for ca, (toX_last_step, toY_last_step) in ca_last_step.items():
    if toX_last_step.issubset(explored_step) and toY_last_step.issubset(explored_step):
        potential_causes.remove(ca)
    else:
        explored_step.update(toX_last_step)
        explored_step.update(toY_last_step)

# 4. 计算ATE
for pc in potential_causes:
    T0 = ref_df[pc].mean()
    T1 = data_df[data_df[fair_method] == 1][pc].mean()
    ate_A = compute_ate(pc, metric_A, data_df, ref_df, dg, T0, T1)
    ate_B = compute_ate(pc, metric_B, data_df, ref_df, dg, T0, T1)
    # 判断是否构成trade-off...
```

### C. 决策流程图

```
开始评估
    ↓
检查P0+风险
    ↓
    ├─ ref_df策略未明确 → 采用混合方案（不使用ref_df）
    ├─ T0/T1策略未明确 → 使用25/75分位
    └─ 混淆因素识别 → 使用predecessors()
    ↓
选择实施路径
    ↓
    ├─ 路径A: 完全复现CTF (40-50天, 高风险)
    ├─ 路径B: 混合方案 (10-20天, 中风险) ✅ 推荐
    └─ 路径C: 最小可行方案 (5-7天, 低风险)
    ↓
迭代开发
    ↓
    ├─ 迭代1: MVP (5-7天) → 验证 → 成功/失败
    ├─ 迭代2: V1.0 (+5天) → 改进 → 成功/失败
    └─ 迭代3: V2.0 (+10天, 可选) → 完整功能
    ↓
最终产出
```

---

**报告完成时间**: 2026-01-25
**下次评估**: Phase 0.5完成后（约7天后）
**状态**: ✅ 建议实施渐进式混合方案
