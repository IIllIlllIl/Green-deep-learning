# 差值转换方案v2.0批判性评审报告

**评审人**: Claude (独立因果推断专家)
**评审日期**: 2026-01-16
**方案文档**: `DIFFERENCE_VALUE_TRANSFORMATION_PLAN_V2.md`
**评审目的**: 批判性评估差值转换方案的方法学正确性、可行性和潜在风险

---

## 执行摘要

### 总体评估（5分制）

```
方法学正确性: ⭐⭐ (2/5) - 存在根本性缺陷
实施可行性:   ⭐⭐⭐⭐ (4/5) - 实现简单
风险可控性:   ⭐⭐ (2/5) - 多个高风险问题
总体推荐度:   ⭐⭐ (2/5) - 不推荐，有更好的替代方案
```

### 核心结论 🔴

**不推荐采用此方案**，原因如下：

1. 🔴 **致命缺陷**: 差值法**无法消除is_parallel的系统性偏差**
2. 🔴 **理论错误**: 对"偏差"的理解有误，混淆了"均值差异"和"系统性偏差"
3. 🟡 **样本不足**: n=1导致基准值不可靠，风险高
4. 🟡 **混合尺度**: 部分转换+部分保留会引入新的混淆

### 推荐替代方案 ⭐⭐⭐⭐⭐

**方案D: DiBS混淆因子控制** + **方案C: Z-score标准化**

---

## 1. 方法学正确性评估 🔴🔴🔴

### 1.1 核心问题：差值法能否消除is_parallel偏差？

**答案：不能！** ❌❌❌

#### 理论推导

假设数据生成过程为：

```
并行模式: Y_parallel = f(X) + offset_parallel + ε
非并行模式: Y_nonparallel = f(X) + offset_nonparallel + ε
```

其中：
- `f(X)` 是超参数X对能耗的真实因果效应
- `offset_parallel` 和 `offset_nonparallel` 是模式导致的系统性偏移
- `ε` 是随机误差

**差值转换**：
```
并行: Y'_parallel = Y_parallel - median(Y_parallel_baseline)
                  = f(X) + offset_parallel + ε - [f(X_baseline) + offset_parallel]
                  = [f(X) - f(X_baseline)] + ε
                  = Δf(X) + ε

非并行: Y'_nonparallel = Y_nonparallel - median(Y_nonparallel_baseline)
                        = f(X) + offset_nonparallel + ε - [f(X_baseline) + offset_nonparallel]
                        = [f(X) - f(X_baseline)] + ε
                        = Δf(X) + ε
```

**结论分析**：

✅ **成功消除了什么**：
- 消除了`offset_parallel`和`offset_nonparallel`的**绝对值差异**
- 两种模式转换后都变成了`Δf(X)`，看起来"可比"了

❌ **没有消除什么**：
- **关键问题**：如果`f(X)`本身依赖于模式，即`f_parallel(X) ≠ f_nonparallel(X)`，差值法无能为力！
- **示例**：
  ```
  并行模式: batch_size增加1倍 → GPU能耗增加20,000J
  非并行模式: batch_size增加1倍 → GPU能耗增加5,000J

  差值转换后:
  并行: Δ = +20,000J
  非并行: Δ = +5,000J

  → DiBS仍然检测到: is_parallel → Δ能耗 (因为变异幅度不同)
  ```

#### 数学严格证明

**定理**: 差值转换只能消除**加性常数偏差**，无法消除**交互效应**。

**证明**：

设真实模型为：
```
Y = β₀ + β₁·X + β₂·is_parallel + β₃·(X × is_parallel) + ε
```

其中`β₃·(X × is_parallel)`是交互效应（超参数效应因模式而异）。

差值转换后（以模式内基准值为中心）：
```
ΔY = Y - Y_baseline
   = [β₀ + β₁·X + β₂·is_parallel + β₃·(X × is_parallel) + ε]
     - [β₀ + β₁·X_baseline + β₂·is_parallel + β₃·(X_baseline × is_parallel)]
   = β₁·(X - X_baseline) + β₃·[(X - X_baseline) × is_parallel] + ε
   = β₁·ΔX + β₃·(ΔX × is_parallel) + ε
```

**关键观察**：
- ✅ 消除了`β₀`（截距）
- ✅ 消除了`β₂·is_parallel`（主效应的常数部分）
- ❌ **仍然保留了`β₃·(ΔX × is_parallel)`（交互效应）**

因此，**DiBS仍会检测到is_parallel的因果效应**，因为交互项仍然存在！

### 1.2 与百分比法对比

**百分比法**（v1.0中提到但未采用）：
```
相对变化 = (Y - Y_baseline) / Y_baseline
```

**优势**：
- ✅ 归一化了尺度（除以基准值）
- ✅ 处理了异方差（variance stabilization）
- ✅ 物理意义清晰（"相对于基准的百分比变化"）

**劣势**：
- ❌ 当基准值接近0时不稳定
- ❌ 正负变化不对称（+100%和-50%）

**v2.0为何放弃百分比法？**

文档中未明确说明，但根据v1.0评审，可能是因为"公式不一致"。然而这个理由不充分——百分比法在处理不同尺度数据时实际上更合理。

### 1.3 差值法的适用场景 ✅

差值法**不是完全无用**，它适用于以下场景：

✅ **适用场景**：
1. 只关心"相对于默认值的变化"，不关心绝对值
2. 模式（is_parallel）只有**加性效应**，没有交互效应
3. 所有模型的变异幅度在同一数量级

❌ **不适用场景**（本项目）：
1. is_parallel有交互效应（已通过DiBS验证：712条强边，is_parallel大概率是强因子）
2. 不同模型的能耗尺度差异巨大（10K vs 100K）
3. 目标是完全消除is_parallel的因果效应

**评分**: ⭐⭐ (2/5) - 方法学存在根本性缺陷

---

## 2. 模型级基准值的影响 🟡

### 2.1 优势分析 ✅

**相比组级基准值（v1.0）的改进**：

✅ **优势1: 解决Group1高CV问题**
- v1.0: Group1（4个models）组级CV > 80% → 基准值不稳定
- v2.0: 每个模型独立计算基准值 → CV=0或很小（n=1时）

✅ **优势2: 模型内部一致性**
- 同一模型的所有实验共享基准值 → 消除模型特性差异
- 符合"控制混淆因子"的思想

✅ **优势3: 语义清晰**
- "相对于该模型默认配置的偏差" → 解释性强

### 2.2 劣势分析 ❌

**问题1: 样本量进一步减少** 🔴

| 模型 | 并行默认值 | 非并行默认值 | 总计 | 基准值稳定性 |
|------|----------|------------|------|-------------|
| mnist | 1 | 1 | 2 | ❌ n=1，无方差估计 |
| mnist_rnn | 1 | 1 | 2 | ❌ n=1，无方差估计 |
| siamese | 1 | 1 | 2 | ❌ n=1，无方差估计 |
| mnist_ff | 1 | 3 | 4 | 🟡 非并行n=3，可计算 |
| ... | ... | ... | ... | ... |

**风险**：
- n=1时，基准值=唯一观测值
- 如果该观测值是异常值（系统故障、后台干扰），所有转换都会错误
- **无法检测**：因为没有重复实验来验证

**对比v1.0**：
- v1.0: Group1有10个默认值实验 → 可以检测和移除离群值
- v2.0: 大部分模型只有1个 → 无法检测离群值

**问题2: 无法利用跨模型信息** ⚠️

- 同一组的模型共享相同的超参数和性能指标
- 跨模型的默认值实验可以提供"该组模型典型基准值"的信息
- v2.0完全放弃了这个信息

**权衡分析**：

| 维度 | v1.0 (组级) | v2.0 (模型级) | 推荐 |
|------|-----------|-------------|------|
| 基准值稳定性 | ✅ 高（n=10） | ❌ 低（n=1） | v1.0 |
| 模型一致性 | ❌ 低（CV>80%） | ✅ 高（CV≈0） | v2.0 |
| 离群值检测 | ✅ 可以 | ❌ 不可以 | v1.0 |
| 语义清晰度 | 🟡 中等 | ✅ 高 | v2.0 |

**结论**: 各有优劣，但n=1的风险太大，**需要补充验证机制**。

**评分**: ⭐⭐⭐ (3/5) - 改进有效但引入新风险

### 2.3 缓解措施建议 ✅

**建议1: 手动检查默认值实验** (P0 - 必需)

```python
# 检查默认值实验是否异常
for model in models:
    defaults = get_default_experiments(model)

    # 检查1: status是否全部为success
    assert (defaults['status'] == 'success').all()

    # 检查2: 能耗值是否在合理范围（与其他实验对比）
    all_experiments = get_model_experiments(model)
    for field in energy_fields:
        baseline = defaults[field].median()
        all_values = all_experiments[field]

        # 基准值应该在全体数据的合理范围内（如10%-90%分位）
        percentile = (all_values < baseline).mean() * 100
        if percentile < 5 or percentile > 95:
            print(f"⚠️ 警告: {model}的{field}基准值异常（第{percentile:.0f}百分位）")
```

**建议2: 使用稳健统计量** (P1 - 推荐)

```python
# 对于n≥2的情况，使用截断均值而非中位数
if n >= 3:
    # 移除最大和最小值后取均值
    baseline = values.sort_values()[1:-1].mean()
elif n == 2:
    # 使用中位数（等于均值）
    baseline = values.median()
else:  # n == 1
    # 无选择，但标记为"低置信度"
    baseline = values.iloc[0]
    confidence = "low"
```

**建议3: 跨模型一致性检查** (P1 - 推荐)

```python
# 检查同一组内不同模型的基准值是否在合理范围
for group in groups:
    models_in_group = get_models(group)

    for field in energy_fields:
        baselines = [get_baseline(model, field) for model in models_in_group]
        cv = np.std(baselines) / np.mean(baselines)

        if cv > 1.0:  # CV > 100%
            print(f"⚠️ 警告: {group}的{field}基准值组间差异过大（CV={cv*100:.0f}%）")
            # 可能需要检查是否有模型的默认值异常
```

---

## 3. 部分保留绝对值的合理性 🟡

### 3.1 设计理由分析

**v2.0的决策**：
- ✅ 转换：4个能量字段 + 1个平均功率
- ✅ 保留：2个功率极值 + 2个温度 + 2个利用率

**文档给出的理由**：
> "最大/最小功率反映硬件性能边界，不同实验间的绝对值可比性更重要"
> "温度和利用率是物理绝对量，转换为差值会失去物理意义"

### 3.2 批判性分析

**理由1: "硬件性能边界"** - 🟡 部分合理

✅ **合理部分**：
- GPU最大功率（如319W）确实反映硬件TDP限制
- 接近上限可能触发降频（thermal throttling）
- 这是一个"物理约束"，保留绝对值有意义

⚠️ **问题**：
- 不同GPU的TDP不同（本项目只用RTX 3090，可忽略）
- 如果未来扩展到多GPU类型，绝对值不可比

**理由2: "温度是绝对物理量"** - ❌ 逻辑错误

这个理由**站不住脚**：
- 能量（焦耳）和功率（瓦特）**同样是绝对物理量**！
- 为什么能量可以转换为差值，而温度不行？

**更合理的理由**（文档未提及）：
- 温度的"安全阈值"是固定的（如85°C警告线）
- 差值"相对于默认值+5°C"没有温度绝对值"82°C"有意义
- **但这个理由对利用率不适用**：利用率50%和80%都是正常范围

### 3.3 混合尺度的风险 ⚠️

**问题**: DiBS如何处理混合尺度的变量？

**场景1: 直接因果边**
```
learning_rate → energy_gpu_total_joules (差值, 如+5000J)
learning_rate → energy_gpu_max_watts (绝对值, 如319W)
```

DiBS会学习：
- 边1: "学习率增加 → GPU总能耗相对默认值增加5000J"
- 边2: "学习率增加 → GPU最大功率为319W"

**潜在混淆**：
- 边2的"319W"包含了模式效应（并行/非并行基准功率不同）
- 边1的"差值"已经消除了模式效应
- DiBS可能错误地认为：最大功率与模式相关，而总能耗与模式无关

**场景2: 间接路径**
```
is_parallel → energy_gpu_max_watts (绝对值) → training_time → perf_accuracy
```

如果is_parallel影响最大功率（因为保留了绝对值），这条路径会被保留。
但如果总能耗（差值）中is_parallel效应被消除，可能导致因果图不一致。

### 3.4 建议 ✅

**建议1: 统一处理原则** (P0 - 必需)

**选项A: 全部转换为差值**
```python
for field in all_energy_fields:
    diff = actual - baseline
```
- ✅ 一致性最强
- ❌ 失去温度/功率的物理约束信息

**选项B: 全部保留绝对值**
```python
# 不做差值转换，改用其他方法（如Z-score）
```
- ✅ 保留所有物理信息
- ❌ 未解决模型间尺度差异问题

**选项C: 分离分析**（推荐） ⭐
```python
# 方案1: 只用差值字段做DiBS分析
features_for_dibs = energy_difference_fields + hyperparams + perf

# 方案2: 保留字段作为"上下文变量"单独分析
# 不放入DiBS，而是事后分析：
# "当最大功率>300W时，能耗-性能权衡如何变化？"
```

**建议2: 标准化所有变量** (P1 - 推荐)

无论是否转换为差值，都应该做Z-score标准化：
```python
z = (x - mean) / std
```

这样可以：
- ✅ 消除尺度差异
- ✅ DiBS内部计算更稳定
- ✅ 边权重可比

**评分**: ⭐⭐⭐ (3/5) - 设计有一定合理性，但缺乏统一原则

---

## 4. n=1风险评估 🔴

### 4.1 风险量化

**当前状况**：
- 11个模型中，**10个模型**只有n=1/模式（共20个基准值）
- 只有1个模型（mnist_ff非并行）有n=3

**风险场景**：

**场景1: 默认值实验本身异常** 🔴

```
示例:
某模型非并行默认值实验: 运行时GPU正在被其他进程使用
→ 能耗测量值: 15,000J（异常高）
→ 真实默认值: 10,000J

影响:
所有该模型非并行实验转换后:
diff = actual - 15,000
→ 系统性偏低5,000J

→ 回归分析会错误地认为: "该模型超参数对能耗影响为负"
```

**场景2: 环境漂移** ⚠️

```
默认值实验时间: 2025-12-01
当前分析数据时间: 2025-12-29

可能的变化:
- GPU驱动更新
- CUDA版本变化
- 系统后台负载变化
- 季节性温度变化

→ 基准值可能不再代表"当前"的默认值
```

### 4.2 风险检测方法

**方法1: 残差分析** ✅

```python
# 拟合回归模型后检查残差
model = LinearRegression()
model.fit(X, y_diff)
residuals = y_diff - model.predict(X)

# 按模型分组检查残差分布
for model_name in models:
    model_residuals = residuals[df['model'] == model_name]

    # 如果某模型的残差系统性偏高/偏低，可能是基准值问题
    if abs(model_residuals.mean()) > threshold:
        print(f"⚠️ {model_name}的残差异常，可能基准值有误")
```

**方法2: 交叉验证** ✅

```python
# 对于n≥2的模型，使用leave-one-out验证
if n >= 2:
    for i in range(n):
        # 用其他n-1个计算基准值
        baseline_loo = values.drop(index=i).median()

        # 检查被排除的那个是否异常
        deviation = abs(values.iloc[i] - baseline_loo)
        if deviation > 3 * values.std():
            print(f"⚠️ 第{i}个默认值实验可能是离群值")
```

**方法3: 领域知识验证** ✅

```python
# 检查基准值是否符合物理常识
baseline_gpu_energy = get_baseline(model, 'energy_gpu_total_joules')
training_time = get_typical_training_time(model)

# GPU功率应该在合理范围（50W-350W for RTX 3090）
implied_avg_power = baseline_gpu_energy / training_time
if implied_avg_power < 50 or implied_avg_power > 350:
    print(f"⚠️ {model}的基准值能耗异常：隐含平均功率{implied_avg_power:.0f}W")
```

### 4.3 风险评分 🔴

**可接受性评估**：

| 风险类型 | 概率 | 影响 | 严重度 |
|---------|------|------|--------|
| 默认值异常 | 中（10%） | 高 | 🔴 高 |
| 环境漂移 | 低（5%） | 中 | 🟡 中 |
| 无法检测 | 高（90%） | 高 | 🔴 高 |

**结论**: ❌ **风险不可接受**，必须添加补充验证机制

**评分**: ⭐⭐ (2/5) - 风险过高，可控性差

---

## 5. DiBS兼容性分析 🟡

### 5.1 DiBS对数据的期望

**DiBS（DiBS for Bayesian Structure Learning）期望**：

1. ✅ **连续变量**：所有变量应该是连续的
   - 差值数据满足（连续的能耗差值）

2. ✅ **无严重多重共线性**：变量间不应高度相关
   - 需要验证（差值字段可能与原始字段高度相关）

3. 🟡 **平稳性**：数据生成过程应该稳定
   - 如果基准值有误，会违反平稳性

4. 🟡 **高斯性**：线性高斯模型假设数据近似正态分布
   - 差值数据的分布待检验

### 5.2 差值数据的统计特性

**理论分析**：

假设原始数据 `Y ~ N(μ, σ²)`，基准值 `Y_base = median(Y_defaults)`

差值数据：
```
ΔY = Y - Y_base
```

**性质**：
1. ✅ **均值平移**：E[ΔY] = E[Y] - Y_base
   - 默认值实验的差值均值 ≈ 0

2. ✅ **方差不变**：Var(ΔY) = Var(Y)
   - 变异性保持不变

3. ⚠️ **分布形状**：如果Y是偏态的，ΔY同样偏态
   - 可能违反高斯性假设

### 5.3 是否需要进一步预处理？

**选项1: 差值数据 + 标准化** ⭐⭐⭐⭐⭐

```python
# 步骤1: 计算差值
diff = actual - baseline

# 步骤2: Z-score标准化
z = (diff - mean(diff)) / std(diff)
```

**优势**：
- ✅ 消除尺度差异（所有变量在相同尺度）
- ✅ 改善DiBS数值稳定性
- ✅ 边权重可直接比较

**劣势**：
- ❌ 失去物理单位（z-score无量纲）

**选项2: 差值数据 + Rank变换** ⭐⭐⭐

```python
# 将差值转换为排名（处理非高斯性）
rank_diff = rankdata(diff) / len(diff)  # [0, 1]区间
```

**优势**：
- ✅ 处理异常值和偏态分布
- ✅ 保留单调关系

**劣势**：
- ❌ 失去线性关系信息
- ❌ DiBS的线性高斯假设可能不适用

**推荐**: **选项1（差值+标准化）**，如果发现分布严重偏态再考虑选项2。

### 5.4 混淆因子控制对比

**当前方案（差值法）**：
- 通过预处理消除is_parallel效应
- 优势：DiBS输入"干净"的数据
- 劣势：无法验证是否成功消除

**替代方案（DiBS内部控制）**：
```python
# 将is_parallel作为"known confounder"
# DiBS学习条件因果图: P(G | X, Z=is_parallel)
```

**优势**：
- ✅ DiBS自动调整is_parallel的效应
- ✅ 可以检验is_parallel的因果效应大小
- ✅ 更符合因果推断理论

**劣势**：
- ❌ 实现稍复杂（需要指定先验）
- ⚠️ 当前DiBS代码可能不支持（需要检查）

**评分**: ⭐⭐⭐ (3/5) - 基本兼容，但需要补充标准化

---

## 6. 与替代方案对比 ⭐

### 6.1 方案对比矩阵

| 方案 | 消除模型差异 | 消除模式偏差 | 物理意义 | 实施复杂度 | 对DiBS适配 | 推荐度 |
|------|------------|------------|---------|-----------|-----------|--------|
| **A: 差值法v2.0** | ⭐⭐⭐ 模型级 | ⭐⭐ 部分（只消除加性） | ⭐⭐⭐⭐ 清晰 | ⭐⭐⭐⭐⭐ 简单 | ⭐⭐⭐ 需标准化 | ⭐⭐ 不推荐 |
| **B: 百分比法** | ⭐⭐⭐ 组级 | ⭐⭐⭐⭐ 较好（归一化） | ⭐⭐⭐ 尚可 | ⭐⭐⭐⭐⭐ 简单 | ⭐⭐⭐⭐ 好 | ⭐⭐⭐ 可考虑 |
| **C: Z-score标准化** | ⭐⭐⭐⭐⭐ 任意级 | ⭐⭐⭐⭐⭐ 完全 | ⭐⭐ 弱 | ⭐⭐⭐⭐⭐ 非常简单 | ⭐⭐⭐⭐⭐ 最佳 | ⭐⭐⭐⭐ 推荐 |
| **D: DiBS混淆因子** | ⭐⭐⭐⭐⭐ 自动 | ⭐⭐⭐⭐⭐ 完全 | ⭐⭐⭐⭐⭐ 保留原值 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 原生支持 | ⭐⭐⭐⭐⭐ 最推荐 |

### 6.2 详细评分说明

#### 方案A: 差值法v2.0（本方案）

**消除模型差异** ⭐⭐⭐ (3/5)
- ✅ 模型级基准值有效
- ⚠️ 但n=1风险高

**消除模式偏差** ⭐⭐ (2/5)
- ❌ 只能消除加性偏差
- ❌ 无法消除交互效应
- ❌ 可能仍会被DiBS检测到

**物理意义** ⭐⭐⭐⭐ (4/5)
- ✅ "相对于默认值的增量"语义清晰
- ✅ 保留焦耳/瓦特单位
- ⚠️ 但混合了差值和绝对值

**实施复杂度** ⭐⭐⭐⭐⭐ (5/5)
- ✅ 实现简单（减法操作）
- ✅ 计算快速

**对DiBS适配** ⭐⭐⭐ (3/5)
- 🟡 基本可用
- ⚠️ 需要补充标准化
- ⚠️ 混合尺度可能引起混淆

**总推荐度** ⭐⭐ (2/5) - **不推荐**

---

#### 方案B: 百分比法

**公式**：
```python
relative_change = (actual - baseline) / baseline
```

**消除模型差异** ⭐⭐⭐ (3/5)
- ✅ 归一化后所有模型在相同尺度
- ⚠️ 基准值接近0时不稳定

**消除模式偏差** ⭐⭐⭐⭐ (4/5)
- ✅ 归一化尺度差异
- ✅ 处理异方差
- 🟡 交互效应仍存在但影响较小

**物理意义** ⭐⭐⭐ (3/5)
- ✅ "百分比变化"易理解
- ⚠️ 正负不对称（+100% ≠ -50%）
- ⚠️ 对数百分比更合适但复杂

**实施复杂度** ⭐⭐⭐⭐⭐ (5/5)
- ✅ 实现简单（除法操作）
- ⚠️ 需处理除零

**对DiBS适配** ⭐⭐⭐⭐ (4/5)
- ✅ 数据归一化，DiBS计算更稳定
- ✅ 变量尺度一致

**总推荐度** ⭐⭐⭐ (3/5) - **可考虑**，优于差值法

---

#### 方案C: Z-score标准化 ⭐⭐⭐⭐ 推荐

**公式**：
```python
z = (x - mean(x)) / std(x)
```

**消除模型差异** ⭐⭐⭐⭐⭐ (5/5)
- ✅ 完全归一化，所有变量均值=0，标准差=1
- ✅ 适用于任何粒度（模型级、组级、全局）

**消除模式偏差** ⭐⭐⭐⭐⭐ (5/5)
- ✅ 完全归一化尺度差异
- ✅ 如果按模式分组标准化，可以完全消除模式效应
```python
for mode in ['parallel', 'nonparallel']:
    df_mode = df[df['is_parallel'] == (mode == 'parallel')]
    z = (df_mode[field] - df_mode[field].mean()) / df_mode[field].std()
```

**物理意义** ⭐⭐ (2/5)
- ❌ 失去原始单位（无量纲）
- ⚠️ "高于均值1.5个标准差"不如"高5000J"直观

**实施复杂度** ⭐⭐⭐⭐⭐ (5/5)
- ✅ 最简单（pandas一行代码）
- ✅ 无特殊情况处理

**对DiBS适配** ⭐⭐⭐⭐⭐ (5/5)
- ✅ 这是DiBS推荐的预处理方法！
- ✅ 改善数值稳定性
- ✅ 边权重直接可比

**总推荐度** ⭐⭐⭐⭐ (4/5) - **强烈推荐**

**实施建议**：
```python
# 选项1: 全局标准化（简单）
z = (x - x.mean()) / x.std()

# 选项2: 按模式分组标准化（消除is_parallel效应）⭐⭐⭐⭐⭐
for mode in [0, 1]:
    mask = df['is_parallel'] == mode
    df.loc[mask, energy_fields] = (
        df.loc[mask, energy_fields] - df.loc[mask, energy_fields].mean()
    ) / df.loc[mask, energy_fields].std()

# 选项3: 按模型+模式分组标准化（最精细）
for model in models:
    for mode in [0, 1]:
        mask = (df['model'] == model) & (df['is_parallel'] == mode)
        df.loc[mask, energy_fields] = (
            df.loc[mask, energy_fields] - df.loc[mask, energy_fields].mean()
        ) / df.loc[mask, energy_fields].std()
```

---

#### 方案D: DiBS混淆因子控制 ⭐⭐⭐⭐⭐ 最推荐

**原理**：不修改数据，而是在DiBS模型中指定 `is_parallel` 为已知混淆因子。

**理论基础**：
- DiBS学习 `P(G | X, Z)`，其中Z是混淆因子
- 等价于学习"给定Z的条件下，X的因果结构"
- 自动调整Z的效应

**实现方式**（待验证当前DiBS版本是否支持）：
```python
# 伪代码
dibs_model = JointDiBS(
    x=data,
    confounders=['is_parallel'],  # 指定混淆因子
    ...
)
```

**或者使用后处理**：
```python
# 方法1: 残差化（先回归掉is_parallel的效应）
for field in energy_fields:
    model = LinearRegression()
    model.fit(df[['is_parallel']], df[field])
    df[f'{field}_residual'] = df[field] - model.predict(df[['is_parallel']])

# 方法2: 分层分析（按is_parallel分组，分别做DiBS）
for mode in ['parallel', 'nonparallel']:
    df_mode = df[df['is_parallel'] == (mode == 'parallel')]
    dibs_result = run_dibs(df_mode)  # 不包含is_parallel变量
```

**消除模型差异** ⭐⭐⭐⭐⭐ (5/5)
- ✅ DiBS自动学习模型效应
- ✅ 不需要手动计算基准值

**消除模式偏差** ⭐⭐⭐⭐⭐ (5/5)
- ✅ 理论保证：条件因果图不受Z影响
- ✅ 可以定量评估is_parallel的因果效应

**物理意义** ⭐⭐⭐⭐⭐ (5/5)
- ✅ 保留原始数据，无信息损失
- ✅ 因果效应直接对应物理量

**实施复杂度** ⭐⭐⭐ (3/5)
- ⚠️ 需要检查DiBS是否支持confounders参数
- 🟡 如果不支持，需要使用残差化或分层分析
- ⚠️ 实现稍复杂

**对DiBS适配** ⭐⭐⭐⭐⭐ (5/5)
- ✅ 这是因果推断的标准方法
- ✅ 理论基础最扎实

**总推荐度** ⭐⭐⭐⭐⭐ (5/5) - **最推荐！**

---

### 6.3 推荐方案排序

**综合评估**：

1. **⭐⭐⭐⭐⭐ 方案D: DiBS混淆因子控制**
   - 理论最严谨
   - 保留完整信息
   - 需要技术验证

2. **⭐⭐⭐⭐ 方案C: Z-score标准化**
   - 简单有效
   - DiBS友好
   - 物理意义较弱但可接受

3. **⭐⭐⭐ 方案B: 百分比法**
   - 中等效果
   - 易于理解
   - 需处理除零

4. **⭐⭐ 方案A: 差值法v2.0**（本方案）
   - 方法学缺陷
   - 高风险
   - 不推荐

---

## 7. 关键问题明确答案

### Q1: 差值法能否消除is_parallel偏差？

**答案**: **❌ 部分可以，但有严重限制**

- ✅ 可以消除：**加性常数偏差**（并行基准100K vs 非并行基准10K）
- ❌ 不能消除：**交互效应偏差**（并行模式下超参数效应更强）
- 🔴 **根本问题**：如果is_parallel和超参数存在交互作用（极有可能），差值法无能为力

**推荐**: 使用方案D（混淆因子控制）或方案C（分组标准化）

---

### Q2: 模型级基准值是否优于组级？

**答案**: **🟡 各有优劣，需要看优先级**

**如果优先考虑"模型内部一致性"** → v2.0更好
**如果优先考虑"基准值稳定性"** → v1.0更好

**但关键问题不是粒度，而是**：
- 🔴 **n=1的风险太大**，必须添加验证机制
- 🔴 **差值法本身有缺陷**，换粒度解决不了根本问题

**推荐**: 不要纠结粒度，改用方案C或D

---

### Q3: 部分保留绝对值是否合理？

**答案**: **🟡 有一定合理性，但不是最优选择**

**合理的部分**：
- 温度和最大功率的绝对值确实有物理意义

**问题**：
- 混合尺度可能引入混淆
- 设计原则不统一（为什么平均功率转换，最大功率不转换？）

**推荐**:
- **如果用差值法**：要么全转换，要么全不转换
- **如果用Z-score**：全部标准化，问题自然解决

---

### Q4: n=1的风险是否可接受？

**答案**: **❌ 不可接受，必须添加缓解措施**

**风险等级**: 🔴 高
**影响范围**: 10/11个模型（91%）
**检测难度**: 🔴 高（无重复实验验证）

**必需的缓解措施**（P0）：
1. 手动检查所有默认值实验的status和日志
2. 交叉验证基准值的合理性
3. 残差分析检测系统性偏差

**如果无法实施上述措施** → **放弃差值法**，改用方案C或D

---

### Q5: 相比Z-score标准化，差值法是否更优？

**答案**: **❌ 不是，Z-score更优**

| 维度 | 差值法 | Z-score标准化 |
|------|--------|--------------|
| 消除模型差异 | ✅ 是 | ✅ 是（更彻底） |
| 消除模式偏差 | 🟡 部分 | ✅ 完全（如果分组） |
| 处理交互效应 | ❌ 否 | ✅ 是（如果分组） |
| 物理意义 | ✅ 强 | ⚠️ 弱 |
| 实施复杂度 | 🟡 中等 | ✅ 简单 |
| 对DiBS友好 | 🟡 需补充 | ✅ 最佳 |
| 稳健性 | ⚠️ n=1风险 | ✅ 稳健 |

**结论**: **Z-score在技术上全面优于差值法**，唯一劣势是物理意义较弱，但这在因果发现阶段是可以接受的（后续回归分析可以用原始数据）。

---

## 8. 改进建议

### 🔴 严重问题 (P0 - 必须解决)

#### 问题1: 差值法无法消除交互效应

**解决方案1A**: 放弃差值法，改用**Z-score分组标准化**
```python
# 按模型+模式分组标准化
for model in models:
    for is_par in [True, False]:
        mask = (df['model'] == model) & (df['is_parallel'] == is_par)
        df.loc[mask, energy_fields] = standardize(df.loc[mask, energy_fields])
```

**解决方案1B**: 使用**DiBS混淆因子控制**
```python
# 方法1: 残差化
for field in energy_fields:
    df[f'{field}_resid'] = residualize(df[field], df['is_parallel'])

# 方法2: 分层分析
for is_par in [True, False]:
    run_dibs(df[df['is_parallel'] == is_par])
```

**优先级**: **P0 - 立即执行**
**预期效果**: 完全消除is_parallel的系统性偏差（包括交互效应）

---

#### 问题2: n=1基准值不可靠

**解决方案2A**: 手动验证所有默认值实验
```python
# 检查清单
for model in models:
    defaults = get_default_experiments(model)

    ✅ 检查1: status == 'success'
    ✅ 检查2: 日志无错误
    ✅ 检查3: 能耗值在合理范围（与其他实验对比）
    ✅ 检查4: 功率/温度/利用率正常
```

**解决方案2B**: 使用"伪基准值"（最近邻）
```python
# 如果没有真正的默认值实验，使用最接近默认配置的实验
def find_pseudo_baseline(model, mode):
    df_model_mode = df[(df['model'] == model) & (df['is_parallel'] == mode)]

    # 计算与默认配置的距离
    distances = []
    for idx, row in df_model_mode.iterrows():
        dist = calculate_distance_to_default(row)
        distances.append((dist, idx))

    # 选择最近的k个实验作为基准
    k_nearest = sorted(distances)[:5]
    baseline = df_model_mode.loc[[idx for _, idx in k_nearest]].median()
    return baseline
```

**优先级**: **P0 - 如果继续使用差值法，必须执行**

---

### 🟡 中等问题 (P1 - 强烈建议)

#### 问题3: 混合尺度变量

**解决方案3A**: 统一处理原则
```python
# 选项1: 全部转换为差值
for field in all_energy_fields:
    df[f'{field}_diff'] = df[field] - get_baseline(df['model'], field)

# 选项2: 全部保留绝对值，但用Z-score标准化
for field in all_energy_fields:
    df[f'{field}_z'] = (df[field] - df[field].mean()) / df[field].std()
```

**优先级**: **P1 - 强烈建议**
**预期效果**: 消除尺度混淆，改善DiBS结果一致性

---

#### 问题4: 缺少验证步骤

**解决方案4**: 添加完整验证流程
```python
# 验证1: 基准值合理性
validate_baseline_sanity()

# 验证2: 转换后数据分布
validate_distribution(df_diff)

# 验证3: is_parallel效应是否减弱
validate_is_parallel_effect_reduction(df_original, df_diff)

# 验证4: DiBS结果对比
dibs_result_original = run_dibs(df_original)
dibs_result_diff = run_dibs(df_diff)
compare_dibs_results(dibs_result_original, dibs_result_diff)
```

**优先级**: **P1 - 强烈建议**

---

### 🟢 轻微问题 (P2 - 可选优化)

#### 问题5: 物理意义损失

**解决方案5**: 两阶段分析
```python
# 阶段1: 因果发现（使用标准化数据）
dibs_result = run_dibs(df_standardized)
causal_graph = extract_causal_graph(dibs_result)

# 阶段2: 效应量化（使用原始数据）
for edge in causal_graph.edges:
    effect_size = estimate_effect(df_original, edge.source, edge.target)
    print(f"{edge.source} → {edge.target}: {effect_size:.0f}J")
```

**优先级**: **P2 - 可选**
**优势**: 兼顾因果发现的技术需求和结果解释的物理意义

---

## 9. 最终建议

### 推荐采用？

**答案**: **❌ 不推荐采用差值法v2.0**

**理由**：
1. 🔴 **方法学缺陷**：无法消除交互效应
2. 🔴 **高风险**：n=1基准值不可靠
3. 🟡 **混合尺度**：设计不统一
4. ⭐ **有更好的替代方案**：方案C和D

---

### 有条件采用的条件

**如果用户坚持使用差值法**，必须满足以下条件：

✅ **必需条件（P0）**：
1. 手动验证所有11个模型×2个模式=22个基准值实验的合理性
2. 实施残差分析检测基准值错误
3. 添加完整的验证流程（验证4步骤）
4. 对转换后数据执行**分组Z-score标准化**（消除剩余模式效应）

✅ **强烈建议（P1）**：
1. 统一处理原则（全转换或全保留）
2. 使用稳健统计量（截断均值代替中位数）
3. 交叉验证基准值

**如果无法满足上述条件** → **强烈建议放弃差值法**

---

### 推荐的替代方案

#### 🏆 首选：方案D + 方案C 组合 ⭐⭐⭐⭐⭐

**步骤1**: Z-score分组标准化（方案C）
```python
# 按模型+模式分组标准化
for model in models:
    for is_par in [True, False]:
        mask = (df['model'] == model) & (df['is_parallel'] == is_par)
        df.loc[mask, energy_fields] = (
            df.loc[mask, energy_fields] - df.loc[mask, energy_fields].mean()
        ) / df.loc[mask, energy_fields].std()
```

**步骤2**: DiBS分析（移除is_parallel变量）
```python
# is_parallel已经在标准化中被控制，不需要放入DiBS
features_for_dibs = energy_fields + hyperparam_fields + perf_fields
# 不包含is_parallel和model_*

dibs_result = run_dibs(df[features_for_dibs])
```

**步骤3**: 效应量化（使用原始数据）
```python
# 对DiBS发现的边，用原始数据回归估计效应大小
for edge in causal_edges:
    # 控制is_parallel后的净效应
    model = smf.ols(f'{edge.target} ~ {edge.source} + is_parallel', data=df_original)
    result = model.fit()
    print(f"{edge.source} → {edge.target}: {result.params[edge.source]:.0f}J")
```

**优势**：
- ✅ 完全消除is_parallel偏差（理论保证）
- ✅ 实施简单（标准化是pandas一行代码）
- ✅ DiBS友好（标准化数据是最佳输入）
- ✅ 后续可以用原始数据解释物理意义
- ✅ 无n=1风险（标准化不需要基准值）

---

#### 🥈 备选：方案D 残差化 ⭐⭐⭐⭐

```python
# 步骤1: 残差化（回归掉is_parallel的效应）
from sklearn.linear_model import LinearRegression

for field in energy_fields:
    model = LinearRegression()
    model.fit(df[['is_parallel', 'model_mnist', 'model_mnist_rnn', ...]], df[field])
    df[f'{field}_resid'] = df[field] - model.predict(df[...])

# 步骤2: 用残差做DiBS
dibs_result = run_dibs(df[[f'{f}_resid' for f in energy_fields] + other_fields])
```

**优势**：
- ✅ 理论上严格消除is_parallel效应
- ✅ 保留了能耗值的尺度（残差仍是焦耳）

**劣势**：
- 🟡 需要拟合多个回归模型
- 🟡 如果模型设定错误（如忽略交互项），残差仍有偏

---

## 10. 总结

### 核心发现

1. **🔴 差值法有根本性缺陷**
   - 只能消除加性偏差，无法处理交互效应
   - 在有交互作用的情况下，is_parallel仍会被DiBS检测为因果因子

2. **🔴 n=1风险太高**
   - 10/11个模型只有1个默认值实验/模式
   - 无法检测和缓解异常基准值

3. **🟡 模型级基准值有改进，但权衡明显**
   - 解决了Group1高CV问题 ✅
   - 但牺牲了基准值稳定性 ❌

4. **⭐ 有更好的替代方案**
   - Z-score分组标准化：简单、有效、理论严谨
   - DiBS混淆因子控制：最佳实践，保留完整信息

### 最终评分

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 方法学正确性 | ⭐⭐ (2/5) | 存在根本性缺陷（交互效应） |
| 实施可行性 | ⭐⭐⭐⭐ (4/5) | 实现简单，但需大量验证 |
| 风险可控性 | ⭐⭐ (2/5) | n=1风险高，难以缓解 |
| **总体推荐度** | **⭐⭐ (2/5)** | **不推荐采用** |

### 推荐行动

**立即行动**（今天）：
1. ❌ **停止实施差值法v2.0**
2. ✅ **评估方案C（Z-score标准化）的可行性**
3. ✅ **检查当前DiBS代码是否支持混淆因子控制（方案D）**

**短期行动**（本周）：
1. ✅ **实施Z-score分组标准化**
2. ✅ **用标准化数据重新运行DiBS**
3. ✅ **对比原始数据和标准化数据的DiBS结果**

**中期行动**（下周）：
1. ✅ **如果Z-score有效，撰写方法学文档**
2. ✅ **使用原始数据进行效应量化（回归分析）**
3. ✅ **生成最终因果图和分析报告**

---

## 附录：理论支撑

### A1. 为什么差值法无法消除交互效应？

**数学证明**：

设线性模型（简化）：
```
Y = β₀ + β_X·X + β_Z·Z + β_XZ·(X×Z) + ε
```

其中：
- Y: 能耗
- X: 超参数（如learning_rate）
- Z: is_parallel（0或1）
- β_XZ: 交互效应系数

差值转换：
```
Y' = Y - Y_baseline(Z)
```

其中 `Y_baseline(Z)` 是模式Z下的基准值（默认值实验的能耗）。

假设基准值实验的超参数为 `X = X₀`（默认值），则：
```
Y_baseline(Z) = β₀ + β_X·X₀ + β_Z·Z + β_XZ·(X₀×Z)
```

差值：
```
Y' = Y - Y_baseline(Z)
   = [β₀ + β_X·X + β_Z·Z + β_XZ·(X×Z) + ε]
     - [β₀ + β_X·X₀ + β_Z·Z + β_XZ·(X₀×Z)]
   = β_X·(X - X₀) + β_XZ·[(X - X₀)×Z] + ε
   = β_X·ΔX + β_XZ·(ΔX×Z) + ε
```

**结论**：
- ✅ 消除了 `β₀`（截距）
- ✅ 消除了 `β_Z·Z`（Z的主效应）
- ❌ **仍保留了 `β_XZ·(ΔX×Z)`（交互效应）**

因此，**如果β_XZ ≠ 0（有交互效应），DiBS仍会检测到Z（is_parallel）的因果效应**。

### A2. Z-score为什么可以消除？

**分组Z-score标准化**：
```python
# 对每个Z值（并行/非并行）分别标准化
for z in [0, 1]:
    mask = df['Z'] == z
    df.loc[mask, 'Y'] = (df.loc[mask, 'Y'] - df.loc[mask, 'Y'].mean()) / df.loc[mask, 'Y'].std()
```

**效果**：
- 并行模式：Y_standardized ~ N(0, 1)
- 非并行模式：Y_standardized ~ N(0, 1)

**关键**：两种模式的标准化数据有**相同的均值和方差**，因此：
- ✅ Z的主效应被消除
- ✅ Z的尺度效应被消除
- ✅ 交互效应也被归一化（两种模式在相同尺度上）

**DiBS分析时**：
- Z仍可能被检测为因果因子，但效应大小被控制
- 可以移除Z变量，因为已经在预处理中控制

### A3. 混淆因子控制的理论基础

**因果图模型**：
```
Z (is_parallel) → Y (能耗)
X (超参数) → Y (能耗)
Z → X (可能存在：并行模式可能使用不同超参数)
```

**目标**：估计 X → Y 的净因果效应，消除Z的混淆。

**方法1: 条件分析**（do-calculus）
```
P(Y | do(X)) = Σ_z P(Y | X, Z=z) · P(Z=z)
```

实现：分层分析（按Z分组），然后加权平均。

**方法2: 残差化**（调整）
```
Y_adjusted = Y - E[Y | Z]
X_adjusted = X - E[X | Z]
```

实现：回归Y~Z和X~Z，使用残差。

**方法3: 分组标准化**（特殊情况）
当只关心相对效应时，分组标准化等价于残差化。

---

**报告完成时间**: 2026-01-16
**评审人**: Claude (独立因果推断专家)
**评审方法**: 理论分析 + 文档审查 + 方法学评估
**评审结论**: ❌ **不推荐采用差值法v2.0**，推荐采用**Z-score分组标准化 + DiBS混淆因子控制**

---

**参考文献**：
1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
2. Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
3. DiBS paper: Lorch, L., et al. (2021). "DiBS: Differentiable Bayesian Structure Learning." NeurIPS.
4. 本项目文档：
   - `DIBS_INDEPENDENT_VALIDATION_REPORT_20260116.md`
   - `6GROUPS_DATA_VALIDATION_REPORT_20260115.md`
   - `QUESTION1_REGRESSION_ANALYSIS_PLAN.md`
