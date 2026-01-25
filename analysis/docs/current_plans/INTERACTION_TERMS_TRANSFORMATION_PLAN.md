# 交互项转换方案（原始数据+显式交互项）

**创建日期**: 2026-01-16
**状态**: 待验证
**方法**: 标准化 + 显式交互项 + 保留is_parallel

---

## 📋 目录

- [执行摘要](#执行摘要)
- [❓ 关键问题FAQ](#关键问题faq)（⭐ 新增）
- [研究动机](#研究动机)
- [方法学原理](#方法学原理)
- [转换规则](#转换规则)
- [实施细节](#实施细节)
- [验证步骤](#验证步骤)
- [DiBS分析指南](#dibs分析指南)
- [预期效果](#预期效果)
- [风险评估](#风险评估)

---

## 执行摘要

### 核心问题
**如何让DiBS探测调节效应**：is_parallel（并行/非并行模式）调节超参数对能耗的效应大小。

**研究假设**:
- 某超参数（如batch_size）在并行模式下对能耗的影响 ≠ 非并行模式
- 例如：并行时batch_size每增加1，能耗增加50,000J；非并行时只增加5,000J

### 解决方案
**显式交互项方法**: 手动创建交互项（超参数 × is_parallel）并加入DiBS输入数据。

**转换流程**:
```
1. 标准化能耗和超参数（消除尺度差异）
2. 创建交互项: hyperparameter × is_parallel
3. 输入DiBS: [超参数, is_parallel, 交互项, 能耗]
4. 解释因果图: 如果 交互项 → 能耗，说明存在调节效应
```

### 核心优势
1. ✅ **能探测调节效应**: DiBS可以学到"is_parallel如何改变超参数的效应"
2. ✅ **理论正确**: 交互项是因果推断中处理调节效应的标准方法
3. ✅ **信息完整**: 同时保留主效应和调节效应
4. ✅ **DiBS友好**: 标准化后的数据 + 显式变量最适合DiBS

---

## ❓ 关键问题FAQ

### Q1: 交互项方案是否还会造成测量混淆问题？

**简短答案**: 🟡 会，但与差值方案不同，且这种"混淆"是研究对象的一部分。

#### 详细解释

**测量混淆的本质**:
```
原始数据问题:
  - 并行实验: energy = f(hyperparams) + 并行开销 + 通信成本
  - 非并行实验: energy = f(hyperparams)

问题: is_parallel的直接效应（+90,000 J）非常强，可能遮蔽超参数效应
```

**差值方案的处理**:
```python
# 差值方案（移除is_parallel）
energy_diff = energy_parallel - energy_nonparallel

优点: ✅ 消除is_parallel的加性偏差
缺点: ❌ 假设超参数效应在两种模式下相同
     ❌ 无法探测调节效应
```

**交互项方案的处理**:
```python
# 交互项方案（保留is_parallel）
数据包含:
  1. hyperparam_batch_size (标准化)
  2. is_parallel (0/1)
  3. batch_size_x_parallel (交互项)
  4. energy (标准化)

优点: ✅ 保留所有信息
     ✅ 能探测调节效应
     ✅ 标准化减弱尺度差异
缺点: 🟡 is_parallel的直接效应仍存在（但这是研究对象！）
```

#### 两种方案对比

| 维度 | 差值方案 | 交互项方案 |
|------|---------|-----------|
| **假设** | 超参数效应在两种模式下**相同** | 超参数效应可能**不同** |
| **消除的混淆** | is_parallel的加性偏差 | 通过标准化减弱尺度差异 |
| **能探测调节效应** | ❌ 否 | ✅ 是 ⭐ |
| **适用场景** | 只关心超参数主效应 | 关心调节效应（您的情况） |
| **信息丢失** | 丢失is_parallel的所有信息 | 保留所有信息 |

#### 为什么交互项方案更合理？

根据您的研究假设：
> "我们认为超参数效应应该是被并行模式影响的"

这意味着：
- `batch_size=128` 在并行模式下的能耗效应 **≠** 非并行模式
- `learning_rate` 在两种模式下的效应 **可能不同**

**这正是调节效应的定义！**

因此：
- ✅ **交互项方案**能探测这种现象（符合研究目标）
- ❌ **差值方案**假设这种现象不存在（与研究假设冲突）

#### 结论

交互项方案的"混淆"不是问题，而是研究对象：
- is_parallel的**直接效应**（模式差异）: 我们想量化
- is_parallel的**调节效应**（改变超参数影响）: 我们想探测 ⭐

---

### Q2: 能否沿用昨天生成的6分组数据？

**简短答案**: ✅ 可以直接沿用，无需重新生成！

#### 数据验证

**昨天生成的数据位置**:
```
data/energy_research/6groups_final/
  ├── group1_examples.csv     (304条记录)
  ├── group2_vulberta.csv
  ├── group3_person_reid.csv
  ├── group4_bug_localization.csv
  ├── group5_mrt_oast.csv
  └── group6_resnet.csv
```

**数据包含的列**:
```python
# ✅ 能耗列（未标准化）
energy_cpu_pkg_joules, energy_cpu_ram_joules,
energy_cpu_total_joules, energy_gpu_total_joules, ...

# ✅ 超参数列（未标准化）
hyperparam_batch_size, hyperparam_learning_rate,
hyperparam_epochs, hyperparam_seed, ...

# ✅ 模式变量（True/False格式）
is_parallel

# ✅ 性能指标（未标准化）
perf_test_accuracy, perf_train_loss, ...

# ✅ 控制变量
timestamp, model_mnist_ff, model_mnist_rnn, ...
```

**is_parallel值分布**（以Group1为例）:
```
126条 False (非并行)
178条 True  (并行)
总计: 304条记录
```

#### 需要做的处理

只需添加以下步骤，无需重新生成原始数据：

```python
# 步骤1: 转换is_parallel格式
df['is_parallel'] = df['is_parallel'].map({True: 1, False: 0})

# 步骤2: 标准化超参数
scaler_hp = StandardScaler()
df[hyperparam_cols] = scaler_hp.fit_transform(df[hyperparam_cols])

# 步骤3: 标准化能耗
scaler_energy = StandardScaler()
df[energy_cols] = scaler_energy.fit_transform(df[energy_cols])

# 步骤4: 创建交互项
for hp in hyperparam_cols:
    df[f"{hp}_x_is_parallel"] = df[hp] * df['is_parallel']
```

**预计处理时间**: 10-20分钟（6组数据）

#### 结论

✅ **可以100%复用昨天的数据**，只需添加标准化和交互项转换

---

### Q3: is_parallel是True/False，交互项应该如何建立？如果取0，超参数是否恒为0？

**简短答案**: ✅ 非并行时交互项为0是正确设计，原始超参数列仍然有效！

#### 交互项的正确理解

```python
# 数据转换
is_parallel: True → 1, False → 0
batch_size: 标准化 → batch_size_std (z-score)

# 交互项创建
batch_size_x_parallel = batch_size_std × is_parallel

结果:
  非并行 (is_parallel=0): batch_size_x_parallel = 0
  并行 (is_parallel=1):   batch_size_x_parallel = batch_size_std
```

#### ⚠️ 澄清误解："超参数恒为0？"

**错误理解**:
> "如果is_parallel=0，交互项=0，那么超参数不就没有效应了吗？"

**正确理解**:
> 原始超参数列（`batch_size_std`）仍然存在，交互项只是**额外变量**！

#### DiBS输入的完整变量列表

```python
DiBS分析的变量（完整列表）:
  1. batch_size_std              ← 主效应（所有模式）⭐
  2. learning_rate_std           ← 主效应（所有模式）
  3. is_parallel                 ← 模式效应
  4. batch_size_x_parallel       ← 调节效应（仅并行）
  5. learning_rate_x_parallel    ← 调节效应（仅并行）
  6. energy_gpu_total            ← 因变量
```

#### 数据示例

| batch_size_std | is_parallel | batch_size_x_parallel | 解释 |
|----------------|-------------|-----------------------|------|
| 0.5 | 0 (非并行) | 0.0 | batch_size仍然有效（通过列1）！ |
| -0.3 | 0 (非并行) | 0.0 | batch_size=-0.3仍在列1中 |
| 0.8 | 1 (并行) | 0.8 | batch_size效应 = 列1 + 列3 |
| -1.2 | 1 (并行) | -1.2 | batch_size效应 = 列1 + 列3 |

**关键点**:
- 列1（`batch_size_std`）: 捕捉**所有模式**下batch_size的基础效应
- 列3（`batch_size_x_parallel`）: 捕捉**仅并行模式**下的额外效应

#### 因果效应的数学表达

DiBS可能学到的因果图：
```
batch_size_std ──→ energy (β₁)
is_parallel ──→ energy (β₂)
batch_size_x_parallel ──→ energy (β₃)
```

**回归方程**:
```
energy = β₁·batch_size_std + β₂·is_parallel + β₃·batch_size_x_parallel

非并行模式 (is_parallel=0):
  energy = β₁·batch_size_std
  → batch_size的效应是 β₁ ✅ （不是0！）

并行模式 (is_parallel=1):
  energy = β₁·batch_size_std + β₂ + β₃·batch_size_std
         = (β₁ + β₃)·batch_size_std + β₂
  → batch_size的效应是 (β₁ + β₃) ⭐ （调节效应）
```

**关键发现**:
- 如果 β₃ = 0 → 无调节效应（batch_size在两种模式下效应相同）
- 如果 β₃ > 0 → 并行模式**放大**了batch_size的效应
- 如果 β₃ < 0 → 并行模式**减弱**了batch_size的效应

#### 为什么非并行模式的交互项是0？

这是**特征工程的标准设计**：

**正确方法（我们的方案）**:
```python
batch_size_x_parallel = batch_size_std × is_parallel

优点:
  - 非并行: 交互项=0，不影响能耗（主效应由batch_size_std捕捉）✅
  - 并行: 交互项=batch_size_std，捕捉额外的调节效应 ✅
  - 物理意义清晰：只有并行时才有"调节效应"
```

**错误方法（中心化交互项）**:
```python
batch_size_x_parallel = (batch_size_std - 均值) × is_parallel

问题:
  - 破坏了is_parallel的0/1含义 ❌
  - 难以解释 ❌
```

#### 结论

✅ **非并行模式的交互项为0是正确且必要的设计**

原因：
1. 原始超参数列仍然存在，捕捉主效应
2. 交互项只捕捉"并行模式的额外效应"
3. 这正是"调节效应"的定义

---

## 研究动机

### 为什么需要探测调节效应？

**问题场景**:
```
观察1: 并行训练能耗高（均值100,000 J）
观察2: 非并行训练能耗低（均值10,000 J）

但这不是完整故事！我们还想知道：

观察3: batch_size增加时...
  - 并行模式: 能耗增加很多（+50,000 J）⭐
  - 非并行模式: 能耗增加很少（+5,000 J）⭐

这说明: is_parallel调节了batch_size的效应！
```

**研究价值**:
- ✅ 识别哪些超参数的效应受并行模式影响
- ✅ 量化调节效应的大小
- ✅ 指导实践：在不同模式下如何调整超参数

### 为什么DiBS默认探测不到？

**DiBS假设线性加性模型**:
```
Energy = β₁·batch_size + β₂·learning_rate + β₃·is_parallel + ε
         ↑ 独立效应       ↑ 独立效应           ↑ 独立效应

问题: 这个模型假设batch_size的效应（β₁）在并行/非并行下相同
     无法表达"is_parallel改变了batch_size的效应"
```

**解决方案：显式添加交互项**:
```
Energy = β₁·batch_size + β₂·learning_rate + β₃·is_parallel
       + β₄·(batch_size × is_parallel)  # 👈 交互项（调节效应）
       + β₅·(learning_rate × is_parallel)
       + ε

如果 β₄ ≠ 0: is_parallel调节batch_size的效应
如果 β₅ ≠ 0: is_parallel调节learning_rate的效应
```

---

## 方法学原理

### 1. 标准化（Standardization）

**目的**: 消除变量间的尺度差异，使DiBS更容易学习因果关系

**方法**: Z-score标准化
```
z = (x - μ) / σ

其中:
  μ = 变量均值
  σ = 变量标准差

转换后: 均值=0, 标准差=1
```

**应用范围**:
- ✅ 能耗变量（所有能量、功率字段）
- ✅ 超参数（batch_size, learning_rate, epochs等）
- ✅ 连续型控制变量
- ❌ is_parallel（保持0/1二值，不标准化）

**效果**:
```
标准化前:
  并行能耗: 50,000 ~ 200,000 J (巨大数值)
  非并行能耗: 5,000 ~ 20,000 J
  → 并行/非并行差异主导数据

标准化后:
  并行能耗: z-score -2 ~ +2 (标准正态分布)
  非并行能耗: z-score -2 ~ +2
  → 两组在相同尺度上，DiBS更容易比较
```

### 2. 交互项（Interaction Terms）

**定义**: 两个变量的乘积，表示一个变量如何调节另一个变量的效应

**创建方法**:
```python
interaction = hyperparameter × is_parallel

例如:
  batch_size_x_is_parallel = batch_size × is_parallel

  当 is_parallel=0 (非并行): 交互项 = 0
  当 is_parallel=1 (并行):   交互项 = batch_size的值
```

**解释**:
```
如果DiBS学到: batch_size_x_is_parallel → energy

含义:
  1. batch_size对能耗的效应在并行/非并行下不同
  2. 具体地，并行模式额外增加了batch_size的效应
  3. 效应大小 = 交互项在因果图中的权重
```

### 3. 因果图解释

**可能学到的因果结构**:

**场景1: 仅主效应，无调节**
```
batch_size ──→ energy
is_parallel ──→ energy

解释:
  - batch_size和is_parallel独立影响能耗
  - batch_size的效应在两种模式下相同
```

**场景2: 存在调节效应**
```
batch_size ──→ energy
is_parallel ──→ energy
(batch_size × is_parallel) ──→ energy  # 👈 调节效应！

解释:
  - batch_size在并行/非并行下效应不同
  - is_parallel不仅直接影响能耗，还调节batch_size的效应
```

**场景3: 仅调节效应，无主效应**
```
(batch_size × is_parallel) ──→ energy
is_parallel ──→ energy

解释:
  - batch_size本身不影响能耗（无主效应）
  - 但在并行模式下才有效应（纯调节效应）
  - 罕见，但可能发生
```

---

## 转换规则

### 转换字段分类

| 类别 | 字段 | 转换方式 | 原因 |
|------|------|---------|------|
| **能耗变量** | `energy_cpu_pkg_joules` | ✅ 标准化 | 因变量，需要统一尺度 |
|  | `energy_cpu_ram_joules` | ✅ 标准化 |  |
|  | `energy_cpu_total_joules` | ✅ 标准化 |  |
|  | `energy_gpu_total_joules` | ✅ 标准化 |  |
|  | `energy_gpu_avg_watts` | ✅ 标准化 |  |
|  | `energy_gpu_max_watts` | ✅ 标准化 | 改为标准化（与其他能耗一致）|
|  | `energy_gpu_min_watts` | ✅ 标准化 |  |
| **温度/利用率** | `energy_gpu_temp_avg_celsius` | ✅ 标准化 | 改为标准化（统一处理）|
|  | `energy_gpu_temp_max_celsius` | ✅ 标准化 |  |
|  | `energy_gpu_util_avg_percent` | ✅ 标准化 |  |
|  | `energy_gpu_util_max_percent` | ✅ 标准化 |  |
| **超参数** | `hyperparam_batch_size` | ✅ 标准化 | 自变量，需要统一尺度 |
|  | `hyperparam_learning_rate` | ✅ 标准化 |  |
|  | `hyperparam_epochs` | ✅ 标准化 |  |
|  | `hyperparam_seed` | ❌ 不处理 | 随机种子，非因果变量 |
|  | （其他超参数）| ✅ 标准化 | 按组不同 |
| **模式变量** | `is_parallel` | ❌ 保持0/1 | 二值变量，不标准化 |
| **性能指标** | `perf_test_accuracy` | ✅ 标准化 | 可能的中介变量 |
|  | `perf_train_loss` | ✅ 标准化 |  |
|  | （其他性能指标）| ✅ 标准化 | 按组不同 |

### 交互项创建规则

**为每个超参数创建交互项**:
```python
for hyperparam in [超参数列表]:
    interaction = hyperparam × is_parallel
    新列名 = f"{hyperparam}_x_is_parallel"
```

**示例**（Group1: examples）:
```
原始超参数:
  - hyperparam_batch_size
  - hyperparam_learning_rate
  - hyperparam_epochs

创建交互项:
  - hyperparam_batch_size_x_is_parallel
  - hyperparam_learning_rate_x_is_parallel
  - hyperparam_epochs_x_is_parallel
```

**不创建交互项的变量**:
- ❌ `hyperparam_seed`: 随机种子，非因果变量
- ❌ 性能指标: 它们是中介变量，不是自变量
- ❌ 温度/利用率: 它们是中间过程变量

---

## 实施细节

### 步骤1: 数据准备

**输入**:
- 6groups原始数据: `data/energy_research/6groups_*.csv`

**脚本**: `scripts/generate_interaction_terms_data.py`

**输出**:
- 6组交互项数据: `data/energy_research/6groups_*_interaction.csv`
- 标准化参数: `data/energy_research/standardization_params.json`

**核心逻辑**:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取数据
df = pd.read_csv('6groups_group1_examples.csv')

# 定义变量组
energy_cols = [
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules',
    'energy_cpu_total_joules', 'energy_gpu_total_joules',
    'energy_gpu_avg_watts', 'energy_gpu_max_watts', 'energy_gpu_min_watts',
    'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
    'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent'
]

hyperparam_cols = [
    'hyperparam_batch_size', 'hyperparam_learning_rate', 'hyperparam_epochs'
    # 不包括 hyperparam_seed
]

perf_cols = [
    'perf_test_accuracy'  # 按组不同
]

# 步骤1: 标准化
scaler_energy = StandardScaler()
scaler_hyperparam = StandardScaler()
scaler_perf = StandardScaler()

df[energy_cols] = scaler_energy.fit_transform(df[energy_cols])
df[hyperparam_cols] = scaler_hyperparam.fit_transform(df[hyperparam_cols])
df[perf_cols] = scaler_perf.fit_transform(df[perf_cols])

# 步骤2: 创建交互项
for hp in hyperparam_cols:
    interaction_col = f"{hp}_x_is_parallel"
    df[interaction_col] = df[hp] * df['is_parallel'].astype(float)

# 步骤3: 保存
df.to_csv('6groups_group1_examples_interaction.csv', index=False)

# 步骤4: 保存标准化参数（用于后续逆变换）
standardization_params = {
    'energy': {
        'mean': scaler_energy.mean_.tolist(),
        'std': scaler_energy.scale_.tolist(),
        'columns': energy_cols
    },
    'hyperparam': {
        'mean': scaler_hyperparam.mean_.tolist(),
        'std': scaler_hyperparam.scale_.tolist(),
        'columns': hyperparam_cols
    },
    'perf': {
        'mean': scaler_perf.mean_.tolist(),
        'std': scaler_perf.scale_.tolist(),
        'columns': perf_cols
    }
}

import json
with open('standardization_params.json', 'w') as f:
    json.dump(standardization_params, f, indent=2)
```

---

### 步骤2: DiBS输入数据结构

**变量顺序**（重要！）:
```
DiBS输入列顺序:
1. 超参数（主效应）
   - hyperparam_batch_size
   - hyperparam_learning_rate
   - hyperparam_epochs

2. is_parallel（主效应）
   - is_parallel (0/1)

3. 交互项（调节效应）
   - hyperparam_batch_size_x_is_parallel
   - hyperparam_learning_rate_x_is_parallel
   - hyperparam_epochs_x_is_parallel

4. 性能指标（中介变量）
   - perf_test_accuracy

5. 能耗变量（因变量）
   - energy_cpu_total_joules
   - energy_gpu_total_joules
   - energy_gpu_avg_watts
   - ... (其他能耗字段)
```

**示例数据**（Group1前3行）:
```csv
hyperparam_batch_size,hyperparam_learning_rate,hyperparam_epochs,is_parallel,hyperparam_batch_size_x_is_parallel,hyperparam_learning_rate_x_is_parallel,hyperparam_epochs_x_is_parallel,perf_test_accuracy,energy_cpu_total_joules,energy_gpu_total_joules,...
-0.52,1.34,-1.01,0,0.00,0.00,0.00,0.45,-0.31,-0.89,...
0.84,-0.67,0.32,1,0.84,-0.67,0.32,-1.23,1.56,2.01,...
-1.12,0.89,1.45,0,0.00,0.00,0.00,1.89,-0.78,-1.34,...
...
```

**说明**:
- 超参数和能耗都是z-score（均值0，标准差1）
- is_parallel保持0/1
- 当is_parallel=0时，所有交互项=0
- 当is_parallel=1时，交互项=超参数的标准化值

---

### 步骤3: DiBS分析

**运行DiBS**:
```python
import numpy as np
from dibs import DiBS  # 假设DiBS库可用

# 读取交互项数据
df = pd.read_csv('6groups_group1_examples_interaction.csv')

# 选择DiBS输入列
dibs_cols = (
    hyperparam_cols +
    ['is_parallel'] +
    [f"{hp}_x_is_parallel" for hp in hyperparam_cols] +
    perf_cols +
    energy_cols
)

X = df[dibs_cols].values
var_names = dibs_cols

# DiBS参数
dibs_params = {
    'n_particles': 20,
    'n_steps': 1000,
    'alpha_linear': 0.05,
    # ... 其他参数
}

# 运行DiBS
print("运行DiBS因果发现...")
result = DiBS(
    data=X,
    var_names=var_names,
    **dibs_params
)

# 获取因果图
causal_graph = result.sample_graph()  # 或 result.get_mle_graph()
```

---

### 步骤4: 结果解释

**解释因果图**:
```python
import networkx as nx

# 转换为NetworkX图
G = nx.DiGraph()
for i, parent in enumerate(var_names):
    for j, child in enumerate(var_names):
        if causal_graph[i, j] > 0.5:  # 边存在的概率 > 0.5
            G.add_edge(parent, child, weight=causal_graph[i, j])

# 分析主效应
print("\n=== 主效应分析 ===")
for energy_var in energy_cols[:4]:  # 主要能耗变量
    parents = list(G.predecessors(energy_var))
    print(f"\n{energy_var} 的因果父节点:")

    for parent in parents:
        if '_x_is_parallel' not in parent and parent != 'is_parallel':
            print(f"  ✅ {parent} → {energy_var} (主效应)")

# 分析调节效应（关键！）
print("\n=== 调节效应分析 ===")
for energy_var in energy_cols[:4]:
    parents = list(G.predecessors(energy_var))

    for parent in parents:
        if '_x_is_parallel' in parent:
            base_hyperparam = parent.replace('_x_is_parallel', '')
            print(f"\n  ⭐ 发现调节效应!")
            print(f"     {base_hyperparam} 对 {energy_var} 的效应")
            print(f"     在并行/非并行模式下不同")
            print(f"     调节效应强度: {G[parent][energy_var]['weight']:.3f}")

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=2, iterations=50)

# 节点分类着色
node_colors = []
for node in G.nodes():
    if 'energy' in node:
        node_colors.append('lightcoral')  # 红色：能耗
    elif '_x_is_parallel' in node:
        node_colors.append('lightgreen')  # 绿色：交互项
    elif 'is_parallel' in node:
        node_colors.append('yellow')  # 黄色：is_parallel
    elif 'hyperparam' in node:
        node_colors.append('lightblue')  # 蓝色：超参数
    else:
        node_colors.append('lightgray')  # 灰色：其他

nx.draw(G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=3000,
        font_size=8,
        arrows=True,
        arrowsize=20)

plt.title('Causal Graph with Moderation Effects', fontsize=16)
plt.tight_layout()
plt.savefig('causal_graph_interaction.png', dpi=300)
print("\n✅ 因果图已保存到 causal_graph_interaction.png")
```

---

## 验证步骤

### 验证1: 标准化正确性

**目的**: 确认所有变量标准化为均值0、标准差1

**方法**:
```python
df = pd.read_csv('6groups_group1_examples_interaction.csv')

print("标准化验证:")
for col in energy_cols + hyperparam_cols + perf_cols:
    mean = df[col].mean()
    std = df[col].std()
    print(f"{col}:")
    print(f"  均值: {mean:.6f} (应接近0)")
    print(f"  标准差: {std:.6f} (应接近1)")

    assert abs(mean) < 0.01, f"{col}均值不为0"
    assert abs(std - 1.0) < 0.01, f"{col}标准差不为1"
```

**预期结果**: 所有标准化字段均值≈0，标准差≈1

---

### 验证2: 交互项正确性

**目的**: 确认交互项计算正确

**方法**:
```python
# 检查交互项的计算
for hp in hyperparam_cols:
    interaction_col = f"{hp}_x_is_parallel"

    # 非并行数据: 交互项应为0
    nonparallel_interactions = df[df['is_parallel'] == 0][interaction_col]
    assert (nonparallel_interactions == 0).all(), f"{interaction_col}在非并行时应为0"

    # 并行数据: 交互项应等于超参数值
    parallel_data = df[df['is_parallel'] == 1]
    diff = parallel_data[interaction_col] - parallel_data[hp]
    assert (abs(diff) < 1e-10).all(), f"{interaction_col}在并行时应等于{hp}"

print("✅ 交互项计算正确")
```

**预期结果**:
- 非并行时，所有交互项=0
- 并行时，交互项=对应超参数的值

---

### 验证3: 分层回归验证

**目的**: 用传统回归分析验证调节效应是否存在

**方法**:
```python
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# 对每个能耗变量，检查batch_size的效应是否在两种模式下不同
for energy_var in ['energy_gpu_total_joules']:
    print(f"\n{energy_var}:")

    # 并行模式回归
    df_parallel = df[df['is_parallel'] == 1]
    X_par = df_parallel[['hyperparam_batch_size']].values
    y_par = df_parallel[energy_var].values
    reg_par = LinearRegression().fit(X_par, y_par)

    # 非并行模式回归
    df_nonparallel = df[df['is_parallel'] == 0]
    X_nonpar = df_nonparallel[['hyperparam_batch_size']].values
    y_nonpar = df_nonparallel[energy_var].values
    reg_nonpar = LinearRegression().fit(X_nonpar, y_nonpar)

    # 比较系数
    coef_par = reg_par.coef_[0]
    coef_nonpar = reg_nonpar.coef_[0]

    print(f"  并行模式: batch_size系数 = {coef_par:.4f}")
    print(f"  非并行模式: batch_size系数 = {coef_nonpar:.4f}")
    print(f"  差异: {abs(coef_par - coef_nonpar):.4f}")

    if abs(coef_par - coef_nonpar) > 0.1:
        print(f"  ⭐ 存在调节效应（系数差异显著）")
    else:
        print(f"  ⚠️  调节效应不明显（系数差异小）")
```

**预期结果**:
- 如果调节效应存在，两种模式下的系数应有明显差异
- 这为DiBS学到的交互项提供了独立验证

---

### 验证4: 数据完整性

**目的**: 确认转换后数据记录数不变、无意外缺失

**方法**:
```python
df_original = pd.read_csv('6groups_group1_examples.csv')
df_interaction = pd.read_csv('6groups_group1_examples_interaction.csv')

assert len(df_original) == len(df_interaction), "记录数改变"

# 检查原始列是否保留
original_cols = set(df_original.columns)
interaction_cols = set(df_interaction.columns)
assert original_cols.issubset(interaction_cols), "原始列丢失"

# 检查新增列
new_cols = interaction_cols - original_cols
expected_new_cols = {f"{hp}_x_is_parallel" for hp in hyperparam_cols}
assert new_cols == expected_new_cols, f"交互项列不匹配: {new_cols} vs {expected_new_cols}"

print("✅ 数据完整性验证通过")
```

---

## DiBS分析指南

### 因果图解读

**节点类型**:
```
蓝色: 超参数（自变量）
  - hyperparam_batch_size
  - hyperparam_learning_rate
  - hyperparam_epochs

黄色: 模式变量
  - is_parallel

绿色: 交互项（调节效应）
  - hyperparam_batch_size_x_is_parallel
  - hyperparam_learning_rate_x_is_parallel
  - hyperparam_epochs_x_is_parallel

灰色: 性能指标（中介变量）
  - perf_test_accuracy

红色: 能耗变量（因变量）
  - energy_cpu_total_joules
  - energy_gpu_total_joules
  - ...
```

**边的解释**:
```
A → C: A直接因果影响C

特殊情况:
  batch_size → energy: 主效应（batch_size总是影响能耗）

  (batch_size × is_parallel) → energy: 调节效应
    含义: batch_size的效应大小取决于is_parallel

  is_parallel → energy: 模式直接效应
    含义: 并行/非并行本身影响能耗基线
```

### 研究问题回答

**问题1: 哪些超参数影响能耗？**
```
查找: 超参数 → 能耗 的边

如果存在: batch_size → energy_gpu_total_joules
答案: batch_size影响GPU能耗（主效应）
```

**问题2: is_parallel是否调节超参数的效应？**（核心！）
```
查找: (超参数 × is_parallel) → 能耗 的边

如果存在: (batch_size × is_parallel) → energy_gpu_total_joules
答案: ⭐ is_parallel调节batch_size对GPU能耗的效应
      即：batch_size在并行/非并行下的效应不同
```

**问题3: 调节效应有多强？**
```
查看: 交互项 → 能耗 边的权重

权重 > 0.5: 强调节效应
权重 0.2-0.5: 中等调节效应
权重 < 0.2: 弱调节效应
```

---

## 预期效果

### 1. 能探测调节效应 ✅⭐⭐⭐⭐⭐

**场景**: batch_size在并行模式下效应更大

**DiBS可能学到**:
```
batch_size ──→ energy_gpu_total_joules (主效应)
is_parallel ──→ energy_gpu_total_joules (模式效应)
(batch_size × is_parallel) ──→ energy_gpu_total_joules (调节效应) ⭐

解释:
  1. batch_size本身影响GPU能耗
  2. 并行模式本身增加GPU能耗基线
  3. 并行模式放大了batch_size的效应（调节作用）
```

**实际含义**:
```
非并行模式下:
  batch_size从64→128, GPU能耗增加5,000 J

并行模式下:
  batch_size从64→128, GPU能耗增加50,000 J

调节效应: +45,000 J (并行模式放大了9倍效应)
```

---

### 2. 信息完整 ✅

**对比其他方案**:

| 方案 | 主效应 | 模式效应 | 调节效应 |
|------|--------|---------|---------|
| **原始数据（无交互项）** | ✅ | ✅ | ❌ |
| **差值数据（移除is_parallel）** | ✅ | ❌ | ❌ |
| **交互项方案（推荐）** | ✅ | ✅ | ✅ |

只有交互项方案能回答所有研究问题！

---

### 3. 标准化消除尺度差异 ✅

**问题**: 原始数据中is_parallel的直接效应过强（90,000 J差异）

**解决**: 标准化后
```
转换前:
  并行能耗: 100,000 J
  非并行能耗: 10,000 J
  → 差异90,000 J主导一切

转换后:
  并行能耗: z-score +1.5 (高于均值1.5个标准差)
  非并行能耗: z-score -0.5 (低于均值0.5个标准差)
  → 差异2个标准差单位，与超参数效应可比
```

DiBS现在可以公平地比较is_parallel效应和超参数效应！

---

## 风险评估

### 风险1: DiBS可能学到虚假交互项 ⚠️⚠️

**问题**: 如果数据中真实不存在调节效应，DiBS可能错误学到

**示例**:
```
真实情况: batch_size在两种模式下效应相同

但由于噪声/偶然性:
  DiBS错误学到: (batch_size × is_parallel) → energy
```

**缓解措施**:
1. ✅ **验证3（分层回归）**: 独立验证调节效应是否真实存在
2. ✅ **重复实验**: 多次运行DiBS，检查结果稳定性
3. ✅ **因果图后验概率**: 检查边的置信度（> 0.8才认为可靠）
4. ✅ **敏感性分析**: 对比有/无交互项的DiBS结果

**评估**: 🟡 中等风险，可通过验证缓解

---

### 风险2: 交互项增加模型复杂度 ⚠️

**问题**: 变量数量显著增加

**示例**:
```
原始: 3个超参数 + is_parallel + 4个能耗 = 8个变量

添加交互项后:
  3个超参数 + is_parallel + 3个交互项 + 4个能耗 = 11个变量

可能的边数: 11 × 11 = 121 条
```

**影响**:
- DiBS运行时间增加
- 可能过拟合（学到虚假因果关系）
- 因果图更复杂，难以解释

**缓解措施**:
1. ✅ 只为核心超参数创建交互项（不是所有超参数）
2. ✅ 增加DiBS正则化参数（惩罚过多的边）
3. ✅ 使用更强的先验（prefer sparse graphs）

**评估**: 🟢 轻微风险，可控

---

### 风险3: 标准化后物理意义减弱 ⚠️

**问题**: z-score不直观，难以向非技术人员解释

**示例**:
```
原始: "batch_size从64增加到128，GPU能耗增加50,000 J"
      → 清晰的物理含义

标准化后: "batch_size标准化值从-0.5增加到+0.5，GPU能耗z-score增加1.2"
          → 需要解释什么是z-score
```

**解决方案**:
1. ✅ **保存标准化参数**: 可以逆变换回原始尺度
2. ✅ **双层解释**:
   - 因果发现阶段：使用z-score（技术准确）
   - 结果报告阶段：转换回原始尺度（易于理解）
3. ✅ **可视化**: 用原始单位绘制效应图

**评估**: 🟢 轻微风险，可通过报告技巧解决

---

### 风险4: 假设线性交互 ⚠️

**问题**: 交互项 = 超参数 × is_parallel 假设线性关系

**如果真实关系是非线性**:
```
真实: batch_size在并行模式下的效应是指数增长
模型: 假设为线性交互

结果: 交互项可能捕捉不到完整的调节效应
```

**检测方法**:
- 绘制残差图，检查非线性模式
- 尝试多项式交互项（如 batch_size² × is_parallel）

**评估**: 🟢 轻微风险，大多数情况下线性假设合理

---

## 与其他方案对比

| 方案 | 探测调节效应 | 物理意义 | 实施复杂度 | DiBS适配 | 推荐度 |
|------|------------|---------|-----------|---------|--------|
| **交互项方案** | ✅⭐⭐⭐⭐⭐ | 🟡 需逆变换 | ⭐⭐⭐ 中 | ⭐⭐⭐⭐⭐ 完美 | ⭐⭐⭐⭐⭐ |
| 差值法v2.0 | ❌ | ⭐⭐⭐⭐ 清晰 | ⭐⭐⭐ 中 | ⭐⭐⭐ 一般 | ⭐⭐ |
| Z-score（移除is_parallel） | ❌ | ❌ 丢失 | ⭐⭐⭐⭐⭐ 最简 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 分层分析 | ✅ | ⭐⭐⭐⭐⭐ 最清晰 | ⭐⭐⭐⭐ 简单 | ⭐⭐⭐ 需两次 | ⭐⭐⭐⭐ |

**推荐**:
1. **首选**: 交互项方案（本方案）- 信息最完整
2. **备选**: 分层分析 - 最易解释，但需要足够样本量

---

## 实施路线图

### 阶段1: 数据准备（1天）

1. 实现 `scripts/generate_interaction_terms_data.py`
2. 生成6组交互项数据
3. 执行验证1-2（标准化和交互项正确性）

### 阶段2: 初步验证（0.5天）

1. 执行验证3（分层回归）
2. 确认数据中是否真实存在调节效应
3. 如果没有调节效应，考虑简化方案

### 阶段3: DiBS分析（1-2天）

1. 对1个组（如Group1）运行DiBS
2. 解释因果图，检查是否学到交互项
3. 可视化结果

### 阶段4: 全面分析（1周）

1. 对所有6组运行DiBS
2. 对比不同组的调节效应
3. 生成综合报告

---

## 相关文档

| 文档 | 用途 |
|------|------|
| `DIFFERENCE_VALUE_TRANSFORMATION_PLAN_V2.md` | 差值法方案（备选） |
| `RELATIVE_VALUE_TRANSFORMATION_PLAN.md` | 百分比法方案（已废弃） |
| `data/energy_research/6groups_*.csv` | 原始输入数据 |
| `data/energy_research/6groups_*_interaction.csv` | 交互项输出数据（待生成） |
| `data/energy_research/standardization_params.json` | 标准化参数（待生成） |

---

## 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|---------|
| v1.0 | 2026-01-16 | 初始版本：原始数据+显式交互项方案 |
| v1.1 | 2026-01-16 | 新增"关键问题FAQ"章节，澄清3个关键疑问 |

---

**维护者**: Claude
**最后更新**: 2026-01-16
**状态**: ✅ 已澄清关键问题，待用户确认
