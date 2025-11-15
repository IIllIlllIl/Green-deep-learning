# hyperparams.py 变异机制分析与设计对比

**分析时间**: 2025-11-15 17:45
**代码文件**: `mutation/hyperparams.py`
**配置文件**: `mutation/models_config.json`

---

## 1. 当前变异机制详解

### 1.1 核心函数：`mutate_hyperparameter()`

```python
def mutate_hyperparameter(param_config, param_name="", random_seed=None, logger=None):
    """
    从配置中读取:
    - param_config["type"]: "int" 或 "float"
    - param_config["range"]: [min, max]
    - param_config["distribution"]: "uniform" 或 "log_uniform"
    - param_config["zero_probability"]: 0.0 ~ 1.0 (可选)

    返回: 在指定范围内的随机值
    """
```

### 1.2 变异流程图

```
开始
  ↓
读取配置 (type, range, distribution, zero_probability)
  ↓
[zero_probability > 0?]
  ├── Yes → [随机数 < zero_probability?]
  │          ├── Yes → 返回 0
  │          └── No → 继续
  └── No → 继续
  ↓
[distribution == "log_uniform"?]
  ├── Yes → log_min = log(min)
  │         log_max = log(max)
  │         log_value = random.uniform(log_min, log_max)
  │         value = exp(log_value)
  │         返回 value
  │
  └── No → [distribution == "uniform"?]
            ├── Yes → 返回 random.uniform(min, max)
            └── No → 报错
```

---

## 2. 各超参数的变异策略

### 2.1 Epochs

**配置示例** (DenseNet121):
```json
{
  "epochs": {
    "type": "int",
    "default": 60,
    "range": [30, 90],
    "distribution": "log_uniform"
  }
}
```

**变异方式**:
```python
# Step 1: 对数空间均匀采样
log_min = log(30) = 3.401
log_max = log(90) = 4.500
log_value = random.uniform(3.401, 4.500)  # 例如: 3.912

# Step 2: 转回原始空间
value = exp(3.912) = 50.0

# Step 3: 取整
epochs = round(50.0) = 50
```

**特点**:
- ✅ 对数均匀分布，偏向较小值
- ✅ 生成连续随机整数（30-90任意值）
- ⚠️ **每次运行生成不同值**（不可重复）

**示例输出**:
```
运行1: epochs = 43
运行2: epochs = 67
运行3: epochs = 38
运行4: epochs = 81
运行5: epochs = 52
```

---

### 2.2 Learning Rate

**配置示例** (DenseNet121):
```json
{
  "learning_rate": {
    "type": "float",
    "default": 0.05,
    "range": [0.025, 0.1],
    "distribution": "log_uniform"
  }
}
```

**变异方式**:
```python
# Step 1: 对数空间均匀采样
log_min = log(0.025) = -3.689
log_max = log(0.1) = -2.303
log_value = random.uniform(-3.689, -2.303)  # 例如: -2.996

# Step 2: 转回原始空间
lr = exp(-2.996) = 0.050085

# Step 3: 限制在范围内
lr = clip(0.050085, 0.025, 0.1) = 0.050085
```

**特点**:
- ✅ 对数均匀分布，适合指数敏感的参数
- ✅ 生成连续浮点数（0.025-0.1任意值）
- ⚠️ **每次运行生成不同值**（不可重复）

**示例输出**:
```
运行1: lr = 0.031335
运行2: lr = 0.075394
运行3: lr = 0.094041
运行4: lr = 0.043210
运行5: lr = 0.062987
```

**对数分布的特性**:
```
范围 [0.025, 0.1] 的对数分布:
- 0.025-0.040 (0.5×-0.8×): 概率约33%
- 0.040-0.063 (0.8×-1.25×): 概率约33%
- 0.063-0.1 (1.25×-2×): 概率约33%

对比线性分布:
- 0.025-0.05 (0.5×-1×): 概率33%
- 0.05-0.075 (1×-1.5×): 概率33%
- 0.075-0.1 (1.5×-2×): 概率33%

✅ 对数分布在小值区域采样更密集
```

---

### 2.3 Weight Decay

**配置示例** (ResNet20):
```json
{
  "weight_decay": {
    "type": "float",
    "default": 0.0001,
    "range": [0.00001, 0.01],
    "distribution": "log_uniform",
    "zero_probability": 0.3
  }
}
```

**变异方式**:
```python
# Step 1: 30%概率直接返回0
if random.random() < 0.3:
    return 0.0  # 30%的情况

# Step 2: 70%情况下，对数均匀采样
log_min = log(0.00001) = -11.513
log_max = log(0.01) = -4.605
log_value = random.uniform(-11.513, -4.605)  # 例如: -8.059
wd = exp(-8.059) = 0.000316
```

**特点**:
- ✅ **30%概率为0**（探索不使用weight decay）
- ✅ 70%概率从宽范围采样（0.00001-0.01，跨3个数量级）
- ⚠️ **每次运行生成不同值**（不可重复）

**示例输出**:
```
运行1: wd = 0.0       (零概率)
运行2: wd = 0.000316  (对数采样)
运行3: wd = 0.0       (零概率)
运行4: wd = 0.002145  (对数采样)
运行5: wd = 0.000089  (对数采样)
```

**零概率的实际效果**:
```
100次采样统计:
- wd = 0.0: 约30次
- wd ∈ (0.00001, 0.0001): 约23次 (小值区域)
- wd ∈ (0.0001, 0.001): 约23次 (中值区域)
- wd ∈ (0.001, 0.01): 约24次 (大值区域)
```

---

### 2.4 Dropout

**配置示例** (DenseNet121):
```json
{
  "dropout": {
    "type": "float",
    "default": 0.5,
    "range": [0.0, 0.5],
    "distribution": "uniform"
  }
}
```

**变异方式**:
```python
# 线性均匀分布
dropout = random.uniform(0.0, 0.5)  # 任意值
```

**特点**:
- ✅ 线性均匀分布（每个值概率相等）
- ✅ 生成连续浮点数（0.0-0.5任意值）
- ⚠️ **每次运行生成不同值**（不可重复）

**示例输出**:
```
运行1: dropout = 0.123
运行2: dropout = 0.456
运行3: dropout = 0.089
运行4: dropout = 0.321
运行5: dropout = 0.478
```

**线性分布的特性**:
```
范围 [0.0, 0.5] 的均匀分布:
- 0.0-0.167: 概率33%
- 0.167-0.333: 概率33%
- 0.333-0.5: 概率33%

✅ 每个区间概率相等
```

---

### 2.5 Seed

**配置示例**:
```json
{
  "seed": {
    "type": "int",
    "default": 42,
    "range": [0, 9999],
    "distribution": "uniform"
  }
}
```

**变异方式**:
```python
# 整数均匀分布
seed = random.randint(0, 9999)  # 0-9999任意整数
```

**特点**:
- ✅ 离散均匀分布
- ✅ 生成随机整数（0-9999任意值）
- ⚠️ **每次运行生成不同值**（不可重复）

**示例输出**:
```
运行1: seed = 3142
运行2: seed = 8765
运行3: seed = 0127
运行4: seed = 9012
运行5: seed = 4567
```

---

## 3. 与设计预期的对比

### 3.1 设计预期（基于您的要求）

```
Epochs:        [default × 0.5, default × 1.5] - 离散测试点
Learning Rate: [default × 0.5, default × 2.0] - 离散测试点
Weight Decay:  [0.00001, 0.01] - 离散测试点或倍数
Dropout:       [0.0, 0.5] - 离散测试点或偏移量
Seed:          [0, 9999] - 离散固定值
```

**预期方法**: 离散边界测试（固定几个点）

---

### 3.2 当前实现

```
Epochs:        连续对数均匀采样 [30, 90]
Learning Rate: 连续对数均匀采样 [0.025, 0.1]
Weight Decay:  30%零概率 + 70%对数均匀采样 [0.00001, 0.01]
Dropout:       连续线性均匀采样 [0.0, 0.5]
Seed:          连续整数均匀采样 [0, 9999]
```

**当前方法**: 连续随机采样（AutoML风格）

---

### 3.3 详细对比表

| 超参数 | 设计预期 | 当前实现 | 是否符合 | 差异 |
|--------|---------|---------|---------|------|
| **Epochs** | 固定点：[0.5×, 1.0×, 1.5×] | 连续采样：[0.5×~1.5×任意值] | ❌ 不符合 | 方法论完全不同 |
| **Learning Rate** | 固定点：[0.5×, 1.0×, 2.0×] | 连续采样：[0.5×~2×任意值] | ❌ 不符合 | 方法论完全不同 |
| **Weight Decay** | 固定点：[0.1×, 1×, 10×] 或 [0, 0.001, 0.01] | 30%零 + 70%连续采样 | ⚠️ 部分符合 | 范围符合，方法不同 |
| **Dropout** | 固定点：[0.0, 0.25, 0.5] | 连续采样：[0~0.5任意值] | ⚠️ 部分符合 | 范围符合，方法不同 |
| **Seed** | 固定值：[0, 42, 9999] | 连续整数：[0~9999任意值] | ❌ 不符合 | 意图不同 |

---

## 4. 核心差异分析

### 4.1 方法论差异

| 维度 | 设计预期 | 当前实现 |
|------|---------|---------|
| **采样方式** | 离散（固定几个点） | 连续（范围内任意值） |
| **可重复性** | ✅ 可重复（固定点） | ❌ 不可重复（随机） |
| **实验数量** | 可控（3-5个点） | 需指定（`--runs N`） |
| **分析难度** | ✅ 易于对比 | ⚠️ 难以对比 |
| **探索性** | ⚠️ 可能错过最优 | ✅ 覆盖全空间 |

### 4.2 具体示例对比

#### 示例：DenseNet121 Learning Rate变异

**设计预期**:
```python
# 固定3个测试点
lr_values = [0.025, 0.05, 0.1]  # [0.5×, 1×, 2×]

实验1: lr = 0.025 (0.5×)
实验2: lr = 0.05 (1.0×, baseline)
实验3: lr = 0.1 (2.0×)

结果可直接对比：
- 0.5× vs baseline
- 2× vs baseline
```

**当前实现**:
```python
# 连续随机采样
lr_values = [random_log_uniform(0.025, 0.1) for _ in range(3)]

实验1: lr = 0.031335 (0.63×)
实验2: lr = 0.075394 (1.51×)
实验3: lr = 0.094041 (1.88×)

结果难以对比：
- 没有baseline (1.0×)
- 没有测试边界 (0.5×, 2×)
- 每次运行值不同（不可重复）
```

---

## 5. 优缺点分析

### 5.1 当前实现（连续随机采样）

#### 优点 ✅

1. **探索性强**
   - 覆盖整个参数空间
   - 可能发现意外的最优点
   - 适合AutoML和超参数优化

2. **实现简单**
   - 代码简洁（100行）
   - 逻辑清晰
   - 易于扩展新参数

3. **分布合理**
   - Learning Rate使用对数分布（符合指数敏感性）
   - Weight Decay的零概率机制（探索无正则化）
   - Dropout使用线性分布（符合概率特性）

#### 缺点 ❌

1. **不可重复**
   ```python
   # 同样的命令，每次结果不同
   python mutation.py -r Person_reID -m densenet121 --mutate learning_rate --runs 3

   第1次运行: [0.031, 0.075, 0.094]
   第2次运行: [0.043, 0.062, 0.089]  # 完全不同！
   ```

2. **难以系统分析**
   - 无法直接对比baseline vs 边界
   - 性能差异可能来自随机性而非超参数
   - 不适合能耗研究（需要可比性）

3. **可能错过关键点**
   ```python
   # 生成了5个随机值，但都没测试baseline
   lr_values = [0.031, 0.043, 0.062, 0.075, 0.089]
   # 缺失: 0.05 (baseline, 1.0×)
   # 缺失: 0.025 (下界, 0.5×)
   # 缺失: 0.1 (上界, 2.0×)
   ```

4. **浪费计算资源**
   - 随机采样可能集中在某个区域
   - 边界（最重要的点）可能未充分测试

---

### 5.2 设计预期（离散边界测试）

#### 优点 ✅

1. **可重复**
   ```python
   # 每次运行结果相同
   lr_values = [0.025, 0.05, 0.1]  # 固定
   ```

2. **易于分析**
   ```
   Baseline:  mAP = 73.01% (lr=0.05)
   LR 0.5×:   mAP = 71.5% (-1.5%)  ← 直接对比
   LR 2×:     mAP = 71.8% (-1.2%)  ← 直接对比
   ```

3. **高效验证**
   - 直接测试关键边界
   - 适合安全范围验证
   - 适合能耗研究（固定训练时长）

4. **符合实验设计原则**
   - 单因素分析
   - 控制变量
   - 结果可重复

#### 缺点 ❌

1. **探索性弱**
   - 可能错过最优点（如果最优在0.7×）
   - 不适合黑盒优化

2. **需要先验知识**
   - 需要知道合理的边界
   - 需要baseline验证

---

## 6. 是否符合设计预期？

### 6.1 总体评估

**答案**: ❌ **不符合**

当前实现是**连续随机采样**（AutoML风格），而设计预期是**离散边界测试**（实验设计风格）。

### 6.2 具体不符合的方面

| 方面 | 设计预期 | 当前实现 | 符合度 |
|------|---------|---------|--------|
| **采样方式** | 离散固定点 | 连续随机值 | ❌ 0% |
| **可重复性** | 可重复 | 不可重复 | ❌ 0% |
| **测试点选择** | 边界+baseline | 随机分布 | ❌ 0% |
| **实验数量** | 3-5个固定点 | N个随机点 | ⚠️ 50% |
| **分析方式** | 对比分析 | 探索性分析 | ❌ 0% |

### 6.3 范围符合度

虽然方法不符合，但**范围本身是符合的**（已修改配置）：

| 超参数 | 设计范围 | 当前配置范围 | 符合度 |
|--------|---------|------------|--------|
| Epochs | [0.5×, 1.5×] | [0.5×, 1.5×] | ✅ 100% |
| Learning Rate | [0.5×, 2×] | [0.5×, 2×] | ✅ 100% |
| Weight Decay | [0.00001, 0.01] | [0.00001, 0.01] | ✅ 100% |
| Dropout | [0.0, 0.5] | [0.0, 0.5] | ✅ 100% |
| Seed | [0, 9999] | [0, 9999] | ✅ 100% |

---

## 7. 推荐的修改方案

### 7.1 选项A: 添加离散模式（推荐）⭐

**修改策略**: 保留当前连续模式，添加新的离散模式

```python
def mutate_hyperparameter(param_config, param_name="", mode="continuous", logger=None):
    """
    新增 mode 参数:
    - mode="continuous": 当前的连续随机采样
    - mode="discrete": 新的离散固定点采样
    """

    if mode == "discrete":
        # 新增逻辑
        return discrete_mutation(param_config, param_name)
    else:
        # 保留原有逻辑
        return continuous_mutation(param_config, param_name)

def discrete_mutation(param_config, param_name):
    """离散边界测试"""
    default = param_config.get("default")

    if param_name == "learning_rate":
        # 固定测试点: [0.5×, 1×, 2×]
        return random.choice([default * 0.5, default, default * 2.0])

    elif param_name == "epochs":
        # 固定测试点: [0.5×, 1×, 1.5×]
        return random.choice([
            int(default * 0.5),
            default,
            int(default * 1.5)
        ])

    elif param_name == "weight_decay":
        if default > 0:
            return random.choice([default * 0.1, default, default * 10])
        else:
            return random.choice([0.0, 0.001, 0.01])

    elif param_name == "dropout":
        # 偏移量: [-0.3, 0, +0.2]
        return max(0, min(1, default + random.choice([-0.3, 0.0, 0.2])))

    elif param_name == "seed":
        return random.choice([0, 42, 9999])
```

**使用方式**:
```bash
# 连续模式（当前）
python mutation.py -r Person_reID -m densenet121 --mode continuous --runs 10

# 离散模式（新增）
python mutation.py -r Person_reID -m densenet121 --mode discrete --runs 3
```

---

### 7.2 选项B: 完全替换为离散模式

**修改策略**: 直接修改`mutate_hyperparameter()`使用固定点

**优点**:
- 简单直接
- 符合设计预期

**缺点**:
- 失去探索性
- 破坏现有功能

---

### 7.3 选项C: 配置文件驱动

**修改策略**: 在配置文件中指定离散点

```json
{
  "learning_rate": {
    "type": "float",
    "default": 0.05,
    "range": [0.025, 0.1],
    "distribution": "discrete",
    "values": [0.025, 0.05, 0.1],
    "note": "Discrete boundary test points"
  }
}
```

```python
def mutate_hyperparameter(param_config, param_name="", logger=None):
    distribution = param_config.get("distribution", "uniform")

    if distribution == "discrete":
        # 从预定义值中随机选择
        values = param_config.get("values", [])
        return random.choice(values)

    elif distribution == "log_uniform":
        # 原有逻辑
        ...
```

---

## 8. 总结

### 8.1 当前状态

| 项目 | 状态 |
|------|------|
| **配置范围** | ✅ 已修改为统一标准 |
| **变异方法** | ❌ 连续采样，不符合离散测试预期 |
| **代码质量** | ✅ 实现清晰，文档完善 |
| **功能正确性** | ✅ 在连续采样框架下正常工作 |

### 8.2 核心问题

```
设计预期: 离散边界测试 (如 [0.5×, 1×, 2×])
当前实现: 连续随机采样 (如 [0.63×, 1.51×, 1.88×])

差异: 方法论完全不同
```

### 8.3 建议

1. **短期**: 保持当前实现，但注意：
   - 连续采样适合AutoML和探索
   - 不适合能耗研究（需要可重复性）
   - 可能需要大量运行才能覆盖边界

2. **中期**: 添加离散模式（选项A）
   - 兼容两种方法
   - 根据研究目标选择

3. **长期**: 根据实际使用反馈决定

---

**报告版本**: 1.0
**生成时间**: 2025-11-15 17:45
**关键结论**: 当前实现是连续随机采样，不符合离散边界测试的设计预期，建议添加离散模式
