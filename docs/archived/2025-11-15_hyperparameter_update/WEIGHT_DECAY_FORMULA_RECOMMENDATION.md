# Weight Decay 变异范围推荐

**分析时间**: 2025-11-15 16:45
**目标**: 为Weight Decay设计合理的变异范围公式

---

## 1. Weight Decay 特性分析

### 1.1 典型数值范围

Weight Decay在深度学习中的常见取值：

| 任务类型 | 典型范围 | 常用默认值 |
|---------|---------|-----------|
| 图像分类 (CNN) | 1e-5 ~ 1e-3 | **1e-4 (0.0001)** |
| 目标检测 | 1e-5 ~ 5e-4 | 1e-4 |
| NLP/序列模型 | 0 ~ 1e-2 | **0.01** 或 **0** |
| Person ReID | 1e-4 ~ 5e-4 | 5e-4 |

**关键特点**:
- ✅ 数值跨度大（3-5个数量级）
- ✅ 常用科学记数法（1e-4）
- ⚠️ **许多模型默认值为0**（不使用weight decay）

---

### 1.2 实测数据

**ResNet20 (CIFAR-10)**:
- 实验#10, #11, #12: 全部使用 `wd=0.0001` (固定)
- **未测试weight decay变化**

**DenseNet121 (Person ReID)**:
- 日志中未明确显示weight decay
- 推测：可能为0或默认值5e-4

**MRT-OAST (Clone Detection)**:
- 实验配置：`wd=0.0` (无weight decay)

---

## 2. Weight Decay 公式选项

### 选项1: 倍数公式（推荐用于 default > 0）✅

```json
{
  "weight_decay": {
    "type": "multiplier",
    "formula": "WD_mutated = WD_default × multiplier",
    "multipliers": [0.1, 0.5, 1.0, 2.0, 10.0],
    "range": "[0.1×, 10×]"
  }
}
```

**示例应用**:

| 模型 | Default WD | 下界 (0.1×) | 中值 | 上界 (10×) | 覆盖范围 |
|------|-----------|------------|------|-----------|---------|
| ResNet20 | 0.0001 | 0.00001 | 0.0001 | 0.001 | 1e-5 ~ 1e-3 ✅ |
| DenseNet121 | 0.0005* | 0.00005 | 0.0005 | 0.005 | 5e-5 ~ 5e-3 ✅ |

*假设值

**优点**:
- ✅ 适应不同默认值的数量级
- ✅ 0.1× ~ 10× 覆盖2个数量级
- ✅ 保持相对关系一致

**缺点**:
- ❌ **对default=0无法使用**（0 × 任何数 = 0）

---

### 选项2: 对数空间倍数（科学严谨）⭐ 推荐

```json
{
  "weight_decay": {
    "type": "log_multiplier",
    "formula": "WD_mutated = WD_default × 10^offset",
    "offsets": [-1, -0.5, 0, 0.5, 1],
    "range": "[10^-1, 10^1] = [0.1×, 10×]"
  }
}
```

**等价于选项1的倍数公式**，但更明确表达对数关系

**测试点**:
- 10^-1 = 0.1× (下界)
- 10^-0.5 ≈ 0.316×
- 10^0 = 1.0× (baseline)
- 10^0.5 ≈ 3.16×
- 10^1 = 10× (上界)

---

### 选项3: 绝对值范围（处理default=0）

```json
{
  "weight_decay": {
    "type": "absolute",
    "formula": "WD_mutated ∈ [0.0, 0.01]",
    "test_points": [0.0, 0.0001, 0.001, 0.01]
  }
}
```

**适用场景**:
- default = 0 的模型（如MRT-OAST）
- 探索引入weight decay的效果

**问题**:
- ❌ 违反"固定表达式"原则
- ⚠️ 对不同默认值的模型不统一

---

### 选项4: 混合策略（兼顾default=0和default>0）⭐⭐ 最推荐

```json
{
  "weight_decay": {
    "type": "adaptive",
    "formula": "if WD_default > 0: WD_mutated = WD_default × multiplier\n           else: WD_mutated ∈ [0.0, 0.0001, 0.001, 0.01]",
    "multipliers_if_nonzero": [0.1, 0.5, 1.0, 2.0, 10.0],
    "absolute_values_if_zero": [0.0, 0.0001, 0.001, 0.01]
  }
}
```

**示例应用**:

| 模型 | Default WD | 变异范围 | 方法 |
|------|-----------|---------|------|
| ResNet20 | 0.0001 | [0.00001, 0.00005, 0.0001, 0.0002, 0.001] | 倍数 |
| MRT-OAST | 0.0 | [0.0, 0.0001, 0.001, 0.01] | 绝对值 |
| DenseNet121 | 0.0005 | [0.00005, 0.00025, 0.0005, 0.001, 0.005] | 倍数 |

**优点**:
- ✅ 兼容default=0和default>0
- ✅ 对非零默认值保持相对关系
- ✅ 对零默认值探索引入weight decay

**缺点**:
- ⚠️ 需要条件判断（略复杂）

---

## 3. 最终推荐

### 3.1 主推荐：混合策略 ⭐⭐

```json
{
  "weight_decay_mutation": {
    "strategy": "adaptive",
    "rule": "使用倍数（default>0）或绝对值（default=0）",

    "case_1_nonzero_default": {
      "condition": "WD_default > 0",
      "formula": "WD_mutated = WD_default × multiplier",
      "multipliers": [0.1, 1.0, 10.0],
      "range": "[0.1×, 10×]"
    },

    "case_2_zero_default": {
      "condition": "WD_default = 0",
      "formula": "WD_mutated ∈ absolute_values",
      "absolute_values": [0.0, 0.0001, 0.001, 0.01],
      "rationale": "探索引入weight decay的效果"
    }
  }
}
```

**简化版（3个测试点）**:
- default > 0: `[0.1×, 1.0×, 10×]`
- default = 0: `[0.0, 0.001, 0.01]`

---

### 3.2 备选：仅倍数公式（简单但不完整）

如果不想处理default=0的情况：

```json
{
  "weight_decay": {
    "type": "multiplier",
    "formula": "WD_mutated = WD_default × multiplier",
    "multipliers": [0.1, 1.0, 10.0],
    "range": "[0.1×, 10×]",
    "note": "对default=0的模型，保持WD=0不变"
  }
}
```

**优点**: 简单
**缺点**: 无法测试default=0模型引入weight decay的效果

---

## 4. 与Epochs和Seed的协调

既然Epochs和Seed都要变异，需要考虑组合爆炸：

### 4.1 总实验数量估算

假设每个超参数的测试点：
- Epochs: 3个值 `[0.5×, 1.0×, 2.0×]`
- Learning Rate: 3个值 `[0.5×, 1.0×, 2.0×]`
- Weight Decay: 3个值 `[0.1×, 1.0×, 10×]` 或 `[0, 0.001, 0.01]`
- Dropout: 3个值 `[default-0.3, default, default+0.2]` 或 `[0.0, 0.35, 0.7]`
- Seed: 3个值 `[0, 42, 9999]`

**全组合**: 3^5 = **243个实验/模型** ❌ 不可行

### 4.2 推荐的采样策略

#### 策略A: 单因素分析（推荐）✅

每次只变化一个超参数，其他保持默认：

```
变异Epochs:  [0.5×, 1.0×, 2.0×] + 默认其他 = 3个实验
变异LR:      [0.5×, 1.0×, 2.0×] + 默认其他 = 3个实验
变异WD:      [0.1×, 1.0×, 10×] + 默认其他 = 3个实验
变异Dropout: [d-0.3, d, d+0.2] + 默认其他 = 3个实验
变异Seed:    [0, 42, 9999] + 默认其他 = 3个实验

总计：15个实验/模型
```

**优点**:
- ✅ 可以独立分析每个超参数的影响
- ✅ 实验数量可控

**缺点**:
- ⚠️ 无法捕获超参数交互作用

#### 策略B: 边界测试（当前boundary_test_v2的方法）

只测试极值：

```
Baseline: 所有默认值
LR下界: LR=0.5×, 其他默认
LR上界: LR=2×, 其他默认
WD下界: WD=0.1×, 其他默认
WD上界: WD=10×, 其他默认
Dropout下界: Dropout=default-0.3, 其他默认
Dropout上界: Dropout=default+0.2, 其他默认
Epochs下界: Epochs=0.5×, 其他默认
Epochs上界: Epochs=2×, 其他默认
Seed变化: Seed变化3次

总计：10个实验/模型
```

#### 策略C: Latin Hypercube Sampling（高级）

从5维空间中采样20-30个代表性点

**优点**: 覆盖全空间
**缺点**: 复杂，难以解释

---

## 5. Seed变异的特殊考虑

### 5.1 Seed的作用

Seed变异的目的：
1. **评估稳定性**: 性能波动有多大？
2. **统计显著性**: 超参数影响是否超过随机性？

### 5.2 建议的Seed策略

#### 选项A: 对每个配置重复3次（推荐）

```json
{
  "seed_strategy": "repetition",
  "seeds": [0, 42, 9999],
  "method": "每个超参数配置运行3次，取平均值",
  "example": "LR=0.5× 运行3次 (seed=0,42,9999)，报告均值和标准差"
}
```

**实验数量**: 基础配置数 × 3

#### 选项B: 单因素分析时固定Seed

```json
{
  "seed_strategy": "fixed_for_hyperparameter_analysis",
  "seed": 42,
  "note": "单因素分析时固定seed=42",
  "separate_seed_analysis": "单独做3次seed变化实验评估随机性"
}
```

**实验数量**: 基础配置数 + 3

---

## 6. 最终配置建议

### 完整的5个超参数变异范围

```json
{
  "hyperparameter_mutation_config": {
    "epochs": {
      "type": "multiplier",
      "formula": "Epochs_mutated = Epochs_default × multiplier",
      "range": "[0.5×, 2.0×]",
      "test_points": [0.5, 1.0, 2.0]
    },

    "learning_rate": {
      "type": "multiplier",
      "formula": "LR_mutated = LR_default × multiplier",
      "range": "[0.5×, 2.0×]",
      "test_points": [0.5, 1.0, 2.0]
    },

    "weight_decay": {
      "type": "adaptive_multiplier",
      "formula_if_nonzero": "WD_mutated = WD_default × multiplier",
      "formula_if_zero": "WD_mutated ∈ [0.0, 0.001, 0.01]",
      "multipliers": [0.1, 1.0, 10.0],
      "absolute_values_for_zero": [0.0, 0.001, 0.01],
      "range": "[0.1×, 10×] or [0, 0.01]"
    },

    "dropout": {
      "type": "offset",
      "formula": "Dropout_mutated = clip(Dropout_default + offset, 0, 1)",
      "range": "[-0.3, +0.2]",
      "test_points": [-0.3, 0.0, 0.2]
    },

    "seed": {
      "type": "discrete",
      "formula": "Seed ∈ {0, 42, 9999}",
      "values": [0, 42, 9999],
      "usage": "每个配置重复3次或单独分析"
    }
  },

  "sampling_strategy": "single_factor_analysis",
  "total_experiments_per_model": "15 (不含seed重复) 或 45 (含seed重复)"
}
```

---

## 7. 实验设计示例

### 以ResNet20为例（default: epochs=200, lr=0.1, wd=0.0001, dropout=0, seed=42）

#### 单因素分析（15个实验）:

| # | 变异参数 | Epochs | LR | WD | Dropout | Seed | 说明 |
|---|---------|--------|----|----|---------|------|------|
| 1 | Baseline | 200 | 0.1 | 0.0001 | 0 | 42 | 基准 |
| 2 | Epochs | **100** | 0.1 | 0.0001 | 0 | 42 | 0.5× |
| 3 | Epochs | **400** | 0.1 | 0.0001 | 0 | 42 | 2× |
| 4 | LR | 200 | **0.05** | 0.0001 | 0 | 42 | 0.5× |
| 5 | LR | 200 | **0.2** | 0.0001 | 0 | 42 | 2× |
| 6 | WD | 200 | 0.1 | **0.00001** | 0 | 42 | 0.1× |
| 7 | WD | 200 | 0.1 | **0.001** | 0 | 42 | 10× |
| 8 | Dropout | 200 | 0.1 | 0.0001 | **0** | 42 | d-0.3=0 |
| 9 | Dropout | 200 | 0.1 | 0.0001 | **0.2** | 42 | d+0.2 |
| 10 | Seed | 200 | 0.1 | 0.0001 | 0 | **0** | - |
| 11 | Seed | 200 | 0.1 | 0.0001 | 0 | **9999** | - |

**总计**: 11个实验（Baseline + 10个变异）

---

**报告版本**: 1.0
**生成时间**: 2025-11-15 16:45
**推荐**: Weight Decay使用 **混合策略（adaptive multiplier）**
