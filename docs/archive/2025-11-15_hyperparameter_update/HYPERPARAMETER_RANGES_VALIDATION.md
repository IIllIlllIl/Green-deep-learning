# 超参数变异范围验证与更新

**分析时间**: 2025-11-15 16:30
**数据基础**: boundary_test_v2 实验结果（12个实验）
**目标**: 验证并更新5个超参数的变异范围

---

## 1. 提议的变异范围分析

### 用户提议的范围

```json
{
  "epochs": "[default × 0.5, default × 2.0]",
  "learning_rate": "[default × 0.5, default × 2.0]",
  "weight_decay": "[max(0, default - 0.3), min(1, default + 0.2)]",
  "dropout": "[0.0, 0.7]",
  "seed": "[0, 9999]"
}
```

---

## 2. 逐个参数验证

### 2.1 Epochs - ❓ 需要讨论

**提议范围**: `[default × 0.5, default × 2.0]`

**问题分析**:

1. **Epochs与能耗研究目标的冲突**
   - 能耗研究的核心目标：固定训练质量，测量能耗差异
   - 变异Epochs会直接改变训练时长，导致：
     - 能耗线性变化（训练时间×2 → 能耗×2）
     - 性能变化（epochs不足 → 欠拟合，epochs过多 → 过拟合）
   - **能耗差异无法归因于超参数本身**

2. **与Learning Rate的耦合效应**
   - Epochs × LR 共同决定收敛性
   - 例如：LR×2 + Epochs×0.5 可能等效于 LR×1 + Epochs×1
   - 难以独立分析各参数影响

3. **实际测试数据**
   - boundary_test_v2中，所有实验**固定了epochs**：
     - DenseNet121: 60 epochs（所有5个实验）
     - MRT-OAST: 10 epochs（所有4个实验）
     - ResNet20: 200 epochs（所有3个实验）
   - 未测试epochs变化的影响

**建议**:

#### 选项A: **不变异Epochs**（推荐）✅

```json
{
  "epochs": "fixed at default",
  "rationale": "保持训练充分性一致，能耗差异仅来自算法效率"
}
```

**优点**:
- ✅ 能耗可比性强（训练时长一致）
- ✅ 性能可比性强（都达到充分训练）
- ✅ 简化分析（减少变量）

**缺点**:
- ⚠️ 无法研究"早停"策略的能耗优化潜力

#### 选项B: 变异Epochs但设计特殊实验

```json
{
  "epochs": "[default × 0.5, default × 1.0, default × 2.0]",
  "special_analysis": "单独分组分析，不与其他参数混合"
}
```

**使用场景**: 研究"训练时长vs性能"权衡

---

### 2.2 Learning Rate - ✅ 基本正确，需微调

**提议范围**: `[default × 0.5, default × 2.0]`

**实测数据验证**:

| 模型 | LR 0.5× | LR 1.0× | LR 2.0× | LR 4.0× |
|------|---------|---------|---------|---------|
| DenseNet121 | 📊需验证 | 73.01% | 📊需验证 | 0.17% ❌崩溃 |
| MRT-OAST | 📊需验证 | 93.34% | 📊需验证 | **99.23% ✅最佳** |
| ResNet20 | 📊需验证 | 91.70% | 📊需验证 | 90.85% (-0.85%) |

**已测试的边界**:
- 下界0.25×: DenseNet -4.0%, MRT-OAST -10.24%, ResNet20 -0.84%
- 上界4×: DenseNet崩溃, MRT-OAST最佳, ResNet20 -0.85%

**分析**:
- ✅ **0.5×下界合理**: 比已测试的0.25×更保守，预期性能下降<3%
- ✅ **2×上界安全**: 避免DenseNet的4×崩溃
- ⚠️ **但牺牲了MRT-OAST的最佳性能区域**（99%@4× vs 预计96%@2×）

**建议**: ✅ **保持 `[0.5×, 2.0×]`**

---

### 2.3 Weight Decay - ❌ 公式错误

**提议范围**: `[max(0, default - 0.3), min(1, default + 0.2)]`

**问题**: **这是Dropout的公式，不是Weight Decay的公式！**

**Weight Decay特性**:
1. **数值范围通常很小**: 0.0001 ~ 0.01
2. **使用偏移量不合理**:
   ```
   示例：default = 0.0001
   提议公式: [max(0, 0.0001-0.3), min(1, 0.0001+0.2)]
            = [0, 0.2001]

   问题：0.2的weight decay过大，会严重损害性能
   ```

3. **应该使用倍数关系**:
   ```
   合理范围: [default × 0.1, default × 10]
   示例：default = 0.0001
         范围 = [0.00001, 0.001]
   ```

**实测数据**:
- ResNet20使用 wd=0.0001（所有3个实验固定）
- boundary_test_v2 **未测试weight decay变化**

**建议**: ✅ **修正为倍数公式**

```json
{
  "weight_decay": {
    "formula": "[default × 0.1, default × 10]",
    "special_case": "if default = 0, use [0.0, 0.01]"
  }
}
```

**理由**:
- 0.1× ~ 10× 覆盖2个数量级，足够探索
- 对于default=0的模型，提供小范围探索[0, 0.01]

---

### 2.4 Dropout - ⚠️ 需要修正公式类型

**提议范围**: `[0.0, 0.7]`

**实测数据验证**:

| 模型 | Default | Dropout 0.0 | Dropout 0.4 | Dropout 0.7 |
|------|---------|-------------|-------------|-------------|
| DenseNet121 | 0.5 | 75.42% (+2.4%) ✅ | 73.19% (+0.2%) ✅ | 📊需验证 |
| MRT-OAST | 0.2 | 99.15% (+5.81%) ✅ | - | 📊需验证 |

**分析**:

1. **绝对值范围 vs 偏移量范围**

   **提议**: `[0.0, 0.7]` - 绝对值范围

   **问题**: 违反"固定表达式"要求
   ```
   DenseNet default=0.5: 范围 [0.0, 0.7] → 可测试
   MRT-OAST default=0.2: 范围 [0.0, 0.7] → 可测试
   ResNet20 default=0.0: 范围 [0.0, 0.7] → 引入dropout（改变模型特性）
   ```

2. **之前推荐的偏移量公式**
   ```json
   {
     "dropout": "[max(0, default - 0.3), min(1, default + 0.2)]"
   }
   ```

   **测试**:
   - DenseNet (d=0.5): [0.2, 0.7] ✅ 包含已测试的0.0, 0.4
   - MRT-OAST (d=0.2): [0.0, 0.4] ✅ 包含已测试的0.0
   - ResNet (d=0.0): [0.0, 0.2] ✅ 保持无dropout特性

**建议**: ⚠️ **需要明确选择公式类型**

#### 选项A: 偏移量公式（符合"固定表达式"）

```json
{
  "dropout": {
    "formula": "[max(0, default - 0.3), min(1, default + 0.2)]",
    "rationale": "适应不同模型的默认值，避免改变模型特性"
  }
}
```

#### 选项B: 绝对值范围（简单但违反统一公式）

```json
{
  "dropout": {
    "formula": "[0.0, 0.7]",
    "note": "对default=0的模型会引入dropout",
    "rationale": "简单直接，0.7基于DenseNet default=0.5 + 0.2"
  }
}
```

**如果选择绝对值范围**，建议测试上界0.7:
- DenseNet已测试0.0(+2.4%), 0.4(+0.2%), 0.5(baseline)
- 0.7需要验证（预期性能下降<3%）

---

### 2.5 Seed - ⚠️ 需要明确目的

**提议范围**: `[0, 9999]`

**问题分析**:

1. **Seed的作用**
   - 控制随机数生成器（权重初始化、数据增强、dropout mask等）
   - 不同seed → 不同的训练轨迹 → 性能波动

2. **与能耗研究的关系**
   ```
   问题：不同seed的性能差异是什么？

   示例：
   Seed=0:    Acc=91.5%, Energy=1000J
   Seed=42:   Acc=91.8%, Energy=1005J
   Seed=9999: Acc=91.3%, Energy=998J

   如何解释？
   - Energy差异来自随机性，还是有实际意义？
   - 如果只是噪声，变异seed有何价值？
   ```

3. **实际测试情况**
   - boundary_test_v2 **未明确记录seed**
   - 可能所有实验使用相同seed（通常为默认值）

**建议**: ⚠️ **需要明确Seed的研究目标**

#### 选项A: **不变异Seed**（推荐用于单次实验）✅

```json
{
  "seed": "fixed at default (e.g., 42)",
  "rationale": "保证可重复性，能耗差异来自超参数而非随机性"
}
```

#### 选项B: 变异Seed用于统计分析（多次重复实验）

```json
{
  "seed": "[0, 1, 2, 3, 4]",
  "rationale": "每个超参数配置运行5次，计算均值和方差",
  "note": "需要5×实验数量，成本高"
}
```

**使用场景**:
- 研究超参数的稳定性
- 论文发表需要统计显著性

#### 选项C: 单次随机Seed

```json
{
  "seed": "random in [0, 9999]",
  "rationale": "避免lucky seed，但不做统计分析"
}
```

**问题**: 单次随机无法区分性能差异来自超参数还是运气

---

## 3. 推荐的最终变异范围

### 3.1 主推荐方案（保守型，适合能耗研究）

```json
{
  "hyperparameter_mutation_ranges": {
    "epochs": {
      "type": "fixed",
      "value": "default",
      "rationale": "保持训练充分性一致，能耗可比性强"
    },
    "learning_rate": {
      "type": "multiplier",
      "range": [0.5, 2.0],
      "formula": "LR_mutated = LR_default × multiplier",
      "rationale": "避免DenseNet崩溃，性能下降预期<3%"
    },
    "weight_decay": {
      "type": "multiplier",
      "range": [0.1, 10.0],
      "formula": "WD_mutated = WD_default × multiplier",
      "special_case": "if WD_default = 0, use [0.0, 0.001, 0.01]",
      "rationale": "覆盖2个数量级，适应小数值特性"
    },
    "dropout": {
      "type": "offset",
      "range": "[-0.3, +0.2]",
      "formula": "Dropout_mutated = clip(Dropout_default + offset, 0, 1)",
      "rationale": "适应不同默认值，已测试的0.0表现最佳"
    },
    "seed": {
      "type": "fixed",
      "value": 42,
      "rationale": "保证可重复性，避免随机性干扰"
    }
  }
}
```

**性能预期**: 所有模型性能下降<3%（除个别已知崩溃情况）

---

### 3.2 备选方案（如果必须使用绝对值范围）

```json
{
  "hyperparameter_mutation_ranges": {
    "epochs": {
      "type": "fixed",
      "value": "default"
    },
    "learning_rate": {
      "type": "multiplier",
      "range": [0.5, 2.0],
      "formula": "LR_mutated = LR_default × multiplier"
    },
    "weight_decay": {
      "type": "multiplier",
      "range": [0.1, 10.0],
      "formula": "WD_mutated = WD_default × multiplier",
      "special_case": "if WD_default = 0, use absolute range [0.0, 0.01]"
    },
    "dropout": {
      "type": "absolute",
      "range": [0.0, 0.7],
      "formula": "Dropout_mutated ∈ [0.0, 0.7]",
      "note": "对default=0的模型会引入dropout",
      "rationale": "简单直接，0.7基于测试数据外推"
    },
    "seed": {
      "type": "fixed",
      "value": 42
    }
  }
}
```

---

## 4. 对比表格

### 用户提议 vs 推荐方案对比

| 超参数 | 用户提议 | 主推荐 | 差异说明 |
|--------|---------|--------|---------|
| **Epochs** | `[0.5×, 2×]` | **Fixed** | ⚠️ 建议固定，避免能耗归因问题 |
| **Learning Rate** | `[0.5×, 2×]` | **`[0.5×, 2×]`** | ✅ 完全一致 |
| **Weight Decay** | `[default-0.3, default+0.2]` | **`[0.1×, 10×]`** | ❌ 用户公式错误（混淆了dropout） |
| **Dropout** | `[0.0, 0.7]` | **`[default-0.3, default+0.2]`** | ⚠️ 绝对值vs偏移量，各有优劣 |
| **Seed** | `[0, 9999]` | **Fixed at 42** | ⚠️ 建议固定，除非做统计分析 |

---

## 5. 需要验证的实验

为了完全验证推荐范围，需要补充以下实验：

| # | 模型 | 超参数 | 目的 |
|---|------|--------|------|
| 1 | DenseNet121 | LR=0.025 (0.5×) | 验证LR下界 |
| 2 | DenseNet121 | LR=0.1 (2×) | 验证LR上界 |
| 3 | DenseNet121 | Dropout=0.7 | 验证Dropout上界 |
| 4 | MRT-OAST | LR=0.00005 (0.5×) | 验证LR下界 |
| 5 | MRT-OAST | LR=0.0002 (2×) | 验证LR上界（预期96-97%） |
| 6 | DenseNet/MRT | WD变化 | 首次测试weight decay影响 |

**最小验证集**: 实验#1, #2, #5（3个实验，约2小时）

---

## 6. 最终建议

### 如果目标是"统一的固定公式"：

```
✅ Epochs: Fixed at default
✅ Learning Rate: [default × 0.5, default × 2.0]
✅ Weight Decay: [default × 0.1, default × 10] (或 [0, 0.01] if default=0)
✅ Dropout: [max(0, default - 0.3), min(1, default + 0.2)]
✅ Seed: Fixed at 42
```

### 如果可以接受"简单绝对值"：

```
✅ Epochs: Fixed at default
✅ Learning Rate: [default × 0.5, default × 2.0]
✅ Weight Decay: [default × 0.1, default × 10]
✅ Dropout: [0.0, 0.7]  (需验证0.7)
✅ Seed: Fixed at 42
```

---

**报告版本**: 1.0
**生成时间**: 2025-11-15 16:30
**建议状态**: 等待用户确认
**关键问题**:
1. ❓ Epochs是否固定？
2. ❓ Weight Decay使用倍数还是绝对值？
3. ❓ Dropout使用偏移量还是绝对值？
4. ❓ Seed是否固定？
