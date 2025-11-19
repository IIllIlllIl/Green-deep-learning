# 超参数变异范围和方式设计

**日期**: 2025-11-09
**作者**: Green
**项目**: Mutation-Based Training Energy Profiler

---

## 📋 概述

本文档为每个超参数提供科学合理的变异范围和变异方式建议，确保变异后的模型既能保持性能合理性，又能有效探索能耗-性能权衡空间。

**设计原则**:
1. ✅ **性能下界**: 避免性能过差导致能耗研究无意义
2. ✅ **探索充分**: 范围足够大以观察能耗变化趋势
3. ✅ **实用性**: 基于机器学习领域最佳实践
4. ✅ **可复现性**: 明确的变异分布和采样策略

---

## 🎯 通用超参数变异设计

### 1. Epochs（训练轮数）

#### 推荐变异范围

**表达式**: `[default * 0.5, default * 2.0]`

**变异方式**: **对数均匀分布** (Log-Uniform Distribution)

**具体实现**:
```python
import numpy as np

def mutate_epochs(default, min_epochs=5):
    """
    对数均匀分布采样epochs

    Args:
        default: 默认epochs值
        min_epochs: 最小epochs约束（避免过小）

    Returns:
        变异后的epochs值
    """
    lower = max(default * 0.5, min_epochs)
    upper = default * 2.0

    # 对数均匀分布采样
    log_lower = np.log(lower)
    log_upper = np.log(upper)
    log_value = np.random.uniform(log_lower, log_upper)

    return int(np.exp(log_value))
```

**分档建议**（离散变异）:
```python
# 对于需要离散采样的场景
def mutate_epochs_discrete(default):
    """
    离散分档采样

    返回: default * {0.5, 0.75, 1.0, 1.5, 2.0} 之一
    """
    factors = [0.5, 0.75, 1.0, 1.5, 2.0]
    factor = np.random.choice(factors)
    return int(default * factor)
```

#### 各模型具体建议

| 模型组 | 默认值 | 推荐范围 | 变异方式 | 说明 |
|--------|--------|---------|---------|------|
| **MRT-OAST** | 10 | [5, 20] | 离散: {5, 8, 10, 15, 20} | 小数据集，过多epoch易过拟合 |
| **pytorch_resnet** | 200 | [100, 300] | 对数: [100, 400] | 大数据集(CIFAR-10)，需充分训练 |
| **VulBERTa** | 10 | [5, 20] | 离散: {5, 8, 10, 15, 20} | 预训练模型微调，少epoch足够 |
| **Person_reID** | 60 | [30, 120] | 对数: [30, 120] | 检索任务，需较多epoch |
| **examples** | 10 | [5, 20] | 离散: {5, 8, 10, 15, 20} | MNIST简单任务 |

#### 原因和参考

**理论依据**:
1. **能耗影响**: epochs与能耗成线性关系，是最直接的能耗控制因素
2. **性能权衡**: 过少epoch导致欠拟合，过多导致过拟合和能耗浪费
3. **0.5-2.0倍范围**:
   - 下界0.5倍：保证基本收敛（参考：Deep Learning, Goodfellow et al., 2016）
   - 上界2.0倍：探索过拟合边界和能耗上限

**参考文献**:
- Bengio, Y. (2012). "Practical recommendations for gradient-based training of deep architectures"
- Smith, L. N. (2018). "A disciplined approach to neural network hyper-parameters"

---

### 2. Learning Rate（学习率）

#### 推荐变异范围

**表达式**: `[default * 0.1, default * 10.0]`

**变异方式**: **对数均匀分布** (Log-Uniform Distribution) ⭐ **强烈推荐**

**具体实现**:
```python
def mutate_learning_rate(default):
    """
    对数均匀分布采样学习率

    学习率在对数空间均匀分布，在原始空间呈现指数分布
    这是因为学习率的影响是指数级的

    Args:
        default: 默认学习率

    Returns:
        变异后的学习率
    """
    lower = default * 0.1
    upper = default * 10.0

    # 对数空间均匀采样
    log_lower = np.log10(lower)
    log_upper = np.log10(upper)
    log_value = np.random.uniform(log_lower, log_upper)

    return 10 ** log_value
```

**分档建议**（网格搜索）:
```python
def mutate_learning_rate_grid(default):
    """
    网格搜索式采样

    返回: default * {0.1, 0.3, 1.0, 3.0, 10.0} 之一
    """
    factors = [0.1, 0.3, 1.0, 3.0, 10.0]
    factor = np.random.choice(factors)
    return default * factor
```

#### 各模型具体建议

| 模型组 | 默认值 | 推荐范围 | 变异方式 | 说明 |
|--------|--------|---------|---------|------|
| **MRT-OAST** | 1e-4 | [1e-5, 1e-3] | 对数均匀 | Adam优化器，较小学习率 |
| **pytorch_resnet** | 0.1 | [0.01, 1.0] | 对数均匀 | SGD+Momentum，可用较大学习率 |
| **VulBERTa** | 3e-5 | [3e-6, 3e-4] | 对数均匀 | BERT微调，极小学习率 |
| **Person_reID** | 0.05 | [0.005, 0.5] | 对数均匀 | 特征提取任务 |
| **examples** | 0.01 | [0.001, 0.1] | 对数均匀 | MNIST简单任务 |
| **bug-localization** | 1e-5 (alpha) | [1e-6, 1e-4] | 对数均匀 | 正则化参数 |

#### 原因和参考

**理论依据**:
1. **指数敏感性**: 学习率对训练的影响是指数级的（相差10倍可能导致天壤之别）
2. **对数采样**: 在对数空间均匀采样确保小值和大值都有充分探索
3. **0.1-10倍范围**:
   - 下界0.1倍：避免收敛过慢导致欠拟合
   - 上界10倍：避免梯度爆炸和不收敛

**能耗影响**:
- 过小学习率 → 收敛慢 → 需要更多epoch → 能耗增加
- 过大学习率 → 不收敛/震荡 → 无效计算 → 能耗浪费
- 最优学习率 → 快速收敛 → 能耗最低

**参考文献**:
- Bengio, Y. (2012). "Practical recommendations for gradient-based training"
- Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks"
- You, Y., et al. (2017). "Large Batch Training of Convolutional Networks"

---

### 3. Weight Decay（权重衰减 / L2正则化）

#### 推荐变异范围

**表达式**: `[0.0, default * 100.0]`

**变异方式**: **对数均匀分布**（非零情况）

**具体实现**:
```python
def mutate_weight_decay(default):
    """
    权重衰减变异

    允许0值（无正则化）+ 对数均匀分布

    Args:
        default: 默认weight decay值

    Returns:
        变异后的weight decay值
    """
    # 30% 概率采样为0（无正则化）
    if np.random.random() < 0.3:
        return 0.0

    # 70% 概率在对数空间采样
    if default == 0.0:
        # 如果默认值为0，使用典型范围
        lower = 1e-6
        upper = 1e-2
    else:
        lower = default * 0.1
        upper = default * 100.0

    log_lower = np.log10(lower)
    log_upper = np.log10(upper)
    log_value = np.random.uniform(log_lower, log_upper)

    return 10 ** log_value
```

**分档建议**:
```python
def mutate_weight_decay_discrete(default):
    """
    离散采样

    返回: {0.0, default*0.1, default, default*10, default*100} 之一
    """
    if default == 0.0:
        # 默认无正则化时的候选值
        candidates = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    else:
        candidates = [0.0, default*0.1, default, default*10, default*100]

    return np.random.choice(candidates)
```

#### 各模型具体建议

| 模型组 | 默认值 | 推荐范围 | 变异方式 | 说明 |
|--------|--------|---------|---------|------|
| **MRT-OAST** | 0.0 | [0.0, 1e-2] | 混合: 30%零 + 70%对数 | 可能不需要正则化 |
| **pytorch_resnet** | 1e-4 | [0.0, 1e-2] | 混合: 20%零 + 80%对数 | CNN常用1e-4 |
| **VulBERTa** | 0.0 | [0.0, 1e-2] | 混合: 40%零 + 60%对数 | 预训练模型可能不需要 |

#### 原因和参考

**理论依据**:
1. **正则化作用**: 防止过拟合，提升泛化能力
2. **对数敏感性**: 类似学习率，影响是指数级的
3. **包含零值**: 允许探索"无正则化"的情况

**能耗影响**:
- 适度正则化 → 减少过拟合 → 可能需要更少epoch → 能耗降低
- 过强正则化 → 欠拟合 → 需要更多epoch/调整 → 能耗增加

**参考文献**:
- Krogh, A., & Hertz, J. A. (1992). "A simple weight decay can improve generalization"
- Zhang, C., et al. (2018). "Three mechanisms of weight decay regularization"

---

### 4. Dropout（丢弃率）

#### 推荐变异范围

**表达式**: `[0.0, 0.7]`

**变异方式**: **均匀分布**（线性空间）

**具体实现**:
```python
def mutate_dropout(default):
    """
    Dropout率变异

    在[0.0, 0.7]范围内均匀采样

    Args:
        default: 默认dropout率

    Returns:
        变异后的dropout率
    """
    # 在合理范围内均匀采样
    lower = 0.0
    upper = 0.7

    return np.random.uniform(lower, upper)
```

**分档建议**:
```python
def mutate_dropout_discrete(default):
    """
    离散采样常用dropout率

    返回: {0.0, 0.1, 0.2, 0.3, 0.5, 0.7} 之一
    """
    candidates = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    return np.random.choice(candidates)
```

#### 各模型具体建议

| 模型组 | 默认值 | 推荐范围 | 变异方式 | 说明 |
|--------|--------|---------|---------|------|
| **MRT-OAST** | 0.2 | [0.0, 0.6] | 均匀分布 | 全连接层dropout |
| **Person_reID** | 0.5 | [0.0, 0.7] | 均匀分布 | 特征提取，可用较大dropout |

#### 原因和参考

**理论依据**:
1. **上界0.7**: 超过0.7会严重影响信息流动（Srivastava et al., 2014）
2. **下界0.0**: 允许无dropout，探索是否必要
3. **均匀分布**: Dropout在线性空间的影响较为均匀

**能耗影响**:
- Dropout → 减少过拟合 → 可能更快收敛 → 能耗降低
- 但Dropout → 训练时额外计算 → 单epoch能耗略增
- 总体影响：以训练效率为主

**参考文献**:
- Srivastava, N., et al. (2014). "Dropout: A simple way to prevent neural networks from overfitting"
- Gal, Y., & Ghahramani, Z. (2016). "A theoretically grounded application of dropout"

---

### 5. Seed（随机种子）

#### 推荐变异范围

**表达式**: `[0, 9999]`

**变异方式**: **均匀分布**（整数）

**具体实现**:
```python
def mutate_seed():
    """
    随机种子变异

    在[0, 9999]范围内均匀采样

    Returns:
        随机生成的种子值
    """
    return np.random.randint(0, 10000)
```

**说明**: Seed不影响能耗，但影响性能方差，用于评估结果稳定性。

#### 原因和参考

**理论依据**:
1. **范围充分**: 10000个不同种子足够评估方差
2. **均匀采样**: 无偏地探索不同初始化

**实验设计**:
- 固定其他参数，变异seed → 评估性能方差
- 变异其他参数，固定seed → 评估能耗影响

**参考文献**:
- Henderson, P., et al. (2018). "Deep reinforcement learning that matters"

---

### 6. 特殊参数

#### 6.1 max_iter（最大迭代次数 - bug-localization）

**推荐范围**: `[default * 0.5, default * 2.0]`
**变异方式**: 对数均匀分布
**默认值**: 10000
**推荐**: [5000, 20000]

**原因**: 类似epochs，控制训练时长和能耗。

#### 6.2 kfold（交叉验证折数 - bug-localization）

**推荐范围**: `[2, 10]`
**变异方式**: 均匀整数分布
**默认值**: 10
**推荐**: {2, 5, 10}（离散）

**原因**:
- 影响训练时间（kfold倍）
- 过小(<3)：统计不可靠
- 过大(>10)：计算成本高，收益递减

#### 6.3 alpha（正则化参数 - bug-localization）

**推荐范围**: `[default * 0.1, default * 10.0]`
**变异方式**: 对数均匀分布
**默认值**: 1e-5
**推荐**: [1e-6, 1e-4]

**原因**: 本质是学习率/正则化参数，遵循对数规律。

---

## 🔬 变异策略建议

### 策略1: 单参数变异（推荐用于能耗研究）

**目的**: 隔离单个参数对能耗的影响

**方法**:
```python
# 固定其他参数，只变异epochs
experiments = [
    {"epochs": 5, "lr": 0.1, "seed": 42},
    {"epochs": 10, "lr": 0.1, "seed": 42},
    {"epochs": 20, "lr": 0.1, "seed": 42},
]
```

**适用**: 理解单个超参数-能耗关系

---

### 策略2: 配对变异

**目的**: 探索参数间交互作用

**方法**:
```python
# epochs和learning_rate配对变异
experiments = [
    {"epochs": 10, "lr": 0.01},   # 少epoch，小lr
    {"epochs": 10, "lr": 0.1},    # 少epoch，大lr
    {"epochs": 100, "lr": 0.01},  # 多epoch，小lr
    {"epochs": 100, "lr": 0.1},   # 多epoch，大lr
]
```

**适用**: 研究超参数交互效应

---

### 策略3: 随机全变异

**目的**: 探索全局空间，发现意外模式

**方法**:
```python
# 所有参数同时随机变异
for _ in range(n_experiments):
    experiment = {
        "epochs": mutate_epochs(default_epochs),
        "lr": mutate_learning_rate(default_lr),
        "weight_decay": mutate_weight_decay(default_wd),
        "seed": mutate_seed()
    }
```

**适用**: 大规模探索性研究

---

## 📊 实施优先级

### 高优先级（强烈推荐变异）

1. **Epochs** - 直接影响能耗（线性关系）
2. **Learning Rate** - 影响收敛速度和能耗效率

### 中优先级

3. **Weight Decay** - 影响泛化和训练效率
4. **Dropout** - 影响训练时计算量

### 低优先级

5. **Seed** - 主要用于评估方差，不直接影响能耗

---

## 🎯 模型特定建议

### ResNet (CIFAR-10)

**重点变异**: epochs, learning_rate
**次要变异**: weight_decay
**原因**: CNN训练时间长，epochs和lr影响最大

**推荐实验**:
```python
{
    "epochs": [100, 150, 200, 300],  # 离散
    "learning_rate": log_uniform(0.01, 1.0),
    "weight_decay": [0.0, 1e-4, 1e-3],  # 离散
}
```

---

### VulBERTa (预训练模型微调)

**重点变异**: learning_rate, epochs
**次要变异**: weight_decay
**原因**: 预训练模型对学习率极敏感

**推荐实验**:
```python
{
    "learning_rate": log_uniform(1e-6, 1e-3),  # 重点
    "epochs": [5, 10, 15, 20],
    "weight_decay": [0.0, 1e-4, 1e-3],
}
```

---

### Person ReID

**重点变异**: epochs, learning_rate, dropout
**次要变异**: 无
**原因**: 检索任务需要充分训练，dropout影响特征质量

**推荐实验**:
```python
{
    "epochs": log_uniform(30, 120),
    "learning_rate": log_uniform(0.005, 0.5),
    "dropout": uniform(0.0, 0.7),
}
```

---

## 📚 完整配置建议

### 配置文件格式（新增字段）

```json
{
  "models": {
    "pytorch_resnet_cifar10": {
      "supported_hyperparams": {
        "epochs": {
          "flag": "-e",
          "type": "int",
          "default": 200,
          "range": [100, 400],
          "mutation_strategy": {
            "distribution": "log_uniform",
            "min_factor": 0.5,
            "max_factor": 2.0,
            "discrete_values": [100, 150, 200, 300, 400],
            "description": "对数均匀分布或离散采样"
          }
        },
        "learning_rate": {
          "flag": "--lr",
          "type": "float",
          "default": 0.1,
          "range": [0.01, 1.0],
          "mutation_strategy": {
            "distribution": "log_uniform",
            "min_factor": 0.1,
            "max_factor": 10.0,
            "discrete_values": [0.01, 0.03, 0.1, 0.3, 1.0],
            "description": "对数空间均匀分布，学习率敏感"
          }
        }
      }
    }
  }
}
```

---

## 🔍 参考文献

### 综合性文献
1. Bengio, Y. (2012). "Practical recommendations for gradient-based training of deep architectures"
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" - Chapter 11: Practical Methodology

### 学习率
3. Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks"
4. You, Y., et al. (2017). "Large Batch Training of Convolutional Networks"

### 正则化
5. Srivastava, N., et al. (2014). "Dropout: A simple way to prevent neural networks from overfitting"
6. Krogh, A., & Hertz, J. A. (1992). "A simple weight decay can improve generalization"

### 实验设计
7. Henderson, P., et al. (2018). "Deep reinforcement learning that matters"
8. Liashchynskyi, P., & Liashchynskyi, P. (2019). "Grid search, random search, genetic algorithm"

---

**维护者**: Green
**项目**: Mutation-Based Training Energy Profiler
**最后更新**: 2025-11-09
