# 超参数变异策略实现文档

**日期**: 2025-11-10
**版本**: v2.0 - 高级分布策略
**状态**: ✅ 已实现并测试通过

---

## 概述

实现了基于分布的高级超参数变异策略，支持对数均匀分布、零值概率和标准均匀分布。

---

## 变异策略设计

### 参数分类与变异范围

| 参数 | 变异范围 | 分布方式 | 零值概率 | 设计原因 |
|------|---------|---------|---------|---------|
| **Epochs** | `[default×0.5, default×2.0]` | 对数均匀 | 0% | 直接影响能耗(线性)，避免欠拟合和过拟合 |
| **Learning Rate** | `[default×0.1, default×10.0]` | 对数均匀 ⭐ | 0% | 指数敏感性，影响收敛速度和能耗效率 |
| **Weight Decay** | `[0.0, default×100]` | 对数均匀 | 30% | 防过拟合，对数敏感，允许无正则化 |
| **Dropout** | `[0.0, 0.7]` | 均匀分布 | 0% | 线性影响，超过0.7严重阻碍信息流动 |
| **Seed** | `[0, 9999]` | 均匀整数 | 0% | 评估稳定性，不直接影响能耗 |

---

## 实现细节

### 1. 对数均匀分布 (Log-Uniform)

**用途**: Epochs, Learning Rate, Weight Decay (非零值)

**数学原理**:
```
log_min = log(min_val)
log_max = log(max_val)
log_value = uniform(log_min, log_max)
value = exp(log_value)
```

**特点**:
- 在对数空间中均匀采样
- 适合指数敏感的参数
- 自动倾向于较小值（符合大多数最优超参数的特性）

**实际效果** (测试结果):
- Learning Rate: 在 [0.001, 0.1] 范围内，log10值均匀分布在 [-3.0, -1.0]
- 30% 值落在 0.001-0.003, 31% 在 0.003-0.01, 19% 在 0.01-0.03, 20% 在 0.03-0.1
- Epochs: 61% 值 < 12, 39% 值 ≥ 12 (范围 [5, 20])

### 2. 零值概率 (Zero Probability)

**用途**: Weight Decay

**实现**:
```python
if zero_probability > 0 and random.random() < zero_probability:
    return 0.0
```

**特点**:
- 30% 概率返回 0.0 (无正则化)
- 70% 概率使用对数均匀分布
- 允许评估正则化的真实影响

**实际效果** (测试结果):
- 1000次采样: 304次零值 (30.4%)
- 非零值: 均值 0.001577, 中位数 0.000376
- 完美符合30%零值的设计目标

### 3. 标准均匀分布 (Uniform)

**用途**: Dropout, Seed

**实现**:
```python
# Float
value = random.uniform(min_val, max_val)

# Integer
value = random.randint(min_val, max_val)
```

**特点**:
- 线性空间中均匀采样
- 适合线性影响的参数

**实际效果** (测试结果):
- Dropout [0.0, 0.7]: 均值 0.356 (接近0.35), 各区间分布均匀
- Seed [0, 9999]: 均值 4811.6 (接近5000), 四分位数大致平衡

---

## 代码实现

### 修改文件: `mutation.py`

#### 1. 导入math模块
```python
import math
```

#### 2. 更新`mutate_hyperparameter`方法

```python
def mutate_hyperparameter(self, param_config: Dict, param_name: str = "") -> Any:
    """Mutate a single hyperparameter with advanced strategies

    Implements parameter-specific mutation strategies:
    - Epochs: Log-uniform distribution [default×0.5, default×2.0]
    - Learning Rate: Log-uniform distribution [default×0.1, default×10.0]
    - Weight Decay: 30% zero + 70% log-uniform [0.0, default×100]
    - Dropout: Uniform distribution [0.0, 0.7]
    - Seed: Uniform integer [0, 9999]
    """
    param_type = param_config["type"]
    param_range = param_config["range"]

    # Get distribution strategy from config
    distribution = param_config.get("distribution", "uniform")
    zero_probability = param_config.get("zero_probability", 0.0)

    # Handle zero probability
    if zero_probability > 0 and random.random() < zero_probability:
        return 0.0 if param_type == "float" else 0

    min_val, max_val = param_range[0], param_range[1]

    # Log-uniform distribution
    if distribution == "log_uniform":
        if min_val <= 0:
            raise ValueError(f"Log-uniform requires min_val > 0")

        log_min = math.log(min_val)
        log_max = math.log(max_val)
        log_value = random.uniform(log_min, log_max)
        value = math.exp(log_value)

        if param_type == "int":
            return max(min_val, min(max_val, int(round(value))))
        else:
            return max(min_val, min(max_val, value))

    # Standard uniform distribution
    elif distribution == "uniform":
        if param_type == "int":
            return random.randint(min_val, max_val)
        else:
            return random.uniform(min_val, max_val)
```

#### 3. 更新`generate_mutations`调用

```python
# Pass parameter name to mutate_hyperparameter
mutation[param] = self.mutate_hyperparameter(param_config, param)
```

---

## 配置文件格式

### 超参数配置示例

```json
{
  "supported_hyperparams": {
    "epochs": {
      "type": "int",
      "default": 10,
      "range": [5, 20],
      "flag": "--epochs",
      "distribution": "log_uniform"
    },
    "learning_rate": {
      "type": "float",
      "default": 0.01,
      "range": [0.001, 0.1],
      "flag": "--lr",
      "distribution": "log_uniform"
    },
    "weight_decay": {
      "type": "float",
      "default": 0.0001,
      "range": [0.00001, 0.01],
      "flag": "--weight-decay",
      "distribution": "log_uniform",
      "zero_probability": 0.3
    },
    "dropout": {
      "type": "float",
      "default": 0.5,
      "range": [0.0, 0.7],
      "flag": "--dropout",
      "distribution": "uniform"
    },
    "seed": {
      "type": "int",
      "default": 42,
      "range": [0, 9999],
      "flag": "--seed",
      "distribution": "uniform"
    }
  }
}
```

### 新增字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `distribution` | string | 否 | 分布类型: "log_uniform" 或 "uniform" (默认) |
| `zero_probability` | float | 否 | 零值概率 (0.0-1.0)，默认0.0 |

---

## 测试验证

### 测试文件: `test/test_mutation_strategies.py`

运行测试:
```bash
python3 test/test_mutation_strategies.py
```

### 测试结果摘要

| 测试项 | 状态 | 关键指标 |
|--------|------|---------|
| Log-Uniform (Epochs) | ✅ | 61% < 中位数, 符合对数分布特征 |
| Log-Uniform (Learning Rate) | ✅ | Log空间均匀分布, 均值-2.13 |
| Zero Probability (Weight Decay) | ✅ | 30.4% 零值, 误差<1% |
| Uniform (Dropout) | ✅ | 均值0.356, 接近理论值0.35 |
| Uniform (Seed) | ✅ | 均值4811.6, 接近理论值5000 |
| Mutation Uniqueness | ✅ | 20/20 变异全部唯一 |

### 完整测试输出

```
✅ ALL TESTS COMPLETED SUCCESSFULLY

Summary:
- Log-uniform distribution working for epochs and learning_rate
- Zero probability working for weight_decay (30% zeros)
- Uniform distribution working for dropout and seed
- Mutation uniqueness guaranteed

✨ Mutation strategies are ready for use!
```

---

## 使用示例

### 1. 命令行模式

```bash
# 变异epochs和learning_rate (自动使用配置中的分布策略)
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 \
                    -mt epochs,learning_rate -n 10

# 变异所有参数
python3 mutation.py -r VulBERTa -m mlp -mt all -n 20
```

### 2. 配置文件模式

创建实验配置 `settings/mutation_test.json`:

```json
{
  "experiment_name": "mutation_strategy_test",
  "description": "Test advanced mutation strategies",
  "mode": "mutation",
  "runs_per_config": 10,
  "experiments": [
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "mutate": ["epochs", "learning_rate", "weight_decay"]
    }
  ]
}
```

运行:
```bash
python3 mutation.py -ec settings/mutation_test.json
```

### 3. Python API

```python
from mutation import MutationRunner

runner = MutationRunner(random_seed=42)

# 生成10个变异
mutations = runner.generate_mutations(
    repo="pytorch_resnet_cifar10",
    model="resnet20",
    mutate_params=["epochs", "learning_rate", "weight_decay"],
    num_mutations=10
)

# 每个mutation都是唯一的超参数组合
for i, m in enumerate(mutations, 1):
    print(f"Mutation {i}: {m}")
```

---

## 变异策略优势

### 1. 更智能的采样

| 策略 | 传统方法 | 新方法 | 优势 |
|------|---------|--------|------|
| Learning Rate | 均匀分布 [0.001, 0.1] | 对数均匀 | 更多采样点在有效范围(0.001-0.01) |
| Weight Decay | 均匀分布 | 30%零值 + 对数均匀 | 显式评估无正则化效果 |
| Epochs | 均匀分布 | 对数均匀 | 倾向较少epoch(节能) |

### 2. 能耗-性能权衡

- **对数分布**倾向较小值：自然减少训练时间和能耗
- **零值概率**允许评估极端配置(无正则化)
- **均匀分布**确保充分探索dropout空间

### 3. 科学性和可重复性

- 基于概率论和统计学原理
- 支持随机种子设置
- 确保变异唯一性
- 所有策略经过测试验证

---

## 最佳实践

### 1. 选择合适的分布

| 参数特征 | 推荐分布 | 示例参数 |
|---------|---------|---------|
| 指数敏感 | log_uniform | learning_rate, weight_decay |
| 线性影响 | uniform | dropout, temperature |
| 计数/标识符 | uniform (int) | epochs, seed, batch_size |

### 2. 设置变异范围

```python
# 基于默认值的倍数
"range": [default * 0.5, default * 2.0]  # Epochs
"range": [default * 0.1, default * 10.0]  # Learning Rate

# 绝对范围
"range": [0.0, 0.7]  # Dropout (物理约束)
"range": [0, 9999]   # Seed (任意有效范围)
```

### 3. 使用零值概率

适用场景:
- Weight Decay: 评估无正则化
- Dropout: 评估无dropout
- 任何可以"关闭"的技术

建议值: 0.2-0.3 (20-30%)

---

## 后续扩展

### 计划支持的分布

1. **正态分布** (Normal): 围绕默认值采样
2. **Beta分布**: 更灵活的有界分布
3. **离散集合**: 从预定义值中选择
4. **条件分布**: 基于其他参数的依赖采样

### 配置文件增强

```json
{
  "learning_rate": {
    "distribution": "normal",
    "mean": 0.01,
    "std": 0.003,
    "range": [0.001, 0.1]
  },
  "optimizer": {
    "distribution": "discrete",
    "values": ["adam", "sgd", "rmsprop"]
  }
}
```

---

## 故障排除

### 问题1: Log-uniform报错 "min_val > 0"

**原因**: 对数函数要求输入 > 0

**解决**: 确保 range[0] > 0
```json
// ❌ 错误
"range": [0.0, 0.1]

// ✅ 正确
"range": [0.0001, 0.1]
```

### 问题2: 零值太多/太少

**原因**: 随机波动

**解决**: 增加采样数量 (统计规律在大样本下更稳定)

### 问题3: 变异不够多样

**检查**:
1. 是否范围太窄?
2. 是否精度太低 (int vs float)?
3. 是否参数数量太少?

**解决**: 扩大范围或增加变异参数

---

## 总结

### ✅ 已完成

1. ✅ 实现对数均匀分布 (log_uniform)
2. ✅ 实现零值概率机制 (zero_probability)
3. ✅ 保留标准均匀分布 (uniform)
4. ✅ 支持整数和浮点数
5. ✅ 确保变异唯一性
6. ✅ 完整测试套件
7. ✅ 文档和示例

### 📈 性能影响

- **代码复杂度**: 低 (仅增加~40行核心代码)
- **运行性能**: 无影响 (O(1)采样算法)
- **可维护性**: 高 (清晰的分布抽象)
- **可扩展性**: 强 (易于添加新分布)

### 🎯 推荐配置

对于大多数深度学习模型:

```json
{
  "epochs": {"distribution": "log_uniform", "range": "[default*0.5, default*2]"},
  "learning_rate": {"distribution": "log_uniform", "range": "[default*0.1, default*10]"},
  "weight_decay": {"distribution": "log_uniform", "zero_probability": 0.3},
  "dropout": {"distribution": "uniform", "range": "[0.0, 0.7]"},
  "seed": {"distribution": "uniform", "range": "[0, 9999]"}
}
```

---

**文档版本**: 2.0
**最后更新**: 2025-11-10
**测试状态**: 全部通过 ✅
