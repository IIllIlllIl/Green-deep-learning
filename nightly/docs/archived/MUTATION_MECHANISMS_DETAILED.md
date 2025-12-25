# 变异机制详细说明：Epochs、Weight Decay 和重复检查

**分析时间**: 2025-11-15 18:00
**代码文件**: `mutation/hyperparams.py`
**配置文件**: `mutation/models_config.json`

---

## 1. Epochs变异机制

### 1.1 当前配置状态

**所有模型的Epochs当前使用**: `log_uniform` (对数均匀分布)

```python
# models_config.json 中的配置
"epochs": {
    "distribution": "log_uniform",  # ← 当前是对数分布
    "range": [30, 90],
    "default": 60
}
```

### 1.2 对数分布 vs 线性分布的区别

#### 对数分布（当前）

```python
# hyperparams.py 行140-156
log_min = math.log(30)     # = 3.401
log_max = math.log(90)     # = 4.500
log_value = random.uniform(3.401, 4.500)  # 对数空间均匀
epochs = exp(log_value)    # 转回原始空间
epochs = round(epochs)     # 取整

# 采样分布特性
范围 [30, 90]:
- 30-45 (小值区域): 概率约 40%
- 45-65 (中值区域): 概率约 35%
- 65-90 (大值区域): 概率约 25%

→ 偏向较小值
```

**示例输出（对数分布）**:
```
运行1: epochs = 38
运行2: epochs = 52
运行3: epochs = 43
运行4: epochs = 67
运行5: epochs = 81
平均值 ≈ 52 (小于线性分布的60)
```

#### 线性分布（您要求的）

```python
# 应该改为
epochs = random.randint(30, 90)  # 整数均匀分布

# 采样分布特性
范围 [30, 90]:
- 30-50: 概率 33.3%
- 50-70: 概率 33.3%
- 70-90: 概率 33.3%

→ 每个值概率相等
```

**示例输出（线性分布）**:
```
运行1: epochs = 62
运行2: epochs = 45
运行3: epochs = 78
运行4: epochs = 33
运行5: epochs = 56
平均值 ≈ 60 (正好是范围中点)
```

### 1.3 两种分布的对比

| 特性 | 对数分布（当前） | 线性分布（需改为） |
|------|----------------|-----------------|
| **公式** | `exp(uniform(log(30), log(90)))` | `randint(30, 90)` |
| **小值概率** | 高（约40%） | 均等（33.3%） |
| **大值概率** | 低（约25%） | 均等（33.3%） |
| **平均值** | ≈52 (偏小) | ≈60 (居中) |
| **适用场景** | 指数敏感参数（如LR） | 线性参数（如epochs） |

### 1.4 为什么Epochs应该用线性分布？

```
Epochs的特性:
✅ 线性影响训练充分性
   - 30 epochs: 欠拟合风险
   - 60 epochs: 充分训练
   - 90 epochs: 可能过拟合

❌ 不是指数敏感的
   - epochs从30→60的影响 ≈ 60→90的影响
   - 不像LR那样0.01→0.1和0.1→1.0影响差异巨大

结论: 应该用线性分布，每个值概率相等
```

---

## 2. Weight Decay变异机制

### 2.1 当前配置状态

**当前配置**:
```json
{
  "weight_decay": {
    "distribution": "log_uniform",
    "zero_probability": 0.3,  // ← 30%概率为0
    "range": [0.00001, 0.01]
  }
}
```

### 2.2 当前变异流程（30%零概率）

```python
# hyperparams.py 行132-134
if zero_probability > 0 and random.random() < zero_probability:
    return 0.0  # 30%的情况直接返回0

# 否则（70%情况）对数均匀采样
log_min = log(0.00001) = -11.513
log_max = log(0.01) = -4.605
log_value = random.uniform(-11.513, -4.605)
wd = exp(log_value)
```

**实际采样分布**:
```
100次采样统计:
- wd = 0.0:          约30次 (零概率)
- wd ∈ (0.00001, 0.0001): 约23次 (小值，对数分布)
- wd ∈ (0.0001, 0.001):  约23次 (中值)
- wd ∈ (0.001, 0.01):    约24次 (大值)
```

### 2.3 100%对数分布（您要求的）

**修改后的行为**:
```python
# 移除零概率判断，直接对数采样
# zero_probability 设为 0 或删除

log_min = log(0.00001) = -11.513
log_max = log(0.01) = -4.605
log_value = random.uniform(-11.513, -4.605)
wd = exp(log_value)  # 始终从范围采样
```

**修改后采样分布**:
```
100次采样统计:
- wd = 0.0:          0次 (不再有零概率)
- wd ∈ (0.00001, 0.0001): 约33次 (小值，对数分布)
- wd ∈ (0.0001, 0.001):  约33次 (中值)
- wd ∈ (0.001, 0.01):    约34次 (大值)
```

### 2.4 对比表格

| 方面 | 30%零概率（当前） | 100%对数（需改为） |
|------|----------------|-----------------|
| **wd=0出现概率** | 30% | 0% |
| **小值(1e-5~1e-4)** | 23% | 33% |
| **中值(1e-4~1e-3)** | 23% | 33% |
| **大值(1e-3~0.01)** | 24% | 34% |
| **探索无正则化** | ✅ 能探索 | ❌ 不能探索 |
| **覆盖范围** | [0, 0.01] | [0.00001, 0.01] |

### 2.5 为什么要100%对数分布？

#### 选项A: 保留30%零概率

**优点**:
- ✅ 可以探索"无weight decay"的效果
- ✅ 符合实际应用（很多模型wd=0）
- ✅ 对default=0的模型友好

**缺点**:
- ⚠️ 30%的运行"浪费"在wd=0上
- ⚠️ 无法保证每次运行都测试非零值

#### 选项B: 100%对数分布（您的要求）

**优点**:
- ✅ 充分利用每次采样
- ✅ 覆盖整个对数范围
- ✅ 适合研究wd影响

**缺点**:
- ❌ 无法探索wd=0的情况
- ❌ 对default=0的模型需要特殊处理

**建议**:
```json
// 如果要100%对数，建议
{
  "weight_decay": {
    "distribution": "log_uniform",
    "zero_probability": 0,  // 设为0或删除此行
    "range": [0.00001, 0.01]
  }
}

// 如果要测试wd=0，可以
// 1. 手动添加baseline实验（wd=0）
// 2. 或者将0.00001降低到非常小的值（如1e-8）
```

---

## 3. 重复检查机制详解

### 3.1 重复检查的必要性

**问题场景**:
```python
# 请求生成10个mutation
generate_mutations(params, mutate_params=["learning_rate"], num_mutations=10)

# 随机采样可能生成重复值
mutation_1 = {"learning_rate": 0.053421}
mutation_2 = {"learning_rate": 0.053422}  # 非常接近!
mutation_3 = {"learning_rate": 0.053421}  # 完全重复!

问题：浪费计算资源
```

### 3.2 重复检查实现机制

#### Step 1: 归一化（Normalization）

**代码位置**: `hyperparams.py` 行42-67

```python
def _normalize_mutation_key(mutation: Dict[str, Any]) -> tuple:
    """将mutation转换为可哈希的唯一key"""

    normalized_items = []
    for param, value in mutation.items():
        # 浮点数归一化到固定精度
        if isinstance(value, float):
            normalized_value = f"{value:.6f}"  # 6位小数
        elif isinstance(value, int):
            normalized_value = str(int(value))
        else:
            normalized_value = str(value)

        normalized_items.append((param, normalized_value))

    # 排序确保确定性
    return tuple(sorted(normalized_items))
```

**示例**:
```python
# 原始mutation
mutation_1 = {"learning_rate": 0.053421234567}
mutation_2 = {"learning_rate": 0.053421789012}
mutation_3 = {"learning_rate": 0.053421234567}

# 归一化后的key
key_1 = (('learning_rate', '0.053421'),)  # 保留6位小数
key_2 = (('learning_rate', '0.053422'),)  # 不同
key_3 = (('learning_rate', '0.053421'),)  # 与key_1相同!

# 判断
key_1 == key_2  # False（不同）
key_1 == key_3  # True（重复！）
```

#### Step 2: 唯一性检查

**代码位置**: `hyperparams.py` 行209-233

```python
def generate_mutations(..., num_mutations=10):
    mutations = []
    seen_mutations = set()  # 集合，存储已见过的key
    attempts = 0

    while len(mutations) < num_mutations and attempts < 1000:
        attempts += 1

        # 生成新mutation
        mutation = {}
        for param in params_to_mutate:
            mutation[param] = mutate_hyperparameter(...)

        # 归一化为key
        mutation_key = _normalize_mutation_key(mutation)

        # 检查是否重复
        if mutation_key not in seen_mutations:
            seen_mutations.add(mutation_key)  # 记录
            mutations.append(mutation)         # 保存
            print(f"   Mutation {len(mutations)}: {mutation}")
        # else: 重复，丢弃，重新生成

    return mutations
```

**流程图**:
```
开始 (mutations=[], seen={})
  ↓
[len(mutations) < num_mutations?]
  ├── No → 返回mutations
  └── Yes ↓
    生成新mutation
    ↓
    计算 key = normalize(mutation)
    ↓
    [key in seen?]
    ├── Yes (重复) → 丢弃，重新生成
    └── No (新的) →
        seen.add(key)
        mutations.append(mutation)
        ↓
        返回开始
```

### 3.3 重复检查详细示例

```python
# 请求10个learning_rate变异
generate_mutations(params, mutate_params=["learning_rate"], num_mutations=10)

attempts = 1: lr=0.053421234 → key='0.053421' → 新的 → mutations=[1]
attempts = 2: lr=0.078923456 → key='0.078923' → 新的 → mutations=[2]
attempts = 3: lr=0.053421789 → key='0.053421' → 重复! → 丢弃
attempts = 4: lr=0.031234567 → key='0.031235' → 新的 → mutations=[3]
attempts = 5: lr=0.078923111 → key='0.078923' → 重复! → 丢弃
attempts = 6: lr=0.094567890 → key='0.094568' → 新的 → mutations=[4]
...
attempts = 25: lr=0.067890123 → key='0.067890' → 新的 → mutations=[10]

最终返回10个唯一的mutation
```

### 3.4 精度控制

**精度常量**: `FLOAT_PRECISION = 6` (6位小数)

```python
# hyperparams.py 行20
FLOAT_PRECISION = 6

# 归一化时使用
normalized_value = f"{value:.{FLOAT_PRECISION}f}"
# 等价于
normalized_value = f"{value:.6f}"
```

**精度影响**:
```python
# 6位小数精度
0.0534212345 → '0.053421' }
0.0534218765 → '0.053422' } 不同（第5位不同）
0.0534219999 → '0.053422' }

0.0534210001 → '0.053421' }
0.0534214999 → '0.053421' } 相同（前6位相同）

结论: 前6位小数相同的值被认为是重复
```

**为什么是6位？**
```
Learning Rate示例:
- 范围 [0.025, 0.1]
- 6位精度: 可区分 75,000 个不同值 (0.000001的倍数)
- 足够精确，又避免浮点数比较问题

Epochs示例:
- 范围 [30, 90]
- 整数: 可区分 61 个不同值
- 直接用int()转换，无精度问题
```

### 3.5 失败保护机制

**最大尝试次数**: `MAX_MUTATION_ATTEMPTS = 1000`

```python
# hyperparams.py 行21
MAX_MUTATION_ATTEMPTS = 1000

# 使用
while len(mutations) < num_mutations and attempts < 1000:
    attempts += 1
    ...

# 超出限制后
if len(mutations) < num_mutations:
    logger.warning(
        f"Could only generate {len(mutations)} unique mutations "
        f"after {attempts} attempts."
    )
    print(f"⚠️  Warning: Could only generate {len(mutations)} unique mutations")
```

**触发场景**:
```python
# 场景1: 范围太窄
params = {
    "learning_rate": {
        "range": [0.05, 0.051],  # 极窄范围
        "distribution": "log_uniform"
    }
}
generate_mutations(..., num_mutations=100)
# 结果: 只能生成约10个唯一值（6位精度下）

# 场景2: 请求过多
params = {
    "seed": {
        "range": [0, 10],  # 只有11个可能值
        "distribution": "uniform"
    }
}
generate_mutations(..., num_mutations=20)
# 结果: 最多生成11个唯一值
```

### 3.6 多参数重复检查

```python
# 多个参数同时变异
mutation_1 = {"learning_rate": 0.053421, "dropout": 0.234567}
mutation_2 = {"learning_rate": 0.053421, "dropout": 0.345678}
mutation_3 = {"learning_rate": 0.053421, "dropout": 0.234567}

# 归一化
key_1 = (('dropout', '0.234567'), ('learning_rate', '0.053421'))
key_2 = (('dropout', '0.345678'), ('learning_rate', '0.053421'))
key_3 = (('dropout', '0.234567'), ('learning_rate', '0.053421'))

# 注意: 参数会被排序（字母序）
# dropout在前，learning_rate在后

# 判断
key_1 == key_2  # False（dropout不同）
key_1 == key_3  # True（完全相同）
```

---

## 4. 完整流程示例

### 4.1 实际运行示例

```bash
python mutation.py -r Person_reID_baseline_pytorch \
    -m densenet121 \
    --mutate learning_rate,dropout \
    --runs 5
```

**内部执行**:
```python
Step 1: 读取配置
config = {
    "learning_rate": {
        "default": 0.05,
        "range": [0.025, 0.1],
        "distribution": "log_uniform"
    },
    "dropout": {
        "default": 0.5,
        "range": [0.0, 0.5],
        "distribution": "uniform"
    }
}

Step 2: 生成mutations
mutations = []
seen = set()
attempts = 0

尝试1:
  lr = 0.053421234, dropout = 0.234567890
  key = (('dropout', '0.234568'), ('learning_rate', '0.053421'))
  → 新的，保存 → mutations=[1]

尝试2:
  lr = 0.078923456, dropout = 0.345678901
  key = (('dropout', '0.345679'), ('learning_rate', '0.078923'))
  → 新的，保存 → mutations=[2]

尝试3:
  lr = 0.053421789, dropout = 0.234567123
  key = (('dropout', '0.234567'), ('learning_rate', '0.053422'))
  → 新的，保存 → mutations=[3]

尝试4:
  lr = 0.078923111, dropout = 0.345678999
  key = (('dropout', '0.345679'), ('learning_rate', '0.078923'))
  → 重复（与尝试2的key相同），丢弃

尝试5:
  lr = 0.094567890, dropout = 0.456789012
  key = (('dropout', '0.456789'), ('learning_rate', '0.094568'))
  → 新的，保存 → mutations=[4]

尝试6:
  lr = 0.031234567, dropout = 0.123456789
  key = (('dropout', '0.123457'), ('learning_rate', '0.031235'))
  → 新的，保存 → mutations=[5]

返回5个唯一的mutation
```

---

## 5. 配置修改建议

### 5.1 Epochs改为线性分布

**当前配置** (所有模型):
```json
{
  "epochs": {
    "distribution": "log_uniform"  // ← 需要改
  }
}
```

**修改为**:
```json
{
  "epochs": {
    "distribution": "uniform"  // ← 改为线性
  }
}
```

**影响**:
- 改前: 偏向小值（如更多38, 43, 52...）
- 改后: 均匀分布（如30-90各值概率相等）

### 5.2 Weight Decay移除零概率

**当前配置**:
```json
{
  "weight_decay": {
    "distribution": "log_uniform",
    "zero_probability": 0.3  // ← 需要改
  }
}
```

**修改为**:
```json
{
  "weight_decay": {
    "distribution": "log_uniform",
    "zero_probability": 0  // ← 改为0（或删除此行）
  }
}
```

**影响**:
- 改前: 30%返回0，70%对数采样
- 改后: 100%对数采样，无0值

---

## 6. 总结

### 6.1 当前机制

| 组件 | 当前状态 | 您的要求 | 是否需要修改 |
|------|---------|---------|------------|
| **Epochs分布** | log_uniform | uniform | ✅ 需要修改 |
| **Weight Decay零概率** | 0.3 (30%) | 0 (100%对数) | ✅ 需要修改 |
| **重复检查** | ✅ 已实现 | - | ✅ 已符合 |

### 6.2 重复检查机制总结

```
核心机制:
1. 归一化: 浮点数→6位小数字符串
2. 哈希: 转为tuple作为集合key
3. 去重: 集合membership检查
4. 保护: 最多尝试1000次

优点:
✅ 高效（集合O(1)查找）
✅ 精确（6位小数足够）
✅ 健壮（有失败保护）
✅ 通用（支持多参数）

限制:
⚠️ 精度固定6位（不可配置）
⚠️ 极窄范围可能生成不足
```

### 6.3 需要的修改

两处配置需要修改：

1. **所有Epochs**: `"distribution": "log_uniform"` → `"uniform"`
2. **所有Weight Decay**: `"zero_probability": 0.3` → `0` 或删除

---

**报告版本**: 1.0
**生成时间**: 2025-11-15 18:00
**关键发现**:
- Epochs当前是对数分布，需改为线性
- Weight Decay当前有30%零概率，需改为100%对数
- 重复检查机制已完善实现（6位精度+集合去重）
