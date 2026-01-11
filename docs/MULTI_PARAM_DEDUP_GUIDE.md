# 多超参数组合去重功能使用指南

**版本**: v1.0
**创建日期**: 2026-01-11
**作者**: Energy DL Team

---

## 功能概述

新的多超参数组合去重功能允许在实验配置中，只比较正在变异的超参数组合，而忽略历史数据中的其他参数。这解决了多参数变异实验中的去重问题。

### 问题场景

**之前的问题**:
- 历史数据: `{"learning_rate": 0.01, "batch_size": 32, "dropout": 0.5}`
- 当前变异: `{"learning_rate": 0.01, "batch_size": 32}` （只变异 lr 和 bs）
- 结果: **不会去重**（因为 normalized key 不同）

**现在的解决方案**:
- 使用 `filter_params=["learning_rate", "batch_size"]`
- 只比较这两个参数的组合
- 结果: **正确去重**✅

---

## 核心修改

### 1. `mutation/dedup.py` - `build_dedup_set()` 函数

新增 `filter_params` 参数：

```python
def build_dedup_set(
    mutations: List[Dict],
    filter_params: Optional[List[str]] = None,  # 新增参数
    logger: Optional[logging.Logger] = None
) -> Set[tuple]:
    """Build a set of normalized mutation keys for deduplication

    Args:
        mutations: List of mutation dictionaries
        filter_params: Optional list of parameter names to compare.
                      If provided, only these parameters will be used.
                      If None, all parameters will be used (原有行为).
        logger: Optional logger instance

    Returns:
        Set of normalized mutation keys
    """
```

### 2. `mutation/runner.py` - 自动应用

在 `run_from_experiment_config()` 中，系统会自动根据当前实验的 `mutate_params` 构建针对性的去重集：

```python
# 为每个实验配置构建针对性的去重集
if historical_mutations is not None:
    # 根据当前变异的参数过滤
    if "all" in mutate_params:
        filter_params = list(supported_params.keys())
    else:
        filter_params = mutate_params

    dedup_set = build_dedup_set(
        historical_mutations,
        filter_params=filter_params,  # 只比较正在变异的参数
        logger=self.logger
    )
```

---

## 使用示例

### 示例 1: 基本用法

```python
from mutation.dedup import build_dedup_set

# 历史数据（包含多个参数）
historical_mutations = [
    {"lr": 0.01, "batch_size": 32, "dropout": 0.5, "__mode__": "nonparallel"},
    {"lr": 0.001, "batch_size": 64, "dropout": 0.3, "__mode__": "nonparallel"},
]

# 只比较 lr 和 batch_size
dedup_set = build_dedup_set(
    historical_mutations,
    filter_params=["lr", "batch_size"]  # 只看这两个参数
)

# dedup_set 将只包含 (lr, batch_size) 的组合
# 即使历史数据中有 dropout，也会被忽略
```

### 示例 2: 实验配置文件中使用

```json
{
  "experiment_name": "multi_param_mutation",
  "use_deduplication": true,
  "historical_csvs": ["data/raw_data.csv"],
  "experiments": [
    {
      "repo": "examples",
      "model": "mnist",
      "mode": "mutation",
      "mutate": ["learning_rate", "batch_size"],
      "runs_per_config": 10
    }
  ]
}
```

**系统行为**:
1. 加载 `data/raw_data.csv` 中的历史数据（可能包含 lr, bs, dropout, epochs 等多个参数）
2. 自动构建去重集，**只比较 learning_rate 和 batch_size**
3. 生成新的变异时，只要 (lr, bs) 组合不重复即可，不管 dropout 等其他参数

---

## 用户需求示例

根据您的需求，以下是具体的行为：

```python
# 模型有 a, b, c 三个超参数，默认值都是 1

mutations = []

# 第1次变异: (1, 2, 3) - 不去重
mutations.append({"a": 1, "b": 2, "c": 3})
dedup_set = build_dedup_set(mutations, filter_params=["a", "b", "c"])
assert len(dedup_set) == 1  # ✓

# 第2次变异: (1, 2, 1) - 不去重（c不同）
mutations.append({"a": 1, "b": 2, "c": 1})
dedup_set = build_dedup_set(mutations, filter_params=["a", "b", "c"])
assert len(dedup_set) == 2  # ✓

# 第3次变异: (2, 2, 2) - 不去重
mutations.append({"a": 2, "b": 2, "c": 2})
dedup_set = build_dedup_set(mutations, filter_params=["a", "b", "c"])
assert len(dedup_set) == 3  # ✓

# 第4次变异: (2, 2, 3) - 不去重（没有完全相同的组合）
mutations.append({"a": 2, "b": 2, "c": 3})
dedup_set = build_dedup_set(mutations, filter_params=["a", "b", "c"])
assert len(dedup_set) == 4  # ✓

# 第5次变异: (1, 2, 1) - 去重！（与第2次完全相同）
test_mutation = {"a": 1, "b": 2, "c": 1}
key = _normalize_mutation_key(test_mutation, mode="nonparallel")
assert key in dedup_set  # ✓ 会被检测为重复
```

---

## 向后兼容性

**保证向后兼容**：
- 如果不提供 `filter_params` 参数（默认 `None`），行为与之前完全相同
- 现有代码无需修改即可继续工作
- 只有需要多参数组合去重时才需要使用新功能

```python
# 原有用法（不变）
dedup_set = build_dedup_set(mutations)  # 比较所有参数

# 新用法（可选）
dedup_set = build_dedup_set(mutations, filter_params=["lr", "bs"])  # 只比较指定参数
```

---

## 测试验证

运行测试验证功能：

```bash
# 运行多参数去重测试
python3 tests/test_multi_param_dedup.py

# 运行现有去重测试（验证兼容性）
python3 tests/unit/test_dedup_mechanism.py
```

**测试结果**:
- ✅ 所有新功能测试通过（6/6）
- ✅ 现有功能测试通过（3/3 核心测试）
- ✅ 向后兼容性验证通过

---

## 关键优势

1. **精确去重**: 只比较正在变异的参数组合，而不是所有参数
2. **向后兼容**: 不影响现有代码和实验配置
3. **自动应用**: 在实验配置文件中自动根据 `mutate` 字段应用
4. **灵活性**: 支持单参数、多参数、"all" 参数的去重

---

## 相关文件

### 修改的文件
- `mutation/dedup.py`: 核心去重函数
- `mutation/runner.py`: 实验运行器（自动应用去重）

### 新增文件
- `tests/test_multi_param_dedup.py`: 多参数去重测试
- `docs/MULTI_PARAM_DEDUP_GUIDE.md`: 本文档

### 相关文档
- [CLAUDE.md](../CLAUDE.md): 项目快速指南
- [CLAUDE_FULL_REFERENCE.md](CLAUDE_FULL_REFERENCE.md): 完整参考文档

---

## 常见问题

### Q1: 什么时候需要使用这个功能？

**A**: 当您同时变异多个超参数，并且希望避免生成历史数据中已有的参数组合时。

### Q2: 如果历史数据中包含的参数比当前变异的多，会怎样？

**A**: 系统会自动只比较您正在变异的那些参数，忽略历史数据中的其他参数。

### Q3: 如果历史数据中缺少某些参数，会怎样？

**A**: `filter_params` 只会比较存在的参数。如果某个参数在历史数据中不存在，该条历史记录对去重没有影响。

### Q4: 性能影响如何？

**A**: 几乎没有性能影响。过滤操作是简单的字典键值检查，时间复杂度为 O(n)，其中 n 是历史变异数量。

---

## 技术细节

### 去重键的生成

```python
# 不使用 filter_params（原有行为）
mutation = {"lr": 0.01, "bs": 32, "dropout": 0.5}
key = (('bs', '32'), ('dropout', '0.5'), ('lr', '0.01'), ('__mode__', 'nonparallel'))

# 使用 filter_params=["lr", "bs"]
filtered_mutation = {"lr": 0.01, "bs": 32}  # dropout 被过滤掉
key = (('bs', '32'), ('lr', '0.01'), ('__mode__', 'nonparallel'))
```

### Mode 区分

去重时会区分 parallel 和 nonparallel 模式：
- 相同参数在不同模式下**不**会被去重
- 例如: `(lr=0.01, mode=parallel)` ≠ `(lr=0.01, mode=nonparallel)`
- Mode 识别使用精确的后缀匹配（`endswith("_parallel")`）避免误判

### 按模型过滤

**重要**：去重自动按 repository 和 model 过滤，**只比较同一模型的历史数据**。

- 系统自动从历史数据中提取每条记录的 repo 和 model 信息
- 在构建去重集时，**只包含当前模型的历史变异**
- 这避免了跨模型去重的问题（例如 mnist 的参数不会与 cifar10 的参数比较）

**示例**：
```python
# 历史数据中有
# examples/mnist: lr=0.01, bs=32
# examples/cifar10: lr=0.01, bs=32

# 当前运行 examples/mnist
# → 去重集只包含 mnist 的历史数据
# → cifar10 的 (0.01, 32) 不会影响 mnist 的去重
```

---

## 更新日志

### v1.1 - 2026-01-11
- ✅ **修复 Mode 识别**：改用 `endswith("_parallel")` 避免误判
- ✅ **添加按模型过滤**：自动过滤到当前 repo/model，避免跨模型去重
- ✅ **改进日志输出**：显示过滤的 repo/model 和排除的记录数

### v1.0 - 2026-01-11
- ✅ 实现 `filter_params` 参数
- ✅ 在 `runner.py` 中自动应用
- ✅ 添加完整的测试套件
- ✅ 向后兼容性验证
- ✅ 文档编写

---

**维护者**: Energy DL Team
**最后更新**: 2026-01-11
