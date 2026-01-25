# 浮点数归一化与去重比较机制说明

**日期**: 2025-11-26
**版本**: Current Implementation

---

## 🔍 当前实现方式

### 归一化过程

在 `mutation/hyperparams.py` 的 `_normalize_mutation_key()` 函数中：

```python
FLOAT_PRECISION = 6  # 全局常量

def _normalize_mutation_key(mutation: Dict[str, Any]) -> tuple:
    normalized_items = []
    for param, value in mutation.items():
        if isinstance(value, float):
            # 关键：格式化为字符串（6位小数）
            normalized_value = f"{value:.{FLOAT_PRECISION}f}"
        elif isinstance(value, int):
            normalized_value = str(int(value))
        else:
            normalized_value = str(value)

        normalized_items.append((param, normalized_value))

    return tuple(sorted(normalized_items))
```

### 比较机制

**当前方式：字符串精确匹配**

- **不是** 数值容差比较（±1e-6）
- **而是** 字符串精确匹配

```python
# 步骤 1: 归一化为字符串
0.01       → "0.010000"
0.0100001  → "0.010000"  # 四舍五入
0.0100005  → "0.010001"  # 四舍五入
0.0100006  → "0.010001"  # 四舍五入

# 步骤 2: 字符串比较
"0.010000" == "0.010000"  # True  ✅ 认为重复
"0.010000" == "0.010001"  # False ✗ 认为不重复
```

---

## 📊 实际效果

### 示例 1: 学习率 (learning_rate)

```python
# 场景：两个学习率值
lr1 = 0.01
lr2 = 0.010000049

# 归一化
key1 = f"{lr1:.6f}"  # "0.010000"
key2 = f"{lr2:.6f}"  # "0.010000" (四舍五入)

# 比较结果
key1 == key2  # True ✅ 被认为是重复
```

**结论**: 差异 < 5×10^-7 的值会被认为相同

---

### 示例 2: Epochs (整数)

```python
# 场景：epochs 是整数
epochs1 = 10
epochs2 = 10.0

# 归一化
key1 = str(int(10))      # "10"
key2 = f"{10.0:.6f}"     # "10.000000"

# 比较结果
key1 != key2  # False ✗ 不同类型，会被认为是不同的
```

**注意**: 整数和浮点数会被区别对待

---

### 示例 3: 完整超参数组合

```python
# 历史实验
mutation1 = {
    "epochs": 10,
    "learning_rate": 0.01,
    "batch_size": 32
}

# 新生成的实验
mutation2 = {
    "epochs": 10,
    "learning_rate": 0.0100000123,  # 非常接近
    "batch_size": 32
}

# 归一化
key1 = (("batch_size", "32"), ("epochs", "10"), ("learning_rate", "0.010000"))
key2 = (("batch_size", "32"), ("epochs", "10"), ("learning_rate", "0.010000"))

# 比较
key1 == key2  # True ✅ 被认为是重复，mutation2 会被丢弃
```

---

## 🎯 精度分析

### 6 位小数的含义

对于不同数量级的值，6 位小数提供的精度不同：

| 值的范围 | 示例 | 最小可区分差异 | 相对精度 |
|---------|------|--------------|---------|
| 0.001 - 0.01 | learning_rate = 0.001 | 0.000001 | 0.1% |
| 0.01 - 0.1 | learning_rate = 0.01 | 0.000001 | 0.01% |
| 0.1 - 1.0 | learning_rate = 0.1 | 0.000001 | 0.001% |
| 1 - 10 | batch_size = 8.5 | 0.000001 | 0.00001% |
| 10 - 100 | epochs = 50.5 | 0.000001 | 0.000002% |

### 四舍五入边界

```python
# 临界点分析
0.0100004999999  → "0.010000"  # < 0.5 in 7th decimal
0.0100005000000  → "0.010001"  # ≥ 0.5 in 7th decimal
0.0100005000001  → "0.010001"
```

**关键**: 差异小于 **5×10^-7** 的值会四舍五入到相同字符串

---

## ❓ 为什么不用 ±1e-6 容差比较？

### 当前方式（字符串比较）的优势

✅ **1. 确定性**
```python
# 字符串比较：完全确定
"0.010000" == "0.010000"  # 总是 True

# 容差比���：需要定义容差
abs(0.01 - 0.0100000123) < 1e-6  # True
abs(0.01 - 0.0100001500) < 1e-6  # False，但很接近
```

✅ **2. 可哈希 (Hashable)**
```python
# 字符串可以放入 Set 中，查找是 O(1)
seen_mutations = set()
key = (("epochs", "10"), ("lr", "0.010000"))
seen_mutations.add(key)  # ✓ 可以

# 如果用浮点数，不能直接用 Set
mutation = {"epochs": 10, "lr": 0.01}
seen_mutations.add(tuple(mutation.items()))  # ✗ 浮点数精度问题
```

✅ **3. 性能**
```python
# Set 查找：O(1)
key in seen_mutations  # 极快

# 容差比较：O(n)
any(is_close(new, existing, tol=1e-6) for existing in seen_mutations)  # 慢
```

✅ **4. 简单明确**
- 归一化到字符串，比较逻辑清晰
- 不需要处理边界情况（0.9999999 vs 1.0）

---

## 🔧 如果需要容差比较？

### 方案 A: 保持当前实现（推荐）

**理由**:
- 6 位小数精度对于超参数已经足够
  - learning_rate: 0.000001 的差异通常不影响训练
  - epochs: 整数，不需要容差
  - batch_size: 整数，不需要容差
- Set 查找性能优秀 (O(1))
- 实现简单，不易出错

---

### 方案 B: 实现容差比较（如果真的需要）

```python
import math

def _is_close_mutation(mutation1: Dict, mutation2: Dict, rel_tol=1e-6, abs_tol=1e-6) -> bool:
    """使用容差比较两个超参数组合是否相同"""
    if set(mutation1.keys()) != set(mutation2.keys()):
        return False

    for key in mutation1:
        val1 = mutation1[key]
        val2 = mutation2[key]

        # 整数：精确比较
        if isinstance(val1, int) and isinstance(val2, int):
            if val1 != val2:
                return False

        # 浮点数：容差比较
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if not math.isclose(float(val1), float(val2), rel_tol=rel_tol, abs_tol=abs_tol):
                return False

        # 其他类型：精确比较
        else:
            if val1 != val2:
                return False

    return True

# 使用时需要 O(n) 遍历
def generate_mutations_with_tolerance(...):
    # ...
    for existing_mutation in seen_mutations_list:  # 不能用 Set，需要 List
        if _is_close_mutation(mutation, existing_mutation):
            # 重复，跳过
            continue
```

**缺点**:
- ❌ 性能下降：O(n) vs O(1)
- ❌ 复杂度增加
- ❌ 需要定义 rel_tol 和 abs_tol
- ❌ 边界情况复杂

---

### 方案 C: 调整精度（如果6位不够）

```python
# 增加精度到 8 位小数
FLOAT_PRECISION = 8

# 或减少到 4 位小数
FLOAT_PRECISION = 4
```

**权衡**:
- 精度更高 → 更多"唯一"组合，但可能实际效果相同
- 精度更低 → 更少"唯一"组合，可能误判为重复

---

## 📋 当前实现的实际效果

### 从 summary_all.csv 的数据

```
总实验记录: 211 条
唯一超参数: 177 个
重复率: 16.1% (34 条记录)
```

**分析**:
- 34 条记录被识别为重复（完全相同的字符串）
- 这些是真正的重复，不是因为浮点精度问题
- 归一化机制工作正常

### 测试验证

```bash
$ python3 tests/unit/test_dedup_mechanism.py
```

```python
# Test 4: Build dedup set
test_mutations = [
    {"epochs": 10.0, "learning_rate": 0.001},
    {"epochs": 20.0, "learning_rate": 0.01},
    {"epochs": 10.0, "learning_rate": 0.001},  # 精确重复
]

dedup_set = build_dedup_set(test_mutations)
assert len(dedup_set) == 2  # ✓ 检测到重复
```

---

## 🎯 结论与建议

### 当前机制总结

1. **归一化**: 浮点数 → 6位小数字符串
2. **比较**: 字符串精确匹配（不是容差比较）
3. **效果**: 差异 < 5×10^-7 的值被认为相同
4. **性能**: O(1) Set 查找

### 是否需要改为容差比较？

**建议：不需要** ✅

**理由**:
1. **精度足够**: 6位小数对超参数已经过于精确
   - learning_rate: 0.01 vs 0.0100001 在训练中无法区分
   - 实际训练中，这种微小差异不会导致不同结果

2. **性能优秀**: Set 查找 O(1) vs 容差比较 O(n)

3. **实现简单**: 字符串比较清晰明了，不易出错

4. **实际验证**: 从 211 条记录中正确识别了 177 个唯一组合

### 如果确实需要更粗粒度的去重

**方案**: 降低精度（不是改为容差比较）

```python
# 从 6 位降到 4 位小数
FLOAT_PRECISION = 4

# 效果
0.01     → "0.0100"
0.010001 → "0.0100"  # 被认为相同
0.0101   → "0.0101"  # 被认为不同
```

**优势**:
- 保持字符串比较的所有优势
- 更粗粒度的去重
- 不需要改变核心逻辑

---

## 📚 相关代码

### 核心函数

**`mutation/hyperparams.py:42-67`**
```python
def _normalize_mutation_key(mutation: Dict[str, Any]) -> tuple:
    """归一化并创建可哈希的键"""
    normalized_items = []
    for param, value in mutation.items():
        if isinstance(value, float):
            normalized_value = f"{value:.{FLOAT_PRECISION}f}"  # 字符串格式化
        # ...
    return tuple(sorted(normalized_items))
```

**使用位置**: `mutation/hyperparams.py:201`
```python
mutation_key = _normalize_mutation_key(mutation)
if mutation_key not in seen_mutations:  # Set 查找，O(1)
    seen_mutations.add(mutation_key)
    mutations.append(mutation)
```

---

**作者**: Mutation-Based Training Energy Profiler Team
**日期**: 2025-11-26
**版本**: 1.0
