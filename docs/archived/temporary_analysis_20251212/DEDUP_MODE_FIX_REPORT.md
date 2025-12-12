# 去重机制修复报告 - 区分并行/非并行模式

**修复日期**: 2025-12-05
**问题识别**: 去重机制未区分并行和非并行运行模式
**修复状态**: ✅ 已完成

---

## 一、问题分析

### 1.1 问题描述

去重机制在检查历史实验时，只比较超参数值，**没有区分运行模式**（并行/非并行）。这导致：

- 相同超参数在非并行模式运行后
- 在并行模式尝试运行相同超参数时被错误跳过
- 并行模式的唯一值数量远低于预期

### 1.2 问题证据

**当前数据统计**：
```
非并行模式达标率: 97.8% (44/45参数)
并行模式达标率:   62.2% (28/45参数)
总体达标率:       80.0% (72/90参数-模式组合)
```

**并行模式缺失的参数**（18个）：
- hrnet18: 4个参数（dropout, epochs, learning_rate, seed）
- pcb: 4个参数（dropout, epochs, learning_rate, seed）
- mnist_ff: 4个参数（batch_size, epochs, learning_rate, seed）
- bug-localization: kfold
- mnist系列: batch_size (3个模型)
- mnist_rnn: epochs
- MRT-OAST: epochs (非并行)

### 1.3 根本原因

**代码层面**：

1. **`_normalize_mutation_key()` (hyperparams.py:42)**
   - 只基于超参数名和值创建key
   - 没有包含运行模式信息

2. **`extract_mutations_from_csv()` (dedup.py:53)**
   - 只提取超参数值
   - 不提取experiment_id中的模式标识

3. **去重key格式**：
   ```python
   # 原来的key（错误）
   (('learning_rate', '0.01'),)

   # 问题：相同超参数在两种模式下生成相同的key
   ```

---

## 二、修复方案

### 2.1 修改概览

修改了3个核心文件：
1. `mutation/hyperparams.py` - 增加mode参数支持
2. `mutation/dedup.py` - 从CSV提取模式信息
3. `mutation/runner.py` - 传入模式参数

### 2.2 详细修改

#### 修改1：hyperparams.py

**文件**: mutation/hyperparams.py
**函数**: `_normalize_mutation_key()`

**修改前**：
```python
def _normalize_mutation_key(mutation: Dict[str, Any]) -> tuple:
    normalized_items = []
    for param, value in mutation.items():
        # ... normalize value
        normalized_items.append((param, normalized_value))
    return tuple(sorted(normalized_items))
```

**修改后**：
```python
def _normalize_mutation_key(mutation: Dict[str, Any], mode: Optional[str] = None) -> tuple:
    normalized_items = []

    # Add mode as the first element if provided
    if mode is not None:
        normalized_items.append(('__mode__', mode))

    for param, value in mutation.items():
        # ... normalize value
        normalized_items.append((param, normalized_value))
    return tuple(sorted(normalized_items))
```

**新的key格式**：
```python
# 非并行模式
(('__mode__', 'nonparallel'), ('learning_rate', '0.010000'))

# 并行模式
(('__mode__', 'parallel'), ('learning_rate', '0.010000'))

# 现在两种模式的key不同，不会被错误去重！
```

**同时修改**：
- `generate_mutations()` 函数增加 `mode` 参数
- 在两处调用 `_normalize_mutation_key()` 时传入 `mode`

---

#### 修改2：dedup.py

**文件**: mutation/dedup.py
**函数**: `extract_mutations_from_csv()`, `build_dedup_set()`

**修改内容**：

1. **提取模式信息**（第109-123行）：
```python
# Extract execution mode from experiment_id
exp_id = row.get("experiment_id", "")
mode = "parallel" if "parallel" in exp_id else "nonparallel"

# Extract hyperparameters (only non-None values)
mutation = {}
for csv_col, param_name in HYPERPARAM_COLUMNS.items():
    value = _parse_hyperparam_value(row.get(csv_col, ""))
    if value is not None:
        mutation[param_name] = value

# Only add if we extracted at least one hyperparameter
if mutation:
    # Include mode information in the mutation dict
    mutation['__mode__'] = mode
    mutations.append(mutation)
```

2. **构建去重集合时传入mode**（第201-232行）：
```python
def build_dedup_set(mutations: List[Dict], logger: Optional[logging.Logger] = None) -> Set[tuple]:
    dedup_set = set()

    for mutation in mutations:
        # Extract and remove mode from mutation dict
        mode = mutation.pop('__mode__', None)

        # Create normalized key with mode
        key = _normalize_mutation_key(mutation, mode)
        dedup_set.add(key)

    return dedup_set
```

---

#### 修改3：runner.py

**文件**: mutation/runner.py
**修改位置**: 3处调用 `generate_mutations()` 的地方

**修改1 - 非并行模式**（第773行）：
```python
mutations = generate_mutations(
    supported_params=supported_params,
    mutate_params=mutate_params,
    num_mutations=num_runs,
    random_seed=self.random_seed,
    logger=self.logger,
    mode="nonparallel"  # Default to non-parallel mode
)
```

**修改2 - 并行模式前景**（第1009行）：
```python
fg_mutations = generate_mutations(
    supported_params=fg_supported_params,
    mutate_params=mutate_params,
    num_mutations=runs_per_config,
    random_seed=self.random_seed,
    logger=self.logger,
    existing_mutations=dedup_set,
    mode="parallel"  # Distinguish parallel mode for deduplication
)
```

**修改3 - 配置文件非并行模式**（第1107行）：
```python
mutations = generate_mutations(
    supported_params=supported_params,
    mutate_params=mutate_params,
    num_mutations=runs_per_config,
    random_seed=self.random_seed,
    logger=self.logger,
    existing_mutations=dedup_set,
    mode="nonparallel"  # Distinguish non-parallel mode for deduplication
)
```

---

## 三、修复效果预期

### 3.1 预期改进

修复后，去重机制将正确区分两种模式：

**并行模式**：
- 不再被错误地跳过非并行模式的历史实验
- 可以独立累积唯一值
- 预计达标率从62.2%提升到100%

**需要补充的实验**：
- 18个参数-模式组合
- 共需补充50个唯一值
- 预计需要74-116个新实验（考虑去重）

### 3.2 预计总实验数

```
当前: 381个实验
补充: 74-116个实验
总计: 455-497个实验
```

达成目标：所有90个参数-模式组合都有5个唯一值。

---

## 四、验证方法

### 4.1 功能验证

运行测试验证修复：

```bash
# 1. 测试非并行模式去重
python3 -c "
from mutation.hyperparams import _normalize_mutation_key
key1 = _normalize_mutation_key({'learning_rate': 0.01}, mode='nonparallel')
key2 = _normalize_mutation_key({'learning_rate': 0.01}, mode='parallel')
print(f'非并行key: {key1}')
print(f'并行key:   {key2}')
print(f'不同: {key1 != key2}')
"

# 2. 测试CSV提取
python3 << 'EOF'
from pathlib import Path
from mutation.dedup import extract_mutations_from_csv, build_dedup_set
import logging

logger = logging.getLogger()
mutations, stats = extract_mutations_from_csv(
    Path('results/summary_all.csv'),
    logger=logger
)

# 检查前5个mutation
for i, m in enumerate(mutations[:5]):
    print(f"{i+1}. mode={m.get('__mode__')}, params={list(m.keys())}")

dedup_set = build_dedup_set(mutations)
print(f"\n去重集合大小: {len(dedup_set)}")
EOF
```

### 4.2 实际运行验证

创建小规模测试配置：

```json
{
  "experiment_name": "dedup_mode_test",
  "description": "测试去重机制是否区分并行/非并行模式",
  "use_deduplication": true,
  "historical_csvs": ["results/summary_all.csv"],
  "experiments": [
    {
      "repo": "examples",
      "model": "mnist_ff",
      "mode": "parallel",
      "foreground": {
        "repo": "examples",
        "model": "mnist_ff",
        "mode": "mutation",
        "mutate": ["learning_rate"]
      },
      "background": {
        "repo": "examples",
        "model": "mnist",
        "hyperparameters": {}
      },
      "runs_per_config": 2
    }
  ]
}
```

**预期结果**：
- 生成的learning_rate值不会与历史非并行模式的值冲突
- 可以成功运行2个并行实验（而不是被全部跳过）

---

## 五、补充实验配置

基于修复后的去重机制，创建Stage7配置补充缺失的并行模式数据。

**配置文件**: `settings/stage7_parallel_补充.json`

**建议内容**：
```json
{
  "experiment_name": "stage7_parallel_completion",
  "description": "补充并行模式缺失的参数唯一值（修复去重机制后）",
  "mode": "mutation",
  "use_deduplication": true,
  "historical_csvs": ["results/summary_all.csv"],
  "runs_per_config": 4,
  "max_retries": 2,
  "governor": "performance",
  "experiments": [
    // hrnet18 (4参数 × 3缺失/参数 = 12个实验)
    {"repo": "Person_reID_baseline_pytorch", "model": "hrnet18",
     "mode": "parallel",
     "foreground": {..., "mutate": ["dropout"]},
     "runs_per_config": 4},
    // ... 其他17个参数-模式组合
  ]
}
```

**预计时间**：15-20小时（基于历史数据）

---

## 六、向后兼容性

### 6.1 兼容性说明

**Mode参数是可选的**：
- `_normalize_mutation_key(mutation)`  ← 仍然有效（向后兼容）
- `_normalize_mutation_key(mutation, mode="parallel")`  ← 新用法

**历史数据**：
- 旧的去重集合（不含mode）仍可使用
- 但不会区分模式（与修复前行为一致）

**渐进迁移**：
- 修复后的代码会为所有新实验添加mode信息
- 历史CSV已包含足够信息（experiment_id）来提取mode
- 下次加载历史数据时自动转换为新格式

---

## 七、总结

### 7.1 修复内容

✅ 修改3个核心文件
✅ 增加mode参数支持
✅ 从CSV自动提取模式
✅ 所有调用处传入mode
✅ 保持向后兼容

### 7.2 预期收益

- **并行模式达标率**: 62.2% → 100%
- **总体达标率**: 80.0% → 100%
- **避免重复实验**: 正确识别不同模式的实验
- **数据完整性**: 支持两种模式独立收集数据

### 7.3 后续步骤

1. **立即验证**: 运行测试确认修复有效
2. **创建Stage7配置**: 补充并行模式缺失数据
3. **执行实验**: 预计74-116个实验，15-20小时
4. **验证结果**: 确认所有90个参数-模式组合达标
5. **更新文档**: 在CLAUDE.md中记录此次修复

---

**修复完成日期**: 2025-12-05
**修复人员**: Claude Code Assistant
**审核状态**: 待用户确认

---

## 附录：修改文件列表

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| mutation/hyperparams.py | _normalize_mutation_key增加mode参数 | 42-75 |
| mutation/hyperparams.py | generate_mutations增加mode参数 | 180-188 |
| mutation/hyperparams.py | 调用_normalize_mutation_key时传入mode | 227, 260 |
| mutation/dedup.py | extract_mutations_from_csv提取mode | 109-129 |
| mutation/dedup.py | build_dedup_set传入mode | 201-232 |
| mutation/runner.py | run_mutation_experiments传入mode | 773-780 |
| mutation/runner.py | 并行模式generate_mutations传入mode | 1009-1017 |
| mutation/runner.py | 非并行模式generate_mutations传入mode | 1107-1115 |

总计修改行数：约30行
新增代码行数：约15行
删除代码行数：约0行（保持兼容）
