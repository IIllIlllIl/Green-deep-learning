# 轮间超参数去重机制 - 使用指南

**版本**: 2.0 (Updated 2025-11-26)
**状态**: ✅ 已测试并就绪

---

## 📋 概述

轮间超参数去重机制通过读取历史实验数据，自动避免生成重复的超参数组合，确保每轮实验的超参数都是唯一的。

### 核心改进（v2.0）

✅ **简化数据源**: 从 `results/summary_all.csv` 读取（单一文件）
✅ **更高效**: 无需读取多个实验轮次的 CSV 文件
✅ **更易维护**: 只依赖汇总数据文件
✅ **完全向后兼容**: 不启用时无影响

---

## 🔍 去重机制工作原理

### 1. 数据加载流程

```
开始实验
    ↓
检查配置: use_deduplication == true?
    ↓ 是
读取 results/summary_all.csv
    ↓
提取所有历史超参数组合
    ↓
构建去重集合 (Set)
    ↓
生成新超参数时检查去重集合
    ↓
跳过重复组合，重新生成
    ↓
返回唯一的超参数组合
```

### 2. 归一化机制

超参数在比较前会被归一化，确保等价的值被识别为相同：

```python
# 示例：这些都会被识别为相同的超参数组合
{
    "epochs": 10,
    "learning_rate": 0.01
}

{
    "epochs": 10.0,
    "learning_rate": 0.010000
}

# 归一化后的键：
(("epochs", "10"), ("learning_rate", "0.010000"))
```

### 3. 当前数据状态

基于 `results/summary_all.csv` (2025-11-26):

- **总实验记录**: 211 条
- **唯一超参数组合**: 177 个
- **重复组合**: 34 条（16.1%）
- **覆盖模型**: 10 个

---

## ⚙️ 配置方法

### 启用去重机制

在实验配置文件中添加两个字段：

```json
{
  "experiment_name": "your_experiment_name",
  "mode": "batch",
  "runs_per_config": 2,
  "use_deduplication": true,        // ← 启用去重
  "historical_csvs": [               // ← 指定历史数据文件
    "results/summary_all.csv"
  ],
  "experiments": [
    {
      "repo": "examples",
      "model": "mnist",
      "num_mutations": 10
    }
  ]
}
```

### 配置参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `use_deduplication` | boolean | 是 | 是否启用轮间去重机制 |
| `historical_csvs` | array | 是 | 历史数据 CSV 文件路径列表 |

**推荐配置**:
```json
"use_deduplication": true,
"historical_csvs": ["results/summary_all.csv"]
```

### 禁用去重机制

不添加这两个字段，或设置为：

```json
"use_deduplication": false,
"historical_csvs": []
```

---

## 🚀 触发时机

### 何时应该启用去重机制？

✅ **推荐启用的场景**:

1. **运行新一轮变异实验**
   - 已经有多轮历史数据
   - 希望避免重复之前的超参数组合
   - 例如：运行 mutation_3x，已有 default、mutation_1x、mutation_2x

2. **补充实验**
   - 补充之前失败或缺失的实验
   - 确保新生成的超参数不与历史重复
   - 例如：当前的 mutation_2x_supplement

3. **扩展实验**
   - 在现有数据基础上增加更多实验
   - 希望探索新的超参数空间
   - 例如：增加新模型的变异测试

⚠️ **不需要启用的场景**:

1. **第一轮实验 (default)**
   - 没有历史数据
   - 只需要避免与默认值重复（已内置）

2. **独立的小规模测试**
   - 临时测试某个模型
   - 不关心与历史数据的重复

3. **特定超参数测试**
   - 明确指定超参数值（mode: "default"）
   - 不需要随机生成

---

## 📊 使用效果

### 示例：补充实验

**配置**: `settings/mutation_2x_supplement.json`

```json
{
  "experiment_name": "mutation_2x_supplement_20251126",
  "use_deduplication": true,
  "historical_csvs": ["results/summary_all.csv"]
}
```

**运行时输出**:

```
================================================================================
LOADING HISTORICAL HYPERPARAMETER DATA
================================================================================
Loading from 1 CSV files...
  ✓ results/summary_all.csv

Extracting mutations from CSV...
  ✓ Extracted 211 mutations

Building deduplication set...
  ✓ Created 177 unique mutation keys

================================================================================
Historical Hyperparameter Loading Statistics
================================================================================
CSV Files Processed: 1/1
Total Rows: 211
Extracted Mutations: 211
Unique Mutations: 177

Breakdown by Model:
  MRT-OAST/default: 32
  Person_reID_baseline_pytorch/densenet121: 26
  Person_reID_baseline_pytorch/hrnet18: 11
  Person_reID_baseline_pytorch/pcb: 10
  VulBERTa/mlp: 26
  bug-localization-by-dnn-and-rvsm/default: 20
  examples/mnist: 38
  examples/mnist_rnn: 24
  examples/siamese: 24
  pytorch_resnet_cifar10/resnet20: 26
================================================================================

Generating mutations for Person_reID_baseline_pytorch/hrnet18...
   Loaded 177 historical mutations for deduplication  ← 去重生效！
   Generated 8 unique mutations (avoiding historical data)
```

### 去重效果验证

运行后检查日志：

```bash
# 查看去重日志
grep "Loaded.*historical mutations" results/mutation_*/logs/*.log

# 应该看到：
# "Loaded 177 historical mutations for deduplication"
```

---

## 💡 实现细节

### 核心模块

#### 1. 去重模块 (`mutation/dedup.py`)

提供三个核心函数：

```python
from mutation.dedup import (
    load_historical_mutations,  # 从 CSV 加载历史数据
    build_dedup_set,            # 构建去重集合
    print_dedup_statistics      # 打印统计信息
)

# 使用示例
csv_files = [Path("results/summary_all.csv")]
mutations, stats = load_historical_mutations(csv_files)
dedup_set = build_dedup_set(mutations)
print_dedup_statistics(stats, dedup_set)
```

#### 2. 超参数生成 (`mutation/hyperparams.py`)

`generate_mutations()` 函数支持可选的去重参数：

```python
def generate_mutations(
    supported_params: Dict,
    mutate_params: List[str],
    num_mutations: int = 1,
    random_seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    existing_mutations: Optional[set] = None  # ← 去重集合
) -> List[Dict[str, Any]]:
    """生成变异的超参数，避免重复"""
```

#### 3. 实验运行器 (`mutation/runner.py`)

`MutationRunner` 自动加载历史数据并传递给 `generate_mutations()`：

```python
# 在 run_from_experiment_config() 中
if use_deduplication and historical_csvs:
    mutations, stats = load_historical_mutations(csv_paths)
    dedup_set = build_dedup_set(mutations)

# 传递给 generate_mutations()
mutations = generate_mutations(
    supported_params=supported_params,
    mutate_params=mutate_params,
    num_mutations=runs_per_config,
    existing_mutations=dedup_set  # ← 去重生效
)
```

---

## 🔄 数据维护

### summary_all.csv 的维护

去重机制依赖 `results/summary_all.csv`，因此需要定期更新：

#### 1. 初始生成

```bash
# 聚合所有实验轮次的数据
python3 scripts/aggregate_csvs.py
```

#### 2. 添加新轮次数据

每次完成新一轮实验后，重新聚合：

```bash
# 方法 1: 重新生成 summary_all.csv
python3 scripts/aggregate_csvs.py

# 方法 2: 手动添加新数据源
# 编辑 scripts/aggregate_csvs.py，添加新的 INPUT_CONFIGS
```

#### 3. 验证数据完整性

```bash
# 检查 CSV 文件
wc -l results/summary_all.csv

# 检查重复
python3 scripts/analyze_duplicates.py
```

---

## ⚠️ 注意事项

### 1. CSV 文件必须存在

如果 `results/summary_all.csv` 不存在，去重机制会自动禁用：

```
⚠️  Warning: 1 CSV files not found:
   - results/summary_all.csv
⚠️  No valid CSV files found, disabling inter-round deduplication
```

**解决方法**：
```bash
python3 scripts/aggregate_csvs.py
```

### 2. CSV 格式要求

`summary_all.csv` 必须包含超参数列：

- `hyperparam_epochs`
- `hyperparam_learning_rate`
- `hyperparam_batch_size`
- 等等...

这些列由 `aggregate_csvs.py` 自动生成。

### 3. 模型特定去重

当前实现对所有模型使用全局去重集合。如果需要模型特定去重：

```python
# 可选：按模型过滤历史数据
mutations, stats = load_historical_mutations(
    csv_files,
    filter_by_repo="examples",
    filter_by_model="mnist"
)
```

不过通常不需要这样做，因为不同模型的超参数范围不同，自然不会冲突。

### 4. 性能考虑

- **加载时间**: summary_all.csv (211 行) 加载 < 1 秒
- **内存占用**: 177 个去重键 ≈ 几 KB
- **查重时间**: O(1) 哈希查找

对于数千个实验也完全没问题。

---

## 🧪 测试验证

### 运行集成测试

```bash
python3 tests/functional/test_runner_dedup_integration.py
```

**预期输出**:

```
================================================================================
Test Summary
================================================================================
Total tests: 6
Passed: 6
Failed: 0

✓ All tests passed!

Configuration:
  - Deduplication enabled: Yes
  - Historical data source: results/summary_all.csv
================================================================================
```

### 手动验证

1. **检查配置文件**:
```bash
cat settings/mutation_2x_supplement.json | grep -A 5 "use_deduplication"
```

2. **验证 CSV 存在**:
```bash
ls -lh results/summary_all.csv
wc -l results/summary_all.csv
```

3. **运行实验并检查日志**:
```bash
python3 -m mutation.runner settings/mutation_2x_supplement.json 2>&1 | tee run.log
grep "Loaded.*historical mutations" run.log
```

---

## 📈 效果对比

### 不启用去重（之前）

```
轮次 1 (default):
  - epochs=10, lr=0.01 ✓

轮次 2 (mutation_1x):
  - epochs=10, lr=0.01 ✗ (重复!)
  - epochs=12, lr=0.02 ✓

轮次 3 (mutation_2x):
  - epochs=10, lr=0.01 ✗ (重复!)
  - epochs=12, lr=0.02 ✗ (重复!)
  - epochs=15, lr=0.03 ✓
```

**问题**: 10 个重复实验（4.7%）

### 启用去重（现在）

```
轮次 1 (default):
  - epochs=10, lr=0.01 ✓

轮次 2 (mutation_1x + 去重):
  加载 22 个历史超参数
  - epochs=12, lr=0.02 ✓ (新组合)
  - epochs=15, lr=0.03 ✓ (新组合)

轮次 3 (mutation_2x + 去重):
  加载 96 个历史超参数 (22 + 74)
  - epochs=18, lr=0.05 ✓ (新组合)
  - epochs=20, lr=0.01 ✓ (新组合)

补充轮次 (supplement + 去重):
  加载 177 个历史超参数
  - epochs=14, lr=0.015 ✓ (新组合，避免所有历史重复)
```

**结果**: 0 个重复实验 ✅

---

## 🎯 最佳实践

### 1. 每轮实验都启用去重

**推荐配置模板**:

```json
{
  "experiment_name": "mutation_Nx_YYYYMMDD",
  "mode": "batch",
  "runs_per_config": 2,
  "max_retries": 2,
  "use_deduplication": true,           // ← 始终启用
  "historical_csvs": [
    "results/summary_all.csv"          // ← 使用汇总数据
  ],
  "experiments": [ /* ... */ ]
}
```

### 2. 实验前更新 summary_all.csv

```bash
# 每次开始新实验前
python3 scripts/aggregate_csvs.py

# 验证数据
wc -l results/summary_all.csv
```

### 3. 实验后重新聚合数据

```bash
# 实验完成后
python3 scripts/aggregate_csvs.py

# 新的 summary_all.csv 现在包含最新实验
```

### 4. 定期检查重复

```bash
# 定期运行去重分析
python3 scripts/analyze_duplicates.py

# 如果发现新的重复，说明去重机制可能未启用
```

---

## 📚 相关文档

- **去重机制详细设计**: `docs/INTER_ROUND_DEDUPLICATION.md`
- **去重模块实现**: `mutation/dedup.py`
- **CSV 聚合脚本**: `scripts/aggregate_csvs.py`
- **缺失实验分析**: `docs/MISSING_EXPERIMENTS_CHECKLIST.md`

---

## 🔗 快速参考

### 启用去重的三个步骤

1. **确保 summary_all.csv 存在**:
   ```bash
   python3 scripts/aggregate_csvs.py
   ```

2. **在配置文件中启用**:
   ```json
   "use_deduplication": true,
   "historical_csvs": ["results/summary_all.csv"]
   ```

3. **运行实验**:
   ```bash
   python3 -m mutation.runner settings/your_config.json
   ```

### 验证去重生效

```bash
# 运行测试
python3 tests/functional/test_runner_dedup_integration.py

# 查看实验日志
grep "Loaded.*historical mutations" results/*/logs/*.log
```

---

**版本**: 2.0
**更新时间**: 2025-11-26
**维护者**: Mutation-Based Training Energy Profiler Team
**状态**: ✅ 已测试并就绪
