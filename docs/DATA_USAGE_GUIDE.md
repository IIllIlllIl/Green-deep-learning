# 数据使用主指南

**文档版本**: 2.0 (统一版)
**创建日期**: 2025-12-28
**最后更新**: 2026-01-25
**适用数据**: data.csv, raw_data.csv

> **文档合并说明 (2026-01-25)**:
> 本文档整合了原有的 `DATA_MASTER_GUIDE.md` (574行) 和 `RAW_DATA_CSV_USAGE_GUIDE.md` (720行)，
> 消除了约50%的重复内容，提供统一的数据使用指南。

---

## 📑 目录

1. [快速开始](#快速开始)
2. [数据文件选择决策](#数据文件选择决策)
3. [关键注意事项](#关键注意事项)
4. [数据结构详解](#数据结构详解)
5. [代码示例](#代码示例)
6. [常见错误与解决方案](#常见错误与解决方案)
7. [数据质量现状](#数据质量现状)
8. [参考文档](#参考文档)

---

## 快速开始

### ⚡ 3秒决策：使用哪个数据文件？

```
你需要：
├─ 简单易用，快速上手？         → 使用 data.csv ⭐⭐⭐⭐⭐ 推荐
├─ 高质量数据，统一格式？       → 使用 data.csv ⭐⭐⭐⭐⭐
├─ 原始数据，完整列信息？       → 使用 raw_data.csv ⭐⭐⭐
└─ 特定研究需求，处理fg_前缀？  → 使用 raw_data.csv ⭐⭐⭐
```

**推荐方案**: 95%的情况下使用 `data.csv`

### ✅ 推荐：使用 data.csv

```python
import pandas as pd

# 读取数据
df = pd.read_csv('data/data.csv')

# 直接使用，无需考虑并行模式
learning_rate = df['learning_rate']
model = df['model']
energy = df['energy_cpu_total_joules']

# 筛选可用数据
df_usable = df[
    (df['training_success'] == True) &  # 训练成功
    (df['energy_cpu_total_joules'].notna())  # 有能耗数据
]

print(f"可用数据: {len(df_usable)}条")
```

### ⚠️ 备选：使用 raw_data.csv

```python
import pandas as pd

# 需要特殊处理函数
def get_field(df, row, field_name):
    """智能获取字段，自动处理并行/非并行模式"""
    if row['mode'] == 'parallel':
        fg_field = f'fg_{field_name}'
        return row[fg_field] if fg_field in df.columns else None
    else:
        return row[field_name]

# 使用示例
df = pd.read_csv('data/raw_data.csv')
for idx, row in df.iterrows():
    model = get_field(df, row, 'model')
    learning_rate = get_field(df, row, 'hyperparam_learning_rate')
```

**何时必须使用 raw_data.csv**:
- 需要87列完整数据（data.csv只有56列）
- 需要分析原始fg_字段
- 需要访问archived数据的特殊字段

---

## 数据文件选择决策

### data.csv vs raw_data.csv 详细对比

| 特性 | data.csv | raw_data.csv |
|------|----------|--------------|
| **行数** | 970行（含header） | 970行（含header） |
| **列数** | 56列 | 87列 |
| **并行模式** | ✅ 已统一字段 | ⚠️ 需处理fg_前缀 |
| **易用性** | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐⭐ 中等 |
| **数据完整性** | 818条可用 (84.3%) | 577条完全可用 (59.5%) |
| **适用场景** | 95%的分析任务 | 高级分析、完整列访问 |

### 什么时候使用 raw_data.csv？

✅ **应该使用 raw_data.csv**:
- 需要87列完整数据（data.csv只有56列）
- 需要访问原始fg_字段进行特殊分析
- 需要与archived数据对比
- 研究并行模式的内部机制

❌ **不应该使用 raw_data.csv**:
- 一般的数据分析和可视化
- 机器学习建模（推荐data.csv）
- 快速探索性分析

---

## 关键注意事项

### 🚨 极其重要：唯一标识符

**❌ 错误认识**: `experiment_id` 是唯一的
**✅ 正确认识**: `timestamp` 才是唯一键

```python
# ❌ 错误！会丢失大量数据！
df_unique = df.drop_duplicates(subset=['experiment_id'])

# ✅ 正确！
df_unique = df.drop_duplicates(subset=['timestamp'])
```

**原因**:
- `experiment_id`: 代表实验**配置**（可运行多次）
- `timestamp`: 代表实验**运行实例**（唯一）

**验证唯一性**:
```python
assert df['timestamp'].nunique() == len(df), "timestamp必须是唯一的！"
```

### 🚨 重要：并行模式数据处理

在 `raw_data.csv` 中:
- ❌ 并行模式数据在 `fg_` 前缀字段中
- ❌ 非并行模式数据在顶层字段中
- ❌ **不能直接使用顶层字段名！**

**解决方案**:
1. **推荐**: 使用 `data.csv`（已自动处理）
2. **备选**: 使用 `get_field()` 辅助函数（见上方代码）

### 🚨 重要：空字符串 vs 缺失值

```python
# ❌ 错误：将空字符串当作缺失值
df_clean = df.replace('', np.nan).dropna()

# ✅ 正确：空字符串是有效数据
df_clean = df.dropna()
# 只有True的NaN才是真正的缺失值
```

---

## 数据结构详解

### 总体结构

```
data/raw_data.csv (87列)
├── 基础字段 (7列)
│   ├── experiment_id, timestamp, repository, model
│   ├── training_success, duration_seconds, retries
│
├── 超参数字段 (9列)
│   ├── hyperparam_alpha, hyperparam_batch_size
│   ├── hyperparam_dropout, hyperparam_epochs
│   ├── hyperparam_kfold, hyperparam_learning_rate
│   ├── hyperparam_max_iter, hyperparam_seed
│   └── hyperparam_weight_decay
│
├── 性能指标字段 (9列)
│   ├── perf_test_accuracy, perf_best_val_accuracy
│   ├── perf_rank1, perf_rank5, perf_map
│   ├── perf_precision, perf_recall, perf_f1
│   └── perf_train_time
│
├── 能耗字段 (13列)
│   ├── energy_cpu_pkg_joules, energy_cpu_ram_joules
│   ├── energy_cpu_total_joules
│   ├── energy_gpu_avg_watts, energy_gpu_max_watts
│   ├── energy_gpu_min_watts, energy_gpu_total_joules
│   └── energy_gpu_temp_avg_celsius, energy_gpu_temp_max_celsius
│   └── energy_gpu_util_avg_percent, energy_gpu_util_max_percent
│
└── 元数据字段 (5列)
    ├── experiment_source, num_mutated_params
    ├── mutated_param, mode, error_message

data/data.csv (56列)
└── 统一格式，合并了顶层和fg_字段，添加了is_parallel列
```

### 并行vs非并行模式

**核心区别**:
- **非并行模式**: 数据在顶层字段（`repository`, `model`, `hyperparam_*`）
- **并行模式**: 数据在 `fg_` 前缀字段（`fg_repository`, `fg_model`, `fg_hyperparam_*`）

**识别方法**:
```python
# 检查是否为并行模式
df['is_parallel'] = (df['mode'] == 'parallel')

# data.csv已包含此列，可直接使用
```

### 字段详细说明

#### 基础字段

| 字段名 | 类型 | 说明 | 空值含义 |
|--------|------|------|---------|
| `experiment_id` | string | 实验唯一标识符 | 不应为空 |
| `timestamp` | string | ISO 8601格式时间戳 | 不应为空 |
| `repository` | string | 仓库名称 | **并行模式下为空** |
| `model` | string | 模型名称 | **并行模式下为空** |
| `training_success` | boolean | 训练是否成功 | **并行模式下为空** |
| `duration_seconds` | float | 训练时长（秒） | **并行模式下为空** |
| `retries` | int | 重试次数 | 0或空 |

#### 超参数字段

| 字段名 | 类型 | 适用模型 | 空值含义 |
|--------|------|---------|---------|
| `hyperparam_alpha` | float | bug-localization | 该模型不使用此参数 |
| `hyperparam_batch_size` | int | 大多数模型 | 使用默认值 |
| `hyperparam_dropout` | float | Person_reID, VulBERTa, MRT-OAST | 该模型不使用dropout |
| `hyperparam_epochs` | int | 所有模型 | 使用默认值 |
| `hyperparam_learning_rate` | float | 所有模型 | 使用默认值 |
| `hyperparam_seed` | int | 大多数模型 | 未设置或使用默认 |

**⚠️ 重要**：空值不代表数据缺失，而是：
1. 该模型不使用此超参数
2. 使用模型的默认值
3. 该参数在本次实验中未被变异

#### 性能指标字段

不同模型使用不同的性能指标：

| 模型 | 主要性能指标 |
|------|------------|
| Person_reID (densenet121, hrnet18, pcb) | `perf_rank1`, `perf_rank5`, `perf_map` |
| ResNet (pytorch_resnet_cifar10) | `perf_test_accuracy`, `perf_best_val_accuracy` |
| VulBERTa | `perf_precision`, `perf_recall`, `perf_f1` |
| Examples (mnist, mnist_rnn, siamese) | `perf_test_accuracy` |
| MRT-OAST | `perf_precision` |
| bug-localization | 无标准化指标 |

**空值含义**：该模型不使用此指标

#### 能耗字段

| 字段名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| `energy_cpu_pkg_joules` | float | 焦耳 | CPU Package能耗 |
| `energy_cpu_ram_joules` | float | 焦耳 | RAM能耗 |
| `energy_cpu_total_joules` | float | 焦耳 | CPU总能耗 (pkg + ram) |
| `energy_gpu_avg_watts` | float | 瓦特 | GPU平均功率 |
| `energy_gpu_total_joules` | float | 焦耳 | GPU总能耗 |

**空值含义**：能耗监控失败（权限问题或nvidia-smi不可用）

---

## 代码示例

### 示例1: 读取和预处理数据

```python
import pandas as pd
import numpy as np

# ✅ 推荐：使用 data.csv
df = pd.read_csv('data/data.csv')

# 验证数据完整性
assert df['timestamp'].nunique() == len(df), "timestamp必须唯一"

# 筛选可用数据
df_usable = df[
    (df['training_success'] == True) &  # 训练成功
    (df['energy_cpu_total_joules'].notna()) &  # 有能耗数据
    (df['perf_test_accuracy'].notna())  # 有性能指标
]

print(f"总数据: {len(df)}条")
print(f"可用数据: {len(df_usable)}条")
```

### 示例2: 按模型分组分析

```python
def analyze_by_model(df):
    """按模型统计实验数量和成功率"""
    results = []

    for repo_model in df.groupby(['repository', 'model']):
        repo, model = repo_model[0]
        data = repo_model[1]

        if not repo or not model:
            continue

        total = len(data)
        success_rate = (data['training_success'] == True).sum() / total * 100
        energy_rate = data['energy_cpu_total_joules'].notna().sum() / total * 100

        results.append({
            'repository': repo,
            'model': model,
            'total': total,
            'success_rate': f'{success_rate:.1f}%',
            'energy_coverage': f'{energy_rate:.1f}%'
        })

    return pd.DataFrame(results).sort_values('total', ascending=False)

# 使用
summary = analyze_by_model(df)
print(summary.to_string(index=False))
```

### 示例3: 能耗分析

```python
def analyze_energy_consumption(df, group_by='model'):
    """分析能耗消耗"""
    df_energy = df[df['energy_cpu_total_joules'].notna()].copy()

    # 按模型统计
    stats = df_energy.groupby(group_by).agg({
        'energy_cpu_total_joules': ['mean', 'median', 'std'],
        'energy_gpu_total_joules': ['mean', 'median'],
        'timestamp': 'count'
    }).round(2)

    return stats

# 使用
energy_stats = analyze_energy_consumption(df, group_by='model')
print(energy_stats)
```

### 示例4: 筛选高质量数据

```python
def get_high_quality_data(df):
    """获取高质量数据：训练成功 + 有能耗 + 有性能指标"""
    return df[
        (df['training_success'] == True) &
        (df['energy_cpu_total_joules'].notna()) &
        (df[['col for col in df.columns if col.startswith('perf_')]].notna().any(axis=1))
    ].copy()

df_hq = get_high_quality_data(df)
print(f"高质量数据: {len(df_hq)}条")
```

---

## 常见错误与解决方案

### ❌ 错误1: 使用 experiment_id 作为唯一键

**问题**: 不同批次的实验会产生相同的experiment_id

**错误代码**:
```python
df_unique = df.drop_duplicates(subset=['experiment_id'])
```

**正确代码**:
```python
# ✅ 使用 timestamp 作为唯一键
df_unique = df.drop_duplicates(subset=['timestamp'])

# ✅ 或使用复合键
df['composite_key'] = df['experiment_id'] + '|' + df['timestamp']
df_unique = df.drop_duplicates(subset=['composite_key'])
```

### ❌ 错误2: 直接读取 raw_data.csv 不处理并行模式

**问题**: 并行模式数据在fg_字段中，直接读取会遗漏

**错误代码**:
```python
df = pd.read_csv('data/raw_data.csv')
df_resnet = df[df['repository'] == 'pytorch_resnet_cifar10']
# 结果：遗漏了所有并行模式的resnet实验！
```

**正确代码**:
```python
# ✅ 方案1：使用 data.csv
df = pd.read_csv('data/data.csv')
df_resnet = df[df['repository'] == 'pytorch_resnet_cifar10']

# ✅ 方案2：使用 raw_data.csv + 处理函数
def get_field(df, row, field_name):
    if row['mode'] == 'parallel':
        fg_field = f'fg_{field_name}'
        return row[fg_field] if fg_field in df.columns else None
    return row[field_name]

df_resnet = df[df.apply(lambda x: get_field(df, x, 'repository') == 'pytorch_resnet_cifar10', axis=1)]
```

### ❌ 错误3: 误判空字符串为缺失值

**问题**: 空字符串是有效数据，不应该被当作缺失值

**错误代码**:
```python
df_clean = df.replace('', np.nan).dropna()
```

**正确代码**:
```python
# ✅ 只删除真正的缺失值
df_clean = df.dropna()
# 空字符串会被保留
```

### ❌ 错误4: 忽略训练失败的记录

**问题**: 应该根据分析目的决定是否包含失败记录

**错误代码**:
```python
df_success = df[df['training_success'] == True]  # 无条件过滤
```

**正确代码**:
```python
# ✅ 根据分析目的选择
if analysis_type == 'performance_analysis':
    df_filtered = df[df['training_success'] == True]
elif analysis_type == 'success_rate_analysis':
    df_filtered = df  # 包含失败记录
```

### ❌ 错误5: 不验证数据质量就开始分析

**问题**: 应该先验证数据质量，了解数据分布

**错误代码**:
```python
df = pd.read_csv('data/data.csv')
model.fit(df[['hyperparam_learning_rate']], df['perf_test_accuracy'])  # 直接建模
```

**正确代码**:
```python
# ✅ 先验证数据质量
df = pd.read_csv('data/data.csv')

# 1. 验证唯一性
assert df['timestamp'].nunique() == len(df)

# 2. 检查缺失值
print(df.isnull().sum())

# 3. 筛选可用数据
df_usable = df[
    (df['training_success'] == True) &
    (df['energy_cpu_total_joules'].notna())
]

print(f"数据质量: {len(df_usable)}/{len(df)} ({len(df_usable)/len(df)*100:.1f}%)")

# 4. 然后再建模
if len(df_usable) > 100:
    model.fit(df_usable[['hyperparam_learning_rate']], df_usable['perf_test_accuracy'])
```

---

## 数据质量现状

### 最新统计（2026-01-15更新）

#### data.csv 数据质量 ✅ 优秀

- **总记录数**: 970条（含header，实际969条数据）
- **✅ 可用记录**: 约818条 (84.3%)
  - 训练成功: 853条 (88.0%)
  - 有能耗数据: 828条 (85.4%)
- **推荐使用**: ⭐⭐⭐⭐⭐ 95%的分析任务

#### raw_data.csv 数据质量 ⚠️ 混合

- **总记录数**: 970条（含header，实际969条数据）
- **✅ 完全可用记录**: 577条 (59.5%)
  - 训练成功 + 有能耗 + 有性能指标
- **能耗数据**: 828条 (85.4%)
- **适用场景**: 高级分析、完整列访问

### 推荐使用的高质量数据

**8个100%可用模型** (487条记录):
- pytorch_resnet_cifar10/resnet20 (53条)
- Person_reID系列: densenet121, hrnet18, pcb (159条)
- examples系列: mnist, mnist_rnn, siamese, mnist_ff (275条)

**详细分析**: 参见 [docs/DATA_USABILITY_SUMMARY_20260113.md](DATA_USABILITY_SUMMARY_20260113.md)

---

## 参考文档

### 核心文档 ⭐⭐⭐⭐⭐

| 文档 | 用途 |
|------|------|
| [CLAUDE_FULL_REFERENCE.md](CLAUDE_FULL_REFERENCE.md) | 完整项目参考 |
| [CLAUDE.md](../CLAUDE.md) | 5分钟快速指南 |
| [analysis/docs/INDEX.md](../analysis/docs/INDEX.md) | 分析模块文档 |

### 数据质量报告

- [docs/results_reports/DATA_REPAIR_REPORT_20260104.md](results_reports/DATA_REPAIR_REPORT_20260104.md) - 数据完整性修复报告
- [docs/DATA_USABILITY_SUMMARY_20260113.md](DATA_USABILITY_SUMMARY_20260113.md) - 数据可用性分析
- [analysis/docs/DATA_FILES_COMPARISON.md](../analysis/docs/DATA_FILES_COMPARISON.md) - 文件对比分析

### 数据处理工具

- `tools/data_management/validate_raw_data.py` - 验证数据完整性
- `tools/data_management/analyze_experiment_status.py` - 分析实验状况
- `tools/data_management/analyze_missing_energy_data.py` - 分析缺失能耗
- `tools/data_management/repair_missing_energy_data.py` - 修复缺失能耗

---

**文档维护**: 本文档合并了原有的 DATA_MASTER_GUIDE 和 RAW_DATA_CSV_USAGE_GUIDE
**归档位置**: `archived/DATA_*_GUIDE.md.backup_20260125`
**合并日期**: 2026-01-25
**重复内容消除**: 约50%
