# data.csv vs raw_data.csv 文件对比

**创建日期**: 2025-12-22
**目的**: 说明主项目两个CSV文件的区别，帮助选择合适的数据源

---

## 📊 文件基本信息

| 文件 | 路径 | 大小 | 行数 | 列数 |
|------|------|------|------|------|
| **data.csv** | `results/data.csv` | 276KB | 677 | **54** |
| **raw_data.csv** | `results/raw_data.csv` | 302KB | 677 | **87** |

**共同点**:
- 行数相同（676个有效实验 + 1行表头）
- 训练成功率：100%
- 能耗数据完整率：91.1% (616/676)

---

## 🔍 列差异详解

### data.csv 包含的列（54列）

#### 1. 基础信息（8列）
```
experiment_id, timestamp, repository, model, is_parallel,
training_success, duration_seconds, retries, error_message
```

#### 2. 超参数（9列）
```
hyperparam_alpha, hyperparam_batch_size, hyperparam_dropout,
hyperparam_epochs, hyperparam_kfold, hyperparam_learning_rate,
hyperparam_max_iter, hyperparam_seed, hyperparam_weight_decay
```

#### 3. 性能指标（14列）
```
perf_best_val_accuracy, perf_map, perf_precision, perf_rank1, perf_rank5,
perf_recall, perf_test_accuracy, perf_test_loss, perf_final_training_loss,
perf_eval_samples_per_second, perf_top1_accuracy, perf_top5_accuracy,
perf_top10_accuracy, perf_top20_accuracy
```

#### 4. 能耗指标（10列）
```
energy_cpu_pkg_joules, energy_cpu_ram_joules, energy_cpu_total_joules,
energy_gpu_avg_watts, energy_gpu_max_watts, energy_gpu_min_watts,
energy_gpu_total_joules, energy_gpu_temp_avg_celsius,
energy_gpu_temp_max_celsius, energy_gpu_util_avg_percent,
energy_gpu_util_max_percent
```

#### 5. 实验元数据（7列）
```
experiment_source, num_mutated_params, mutated_param, mode,
bg_repository, bg_model, bg_note
```

#### 6. 并行模式元数据（6列）
```
bg_log_directory, fg_duration_seconds, fg_retries, fg_error_message
```

**总计**: 54列

---

### raw_data.csv 额外包含的列（+33列）

#### 1. 前景任务超参数（+9列）
```
fg_hyperparam_alpha, fg_hyperparam_batch_size, fg_hyperparam_dropout,
fg_hyperparam_epochs, fg_hyperparam_kfold, fg_hyperparam_learning_rate,
fg_hyperparam_max_iter, fg_hyperparam_seed, fg_hyperparam_weight_decay
```

#### 2. 前景任务性能指标（+9列）
```
fg_perf_accuracy, fg_perf_best_val_accuracy, fg_perf_map,
fg_perf_precision, fg_perf_rank1, fg_perf_rank5,
fg_perf_recall, fg_perf_test_accuracy, fg_perf_test_loss
```

#### 3. 前景任务能耗指标（+12列）
```
fg_energy_cpu_pkg_joules, fg_energy_cpu_ram_joules, fg_energy_cpu_total_joules,
fg_energy_gpu_avg_watts, fg_energy_gpu_max_watts, fg_energy_gpu_min_watts,
fg_energy_gpu_total_joules, fg_energy_gpu_temp_avg_celsius,
fg_energy_gpu_temp_max_celsius, fg_energy_gpu_util_avg_percent,
fg_energy_gpu_util_max_percent
```

#### 4. 前景任务元数据（+2列）
```
fg_repository, fg_model, fg_training_success
```

#### 5. 其他性能指标（+2列）
```
perf_accuracy, perf_eval_loss
```

**总计**: 87列（54 + 33）

---

## 🎯 使用场景建议

### 场景1: 单任务分析（非并行或整体）

**推荐文件**: **data.csv** ✅

**原因**:
- 包含所有核心超参数、能耗、性能数据
- 文件更小，加载更快
- 列结构简洁，易于理解

**适用研究问题**:
- 超参数对能耗的影响
- 超参数对性能的影响
- 能耗-性能权衡分析
- 模型类型对比

**示例**:
```python
import pandas as pd
df = pd.read_csv('data/energy_research/raw/energy_data_original.csv')

# 分析learning_rate对GPU能耗的影响
df[['learning_rate', 'energy_gpu_avg_watts', 'perf_test_accuracy']]
```

---

### 场景2: 并行模式深入分析

**推荐文件**: **raw_data.csv** ✅

**原因**:
- 包含并行模式下前景任务（fg）和背景任务（bg）的完整数据
- 可以对比前景-背景任务的差异
- 可以分析并行模式的资源分配

**适用研究问题**:
- 并行训练的前景任务性能如何？
- 前景任务与背景任务的能耗分配比例？
- 并行模式下的干扰效应？

**示例**:
```python
import pandas as pd
df = pd.read_csv('results/raw_data.csv')

# 筛选并行实验
df_parallel = df[df['fg_repository'].notna()]

# 对比前景和背景能耗
df_parallel[['energy_gpu_avg_watts', 'fg_energy_gpu_avg_watts']].describe()
```

---

### 场景3: 因果分析（本项目）

**推荐文件**: **data.csv** ✅ **（首选）**

**原因**:
- DiBS+DML分析需要简洁的变量集合
- 54列已包含所有关键因果变量
- 避免fg_/bg_前缀带来的复杂性

**变量设计**:
```python
# 输入变量（因）
inputs = [
    'learning_rate', 'batch_size', 'epochs', 'dropout', 'weight_decay',
    'is_parallel',  # 并行模式
    'model_*',      # 模型类型（One-Hot编码后）
]

# 输出变量（果）
outputs = [
    'energy_gpu_avg_watts', 'energy_gpu_total_joules',  # 能耗
    'perf_test_accuracy', 'perf_test_loss',              # 性能
    'duration_seconds'                                    # 时长
]
```

**如需分析并行模式的调节效应**:
- 可使用`is_parallel`作为分组变量
- 不需要fg_/bg_的详细数据

---

## 📋 关键差异总结

| 维度 | data.csv | raw_data.csv |
|------|----------|--------------|
| **列数** | 54 | 87 |
| **文件大小** | 276KB | 302KB |
| **前景任务详细数据** | ❌ 无 | ✅ 有（fg_*字段） |
| **适用场景** | 单任务分析、因果分析 | 并行模式深入分析 |
| **复杂度** | 低（简洁） | 高（详细） |
| **推荐度** | ⭐⭐⭐ | ⭐⭐ |

---

## 🚀 本项目选择

**当前使用**: `data.csv` → 复制为 `energy_data_original.csv`

**理由**:
1. 因果分析不需要前景任务的详细分解数据
2. `is_parallel`字段已足够表示并行模式
3. 简洁的54列结构更适合DiBS变量选择
4. 文件更小，处理更快

**如果未来需要**:
- 分析"前景任务 vs 背景任务"的资源竞争
- 研究并行模式的详细因果机制

**则切换到**: `raw_data.csv`

---

## 📊 数据完整性对比

| 指标 | data.csv | raw_data.csv | 一致性 |
|------|----------|--------------|--------|
| 训练成功 | 676/676 (100%) | 676/676 (100%) | ✅ |
| 能耗数据 | 616/676 (91.1%) | 616/676 (91.1%) | ✅ |
| 性能数据 | 616/676 (91.1%) | 616/676 (91.1%) | ✅ |
| 并行实验 | 有`is_parallel`标识 | 有`fg_*`完整数据 | ✅ |

**结论**: 两个文件的有效样本数完全一致，data.csv已足够满足大部分分析需求。

---

**文档维护者**: Analysis模块
**最后更新**: 2025-12-22
