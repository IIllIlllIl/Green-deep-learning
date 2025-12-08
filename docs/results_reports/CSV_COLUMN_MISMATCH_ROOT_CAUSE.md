# CSV列数不匹配根本原因分析

**问题日期**: 2025-12-03
**问题**: summary_all.csv第321-332行只有29列，而标准格式需要37列
**影响**: GitHub无法正确解析CSV文件

---

## 缺失的8个列详细信息

### 1. 超参数列 (4个)

| 列名 | 索引 | 说明 | Stage1为空原因 |
|------|------|------|---------------|
| `hyperparam_alpha` | 7 | bug-localization的alpha参数 | Stage1不包含bug-localization模型 |
| `hyperparam_kfold` | 11 | bug-localization的kfold参数 | Stage1不包含bug-localization模型 |
| `hyperparam_max_iter` | 13 | bug-localization的max_iter参数 | Stage1不包含bug-localization模型 |
| `hyperparam_weight_decay` | 15 | 权重衰减参数（多模型使用） | Stage1模型未变异此参数 |

### 2. 性能指标列 (3个)

| 列名 | 索引 | 说明 | Stage1为空原因 |
|------|------|------|---------------|
| `perf_best_val_accuracy` | 17 | 训练过程中最佳验证准确率 | Stage1模型不记录此指标 |
| `perf_test_accuracy` | 23 | 测试集准确率 | Stage1不包含记录此指标的模型 |
| `perf_test_loss` | 24 | 测试集损失值 | Stage1不包含记录此指标的模型 |

### 3. 元数据列 (1个)

| 列名 | 索引 | 说明 | Stage1为空原因 |
|------|------|------|---------------|
| `experiment_source` | 36 | 实验来源标记 | Stage1配置未包含此字段 |

---

## Stage1实验内容分析

### Stage1包含的12个实验

```
1. MRT-OAST_default_001 (1300s)
   - 变异参数: dropout=0.0427
   - 性能指标: accuracy, precision, recall

2-4. Person_reID_baseline_pytorch_hrnet18 (4155-4285s) × 3个
   - 变异参数: learning_rate, dropout, seed
   - 性能指标: map, rank1, rank5

5-8. Person_reID_baseline_pytorch_pcb (4126-5951s) × 4个
   - 变异参数: epochs, learning_rate, dropout, seed
   - 性能指标: map, rank1, rank5

9-12. examples_mnist_ff (10-18s) × 4个
   - 变异参数: batch_size, epochs, learning_rate, seed
   - 性能指标: (无性能指标记录)
```

### Stage1实际使用的29列

**元数据** (7个):
- experiment_id
- timestamp
- repository
- model
- training_success
- duration_seconds
- retries

**超参数** (5个):
- hyperparam_batch_size
- hyperparam_dropout
- hyperparam_epochs
- hyperparam_learning_rate
- hyperparam_seed

**性能指标** (6个):
- perf_accuracy
- perf_map
- perf_precision
- perf_rank1
- perf_rank5
- perf_recall

**能耗指标** (11个):
- energy_cpu_pkg_joules
- energy_cpu_ram_joules
- energy_cpu_total_joules
- energy_gpu_avg_watts
- energy_gpu_max_watts
- energy_gpu_min_watts
- energy_gpu_total_joules
- energy_gpu_temp_avg_celsius
- energy_gpu_temp_max_celsius
- energy_gpu_util_avg_percent
- energy_gpu_util_max_percent

---

## 代码缺陷分析

### 缺陷位置

**文件**: `mutation/runner.py`
**方法**: `_append_to_summary_all()`
**代码行**: 167-189

### 问题代码

```python
# 第167-170行: 读取session数据
with open(session_summary, 'r', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames  # ❌ 问题1: 使用session的29列
    session_rows = list(reader)

# 第179-189行: 写入summary_all
with open(summary_all_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)  # ❌ 问题2: 用29列写入

    if write_header:
        writer.writeheader()

    writer.writerows(session_rows)  # ❌ 结果: 只写入29列
```

### 为什么会出错

1. **第169行**: `fieldnames = reader.fieldnames`
   - 获取的是 session CSV 的列名（29列）
   - 不是 summary_all.csv 的列名（37列）

2. **第181行**: `writer = csv.DictWriter(f, fieldnames=fieldnames)`
   - 使用29列的fieldnames创建writer
   - DictWriter只会写入这29列
   - 忽略了summary_all.csv已有的37列结构

3. **第189行**: `writer.writerows(session_rows)`
   - 追加数据时只写入29列
   - 导致CSV格式不一致：
     - 前320行: 37列
     - 后12行: 29列

### 错误的假设

代码假设：
- ✗ session CSV 和 summary_all CSV 有相同的列结构
- ✗ 直接使用session的fieldnames就能正确追加

实际情况：
- ✓ session CSV 只包含当前实验使用的列（动态列集）
- ✓ summary_all CSV 包含所有可能的列（静态标准模板）
- ✓ 不同session的列数和列名可能不同

---

## 为什么Stage1只有29列

### 列生成机制

`mutation/session.py` 中的 CSV writer 根据**实际数据**动态生成列：

```python
# 只有实际存在的字段才会成为列
if experiment_data.get('hyperparameters'):
    for param, value in experiment_data['hyperparameters'].items():
        row[f'hyperparam_{param}'] = value  # 只添加实际使用的超参数

if experiment_data.get('performance'):
    for metric, value in experiment_data['performance'].items():
        row[f'perf_{metric}'] = value  # 只添加实际记录的性能指标
```

### Stage1的模型特点

| 模型 | 使用的超参数 | 记录的性能指标 |
|------|------------|---------------|
| MRT-OAST | dropout | accuracy, precision, recall |
| hrnet18/pcb | epochs, lr, dropout, seed | map, rank1, rank5 |
| mnist_ff | batch_size, epochs, lr, seed | (无) |

**缺失的参数**:
- 没有bug-localization → 没有 alpha, kfold, max_iter
- 没有变异weight_decay → 没有 weight_decay列
- 没有记录validation/test指标 → 没有 best_val_accuracy, test_accuracy, test_loss
- 没有experiment_source字段 → 没有 experiment_source列

---

## summary_all.csv的37列来源

### 历史数据包含所有列

summary_all.csv的37列是由**所有历史实验**的并集形成的：

1. **319个初始实验** (2025-11-25之前):
   - 包含11个不同仓库的模型
   - 包含bug-localization (alpha, kfold, max_iter)
   - 包含examples模型 (test_accuracy, test_loss)
   - 包含多种超参数变异 (weight_decay)
   - 包含各种性能指标 (best_val_accuracy)

2. **列的形成过程**:
   ```
   第1批实验 → 20列
   第2批实验 → 新增5列 → 25列
   第3批实验 → 新增8列 → 33列
   第N批实验 → 新增4列 → 37列 (当前)
   ```

3. **Stage1追加时的冲突**:
   ```
   summary_all.csv: 37列 (所有历史实验的并集)
   stage1 summary.csv: 29列 (只有Stage1使用的列)
   追加操作: 用29列覆盖 → 格式破坏
   ```

---

## 为什么这些列应该为空

### 合理性验证

对于Stage1的12个实验，这8个缺失列**应该**为空：

| 缺失列 | 应该为空吗？ | 原因 |
|--------|------------|------|
| hyperparam_alpha | ✓ 是 | Stage1没有bug-localization实验 |
| hyperparam_kfold | ✓ 是 | Stage1没有bug-localization实验 |
| hyperparam_max_iter | ✓ 是 | Stage1没有bug-localization实验 |
| hyperparam_weight_decay | ✓ 是 | Stage1没有变异weight_decay参数 |
| perf_best_val_accuracy | ✓ 是 | Stage1模型不记录此指标 |
| perf_test_accuracy | ✓ 是 | Stage1的examples模型未记录测试指标 |
| perf_test_loss | ✓ 是 | Stage1的examples模型未记录测试指标 |
| experiment_source | ✓ 是 | Stage1配置未设置此字段 |

**结论**: 修复方案（填充空值）是正确的。

---

## 修复验证

### 修复后的第321行

```csv
MRT-OAST_default_001,2025-12-02T19:20:10.686267,MRT-OAST,default,True,1300.203...,0,,,0.0427...,,,,4934.0,,,0.9991,38422.61,2977.18,...
                                                                                     ↑   ↑   ↑       ↑  ↑
                                                                                     7   11  13  15  17 23 24
                                                                                     缺失列位置（空值）
```

### 验证结果

```python
✓ 所有331行都有37列
✓ 缺失列正确填充为空值
✓ 数据内容保持不变
✓ CSV格式完全一致
✓ GitHub可以正常解析
```

---

## 根本原因总结

1. **直接原因**: `_append_to_summary_all()` 使用session的fieldnames而非summary_all的fieldnames

2. **设计缺陷**:
   - 假设所有CSV有相同的列结构
   - 未考虑动态列集 vs 静态标准模板的差异

3. **数据不一致**:
   - Session CSV是动态生成（只包含实际使用的列）
   - summary_all CSV是累积结果（包含所有可能的列）

4. **修复方向**:
   - 使用summary_all的列作为标准模板
   - 让DictWriter自动处理缺失列（填充空值）
   - 或在生成session CSV时就使用标准37列模板

---

**分析者**: Claude Code
**分析日期**: 2025-12-03
**状态**: ✅ 根本原因已识别，修复方案已验证
