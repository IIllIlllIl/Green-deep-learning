# raw_data.csv 去重报告

**日期**: 2025-12-13
**问题**: 发现重复实验数据
**操作**: 去除重复数据并重新评估实验完成度

---

## 📋 问题发现

### 重复数据识别

在检查复合键去重方案时，发现raw_data.csv中存在1对重复数据：

**重复的实验对**:
- **Timestamp**: `2025-11-19T01:29:07.726147` (精确到微秒完全相同)
- **实验1**: `default__examples_mnist_008`
  - experiment_source: `default`
- **实验2**: `mutation_1x__examples_mnist_008`
  - experiment_source: `mutation_1x`

### 重复验证

**数据对比结果**:
- 总字段数: 80
- 相同字段: 78/80 (97.5%)
- 不同字段: 2/80 (2.5%)

**不同的字段**:
1. `experiment_id`: 仅ID前缀不同
2. `experiment_source`: `default` vs `mutation_1x`

**完全相同的字段**（证明是重复数据）:
- `timestamp`: 2025-11-19T01:29:07.726147 (精确到微秒)
- `duration_seconds`: 142.67011833190918 (完全一致)
- `energy_cpu_total_joules`: 4810.22
- `energy_gpu_total_joules`: 13123.7
- `perf_test_accuracy`: 96.0
- `perf_test_loss`: 0.1205
- 所有超参数值完全相同

**结论**: 这是同一个实验被错误地记录了两次，使用了不同的ID前缀。

---

## 🔧 去重操作

### 操作步骤

1. **备份原始数据**
   ```bash
   cp data/raw_data.csv data/raw_data.csv.backup_before_dedup_20251213_HHMMSS
   ```

2. **删除重复记录**
   - 保留: `default__examples_mnist_008` (experiment_source='default')
   - 删除: `mutation_1x__examples_mnist_008` (experiment_source='mutation_1x')
   - 理由: `default` 更准确地描述该实验（所有参数都是默认值）

3. **验证去重结果**
   - 原始数据: 480行
   - 去重后: 479行
   - 删除: 1行

### 去重后的数据验证

运行 `tools/data_management/validate_raw_data.py` 验证结果：

```
✅ 验证完成: raw_data.csv 数据完整且安全

📊 数据摘要:
  - 总实验数: 479
  - 训练成功: 479 (100.0%)
  - CPU能耗完整: 475 (99.2%)
  - GPU能耗完整: 475 (99.2%)
  - 性能指标完整: 372 (77.7%)
  - 数据格式: 80列标准格式
```

**验证通过** ✅

---

## 📊 去重后的实验完成度统计

### 总体统计

| 指标 | 数值 |
|------|------|
| 总参数-模式组合数 | 96 |
| **完全达标** | **57/96 (59.4%)** |
| 部分完成 | 33/96 (34.4%) |
| 完全缺失 | 6/96 (6.2%) |

**完全达标标准**: 每个参数-模式组合需要：
- 1个默认值实验（num_mutated_params=0）
- 5个唯一单参数变异实验

### 与去重前的对比

| 指标 | 去重前 | 去重后 | 变化 |
|------|--------|--------|------|
| 总实验数 | 480 | 479 | -1 |
| 完全达标组合 | 57/96 | 57/96 | 无变化 |
| 部分完成组合 | 33/96 | 33/96 | 无变化 |
| 完全缺失组合 | 6/96 | 6/96 | 无变化 |

**结论**: 去除的重复数据对实验完成度统计无影响，因为：
- 重复的实验是同一个模型（examples/mnist）的默认值实验
- 该模型在nonparallel模式下已有默认值实验覆盖
- 删除重复数据不影响任何参数-模式组合的达标状态

---

## 🎯 实验缺口分析

### 需要补充的实验数

| 类型 | 数量 | 说明 |
|------|------|------|
| 默认值实验 | 32个 | 部分组合缺少默认值基线 |
| 变异实验 | 1个 | MRT-OAST/default的epochs参数需要1个变异 |
| 完全缺失组合 | 36个 | 6个组合 × (1默认+5变异) |
| **总计** | **69个** | 需补充实验总数 |

### 完全缺失的组合（6个）

所有缺失组合都是Person_reID系列模型的 `weight_decay` 参数：

1. `Person_reID_baseline_pytorch/densenet121 | weight_decay | nonparallel`
2. `Person_reID_baseline_pytorch/densenet121 | weight_decay | parallel`
3. `Person_reID_baseline_pytorch/hrnet18 | weight_decay | nonparallel`
4. `Person_reID_baseline_pytorch/hrnet18 | weight_decay | parallel`
5. `Person_reID_baseline_pytorch/pcb | weight_decay | nonparallel`
6. `Person_reID_baseline_pytorch/pcb | weight_decay | parallel`

**原因**: 这3个模型的所有实验都未变异weight_decay参数

### 部分完成的典型案例（前5个）

| 模型 | 参数 | 模式 | 默认值 | 变异数 | 缺口 |
|------|------|------|--------|--------|------|
| MRT-OAST/default | epochs | nonparallel | 1 | 4 | 需1个变异 |
| Person_reID/densenet121 | dropout | nonparallel | 0 | 6 | 需1个默认值 |
| Person_reID/densenet121 | dropout | parallel | 0 | 5 | 需1个默认值 |
| Person_reID/densenet121 | epochs | nonparallel | 0 | 6 | 需1个默认值 |
| Person_reID/densenet121 | epochs | parallel | 0 | 5 | 需1个默认值 |

**主要问题**: Person_reID系列模型缺少默认值实验（32个中的大部分）

---

## 📈 完成度提升路径

### 短期目标（完成度 → 70%）

补充13个实验即可达到70%完成度（67/96）：

1. **MRT-OAST/default**:
   - epochs参数nonparallel模式：1个变异实验

2. **Person_reID系列（3个模型）**:
   - 每个模型4个参数 × 2种模式 = 24个默认值实验
   - 优先补充：dropout, epochs, learning_rate, seed的默认值

**预计时间**: 约5-8小时（Person_reID模型训练时间较长）

### 中期目标（完成度 → 90%）

补充额外的23个实验（总计36个）：

- 完成Person_reID系列的所有默认值实验
- 补充weight_decay参数的实验（6个组合 × 6实验 = 36个）

**预计时间**: 约20-30小时

### 长期目标（完成度 → 100%）

补充所有69个缺失实验：
- 32个默认值实验
- 1个变异实验
- 36个完全缺失组合的实验

**预计时间**: 约40-50小时

---

## ✅ 结论

### 去重效果

1. ✅ **成功去除1条重复数据**
   - 删除: `mutation_1x__examples_mnist_008`
   - 保留: `default__examples_mnist_008`

2. ✅ **数据完整性验证通过**
   - 479个实验数据完整
   - 100%训练成功率
   - 99.2%能耗数据完整
   - 77.7%性能数据完整

3. ✅ **实验完成度明确**
   - 当前: 57/96 (59.4%) 完全达标
   - 缺口: 69个实验
   - 主要缺失: 32个默认值实验 + 6个weight_decay组合

### 建议

1. **优先补充默认值实验**
   - 特别是Person_reID系列（3个模型 × 4参数 × 2模式 = 24个）
   - 这些实验对建立基线至关重要

2. **其次补充weight_decay参数实验**
   - 6个完全缺失的组合
   - 每个需要6个实验（1默认+5变异）

3. **保持数据质量**
   - 继续使用复合键去重方案（experiment_id + timestamp）
   - 定期验证数据完整性
   - 避免重复记录同一实验

---

**报告生成**: 2025-12-13
**维护者**: Green + Claude (AI Assistant)
**状态**: ✅ 完成
