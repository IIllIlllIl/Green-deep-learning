# 性能指标合并可行性分析

**分析日期**: 2025-12-19
**版本**: v1.0
**目标**: 评估同类性能指标是否可以合并，减少列数，提高数据一致性

---

## 📊 核心问题

当前data.csv有**16个性能指标列**，分为5大类。问题：
- 同一类中的多个指标是否能合并？
- 合并的标准是什么？
- 合并后是否会丢失信息？

---

## 🔍 逐类分析

### 1. 标准准确率组 ⚠️ **不建议合并**

#### 指标列表
- `test_accuracy` - 测试集准确率
- `best_val_accuracy` - 最佳验证集准确率
- `accuracy` - 通用准确率

#### 共现模式
```
test_accuracy: 219次 (独立使用)
accuracy: 46次 (独立使用)
best_val_accuracy+test_accuracy: 39次 (同时使用)
```

#### 使用场景
| 模型 | 使用指标 | 说明 |
|------|---------|------|
| mnist, mnist_ff, mnist_rnn, siamese | `test_accuracy` | 仅测试集准确率 |
| resnet20 | `test_accuracy` + `best_val_accuracy` | **同时记录两个** |
| MRT-OAST | `accuracy` | 通用准确率（已修复） |

#### 语义差异分析

| 指标 | 评估阶段 | 评估集 | 是否可替代 |
|------|---------|--------|-----------|
| `test_accuracy` | 训练后 | 测试集 | ❌ 不可替代 |
| `best_val_accuracy` | 训练中 | 验证集 | ❌ 不可替代 |
| `accuracy` | 训练后 | 测试集 | ✅ 可映射到test_accuracy |

**关键差异**:
- `test_accuracy` vs `best_val_accuracy`: **不同评估阶段和数据集**
  - test_accuracy: 训练结束后在测试集上评估（未见过的数据）
  - best_val_accuracy: 训练过程中验证集的最佳表现
  - resnet20同时记录两者是为了**检测过拟合**（val > test 说明过拟合）

- `accuracy` vs `test_accuracy`: **名称差异，语义相同**
  - MRT-OAST使用`accuracy`仅因命名习惯
  - 实际都是测试集准确率

#### 合并建议 ⭐

**方案**: 部分统一

1. **保留** `test_accuracy` - 主要测试指标
2. **保留** `best_val_accuracy` - 训练监控指标（仅resnet20使用）
3. **合并** `accuracy` → `test_accuracy` - 统一命名

**理由**:
- ✅ test_accuracy 和 best_val_accuracy 含义不同，必须保留
- ✅ accuracy 可以重命名为 test_accuracy（无信息损失）
- ✅ 合并后减少1列

**影响**: MRT-OAST的46个实验的`accuracy`列改为`test_accuracy`列

---

### 2. Top-K准确率组 ❌ **不建议合并**

#### 指标列表
- `top1_accuracy` - Top-1准确率
- `top5_accuracy` - Top-5准确率
- `top10_accuracy` - Top-10准确率
- `top20_accuracy` - Top-20准确率

#### 共现模式
```
top1+top5+top10+top20: 40次 (总是同时出现)
```

#### 使用场景
| 模型 | 使用指标 | 任务类型 |
|------|---------|---------|
| bug-localization | 全部4个 | Bug定位排序 |

#### 语义差异分析

这4个指标是**递进关系**，不是冗余关系：

| 指标 | 含义 | 典型值 | 信息增益 |
|------|------|--------|---------|
| `top1_accuracy` | 首位正确率 | 37.7% | 基准 |
| `top5_accuracy` | 前5位包含正确答案 | 62.4% | +24.7% |
| `top10_accuracy` | 前10位包含正确答案 | 73.5% | +11.1% |
| `top20_accuracy` | 前20位包含正确答案 | 82.4% | +8.9% |

**关键特性**:
- 单调递增: top1 ≤ top5 ≤ top10 ≤ top20 ✅
- 每个指标提供独立信息（排序质量曲线）
- 用于评估推荐系统在不同截断点的表现

#### 合并建议 ❌

**方案**: 不合并

**理由**:
- ❌ 4个指标描述不同的性能维度
- ❌ 合并会丢失排序质量曲线信息
- ❌ 仅1个模型使用（bug-localization），合并收益小

**替代方案**: 如果必须减少列数，可以考虑：
- 保留 top1 和 top5（最常用的两个）
- 或创建一个"top_k_accuracies"JSON列存储所有4个值

**不推荐**，会破坏数据完整性。

---

### 3. 检索指标组 ⚠️ **部分可合并**

#### 指标列表
- `map` - Mean Average Precision (平均精度均值)
- `rank1` - Rank-1准确率
- `rank5` - Rank-5准确率

#### 共现模式
```
map+rank1+rank5: 116次 (总是同时出现)
```

#### 使用场景
| 模型 | 使用指标 | 任务类型 |
|------|---------|---------|
| densenet121, hrnet18, pcb | 全部3个 | 人员重识别 |

#### 语义差异分析

| 指标 | 含义 | 典型值 | 可否代表整体性能 |
|------|------|--------|----------------|
| `map` | 综合检索质量（所有查询的平均精度） | 75% | ✅ **主指标** |
| `rank1` | 首位匹配率 | 91% | ⚠️ 辅助指标 |
| `rank5` | 前5位匹配率 | 97% | ⚠️ 辅助指标 |

**关键差异**:
- **mAP**: 综合评估所有排序位置的精度（**最全面**）
- **Rank-1**: 仅关注首位匹配（最严格）
- **Rank-5**: 允许前5位匹配（较宽松）

**关系**:
- mAP 包含了整个排序列表的信息
- Rank-1/5 只关注特定截断点
- 但 Rank-1/5 更直观，更易解释

#### 合并建议 ⭐

**方案1**: 保留mAP作为主指标（激进方案）
- 保留 `map`
- 删除 `rank1`, `rank5`
- 理由: mAP已包含所有排序信息

**方案2**: 保留mAP和Rank-1（保守方案）⭐ **推荐**
- 保留 `map` - 综合指标
- 保留 `rank1` - 最严格指标（业界标准）
- 删除 `rank5` - 与rank1冗余
- 理由:
  - mAP + Rank-1 是人员重识别领域的**标准报告指标**
  - Rank-5 虽有价值，但优先级低于Rank-1

**方案3**: 全部保留（最保守）
- 理由: 仅3个模型使用，数据量不大，保留完整信息

#### 推荐 ⭐

**方案2 - 保留 map + rank1**

**影响**:
- 减少1列（删除rank5）
- 116个实验的rank5数据移至归档
- 保留最关键的综合指标（map）和业界标准指标（rank1）

---

### 4. 精确召回组 ❌ **不建议合并**

#### 指标列表
- `precision` - 精确率（查准率）
- `recall` - 召回率（查全率）

#### 共现模式
```
precision+recall: 58次 (总是同时出现)
```

#### 使用场景
| 模型 | 使用指标 | 任务类型 |
|------|---------|---------|
| MRT-OAST | precision + recall | 二分类 |

#### 语义差异分析

| 指标 | 含义 | 典型值 | 关注点 |
|------|------|--------|--------|
| `precision` | TP / (TP + FP) | 98.3% - 99.9% | 预测为正的有多少真正确 |
| `recall` | TP / (TP + FN) | 74.3% - 94.7% | 真正例有多少被找到 |

**关键差异**:
- Precision和Recall是**对立的两个维度**
- 高precision ≠ 高recall（经典trade-off）
- 两者缺一不可

#### 合并建议 ❌

**方案**: 不合并

**理由**:
- ❌ 两者描述不同维度（查准 vs 查全）
- ❌ 合并会丢失关键信息
- ❌ 分类任务的标准评估需要两者

**替代方案**: 如果必须合并，可以计算F1-score
- F1 = 2 × (precision × recall) / (precision + recall)
- 但这会丢失precision和recall的独立信息

**不推荐**，两个指标都应保留。

---

### 5. 损失函数组 ⚠️ **部分可合并**

#### 指标列表
- `test_loss` - 测试集损失
- `eval_loss` - 评估集损失
- `final_training_loss` - 最终训练损失

#### 共现模式
```
test_loss: 122次 (独立使用)
eval_loss+final_training_loss: 72次 (同时使用)
```

#### 使用场景
| 模型 | 使用指标 | 说明 |
|------|---------|------|
| mnist, mnist_rnn, siamese | `test_loss` | 测试集损失 |
| VulBERTa/mlp | `eval_loss` + `final_training_loss` | 评估损失 + 训练损失 |

#### 语义差异分析

| 指标 | 评估阶段 | 评估集 | 是否可替代 |
|------|---------|--------|-----------|
| `test_loss` | 训练后 | 测试集 | ❌ 不可替代 |
| `eval_loss` | 训练后 | 评估集 | ⚠️ 可能等同于test_loss |
| `final_training_loss` | 训练结束 | 训练集 | ❌ 不可替代 |

**关键差异**:
- `test_loss` vs `eval_loss`: **可能是同义词**
  - 不同框架的命名差异（PyTorch vs Transformers）
  - 都表示在独立数据集上的损失评估

- `final_training_loss` vs 其他: **不同数据集**
  - 训练集损失（已见过的数据）
  - 用于对比过拟合程度（training_loss << eval_loss → 过拟合）

#### 合并建议 ⭐

**方案**: 统一命名

1. **合并** `eval_loss` → `test_loss` - 统一命名
2. **保留** `final_training_loss` - 独立信息

**理由**:
- ✅ test_loss 和 eval_loss 语义相同（评估集损失）
- ✅ final_training_loss 提供独立信息（训练集损失）
- ✅ 合并后减少1列

**影响**: VulBERTa/mlp的72个实验的`eval_loss`列改为`test_loss`列

---

## 📋 合并方案总结

### 推荐合并方案 ⭐⭐⭐

| 操作 | 源列 | 目标列 | 影响实验数 | 减少列数 |
|------|------|--------|-----------|---------|
| 重命名 | `accuracy` | `test_accuracy` | 46 | -1 |
| 重命名 | `eval_loss` | `test_loss` | 72 | -1 |
| 删除 | `rank5` | - | 116 | -1 |

**总计**: 减少 **3列** (16 → 13列)

### 保守合并方案 ⭐⭐

| 操作 | 源列 | 目标列 | 影响实验数 | 减少列数 |
|------|------|--------|-----------|---------|
| 重命名 | `accuracy` | `test_accuracy` | 46 | -1 |
| 重命名 | `eval_loss` | `test_loss` | 72 | -1 |

**总计**: 减少 **2列** (16 → 14列)

**保留** `rank5`，理由：数据完整性优先

### 不推荐合并的指标

| 指标组 | 原因 |
|--------|------|
| `test_accuracy` vs `best_val_accuracy` | 不同评估阶段，用于检测过拟合 |
| Top-K准确率 (4个) | 描述排序质量曲线，缺一不可 |
| `precision` vs `recall` | 对立维度，经典trade-off |
| `test_loss` vs `final_training_loss` | 不同数据集（测试集 vs 训练集） |

---

## 🎯 具体实施建议

### 阶段1: 低风险合并（推荐立即执行）⭐

#### 1.1 统一accuracy命名
```python
# 将 MRT-OAST 的 accuracy 列重命名为 test_accuracy
for row in data:
    if row['repository'] == 'MRT-OAST' and row['perf_accuracy']:
        row['perf_test_accuracy'] = row['perf_accuracy']
        row['perf_accuracy'] = ''
```

**影响**: 46个实验，无数据损失

#### 1.2 统一loss命名
```python
# 将 VulBERTa/mlp 的 eval_loss 重命名为 test_loss
for row in data:
    if row['repository'] == 'VulBERTa' and row['perf_eval_loss']:
        row['perf_test_loss'] = row['perf_eval_loss']
        row['perf_eval_loss'] = ''
```

**影响**: 72个实验，无数据损失

### 阶段2: 谨慎合并（可选）⚠️

#### 2.1 删除rank5
```python
# 备份 rank5 数据到归档文件
# 从 data.csv 中删除 perf_rank5 列
```

**影响**: 116个实验，**有数据丢失**（但可从归档恢复）

**建议**:
- 先备份 rank5 数据到独立CSV文件
- 评估一段时间后再决定是否永久删除
- 如果有用户需要 rank5，可快速恢复

---

## 📊 合并前后对比

### 当前状态 (16列)
```
准确率类(3): test_accuracy, best_val_accuracy, accuracy
Top-K准确率(4): top1_accuracy, top5_accuracy, top10_accuracy, top20_accuracy
检索指标(3): map, rank1, rank5
精确召回(2): precision, recall
损失类(3): test_loss, eval_loss, final_training_loss
吞吐量(1): eval_samples_per_second
```

### 推荐方案后 (13列) ⭐
```
准确率类(2): test_accuracy, best_val_accuracy  [合并accuracy]
Top-K准确率(4): top1_accuracy, top5_accuracy, top10_accuracy, top20_accuracy
检索指标(2): map, rank1  [删除rank5]
精确召回(2): precision, recall
损失类(2): test_loss, final_training_loss  [合并eval_loss]
吞吐量(1): eval_samples_per_second
```

### 保守方案后 (14列) ⭐⭐
```
准确率类(2): test_accuracy, best_val_accuracy  [合并accuracy]
Top-K准确率(4): top1_accuracy, top5_accuracy, top10_accuracy, top20_accuracy
检索指标(3): map, rank1, rank5  [保留]
精确召回(2): precision, recall
损失类(2): test_loss, final_training_loss  [合并eval_loss]
吞吐量(1): eval_samples_per_second
```

---

## ✅ 推荐执行方案

**建议**: 采用**保守合并方案**（14列）⭐⭐

**理由**:
1. ✅ 减少2列（12.5%精简）
2. ✅ 无数据丢失（仅重命名）
3. ✅ 向后兼容（保留所有原始信息）
4. ✅ 语义更统一（test_accuracy, test_loss）

**下一步**:
1. 实施accuracy和eval_loss的重命名
2. 验证合并后的数据完整性
3. 评估是否需要删除rank5（可延后决定）

---

**分析完成日期**: 2025-12-19
**推荐方案**: 保守合并（减少2列，无数据损失）
**核心原则**: 语义统一 > 列数减少，信息完整性优先
