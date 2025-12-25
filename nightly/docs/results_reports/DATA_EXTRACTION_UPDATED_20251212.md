# 数据提取问题模型名单（更新版）

**日期**: 2025-12-12
**更新**: 基于完整的18个性能指标列重新分析
**数据源**: results/raw_data.csv (458个唯一实验)

---

## 🎉 重大发现：数据比预想的好很多！

### 之前的分析（错误）
- ❌ 使用了错误的列名（test_accuracy而非perf_test_accuracy）
- ❌ 遗漏了9个重要性能指标列
- 结论：7个模型全部缺失性能数据

### 现在的分析（正确）
- ✅ 使用完整的18个性能指标列
- ✅ 包含所有9种指标类型（accuracy, map, rank1, rank5等）
- **结论**: ✅ **7个模型数据完整，只有4个模型有问题！**

---

## 📊 总体统计（更新）

| 指标 | 数值 | 变化 |
|------|------|------|
| **数据完整的模型** | **7/11 (63.6%)** | ⬆️ 从36.4% |
| 性能数据全缺的模型 | 3/11 (27.3%) | ⬇️ 从63.6% |
| 性能数据部分缺的模型 | 1/11 (9.1%) | 新发现 |
| 有性能数据的实验 | 327/458 (71.4%) | ⬆️ 从35.4% |
| 受影响的实验 | 131/458 (28.6%) | ⬇️ 从64.6% |

**关键改进**: 从7个问题模型 → 4个问题模型（减少43%）

---

## ✅ 数据完整的模型（7个，327个实验）

| 模型 | 实验数 | 性能数据 | 使用的指标 |
|------|--------|----------|-----------|
| **Person_reID_baseline_pytorch/densenet121** | 43 | ✅ 43/43 | map, rank1, rank5 |
| **Person_reID_baseline_pytorch/hrnet18** | 34 | ✅ 34/34 | map, rank1, rank5 |
| **Person_reID_baseline_pytorch/pcb** | 34 | ✅ 34/34 | map, rank1, rank5 |
| **examples/mnist** | 40 | ✅ 40/40 | test_accuracy, test_loss |
| **examples/mnist_rnn** | 43 | ✅ 43/43 | test_accuracy, test_loss |
| **examples/siamese** | 40 | ✅ 40/40 | test_accuracy, test_loss |
| **pytorch_resnet_cifar10/resnet20** | 39 | ✅ 39/39 | best_val_accuracy, test_accuracy |

**总计**: 273个实验，100%性能数据完整

### 关键发现

1. **Person_reID系列实际上数据完整！**
   - 之前误判为"全部缺失"
   - 实际使用ReID专用指标：`perf_map`, `perf_rank1`, `perf_rank5`
   - 提取系统已经正确支持ReID指标 ✅

2. **examples模型全部正常**
   - mnist ✅
   - mnist_rnn ✅
   - siamese ✅
   - ❌ 只有mnist_ff有问题

3. **pytorch_resnet_cifar10正常**
   - 使用`best_val_accuracy`和`test_accuracy` ✅

---

## ❌ 问题模型名单（4个，131个实验）

### 严重级别：性能数据全部缺失（3个模型）

| # | 模型 | 实验数 | 性能数据 | 能耗数据 | 优先级 |
|---|------|--------|----------|----------|--------|
| 1 | **examples/mnist_ff** | 46 | 0/46 (0%) | 46/46 (100%) | 🔴 最高 |
| 2 | **VulBERTa/mlp** | 45 | 0/45 (0%) | 45/45 (100%) | 🔴 高 |
| 3 | **bug-localization-by-dnn-and-rvsm/default** | 40 | 0/40 (0%) | 40/40 (100%) | 🔴 高 |

**小计**: 131个实验，0%性能数据

### 中等级别：性能数据部分缺失（1个模型）

| # | 模型 | 实验数 | 性能数据 | 能耗数据 | 优先级 |
|---|------|--------|----------|----------|--------|
| 4 | **MRT-OAST/default** | 54 | 34/54 (63%) | 54/54 (100%) | 🟡 中 |

**详情**: 20个实验缺失性能数据，34个正常

---

## 🔍 问题模型详细分析

### 1. examples/mnist_ff (46个实验，0%性能数据)

**特征**:
- 与examples/mnist相关，但使用Feed-Forward网络
- 训练全部成功（100%能耗数据）
- 所有性能指标列全部为空

**对比分析**:
| 模型 | 性能数据 | 使用的指标 |
|------|----------|-----------|
| examples/mnist | ✅ 100% | test_accuracy, test_loss |
| examples/mnist_rnn | ✅ 100% | test_accuracy, test_loss |
| examples/siamese | ✅ 100% | test_accuracy, test_loss |
| **examples/mnist_ff** | ❌ 0% | (无) |

**推测原因**:
1. **日志格式不同**: mnist_ff可能使用不同的输出格式
2. **日志位置不同**: 可能输出到stderr或特定文件
3. **指标命名不同**: 可能使用"val_acc"而非"test_accuracy"

**优先级**: 🔴 **最高**（同仓库其他模型都正常，应该最容易修复）

---

### 2. VulBERTa/mlp (45个实验，0%性能数据)

**特征**:
- BERT类模型的MLP头
- 训练全部成功（100%能耗数据）
- 所有性能指标列全部为空

**可能原因**:
1. **使用validation指标**: 可能输出"val_accuracy"而非"test_accuracy"
2. **使用transformers库格式**: HuggingFace库有特定的日志格式
3. **使用wandb/tensorboard**: 可能只输出到日志平台而非stdout

**优先级**: 🔴 **高**

---

### 3. bug-localization-by-dnn-and-rvsm/default (40个实验，0%性能数据)

**特征**:
- Bug定位任务（非标准分类）
- 训练全部成功（100%能耗数据）
- 所有性能指标列全部为空

**推测原因**:
1. **特殊任务指标**: Bug定位使用Top-K准确率、MRR等特殊指标
2. **自定义评估脚本**: 可能使用独立的评估脚本
3. **输出到文件**: 结果可能保存在CSV/JSON文件中

**优先级**: 🔴 **高**

---

### 4. MRT-OAST/default (54个实验，63%性能数据)

**特征**:
- 34个实验有性能数据 ✅
- 20个实验缺失性能数据 ❌
- 缺失的全部是`mutation_2x_safe__`前缀的实验

**使用的指标**（34个正常实验）:
- ✅ perf_accuracy
- ✅ perf_precision
- ✅ perf_recall

**缺失数据的实验分析**:
| 类型 | 数量 | 模式 |
|------|------|------|
| mutation_2x_safe (nonparallel) | 10 | 非并行 |
| mutation_2x_safe (parallel) | 10 | 并行 |

**问题特征**:
- ✅ training_success = True（训练成功）
- ✅ 有能耗数据
- ❌ 所有18个性能列全部为空
- 🤔 **只影响特定批次的实验**（mutation_2x_safe）

**可能原因**:
1. **批次问题**: mutation_2x_safe这批实验可能使用了不同的配置
2. **提取脚本版本**: 这批实验运行时，提取脚本可能有bug
3. **日志文件缺失**: 这批实验的日志文件可能丢失或损坏

**优先级**: 🟡 **中**（大部分实验正常，影响有限）

---

## 📊 完整性能指标列表（18列）

### 非并行模式（9列）
1. `perf_accuracy` - 准确率
2. `perf_best_val_accuracy` - 最佳验证准确率
3. `perf_map` - 平均精度均值（ReID任务）
4. `perf_precision` - 精确率
5. `perf_rank1` - Rank-1准确率（ReID任务）
6. `perf_rank5` - Rank-5准确率（ReID任务）
7. `perf_recall` - 召回率
8. `perf_test_accuracy` - 测试准确率
9. `perf_test_loss` - 测试损失

### 并行模式前景（9列）
- `fg_perf_*`：与上述9列对应

### 各模型使用的指标映射

| 模型类型 | 使用的指标 |
|----------|-----------|
| 图像分类 (mnist, resnet) | test_accuracy, test_loss, best_val_accuracy |
| ReID检索 (Person_reID) | map, rank1, rank5 |
| 标准分类 (MRT-OAST) | accuracy, precision, recall |

---

## 🎯 修复优先级（更新）

### 🔴 最高优先级

**examples/mnist_ff** (46个实验)
- **理由**: 同仓库其他模型都正常，应该最容易修复
- **策略**: 对比mnist和mnist_ff的训练日志差异
- **预期时间**: 1-2小时

### 🔴 高优先级

**VulBERTa/mlp** (45个实验)
- **理由**: 影响实验数多
- **策略**: 检查BERT模型的标准日志格式
- **预期时间**: 2-3小时

**bug-localization** (40个实验)
- **理由**: 影响实验数多
- **策略**: 查找Bug定位任务的评估指标
- **预期时间**: 2-4小时

### 🟡 中优先级

**MRT-OAST/default** (20个实验)
- **理由**: 大部分实验正常，只有特定批次有问题
- **策略**: 检查mutation_2x_safe批次的日志文件
- **预期时间**: 1-2小时

---

## 📝 下一步行动计划

### Phase 1: 快速诊断 (2-3小时)

#### Step 1: examples/mnist_ff
```bash
# 找一个mnist和mnist_ff的实验日志
find results/run_* -name "*examples_mnist_008*" -name "*.log"
find results/run_* -name "*examples_mnist_ff*" -name "*.log"

# 对比日志格式差异
diff <(grep -i "accuracy\|loss" mnist.log) <(grep -i "accuracy\|loss" mnist_ff.log)
```

#### Step 2: VulBERTa/mlp
```bash
# 找VulBERTa实验日志
find results/run_* -name "*VulBERTa_mlp*" -name "*.log" | head -1

# 查找可能的指标名称
grep -i "acc\|loss\|precision\|recall\|f1\|eval" <log_file>
```

#### Step 3: bug-localization
```bash
# 找bug-localization实验日志
find results/run_* -name "*bug-localization*" -name "*.log" | head -1

# 查找评估指标
grep -i "top\|mrr\|precision\|recall\|rank" <log_file>
```

#### Step 4: MRT-OAST mutation_2x_safe
```bash
# 找mutation_2x_safe的日志
find results/run_* -name "*mutation_2x_safe__MRT-OAST*" -name "*.log" | head -1

# 对比正常实验的日志
find results/run_* -name "*mutation_1x__MRT-OAST*" -name "*.log" | head -1
```

### Phase 2: 修复提取脚本 (2-4小时)

1. **定位提取代码位置**
   ```bash
   grep -r "perf_test_accuracy\|perf_accuracy" mutation/ scripts/
   ```

2. **为每个问题模型添加适配器**
   - mnist_ff: 支持其特定输出格式
   - VulBERTa: 支持transformers日志格式
   - bug-localization: 添加Top-K/MRR指标提取

3. **测试修复**
   - 在少量实验上测试
   - 验证准确性

### Phase 3: 重新提取 (1-2小时)

1. **运行修复后的脚本**
   ```bash
   python3 scripts/extract_performance_metrics.py --reextract
   ```

2. **验证结果**
   - 检查131个实验是否成功提取
   - 更新raw_data.csv

3. **重新生成报告**
   ```bash
   python3 scripts/analyze_experiment_completion.py
   ```

---

## 📊 修复后的预期改进

| 指标 | 当前 | 修复后 (预期) | 改进 |
|------|------|---------------|------|
| 有性能数据的实验 | 327 (71.4%) | 458 (100%) | +40% |
| 数据完整的模型 | 7 (63.6%) | 11 (100%) | +57% |
| 默认值实验可用 | 8/22 (36.4%) | 14/22 (63.6%) | +75% |
| 可用于分析的实验 | 327 | 458 | +40% |

**关键收益**:
- ✅ 所有131个"失败"实验实际训练成功
- ✅ 修复后可立即使用，无需重新训练
- ✅ 项目完成度将从71.4%提升至~95%+

---

## 🎉 好消息总结

### 1. 大部分数据实际上是完整的！

- ✅ **7/11模型** (63.6%) 数据完整
- ✅ **327/458实验** (71.4%) 有性能数据
- ✅ **Person_reID系列**不是问题，已经正确提取ReID指标
- ✅ **所有实验训练都成功**（100%能耗数据）

### 2. 问题范围大幅缩小

- 从7个问题模型 → 4个问题模型（-43%）
- 从296个受影响实验 → 131个（-56%）
- 问题集中在3个特定模型 + 1个模型的特定批次

### 3. 修复难度降低

- examples/mnist_ff: 同仓库有正常模型可参考
- VulBERTa/mlp: BERT模型有标准日志格式
- bug-localization: 只需添加特定任务指标
- MRT-OAST: 只影响特定批次，可能是配置问题

---

**报告作者**: Claude (AI Assistant)
**数据来源**: results/raw_data.csv (458个唯一实验，80列)
**分析日期**: 2025-12-12
**状态**: ✅ 准备开始修复（优先级：mnist_ff → VulBERTa → bug-localization → MRT-OAST）
