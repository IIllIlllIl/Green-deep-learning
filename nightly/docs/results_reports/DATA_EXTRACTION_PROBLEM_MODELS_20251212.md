# 数据提取问题模型名单

**日期**: 2025-12-12
**目的**: 识别性能数据提取失败的模型，优先修复
**数据源**: results/raw_data.csv (458个唯一实验)
**分析范围**: 11个模型，全部实验

---

## 📊 总体统计

| 指标 | 数值 |
|------|------|
| 总模型数 | 11 |
| **性能数据全缺的模型** | **7个 (63.6%)** |
| 性能数据完整的模型 | 4个 (36.4%) |
| 能耗数据全缺的模型 | 0个 (0%) |

**关键发现**:
- ✅ **能耗数据100%完整** - 能耗监控系统工作正常
- ❌ **7个模型性能数据0%** - 性能指标提取系统存在严重问题

---

## ❌ 问题模型名单（按实验数量排序）

### 严重级别：性能数据全部缺失

| # | 模型 | 实验数 | 性能数据 | 能耗数据 | 优先级 |
|---|------|--------|----------|----------|--------|
| 1 | **MRT-OAST/default** | 54 | 0/54 (0%) | 54/54 (100%) | 🔴 高 |
| 2 | **examples/mnist_ff** | 46 | 0/46 (0%) | 46/46 (100%) | 🔴 高 |
| 3 | **VulBERTa/mlp** | 45 | 0/45 (0%) | 45/45 (100%) | 🔴 高 |
| 4 | **Person_reID_baseline_pytorch/densenet121** | 43 | 0/43 (0%) | 43/43 (100%) | 🔴 高 |
| 5 | **bug-localization-by-dnn-and-rvsm/default** | 40 | 0/40 (0%) | 40/40 (100%) | 🔴 高 |
| 6 | **Person_reID_baseline_pytorch/hrnet18** | 34 | 0/34 (0%) | 34/34 (100%) | 🟡 中 |
| 7 | **Person_reID_baseline_pytorch/pcb** | 34 | 0/34 (0%) | 34/34 (100%) | 🟡 中 |

**受影响实验总数**: 296个 (64.6%)

---

## ✅ 正常模型（性能数据完整）

| 模型 | 实验数 | 性能数据 | 能耗数据 |
|------|--------|----------|----------|
| examples/mnist | 40 | 40/40 (100%) | 40/40 (100%) |
| examples/mnist_rnn | 43 | 43/43 (100%) | 43/43 (100%) |
| examples/siamese | 40 | 40/40 (100%) | 40/40 (100%) |
| pytorch_resnet_cifar10/resnet20 | 39 | 39/39 (100%) | 39/39 (100%) |

**正常实验总数**: 162个 (35.4%)

---

## 🔍 问题模型详细分析

### 1. MRT-OAST/default (54个实验)

**特征**:
- 仓库: MRT-OAST
- 模型: default
- 参数数量: 5个 (dropout, epochs, learning_rate, seed, weight_decay)

**问题**:
- 所有54个实验性能数据全部缺失
- 能耗数据100%完整 → 训练过程成功完成
- **推测**: 性能指标提取逻辑不适配此模型的输出格式

**验证示例**: `default__MRT-OAST_default_001`
- 能耗: ✅ CPU=39987.66J, GPU=331876.33J
- 性能: ❌ 所有perf_*列为空

---

### 2. examples/mnist_ff (46个实验)

**特征**:
- 仓库: examples
- 模型: mnist_ff (Feed-Forward网络)
- 参数数量: 4个 (batch_size, epochs, learning_rate, seed)

**问题**:
- 所有46个实验性能数据全部缺失
- 能耗数据100%完整 → 训练过程成功完成
- **推测**: 与examples/mnist (成功) 不同的输出格式

**对比**:
| 模型 | 性能数据 | 说明 |
|------|----------|------|
| examples/mnist | ✅ 100% | 标准MNIST CNN |
| examples/mnist_ff | ❌ 0% | Feed-Forward变体 |
| examples/mnist_rnn | ✅ 100% | RNN变体 |

**关键差异**: mnist_ff可能使用不同的输出格式或日志结构

---

### 3. VulBERTa/mlp (45个实验)

**特征**:
- 仓库: VulBERTa
- 模型: mlp
- 参数数量: 4个 (epochs, learning_rate, seed, weight_decay)

**问题**:
- 所有45个实验性能数据全部缺失
- 能耗数据100%完整
- **推测**: BERT类模型可能使用不同的性能指标命名

**可能的原因**:
1. 输出格式：可能使用"val_accuracy"而非"test_accuracy"
2. 日志位置：可能输出到不同的文件路径
3. 指标命名：可能使用"precision@k"等特殊命名

---

### 4-7. Person_reID系列 (111个实验)

**三个模型**:
- Person_reID_baseline_pytorch/densenet121 (43个)
- Person_reID_baseline_pytorch/hrnet18 (34个)
- Person_reID_baseline_pytorch/pcb (34个)

**共同特征**:
- 所有实验性能数据全部缺失
- 能耗数据100%完整
- 训练代码来自Person Re-identification项目

**关键指标差异**:
- 标准分类: accuracy, precision, recall
- **ReID任务**: mAP (mean Average Precision), Rank-1, Rank-5, CMC

**推测**: 提取脚本查找的是标准分类指标，但ReID输出的是检索指标

---

### 8. bug-localization-by-dnn-and-rvsm/default (40个实验)

**特征**:
- 仓库: bug-localization-by-dnn-and-rvsm
- 模型: default
- 参数数量: 4个 (alpha, kfold, max_iter, seed)

**问题**:
- 所有40个实验性能数据全部缺失
- 能耗数据100%完整
- **任务类型**: Bug定位（非标准分类/回归）

**推测**: 输出的是定位准确率、MRR等特殊指标，不是标准的accuracy/loss

---

## 🎯 问题根源分析

### 1. 提取脚本假设标准输出格式

**当前假设**:
- 指标名称: `test_accuracy`, `test_loss`, `train_accuracy`等
- 日志格式: 标准PyTorch/TensorFlow日志
- 输出位置: `stdout`或特定日志文件

**实际情况**:
- 不同模型使用不同的指标名称
- 不同框架使用不同的日志格式
- 某些模型输出到自定义文件

### 2. 特定任务的特殊指标

| 任务类型 | 标准指标 | 实际使用的指标 |
|----------|----------|----------------|
| 分类 | accuracy, precision | ✅ 标准 |
| 检索(ReID) | accuracy | ❌ 使用mAP, Rank-1, CMC |
| Bug定位 | accuracy | ❌ 使用Top-K准确率, MRR |
| 语言模型 | accuracy | ❌ 使用perplexity, BLEU |

### 3. 日志输出位置不统一

**可能的位置**:
1. 标准输出 (stdout) - ✅ 大多数examples模型
2. 标准错误 (stderr) - 某些框架
3. 日志文件 (logs/*.log) - 某些项目
4. TensorBoard文件 - 深度学习项目
5. 自定义JSON/CSV文件 - 特殊项目

---

## 🔧 修复优先级

### 高优先级（影响最大）

1. **MRT-OAST/default** (54个实验)
   - 检查: 训练日志输出位置
   - 查找: 实际使用的指标名称
   - 修复: 更新提取逻辑

2. **examples/mnist_ff** (46个实验)
   - 对比: 与examples/mnist的差异
   - 检查: 是否输出到不同文件
   - 修复: 适配mnist_ff的输出格式

3. **VulBERTa/mlp** (45个实验)
   - 检查: BERT模型的标准输出格式
   - 查找: validation指标位置
   - 修复: 支持BERT类模型

### 中优先级

4. **Person_reID系列** (111个实验)
   - 理解: ReID任务的评估指标
   - 添加: mAP, Rank-1, Rank-5提取支持
   - 映射: 将ReID指标映射到perf_*列

5. **bug-localization** (40个实验)
   - 理解: Bug定位的评估方式
   - 添加: Top-K, MRR指标提取
   - 映射: 到标准性能列

---

## 📝 下一步行动计划

### Phase 1: 诊断 (1-2小时)

对每个问题模型：
1. **找一个实验的完整日志**
   ```bash
   # 示例
   find results/run_* -name "*MRT-OAST_default_001*" -type f
   ```

2. **手动检查日志内容**
   - 查找性能指标在哪里
   - 确认指标的确切名称
   - 记录日志格式

3. **记录实际输出格式**
   - 创建每个模型的输出格式文档
   - 标注关键字和位置

### Phase 2: 修复提取脚本 (2-4小时)

1. **定位提取代码**
   ```bash
   # 查找性能提取逻辑
   grep -r "test_accuracy\|train_accuracy" mutation/ scripts/
   ```

2. **添加模型特定的提取规则**
   - 为每个问题模型添加适配器
   - 支持多种指标命名方案
   - 处理不同的日志格式

3. **测试修复效果**
   - 在少量实验上测试
   - 验证提取的准确性
   - 确保不破坏现有的正常模型

### Phase 3: 重新提取数据 (1-2小时)

1. **运行修复后的提取脚本**
   ```bash
   # 重新提取296个实验的性能数据
   python3 scripts/extract_performance_metrics.py --models MRT-OAST,VulBERTa,Person_reID,bug-localization,examples/mnist_ff
   ```

2. **验证数据完整性**
   - 检查是否成功提取
   - 抽样验证准确性
   - 更新raw_data.csv

3. **重新生成分析报告**
   ```bash
   python3 scripts/analyze_experiment_completion.py
   ```

---

## 🔍 诊断命令速查

### 查找实验日志
```bash
# MRT-OAST
find results/run_* -name "*MRT-OAST*" -name "*.log" | head -3

# Person_reID
find results/run_* -name "*Person_reID*densenet121*" -name "*.log" | head -3

# VulBERTa
find results/run_* -name "*VulBERTa*" -name "*.log" | head -3

# mnist_ff
find results/run_* -name "*mnist_ff*" -name "*.log" | head -3
```

### 查看日志内容
```bash
# 查找包含accuracy/loss的行
grep -i "accuracy\|loss\|error\|precision\|recall\|mAP\|rank" <log_file>

# 查看最后100行（通常包含最终结果）
tail -100 <log_file>
```

### 查找提取代码
```bash
# 查找性能提取相关代码
grep -rn "perf_test\|test_accuracy\|extract.*performance" mutation/ scripts/

# 查找JSON解析代码
grep -rn "experiment\.json\|performance" scripts/
```

---

## 📊 预期影响

### 修复后的改进

| 指标 | 修复前 | 修复后 (预期) | 改进 |
|------|--------|---------------|------|
| 有性能数据的实验 | 162 (35.4%) | 458 (100%) | +183% |
| 可用于分析的模型 | 4 (36.4%) | 11 (100%) | +175% |
| 默认值实验可用 | 4/22 (18.2%) | 14/22 (63.6%) | +250% |
| 参数-模式组合完成度 | 28/90 (31.1%) | 提升至70%+ | +126% |

**关键收益**:
- ✅ 所有296个"失败"实验实际训练成功（有能耗数据）
- ✅ 修复后可立即使用，无需重新训练
- ✅ 大幅提升实验完成度评估准确性

---

## 总结

### 问题本质

**不是训练失败，是数据提取失败**

- ✅ 训练成功率: 100% (所有实验都有能耗数据)
- ❌ 性能提取成功率: 35.4% (只有4个模型)
- 🎯 **核心问题**: 提取脚本假设统一的输出格式

### 7个问题模型

1. MRT-OAST/default (54个实验) - 🔴 高优先级
2. examples/mnist_ff (46个实验) - 🔴 高优先级
3. VulBERTa/mlp (45个实验) - 🔴 高优先级
4. Person_reID_baseline_pytorch/densenet121 (43个实验) - 🔴 高优先级
5. bug-localization-by-dnn-and-rvsm/default (40个实验) - 🔴 高优先级
6. Person_reID_baseline_pytorch/hrnet18 (34个实验) - 🟡 中优先级
7. Person_reID_baseline_pytorch/pcb (34个实验) - 🟡 中优先级

### 下一步

1. **立即**: 诊断每个模型的实际输出格式
2. **今天**: 修复提取脚本，支持多种格式
3. **明天**: 重新提取296个实验的性能数据

---

**报告作者**: Claude (AI Assistant)
**数据来源**: results/raw_data.csv (458个唯一实验)
**分析日期**: 2025-12-12
**状态**: ✅ 准备开始修复
