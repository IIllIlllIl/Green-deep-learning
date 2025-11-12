# 模型对抽样策略对比分析
# Model Pair Sampling Strategy Comparison

**日期**: 2025-11-11
**目的**: 对比两种抽样策略的结果，帮助选择最终方案

---

## 📊 两种策略对比

### 策略A: 混合策略（实用性 + 分层）
**来源**: `concurrent_hp_study_experimental_design.md`
**原理**: 6个实用性抽样（优先有数据、安全） + 6个分层抽样（覆盖不同类型）

### 策略B: 纯显存分层抽样
**来源**: `memory_stratified_sample.json`
**原理**: 完全基于显存占用分层，按比例分配（1+5+5+1=12）

---

## 🔍 策略A详细组合（混合策略）

### 实用性抽样部分（6个）- 优先安全+有历史数据

| # | 模型A | 模型B | 显存 | GPU利用率 | 特点 |
|---|------|------|------|----------|------|
| 1 | examples_mnist | Person_reID_baseline_pytorch_densenet121 | 3750MB | 84% | ⭐最佳互补 |
| 2 | examples_mnist | MRT-OAST_default | 2400MB | 105% | ⭐完美互补 |
| 3 | Person_reID_baseline_pytorch_densenet121 | MRT-OAST_default | 5250MB | 165% | ⚠️竞争型 |
| 4 | examples_mnist | examples_mnist_rnn | 900MB | 62% | 同族模型 |
| 5 | examples_mnist | examples_mnist_ff | 900MB | 62% | 同族模型 |
| 6 | examples_mnist | pytorch_resnet_cifar10_resnet20 | 1170MB | 62% | 跨领域 |

### 分层抽样部分（6个）- 覆盖不同类型

| # | 模型A | 模型B | 显存 | GPU利用率 | 特点 |
|---|------|------|------|----------|------|
| 7 | pytorch_resnet_cifar10_resnet44 | pytorch_resnet_cifar10_resnet56 | 2430MB | 100% | 同族扩展 |
| 8 | examples_mnist | VulBERTa_cnn | 1800MB | 62% | 跨领域 |
| 9 | pytorch_resnet_cifar10_resnet44 | Person_reID_baseline_pytorch_hrnet18 | 3330MB | 100% | 中型组合 |
| 10 | examples_mnist | bug-localization-by-dnn-and-rvsm_default | 2250MB | 62% | 跨领域 |
| 11 | examples_mnist | Person_reID_baseline_pytorch_hrnet18 | 2700MB | 62% | 互补型 |
| 12 | pytorch_resnet_cifar10_resnet44 | Person_reID_baseline_pytorch_densenet121 | 4380MB | 122% | 混合型 |

**显存分布**:
- 超低(<1.5GB): 2个 (组4, 5)
- 低(1.5-3GB): 6个 (组2, 6, 7, 8, 10, 11)
- 中(3-5GB): 3个 (组1, 9, 12)
- 高(5-7GB): 1个 (组3)

---

## 🔍 策略B详细组合（纯显存分层）

### 各层抽样结果

| # | 模型A | 模型B | 显存 | 层级 | 特点 |
|---|------|------|------|------|------|
| 1 | examples_mnist_ff | pytorch_resnet_cifar10_resnet20 | 1300MB | ultra_low | 两个小模型 |
| 2 | examples_mnist | bug-localization-by-dnn-and-rvsm_default | 2500MB | low | 跨领域 |
| 3 | examples_mnist | pytorch_resnet_cifar10_resnet44 | 1700MB | low | 视觉任务 |
| 4 | pytorch_resnet_cifar10_resnet44 | VulBERTa_cnn | 2700MB | low | 跨领域 |
| 5 | examples_mnist_rnn | bug-localization-by-dnn-and-rvsm_default | 2500MB | low | 跨领域 |
| 6 | examples_mnist_rnn | VulBERTa_mlp | 2000MB | low | 跨领域 |
| 7 | pytorch_resnet_cifar10_resnet44 | Person_reID_baseline_pytorch_densenet121 | 4700MB | medium | 视觉任务 |
| 8 | pytorch_resnet_cifar10_resnet20 | MRT-OAST_default | 3000MB | medium | 视觉任务 |
| 9 | VulBERTa_mlp | examples_siamese | 3000MB | medium | 跨领域 |
| 10 | pytorch_resnet_cifar10_resnet20 | Person_reID_baseline_pytorch_densenet121 | 4300MB | medium | 视觉任务 |
| 11 | MRT-OAST_default | examples_siamese | 3700MB | medium | 视觉任务 |
| 12 | Person_reID_baseline_pytorch_densenet121 | examples_word_lm | 5000MB | high | 跨领域 |

**显存分布**:
- 超低(<1.5GB): 1个 (16.7%抽样率)
- 低(1.5-3GB): 5个 (10.0%抽样率)
- 中(3-5GB): 5个 (9.1%抽样率)
- 高(5-7GB): 1个 (11.1%抽样率)

**显存统计**:
- 最小: 1300MB
- 最大: 5000MB
- 平均: 3033MB
- 中位数: 2850MB

---

## 📈 两种策略对比分析

### 1. 显存覆盖对比

| 显存层级 | 策略A（混合） | 策略B（显存分层） |
|---------|-------------|-----------------|
| 超低(<1.5GB) | 2个 (17%) | 1个 (8%) |
| 低(1.5-3GB) | 6个 (50%) | 5个 (42%) |
| 中(3-5GB) | 3个 (25%) | 5个 (42%) |
| 高(5-7GB) | 1个 (8%) | 1个 (8%) |

**分析**:
- 策略A更偏向低显存组合（67%在3GB以下）
- 策略B更均衡（50% vs 50%）

### 2. 模型覆盖对比

| 模型 | 策略A出现次数 | 策略B出现次数 |
|------|-------------|-------------|
| examples_mnist | 7次 | 2次 |
| examples_mnist_rnn | 1次 | 2次 |
| examples_mnist_ff | 1次 | 1次 |
| pytorch_resnet_cifar10_resnet20 | 1次 | 2次 |
| pytorch_resnet_cifar10_resnet44 | 3次 | 3次 |
| pytorch_resnet_cifar10_resnet56 | 1次 | 0次 |
| Person_reID_baseline_pytorch_densenet121 | 3次 | 2次 |
| Person_reID_baseline_pytorch_hrnet18 | 1次 | 0次 |
| MRT-OAST_default | 1次 | 2次 |
| VulBERTa_cnn | 1次 | 1次 |
| VulBERTa_mlp | 0次 | 2次 |
| bug-localization-by-dnn-and-rvsm_default | 1次 | 2次 |
| examples_siamese | 0次 | 2次 |
| examples_word_lm | 0次 | 1次 |

**分析**:
- 策略A高度依赖examples_mnist（7次），可能过度使用
- 策略B模型分布更均匀，最多3次

### 3. 领域覆盖对比

| 领域组合 | 策略A | 策略B |
|---------|------|------|
| vision+vision | 9个 | 7个 |
| vision+code | 2个 | 4个 |
| vision+nlp | 0个 | 1个 |
| code+code | 0个 | 0个 |

**分析**:
- 策略A更聚焦视觉领域（75%）
- 策略B跨领域更均衡（33%跨领域）

### 4. 实用性对比

#### 策略A优势：
- ✅ 优先包含有历史数据的模型（前6个都有数据）
- ✅ 包含已知的最佳互补组合
- ✅ 包含故意的竞争型组合用于对比
- ✅ 低显存组合更多，实验成功率更高

#### 策略B优势：
- ✅ 统计上更无偏，代表性更强
- ✅ 模型分布更均匀，避免过度使用某个模型
- ✅ 显存层级覆盖更均衡
- ✅ 跨领域组合更多，泛化性更好

#### 策略A劣势：
- ⚠️ examples_mnist使用7次，可能过度依赖
- ⚠️ 偏向低显存，中高显存覆盖不足
- ⚠️ 主观性强，可能引入偏差

#### 策略B劣势：
- ⚠️ 未考虑历史数据可用性
- ⚠️ 未考虑GPU利用率互补性
- ⚠️ 随机抽样，可能遗漏关键组合（如MNIST+DenseNet121）

---

## 💡 最终推荐

### 推荐方案：混合两种策略的优点

#### 方案C: 优化混合策略（推荐⭐⭐⭐⭐⭐）

**核心思想**: 保留策略A的核心组合，用策略B的部分组合替换examples_mnist过度使用的情况

**具体方案**（12个组合）:

| # | 来源 | 模型A | 模型B | 显存 | 理由 |
|---|-----|------|------|------|------|
| 1 | 策略A | examples_mnist | Person_reID_baseline_pytorch_densenet121 | 3750MB | ⭐最佳互补，必须保留 |
| 2 | 策略A | examples_mnist | MRT-OAST_default | 2400MB | ⭐完美互补，必须保留 |
| 3 | 策略A | Person_reID_baseline_pytorch_densenet121 | MRT-OAST_default | 5250MB | ⚠️竞争型，必须保留 |
| 4 | 策略B | pytorch_resnet_cifar10_resnet44 | VulBERTa_cnn | 2700MB | 替代，减少MNIST使用 |
| 5 | 策略B | examples_mnist_rnn | bug-localization-by-dnn-and-rvsm_default | 2500MB | 替代，增加RNN覆盖 |
| 6 | 策略B | examples_mnist_rnn | VulBERTa_mlp | 2000MB | 替代，增加跨领域 |
| 7 | 策略A | pytorch_resnet_cifar10_resnet44 | pytorch_resnet_cifar10_resnet56 | 2430MB | 同族扩展 |
| 8 | 策略B | VulBERTa_mlp | examples_siamese | 3000MB | 代码+视觉 |
| 9 | 策略A | pytorch_resnet_cifar10_resnet44 | Person_reID_baseline_pytorch_hrnet18 | 3330MB | 中型组合 |
| 10 | 策略B | MRT-OAST_default | examples_siamese | 3700MB | 增加MRT-OAST使用 |
| 11 | 策略B | Person_reID_baseline_pytorch_densenet121 | examples_word_lm | 5000MB | 跨领域NLP |
| 12 | 策略A | pytorch_resnet_cifar10_resnet44 | Person_reID_baseline_pytorch_densenet121 | 4380MB | 混合型 |

**优化后的模型使用频率**:
- examples_mnist: 2次（从7次减少）✅
- examples_mnist_rnn: 2次（从1次增加）✅
- pytorch_resnet_cifar10_resnet44: 4次（保持）
- Person_reID_baseline_pytorch_densenet121: 4次（从3次增加）
- MRT-OAST_default: 2次（从1次增加）✅
- VulBERTa系列: 2次（从1次增加）✅

**显存分布**:
- 超低(<1.5GB): 0个
- 低(1.5-3GB): 6个 (50%)
- 中(3-5GB): 4个 (33%)
- 高(5-7GB): 2个 (17%)

**优势**:
1. ✅ 保留了最关键的互补型和竞争型组合
2. ✅ 模型使用更均衡，避免过度依赖
3. ✅ 跨领域覆盖更好（增加NLP、代码分析）
4. ✅ 显存覆盖更合理
5. ✅ 兼顾实用性和代表性

---

## 🎯 实施建议

### 短期实施（Pilot阶段）
**推荐3个组合进行pilot测试**:
1. examples_mnist + Person_reID_baseline_pytorch_densenet121 (最佳互补)
2. examples_mnist + MRT-OAST_default (完美互补)
3. Person_reID_baseline_pytorch_densenet121 + MRT-OAST_default (竞争型)

**时间**: 20-25小时
**目的**: 验证实验设计可行性，发现潜在问题

### 长期实施（全量实验）
**使用方案C（优化混合策略）的12个组合**

**时间**: 40-50小时（串行），25-30小时（并发优化）
**产出**: 132次训练数据，完整的并发训练超参数影响分析

---

## 📝 结论

### 如果选择策略A（混合策略）:
- 优先保证实验成功率
- 聚焦已知关键组合
- 适合资源有限、时间紧张的场景

### 如果选择策略B（纯显存分层）:
- 追求统计严谨性
- 需要更全面的模型覆盖
- 适合长期深入研究

### 如果选择方案C（优化混合）⭐推荐:
- 兼顾实用性和代表性
- 模型使用更均衡
- **最推荐用于本研究**

---

**文档版本**: 1.0
**生成时间**: 2025-11-11
**相关文件**:
- `memory_stratified_sample.json` - 策略B抽样结果
- `concurrent_hp_study_experimental_design.md` - 策略A完整设计
- `stratified_sampling_strategies_guide.md` - 分层策略详解
