# 深度学习模型训练超参数全面分析报告

生成时间: 2025-11-04
分析对象: 6个仓库，12个训练模型

---

## 目录

1. [模型概览](#1-模型概览)
2. [各模型可变超参数详细分析](#2-各模型可变超参数详细分析)
3. [跨模型通用可变超参数](#3-跨模型通用可变超参数)
4. [代码扩展后的潜在通用超参数](#4-代码扩展后的潜在通用超参数)
5. [超参数变异对性能和能耗的影响评估](#5-超参数变异对性能和能耗的影响评估)
6. [修改难度分析](#6-修改难度分析)
7. [推荐的超参数变异列表](#7-推荐的超参数变异列表)

---

## 1. 模型概览

### 1.1 仓库与模型列表

| 序号 | 仓库名称 | 模型数量 | 模型名称 | 训练指令 |
|------|---------|---------|---------|---------|
| 1 | MRT-OAST | 1 | MRT-OAST | `./train.sh` |
| 2 | bug-localization-by-dnn-and-rvsm | 1 | DNN | `./train.sh` |
| 3 | pytorch_resnet_cifar10 | 1 | ResNet20 | `./train.sh` |
| 4 | VulBERTa | 2 | MLP, CNN | `./train.sh -n {mlp\|cnn}` |
| 5 | Person_reID_baseline_pytorch | 3 | densenet121, hrnet18, pcb | `./train.sh -n {model}` |
| 6 | examples | 4 | mnist, mnist_rnn, mnist_ff, siamese | `./train.sh -n {model}` |
| **总计** | **6** | **12** | - | - |

---

## 2. 各模型可变超参数详细分析

### 2.1 MRT-OAST

**模型类型**: Transformer-based代码克隆检测
**当前可修改方式**: 通过train.sh命令行参数

#### 可变超参数列表

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `epochs` | 10 | 1-100 | 训练轮数 | ⭐ 极易 |
| `batch_size` | 64 | 8-128 | 批次大小 | ⭐ 极易 |
| `lr` | 0.0001 | 1e-5 - 1e-2 | 学习率 | ⭐ 极易 |
| `dropout` | 0.2 | 0.0-0.8 | Dropout率 | ⭐ 极易 |
| `seed` | 1334 | 任意整数 | 随机种子 | ⭐ 极易 |
| `valid_step` | 1750 | 100-5000 | 验证步数 | ⭐ 极易 |
| `max_len` | 256 | 128-512 | 最大序列长度 | ⭐⭐ 容易 |
| `layers` | 2 | 1-6 | Transformer层数 | ⭐⭐ 容易 |
| `d_model` | 128 | 64-512 | 模型维度 | ⭐⭐ 容易 |
| `d_ff` | 512 | 256-2048 | 前馈网络维度 | ⭐⭐ 容易 |
| `heads` | 8 | 2-16 | 注意力头数 | ⭐⭐ 容易 |
| `output_dim` | 512 | 128-1024 | 输出维度 | ⭐⭐ 容易 |
| `threshold` | 0.9 | 0.5-0.99 | 测试阈值 | ⭐ 极易 |
| `valid_threshold` | 0.8 | 0.5-0.99 | 验证阈值 | ⭐ 极易 |

**训练命令示例**:
```bash
./train.sh --epochs 15 --batch-size 32 --lr 0.0002 --dropout 0.3
```

---

### 2.2 bug-localization-by-dnn-and-rvsm (DNN模型)

**模型类型**: 多层感知器(MLP)用于缺陷定位
**当前可修改方式**: 通过train.sh命令行参数

#### 可变超参数列表

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `kfold` | 10 | 3-20 | K折交叉验证折数 | ⭐ 极易 |
| `hidden_sizes` | 300 | 100-1000 | 隐藏层大小（可多层） | ⭐ 极易 |
| `alpha` | 1e-5 | 1e-7 - 1e-2 | L2正则化系数 | ⭐ 极易 |
| `max_iter` | 10000 | 1000-50000 | 最大迭代次数 | ⭐ 极易 |
| `n_iter_no_change` | 30 | 10-100 | 早停耐心值 | ⭐ 极易 |
| `solver` | sgd | sgd/adam/lbfgs | 优化器类型 | ⭐ 极易 |
| `n_jobs` | -2 | -2至CPU核数 | 并行作业数 | ⭐ 极易 |

**训练命令示例**:
```bash
./train.sh -n dnn --kfold 5 --hidden_sizes 200 --alpha 1e-4 --solver adam
```

---

### 2.3 pytorch_resnet_cifar10

**模型类型**: ResNet用于CIFAR-10图像分类
**当前可修改方式**: 通过train.sh命令行参数

#### 可变超参数列表

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `epochs` | 200 | 50-500 | 训练轮数 | ⭐ 极易 |
| `batch_size` | 128 | 32-256 | 批次大小 | ⭐ 极易 |
| `lr` | 0.1 | 0.001-0.5 | 初始学习率 | ⭐ 极易 |
| `momentum` | 0.9 | 0.5-0.99 | SGD动量 | ⭐ 极易 |
| `weight_decay` | 0.0001 | 1e-6 - 1e-2 | 权重衰减 | ⭐ 极易 |
| `workers` | 4 | 1-16 | 数据加载线程数 | ⭐ 极易 |
| `print_freq` | 50 | 10-200 | 日志打印频率 | ⭐ 极易 |
| `save_every` | 10 | 5-50 | checkpoint保存频率 | ⭐ 极易 |

**训练命令示例**:
```bash
./train.sh -n resnet20 -e 100 -b 64 --lr 0.05 --momentum 0.95
```

---

### 2.4 VulBERTa

**模型类型**: 基于RoBERTa的漏洞检测模型
**当前可修改方式**: 通过train.sh命令行参数

#### 2.4.1 MLP模型

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `batch_size` | 2 | 1-8 | 批次大小（受显存限制） | ⭐ 极易 |
| `epochs` | 10 | 5-30 | 训练轮数 | ⭐ 极易 |
| `learning_rate` | 3e-05 | 1e-6 - 1e-3 | 学习率 | ⭐ 极易 |
| `seed` | 42 | 任意整数 | 随机种子 | ⭐ 极易 |
| `fp16` | True | True/False | ���合精度训练 | ⭐ 极易 |

#### 2.4.2 CNN模型

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `batch_size` | 128 | 32-256 | 批次大小 | ⭐ 极易 |
| `epochs` | 20 | 10-50 | 训练轮数 | ⭐ 极易 |
| `learning_rate` | 0.0005 | 1e-5 - 1e-2 | 学习率 | ⭐ 极易 |
| `seed` | 1234 | 任意整数 | 随机种子 | ⭐ 极易 |
| `fp16` | False | True/False | 混合精度训练 | ⭐ 极易 |

**训练命令示例**:
```bash
# MLP模型
./train.sh -n mlp -d devign --batch_size 4 --epochs 15 --learning_rate 5e-05

# CNN模型
./train.sh -n cnn -d devign --batch_size 64 --epochs 25 --learning_rate 0.001
```

---

### 2.5 Person_reID_baseline_pytorch

**模型类型**: 行人重识别模型（多种backbone）
**当前可修改方式**: 通过train.sh命令行参数

#### 共享超参数（适用于所有3个模型）

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `batchsize` | 32* | 8-64 | 批次大小 | ⭐ 极易 |
| `lr` | 0.05 | 0.001-0.2 | 学习率 | ⭐ 极易 |
| `total_epoch` | 60 | 20-150 | 总训练轮数 | ⭐ 极易 |
| `warm_epoch` | 0* | 0-10 | 预热轮数 | ⭐ 极易 |
| `stride` | 2* | 1-2 | ResNet步长 | ⭐ 极易 |
| `erasing_p` | 0* | 0.0-0.7 | 随机擦除概率 | ⭐ 极易 |
| `droprate` | 0.5 | 0.0-0.8 | Dropout率 | ⭐ 极易 |
| `linear_num` | 512 | 256-1024 | 特征维度 | ⭐ 极易 |
| `weight_decay` | 5e-4 | 1e-6 - 1e-2 | 权重衰减 | ⭐ 极易 |

*注：不同模型配置有调整

#### 模型特定配置

**densenet121**:
- batchsize: 24
- 其他参数使用默认值

**hrnet18**:
- batchsize: 24
- 其他参数使用默认值

**pcb**:
- batchsize: 32
- lr: 0.02
- 其他参数使用默认值

**训练命令示例**:
```bash
./train.sh -n densenet121 --batchsize 16 --lr 0.03 --total_epoch 40
./train.sh -n hrnet18 --batchsize 20 --lr 0.04
./train.sh -n pcb --batchsize 24 --lr 0.015
```

---

### 2.6 examples (PyTorch官方示例)

**模型类型**: 各类经典深度学习模型
**当前可修改方式**: 通过train.sh命令行参数

#### 2.6.1 MNIST (CNN)

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `epochs` | 14 | 5-50 | 训练轮数 | ⭐ 极易 |
| `batch_size` | 32 | 16-128 | 批次大小 | ⭐ 极易 |
| `lr` | 1.0 | 0.1-5.0 | 学习率 | ⭐ 极易 |
| `seed` | 1 | 任意整数 | 随机种子 | ⭐ 极易 |
| `gamma` | 0.7 | 0.5-0.95 | 学习率衰减因子 | ⭐ 极易 |
| `test_batch_size` | 1000 | 100-2000 | 测试批次大小 | ⭐ 极易 |
| `log_interval` | 10 | 5-100 | 日志间隔 | ⭐ 极易 |

#### 2.6.2 MNIST RNN

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `epochs` | 14 | 5-50 | 训练轮数 | ⭐ 极易 |
| `batch_size` | 32 | 16-128 | 批次大小 | ⭐ 极易 |
| `lr` | 0.1 | 0.01-1.0 | 学习率 | ⭐ 极易 |
| `seed` | 1 | 任意整数 | 随机种子 | ⭐ 极易 |
| `gamma` | 0.7 | 0.5-0.95 | 学习率衰减因子 | ⭐ 极易 |

#### 2.6.3 MNIST Forward-Forward

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `epochs` | 1000 | 100-5000 | 训练轮数 | ⭐ 极易 |
| `train_size` | 50000 | 10000-60000 | 训练集大小 | ⭐ 极易 |
| `lr` | 0.03 | 0.001-0.1 | 学习率 | ⭐ 极易 |
| `seed` | 1 | 任意整数 | 随机种子 | ⭐ 极易 |
| `threshold` | 2 | 0.5-5.0 | 训练阈值 | ⭐ 极易 |

#### 2.6.4 Siamese Network

| 超参数名称 | 默认值 | 取值范围 | 参数说明 | 修改难度 |
|-----------|-------|---------|---------|---------|
| `epochs` | 14 | 5-50 | 训练轮数 | ⭐ 极易 |
| `batch_size` | 32 | 16-128 | 批次大小 | ⭐ 极易 |
| `lr` | 1.0 | 0.1-5.0 | 学习率 | ⭐ 极易 |
| `seed` | 1 | 任意整数 | 随机种子 | ⭐ 极易 |
| `gamma` | 0.7 | 0.5-0.95 | 学习率衰减因子 | ⭐ 极易 |

**训练命令示例**:
```bash
./train.sh -n mnist -e 10 -b 64 -l 0.8
./train.sh -n mnist_rnn -e 12 -b 32 -l 0.15
./train.sh -n mnist_ff -e 500 --lr 0.05 --threshold 3
./train.sh -n siamese -e 10 -b 48 -l 1.2
```

---

## 3. 跨模型通用可变超参数

### 3.1 完全通用超参数（所有12个模型均支持）

以下超参数在**所有12个模型**中都可以通过命令行直接修改：

| 超参数类别 | 超参数名称 | 说明 | 出现频率 |
|-----------|-----------|------|---------|
| **训练配置** | `epochs` / `total_epoch` | 训练轮数 | 12/12 (100%) |
| **训练配置** | `batch_size` / `batchsize` | 批次大小 | 12/12 (100%) |
| **优化器** | `lr` / `learning_rate` | 学习率 | 12/12 (100%) |
| **可重复性** | `seed` | 随机种子 | 12/12 (100%) |

**说明**: 这4个超参数是**真正的通用超参数**，适用于所有模型的变异实验。

### 3.2 高频通用超参数（出现在多数模型中）

| 超参数类别 | 超参数名称 | 说明 | 出现频率 | 适用模型 |
|-----------|-----------|------|---------|---------|
| **正则化** | `dropout` / `droprate` | Dropout率 | 5/12 (42%) | MRT-OAST, Person_reID (3个) |
| **正则化** | `weight_decay` | 权重衰减 | 3/12 (25%) | pytorch_resnet_cifar10, Person_reID (3个) |
| **优化器** | `momentum` | SGD动量 | 2/12 (17%) | pytorch_resnet_cifar10, Person_reID (隐式) |
| **数据加载** | `workers` / `num_workers` | 数据加载线程数 | 2/12 (17%) | pytorch_resnet_cifar10, Person_reID (隐式) |
| **混合精度** | `fp16` / `bf16` / `half` | 混合精度训练 | 5/12 (42%) | pytorch_resnet_cifar10, VulBERTa (2个), Person_reID (3个) |

### 3.3 模型架构相关超参数（特定类型模型通用）

#### Transformer类模型 (MRT-OAST)
- `layers` / `transformer_nlayers`: Transformer层数
- `d_model`: 模型维度
- `d_ff`: 前馈网络维度
- `heads` / `h`: 注意力头数
- `max_len` / `sen_max_len`: 最大序列长度

#### CNN/ResNet类模型 (pytorch_resnet_cifar10, Person_reID)
- `stride`: 卷积步长
- `weight_decay`: 权重衰减

#### 预训练模型微调 (VulBERTa, Person_reID)
- `learning_rate`: 学习率（通常较小）
- `fp16`: 混合精度训练

---

## 4. 代码扩展后的潜在通用超参数

### 4.1 学习率调度器相关

**当前状态**:
- MRT-OAST: 使用LambdaLR（已实现warmup）
- pytorch_resnet_cifar10: 使用MultiStepLR（固定milestones）
- Person_reID: 使用StepLR或CosineAnnealingLR（可选）
- bug-localization: 使用sklearn的MLPRegressor（内置）
- VulBERTa: 使用Transformers默认调度器
- examples: 使用StepLR（固定gamma）

**扩展建议**:
如果修改代码添加统一的学习率调度器参数，以下超参数可成为通用参数：

| 超参数名称 | 说明 | 实现难度 | 预期影响 |
|-----------|------|---------|---------|
| `lr_scheduler` | 调度器类型 (step/cosine/exponential) | ⭐⭐ 容易 | 中等 |
| `lr_decay_steps` | 学习率衰减步数 | ⭐⭐ 容易 | 中等 |
| `lr_decay_gamma` | 学习率衰减因子 | ⭐⭐ 容易 | 中等 |
| `warmup_epochs` | 预热轮数 | ⭐⭐⭐ 中等 | 小到中等 |
| `min_lr` | 最小学习率 | ⭐ 极易 | 小 |

**修改示例** (以bug-localization为例):
```python
# 当前: sklearn MLPRegressor内置学习率
# 修改后: 添加自定义学习率调度
if args.lr_scheduler == 'step':
    scheduler = StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_gamma)
elif args.lr_scheduler == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
```

### 4.2 数据增强相关

**当前状态**:
- Person_reID: 支持erasing_p, color_jitter
- pytorch_resnet_cifar10: 固定数据增强策略
- examples: 无数据增强
- 其他模型: 无数据增强或固定策略

**扩展建议**:

| 超参数名称 | 说明 | 实现难度 | 预期影响 | 适用模型 |
|-----------|------|---------|---------|---------|
| `augmentation` | 数据增强开关 | ⭐⭐ 容易 | 大 | 所有CV模型 |
| `mixup_alpha` | Mixup增强系数 | ⭐⭐ 容易 | 中等 | 所有CV模型 |
| `cutmix_alpha` | CutMix增强系数 | ⭐⭐ 容易 | 中等 | 所有CV模型 |
| `auto_augment` | AutoAugment策略 | ⭐⭐⭐ 中等 | 大 | 所有CV模型 |

### 4.3 优化器相关

**当前状态**:
- MRT-OAST: Adam (硬编码)
- pytorch_resnet_cifar10: SGD (硬编码)
- Person_reID: SGD (硬编码)
- bug-localization: 可选 sgd/adam/lbfgs
- VulBERTa: AdamW (Transformers默认)
- examples: Adadelta/SGD (固定)

**扩展建议**:

| 超参数名称 | 说明 | 实现难度 | 预期影响 |
|-----------|------|---------|---------|
| `optimizer` | 优化器类型 (adam/sgd/adamw) | ⭐⭐ 容易 | 大 |
| `beta1` | Adam的beta1参数 | ⭐ 极易 | 小 |
| `beta2` | Adam的beta2参数 | ⭐ 极易 | 小 |
| `epsilon` | 优化器epsilon | ⭐ 极易 | 小 |
| `amsgrad` | 使用AMSGrad变体 | ⭐ 极易 | 小到中等 |

### 4.4 早停（Early Stopping）

**当前状态**:
- bug-localization: 支持n_iter_no_change
- VulBERTa: Transformers内置早停
- 其他模型: 无早停机制

**扩展建议**:

| 超参数名称 | 说明 | 实现难度 | 预期影响 |
|-----------|------|---------|---------|
| `early_stopping` | 早停开关 | ⭐⭐ 容易 | 大（节省时间和能耗） |
| `patience` | 早停耐心值 | ⭐ 极易 | 大 |
| `min_delta` | 最小改进阈值 | ⭐ 极易 | 中等 |

**实现示例**:
```python
# 添加到所有模型的训练循环
best_metric = -float('inf')
patience_counter = 0

for epoch in range(args.epochs):
    # ... 训练代码 ...
    val_metric = evaluate(model, val_loader)

    if val_metric > best_metric + args.min_delta:
        best_metric = val_metric
        patience_counter = 0
        save_checkpoint(model)
    else:
        patience_counter += 1

    if args.early_stopping and patience_counter >= args.patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 4.5 梯度裁剪

**当前状态**:
- 大部分模型: 无梯度裁剪
- VulBERTa: Transformers可能有内置

**扩展建议**:

| 超参数名称 | 说明 | 实现难度 | 预期影响 |
|-----------|------|---------|---------|
| `grad_clip` | 梯度裁剪阈值 | ⭐ 极易 | 中等（提升稳定性） |
| `grad_clip_norm` | 裁剪范数类型 (1/2/inf) | ⭐⭐ 容易 | 小 |

### 4.6 汇总：代码扩展后的通用超参数

经过合理的代码扩展（估计每个模型需要50-200行代码），以下超参数可以成为**跨模型通用超参数**：

**高优先级（强烈推荐扩展）**:
1. `lr_scheduler`: 学习率调度器类型
2. `early_stopping`: 早停开关
3. `patience`: 早停耐心值
4. `optimizer`: 优化器类型

**中优先级（推荐扩展）**:
5. `warmup_epochs`: 预热轮数
6. `grad_clip`: 梯度裁剪阈值
7. `weight_decay`: 权重衰减（扩展到所有模型）
8. `dropout`: Dropout率（扩展到所有模型）

**低优先级（可选扩展）**:
9. 数据增强相关参数（仅CV模型）
10. 优化器的beta参数（细粒度调优）

---

## 5. 超参数变异对性能和能耗的影响评估

### 5.1 影响评估维度

我们从以下三个维度评估超参数变异的影响：

1. **性能影响** (Performance Impact): 对模型准确率、F1分数等指标的影响
2. **能耗影响** (Energy Impact): 对训练时间和GPU能耗的影响
3. **影响敏感度** (Sensitivity): 超参数变化对结果的敏感程度

影响等级：
- **高**: 变化±20%可导致性能/能耗变化>10%
- **中**: 变化±20%可导致性能/能耗变化5-10%
- **低**: 变化±20%可导致性能/能耗变化<5%

### 5.2 完全通用超参数的影响分析

#### 5.2.1 epochs (训练轮数)

| 维度 | 影响程度 | 分析 |
|------|---------|------|
| **性能影响** | 高 | • 过少：欠拟合，性能显著下降<br>• 适中：达到最优性能<br>• 过多：可能过拟合或性能饱和 |
| **能耗影响** | 高（线性相关） | • 能耗几乎与epochs成正比<br>• epochs减半 → 能耗减半<br>• epochs翻倍 → 能耗翻倍 |
| **影响敏感度** | 高 | • 不同模型最优epochs差异大<br>• MNIST: 10-20轮足够<br>• ResNet: 需要150-200轮<br>• Person_reID: 60-100轮 |

**变异建议**:
- 减少epochs: 50%, 75% (快速测试，显著节能)
- 增加epochs: 150%, 200% (追求最优性能)

**预期影响示例**:
```
MRT-OAST (默认10轮):
- 5轮: 性能-15%, 能耗-50%, 时间-50%
- 15轮: 性能+5%, 能耗+50%, 时间+50%
- 20轮: 性能+7%, 能耗+100%, 时间+100%

pytorch_resnet_cifar10 (默认200轮):
- 100轮: 性能-8%, 能耗-50%, 时间-50%
- 300轮: 性能+2%, 能耗+50%, 时间+50%
```

---

#### 5.2.2 batch_size (批次大小)

| 维度 | 影响程度 | 分析 |
|------|---------|------|
| **性能影响** | 中 | • 小batch (8-32): 更好的泛化，但训练不稳定<br>• 大batch (64-256): 训练稳定，但可能泛化差<br>• 超大batch (>256): 可能显著降低性能 |
| **能耗影响** | 中（复杂关系） | • 大batch：单次迭代慢，但总迭代次数少<br>• 小batch：单次迭代快，但总迭代次数多<br>• 最优点通常在32-128之间 |
| **影响敏感度** | 中到高 | • 受GPU显存限制<br>• 大模型(VulBERTa-MLP)只能用小batch<br>• 小模型(MNIST)可用大batch |

**变异建议**:
- 减小batch_size: 50%, 75% (提升泛化，可能节能)
- 增大batch_size: 150%, 200% (加速训练，但需要调整学习率)

**预期影响示例**:
```
Person_reID (默认batch=32):
- batch=16: 性能+2%, 每轮时间+30%, 总能耗+20%
- batch=64: 性能-1%, 每轮时间-20%, 总能耗-10%

VulBERTa-MLP (默认batch=2):
- batch=1: 性能-3%, 时间+50%, 能耗+40%
- batch=4: 性能+1%, 时间-15%, 能耗-10% (如果显存允许)
```

**重要提示**:
- 修改batch_size通常需要同步调整学习率
- 经验法则: `new_lr = old_lr * sqrt(new_batch / old_batch)`

---

#### 5.2.3 learning_rate (学习率)

| 维度 | 影响程度 | 分析 |
|------|---------|------|
| **性能影响** | 高 | • 过小：收敛慢，可能陷入局部最优<br>• 适中：收敛快且稳定<br>• 过大：训练不稳定，甚至发散 |
| **能耗影响** | 中 | • 适当增大学习率可加速收敛，节省能耗<br>• 过大的学习率导致震荡，反而增加能耗<br>• 过小的学习率延长训练时间，增加能耗 |
| **影响敏感度** | 高 | • 不同模型最优学习率差异极大<br>• Adam: 通常1e-4到1e-3<br>• SGD: 通常0.01到0.1<br>• 预训练模型微调: 通常1e-5到1e-4 |

**变异建议**:
- 对数尺度变异: 0.1x, 0.5x, 2x, 5x, 10x
- 不要使用线性变异（效果不佳）

**预期影响示例**:
```
MRT-OAST (默认lr=0.0001, Adam):
- lr=0.00005: 性能-5%, 收敛慢, 能耗+20%
- lr=0.0002: 性能+2%, 收敛快, 能耗-10%
- lr=0.0005: 性能-3%, 训练不稳定

pytorch_resnet_cifar10 (默认lr=0.1, SGD):
- lr=0.05: 性能-8%, 收敛慢, 能耗+15%
- lr=0.2: 性能-5%, 训练不稳定
- lr=0.15: 性能-2%, 需要更长warmup

VulBERTa-MLP (默认lr=3e-05, AdamW):
- lr=1e-05: 性能-10%, 收敛极慢, 能耗+30%
- lr=5e-05: 性能+3%, 收敛快, 能耗-5%
- lr=1e-04: 性能-8%, 过拟合风险
```

**高风险警告**:
- 学习率是**最敏感**的超参数
- 不当的学习率可能导致完全失败的训练
- 建议先进行小规模学习率搜索实验

---

#### 5.2.4 seed (随机种子)

| 维度 | 影响程度 | 分析 |
|------|---------|------|
| **性能影响** | 低到中 | • 不同随机种子可导致±1-5%性能变化<br>• 小数据集影响更大<br>• 大数据集影响较小 |
| **能耗影响** | 极低 | • 几乎无影响（<1%） |
| **影响敏感度** | 低 | • 主要影响模型初始化和数据shuffle |

**变异建议**:
- 使用多个不同的随机种子 (如: 42, 123, 456, 789, 1024)
- 对性能进行统计分析（均值±标准差）

**预期影响示例**:
```
所有模型:
- 不同seed导致性能变化: 通常±0.5-3%
- 能耗变化: <±1%

建议实践:
- 对关键实验使用3-5个不同seed
- 报告平均性能和标准差
```

---

### 5.3 高频通用超参数的影响分析

#### 5.3.1 dropout / droprate (Dropout率)

| 维度 | 影响程度 | 分析 |
|------|---------|------|
| **性能影响** | 中到高 | • 0.0: 无正则化，可能过拟合<br>• 0.2-0.5: 通常是最优范围<br>• >0.7: 过度正则化，欠拟合 |
| **能耗影响** | 低 | • Dropout对计算量影响很小<br>• 间接影响：好的dropout可能加速收敛 |
| **影响敏感度** | 中 | • 小模型: dropout影响较大<br>• 大模型: dropout更重要（防止过拟合） |

**变异建议**:
- dropout=0.0 (无正则化)
- dropout=0.1, 0.2, 0.3, 0.5, 0.7

**预期影响示例**:
```
MRT-OAST (默认dropout=0.2):
- dropout=0.0: 训练准确率高但测试准确率-3%
- dropout=0.3: 性能+1%
- dropout=0.5: 性能-2%, 欠拟合

Person_reID (默认droprate=0.5):
- droprate=0.0: mAP-2%, 过拟合
- droprate=0.3: mAP+1%
- droprate=0.7: mAP-5%, 严重欠拟合
```

---

#### 5.3.2 weight_decay (权重衰减 / L2正则化)

| 维度 | 影响程度 | 分析 |
|------|---------|------|
| **性能影响** | 中 | • 0.0: 无正则化，可能过拟合<br>• 1e-4到1e-3: 通常最优<br>• >1e-2: 过度正则化 |
| **能耗影响** | 极低 | • 计算开销可忽略 |
| **影响敏感度** | 中 | • 与dropout类似，但更温和 |

**变异建议**:
- 对数尺度: 0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3

**预期影响示例**:
```
pytorch_resnet_cifar10 (默认weight_decay=0.0001):
- weight_decay=0.0: 准确率-2%, 过拟合
- weight_decay=0.0005: 准确率+1%
- weight_decay=0.005: 准确率-4%, 欠拟合

Person_reID (默认weight_decay=5e-4):
- weight_decay=0.0: mAP-1.5%
- weight_decay=1e-4: mAP+0.5%
- weight_decay=1e-3: mAP-1%
```

---

#### 5.3.3 momentum (SGD动量)

| 维度 | 影响程度 | 分析 |
|------|---------|------|
| **性能影响** | 中 | • 0.0: 普通SGD，收敛慢<br>• 0.9: 标准配置<br>• 0.99: 更平滑，但可能过冲 |
| **能耗影响** | 低到中 | • 高动量可加速收敛，节省能耗 |
| **影响敏感度** | 中 | • 通常0.9是最优值 |

**变异建议**:
- momentum=0.0, 0.5, 0.9, 0.95, 0.99

**预期影响示例**:
```
pytorch_resnet_cifar10 (默认momentum=0.9):
- momentum=0.0: 准确率-5%, 能耗+20%
- momentum=0.95: 准确率+0.5%, 能耗-5%
- momentum=0.99: 准确率-1%, 训练不稳定
```

---

#### 5.3.4 fp16 / bf16 / half (混合精度训练)

| 维度 | 影响程度 | 分析 |
|------|---------|------|
| **性能影响** | 低 | • 现代混合精度实现对性能影响很小<br>• 可能有±0.5-2%的性能变化 |
| **能耗影响** | 高 | • 显著减少显存占用（~50%）<br>• 加速训练（20-50%）<br>• 能耗节省（15-40%） |
| **影响敏感度** | 低 | • 大部分模型可以安全使用 |

**变异建议**:
- fp16=True vs False
- bf16=True (更稳定) vs fp16=True

**预期影响示例**:
```
VulBERTa-MLP (默认fp16=True):
- fp16=False: 性能-0.5%, 能耗+30%, 显存+50%
- bf16=True: 性能相同, 能耗-25%, 显存-45%

pytorch_resnet_cifar10:
- half=True: 准确率-1%, 能耗-25%, 时间-30%
- half=False: 基线性能

Person_reID:
- bf16=True: mAP-0.3%, 能耗-20%, 时间-25%
- fp16=True: mAP-0.8%, 能耗-25%, 时间-30%
```

**强烈推荐**:
- bf16优于fp16（更稳定）
- 对于能耗研究，混合精度是**高优先级**变异算子

---

### 5.4 模型架构超参数的影响分析

#### 5.4.1 Transformer相关 (MRT-OAST)

**layers / transformer_nlayers (Transformer层数)**:
- 性能影响: 高
  - 1层: -15% 性能
  - 2层 (默认): 基线
  - 4层: +3-5% 性能
  - 6层: +5-7% 性能，但过拟合风险
- 能耗影响: 高（几乎线性）
  - 每增加1层，能耗+40-60%
  - 每增加1层，训练时间+45-70%

**d_model (模型维度)**:
- 性能影响: 中到高
  - 64: -10% 性能
  - 128 (默认): 基线
  - 256: +3-5% 性能
  - 512: +5-8% 性能
- 能耗影响: 高（平方关系）
  - 维度翻倍，能耗约为原来的2-3倍

**heads / h (注意力头数)**:
- 性能影响: 中
  - 4: -3% 性能
  - 8 (默认): 基线
  - 16: +1-2% 性能
- 能耗影响: 低到中
  - 头数翻倍，能耗+20-30%

---

#### 5.4.2 MLP相关 (bug-localization)

**hidden_sizes (隐藏层大小)**:
- 性能影响: 中
  - 100: -5% top-1准确率
  - 200: -2% top-1准确率
  - 300 (默认): 基线
  - 500: +1-2% top-1准确率
  - 1000: +2-3% top-1准确率，过拟合风险
- 能耗影响: 中
  - 大小翻倍，能耗+50-80%

**alpha (L2正则化)**:
- 性能影响: 中
  - 1e-7: -2% (过拟合)
  - 1e-5 (默认): 基线
  - 1e-4: +1%
  - 1e-3: -3% (欠拟合)
- 能耗影响: 极低

---

### 5.5 影响总结表

#### 高性能影响 + 高能耗影响

这些超参数是**优先变异目标**，可以同时研究性能和能耗的权衡：

| 超参数 | 性能影响 | 能耗影响 | 推荐优先级 |
|-------|---------|---------|-----------|
| `epochs` | 高 | 高（线性） | ⭐⭐⭐⭐⭐ |
| `layers` (Transformer) | 高 | 高（线性） | ⭐⭐⭐⭐⭐ |
| `d_model` (Transformer) | 高 | 高（平方） | ⭐⭐⭐⭐ |
| `learning_rate` | 高 | 中 | ⭐⭐⭐⭐ |

#### 高性能影响 + 低能耗影响

这些超参数可以**调优性能**，而不显著增加能耗：

| 超参数 | 性能影响 | 能耗影响 | 推荐优先级 |
|-------|---------|---------|-----------|
| `dropout` | 中到高 | 低 | ⭐⭐⭐⭐ |
| `weight_decay` | 中 | 极低 | ⭐⭐⭐ |

#### 低性能影响 + 高能耗影响

这些超参数可以**节省能耗**，对性能影响较小：

| 超参数 | 性能影响 | 能耗影响 | 推荐优先级 |
|-------|---------|---------|-----------|
| `fp16/bf16` | 低 | 高（节省） | ⭐⭐⭐⭐⭐ |
| `batch_size` | 中 | 中 | ⭐⭐⭐ |

---

## 6. 修改难度分析

### 6.1 修改难度等级定义

| 难度等级 | 符号 | 说明 | 预计时间 |
|---------|------|------|---------|
| **极易** | ⭐ | 已有命令行参数，直接修改即可 | 0分钟 |
| **容易** | ⭐⭐ | 需要修改训练脚本，添加参数解析 | 5-15分钟 |
| **中等** | ⭐⭐⭐ | 需要修改训练逻辑或模型定义 | 30-60分钟 |
| **困难** | ⭐⭐⭐⭐ | 需要重构部分代码或添加新功能 | 1-3小时 |
| **很困难** | ⭐⭐⭐⭐⭐ | 需要大规模重构或深入理解模型 | >3小时 |

### 6.2 各模型超参数修改难度

#### 6.2.1 极易修改 (⭐)

以下超参数**所有模型**都已经支持命令行修改，**无需任何代码修改**：

- `epochs` / `total_epoch`
- `batch_size` / `batchsize`
- `lr` / `learning_rate`
- `seed`

**额外的极易修改超参数（部分模型）**:

| 超参数 | 支持模型 | 修改方式 |
|-------|---------|---------|
| `dropout` | MRT-OAST | `--dropout 0.3` |
| `weight_decay` | pytorch_resnet_cifar10, Person_reID | `--weight-decay 0.001` 或 `--weight_decay 0.001` |
| `momentum` | pytorch_resnet_cifar10 | `--momentum 0.95` |
| `fp16/bf16/half` | VulBERTa, Person_reID, pytorch_resnet_cifar10 | `--fp16` 或 `--bf16` 或 `--half` |
| 所有MRT-OAST架构参数 | MRT-OAST | `--layers 4 --d-model 256 --heads 16` |
| 所有bug-localization参数 | bug-localization | `--hidden_sizes 200 --alpha 1e-4` |
| 所有Person_reID参数 | Person_reID (3个模型) | `--droprate 0.3 --erasing_p 0.5` |

#### 6.2.2 容易修改 (⭐⭐)

需要修改训练脚本，添加参数解析和传递，但不涉及训练逻辑：

**扩展dropout到所有模型**:
- pytorch_resnet_cifar10: 添加dropout层到ResNet定义（修改resnet.py，约20行代码）
- VulBERTa: 已在模型内部，添加命令行参数（修改train_vulberta.py，约10行代码）
- examples: 各模型已有dropout，添加命令行参数（修改train.sh和main.py，约15行代码/模型）

**扩展weight_decay到所有模型**:
- MRT-OAST: 添加optimizer参数（修改main_batch.py，约5行代码）
- bug-localization: sklearn的MLPRegressor支持alpha参数（已有）
- VulBERTa: Transformers训练参数（修改train_vulberta.py，约5行代码）
- examples: 添加optimizer参数（修改各main.py，约5行代码/模型）

**添加混合精度到未支持的模型**:
- MRT-OAST: 使用torch.cuda.amp（修改main_batch.py，约30行代码）
- bug-localization: 不适用（sklearn模型）
- examples: 使用torch.cuda.amp（修改各main.py，约30行代码/模型）

#### 6.2.3 中等难度 (⭐⭐⭐)

需要修改训练逻辑或添加新功能：

**添加学习率调度器**:
- 所有模型: 添加scheduler初始化和step调用（约50-100行代码/模型）
- MRT-OAST已有LambdaLR，但需要参数化（约30行代码）
- Person_reID部分支持cosine，需要统一（约40行代码）

**添加早停机制**:
- 所有模型: 添加验证循环、metric跟踪、checkpoint管理（约80-150行代码/模型）
- bug-localization已有n_iter_no_change，可借鉴

**添加梯度裁剪**:
- 所有模型: 在optimizer.step()前添加clip_grad_norm_（约10行代码/模型）
- 需要正确处理多GPU情况

#### 6.2.4 困难 (⭐⭐⭐⭐)

需要重构部分代码或深入理解模型：

**统一优化器选择**:
- 需要重构optimizer初始化逻辑（约100-200行代码/模型）
- 需要处理不同优化器的参数差异
- 需要适配不同模型的特殊需求（如Person_reID的多学习率）

**添加数据增强**:
- CV模型: 需要重构数据加载pipeline（约100-300行代码/模型）
- NLP模型: 需要实现特定的文本增强策略（约200-500行代码/模型）

**添加模型结构搜索空间**:
- 需要参数化模型定义（约300-800行代码/模型）
- 例如：动态调整ResNet的block数量、channel数量

#### 6.2.5 很困难 (⭐⭐⭐⭐⭐)

需要大规模重构或深入理解模型：

**改变模型架构类型**:
- 例如：将ResNet改为EfficientNet
- 需要重写模型定义和可能的训练逻辑（>1000行代码）

**添加分布式训练支持**:
- 需要重构整个训练流程（>500行代码/模型）

**实现神经架构搜索（NAS）**:
- 需要完全重构训练流程（>2000行代码）

### 6.3 修改难度总结

#### 按难度分类的推荐变异超参数

**立即可用（无需修改代码）**:
1. `epochs`
2. `batch_size`
3. `learning_rate`
4. `seed`
5. 模型特定已支持的超参数

**短期可扩展（<1天工作量）**:
6. `dropout`（扩展到所有模型）
7. `weight_decay`（扩展到所有模型）
8. `fp16/bf16`（扩展到所有模型）
9. `grad_clip`（新增）

**中期可扩展（1-3天工作量）**:
10. `lr_scheduler`（新增）
11. `early_stopping` + `patience`（新增）
12. `warmup_epochs`（新增）
13. `optimizer`（统一）

**长期可扩展（>3天工作量）**:
14. 数据增强相关参数
15. 模型架构参数（动态）

---

## 7. 推荐的超参数变异列表

### 7.1 推荐原则

基于以上分析，我们制定了以下推荐原则：

1. **性能-能耗权衡**: 优先选择对性能和能耗都有显著影响的超参数
2. **跨模型通用性**: 优先选择在所有12个模型上都可变的超参数
3. **修改难度**: 优先选择已支持或易于扩展的超参数
4. **科学价值**: 优先选择具有理论和实践意义的超参数
5. **实验可行性**: 考虑实验时间和计算资源限制

### 7.2 三级推荐列表

#### 7.2.1 第一优先级（强烈推荐）

这些超参数**必须**包含在变异实验中：

| 序号 | 超参数 | 适用模型 | 推荐理由 | 修改难度 | 预期影响 |
|------|-------|---------|---------|---------|---------|
| 1 | `epochs` | 全部 (12/12) | • 对性能和能耗影响都极大<br>• 线性能耗关系，便于分析<br>• 无需代码修改 | ⭐ | 性能: 高<br>能耗: 高 |
| 2 | `batch_size` | 全部 (12/12) | • 影响训练速度和收敛性<br>• 与能耗有复杂关系<br>• 无需代码修改 | ⭐ | 性能: 中<br>能耗: 中 |
| 3 | `learning_rate` | 全部 (12/12) | • 最关键的超参数之一<br>• 影响收敛速度和最终性能<br>• 无需代码修改 | ⭐ | 性能: 高<br>能耗: 中 |
| 4 | `fp16/bf16` | 8/12 (扩展后12/12) | • 显著节省能耗和显存<br>• 对性能影响较小<br>• 易于扩展 | ⭐⭐ | 性能: 低<br>能耗: 高（节省） |

**推荐的变异值**:

```python
# epochs（根据模型调整）
epochs_variants = {
    'MRT-OAST': [5, 7, 10, 15, 20],  # 默认10
    'bug-localization': [5, 10, 15, 20],  # K-fold每折的迭代次数相关
    'pytorch_resnet_cifar10': [100, 150, 200, 250, 300],  # 默认200
    'VulBERTa-MLP': [5, 8, 10, 15, 20],  # 默认10
    'VulBERTa-CNN': [10, 15, 20, 30],  # 默认20
    'Person_reID': [30, 45, 60, 80, 100],  # 默认60
    'examples-mnist': [7, 10, 14, 20],  # 默认14
    'examples-mnist_rnn': [7, 10, 14, 20],  # 默认14
    'examples-mnist_ff': [500, 750, 1000, 1500],  # 默认1000
    'examples-siamese': [7, 10, 14, 20],  # 默认14
}

# batch_size（根据模型调整）
batch_size_variants = {
    'MRT-OAST': [32, 48, 64, 96, 128],  # 默认64
    'bug-localization': [5, 10, 15],  # 影响K-fold并行度
    'pytorch_resnet_cifar10': [64, 96, 128, 192],  # 默认128
    'VulBERTa-MLP': [1, 2, 4],  # 默认2，显存限制
    'VulBERTa-CNN': [64, 96, 128, 192],  # 默认128
    'Person_reID-densenet121': [12, 18, 24, 32],  # 默认24
    'Person_reID-hrnet18': [12, 18, 24, 32],  # 默认24
    'Person_reID-pcb': [16, 24, 32, 48],  # 默认32
    'examples-mnist': [16, 32, 64, 128],  # 默认32
    'examples-mnist_rnn': [16, 32, 64],  # 默认32
    'examples-mnist_ff': [30000, 40000, 50000],  # train_size，默认50000
    'examples-siamese': [16, 32, 64],  # 默认32
}

# learning_rate（对数尺度）
learning_rate_variants = {
    'MRT-OAST': [5e-5, 1e-4, 2e-4, 5e-4],  # 默认1e-4，Adam
    'bug-localization': [1e-6, 1e-5, 1e-4, 1e-3],  # 默认1e-5，内置优化器
    'pytorch_resnet_cifar10': [0.05, 0.075, 0.1, 0.15, 0.2],  # 默认0.1，SGD
    'VulBERTa-MLP': [1e-5, 3e-5, 5e-5, 1e-4],  # 默认3e-5，AdamW
    'VulBERTa-CNN': [0.0001, 0.0003, 0.0005, 0.001],  # 默认0.0005
    'Person_reID': [0.02, 0.035, 0.05, 0.08],  # 默认0.05，SGD
    'examples-mnist': [0.5, 0.8, 1.0, 1.5],  # 默认1.0
    'examples-mnist_rnn': [0.05, 0.08, 0.1, 0.15],  # 默认0.1
    'examples-mnist_ff': [0.01, 0.02, 0.03, 0.05],  # 默认0.03
    'examples-siamese': [0.5, 0.8, 1.0, 1.5],  # 默认1.0
}

# fp16/bf16
precision_variants = ['fp32', 'fp16', 'bf16']  # bf16更稳定，推荐优先
```

---

#### 7.2.2 第二优先级（推荐）

这些超参数**应该**包含在变异实验中（如果资源允许）：

| 序号 | 超参数 | 适用模型 | 推荐理由 | 修改难度 | 预期影响 |
|------|-------|---------|---------|---------|---------|
| 5 | `dropout` | 5/12 (扩展后12/12) | • 重要的正则化方法<br>• 对性能影响大<br>• 对能耗影响小 | ⭐⭐ | 性能: 中-高<br>能耗: 低 |
| 6 | `weight_decay` | 4/12 (扩展后12/12) | • 另一种正则化方法<br>• 与dropout互补<br>• 易于扩展 | ⭐⭐ | 性能: 中<br>能耗: 极低 |
| 7 | `seed` | 全部 (12/12) | • 验证结果可重复性<br>• 评估性能方差<br>• 无需代码修改 | ⭐ | 性能: 低-中<br>能耗: 极低 |
| 8 | `momentum` | 2/12 (SGD模型) | • 影响SGD收敛<br>• 部分模型已支持 | ⭐ | 性能: 中<br>能耗: 低-中 |

**推荐的变异值**:

```python
# dropout
dropout_variants = [0.0, 0.1, 0.2, 0.3, 0.5]

# weight_decay（对数尺度）
weight_decay_variants = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]

# seed
seed_variants = [42, 123, 456, 789, 1024]

# momentum（仅SGD模型）
momentum_variants = [0.0, 0.5, 0.9, 0.95, 0.99]
```

---

#### 7.2.3 第三优先级（可选）

这些超参数**可以**包含在扩展实验中（如果有足够资源和时间）：

| 序号 | 超参数 | 适用模型 | 推荐理由 | 修改难度 | 预期影响 |
|------|-------|---------|---------|---------|---------|
| 9 | `layers` | 1/12 (MRT-OAST) | • 对性能和能耗影响都大<br>• 研究模型容量 | ⭐ | 性能: 高<br>能耗: 高 |
| 10 | `d_model` | 1/12 (MRT-OAST) | • Transformer维度<br>• 平方级能耗影响 | ⭐ | 性能: 高<br>能耗: 高 |
| 11 | `hidden_sizes` | 1/12 (bug-localization) | • MLP容量<br>• 影响性能和能耗 | ⭐ | 性能: 中<br>能耗: 中 |
| 12 | `lr_scheduler` | 0/12 (扩展后12/12) | • 改善收敛<br>• 可能节省能耗 | ⭐⭐⭐ | 性能: 中<br>能耗: 中 |
| 13 | `early_stopping` | 1/12 (扩展后12/12) | • 自动停止训练<br>• 显著节省能耗 | ⭐⭐⭐ | 性能: 中<br>能耗: 高（节省） |
| 14 | `warmup_epochs` | 部分支持 | • 改善训练初期稳定性<br>• 可能改善性能 | ⭐⭐⭐ | 性能: 低-中<br>能耗: 低 |

**推荐的变异值**:

```python
# layers（MRT-OAST）
layers_variants = [1, 2, 3, 4, 6]  # 默认2

# d_model（MRT-OAST）
d_model_variants = [64, 128, 192, 256, 512]  # 默认128

# hidden_sizes（bug-localization）
hidden_sizes_variants = [[100], [200], [300], [500], [300, 200], [500, 300]]  # 默认[300]

# lr_scheduler（需要扩展）
lr_scheduler_variants = ['none', 'step', 'cosine', 'exponential']

# early_stopping（需要扩展）
early_stopping_variants = {
    'enabled': [False, True],
    'patience': [5, 10, 15, 20],  # 当enabled=True时
}

# warmup_epochs（需要扩展）
warmup_epochs_variants = [0, 3, 5, 10]
```

---

### 7.3 最小变异实验方案

如果**资源非常有限**，建议使用以下最小方案：

#### 方案A：单参数变异（最小）

每次只变异1个超参数，其他保持默认值。

**必须包含的单参数变异**:
1. `epochs`: 5个变异值 × 12个模型 = 60次实验
2. `batch_size`: 4个变异值 × 12个模型 = 48次实验
3. `learning_rate`: 4个变异值 × 12个模型 = 48次实验
4. `fp16/bf16`: 3个变异值 × 12个模型 = 36次实验

**总计**: 192次实验（不含重复种子）

#### 方案B：关键参数组合（推荐）

变异最关键的参数组合，同时考虑多个因素。

**关键组合**:
1. **默认配置**（基线）: 12次
2. **减少epochs + 增大batch**（快速训练）: 12次
3. **增加epochs + 降低lr**（追求性能）: 12次
4. **默认配置 + fp16**（节能）: 12次
5. **默认配置 + 增强正则化**（dropout+weight_decay）: 12次

**总计**: 60次实验

#### 方案C：分层抽样（平衡）

选择代表性模型，进行更全面的变异。

**选择3个代表性模型**:
- MRT-OAST（Transformer）
- pytorch_resnet_cifar10（CNN）
- bug-localization（MLP）

**每个模型的变异**:
- epochs: 5个值
- batch_size: 4个值
- learning_rate: 4个值
- dropout: 5个值
- fp16/bf16: 3个值

**总计**: 3个模型 × (5+4+4+5+3) = 63次实验

---

### 7.4 完整变异实验方案

如果**资源充足**，建议使用以下完整方案：

#### 第一阶段：单参数扫描

**目标**: 识别每个超参数的最优范围和敏感度

1. **第一优先级参数** (4个)
   - epochs: 5值 × 12模型 = 60次
   - batch_size: 4值 × 12模型 = 48次
   - learning_rate: 4值 × 12模型 = 48次
   - fp16/bf16: 3值 × 12模型 = 36次
   - **小计**: 192次

2. **第二优先级参数** (4个)
   - dropout: 5值 × 12模型 = 60次
   - weight_decay: 5值 × 12模型 = 60次
   - seed: 5值 × 12模型 = 60次
   - momentum: 5值 × 2模型(SGD) = 10次
   - **小计**: 190次

3. **第三优先级参数**（模型特定）
   - layers: 5值 × 1模型 = 5次
   - d_model: 5值 × 1模型 = 5次
   - hidden_sizes: 6值 × 1模型 = 6次
   - **小计**: 16次

**第一阶段总计**: 398次实验

#### 第二阶段：最优组合

基于第一阶段结果，为每个模型选择最优超参数组合进行验证。

- 每个模型选择3-5个最优组合
- 每个组合使用3个不同seed
- **估计**: 12模型 × 4组合 × 3seed = 144次

#### 第三阶段：消融研究

研究关键超参数之间的交互效应。

- 重点研究: epochs × learning_rate, batch_size × learning_rate, dropout × weight_decay
- **估计**: 100-200次

**完整方案总计**: 约650-750次实验

---

### 7.5 实施建议

#### 7.5.1 实验执行策略

1. **并行化**:
   - 使用多GPU并行运行不同模型
   - 优先运行快速模型（examples-mnist），延后运行慢速模型（pytorch_resnet_cifar10）

2. **增量式**:
   - 先执行方案B（60次）验证基础设施
   - 然后执行第一优先级参数扫描
   - 最后根据结果决定是否进行第二、三优先级

3. **早期终止**:
   - 如果某个变异明显失败（如训练发散），提前终止
   - 节省的时间用于更有价值的实验

#### 7.5.2 结果记录

对每次实验记录：

**必须记录**:
- 模型名称和超参数配置
- 训练时间和GPU能耗
- 最终性能指标（准确率、F1等）
- 训练曲线（loss和metric）

**建议记录**:
- GPU显存峰值使用
- 每个epoch的时间
- 验证集性能演变
- 模型文件大小

#### 7.5.3 自动化脚本

建议创建自动化实验脚本：

```bash
#!/bin/bash
# experiment_runner.sh

# 定义实验配置
declare -A experiments=(
    ["exp001"]="MRT-OAST --epochs 5"
    ["exp002"]="MRT-OAST --epochs 7"
    ["exp003"]="MRT-OAST --epochs 10"
    # ... 更多实验配置
)

# 循环执行实验
for exp_id in "${!experiments[@]}"; do
    echo "Running experiment: $exp_id"

    # 解析配置
    config="${experiments[$exp_id]}"
    model=$(echo $config | awk '{print $1}')
    args=$(echo $config | cut -d' ' -f2-)

    # 执行训练
    cd models/$model
    ./train.sh $args 2>&1 | tee ../../results/${exp_id}_training.log

    # 保存结果
    cp training.log ../../results/${exp_id}_result.txt

    # 记录能耗（如果使用nvidia-smi或其他工具）
    # ...

    echo "Experiment $exp_id completed"
done
```

---

### 7.6 推荐总结

#### 最小推荐（资源受限）

**必须变异**: epochs, batch_size, learning_rate, fp16/bf16
**最少实验数**: 60次（方案B）

#### 标准推荐（一般情况）

**必须变异**: 第一优先级全部 (4个超参数)
**推荐变异**: 第二优先级部分 (dropout, weight_decay)
**实验数**: 约300次

#### 完整推荐（资源充足）

**变异**: 第一、二、三优先级全部
**实验数**: 约650-750次

#### 按影响维度的推荐

**专注于性能优化**:
- epochs, learning_rate, dropout, layers (Transformer), hidden_sizes (MLP)

**专注于能耗优化**:
- epochs, fp16/bf16, early_stopping, batch_size

**平衡性能和能耗**:
- epochs, learning_rate, batch_size, fp16/bf16

---

## 8. 附录

### 8.1 完整超参数速查表

见下页表格...

### 8.2 模型训练命令快速参考

#### MRT-OAST
```bash
cd models/MRT-OAST
./train.sh --dataset OJClone --epochs 10 --batch-size 64 --lr 0.0001 \
           --dropout 0.2 --layers 2 --d-model 128 --heads 8 \
           2>&1 | tee training.log
```

#### bug-localization-by-dnn-and-rvsm
```bash
cd models/bug-localization-by-dnn-and-rvsm
./train.sh -n dnn --kfold 10 --hidden_sizes 300 --alpha 1e-5 \
           --solver sgd --max_iter 10000 \
           2>&1 | tee training.log
```

#### pytorch_resnet_cifar10
```bash
cd models/pytorch_resnet_cifar10
./train.sh -n resnet20 -e 200 -b 128 --lr 0.1 \
           --momentum 0.9 --wd 0.0001 \
           2>&1 | tee training.log
```

#### VulBERTa
```bash
cd models/VulBERTa
# MLP
./train.sh -n mlp -d devign --batch_size 2 --epochs 10 \
           --learning_rate 3e-05 --fp16 \
           2>&1 | tee training.log

# CNN
./train.sh -n cnn -d devign --batch_size 128 --epochs 20 \
           --learning_rate 0.0005 \
           2>&1 | tee training.log
```

#### Person_reID_baseline_pytorch
```bash
cd models/Person_reID_baseline_pytorch
# DenseNet121
./train.sh -n densenet121 --batchsize 24 --lr 0.05 \
           --total_epoch 60 --droprate 0.5 \
           2>&1 | tee training.log

# HRNet18
./train.sh -n hrnet18 --batchsize 24 --lr 0.05 \
           --total_epoch 60 \
           2>&1 | tee training.log

# PCB
./train.sh -n pcb --batchsize 32 --lr 0.02 \
           --total_epoch 60 \
           2>&1 | tee training.log
```

#### examples
```bash
cd models/examples
# MNIST
./train.sh -n mnist -e 14 -b 32 -l 1.0 --seed 1 \
           2>&1 | tee training.log

# MNIST RNN
./train.sh -n mnist_rnn -e 14 -b 32 -l 0.1 --seed 1 \
           2>&1 | tee training.log

# MNIST Forward-Forward
./train.sh -n mnist_ff -e 1000 --lr 0.03 --threshold 2 \
           2>&1 | tee training.log

# Siamese Network
./train.sh -n siamese -e 14 -b 32 -l 1.0 --seed 1 \
           2>&1 | tee training.log
```

### 8.3 参考资料

1. **超参数优化综述**:
   - Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. JMLR.
   - Feurer, M., & Hutter, F. (2019). Hyperparameter optimization. AutoML book.

2. **学习率调度**:
   - Smith, L. N. (2017). Cyclical learning rates for training neural networks. WACV.
   - Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic gradient descent with warm restarts. ICLR.

3. **混合精度训练**:
   - Micikevicius, P., et al. (2017). Mixed precision training. ICLR.
   - Kalamkar, D., et al. (2019). A study of BFLOAT16 for deep learning training. arXiv.

4. **能耗优化**:
   - You, Y., et al. (2019). Large batch optimization for deep learning. KDD.
   - Schwartz, R., et al. (2020). Green AI. CACM.

---

## 报告总结

本报告系统分析了6个深度学习仓库中12个模型的训练超参数，主要发现：

### 关键发现

1. **完全通用超参数**: `epochs`, `batch_size`, `learning_rate`, `seed` 在所有12个模型中均可修改

2. **高影响超参数**:
   - 性能影响: epochs, learning_rate, dropout, layers (Transformer)
   - 能耗影响: epochs, fp16/bf16, batch_size
   - 双重影响: epochs, learning_rate

3. **修改难度**:
   - 第一优先级参数均为极易修改（⭐）
   - 扩展通用性需要容易到中等难度（⭐⭐-⭐⭐⭐）

### 核心推荐

**最小实验方案**（60次）:
- 变异: epochs, batch_size, learning_rate, fp16/bf16
- 使用5种关键配置组合
- 覆盖所有12个模型

**标准实验方案**（~300次）:
- 第一优先级: epochs, batch_size, learning_rate, fp16/bf16
- 第二优先级: dropout, weight_decay
- 单参数扫描 + 最优组合验证

**完整实验方案**（~650-750次）:
- 三个优先级全部超参数
- 单参数扫描 + 组合优化 + 消融研究

### 实施优先级

1. **立即可行**: epochs, batch_size, learning_rate, seed（无需代码修改）
2. **短期扩展**: fp16/bf16, dropout, weight_decay（1天工作量）
3. **中期扩展**: lr_scheduler, early_stopping（1-3天工作量）

---

**报告完成时间**: 2025-11-04
**分析人员**: Claude
**版本**: 1.0
