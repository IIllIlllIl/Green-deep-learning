# 深度学习模型训练超参数分析报告

**生成日期：** 2025-11-03
**项目：** 6个仓库11个深度学习模型的训练超参数变异测试分析
**目的：** 为变异测试提供超参数修改方案，以研究超参数对模型训练过程的影响

---

## 目录

1. [项目概述](#项目概述)
2. [模型列表](#模型列表)
3. [各模型超参数详细分析](#各模型超参数详细分析)
4. [多模型共有超参数](#多模型共有超参数)
5. [代码级可变超参数](#代码级可变超参数)
6. [变异测试实施建议](#变异测试实施建议)
7. [附录：超参数速查表](#附录超参数速查表)

---

## 项目概述

本项目包含6个仓库，共计11个深度学习模型，涵盖代码克隆检测、漏洞检测、图像分类、行人重识别、缺陷定位等多个领域。

### 仓库与模型对应关系

| 仓库名 | 模型数量 | 模型名称 | 领域 |
|--------|---------|---------|------|
| MRT-OAST | 1 | MRT-OAST | 代码克隆检测 |
| VulBERTa | 1 | VulBERTa (MLP/CNN) | 漏洞检测 |
| pytorch_resnet_cifar10 | 1 | ResNet | 图像分类 |
| bug-localization-by-dnn-and-rvsm | 1 | DNN | 缺陷定位 |
| Person_reID_baseline_pytorch | 3 | DenseNet121, HRNet18, PCB | 行人重识别 |
| examples | 4 | MNIST(CNN), MNIST RNN, MNIST Forward-Forward, Siamese Network | 基础示例 |

### 训练命令格式

```bash
# 单模型仓库
./train.sh 2>&1 | tee training.log

# 多模型仓库
./train.sh -n model_name 2>&1 | tee training.log
```

---

## 模型列表

共计11个模型：

1. **MRT-OAST** - 基于Transformer的代码克隆检测模型
2. **VulBERTa** - 基于BERT的漏洞检测模型
3. **ResNet-CIFAR10** - ResNet在CIFAR-10上的实现
4. **DNN (Bug Localization)** - 用于缺陷定位的深度神经网络
5. **DenseNet121 (Person ReID)** - 行人重识别DenseNet变体
6. **HRNet18 (Person ReID)** - 行人重识别HRNet变体
7. **PCB (Person ReID)** - 基于局部特征的行人重识别
8. **MNIST CNN** - MNIST数据集的卷积神经网络
9. **MNIST RNN** - MNIST数据集的循环神经网络
10. **MNIST Forward-Forward** - Forward-Forward算法实现
11. **Siamese Network** - 孪生网络示例

---

## 各模型超参数详细分析

### 1. MRT-OAST (代码克隆检测)

**训练命令：** `./train.sh [OPTIONS]`

**可通过命令行修改的超参数：**

| 参数名 | 命令行选项 | 默认值 | 类型 | 说明 |
|--------|-----------|--------|------|------|
| 训练轮数 | `--epochs` | 10 | int | 训练epoch数 |
| 批次大小 | `--batch-size` | 64 | int | 训练批次大小 |
| 学习率 | `--lr` | 0.0001 | float | 初始学习率 |
| Dropout率 | `--dropout` | 0.2 | float | Dropout概率 |
| 随机种子 | `--seed` | 1334 | int | 随机数种子 |
| 验证步数 | `--valid-step` | 1750 | int | 验证频率（0表示每epoch验证）|
| 最大序列长度 | `--max-len` | 256 | int | 输入序列最大长度 |
| Transformer层数 | `--layers` | 2 | int | Transformer编码器层数 |
| 模型维度 | `--d-model` | 128 | int | 模型隐藏层维度 |
| 前馈网络维度 | `--d-ff` | 512 | int | FFN中间层维度 |
| 注意力头数 | `--heads` | 8 | int | 多头注意力头数 |
| 输出维度 | `--output-dim` | 512 | int | 最终输出维度 |
| 测试阈值 | `--threshold` | 0.9 | float | 测试时相似度阈值 |
| 验证阈值 | `--valid-threshold` | 0.8 | float | 验证时相似度阈值 |

**代码中定义但未暴露的超参数：**
- `gamma` (默认: 0.5) - 学习率衰减系数

**示例命令：**
```bash
# 使用默认参数
./train.sh

# 自定义参数
./train.sh --epochs 20 --batch-size 32 --lr 0.0005 --layers 4
```

---

### 2. VulBERTa (漏洞检测)

**训练命令：** `./train.sh [OPTIONS]`

**可通过命令行修改的超参数：**

| 参数名 | 命令行选项 | MLP默认值 | CNN默认值 | 类型 | 说明 |
|--------|-----------|----------|----------|------|------|
| 批次大小 | `--batch_size` | 4 | 128 | int | 训练批次大小 |
| 训练轮数 | `--epochs` | 10 | 20 | int | 训练epoch数 |
| 学习率 | `--learning_rate` | 3e-05 | 0.0005 | float | 初始学习率 |
| 随机种子 | `--seed` | 42 | 1234 | int | 随机数种子 |
| 混合精度 | `--fp16` | False | False | bool | 使用FP16训练 |

**注意事项：**
- VulBERTa支持两种模型架构：MLP和CNN，它们使用不同的默认超参数
- 必须指定模型名称（`-n mlp` 或 `-n cnn`）和数据集（`-d dataset_name`）

**示例命令：**
```bash
# 训练MLP模型
./train.sh -n mlp -d devign --batch_size 2 --epochs 5

# 训练CNN模型
./train.sh -n cnn -d devign --batch_size 64 --epochs 10
```

---

### 3. pytorch_resnet_cifar10 (图像分类)

**训练命令：** `./train.sh [OPTIONS]`

**可通过命令行修改的超参数：**

| 参数名 | 命令行选项 | 默认值 | 类型 | 说明 |
|--------|-----------|--------|------|------|
| 模型架构 | `-n, --name` | resnet20 | str | ResNet变体（20/32/44/56/110/1202）|
| 训练轮数 | `-e, --epochs` | 200 | int | 训练epoch数 |
| 批次大小 | `-b, --batch-size` | 128 | int | 训练批次大小 |
| 学习率 | `--lr` | 0.1 | float | 初始学习率 |
| SGD动量 | `--momentum` | 0.9 | float | SGD优化器动量 |
| 权重衰减 | `--wd` | 0.0001 | float | L2正则化系数 |
| 数据加载线程 | `-j, --workers` | 4 | int | 数据加载的worker数 |
| 打印频率 | `--print-freq` | 50 | int | 日志打印间隔（批次）|
| 保存频率 | `--save-every` | 10 | int | 模型保存间隔（epochs）|
| 半精度训练 | `--half` | False | bool | 使用FP16训练 |

**可选的ResNet架构：**
- resnet20 (20层)
- resnet32 (32层)
- resnet44 (44层)
- resnet56 (56层)
- resnet110 (110层)
- resnet1202 (1202层，需要更多显存）

**示例命令：**
```bash
# 训练ResNet20
./train.sh -n resnet20

# 训练ResNet56，自定义参数
./train.sh -n resnet56 -e 100 -b 64 --lr 0.05

# 使用半精度训练ResNet1202
./train.sh -n resnet1202 -b 32 --half
```

---

### 4. bug-localization-by-dnn-and-rvsm (缺陷定位)

**训练命令：** `./train.sh -n dnn [OPTIONS]`

**可通过命令行修改的超参数（仅DNN模型）：**

| 参数名 | 命令行选项 | 默认值 | 类型 | 说明 |
|--------|-----------|--------|------|------|
| K折交叉验证 | `--kfold` | 10 | int | 交叉验证折数 |
| 隐藏层大小 | `--hidden_sizes` | 300 | int/list | 隐藏层神经元数（可多个）|
| L2正则化 | `--alpha` | 1e-5 | float | L2惩罚参数 |
| 最大迭代次数 | `--max_iter` | 10000 | int | 最大训练迭代数 |
| 早停patience | `--n_iter_no_change` | 30 | int | 无改进时的容忍次数 |
| 优化器 | `--solver` | sgd | str | 优化器类型（sgd/adam/lbfgs）|
| 并行作业数 | `--n_jobs` | -2 | int | 并行训练的作业数 |

**注意事项：**
- 此模型使用`max_iter`而非`epochs`作为训练长度控制
- `hidden_sizes`可以指定多个值来创建多层网络，例如：`--hidden_sizes 300 200`

**示例命令：**
```bash
# 默认配置训练DNN
./train.sh -n dnn

# 自定义配置
./train.sh -n dnn --hidden_sizes 200 --kfold 5 --solver adam
```

---

### 5. Person_reID_baseline_pytorch (行人重识别)

此仓库包含3个模型：**DenseNet121**, **HRNet18**, **PCB**

**训练命令：** `./train.sh -n model_name [OPTIONS]`

#### 通用超参数（3个模型共享）

| 参数名 | 命令行选项 | 默认值范围 | 类型 | 说明 |
|--------|-----------|-----------|------|------|
| 批次大小 | `--batchsize` | 24-32 | int | 训练批次大小 |
| 学习率 | `--lr` | 0.02-0.05 | float | 初始学习率 |
| 训练轮数 | `--total_epoch` | 60 | int | 总训练epoch数 |
| 预热轮数 | `--warm_epoch` | 0 | int | 学习率预热的epoch数 |
| ResNet步长 | `--stride` | 2 | int | ResNet最后卷积层步长 |
| 随机擦除概率 | `--erasing_p` | 0 | float | Random Erasing概率[0,1] |
| Dropout率 | `--droprate` | 0.5 | float | Dropout概率 |
| 线性特征维度 | `--linear_num` | 512 | int | 全连接层特征维度 |
| 权重衰减 | `--weight_decay` | 5e-4 | float | L2正则化系数 |

#### 精度选项

| 参数名 | 命令行选项 | 默认值 | 说明 |
|--------|-----------|--------|------|
| FP16 | `--fp16` | False | 使用float16精度 |
| BF16 | `--bf16` | False | 使用bfloat16精度 |

#### 损失函数选项

| 参数名 | 命令行选项 | 默认值 | 说明 |
|--------|-----------|--------|------|
| Circle Loss | `--circle` | False | 使用Circle损失 |
| Triplet Loss | `--triplet` | False | 使用Triplet损失 |
| Contrastive Loss | `--contrast` | False | 使用对比损失 |
| ArcFace Loss | `--arcface` | False | 使用ArcFace损失 |
| CosFace Loss | `--cosface` | False | 使用CosFace损失 |

#### 学习率调度

| 参数名 | 命令行选项 | 默认值 | 说明 |
|--------|-----------|--------|------|
| 余弦调度 | `--cosine` | False | 使用余弦退火学习率 |

#### 各模型特定默认值

| 模型 | 批次大小 | 学习率 | 特殊配置 |
|------|---------|--------|---------|
| **DenseNet121** | 24 | 0.05 | `--use_dense` |
| **HRNet18** | 24 | 0.05 | `--use_hr` |
| **PCB** | 32 | 0.02 | `--PCB` |

**示例命令：**
```bash
# 训练DenseNet121（默认配置）
./train.sh -n densenet121

# 训练HRNet18，使用Circle Loss
./train.sh -n hrnet18 --circle --warm_epoch 5

# 训练PCB，自定义参数
./train.sh -n pcb --batchsize 16 --lr 0.01 --total_epoch 40
```

---

### 6. examples (PyTorch示例模型)

此仓库包含4个示例模型：**MNIST CNN**, **MNIST RNN**, **MNIST Forward-Forward**, **Siamese Network**

**训练命令：** `./train.sh -n model_name [OPTIONS]`

#### 6.1 MNIST CNN

| 参数名 | 命令行选项 | 默认值 | 类型 | 说明 |
|--------|-----------|--------|------|------|
| 批次大小 | `-b, --batch-size` | 32 | int | 训练批次大小 |
| 测试批次大小 | `--test-batch-size` | 1000 | int | 测试批次大小 |
| 训练轮数 | `-e, --epochs` | 14 | int | 训练epoch数 |
| 学习率 | `-l, --lr` | 1.0 | float | 初始学习率 |
| Gamma | `--gamma` | 0.7 | float | 学习率衰减系数 |
| 随机种子 | `--seed` | 1 | int | 随机数种子 |
| 日志间隔 | `--log-interval` | 10 | int | 日志打印间隔（批次）|

**示例命令：**
```bash
./train.sh -n mnist -e 10 -b 64 -l 0.5
```

#### 6.2 MNIST RNN

| 参数名 | 命令行选项 | 默认值 | 类型 | 说明 |
|--------|-----------|--------|------|------|
| 批次大小 | `-b, --batch-size` | 32 | int | 训练批次大小 |
| 测试批次大小 | `--test-batch-size` | 1000 | int | 测试批次大小 |
| 训练轮数 | `-e, --epochs` | 14 | int | 训练epoch数 |
| 学习率 | `-l, --lr` | 0.1 | float | 初始学习率 |
| Gamma | `--gamma` | 0.7 | float | 学习率衰减系数 |
| 随机种子 | `--seed` | 1 | int | 随机数种子 |
| 日志间隔 | `--log-interval` | 10 | int | 日志打印间隔（批次）|

**示例命令：**
```bash
./train.sh -n mnist_rnn -e 10 -b 64 -l 0.05
```

#### 6.3 MNIST Forward-Forward

| 参数名 | 命令行选项 | 默认值 | 类型 | 说明 |
|--------|-----------|--------|------|------|
| 训练轮数 | `-e, --epochs` | 1000 | int | 每层训练的epoch数 |
| 学习率 | `-l, --lr` | 0.03 | float | 学习率 |
| 随机种子 | `--seed` | 1 | int | 随机数种子 |
| 训练集大小 | `--train-size` | 50000 | int | 训练样本数 |
| 测试集大小 | `--test-size` | 10000 | int | 测试样本数 |
| 阈值 | `--threshold` | 2 | float | Forward-Forward训练阈值 |
| 日志间隔 | `--log-interval` | 10 | int | 日志打印间隔 |

**注意：** 此模型使用`train_size`和`test_size`而非传统的`batch_size`

**示例命令：**
```bash
./train.sh -n mnist_ff -e 500 -l 0.01 --threshold 1.5
```

#### 6.4 Siamese Network

| 参数名 | 命令行选项 | 默认值 | 类型 | 说明 |
|--------|-----------|--------|------|------|
| 批次大小 | `-b, --batch-size` | 32 | int | 训练批次大小 |
| 测试批次大小 | `--test-batch-size` | 1000 | int | 测试批次大小 |
| 训练轮数 | `-e, --epochs` | 14 | int | 训练epoch数 |
| 学习率 | `-l, --lr` | 1.0 | float | 初始学习率 |
| Gamma | `--gamma` | 0.7 | float | 学习率衰减系数 |
| 随机种子 | `--seed` | 1 | int | 随机数种子 |
| 日志间隔 | `--log-interval` | 10 | int | 日志打印间隔（批次）|

**示例命令：**
```bash
./train.sh -n siamese -e 10 -b 64 -l 0.5
```

---

## 多模型共有超参数

此节列出可以在多个模型中同时修改的超参数，适合进行横向对比的变异测试。

### 1. 核心训练超参数

#### 1.1 epochs (训练轮数)

**覆盖范围：** 10/11个模型（除bug-localization）

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| MRT-OAST | `--epochs` | 10 |
| VulBERTa | `--epochs` | 10-20 |
| pytorch_resnet_cifar10 | `--epochs` | 200 |
| Person_reID (3个) | `--total_epoch` | 60 |
| examples (4个) | `--epochs` | 14-1000 |

**变异建议：**
- ×0.5: 快速测试（5, 10, 100）
- ×2: 长时间训练（20, 40, 400）
- ±20%: 微调（8, 12, 48, 72）

#### 1.2 batch_size (批次大小)

**覆盖范围：** 10/11个模型（除bug-localization）

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| MRT-OAST | `--batch-size` | 64 |
| VulBERTa | `--batch_size` | 4-128 |
| pytorch_resnet_cifar10 | `--batch-size` | 128 |
| Person_reID (3个) | `--batchsize` | 24-32 |
| examples (4个) | `--batch-size` | 32-64 |

**变异建议：**
- 标准值：16, 32, 64, 128, 256
- 注意：需根据GPU显存调整

#### 1.3 learning_rate (学习率)

**覆盖范围：** 10/11个模型（除bug-localization）

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| MRT-OAST | `--lr` | 0.0001 |
| VulBERTa | `--learning_rate` | 0.00003-0.0005 |
| pytorch_resnet_cifar10 | `--lr` | 0.1 |
| Person_reID (3个) | `--lr` | 0.02-0.05 |
| examples (4个) | `--lr` | 0.03-1.0 |

**变异建议：**
- ×10: 0.001, 0.001, 1.0
- ×0.1: 0.00001, 0.00003, 0.01
- ×2: 0.0002, 0.0001, 0.2
- ×0.5: 0.00005, 0.00015, 0.05

#### 1.4 seed (随机种子)

**覆盖范围：** 6/11个模型

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| MRT-OAST | `--seed` | 1334 |
| VulBERTa | `--seed` | 42-1234 |
| examples (4个) | `--seed` | 1 |

**变异建议：**
- 常用种子：1, 42, 123, 1234, 2024, 9999

### 2. 正则化超参数

#### 2.1 dropout (Dropout率)

**覆盖范围：** 4/11个模型

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| MRT-OAST | `--dropout` | 0.2 |
| Person_reID (3个) | `--droprate` | 0.5 |

**变异建议：**
- 0.0 (无Dropout)
- 0.1, 0.2, 0.3, 0.5, 0.7

#### 2.2 weight_decay (权重衰减/L2正则化)

**覆盖范围：** 4/11个模型

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| pytorch_resnet_cifar10 | `--weight-decay` | 0.0001 |
| bug-localization | `--alpha` | 0.00001 |
| Person_reID (3个) | `--weight_decay` | 0.0005 |

**变异建议：**
- 0.0001, 0.0005, 0.001, 0.00001, 0.000001

### 3. 学习率调度超参数

#### 3.1 gamma (学习率衰减系数)

**覆盖范围：** 4/11个模型

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| examples (MNIST, RNN, Siamese) | `--gamma` | 0.7 |

**变异建议：**
- 0.5, 0.7, 0.9, 0.95

### 4. 混合精度训练

#### 4.1 fp16/half/bf16

**覆盖范围：** 4/11个模型

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| VulBERTa | `--fp16` | False |
| pytorch_resnet_cifar10 | `--half` | False |
| Person_reID (3个) | `--fp16, --bf16` | False |

**变异建议：**
- 开启/关闭混合精度，观察速度和精度变化

---

## 代码级可变超参数

如果允许修改训练代码，以下超参数可以在所有或大部分模型中统一进行变异测试。

### 1. 优化器相关（11/11模型）

#### 1.1 optimizer_type (优化器类型)

**当前使用情况：**

| 模型 | 当前优化器 | 可替换为 |
|------|-----------|---------|
| MRT-OAST | Adam | SGD, AdamW, RMSprop |
| VulBERTa | AdamW | Adam, SGD |
| pytorch_resnet_cifar10 | SGD | Adam, AdamW |
| bug-localization | SGD/Adam/LBFGS | 互相替换 |
| Person_reID (3个) | SGD | Adam, AdamW |
| examples (4个) | Adadelta/Adam | SGD, AdamW |

**实施方法：**
```python
# 原代码示例
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# 修改为
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
```

#### 1.2 momentum (SGD动量)

**当前暴露情况：**
- pytorch_resnet_cifar10: 已暴露（`--momentum`，默认0.9）
- 其他模型：需要在代码中修改

**变异建议：** 0.0, 0.5, 0.9, 0.95, 0.99

#### 1.3 beta1, beta2 (Adam优化器参数)

**适用模型：** 所有使用Adam/AdamW的模型（约6个）

**默认值：** (0.9, 0.999)

**变异建议：**
- beta1: 0.5, 0.9, 0.95
- beta2: 0.99, 0.999, 0.9999

### 2. 学习率调度策略（11/11模型）

#### 2.1 lr_scheduler_type (调度器类型)

**当前使用情况：**

| 模型 | 当前调度器 |
|------|-----------|
| MRT-OAST | 固定学习率 |
| VulBERTa | 线性调度 |
| pytorch_resnet_cifar10 | MultiStepLR |
| Person_reID | StepLR（可选Cosine）|
| examples | StepLR |

**可替换选项：**
- StepLR: 每隔固定epoch降低学习率
- MultiStepLR: 在指定epoch降低学习率
- ExponentialLR: 指数衰减
- CosineAnnealingLR: 余弦退火
- ReduceLROnPlateau: 基于验证指标自适应调整

**实施方法：**
```python
if args.scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.scheduler == 'multistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
```

#### 2.2 lr_scheduler_params (调度器参数)

**StepLR参数：**
- `step_size`: 10, 20, 30, 50
- `gamma`: 0.1, 0.2, 0.5

**MultiStepLR参数：**
- `milestones`: [60, 120], [80, 150], [100, 200]
- `gamma`: 0.1, 0.2

**CosineAnnealingLR参数：**
- `T_max`: 等于总epochs或总epochs的一半

### 3. 梯度处理（11/11模型）

#### 3.1 gradient_clipping (梯度裁剪)

**当前使用情况：** 大部分模型未使用

**实施方法：**
```python
# 在optimizer.step()之前添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
```

**变异建议：**
- max_norm: 0.5, 1.0, 5.0, 10.0
- 或不使用梯度裁剪

### 4. 训练策略（11/11模型）

#### 4.1 early_stopping_patience (早停)

**当前使用情况：**
- bug-localization: 已实现（`n_iter_no_change=30`）
- 其他模型：大部分未实现

**实施方法：**
```python
best_loss = float('inf')
patience_counter = 0

for epoch in range(args.epochs):
    train_loss = train(...)
    val_loss = validate(...)

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        save_checkpoint(...)
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

**变异建议：** patience=5, 10, 20, 30

#### 4.2 warm_up_epochs (学习率预热)

**当前使用情况：**
- Person_reID: 已暴露（`--warm_epoch`，默认0）
- 其他模型：需要在代码中添加

**实施方法：**
```python
if epoch < args.warm_epochs:
    warmup_lr = args.lr * (epoch + 1) / args.warm_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_lr
```

**变异建议：** 0（无预热），5, 10, 20 epochs

### 5. 数据增强（适用于图像模型：8/11）

#### 5.1 数据增强策略

**适用模型：**
- pytorch_resnet_cifar10
- Person_reID (3个)
- examples (MNIST相关4个)

**当前使用情况：**
- Person_reID: 支持Random Erasing（`--erasing_p`），Color Jitter（`--color_jitter`）
- pytorch_resnet_cifar10: 基础增强（RandomCrop, RandomFlip）

**可添加的增强：**
- RandomRotation
- RandomAffine
- ColorJitter
- RandomGrayscale
- GaussianBlur
- Cutout/Random Erasing
- MixUp
- CutMix

### 6. 模型架构参数

#### 6.1 hidden_size / hidden_units (隐藏层大小)

**当前使用情况：**

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| MRT-OAST | `d_model` | 128 |
| bug-localization | `hidden_sizes` | 300 |
| Person_reID | `linear_num` | 512 |
| examples-FF | 硬编码 | [784, 500, 500] |

**变异建议：**
- 减半：64, 150, 256, [784, 250, 250]
- 加倍：256, 600, 1024, [784, 1000, 1000]

#### 6.2 num_layers (网络层数)

**当前使用情况：**

| 模型 | 参数名 | 默认值 |
|------|--------|--------|
| MRT-OAST | `transformer_nlayers` | 2 |
| pytorch_resnet_cifar10 | 通过arch选择 | 20-1202 |
| examples-FF | 硬编码 | 2 |

**变异建议：**
- Transformer: 1, 2, 4, 6层
- ResNet: 20, 32, 56, 110层
- MLP: 2, 3, 4层

---

## 变异测试实施建议

### 1. 三层变异测试方案

#### 第一层：命令行参数变异（无需改代码）

**推荐优先级：🔥 最高**

专注于可以通过修改train.sh调用参数实现的变异：

| 超参数 | 覆盖模型 | 变异方案 | 预期影响 |
|--------|---------|---------|---------|
| **epochs** | 10/11 | ×0.5, ×2 | 训练时间、收敛性 |
| **batch_size** | 10/11 | 16, 32, 64, 128 | 内存占用、收敛速度 |
| **learning_rate** | 10/11 | ×0.1, ×0.5, ×2, ×10 | 收敛速度、最终精度 |
| **seed** | 6/11 | 1, 42, 123, 1234 | 结果可重复性 |

**实施步骤：**

1. 创建变异参数配置文件：
```bash
# mutation_configs.txt
epochs: 5, 10, 20, 40
batch_size: 16, 32, 64, 128
lr_multiplier: 0.1, 0.5, 1.0, 2.0, 10.0
seed: 1, 42, 123, 1234
```

2. 编写批量测试脚本：
```bash
#!/bin/bash
# batch_mutation_test.sh

for epochs in 5 10 20; do
    for batch in 16 32 64; do
        for lr_mult in 0.1 0.5 2.0; do
            lr=$(echo "scale=6; $DEFAULT_LR * $lr_mult" | bc)
            echo "Testing: epochs=$epochs, batch=$batch, lr=$lr"
            ./train.sh --epochs $epochs --batch-size $batch --lr $lr \
                2>&1 | tee "logs/mutation_e${epochs}_b${batch}_lr${lr}.log"
        done
    done
done
```

#### 第二层：代码级常见参数变异（简单修改）

**推荐优先级：🔥 高**

修改代码添加常用超参数：

**需要修改的内容：**

1. **添加优化器选择：**
```python
# 在argparse中添加
parser.add_argument('--optimizer', type=str, default='adam',
                   choices=['sgd', 'adam', 'adamw', 'rmsprop'])

# 在训练代码中添加
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
```

2. **添加学习率调度器选择：**
```python
# 在argparse中添加
parser.add_argument('--scheduler', type=str, default='step',
                   choices=['none', 'step', 'cosine', 'multistep'])

# 在训练代码中添加
if args.scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
```

3. **添加梯度裁剪：**
```python
# 在argparse中添加
parser.add_argument('--clip_grad', type=float, default=0.0,
                   help='gradient clipping max norm (0 means no clipping)')

# 在训练循环中添加（optimizer.step()之前）
if args.clip_grad > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
```

#### 第三层：深度架构参数变异（复杂修改）

**推荐优先级：中**

修改模型架构相关参数：

- 隐藏层大小
- 网络层数
- 卷积核大小
- 注意力头数

**实施难度：** 需要对模型架构有深入理解

### 2. 分阶段实施计划

#### Phase 1: 快速探索（1-2周）

**目标：** 验证变异测试框架可行性

**范围：** 仅测试3个最常用超参数
- epochs: ×0.5, ×2
- batch_size: ×0.5, ×2
- learning_rate: ×0.1, ×10

**预期实验数量：**
- 11个模型 × 3个参数 × 2个变异值 = 66次实验

**实施方式：** 手动修改train.sh参数

#### Phase 2: 系统化测试（2-4周）

**目标：** 全面测试命令行可修改参数

**范围：**
- epochs: 3-5个不同值
- batch_size: 4-5个不同值
- learning_rate: 5个不同值
- seed: 4个不同值
- dropout: 3个不同值（适用模型）

**预期实验数量：** 约200-300次

**实施方式：** 编写自动化批量测试脚本

#### Phase 3: 深度探索（4-8周）

**目标：** 测试代码级可变参数

**范围：**
- 优化器类型：4种
- 学习率调度器：4种
- 梯度裁剪：3个值
- 早停策略：3个值
- 数据增强：多种组合

**预期实验数量：** 约500+次

**实施方式：** 修改代码，使用配置文件管理参数

### 3. 变异测试自动化框架

#### 3.1 配置文件驱动方案

**创建统一配置格式（YAML）：**

```yaml
# config/mutation_test_001.yaml
model: MRT-OAST
hyperparameters:
  epochs: 20
  batch_size: 32
  lr: 0.0002
  dropout: 0.3
  seed: 42

# config/mutation_test_002.yaml
model: VulBERTa
hyperparameters:
  model_name: mlp
  epochs: 15
  batch_size: 8
  learning_rate: 0.00006
  seed: 123
```

**创建配置文件生成器：**

```python
# generate_configs.py
import yaml
import itertools

base_config = {
    'model': 'MRT-OAST',
    'hyperparameters': {
        'epochs': 10,
        'batch_size': 64,
        'lr': 0.0001,
        'seed': 1334
    }
}

mutations = {
    'epochs': [5, 10, 20],
    'batch_size': [32, 64, 128],
    'lr': [0.00001, 0.0001, 0.001]
}

# 生成所有组合
configs = []
for epochs, batch, lr in itertools.product(
    mutations['epochs'],
    mutations['batch_size'],
    mutations['lr']
):
    config = base_config.copy()
    config['hyperparameters'] = {
        'epochs': epochs,
        'batch_size': batch,
        'lr': lr,
        'seed': base_config['hyperparameters']['seed']
    }
    configs.append(config)

# 保存配置文件
for i, config in enumerate(configs):
    with open(f'config/mutation_{i:03d}.yaml', 'w') as f:
        yaml.dump(config, f)
```

#### 3.2 批量执行框架

```bash
#!/bin/bash
# run_mutation_tests.sh

CONFIG_DIR="config"
LOG_DIR="mutation_logs"
mkdir -p "$LOG_DIR"

for config_file in "$CONFIG_DIR"/mutation_*.yaml; do
    config_name=$(basename "$config_file" .yaml)
    log_file="$LOG_DIR/${config_name}.log"

    echo "Running test: $config_name"

    # 从YAML读取参数并执行训练
    python run_from_config.py --config "$config_file" \
        2>&1 | tee "$log_file"

    # 记录退出状态
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $config_name" >> "$LOG_DIR/summary.txt"
    else
        echo "FAILED: $config_name" >> "$LOG_DIR/summary.txt"
    fi
done
```

#### 3.3 结果收集与分析

```python
# analyze_results.py
import os
import re
import pandas as pd

def extract_metrics(log_file):
    """从日志文件提取性能指标"""
    with open(log_file, 'r') as f:
        content = f.read()

    metrics = {}

    # 提取准确率（根据不同模型调整正则表达式）
    acc_match = re.search(r'Accuracy[:\s]+(\d+\.?\d*)', content)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))

    # 提取损失
    loss_match = re.search(r'(?:Final|Test)\s+[Ll]oss[:\s]+(\d+\.?\d*)', content)
    if loss_match:
        metrics['loss'] = float(loss_match.group(1))

    # 提取训练时间
    time_match = re.search(r'Total [Dd]uration[:\s]+(\d+)h\s*(\d+)m', content)
    if time_match:
        hours = int(time_match.group(1))
        minutes = int(time_match.group(2))
        metrics['training_time_minutes'] = hours * 60 + minutes

    return metrics

# 收集所有结果
results = []
for log_file in os.listdir('mutation_logs'):
    if log_file.endswith('.log'):
        config_name = log_file.replace('.log', '')
        metrics = extract_metrics(f'mutation_logs/{log_file}')

        # 从配置名解析参数
        # 例如: mutation_e10_b64_lr0.001.log
        params = parse_config_name(config_name)

        results.append({
            'config': config_name,
            **params,
            **metrics
        })

# 保存为CSV
df = pd.DataFrame(results)
df.to_csv('mutation_test_results.csv', index=False)

# 生成分析报告
print("=== Mutation Test Summary ===")
print(f"Total tests: {len(df)}")
print(f"\nBest accuracy: {df['accuracy'].max():.4f}")
print(f"Worst accuracy: {df['accuracy'].min():.4f}")
print(f"\nBest config:\n{df.loc[df['accuracy'].idxmax()]}")
```

### 4. 注意事项与最佳实践

#### 4.1 资源管理

1. **GPU资源：**
   - 使用队列管理系统避免资源冲突
   - 考虑使用`nvidia-smi`监控GPU使用率
   - 为不同实验分配不同GPU：`CUDA_VISIBLE_DEVICES=0 ./train.sh ...`

2. **存储空间：**
   - 定期清理中间模型检查点
   - 仅保存最佳模型
   - 压缩日志文件

3. **时间管理：**
   - 优先测试快速模型（examples系列）
   - 对长时间训练模型（ResNet200 epochs）减少变异数量
   - 使用`--epochs 5`快速验证脚本正确性

#### 4.2 实验设计原则

1. **控制变量：**
   - 每次只改变一个超参数
   - 保持其他参数为默认值
   - 使用相同的随机种子（除非测试seed影响）

2. **重复实验：**
   - 对关键发现进行3-5次重复实验
   - 使用不同seed验证结果稳定性

3. **记录完整信息：**
   - 保存完整的训练日志
   - 记录环境信息（GPU型号、PyTorch版本等）
   - 记录数据集版本

#### 4.3 常见问题处理

1. **OOM (Out of Memory)：**
   - 减小batch_size
   - 启用混合精度训练（--fp16）
   - 使用梯度累积

2. **训练不收敛：**
   - 减小learning_rate
   - 增加warm_up_epochs
   - 检查数据预处理

3. **训练过慢：**
   - 增大batch_size（在显存允许范围内）
   - 减少数据加载workers
   - 启用混合精度训练

---

## 附录：超参数速查表

### 表A1: 所有模型超参数对比矩阵

| 超参数 | MRT-OAST | VulBERTa | ResNet | Bug-Loc | DenseNet | HRNet | PCB | MNIST | MNIST-RNN | MNIST-FF | Siamese |
|--------|---------|---------|--------|---------|---------|-------|-----|-------|-----------|----------|---------|
| **epochs** | ✓ (10) | ✓ (10-20) | ✓ (200) | ✗ | ✓ (60) | ✓ (60) | ✓ (60) | ✓ (14) | ✓ (14) | ✓ (1000) | ✓ (14) |
| **batch_size** | ✓ (64) | ✓ (4-128) | ✓ (128) | ✗ | ✓ (24) | ✓ (24) | ✓ (32) | ✓ (32) | ✓ (32) | ✗ | ✓ (32) |
| **learning_rate** | ✓ (0.0001) | ✓ (3e-5~5e-4) | ✓ (0.1) | ✗ | ✓ (0.05) | ✓ (0.05) | ✓ (0.02) | ✓ (1.0) | ✓ (0.1) | ✓ (0.03) | ✓ (1.0) |
| **seed** | ✓ (1334) | ✓ (42-1234) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (1) | ✓ (1) | ✓ (1) | ✓ (1) |
| **dropout** | ✓ (0.2) | ✗ | ✗ | ✗ | ✓ (0.5) | ✓ (0.5) | ✓ (0.5) | ✗ | ✗ | ✗ | ✗ |
| **weight_decay** | ✗ | ✗ | ✓ (1e-4) | ✓ (1e-5) | ✓ (5e-4) | ✓ (5e-4) | ✓ (5e-4) | ✗ | ✗ | ✗ | ✗ |
| **momentum** | ✗ | ✗ | ✓ (0.9) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **gamma** | 代码 (0.5) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (0.7) | ✓ (0.7) | ✗ | ✓ (0.7) |
| **mixed_precision** | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **warm_epoch** | ✗ | ✗ | ✗ | ✗ | ✓ (0) | ✓ (0) | ✓ (0) | ✗ | ✗ | ✗ | ✗ |

**图例：**
- ✓ (值): 支持且可通过命令行修改，括号内为默认值
- 代码 (值): 在代码中定义但未暴露到命令行
- ✗: 不支持或未实现

### 表A2: 推荐变异优先级

| 优先级 | 超参数 | 覆盖模型数 | 变异难度 | 预期影响 |
|-------|--------|-----------|---------|---------|
| 🔥 P0 | epochs | 10/11 | ⭐ 低 | 训练时间、收敛性 |
| 🔥 P0 | batch_size | 10/11 | ⭐ 低 | 收敛速度、内存 |
| 🔥 P0 | learning_rate | 10/11 | ⭐ 低 | 收敛速度、精度 |
| 🔥 P1 | seed | 6/11 | ⭐ 低 | 结果稳定性 |
| 📊 P1 | optimizer | 11/11 | ⭐⭐ 中 | 收敛性能 |
| 📊 P1 | lr_scheduler | 11/11 | ⭐⭐ 中 | 训练稳定性 |
| 📊 P2 | dropout | 4/11 | ⭐ 低 | 过拟合控制 |
| 📊 P2 | weight_decay | 4/11 | ⭐ 低 | 泛化能力 |
| 📊 P2 | gradient_clip | 11/11 | ⭐⭐ 中 | 训练稳定性 |
| 🔧 P3 | hidden_size | 11/11 | ⭐⭐⭐ 高 | 模型容量 |
| 🔧 P3 | num_layers | 11/11 | ⭐⭐⭐ 高 | 模型深度 |

### 表A3: 各仓库train.sh位置

| 仓库 | train.sh路径 | 训练脚本路径 |
|------|------------|------------|
| MRT-OAST | `/home/green/energy_dl/success/MRT-OAST/train.sh` | `main_batch.py` |
| VulBERTa | `/home/green/energy_dl/success/VulBERTa/train.sh` | `train_vulberta.py` |
| pytorch_resnet_cifar10 | `/home/green/energy_dl/success/pytorch_resnet_cifar10/train.sh` | `trainer.py` |
| bug-localization | `/home/green/energy_dl/success/bug-localization-by-dnn-and-rvsm/train.sh` | `train_wrapper.py` |
| Person_reID | `/home/green/energy_dl/success/Person_reID_baseline_pytorch/train.sh` | `train.py` |
| examples | `/home/green/energy_dl/success/examples/train.sh` | 各子目录的`main.py` |

---

## 版本历史

- **v1.0** (2025-11-03): 初始版本，完成11个模型的超参数分析

---

**文档维护者：** Claude Code
**最后更新：** 2025-11-03
