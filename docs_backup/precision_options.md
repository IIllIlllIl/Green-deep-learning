# 深度学习训练精度选项全面指南

生成时间: 2025-11-04

---

## 目录

1. [精度类型概览](#1-精度类型概览)
2. [各精度选项详细说明](#2-各精度选项详细说明)
3. [精度对比分析](#3-精度对比分析)
4. [实现方法](#4-实现方法)
5. [推荐配置](#5-推荐配置)

---

## 1. 精度类型概览

### 1.1 完整精度选项列表

| 精度类型 | 位数 | 范围 | 精度 | GPU支持 | 推荐场景 |
|---------|------|------|------|---------|---------|
| **FP64** | 64位 | ±1.7e308 | 15-17位 | 所有GPU | 科学计算 |
| **FP32** | 32位 | ±3.4e38 | 6-9位 | 所有GPU | 标准训练（默认） |
| **TF32** | 19位 | 同FP32 | 3位尾数 | Ampere+ | 加速FP32训练 |
| **BF16** | 16位 | 同FP32 | 2-3位 | Ampere+ | 稳定混合精度 |
| **FP16** | 16位 | ±65504 | 3-4位 | Volta+ | 传统混合精度 |
| **INT8** | 8位 | -128~127 | 整数 | 所有GPU | 推理/量化训练 |
| **INT4** | 4位 | -8~7 | 整数 | 特定硬件 | 极限压缩 |

### 1.2 精度层次结构

```
高精度（慢，准确）
    ↓
FP64 (双精度)
    ↓
FP32 (单精度) ← 默认
    ↓
TF32 (Tensor Float 32) ← Ampere GPU自动启用
    ↓
BF16 (Brain Float 16) ← 推荐混合精度
    ↓
FP16 (Half Precision) ← 传统混合精度
    ↓
INT8 (8位整数) ← 量化
    ↓
INT4/INT2/INT1 (极低精度)
    ↓
低精度（快，能耗低）
```

---

## 2. 各精度选项详细说明

### 2.1 FP64 (双精度浮点数)

**技术规格**:
- **位数**: 64位 (1符号位 + 11指数位 + 52尾数位)
- **数值范围**: ±1.7e308
- **精度**: 约15-17位有效数字

**优点**:
- ✅ 最高精度，几乎无舍入误差
- ✅ 适合需要极高精度的科学计算
- ✅ 所有GPU都支持

**缺点**:
- ❌ 速度慢（通常比FP32慢2-4倍）
- ❌ 内存占用大（是FP32的2倍）
- ❌ 能耗高（约为FP32的2倍）

**适用场景**:
- 科学计算、数值模拟
- 对精度要求极高的研究
- **深度学习训练**: 几乎不使用

**性能影响**:
- 速度: -50% 到 -75% (vs FP32)
- 能耗: +100% (vs FP32)
- 性能: 无明显提升（深度学习对FP64不敏感）

**PyTorch实现**:
```python
model = model.double()  # 转换为FP64
optimizer = torch.optim.Adam(model.parameters())

# 或者指定dtype
model = MyModel().to(dtype=torch.float64)
```

**不推荐用于深度学习训练**。

---

### 2.2 FP32 (单精度浮点数) ⭐ 默认

**技术规格**:
- **位数**: 32位 (1符号位 + 8指数位 + 23尾数位)
- **数值范围**: ±3.4e38
- **精度**: 约6-9位有效数字

**优点**:
- ✅ 深度学习的标准精度
- ✅ 所有框架和硬件完美支持
- ✅ 稳定可靠，不需要特殊处理
- ✅ 精度足够高，适合绝大多数任务

**缺点**:
- ❌ 相比低精度，速度和内存不是最优
- ❌ 能耗相对较高

**适用场景**:
- **默认选择**: 所有深度学习训练
- 对精度有要求的任务
- 不支持低精度硬件的情况

**性能影响**:
- 速度: 基线
- 能耗: 基线
- 性能: 基线

**PyTorch实现**:
```python
# 默认就是FP32，无需特殊设置
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
```

**推荐作为基线配置**。

---

### 2.3 TF32 (TensorFloat-32) ⭐⭐⭐

**技术规格**:
- **位数**: 19位 (1符号位 + 8指数位 + 10尾数位)
- **数值范围**: 同FP32
- **精度**: 约3位有效数字

**特点**:
- NVIDIA Ampere架构（A100, RTX 30系列）及以后的GPU专有
- 在FP32模式下**自动启用**，无需代码修改
- 内部使用TF32加速矩阵乘法，但保持FP32的数值范围

**优点**:
- ✅ **无需代码修改**，自动加速
- ✅ 保持FP32的数值稳定性（相同范围）
- ✅ 速度提升显著（约1.5-2倍）
- ✅ 能耗降低（约20-30%）
- ✅ 精度损失极小（<1%）

**缺点**:
- ❌ 仅Ampere及更新架构支持
- ❌ 某些精度敏感任务可能受影响

**适用场景**:
- **Ampere+ GPU的默认选择**
- 所有深度学习训练
- 希望加速但不想改代码的情况

**性能影响** (vs FP32):
- 速度: +50% 到 +100%
- 能耗: -20% 到 -30%
- 性能: -0.1% 到 -0.5%（几乎无影响）

**PyTorch实现**:
```python
# Ampere+ GPU上默认启用，可显式控制

# 启用TF32（默认）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 禁用TF32（如果需要完整FP32精度）
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

**强烈推荐在Ampere+ GPU上使用**。

---

### 2.4 BF16 (Brain Float 16) ⭐⭐⭐⭐⭐

**技术规格**:
- **位数**: 16位 (1符号位 + 8指数位 + 7尾数位)
- **数值范围**: 同FP32 (±3.4e38)
- **精度**: 约2-3位有效数字

**特点**:
- Google为深度学习设计的浮点格式
- **关键优势**: 保持FP32的数值范围，避免溢出/下溢问题
- 需要Ampere架构（A100, RTX 30系列）及以后的GPU

**优点**:
- ✅ **最稳定的混合精度方案**
- ✅ 不需要loss scaling（FP16需要）
- ✅ 数值范围与FP32相同，不易溢出
- ✅ 速度快（接近FP16）
- ✅ 内存减半（vs FP32）
- ✅ 能耗降低显著（20-40%）
- ✅ 精度损失小（0.5-2%）

**缺点**:
- ❌ 仅Ampere+架构支持
- ❌ 某些精度敏感的任务可能受影响

**适用场景**:
- **Ampere+ GPU的首选混合精度**
- 所有深度学习训练（替代FP16）
- Transformer等大模型训练
- 不希望调整loss scaling的情况

**性能影响** (vs FP32):
- 速度: +30% 到 +60%
- 能耗: -20% 到 -40%
- 显存: -50%
- 性能: -0.5% 到 -2%

**PyTorch实现**:
```python
# 方法1: 使用autocast（推荐）
from torch.cuda.amp import autocast

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = torch.cuda.amp.GradScaler()  # BF16也可以用scaler，但非必需

for epoch in range(epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()

        # 使用bfloat16自动混合精度
        with autocast(dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # BF16不一定需要scaler，但使用也无害
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# 方法2: 全局BF16
model = model.to(dtype=torch.bfloat16)
```

**能耗研究强烈推荐**。

---

### 2.5 FP16 (Half Precision) ⭐⭐⭐⭐

**技术规格**:
- **位数**: 16位 (1符号位 + 5指数位 + 10尾数位)
- **数值范围**: ±65504 (非常小！)
- **精度**: 约3-4位有效数字

**特点**:
- 最早广泛使用的混合精度方案
- **数值范围小**，容易发生溢出/下溢
- 需要loss scaling来防止梯度下溢

**优点**:
- ✅ 广泛支持（Volta, Turing, Ampere架构）
- ✅ 速度快（比FP32快1.5-3倍）
- ✅ 内存减半
- ✅ 能耗降低显著（25-50%）

**缺点**:
- ❌ **数值范围小**，容易溢出/下溢
- ❌ **需要loss scaling**，增加调试复杂度
- ❌ 某些层仍需FP32（如BatchNorm）
- ❌ 精度损失可能较大（0.5-3%）

**适用场景**:
- Volta/Turing GPU（不支持BF16时）
- 对显存要求极高的情况
- 已有成熟FP16训练经验的项目

**性能影响** (vs FP32):
- 速度: +50% 到 +200%
- 能耗: -25% 到 -50%
- 显存: -50%
- 性能: -0.5% 到 -3%

**PyTorch实现**:
```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # 必须使用scaler

for epoch in range(epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()

        # 自动混合精度
        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 使用scaler防止梯度下溢
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**在Ampere+ GPU上推荐使用BF16代替**。

---

### 2.6 INT8 量化训练 ⭐⭐

**技术规格**:
- **位数**: 8位整数
- **数值范围**: -128 到 127 (signed) 或 0 到 255 (unsigned)
- **精度**: 256个离散值

**特点**:
- 需要量化感知训练（Quantization-Aware Training, QAT）
- 通常用于模型部署和推理
- 训练时复杂度较高

**优点**:
- ✅ **极小的模型大小**（是FP32的1/4）
- ✅ **极快的推理速度**
- ✅ **极低的能耗**（可降低50-75%）
- ✅ 适合边缘设备部署

**缺点**:
- ❌ **训练复杂**，需要特殊技术
- ❌ **精度损失明显**（可能5-15%）
- ❌ 需要仔细调整量化参数
- ❌ 不是所有层都适合量化

**适用场景**:
- 模型部署到移动设备
- 对推理速度和能耗要求极高
- 可以容忍一定精度损失
- **训练**: 较少使用，主要用于QAT

**性能影响** (vs FP32):
- **推理**速度: +200% 到 +400%
- **推理**能耗: -50% 到 -75%
- 模型大小: -75%
- 性能: -5% 到 -15%（视任务而定）

**PyTorch实现**:
```python
import torch.quantization

# 量化感知训练（QAT）
model = MyModel()

# 配置量化
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# 正常训练
for epoch in range(epochs):
    train_one_epoch(model, dataloader, optimizer)

    # 在某个epoch后冻结观察器
    if epoch > 3:
        model.apply(torch.quantization.disable_observer)

    # 冻结BN统计
    if epoch > 2:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

# 转换为量化模型
model_int8 = torch.quantization.convert(model, inplace=False)
```

**主要用于推理优化，训练较少使用**。

---

### 2.7 极低精度（INT4/INT2/INT1）

**技术规格**:
- **INT4**: 4位整数，16个离散值
- **INT2**: 2位整数，4个离散值
- **INT1**: 1位（二值），2个值（-1, +1）

**特点**:
- 实验性质，研究前沿
- 需要专门的训练技术（如二值网络、三值网络）

**优点**:
- ✅ **极致压缩**
- ✅ **极快速度**（理论上）
- ✅ **极低能耗**

**缺点**:
- ❌ **精度损失巨大**（10-30%）
- ❌ 训练非常困难
- ❌ 硬件支持有限
- ❌ 实用性存疑

**适用场景**:
- 学术研究
- 极端资源受限的边缘设备
- **不推荐用于一般训练**

---

## 3. 精度对比分析

### 3.1 综合对比表

| 精度 | 速度 | 能耗 | 显存 | 精度损失 | 稳定性 | 易用性 | 硬件要求 | 推荐度 |
|------|------|------|------|---------|-------|-------|---------|--------|
| FP64 | 很慢 | 很高 | 2x | 0% | 极佳 | 简单 | 所有GPU | ⭐ |
| FP32 | 基线 | 基线 | 1x | 0% | 极佳 | 简单 | 所有GPU | ⭐⭐⭐ |
| TF32 | +50-100% | -20-30% | 1x | <0.5% | 优秀 | 极简 | Ampere+ | ⭐⭐⭐⭐ |
| BF16 | +30-60% | -20-40% | 0.5x | 0.5-2% | 优秀 | 简单 | Ampere+ | ⭐⭐⭐⭐⭐ |
| FP16 | +50-200% | -25-50% | 0.5x | 0.5-3% | 良好 | 中等 | Volta+ | ⭐⭐⭐⭐ |
| INT8 | +200-400% | -50-75% | 0.25x | 5-15% | 一般 | 困难 | 所有GPU | ⭐⭐ |

### 3.2 性能-精度权衡曲线

```
高精度
  ↑
  |  FP64 (几乎不用)
  |
  |  FP32 (标准)
  |    ↘
  |      TF32 (Ampere自动)
  |        ↘
  |          BF16 (推荐)
  |            ↘
  |              FP16 (传统)
  |                ↘
  |                  INT8 (推理)
  |                    ↘
  |                      INT4/2/1 (实验)
  └────────────────────────────────→ 速度/能耗优化
                                    低能耗
```

### 3.3 GPU架构支持矩阵

| GPU架构 | FP64 | FP32 | TF32 | BF16 | FP16 | INT8 |
|---------|------|------|------|------|------|------|
| **Pascal** (GTX 10系列) | ✅ | ✅ | ❌ | ❌ | ⚠️ | ✅ |
| **Volta** (V100) | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Turing** (RTX 20系列) | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Ampere** (A100, RTX 30系列) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Hopper** (H100) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

⚠️ Pascal的FP16支持有限（性能提升不明显）

### 3.4 任务类型推荐

| 任务类型 | 推荐精度 | 备选精度 | 原因 |
|---------|---------|---------|------|
| **CV - 图像分类** | BF16/FP16 | FP32 | 对精度不敏感，可大幅节能 |
| **CV - 目标检测** | BF16/FP16 | FP32 | 对精度不敏感 |
| **CV - 语义分割** | BF16/FP16 | FP32 | 对精度不敏感 |
| **NLP - Transformer** | BF16 | FP32 | BF16稳定性更好 |
| **NLP - RNN/LSTM** | FP32 | BF16/FP16 | RNN对精度较敏感 |
| **强化学习** | FP32 | BF16 | 数值稳定性重要 |
| **GAN训练** | FP32 | BF16 | 对抗训练敏感 |
| **小模型（<10M参数）** | FP32 | BF16 | 混合精度优势不明显 |
| **大模型（>100M参数）** | BF16 | FP16 | 必须减少显存占用 |

---

## 4. 实现方法

### 4.1 PyTorch自动混合精度（推荐）

**BF16混合精度**:
```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()  # BF16可选，FP16必须

for epoch in range(epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()

        # 自动混合精度上下文
        with autocast(dtype=torch.bfloat16):  # 或 torch.float16
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 梯度缩放（BF16可选，FP16必须）
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**FP16混合精度**:
```python
# 与BF16相同，只需改变dtype
with autocast(dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### 4.2 手动精度控制

**全模型FP16**:
```python
model = model.half()  # 转换为FP16
# 或
model = model.to(dtype=torch.float16)

# 注意：输入也需要转换
inputs = inputs.half()
```

**全模型BF16**:
```python
model = model.to(dtype=torch.bfloat16)
inputs = inputs.to(dtype=torch.bfloat16)
```

### 4.3 混合精度最佳实践

**1. 使用autocast而非手动转换**:
```python
# 好 ✅
with autocast(dtype=torch.bfloat16):
    output = model(input)

# 不好 ❌
model = model.half()
output = model(input.half())
```

**2. 某些操作保持FP32**:
```python
# autocast会自动处理，某些操作保持FP32
with autocast(dtype=torch.bfloat16):
    # 矩阵乘法使用BF16
    hidden = torch.matmul(x, weight)

    # BatchNorm自动使用FP32
    normalized = batch_norm(hidden)

    # Softmax自动使用FP32
    probs = torch.softmax(logits, dim=-1)
```

**3. 梯度裁剪与混合精度**:
```python
scaler = GradScaler()

for inputs, labels in dataloader:
    optimizer.zero_grad()

    with autocast(dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()

    # 梯度裁剪（在scaler.step之前）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

### 4.4 TF32控制

**启用TF32**（Ampere+ GPU默认启用）:
```python
import torch

# 启用TF32加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 正常FP32训练即可，内部自动使用TF32加速
model = MyModel()
# ... 正常训练代码
```

**禁用TF32**（需要完整FP32精度时）:
```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

### 4.5 INT8量化训练

```python
import torch.quantization as quant

# 1. 准备模型
model = MyModel()
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
quant.prepare_qat(model, inplace=True)

# 2. 训练几个epoch
for epoch in range(10):
    train_one_epoch(model, dataloader, optimizer)

    # epoch 3之后禁用观察器
    if epoch == 3:
        model.apply(quant.disable_observer)

    # epoch 2之后冻结BN
    if epoch == 2:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

# 3. 转换为量化模型
model_quantized = quant.convert(model.eval(), inplace=False)

# 4. 评估量化模型
evaluate(model_quantized, test_loader)
```

---

## 5. 推荐配置

### 5.1 按GPU选择精度

#### Ampere架构（A100, RTX 3080/3090, RTX 40系列）

**推荐顺序**:
1. **BF16混合精度**（首选）
   - 最稳定
   - 性能-能耗平衡最佳
   - 无需调试

2. **TF32**（自动启用）
   - 无需代码修改
   - 中等加速

3. **FP16混合精度**（备选）
   - 速度可能略快于BF16
   - 需要更多调试

**配置示例**:
```python
# 方案1: BF16混合精度（推荐）
with autocast(dtype=torch.bfloat16):
    output = model(input)

# 方案2: TF32（自动，无需配置）
# 正常FP32训练即可

# 方案3: FP16混合精度
with autocast(dtype=torch.float16):
    output = model(input)
```

#### Turing架构（RTX 2080/2080Ti）

**推荐顺序**:
1. **FP16混合精度**（首选）
2. **FP32**（稳定性优先时）

**配置示例**:
```python
# FP16混合精度
scaler = GradScaler()
with autocast(dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Volta架构（V100）

**推荐顺序**:
1. **FP16混合精度**（首选）
2. **FP32**（稳定性优先时）

**配置**: 同Turing

#### Pascal架构（GTX 1080/1080Ti）

**推荐**:
- **FP32**（FP16性能提升不明显）

### 5.2 按任务选择精度

#### 计算机视觉任务

**图像分类、目标检测、分割**:
```python
# Ampere+ GPU
with autocast(dtype=torch.bfloat16):
    outputs = model(images)
    loss = criterion(outputs, labels)

# Volta/Turing GPU
with autocast(dtype=torch.float16):
    outputs = model(images)
    loss = criterion(outputs, labels)
```

**GAN训练**（更敏感）:
```python
# 建议先FP32训练，稳定后再尝试混合精度
# 如果使用混合精度，BF16更稳定
with autocast(dtype=torch.bfloat16):
    fake = generator(noise)
    d_fake = discriminator(fake)
    g_loss = criterion(d_fake, real_labels)
```

#### NLP任务

**Transformer模型**:
```python
# BF16是首选（特别是大模型）
with autocast(dtype=torch.bfloat16):
    outputs = transformer(input_ids, attention_mask)
    loss = outputs.loss
```

**RNN/LSTM**:
```python
# 建议FP32（数值稳定性重要）
# 或谨慎使用BF16
outputs = rnn(embeddings)  # FP32
```

### 5.3 超参数调整建议

#### 使用混合精度时的学习率调整

**通常不需要调整学习率**，但如果性能不佳：

```python
# FP32基线
lr = 0.001

# 混合精度可以尝试略微增大学习率
lr_mixed = 0.001 * 1.0  # BF16通常不需要调整
lr_mixed = 0.001 * 1.2  # FP16可能需要略微增大
```

#### Loss Scaling调整

**BF16**: 通常不需要loss scaling
```python
# BF16可以不用scaler，但用了也无害
scaler = GradScaler()  # 使用默认设置即可
```

**FP16**: 必须使用loss scaling
```python
# FP16必须用scaler
scaler = GradScaler(
    init_scale=2**16,  # 初始scale（默认65536）
    growth_factor=2.0,  # scale增长因子
    backoff_factor=0.5,  # scale减小因子
    growth_interval=2000,  # 多少步增长一次scale
)
```

### 5.4 调试建议

#### 检查数值溢出

```python
import torch

# 训练过程中检查
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

# 检查是否有NaN或Inf
if torch.isnan(loss) or torch.isinf(loss):
    print("警告：检测到NaN或Inf！")
    # 切换回FP32或调整loss scaling
```

#### 对比FP32和混合精度

```python
# 1. 训练FP32基线
model_fp32 = train_model(precision='fp32')
acc_fp32 = evaluate(model_fp32)

# 2. 训练混合精度
model_bf16 = train_model(precision='bf16')
acc_bf16 = evaluate(model_bf16)

# 3. 对比
print(f"FP32精度: {acc_fp32:.4f}")
print(f"BF16精度: {acc_bf16:.4f}")
print(f"精度差异: {abs(acc_fp32 - acc_bf16):.4f}")
```

---

## 6. 能耗研究专用配置

### 6.1 推荐的精度变异方案

对于能耗-性能权衡研究，建议测试以下配置：

```python
precision_configs = {
    'fp32': {
        'dtype': torch.float32,
        'use_amp': False,
        'description': '标准精度基线'
    },
    'tf32': {
        'dtype': torch.float32,
        'use_amp': False,
        'enable_tf32': True,
        'description': 'Ampere+ GPU自动加速'
    },
    'bf16': {
        'dtype': torch.bfloat16,
        'use_amp': True,
        'description': '推荐混合精度'
    },
    'fp16': {
        'dtype': torch.float16,
        'use_amp': True,
        'use_scaler': True,
        'description': '传统混合精度'
    },
}
```

### 6.2 实验执行脚本

```python
def train_with_precision(model, config_name, precision_config):
    """使用指定精度训练模型"""
    if config_name == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_amp = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        use_amp = precision_config.get('use_amp', False)

    dtype = precision_config['dtype']
    use_scaler = precision_config.get('use_scaler', False)
    scaler = GradScaler() if use_scaler else None

    # 训练循环
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()

            if use_amp:
                with autocast(dtype=dtype):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    return model

# 运行所有配置
results = {}
for config_name, config in precision_configs.items():
    print(f"训练 {config_name}...")
    model = MyModel()
    start_time = time.time()
    trained_model = train_with_precision(model, config_name, config)
    training_time = time.time() - start_time

    accuracy = evaluate(trained_model)
    results[config_name] = {
        'accuracy': accuracy,
        'training_time': training_time,
        'description': config['description']
    }

# 输出结果
for name, result in results.items():
    print(f"{name}: 精度={result['accuracy']:.4f}, "
          f"时间={result['training_time']:.2f}s")
```

### 6.3 预期结果（参考）

基于文献和经验，预期结果（相对FP32）：

| 配置 | 精度变化 | 时间变化 | 能耗变化 | 推荐用于 |
|------|---------|---------|---------|---------|
| FP32 | 基线 | 基线 | 基线 | 基线对照 |
| TF32 | -0.1% ~ -0.5% | -30% ~ -50% | -20% ~ -30% | Ampere+ GPU基线 |
| BF16 | -0.5% ~ -2% | -30% ~ -40% | -20% ~ -40% | 主要混合精度实验 |
| FP16 | -0.5% ~ -3% | -40% ~ -60% | -25% ~ -50% | Volta/Turing GPU |

---

## 7. 常见问题

### Q1: BF16和FP16哪个更好？

**A**: 取决于GPU架构：
- **Ampere+**: BF16更好（更稳定，不需要loss scaling）
- **Volta/Turing**: 只能用FP16（不支持BF16）

### Q2: 混合精度会损失多少精度？

**A**: 通常很小：
- **BF16**: 0.5-2%
- **FP16**: 0.5-3%
- **实践中**: 很多任务精度几乎无损失

### Q3: 是否需要调整学习率？

**A**: 通常不需要：
- **BF16**: 完全不需要调整
- **FP16**: 大多数情况不需要，个别情况可略微增大（1.2倍）

### Q4: 所有模型都适合混合精度吗？

**A**: 大多数适合，但注意：
- **不适合**: 对数值敏感的任务（如某些RL算法）
- **需谨慎**: GAN训练、某些RNN/LSTM
- **完全适合**: CNN、Transformer、大多数CV和NLP任务

### Q5: 如何判断混合精度训练是否成功？

**A**: 检查以下指标：
1. **Loss不应该发散**: 如果loss变为NaN或Inf，说明数值不稳定
2. **精度损失<3%**: 超过3%可能需要调整
3. **训练曲线正常**: 应该与FP32曲线形状相似

### Q6: Ampere GPU上TF32和BF16可以同时用吗？

**A**: 可以，但通常选择其中之一：
- **只用TF32**: 最简单，无需改代码，中等加速
- **只用BF16**: 更大加速，显存减半
- **同时用**: BF16自动使用TF32加速内部操作

---

## 8. 总结与推荐

### 8.1 快速决策树

```
是Ampere+架构（RTX 30/40系, A100）？
├─ 是 → 使用BF16混合精度 ⭐⭐⭐⭐⭐
│      （TF32自动启用，无需关心）
└─ 否 → 是Volta/Turing（V100, RTX 20系）？
       ├─ 是 → 使用FP16混合精度 ⭐⭐⭐⭐
       └─ 否 → 使用FP32 ⭐⭐⭐
              （Pascal及更老架构）
```

### 8.2 能耗研究推荐

对于能耗-性能权衡研究，推荐对比以下配置：

**必须测试**:
1. FP32（基线）
2. BF16或FP16（主要混合精度，取决于GPU）

**建议测试**:
3. TF32（如果是Ampere+）

**可选测试**:
4. INT8量化（如果关注推理）

### 8.3 最佳实践总结

1. **默认选择**: Ampere+用BF16，其他用FP32
2. **能耗优化**: 优先使用混合精度（BF16/FP16）
3. **稳定性优先**: 使用FP32或BF16
4. **显存受限**: 必须使用混合精度
5. **调试阶段**: 先用FP32，稳定后再切换

---

**文档版本**: 1.0
**更新日期**: 2025-11-04
**相关文档**: [hyperparameter_analysis.md](./hyperparameter_analysis.md)
