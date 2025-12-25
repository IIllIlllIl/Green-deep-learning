# 11个模型的深度学习架构详解

**版本**: v4.3.0
**最后更新**: 2025-11-18

本文档详细描述实验中使用的11个深度学习模型的网络架构。

---

## 📊 架构总览表

| # | 模型 | 架构类型 | 主要组件 | 层数 | 参数量 | 特点 |
|---|------|---------|---------|------|--------|------|
| 1 | MRT-OAST | Transformer | Multi-head Attention + MLP + ResNet | ~12层 | ~10M | 多表示融合 |
| 2 | bug-localization | DNN + IR | DNN (2隐藏层) | 3层 | ~1K | 混合方法 |
| 3 | resnet20 | ResNet | Residual Blocks | 20层 | 0.27M | 残差连接 |
| 4 | VulBERTa_mlp | Transformer + MLP | BERT + MLP | 12+2层 | 125M | 预训练+微调 |
| 5 | densenet121 | DenseNet | Dense Blocks | 121层 | 7.98M | 密集连接 |
| 6 | hrnet18 | HRNet | Parallel Multi-scale | 18层 | 21.3M | 高分辨率保持 |
| 7 | pcb | CNN + Partition | ResNet-50 + 6部分 | 50+6层 | 25.6M | 部分特征提取 |
| 8 | mnist | CNN | Conv + Pooling | 3层 | ~44K | 经典CNN |
| 9 | mnist_rnn | RNN | LSTM | 2层 | ~100K | 序列处理 |
| 10 | mnist_ff | Forward-Forward | FF Layers | 4层 | ~50K | 新型学习算法 |
| 11 | siamese | Siamese CNN | 孪生网络 | 3×2层 | ~88K | 对比学习 |

---

## 🔍 详细架构分析

### 1. MRT-OAST (Multi-Representation Transformer)

**任务**: 代码克隆检测

**架构**: Transformer-based Multi-Representation Learning

```
输入层:
├─ Code Tokens (代码词元)
├─ AST (抽象语法树)
└─ OAST (优化的抽象语法树)
         ↓
    Embedding Layer (词嵌入层)
    - Dimension: 512
         ↓
    Transformer Encoder
    ├─ Multi-head Self-Attention (8 heads)
    ├─ Feed-Forward Network
    ├─ Layer Normalization
    └─ Residual Connection
    × 12 layers
         ↓
    Representation Fusion (表示融合)
    - Token representation
    - AST representation
    - OAST representation
         ↓
    Similarity Network (相似度网络)
    - MLP: 512 → 256 → 128 → 1
         ↓
    Output: Similarity Score (0-1)
```

**关键特点**:
- 🔥 多表示学习（Token + AST + OAST）
- 🔥 Transformer架构（自注意力机制）
- 🔥 端到端训练

**参数量**: ~10M

---

### 2. bug-localization (DNN + rVSM)

**任务**: 软件缺陷定位

**架构**: 简单的多层感知机 (MLP)

```
输入层 (5个特征):
├─ rVSM similarity (文本相似度)
├─ Collaborative filter (协同过滤)
├─ Classname similarity (类名相似度)
├─ Bug recency (bug新近度)
└─ Bug frequency (bug频率)
         ↓
    Hidden Layer 1
    - Neurons: 10
    - Activation: ReLU
    - Dropout: 0.2
         ↓
    Hidden Layer 2
    - Neurons: 10
    - Activation: ReLU
    - Dropout: 0.2
         ↓
    Output Layer
    - Neurons: 1
    - Activation: Sigmoid
         ↓
    Output: Relevance Score (0-1)
```

**关键特点**:
- 🔥 简单但有效的MLP
- 🔥 结合传统IR特征（rVSM）和DNN
- 🔥 10折交叉验证

**参数量**: ~1,000 (非常小)

**网络结构**:
```
Input(5) → Dense(10, relu) → Dense(10, relu) → Dense(1, sigmoid)
```

---

### 3. resnet20 (ResNet-20 for CIFAR-10)

**任务**: 图像分类（CIFAR-10）

**架构**: 残差网络 (Residual Network)

```
Input: 32×32×3 (CIFAR-10图像)
         ↓
    Conv1: 3×3, 16 filters
         ↓
    Residual Block Stack 1 (16 filters)
    ├─ Residual Block × 3
    │  ├─ Conv: 3×3, 16
    │  ├─ BatchNorm + ReLU
    │  ├─ Conv: 3×3, 16
    │  ├─ BatchNorm
    │  └─ Shortcut (identity)
         ↓
    Residual Block Stack 2 (32 filters, stride=2)
    ├─ Residual Block × 3
         ↓
    Residual Block Stack 3 (64 filters, stride=2)
    ├─ Residual Block × 3
         ↓
    Global Average Pooling
         ↓
    Fully Connected: 10 classes
         ↓
    Softmax
```

**关键特点**:
- 🔥 残差连接（Skip Connection）解决梯度消失
- 🔥 BatchNorm加速训练
- 🔥 严格按照原论文实现

**参数量**: 0.27M

**Residual Block结构**:
```
x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
↓_________________________________↑
        (shortcut/identity)
```

---

### 4. VulBERTa_mlp (代码漏洞检测)

**任务**: 源代码漏洞检测

**架构**: BERT + MLP Classifier

```
Input: C/C++ Source Code
         ↓
    Custom Tokenizer (代码专用)
    - Vocabulary: 50,000
         ↓
    RoBERTa Base Encoder
    ├─ Embedding Layer: 768
    ├─ Transformer Encoder × 12
    │  ├─ Multi-head Attention (12 heads)
    │  ├─ Feed-Forward (3072)
    │  └─ Layer Norm + Residual
         ↓
    [CLS] Token Representation (768-dim)
         ↓
    MLP Classifier
    ├─ Dense: 768 → 256
    ├─ ReLU + Dropout(0.1)
    ├─ Dense: 256 → 128
    ├─ ReLU + Dropout(0.1)
    └─ Dense: 128 → 2
         ↓
    Softmax
         ↓
    Output: [Vulnerable, Non-Vulnerable]
```

**关键特点**:
- 🔥 基于RoBERTa预训练模型
- 🔥 自定义代码tokenizer
- 🔥 在Devign数据集上fine-tune

**参数量**: ~125M (BERT: 123M + MLP: 2M)

**预训练**: DrapGH数据集（开源C/C++代码）

---

### 5. densenet121 (DenseNet-121)

**任务**: 行人重识别

**架构**: 密集连接网络 (Densely Connected Network)

```
Input: 256×128×3 (行人图像)
         ↓
    Conv1: 7×7, 64, stride=2
    MaxPool: 3×3, stride=2
         ↓
    Dense Block 1 (6 layers)
    ├─ [BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)] × 6
    ├─ Growth rate: 32
    └─ Dense connections (每层连接前面所有层)
         ↓
    Transition Layer 1
    ├─ BN-Conv(1×1)-AvgPool(2×2)
    └─ Compression: 0.5
         ↓
    Dense Block 2 (12 layers)
         ↓
    Transition Layer 2
         ↓
    Dense Block 3 (24 layers)
         ↓
    Transition Layer 3
         ↓
    Dense Block 4 (16 layers)
         ↓
    Global Average Pooling
         ↓
    Fully Connected: 751 IDs (Market-1501)
         ↓
    Output: Person ID
```

**关键特点**:
- 🔥 Dense Connection（每层与前面所有层连接）
- 🔥 特征重用，参数效率高
- 🔥 缓解梯度消失

**参数量**: 7.98M

**Dense Block核心**:
```
x₀ → H₁ → x₁ ─┐
x₀ ─┬────────→ Concat → H₂ → x₂ ─┐
    └─────────────────────────→ Concat → H₃ → x₃
```

---

### 6. hrnet18 (High-Resolution Net)

**任务**: 行人重识别

**架构**: 高分辨率网络 (并行多尺度)

```
Input: 256×128×3
         ↓
    Stem: 2× Conv(3×3, 64)
         ↓
    Stage 1: Single Resolution (1/4)
    └─ Bottleneck × 4
         ↓
    Stage 2: Parallel Branches (1/4 + 1/8)
    ├─ High-Res Branch (1/4)
    └─ Low-Res Branch (1/8)
    └─ Multi-scale Fusion
         ↓
    Stage 3: Parallel Branches (1/4 + 1/8 + 1/16)
    ├─ High-Res Branch (1/4)
    ├─ Med-Res Branch (1/8)
    └─ Low-Res Branch (1/16)
    └─ Multi-scale Fusion
         ↓
    Stage 4: Parallel Branches (1/4 + 1/8 + 1/16 + 1/32)
    └─ Multi-scale Fusion
         ↓
    Global Average Pooling
         ↓
    FC: 751 classes
```

**关键特点**:
- 🔥 始终保持高分辨率表示
- 🔥 并行多尺度分支
- 🔥 跨分支信息融合

**参数量**: 21.3M

**Multi-scale Fusion**:
```
High-Res ──┬─→ High-Res Output
           │
Med-Res ───┼─→ Med-Res Output
           │
Low-Res ───┴─→ Low-Res Output
(通过上采样/下采样互相交换信息)
```

---

### 7. pcb (Part-based Convolutional Baseline)

**任务**: 行人重识别

**架构**: ResNet-50 + 部分分割

```
Input: 384×128×3
         ↓
    ResNet-50 Backbone (去掉最后的stride)
    ├─ Conv1: 7×7, 64
    ├─ MaxPool: 3×3
    ├─ Layer1 (Bottleneck × 3)
    ├─ Layer2 (Bottleneck × 4)
    ├─ Layer3 (Bottleneck × 6)
    └─ Layer4 (Bottleneck × 3)
         ↓
    Feature Map: 24×8×2048
         ↓
    Uniform Partition (均匀分割)
    ├─ Part 1: [0:4, :, :]   (头部)
    ├─ Part 2: [4:8, :, :]   (上躯干)
    ├─ Part 3: [8:12, :, :]  (腰部)
    ├─ Part 4: [12:16, :, :] (大腿)
    ├─ Part 5: [16:20, :, :] (小腿)
    └─ Part 6: [20:24, :, :] (脚部)
         ↓
    Part-level Feature Extraction (每部分独立)
    ├─ Global Average Pooling (每个part)
    ├─ FC: 2048 → 256 (每个part)
    └─ L2 Normalization
         ↓
    Concatenation: 256×6 = 1536-dim
         ↓
    Classification: 751 classes
```

**关键特点**:
- 🔥 部分分割（6个水平条带）
- 🔥 细粒度特征提取
- 🔥 对遮挡鲁棒

**参数量**: 25.6M (ResNet-50: 23.5M + 6×FC: 2.1M)

---

### 8. mnist (经典CNN)

**任务**: 手写数字识别

**架构**: 简单3层CNN

```
Input: 28×28×1 (灰度图像)
         ↓
    Conv1: 3×3, 32 filters
    ReLU
    MaxPool: 2×2
         ↓
    Conv2: 3×3, 64 filters
    ReLU
    MaxPool: 2×2
         ↓
    Flatten: 7×7×64 = 3136
         ↓
    FC1: 3136 → 128
    ReLU
    Dropout: 0.5
         ↓
    FC2: 128 → 10
         ↓
    Softmax
         ↓
    Output: 10 classes (0-9)
```

**关键特点**:
- 🔥 经典CNN结构
- 🔥 简单但有效
- 🔥 教学示例

**参数量**: ~44K

---

### 9. mnist_rnn (LSTM for MNIST)

**任务**: 手写数字识别（序列方式）

**架构**: LSTM循环神经网络

```
Input: 28×28 → 重塑为序列 28 steps × 28 features
         ↓
    LSTM Layer 1
    - Hidden units: 128
    - Return sequences: True
    - Dropout: 0.2
         ↓
    LSTM Layer 2
    - Hidden units: 128
    - Return sequences: False
    - Dropout: 0.2
         ↓
    Last Hidden State: 128-dim
         ↓
    Fully Connected: 128 → 10
         ↓
    Softmax
         ↓
    Output: 10 classes
```

**关键特点**:
- 🔥 将图像视为序列（28行，每行28像素）
- 🔥 LSTM处理时序依赖
- 🔥 演示RNN在CV中的应用

**参数量**: ~100K

**LSTM Cell结构**:
```
Input Gate:  i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
Forget Gate: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
Cell State:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_c·[h_{t-1}, x_t] + b_c)
Output Gate: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
Hidden:      h_t = o_t ⊙ tanh(C_t)
```

---

### 10. mnist_ff (Forward-Forward Network)

**任务**: 手写数字识别

**架构**: Forward-Forward算法（Hinton 2022）

```
Input: 28×28×1 → Flatten: 784
         ↓
    FF Layer 1
    - Input: 784
    - Output: 500
    - Positive pass: Real data
    - Negative pass: Negative data
    - Local loss: Goodness function
         ↓
    FF Layer 2
    - Input: 500
    - Output: 500
    - Local loss: Goodness function
         ↓
    FF Layer 3
    - Input: 500
    - Output: 500
    - Local loss: Goodness function
         ↓
    FF Layer 4
    - Input: 500
    - Output: 10
    - Local loss: Goodness function
         ↓
    Output: 10 classes
```

**关键特点**:
- 🔥 无反向传播（不需要BP）
- 🔥 局部学习规则
- 🔥 每层独立训练

**参数量**: ~50K

**Goodness Function (核心)**:
```python
# Positive data: 最大化 goodness
goodness_pos = Σ(activation²)

# Negative data: 最小化 goodness
goodness_neg = Σ(activation²)

# Local loss
loss = -log(σ(goodness_pos - threshold))
       -log(σ(threshold - goodness_neg))
```

**训练方式**:
- ✅ 每层局部训练（不需要全局梯度）
- ✅ Positive samples: 真实数据
- ✅ Negative samples: 标签错误的数据

---

### 11. siamese (孪生网络)

**任务**: 相似度学习

**架构**: Siamese CNN

```
Image 1: 28×28×1 ──┐
                    ├─→ Shared CNN ──┐
Image 2: 28×28×1 ──┘                 │
                                     │
Shared CNN:                          │
├─ Conv1: 3×3, 64                   │
├─ ReLU + MaxPool                   │
├─ Conv2: 3×3, 128                  │
├─ ReLU + MaxPool                   │
├─ Flatten                          │
└─ FC: 128                          │
                                     ↓
                          [Feature 1, Feature 2]
                                     ↓
                          Distance Calculation
                          (Euclidean / Cosine)
                                     ↓
                          Contrastive Loss
                          - Similar pairs: minimize distance
                          - Dissimilar pairs: maximize distance
                                     ↓
                          Output: Similarity Score
```

**关键特点**:
- 🔥 权重共享（两个分支使用相同的CNN）
- 🔥 对比学习（Contrastive Learning）
- 🔥 学习距离度量

**参数量**: ~88K (单个分支44K × 权重共享)

**Contrastive Loss**:
```python
loss = (1 - Y) * 0.5 * D² + Y * 0.5 * max(margin - D, 0)²

其中:
- Y: 标签 (0=相似, 1=不相似)
- D: 欧氏距离
- margin: 间隔（通常为1.0）
```

---

## 📊 架构分类统计

### 按架构类型分类

| 架构类型 | 模型数量 | 模型列表 |
|---------|---------|---------|
| **CNN** | 5 | resnet20, densenet121, hrnet18, pcb, mnist |
| **RNN/LSTM** | 1 | mnist_rnn |
| **Transformer** | 2 | MRT-OAST, VulBERTa_mlp |
| **MLP** | 1 | bug-localization |
| **Siamese** | 1 | siamese |
| **Forward-Forward** | 1 | mnist_ff |

### 按参数规模分类

| 规模 | 范围 | 模型数量 | 模型列表 |
|------|------|---------|---------|
| **极小** | < 100K | 4 | bug-localization, mnist, mnist_rnn, mnist_ff, siamese |
| **小** | 100K-1M | 1 | resnet20 |
| **中** | 1M-10M | 2 | densenet121, MRT-OAST |
| **大** | 10M-30M | 2 | hrnet18, pcb |
| **超大** | > 100M | 1 | VulBERTa_mlp |

### 按创新性分类

| 类别 | 模型 | 创新点 |
|------|------|--------|
| **经典架构** | mnist, mnist_rnn | 教学示例 |
| **残差学习** | resnet20 | Skip Connection |
| **密集连接** | densenet121 | Dense Connection |
| **多尺度** | hrnet18 | Parallel Multi-scale |
| **部分分割** | pcb | Part-based Features |
| **预训练** | VulBERTa_mlp | BERT for Code |
| **多表示** | MRT-OAST | Token + AST + OAST |
| **对比学习** | siamese | Contrastive Learning |
| **新型学习** | mnist_ff | No Backpropagation |

---

## 🎯 架构选择建议

### 图像分类任务

- **小数据集**: resnet20 (轻量级)
- **中等数据集**: densenet121 (特征重用)
- **需要多尺度**: hrnet18 (保持高分辨率)

### 行人重识别任务

- **标准方法**: densenet121
- **细粒度**: pcb (部分特征)
- **高精度**: hrnet18 (多尺度)

### 代码分析任务

- **克隆检测**: MRT-OAST (多表示)
- **漏洞检测**: VulBERTa (预训练)
- **缺陷定位**: bug-localization (轻量级)

### 教学示例

- **CNN入门**: mnist
- **RNN入门**: mnist_rnn
- **度量学习**: siamese
- **新算法**: mnist_ff

---

## 📚 参考资料

### 原始论文

1. **ResNet**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
2. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
3. **HRNet**: Wang et al., "Deep High-Resolution Representation Learning", CVPR 2019
4. **PCB**: Sun et al., "Beyond Part Models: Person Retrieval with Refined Part Pooling", ECCV 2018
5. **VulBERTa**: Hanif & Maffeis, "VulBERTa: Simplified Source Code Pre-Training", IJCNN 2022
6. **Forward-Forward**: Hinton, "The Forward-Forward Algorithm", 2022
7. **Siamese**: Bromley et al., "Signature Verification using a Siamese Time Delay Neural Network", 1993

### 实现参考

- PyTorch官方: https://pytorch.org/docs/
- timm库: https://github.com/rwightman/pytorch-image-models
- HuggingFace Transformers: https://huggingface.co/docs/transformers

---

**文档版本**: v1.0
**创建日期**: 2025-11-18
**作者**: Green
