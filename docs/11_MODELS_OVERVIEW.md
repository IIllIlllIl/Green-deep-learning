# 实验模型概览 - 11个选定模型

**版本**: v4.3.0
**最后更新**: 2025-11-18

本文档详细描述实验中使用的11个深度学习模型，涵盖6个不同的应用领域。

---

## 📊 模型总览表

| # | 仓库 | 模型 | 任务类型 | 数据集 | 模型架构 | 显存占用 | 1 Epoch时间 | 应用领域 |
|---|------|------|---------|--------|---------|----------|-----------|---------|
| 1 | MRT-OAST | default | 时空预测 | AQI36 | MLP + ResNet | 2700 MB | 5-8 min | 空气质量预测 |
| 2 | bug-localization | default | 缺陷定位 | AspectJ等5个项目 | DNN + RVSM | 4000 MB | 20-25 min* | 软件工程 |
| 3 | pytorch_resnet_cifar10 | resnet20 | 图像分类 | CIFAR-10 | ResNet-20 | 1300 MB | 3-4 min | 计算机视觉 |
| 4 | VulBERTa | mlp | 漏洞检测 | Devign | BERT + MLP | 2000 MB | 15-20 min | 代码安全 |
| 5 | Person_reID_baseline_pytorch | densenet121 | 行人重识别 | Market-1501 | DenseNet-121 | 4000 MB | 6-8 min | 计算机视觉 |
| 6 | Person_reID_baseline_pytorch | hrnet18 | 行人重识别 | Market-1501 | HRNet-W18 | 3000 MB | 6-8 min | 计算机视觉 |
| 7 | Person_reID_baseline_pytorch | pcb | 行人重识别 | Market-1501 | ResNet-50 + PCB | 3000 MB | 5-7 min | 计算机视觉 |
| 8 | examples | mnist | 图像分类 | MNIST | CNN (3层) | 800 MB | 0.3-0.5 min | 基础CV |
| 9 | examples | mnist_rnn | 图像分类 | MNIST | LSTM | 1200 MB | 0.5-1 min | 序列学习 |
| 10 | examples | mnist_ff | 图像分类 | MNIST | Forward-Forward | 900 MB | 0.3-0.5 min | 新型学习算法 |
| 11 | examples | siamese | 相似度学习 | MNIST | Siamese CNN | 1500 MB | 1-2 min | 度量学习 |

\* bug-localization的时间指max_iter=500时（快速验证版本）

---

## 🔍 详细模型信息

### 1. MRT-OAST (空气质量预测)

**完整名称**: Multi-Receptive-field Temporal-series Ordinary-differential-equation Attention Spatial-Transcoder

**任务**: 时空序列预测
- 预测未来1-3小时的空气质量指数（AQI）
- 同时考虑时间和空间维度

**架构特点**:
- 多感受野时序建模
- 常微分方程（ODE）注意力机制
- 空间转换器网络

**数据集**: AQI36
- 36个监测站点
- 6种空气污染物（PM2.5, PM10, SO2, NO2, CO, O3）
- 时序数据

**默认超参数**:
```python
{
  "epochs": 1,
  "learning_rate": 0.0001,
  "dropout": 0.2,
  "weight_decay": 0.0,
  "seed": 1334
}
```

**性能指标**: MAE, RMSE, R²

**应用场景**: 城市空气质量监测与预报

---

### 2. bug-localization (软件缺陷定位)

**完整名称**: Bug Localization by Deep Neural Network and RVSM

**任务**: 软件缺陷定位
- 根据bug报告定位源代码中的缺陷位置
- 结合深度学习和信息检索技术

**架构特点**:
- DNN用于学习bug报告和代码的语义表示
- RVSM (Revised Vector Space Model) 用于相似度计算
- 混合方法提升定位精度

**数据集**: 5个Java开源项目
- AspectJ
- Eclipse
- SWT
- Tomcat
- JDT

**默认超参数**:
```python
{
  "max_iter": 500,    # 训练迭代次数（快速验证版本）
  "alpha": 1e-05,     # RVSM权重
  "kfold": 10,        # 10折交叉验证
  "seed": 42
}
```

**性能指标**: Top-1, Top-5, Top-10准确率, MRR (Mean Reciprocal Rank)

**应用场景**: 软件维护、自动化调试

**注意事项**:
- 10折交叉验证导致训练时间较长
- 完整版max_iter=10000需要约8小时

---

### 3. resnet20 (图像分类 - CIFAR-10)

**完整名称**: Residual Network with 20 layers

**任务**: 小尺寸图像分类
- CIFAR-10: 10类32×32彩色图像
- 飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车

**架构特点**:
- 20层深度残差网络
- 残差连接解决梯度消失问题
- 适合小图像的轻量级设计

**数据集**: CIFAR-10
- 训练集: 50,000张
- 测试集: 10,000张
- 图像尺寸: 32×32×3

**默认超参数**:
```python
{
  "epochs": 1,
  "learning_rate": 0.1,
  "weight_decay": 0.0001,
  "seed": 1334
}
```

**性能指标**: Top-1准确率 (目标: >91%)

**应用场景**: 图像分类基准测试、算法验证

---

### 4. VulBERTa_mlp (代码漏洞检测)

**完整名称**: VulBERTa with MLP Classifier

**任务**: 源代码漏洞检测
- 二分类：漏洞代码 vs 正常代码
- C/C++源代码分析

**架构特点**:
- BERT预训练模型（CodeBERT）用于代码表示学习
- MLP分类器
- 针对代码的特殊tokenization

**数据集**: Devign
- 27,318个C语言函数
- 标注漏洞与非漏洞代码

**默认超参数**:
```python
{
  "epochs": 1,
  "learning_rate": 3e-05,
  "weight_decay": 0.0,
  "seed": 42
}
```

**性能指标**: Accuracy, Precision, Recall, F1-score

**应用场景**: 代码安全审计、自动化漏洞扫描

**注意事项**:
- BERT模型较大，训练时间较长
- 需要预下载预训练权重用于离线训练

---

### 5-7. Person_reID (行人重识别)

**完整名称**: Person Re-Identification Baseline

**任务**: 行人重识别
- 跨摄像头匹配同一行人
- 检索特定行人的所有出现

**数据集**: Market-1501
- 1,501个行人身份
- 32,668张图像
- 6个摄像头视角

**共同默认超参数**:
```python
{
  "epochs": 1,
  "learning_rate": 0.05,
  "dropout": 0.5,
  "seed": 1334
}
```

**性能指标**: mAP, Rank-1, Rank-5, Rank-10

**应用场景**: 视频监控、安防系统、智慧城市

#### 5. densenet121
- **架构**: DenseNet-121 (密集连接网络)
- **特点**: 每层与前面所有层连接，特征重用
- **显存**: 4000 MB (最高)
- **时间**: 6-8 min

#### 6. hrnet18
- **架构**: HRNet-W18 (高分辨率网络)
- **特点**: 保持高分辨率表示，适合密集预测
- **显存**: 3000 MB
- **时间**: 6-8 min
- **注意**: 需要预下载timm模型权重

#### 7. pcb
- **架构**: ResNet-50 + Part-based Convolutional Baseline
- **特点**: 将行人图像分成6个部分分别提取特征
- **显存**: 3000 MB
- **时间**: 5-7 min

---

### 8-11. MNIST示例模型

**数据集**: MNIST
- 手写数字识别 (0-9)
- 训练集: 60,000张
- 测试集: 10,000张
- 图像尺寸: 28×28 (灰度)

**共同默认超参数**:
```python
{
  "epochs": 1,
  "learning_rate": 0.01,
  "batch_size": 32,
  "seed": 1
}
```

**性能指标**: Test Accuracy, Test Loss

#### 8. mnist (标准CNN)
- **架构**: 3层卷积神经网络
- **特点**: 简单基础的CNN结构
- **显存**: 800 MB (最低)
- **时间**: 0.3-0.5 min (最快)
- **应用**: 基础CV教学、快速验证

#### 9. mnist_rnn (循环神经网络)
- **架构**: LSTM
- **特点**: 将图像每行作为序列输入
- **显存**: 1200 MB
- **时间**: 0.5-1 min
- **应用**: 序列学习演示、RNN基础

#### 10. mnist_ff (Forward-Forward算法)
- **架构**: Forward-Forward Network
- **特点**: 新型学习算法，无反向传播
- **显存**: 900 MB
- **时间**: 0.3-0.5 min
- **应用**: 新型学习算法研究

#### 11. siamese (孪生网络)
- **架构**: Siamese CNN
- **特点**: 学习相似度度量，对比学习
- **显存**: 1500 MB
- **时间**: 1-2 min
- **应用**: 度量学习、相似度判断

---

## 📊 模型分组统计

### 按应用领域分组

| 领域 | 模型数量 | 模型列表 |
|------|---------|---------|
| **计算机视觉** | 4 | resnet20, densenet121, hrnet18, pcb |
| **基础CV示例** | 4 | mnist, mnist_rnn, mnist_ff, siamese |
| **代码分析** | 2 | VulBERTa_mlp, bug-localization |
| **时空预测** | 1 | MRT-OAST |

### 按显存占用分组

| 显存范围 | 模型数量 | 模型列表 |
|---------|---------|---------|
| **超低 (< 1000 MB)** | 2 | mnist (800), mnist_ff (900) |
| **低 (1000-2000 MB)** | 3 | mnist_rnn (1200), resnet20 (1300), siamese (1500) |
| **中低 (2000-3000 MB)** | 3 | VulBERTa_mlp (2000), MRT-OAST (2700), hrnet18 (3000), pcb (3000) |
| **中高 (3000-4000 MB)** | 2 | bug-localization (4000), densenet121 (4000) |

### 按训练时间分组 (1 epoch)

| 时间范围 | 模型数量 | 模型列表 |
|---------|---------|---------|
| **极快 (< 1 min)** | 3 | mnist, mnist_ff, mnist_rnn |
| **快 (1-5 min)** | 2 | siamese, resnet20 |
| **中速 (5-10 min)** | 5 | MRT-OAST, pcb, hrnet18, densenet121 |
| **慢 (15-25 min)** | 2 | VulBERTa_mlp, bug-localization |

---

## 🎯 模型选择策略

### 多样性考虑

选择这11个模型的原因：

1. **任务多样性** ✅
   - 图像分类 (5个)
   - 行人重识别 (3个)
   - 代码分析 (2个)
   - 时空预测 (1个)

2. **架构多样性** ✅
   - CNN: mnist, resnet20, siamese
   - ResNet系列: resnet20, pcb (ResNet-50)
   - Dense连接: densenet121
   - 高分辨率网络: hrnet18
   - RNN: mnist_rnn
   - Transformer: VulBERTa_mlp (BERT)
   - 混合架构: MRT-OAST, bug-localization

3. **规模多样性** ✅
   - 轻量级: mnist系列 (< 1 MB)
   - 中等规模: resnet20, pcb (10-100 MB)
   - 大规模: VulBERTa_mlp, densenet121 (> 100 MB)

4. **训练时间梯度** ✅
   - 从0.3分钟到25分钟
   - 便于不同时间预算的实验

5. **显存占用梯度** ✅
   - 从800 MB到4000 MB
   - 测试不同GPU负载场景

6. **实际应用价值** ✅
   - 涵盖工业界常见任务
   - 代表性强

---

## 🔄 并行组合策略

### 显存互补原则

在并行实验中，前景和背景模型的显存总和控制在5-6 GB以内：

| 组合 | 前景显存 | 背景显存 | 总显存 | 策略 |
|------|---------|---------|--------|------|
| resnet20 + mnist_ff | 1300 | 900 | 2200 | 超低显存组合 |
| VulBERTa_mlp + mnist | 2000 | 800 | 2800 | 低显存组合 |
| MRT-OAST + mnist_rnn | 2700 | 1200 | 3900 | 中显存组合 |
| pcb + mnist_rnn | 3000 | 1200 | 4200 | 中显存组合 |
| hrnet18 + mnist_rnn | 3000 | 1200 | 4200 | 中显存组合 |
| densenet121 + VulBERTa_mlp | 4000 | 2000 | 6000 | 高显存组合（接近上限）|

### 时间互补原则

尽量让快速模型作为背景，避免总训练时间过长。

---

## 📖 相关文档

- [超参数变异范围](MUTATION_RANGES_QUICK_REFERENCE.md) - 各模型的超参数配置
- [实验配置指南](SETTINGS_CONFIGURATION_GUIDE.md) - 如何配置这些模型
- [并行训练使用](PARALLEL_TRAINING_USAGE.md) - 并行组合详解
- [快速参考](QUICK_REFERENCE.md) - 命令速查

---

## 🔗 数据集和模型链接

### 数据集

- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html
- **MNIST**: http://yann.lecun.com/exdb/mnist/
- **Market-1501**: https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
- **Devign**: https://github.com/microsoft/CodeXGLUE
- **Bug-localization datasets**: 各项目官方仓库

### 预训练权重

- **HRNet-W18**: HuggingFace timm库 (`timm/hrnet_w18`)
- **ResNet-50**: torchvision预训练权重
- **DenseNet-121**: torchvision预训练权重
- **CodeBERT**: HuggingFace transformers库

**离线下载脚本**: `scripts/download_pretrained_models.py`

---

**文档版本**: v4.3.0
**维护者**: Green
**最后更新**: 2025-11-18
