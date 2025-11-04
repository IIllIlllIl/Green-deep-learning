# VulBERTa 训练时间估算

生成时间：2025-11-03
硬件环境：NVIDIA GeForce RTX 3080 (10GB VRAM)

## 概述

本文档提供 VulBERTa-MLP 和 VulBERTa-CNN 模型在不同数据集上的训练时间估算。估算基于：
- 实际硬件配置（RTX 3080 10GB）
- 数据集样本数量
- 训练超参数配置
- 模型架构和参数量

## 硬件环境

- **GPU**: NVIDIA GeForce RTX 3080
- **显存**: 10GB
- **CUDA**: sm_86 架构
- **注意**: 当前PyTorch版本与RTX 3080不完全兼容（需要升级）

## 数据集统计

| 数据集 | 训练集 | 验证集 | 总计 | 相对大小 |
|--------|--------|--------|------|----------|
| d2a | 4,643 | 596 | 5,239 | 最小 |
| reveal | 18,187 | 2,273 | 20,460 | 小 |
| devign | 21,854 | 2,732 | 24,586 | 小 |
| vuldeepecker | 128,118 | 16,015 | 144,133 | 大 |
| mvd | 123,515 | 21,797 | 145,312 | 大 |
| draper | 1,019,471 | 127,476 | 1,146,947 | 超大 |

## 模型配置

### VulBERTa-MLP

**架构**: RoBERTa-base + 分类头
**参数量**: 124,836,866 (~125M)
**基础模型**: ./models/VulBERTa/ (预训练权重)

**默认超参数**:
```bash
batch_size: 4 (推荐使用 2 避免OOM)
epochs: 10
learning_rate: 3e-05
seed: 42
fp16: True
max_length: 1024 tokens
```

**显存使用**:
- 模型权重: ~500MB
- 优化器状态: ~1GB
- 激活值 (batch_size=2): ~6-8GB
- **总计**: ~8-9GB (接近10GB上限)

### VulBERTa-CNN

**架构**: RoBERTa-base (frozen) + CNN层
**状态**: 尚未实现在train_vulberta.py中

**默认超参数**:
```bash
batch_size: 128
epochs: 20
learning_rate: 0.0005
seed: 1234
fp16: False
```

## 训练时间估算

### 估算方法

训练时间主要取决于：
1. **每步时间**: 前向传播 + 反向传播 + 优化器更新
2. **总步数**: (训练样本数 / batch_size) × epochs
3. **评估时间**: 每个epoch结束后的验证

**每步时间估算** (基于RTX 3080):
- MLP (batch_size=2): ~0.6-0.8秒/步
- MLP (batch_size=4): ~1.0-1.2秒/步 (可能OOM)
- CNN (batch_size=128): ~0.3-0.5秒/步

### VulBERTa-MLP 训练时间估算

使用 `batch_size=2` (推荐值，避免显存溢出):

| 数据集 | 训练样本 | Steps/Epoch | 总Steps (10 epochs) | 预计时间 (保守估算) | 预计时间 (乐观估算) |
|--------|----------|-------------|---------------------|---------------------|---------------------|
| **d2a** | 4,643 | 2,322 | 23,220 | **4.6小时** | **3.9小时** |
| **reveal** | 18,187 | 9,094 | 90,940 | **18.2小时** | **15.2小时** |
| **devign** | 21,854 | 10,927 | 109,270 | **21.9小时** | **18.2小时** |
| **vuldeepecker** | 128,118 | 64,059 | 640,590 | **128小时 (5.3天)** | **107小时 (4.5天)** |
| **mvd** | 123,515 | 61,758 | 617,580 | **124小时 (5.1天)** | **103小时 (4.3天)** |
| **draper** | 1,019,471 | 509,736 | 5,097,360 | **1,020小时 (42天)** | **849小时 (35天)** |

**计算公式**:
- 保守估算: Steps × 0.8秒
- 乐观估算: Steps × 0.6秒
- 包含约5%的评估和保存开销

**实际时间可能受影响因素**:
- ✓ 数据加载速度
- ✓ GPU利用率
- ✓ 系统负载
- ✓ CUDA版本兼容性问题（当前存在警告）

### VulBERTa-CNN 训练时间估算

使用 `batch_size=128`, `epochs=20`:

**注意**: CNN模型训练功能尚未在train_vulberta.py中实现。以下估算基于理论分析。

CNN模型特点：
- RoBERTa编码器通常被冻结 (frozen)
- 只训练CNN分类层
- 计算量相对较小
- 可以使用更大的batch_size

| 数据集 | 训练样本 | Steps/Epoch | 总Steps (20 epochs) | 预计时间 (保守估算) | 预计时间 (乐观估算) |
|--------|----------|-------------|---------------------|---------------------|---------------------|
| **d2a** | 4,643 | 37 | 740 | **0.2小时** | **0.1小时** |
| **reveal** | 18,187 | 143 | 2,860 | **0.7小时** | **0.5小时** |
| **devign** | 21,854 | 171 | 3,420 | **0.9小时** | **0.6小时** |
| **vuldeepecker** | 128,118 | 1,001 | 20,020 | **5.0小时** | **3.3小时** |
| **mvd** | 123,515 | 965 | 19,300 | **4.8小时** | **3.2小时** |
| **draper** | 1,019,471 | 7,965 | 159,300 | **39.8小时 (1.7天)** | **26.5小时 (1.1天)** |

**计算公式**:
- 保守估算: Steps × 0.5秒
- 乐观估算: Steps × 0.3秒

### 推荐训练顺序

基于时间成本和数据集规模，推荐以下训练顺序：

#### 快速验证 (1-2天)
1. **d2a** (MLP: ~4小时, CNN: ~0.2小时)
2. **reveal** (MLP: ~18小时, CNN: ~0.7小时)
3. **devign** (MLP: ~22小时, CNN: ~0.9小时)

#### 中等规模 (4-6天)
4. **vuldeepecker** (MLP: ~5天, CNN: ~5小时)
5. **mvd** (MLP: ~5天, CNN: ~5小时)

#### 大规模数据集 (35-42天)
6. **draper** (MLP: ~42天, CNN: ~2天)

## 优化建议

### 减少训练时间

1. **减少epochs数量**
   ```bash
   ./train.sh -n mlp -d devign --epochs 5  # 时间减半
   ```

2. **使用梯度累积** (如果实现)
   - 可以模拟更大的batch_size
   - 不增加显存使用

3. **Early Stopping**
   - 配置在验证集上无提升时提前停止
   - 可能节省30-50%时间

4. **数据采样** (仅用于快速实验)
   ```bash
   # 使用10%数据快速测试
   # 需要修改代码实现
   ```

### 避免显存溢出 (OOM)

当前配置下，MLP模型容易发生OOM错误：

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate 192.00 MiB
(GPU 0; 9.77 GiB total capacity; 8.10 GiB already allocated)
```

**解决方案**:

1. **减小batch_size** (推荐)
   ```bash
   ./train.sh -n mlp -d devign --batch_size 2  # 默认已是2
   ./train.sh -n mlp -d devign --batch_size 1  # 极端情况
   ```

2. **禁用FP16** (如果问题持续)
   ```bash
   # 修改train.sh或train_vulberta.py，设置fp16=False
   ```

3. **使用CPU训练** (最后手段，会非常慢)
   ```bash
   ./train.sh -n mlp -d devign --cpu
   ```

4. **使用梯度检查点** (需要代码修改)
   - 可以减少50%显存使用
   - 会增加20-30%训练时间

### CUDA兼容性问题

当前环境存在CUDA兼容性警告：
```
NVIDIA GeForce RTX 3080 with CUDA capability sm_86 is not compatible
with the current PyTorch installation.
```

**影响**: 可能无法使用GPU，或性能下降

**解决方案**: 参见 `GPU_UPGRADE_GUIDE.md`
```bash
# 升级PyTorch到支持sm_86的版本
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

## 并行训练策略

### 单数据集，多模型并行

如果有多个GPU或机器，可以同时训练MLP和CNN：

```bash
# GPU 0: MLP
CUDA_VISIBLE_DEVICES=0 ./train.sh -n mlp -d devign &

# GPU 1: CNN (如果实现)
CUDA_VISIBLE_DEVICES=1 ./train.sh -n cnn -d devign &
```

### 多数据集并行

在单GPU上串行训练多个小数据集：

```bash
# 自动化脚本
for dataset in d2a reveal devign; do
    ./train.sh -n mlp -d $dataset
done
```

预计总时间: ~44小时 (4+18+22)

## 实际训练记录

### 已知问题 (从日志中)

1. **CUDA兼容性**: RTX 3080需要PyTorch升级
2. **显存不足**: batch_size=4会OOM，需要使用batch_size=2
3. **数据加载**: Devign数据集加载时间约2分钟

### 成功案例

目前尚无完整成功的训练记录。所有尝试都因以下原因失败：
- CUDA版本不兼容
- 显存溢出 (OOM)
- 数据格式错误

**建议**: 先解决CUDA兼容性问题，再开始正式训练。

## 验证脚本

创建了一个时间估算验证脚本：`scripts/estimate_training_time.py`

使用方法：
```bash
python scripts/estimate_training_time.py --model mlp --dataset devign
python scripts/estimate_training_time.py --model cnn --dataset all
```

## 总结

### MLP模型训练时间 (10 epochs, batch_size=2)

- 小数据集 (d2a, reveal, devign): **4-22小时**
- 大数据集 (vuldeepecker, mvd): **4-5天**
- 超大数据集 (draper): **35-42天**

### CNN模型训练时间 (20 epochs, batch_size=128)

- 小数据集: **0.1-1小时**
- 大数据集: **3-5小时**
- 超大数据集: **1-2天**

### 关键建议

1. **优先级**: 先训练小数据集验证环境和代码
2. **显存管理**: 使用batch_size=2或更小
3. **环境准备**: 先升级PyTorch解决CUDA兼容性
4. **时间规划**: draper数据集需要1个多月，建议最后训练
5. **监控**: 使用tensorboard或日志监控训练进度

## 更新日志

- 2025-11-03: 初始版本，基于数据集分析和硬件规格估算
- 待更新: 实际训练完成后，更新真实训练时间数据

## 参考资料

- 训练脚本: `train.sh`
- 训练代码: `train_vulberta.py`
- GPU升级指南: `GPU_UPGRADE_GUIDE.md`
- 快速开始: `QUICKSTART.md`
