# VulBERTa 训练时长评估和优化建议

## 硬件环境
- **GPU**: NVIDIA GeForce RTX 3080 10GB VRAM
- **训练框架**: PyTorch 1.7.0 + CUDA 10.1
- **显存**: 10GB

## 训练配置分析

### 预训练 (Pretraining_VulBERTa.ipynb)

#### 配置参数
```python
TrainingArguments(
    output_dir="models/VulBERTa",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    save_steps=10000,
    save_total_limit=4,
    seed=42
)
```

#### 模型配置选项

1. **VulBERTa-small** (推荐用于快速实验)
   - num_hidden_layers: 3
   - num_attention_heads: 3
   - 参数量: ~10M
   - 预计训练时间: **1-2小时** (取决于数据集大小)

2. **VulBERTa-medium**
   - num_hidden_layers: 6
   - num_attention_heads: 12
   - 参数量: ~40M
   - 预计训练时间: **4-6小时**

3. **VulBERTa-base**
   - num_hidden_layers: 12
   - num_attention_heads: 12
   - 参数量: ~84M
   - 预计训练时间: **8-12小时**

4. **VulBERTa-large**
   - num_hidden_layers: 24
   - num_attention_heads: 16
   - hidden_size: 1024
   - 参数量: ~300M
   - 预计训练时间: **>24小时**
   - ⚠️ **警告**: 可能超出RTX 3080 10GB显存限制

### 微调 (Finetuning)
- 预计时间: 30分钟 - 2小时 (取决于数据集和模型大小)
- 显存需求: 相对较小

## 训练时长估算

### 保守估计 (基于单卡RTX 3080)

| 任务 | 配置 | 预计时长 | 是否超过2h |
|-----|------|---------|----------|
| 预训练-small | 10 epochs, bs=8 | 1-2h | ❌ |
| 预训练-medium | 10 epochs, bs=8 | 4-6h | ✅ |
| 预训练-base | 10 epochs, bs=8 | 8-12h | ✅ |
| 预训练-large | 10 epochs, bs=8 | >24h | ✅ |
| 微调-MLP | 5 epochs | 0.5-1h | ❌ |
| 微调-CNN | 5 epochs | 1-2h | ❌ |

## 优化建议

### 1. 缩短训练时间的方法

#### 方法A: 使用更小的模型配置
**建议**: 使用 VulBERTa-small 进行初步验证
```python
# 在 Pretraining_VulBERTa.ipynb 中选择 small 配置
config = RobertaConfig(
    vocab_size=50000,
    max_position_embeddings=1026,
    num_attention_heads=3,
    num_hidden_layers=3,  # 最小配置
    type_vocab_size=1,
)
```
**时间节省**: 可将预训练时间从8-12h缩短至1-2h

#### 方法B: 减少训练轮数
**建议**: 将epochs从10减少到3-5
```python
training_args = TrainingArguments(
    ...
    num_train_epochs=3,  # 从10降到3
    ...
)
```
**时间节省**: 60-70%
**影响**: 模型性能可能略有下降，但对于验证流程足够

#### 方法C: 增加批次大小 (需要足够显存)
**建议**: 在显存允许的情况下增加batch size
```python
training_args = TrainingArguments(
    ...
    per_device_train_batch_size=16,  # 从8增加到16
    gradient_accumulation_steps=2,   # 如果显存不足
    ...
)
```
**时间节省**: 30-40%
**注意**: RTX 3080 10GB可能无法支持大模型的大batch size

#### 方法D: 使用混合精度训练
**建议**: 启用FP16训练
```python
training_args = TrainingArguments(
    ...
    fp16=True,  # 启用混合精度
    ...
)
```
**时间节省**: 40-50%
**额外好处**: 减少显存占用，可以增加batch size

#### 方法E: 减少数据集大小
**建议**: 在数据加载时采样部分数据
```python
# 在 PretrainDataset 中
mydata = mydata.sample(frac=0.3)  # 只使用30%的数据
```
**时间节省**: 70%
**影响**: 仅用于快速验证，不适合正式实验

#### 方法F: 使用预训练模型直接微调
**建议**: 跳过预训练阶段，直接下载并使用提供的预训练模型
```python
# 直接加载预训练模型进行微调
from transformers import RobertaForSequenceClassification
model = RobertaForSequenceClassification.from_pretrained("models/VulBERTa")
```
**时间节省**: 跳过全部预训练时间
**适用场景**: 只想验证微调和评估流程

### 2. 推荐的快速验证方案

#### 方案1: 最小验证 (预计总时长: 2-3小时)
```python
# 预训练配置
config = VulBERTa-small
num_train_epochs = 3
per_device_train_batch_size = 16
fp16 = True

# 微调配置
使用small模型
num_train_epochs = 3
```

#### 方案2: 中等验证 (预计总时长: 4-6小时)
```python
# 预训练配置
config = VulBERTa-medium
num_train_epochs = 5
per_device_train_batch_size = 8
fp16 = True

# 微调配置
使用medium模型
num_train_epochs = 5
```

#### 方案3: 跳过预训练 (预计总时长: <2小时)
```
1. 下载预训练模型
2. 直接运行微调
3. 运行评估
```

### 3. 显存优化建议

#### 监控显存使用
```python
# 在训练代码中添加显存监控
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

#### 如果遇到OOM (Out of Memory)
1. 减小batch_size
2. 启用gradient_accumulation_steps
3. 启用gradient_checkpointing
4. 选择更小的模型配置

```python
# 显存不足时的配置
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # 减小到4
    gradient_accumulation_steps=4,  # 累积4步
    gradient_checkpointing=True,    # 启用检查点
    fp16=True,                      # 混合精度
)
```

### 4. 完整的优化训练脚本示例

创建 `scripts/train_optimized.py`:

```python
#!/usr/bin/env python3
"""
VulBERTa 优化训练脚本
适用于RTX 3080 10GB，目标训练时间<2小时
"""

from transformers import TrainingArguments

# 优化的训练参数
training_args = TrainingArguments(
    output_dir="models/VulBERTa_optimized",
    do_train=True,
    overwrite_output_dir=True,

    # 核心优化参数
    per_device_train_batch_size=16,  # 增加batch size
    gradient_accumulation_steps=1,
    num_train_epochs=3,              # 减少epochs
    fp16=True,                       # 混合精度训练

    # 其他参数
    save_steps=5000,                 # 减少保存频率
    save_total_limit=2,              # 只保留2个checkpoint
    logging_steps=100,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,

    # 性能优化
    dataloader_num_workers=4,        # 多进程数据加载
    dataloader_pin_memory=True,

    seed=42
)
```

## 总结

### 对于2小时以内的训练目标

**推荐方案**:
1. 使用 VulBERTa-small 配置
2. 减少epochs到3-5
3. 启用FP16混合精度
4. 增加batch size到16

**或者**:
- 直接下载预训练模型，跳过预训练阶段
- 只进行微调和评估 (总时长: <2小时)

### 最佳实践
1. **首次运行**: 使用small配置 + 少量epochs验证流程是否正常
2. **正式训练**: 根据需要选择medium或base配置
3. **长时间训练**: 建议使用tmux或screen，避免断线中断训练

## 监控训练进度

创建 `scripts/monitor_training.sh`:
```bash
#!/bin/bash
# 监控GPU使用情况
watch -n 1 nvidia-smi
```

在另一个终端查看训练日志：
```bash
tail -f models/VulBERTa/trainer_state.json
```
