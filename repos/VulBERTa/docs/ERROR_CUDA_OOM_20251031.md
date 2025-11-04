# 训练错误诊断报告

## 错误信息

**错误类型**: CUDA Out of Memory (OOM)

**错误详情**:
```
RuntimeError: CUDA out of memory. Tried to allocate 192.00 MiB
(GPU 0; 9.77 GiB total capacity; 8.10 GiB already allocated; 53.81 MiB free; 8.28 GiB reserved in total by PyTorch)
```

## 问题分析

### GPU内存使用情况
- **GPU型号**: NVIDIA GeForce RTX 3080
- **总显存**: 10 GB (实际可用 9.77 GiB)
- **已分配**: 8.10 GiB
- **剩余可用**: 53.81 MiB
- **尝试分配**: 192.00 MiB
- **PyTorch保留**: 8.28 GiB

### 训练配置
- **模型**: VulBERTa-MLP (RobertaForSequenceClassification)
- **模型参数量**: 124,836,866 (~125M)
- **训练样本**: 21,854 (train) + 2,732 (val)
- **序列长度**: 1024 tokens (最大)
- **Batch size**: 4
- **FP16**: True (启用混合精度)

### 根本原因

VulBERTa是基于RoBERTa-base的大型模型，内存需求估算：

1. **模型权重** (~500 MB FP16):
   - 125M参数 × 2 bytes (FP16) = 250 MB
   - 梯度 (FP16): 250 MB
   - 优化器状态 (Adam): ~500 MB (FP32 master weights + momentum)
   - **总计**: ~1 GB

2. **激活值** (最消耗内存):
   - Batch size = 4
   - Sequence length = 1024
   - Hidden size = 768
   - 12层Transformer
   - 每层中间激活: 4 × 1024 × 768 × 4 bytes ≈ 12 MB
   - 所有层: 12 MB × 12 layers × 多个中间张量 ≈ **6-8 GB**

3. **总内存需求**: 1 GB (模型) + 6-8 GB (激活) = **7-9 GB**

这已经接近或超过RTX 3080的10GB显存。

## 解决方案

### 方案1: 减小Batch Size（推荐）

将batch_size从4降低到2或1：

```bash
# Batch size = 2 (推荐)
./train.sh -n mlp -d devign --batch_size 2 2>&1 | tee training.log

# Batch size = 1 (最保守)
./train.sh -n mlp -d devign --batch_size 1 2>&1 | tee training.log
```

**预期内存减少**:
- Batch size 4 → 2: 减少约3-4 GB
- Batch size 4 → 1: 减少约6-7 GB

### 方案2: 梯度累积（保持有效batch size）

如果需要保持batch size=4的效果（for reproducibility），可以使用梯度累积：
- Physical batch size = 1 or 2
- Accumulation steps = 4 or 2
- Effective batch size = 4

**需要修改训练代码**添加`gradient_accumulation_steps`参数。

### 方案3: 减少序列长度

将max_length从1024降低到512：
- 内存减少约50%
- 但可能影响长代码片段的处理

**不推荐**：可能影响模型性能和复现性。

## 推荐行动

**立即执行**: 使用batch_size=2重新训练

```bash
./train.sh -n mlp -d devign --batch_size 2 2>&1 | tee training.log
```

## 内存优化效果预测

| Batch Size | 预计显存使用 | 可行性 | 训练时间影响 |
|-----------|------------|--------|------------|
| 4 | ~9 GB | ❌ OOM | - |
| 2 | ~5-6 GB | ✅ 可行 | +50% |
| 1 | ~3-4 GB | ✅ 非常安全 | +100% |

## 文档更新

需要更新以下文档的默认batch size建议：
- [ ] TRAINING_GUIDE.md - 添加GPU内存限制说明
- [ ] QUICKSTART.md - 更新默认batch size推荐
- [ ] train_vulberta.py - 可能需要调整默认值注释

## 时间记录

- **错误发生时间**: 2025-10-31 16:07:11
- **训练持续时间**: 2分20秒 (仅初始化+第一个batch)
- **错误位置**: 第一个训练batch的前向传播
