# VulBERTa 训练脚本使用说明

## 概述

已成功创建train.sh脚本用于在screen会话中自动执行VulBERTa模型训练。脚本支持命令行参数修改超参数，并生成详细的训练报告。

## 文件说明

- **train.sh**: Shell包装脚本，处理环境设置和日志输出
- **train_vulberta.py**: Python训练脚本，实现模型训练逻辑

## 基本用法

```bash
# 在screen会话中运行训练
./train.sh -n model_name -d dataset 2>&1 | tee training.log
```

### 必需参数

- `-n, --model_name`: 模型架构
  - `mlp`: VulBERTa-MLP模型
  - `cnn`: VulBERTa-CNN模型（当前需使用Jupyter notebook）

- `-d, --dataset`: 训练数据集
  - `devign`, `draper`, `reveal`, `mvd`, `vuldeepecker`, `d2a`

### 可选参数

- `--batch_size`: 批次大小（默认：MLP=4, CNN=128）
- `--epochs`: 训练轮数（默认：MLP=10, CNN=20）
- `--learning_rate`: 学习率（默认：MLP=3e-05, CNN=0.0005）
- `--seed`: 随机种子（默认：MLP=42, CNN=1234）
- `--fp16`: 使用混合精度训练（默认：MLP=True, CNN=False）
- `--cpu`: 强制使用CPU训练（用于解决CUDA兼容性问题）

## 使用示例

```bash
# 1. 使用默认参数训练MLP模型（复现原始性能）
./train.sh -n mlp -d devign 2>&1 | tee training.log

# 2. 减小批次大小以适应GPU内存限制
./train.sh -n mlp -d devign --batch_size 2 2>&1 | tee training.log

# 3. 快速测试（减少epochs）
./train.sh -n mlp -d devign --epochs 1 --batch_size 2 2>&1 | tee training_test.log

# 4. 在CPU上训练（解决CUDA兼容性问题）
./train.sh -n mlp -d devign --cpu 2>&1 | tee training_cpu.log
```

## 默认超参数

### MLP模型（VulBERTa-MLP）
- Batch size: 4 (推荐使用2以避免OOM)
- Epochs: 10
- Learning rate: 3e-05
- Seed: 42
- FP16: True

### CNN模型（VulBERTa-CNN）
- Batch size: 128
- Epochs: 20
- Learning rate: 0.0005
- Seed: 1234
- FP16: False

## Screen会话管理

```bash
# 创建新的screen会话
screen -S test

# 从screen会话分离
Ctrl+A, D

# 重新连接到screen会话
screen -r test

# 列出所有screen会话
screen -list

# 在screen会话中运行训练
cd /home/green/energy_dl/test/VulBERTa
./train.sh -n mlp -d devign 2>&1 | tee training.log
```

## 训练报告

训练完成后会生成两份报告：

1. **命令行输出**: 包含训练进度、性能指标、错误信息、时间统计
2. **文件报告**: `training_report_<model>_<dataset>_<timestamp>.txt`

报告内容包括：
- 模型和数据集信息
- 训练开始/结束时间
- 总耗时
- 超参数配置
- 训练/验证性能指标
- 错误日志（如有）

## 已知问题和解决方案

### 1. CUDA兼容性问题

**问题**: PyTorch 1.7.0+cu101不支持RTX 3080 GPU (sm_86)
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案**:
```bash
# 方案1: 使用CPU训练
./train.sh -n mlp -d devign --cpu 2>&1 | tee training.log

# 方案2: 升级PyTorch到支持sm_86的版本（推荐）
# 需要PyTorch 1.8.0+cu111或更高版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.1 -c pytorch -c nvidia
```

### 2. GPU内存不足

**问题**: 批次大小过大导致OOM错误

**解决方案**:
```bash
# 减小批次大小到2（推荐值）
./train.sh -n mlp -d devign --batch_size 2 2>&1 | tee training.log

# 如果还不够，进一步减小到1
./train.sh -n mlp -d devign --batch_size 1 2>&1 | tee training.log
```

**注意**: 根据实际测试，在RTX 3080上batch_size=4会导致OOM，推荐默认使用batch_size=2。

### 3. 数据集列名问题

**问题**: 已修正。devign数据集使用'target'列而非'label'列

**状态**: ✅ 已在train_vulberta.py中修复

## 环境要求

- Python: 3.8.20 (vulberta conda环境)
- PyTorch: 1.7.0+cu101 (当前版本)
- Transformers: 4.4.1
- CUDA: 可用但需要兼容GPU
- GPU: NVIDIA GPU (推荐) 或使用 --cpu 选项

## 数据和模型位置

- 数据集: `data/finetune/<dataset>/`
- 预训练模型: `models/VulBERTa/`
- 微调模型输出: `models/VB-MLP_<dataset>/` 或 `models/VB-CNN_<dataset>/`
- Tokenizer: `tokenizer/`

## 性能优化建议

针对RTX 3080 GPU和当前环境限制：

1. **升级PyTorch**: 优先升级PyTorch到支持sm_86的版本以使用GPU（参见 `GPU_UPGRADE_GUIDE.md`）
2. **使用推荐batch_size**: 使用batch_size=2以避免OOM（即使升级PyTorch后）
3. **快速验证**: 使用`--epochs 1`进行初步测试
4. **备选方案--cpu**: 如果CUDA版本不兼容，可使用CPU训练（速度较慢）

## 训练时间估算

详细的训练时间估算请参见：`docs/TRAINING_TIME_ESTIMATION.md`

**快速参考** (使用 batch_size=2, RTX 3080):

### MLP模型 (10 epochs)
- 小数据集 (d2a, reveal, devign): 4-24小时
- 大数据集 (vuldeepecker, mvd): 4-5天
- 超大数据集 (draper): 35-42天

### CNN模型 (20 epochs)
- 小数据集: 0.1-1小时
- 大数据集: 3-5小时
- 超大数据集: 1-2天

使用估算工具：
```bash
python scripts/estimate_training_time.py --model mlp --dataset devign
python scripts/estimate_training_time.py --model cnn --dataset all
```

## 故障排除

查看详细错误信息：
```bash
# 查看最新的训练日志
tail -100 training.log

# 查看训练报告
cat training_report_mlp_devign_*.txt

# 检查Python进程
ps aux | grep train_vulberta.py
```

## 复现原始性能

使用默认参数复现论文中的原始性能：

```bash
# 对于各个数据集使用默认参数
./train.sh -n mlp -d devign 2>&1 | tee training_devign.log
./train.sh -n mlp -d draper 2>&1 | tee training_draper.log
./train.sh -n mlp -d reveal 2>&1 | tee training_reveal.log
./train.sh -n mlp -d mvd 2>&1 | tee training_mvd.log
./train.sh -n mlp -d vuldeepecker 2>&1 | tee training_vuldeepecker.log
./train.sh -n mlp -d d2a 2>&1 | tee training_d2a.log
```

**注意**: 由于GPU兼容性问题，建议在所有命令中添加`--cpu`选项，或先升级PyTorch版本。

## 总结

train.sh脚本已完全实现以下功能：
- ✅ 支持命令行参数修改超参数
- ✅ 默认值与原始仓库一致
- ✅ 在screen会话中运行
- ✅ 生成详细的训练报告（包含性能、错误、时间）
- ✅ 错误处理和报告
- ✅ GPU/CPU灵活切换
- ✅ 修复了数据集兼容性问题

已知限制：
- ⚠️ 当前PyTorch版本不支持RTX 3080（需要使用--cpu或升级PyTorch）
- ⚠️ CNN模型训练暂未在脚本中实现（需使用Jupyter notebook）
