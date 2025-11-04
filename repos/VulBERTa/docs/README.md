# VulBERTa 复现指南总结

## 概述
本文档总结了VulBERTa模型复现的完整流程，包括环境搭建、数据准备、训练配置和优化建议。

## 完成状态

### ✓ 已完成
1. **仓库代码检查** ✓
   - 确认为本地训练
   - 包含完整的训练、微调、评估流程
   - 使用Jupyter Notebook

2. **环境配置** ✓
   - 创建conda环境: `vulberta`
   - Python 3.8.20
   - PyTorch 1.7.0+cu101
   - Transformers 4.4.1
   - 所有依赖已安装

3. **文档和脚本** ✓
   - 环境搭建文档: `docs/reproduction_setup.md`
   - 数据下载指南: `docs/download_guide.md`
   - 训练时长评估: `docs/training_time_estimation.md`
   - 数据验证脚本: `scripts/verify_data.py`

### ⚠️ 待完成
1. **数据下载** (需要手动操作)
   - 预训练数据
   - 微调数据
   - (可选) 预训练模型

## 快速开始

### 1. 激活环境
```bash
cd /home/green/energy_dl/test/VulBERTa
conda activate vulberta
```

### 2. 下载数据
按照 `docs/download_guide.md` 中的说明从OneDrive下载数据：
- 必需: `data.zip` - 包含训练数据
- 可选: `pretraining_model.zip` - 如果要跳过预训练
- 可选: `finetuning_models.zip` - 如果只想评估

### 3. 验证数据
```bash
python scripts/verify_data.py
```

### 4. 开始训练

#### 选项A: 完整流程 (从头预训练)
```bash
# 启动JupyterLab
jupyter lab

# 然后按顺序运行:
# 1. Tokenizer_training.ipynb (tokenizer已存在,可跳过)
# 2. Pretraining_VulBERTa.ipynb (选择small配置以节省时间)
# 3. Finetuning_VulBERTa-MLP.ipynb
# 4. Evaluation_VulBERTa-MLP.ipynb
```

#### 选项B: 快速验证 (使用预训练模型)
```bash
# 1. 下载预训练模型
# 2. 启动JupyterLab
jupyter lab

# 3. 运行微调和评估:
# - Finetuning_VulBERTa-MLP.ipynb
# - Evaluation_VulBERTa-MLP.ipynb
```

## 训练时长预估 (RTX 3080 10GB)

### 原始配置 (base模型, 10 epochs)
- **预训练**: 8-12小时 ⚠️ **超过2小时**
- **微调**: 0.5-1小时
- **评估**: <30分钟

### 优化配置 (small模型, 3 epochs, FP16)
- **预训练**: 1-2小时 ✓ **满足要求**
- **微调**: 0.5小时
- **评估**: <30分钟
- **总计**: 约2-3小时

### 跳过预训练方案
- **下载预训练模型**: 根据网速
- **微调**: 0.5-1小时
- **评估**: <30分钟
- **总计**: <2小时 ✓ **推荐用于快速验证**

## 推荐的优化策略

### 如果预计训练超过2小时，建议：

1. **使用small模型配置**
   ```python
   config = RobertaConfig(
       vocab_size=50000,
       max_position_embeddings=1026,
       num_attention_heads=3,
       num_hidden_layers=3,  # small配置
       type_vocab_size=1,
   )
   ```

2. **减少训练轮数**
   ```python
   training_args = TrainingArguments(
       ...
       num_train_epochs=3,  # 从10减到3
       ...
   )
   ```

3. **启用混合精度训练**
   ```python
   training_args = TrainingArguments(
       ...
       fp16=True,  # 启用FP16
       per_device_train_batch_size=16,  # 可以增加batch size
       ...
   )
   ```

4. **或直接使用预训练模型**
   - 下载 `pretraining_model.zip`
   - 跳过预训练步骤
   - 直接进行微调

详细优化方案请参考: `docs/training_time_estimation.md`

## 硬件环境

- **GPU**: NVIDIA GeForce RTX 3080 10GB VRAM
- **Driver**: 535.183.01
- **CUDA**: 12.2 (运行时使用CUDA 10.1)
- **系统**: Linux 6.2.0-39-generic

## 目录结构

```
VulBERTa/
├── docs/
│   ├── reproduction_setup.md         # 环境搭建文档
│   ├── download_guide.md             # 数据下载指南
│   └── training_time_estimation.md   # 训练时长评估
├── scripts/
│   ├── verify_data.py                # 数据验证脚本
│   ├── download_data.sh              # 下载辅助脚本
│   └── download_data.py              # 自动下载脚本(可能需要手动)
├── data/
│   ├── pretrain/                     # 预训练数据 (待下载)
│   ├── finetune/                     # 微调数据 (待下载)
│   └── tokenizer/                    # tokenizer训练数据 (待下载)
├── models/
│   └── VulBERTa/                     # 预训练模型 (可选,待下载)
├── tokenizer/
│   ├── drapgh-vocab.json             # ✓ 已存在
│   └── drapgh-merges.txt             # ✓ 已存在
├── custom.py                          # 自定义数据处理
├── models.py                          # 模型定义
└── *.ipynb                            # 训练notebooks
```

## 常见问题

### Q1: 数据下载很慢怎么办？
A: OneDrive在国内访问较慢，建议：
- 使用代理
- 在网络较好时下载
- 或联系论文作者获取其他下载源

### Q2: 显存不足(OOM)怎么办？
A: 尝试以下方法：
- 减小batch_size到4或更小
- 使用gradient_accumulation_steps
- 选择更小的模型配置(small而非base)
- 启用gradient_checkpointing

### Q3: 我只想验证部分流程，怎么办？
A: 根据需求选择：
- **只验证预训练**: 下载`data.zip`，运行Pretraining notebook
- **只验证微调**: 下载`data.zip`和`pretraining_model.zip`，运行Finetuning notebook
- **只验证评估**: 下载所有zip文件，运行Evaluation notebook

### Q4: 训练中断了怎么办？
A:
- 训练会自动保存checkpoint (每10000步)
- 可以从最近的checkpoint恢复训练
- 建议使用tmux或screen运行长时间训练

### Q5: 如何监控训练进度？
A:
```bash
# 终端1: 监控GPU
watch -n 1 nvidia-smi

# 终端2: 查看训练日志
tail -f models/VulBERTa/trainer_state.json
```

## 下一步

1. **下载数据**: 按照`docs/download_guide.md`操作
2. **验证数据**: 运行`python scripts/verify_data.py`
3. **选择训练方案**: 根据时间要求选择配置
4. **开始训练**: 启动JupyterLab运行notebooks

## 相关文档

- [环境搭建详细文档](docs/reproduction_setup.md)
- [数据下载详细指南](docs/download_guide.md)
- [训练时长评估和优化](docs/training_time_estimation.md)

## 联系和反馈

如有问题，请参考：
- VulBERTa论文: https://ieeexplore.ieee.org/document/9892280
- GitHub仓库: https://github.com/ICL-ml4csec/VulBERTa
