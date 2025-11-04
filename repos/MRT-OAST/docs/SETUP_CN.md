# MRT-OAST 环境配置说明

## 环境概览

已成功创建名为 `mrt-oast` 的conda环境，包含以下依赖：

### 已安装的包
- Python 3.7
- PyTorch 1.13.1 (with CUDA 11.7)
- TorchVision 0.14.1
- TorchAudio 0.13.1
- NumPy 1.21.6
- Matplotlib 3.5.3
- tqdm
- javalang
- tensorboard

### 硬件支持
- CUDA 11.7
- GPU: NVIDIA GeForce RTX 3080

## 使用方法

### 1. 激活环境
```bash
conda activate mrt-oast
```

### 2. 训练模型
```bash
# 训练和验证MRT模型
python main_batch.py --cuda

# 查看更多参数选项
python main_batch.py --help
```

### 3. 测试模型
```bash
# 快速测试
python main_batch.py --cuda --is_test --quick_test

# 详细测试
python main_batch.py --cuda --is_test
```

### 4. 退出环境
```bash
conda deactivate
```

## 常用命令参数

主要训练参数（在 `main_batch.py` 中）：
- `--cuda`: 使用CUDA加速
- `--epochs`: 训练轮数（默认：10）
- `--batch_size`: 批次大小（默认：64）
- `--lr`: 学习率（默认：0.0001）
- `--sen_max_len`: 最大序列长度（默认：256）
- `--data`: 数据集路径
- `--ast_type`: AST类型（AST/OAST）

## 环境维护

### 查看已安装的包
```bash
conda activate mrt-oast
conda list
```

### 导出环境（用于备份）
```bash
conda env export > environment_backup.yml
```

### 删除环境（如需重建）
```bash
conda deactivate
conda env remove -n mrt-oast
```

### 重新创建环境
```bash
conda env create -f environment.yml
conda activate mrt-oast
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## 故障排除

### 如果CUDA不可用
1. 检查NVIDIA驱动是否正确安装：`nvidia-smi`
2. 确认CUDA版本兼容性
3. 重新安装PyTorch

### 如果内存不足
- 减小batch_size参数
- 减小模型参数（d_model, d_ff等）
- 使用梯度累积

## 数据集准备

确保以下数据文件存在于 `origindata/` 目录：
- `OJClone_with_AST+OAST.csv`
- `GCJ_with_AST+OAST.csv`
- `BCB_with_AST+OAST.csv`
- 对应的字典文件（`*_dictionary.txt`）
- 训练/验证/测试划分文件

## 注意事项

1. 首次运行会预处理数据，可能需要较长时间
2. 训练过程中会自动保存最佳模型到 `model/` 目录
3. TensorBoard日志保存在 `model/log_*/` 目录
4. 确保有足够的磁盘空间存储模型和日志
