# 环境复现配置文档

## 项目信息

- **项目名称**: PyTorch ResNet CIFAR-10
- **项目地址**: https://github.com/akamaster/pytorch_resnet_cifar10
- **任务**: 在CIFAR-10数据集上训练ResNet模型
- **训练方式**: 本地GPU训练

## 硬件要求

### 最低配置
- **GPU**: NVIDIA GPU，至少4GB显存（用于训练ResNet20-56）
- **推荐GPU**: 8GB+显存（用于训练ResNet110）
- **ResNet1202要求**: 16GB显存（或使用优化配置在10GB显存上运行）

### 测试环境配置
- **GPU**: NVIDIA GeForce RTX 3080 (10GB)
- **CUDA版本**: 12.2
- **操作系统**: Linux (Ubuntu/类似发行版)
- **驱动版本**: 535.183.01

## 软件环境

### 1. Conda环境

#### 创建环境
```bash
conda create -n pytorch_resnet_cifar10 python=3.10 -y
```

#### 激活环境
```bash
conda activate pytorch_resnet_cifar10
```

### 2. Python依赖包

#### 核心依赖
安装PyTorch和torchvision（CUDA 12.1版本）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 依赖包版本
经测试的版本组合：
- Python: 3.10.19
- PyTorch: 2.5.1+cu121
- torchvision: 0.20.1+cu121
- CUDA: 12.1 (PyTorch内置)
- cuDNN: 9.1.0.70

其他自动安装的依赖：
- numpy: 2.1.2
- pillow: 11.3.0
- nvidia-cuda-runtime-cu12: 12.1.105
- nvidia-cudnn-cu12: 9.1.0.70

### 3. 环境验证

运行验证脚本：
```bash
python scripts/verify_environment.py
```

或手动验证：
```python
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
```

预期输出：
```
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU Name: NVIDIA GeForce RTX 3080
```

## 数��集

### CIFAR-10数据集
- **大小**: ~170MB
- **类别**: 10个类别
- **训练集**: 50,000张图片
- **测试集**: 10,000张图片
- **图片尺寸**: 32x32 RGB

### 自动下载
代码会在首次运行时自动下载数据集到 `./data` 目录（trainer.py:92）。
不需要手动下载。

### 手动下载（可选）
如果自动下载失败，可以手动下载：
```bash
mkdir -p data
# 下载并解压CIFAR-10数据集到data目录
# 可从 https://www.cs.toronto.edu/~kriz/cifar.html 下载
```

## 模型信息

本仓库提供6个ResNet模型的实现：

| 模型 | 层数 | 参数量 | 论文测试误差 | 本实现测试误差 | 训练时长(RTX 3080) |
|------|------|--------|------------|--------------|-------------------|
| ResNet20 | 20 | 0.27M | 8.75% | 8.27% | ~0.2小时 |
| ResNet32 | 32 | 0.46M | 7.51% | 7.37% | ~0.3小时 |
| ResNet44 | 44 | 0.66M | 7.17% | 6.90% | ~0.4小时 |
| ResNet56 | 56 | 0.85M | 6.97% | 6.61% | ~0.5小时 |
| ResNet110 | 110 | 1.7M | 6.43% | 6.32% | ~0.9小时 |
| ResNet1202 | 1202 | 19.4M | 7.93% | 6.18% | ~需要优化配置 |

## 训练配置

### 默认超参数
- **Epochs**: 200
- **Batch Size**: 128
- **初始学习率**: 0.1
- **学习率调度**: MultiStepLR，在epoch 100和150降低学习率
- **优化器**: SGD，momentum=0.9，weight_decay=1e-4
- **数据增强**: RandomHorizontalFlip, RandomCrop(32, 4)
- **归一化**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### 训练命令

#### 训练所有模型
```bash
chmod +x run.sh
./run.sh
```

#### 训练单个模型
```bash
# ResNet56示例
python -u trainer.py --arch=resnet56 --save-dir=save_resnet56
```

#### 自定义配置
```bash
python -u trainer.py \
    --arch=resnet56 \
    --epochs=100 \
    --batch-size=128 \
    --lr=0.1 \
    --save-dir=save_resnet56_custom
```

#### ResNet1202特殊配置（10GB显存）
```bash
# 方案1: 减小批次+半精度
python -u trainer.py --arch=resnet1202 --batch-size=32 --half --save-dir=save_resnet1202

# 方案2: 仅减小批次
python -u trainer.py --arch=resnet1202 --batch-size=16 --save-dir=save_resnet1202
```

## 预训练模型

### 下载预训练模型
仓库提供了所有模型的预训练权重：

```bash
# 创建目录
mkdir -p pretrained_models

# 下载示例（ResNet56）
wget https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet56-4bfd9763.th -O pretrained_models/resnet56.th
```

### 使用预训练模型评估
```bash
python trainer.py \
    --arch=resnet56 \
    --resume=pretrained_models/resnet56.th \
    --evaluate
```

## 项目结构

```
pytorch_resnet_cifar10/
├── trainer.py              # 主训练脚本
├── resnet.py              # ResNet模型定义
├── hubconf.py             # PyTorch Hub配置
├── run.sh                 # 批量训练脚本
├── LICENSE                # BSD 2-Clause许可证
├── README.md              # 项目说明
├── data/                  # CIFAR-10数据集（自动下载）
├── pretrained_models/     # 预训练模型（需手动下载）
├── scripts/               # 辅助脚本
│   ├── estimate_training_time.py  # 训练时间估算
│   └── verify_environment.py      # 环境验证
└── docs/                  # 文档
    ├── environment_setup.md       # 本文档
    └── training_optimization.md   # 训练优化建议
```

## 输出文件

### 训练过程输出
- **模型检查点**: `save_<model_name>/checkpoint.th`（每10个epoch保存一次）
- **最佳模型**: `save_<model_name>/model.th`（最佳准确率的模型）
- **训练日志**: `log_<model_name>`（如果使用run.sh）

### 检查点内容
```python
{
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'best_prec1': best_prec1
}
```

## 快速开始指南

### 1. 环境准备（首次使用）
```bash
# 创建conda环境
conda create -n pytorch_resnet_cifar10 python=3.10 -y

# 激活环境
conda activate pytorch_resnet_cifar10

# 安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 验证环境
python scripts/verify_environment.py
```

### 2. 估算训练时间
```bash
python scripts/estimate_training_time.py
```

### 3. 开始训练
```bash
# 训练单个模型（推荐先测试）
python -u trainer.py --arch=resnet20 --save-dir=save_resnet20

# 或训练所有模型
./run.sh
```

### 4. 监控训练
训练过程会定期输出：
```
Epoch: [0][0/390]    Time 0.123 (0.123)    Loss 2.3456 (2.3456)    Prec@1 10.000 (10.000)
```

## 常见问题

### 1. CUDA Out of Memory
**问题**: 显存不足
**解决**:
- 减小batch size: `--batch-size=64`或更小
- 使用半精度: `--half`
- 参考 `docs/training_optimization.md`

### 2. 数据下载缓慢
**问题**: CIFAR-10下载速度慢
**解决**:
- 等待自动下载完成（仅首次需要）
- 或手动下载后放到`./data`目录

### 3. 训练精度不如论文
**原因**:
- 代码不使用验证集选择最佳模型（与论文不同）
- 随机性影响
**解决**: 多次训练取最佳结果

### 4. 想要加快训练
**解决**:
- 减少epochs: `--epochs=100`
- 增大batch size（如果显存允许）
- 使用多GPU（需修改代码）

## 环境导出与迁移

### 导出环境配置
```bash
# 导出完整环境
conda env export > environment.yml

# 或仅导出pip包
pip list --format=freeze > requirements.txt
```

### 在新机器上重建环境
```bash
# 方法1: 使用conda环境文件
conda env create -f environment.yml

# 方法2: 使用pip requirements
conda create -n pytorch_resnet_cifar10 python=3.10 -y
conda activate pytorch_resnet_cifar10
pip install -r requirements.txt
```

## 参考文献

1. **原始论文**: Kaiming He, et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
2. **CIFAR-10数据集**: https://www.cs.toronto.edu/~kriz/cifar.html
3. **PyTorch文档**: https://pytorch.org/docs/

## 技术支持

- **项目Issues**: https://github.com/akamaster/pytorch_resnet_cifar10/issues
- **PyTorch论坛**: https://discuss.pytorch.org/

## 更新日志

- **2025-11-01**: 创建环境配置文档
- **2025-11-01**: 添加RTX 3080训练时间估算
- **2025-11-01**: 添加ResNet1202显存优化方案
