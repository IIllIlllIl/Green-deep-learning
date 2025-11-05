# Environment Setup Guide

本目录包含所有项目所需的conda环境配置文件，方便在新机器上快速重建环境。

## 📁 文件列表

### Conda环境文件 (*.yml)

| 环境文件 | 环境名称 | 用途 | Python版本 |
|---------|---------|------|-----------|
| `mrt-oast.yml` | mrt-oast | MRT-OAST代码克隆检测 | 3.8 |
| `dnn_rvsm.yml` | dnn_rvsm | Bug定位模型 | 3.7 |
| `pytorch_resnet_cifar10.yml` | pytorch_resnet_cifar10 | ResNet CIFAR-10分类 | 3.10 |
| `vulberta.yml` | vulberta | VulBERTa漏洞检测 | 3.9 |
| `reid_baseline.yml` | reid_baseline | Person Re-ID | 3.9 |
| `pytorch_examples.yml` | pytorch_examples | PyTorch基��示例 | 3.10 |
| `mutation_runner.yml` | mutation_runner | 主程序运行环境 | 3.10 |

### 映射文件

- `environment_mapping.yml` - 仓库与环境的对应关系

## 🚀 快速开始

### 方法1：创建所有环境（推荐用于新机器）

```bash
cd /home/green/energy_dl/nightly/environment

# 创建所有环境
for env_file in *.yml; do
    echo "Creating environment from $env_file..."
    conda env create -f "$env_file"
done
```

### 方法2：只创建需要的环境

```bash
cd /home/green/energy_dl/nightly/environment

# 示例：只创建mutation_runner和pytorch_resnet_cifar10
conda env create -f mutation_runner.yml
conda env create -f pytorch_resnet_cifar10.yml
```

### 方法3：更新已存在的环境

```bash
# 如果环境已存在，使用update而非create
conda env update -n pytorch_resnet_cifar10 -f pytorch_resnet_cifar10.yml --prune
```

## 📋 详细说明

### 1. mutation_runner 环境

**用途**: 运行 `mutation_runner.py` 主程序

```bash
# 创建环境
conda env create -f mutation_runner.yml

# 激活环境
conda activate mutation_runner

# 运行程序
python3 ../mutation_runner.py --list
```

**特点**:
- 🔹 只需Python 3.10标准库
- 🔹 无特殊依赖（subprocess, json, argparse等都是标准库）
- 🔹 可选安装pandas/matplotlib用于结果分析

### 2. pytorch_resnet_cifar10 环境

**用途**: 训练ResNet CIFAR-10模型

```bash
# 创建环境
conda env create -f pytorch_resnet_cifar10.yml

# 激活环境
conda activate pytorch_resnet_cifar10

# 运行训练
cd ../repos/pytorch_resnet_cifar10
./train.sh
```

**主要依赖**:
- PyTorch >= 1.13
- TorchVision
- NumPy, Pillow

### 3. vulberta 环境

**用途**: 训练VulBERTa漏洞检测模型（MLP和CNN）

```bash
# 创建环境
conda env create -f vulberta.yml

# 激活环境
conda activate vulberta

# 运行训练
cd ../repos/VulBERTa
./train.sh -n mlp -d d2a
```

**主要依赖**:
- Transformers (HuggingFace)
- PyTorch
- scikit-learn
- pandas

### 4. reid_baseline 环境

**用途**: 训练Person Re-ID模型

```bash
# 创建环境
conda env create -f reid_baseline.yml

# 激活环境
conda activate reid_baseline

# 运行训练
cd ../repos/Person_reID_baseline_pytorch
./train.sh -n densenet121
```

**主要依赖**:
- PyTorch
- TorchVision
- scipy
- scikit-learn

### 5. mrt-oast 环境

**用途**: 训练MRT-OAST代码克隆检测模型

```bash
# 创建环境
conda env create -f mrt-oast.yml

# 激活环境
conda activate mrt-oast

# 运行训练
cd ../repos/MRT-OAST
./train.sh
```

**主要依赖**:
- PyTorch
- transformers
- scikit-learn

### 6. dnn_rvsm 环境

**用途**: 训练DNN+RVSM Bug定位模型

```bash
# 创建环境
conda env create -f dnn_rvsm.yml

# 激活环境
conda activate dnn_rvsm

# 运行训练
cd ../repos/bug-localization-by-dnn-and-rvsm
./train.sh
```

**主要依赖**:
- TensorFlow 1.x
- Keras
- scikit-learn

### 7. pytorch_examples 环境

**用途**: 运行PyTorch基础示例（MNIST等）

```bash
# 创建环境
conda env create -f pytorch_examples.yml

# 激活环境
conda activate pytorch_examples

# 运行训练
cd ../repos/examples
./train.sh -n mnist_cnn
```

**主要依赖**:
- PyTorch
- TorchVision
- matplotlib

## 🔧 故障排除

### 问题1: 环境创建失败 - 包版本冲突

**原因**: yml文件中的包版本在新系统上不可用

**解决方案1**: 使用 `--no-builds` 导出（已使用）

**解决方案2**: 移除版本号，让conda自动解决依赖
```bash
# 编辑yml文件，将
# - pytorch=1.13.1=py310_cuda11.7_cudnn8.5.0_0
# 改为
# - pytorch>=1.13.0

conda env create -f modified.yml
```

### 问题2: CUDA版本不匹配

**原因**: yml文件中的PyTorch是为特定CUDA版本编译的

**解决方案**: 根据目标机器的CUDA版本重新安装PyTorch

```bash
# 创建环境（可能使用错误的CUDA版本）
conda env create -f pytorch_resnet_cifar10.yml

# 重新安装正确的PyTorch版本
conda activate pytorch_resnet_cifar10

# 对于CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 对于CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 对于CPU only
conda install pytorch torchvision cpuonly -c pytorch
```

### 问题3: 下载速度慢

**解决方案**: 配置国内镜像源

```bash
# 添加清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes

# 或者使用国内pip镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题4: 环境已存在

**解决方案1**: 删除旧环境后重新创建
```bash
conda env remove -n pytorch_resnet_cifar10
conda env create -f pytorch_resnet_cifar10.yml
```

**解决方案2**: 更新现有环境
```bash
conda env update -n pytorch_resnet_cifar10 -f pytorch_resnet_cifar10.yml --prune
```

## 📦 自定义环境导出

如果你修改了环境并想导出新的配置：

```bash
# 导出环境（不包含build字符串，增强可移植性）
conda env export -n environment_name --no-builds > environment_name.yml

# 导出为requirements.txt（仅pip包）
pip list --format=freeze > requirements.txt
```

## 🔍 验证环境

创建环境后，验证是否正确安装：

```bash
# 激活环境
conda activate pytorch_resnet_cifar10

# 检查Python版本
python --version

# 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查CUDA版本
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# 列出所有包
conda list
```

## 📊 环境大小参考

| 环境 | 大小（约） | 包数量 |
|------|----------|--------|
| mutation_runner | ~200MB | ~50 |
| pytorch_resnet_cifar10 | ~3GB | ~80 |
| vulberta | ~5GB | ~150 |
| reid_baseline | ~4GB | ~100 |
| mrt-oast | ~5GB | ~150 |
| dnn_rvsm | ~2GB | ~60 |
| pytorch_examples | ~3GB | ~70 |

## 🌟 最佳实践

### 1. 环境隔离

每个项目使用独立的环境，避免依赖冲突：

```bash
# ✅ 好的做法
conda activate pytorch_resnet_cifar10
cd repos/pytorch_resnet_cifar10
./train.sh

# ❌ 不要在base环境中运行
conda activate base  # 不推荐
./train.sh
```

### 2. 定期更新环境文件

修改环境后重新导出：

```bash
# 安装新包后
conda install new_package

# 重新导出
conda env export -n environment_name --no-builds > environment/environment_name.yml
```

### 3. 版本锁定vs版本范围

- **生产环境**: 使用精确版本（pytorch=1.13.1）
- **开发环境**: 使用版本范围（pytorch>=1.13.0）

### 4. 使用mamba加速

```bash
# 安装mamba（conda的C++实现，更快）
conda install mamba -n base -c conda-forge

# 使用mamba创建环境
mamba env create -f pytorch_resnet_cifar10.yml
```

## 🔗 相关文档

- [Conda用户指南](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [PyTorch安装指南](https://pytorch.org/get-started/locally/)
- [项目主文档](../README.md)

## 📝 环境更新记录

| 日期 | 环境 | 变更 |
|------|------|------|
| 2025-11-05 | 所有环境 | 初始导出 |

## ❓ 常见问题

**Q: 为什么有这么多环境？**

A: 不同项目使用不同的深度学习框架和版本（PyTorch 1.x, 2.x, TensorFlow等），需要隔离避免冲突。

**Q: mutation_runner需要特殊环境吗？**

A: 不需要。mutation_runner.py只使用Python标准库，可以在任何Python 3.6+环境中运行。但推荐使用专门的环境保持整洁。

**Q: 可以使用venv代替conda吗？**

A: 可以，但conda更适合深度学习项目，因为它可以管理CUDA、cuDNN等系统级依赖。

**Q: 环境文件太大，可以精简吗？**

A: 可以。创建minimal版本的环境文件，只包含核心依赖，让conda自动解决其他依赖。

## 💡 提示

- 所有yml文件已使用 `--no-builds` 导出，提高跨平台兼容性
- 建议在新机器上先创建mutation_runner环境，再根据需要创建其他环境
- GPU环境需要预先安装NVIDIA驱动和CUDA toolkit
- 使用 `conda clean --all` 定期清理缓存节省空间
