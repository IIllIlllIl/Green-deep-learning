# Energy DL 项目环境配置

本项目使用 conda 和 venv 管理多个独立的 Python 环境。

## 快速开始

### 一键安装所有环境

```bash
cd environment
./install.sh
```

### 检查环境状态

```bash
./install.sh --check
```

---

## 环境列表

### 训练环境 (conda)

| 环境名 | Python | 对应仓库 | 核心依赖 |
|--------|--------|----------|----------|
| `dnn_rvsm` | 3.7 | bug-localization-by-dnn-and-rvsm | scikit-learn, nltk |
| `mrt-oast` | 3.7 | MRT-OAST | torch 1.13, javalang |
| `pytorch_resnet_cifar10` | 3.10 | pytorch_resnet_cifar10 | torch 2.5 |
| `reid_baseline` | 3.10 | Person_reID_baseline_pytorch | torch 2.5, timm |
| `vulberta` | 3.8 | VulBERTa | torch 1.8, transformers |

### 分析环境 (conda)

| 环境名 | Python | 用途 | 核心依赖 |
|--------|--------|------|----------|
| `causal-research` | 3.12 | 因果推断分析 | dibs-lib, jax, pandas |

### Venv 环境

| 环境名 | 位置 | 用途 |
|--------|------|------|
| `pytorch_examples` | `repos/examples/venv` | PyTorch 示例模型 |

---

## 安装脚本用法

```bash
# 安装所有环境
./install.sh

# 仅安装训练环境
./install.sh --training

# 仅安装分析环境
./install.sh --analysis

# 安装单个环境
./install.sh --env dnn_rvsm
./install.sh --env causal-research
./install.sh --env pytorch_examples

# 跳过已存在的环境
./install.sh --skip-existing

# 强制重新创建环境
./install.sh --force

# 检查环境状态
./install.sh --check

# 显示帮助
./install.sh --help
```

---

## 目录结构

```
environment/
├── install.sh              # 主安装脚本
├── README.md               # 本文件
├── conda/                  # Conda 环境配置 (精简版)
│   ├── dnn_rvsm.yml
│   ├── mrt-oast.yml
│   ├── pytorch_resnet_cifar10.yml
│   ├── reid_baseline.yml
│   ├── vulberta.yml
│   └── causal-research.yml
├── venv/                   # Venv 环境配置
│   └── examples/
│       └── requirements.txt
└── current/                # 当前完整版配置 (保留用于参考)
    ├── dnn_rvsm.yml
    ├── mrt-oast.yml
    └── ...
```

---

## 在其他机器上使用

### 前置要求

1. 安装 Miniconda 或 Anaconda
2. Python 3.10+ (用于 venv)

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository>
cd energy_dl/nightly

# 2. 运行安装脚本
cd environment
./install.sh

# 3. 验证安装
./install.sh --check
```

---

## 手动安装单个环境

如果自动安装失败，可以手动安装：

### Conda 环境

```bash
conda env create -f environment/conda/dnn_rvsm.yml
conda activate dnn_rvsm
```

### Venv 环境

```bash
python3 -m venv repos/examples/venv
source repos/examples/venv/bin/activate
pip install -r environment/venv/examples/requirements.txt
```

---

## 常见问题

### Q: conda 安装很慢？
A: 使用国内镜像源：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
```

### Q: GPU 相关的包安装失败？
A: 确保安装了正确的 NVIDIA 驱动和 CUDA 工具包。检查：
```bash
nvidia-smi
nvcc --version
```

### Q: venv 中的 torch 没有 GPU 支持？
A: venv 安装的是 CPU 版本。如需 GPU 支持，手动安装：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 更新环境

更新环境配置后，使用 `--force` 重新创建：

```bash
./install.sh --env dnn_rvsm --force
```

或手动更新：

```bash
conda activate dnn_rvsm
pip install -r requirements.txt
# 或
conda env update -f environment/conda/dnn_rvsm.yml --prune
```
