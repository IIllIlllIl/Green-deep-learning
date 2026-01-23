# 项目环境配置分析报告

**日期**: 2026-01-23
**目的**: 让项目可以在其他机器上运行，梳理所有conda环境

---

## 一、环境总览

### 1.1 已创建的Conda环境

| 环境名称 | Python版本 | 对应仓库/用途 | 配置文件位置 |
|---------|-----------|-------------|-------------|
| `dnn_rvsm` | 3.7 | bug-localization-by-dnn-and-rvsm | `environment/dnn_rvsm.yml` |
| `mrt-oast` | 3.7 | MRT-OAST | `environment/mrt-oast.yml` |
| `pytorch_resnet_cifar10` | 3.10 | pytorch_resnet_cifar10 | `environment/pytorch_resnet_cifar10.yml` |
| `reid_baseline` | 3.10 | Person_reID_baseline_pytorch | `environment/reid_baseline.yml` |
| `vulberta` | 3.8 | VulBERTa | `environment/vulberta.yml` |
| `pytorch_examples` | 3.10 | examples | `repos/examples/environment.yml` |
| `causal-research` | 3.12 | 分析模块(DiBS/DML) | `analysis/environment.yaml` |

### 1.2 不需要特殊环境的组件

- **mutation.py**: 仅使用Python标准库（json, subprocess, argparse, pathlib, re, time, datetime）
- **mutation_runner**: 虚拟环境名，不需要单独创建

---

## 二、��练脚本环境激活方式验证

### 2.1 各repo的train.sh环境设置

| 仓库 | train.sh中的环境设置 | 实际激活的方式 |
|------|---------------------|---------------|
| bug-localization-by-dnn-and-rvsm | `CONDA_ENV="dnn_rvsm"` | 硬编码环境路径 |
| MRT-OAST | `source /home/green/miniconda3/bin/activate mrt-oast` | conda activate |
| pytorch_resnet_cifar10 | `CONDA_ENV_PATH="/home/.../pytorch_resnet_cifar10"` | 硬编码环境路径 |
| Person_reID_baseline_pytorch | `CONDA_ENV="reid_baseline"` | 硬编码环境路径 |
| VulBERTa | `CONDA_ENV_NAME="vulberta"` | 硬编码环境路径 |
| examples | `find_python()`函数 | 查找pytorch_examples环境或venv |

### 2.2 调用链

```
mutation.py
    └── mutation/run.sh
            └── repos/<repo>/train.sh
                    └── 激活对应的conda环境
```

`mutation/run.sh` **不**激活conda环境，由各repo的train.sh自行处理。

---

## 三、环境依赖清单

### 3.1 dnn_rvsm (bug-localization-by-dnn-and-rvsm)

**Python版本**: 3.7.12

**核心依赖**:
- joblib==0.13.2
- nltk==3.4.3
- numpy==1.16.4
- scikit-learn==0.21.2
- scipy==1.5.4

**配置文件**: `environment/dnn_rvsm.yml`

---

### 3.2 mrt-oast (MRT-OAST)

**Python版本**: 3.7.12

**核心依赖**:
- torch==1.13.1+cu117
- torchvision==0.14.1+cu117
- torchaudio==0.13.1
- javalang==0.13.0
- matplotlib==3.5.3
- numpy==1.21.6
- tensorboard==2.11.2

**配置文件**: `environment/mrt-oast.yml`

---

### 3.3 pytorch_resnet_cifar10

**Python版本**: 3.10.19

**核心依赖**:
- torch 2.x (CUDA 12.1)
- torchvision
- numpy

**配置文件**: `environment/pytorch_resnet_cifar10.yml`

---

### 3.4 reid_baseline (Person_reID_baseline_pytorch)

**Python版本**: 3.10.19

**核心依赖**:
- torch==2.5.1+cu121
- torchvision==0.20.1+cu121
- timm==1.0.21
- efficientnet-pytorch==0.7.1
- pytorch-metric-learning==2.9.0
- pretrainedmodels==0.7.4

**配置文件**: `environment/reid_baseline.yml`

---

### 3.5 vulberta (VulBERTa)

**Python版本**: 3.8.20

**核心依赖**:
- torch==1.8.0+cu111
- torchvision==0.9.0
- torchaudio==0.8.0
- transformers==4.4.1
- tokenizers==0.10.1

**配置文件**: `environment/vulberta.yml`

---

### 3.6 pytorch_examples (examples)

**Python版本**: 3.10

**核心依赖**:
- pytorch>=2.6
- torchvision
- torchaudio

**配置文件**: `repos/examples/environment.yml`

---

### 3.7 causal-research (分析模块)

**Python版本**: 3.12.12

**核心依赖**:
- dibs-lib==1.2.0
- jax==0.4.25
- jaxlib==0.4.25
- torch==2.9.0
- pandas==2.3.3
- scikit-learn==1.7.2
- statsmodels==0.14.5
- matplotlib==3.10.6
- seaborn==0.13.2

**配置文件**: `analysis/environment.yaml`

---

## 四、依赖验证结果

### 4.1 配置文件完整性检查

| 环境 | 核心依赖完整性 | 说明 |
|------|---------------|------|
| dnn_rvsm | ✅ 完整 | 所有核心依赖都在配置中 |
| mrt-oast | ✅ 完整 | numpy/matplotlib为conda依赖，torch为pip依赖 |
| pytorch_resnet_cifar10 | ✅ 完整 | 所有核心依赖都在配置中 |
| reid_baseline | ✅ 完整 | 所有核心依赖都在配置中 |
| vulberta | ✅ 完整 | 所有核心依赖都在配置中 |
| pytorch_examples | ⚠️ 环境不存在 | 需要创建或复用其他环境 |
| causal-research | ✅ 完整 | 所有核心依赖都在配置中 |

### 4.2 pytorch_examples环境问题 ⚠️

**问题**: `pytorch_examples` conda环境不存在

**影响**:
- `repos/examples/train.sh`的`find_python()`函数会查找此环境
- 查找顺序：venv → pytorch_examples → python3

**解决方案**:
1. **方案A**: 创建`pytorch_examples`环境（推荐）
2. **方案B**: 修改`train.sh`复用`pytorch_resnet_cifar10`环境

### 4.3 各环境核心依赖详情

#### dnn_rvsm (bug-localization-by-dnn-and-rvsm)
```
核心依赖:
- joblib==0.13.2
- nltk==3.4.3
- numpy==1.16.4
- scikit-learn==0.21.2
- scipy==1.5.4
```

#### mrt-oast
```
核心依赖 (conda):
- numpy=1.21.6
- matplotlib=3.5.3

核心依赖 (pip):
- torch==1.13.1+cu117
- torchvision==0.14.1+cu117
- torchaudio==0.13.1+cu117
- javalang==0.13.0
- tensorboard==2.11.2
```

#### pytorch_resnet_cifar10
```
核心依赖 (pip):
- torch==2.5.1+cu121
- torchvision==0.20.1+cu121
- numpy==2.1.2
```

#### reid_baseline (Person_reID_baseline_pytorch)
```
核心依赖 (pip):
- torch==2.5.1+cu121
- torchvision==0.20.1+cu121
- timm==1.0.21
- efficientnet-pytorch==0.7.1
- pytorch-metric-learning==2.9.0
- pretrainedmodels==0.7.4
```

#### vulberta
```
核心依赖 (pip):
- torch==1.8.0+cu111
- torchvision==0.9.0+cu111
- torchaudio==0.8.0
- transformers==4.4.1
- tokenizers==0.10.1
```

#### pytorch_examples (待创建)
```
计划依赖 (repos/examples/environment.yml):
- python=3.10
- pytorch>=2.6
- torchvision
- torchaudio
- numpy
- matplotlib
- lmdb
- six
- tqdm
- spacy
```

#### causal-research (分析模块)
```
核心依赖 (pip):
- dibs-lib==1.2.0
- jax==0.4.25
- jaxlib==0.4.25
- torch==2.9.0
- pandas==2.3.3
- scikit-learn==1.7.2
- statsmodels==0.14.5
- matplotlib==3.10.6
- seaborn==0.13.2
```

---

## 五、问题与建议

### 5.1 需要解决的问题

1. **pytorch_examples环境缺失**
   - 状态: 环境不存在，但train.sh会尝试查找
   - 影响: examples模型训练可能回退到系统python
   - 建议: 创建`pytorch_examples`环境

2. **配置文件中的prefix路径**
   - 状态: 所有yml文件都包含硬编码的`prefix`路径
   - 影响: 在其他机器上创建环境时需要手动修改
   - 建议: 安装时使用`conda env create --prefix`指定路径

### 5.2 配置文件评估

**优点**:
- ✅ 所有核心依赖都已记录
- ✅ 配置文件与实际环境一致
- ✅ 包含了必要的传递依赖

**可改进**:
- ⚠️ 配置文件包含大量系统库依赖（可简化）
- ⚠️ prefix路径硬编码（可移除）

---

## 六、环境安装脚本

### 6.1 快速安装

已创建统一的环境安装脚本 `environment/install.sh`：

```bash
cd environment
./install.sh              # 安装所有环境
./install.sh --check      # 检查环境状态
```

### 6.2 新配置文件结构

```
environment/
├── install.sh           # 主安装脚本
├── README.md            # 使用说明
├── conda/               # 精简版 conda 配置
│   ├── dnn_rvsm.yml
│   ├── mrt-oast.yml
│   ├── pytorch_resnet_cifar10.yml
│   ├── reid_baseline.yml
│   ├── vulberta.yml
│   └── causal-research.yml
├── venv/                # venv 配置
│   └── examples/
│       └── requirements.txt
└── current/             # 完整版配置（参考用）
    └── *.yml
```

### 6.3 使用方法

| 命令 | 说明 |
|------|------|
| `./install.sh` | 安装所有环境 |
| `./install.sh --training` | 仅安装训练环境 |
| `./install.sh --analysis` | 仅安装���析环境 |
| `./install.sh --env <name>` | 安装单个环境 |
| `./install.sh --check` | 检查环境状态 |
| `./install.sh --skip-existing` | 跳过已存在的环境 |
| `./install.sh --force` | 强制重新创建 |

详细说明: `environment/README.md`

---

## 七、下一步行动
