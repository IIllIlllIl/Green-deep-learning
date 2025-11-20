# Mutation-Based Training Energy Profiler

自动化深度学习模型训练的超参数变异与能耗性能分析框架

**当前版本**: v4.3.0 (2025-11-19)
**状态**: ✅ Production Ready

---

## 项目概述

研究深度学习训练超参数对能耗和性能的影响。通过自动化变异超参数、监控能耗、收集性能指标,支持大规模实验研究。

### 核心功能

- ✅ **超参数变异** - 自动生成超参数变体（log-uniform/uniform分布）
- ✅ **能耗监控** - CPU (perf) + GPU (nvidia-smi),CPU误差<2%
- ✅ **并行训练** - 支持前台监控+后台负载的并行训练模式
- ✅ **完整元数据** - 并行实验记录完整前景+背景模型信息
- ✅ **离线训练** - 支持完全离线运行，避免网络依赖
- ✅ **快速验证** - 1-epoch配置，15-20分钟验证全部模型
- ✅ **结果组织** - 分层目录结构 + CSV汇总 + JSON详细数据
- ✅ **批量实验** - 配置文件支持复杂实验设计

---

## 快速开始

### 1. 列出可用模型

```bash
python3 mutation.py --list
```

### 2. 运行单个实验

```bash
# 基本用法
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs,learning_rate -n 3
```

### 3. 运行批量实验（推荐）

```bash
# 快速验证（15-20分钟，1 epoch）
HF_HUB_OFFLINE=1 python3 mutation.py -ec settings/11_models_quick_validation_1epoch.json

# 变异实验验证（1次变异，快速测试）
sudo -E python3 mutation.py -ec settings/mutation_validation_1x.json -g performance

# 完整变异实验（3次变异，完整数据）
sudo -E python3 mutation.py -ec settings/mutation_all_models_3x_dynamic.json -g performance

# 完整基线实验（9+小时）
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py -ec settings/11_models_sequential_and_parallel_training.json -g performance
```

---

## 支持的模型

| 仓库 | 模型 | 超参数 | Epochs范围 |
|------|------|--------|-----------|
| **pytorch_resnet_cifar10** | resnet20/32/44/56 | epochs, lr, weight_decay, seed | [100, 300] |
| **Person_reID_baseline_pytorch** | densenet121, hrnet18, pcb | epochs, lr, dropout, seed | [30, 90] |
| **VulBERTa** | mlp, cnn | epochs, lr, weight_decay, seed | [5, 20] |
| **MRT-OAST** | default | epochs, lr, dropout, weight_decay, seed | [5, 15] |
| **bug-localization** | default | max_iter, alpha, kfold, seed | - |
| **examples** | mnist, mnist_rnn, mnist_ff, siamese | epochs, lr, batch_size, seed | [5, 15] |

**详细范围**: [docs/MUTATION_RANGES_QUICK_REFERENCE.md](docs/MUTATION_RANGES_QUICK_REFERENCE.md)

---

## 结果输出

每次运行创建独立session,包含CSV汇总和详细JSON数据：

```
results/run_YYYYMMDD_HHMMSS/
├── summary.csv                    # 所有实验汇总
└── {repo}_{model}_{id}_parallel/  # 并行实验（或不带_parallel为顺序实验）
    ├── experiment.json            # 完整数据（超参数+性能+能耗）
    │                              # 并行实验包含foreground和background信息
    ├── training.log               # 前景训练日志
    ├── energy/                    # 能耗原始数据
    └── background_logs/           # 后台训练日志（仅并行实验）
```

**详细说明**: [docs/OUTPUT_STRUCTURE_QUICKREF.md](docs/OUTPUT_STRUCTURE_QUICKREF.md)

---

## 常用命令

```bash
# 列出所有模型
python3 mutation.py --list

# 单次训练
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs -n 3

# 快速验证（15-20分钟）
HF_HUB_OFFLINE=1 python3 mutation.py -ec settings/11_models_quick_validation_1epoch.json

# 变异实验（动态生成超参数）
sudo -E python3 mutation.py -ec settings/mutation_validation_1x.json -g performance

# 完整实验（9+小时）
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py -ec settings/11_models_sequential_and_parallel_training.json -g performance

# 查看结果
cat results/run_*/summary.csv | column -t -s,
```

**完整参数说明**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

---

## 系统要求

- **Python**: 3.6+ (仅标准库)
- **能耗监控**: `perf` (CPU), `nvidia-smi` (GPU)
- **权限**: 需要sudo以获取准确的CPU能量数据

```bash
# 安装perf
sudo apt-get install linux-tools-common linux-tools-generic
sudo sysctl -w kernel.perf_event_paranoid=-1
```

---

## 📚 文档

| 文档 | 说明 |
|-----|------|
| [超参数变异范围](docs/MUTATION_RANGES_QUICK_REFERENCE.md) | 变异范围速查 ⭐⭐⭐ |
| [快速参考](docs/QUICK_REFERENCE.md) | 命令速查表 ⭐⭐ |
| [实验配置指南](docs/SETTINGS_CONFIGURATION_GUIDE.md) | Settings JSON编写 ⭐⭐ |
| [11个模型概览](docs/11_MODELS_OVERVIEW.md) | 模型详细信息 ⭐⭐ |
| [并行训练使用](docs/PARALLEL_TRAINING_USAGE.md) | 并行训练配置 ⭐⭐ |
| [输出结构](docs/OUTPUT_STRUCTURE_QUICKREF.md) | 结果目录结构 ⭐ |
| [功能总览](docs/FEATURES_OVERVIEW.md) | 所有功能说明 ⭐⭐ |
| [完整文档索引](docs/README.md) | 所有文档列表 |

---

## Settings配置文件

### 可用配置（settings/目录）

| 配置文件 | 说明 | 预计时间 |
|---------|------|---------|
| `11_models_quick_validation_1epoch.json` | 快速验证（1 epoch） | 15-20分钟 |
| `mutation_validation_1x.json` | 变异验证（1次） | 按模型而定 |
| `mutation_all_models_3x_dynamic.json` | 完整变异（3次） | 较长时间 |
| `11_models_sequential_and_parallel_training.json` | 完整基线 | 9+小时 |
| `person_reid_dropout_boundary_test.json` | Dropout边界测试 | ~6.5小时 |

### 配置格式

**顺序训练**:
```json
{
  "repo": "examples",
  "model": "mnist",
  "mode": "mutation",
  "mutate": ["epochs", "learning_rate"]
}
```

**并行训练**:
```json
{
  "mode": "parallel",
  "foreground": {
    "repo": "examples",
    "model": "mnist",
    "mode": "mutation",
    "mutate": ["epochs"]
  },
  "background": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "densenet121",
    "hyperparameters": {"epochs": 60, "learning_rate": 0.05}
  }
}
```

**详细指南**: [docs/SETTINGS_CONFIGURATION_GUIDE.md](docs/SETTINGS_CONFIGURATION_GUIDE.md)

---

## 版本信息

**v4.3.0** (2025-11-19)
- ✅ 11个模型完整支持（基线+变异）
- ✅ 动态变异系统（log-uniform/uniform分布）
- ✅ 并行训练（前景+背景GPU同时利用）
- ✅ 离线训练（HF_HUB_OFFLINE=1）
- ✅ 高精度能耗监控（CPU误差<2%）
- ✅ Dropout范围优化：统一采用d-0.2到d+0.1策略
  - MRT-OAST: [0.0, 0.3] (d=0.2)
  - Person_reID: [0.3, 0.6] (d=0.5)
  - 基于边界测试验证，dropout=0.6性能变化<1%，dropout=0.7导致严重劣化(-3%~-8%)

**完整版本历史**: [docs/FEATURES_OVERVIEW.md](docs/FEATURES_OVERVIEW.md)

---

**维护者**: Green | **状态**: ✅ Production Ready | **更新**: 2025-11-19
