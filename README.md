# Mutation-Based Training Energy Profiler

自动化深度学习模型训练的超参数变异与能耗性能分析框架

**当前版本**: v4.3.0 - Enhanced Parallel Experiments & Offline Training
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

# 完整实验（9+小时，使用sudo确保能量数据准确）
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

# 完整批量实验（9+小时，使用sudo确保能量数据准确）
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
| [超参数变异范围](docs/MUTATION_RANGES_QUICK_REFERENCE.md) | 各模型的超参数范围配置 ⭐⭐⭐ |
| [快速参考](docs/QUICK_REFERENCE.md) | 命令速查表 ⭐⭐ |
| [实验配置指南](docs/SETTINGS_CONFIGURATION_GUIDE.md) | 配置文件编写指南 ⭐⭐ |
| [并行训练使用](docs/PARALLEL_TRAINING_USAGE.md) | 并行训练模式说明 ⭐⭐ |
| [输出结构](docs/OUTPUT_STRUCTURE_QUICKREF.md) | 结果目录结构 ⭐ |
| [功能总览](docs/FEATURES_OVERVIEW.md) | 所有功能说明 ⭐⭐ |
| [更新日志 2025-11-18](docs/CHANGELOG_20251118.md) | v4.3.0 更新详情 🆕 |
| [完整索引](docs/README.md) | 所有文档列表 |

---

## 版本信息

**当前版本**: v4.3.0 (2025-11-18)
- ✅ 并行实验JSON增强 - 完整记录前景+背景模型信息
- ✅ 离线训练支持 - HF_HUB_OFFLINE=1 完全离线运行
- ✅ 快速验证配置 - 1-epoch版本，15-20分钟完成全模型测试
- ✅ 实验数据完整性 - 改进目录结构，确保数据不丢失

**主要里程碑**:
- v4.3.0: 并行实验元数据增强 + 离线训练 + 快速验证工具
- v4.2.0: Sequential和Parallel训练完整配置
- v4.1.0: 模型独立的超参数范围，完成范围测试和并行V3测试
- v4.0: 模块化重构，33个测试
- v3.0: 分层目录结构 + CSV汇总
- v2.0: 高精度能耗监控（误差<2%）

**完整版本历史**: [docs/FEATURES_OVERVIEW.md](docs/FEATURES_OVERVIEW.md) | **v4.3.0更新详情**: [docs/CHANGELOG_20251118.md](docs/CHANGELOG_20251118.md)

---

**维护者**: Green | **项目状态**: ✅ Production Ready | **最后更新**: 2025-11-18
