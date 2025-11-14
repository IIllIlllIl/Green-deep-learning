# Mutation-Based Training Energy Profiler

自动化深度学习模型训练的超参数变异与能耗性能分析框架

**当前版本**: v4.0.1 - Modular Architecture

---

## 项目概述

研究深度学习训练超参数对能耗和性能的影响。通过自动化变异超参数、监控能耗、收集性能指标，支持大规模实验研究。

### 核心功能

- ✅ **超参数变异** - 自动生成超参数变体（log-uniform/uniform分布）
- ✅ **能耗监控** - CPU (perf) + GPU (nvidia-smi)，CPU误差<2%
- ✅ **结果组织** - 分层目录结构 + CSV汇总
- ✅ **自动重试** - 训练失败时智能重试
- ✅ **批量实验** - 配置文件支持复杂实验设计

---

## 快速开始

### 1. 列出可用模型

```bash
python3 mutation.py --list
```

### 2. 运行单个实验

```bash
# 完整命令
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 --mutate epochs,learning_rate --runs 3

# 简写形式
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs,learning_rate -n 3
```

### 3. 运行批量实验（推荐）

```bash
# 使用预设配置文件
sudo python3 mutation.py -ec settings/default.json

# 可用配置：
# - default.json          复现所有模型原始训练
# - all.json              变异所有模型所有超参数
# - boundary_test_v2.json 超参数边界测试
```

### 4. 设置CPU Governor（可选）

```bash
sudo python3 mutation.py -ec settings/all.json -g performance
```

---

## 架构

### 模块化设计 (v4.0)

从1,851行单体文件重构为8个专注模块：

```
mutation/                        # 自包含核心包
├── models_config.json          # 模型配置
├── runner.py                   # 主编排器 (841行)
├── session.py                  # 会话管理
├── hyperparams.py              # 超参数变异
├── command_runner.py           # 进程执行
├── energy.py                   # 能耗/性能解析
├── utils.py                    # 工具函数
├── exceptions.py               # 异常层次
├── run.sh                      # 训练包装脚本
├── governor.sh                 # Governor控制
└── background_training_template.sh
```

### 测试套件

- **33个测试** (32 passed, 1 skipped)
- 单元测试: `tests/unit/` (25个)
- 功能测试: `tests/functional/` (8个)

```bash
# 运行所有测试
python3 -m unittest discover -s tests/unit
python3 tests/functional/test_refactoring.py
```

---

## 支持的模型

| 仓库 | 模型 | 超参数 |
|------|------|--------|
| **pytorch_resnet_cifar10** | resnet20/32/44/56/110/1202 | epochs, lr, seed, weight_decay |
| **Person_reID_baseline_pytorch** | densenet121, hrnet18, pcb | epochs, lr, seed, dropout |
| **VulBERTa** | mlp, cnn | epochs, lr, seed, weight_decay |
| **MRT-OAST** | default | epochs, lr, seed, dropout, weight_decay |
| **bug-localization-by-dnn-and-rvsm** | default | epochs, lr, seed |
| **examples** | mnist_cnn, mnist_rnn, siamese | epochs, lr, seed |

查看详情：`python3 mutation.py --list`

---

## 结果目录

每次运行创建独立session，自动递增实验编号：

```
results/
└── run_20251113_150000/              # Session目录
    ├── summary.csv                   # 汇总CSV（所有实验）
    ├── pytorch_resnet_cifar10_resnet20_001/
    │   ├── experiment.json           # 详细数据（JSON）
    │   ├── training.log              # 训练日志
    │   └── energy/                   # 能耗原始数据
    │       ├── cpu_energy.txt
    │       ├── gpu_power.csv
    │       └── gpu_temperature.csv
    └── pytorch_resnet_cifar10_resnet20_002/
        └── ...
```

### CSV汇总内容

- 实验元数据（ID、时间、repo、model）
- 超参数（动态列，适配不同实验）
- 性能指标（accuracy, mAP, rank-1等）
- 能耗数据（CPU/GPU全指标）

---

## 命令行参数

### 必需参数（CLI模式）

- `-r`, `--repo` - 仓库名
- `-m`, `--model` - 模型名
- `-mt`, `--mutate` - 超参数（逗号分隔或`all`）

### 常用选项

- `-n`, `--runs` - 运行次数（默认：1）
- `-g`, `--governor` - CPU调度器（performance/powersave/ondemand）
- `-ec`, `--experiment-config` - 配置文件路径
- `-s`, `--seed` - 随机种子
- `-l`, `--list` - 列出所有模型
- `-mr`, `--max-retries` - 最大重试次数（默认：2）

---

## 工作流程

```
1. 加载配置 → 2. 创建Session → 3. 生成变异
        ↓
4. 对每个实验：
   a. 创建实验目录
   b. 启动训练 + 能耗监控
   c. 等待完成
   d. 解析能耗/性能数据
   e. 保存结果
   f. 冷却60秒
        ↓
5. 生成CSV汇总 → 6. 显示摘要
```

---

## 依赖项

### Python
- Python 3.6+ (仅标准库，无需pip包)

### 系统工具
```bash
# 必需
sudo apt-get install linux-tools-common linux-tools-generic  # perf
sudo sysctl -w kernel.perf_event_paranoid=-1                 # 启用perf

# 可选（GPU监控）
nvidia-smi  # 通常随NVIDIA驱动安装
```

---

## 最佳实践

### 1. 使用配置文件进行批量实验

```bash
# ✅ 推荐
sudo python3 mutation.py -ec settings/default.json

# ❌ 不推荐（难以复现）
python3 mutation.py -r repo1 -m model1 -mt all -n 5
python3 mutation.py -r repo2 -m model2 -mt all -n 5
...
```

### 2. 设置Performance Governor减少干扰

```bash
sudo python3 mutation.py -ec settings/all.json -g performance
```

### 3. 使用随机种子确保可复现

```bash
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt all -n 5 -s 42
```

### 4. 分析结果

```bash
# 查看CSV
cat results/run_*/summary.csv | column -t -s,

# 使用jq分析JSON
cat results/run_*/*/experiment.json | jq '.performance_metrics'

# 提取能耗数据
cat results/run_*/*/experiment.json | jq -r '[.experiment_id, .energy_metrics.cpu_energy_total_joules] | @csv'
```

---

## 常见问题

### 1. 训练失败
查看错误信息：
```bash
cat results/run_*/*/experiment.json | jq '.error_message'
```
框架会自动重试（默认最多2次）

### 2. 能耗监控无数据
检查权限：
```bash
sudo sysctl kernel.perf_event_paranoid  # 应该为-1或0
nvidia-smi                               # GPU监控
```

### 3. Governor设置失败
需要sudo权限：
```bash
sudo python3 mutation.py ... -g performance
```

---

## 📚 文档

### 快速参考
- [功能特性总览](docs/FEATURES_OVERVIEW.md)
- [实验配置指南](docs/SETTINGS_CONFIGURATION_GUIDE.md)
- [快速参考卡片](docs/QUICK_REFERENCE.md)

### 深入学习
- [模块化架构说明](docs/REFACTORING_SUMMARY.md)
- [配置迁移记录](docs/CONFIG_MIGRATION.md)
- [超参数变异策略](docs/HYPERPARAMETER_MUTATION_STRATEGY.md)
- [能耗监控改进](docs/energy_monitoring_improvements.md)

### 完整文档索引
[docs/README.md](docs/README.md)

---

## 版本历史

### v4.0.1 (2025-11-13) - Current
- ✅ 配置文件移至mutation/包（自包含模块）
- ✅ 目录清理整合（tests统一结构）
- ✅ 文档完善

### v4.0 (2025-11-13)
- ✅ 模块化架构重构（8个模块）
- ✅ 完整测试套件（33个测试）
- ✅ 100%向后兼容
- ✅ 代码质量提升（-54.6%行数）

### v3.0 (2025-11-12)
- ✅ 分层目录结构 + CSV汇总

### v2.0 (2025-11-10)
- ✅ 高精度能耗监控（CPU误差<2%）

### v1.0 (2025-11-08)
- ✅ 核心超参数变异功能

---

**状态**: ✅ Production Ready
**版本**: v4.0.1
**更新**: 2025-11-13
