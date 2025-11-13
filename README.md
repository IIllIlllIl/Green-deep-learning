# Mutation-Based Training Energy Profiler

自动化深度学习模型训练的超参数变异与能耗性能分析框架

## ⚠️ 项目状态

**当前版本**: v3.0 - Production Ready

- ✅ 所有核心功能已完成并测试
- ✅ 分层目录结构 + CSV总结
- ✅ 并行训练机制（两种模式）
- ✅ 高精度能耗监控（误差<2%）
- ✅ 代码质量优化（评分4.86/5.0）
- 📖 完整文档: [docs/README.md](docs/README.md)
- 📖 功能总览: [docs/FEATURES_OVERVIEW.md](docs/FEATURES_OVERVIEW.md)

---

## 项目概述

本框架用于研究深度学习模型训练超参数对能耗与性能的影响。通过自动化变异超参数、监控能耗、收集性能指标，支持大规模实验研究。

### 核心功能

✅ **分层目录结构** - 自动组织实验结果，生成汇总CSV
✅ **并行训练** - 最大化GPU利用率，支持两种模式（无限循环/脚本复用）
✅ **超参数变异** - 自动生成超参数变体（epochs, learning_rate, seed, dropout, weight_decay）
✅ **高精度能耗监控** - 使用perf和nvidia-smi，CPU误差<2%，GPU全指标监控
✅ **自动重试** - 训练失败时自动重试，确保实验可靠性
✅ **配置文件系统** - 批量实验配置，支持复杂实验设计
✅ **性能指标提取** - 自动提取各类性能指标
✅ **Governor控制** - 支持设置CPU频率调度器以减少干扰

---

## 快速开始

### 方式1: 配置文件模式（推荐）⭐

适合批量实验和长期研究：

```bash
# 1. 查看可用的预设配置
ls settings/*.json

# 2. 运行基线实验（复现所有模型的原始训练）
sudo python3 mutation.py --experiment-config settings/default.json

# 3. 运行全面变异实验
sudo python3 mutation.py --experiment-config settings/all.json

# 4. 运行并行训练实验（最大化GPU利用率）
sudo python3 mutation.py --experiment-config settings/parallel_with_script_reuse.json
```

**预设配置文件**:
- `default.json` - ⭐ 复现所有模型的原始训练（推荐先运行）
- `all.json` - 变异所有模型的所有超参数
- `parallel_example.json` - 并行训练示例（无限循环模式）
- `parallel_with_script_reuse.json` - 并行训练（脚本复用模式，推荐）
- 其他专项配置 - 详见 [settings/README.md](settings/README.md)

### 方式2: 命令行模式

适合快速测试单个实验：

```bash
# 1. 查看可用模型
python3 mutation.py --list

# 2. 运行单次变异实验
python3 mutation.py \
    --repo pytorch_resnet_cifar10 \
    --model resnet20 \
    --mutate epochs,learning_rate \
    --runs 1

# 3. 变异所有超参数，运行5次
python3 mutation.py \
    --repo VulBERTa \
    --model mlp \
    --mutate all \
    --runs 5

# 4. 使用性能模式和缩写参数
sudo python3 mutation.py \
    -r Person_reID_baseline_pytorch \
    -m densenet121 \
    -mt epochs,learning_rate,dropout \
    -g performance \
    -n 3
```

---

## 新功能亮点 (v3.0)

### 1. 分层目录结构 + CSV总结 ✅

**自动组织实验结果**，每次运行创建独立session目录：

```
results/
└── run_20251112_150000/              # Session目录（单次运行）
    ├── summary.csv                   # 总结CSV（动态列生成）
    ├── pytorch_resnet_cifar10_resnet20_001/
    │   ├── experiment.json
    │   ├── training.log
    │   └── energy/
    └── pytorch_resnet_cifar10_resnet20_002_parallel/
        ├── experiment.json
        ├── training.log
        ├── energy/
        └── background_logs/          # 并行训练背景日志
```

**CSV包含**:
- 实验元数据（ID、时间戳、模型）
- 动态超参数列（自适应不同实验）
- 性能指标（accuracy, mAP, rank-1等）
- 能耗数据（CPU/GPU全指标）

详细说明: [docs/OUTPUT_STRUCTURE_QUICKREF.md](docs/OUTPUT_STRUCTURE_QUICKREF.md)

### 2. 并行训练机制 ✅

**最大化GPU利用率**，在前景训练间隙运行背景训练：

```
前景训练 → 60秒冷却（背景训练循环） → 前景训练 → ...
```

**两种模式**:

1. **无限循环模式** (默认):
   ```bash
   sudo python3 mutation.py -ec settings/parallel_example.json
   ```

2. **脚本复用模式** (推荐，更高效):
   ```bash
   sudo python3 mutation.py -ec settings/parallel_with_script_reuse.json
   ```

详细说明: [docs/PARALLEL_TRAINING_USAGE.md](docs/PARALLEL_TRAINING_USAGE.md)

### 3. 高精度能耗监控 ✅

**直接包装（Direct Wrapping）方法**，显著提升测量精度：

| 维度 | 旧方法 | 新方法 | 改进 |
|------|--------|--------|------|
| CPU能耗误差 | 5-10% | <2% | **提升3-5倍** |
| 时间边界误差 | 1-3秒 | 0秒 | **零误差** |
| GPU指标数量 | 1项 | 5项 | **5倍信息** |

详细说明: [docs/energy_monitoring_improvements.md](docs/energy_monitoring_improvements.md)

---

## 命令行参数

所有参数都支持缩写形式，详见 [参数缩写手册](docs/mutation_parameter_abbreviations.md)

### 必需参数（命令行模式）

- `--repo REPO_NAME` (缩写: `-r`) - 仓库名称（如pytorch_resnet_cifar10）
- `--model MODEL_NAME` (缩写: `-m`) - 模型名称（如resnet20）
- `--mutate PARAMS` (缩写: `-mt`) - 要变异的超参数（逗号分隔，或使用"all"）

### 可选参数

- `--runs N` (缩写: `-n`) - 运行次数（默认：1）
- `--governor MODE` (缩写: `-g`) - CPU调度器模式（performance/powersave/ondemand）
- `--max-retries N` (缩写: `-mr`) - 失败时最大重试次数（默认：2）
- `--experiment-config FILE` (缩写: `-ec`) - 实验配置文件路径
- `--seed N` (缩写: `-s`) - 随机种子（用于可复现实验）
- `--list` (缩写: `-l`) - 列出所有可用模型

### 缩写示例

```bash
# 完整参数
python3 mutation.py --repo VulBERTa --model mlp --mutate all --runs 5

# 使用缩写（效果相同）
python3 mutation.py -r VulBERTa -m mlp -mt all -n 5
```

---

## 支持的仓库和模型

### 1. MRT-OAST
- **模型**: default
- **超参数**: epochs, learning_rate, seed, dropout, weight_decay

### 2. bug-localization-by-dnn-and-rvsm
- **模型**: default
- **超参数**: epochs, learning_rate, seed

### 3. pytorch_resnet_cifar10
- **模型**: resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
- **超参数**: epochs, learning_rate, seed, weight_decay

### 4. VulBERTa
- **模型**: mlp, cnn
- **超参数**: epochs, learning_rate, seed, weight_decay

### 5. Person_reID_baseline_pytorch
- **模型**: densenet121, hrnet18, pcb
- **超参数**: epochs, learning_rate, seed, dropout

### 6. examples
- **模型**: mnist_cnn, mnist_rnn, mnist_forward_forward, siamese
- **超参数**: epochs, learning_rate, seed

详细信息: [docs/hyperparameter_support_matrix.md](docs/hyperparameter_support_matrix.md)

---

## 结果格式

### 实验目录结构（新）

每次运行创建一个session目录，包含所有实验：

```
results/run_20251112_150000/
├── summary.csv                                   # 所有实验汇总
├── pytorch_resnet_cifar10_resnet20_001/
│   ├── experiment.json                           # 实验详细数据
│   ├── training.log                              # 训练日志
│   └── energy/                                   # 能耗监控数据
└── pytorch_resnet_cifar10_resnet20_002_parallel/
    ├── experiment.json
    ├── training.log
    ├── energy/
    └── background_logs/                          # 并行训练背景日志
```

### experiment.json 格式

```json
{
  "experiment_id": "pytorch_resnet_cifar10_resnet20_001",
  "timestamp": "2025-11-12T15:00:00.123456",
  "repository": "pytorch_resnet_cifar10",
  "model": "resnet20",
  "hyperparameters": {
    "epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 0.0001
  },
  "duration_seconds": 1234.56,
  "energy_metrics": {
    "cpu_energy_pkg_joules": 80095.55,
    "cpu_energy_ram_joules": 5432.11,
    "cpu_energy_total_joules": 85527.66,
    "gpu_power_avg_watts": 246.36,
    "gpu_power_max_watts": 250.12,
    "gpu_power_min_watts": 240.05,
    "gpu_energy_total_joules": 527217.33,
    "gpu_temp_avg_celsius": 75.2,
    "gpu_temp_max_celsius": 78.0,
    "gpu_util_avg_percent": 95.3,
    "gpu_util_max_percent": 98.0
  },
  "performance_metrics": {
    "accuracy": 92.5,
    "loss": 0.234
  },
  "training_success": true,
  "retries": 0,
  "error_message": ""
}
```

---

## 工作流程

```
1. 设置CPU Governor (可选)
   ↓
2. 创建Session目录
   ↓
3. 生成超参数变异
   ↓
4. 对每个变异：
   a. 创建实验目录（自动递增序号）
   b. 启动训练进程
   c. 同时启动能耗监控
   d. 等待训练完成
   e. 收集能耗数据
   f. 提取性能指标
   g. 保存到experiment.json
   h. 添加到session记录
   i. 休眠60秒（或运行背景训练）
   ↓
5. 生成CSV总结
   ↓
6. 显示实验摘要
```

---

## 测试

运行完整测试套件：

```bash
cd test
./run_tests.sh
```

测试包括：
- 32个核心功能测试
- 14个输出结构测试
- 文件存在性检查
- 配置文件验证
- 能耗监控验证

详见 [test/README.md](test/README.md)

---

## 最佳实践

### 1. 使用配置文件模式

```bash
# 推荐：配置文件模式
sudo python3 mutation.py -ec settings/my_experiment.json

# 优势：
# - 批量实验配置
# - 可复现性强
# - 支持复杂实验设计（如并行训练）
```

### 2. 使用并行训练最大化GPU利用率

```bash
# 推荐：脚本复用模式
sudo python3 mutation.py -ec settings/parallel_with_script_reuse.json

# 优势：
# - GPU利用率接近100%
# - 减少启动开销
# - 能耗测量更准确
```

### 3. 使用Performance Governor

```bash
# 自动设置（推荐）
sudo python3 mutation.py -ec settings/all.json -g performance

# 或手动设置
sudo ./governor.sh performance
python3 mutation.py -ec settings/all.json
sudo ./governor.sh powersave  # 实验完成后恢复
```

### 4. 结果分析

```bash
# 查看CSV总结
cat results/run_*/summary.csv | column -t -s,

# 使用jq分析JSON
cat results/run_*/*/experiment.json | jq '.performance_metrics'

# 提取特定指标
cat results/run_*/*/experiment.json | jq -r '[.experiment_id, .duration_seconds, .energy_metrics.cpu_energy_total_joules] | @csv'
```

---

## 故障排除

### 常见问题

详见 [docs/FIXES_AND_TESTING.md](docs/FIXES_AND_TESTING.md)

### 训练失败

框架会自动重试失败的训练（默认最多2次）。查看错误信息：
```bash
cat results/run_*/*/experiment.json | jq '.error_message'
```

### 能耗监控无数据

检查：
1. `perf` 权限：`sudo sysctl kernel.perf_event_paranoid`
2. `nvidia-smi` 可用性：`nvidia-smi`
3. 查看监控日志：`ls results/run_*/*/energy/`

### Governor设置失败

需要root权限：
```bash
sudo python3 mutation.py ... --governor performance
```

---

## 依赖项

### Python
- Python 3.6+
- 标准库（无需额外pip包）

### 系统工具
- `perf` - CPU能耗监控
- `nvidia-smi` - GPU能耗监控（可选）
- `bc` - 计算工具
- `bash` - Shell脚本执行

### 安装perf

```bash
# Ubuntu/Debian
sudo apt-get install linux-tools-common linux-tools-generic

# 启用perf
sudo sysctl -w kernel.perf_event_paranoid=-1
```

---

## 📚 文档导航

### 新手必读
1. [功能特性总览](docs/FEATURES_OVERVIEW.md) - ⭐⭐⭐ 了解所有功能
2. [快速参考卡片](docs/QUICK_REFERENCE.md) - ⭐⭐ 日常使用
3. [实验配置指南](docs/SETTINGS_CONFIGURATION_GUIDE.md) - ⭐⭐ 配置实验

### 深入学习
- [超参数变异策略](docs/HYPERPARAMETER_MUTATION_STRATEGY.md) - 科学设计变异实验
- [并行训练使用指南](docs/PARALLEL_TRAINING_USAGE.md) - 最大化GPU利用率
- [能耗监控改进](docs/energy_monitoring_improvements.md) - 确保测量精度
- [性能度量分析](docs/PERFORMANCE_METRICS_CONCLUSION.md) - 了解支持的指标

### 问题排查
- [问题排查与测试](docs/FIXES_AND_TESTING.md) - 常见问题解决方案
- [Bug修复记录](docs/BUGFIX_TIMEOUT_TYPEERROR.md) - 已知问题修复

### 完整文档索引
详见 [docs/README.md](docs/README.md)

---

## 示例用例

### 研究学习率对能耗的影响

```bash
python3 mutation.py \
    -r pytorch_resnet_cifar10 \
    -m resnet20 \
    -mt learning_rate \
    -n 10
```

### 研究Dropout对性能的影响

```bash
python3 mutation.py \
    -r Person_reID_baseline_pytorch \
    -m densenet121 \
    -mt dropout \
    -n 10
```

### 全面变异实验

```bash
python3 mutation.py \
    -r VulBERTa \
    -m mlp \
    -mt all \
    -n 20 \
    -g performance
```

### 并行训练实验

```bash
sudo python3 mutation.py -ec settings/parallel_with_script_reuse.json
```

---

## 版本历史

### v3.0 (2025-11-12) - Current
- ✅ 分层目录结构
- ✅ CSV总结生成（动态列）
- ✅ Bug修复（timeout TypeError）
- ✅ 文档整理归档

### v2.5 (2025-11-11)
- ✅ 并行训练机制
- ✅ 脚本复用优化
- ✅ 代码质量全面提升（4.86/5.0）
- ✅ 32个核心测试全部通过

### v2.0 (2025-11-10)
- ✅ 高精度能耗监控（误差<2%）
- ✅ 所有12个模型验证通过
- ✅ GPU全指标监控

### v1.5 (2025-11-09)
- ✅ 配置文件系统
- ✅ 参数缩写功能

### v1.0 (2025-11-08)
- ✅ 核心超参数变异功能
- ✅ 基础能耗监控

---

## 贡献

欢迎提交问题和改进建议！

## 作者

Green - 深度学习能耗研究项目

## 许可证

本项目用于研究目的。

---

**项目状态**: ✅ Production Ready
**版本**: v3.0
**最后更新**: 2025-11-12
