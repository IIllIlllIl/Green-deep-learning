# Mutation-Based Training Energy Profiler

自动化深度学习模型训练的超参数变异与能耗性能分析框架

## 项目概述

本框架用于研究深度学习模型训练超参数对能耗与性能的影响。通过自动化变异超参数、监控能耗、收集性能指标，支持大规模实验研究。

### 核心功能

✅ **超参数变异** - 自动生成超参数变体（epochs, learning_rate, seed, dropout, weight_decay）
✅ **能耗监控** - 使用perf和nvidia-smi实时监控CPU/GPU能耗
✅ **自动重试** - 训练失败时自动重试，确保实验可靠性
✅ **结果收集** - 自动提取性能指标和能耗数据，保存为JSON
✅ **Governor控制** - 支持设置CPU频率调度器以减少干扰
✅ **防干扰休眠** - 训练之间自动休眠60秒，防止能耗干扰

## 项目结构

```
nightly/
├── mutation.py          # 主程序：协调整个实验流程
├── governor.sh                 # CPU频率调度器控制脚本
├── config/
│   └── models_config.json      # 模型配置：定义支持的超参数
├── scripts/
│   └── run.sh                   # 训练包装脚本（集成能耗监控）
├── test/                        # 测试目录
│   ├── run_tests.sh             # 测试运行脚本
│   ├── validate_energy_monitoring.sh  # 能耗监控验证脚本
│   └── README.md                # 测试文档
├── experiments/ → settings/     # 实验配置文件目录（新增）
│   ├── all.json                # 全面变异所有模型
│   ├── default.json            # 复现原始训练（基线）
│   ├── learning_rate_study.json # 学习率影响研究
│   └── README.md               # 配置文件使用说明
├── results/                    # 实验结果目录（JSON格式）
├── repos/                      # 模型仓库目录
│   ├── MRT-OAST/
│   ├── bug-localization-by-dnn-and-rvsm/
│   ├── pytorch_resnet_cifar10/
│   ├── VulBERTa/
│   ├── Person_reID_baseline_pytorch/
│   └── examples/
├── environment/                # Conda环境配置
├── test/                       # 测试环境
│   ├── run_tests.sh            # 测试运行脚本
│   └── README.md               # 测试文档
└── docs/                       # 项目文档
```

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
```

**预设配置文件**:
- `default.json` - ⭐ 复现所有模型的原始训练（推荐先运行）
- `all.json` - 变异所有模型的所有超参数
- `learning_rate_study.json` - 研究学习率影响
- `resnet_all_models.json` - ResNet家族实验

详见 [settings/README.md](settings/README.md)

### 方式2: 命令行模式

适合快速测试单个实验：

### 1. 查看可用模型

```bash
python3 mutation.py --list
```

输出示例：
```
📋 Available Repositories and Models:

  pytorch_resnet_cifar10:
    Models: resnet20, resnet32, resnet44, resnet56
    Supported hyperparameters: epochs, learning_rate, seed, weight_decay

  VulBERTa:
    Models: mlp, cnn
    Supported hyperparameters: epochs, learning_rate, seed, weight_decay

  ...
```

### 2. 运行单次变异实验

```bash
# 变异ResNet20的epochs和learning_rate
python3 mutation.py \
    --repo pytorch_resnet_cifar10 \
    --model resnet20 \
    --mutate epochs,learning_rate \
    --runs 1
```

### 3. 运行多次变异实验

```bash
# 变异所有支持的超参数，运行5次
python3 mutation.py \
    --repo VulBERTa \
    --model mlp \
    --mutate all \
    --runs 5
```

### 4. 使用性能模式运行

```bash
# 设置CPU为performance模式，减少干扰
sudo python3 mutation.py \
    --repo Person_reID_baseline_pytorch \
    --model densenet121 \
    --mutate epochs,learning_rate,dropout \
    --governor performance \
    --runs 3
```

## 命令行参数

### 必需参数

- `--repo REPO_NAME` - 仓库名称（如pytorch_resnet_cifar10）
- `--model MODEL_NAME` - 模型名称（如resnet20）
- `--mutate PARAMS` - 要变异的超参数（逗号分隔，或使用"all"）

### 可选参数

- `--runs N` - 运行次数（默认：1）
- `--governor MODE` - CPU调度器模式（performance/powersave/ondemand）
- `--max-retries N` - 失败时最大重试次数（默认：2）
- `--config PATH` - 配置文件路径（默认：config/models_config.json）
- `--list` - 列出所有可用模型
- `-h, --help` - 显示帮助信息

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

## 结果格式

每次实验生成一个JSON文件，包含完整的实验信息：

```json
{
  "experiment_id": "20251105_174723_test_repo_model_a",
  "timestamp": "2025-11-05T17:47:45.528255",
  "repository": "test_repo",
  "model": "model_a",
  "hyperparameters": {
    "epochs": 19,
    "learning_rate": 0.004356
  },
  "duration_seconds": 19.09,
  "energy_metrics": {
    "cpu_energy_pkg_joules": 406.32,
    "cpu_energy_ram_joules": 30.54,
    "cpu_energy_total_joules": 436.86,
    "gpu_power_avg_watts": 68.59,
    "gpu_power_max_watts": 68.85,
    "gpu_power_min_watts": 68.44,
    "gpu_energy_total_joules": 754.54,
    "gpu_temp_avg_celsius": 75.2,
    "gpu_temp_max_celsius": 78.0,
    "gpu_util_avg_percent": 95.3,
    "gpu_util_max_percent": 98.0
  },
  "performance_metrics": {
    "accuracy": 85.0,
    "loss": 0.6337
  },
  "training_success": true,
  "retries": 0,
  "error_message": ""
}
```

**新增能耗指标**（v2.0）：
- `gpu_temp_avg_celsius` / `gpu_temp_max_celsius` - GPU温度统计
- `gpu_util_avg_percent` / `gpu_util_max_percent` - GPU利用率统计

## 工作流程

```
1. 设置CPU Governor (可选)
   ↓
2. 生成超参数变异
   ↓
3. 对每个变异：
   a. 启动训练进程
   b. 同时启动能耗监控
   c. 等待训练完成
   d. 收集能耗数据
   e. 提取性能指标
   f. 检查训练成功性
   g. 失败则重试
   h. 保存结果到JSON
   i. 休眠60秒
   ↓
4. 生成实验总结
```

## 测试

运行完整测试套件：

```bash
cd test
./run_tests.sh
```

测试包括：
- 文件存在性检查
- 脚本可执行性检查
- 配置文件验证
- 模拟训练测试
- 能耗监控测试
- 完整集成测试

详见 [test/README.md](test/README.md)

## 能耗监控

### 监控方法改进（v2.0）⭐

本项目已采用**直接包装**（Direct Wrapping）的能耗监控方法，显著提升测量精度：

| 改进维度 | 旧方法 | 新方法（当前） | 精度提升 |
|---------|--------|--------------|---------|
| CPU能耗度量 | 间隔采样求和 | 直接包装命令 | **误差从5-10%降至<2%** |
| 时间边界 | ±2秒轮询误差 | 精确对齐 | **零边界误差** |
| GPU指标 | 仅功耗 | 功耗+温度+利用率 | **5项指标** |

**关键优势**：
- ✅ CPU能耗：使用 `perf stat` 直接包装训练命令，获取总能耗（无累积误差）
- ✅ GPU监控：一次查询获取功耗、温度、利用率，保证时间一致性
- ✅ 优雅停止：使用SIGTERM而非SIGKILL，避免数据丢失
- ✅ 进程精度：只监控目标进程树，无其他进程干扰

详细技术说明请查看：[docs/energy_monitoring_improvements.md](docs/energy_monitoring_improvements.md)

### 验证改进效果

运行验证脚本测试新方法的准确性：
```bash
./test/validate_energy_monitoring.sh
```

### CPU能耗监控

使用Linux `perf` 工具直接包装训练命令：
- **Package Energy** - CPU封装能耗
- **RAM Energy** - 内存能耗
- **监控方式** - 进程树级精确监控（包含所有子进程）

需要权限：
```bash
# 临时允许
sudo sysctl -w kernel.perf_event_paranoid=-1

# 永久设置
echo 'kernel.perf_event_paranoid=-1' | sudo tee -a /etc/sysctl.conf
```

### GPU能耗监控

使用 `nvidia-smi` 异步监控（每秒采样）：
- **Power Draw** - 实时功耗
- **GPU Temperature** - GPU核心温度
- **Memory Temperature** - 显存温度
- **GPU Utilization** - GPU利用率
- **Memory Utilization** - 显存利用率
- **统计数据** - 平均/最大/最小值

能耗数据保存位置：`results/energy_<experiment_id>/`
```
├── cpu_energy.txt              # CPU能耗总结
├── cpu_energy_raw.txt          # perf原始输出
├── gpu_power.csv               # GPU功耗时间序列
├── gpu_temperature.csv         # GPU温度时间序列
└── gpu_utilization.csv         # GPU利用率时间序列
```

## 配置文件

### models_config.json 结构

```json
{
  "models": {
    "repository_name": {
      "path": "repos/repository_name",
      "train_script": "./train.sh",
      "models": ["model1", "model2"],
      "supported_hyperparams": {
        "epochs": {
          "flag": "--epochs",
          "type": "int",
          "default": 10,
          "range": [5, 20]
        }
      },
      "model_flag": "-n",
      "performance_metrics": {
        "log_patterns": {
          "accuracy": "Accuracy[:\\s]+([0-9.]+)"
        }
      }
    }
  }
}
```

### 添加新模型

1. 在 `config/models_config.json` 中添加配置
2. 确保训练脚本支持命令行参数
3. 定义性能指标提取的正则表达式
4. 测试配置：`python3 mutation.py --list`

## 最佳实践

### 1. 使用Performance Governor

```bash
# 运行实验前设置
sudo ./governor.sh performance

# 实验完成后恢复
sudo ./governor.sh powersave
```

或使用 `--governor` 参数自动设置：
```bash
sudo python3 mutation.py ... --governor performance
```

### 2. 批量实验

```bash
# 示例：对多个模型运行实验
for model in resnet20 resnet32 resnet44; do
    python3 mutation.py \
        --repo pytorch_resnet_cifar10 \
        --model $model \
        --mutate all \
        --runs 5 \
        --governor performance
    sleep 300  # 额外休眠5分钟
done
```

### 3. 结果分析

```bash
# 查看所有结果
ls -lh results/*.json

# 使用jq分析结果
cat results/*.json | jq '.performance_metrics'

# 提取特定指标
cat results/*.json | jq -r '[.experiment_id, .duration_seconds, .energy_metrics.cpu_energy_total_joules] | @csv'
```

## 故障排除

### 训练失败

框架会自动重试失败的训练（默认最多2次）。查看错误信息：
```bash
cat results/<experiment_id>.json | jq '.error_message'
```

### 能耗监控无数据

检查：
1. `perf` 权限：`sudo sysctl kernel.perf_event_paranoid`
2. `nvidia-smi` 可用性：`nvidia-smi`
3. 查看监控日志：`ls results/energy_*/`

### Governor设置失败

需要root权限：
```bash
sudo python3 mutation.py ... --governor performance
```

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

## 示例用例

### 研究学习率对能耗的影响

```bash
python3 mutation.py \
    --repo pytorch_resnet_cifar10 \
    --model resnet20 \
    --mutate learning_rate \
    --runs 10
```

### 研究Dropout对性能的影响

```bash
python3 mutation.py \
    --repo Person_reID_baseline_pytorch \
    --model densenet121 \
    --mutate dropout \
    --runs 10
```

### 全面变异实验

```bash
python3 mutation.py \
    --repo VulBERTa \
    --model mlp \
    --mutate all \
    --runs 20 \
    --governor performance
```

## 贡献

欢迎提交问题和改进建议！

## 作者

Green - 深度学习能耗研究项目

## 许可证

本项目用于研究目的。
