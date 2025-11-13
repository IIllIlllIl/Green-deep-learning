# 并行训练功能使用指南

## 概述

并行训练功能允许同时运行两个模型训练：
- **前景模型（Foreground）**: 完整监控的训练，包括能耗测量和性能指标
- **背景模型（Background）**: 持续循环训练，仅作为GPU负载，直到前景训练完成

## 实现方案

采用**方案1: subprocess.Popen + Shell脚本循环**

### 核心机制

```
mutation.py (主进程)
    ├─ 生成 background_training.sh (Shell脚本)
    │    └─ while true; do train.sh; done
    │
    ├─ 启动后台进程 (subprocess.Popen + os.setsid)
    │
    ├─ 运行前景训练 (正常监控 + 能耗测量)
    │
    └─ 终止后台进程 (os.killpg)
```

### 优势

1. ✅ 简单可靠：约200行代码实现
2. ✅ 完全隔离：前景和背景在独立进程中运行
3. ✅ 清理彻底：使用进程组管理，确保无僵尸进程
4. ✅ 向后兼容：完全不影响现有功能
5. ✅ 易于调试：可以直接查看生成的Shell脚本

---

## 配置文件格式

### 基本结构

```json
{
  "experiment_name": "parallel_experiment",
  "description": "Description of the parallel experiment",
  "mode": "parallel",
  "governor": "performance",
  "runs_per_config": 1,
  "max_retries": 2,
  "experiments": [
    {
      "repo": "placeholder",
      "model": "placeholder",
      "mode": "parallel",
      "foreground": {
        "repo": "pytorch_resnet_cifar10",
        "model": "resnet20",
        "mode": "mutation",
        "mutate": ["learning_rate"]
      },
      "background": {
        "repo": "VulBERTa",
        "model": "mlp",
        "hyperparameters": {
          "epochs": 1,
          "learning_rate": 0.001,
          "dropout": 0.2
        }
      }
    }
  ]
}
```

### 配置字段说明

#### 全局配置

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `experiment_name` | string | 是 | 实验名称 |
| `description` | string | 否 | 实验描述 |
| `mode` | string | 是 | 必须设置为 `"parallel"` |
| `governor` | string | 否 | CPU调度策略 (performance/powersave) |
| `runs_per_config` | integer | 否 | 每个配置运行次数 (默认: 1) |
| `max_retries` | integer | 否 | 最大重试次数 (默认: 2) |

#### 前景训练配置 (`foreground`)

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `repo` | string | 是 | 仓库名称 |
| `model` | string | 是 | 模型名称 |
| `mode` | string | 是 | `"mutation"` (变异) 或 `"default"` (固定参数) |
| `mutate` | array | 条件 | 变异参数列表 (mode=mutation时必需) |
| `hyperparameters` | object | 条件 | 固定超参数 (mode=default时必需) |

#### 背景训练配置 (`background`)

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `repo` | string | 是 | 仓库名称 |
| `model` | string | 是 | 模型名称 |
| `hyperparameters` | object | 是 | 训练超参数 (通常使用默认值) |

---

## 使用示例

### 示例1: 基础并行训练

**配置**: `settings/parallel_example.json`

```json
{
  "experiment_name": "parallel_training_example",
  "mode": "parallel",
  "governor": "performance",
  "runs_per_config": 2,
  "experiments": [
    {
      "repo": "placeholder",
      "model": "placeholder",
      "mode": "parallel",
      "foreground": {
        "repo": "pytorch_resnet_cifar10",
        "model": "resnet20",
        "mode": "mutation",
        "mutate": ["learning_rate"]
      },
      "background": {
        "repo": "VulBERTa",
        "model": "mlp",
        "hyperparameters": {
          "epochs": 1,
          "learning_rate": 0.001,
          "dropout": 0.2,
          "weight_decay": 0.0
        }
      }
    }
  ]
}
```

**运行命令**:
```bash
sudo python3 mutation.py -ec settings/parallel_example.json
```

**说明**:
- 前景: ResNet20 with mutated learning rate (完整监控)
- 背景: VulBERTa MLP (仅作为GPU负载)
- 运行2次变异实验

### 示例2: 真实工作负载

**配置**: `settings/parallel_densenet_reid.json`

```json
{
  "experiment_name": "parallel_densenet_reid",
  "mode": "parallel",
  "governor": "performance",
  "runs_per_config": 1,
  "experiments": [
    {
      "repo": "placeholder",
      "model": "placeholder",
      "mode": "parallel",
      "foreground": {
        "repo": "Person_reID_baseline_pytorch",
        "model": "densenet121",
        "mode": "default",
        "hyperparameters": {
          "epochs": 60,
          "learning_rate": 0.05,
          "dropout": 0.5
        }
      },
      "background": {
        "repo": "pytorch_resnet_cifar10",
        "model": "resnet20",
        "hyperparameters": {
          "epochs": 164,
          "learning_rate": 0.1
        }
      }
    }
  ]
}
```

**运行命令**:
```bash
sudo python3 mutation.py -ec settings/parallel_densenet_reid.json
```

**说明**:
- 前景: DenseNet121 with default hyperparameters (Person Re-ID任务)
- 背景: ResNet20 (CIFAR-10训练作为负载)
- 模拟真实GPU负载环境

---

## 输出结构

### 文件组织

```
results/
├── background_training_{experiment_id}.sh      # 生成的后台训练脚本
├── background_logs_{experiment_id}/            # 后台训练日志
│   ├── run_1.log
│   ├── run_2.log
│   └── ...
├── energy_{experiment_id}_attempt0/            # 前景能耗数据
│   ├── cpu_energy.txt
│   ├── gpu_power.csv
│   ├── gpu_temperature.csv
│   └── gpu_utilization.csv
├── training_{repo}_{model}_{timestamp}.log     # 前景训练日志
└── {experiment_id}.json                        # 实验结果JSON
```

### 结果JSON格式

```json
{
  "experiment_id": "20251112_123456_pytorch_resnet_cifar10_resnet20_parallel",
  "mode": "parallel",
  "foreground_result": {
    "experiment_id": "20251112_123456_pytorch_resnet_cifar10_resnet20",
    "success": true,
    "duration": 3600.5,
    "retries": 0
  },
  "background_info": {
    "repo": "VulBERTa",
    "model": "mlp",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 0.001,
      "dropout": 0.2
    },
    "note": "Background training served as GPU load only (not monitored)"
  }
}
```

**注意**: 能耗数据包含前景+背景总和，无法在硬件层面分离。

---

## 测试

### 运行单元测试

```bash
python3 test/test_parallel_training.py
```

### 测试覆盖

1. ✅ `test_build_training_args`: 训练参数构建
2. ✅ `test_background_script_generation`: 后台脚本生成
3. ✅ `test_background_process_termination`: 进程终止清理
4. ✅ `test_parallel_experiment_structure`: 实验结构验证
5. ✅ `test_parallel_config_validation`: 配置格式验证

### 预期输出

```
================================================================================
PARALLEL TRAINING TEST SUITE
================================================================================
...
================================================================================
TEST SUMMARY
================================================================================
Tests run: 5
Successes: 5
Failures: 0
Errors: 0

✅ All tests passed!
```

---

## 常见问题

### Q1: 如何验证后台训练正在运行？

**A**: 检查后台日志目录:
```bash
ls -la results/background_logs_{experiment_id}/
# 应该看到 run_1.log, run_2.log, ... 不断增加
```

### Q2: 如何手动停止后台训练？

**A**: 后台训练会在前景完成后自动停止。如需手动停止:
```bash
# 查找后台进程
ps aux | grep background_training

# 终止进程组 (替换PID)
sudo kill -TERM -<PID>
```

### Q3: 能耗数据是否包含背景训练？

**A**: 是的。能耗数据包含前景+背景的总和，无法在硬件层面分离。这是预期行为，目的是测量真实GPU负载下的能耗。

### Q4: 背景训练失败会影响前景训练吗？

**A**: 不会。后台训练脚本包含无限循环，即使某次训练失败也会自动重启。错误会记录在 `background_logs/` 中。

### Q5: 如何调整背景训练强度？

**A**: 修改背景模型的超参数，例如：
- 增加 `epochs`: 更长训练时间
- 增加 `batch_size`: 更高GPU利用率
- 选择更大模型: DenseNet121 > ResNet20

### Q6: 可以同时运行多个背景模型吗？

**A**: 当前版本仅支持一个背景模型。如需多个背景负载，可以：
1. 手动启动额外训练进程
2. 或考虑扩展实现 (future work)

---

## 技术细节

### 进程管理

1. **进程组创建**: `preexec_fn=os.setsid`
2. **优雅终止**: `os.killpg(pgid, SIGTERM)` + 10秒超时
3. **强制终止**: `os.killpg(pgid, SIGKILL)` (超时后)

### 错误处理

- ✅ 后台进程崩溃: 自动重启 (while循环)
- ✅ 前景训练失败: 正常重试机制
- ✅ 进程清理失败: 尝试SIGKILL
- ✅ 配置验证: 跳过无效实验并警告

### 向后兼容性

- ✅ 不修改 `config/models_config.json`
- ✅ 不影响现有 `settings/` 配置
- ✅ 默认行为不变 (单模型训练)
- ✅ 新增功能通过 `mode="parallel"` 启用

---

## 开发者信息

### 新增方法

**mutation.py**:
1. `_build_training_args()`: 构建训练参数字符串
2. `_start_background_training()`: 启动后台训练循环
3. `_stop_background_training()`: 停止后台进程组
4. `run_parallel_experiment()`: 协调并行实验

### 修改方法

**mutation.py**:
1. `run_from_experiment_config()`: 添加 `mode="parallel"` 处理

### 代码量统计

- 新增代码: ~235行
- 修改代码: ~60行
- 测试代码: ~240行
- 总计: **~535行**

---

## 参考文档

- [并行训练方案对比](docs/PARALLEL_TRAINING_OPTIONS.md): 4种方案详细对比
- [方案1设计文档](docs/PARALLEL_TRAINING_DESIGN.md): 技术实现细节
- [超参数范围v2](docs/HYPERPARAMETER_RANGES_V2.md): 变异范围说明

---

**最后更新**: 2025-11-12
**版本**: v1.0
**实现方案**: subprocess.Popen + Shell script (方案1)
