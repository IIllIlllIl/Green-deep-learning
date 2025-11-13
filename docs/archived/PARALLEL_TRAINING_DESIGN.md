# 并行训练模式设计方案

## 文档信息

- **版本**: v1.0
- **日期**: 2025-11-11
- **状态**: 设计方案
- **作者**: Claude Code

---

## 目录

1. [概述](#概述)
2. [需求分析](#需求分析)
3. [架构设计](#架构设计)
4. [实现细节](#实现细节)
5. [配置文件格式](#配置文件格式)
6. [技术要点](#技术要点)
7. [可行性评估](#可行性评估)
8. [实现计划](#实现计划)

---

## 概述

### 背景

当前的mutation.py支持单模型训练和能耗监控。为了研究在真实GPU负载环境下的能耗特性，需要支持**并行训练模式**：

- **前景模型A**：执行变异训练，需要完整的性能度量和能耗监控
- **背景模型B**：使用默认超参数持续训练，仅作为GPU负载，不需要监控和度量

### 目标

设计并实现一个简单、可靠的并行训练模式，满足以下要求：

1. ✅ 支持同时运行两个模型（A变异训练 + B背景负载）
2. ✅ B模型持续循环训练，直到A训练完成
3. ✅ 完全向后兼容，不影响现有单模型训练功能
4. ✅ 不修改config/models_config.json
5. ✅ 资源分配交给操作系统和GPU自动调度

---

## 需求分析

### 功能需求

| 需求 | 描述 | 优先级 |
|------|------|--------|
| **FR-1** | 前景模型A执行变异训练，记录完整的能耗和性能数据 | P0 |
| **FR-2** | 背景模型B使用默认超参数，持续循环训练 | P0 |
| **FR-3** | B训练在A训练期间持续运行，A完成后自动停止 | P0 |
| **FR-4** | 仅监控和度量前景模型A | P0 |
| **FR-5** | 支持通过配置文件指定并行训练参数 | P0 |
| **FR-6** | 向后兼容现有单模型训练模式 | P0 |

### 非功能需求

| 需求 | 描述 | 优先级 |
|------|------|--------|
| **NFR-1** | 代码改动量小于200行 | P1 |
| **NFR-2** | 不修改现有配置文件格式 | P0 |
| **NFR-3** | 资源分配交给OS，无需手动管理GPU/CPU | P1 |
| **NFR-4** | 进程清理可靠，不留僵尸进程 | P0 |
| **NFR-5** | 日志隔离，背景训练不污染主日志 | P1 |

### 约束条件

1. ⚠️ **能耗数据限制**：CPU和GPU能耗无法精确分离前景和背景训练，结果为总和
2. ⚠️ **GPU内存**：两个模型同时训练，需要GPU有足够内存
3. ⚠️ **Python版本**：使用标准库，兼容Python 3.6+

---

## 架构设计

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    MutationRunner                           │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  run_from_experiment_config()                         │ │
│  │  - 读取配置文件                                        │ │
│  │  - 判断模式：default/mutation/parallel                │ │
│  └───────────────────────────────────────────────────────┘ │
│                           │                                 │
│              ┌────────────┴────────────┐                   │
│              │ mode == "parallel"      │                   │
│              └──────────┬──────────────┘                   │
│                         ▼                                   │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  run_parallel_experiment()                            │ │
│  │  1. 启动背景训练进程 (_start_background_training)      │ │
│  │  2. 运行前景训练 (run_experiment)                     │ │
│  │  3. 停止背景训练进程 (_stop_background_training)      │ │
│  │  4. 返回前景训练结果                                   │ │
│  └───────────────────────────────────────────────────────┘ │
│         │                                       │            │
│         ▼                                       ▼            │
│  ┌──────────────┐                      ┌──────────────┐    │
│  │ Background   │                      │ Foreground   │    │
│  │ Process      │                      │ Training     │    │
│  │              │                      │              │    │
│  │ - 循环训练   │                      │ - 完整监控   │    │
│  │ - 无监控     │                      │ - 性能度量   │    │
│  │ - 自动重启   │                      │ - 能耗数据   │    │
│  └──────────────┘                      └──────────────┘    │
│         │                                       │            │
│         └───────────────┬───────────────────────┘            │
│                         ▼                                   │
│              ┌────────────────────┐                         │
│              │  GPU (OS调度)      │                         │
│              │  - 自动资源分配    │                         │
│              └────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. `run_parallel_experiment()`

**职责**：协调并行训练流程

**输入**：
- 前景模型配置（repo, model, mutation）
- 背景模型配置（repo, model, hyperparameters）
- 重试参数（max_retries）

**输出**：
```python
{
    "experiment_id": str,
    "mode": "parallel",
    "foreground_result": {
        "repository": str,
        "model": str,
        "hyperparameters": dict,
        "duration_seconds": float,
        "energy_metrics": dict,
        "performance_metrics": dict,
        "training_success": bool,
        "retries": int
    },
    "background_info": {
        "repo": str,
        "model": str,
        "hyperparameters": dict,
        "note": "Background training served as GPU load only, not monitored"
    }
}
```

#### 2. `_start_background_training()`

**职责**：启动背景训练循环进程

**实现方式**：
1. 生成shell脚本，包含无限训练循环
2. 使用`subprocess.Popen`启动后台进程
3. 使用`os.setsid()`创建新进程组，便于清理

**脚本逻辑**：
```bash
#!/bin/bash
# 无限循环
run_count=0
while true; do
    run_count=$((run_count + 1))
    echo "[Background] Starting run #$run_count"

    # 运行训练（输出到独立日志）
    $TRAIN_SCRIPT $TRAIN_ARGS > "$LOG_DIR/run_$run_count.log" 2>&1

    # 短暂休眠
    sleep 2
done
```

#### 3. `_stop_background_training()`

**职责**：停止背景训练进程及所有子进程

**实现方式**：
1. 向进程组发送`SIGTERM`信号
2. 等待10秒
3. 如果仍未终止，发送`SIGKILL`强制终止

---

## 实现细节

### 代码结构

#### 文件改动

| 文件 | 改动类型 | 改动量 |
|------|---------|--------|
| `mutation.py` | 新增方法 | ~160行 |
| `config/models_config.json` | 无改动 | 0行 |
| `settings/*.json` | 新增配置文件（可选） | ~50行 |

#### 新增方法列表

```python
# mutation.py

class MutationRunner:
    # ... 现有代码 ...

    def run_parallel_experiment(self, ...) -> Dict[str, Any]:
        """协调并行训练"""
        # ~60行

    def _start_background_training(self, ...) -> subprocess.Popen:
        """启动背景训练循环"""
        # ~70行

    def _stop_background_training(self, process: subprocess.Popen) -> None:
        """停止背景训练进程"""
        # ~25行

    # 修改现有方法
    def run_from_experiment_config(self, config_file: str) -> None:
        """添加parallel模式处理逻辑"""
        # 新增 ~30行
```

### 伪代码

#### run_parallel_experiment

```python
def run_parallel_experiment(self,
                           foreground_repo, foreground_model, foreground_mutation,
                           background_repo, background_model, background_hyperparams,
                           max_retries=2):
    """
    并行运行前景训练（监控）和背景训练（负载）
    """
    # 1. 生成实验ID
    experiment_id = generate_experiment_id(foreground_repo, foreground_model, "parallel")

    # 2. 初始化
    background_process = None

    try:
        # 3. 启动背景训练循环
        background_process = self._start_background_training(
            background_repo, background_model, background_hyperparams, experiment_id
        )
        print(f"✓ Background training started (PID: {background_process.pid})")

        # 4. 等待背景进程启动
        sleep(5)

        # 5. 运行前景训练（完整监控）
        foreground_result = self.run_experiment(
            foreground_repo, foreground_model, foreground_mutation, max_retries
        )

    finally:
        # 6. 停止背景训练
        if background_process and is_running(background_process):
            self._stop_background_training(background_process)

    # 7. 返回结果（仅前景）
    return {
        "experiment_id": experiment_id,
        "mode": "parallel",
        "foreground_result": foreground_result,
        "background_info": {
            "repo": background_repo,
            "model": background_model,
            "hyperparameters": background_hyperparams,
            "note": "Background training served as GPU load only"
        }
    }
```

#### _start_background_training

```python
def _start_background_training(self, repo, model, hyperparams, experiment_id):
    """
    创建并启动背景训练脚本
    """
    # 1. 构建训练命令参数
    cmd_args = build_command_args(repo, model, hyperparams)

    # 2. 生成shell脚本
    script_path = f"{results_dir}/background_training_{experiment_id}.sh"
    script_content = f"""#!/bin/bash
        REPO_PATH="{repo_path}"
        TRAIN_SCRIPT="{train_script}"
        TRAIN_ARGS="{cmd_args}"
        LOG_DIR="{results_dir}/background_logs_{experiment_id}"

        mkdir -p "$LOG_DIR"
        cd "$REPO_PATH"

        run_count=0
        while true; do
            run_count=$((run_count + 1))
            echo "[Background] Run #$run_count at $(date)"
            $TRAIN_SCRIPT $TRAIN_ARGS > "$LOG_DIR/run_$run_count.log" 2>&1
            sleep 2
        done
    """

    write_file(script_path, script_content)
    chmod(script_path, 0o755)

    # 3. 启动后台进程
    process = subprocess.Popen(
        [script_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid  # 创建新进程组
    )

    return process
```

#### _stop_background_training

```python
def _stop_background_training(self, process):
    """
    停止背景训练进程组
    """
    try:
        # 1. 发送SIGTERM终止信号
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        # 2. 等待进程终止
        process.wait(timeout=10)
        print("✓ Background training stopped gracefully")

    except subprocess.TimeoutExpired:
        # 3. 超时强制终止
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
        print("⚠️ Background training force killed")

    except ProcessLookupError:
        # 进程已结束
        print("✓ Background training already stopped")
```

---

## 配置文件格式

### settings/parallel_test.json

```json
{
  "experiment_name": "parallel_mutation_test",
  "description": "Parallel training: mutated ResNet20 + background DenseNet121",
  "mode": "parallel",
  "governor": "performance",
  "runs_per_config": 3,
  "max_retries": 2,
  "experiments": [
    {
      "foreground": {
        "repo": "pytorch_resnet_cifar10",
        "model": "resnet20",
        "mutate": ["epochs", "learning_rate"]
      },
      "background": {
        "repo": "Person_reID_baseline_pytorch",
        "model": "densenet121",
        "hyperparameters": {
          "epochs": 60,
          "learning_rate": 0.05,
          "dropout": 0.5
        }
      }
    }
  ]
}
```

### 配置说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `mode` | string | 是 | 必须为`"parallel"` |
| `foreground` | object | 是 | 前景模型配置 |
| `foreground.repo` | string | 是 | 仓库名 |
| `foreground.model` | string | 是 | 模型名 |
| `foreground.mutate` | array | 是 | 要变异的超参数列表 |
| `background` | object | 是 | 背景模型配置 |
| `background.repo` | string | 是 | 仓库名 |
| `background.model` | string | 是 | 模型名 |
| `background.hyperparameters` | object | 是 | 超参数字典（使用默认值） |

### 向后兼容

现有配置文件（`default.json`, `all.json`等）**完全不需要修改**：

```json
{
  "mode": "default",  // 或 "mutation"
  "experiments": [
    {
      "repo": "pytorch_resnet_cifar10",
      "model": "resnet20",
      "hyperparameters": {...}
    }
  ]
}
```

---

## 技术要点

### 1. 进程组管理

**问题**：如何确保背景训练脚本及其子进程都能被清理？

**解决方案**：使用`os.setsid()`创建新进程组

```python
process = subprocess.Popen(
    [script_path],
    preexec_fn=os.setsid  # 关键：创建新进程组
)

# 终止时杀死整个进程组
os.killpg(os.getpgid(process.pid), signal.SIGTERM)
```

**原理**：
- `os.setsid()`使子进程成为新会话的领导者
- 新会话中的所有进程共享相同的进程组ID（PGID）
- `os.killpg()`可以一次性终止整个进程组

### 2. 日志隔离

**问题**：背景训练的输出可能污染前景训练日志。

**解决方案**：重定向到独立目录

```bash
LOG_DIR="results/background_logs_{experiment_id}"
mkdir -p "$LOG_DIR"
$TRAIN_SCRIPT $TRAIN_ARGS > "$LOG_DIR/run_$run_count.log" 2>&1
```

**目录结构**：
```
results/
├── training_pytorch_resnet_cifar10_resnet20_20251111_180000_foreground.log
├── background_logs_20251111_180000_parallel_exp1/
│   ├── run_1.log
│   ├── run_2.log
│   └── run_3.log
├── energy_20251111_180000_parallel_exp1_foreground/
│   ├── cpu_energy.txt
│   └── gpu_power.csv
└── 20251111_180000_parallel_exp1.json
```

### 3. 能耗数据归因

**问题**：CPU和GPU能耗是整个系统级别的，无法精确分离两个训练进程。

**解决方案**：明确标注能耗为"总和"

```json
{
  "energy_metrics": {
    "cpu_energy_total_joules": 50000.0,
    "gpu_energy_total_joules": 120000.0,
    "attribution": "combined",
    "note": "Energy metrics include both foreground (monitored) and background (load) training. Cannot be separated at hardware level."
  }
}
```

### 4. 异常处理

**问题**：如果前景训练失败或被中断，背景进程可能变成僵尸进程。

**解决方案**：使用`try-finally`确保清理

```python
try:
    background_process = start_background(...)
    foreground_result = run_experiment(...)
finally:
    if background_process and background_process.poll() is None:
        stop_background(background_process)
```

### 5. 启动延迟

**问题**：背景进程可能需要时间初始化。

**解决方案**：启动后等待5秒

```python
background_process = start_background(...)
print(f"✓ Background training started (PID: {background_process.pid})")
time.sleep(5)  # 给予启动时间
foreground_result = run_experiment(...)
```

---

## 可行性评估

### 技术可行性：⭐⭐⭐⭐⭐

| 评估维度 | 分数 | 说明 |
|---------|------|------|
| **实现复杂度** | 5/5 | 使用标准库`subprocess`，逻辑清晰 |
| **资源管理** | 5/5 | 完全交给OS和GPU调度，无需手动干预 |
| **进程清理** | 5/5 | 进程组管理确保可靠清理 |
| **日志隔离** | 5/5 | 独立目录，避免污染 |
| **异常处理** | 5/5 | `try-finally`确保清理 |

### 向后兼容性：⭐⭐⭐⭐⭐

| 评估维度 | 分数 | 说明 |
|---------|------|------|
| **命令行模式** | 5/5 | 完全不变 |
| **配置文件模式** | 5/5 | 现有配置无需修改 |
| **config/models_config.json** | 5/5 | 无需修改 |
| **默认行为** | 5/5 | 仅在`mode="parallel"`时启用 |

### 能耗监控：⭐⭐⭐⭐☆

| 评估维度 | 分数 | 说明 |
|---------|------|------|
| **GPU能耗** | 4/5 | 可监控总和，标注为combined |
| **CPU能耗** | 4/5 | 可监控总和，标注为combined |
| **时间记录** | 5/5 | 精确记录前景训练时间 |
| **性能指标** | 5/5 | 仅监控前景训练 |

**扣分原因**：CPU/GPU能耗无法精确分离，但已明确标注。

### 资源需求：⭐⭐⭐☆☆

| 评估维度 | 分数 | 说明 |
|---------|------|------|
| **GPU内存** | 3/5 | 需要GPU有足够内存同时运行两个模型 |
| **CPU使用** | 5/5 | 自动调度 |
| **磁盘空间** | 5/5 | 背景日志占用较小 |

**注意事项**：
- 如果GPU内存不足，可能需要调整模型批次大小
- 建议在GPU内存≥8GB的环境中使用

### 总体可行性：⭐⭐⭐⭐⭐

**结论**：技术方案完全可行，建议实施。

---

## 实现计划

### Phase 1: 核心功能实现（优先级：P0）

**时间估计**：2-3小时

**任务列表**：

1. ✅ 在`mutation.py`中添加`run_parallel_experiment()`方法
2. ✅ 实现`_start_background_training()`方法
3. ✅ 实现`_stop_background_training()`方法
4. ✅ 修改`run_from_experiment_config()`支持parallel模式
5. ✅ 添加单元测试（可选）

**验收标准**：
- 可以通过配置文件运行并行训练
- 前景训练正常监控
- 背景训练持续运行，前景完成后停止
- 进程清理干净，无僵尸进程

### Phase 2: 文档和测试（优先级：P1）

**时间估计**：1-2小时

**任务列表**：

1. ✅ 更新`README.md`
2. ✅ 创建示例配置文件`settings/parallel_example.json`
3. ✅ 添加使用说明到文档
4. ✅ 创建测试脚本验证功能

**验收标准**：
- 文档完整，用户可以按照说明使用
- 示例配置可以运行

### Phase 3: 优化和扩展（优先级：P2）

**时间估计**：2-3小时

**任务列表**：

1. ⏸️ 添加GPU内存限制支持（可选）
2. ⏸️ 优化日志输出格式
3. ⏸️ 添加进度条显示（可选）
4. ⏸️ 支持多个背景模型（可选）

---

## 使用示例

### 命令行使用

```bash
# 运行并行训练
sudo python3 mutation.py --experiment-config settings/parallel_test.json

# 或使用缩写
sudo python3 mutation.py -ec settings/parallel_test.json
```

### 预期输出

```
================================================================================
🔬 PARALLEL EXPERIMENT: 20251111_180000_pytorch_resnet_cifar10_resnet20_parallel
   Foreground (monitored): pytorch_resnet_cifar10/resnet20
   Background (load only): Person_reID_baseline_pytorch/densenet121
================================================================================

✓ Background training started (PID: 12345)

🚀 Starting foreground training with full monitoring...
   Command: ./scripts/run.sh repos/pytorch_resnet_cifar10 ./train.sh ...
   Log: results/training_pytorch_resnet_cifar10_resnet20_20251111_180000_foreground.log
   Energy directory: results/energy_20251111_180000_parallel_exp1_foreground
   Timeout: 36000s (10.0h)

[训练输出...]

✓ Training finished in 3600.5s with exit code 0
   CPU Energy: 50000.00 J
   GPU Energy: 120000.00 J

✅ Foreground training completed

🛑 Stopping background training (PID: 12345)...
✓ Background training stopped gracefully

💾 Results saved to: results/20251111_180000_parallel_exp1.json
```

### 结果JSON示例

```json
{
  "experiment_id": "20251111_180000_pytorch_resnet_cifar10_resnet20_parallel",
  "timestamp": "2025-11-11T18:00:00.123456",
  "mode": "parallel",
  "foreground_result": {
    "repository": "pytorch_resnet_cifar10",
    "model": "resnet20",
    "hyperparameters": {
      "epochs": 150,
      "learning_rate": 0.05,
      "weight_decay": 0.0001
    },
    "duration_seconds": 3600.5,
    "energy_metrics": {
      "cpu_energy_pkg_joules": 35000.0,
      "cpu_energy_ram_joules": 2500.0,
      "cpu_energy_total_joules": 37500.0,
      "gpu_power_avg_watts": 230.5,
      "gpu_power_max_watts": 280.0,
      "gpu_power_min_watts": 180.0,
      "gpu_energy_total_joules": 120000.0,
      "gpu_temp_avg_celsius": 78.5,
      "gpu_temp_max_celsius": 82.0,
      "gpu_util_avg_percent": 85.3,
      "gpu_util_max_percent": 95.0,
      "attribution": "combined",
      "note": "Energy metrics include both foreground and background training"
    },
    "performance_metrics": {
      "test_accuracy": 91.5,
      "best_val_accuracy": 92.3
    },
    "training_success": true,
    "retries": 0,
    "error_message": "Training completed successfully"
  },
  "background_info": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "densenet121",
    "hyperparameters": {
      "epochs": 60,
      "learning_rate": 0.05,
      "dropout": 0.5
    },
    "note": "Background training served as GPU load only, not monitored"
  }
}
```

---

## 常见问题

### Q1: 为什么能耗数据无法精确分离？

**A**: CPU和GPU的能耗监控是硬件级别的：
- `perf stat`测量的是整个CPU package的能耗
- `nvidia-smi`测量的是整个GPU的功耗

这些都是系统级别的指标，无法区分具体是哪个进程消耗的。因此，并行训练的能耗数据是两个训练的**总和**。

**解决方案**：在结果中明确标注`"attribution": "combined"`。

### Q2: 如果GPU内存不足怎么办？

**A**: 有以下几种方案：

1. **降低批次大小**：在配置中调整batch_size
2. **使用更小的模型**：选择参数量较少的模型作为背景
3. **多GPU环境**：如果有多个GPU，可以手动指定（需要扩展实现）

### Q3: 背景训练会影响前景训练的性能吗？

**A**: 会有一定影响，这正是并行训练的目的：

- 模拟真实的多任务GPU环境
- 研究在有背景负载时的能耗特性
- GPU会自动调度资源给两个训练进程

**注意**：如果要对比，建议也运行单独的前景训练作为baseline。

### Q4: 可以使用三个或更多模型并行吗？

**A**: 当前设计仅支持两个模型（1个前景 + 1个背景）。如果需要多个背景模型，可以在Phase 3中扩展实现。

### Q5: 背景训练的日志存在哪里？

**A**: 背景训练的日志存储在：
```
results/background_logs_{experiment_id}/
├── run_1.log
├── run_2.log
└── run_3.log
```

这些日志仅用于调试，不会被自动分析。

---

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| GPU内存不足 | 高 | 中 | 在文档中明确GPU内存要求，建议≥8GB |
| 进程清理失败 | 中 | 低 | 使用进程组管理 + `try-finally` |
| 能耗数据误读 | 中 | 中 | 在结果中明确标注为"combined" |
| 背景训练崩溃 | 低 | 低 | 自动重启循环，不影响前景训练 |
| 配置文件格式错误 | 低 | 低 | 添加配置验证逻辑 |

---

## 版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|---------|
| v1.0 | 2025-11-11 | Claude Code | 初始版本 |

---

## 参考资料

1. [Python subprocess 文档](https://docs.python.org/3/library/subprocess.html)
2. [Linux Process Groups](https://man7.org/linux/man-pages/man2/setpgid.2.html)
3. [nvidia-smi 文档](https://developer.nvidia.com/nvidia-system-management-interface)
4. [perf stat 文档](https://perf.wiki.kernel.org/index.php/Tutorial)

---

## 附录

### A. 完整的配置文件示例

见 `settings/parallel_example.json`

### B. 代码片段

见实现计划中的各个方法

### C. 测试用例

见 Phase 2 测试脚本
