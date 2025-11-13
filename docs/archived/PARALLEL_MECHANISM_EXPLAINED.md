# 并行训练机制详解

## 问题1: 如何实现并行？

### 1.1 核心原理

我们实现的是**进程级真并行**，而非线程级并行或伪并行。

```
时间轴:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│                                                                │
│  T0: 启动后台进程 (subprocess.Popen)                            │
│      ↓                                                         │
│      创建独立进程组 (os.setsid)                                 │
│      ↓                                                         │
│      执行 background_training.sh                               │
│                                                                │
│  T1: 等待 5 秒 (确保后台训练启动)                                │
│                                                                │
│  T2: 启动前景训练 (run_experiment)                              │
│      ┌────────────────────────────────────┐                   │
│      │ 前景训练 (GPU 0, 完整监控)          │                   │
│      │                                    │                   │
│      │   同时...                           │                   │
│      │                                    │                   │
│      │ 后台训练 (GPU 0, 无监控, 循环)      │                   │
│      └────────────────────────────────────┘                   │
│                                                                │
│  T3: 前景训练完成                                               │
│      ↓                                                         │
│      发送 SIGTERM 到后台进程组                                   │
│      ↓                                                         │
│      后台进程终止                                               │
│                                                                │
│  T4: 清理临时脚本                                               │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 1.2 进程隔离机制

#### **进程组 (Process Group)**

```python
# mutation.py:735-740
process = subprocess.Popen(
    [str(script_path)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    preexec_fn=os.setsid  # 关键！创建新进程组
)
```

**为什么使用 `os.setsid`？**

- `os.setsid()` 创建新的会话 (session) 和进程组
- 后台进程及其所有子进程都属于这个新进程组
- 终止时，一次性杀死整个进程组（包括所有子进程）
- 避免僵尸进程残留

**进程树示例**:
```
PID 1000: mutation.py (主进程)
    │
    ├─ PID 1001: bash background_training.sh (后台脚本, 进程组 1001)
    │       │
    │       ├─ PID 1002: python train.py (第1次训练)
    │       ├─ PID 1003: python train.py (第2次训练)
    │       └─ PID 1004: python train.py (第3次训练, 当前运行)
    │
    └─ PID 1100: bash run.sh (前景训练)
            ├─ PID 1101: python train.py (前景训练)
            ├─ PID 1102: perf stat (CPU能耗监控)
            └─ PID 1103: nvidia-smi loop (GPU能耗监控)
```

**终止时**:
```bash
# 杀死整个进程组 1001 (包括 1001, 1002, 1003, 1004)
os.killpg(1001, signal.SIGTERM)
```

### 1.3 Shell脚本循环机制

生成的脚本 `background_training_{experiment_id}.sh`:

```bash
#!/bin/bash
# Background training loop script for parallel experiments

REPO_PATH="/path/to/repo"
TRAIN_SCRIPT="train.py"
TRAIN_ARGS="--epochs 10 --lr 0.01"
LOG_DIR="results/background_logs_xxx"
RESTART_DELAY=2  # 从类常量传入，可配置

cd "$REPO_PATH" || exit 1

echo "[Background] Starting training loop at $(date)"

run_count=0
while true; do
    run_count=$((run_count + 1))
    echo "[Background] Run #$run_count starting at $(date)"

    # Run training, output to separate log file
    $TRAIN_SCRIPT $TRAIN_ARGS > "$LOG_DIR/run_$run_count.log" 2>&1

    exit_code=$?
    echo "[Background] Run #$run_count finished with exit code $exit_code at $(date)"

    # Brief sleep to avoid excessive restarts
    sleep $RESTART_DELAY
done
```

**关键特性**:
1. **无限循环**: `while true` 确保后台训练持续运行
2. **自动重启**: 每次训练完成后自动开始下一轮
3. **独立日志**: 每次运行输出到 `run_N.log`
4. **避免过载**: `sleep $RESTART_DELAY` 防止连续重启过快

### 1.4 为什么是真并行？

#### **两个独立的操作系统进程**

```
CPU/GPU 视角:

时刻 T:
  GPU 核心:
    ┌─────────────────────────────────────────┐
    │  前景训练任务 (CUDA kernels)              │
    │  + 后台训练任务 (CUDA kernels)            │
    │  = GPU 同时处理两个训练                   │
    └─────────────────────────────────────────┘

  CPU 核心:
    Core 0-3: 前景训练进程
    Core 4-7: 后台训练进程
    (由操作系统调度)
```

#### **非线程并行，而是进程并行**

| 对比项 | 线程并行 (Threading) | 进程并行 (我们的方案) |
|--------|---------------------|---------------------|
| GIL限制 | 受Python GIL限制 | 完全独立，无GIL限制 |
| 内存隔离 | 共享内存空间 | 完全隔离的内存空间 |
| GPU访问 | 可能冲突 | 操作系统调度，自动管理 |
| 崩溃影响 | 一个崩溃全崩溃 | 后台崩溃不影响前景 |
| 清理难度 | 简单 | 需要进程组管理 |

### 1.5 GPU 并行机制

**GPU 如何处理两个训练？**

```
NVIDIA GPU (CUDA):

┌────────────────────────────────────────────────────────┐
│  GPU 显存 (例如 12GB)                                   │
│                                                        │
│  前景训练: 4GB (模型 + 数据)                             │
│  后台训练: 4GB (模型 + 数据)                             │
│  剩余:     4GB (系统保留)                               │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  GPU 计算单元 (SM - Streaming Multiprocessor)           │
│                                                        │
│  前景训练: 占用部分 SM                                   │
│  后台训练: 占用部分 SM                                   │
│                                                        │
│  CUDA 驱动自动调度两个训练任务                           │
│  - 时间片轮转 (Time Slicing)                           │
│  - 或者 MPS (Multi-Process Service)                    │
└────────────────────────────────────────────────────────┘
```

**实际效果**:
- 两个训练同时运行，但速度都会降低（竞争GPU资源）
- GPU利用率接近100%（这正是我们想要的，模拟真实负载）

---

## 问题2: 能否得知背景模型B的训练是否覆盖了模型A的训练？

### 2.1 答案：**可以**，通过查看日志文件

背景训练的每一轮都记录在独立的日志文件中：

```bash
results/
├── background_logs_{experiment_id}/
│   ├── run_1.log      # 第1轮训练
│   ├── run_2.log      # 第2轮训练
│   ├── run_3.log      # 第3轮训练
│   └── ...
```

### 2.2 如何判断覆盖情况

#### **方法1: 比较时间戳**

```bash
# 查看前景训练日志的时间范围
grep "Starting" results/training_pytorch_resnet_cifar10_*.log
# 输出: 2025-11-12 10:00:00 - Starting training...

grep "Training completed" results/training_pytorch_resnet_cifar10_*.log
# 输出: 2025-11-12 12:30:00 - Training completed

# 查看后台训练的时间范围
ls -lt results/background_logs_*/
# 输出:
# run_1.log  (10:00:05 - 10:45:20)  ✓ 覆盖了前景训练开始阶段
# run_2.log  (10:45:22 - 11:30:15)  ✓ 覆盖了前景训练中期
# run_3.log  (11:30:17 - 12:15:30)  ✓ 覆盖了前景训练后期
# run_4.log  (12:15:32 - 12:30:10)  ✓ 覆盖了前景训练结束阶段
```

#### **方法2: 查看后台日志内容**

后台脚本在每次训练前后都会打印时间戳：

```bash
cat results/background_logs_xxx/run_1.log

# 输出类似:
[Background] Run #1 starting at Tue Nov 12 10:00:05 CST 2025
Epoch 1/10: loss=2.3, acc=0.1
Epoch 2/10: loss=2.1, acc=0.15
...
Epoch 10/10: loss=0.8, acc=0.72
[Background] Run #1 finished with exit code 0 at Tue Nov 12 10:45:20 CST 2025

[Background] Run #2 starting at Tue Nov 12 10:45:22 CST 2025
Epoch 1/10: loss=2.3, acc=0.1
...
```

### 2.3 编写检查脚本

创建一个脚本来自动检查覆盖情况：

```python
#!/usr/bin/env python3
"""检查后台训练是否覆盖了前景训练"""

import re
from pathlib import Path
from datetime import datetime

def parse_timestamp(log_line):
    """从日志行提取时间戳"""
    match = re.search(r'at (.+ CST \d{4})', log_line)
    if match:
        return datetime.strptime(match.group(1), '%a %b %d %H:%M:%S CST %Y')
    return None

def check_coverage(experiment_id):
    """检查后台训练是否覆盖前景训练"""

    # 1. 获取前景训练时间范围
    fg_log = list(Path("results").glob(f"training_*{experiment_id}*.log"))[0]
    with open(fg_log) as f:
        content = f.read()
        # 提取开始和结束时间（需要根据实际日志格式调整）
        fg_start = # ... 解析前景开始时间
        fg_end = # ... 解析前景结束时间

    # 2. 获取后台训练时间范围
    bg_log_dir = Path(f"results/background_logs_{experiment_id}")
    bg_runs = []

    for log_file in sorted(bg_log_dir.glob("run_*.log")):
        with open(log_file) as f:
            lines = f.readlines()
            start_line = [l for l in lines if "starting at" in l][0]
            end_line = [l for l in lines if "finished" in l][0]

            start_time = parse_timestamp(start_line)
            end_time = parse_timestamp(end_line)

            bg_runs.append({
                "run": log_file.stem,
                "start": start_time,
                "end": end_time
            })

    # 3. 检查覆盖情况
    print(f"前景训练: {fg_start} -> {fg_end}")
    print(f"持续时间: {(fg_end - fg_start).total_seconds():.0f} 秒\n")

    print("后台训练轮次:")
    for run in bg_runs:
        overlap = check_overlap(fg_start, fg_end, run['start'], run['end'])
        print(f"  {run['run']}: {run['start']} -> {run['end']} ({overlap})")

    # 4. 计算总覆盖率
    total_bg_time = sum((r['end'] - r['start']).total_seconds() for r in bg_runs)
    fg_duration = (fg_end - fg_start).total_seconds()

    coverage = (total_bg_time / fg_duration) * 100 if fg_duration > 0 else 0

    print(f"\n覆盖率: {coverage:.1f}%")
    if coverage >= 95:
        print("✅ 后台训练几乎完全覆盖了前景训练")
    elif coverage >= 80:
        print("⚠️  后台训练大部分覆盖了前景训练")
    else:
        print("❌ 后台训练覆盖不足")
```

### 2.4 预期结果

**理想情况**（后台训练完全覆盖前景训练）:

```
前景训练: 10:00:00 -> 12:30:00 (9000秒)

后台训练轮次:
  run_1: 10:00:05 -> 10:45:20 (2715秒) ✓
  run_2: 10:45:22 -> 11:30:15 (2693秒) ✓
  run_3: 11:30:17 -> 12:15:30 (2713秒) ✓
  run_4: 12:15:32 -> 12:30:10 (878秒)  ✓

后台总时间: 8999秒
覆盖率: 99.9%
✅ 后台训练几乎完全覆盖了前景训练
```

**不理想情况**（后台训练轮次太长，没有持续循环）:

```
前景训练: 10:00:00 -> 12:30:00 (9000秒)

后台训练轮次:
  run_1: 10:00:05 -> 12:00:00 (7195秒) ✓
  run_2: 12:00:02 -> 未完成 (被SIGTERM终止)

后台总时间: 7195秒
覆盖率: 79.9%
⚠️  后台训练大部分覆盖了前景训练

建议: 减少后台训练的 epochs，使每轮训练更短，循环更多次
```

---

## 问题3: 模型B是否能在启动模型训练之间的60s停止？

### 3.1 答案：**不能**，也**不应该**停止

#### **为什么不能停止？**

查看代码 `mutation.py:337-339`:

```python
# Sleep between runs
if run < runs_per_config:
    print(f"\n⏳ Sleeping {self.RUN_SLEEP_SECONDS} seconds to prevent energy interference...")
    time.sleep(self.RUN_SLEEP_SECONDS)
```

这个60秒的休眠发生在**主进程**中，而后台训练在**独立进程**中：

```
时间线:

前景训练1完成 -> 主进程休眠60s -> 前景训练2开始
                    │
                    └─> 后台训练仍在运行（独立进程不受影响）
```

#### **代码流程分析**:

```python
# mutation.py:323-339 (简化版)
for run, fg_mutation in enumerate(fg_mutations, 1):
    # 1. 启动后台训练（只在第一次run时启动）
    if run == 1:
        background_process, script_path = self._start_background_training(...)

    # 2. 运行前景训练
    result = self.run_parallel_experiment(...)

    # 3. 休眠60秒（主进程休眠）
    if run < runs_per_config:
        time.sleep(self.RUN_SLEEP_SECONDS)  # 60秒

    # 后台训练进程：持续运行，不受主进程休眠影响
```

**实际情况**:
```
T0:  启动后台训练 (while true 循环)
T1:  前景训练1 (1小时)
T2:  前景训练1完成
T3:  主进程休眠 60秒
     ↓
     后台训练继续运行！(独立进程)
     ↓
T4:  主进程唤醒
T5:  前景训练2 (1小时)
T6:  前景训练2完成
T7:  终止后台训练 (os.killpg)
```

### 3.2 为什么不应该停止？

#### **设计目的**

60秒休眠的目的是**避免前景训练之间的能耗干扰**，而不是让GPU空闲：

```
不正确的理解:
  训练1 -> [60s空闲] -> 训练2
  目的: 让GPU冷却

正确的理解:
  训练1 -> [60s + 后台训练持续] -> 训练2
  目的: 分隔前景训练的能耗数据，避免混淆
```

#### **如果停止后台训练会怎样？**

```
场景1: 后台训练在60s内停止

T0: 前景训练1完成
T1: 停止后台训练
T2: 休眠60秒
T3: 重启后台训练
T4: 前景训练2开始

问题:
❌ 前景训练2的能耗数据不包含启动阶段的后台负载
❌ GPU在休眠期间利用率下降，不符合真实场景
❌ 增加了启动/停止的开销
```

### 3.3 如何验证后台训练在60秒内仍在运行？

#### **方法1: 查看进程**

```bash
# 在60秒休眠期间执行
ps aux | grep background_training

# 输出应该显示进程仍在运行:
user  12345  ... python train.py ...
user  12300  ... bash background_training_xxx.sh
```

#### **方法2: 查看GPU使用**

```bash
# 在60秒休眠期间执行
watch -n 1 nvidia-smi

# 输出应该显示GPU仍在使用:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|   0  RTX 3090       Off  | 00000000:01:00.0 Off |                  N/A |
| 50%   72C    P2   250W / 350W |   8000MiB / 24576MiB |     85%      Default |
+-------------------------------+----------------------+----------------------+

# GPU利用率 85% -> 后台训练仍在运行 ✓
```

#### **方法3: 查看日志时间戳**

```bash
# 查看后台日志的连续性
tail -f results/background_logs_xxx/run_*.log

# 输出应该显示连续的训练：
[Background] Run #3 finished at 11:30:15
[Background] Run #4 starting at 11:30:17  <- 仅2秒间隔
[Background] Run #4 finished at 12:15:32
[Background] Run #5 starting at 12:15:34  <- 前景训练间隔60s，但后台仍在运行
[Background] Run #5 finished at 13:00:45
```

### 3.4 如果真的需要在60秒内停止后台训练怎么办？

**需要修改代码**（但不推荐）:

```python
# 修改 mutation.py run_from_experiment_config 方法

for run, fg_mutation in enumerate(fg_mutations, 1):
    # 启动后台训练
    background_process, script_path = self._start_background_training(...)

    # 运行前景训练
    result = self.run_parallel_experiment(...)

    # ⚠️ 停止后台训练
    self._stop_background_training(background_process, script_path)

    # 休眠60秒
    if run < runs_per_config:
        time.sleep(self.RUN_SLEEP_SECONDS)
```

**后果**:
- ❌ 每次都需要重新启动后台训练（增加开销）
- ❌ GPU在60秒休眠期间利用率降为0（不真实）
- ❌ 前景训练的能耗数据不一致（有的有后台负载，有的没有）

---

## 总结

### 问题1: 并行实现
✅ **操作系统级进程并行**
- 使用 `subprocess.Popen` + `os.setsid` 创建独立进程组
- 后台训练通过 Shell 脚本 `while true` 循环
- 前景和后台训练完全独立，互不影响
- GPU 由 CUDA 驱动自动调度两个训练任务

### 问题2: 覆盖检测
✅ **可以通过日志验证**
- 后台日志: `results/background_logs_{id}/run_N.log`
- 每个日志包含训练开始/结束时间戳
- 可以编写脚本自动计算覆盖率
- 预期覆盖率: >95%（几乎完全覆盖）

### 问题3: 60秒休眠
✅ **后台训练不会停止，也不应该停止**
- 60秒休眠只影响主进程（前景训练之间）
- 后台训练在独立进程中持续运行
- 这是符合设计目标的（模拟真实GPU负载）
- 可以通过 `ps` / `nvidia-smi` / 日志验证

---

**设计理念**: 后台训练的作用是提供**持续的GPU负载**，模拟真实生产环境中多任务并行的场景，从而获得更真实的能耗数据。

**验证方法**: 建议运行一次并行训练，然后检查：
1. 后台日志是否连续
2. GPU利用率是否在整个实验期间保持高位
3. 能耗数据是否明显高于单独训练
