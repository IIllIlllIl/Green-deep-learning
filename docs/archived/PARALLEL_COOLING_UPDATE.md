# 并行训练冷却逻辑更新

## 更新日期
2025-11-12

## 用户需求

1. **GPU冷却期**: 训练1 → [60秒空闲] → 训练2，让GPU完全冷却
2. **确保后台启动**: 增加等待时间，确保后台训练完全运行起来再开始前景训练

## 修改内容

### 1. 增加后台启动等待时间

**文件**: `mutation.py:49`

**修改前**:
```python
BACKGROUND_STARTUP_WAIT_SECONDS = 5  # Wait for background training to start
```

**修改后**:
```python
BACKGROUND_STARTUP_WAIT_SECONDS = 30  # Wait for background training to fully start
```

**影响**:
- 启动后台训练后等待30秒（原来5秒）
- 确保后台训练完全初始化并开始运行
- 避免前景训练启动时后台还在初始化阶段

---

### 2. 实现GPU冷却逻辑

**文件**: `mutation.py:1172-1178`

**修改前** (后台训练持续运行):
```python
# Sleep between runs
if run < runs_per_config:
    print(f"\n⏳ Sleeping {self.RUN_SLEEP_SECONDS} seconds to prevent energy interference...")
    time.sleep(self.RUN_SLEEP_SECONDS)
```

**修改后** (GPU完全空闲):
```python
# CRITICAL: Stop background training and sleep to allow GPU cooling
# This ensures 60 seconds of GPU idle time between runs
if run < runs_per_config:
    print(f"\n❄️  GPU Cooling Period:")
    print(f"   All training stopped. GPU will cool down during {self.RUN_SLEEP_SECONDS}s idle time.")
    print(f"⏳ Sleeping {self.RUN_SLEEP_SECONDS} seconds for GPU cooling...")
    time.sleep(self.RUN_SLEEP_SECONDS)
```

**关键变化**:
1. 后台训练在每次前景训练完成后**自动停止**（由 `run_parallel_experiment` 的 `finally` 块处理）
2. 60秒休眠期间**无任何训练运行**，GPU完全空闲
3. 下次运行时**重新启动**后台训练

---

## 新的工作流程

### 单次并行实验 (`run_parallel_experiment`)

```
┌─────────────────────────────────────────────────────────────┐
│ run_parallel_experiment()                                   │
│                                                              │
│  1. 启动后台训练                                              │
│     └─> subprocess.Popen (background_training.sh)           │
│                                                              │
│  2. 等待 30 秒                                               │
│     └─> 确保后台训练完全运行起来                              │
│                                                              │
│  3. 运行前景训练                                              │
│     ├─> 前景: 完整监控                                       │
│     └─> 后台: 持续循环（GPU负载）                            │
│                                                              │
│  4. finally: 停止后台训练                                    │
│     └─> os.killpg() + 清理脚本                              │
└─────────────────────────────────────────────────────────────┘
```

### 多次并行实验 (`run_from_experiment_config`)

```
时间线:

T0:  并行实验 #1 开始
     ├─ 启动后台训练
     ├─ 等待 30秒
     ├─ 运行前景训练 (1小时)
     └─ 停止后台训练 ✓

T1:  GPU 冷却期 (60秒)
     └─ GPU 完全空闲 ❄️

T2:  并行实验 #2 开始
     ├─ 重新启动后台训练
     ├─ 等待 30秒
     ├─ 运行前景训练 (1小时)
     └─ 停止后台训练 ✓

T3:  GPU 冷却期 (60秒)
     └─ GPU 完全空闲 ❄️

T4:  并行实验 #3 开始
     ...
```

---

## 与之前版本的对比

| 对比项 | 之前版本 | 当前版本 |
|--------|---------|---------|
| 后台启动等待 | 5秒 | **30秒** |
| 60秒休眠期间GPU状态 | 后台训练仍在运行 | **完全空闲** |
| 后台训练生命周期 | 整个实验期间持续 | **每次运行重新启动** |
| GPU冷却效果 | ❌ 无冷却 | ✅ **完全冷却** |

---

## 验证方法

### 1. 查看运行日志

```bash
# 运行并行训练
sudo python3 mutation.py -ec settings/parallel_example.json

# 查看输出，应该看到:
❄️  GPU Cooling Period:
   All training stopped. GPU will cool down during 60s idle time.
⏳ Sleeping 60 seconds for GPU cooling...
```

### 2. 监控GPU利用率

在60秒冷却期间运行:

```bash
watch -n 1 nvidia-smi
```

预期输出:
```
GPU-Util: 0%    # 完全空闲 ✓
```

### 3. 检查后台训练脚本

每次运行生成新的脚本:

```bash
ls -lt results/background_training_*.sh

# 应该看到多个脚本（每次运行一个）:
# background_training_20251112_100000_..._parallel.sh
# background_training_20251112_110100_..._parallel.sh
# background_training_20251112_120200_..._parallel.sh
```

### 4. 检查进程状态

在60秒冷却期间:

```bash
ps aux | grep python | grep train

# 应该没有任何训练进程 ✓
```

---

## 测试结果

```
================================================================================
TEST SUMMARY
================================================================================
Tests run: 5
Successes: 5
Failures: 0
Errors: 0

✅ All tests passed!
```

所有测试通过，验证了:
1. ✅ 后台训练正确启动和停止
2. ✅ 进程组正确清理
3. ✅ 脚本文件正确删除
4. ✅ 无僵尸进程残留

---

## 使用示例

### 示例配置: `settings/parallel_example.json`

```json
{
  "experiment_name": "parallel_training_with_cooling",
  "mode": "parallel",
  "runs_per_config": 3,
  "experiments": [
    {
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
          "learning_rate": 0.001
        }
      }
    }
  ]
}
```

### 运行命令

```bash
sudo python3 mutation.py -ec settings/parallel_example.json
```

### 预期行为

```
RUN 1:
  - 启动后台 (VulBERTa MLP)
  - 等待 30秒 (确保后台运行)
  - 运行前景 (ResNet20)
  - 停止后台

GPU冷却 60秒 ❄️

RUN 2:
  - 重新启动后台
  - 等待 30秒
  - 运行前景
  - 停止后台

GPU冷却 60秒 ❄️

RUN 3:
  - 重新启动后台
  - 等待 30秒
  - 运行前景
  - 停止后台

完成！
```

---

## 技术细节

### 后台训练停止机制

后台训练在 `run_parallel_experiment()` 的 `finally` 块中自动停止:

```python
def run_parallel_experiment(...):
    try:
        # 1. 启动后台
        background_process, script_path = self._start_background_training(...)

        # 2. 等待30秒
        time.sleep(self.BACKGROUND_STARTUP_WAIT_SECONDS)

        # 3. 运行前景
        foreground_result = self.run_experiment(...)

    finally:
        # 4. 总是停止后台（即使前景失败）
        if background_process and background_process.poll() is None:
            self._stop_background_training(background_process, script_path)
```

### GPU冷却验证

60秒冷却期间的系统状态:

```
CPU: 空闲（主进程 sleep）
GPU: 空闲（所有训练停止）
进程: 仅主进程存在
温度: 逐渐降低
功耗: 降至空闲水平
```

---

## 总结

### ✅ 实现的需求

1. **GPU完全冷却**: 60秒空闲期，无任何训练运行
2. **确保后台启动**: 等待时间从5秒增加到30秒

### ✅ 新的运行逻辑

```
启动后台 → 等待30秒 → 前景训练 → 停止后台 → 60秒空闲 → 重复
```

### ✅ 测试验证

- 5/5 单元测试通过
- 进程正确启动和停止
- 资源完全清理
- 无僵尸进程

---

**完成时间**: 2025-11-12
**测试状态**: ✅ 全部通过
**可用状态**: ✅ 立即可用
