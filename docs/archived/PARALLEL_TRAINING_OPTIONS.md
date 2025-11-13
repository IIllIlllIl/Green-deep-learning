# 并行训练方案对比与推荐

## 文档信息

- **版本**: v1.0
- **日期**: 2025-11-12
- **状态**: 方案对比
- **作者**: Claude Code

---

## 核心需求回顾

1. ✅ 并行运行两个模型：变异模型A（前景）+ 背景模型B
2. ✅ A执行变异训练并完整监控，B使用默认参数仅作为GPU负载
3. ✅ B持续循环训练，直到A完成
4. ✅ 资源由OS/GPU自动调度
5. ✅ 完全向后兼容，不修改config和现有settings
6. ✅ 默认执行原有单模型训练

---

## 方案一览表

| 方案 | 实现方式 | 复杂度 | 向后兼容 | 推荐度 |
|------|---------|--------|---------|--------|
| **方案1** | subprocess.Popen后台进程 + Shell循环脚本 | ⭐⭐⭐⭐⭐ 简单 | ⭐⭐⭐⭐⭐ 完美 | ⭐⭐⭐⭐⭐ **强烈推荐** |
| **方案2** | Python threading多线程 | ⭐⭐⭐⭐☆ 中等 | ⭐⭐⭐⭐⭐ 完美 | ⭐⭐⭐⭐☆ 推荐 |
| **方案3** | Python multiprocessing多进程 | ⭐⭐⭐☆☆ 复杂 | ⭐⭐⭐⭐☆ 良好 | ⭐⭐⭐☆☆ 备选 |
| **方案4** | 修改run.sh支持后台模式 | ⭐⭐☆☆☆ 复杂 | ⭐⭐⭐☆☆ 一般 | ⭐⭐☆☆☆ 不推荐 |

---

## 方案1: subprocess.Popen + Shell循环脚本 ⭐ 推荐

### 核心思想

使用Python的`subprocess.Popen`启动一个独立的Shell脚本，该脚本包含无限循环逻辑，持续运行背景训练，直到被前景训练完成后终止。

### 架构图

```
mutation.py (主进程)
    │
    ├─ 启动 background_training.sh (独立进程)
    │    └─ while true; do train.sh; done
    │
    ├─ 运行 run_experiment() (前景训练)
    │    └─ 完整监控 + 能耗测量
    │
    └─ 终止 background_training.sh (killpg)
```

### 实现要点

#### 1. 生成后台训练脚本

```python
def _start_background_training(self, repo, model, hyperparams, experiment_id):
    """创建并启动背景训练Shell脚本"""

    # 构建训练命令
    cmd_args = self._build_training_args(repo, model, hyperparams)

    # 生成Shell脚本
    script_path = self.results_dir / f"background_training_{experiment_id}.sh"
    script_content = f"""#!/bin/bash
# 背景训练循环脚本
REPO_PATH="{self.project_root / repo_config['path']}"
TRAIN_SCRIPT="{repo_config['train_script']}"
TRAIN_ARGS="{cmd_args}"
LOG_DIR="{self.results_dir}/background_logs_{experiment_id}"

mkdir -p "$LOG_DIR"
cd "$REPO_PATH"

echo "[Background] Starting training loop at $(date)"

run_count=0
while true; do
    run_count=$((run_count + 1))
    echo "[Background] Run #$run_count starting at $(date)"

    # 运行训练，输出到独立日志
    $TRAIN_SCRIPT $TRAIN_ARGS > "$LOG_DIR/run_$run_count.log" 2>&1

    exit_code=$?
    echo "[Background] Run #$run_count finished with exit code $exit_code"

    # 短暂休眠避免过于频繁重启
    sleep 2
done
"""

    # 写入文件并设置可执行权限
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    # 启动后台进程（创建新进程组）
    process = subprocess.Popen(
        [str(script_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid  # 关键：创建新进程组
    )

    return process
```

#### 2. 协调并行训练

```python
def run_parallel_experiment(self, fg_repo, fg_model, fg_mutation,
                           bg_repo, bg_model, bg_hyperparams, max_retries=2):
    """运行并行实验"""

    experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{fg_repo}_{fg_model}_parallel"
    background_process = None

    try:
        # 1. 启动背景训练
        background_process = self._start_background_training(
            bg_repo, bg_model, bg_hyperparams, experiment_id
        )
        print(f"✓ Background training started (PID: {background_process.pid})")

        # 2. 等待背景进程启动
        time.sleep(5)

        # 3. 运行前景训练（正常监控）
        print(f"\n🚀 Starting foreground training...")
        foreground_result = self.run_experiment(
            fg_repo, fg_model, fg_mutation, max_retries
        )

        print(f"\n✅ Foreground training completed")

    finally:
        # 4. 确保停止背景训练
        if background_process and background_process.poll() is None:
            print(f"\n🛑 Stopping background training...")
            self._stop_background_training(background_process)

    # 5. 返回结果（仅前景）
    return {
        "experiment_id": experiment_id,
        "mode": "parallel",
        "foreground_result": foreground_result,
        "background_info": {
            "repo": bg_repo,
            "model": bg_model,
            "hyperparameters": bg_hyperparams,
            "note": "Background training served as GPU load only"
        }
    }
```

#### 3. 停止背景训练

```python
def _stop_background_training(self, process):
    """停止背景训练进程组"""
    try:
        # 向整个进程组发送SIGTERM
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        # 等待进程终止
        process.wait(timeout=10)
        print("✓ Background training stopped gracefully")

    except subprocess.TimeoutExpired:
        # 超时强制终止
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
        print("⚠️ Background training force killed")

    except ProcessLookupError:
        print("✓ Background training already stopped")
```

### 优点

1. ✅ **极其简单**：仅约160行代码
2. ✅ **完全隔离**：背景训练在独立进程中运行
3. ✅ **可靠清理**：进程组管理确保所有子进程被终止
4. ✅ **日志隔离**：背景日志不污染前景日志
5. ✅ **易于调试**：可以直接查看Shell脚本和日志
6. ✅ **资源自动调度**：OS/GPU自动分配资源

### 缺点

1. ⚠️ **需要创建临时脚本**：每次实验生成一个.sh文件
2. ⚠️ **跨平台限制**：依赖Unix/Linux的进程组概念

### 代码量估计

- 新增方法：3个（约160行）
- 修改现有方法：1个（约30行）
- 总计：**约190行**

### 配置文件示例

```json
{
  "mode": "parallel",
  "experiments": [
    {
      "foreground": {
        "repo": "pytorch_resnet_cifar10",
        "model": "resnet20",
        "mutate": ["learning_rate"]
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

---

## 方案2: Python threading多线程

### 核心思想

使用Python标准库的`threading`模块，在主进程中启动一个后台线程持续运行背景训练。

### 架构图

```
mutation.py (主进程)
    │
    ├─ Thread 1 (background_worker)
    │    └─ while not stop_event: train()
    │
    └─ Main Thread
         └─ run_experiment() → set stop_event
```

### 实现要点

```python
import threading
import queue

def _background_training_worker(self, repo, model, hyperparams,
                                stop_event, results_queue, experiment_id):
    """后台线程工作函数"""
    run_count = 0

    while not stop_event.is_set():
        run_count += 1
        print(f"[Background] Starting run #{run_count}")

        try:
            # 运行训练
            result = self._run_training_without_monitoring(
                repo, model, hyperparams,
                log_file=f"background_logs_{experiment_id}/run_{run_count}.log"
            )
            results_queue.put(result)

        except Exception as e:
            print(f"[Background] Run #{run_count} failed: {e}")

        # 检查是否应该停止
        if stop_event.wait(timeout=2):  # 2秒间隔
            break

def run_parallel_experiment(self, ...):
    """运行并行实验"""

    # 创建停止事件和结果队列
    stop_event = threading.Event()
    results_queue = queue.Queue()

    # 启动后台线程
    background_thread = threading.Thread(
        target=self._background_training_worker,
        args=(bg_repo, bg_model, bg_hyperparams,
              stop_event, results_queue, experiment_id),
        daemon=True
    )
    background_thread.start()

    # 运行前景训练
    foreground_result = self.run_experiment(fg_repo, fg_model, fg_mutation, max_retries)

    # 停止后台线程
    stop_event.set()
    background_thread.join(timeout=60)

    # 返回结果
    return {...}
```

### 优点

1. ✅ **标准库支持**：无需外部依赖
2. ✅ **共享内存**：线程间通信简单
3. ✅ **资源开销小**：比多进程轻量
4. ✅ **易于同步**：使用Event和Queue

### 缺点

1. ⚠️ **GIL限制**：Python全局解释器锁，可能影响性能
2. ⚠️ **需要新增训练函数**：`_run_training_without_monitoring()`
3. ⚠️ **复杂度略高**：需要处理线程同步

### 代码量估计

- 新增方法：4个（约220行）
- 修改现有方法：1个（约30行）
- 总计：**约250行**

---

## 方案3: Python multiprocessing多进程

### 核心思想

使用`multiprocessing`模块启动独立的Python进程运行背景训练。

### 架构图

```
mutation.py (主进程)
    │
    ├─ Process 1 (background_process)
    │    └─ while not stop_event: train()
    │
    └─ Main Process
         └─ run_experiment() → set stop_event
```

### 实现要点

```python
import multiprocessing

def run_parallel_experiment(self, ...):
    """运行并行实验"""

    # 创建进程间共享对象
    stop_event = multiprocessing.Event()
    results_queue = multiprocessing.Queue()

    # 启动后台进程
    background_process = multiprocessing.Process(
        target=self._background_training_worker,
        args=(bg_repo, bg_model, bg_hyperparams,
              stop_event, results_queue, experiment_id)
    )
    background_process.start()

    # 运行前景训练
    foreground_result = self.run_experiment(...)

    # 停止后台进程
    stop_event.set()
    background_process.join(timeout=60)
    background_process.terminate()

    return {...}
```

### 优点

1. ✅ **真正并行**：不受GIL限制
2. ✅ **进程隔离**：故障不会相互影响

### 缺点

1. ❌ **内存开销大**：每个进程独立内存空间
2. ❌ **进程间通信复杂**：需要序列化数据
3. ❌ **代码复杂**：需要处理进程同步和共享状态

### 代码量估计

- 新增方法：4个（约250行）
- 修改现有方法：1个（约30行）
- 总计：**约280行**

---

## 方案4: 修改run.sh支持后台模式

### 核心思想

修改`scripts/run.sh`，添加后台运行模式，通过参数控制是否在后台循环运行。

### 实现要点

```bash
# run.sh 新增参数
BACKGROUND_MODE=$5  # 新增参数

if [ "$BACKGROUND_MODE" = "background" ]; then
    # 后台循环模式
    while true; do
        $TRAIN_SCRIPT $TRAIN_ARGS
        sleep 2
    done
else
    # 正常模式
    $TRAIN_SCRIPT $TRAIN_ARGS
fi
```

### 优点

1. ✅ **复用现有脚本**：不需要生成临时文件

### 缺点

1. ❌ **修改核心脚本**：影响现有功能
2. ❌ **向后兼容风险**：可能破坏现有调用
3. ❌ **复杂度高**：需要修改多处代码
4. ❌ **测试成本高**：需要回归测试所有功能

### 代码量估计

- 修改run.sh：约50行
- 修改mutation.py：约200行
- 总计：**约250行**

---

## 方案对比详表

| 维度 | 方案1: Shell脚本 | 方案2: 线程 | 方案3: 多进程 | 方案4: 修改run.sh |
|------|---------------|----------|-----------|---------------|
| **实现复杂度** | ⭐⭐⭐⭐⭐ 非常简单 | ⭐⭐⭐⭐☆ 简单 | ⭐⭐⭐☆☆ 中等 | ⭐⭐☆☆☆ 复杂 |
| **代码量** | 190行 | 250行 | 280行 | 250行 |
| **向后兼容** | ⭐⭐⭐⭐⭐ 完美 | ⭐⭐⭐⭐⭐ 完美 | ⭐⭐⭐⭐☆ 良好 | ⭐⭐⭐☆☆ 一般 |
| **资源管理** | ⭐⭐⭐⭐⭐ 完全隔离 | ⭐⭐⭐⭐☆ 共享内存 | ⭐⭐⭐⭐⭐ 完全隔离 | ⭐⭐⭐☆☆ 依赖修改 |
| **进程清理** | ⭐⭐⭐⭐⭐ 进程组 | ⭐⭐⭐⭐☆ 线程join | ⭐⭐⭐⭐☆ 进程terminate | ⭐⭐⭐☆☆ 手动管理 |
| **日志隔离** | ⭐⭐⭐⭐⭐ 完美 | ⭐⭐⭐⭐☆ 需要处理 | ⭐⭐⭐⭐☆ 需要处理 | ⭐⭐⭐☆☆ 需要修改 |
| **调试难度** | ⭐⭐⭐⭐⭐ 很容易 | ⭐⭐⭐☆☆ 中等 | ⭐⭐☆☆☆ 较难 | ⭐⭐☆☆☆ 较难 |
| **跨平台** | ⭐⭐⭐☆☆ Linux/Mac | ⭐⭐⭐⭐⭐ 全平台 | ⭐⭐⭐⭐⭐ 全平台 | ⭐⭐⭐☆☆ Linux/Mac |
| **测试成本** | ⭐⭐⭐⭐⭐ 很低 | ⭐⭐⭐⭐☆ 低 | ⭐⭐⭐☆☆ 中等 | ⭐⭐☆☆☆ 高 |
| **维护成本** | ⭐⭐⭐⭐⭐ 很低 | ⭐⭐⭐⭐☆ 低 | ⭐⭐⭐☆☆ 中等 | ⭐⭐☆☆☆ 高 |

---

## 推荐决策

### 🥇 **首选方案：方案1 - subprocess.Popen + Shell脚本**

**理由**：
1. ✅ **最简单**：实现清晰，代码量最少
2. ✅ **最可靠**：进程组管理，清理彻底
3. ✅ **最易维护**：逻辑独立，不影响现有代码
4. ✅ **完全兼容**：不修改任何现有文件
5. ✅ **易于调试**：可以直接查看生成的Shell脚本

**适用场景**：
- ✅ Linux/Mac环境（项目当前环境）
- ✅ 不需要跨平台支持
- ✅ 追求简洁和可靠性

### 🥈 **备选方案：方案2 - Python threading**

**理由**：
1. ✅ 标准库支持，跨平台
2. ✅ 共享内存，通信简单
3. ⚠️ 但代码复杂度略高
4. ⚠️ GIL可能影响性能（实际影响不大）

**适用场景**：
- ✅ 需要跨平台支持（Windows）
- ✅ 需要更细粒度的控制
- ⚠️ 愿意接受略高的复杂度

### ❌ **不推荐方案**

- **方案3（多进程）**：复杂度高，收益不明显
- **方案4（修改run.sh）**：风险高，破坏现有功能

---

## 实现建议

### 推荐实施路径

**Phase 1: 快速原型（1-2小时）**
1. 实现方案1的核心功能
2. 创建简单测试配置
3. 验证基本功能

**Phase 2: 完善功能（2-3小时）**
1. 添加错误处理
2. 优化日志输出
3. 添加配置验证

**Phase 3: 文档和测试（1-2小时）**
1. 更新文档
2. 创建示例配置
3. 编写测试脚本

### 验收标准

1. ✅ 可以通过配置文件运行并行训练
2. ✅ 前景训练正常监控，数据准确
3. ✅ 背景训练持续运行，前景完成后停止
4. ✅ 进程清理干净，无僵尸进程
5. ✅ 原有功能完全不受影响

---

## 后续扩展

### 可选功能（Phase 3+）

1. **GPU内存限制**
   ```python
   os.environ['CUDA_VISIBLE_DEVICES'] = '0'
   # 在训练脚本中设置memory_fraction
   ```

2. **多个背景模型**
   ```json
   "background": [
       {"repo": "...", "model": "..."},
       {"repo": "...", "model": "..."}
   ]
   ```

3. **背景训练统计**
   - 记录背景训练完成的轮数
   - 统计背景训练总时间

4. **动态调整**
   - 根据GPU内存动态调整批次大小
   - 根据GPU负载动态调整背景训练强度

---

## 常见问题

### Q: 为什么推荐方案1而不是多线程？

**A**:
1. **简单性**：Shell脚本方案代码量更少，逻辑更清晰
2. **隔离性**：完全独立的进程，不会相互干扰
3. **可靠性**：进程组管理确保清理彻底
4. **调试友好**：可以直接查看Shell脚本和执行过程

### Q: 方案1在Windows上能用吗？

**A**: 不能。方案1依赖Unix/Linux的进程组概念。如果需要Windows支持，应该选择方案2（threading）。但项目当前环境是Linux，方案1完全适用。

### Q: 能耗数据如何处理？

**A**: 所有方案的能耗数据都是前景+背景的总和，无法精确分离（硬件限制）。在结果JSON中会明确标注：
```json
{
  "energy_metrics": {
    "attribution": "combined",
    "note": "Energy includes both foreground and background training"
  }
}
```

### Q: 如果背景训练崩溃怎么办？

**A**: Shell脚本中的while循环会自动重启背景训练，不影响前景训练。崩溃信息会记录在背景日志中。

---

## 决策建议

**强烈推荐使用方案1** - subprocess.Popen + Shell脚本

**理由总结**：
1. ⭐⭐⭐⭐⭐ 最简单
2. ⭐⭐⭐⭐⭐ 最可靠
3. ⭐⭐⭐⭐⭐ 最易维护
4. ⭐⭐⭐⭐⭐ 完全向后兼容
5. ⭐⭐⭐⭐⭐ 适合当前环境

**下一步行动**：
1. 确认方案选择
2. 开始实施Phase 1
3. 创建测试配置验证功能

---

**最后更新**: 2025-11-12
**详细设计**: [PARALLEL_TRAINING_DESIGN.md](PARALLEL_TRAINING_DESIGN.md)
