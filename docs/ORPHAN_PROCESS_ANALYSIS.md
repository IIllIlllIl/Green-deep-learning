# 孤儿进程分析报告

## 进程信息

### 当前状态
```
PID: 1970647
PPID: 1970645 (孤儿，被init收养)
用户: w (sudo运行)
启动时间: 2025-11-25 01:20:48
运行时长: 约1天14小时
命令行: python restarts_count 0
GPU内存: 5540MiB
状态: 运行中
```

### 父进程信息
```
PID: 1970645
PPID: 1 (init)
命令行: python
```

## 根本原因分析

### 1. 进程创建时间点

孤儿进程在 **2025-11-25 01:20:48** 创建，这个时间点：
- 在2x mutation run期间（2025-11-22 18:01 至 2025-11-26 06:27）
- 在hrnet18_130实验运行期间（00:57:31至约03:09:01）
- 距离最后一个parallel实验（densenet121_128_parallel）结束约4小时

### 2. 异常的命令行

进程命令行 `python restarts_count 0` 是异常的：
- 正常的后台训练进程应该是完整的Python命令
- "restarts_count 0"看起来像是参数解析错误
- 父进程命令行只是"python"，同样异常

### 3. 可能的原因

基于代码分析和时间线，**最可能的原因**是：

#### A. 后台训练进程清理失败

从`mutation/runner.py`的代码：

```python
# run_parallel_experiment() 方法
finally:
    # 4. Always stop background training, even if foreground failed
    if background_process and background_process.poll() is None:
        self.cmd_runner.stop_background_training(background_process, script_path)
        if background_process in self._active_background_processes:
            self._active_background_processes.remove(background_process)
```

**问题点**：
1. 如果主进程异常退出（如系统信号、内存错误），`finally`块可能不会执行
2. 如果后台进程已经fork了子进程，kill父进程可能不会清理所有子进程
3. 进程组清理可能失败（特别是在sudo下运行时）

#### B. 进程组管理问题

从`mutation/command_runner.py`的代码：

```python
if self._is_posix:
    # POSIX: Use process group for reliable cleanup
    popen_kwargs["preexec_fn"] = os.setsid
```

**问题点**：
1. 使用sudo运行时，用户权限从`green`切换到`w`（root权限）
2. 进程组属于`w`用户，而清理代码可能在`green`用户下运行
3. 权限不匹配导致进程组无法被完全清理

#### C. Background Training Template脚本的无限循环

从`mutation/background_training_template.sh`:

```bash
# Infinite training loop
while true; do
    run_count=$((run_count + 1))
    # ... training ...
    sleep "$RESTART_DELAY"
done
```

**问题点**：
1. 这是一个无限循环，依赖外部信号终止
2. 如果SIGTERM没有被正确传递，进程会继续运行
3. 脚本本身没有超时或自动退出机制

### 4. 为什么命令行异常？

命令行 `python restarts_count 0` 可能的解释：

1. **参数解析错误**：某个参数包含空格但没有正确引用
2. **cmdline截断**：/proc/PID/cmdline可能因为某些原因被截断
3. **Python子进程**：这可能是一个Python脚本fork的子进程，继承了奇怪的参数

## 验证和检查

### 检查其他parallel实验的后台进程

```bash
# 检查所有parallel实验的背景日志最后时间
for dir in results/run_20251122_175401/*_parallel/background_logs; do
    echo "=== $(basename $(dirname $dir)) ==="
    ls -lt "$dir" | head -3
done
```

### 检查是否还有其他残留进程

```bash
# 查找所有可能的残留Python进程
ps aux | grep -E "w.*python" | grep -v grep

# 查找所有GPU进程
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

## 改进建议

### 1. 增强后台进程清理

在`mutation/runner.py`中添加：

```python
def _cleanup_all_background_processes(self) -> None:
    """Terminate all tracked background processes with enhanced cleanup"""
    for proc in self._active_background_processes[:]:
        if proc.poll() is None:
            try:
                # 1. 首先尝试进程组终止
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)

            except subprocess.TimeoutExpired:
                # 2. 强制kill进程组
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

                # 3. 额外检查：查找所有相关子进程
                try:
                    result = subprocess.run(
                        ['pgrep', '-P', str(proc.pid)],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    child_pids = result.stdout.strip().split()
                    for child_pid in child_pids:
                        try:
                            os.kill(int(child_pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                except Exception:
                    pass
```

### 2. 添加后台进程超时机制

修改`background_training_template.sh`：

```bash
# Add maximum runtime
MAX_RUNS=${6:-999999}  # Optional 6th parameter
run_count=0

while [ $run_count -lt $MAX_RUNS ]; do
    run_count=$((run_count + 1))
    # ... training ...
done

echo "[Background] Maximum runs ($MAX_RUNS) reached, exiting"
```

### 3. 在每个实验前检查残留进程

在`run_experiment()`开始时添加：

```python
def run_experiment(self, ...):
    # Check for stray GPU processes
    self._check_and_cleanup_gpu_processes()
    # ... rest of experiment ...
```

### 4. 使用systemd或supervisor管理后台进程

考虑使用进程管理工具而不是直接fork：
- systemd temporary units
- supervisor
- process监控和自动清理

## 立即行动

1. **清理当前残留进程**（见任务3）
2. **应用上述改进建议**
3. **重新运行测试验证清理机制**
