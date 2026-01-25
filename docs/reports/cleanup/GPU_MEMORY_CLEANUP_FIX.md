# GPU内存清理机制改进方案

## 问题分析

### 根本原因发现

通过对失败实验的分析，发现**根本原因**是：

```
进程 1970647 (python, 用户w, 从11月25日运行至今)
GPU内存占用: 5540MiB (~5.41GiB)
状态: 孤儿进程（PPID=1, 被init收养）
```

**这是一个后台训练的残留进程**，导致所有后续实验的GPU内存不足。

### 失败模式分析

1. **子进程理论上应该释放GPU内存**，但实际存在问题：
   - CUDA上下文可能没有正确销毁
   - 后台进程没有被正确清理，变成孤儿进程
   - GPU内存碎片化

2. **功耗波动分析**：
   - 大部分模型功耗稳定 (CV < 2%)
   - **hrnet18功耗异常波动** (CV=16.2%)
   - densenet121也有轻微波动 (CV=7.8%)
   - **可能与GPU内存压力相关**

## 解决方案

### 1. 添加GPU内存清理机制

创建了 `mutation/gpu_cleanup.py` 脚本，执行多层次清理：

```python
# 清理策略
1. Python垃圾回收 (gc.collect())
2. PyTorch CUDA缓存清理 (torch.cuda.empty_cache())
3. CUDA操作同步 (torch.cuda.synchronize())
4. 重置内存统计 (torch.cuda.reset_peak_memory_stats())
5. 二次垃圾回收
```

### 2. 在关键位置调用清理

修改了 `mutation/runner.py`，在以下时机自动清理：

1. **实验重试前**：训练失败后、重试之前
2. **实验结束后**：每个实验完成后
3. **配置切换时**：不同模型配置之间

清理流程：
```
训练完成 → GPU清理(3秒等待) → 睡眠60秒 → 下一个实验
训练失败 → GPU清理(3秒等待) → 睡眠30秒 → 重试
```

### 3. 改进后台进程清理

现有的后台进程清理机制需要加强：
- 添加进程组完整性检查
- 添加孤儿进程检测和清理
- 在每次实验结束后验证所有GPU进程已清理

## 使用说明

### 清理当前残留进程

**在运行新实验前，必须先清理残留进程**：

```bash
# 查看当前GPU进程
nvidia-smi

# 如果发现进程1970647或其他残留进程
sudo kill -9 1970647

# 确认GPU内存已释放
nvidia-smi
```

### 运行测试验证清理机制

创建了测试配置 `settings/gpu_memory_cleanup_test.json`：

```bash
# 运行测试（约30-60分钟）
sudo -E python3 mutation.py -ec settings/gpu_memory_cleanup_test.json
```

测试流程：
1. VulBERTa (建立baseline)
2. mnist_ff (之前失败的模型)
3. VulBERTa (验证恢复)
4. PCB模型 (之前失败的模型，小batch size)
5. VulBERTa (最终验证)

每个实验运行2次，共10个实验。

### 观察清理过程

运行时会看到清理输出：

```
────────────────────────────────────────────────────────────────────────────────
🧹 Cleaning up GPU memory...
────────────────────────────────────────────────────────────────────────────────
[GPU Cleanup] Memory before: allocated=3.50GB, reserved=3.80GB
[GPU Cleanup] Running Python garbage collection...
[GPU Cleanup]   Collected 150 objects
[GPU Cleanup] Clearing PyTorch CUDA cache...
[GPU Cleanup] Synchronizing CUDA operations...
[GPU Cleanup] Resetting peak memory statistics...
[GPU Cleanup] Memory after: allocated=0.00GB, reserved=0.00GB
[GPU Cleanup] ✓ Freed: allocated=3.50GB, reserved=3.80GB
✓ GPU cleanup completed
────────────────────────────────────────────────────────────────────────────────

Waiting 3 seconds for GPU to stabilize...
```

## 限制和注意事项

### 清理机制的限制

1. **无法清理子进程内存**：
   - 训练运行在独立子进程中
   - Python主进程的清理无法影响子进程
   - 依赖操作系统在子进程退出时释放GPU内存

2. **无法解决所有内存泄漏**：
   - 如果训练代码本身有内存泄漏，清理无法完全解决
   - 孤儿进程需要手动清理

3. **CUDA上下文残留**：
   - 某些情况下CUDA上下文可能不会完全释放
   - 可能需要更激进的清理（如nvidia-smi --gpu-reset，需要sudo）

### 建议的最佳实践

1. **长时间运行建议**：
   - 每运行50-80个实验后重启系统
   - 或者手动检查GPU内存状态

2. **监控GPU状态**：
   ```bash
   # 定期检查GPU内存
   watch -n 5 nvidia-smi
   ```

3. **检查孤儿进程**：
   ```bash
   # 查找可能的残留Python进程
   ps aux | grep python | grep -v grep

   # 查看GPU进程
   nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
   ```

## 预期效果

### 成功指标

如果清理机制有效，应该看到：

1. ✅ **mnist_ff实验成功完成**（之前100%失败）
2. ✅ **PCB实验成功完成**（之前100%失败）
3. ✅ **GPU内存在实验间被释放**（nvidia-smi显示）
4. ✅ **功耗波动减小**（hrnet18 CV < 5%）

### 如果仍然失败

如果测试中仍出现OOM错误，可能需要：

1. **更激进的清理**：
   ```bash
   # 每个实验后重置GPU（需要sudo，会中断所有GPU进程）
   sudo nvidia-smi --gpu-reset
   ```

2. **减小batch size**：
   - mnist_ff: 将batch_size从50000降低到10000
   - PCB: 将batch_size从32降低到16或8

3. **添加PyTorch内存配置**：
   ```python
   # 在训练脚本中添加
   import os
   os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
   ```

## 更新日志

### 新增文件

- `mutation/gpu_cleanup.py` - GPU内存清理工具
- `settings/gpu_memory_cleanup_test.json` - 测试配置

### 修改文件

- `mutation/runner.py`:
  - 添加 `_cleanup_gpu_memory()` 方法
  - 在 `run_experiment()` 重试前添加清理
  - 在 `run_from_experiment_config()` 实验间添加清理
  - 添加 `GPU_CLEANUP_WAIT_SECONDS` 常量

## 下一步行动

1. **清理残留进程**（必须）：
   ```bash
   sudo kill -9 1970647
   nvidia-smi  # 确认GPU内存释放
   ```

2. **运行测试**：
   ```bash
   sudo -E python3 mutation.py -ec settings/gpu_memory_cleanup_test.json
   ```

3. **观察结果**：
   - 检查 `results/run_*/summary.csv`
   - 验证所有实验是否成功
   - 检查GPU内存清理日志

4. **如果测试成功**，可以重新运行完整的2x变异测试
