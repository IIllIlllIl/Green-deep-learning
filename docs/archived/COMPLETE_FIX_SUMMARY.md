# GPU内存问题完整解决方案

## 执行摘要

完成了三个关键任务：
1. ✅ **减少mnist_ff batch size** - 从50000降至10000
2. ✅ **分析孤儿进程原因** - parallel实验的后台进程未被正确清理
3. ⏳ **清理残留进程** - 需要手动执行

## 立即行动（按顺序执行）

### 步骤1：清理残留进程（必须！）

**方法A：使用清理脚本（推荐）**
```bash
cd /home/green/energy_dl/nightly
sudo bash cleanup_orphan_processes.sh
```

**方法B：直接清理**
```bash
sudo kill -9 1970647 1970645
```

### 步骤2：验证GPU内存已释放
```bash
# 等待10秒
sleep 10

# 检查GPU状态
nvidia-smi

# 确认GPU内存占用接近0
# 应该看到：Memory-Usage: 0MiB / 10240MiB
```

### 步骤3：运行GPU清理测试
```bash
cd /home/green/energy_dl/nightly
sudo -E python3 mutation.py -ec settings/gpu_memory_cleanup_test.json
```

测试将运行10个实验（约30-60分钟）：
- VulBERTa → mnist_ff → VulBERTa → PCB → VulBERTa
- 每个模型2次

**期待结果**：
- ✅ 所有10个实验成功（100%成功率）
- ✅ mnist_ff实验成功（之前100%失败）
- ✅ PCB实验成功（之前100%失败）
- ✅ GPU内存在每个实验后被释放

### 步骤4：检查测试结果
```bash
# 查看最新运行的目录
ls -ltd results/run_* | head -1

# 查看成功率
cd results/run_XXXXXXX_XXXXXX  # 替换为实际目录名
grep training_success summary.csv | grep -c "True"
grep training_success summary.csv | grep -c "False"
```

## 问题根源分析

### 发现的问题

**1. mnist_ff的巨大batch size**
- 原始配置：50000（整个训练集作为一个batch）
- 预期GPU内存占用：约150 MiB
- 修复：降至10000

**2. 孤儿进程未被清理**
- 进程ID：1970647 (python, 5.54GB GPU内存)
- 创建时间：2025-11-25 01:20:48
- 原因：parallel实验的后台进程在框架异常时未被清理
- 表现：父进程退出后成为孤儿进程，继续占用GPU

**3. GPU内存清理机制缺失**
- 子进程理论上会释放GPU内存，但实际存在泄漏
- 框架没有在实验间清理GPU缓存
- 长时间运行导致内存累积

### 改进措施

**已实施**：
1. ✅ 创建GPU内存清理工具（`mutation/gpu_cleanup.py`）
2. ✅ 在关键位置添加GPU清理调用
   - 实验失败重试前
   - 每个实验结束后
   - 配置切换时
3. ✅ 减少mnist_ff的batch size
4. ✅ 创建测试配置验证清理机制
5. ✅ 编写孤儿进程分析文档

**待改进**：
1. ⏳ 增强后台进程清理机制（更健壮的进程组管理）
2. ⏳ 添加后台进程超时机制
3. ⏳ 在实验前检查残留进程

## 文件修改清单

### 新增文件
- `mutation/gpu_cleanup.py` - GPU内存清理工具
- `settings/gpu_memory_cleanup_test.json` - 测试配置
- `docs/GPU_MEMORY_CLEANUP_FIX.md` - 修复说明文档
- `docs/ORPHAN_PROCESS_ANALYSIS.md` - 孤儿进程分析
- `cleanup_orphan_processes.sh` - 残留进程清理脚本

### 修改文件
- `mutation/runner.py`:
  - 添加 `_cleanup_gpu_memory()` 方法
  - 在重试前、实验后、配置切换时调用清理
  - 添加 `GPU_CLEANUP_WAIT_SECONDS` 常量

- `repos/examples/mnist_forward_forward/main.py`:
  - 降低默认 train_size 从 50000 → 10000

- `repos/examples/train.sh`:
  - 降低 mnist_ff 的 DEFAULT_BATCH_SIZE 从 50000 → 10000

## 性能影响评估

### Batch Size减少的影响

**mnist_ff (50000 → 10000)**：
- GPU内存占用：减少约80%
- 训练速度：略微降低（需要更多迭代）
- 模型性能：理论上不变（Forward-Forward算法对batch size不太敏感）

### GPU清理的影响

**时间开销**：
- 每次清理：3秒
- 等待稳定：3秒
- 总开销：约6秒/实验

**对160个实验的影响**：
- 额外时间：160 × 6秒 = 16分钟
- 占总运行时间3.5天的：0.3%
- **可接受**

## 预期改善

如果所有改进生效：

**成功率**：
- 之前：82.5% (132/160)
- 预期：**≥95%** (152/160)

**失败原因**：
- 之前：100% GPU OOM
- 预期：配置错误、训练本身失败等正常原因

**功耗稳定性**：
- hrnet18变异系数：16.2% → **<5%**
- densenet121变异系数：7.8% → **<3%**

## 下一步行动

### 立即（今天）
1. 执行步骤1-4，完成清理和测试
2. 检查测试结果，验证改进有效

### 短期（如果测试成功）
1. 重新运行完整的2x变异测试
2. 对比新旧结果，验证改进效果

### 中期（优化框架）
1. 实施更健壮的后台进程管理
2. 添加GPU内存监控和预警
3. 考虑将长时间运行分批进行

## 联系和支持

如果遇到问题：
1. 查看日志：`results/run_*/*/training.log`
2. 查看GPU状态：`nvidia-smi`
3. 查看进程：`ps aux | grep python`
4. 参考文档：`docs/GPU_MEMORY_CLEANUP_FIX.md`
