# Mutation 2x Supplementary Experiments Configuration

**配置文件**: `settings/mutation_2x_supplement.json`
**创建时间**: 2025-11-26
**目的**: 补充 `results/summary_all.csv` 中缺失的 51 个实验

---

## 📋 实验概述

### 补充目标

基于 `docs/MISSING_EXPERIMENTS_CHECKLIST.md` 的分析，本配置旨在补充以下缺失实验：

| 模型 | 缺失数量 | 原因 | 本次补充 |
|------|----------|------|----------|
| Person_reID/hrnet18 | 15 | 孤儿进程影响 | 16 次 (8×2) |
| Person_reID/pcb | 16 | GPU OOM | 16 次 (8×2) |
| examples/mnist_ff | 20 | GPU OOM + 被过滤 | 20 次 (10×2) |
| **总计** | **51** | - | **52** |

### 关键特性

✅ **启用轮间去重机制** (`use_deduplication: true`)
- 加载历史数据：`results/summary_all.csv`
- 避免生成已有的超参数组合
- 自动过滤 177 个历史超参数

✅ **GPU 内存清理机制**
- `cleanup_gpu_memory: true`: 启用 GPU 内存清理
- `cleanup_between_experiments: true`: 实验间清理

✅ **增强重试机制**
- `max_retries: 2`: 从 1 增加到 2
- 针对容易 OOM 的模型提供更多重试机会

✅ **针对性优化配置**
- **pcb**: batch_size 从 16 降到 8，防止 OOM
- **mnist_ff**: batch_size 固定为 10000 (已修复代码)

---

## 🔧 配置详解

### 1. Person_reID/hrnet18 (补充 15 个缺失)

```json
{
  "repo": "Person_reID_baseline_pytorch",
  "model": "hrnet18",
  "num_mutations": 8,
  "mutate": ["epochs"]
}
```

**缺失原因**:
- mutation_2x 原始数据中有 14 个 hrnet18 实验在孤儿进程影响区间 (2025-11-25 01:20:48 之后)
- 这些数据被排除在 `summary_safe.csv` 之外
- 只有 1 个 hrnet18 实验在安全区间

**补充策略**:
- 生成 8 个变异 × 2 次运行 = 16 次实验
- 使用默认配置 (历史数据显示稳定)
- 变异参数: epochs

**预期结果**:
- hrnet18 总实验数: 11 → 27 (超出预期 1 个)
- 平均运行时间: ~74 分钟/次
- 总时间: ~19.7 小时

---

### 2. Person_reID/pcb (补充 16 个缺失)

```json
{
  "repo": "Person_reID_baseline_pytorch",
  "model": "pcb",
  "num_mutations": 8,
  "mutate": ["epochs"],
  "hyperparameters": {
    "batchsize": 8
  }
}
```

**缺失原因**:
- mutation_2x 原始数据中所有 16 个 pcb 实验都失败了
- 失败原因: GPU 内存溢出 (OOM)
- 原配置使用 batch_size=16

**补充策略**:
- 生成 8 个变异 × 2 次运行 = 16 次实验
- **关键优化**: batch_size 从 16 降到 8
- 变异参数: epochs
- 启用增强重试机制 (max_retries=2)

**预期结果**:
- pcb 总实验数: 10 → 26 (完整)
- 平均运行时间: ~70 分钟/次
- 总时间: ~18.7 小时
- **风险**: 仍需监控 GPU 内存使用

---

### 3. examples/mnist_ff (补充 20 个缺失)

```json
{
  "repo": "examples",
  "model": "mnist_ff",
  "num_mutations": 10,
  "mutate": ["epochs", "learning_rate"],
  "hyperparameters": {
    "batch_size": 10000
  }
}
```

**缺失原因**:
- 所有 mnist_ff 实验都因为 batch_size=50000 导致 GPU OOM
- 在 CSV 聚合时被过滤掉 (`aggregate_csvs.py` 默认过滤 mnist_ff)
- 缺失实验: default (2个), mutation_1x (6个), mutation_2x (12个)

**补充策略**:
- 生成 10 个变异 × 2 次运行 = 20 次实验
- **关键修复**: batch_size 固定为 10000
  - 代码已修改: `repos/examples/mnist_forward_forward/main.py:123`
  - 代码已修改: `repos/examples/train.sh:196`
- 变异参数: epochs, learning_rate (双参数变异)

**预期结果**:
- mnist_ff 总实验数: 0 → 20 (首次完整数据)
- 平均运行时间: ~2 分钟/次 (估计)
- 总时间: ~0.7 小时
- **验证目标**: 确认 batch_size=10000 不会 OOM

---

## ⏱️ 运行时间预估

### 详细时间分解

| 模型 | 实验次数 | 单次时间 | 总时间 |
|------|----------|----------|--------|
| hrnet18 | 16 | ~74 分钟 | ~19.7 小时 |
| pcb | 16 | ~70 分钟 | ~18.7 小时 |
| mnist_ff | 20 | ~2 分钟 | ~0.7 小时 |
| **总计** | **52** | - | **~39 小时** |

### 运行时间建议

**场景 1: 理想情况** (无失败/重试)
- 总时间: **39 小时** (~1.6 天)

**场景 2: 含重试缓冲** (+10%)
- 总时间: **42.9 小时** (~1.8 天)
- 假设: ~10% 的实验需要重试

**场景 3: 推荐预算** (含系统开销)
- 总时间: **45 小时** (~1.9 天) ✅
- 包含:
  - 重试缓冲: +10%
  - 系统开销: ~30 秒/实验
  - GPU 清理时间

### 分批运行建议

**批次 1**: hrnet18 (16 次)
- 预估时间: 19.7 小时
- 适合: 晚上运行 (20:00 → 次日 15:42)

**批次 2**: pcb (16 次)
- 预估时间: 18.7 小时
- 适合: 晚上运行 (20:00 → 次日 14:42)

**批次 3**: mnist_ff (20 次)
- 预估时间: 0.7 小时
- 适合: 任意时间

---

## 🚀 使用方法

### 方法 1: 使用 MutationRunner (推荐)

```bash
# 方法 1a: 直接运行 (如果 MutationRunner 支持 use_deduplication)
python3 -m mutation.runner settings/mutation_2x_supplement.json

# 方法 1b: 使用自定义脚本 (如果需要手动集成 deduplication)
python3 scripts/run_supplement_experiments.py
```

### 方法 2: 手动运行 (用于测试)

```python
#!/usr/bin/env python3
"""
运行补充实验脚本
"""
from pathlib import Path
from mutation.runner import MutationRunner
from mutation.dedup import load_historical_mutations, build_dedup_set, print_dedup_statistics

# Step 1: 加载历史数据
csv_files = [
    Path("results/summary_all.csv"),
]

print("=" * 80)
print("加载历史超参数数据...")
print("=" * 80)

mutations, stats = load_historical_mutations(csv_files)
dedup_set = build_dedup_set(mutations)
print_dedup_statistics(stats, dedup_set)

# Step 2: 运行实验
# 注意: MutationRunner 已集成 deduplication 支持
runner = MutationRunner(
    config_path="settings/mutation_2x_supplement.json"
)

# Deduplication 已通过配置文件自动启用
runner.run_from_experiment_config("settings/mutation_2x_supplement.json")
```

---

## 📊 预期成果

### 实验完成后的数据状态

| 指标 | 当前 | 补充后 | 变化 |
|------|------|--------|------|
| 总实验数 | 211 | 263 | +52 |
| hrnet18 实验数 | 11 | 27 | +16 ✓ |
| pcb 实验数 | 10 | 26 | +16 ✓ |
| mnist_ff 实验数 | 0 | 20 | +20 ✓ |
| 模型完整性 | 90% | 100% | +10% ✓ |

### 数据质量指标

**当前 (`results/summary_all.csv`)**:
- 实验总数: 211
- 成功率: 100% (已过滤失败数据)
- 模型覆盖: 11 个模型
- 缺失实验: 51 个

**补充后 (预期)**:
- 实验总数: 263
- 成功率: 预期 >95% (考虑 pcb 可能 OOM)
- 模型覆盖: 11 个模型 (全部完整)
- 缺失实验: 0 个 (理想情况)

---

## ⚠️ 注意事项与风险

### 1. GPU 内存监控 (高风险)

**pcb 模型**:
- ⚠️ **风险**: 即使 batch_size=8，仍可能 OOM
- 📊 **历史数据**: 所有 16 个 mutation_2x pcb 实验都失败
- 🔍 **监控**: 实时检查 `nvidia-smi` GPU 内存使用
- 🛠️ **后备方案**: 如果仍然 OOM，进一步降低到 batch_size=4

**mnist_ff 模型**:
- ⚠️ **风险**: 无历史成功数据，batch_size=10000 未经验证
- 📊 **历史数据**: 所有 20 个实验都失败 (batch_size=50000)
- 🔍 **监控**: 首次运行时密切观察内存使用
- 🛠️ **后备方案**: 如果 OOM，降低到 batch_size=5000

### 2. 轮间去重机制验证

**当前状态**:
- ✅ 模块已实现: `mutation/dedup.py`
- ✅ 测试已通过: 6/6 tests passing
- ✅ 已集成到 MutationRunner: 自动从配置文件读取

**验证步骤**:
1. 确认配置文件中 `use_deduplication: true` 被正确读取
2. 确认 `historical_csvs: ["results/summary_all.csv"]` 存在
3. 运行时查看日志，应显示 "Loaded 177 historical mutations"
4. 运行后检查生成的超参数是否与历史数据重复

**查看去重日志**:
```bash
# 运行实验时，应该看到类似输出
grep "Loaded.*historical mutations" results/*/logs/*.log

# 预期输出:
# "Loaded 177 historical mutations for deduplication"
```

### 3. 实验运行时间

**长时间运行风险**:
- hrnet18: ~19.7 小时
- pcb: ~18.7 小时
- 建议分批运行，避免单次运行时间过长

**监控建议**:
```bash
# 监控 GPU 使用
watch -n 5 nvidia-smi

# 监控实验日志
tail -f results/mutation_2x_supplement_*/logs/*.log

# 监控 GPU 温度
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -l 5
```

### 4. 磁盘空间

**预估空间需求**:
- 每个实验: ~100-500 MB (日志 + checkpoint)
- 52 个实验: ~5-25 GB
- **检查**: `df -h` 确保有足够空间

### 5. 孤儿进程预防

**确保没有新的孤儿进程**:
```bash
# 运行前检查
nvidia-smi

# 运行后检查
nvidia-smi

# 如果发现新的孤儿进程
sudo kill -15 <PID>
```

---

## 🔍 成功标准

### 必须达到 (P0)

- [x] 配置文件创建并验证
- [ ] 所有 52 个实验成功完成
- [ ] 无 GPU OOM 错误
- [ ] GPU 内存正常清理
- [ ] 实验结果 CSV 生成

### 高优先级 (P1)

- [ ] 轮间去重机制正常工作
- [ ] 无重复超参数生成
- [ ] pcb 实验全部成功 (batch_size=8 足够)
- [ ] mnist_ff 实验全部成功 (batch_size=10000 足够)

### 中优先级 (P2)

- [ ] 总运行时间 < 45 小时
- [ ] 重试次数 < 10% 实验
- [ ] GPU 温度正常 (< 85°C)

---

## 📈 后续步骤

### 实验完成后

1. **聚合新数据**
```bash
# 将新结果添加到 summary_all.csv
python3 scripts/aggregate_csvs.py --add-supplement
```

2. **验证数据完整性**
```bash
# 检查缺失实验是否已补充
python3 scripts/check_missing_experiments.py
```

3. **重新分析重复实验**
```bash
# 检查是否仍有重复
python3 scripts/analyze_duplicates.py
```

4. **更新文档**
- 更新 `docs/MISSING_EXPERIMENTS_CHECKLIST.md`
- 标记已完成的补充实验

### 如果仍有失败

**如果 pcb 仍然 OOM**:
1. 进一步降低 batch_size 到 4
2. 或者考虑减少 epochs 范围
3. 或者使用更小的模型 (densenet121 作为替代)

**如果 mnist_ff 仍然 OOM**:
1. 降低 batch_size 到 5000
2. 或者跳过 mnist_ff，在文档中说明原因

---

## 🔗 相关文档

- **缺失实验分析**: `docs/MISSING_EXPERIMENTS_CHECKLIST.md`
- **轮间去重机制**: `docs/INTER_ROUND_DEDUPLICATION.md`
- **CSV 聚合脚本**: `scripts/aggregate_csvs.py`
- **去重测试脚本**: `tests/unit/test_dedup_mechanism.py`
- **孤儿进程分析**: `docs/ORPHAN_PROCESS_ANALYSIS.md`
- **GPU 清理机制**: `docs/GPU_MEMORY_CLEANUP_FIX.md`

---

**创建时间**: 2025-11-26
**维护者**: Mutation-Based Training Energy Profiler Team
**版本**: 1.0
**状态**: Ready to Run
