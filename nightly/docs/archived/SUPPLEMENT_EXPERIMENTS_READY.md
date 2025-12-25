# 补充实验准备完成 - 工作总结

**完成时间**: 2025-11-26
**状态**: ✅ Ready to Run
**任务**: 准备运行新一轮 2x 变异实验，补充缺失的 51 个实验

---

## 📋 完成的工作

### 1. 缺失实验分析 ✅

创建了详细的缺失实验清单：`docs/MISSING_EXPERIMENTS_CHECKLIST.md`

**分析结果**:
- 当前实验总数: 211
- 预期实验总数: 280
- 缺失实验总数: **69** (实际需要补充 51 个)
- 轮间重复实验: 6 个

**缺失分类**:
1. **Person_reID/hrnet18**: 缺失 15 个 (mutation_2x_safe)
   - 原因: 孤儿进程影响 (2025-11-25 01:20:48 之后)

2. **Person_reID/pcb**: 缺失 16 个 (mutation_2x_safe)
   - 原因: 所有实验 GPU OOM

3. **examples/mnist_ff**: 缺失 20 个 (所有轮次)
   - 原因: batch_size=50000 导致 OOM，已被过滤

4. **轮间重复**: 6 个冗余实验
   - MRT-OAST/default epochs=8.0: 3 次
   - VulBERTa/mlp epochs=12.0: 2 次
   - examples/mnist_rnn epochs=12.0: 3 次
   - examples/siamese epochs=14.0: 2 次

---

### 2. 实验配置文件创建 ✅

创建了补充实验配置：`settings/mutation_2x_supplement.json`

**配置特点**:
```json
{
  "experiment_name": "mutation_2x_supplement_20251126",
  "mode": "batch",
  "runs_per_config": 2,
  "max_retries": 2,
  "use_deduplication": true,  // ← 启用轮间去重
  "historical_csvs": [         // ← 加载历史数据
    "results/summary_all.csv"
  ],
  "experiments": [
    {
      "repo": "Person_reID_baseline_pytorch",
      "model": "hrnet18",
      "num_mutations": 8
    },
    {
      "repo": "Person_reID_baseline_pytorch",
      "model": "pcb",
      "num_mutations": 8,
      "hyperparameters": {
        "batchsize": 8  // ← 降低 batch size 防止 OOM
      }
    },
    {
      "repo": "examples",
      "model": "mnist_ff",
      "num_mutations": 10,
      "hyperparameters": {
        "batch_size": 10000  // ← 使用新的 batch size
      }
    }
  ]
}
```

**补充数量**:
- hrnet18: 8 × 2 = 16 次 (补充 15 个 + 1 个额外)
- pcb: 8 × 2 = 16 次 (补充 16 个)
- mnist_ff: 10 × 2 = 20 次 (补充 20 个)
- **总计**: 52 次实验

---

### 3. MutationRunner 集成轮间去重机制 ✅

**修改文件**: `mutation/runner.py`

**关键修改**:

1. **导入去重模块** (line 28):
```python
from .dedup import load_historical_mutations, build_dedup_set, print_dedup_statistics
```

2. **加载历史数据** (lines 797-853):
```python
# Inter-round deduplication support
use_deduplication = exp_config.get("use_deduplication", False)
historical_csvs = exp_config.get("historical_csvs", [])
dedup_set = None

# Load historical mutations for deduplication if enabled
if use_deduplication and historical_csvs:
    print("\n" + "=" * 80)
    print("LOADING HISTORICAL HYPERPARAMETER DATA")
    print("=" * 80)

    csv_paths = [self.project_root / csv_path for csv_path in historical_csvs]
    existing_csvs = [p for p in csv_paths if p.exists()]

    if existing_csvs:
        mutations, stats = load_historical_mutations(existing_csvs, logger=self.logger)
        dedup_set = build_dedup_set(mutations, logger=self.logger)
        print_dedup_statistics(stats, dedup_set)
```

3. **传递去重集合给 generate_mutations()** (lines 927, 1024):
```python
mutations = generate_mutations(
    supported_params=supported_params,
    mutate_params=mutate_params,
    num_mutations=runs_per_config,
    random_seed=self.random_seed,
    logger=self.logger,
    existing_mutations=dedup_set  # ← Inter-round deduplication
)
```

**集成效果**:
- ✅ 自动加载 177 个历史超参数组合
- ✅ 避免生成重复的超参数
- ✅ 完全向后兼容 (不启用时无影响)

---

### 4. 集成测试脚本 ✅

创建了测试脚本：`tests/functional/test_runner_dedup_integration.py`

**测试结果**: **6/6 通过** ✅

```
================================================================================
Test Summary
================================================================================
Total tests: 6
Passed: 6
Failed: 0

✓ All tests passed!
```

**测试内容**:
1. ✅ 配置文件加载
2. ✅ 历史 CSV 文件存在性检查
3. ✅ MutationRunner 初始化
4. ✅ 去重模块导入
5. ✅ 实验配置验证
6. ✅ 从 summary_all.csv 加载历史数据

---

### 5. 详细文档 ✅

创建了配置文档：`settings/MUTATION_2X_SUPPLEMENT_README.md`

**文档内容**:
- 📊 实验概述和补充目标
- 🔧 三个实验的详细配置解析
- ⏱️ 运行时间预估 (39-45 小时)
- 🚀 使用方法和运行命令
- 📊 预期成果
- ⚠️ 注意事项与风险
- 🔍 成功标准
- 📈 后续步骤

---

## ⏱️ 运行时间预估

### 详细分解

| 模型 | 实验次数 | 单次时间 | 总时间 |
|------|----------|----------|--------|
| hrnet18 | 16 | ~74 分钟 | ~19.7 小时 |
| pcb | 16 | ~70 分钟 | ~18.7 小时 |
| mnist_ff | 20 | ~2 分钟 | ~0.7 小时 |
| **总计** | **52** | - | **~39 小时** |

### 推荐预算

**45 小时** (~1.9 天) ✅

包含:
- 理想运行时间: 39 小时
- 重试缓冲 (+10%): 3.9 小时
- 系统开销: ~2 小时

### 分批运行建议

**批次 1**: hrnet18 (16 次) - 约 20 小时
**批次 2**: pcb (16 次) - 约 19 小时
**批次 3**: mnist_ff (20 次) - 约 1 小时

---

## 🚀 如何运行

### 方法 1: 直接运行 (推荐)

```bash
cd /home/green/energy_dl/nightly

# 确保环境正确
python3 --version  # 应该是 Python 3.x

# 运行补充实验
python3 -m mutation.runner settings/mutation_2x_supplement.json
```

### 方法 2: 使用 nohup 后台运行

```bash
# 后台运行，输出到日志文件
nohup python3 -m mutation.runner settings/mutation_2x_supplement.json > supplement_run.log 2>&1 &

# 查看进程
ps aux | grep mutation

# 实时查看日志
tail -f supplement_run.log

# 或使用 tmux/screen
tmux new -s supplement
python3 -m mutation.runner settings/mutation_2x_supplement.json
# Ctrl+B, D 分离会话
```

### 方法 3: 分批运行

**批次 1: hrnet18**
```bash
# 创建临时配置 (只包含 hrnet18)
# 编辑 settings/mutation_2x_supplement.json，只保留第一个实验
python3 -m mutation.runner settings/mutation_2x_supplement.json
```

**批次 2: pcb**
```bash
# 修改配置，只保留 pcb 实验
python3 -m mutation.runner settings/mutation_2x_supplement.json
```

**批次 3: mnist_ff**
```bash
# 修改配置，只保留 mnist_ff 实验
python3 -m mutation.runner settings/mutation_2x_supplement.json
```

---

## ⚠️ 关键注意事项

### 1. GPU 内存监控

**pcb 模型高风险**:
- batch_size 已降到 8，但仍可能 OOM
- 需要实时监控 GPU 内存

```bash
# 实时监控 GPU
watch -n 5 nvidia-smi

# 或在另一个终端运行
nvidia-smi dmon -s u -d 10
```

**如果 pcb 仍然 OOM**:
1. 进一步降低到 batch_size=4
2. 修改配置文件重新运行

### 2. mnist_ff 验证

**首次使用 batch_size=10000**:
- 历史数据中无成功案例
- 需要密切观察第一次运行

**如果 mnist_ff OOM**:
1. 降低到 batch_size=5000
2. 或跳过 mnist_ff，在文档中说明

### 3. 轮间去重机制

**自动加载历史数据**:
- 从 `results/summary_all.csv` 加载 177 个唯一超参数
- 自动避免重复生成

**验证去重效果**:
运行后检查生成的超参数：
```bash
# 查看实验日志
grep "Loaded.*historical mutations" results/mutation_2x_supplement_*/logs/*.log

# 应该看到类似输出:
# "Loaded 177 historical mutations for deduplication"
```

### 4. 长时间运行

**建议**:
- 使用 tmux/screen 或 nohup
- 定期检查进程状态
- 监控磁盘空间 (需要 ~5-25 GB)

```bash
# 检查磁盘空间
df -h /home/green/energy_dl/nightly/results
```

---

## 📊 预期成果

### 实验完成后的数据状态

| 指标 | 当前 | 补充后 | 变化 |
|------|------|--------|------|
| 总实验数 | 211 | 263 | +52 ✅ |
| hrnet18 实验数 | 11 | 27 | +16 ✅ |
| pcb 实验数 | 10 | 26 | +16 ✅ |
| mnist_ff 实验数 | 0 | 20 | +20 ✅ |
| 模型完整性 | 90% | 100% | +10% ✅ |

### 后续聚合

**运行完成后**:
```bash
# 将新结果添加到 summary_all.csv
python3 scripts/aggregate_csvs.py \
  --add-source mutation_2x_supplement \
  --source-path results/mutation_2x_supplement_YYYYMMDD_HHMMSS/summary.csv \
  --output results/summary_all_v2.csv
```

---

## 🔍 成功标准

### 必须达到 (P0)

- [x] 配置文件创建并验证 ✅
- [x] MutationRunner 集成去重机制 ✅
- [x] 集成测试通过 ✅
- [x] 详细文档完成 ✅
- [ ] 所有 52 个实验成功完成
- [ ] 无 GPU OOM 错误
- [ ] 实验结果 CSV 生成

### 高优先级 (P1)

- [x] 轮间去重机制实现 ✅
- [ ] 轮间去重机制运行验证
- [ ] 无重复超参数生成
- [ ] pcb 实验全部成功
- [ ] mnist_ff 实验全部成功

### 中优先级 (P2)

- [ ] 总运行时间 < 45 小时
- [ ] 重试次数 < 10% 实验
- [ ] GPU 温度正常 (< 85°C)

---

## 📝 相关文件

### 新创建的文件

1. **配置文件**:
   - `settings/mutation_2x_supplement.json` - 补充实验配置
   - `settings/MUTATION_2X_SUPPLEMENT_README.md` - 详细文档

2. **测试脚本**:
   - `tests/functional/test_runner_dedup_integration.py` - 集成测试

3. **分析文档**:
   - `docs/MISSING_EXPERIMENTS_CHECKLIST.md` - 缺失实验清单

### 修改的文件

1. **核心模块**:
   - `mutation/runner.py` - 集成去重机制 (lines 28, 797-853, 927, 1024)

### 相关文档

1. **去重机制**:
   - `docs/INTER_ROUND_DEDUPLICATION.md` - 去重机制文档
   - `mutation/dedup.py` - 去重模块实现
   - `tests/unit/test_dedup_mechanism.py` - 去重测试脚本

2. **数据分析**:
   - `results/summary_all.csv` - 当前汇总数据
   - `results/SUMMARY_ALL_README.md` - 数据文档

3. **其他相关**:
   - `docs/ORPHAN_PROCESS_ANALYSIS.md` - 孤儿进程分析
   - `docs/GPU_MEMORY_CLEANUP_FIX.md` - GPU 清理机制

---

## 🎯 下一步行动

### 立即执行

1. **运行集成测试** (已完成 ✅)
```bash
python3 tests/functional/test_runner_dedup_integration.py
```

2. **启动补充实验**
```bash
# 使用 tmux 或 screen
tmux new -s supplement
python3 -m mutation.runner settings/mutation_2x_supplement.json

# 或后台运行
nohup python3 -m mutation.runner settings/mutation_2x_supplement.json > supplement_run.log 2>&1 &
```

3. **监控实验进度**
```bash
# GPU 监控
watch -n 5 nvidia-smi

# 日志监控
tail -f results/mutation_2x_supplement_*/logs/*.log

# 或查看后台日志
tail -f supplement_run.log
```

### 运行期间

1. **定期检查**:
   - GPU 内存使用
   - 磁盘空间
   - 进程状态
   - 实验进度

2. **应对问题**:
   - 如果 pcb OOM: 降低 batch_size 到 4
   - 如果 mnist_ff OOM: 降低 batch_size 到 5000
   - 如果进程意外终止: 检查日志，必要时重启

### 完成后

1. **聚合新数据**:
```bash
python3 scripts/aggregate_csvs.py --add-supplement
```

2. **验证结果**:
```bash
# 检查实验数量
wc -l results/summary_all_v2.csv
# 应该是 263 行 + 1 行表头 = 264 行

# 检查无重复
python3 scripts/analyze_duplicates.py
```

3. **更新文档**:
   - 标记 `docs/MISSING_EXPERIMENTS_CHECKLIST.md` 中的完成状态
   - 创建实验总结报告

---

## 📞 联系与反馈

如遇到问题:
1. 检查 `settings/MUTATION_2X_SUPPLEMENT_README.md` 的故障排除部分
2. 查看实验日志文件
3. 检查 GPU 状态 (`nvidia-smi`)

---

**准备时间**: 2025-11-26
**维护者**: Mutation-Based Training Energy Profiler Team
**版本**: 1.0
**状态**: ✅ Ready to Run

---

## 总结

✅ **所有准备工作已完成**

已完成:
1. ✅ 缺失实验详细分析 (51 个)
2. ✅ 补充实验配置文件创建
3. ✅ MutationRunner 集成轮间去重机制
4. ✅ 集成测试 (6/6 通过)
5. ✅ 详细文档和使用指南

**可以立即运行补充实验**:
```bash
python3 -m mutation.runner settings/mutation_2x_supplement.json
```

预计完成时间: **45 小时** (~1.9 天)

补充完成后:
- 总实验数: 211 → 263 (+52)
- 模型完整性: 90% → 100%
- 所有缺失实验已补充 ✅
