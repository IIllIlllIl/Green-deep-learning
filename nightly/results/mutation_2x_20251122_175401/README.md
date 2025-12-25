# Mutation 2x 实验结果 (2025-11-22 至 2025-11-26)

## 概览

**实验名称**: mutation_all_models_2x_dynamic
**配置文件**: settings/mutation_all_models_2x_dynamic.json
**运行时间**: 2025-11-22 17:54:01 至 2025-11-26 06:27:53 (约3.5天)
**总实验数**: 160
**完整成功率**: 82.5% (132/160)

## ⚠️ 重要发现：孤儿进程影响

### 问题描述

运行期间出现了一个后台训练的**孤儿进程**，严重影响了后续实验：

- **进程ID**: 1970647
- **创建时间**: 2025-11-25 01:20:48
- **GPU内存占用**: 5.54 GB (持续占用直到 2025-11-26 15:34 手动清理)
- **影响范围**: 31个实验受到影响
- **原因**: parallel实验的后台进程未被正确清理

### 成功率对比

| 时期 | 实验数 | 成功数 | 失败数 | 成功率 | 说明 |
|------|--------|--------|--------|--------|------|
| **孤儿进程创建前** | 129 | 117 | 12 | **90.7%** | 安全数据 |
| **孤儿进程创建后** | 31 | 15 | 16 | **48.4%** | 受影响数据 |
| **整体** | 160 | 132 | 28 | **82.5%** | 完整数据 |

**结论**: 孤儿进程导致成功率下降了 **42.3个百分点**！

## 数据文件说明

### summary.csv
- **内容**: 完整的160个实验结果
- **用途**: 保留原始完整数据供参考
- **注意**: 包含受孤儿进程影响的31个实验

### summary_safe.csv ⭐
- **内容**: 129个安全实验结果（孤儿进程创建前完成）
- **用途**: **推荐用于分析和论文数据**
- **质量**: 90.7%成功率，数据质量高
- **时间范围**: 2025-11-22 17:54:01 至 2025-11-25 01:20:48

## 安全数据统计

### 按模型分布 (summary_safe.csv)

| Repository | 实验数 | 说明 |
|------------|--------|------|
| examples | 48 | mnist, mnist_rnn, mnist_ff (部分), siamese |
| MRT-OAST | 20 | 完整数据 |
| Person_reID_baseline_pytorch | 17 | densenet121 (部分), hrnet18 (部分) |
| pytorch_resnet_cifar10 | 16 | resnet20 完整数据 |
| VulBERTa | 16 | mlp 完整数据 |
| bug-localization-by-dnn-and-rvsm | 12 | 完整数据 |

### 失败实验分析 (安全数据中的12个失败)

安全数据中的12个失败实验**不是由孤儿进程引起**，而是正常的失败原因：

```bash
# 查看失败实验
grep "False" summary_safe.csv | awk -F, '{print $1, $5}'
```

失败分布：
- examples/mnist_ff: 12个失败（batch_size=50000过大，已在后续修复）

**注意**: 这12个mnist_ff失败是因为batch_size配置问题（使用了50000），已在修复中降低至10000。

## 受影响实验列表 (孤儿进程创建后)

受影响的31个实验：
- Person_reID_baseline_pytorch/hrnet18: 130-144 (15个)
- Person_reID_baseline_pytorch/pcb: 145-160 (16个，全部失败)

这些实验的数据**不建议用于分析**，因为GPU内存压力导致结果不可靠。

## 时间线

```
2025-11-22 17:54:01  实验开始
     ↓
[129个实验成功运行，成功率90.7%]
     ↓
2025-11-25 00:06:11  最后一个安全实验完成 (hrnet18_129)
2025-11-25 00:57:31  hrnet18_130 开始
2025-11-25 01:20:48  ⚠️ 孤儿进程创建 (GPU内存泄漏开始)
     ↓
[31个实验受影响，成功率48.4%]
     ↓
2025-11-26 06:27:53  实验结束
2025-11-26 15:34:00  手动清理孤儿进程，GPU内存释放
```

## 后续改进

基于此次问题，已实施以下改进：

1. ✅ 创建GPU内存清理机制 (`mutation/gpu_cleanup.py`)
2. ✅ 在实验间自动清理GPU缓存
3. ✅ 降低mnist_ff的batch_size (50000 → 10000)
4. ✅ 创建GPU清理测试配置
5. ⏳ 待实施：增强后台进程管理

详见：
- `docs/GPU_MEMORY_CLEANUP_FIX.md` - GPU清理机制说明
- `docs/ORPHAN_PROCESS_ANALYSIS.md` - 孤儿进程根因分析
- `docs/COMPLETE_FIX_SUMMARY.md` - 完整修复方案

## 使用建议

### 用于数据分析
```bash
# 推荐：使用安全数据
cat results/mutation_2x_20251122_175401/summary_safe.csv

# 129个实验，90.7%成功率，数据质量高
```

### 查看完整数据
```bash
# 包含受影响的31个实验
cat results/mutation_2x_20251122_175401/summary.csv

# 160个实验，82.5%成功率，但受孤儿进程影响
```

### 对比分析
```python
import pandas as pd

# 安全数据
safe_df = pd.read_csv('results/mutation_2x_20251122_175401/summary_safe.csv')
print(f"安全数据成功率: {safe_df['training_success'].mean():.1%}")

# 完整数据
full_df = pd.read_csv('results/mutation_2x_20251122_175401/summary.csv')
print(f"完整数据成功率: {full_df['training_success'].mean():.1%}")

# 受影响数据
affected_ids = set(full_df['experiment_id']) - set(safe_df['experiment_id'])
affected_df = full_df[full_df['experiment_id'].isin(affected_ids)]
print(f"受影响数据成功率: {affected_df['training_success'].mean():.1%}")
```

## 验证数据完整性

```bash
# 检查summary_safe.csv行数
wc -l summary_safe.csv  # 应该是130行 (129实验 + 1 header)

# 检查没有重复
awk -F, 'NR>1 {print $1}' summary_safe.csv | sort | uniq -d  # 应该无输出

# 验证所有实验ID格式正确
awk -F, 'NR>1 {print $1}' summary_safe.csv | grep -E "^[a-zA-Z0-9_-]+$" | wc -l  # 应该=129
```

## 联系信息

如有问题或需要更多信息，请参考项目文档或联系维护者。

---

**生成时间**: 2025-11-26
**生成工具**: Claude Code
**版本**: Mutation-Based Training Energy Profiler v4.3.0
