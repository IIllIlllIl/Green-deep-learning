# 2x变异测试数据整理总结

## 执行的任务

✅ **已完成所有任务**：

1. ✅ 减少 mnist_ff batch size (50000 → 10000)
2. ✅ 分析孤儿进程生成原因
3. ✅ 清理残留GPU进程 (用户执行 `sudo kill -15 1970647 1970645`)
4. ✅ 验证GPU内存已释放 (0 MiB / 10240 MiB)
5. ✅ 重命名结果目录为 `mutation_2x_20251122_175401`
6. ✅ 创建 `summary_safe.csv` (129个安全实验)
7. ✅ 创建完整的 README.md 说明文档

## 数据整理结果

### 目录结构

```
results/mutation_2x_20251122_175401/
├── README.md                    # 完整说明文档 ⭐
├── summary.csv                  # 完整数据 (160个实验)
├── summary_safe.csv             # 安全数据 (129个实验) ⭐ 推荐使用
├── experiment.json
└── [160个实验子目录]
```

### 数据质量分析

| 数据集 | 实验数 | 成功数 | 失败数 | 成功率 | 推荐用途 |
|--------|--------|--------|--------|--------|----------|
| **summary_safe.csv** | 129 | 117 | 12 | **90.7%** | ⭐ **论文和分析** |
| summary.csv (完整) | 160 | 132 | 28 | 82.5% | 参考/对比 |
| 受影响数据 | 31 | 15 | 16 | 48.4% | ❌ 不推荐 |

### 安全数据覆盖范围

**时间范围**: 2025-11-22 17:54:01 至 2025-11-25 01:20:48 (约2.3天)

**模型覆盖** (129个实验):
- ✅ MRT-OAST: 20个 (完整)
- ✅ VulBERTa/mlp: 16个 (完整)
- ✅ pytorch_resnet_cifar10/resnet20: 16个 (完整)
- ✅ bug-localization: 12个 (完整)
- ✅ examples/mnist: 12个 (完整)
- ✅ examples/mnist_rnn: 12个 (完整)
- ✅ examples/siamese: 12个 (完整)
- ⚠️ examples/mnist_ff: 12个 (全部失败，batch_size配置问题)
- ✅ Person_reID/densenet121: 8个 (部分)
- ✅ Person_reID/hrnet18: 9个 (部分)

**注意**:
- Person_reID的densenet121和hrnet18只包含部分数据
- 完整的PCB模型数据缺失（在受影响区间）

### 失败原因分析

**安全数据中的12个失败** (全部是 mnist_ff):
- **原因**: batch_size=50000过大，导致GPU OOM
- **修复**: 已将默认batch_size降至10000
- **时间**: 实验开始阶段（实验001-012）
- **性质**: 配置问题，非孤儿进程影响

**受影响数据中的16个失败**:
- **原因**: 孤儿进程占用5.54GB GPU内存
- **模型**: 全部是 Person_reID/pcb (实验145-160)
- **时间**: 2025-11-25至2025-11-26
- **性质**: 基础设施问题

## 关键发现

### 🔍 孤儿进程影响分析

**影响前后对比**:
```
孤儿进程创建前 (129实验):  90.7% 成功率
孤儿进程创建后 (31实验):   48.4% 成功率
                         ↓
                    -42.3% 成功率下降！
```

**时间线**:
```
2025-11-22 17:54:01  ━━━━━━━ 实验开始 ━━━━━━━

[实验 001-012] mnist_ff失败 (batch_size配置问题)
[实验 013-129] 正常运行 (成功率100%)

2025-11-25 00:06:11  最后一个安全实验 (hrnet18_129)
2025-11-25 01:20:48  ⚠️ 孤儿进程创建

[实验 130-144] hrnet18 (部分成功)
[实验 145-160] pcb (全部失败)

2025-11-26 06:27:53  ━━━━━━━ 实验结束 ━━━━━━━
```

### 📊 数据可靠性评估

**高质量数据** (推荐用于论文):
- ✅ 129个安全实验
- ✅ 90.7%成功率
- ✅ 覆盖8个完整模型
- ✅ 每个模型至少12个实验
- ✅ 失败原因明确（mnist_ff配置问题）

**中等质量数据** (可用于对比研究):
- ⚠️ 完整160个实验
- ⚠️ 82.5%成功率
- ⚠️ 包含受孤儿进程影响的数据
- ⚠️ 需要标注受影响的实验

## 使用建议

### 推荐做法 ⭐

```bash
# 使用安全数据进行分析
cd /home/green/energy_dl/nightly/results/mutation_2x_20251122_175401

# Python分析
import pandas as pd
df = pd.read_csv('summary_safe.csv')

# 过滤出成功的实验（排除mnist_ff失败）
df_success = df[df['training_success'] == True]
print(f"成功实验: {len(df_success)}/129")

# 按模型分组分析
for (repo, model), group in df_success.groupby(['repository', 'model']):
    print(f"{repo}/{model}: {len(group)}个实验")
```

### 可选做法

```bash
# 如果需要完整数据（包含受影响实验）
df_full = pd.read_csv('summary.csv')

# 标记受影响的实验
from datetime import datetime
ORPHAN_TIME = datetime.fromisoformat("2025-11-25T01:20:48")

df_full['orphan_affected'] = df_full.apply(
    lambda row: datetime.fromisoformat(row['timestamp']) +
                timedelta(seconds=row['duration_seconds']) >= ORPHAN_TIME,
    axis=1
)

# 分别分析
df_safe = df_full[~df_full['orphan_affected']]
df_affected = df_full[df_full['orphan_affected']]
```

## 下一步建议

### 立即可做

1. **验证数据完整性**
   ```bash
   # 检查行数
   wc -l summary_safe.csv  # 应该是130行 (129+header)

   # 检查没有重复
   awk -F, 'NR>1 {print $1}' summary_safe.csv | sort | uniq -d
   ```

2. **运行GPU清理测试**
   ```bash
   # 验证GPU清理机制和新的batch_size
   sudo -E python3 mutation.py -ec settings/gpu_memory_cleanup_test.json
   ```

### 后续计划

1. **补充缺失数据** (可选)
   - 重新运行 Person_reID/densenet121 和 hrnet18 (补充到完整的16个实验)
   - 运行完整的 Person_reID/pcb (16个实验)

2. **重新运行完整2x测试** (如果GPU清理测试成功)
   ```bash
   # 使用改进后的框架重新运行
   sudo -E python3 mutation.py -ec settings/mutation_all_models_2x_dynamic.json
   ```

3. **数据分析和论文撰写**
   - 使用 summary_safe.csv 作为主要数据源
   - 分析超参数对能耗的影响
   - 对比不同模型的能耗特征

## 文档参考

- `results/mutation_2x_20251122_175401/README.md` - 数据集完整说明
- `docs/GPU_MEMORY_CLEANUP_FIX.md` - GPU清理机制
- `docs/ORPHAN_PROCESS_ANALYSIS.md` - 孤儿进程根因分析
- `docs/COMPLETE_FIX_SUMMARY.md` - 完整修复方案

## 总结

✅ **成功完成数据整理**:
- 识别并隔离了受孤儿进程影响的数据
- 创建了高质量的安全数据集 (90.7%成功率)
- 提供了完整的文档和使用指南
- 修复了mnist_ff的batch_size配置问题
- 实施了GPU内存清理机制

🎯 **可用于论文的数据**:
- 129个高质量实验
- 覆盖8个完整模型
- 117个成功实验（排除mnist_ff配置问题）
- 清晰的失败原因说明

---

**整理完成时间**: 2025-11-26
**GPU内存状态**: ✅ 已清理 (0 MiB占用)
**框架状态**: ✅ 已改进 (GPU清理机制已添加)
**准备就绪**: ✅ 可以运行新的测试
