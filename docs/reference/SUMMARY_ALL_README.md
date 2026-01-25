# 汇总实验数据说明

## 文件信息

**文件名**: `summary_all.csv`
**生成时间**: 2025-11-26
**总实验数**: 211
**成功率**: 100%

## 数据来源

本文件汇总了三次独立实验的结果：

### 1. Default 实验 (default)
- **来源**: `results/defualt/summary.csv`
- **说明**: 使用默认超参数的基准实验
- **实验数**: 20
- **时间**: 2025-11-18 至 2025-11-19

### 2. 1x 变异测试 (mutation_1x)
- **来源**: `results/mutation_1x/summary.csv`
- **说明**: 每个配置运行1次的变异实验
- **实验数**: 74
- **时间**: 2025-11-20 至 2025-11-22

### 3. 2x 变异测试安全数据 (mutation_2x_safe)
- **来源**: `results/mutation_2x_20251122_175401/summary_safe.csv`
- **说明**: 每个配置运行2次的变异实验（仅包含安全数据，孤儿进程创建前）
- **实验数**: 117
- **时间**: 2025-11-22 至 2025-11-25 (孤儿进程创建前)

## 数据处理

### 过滤规则

**已过滤的实验**：
- ✗ **mnist_ff** - 共过滤20个实验
  - 原因：batch_size配置问题（使用了50000，已在后续修复为10000）
  - 分布：default (2), mutation_1x (6), mutation_2x_safe (12)

### 实验ID处理

为确保实验ID唯一性，所有ID已添加来源前缀：

**格式**: `{来源}__{原始实验ID}`

**示例**:
- 原始ID: `examples_mnist_001`
- default来源: `default__examples_mnist_001`
- mutation_1x来源: `mutation_1x__examples_mnist_001`
- mutation_2x_safe来源: `mutation_2x_safe__examples_mnist_001`

### 新增列

**experiment_source** - 标识数据来源
- `default`: 默认超参数实验
- `mutation_1x`: 1x变异实验
- `mutation_2x_safe`: 2x变异实验安全数据

## 数据质量

### 整体统计

| 指标 | 数值 |
|------|------|
| 总实验数 | 211 |
| 成功实验 | 211 (100%) |
| 失败实验 | 0 (0%) |
| 唯一ID数 | 211 |
| 数据列数 | 37 |

### 模型分布

| 模型 | 实验数 | 说明 |
|------|--------|------|
| MRT-OAST/default | 32 | 完整数据 |
| pytorch_resnet_cifar10/resnet20 | 26 | 完整数据 |
| VulBERTa/mlp | 26 | 完整数据 |
| Person_reID_baseline_pytorch/densenet121 | 26 | 完整数据 |
| bug-localization-by-dnn-and-rvsm/default | 20 | 完整数据 |
| examples/mnist | 20 | 完整数据 |
| examples/mnist_rnn | 20 | 完整数据 |
| examples/siamese | 20 | 完整数据 |
| Person_reID_baseline_pytorch/hrnet18 | 11 | 部分数据 |
| Person_reID_baseline_pytorch/pcb | 10 | 部分数据 |

**注意**:
- hrnet18 和 pcb 只有部分数据，因为这些模型的部分实验在孤儿进程影响区间内
- 完整的 hrnet18 和 pcb 数据需要重新运行实验

## 使用示例

### Python 读取

```python
import pandas as pd

# 读取汇总数据
df = pd.read_csv('results/summary_all.csv')

print(f"总实验数: {len(df)}")
print(f"成功率: {df['training_success'].mean():.1%}")

# 按来源分组
for source, group in df.groupby('experiment_source'):
    print(f"\n{source}: {len(group)}个实验")
    print(f"  平均GPU功率: {group['energy_gpu_avg_watts'].mean():.2f}W")
    print(f"  平均训练时长: {group['duration_seconds'].mean():.2f}s")

# 按模型分析
model_stats = df.groupby(['repository', 'model']).agg({
    'experiment_id': 'count',
    'energy_gpu_avg_watts': 'mean',
    'energy_gpu_total_joules': 'mean'
})
print("\n模型统计:")
print(model_stats)
```

### 筛选特定来源

```python
# 仅分析1x变异数据
mutation_1x = df[df['experiment_source'] == 'mutation_1x']

# 仅分析2x变异安全数据
mutation_2x_safe = df[df['experiment_source'] == 'mutation_2x_safe']

# 对比默认和变异实验
default_data = df[df['experiment_source'] == 'default']
mutation_data = df[df['experiment_source'].isin(['mutation_1x', 'mutation_2x_safe'])]
```

### 恢复原始实验ID

```python
# 从带前缀的ID中提取原始ID
df['original_experiment_id'] = df['experiment_id'].str.split('__').str[1]
```

## 数据质量说明

### 高质量数据

✅ **所有211个实验都是成功的**
- 过滤了所有失败的mnist_ff实验（batch_size配置问题）
- 使用了2x变异测试的安全数据（孤儿进程影响前）
- 每个实验都有完整的能耗和性能数据

### 数据限制

⚠️ **部分模型数据不完整**:
- **Person_reID/hrnet18**: 只有11个实验（完整应为16-26个）
- **Person_reID/pcb**: 只有10个实验（完整应为16-26个）
- **原因**: 这些模型的部分实验在孤儿进程影响区间（2025-11-25之后）

⚠️ **mnist_ff数据缺失**:
- 所有mnist_ff实验都被过滤（共20个）
- 原因：使用了不当的batch_size=50000
- 已修复：新的默认batch_size=10000
- 建议：重新运行mnist_ff实验

## 相关文档

- **2x变异测试详情**: `results/mutation_2x_20251122_175401/README.md`
- **数据整理总结**: `docs/MUTATION_2X_DATA_SUMMARY.md`
- **GPU清理机制**: `docs/GPU_MEMORY_CLEANUP_FIX.md`
- **孤儿进程分析**: `docs/ORPHAN_PROCESS_ANALYSIS.md`

## 生成脚本

本文件由以下脚本生成：

```bash
# 生成汇总CSV（过滤mnist_ff）
python3 scripts/aggregate_csvs.py --output results/summary_all.csv

# 生成汇总CSV（保留mnist_ff）
python3 scripts/aggregate_csvs.py --output results/summary_all_with_mnist_ff.csv --keep-mnist-ff

# 运行测试验证
python3 tests/functional/test_aggregate_csvs.py
```

## 更新记录

- **2025-11-26**: 初始版本
  - 汇总 default, mutation_1x, mutation_2x_safe 三次实验
  - 过滤 mnist_ff 实验（batch_size配置问题）
  - 添加 experiment_source 列
  - 实验ID唯一化处理

---

**维护者**: Mutation-Based Training Energy Profiler Team
**版本**: v1.0
**生成时间**: 2025-11-26
