# Summary_all.csv 数据整理方案

**方案版本**: v1.0
**创建日期**: 2025-12-11
**项目版本**: v4.7.2
**当前数据**: 476条记录, 37列

---

## 📊 当前数据状态分析

### 基本信息

```
总记录数: 476
总列数: 37
有效模型: 11个
数据完整性: 能耗100%, 超参数/性能指标稀疏(模型特定)
```

### 数据完整性评估

| 类别 | 完整性 | 说明 |
|------|--------|------|
| **基本信息** | 100% | experiment_id, timestamp, repository, model, training_success, duration |
| **能耗数据** | 100% | 所有11个能耗指标完整 |
| **超参数** | 稀疏 | 每个模型只使用4-5个超参数(2.5%-31.7%) |
| **性能指标** | 稀疏 | 模型特定指标(7.8%-34.0%) |
| **实验来源** | 44.3% | experiment_source字段部分缺失 |

---

## 🎯 数据整理目标

### 1. 可读性改进

- 优化列顺序,将相关列分组
- 添加派生列(计算能耗效率、归一化指标等)
- 改善列命名(更直观的名称)

### 2. 信息增强

- 添加模型分类标签
- 计算能耗效率指标
- 添加超参数变化标记
- 补充缺失的元数据

### 3. 分析友好

- 创建针对不同分析任务的视图
- 提供多种格式输出(CSV, Excel, JSON)
- 生成数据字典和说明文档

---

## 📋 方案A: 基础重组 (推荐首选)

### 目标
重新组织列顺序,使CSV更易读,不改变数据内容。

### 建议的列顺序

```
第一组: 实验标识 (5列)
  1. experiment_id
  2. timestamp
  3. repository
  4. model
  5. experiment_source (并行/非并行标记)

第二组: 实验结果 (2列)
  6. training_success
  7. duration_seconds

第三组: 超参数 (9列,按常见性排序)
  8. hyperparam_epochs
  9. hyperparam_learning_rate
  10. hyperparam_batch_size
  11. hyperparam_dropout
  12. hyperparam_weight_decay
  13. hyperparam_seed
  14. hyperparam_alpha
  15. hyperparam_kfold
  16. hyperparam_max_iter

第四组: 性能指标 (9列,按类型分组)
  17. perf_test_accuracy
  18. perf_best_val_accuracy
  19. perf_accuracy
  20. perf_test_loss
  21. perf_map
  22. perf_rank1
  23. perf_rank5
  24. perf_precision
  25. perf_recall

第五组: CPU能耗 (3列)
  26. energy_cpu_pkg_joules
  27. energy_cpu_ram_joules
  28. energy_cpu_total_joules

第六组: GPU能耗 - 功耗 (4列)
  29. energy_gpu_total_joules
  30. energy_gpu_avg_watts
  31. energy_gpu_max_watts
  32. energy_gpu_min_watts

第七组: GPU状态 (4列)
  33. energy_gpu_util_avg_percent
  34. energy_gpu_util_max_percent
  35. energy_gpu_temp_avg_celsius
  36. energy_gpu_temp_max_celsius

第八组: 其他 (1列)
  37. retries
```

### 优点
- ✅ 不改变数据内容,安全性高
- ✅ 相关字段分组,查找容易
- ✅ 实现简单,快速
- ✅ 向后兼容,可轻松恢复原顺序

### 实现
```python
# 定义新的列顺序
new_column_order = [
    # 实验标识
    'experiment_id', 'timestamp', 'repository', 'model', 'experiment_source',
    # 实验结果
    'training_success', 'duration_seconds',
    # 超参数
    'hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_batch_size',
    'hyperparam_dropout', 'hyperparam_weight_decay', 'hyperparam_seed',
    'hyperparam_alpha', 'hyperparam_kfold', 'hyperparam_max_iter',
    # 性能指标
    'perf_test_accuracy', 'perf_best_val_accuracy', 'perf_accuracy',
    'perf_test_loss', 'perf_map', 'perf_rank1', 'perf_rank5',
    'perf_precision', 'perf_recall',
    # CPU能耗
    'energy_cpu_pkg_joules', 'energy_cpu_ram_joules', 'energy_cpu_total_joules',
    # GPU能耗
    'energy_gpu_total_joules', 'energy_gpu_avg_watts', 'energy_gpu_max_watts',
    'energy_gpu_min_watts',
    # GPU状态
    'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent',
    'energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
    # 其他
    'retries'
]

# 重新排序并保存
df = df[new_column_order]
df.to_csv('summary_all_reorganized.csv', index=False)
```

---

## 📋 方案B: 增强版 (推荐用于分析)

### 目标
在方案A基础上,添加派生列和计算字段,增强数据信息量。

### 新增列建议

#### 1. 模型分类列 (4列新增)

```python
# 添加在model列后
- model_type: 模型类型 (CNN/RNN/Transformer/MLP等)
- model_size: 模型规模 (Small/Medium/Large)
- dataset: 数据集名称 (MNIST/CIFAR-10/Market-1501等)
- task: 任务类型 (Classification/ReID/VulnerabilityDetection等)
```

**示例映射**:
```python
model_metadata = {
    'examples/mnist': {
        'model_type': 'CNN',
        'model_size': 'Small',
        'dataset': 'MNIST',
        'task': 'Classification'
    },
    'VulBERTa/mlp': {
        'model_type': 'Transformer+MLP',
        'model_size': 'Large',
        'dataset': 'D2A',
        'task': 'VulnerabilityDetection'
    },
    # ... 其他模型
}
```

#### 2. 模式标识列 (1列新增)

```python
# 添加在experiment_source后
- training_mode: 从experiment_id提取 (parallel/nonparallel)
```

#### 3. 能耗效率列 (5列新增)

```python
# 添加在GPU状态后
- energy_total_joules: CPU总能耗 + GPU总能耗
- energy_per_second: 总能耗 / 训练时长 (Watts平均功耗)
- energy_per_epoch: 总能耗 / epochs数 (仅对有epochs的模型)
- gpu_efficiency: GPU利用率均值 * GPU功耗均值 (衡量GPU使用效率)
- energy_performance_ratio: 总能耗 / 性能指标 (能耗-性能比,越小越好)
```

#### 4. 归一化指标列 (可选,2列)

```python
# 在性能指标后添加
- perf_primary_metric: 主要性能指标(统一名称,便于跨模型比较)
- perf_normalized: 归一化后的性能指标(0-1范围)
```

**示例**:
```python
# 每个模型的主要指标
primary_metrics = {
    'examples/mnist': 'perf_test_accuracy',
    'Person_reID_baseline_pytorch/densenet121': 'perf_rank1',
    'bug-localization-by-dnn-and-rvsm/default': 'perf_map',
    # ...
}
```

### 完整列顺序 (方案B: 37 + 12 = 49列)

```
1-5:   实验标识 (原5列)
6:     training_mode (新增)
7-8:   实验结果 (原2列)
9-12:  模型元数据 (新增4列: model_type, model_size, dataset, task)
13-21: 超参数 (原9列)
22-23: 归一化性能 (新增2列: perf_primary_metric, perf_normalized)
24-32: 性能指标 (原9列)
33-35: CPU能耗 (原3列)
36-39: GPU能耗 (原4列)
40-43: GPU状态 (原4列)
44-48: 能耗效率 (新增5列)
49:    retries (原1列)
```

### 优点
- ✅ 更丰富的分析维度
- ✅ 跨模型比较更容易
- ✅ 能耗效率指标直观
- ✅ 保留原始数据完整性

### 缺点
- ⚠️ 列数增加(37→49)
- ⚠️ 需要编写映射规则
- ⚠️ 计算派生字段需要额外处理

---

## 📋 方案C: 多视图方案 (推荐用于复杂分析)

### 目标
保持原始CSV完整,生成多个针对不同分析任务的专用视图。

### 视图1: 能耗分析视图 (18列)

```
experiment_id, repository, model, training_mode,
duration_seconds, hyperparam_epochs,
energy_cpu_total_joules, energy_gpu_total_joules,
energy_total_joules, energy_per_second, energy_per_epoch,
gpu_avg_watts, gpu_util_avg_percent, gpu_temp_avg_celsius,
perf_primary_metric, perf_normalized,
energy_performance_ratio
```

**用途**: 能耗分析、能效优化、超参数对能耗的影响研究

### 视图2: 性能分析视图 (20列)

```
experiment_id, repository, model, training_mode,
duration_seconds,
所有超参数列 (9列),
所有性能指标列 (9列),
perf_primary_metric
```

**用途**: 超参数调优、性能优化、模型比较

### 视图3: 超参数影响视图 (按模型分组)

每个模型一个CSV文件:
```
model, training_mode,
活跃超参数 (4-5列),
perf_primary_metric,
energy_total_joules,
energy_per_second,
duration_seconds
```

**用途**: 单模型深度分析、超参数敏感性分析

### 视图4: 紧凑总览视图 (12列)

```
experiment_id, repository, model, training_mode,
duration_seconds, epochs,
perf_primary_metric,
energy_cpu_total_joules, energy_gpu_total_joules,
energy_total_joules, energy_per_second,
energy_performance_ratio
```

**用途**: 快速浏览、高层次对比、报告生成

### 优点
- ✅ 原始数据不变
- ✅ 针对性强,每个视图专注特定分析
- ✅ 文件更小,加载更快
- ✅ 可根据需要生成新视图

### 实现
```python
# 生成能耗分析视图
energy_view = df[energy_analysis_columns].copy()
energy_view.to_csv('views/summary_energy_analysis.csv', index=False)

# 生成性能分析视图
performance_view = df[performance_columns].copy()
performance_view.to_csv('views/summary_performance_analysis.csv', index=False)

# 按模型生成超参数影响视图
for model in df['repository/model'].unique():
    model_data = df[df['repository/model'] == model]
    # 只保留该模型使用的超参数列
    model_data.to_csv(f'views/hyperparams_{model.replace("/", "_")}.csv')
```

---

## 📋 方案D: Excel多表格方案

### 目标
生成一个Excel文件,包含多个工作表,便于交互式分析。

### Excel结构

```
summary_all.xlsx
├── Sheet1: 完整数据 (所有476行×37列)
├── Sheet2: 能耗分析 (476行×18列)
├── Sheet3: 性能分析 (476行×20列)
├── Sheet4: 模型汇总 (11行×统计列)
├── Sheet5: 超参数统计 (各超参数的分布统计)
├── Sheet6: 数据字典 (列名解释)
└── Sheet7: 模型元数据 (11个模型的详细信息)
```

### 增强功能

1. **条件格式**: 性能指标高亮,能耗异常标红
2. **数据透视表**: 预配置的分析透视表
3. **图表**: 自动生成的能耗/性能对比图
4. **筛选器**: 每列添加自动筛选
5. **冻结窗格**: 冻结标题行和ID列

### 优点
- ✅ 交互性强,适合探索性分析
- ✅ 可视化内置
- ✅ 非技术用户友好
- ✅ 单文件包含所有信息

### 缺点
- ⚠️ 文件较大
- ⚠️ 需要Excel或兼容软件
- ⚠️ 版本控制不友好

---

## 🎯 推荐实施方案

### 阶段1: 基础重组 (立即执行)

**实施方案A**:
1. 重新排列列顺序
2. 生成 `summary_all_reorganized.csv`
3. 保留原始 `summary_all.csv` 作为备份

**预期时间**: <5分钟
**风险**: 极低

### 阶段2: 信息增强 (短期)

**实施方案B**:
1. 添加模型元数据列
2. 计算能耗效率指标
3. 添加归一化性能指标
4. 生成 `summary_all_enhanced.csv`

**预期时间**: ~30分钟
**风险**: 低(需验证计算逻辑)

### 阶段3: 多视图生成 (中期)

**实施方案C**:
1. 创建 `views/` 目录
2. 生成4个专用视图CSV
3. 创建数据字典文档

**预期时间**: ~1小时
**风险**: 低

### 阶段4: Excel版本 (可选)

**实施方案D**:
1. 安装openpyxl/xlsxwriter
2. 生成多sheet Excel文件
3. 添加格式化和图表

**预期时间**: ~2小时
**风险**: 中(依赖第三方库)

---

## 📊 具体列映射建议

### 模型元数据映射

```python
MODEL_METADATA = {
    'examples/mnist': {
        'model_type': 'CNN',
        'model_size': 'Small',
        'dataset': 'MNIST',
        'task': 'Image Classification',
        'primary_metric': 'perf_test_accuracy'
    },
    'examples/mnist_ff': {
        'model_type': 'Feed-Forward NN',
        'model_size': 'Small',
        'dataset': 'MNIST',
        'task': 'Image Classification',
        'primary_metric': 'perf_test_accuracy'
    },
    'examples/mnist_rnn': {
        'model_type': 'RNN',
        'model_size': 'Small',
        'dataset': 'MNIST',
        'task': 'Image Classification',
        'primary_metric': 'perf_test_accuracy'
    },
    'examples/siamese': {
        'model_type': 'Siamese CNN',
        'model_size': 'Small',
        'dataset': 'MNIST (paired)',
        'task': 'Similarity Learning',
        'primary_metric': 'perf_test_accuracy'
    },
    'Person_reID_baseline_pytorch/densenet121': {
        'model_type': 'CNN (DenseNet)',
        'model_size': 'Medium',
        'dataset': 'Market-1501',
        'task': 'Person Re-Identification',
        'primary_metric': 'perf_rank1'
    },
    'Person_reID_baseline_pytorch/hrnet18': {
        'model_type': 'CNN (HRNet)',
        'model_size': 'Medium',
        'dataset': 'Market-1501',
        'task': 'Person Re-Identification',
        'primary_metric': 'perf_rank1'
    },
    'Person_reID_baseline_pytorch/pcb': {
        'model_type': 'CNN (Part-based)',
        'model_size': 'Medium',
        'dataset': 'Market-1501',
        'task': 'Person Re-Identification',
        'primary_metric': 'perf_rank1'
    },
    'VulBERTa/mlp': {
        'model_type': 'Transformer + MLP',
        'model_size': 'Large',
        'dataset': 'D2A',
        'task': 'Vulnerability Detection',
        'primary_metric': 'perf_accuracy'
    },
    'pytorch_resnet_cifar10/resnet20': {
        'model_type': 'CNN (ResNet)',
        'model_size': 'Small',
        'dataset': 'CIFAR-10',
        'task': 'Image Classification',
        'primary_metric': 'perf_test_accuracy'
    },
    'MRT-OAST/default': {
        'model_type': 'Custom NN',
        'model_size': 'Medium',
        'dataset': 'MRT-OAST',
        'task': 'Code Analysis',
        'primary_metric': 'perf_accuracy'
    },
    'bug-localization-by-dnn-and-rvsm/default': {
        'model_type': 'DNN + RVSM',
        'model_size': 'Medium',
        'dataset': 'Bug Reports',
        'task': 'Bug Localization',
        'primary_metric': 'perf_map'
    }
}
```

### 能耗效率计算

```python
# 总能耗
df['energy_total_joules'] = df['energy_cpu_total_joules'] + df['energy_gpu_total_joules']

# 平均功耗 (Watts)
df['energy_per_second'] = df['energy_total_joules'] / df['duration_seconds']

# 每epoch能耗 (仅对有epochs的记录)
df['energy_per_epoch'] = df.apply(
    lambda row: row['energy_total_joules'] / row['hyperparam_epochs']
    if pd.notna(row['hyperparam_epochs']) and row['hyperparam_epochs'] > 0
    else None,
    axis=1
)

# GPU效率 (利用率 × 功耗)
df['gpu_efficiency'] = df['energy_gpu_util_avg_percent'] * df['energy_gpu_avg_watts'] / 100

# 能耗-性能比 (需要先统一性能指标)
df['energy_performance_ratio'] = df.apply(
    lambda row: row['energy_total_joules'] / row['perf_primary_metric']
    if pd.notna(row['perf_primary_metric']) and row['perf_primary_metric'] > 0
    else None,
    axis=1
)
```

---

## ✅ 验证和质量检查

### 数据一致性检查

```python
# 1. 检查行数是否一致
assert len(df_reorganized) == 476

# 2. 检查列数
assert len(df_reorganized.columns) == expected_columns

# 3. 检查必填列完整性
required_cols = ['experiment_id', 'timestamp', 'repository', 'model']
for col in required_cols:
    assert df_reorganized[col].notna().all()

# 4. 检查能耗数据完整性
energy_cols = [c for c in df_reorganized.columns if c.startswith('energy_')]
for col in energy_cols:
    assert df_reorganized[col].notna().all()

# 5. 检查计算字段合理性
assert (df_reorganized['energy_total_joules'] >= 0).all()
assert (df_reorganized['energy_per_second'] > 0).all()
```

---

## 📚 配套文档

### 1. 数据字典 (data_dictionary.md)

为每一列提供详细说明:
- 列名
- 数据类型
- 单位
- 取值范围
- 计算方法(派生列)
- 使用示例

### 2. 列映射文档 (column_mapping.md)

说明原始列到新列的映射关系:
- 原始列顺序 → 新列顺序
- 新增列及其来源
- 删除列及原因(如有)

### 3. 使用指南 (analysis_guide.md)

针对不同分析任务的数据使用建议:
- 能耗分析: 使用哪些列
- 性能优化: 使用哪些列
- 超参数调优: 使用哪些列

---

## 🎯 总结与建议

### 推荐方案组合

**快速开始** (今天):
- 方案A: 基础重组,生成reorganized版本

**深度分析** (本周):
- 方案B: 增强版,添加派生列和元数据
- 方案C: 多视图,生成专用分析文件

**交互探索** (可选):
- 方案D: Excel版本,便于非编程分析

### 下一步行动

1. ✅ 确认方案选择
2. ✅ 验证模型元数据映射
3. ✅ 编写数据转换脚本
4. ✅ 测试和验证
5. ✅ 生成配套文档

---

**方案作者**: Claude (v4.7.2)
**创建日期**: 2025-12-11
**数据版本**: summary_all.csv (476条记录)
**状态**: 待确认实施
