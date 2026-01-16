# DiBS数据使用快速指南

**最后更新**: 2026-01-15

## 数据文件位置

```
/home/green/energy_dl/nightly/analysis/data/energy_research/dibs_training/
├── group1_examples.csv          # 126样本, 4超参数 ⭐ 最佳
├── group2_vulberta.csv          # 52样本, 0超参数 ⚠️ 需清理
├── group3_person_reid.csv       # 118样本, 4超参数 ⭐ 最佳
├── group4_bug_localization.csv  # 40样本, 0超参数
├── group5_mrt_oast.csv          # 46样本, 0超参数
├── group6_resnet.csv            # 41样本, 4超参数
├── generation_stats.json        # 生成统计
├── detailed_quality_analysis.json  # 详细质量分析
├── DATA_QUALITY_ASSESSMENT_20260115.md  # 详细评估报告
├── QUICK_QUALITY_SUMMARY.md     # 快速总结
└── USAGE_GUIDE.md               # 本文件
```

---

## 研究问题数据选择

### 问题1: 超参数对能耗的影响

**使用文件**（选择任一或全部）:

```python
# 推荐1: examples组（最佳）
df = pd.read_csv('group1_examples.csv')  # 126样本, 4超参数

# 推荐2: Person_reID组（超参数多样性最高）
df = pd.read_csv('group3_person_reid.csv')  # 118样本, 4超参数

# 推荐3: ResNet组
df = pd.read_csv('group6_resnet.csv')  # 41样本, 4超参数

# 合并所有3组（推荐用于综合分析）
df1 = pd.read_csv('group1_examples.csv')
df2 = pd.read_csv('group3_person_reid.csv')
df3 = pd.read_csv('group6_resnet.csv')
# 注意：需要先对齐特征名，因为不同组的特征可能略有差异
```

**特征说明**:
- 超参数: hyperparam_* (如 hyperparam_learning_rate)
- 能耗指标: energy_* (如 energy_gpu_total_joules)
- 性能指标: perf_* (如 perf_test_accuracy)
- 控制变量: duration_seconds, num_mutated_params

### 问题2: 能耗和性能的权衡关系

**使用文件**（推荐使用所有就绪组）:

```python
# 加载所有5个就绪组
groups = [
    'group1_examples.csv',        # 126样本
    'group3_person_reid.csv',     # 118样本
    'group4_bug_localization.csv', # 40样本
    'group5_mrt_oast.csv',        # 46样本
    'group6_resnet.csv'           # 41样本
]

dfs = []
for group in groups:
    df = pd.read_csv(group)
    dfs.append(df)

# 注意：不同组的性能指标不同，需要分组分析或标准化
```

**注意事项**:
- 不同组的性能指标不同（accuracy, loss, map等）
- 建议分组分析或使用通用的标准化方法
- 能耗指标在所有组中一致（11个指标）

### 问题3: 中间变量的中介效应

**使用文件**（推荐高质量组）:

```python
# 推荐1: examples组（最完整）
df = pd.read_csv('group1_examples.csv')  # 4超参数 → 11能耗 → 1性能

# 推荐2: Person_reID组（最丰富）
df = pd.read_csv('group3_person_reid.csv')  # 4超参数 → 11能耗 → 3性能

# 合并分析
df1 = pd.read_csv('group1_examples.csv')
df2 = pd.read_csv('group3_person_reid.csv')
```

**因果路径示例**:
```
超参数 → 能耗指标 → 性能指标
例如: learning_rate → energy_gpu_total_joules → test_accuracy
```

---

## 数据预处理步骤

### 标准工作流（推荐）

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 加载数据
df = pd.read_csv('group1_examples.csv')

# 2. 检查数据质量
print(f"样本数: {len(df)}")
print(f"特征数: {len(df.columns)}")
print(f"缺失值: {df.isnull().sum().sum()}")

# 3. 移除常数特征（如果有）
variance = df.var()
constant_cols = variance[variance == 0].index.tolist()
if constant_cols:
    print(f"移除常数特征: {constant_cols}")
    df = df.drop(columns=constant_cols)

# 4. 标准化（DiBS推荐）
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)

# 5. 验证
print(f"标准化后均值: {df_scaled.mean().mean():.6f}")  # 应接近0
print(f"标准化后标准差: {df_scaled.std().mean():.6f}")  # 应接近1

# 6. 输入到DiBS
# X = df_scaled.values
# 运行DiBS因果图学习...
```

### VulBERTa组特殊处理

```python
# VulBERTa组需要移除常数特征
df = pd.read_csv('group2_vulberta.csv')

# 移除常数特征
df_clean = df.drop(columns=['energy_gpu_util_max_percent'])

# 后续步骤同上
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
```

### 小样本组稳定性处理

```python
from sklearn.model_selection import KFold
from sklearn.utils import resample

# 对于样本量<50的组，使用k-fold交叉验证
df = pd.read_csv('group6_resnet.csv')  # 41样本

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    
    # 标准化训练集
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    
    # 在train_scaled上训练DiBS
    # graph = dibs.learn(train_scaled)
    # results.append(graph)
    
print(f"交叉验证完成，共{len(results)}个模型")
# 分析results的稳定性
```

---

## 特征说明

### 超参数特征（hyperparam_*）

**group1_examples（4个）**:
- batch_size: 批量大小 [19, 10000]
- epochs: 训练轮数 [5, 15]
- learning_rate: 学习率 [0.0056, 0.0184]
- seed: 随机种子 [1, 9809]

**group3_person_reid（4个）**:
- dropout: dropout率 [0.30, 0.59]
- epochs: 训练轮数 [31, 90]
- learning_rate: 学习率 [0.025, 0.075]
- seed: 随机种子 [3, 9974]

**group6_resnet（4个）**:
- epochs: 训练轮数 [108, 297]
- learning_rate: 学习率 [0.051, 0.135]
- seed: 随机种子 [409, 9992]
- weight_decay: 权重衰减 [0.000011, 0.00066]

### 能耗特征（energy_*）（所有组一致，11个）

**CPU能耗**:
- energy_cpu_pkg_joules: CPU包能耗（焦耳）
- energy_cpu_ram_joules: CPU内存能耗（焦耳）
- energy_cpu_total_joules: CPU总能耗（焦耳）

**GPU功耗**:
- energy_gpu_avg_watts: GPU平均功率（瓦特）
- energy_gpu_max_watts: GPU最大功率（瓦特）
- energy_gpu_min_watts: GPU最小功率（瓦特）
- energy_gpu_total_joules: GPU总能耗（焦耳）

**GPU温度**:
- energy_gpu_temp_avg_celsius: GPU平均温度（摄氏度）
- energy_gpu_temp_max_celsius: GPU最大温度（摄氏度）

**GPU利用率**:
- energy_gpu_util_avg_percent: GPU平均利用率（百分比）
- energy_gpu_util_max_percent: GPU最大利用率（百分比）

### 性能特征（perf_*）（各组不同）

**group1_examples（1个）**:
- perf_test_accuracy: 测试准确率

**group3_person_reid（3个）**:
- perf_map: 平均精度均值
- perf_rank1: Rank-1准确率
- perf_rank5: Rank-5准确率

**group6_resnet（2个）**:
- perf_best_val_accuracy: 最佳验证准确率
- perf_test_accuracy: 测试准确率

### 控制变量

- duration_seconds: 训练持续时间（秒）
- num_mutated_params: 变异参数数量

---

## 常见问题

### Q1: 为什么有些组没有超参数？

**A**: 3个组（VulBERTa、bug-localization、MRT-OAST）在数据生成时，原始数据中缺少超参数列（缺失率>40%），因此被移除。这些组只能用于能耗-性能关系分析（问题2），无法研究超参数的因果影响（问题1/3）。

### Q2: 样本量40-46够用吗？

**A**: 满足DiBS最低要求（≥30），但略低于推荐值（50+）。建议：
- 使用k-fold交叉验证评估稳定性
- 或使用bootstrap方法评估不确定性
- 或优先使用大样本组（examples: 126, Person_reID: 118）

### Q3: 如何合并多个组的数据？

**A**: 注意事项：
- 不同组的性能指标不同，需要先对齐
- 超参数也可能不同，需要处理缺失
- 建议分组分析或使用multi-group DiBS方法

```python
# 方法1: 只保留公共特征
df1 = pd.read_csv('group1_examples.csv')
df2 = pd.read_csv('group3_person_reid.csv')

# 找到公共列（能耗指标通常一致）
common_cols = list(set(df1.columns) & set(df2.columns))
df_combined = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)

# 方法2: 分组分析（推荐）
for group_file in ['group1_examples.csv', 'group3_person_reid.csv']:
    df = pd.read_csv(group_file)
    # 在每组上独立运行DiBS
    # 然后比较结果
```

### Q4: 需要处理异常值吗？

**A**: 取决于分析目的：
- DiBS对异常值相对鲁棒，通常不需要特殊处理
- 但如果异常值率>20%（如examples组的某些超参数），可考虑：
  - 使用robust标准化（RobustScaler）
  - 或Winsorization（截断极端值）
  - 或保持原样，让DiBS自动处理

### Q5: VulBERTa组可以用吗？

**A**: 可以，但需要先移除常数特征：

```python
df = pd.read_csv('group2_vulberta.csv')
df_clean = df.drop(columns=['energy_gpu_util_max_percent'])
# 现在可以使用，但只能分析能耗-性能关系（无超参数）
```

---

## 快速开始示例

### 示例1: 分析learning_rate对能耗的影响（问题1）

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
# import dibs  # 假设已安装DiBS

# 加载最佳组
df = pd.read_csv('group1_examples.csv')

# 选择相关特征
features = ['hyperparam_learning_rate', 'energy_gpu_total_joules', 'perf_test_accuracy']
df_subset = df[features]

# 标准化
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_subset), columns=df_subset.columns)

# 运行DiBS
# dag = dibs.learn(df_scaled.values, ...)

print(f"分析{len(df)}个样本的因果关系")
print(f"特征: {features}")
```

### 示例2: 比较不同任务的能耗-性能权衡（问题2）

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载多个组
groups = {
    'examples': 'group1_examples.csv',
    'person_reid': 'group3_person_reid.csv',
    'resnet': 'group6_resnet.csv'
}

results = {}
for name, file in groups.items():
    df = pd.read_csv(file)
    
    # 提取能耗和性能（假设第一个性能指标）
    energy = df['energy_gpu_total_joules']
    perf_cols = [col for col in df.columns if col.startswith('perf_')]
    performance = df[perf_cols[0]]
    
    results[name] = {'energy': energy, 'performance': performance}
    
    # 可视化
    plt.scatter(energy, performance, label=name, alpha=0.6)

plt.xlabel('GPU Total Energy (Joules)')
plt.ylabel('Performance')
plt.legend()
plt.title('Energy-Performance Tradeoff Across Tasks')
plt.show()

# 在每组上运行DiBS
for name, file in groups.items():
    df = pd.read_csv(file)
    # 选择能耗和性能特征
    energy_cols = [col for col in df.columns if col.startswith('energy_')]
    perf_cols = [col for col in df.columns if col.startswith('perf_')]
    df_subset = df[energy_cols + perf_cols]
    
    # 标准化并运行DiBS
    # ...
```

---

## 相关文档

- **详细评估报告**: `DATA_QUALITY_ASSESSMENT_20260115.md`
- **快速总结**: `QUICK_QUALITY_SUMMARY.md`
- **生成统计**: `generation_stats.json`
- **详细分析**: `detailed_quality_analysis.json`

---

**生成时间**: 2026-01-15
**维护者**: Green
