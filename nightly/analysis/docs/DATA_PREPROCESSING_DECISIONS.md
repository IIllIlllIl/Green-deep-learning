# 数据预处理决策文档

**文档版本**: v1.0
**创建时间**: 2025-12-22
**状态**: ✅ 方案确认

---

## 📋 执行摘要

本文档记录因果分析数据预处理的三个关键决策：
1. **数据完整性与缺失值处理** - 从676行筛选到284行严格完整数据
2. **敏感属性二值化** - mode编码为0/1（非并行=0，并行=1）
3. **类别变量数值化** - One-Hot编码方案

---

## 问题1：数据完整性与缺失值处理

### 1.1 原始数据状况

**总数据**: 676行实验

**数据结构差异**：
- **非并行实验** (348行)：数据存储在主列（如 `energy_cpu_total_joules`, `perf_test_accuracy`）
- **并行实验** (328行)：数据存储在fg_*列（如 `fg_energy_cpu_total_joules`, `fg_perf_test_accuracy`）

**数据完整性统计**（使用统一提取逻辑）：

| 数据类别 | 完整行数 | 百分比 |
|---------|---------|-------|
| 有能耗数据 | 537/676 | 79.4% |
| 有性能数据 | 391/676 | 57.8% |
| **能耗+性能** | **465/676** | **68.8%** |

### 1.2 关键列缺失率分析

基于465行有能耗+性能的数据：

#### ✅ 完全完整（100%）

- 能耗指标：`energy_cpu_total_joules`, `energy_gpu_total_joules`
- 能耗中介：`energy_gpu_util_avg_percent`, `energy_gpu_temp_max_celsius`, etc.（5个变量）
- 分组标识：`repository`, `model`

#### ⚠️ 部分缺失

- `hyperparam_learning_rate`: 57.6%
- `hyperparam_epochs`: 58.9%
- `hyperparam_batch_size`: 29.0%
- `mode`: **53.5%** ← 主要缺失来源

#### 🔍 mode列缺失分析

```
mode列取值分布：
  (empty)   : 348个 (51.5%) ← 非并行实验未填充mode
  parallel  : 328个 (48.5%) ← 并行实验有mode值
```

**根本原因**：非并行实验的mode列未填充，但语义上应为"default"（或"single"）。

### 1.3 ✅ 决策：缺失值处理方案

#### 方案确认

**采用"mode填充 + 严格过滤"策略**：

1. **mode列填充规则**：
   ```python
   if row['mode'] == '' or not row['mode']:
       row['mode'] = 'default'  # 填充为'default'表示非并行
   ```

2. **严格完整性标准**（保留以下行）：
   - ✅ 能耗数据：`energy_cpu_total_joules` AND `energy_gpu_total_joules` 都有值
   - ✅ 性能数据：至少一个性能指标有值（`perf_test_accuracy` OR `perf_map` OR `perf_eval_loss` OR `perf_top1_accuracy`）
   - ✅ 训练时长：`hyperparam_epochs` OR `hyperparam_max_iter` 至少一个有值
   - ✅ 分组标识：`repository` AND `model` 都有值
   - ✅ 敏感属性：`mode` 有值（填充后100%满足）

3. **删除缺失值**：
   - 不满足上述任一条件的行，全部删除

#### 处理后数据量

```
原始数据:       676行
严格完整数据:   284行 (42.0%)
删除缺失值:     392行 (58.0%)
```

#### 各任务组样本量

| 任务组 | 样本量 | DiBS可行性 |
|-------|-------|-----------|
| examples (MNIST系列) | 153个 | ✅ 充足 |
| Person_reID_baseline_pytorch | 86个 | ✅ 充足 |
| pytorch_resnet_cifar10 | 21个 | ✅ 可行 |
| VulBERTa | 14个 | ✅ 可行 |
| bug-localization-by-dnn-and-rvsm | 10个 | ✅ 最低线（刚好满足） |

**结论**: 所有任务组都满足DiBS最低要求（10个样本）✅

---

## 问题2：敏感属性二值化（mode编码）

### 2.1 背景

因果推断（特别是公平性分析）要求**敏感属性**必须是**二值**（0/1）。

在我们的研究中：
- **敏感属性** = 训练模式（非并行 vs 并行）
- **研究问题**: 并行训练模式是否会影响能耗和性能？

### 2.2 ✅ 决策：mode二值化编码

#### 编码规则

```python
mode_encoding = {
    'default': 0,   # 非并行模式
    '':        0,   # 空值填充为default后编码为0
    'parallel': 1   # 并行模式
}
```

#### 语义解释

| mode原始值 | 填充后 | 编码值 | 语义 |
|-----------|--------|-------|------|
| `''` (空) | `'default'` | **0** | 单任务训练（非并行） |
| `'default'` | `'default'` | **0** | 单任务训练（非并行） |
| `'parallel'` | `'parallel'` | **1** | 并行训练（前台+后台） |

#### 数据分布

基于284行严格完整数据：

```
mode=0 (非并行):  145个 (51.1%)
mode=1 (并行):    139个 (48.9%)
```

**结论**: 两种模式分布均衡，适合因果分析 ✅

#### 实现代码

```python
def binarize_mode(df):
    """mode二值化编码"""

    # 1. 填充空值
    df['mode'] = df['mode'].fillna('default')
    df['mode'] = df['mode'].replace('', 'default')

    # 2. 编码为0/1
    df['mode_binary'] = df['mode'].map({
        'default': 0,
        'parallel': 1
    })

    # 3. 验证
    assert df['mode_binary'].notna().all(), "存在无法编码的mode值"
    assert df['mode_binary'].isin([0, 1]).all(), "mode_binary必须是0或1"

    return df
```

---

## 问题3：One-Hot编码（类别变量数值化）

### 3.1 什么是One-Hot编码？

**One-Hot编码**（独热编码）是将**类别变量**转换为**数值型**的标准方法。

#### 核心思想

将一个有N个类别的变量，转换为N个二进制列（0/1），每个类别对应一列。

#### 示例1：水果类别

**原始数据**:
```
水果: ['苹果', '香蕉', '苹果', '橙子']
```

**One-Hot编码后**:
```
苹果  香蕉  橙子
 1    0    0     ← 第1行：苹果
 0    1    0     ← 第2行：香蕉
 1    0    0     ← 第3行：苹果
 0    0    1     ← 第4行：橙子
```

**特点**：
- 每行只有一个1（one-hot = 只有一个"热"位）
- 其余全是0
- 3个类别 → 3列二进制变量

#### 示例2：我们的repository列

**原始数据**（5个类别）:
```
repository: ['examples', 'VulBERTa', 'examples', 'Person_reID', ...]
```

**One-Hot编码后**（5列）:
```
repo_examples  repo_VulBERTa  repo_Person_reID  repo_pytorch_resnet  repo_bug_loc
    1              0               0                 0                  0
    0              1               0                 0                  0
    1              0               0                 0                  0
    0              0               1                 0                  0
```

### 3.2 为什么需要One-Hot编码？

#### 原因1：DiBS要求数值型输入

DiBS（因果图学习）算法要求所有输入变量必须是**数值型**（float或int）。

类别变量（如repository='VulBERTa'）无法直接输入DiBS。

#### 原因2：避免错误的数值关系

如果直接编码为整数：
```python
repository_code = {
    'examples': 1,
    'VulBERTa': 2,
    'Person_reID': 3
}
```

**问题**：算法会误认为 `Person_reID (3)` > `VulBERTa (2)` > `examples (1)`，但类别之间没有大小关系！

One-Hot编码避免了这个问题，因为每个类别都是独立的二进制列。

### 3.3 ✅ 决策：我们的One-Hot编码方案

#### 需要编码的列

**在分层分析中，不需要对repository和model进行One-Hot编码**，因为：
- 每个任务组只包含一个repository（如examples）
- 但model可能有多个（如mnist, mnist_ff, mnist_rnn）

**需要One-Hot编码的列**：

1. **model**（在每个任务组内）
   - 例如examples组: mnist, mnist_ff, mnist_rnn, siamese → 4列
   - 例如Person_reID组: densenet121, hrnet18, pcb → 3列

2. **其他类别型超参数**（如果有）
   - 例如optimizer（如果变化）: Adam, SGD, RMSprop → 3列

#### 实现代码

##### 方法1：使用pandas.get_dummies（推荐）

```python
import pandas as pd

def one_hot_encode_model(df, task_name):
    """
    对model列进行One-Hot编码

    Args:
        df: 任务组的DataFrame
        task_name: 任务组名称（用于列命名）

    Returns:
        df: 添加了One-Hot编码列的DataFrame
    """

    # 1. 对model列进行One-Hot编码
    model_dummies = pd.get_dummies(df['model'], prefix='model')

    # 2. 合并到原DataFrame
    df = pd.concat([df, model_dummies], axis=1)

    # 3. 可选：删除原始model列（如果不再需要）
    # df = df.drop('model', axis=1)

    print(f"任务组 {task_name}:")
    print(f"  原始model类别数: {df['model'].nunique()}个")
    print(f"  One-Hot编码后列数: {model_dummies.shape[1]}列")
    print(f"  新增列名: {list(model_dummies.columns)}")

    return df
```

**使用示例**：

```python
# 读取examples任务组数据
df_mnist = pd.read_csv('data/training_data_mnist.csv')

# One-Hot编码
df_mnist = one_hot_encode_model(df_mnist, 'mnist')

# 结果：
# 原始列: model = ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese']
# 新增列: model_mnist, model_mnist_ff, model_mnist_rnn, model_siamese
#         (4个二进制列，每列取值0或1)
```

##### 方法2：使用sklearn（更灵活）

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def one_hot_encode_sklearn(df, column):
    """
    使用sklearn进行One-Hot编码

    优势：可以处理新类别、保存编码器用于预测
    """

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # 1. 编码
    encoded = encoder.fit_transform(df[[column]])

    # 2. 获取列名
    categories = encoder.categories_[0]
    col_names = [f"{column}_{cat}" for cat in categories]

    # 3. 转为DataFrame
    encoded_df = pd.DataFrame(encoded, columns=col_names, index=df.index)

    # 4. 合并
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder  # 返回编码器用于后续数据
```

### 3.4 One-Hot编码的注意事项

#### 1. 虚拟变量陷阱（Dummy Variable Trap）

**问题**：如果有N个类别，One-Hot编码会生成N列。但实际上只需要N-1列，因为最后一列可以通过其他列推导出来。

**示例**：
```
如果有3个类别：A, B, C
One-Hot编码后：
  A  B  C
  1  0  0  ← 类别A
  0  1  0  ← 类别B
  0  0  1  ← 类别C
```

如果前两列都是0，那么C必然是1，所以C列是冗余的。

**在回归分析中**，这会导致多重共线性问题。

**解决方案**：
```python
# pandas: 使用drop_first=True删除第一个类别
pd.get_dummies(df['model'], prefix='model', drop_first=True)

# sklearn: 使用drop='first'
OneHotEncoder(drop='first')
```

**在DiBS中是否需要drop_first？**

对于**因果图学习**（DiBS），通常**不需要**drop_first，因为：
- DiBS关注的是变量间的因果关系，而非回归系数
- 保留所有类别更易于解释（"model是mnist" vs "model不是mnist_ff且不是mnist_rnn"）

**建议**：保留所有类别列，除非出现数值稳定性问题。

#### 2. 稀疏性问题

如果类别数量非常多（如50+），One-Hot编码会产生大量列，且大部分是0（稀疏）。

**解决方案**：
- 合并低频类别（如 "其他"）
- 使用其他编码方法（如Target Encoding，但不适合因果推断）
- 在我们的案例中，每个任务组的model类别数很少（2-4个），无此问题

#### 3. 变量选择

One-Hot编码后，可能需要选择哪些编码列纳入DiBS分析。

**策略**：
```python
# 如果某个model类别样本量 < 5，考虑排除
model_counts = df['model'].value_counts()
rare_models = model_counts[model_counts < 5].index

# 过滤
df = df[~df['model'].isin(rare_models)]
```

### 3.5 完整预处理流程

```python
import pandas as pd
import numpy as np

def preprocess_for_dibs(df, task_name):
    """
    完整的DiBS数据预处理流程

    Args:
        df: 原始DataFrame
        task_name: 任务组名称

    Returns:
        df_processed: 预处理后的DataFrame
        feature_names: DiBS使用的特征列名列表
    """

    print(f"{'='*80}")
    print(f"任务组: {task_name}")
    print(f"{'='*80}\n")

    # 1. 删除缺失值（按问题1的标准）
    print("1. 删除缺失值...")
    df = df.dropna(subset=['energy_cpu_total_joules', 'energy_gpu_total_joules',
                           'repository', 'model'])
    # 至少一个性能指标
    perf_cols = ['perf_test_accuracy', 'perf_map', 'perf_eval_loss', 'perf_top1_accuracy']
    df = df[df[perf_cols].notna().any(axis=1)]
    print(f"   保留行数: {len(df)}")

    # 2. mode二值化（按问题2）
    print("\n2. mode二值化...")
    df['mode'] = df['mode'].fillna('default').replace('', 'default')
    df['mode_binary'] = df['mode'].map({'default': 0, 'parallel': 1})
    print(f"   mode=0: {(df['mode_binary']==0).sum()}个")
    print(f"   mode=1: {(df['mode_binary']==1).sum()}个")

    # 3. One-Hot编码（按问题3）
    print("\n3. One-Hot编码...")
    if df['model'].nunique() > 1:
        model_dummies = pd.get_dummies(df['model'], prefix='model')
        df = pd.concat([df, model_dummies], axis=1)
        print(f"   model类别数: {df['model'].nunique()}个")
        print(f"   新增列: {list(model_dummies.columns)}")
    else:
        print(f"   只有1个model，跳过One-Hot编码")

    # 4. 选择DiBS分析的特征列
    print("\n4. 选择DiBS特征...")

    # 超参数
    hyperparam_cols = []
    for col in ['hyperparam_learning_rate', 'hyperparam_batch_size',
                'hyperparam_training_duration', 'hyperparam_l2_regularization',
                'hyperparam_dropout']:
        if col in df.columns and df[col].notna().sum() > 0:
            hyperparam_cols.append(col)

    # 能耗
    energy_cols = ['energy_cpu_total_joules', 'energy_gpu_total_joules']

    # 能耗中介
    mediator_cols = ['gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
                     'gpu_power_fluctuation', 'gpu_temp_fluctuation']

    # 性能（任务特定）
    perf_col = None
    for col in perf_cols:
        if col in df.columns and df[col].notna().sum() > len(df) * 0.5:
            perf_col = col
            break

    # 敏感属性
    sensitive_col = ['mode_binary']

    # One-Hot编码列
    model_cols = [col for col in df.columns if col.startswith('model_')]

    # 合并所有特征
    feature_names = hyperparam_cols + energy_cols + mediator_cols + [perf_col] + sensitive_col + model_cols
    feature_names = [col for col in feature_names if col is not None]

    print(f"   超参数特征: {len(hyperparam_cols)}个")
    print(f"   能耗特征: {len(energy_cols)}个")
    print(f"   中介变量: {len(mediator_cols)}个")
    print(f"   性能特征: {'1个' if perf_col else '0个'}")
    print(f"   敏感属性: 1个 (mode_binary)")
    print(f"   模型编码: {len(model_cols)}个")
    print(f"   总特征数: {len(feature_names)}个")

    # 5. 提取特征矩阵
    df_processed = df[feature_names].copy()

    # 6. 验证数据类型
    print("\n5. 验证数据类型...")
    for col in feature_names:
        if df_processed[col].dtype == 'object':
            print(f"   ⚠️ {col} 仍是object类型，需转换")
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # 7. 最终检查
    print("\n6. 最终检查...")
    print(f"   数据形状: {df_processed.shape}")
    print(f"   缺失值: {df_processed.isna().sum().sum()}个")
    print(f"   数据类型: 全部数值型={df_processed.dtypes.apply(lambda x: x in [np.float64, np.int64, np.float32, np.int32]).all()}")

    return df_processed, feature_names

# 使用示例
if __name__ == '__main__':
    df_raw = pd.read_csv('../data/raw_data.csv')

    # 筛选MNIST任务组
    df_mnist = df_raw[df_raw['repository'] == 'examples'].copy()

    # 预处理
    df_dibs, features = preprocess_for_dibs(df_mnist, 'mnist')

    # 保存
    df_dibs.to_csv('../data/training_data_mnist_processed.csv', index=False)

    print("\n处理完成！")
```

---

## 📊 预处理前后对比

| 维度 | 预处理前 | 预处理后 |
|------|---------|---------|
| **总行数** | 676行 | 284行 |
| **数据完整性** | 部分缺失 | 100%完整（关键列） |
| **mode列** | 348行缺失(51.5%) | 100%填充+编码 |
| **类别变量** | 字符串（repository, model） | 数值型（One-Hot编码） |
| **敏感属性** | mode（字符串） | mode_binary（0/1） |
| **DiBS兼容性** | ❌ 不兼容 | ✅ 完全兼容 |

---

## 📚 参考文档

- [VARIABLE_EXPANSION_PLAN.md](./reports/VARIABLE_EXPANSION_PLAN.md) - 变量扩展方案
- [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) - 数据迁移指南
- [ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](./reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md) - DiBS基线分析

---

## 📌 版本历史

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2025-12-22 | 初始版本：数据预处理三大决策 | Green + Claude |

---

**维护者**: Green
**文档状态**: ✅ 方案确认完成
**下次更新**: 实现预处理脚本后（添加实际运行结果）
