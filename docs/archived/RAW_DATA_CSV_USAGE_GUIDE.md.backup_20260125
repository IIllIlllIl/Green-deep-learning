# raw_data.csv 数据使用指南

**文档版本**: 1.0
**最后更新**: 2026-01-10
**数据文件**: `data/raw_data.csv`
**当前规模**: 970行实验数据，87列字段

---

## 📋 目录

1. [快速开始](#快速开始)
2. [数据结构概述](#数据结构概述)
3. [并行vs非并行模式](#并行vs非并行模式)
4. [字段详细说明](#字段详细说明)
5. [常见错误和解决方案](#常见错误和解决方案)
6. [代码示例](#代码示例)
7. [模型列表](#模型列表)

---

## 快速开始

### ⚠️ 必读要点

1. **并行模式数据在 `fg_` 前缀字段中** - 不要读取顶层字段！
2. **空字符串表示默认值或不适用** - 不是数据缺失！
3. **mode字段区分并行/非并行** - 必须先检查此字段！
4. **模型名称是固定的** - 参考[模型列表](#模型列表)

### 基本读取代码模板

```python
import pandas as pd

# 读取数据
df = pd.read_csv('data/raw_data.csv')

# 创建统一的访问函数
def get_field(row, field_name):
    """
    智能获取字段值，自动处理并行/非并行模式

    Args:
        row: DataFrame行
        field_name: 字段名（不带fg_前缀）

    Returns:
        字段值（字符串或数值）
    """
    is_parallel = (row['mode'] == 'parallel')

    if is_parallel:
        # 并行模式：优先使用fg_字段
        fg_value = row.get(f'fg_{field_name}', '')
        if pd.notna(fg_value) and str(fg_value).strip():
            return fg_value

    # 非并行模式或fg_字段为空时使用顶层字段
    return row.get(field_name, '')

# 使用示例
for idx, row in df.iterrows():
    repo = get_field(row, 'repository')
    model = get_field(row, 'model')
    learning_rate = get_field(row, 'hyperparam_learning_rate')
    print(f"{repo}/{model}, lr={learning_rate}")
```

---

## 数据结构概述

### 总体结构

```
raw_data.csv (970行 × 87列)
│
├── 基础字段 (1-7列)
│   ├── experiment_id
│   ├── timestamp
│   ├── repository          # ⚠️ 并行模式下为空
│   ├── model               # ⚠️ 并行模式下为空
│   ├── training_success    # ⚠️ 并行模式下为空
│   ├── duration_seconds    # ⚠️ 并行模式下为空
│   └── retries             # ⚠️ 并行模式下为空
│
├── 超参数字段 (8-16列)
│   ├── hyperparam_alpha
│   ├── hyperparam_batch_size
│   ├── hyperparam_dropout
│   ├── hyperparam_epochs
│   ├── hyperparam_kfold
│   ├── hyperparam_learning_rate
│   ├── hyperparam_max_iter
│   ├── hyperparam_seed
│   └── hyperparam_weight_decay
│
├── 性能指标字段 (17-32列)
│   ├── perf_accuracy
│   ├── perf_best_val_accuracy
│   ├── perf_map
│   ├── perf_precision
│   ├── perf_rank1
│   ├── perf_rank5
│   ├── perf_recall
│   ├── perf_test_accuracy
│   ├── perf_test_loss
│   ├── perf_eval_loss
│   ├── perf_final_training_loss
│   ├── perf_eval_samples_per_second
│   ├── perf_top1_accuracy
│   ├── perf_top5_accuracy
│   ├── perf_top10_accuracy
│   └── perf_top20_accuracy
│
├── 能耗字段 (33-43列)
│   ├── energy_cpu_pkg_joules
│   ├── energy_cpu_ram_joules
│   ├── energy_cpu_total_joules
│   ├── energy_gpu_avg_watts
│   ├── energy_gpu_max_watts
│   ├── energy_gpu_min_watts
│   ├── energy_gpu_total_joules
│   ├── energy_gpu_temp_avg_celsius
│   ├── energy_gpu_temp_max_celsius
│   ├── energy_gpu_util_avg_percent
│   └── energy_gpu_util_max_percent
│
├── 元数据字段 (44-48列)
│   ├── experiment_source   # 实验来源标签
│   ├── num_mutated_params  # 变异参数数量
│   ├── mutated_param       # 变异的参数名
│   ├── mode                # ⚠️ 关键: "parallel" 或 空字符串
│   └── error_message
│
├── 前台（Foreground）字段 (49-83列) - 仅并行模式
│   ├── fg_repository       # ⚠️ 并行模式主要数据在这里
│   ├── fg_model
│   ├── fg_duration_seconds
│   ├── fg_training_success
│   ├── fg_retries
│   ├── fg_error_message
│   ├── fg_hyperparam_*     # 前台训练的超参数
│   ├── fg_perf_*           # 前台训练的性能指标
│   └── fg_energy_*         # 前台训练的能耗数据
│
└── 后台（Background）字段 (84-87列) - 仅并行模式
    ├── bg_repository       # 后台干扰任务的仓库
    ├── bg_model            # 后台干扰任务的模型
    ├── bg_note
    └── bg_log_directory
```

### 数据分布统计

- **总实验数**: 970
- **非并行模式**: 436 (44.9%)
- **并行模式**: 534 (55.1%)
- **能耗完整性**: 828/970 (85.4%)

---

## 并行vs非并行模式

### 核心区别

| 特征 | 非并行模式 | 并行模式 |
|------|-----------|---------|
| **mode字段** | 空字符串 `""` | `"parallel"` |
| **数据位置** | 顶层字段 | `fg_` 前缀字段 |
| **repository** | ✅ 有值 | ❌ 空 (在fg_repository) |
| **model** | ✅ 有值 | ❌ 空 (在fg_model) |
| **超参数** | `hyperparam_*` | `fg_hyperparam_*` |
| **性能指标** | `perf_*` | `fg_perf_*` |
| **能耗数据** | `energy_*` | `fg_energy_*` |
| **后台任务** | 无 | `bg_repository`, `bg_model` |

### 实际数据示例

#### 非并行模式示例

```csv
experiment_id: default__MRT-OAST_default_001
mode: ""
repository: MRT-OAST          # ✅ 顶层有值
model: default                # ✅ 顶层有值
training_success: True
hyperparam_epochs: 10         # ✅ 顶层有值
perf_precision: 0.9834
energy_cpu_total_joules: 39987.66

fg_repository: ""             # ❌ fg_字段为空
fg_model: ""
fg_hyperparam_epochs: ""
```

#### 并行模式示例

```csv
experiment_id: default__pytorch_resnet_cifar10_resnet20_012_parallel
mode: "parallel"              # ⚠️ 关键标识
repository: ""                # ❌ 顶层为空
model: ""                     # ❌ 顶层为空
training_success: ""
hyperparam_epochs: ""

fg_repository: pytorch_resnet_cifar10  # ✅ 前台数据在fg_字段
fg_model: resnet20
fg_training_success: True
fg_hyperparam_epochs: 200
fg_perf_test_accuracy: 92.17
fg_energy_cpu_total_joules: 46525.57

bg_repository: examples       # 后台干扰任务
bg_model: mnist_ff
```

### 🚨 常见错误示例

#### ❌ 错误做法1：不检查mode字段

```python
# ❌ 错误：直接读取顶层字段
repo = df['repository']  # 并行模式下全是空值！
```

#### ❌ 错误做法2：只读取顶层或只读取fg_字段

```python
# ❌ 错误：只读fg_字段
repo = df['fg_repository']  # 非并行模式下全是空值！
```

#### ✅ 正确做法：根据mode字段选择

```python
# ✅ 正确
def get_repository(row):
    if row['mode'] == 'parallel':
        return row['fg_repository']
    else:
        return row['repository']

df['repo'] = df.apply(get_repository, axis=1)
```

---

## 字段详细说明

### 基础字段

| 字段名 | 类型 | 说明 | 空值含义 |
|--------|------|------|---------|
| `experiment_id` | string | 实验唯一标识符 | 不应为空 |
| `timestamp` | string | ISO 8601格式时间戳 | 不应为空 |
| `repository` | string | 仓库名称 | **并行模式下为空** |
| `model` | string | 模型名称 | **并行模式下为空** |
| `training_success` | boolean | 训练是否成功 | **并行模式下为空** |
| `duration_seconds` | float | 训练时长（秒） | **并行模式下为空** |
| `retries` | int | 重试次数 | 0或空 |

### 超参数字段

| 字段名 | 类型 | 适用模型 | 空值含义 |
|--------|------|---------|---------|
| `hyperparam_alpha` | float | bug-localization | 该模型不使用此参数 |
| `hyperparam_batch_size` | int | 大多数模型 | 使用默认值 |
| `hyperparam_dropout` | float | Person_reID, VulBERTa, MRT-OAST | 该模型不使用dropout |
| `hyperparam_epochs` | int | 所有模型 | 使用默认值 |
| `hyperparam_kfold` | int | bug-localization | 该模型不使用k-fold |
| `hyperparam_learning_rate` | float | 所有模型 | 使用默认值 |
| `hyperparam_max_iter` | int | bug-localization | 该模型不使用此参数 |
| `hyperparam_seed` | int | 大多数模型 | 未设置或使用默认 |
| `hyperparam_weight_decay` | float | Person_reID, ResNet, MRT-OAST | 该模型不使用 |

**⚠️ 重要**：空值不代表数据缺失，而是：
1. 该模型不使用此超参数
2. 使用模型的默认值
3. 该参数在本次实验中未被变异

### 性能指标字段

不同模型使用不同的性能指标：

| 模型 | 主要性能指标 |
|------|------------|
| Person_reID (densenet121, hrnet18, pcb) | `perf_rank1`, `perf_rank5`, `perf_map` |
| ResNet (pytorch_resnet_cifar10) | `perf_test_accuracy`, `perf_best_val_accuracy` |
| VulBERTa | `perf_precision`, `perf_recall`, `perf_f1` |
| Examples (mnist, mnist_rnn, siamese) | `perf_test_accuracy` |
| MRT-OAST | `perf_precision` |
| bug-localization | 无标准化指标 |

**空值含义**：该模型不使用此指标

### 能耗字段

| 字段名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| `energy_cpu_pkg_joules` | float | 焦耳 | CPU Package能耗 |
| `energy_cpu_ram_joules` | float | 焦耳 | RAM能耗 |
| `energy_cpu_total_joules` | float | 焦耳 | CPU总能耗 (pkg + ram) |
| `energy_gpu_avg_watts` | float | 瓦特 | GPU平均功率 |
| `energy_gpu_max_watts` | float | 瓦特 | GPU峰值功率 |
| `energy_gpu_min_watts` | float | 瓦特 | GPU最小功率 |
| `energy_gpu_total_joules` | float | 焦耳 | GPU总能耗 |
| `energy_gpu_temp_avg_celsius` | float | 摄氏度 | GPU平均温度 |
| `energy_gpu_temp_max_celsius` | float | 摄氏度 | GPU峰值温度 |
| `energy_gpu_util_avg_percent` | float | % | GPU平均利用率 |
| `energy_gpu_util_max_percent` | float | % | GPU峰值利用率 |

**空值含义**：能耗监控失败（权限问题或nvidia-smi不可用）

### 元数据字段

| 字段名 | 类型 | 说明 | 空值含义 |
|--------|------|------|---------|
| `experiment_source` | string | 实验来源标签 | 旧数据未标记 |
| `num_mutated_params` | int | 变异参数数量 | 旧数据未记录 |
| `mutated_param` | string | 变异的参数名 | 旧数据未记录 |
| `mode` | string | **关键**: "parallel"或空 | 空=非并行 |
| `error_message` | string | 错误或成功信息 | 无 |

---

## 常见错误和解决方案

### 错误1: 不知道并行模式数据在fg_字段

#### ❌ 错误代码
```python
# 读取repository字段，但并行模式下repository为空
df = pd.read_csv('data/raw_data.csv')
df_resnet = df[df['repository'] == 'pytorch_resnet_cifar10']
# 结果：遗漏了所有并行模式的resnet实验！
```

#### ✅ 正确代码
```python
def get_unified_field(df, field_name):
    """创建统一字段，自动合并顶层和fg_字段"""
    result = df[field_name].copy()
    is_parallel = (df['mode'] == 'parallel')
    fg_field = f'fg_{field_name}'

    if fg_field in df.columns:
        # 并行模式使用fg_字段
        result[is_parallel] = df.loc[is_parallel, fg_field]

    return result

df['repo'] = get_unified_field(df, 'repository')
df['model_name'] = get_unified_field(df, 'model')
df_resnet = df[df['repo'] == 'pytorch_resnet_cifar10']
# 结果：包含所有resnet实验（并行+非并行）
```

### 错误2: 把空值当作缺失数据

#### ❌ 错误理解
```python
# 错误：认为空的hyperparam_dropout是数据缺失
df['dropout'].fillna(0.5)  # ❌ 破坏了原始语义
```

#### ✅ 正确理解
```python
# 正确：空值有三种含义
# 1. 该模型不使用此参数（如mnist不使用dropout）
# 2. 使用默认值
# 3. 此参数未被变异

# 如果需要填充，应该查阅models_config.json中的默认值
import json
with open('mutation/models_config.json') as f:
    config = json.load(f)

# 为特定模型获取默认dropout
default_dropout = config['models']['VulBERTa']['hyperparameters']['dropout']['default']
```

### 错误3: 不知道模型名称的准确拼写

#### ❌ 错误代码
```python
# 错误：模型名称拼写错误
df[df['model'] == 'resnet']  # ❌ 应该是 'resnet20'
df[df['model'] == 'densenet']  # ❌ 应该是 'densenet121'
df[df['repository'] == 'Person_reID']  # ❌ 应该是 'Person_reID_baseline_pytorch'
```

#### ✅ 正确代码
```python
# 使用准确的模型名称（参考本文档末尾的模型列表）
df_resnet = df[df['model_name'] == 'resnet20']
df_densenet = df[df['model_name'] == 'densenet121']
df_person_reid = df[df['repo'] == 'Person_reID_baseline_pytorch']
```

### 错误4: 直接过滤数据导致遗漏

#### ❌ 错误代码
```python
# 只筛选了顶层字段，遗漏了并行模式
df_successful = df[df['training_success'] == True]
# 遗漏了所有并行模式的成功实验！
```

#### ✅ 正确代码
```python
def get_training_success(df):
    """统一获取训练成功状态"""
    success = df['training_success'].copy()
    is_parallel = (df['mode'] == 'parallel')
    success[is_parallel] = df.loc[is_parallel, 'fg_training_success']
    return success

df['success'] = get_training_success(df)
df_successful = df[df['success'] == 'True']  # 注意：CSV中是字符串
```

---

## 代码示例

### 示例1: 读取和预处理数据

```python
import pandas as pd
import numpy as np

def load_and_preprocess_data(csv_path='data/raw_data.csv'):
    """
    加载并预处理raw_data.csv，创建统一的访问接口

    Returns:
        DataFrame: 预处理后的数据，添加了统一字段
    """
    df = pd.read_csv(csv_path)

    # 判断是否为并行模式
    df['is_parallel'] = (df['mode'] == 'parallel')

    # 创建统一字段的辅助函数
    def create_unified_field(field_name):
        result = df[field_name].copy()
        fg_field = f'fg_{field_name}'
        if fg_field in df.columns:
            mask = df['is_parallel']
            result[mask] = df.loc[mask, fg_field]
        return result

    # 基础字段
    df['repo'] = create_unified_field('repository')
    df['model_name'] = create_unified_field('model')
    df['success'] = create_unified_field('training_success')
    df['duration'] = create_unified_field('duration_seconds')

    # 超参数字段
    for param in ['alpha', 'batch_size', 'dropout', 'epochs', 'kfold',
                  'learning_rate', 'max_iter', 'seed', 'weight_decay']:
        df[f'hp_{param}'] = create_unified_field(f'hyperparam_{param}')

    # 性能指标字段
    for metric in ['accuracy', 'best_val_accuracy', 'map', 'precision',
                   'rank1', 'rank5', 'recall', 'test_accuracy']:
        df[f'perf_{metric}'] = create_unified_field(f'perf_{metric}')

    # 能耗字段
    for energy in ['cpu_total_joules', 'gpu_total_joules', 'gpu_avg_watts']:
        df[f'energy_{energy}'] = create_unified_field(f'energy_{energy}')

    # 数据类型转换
    df['success'] = df['success'].map({'True': True, 'true': True,
                                       'False': False, 'false': False})

    # 数值列转换
    numeric_cols = ['duration', 'hp_epochs', 'hp_learning_rate',
                   'energy_cpu_total_joules', 'energy_gpu_total_joules']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# 使用示例
df = load_and_preprocess_data()
print(f"总实验数: {len(df)}")
print(f"非并行: {(~df['is_parallel']).sum()}")
print(f"并行: {df['is_parallel'].sum()}")
```

### 示例2: 按模型分组分析

```python
def analyze_by_model(df):
    """按模型统计实验数量和成功率"""
    results = []

    for repo_model in df.groupby(['repo', 'model_name']):
        repo, model = repo_model[0]
        data = repo_model[1]

        if not repo or not model:  # 跳过空值
            continue

        total = len(data)
        parallel = data['is_parallel'].sum()
        nonparallel = total - parallel

        # 成功率
        success_rate = (data['success'] == True).sum() / total * 100

        # 能耗完整性
        has_energy = data['energy_cpu_total_joules'].notna().sum()
        energy_rate = has_energy / total * 100

        results.append({
            'repository': repo,
            'model': model,
            'total': total,
            'parallel': parallel,
            'nonparallel': nonparallel,
            'success_rate': f'{success_rate:.1f}%',
            'energy_coverage': f'{energy_rate:.1f}%'
        })

    return pd.DataFrame(results).sort_values('total', ascending=False)

# 使用
summary = analyze_by_model(df)
print(summary.to_string(index=False))
```

### 示例3: 提取特定模型的数据

```python
def get_model_data(df, repository, model, include_parallel=True):
    """
    提取特定模型的实验数据

    Args:
        df: 预处理后的DataFrame
        repository: 仓库名称
        model: 模型名称
        include_parallel: 是否包含并行模式实验

    Returns:
        DataFrame: 筛选后的数据
    """
    # 基础筛选
    mask = (df['repo'] == repository) & (df['model_name'] == model)

    # 是否包含并行
    if not include_parallel:
        mask = mask & (~df['is_parallel'])

    return df[mask].copy()

# 示例：获取resnet20的所有实验
df_resnet = get_model_data(df, 'pytorch_resnet_cifar10', 'resnet20')
print(f"ResNet20实验数: {len(df_resnet)}")

# 示例：只获取非并行的Person_reID实验
df_person_reid = get_model_data(df, 'Person_reID_baseline_pytorch',
                                'hrnet18', include_parallel=False)
print(f"Person_reID HRNet18非并行实验数: {len(df_person_reid)}")
```

### 示例4: 能耗分析

```python
def analyze_energy_consumption(df, group_by='model_name'):
    """
    分析能耗统计

    Args:
        df: 预处理后的DataFrame
        group_by: 分组依据 ('model_name', 'repo', 或 'is_parallel')

    Returns:
        DataFrame: 能耗统计结果
    """
    # 只分析有能耗数据的实验
    df_energy = df[df['energy_cpu_total_joules'].notna() &
                   df['energy_gpu_total_joules'].notna()].copy()

    # 计算总能耗
    df_energy['total_energy_joules'] = (
        df_energy['energy_cpu_total_joules'] +
        df_energy['energy_gpu_total_joules']
    )

    # 分组统计
    stats = df_energy.groupby(group_by).agg({
        'total_energy_joules': ['count', 'mean', 'std', 'min', 'max'],
        'energy_cpu_total_joules': 'mean',
        'energy_gpu_total_joules': 'mean',
        'energy_gpu_avg_watts': 'mean',
        'duration': 'mean'
    }).round(2)

    return stats

# 使用
energy_stats = analyze_energy_consumption(df, group_by='model_name')
print(energy_stats)
```

### 示例5: 处理超参数数据

```python
def extract_hyperparameters(df, model_repository, model_name):
    """
    提取特定模型的超参数数据，自动过滤空值

    Returns:
        DataFrame: 只包含该模型实际使用的超参数
    """
    # 获取该模型的数据
    model_data = get_model_data(df, model_repository, model_name)

    # 超参数列
    hyperparam_cols = [col for col in df.columns if col.startswith('hp_')]

    # 找出非空的超参数（该模型实际使用的）
    used_params = []
    for col in hyperparam_cols:
        if model_data[col].notna().any():
            used_params.append(col)

    # 只返回相关列
    result_cols = ['experiment_id', 'timestamp', 'success',
                   'duration', 'is_parallel'] + used_params

    return model_data[result_cols].copy()

# 示例：ResNet20使用的超参数
resnet_hyperparams = extract_hyperparameters(df, 'pytorch_resnet_cifar10', 'resnet20')
print(f"ResNet20使用的超参数: {[col for col in resnet_hyperparams.columns if col.startswith('hp_')]}")
```

---

## 模型列表

### 完整的仓库/模型组合（11个模型）

| # | Repository | Model | 实验数量 | 主要超参数 | 主要性能指标 |
|---|-----------|-------|---------|-----------|------------|
| 1 | `MRT-OAST` | `default` | 85 | dropout, epochs, learning_rate, weight_decay | precision |
| 2 | `Person_reID_baseline_pytorch` | `densenet121` | 53 | dropout, epochs, learning_rate, seed | rank1, rank5, map |
| 3 | `Person_reID_baseline_pytorch` | `hrnet18` | 53 | dropout, epochs, learning_rate, seed | rank1, rank5, map |
| 4 | `Person_reID_baseline_pytorch` | `pcb` | 53 | dropout, epochs, learning_rate, seed | rank1, rank5, map |
| 5 | `VulBERTa` | `mlp` | 151 | dropout, epochs, learning_rate, seed | precision, recall, f1 |
| 6 | `bug-localization-by-dnn-and-rvsm` | `default` | 131 | alpha, kfold, max_iter, seed | (无标准化指标) |
| 7 | `examples` | `mnist` | 75 | batch_size, epochs, learning_rate, seed | test_accuracy |
| 8 | `examples` | `mnist_ff` | 87 | batch_size, epochs, learning_rate, seed | test_accuracy |
| 9 | `examples` | `mnist_rnn` | 58 | batch_size, epochs, learning_rate, seed | test_accuracy |
| 10 | `examples` | `siamese` | 55 | batch_size, epochs, learning_rate, seed | test_accuracy |
| 11 | `pytorch_resnet_cifar10` | `resnet20` | 53 | epochs, learning_rate, weight_decay, seed | test_accuracy, best_val_accuracy |

### 模型名称准确拼写

**⚠️ 注意大小写和特殊字符**：

```python
# ✅ 正确拼写
VALID_REPOSITORIES = [
    'MRT-OAST',                              # 注意连字符
    'Person_reID_baseline_pytorch',          # 注意下划线和大小写
    'VulBERTa',                              # 注意大小写
    'bug-localization-by-dnn-and-rvsm',     # 注意连字符
    'examples',
    'pytorch_resnet_cifar10'                 # 注意下划线
]

VALID_MODELS = [
    'default',      # MRT-OAST, bug-localization
    'densenet121',  # 不是 'densenet'
    'hrnet18',      # 不是 'hrnet'
    'pcb',
    'mlp',
    'mnist',
    'mnist_ff',     # 注意下划线
    'mnist_rnn',    # 注意下划线
    'siamese',
    'resnet20'      # 不是 'resnet'
]
```

---

## 快速检查清单

在编写分析脚本前，请检查：

- [ ] ✅ 是否检查了 `mode` 字段来区分并行/非并行？
- [ ] ✅ 是否为并行模式使用了 `fg_` 前缀字段？
- [ ] ✅ 是否理解空值的含义（默认值/不适用）？
- [ ] ✅ 是否使用了正确的模型名称拼写？
- [ ] ✅ 是否创建了统一的字段访问函数？
- [ ] ✅ 是否验证了数据提取的完整性（并行+非并行）？

---

## 获取帮助

- **完整项目文档**: [docs/CLAUDE_FULL_REFERENCE.md](CLAUDE_FULL_REFERENCE.md)
- **分析模块文档**: [analysis/docs/CLAUDE.md](../analysis/docs/CLAUDE.md)
- **数据验证脚本**: `tools/data_management/validate_raw_data.py`
- **数据格式设计**: [docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md)

---

**维护者**: Green
**最后更新**: 2026-01-10
**版本历史**:
- v1.0 (2026-01-10): 初始版本，基于970个实验的数据结构
