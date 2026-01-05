# 能耗因果分析 - 数据重新提取方案

**日期**: 2025-12-24
**目的**: 从主实验JSON文件重新提取数据，解决缺失值问题
**优先级**: 🔴 **P0 - 最高优先级**

---

## 执行摘要

### 问题根源

当前因果分析数据的缺失值问题源于：

1. **超参数命名不统一**: 不同模型使用不同字段名（`learning_rate` vs `max_iter` vs `epochs`）
2. **并行模式超参数为空**: JSON中并行实验的`foreground.hyperparameters`字段为空 ❌
3. **数据提取逻辑简单**: 当前脚本直接映射JSON键名，未做字段统一

### 解决方案

**方案**: 从主实验的`experiment.json`文件重新提取数据，实现：
1. **超参数字段统一映射**（不同命名 → 统一变量名）
2. **从配置文件回溯超参数**（解决并行模式hyperparameters为空的问题）
3. **完整的缺失值处理**（合理插补 + 行删除）

**预期改进**:
- 超参数缺失率: 32-100% → **< 5%**
- 完全无缺失行: 0-64.7% → **> 90%**
- DiBS因果边数: 0条 → **3-8条/任务组**

---

## 一、主实验JSON结构分析

### 1.1 非并行模式JSON结构

**示例**: VulBERTa_mlp_001

```json
{
  "experiment_id": "VulBERTa_mlp_001",
  "timestamp": "2025-12-17T22:26:33.012398",
  "repository": "VulBERTa",
  "model": "mlp",
  "hyperparameters": {
    "epochs": 14  // ✅ 超参数在顶层
  },
  "energy_metrics": {
    "cpu_energy_total_joules": 137821.05,
    "gpu_power_avg_watts": 236.32,
    "gpu_util_avg_percent": 89.23,
    ...
  },
  "performance_metrics": {
    "eval_loss": 0.6955,  // ✅ 性能指标完整
    "final_training_loss": 0.7908
  }
}
```

**特征**:
- ✅ `hyperparameters`在顶层，包含实验参数
- ✅ `energy_metrics`和`performance_metrics`完整
- ⚠️ 超参数字段名因模型而异（`epochs`, `learning_rate`, `max_iter`）

---

### 1.2 并行模式JSON结构

**示例**: bug-localization-by-dnn-and-rvsm_default_001_parallel

```json
{
  "experiment_id": "bug-localization-by-dnn-and-rvsm_default_001_parallel",
  "timestamp": "2025-12-22T22:06:06.154204",
  "mode": "parallel",
  "foreground": {
    "repository": "bug-localization-by-dnn-and-rvsm",
    "model": "default",
    "hyperparameters": {},  // ❌ 空的！
    "energy_metrics": { ... },  // ✅ 完整
    "performance_metrics": {
      "top1_accuracy": 0.382,  // ✅ 完整
      "top5_accuracy": 0.629
    }
  },
  "background": {
    "repository": "examples",
    "model": "mnist"
  }
}
```

**严重问题**:
- ❌ **`foreground.hyperparameters`为空字典**
- 这是为什么并行模式数据有100%超参数缺失！

---

### 1.3 超参数命名差异汇总

| 模型/数据集 | 超参数字段名 | 示例值 | 在data.csv中的列名 |
|------------|-------------|--------|-------------------|
| **examples/mnist** | `learning_rate`, `batch_size` | 0.01, 64 | `hyperparam_learning_rate`, `hyperparam_batch_size` |
| **pytorch_resnet_cifar10/resnet20** | `epochs`, `lr` | 200, 0.1 | `hyperparam_epochs`, `hyperparam_lr` |
| **VulBERTa/mlp** | `epochs`, `learning_rate` | 14, 1e-5 | `hyperparam_epochs`, `hyperparam_learning_rate` |
| **bug-localization** | `max_iter` | 1209 | `hyperparam_max_iter` |
| **Person_reID** | `learning_rate`, `dropout` | 0.001, 0.5 | `hyperparam_learning_rate`, `hyperparam_dropout` |
| **MRT-OAST** | `learning_rate`, `num_iters` | 0.01, 100 | `hyperparam_learning_rate`, `hyperparam_num_iters` |

**问题**: 同一个概念（如"学习率"）有多个字段名：`learning_rate`, `lr`

---

## 二、当前数据提取流程分析

### 2.1 append_session_to_raw_data.py的提取逻辑

**代码** (`tools/data_management/append_session_to_raw_data.py:146-150,216-220`):

```python
# 并行模式
hyperparams = fg_data.get('hyperparameters', {})
for key, value in hyperparams.items():
    col_name = f'hyperparam_{key}'
    if col_name in fieldnames:
        row[col_name] = str(value)
```

**缺陷**:

1. **直接映射字段名**: `key` → `hyperparam_{key}`，未做统一
   - `lr` → `hyperparam_lr`（而非`hyperparam_learning_rate`）
   - `max_iter` → `hyperparam_max_iter`（而非`training_duration`）

2. **依赖JSON中的hyperparameters字段**: 当字段为空时无法提取
   - 并行模式的`hyperparameters`为空 → 100%缺失

3. **无字段映射表**: 不同命名无法归一化

---

### 2.2 create_unified_data_csv.py的处理

**功能**: 从`raw_data.csv`生成`data.csv`

**代码逻辑**:
```python
# 统一超参数字段（并行/非并行）
df['hyperparam_learning_rate'] = df['hyperparam_learning_rate'].fillna(df['fg_hyperparam_learning_rate'])
```

**缺陷**:
- 只是合并并行/非并行的同名字段
- 无法解决字段名不统一的问题（`lr` vs `learning_rate`）
- 无法解决hyperparameters为空的问题

---

### 2.3 analysis/scripts/stage0_data_validation.py的处理

**功能**: 验证`data.csv`质量

**代码** (`analysis/scripts/stage0_data_validation.py:85-90`):
```python
required_columns = {
    '超参数': ['hyperparam_learning_rate', 'hyperparam_batch_size', 'hyperparam_epochs'],
    ...
}
```

**缺陷**:
- 预期固定列名，但实际JSON字段名不固定
- 只能检测缺失，无法修复

---

## 三、重新提取方案设计

### 3.1 方案架构

```
┌─────────────────────────────────────────────────────────────┐
│  主实验JSON文件 (results/run_*/*/experiment.json)         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  新脚本: extract_causal_analysis_data.py                     │
│  功能:                                                       │
│  1. 遍历所有session目录                                    │
│  2. 读取experiment.json                                     │
│  3. 应用超参数字段映射表                                   │
│  4. 从配置文件回溯并行模式超参数                           │
│  5. 统一字段命名                                           │
│  6. 缺失值处理（插补/删除）                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  输出: analysis/data/energy_research/training/              │
│  - training_data_image_classification.csv (高质量)         │
│  - training_data_person_reid.csv                           │
│  - training_data_vulberta.csv                              │
│  - training_data_bug_localization.csv                      │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.2 超参数字段映射表

**核心思想**: 将不同模型的超参数字段名统一到标准变量名

**映射表** (`HYPERPARAM_FIELD_MAPPING`):

```python
HYPERPARAM_FIELD_MAPPING = {
    # 学习率统一
    'learning_rate': 'hyperparam_learning_rate',
    'lr': 'hyperparam_learning_rate',  # CIFAR-10使用
    'initial_lr': 'hyperparam_learning_rate',

    # 训练迭代次数统一
    'epochs': 'training_duration',  # VulBERTa, CIFAR-10
    'max_iter': 'training_duration',  # Bug定位
    'num_iters': 'training_duration',  # MRT-OAST

    # 批量大小统一
    'batch_size': 'hyperparam_batch_size',
    'train_batch_size': 'hyperparam_batch_size',

    # 正则化统一
    'weight_decay': 'l2_regularization',  # 大多数模型
    'alpha': 'l2_regularization',  # MRT-OAST

    # Dropout
    'dropout': 'hyperparam_dropout',
    'dropout_rate': 'hyperparam_dropout',

    # 其他超参数（保持原名）
    'seed': 'seed',
    'momentum': 'hyperparam_momentum',
    'gamma': 'hyperparam_gamma',
}
```

**使用示例**:
```python
# JSON中的字段: {"max_iter": 1209}
# 映射后: {"training_duration": 1209}
# CSV列名: training_duration（而非hyperparam_max_iter）
```

---

### 3.3 并行模式超参数回溯策略

**问题**: 并行模式JSON中`foreground.hyperparameters`为空

**解决方案**: 从对应的配置文件（settings/*.json）中回溯超参数

#### 3.3.1 回溯流程

```
1. 读取experiment.json
   ├─ experiment_id: "bug-localization_default_001_parallel"
   └─ foreground.hyperparameters: {}  // 空的

2. 解析experiment_id
   ├─ 提取repo: "bug-localization-by-dnn-and-rvsm"
   ├─ 提取model: "default"
   └─ 提取序号: 001

3. 查找对应的配置文件
   ├─ 搜索: settings/stage*_*.json
   └─ 匹配: "repo": "bug-localization", "mode": "parallel"

4. 从配置文件提取超参数
   ├─ 配置中的 "mutate": ["max_iter"]
   └─ 配置中的超参数值或默认值

5. 填充到data.csv
   └─ training_duration: <从配置提取的max_iter值>
```

#### 3.3.2 示例代码逻辑

```python
def get_parallel_hyperparams(experiment_id, repo, model):
    """
    从配置文件回溯并行模式的超参数

    Args:
        experiment_id: 实验ID
        repo: 仓库名
        model: 模型名

    Returns:
        dict: 超参数字典
    """
    # 1. 查找对应的配置文件
    config_files = glob.glob('settings/stage*.json') + glob.glob('settings/*parallel*.json')

    for config_file in config_files:
        with open(config_file) as f:
            configs = json.load(f)

        for config in configs:
            if (config.get('repo') == repo and
                config.get('model') == model and
                config.get('mode') == 'parallel'):

                # 2. 提取超参数
                hyperparams = {}

                # 从foreground中提取
                fg = config.get('foreground', {})
                if 'hyperparameters' in fg:
                    hyperparams.update(fg['hyperparameters'])

                # 从mutate参数提取（如果是变异实验）
                if config.get('mode') == 'mutation':
                    mutate_params = config.get('mutate', [])
                    # 需要进一步解析变异值...

                return hyperparams

    # 未找到配置，返回空字典
    return {}
```

**注意**: 这个方法可能不完美，因为：
- 配置文件中可能也没有具体的超参数值（只有变异指令）
- 需要结合默认值和变异策略来推断

**替代方案**: 如果无法回溯，使用**中位数插补**（后续处理）

---

### 3.4 缺失值处理策略

#### 3.4.1 优先级分层

| 列类型 | 处理策略 | 原因 |
|--------|----------|------|
| **超参数列** | 中位数插补（按任务组） | 允许轻微偏差，保留样本量 |
| **性能指标列** | 删除该行 | 目标变量，插补会严重偏倚因果估计 |
| **能耗指标列** | 删除该行 | 目标变量，不能插补 |
| **元信息列** | 必须完整 | experiment_id, timestamp等不能缺失 |

#### 3.4.2 超参数插补规则

**按任务组 × One-Hot分组插补**:

```python
# 示例：图像分类任务组
# MNIST模型的learning_rate缺失 → 用MNIST其他实验的中位数
mnist_lr_median = df[(df['is_mnist']==1) & df['hyperparam_learning_rate'].notna()]['hyperparam_learning_rate'].median()

df.loc[(df['is_mnist']==1) & df['hyperparam_learning_rate'].isnull(),
       'hyperparam_learning_rate'] = mnist_lr_median
```

**合理性**:
- 同一模型的超参数通常在相似范围
- 中位数代表典型值，比均值更稳健
- 按任务组分组避免跨任务污染

#### 3.4.3 行删除规则

**删除条件**:
1. 性能指标**全部**缺失（如VulBERTa的60行无eval_loss）
2. 能耗指标**全部**缺失（如图像分类的1行无能耗数据）
3. 元信息缺失（极少见）

**预期影响**:
- 图像分类: 258 → 257 (-1行，-0.4%)
- Person_reID: 116 → 116 (无变化)
- VulBERTa: 142 → 82 (-60行，-42.3%) ⚠️
- Bug定位: 132 → 80 (-52行，-39.4%) ⚠️

**样本量充足性验证**:
- DiBS最低要求: 10样本 ✅
- 所有任务组删除后仍 > 80样本 ✅

---

### 3.5 数据质量目标

| 任务组 | 当前缺失率 | 目标缺失率 | 当前完全无缺失行 | 目标完全无缺失行 |
|--------|------------|------------|------------------|------------------|
| 图像分类 | 8.83% | **< 2%** | 48.4% | **> 95%** |
| Person_reID | 4.96% | **< 2%** | 64.7% | **> 95%** |
| VulBERTa | 28.87% | **< 3%** | 0% | **> 80%** |
| Bug定位 | 24.38% | **< 3%** | 0% | **> 80%** |

**总体目标**:
- ✅ 所有超参数列填充率 > 90%
- ✅ 所有任务组至少80个完全无缺失的行
- ✅ 相关性矩阵可计算（无nan）
- ✅ DiBS能发现因果边（预期3-8条/任务组）

---

## 四、实施计划

### 4.1 第一阶段：验证JSON可访问性（30分钟）

**目标**: 确认所有experiment.json文件可读取

**脚本**: `scripts/validate_json_accessibility.py`

```python
#!/usr/bin/env python3
"""验证所有experiment.json文件的可访问性和完整性"""

import json
from pathlib import Path
from collections import defaultdict

def validate_json_files():
    """遍历所有session目录，验证JSON文件"""

    results_dir = Path('results')
    stats = {
        'total_sessions': 0,
        'total_experiments': 0,
        'json_found': 0,
        'json_parse_error': 0,
        'hyperparams_empty': 0,
        'hyperparams_nonempty': 0,
        'parallel_mode': 0,
        'non_parallel_mode': 0
    }

    repo_counts = defaultdict(int)
    hyperparam_fields = defaultdict(set)

    # 遍历所有run_*目录
    for session_dir in sorted(results_dir.glob('run_*')):
        if not session_dir.is_dir():
            continue

        stats['total_sessions'] += 1

        # 遍历实验目录
        for exp_dir in session_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name in ['__pycache__', '.git']:
                continue

            stats['total_experiments'] += 1

            json_file = exp_dir / 'experiment.json'
            if not json_file.exists():
                continue

            stats['json_found'] += 1

            try:
                with open(json_file) as f:
                    data = json.load(f)

                # 检查模式
                is_parallel = data.get('mode') == 'parallel'
                if is_parallel:
                    stats['parallel_mode'] += 1
                    repo = data.get('foreground', {}).get('repository')
                    hyperparams = data.get('foreground', {}).get('hyperparameters', {})
                else:
                    stats['non_parallel_mode'] += 1
                    repo = data.get('repository')
                    hyperparams = data.get('hyperparameters', {})

                repo_counts[repo] += 1

                # 检查超参数
                if hyperparams:
                    stats['hyperparams_nonempty'] += 1
                    hyperparam_fields[repo].update(hyperparams.keys())
                else:
                    stats['hyperparams_empty'] += 1

            except Exception as e:
                stats['json_parse_error'] += 1
                print(f"  ❌ 解析失败: {json_file}: {e}")

    # 打印统计
    print("=" * 80)
    print("JSON文件验证统计")
    print("=" * 80)
    print(f"Session目录数: {stats['total_sessions']}")
    print(f"实验目录数: {stats['total_experiments']}")
    print(f"JSON文件找到: {stats['json_found']} ({stats['json_found']/stats['total_experiments']*100:.1f}%)")
    print(f"JSON解析错误: {stats['json_parse_error']}")
    print()

    print(f"并行模式实验: {stats['parallel_mode']}")
    print(f"非并行模式实验: {stats['non_parallel_mode']}")
    print()

    print(f"超参数非空: {stats['hyperparams_nonempty']} ({stats['hyperparams_nonempty']/stats['json_found']*100:.1f}%)")
    print(f"超参数为空: {stats['hyperparams_empty']} ({stats['hyperparams_empty']/stats['json_found']*100:.1f}%)")
    print()

    print("=" * 80)
    print("各仓库实验数量")
    print("=" * 80)
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        print(f"{repo}: {count}")
    print()

    print("=" * 80)
    print("各仓库超参数字段")
    print("=" * 80)
    for repo, fields in sorted(hyperparam_fields.items()):
        if fields:
            print(f"{repo}:")
            for field in sorted(fields):
                print(f"  - {field}")

    return stats

if __name__ == '__main__':
    stats = validate_json_files()
```

**运行**:
```bash
cd /home/green/energy_dl/nightly
python3 scripts/validate_json_accessibility.py
```

**预期输出**:
- 总JSON文件数: ~726
- 超参数为空的比例: ~5-10%（主要是并行模式）
- 各仓库的超参数字段清单（用于构建映射表）

---

### 4.2 第二阶段：构建超参数映射表（1小时）

**目标**: 基于第一阶段的发现，完善超参数字段映射表

**任务**:
1. 汇总所有出现的超参数字段名
2. 根据语义归类（如`lr`和`learning_rate`都是学习率）
3. 确定统一的目标字段名
4. 编写映射字典

**输出**: `HYPERPARAM_FIELD_MAPPING` 字典（见3.2节）

---

### 4.3 第三阶段：实现数据提取脚本（2-3小时）

**脚本**: `analysis/scripts/extract_from_json_direct.py`

**功能模块**:

#### 模块1: JSON遍历与加载
```python
def load_all_experiments(results_dir):
    """遍历所有session目录，加载experiment.json"""
    experiments = []

    for session_dir in results_dir.glob('run_*'):
        for exp_dir in session_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            json_file = exp_dir / 'experiment.json'
            if not json_file.exists():
                continue

            try:
                with open(json_file) as f:
                    data = json.load(f)
                experiments.append(data)
            except Exception as e:
                print(f"⚠️ 跳过 {json_file}: {e}")

    return experiments
```

#### 模块2: 超参数提取与映射
```python
def extract_hyperparams(exp_data, field_mapping):
    """
    从experiment.json提取超参数，并应用字段映射

    Args:
        exp_data: 实验数据字典
        field_mapping: 超参数字段映射表

    Returns:
        dict: 统一后的超参数字典
    """
    is_parallel = exp_data.get('mode') == 'parallel'

    if is_parallel:
        raw_hyperparams = exp_data.get('foreground', {}).get('hyperparameters', {})
    else:
        raw_hyperparams = exp_data.get('hyperparameters', {})

    # 应用字段映射
    unified_hyperparams = {}
    for raw_key, raw_value in raw_hyperparams.items():
        # 查找映射
        if raw_key in field_mapping:
            unified_key = field_mapping[raw_key]
        else:
            # 未映射的字段保持原名（加hyperparam_前缀）
            unified_key = f'hyperparam_{raw_key}'

        unified_hyperparams[unified_key] = raw_value

    return unified_hyperparams
```

#### 模块3: 并行模式超参数回溯（可选）
```python
def backfill_parallel_hyperparams(exp_data, config_files):
    """
    尝试从配置文件回溯并行模式的超参数

    如果失败，返回空字典（后续用中位数插补）
    """
    # 实现省略（复杂度较高，可作为可选增强）
    return {}
```

#### 模块4: 数据转换为DataFrame
```python
def experiments_to_dataframe(experiments, field_mapping):
    """
    将实验列表转换为DataFrame

    Args:
        experiments: 实验数据列表
        field_mapping: 超参数字段映射表

    Returns:
        pd.DataFrame: 包含所有实验的DataFrame
    """
    rows = []

    for exp in experiments:
        row = {}

        is_parallel = exp.get('mode') == 'parallel'

        # 提取基础字段
        if is_parallel:
            fg = exp.get('foreground', {})
            row['experiment_id'] = exp.get('experiment_id')
            row['timestamp'] = exp.get('timestamp')
            row['repository'] = fg.get('repository')
            row['model'] = fg.get('model')
            row['mode'] = 'parallel'
            row['is_parallel'] = 1

            # 超参数
            hyperparams = extract_hyperparams(exp, field_mapping)
            row.update(hyperparams)

            # 能耗和性能
            row.update(extract_energy_metrics(fg.get('energy_metrics', {})))
            row.update(extract_performance_metrics(fg.get('performance_metrics', {})))
        else:
            row['experiment_id'] = exp.get('experiment_id')
            row['timestamp'] = exp.get('timestamp')
            row['repository'] = exp.get('repository')
            row['model'] = exp.get('model')
            row['mode'] = exp.get('mode', 'default')
            row['is_parallel'] = 0

            # 超参数
            hyperparams = extract_hyperparams(exp, field_mapping)
            row.update(hyperparams)

            # 能耗和性能
            row.update(extract_energy_metrics(exp.get('energy_metrics', {})))
            row.update(extract_performance_metrics(exp.get('performance_metrics', {})))

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
```

#### 模块5: 缺失值处理
```python
def handle_missing_values(df):
    """
    处理缺失值

    策略:
    1. 删除性能指标全缺失的行
    2. 删除能耗指标全缺失的行
    3. 对超参数列进行中位数插补（按任务组分组）
    """
    # 1. 删除性能指标全缺失的行
    perf_cols = [c for c in df.columns if c.startswith('perf_')]
    df_clean = df[df[perf_cols].notna().any(axis=1)]
    print(f"  删除性能全缺失: {len(df) - len(df_clean)} 行")

    # 2. 删除能耗指标全缺失的行
    energy_cols = [c for c in df.columns if c.startswith('energy_')]
    df_clean = df_clean[df_clean[energy_cols].notna().any(axis=1)]
    print(f"  删除能耗全缺失: {len(df) - len(df_clean)} 行")

    # 3. 超参数插补（按任务组）
    # 这里需要先做任务分组...

    return df_clean
```

---

### 4.4 第四阶段：数据分层与保存（1小时）

**目标**: 按4个任务组分层保存数据

**任务**:
1. 按`repository`分组（examples, Person_reID, VulBERTa, bug-localization）
2. 添加One-Hot编码列
3. 选择任务特定的变量
4. 保存为`training_data_{task}.csv`

**输出**:
- `analysis/data/energy_research/training/training_data_image_classification.csv`
- `analysis/data/energy_research/training/training_data_person_reid.csv`
- `analysis/data/energy_research/training/training_data_vulberta.csv`
- `analysis/data/energy_research/training/training_data_bug_localization.csv`

---

### 4.5 第五阶段：数据质量验证（30分钟）

**脚本**: `analysis/scripts/validate_extracted_data.py`

**验证项**:
1. ✅ 缺失率 < 目标值
2. ✅ 完全无缺失行比例 > 目标值
3. ✅ 相关性矩阵可计算（无nan）
4. ✅ 样本量充足（每组 > 80）
5. ✅ 超参数填充率 > 90%

**通过条件**: 所有验证项通过

---

### 4.6 第六阶段：重新运行DiBS分析（2小时）

**任务**: 使用新提取的数据重新运行因果分析

**脚本**: `analysis/scripts/experiments/run_energy_causal_analysis.sh`

**预期结果**:
- 图像分类: 发现 **3-6条因果边**
- Person_reID: 发现 **2-5条因果边**
- VulBERTa: 发现 **1-3条因果边**
- Bug定位: 发现 **1-3条因果边**

---

## 五、时间表与里程碑

| 阶段 | 任务 | 预计时间 | 负责人 | 状态 |
|------|------|----------|--------|------|
| **阶段1** | 验证JSON可访问性 | 0.5小时 | Claude | ⏳ 待开始 |
| **阶段2** | 构建超参数映射表 | 1小时 | Claude | ⏳ 待开始 |
| **阶段3** | 实现数据提取脚本 | 2-3小时 | Claude | ⏳ 待开始 |
| **阶段4** | 数据分层与保存 | 1小时 | Claude | ⏳ 待开始 |
| **阶段5** | 数据质量验证 | 0.5小时 | Claude | ⏳ 待开始 |
| **阶段6** | 重新运行DiBS分析 | 2小时 | Claude | ⏳ 待开始 |
| **总计** | - | **7-8小时** | - | - |

**最快完成时间**: 1个工作日
**推荐完成时间**: 2个工作日（留出调试和验证时间）

---

## 六、风险与缓解措施

### 风险1: 并行模式超参数回溯失败

**风险等级**: 🟡 中等

**描述**: 从配置文件回溯超参数可能失败（配置文件缺失或格式不兼容）

**影响**: 并行模式实验仍有超参数缺失

**缓解措施**:
1. **主策略**: 使用中位数插补（按任务组分组）
2. **备选策略**: 手动填充（查看配置文件或实验日志）
3. **最坏情况**: 删除该行（样本量充足，影响有限）

---

### 风险2: 数据质量仍不达标

**风险等级**: 🟢 低

**描述**: 即使重新提取，数据质量可能仍有问题

**影响**: DiBS仍无法学习因果边

**缓解措施**:
1. **诊断**: 使用`validate_extracted_data.py`详细检查
2. **迭代**: 根据验证报告调整提取逻辑
3. **降级方案**: 仅使用完全无缺失的行（删除更多样本）

---

### 风险3: 实施时间超预期

**风险等级**: 🟡 中等

**描述**: 脚本开发和调试可能需要更长时间

**影响**: 延迟因果分析结果

**缓解措施**:
1. **分阶段验证**: 每阶段完成后立即测试
2. **优先级排序**: 先解决超参数缺失（最严重问题）
3. **渐进式改进**: 第一版先达到基本可用，后续迭代优化

---

## 七、成功标准

### 7.1 数据质量标准

| 指标 | 当前值 | 目标值 | 验证方法 |
|------|--------|--------|----------|
| **总体缺失率** | 8-28% | < 3% | `df.isnull().sum().sum()` |
| **超参数填充率** | 32-100%缺失 | > 90% | 每列单独检查 |
| **完全无缺失行** | 0-64.7% | > 90% | `df.dropna()` |
| **相关性可计算** | 失败(nan) | 成功 | `df.corr()` |
| **样本量** | 80-258 | > 80 | `len(df)` |

---

### 7.2 因果分析标准

| 任务组 | 当前因果边数 | 目标因果边数 | DiBS迭代次数 |
|--------|--------------|--------------|--------------|
| 图像分类 | 0 | **3-6条** | 3000 |
| Person_reID | 0 | **2-5条** | 3000 |
| VulBERTa | 0 | **1-3条** | 3000 |
| Bug定位 | 0 | **1-3条** | 3000 |

**参考**: Adult数据集（10样本）发现6条边，能耗数据（80-258样本）应能发现更多

---

### 7.3 可交付成果

✅ **代码**:
1. `scripts/validate_json_accessibility.py` - JSON验证脚本
2. `analysis/scripts/extract_from_json_direct.py` - 数据提取脚本
3. `analysis/scripts/validate_extracted_data.py` - 数据验证脚本

✅ **数据**:
4. `analysis/data/energy_research/training/training_data_*.csv` (4个文件，高质量)

✅ **报告**:
5. `analysis/docs/reports/DATA_REEXTRACTION_EXECUTION_REPORT.md` - 执行报告
6. `analysis/docs/reports/DATA_QUALITY_COMPARISON_REPORT.md` - 新旧数据对比

✅ **因果分析结果**:
7. `analysis/results/energy_research/task_specific/*.npy` - DiBS因果图
8. `analysis/results/energy_research/task_specific/*.pkl` - 因果边和效应

---

## 八、替代方案对比

### 方案A: 重新提取（推荐）⭐⭐⭐

**优点**:
- ✅ 从源头解决问题，数据质量最高
- ✅ 可以完全控制字段映射和缺失值处理
- ✅ 未来可复用（新实验数据也能正确提取）

**缺点**:
- ⚠️ 开发时间较长（7-8小时）
- ⚠️ 需要深入理解JSON结构

**适用场景**: 当前情况（数据质量严重不足，需要彻底修复）

---

### 方案B: 仅使用完全无缺失的行

**优点**:
- ✅ 实施简单（30分钟）
- ✅ 数据质量有保证

**缺点**:
- ❌ 样本量大幅减少（图像分类: 258 → 125, VulBERTa: 142 → 0 ❌）
- ❌ 可能引入选择偏差（只有某些配置无缺失）
- ❌ VulBERTa和Bug定位**完全不可用**（0行无缺失）

**适用场景**: 仅作为快速验证，不适合正式分析

---

### 方案C: 简单插补（未做字段映射）

**优点**:
- ✅ 实施较快（2-3小时）
- ✅ 保留所有样本

**缺点**:
- ❌ 无法解决字段命名不统一问题
- ❌ Bug定位的`learning_rate`仍100%缺失（无法插补）
- ❌ 并行模式超参数仍为空（无法插补）

**适用场景**: 数据质量问题较轻微的情况（但当前不适用）

---

## 九、下一步行动

### 立即行动（今天）

1. **用户确认方案** ✅
   - 确认采用方案A（重新提取）
   - 确认时间预算（7-8小时可接受）

2. **阶段1: JSON验证**（30分钟）
   - 运行`validate_json_accessibility.py`
   - 确认所有JSON文件可访问
   - 汇总超参数字段清单

3. **阶段2: 构建映射表**（1小时）
   - 基于阶段1的发现完善`HYPERPARAM_FIELD_MAPPING`
   - 用户确认映射规则

### 明天行动

4. **阶段3-4: 实现提取脚本**（3-4小时）
   - 编写`extract_from_json_direct.py`
   - 单元测试
   - 生成4个训练数据文件

5. **阶段5-6: 验证与分析**（2.5小时）
   - 数据质量验证
   - 重新运行DiBS
   - 生成对比报告

---

## 十、附录

### 附录A: 主要数据文件清单

| 文件路径 | 类型 | 行数 | 列数 | 用途 |
|---------|------|------|------|------|
| `data/raw_data.csv` | 主数据 | 726 | 87 | 主项目汇总数据 |
| `data/data.csv` | 精简 | 726 | 56 | 主项目精简数据 |
| `analysis/data/energy_research/raw/energy_data_original.csv` | 副本 | 726 | 56 | analysis模块原始数据 |
| `analysis/data/energy_research/training/training_data_*.csv` | 训练 | 80-258 | 13-17 | **待重新生成** ⚠️ |

### 附录B: 关键配置文件位置

- 模型配置: `mutation/models_config.json`
- 实验配置: `settings/stage*.json`, `settings/*parallel*.json`
- DiBS配置: `analysis/config_energy.py`

### 附录C: 相关脚本位置

**主项目**:
- `tools/data_management/append_session_to_raw_data.py` - 从session追加数据
- `tools/data_management/create_unified_data_csv.py` - 生成data.csv

**analysis模块**:
- `analysis/scripts/stage0_data_validation.py` - 数据验证
- `analysis/scripts/stage1_hyperparam_unification.py` - 超参数统一
- `analysis/scripts/experiments/run_energy_causal_analysis.sh` - 运行因果分析

---

**报告人**: Claude
**生成时间**: 2025-12-24
**状态**: ⏳ 等待用户确认方案
**优先级**: 🔴 P0 - 最高优先级
