# 能耗因果分析 - 数据重新提取方案 v2.0

**日期**: 2025-12-24
**版本**: v2.0 (整合默认值回溯机制)
**目的**: 从主实验JSON文件重新提取数据，解决缺失值问题
**优先级**: 🔴 **P0 - 最高优先级**

---

## 📋 版本更新说明

### v2.0 vs v1.0 关键变更 ⭐⭐⭐

**v1.0问题** (已废弃):
- ❌ 假设非并行模式的hyperparameters字段完整
- ❌ 未考虑默认值存储在models_config.json中
- ❌ 缺少从model config回溯默认值的机制

**v2.0改进**:
- ✅ **核心发现**: 实验JSON中只记录**被变异的超参数**，其他使用默认值
- ✅ **默认值来源**: 从`mutation/models_config.json`的`"default"`字段提取
- ✅ **完整回溯机制**: 结合JSON记录值 + model config默认值
- ✅ **字段映射增强**: 处理不同模型的超参数命名差异

---

## 执行摘要

### 问题根源

当前因果分析数据的缺失值问题源于：

1. **部分超参数记录**: experiment.json只记录被变异的超参数 ❌
   - 示例：Bug定位default_017只记录`{"seed": 917}`，缺少max_iter/alpha/kfold
   - 示例：VulBERTa mlp_001只记录`{"epochs": 14}`，缺少learning_rate/weight_decay/seed

2. **默认值未提取**: 未从models_config.json中回溯默认值 ❌
   - Bug定位缺失的max_iter默认值 = 10000 (在models_config.json中)
   - VulBERTa缺失的learning_rate默认值 = 0.00003

3. **并行模式hyperparameters为空**: 需要额外处理 ❌

4. **超参数命名不统一**: 不同模型使用不同字段名 ❌

### 解决方案

**方案**: 从主实验的`experiment.json` + `models_config.json`联合提取数据，实现：

1. **从model config回溯默认值**（解决部分记录问题） ⭐⭐⭐
2. **超参数字段统一映射**（不同命名 → 统一变量名）
3. **并行模式特殊处理**（foreground字段提取）
4. **完整的缺失值处理**（合理插补 + 行删除）

**预期改进**:
- 超参数缺失率: 32-100% → **< 5%**
- 完全无缺失行: 0-64.7% → **> 90%**
- DiBS因果边数: 0条 → **3-8条/任务组**

---

## 一、关键发现：默认值存储机制

### 1.1 实验JSON的超参数记录规则

**规则**: experiment.json **仅记录被变异的超参数**

**证据1: Bug定位默认值实验** (bug-localization_default_017):
```json
{
  "experiment_id": "bug-localization-by-dnn-and-rvsm_default_017",
  "repository": "bug-localization-by-dnn-and-rvsm",
  "hyperparameters": {
    "seed": 917  // ✅ 只记录seed（被变异）
    // ❌ max_iter, alpha, kfold未记录（使用默认值）
  }
}
```

**证据2: VulBERTa变异实验** (VulBERTa_mlp_001):
```json
{
  "experiment_id": "VulBERTa_mlp_001",
  "repository": "VulBERTa",
  "hyperparameters": {
    "epochs": 14  // ✅ 只记录epochs（被变异）
    // ❌ learning_rate, weight_decay, seed未记录（使用默认值）
  }
}
```

**证据3: MRT-OAST默认值实验** (MRT-OAST_default_004):
```json
{
  "hyperparameters": {
    "learning_rate": 0.00011495933376973943  // ✅ 只记录learning_rate
    // ❌ epochs, dropout, weight_decay未记录
  }
}
```

### 1.2 默认值定义位置

**文件**: `mutation/models_config.json`

**结构**: 每个repo的每个超参数都有`"default"`字段

**示例1: Bug定位的默认值**:
```json
"bug-localization-by-dnn-and-rvsm": {
  "supported_hyperparams": {
    "max_iter": {
      "flag": "--max_iter",
      "type": "int",
      "default": 10000,  // ⭐ 默认值
      "range": [1000, 20000]
    },
    "alpha": {
      "flag": "--alpha",
      "type": "float",
      "default": 0.00001,  // ⭐ 默认值
      "range": [0.000005, 0.00002]
    },
    "kfold": {
      "flag": "--kfold",
      "type": "int",
      "default": 10,  // ⭐ 默认值
      "range": [2, 10]
    },
    "seed": {
      "default": 42,  // ⭐ 默认值
      "range": [0, 9999]
    }
  }
}
```

**示例2: VulBERTa的默认值**:
```json
"VulBERTa": {
  "supported_hyperparams": {
    "epochs": {
      "default": 10,  // ⭐
      "range": [5, 20]
    },
    "learning_rate": {
      "default": 0.00003,  // ⭐
      "range": [0.000015, 0.000045]
    },
    "weight_decay": {
      "default": 0.0,  // ⭐
      "range": [0.00001, 0.001]
    },
    "seed": {
      "default": 42  // ⭐
    }
  }
}
```

**示例3: CIFAR-10的默认值**:
```json
"pytorch_resnet_cifar10": {
  "supported_hyperparams": {
    "epochs": {
      "default": 200,  // ⭐
      "range": [100, 300]
    },
    "learning_rate": {
      "flag": "--lr",  // 注意：flag是--lr，但字段名是learning_rate
      "default": 0.1,  // ⭐
      "range": [0.05, 0.15]
    },
    "weight_decay": {
      "flag": "--wd",
      "default": 0.0001,  // ⭐
      "range": [0.00001, 0.001]
    },
    "seed": {
      "default": 1334  // ⭐
    }
  }
}
```

---

## 二、完整超参数提取逻辑

### 2.1 提取流程图

```
┌─────────────────────────────────────────────────────────────┐
│  读取 experiment.json                                        │
│  - experiment_id, repository, model, mode                    │
│  - hyperparameters (部分记录)                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  加载 models_config.json                                     │
│  - 定位 config[repository]["supported_hyperparams"]          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  对于每个超参数 param:                                       │
│  1. 检查 experiment.json["hyperparameters"][param]          │
│     - 如果存在 → 使用记录值 ✅                              │
│     - 如果不存在 → 从model config提取默认值 ⭐              │
│  2. 应用字段映射（统一命名）                                │
│     - max_iter → training_duration                           │
│     - weight_decay → l2_regularization                       │
│     - learning_rate → hyperparam_learning_rate               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  输出完整超参数字典                                          │
│  - 所有超参数都有值（记录值 or 默认值）                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心代码逻辑（伪代码）

```python
def extract_complete_hyperparams(experiment_json, models_config):
    """
    从experiment.json + models_config联合提取完整超参数

    Args:
        experiment_json: 单个实验的JSON数据
        models_config: 从models_config.json加载的配置

    Returns:
        dict: 完整的超参数字典（所有参数都有值）
    """
    repo = experiment_json.get('repository')
    model = experiment_json.get('model')
    is_parallel = experiment_json.get('mode') == 'parallel'

    # 1. 获取实验中记录的超参数（部分）
    if is_parallel:
        recorded_params = experiment_json.get('foreground', {}).get('hyperparameters', {})
    else:
        recorded_params = experiment_json.get('hyperparameters', {})

    # 2. 获取model config中定义的所有支持的超参数
    supported_params = models_config['models'][repo]['supported_hyperparams']

    # 3. 合并：记录值优先，未记录的用默认值
    complete_params = {}

    for param_name, param_config in supported_params.items():
        if param_name in recorded_params:
            # 使用实验中记录的值（被变异的）
            value = recorded_params[param_name]
        else:
            # 使用model config中的默认值（未变异的）
            value = param_config['default']

        # 4. 应用字段映射（统一命名）
        unified_name = apply_field_mapping(repo, param_name)
        complete_params[unified_name] = value

    return complete_params


def apply_field_mapping(repo, param_name):
    """
    应用字段映射规则，统一超参数命名

    Args:
        repo: 仓库名（如'bug-localization-by-dnn-and-rvsm'）
        param_name: 原始参数名（如'max_iter'）

    Returns:
        str: 统一后的字段名（如'training_duration'）
    """
    # 全局映射表
    GLOBAL_MAPPING = {
        # 训练迭代次数统一
        'epochs': 'training_duration',
        'max_iter': 'training_duration',
        'num_iters': 'training_duration',

        # L2正则化统一
        'weight_decay': 'l2_regularization',
        'alpha': 'l2_regularization',  # Bug定位的alpha是L2正则化

        # 学习率统一
        'learning_rate': 'hyperparam_learning_rate',
        'lr': 'hyperparam_learning_rate',

        # 批量大小统一
        'batch_size': 'hyperparam_batch_size',
        'train_batch_size': 'hyperparam_batch_size',

        # Dropout统一
        'dropout': 'hyperparam_dropout',
        'droprate': 'hyperparam_dropout',
        'dropout_rate': 'hyperparam_dropout',

        # 其他参数保持原名
        'seed': 'seed',
        'kfold': 'hyperparam_kfold',
    }

    # 特殊仓库映射（如果全局映射不适用）
    REPO_SPECIFIC_MAPPING = {
        'bug-localization-by-dnn-and-rvsm': {
            'alpha': 'l2_regularization',  # Bug定位的alpha是L2正则
        },
        'MRT-OAST': {
            'alpha': 'l2_regularization',  # MRT-OAST的alpha也是L2正则（根据models_config.json）
        }
    }

    # 先检查仓库特定映射
    if repo in REPO_SPECIFIC_MAPPING and param_name in REPO_SPECIFIC_MAPPING[repo]:
        return REPO_SPECIFIC_MAPPING[repo][param_name]

    # 再检查全局映射
    if param_name in GLOBAL_MAPPING:
        return GLOBAL_MAPPING[param_name]

    # 未映射的保持原名（加hyperparam_前缀）
    return f'hyperparam_{param_name}'
```

### 2.3 示例：Bug定位实验的完整提取

**输入1: experiment.json** (bug-localization_default_017):
```json
{
  "repository": "bug-localization-by-dnn-and-rvsm",
  "hyperparameters": {
    "seed": 917  // 只有这一个被记录
  }
}
```

**输入2: models_config.json**:
```json
"bug-localization-by-dnn-and-rvsm": {
  "supported_hyperparams": {
    "max_iter": {"default": 10000},
    "alpha": {"default": 0.00001},
    "kfold": {"default": 10},
    "seed": {"default": 42}
  }
}
```

**提取过程**:
```python
# 1. 检查max_iter
# experiment.json中没有 → 使用默认值10000
# 应用映射: max_iter → training_duration
# 结果: training_duration = 10000

# 2. 检查alpha
# experiment.json中没有 → 使用默认值0.00001
# 应用映射: alpha → l2_regularization
# 结果: l2_regularization = 0.00001

# 3. 检查kfold
# experiment.json中没有 → 使用默认值10
# 应用映射: kfold → hyperparam_kfold
# 结果: hyperparam_kfold = 10

# 4. 检查seed
# experiment.json中有917 → 使用记录值917（覆盖默认值42）
# 应用映射: seed → seed
# 结果: seed = 917
```

**最终输出**:
```python
{
    'training_duration': 10000,     # 从默认值
    'l2_regularization': 0.00001,   # 从默认值
    'hyperparam_kfold': 10,         # 从默认值
    'seed': 917                      # 从记录值（覆盖默认值42）
}
```

**这解决了Bug定位learning_rate 100%缺失的问题！** ⭐⭐⭐

---

## 三、超参数字段映射表（完整版）

### 3.1 全局映射规则

| 原始字段名 | 统一字段名 | 使用模型 | 说明 |
|-----------|-----------|----------|------|
| **训练迭代次数** ||||
| `epochs` | `training_duration` | examples, VulBERTa, Person_reID, CIFAR-10, MRT-OAST | 训练轮数 |
| `max_iter` | `training_duration` | Bug定位 | 最大迭代次数 |
| **学习率** ||||
| `learning_rate` | `hyperparam_learning_rate` | examples, VulBERTa, MRT-OAST, Person_reID | 标准命名 |
| `lr` | `hyperparam_learning_rate` | CIFAR-10 | 简写形式 |
| **L2正则化** ||||
| `weight_decay` | `l2_regularization` | VulBERTa, CIFAR-10 | 权重衰减 |
| `alpha` | `l2_regularization` | Bug定位, MRT-OAST | L2正则化系数 |
| **批量大小** ||||
| `batch_size` | `hyperparam_batch_size` | examples | 标准命名 |
| **Dropout** ||||
| `dropout` | `hyperparam_dropout` | MRT-OAST | 标准命名 |
| `droprate` | `hyperparam_dropout` | Person_reID | flag形式 |
| **其他超参数** ||||
| `seed` | `seed` | 所有模型 | 随机种子 |
| `kfold` | `hyperparam_kfold` | Bug定位 | K折交叉验证 |

### 3.2 特殊映射规则

**Bug定位 (bug-localization-by-dnn-and-rvsm)**:
```python
SPECIAL_MAPPING = {
    'max_iter': 'training_duration',  # 统一到训练持续时间
    'alpha': 'l2_regularization',      # alpha是L2正则化系数
    'kfold': 'hyperparam_kfold',       # K折验证
    'seed': 'seed'
}

DEFAULT_VALUES = {
    'max_iter': 10000,
    'alpha': 0.00001,
    'kfold': 10,
    'seed': 42
}
```

**VulBERTa**:
```python
SPECIAL_MAPPING = {
    'epochs': 'training_duration',
    'learning_rate': 'hyperparam_learning_rate',
    'weight_decay': 'l2_regularization',
    'seed': 'seed'
}

DEFAULT_VALUES = {
    'epochs': 10,
    'learning_rate': 0.00003,
    'weight_decay': 0.0,
    'seed': 42
}
```

**CIFAR-10 (pytorch_resnet_cifar10)**:
```python
SPECIAL_MAPPING = {
    'epochs': 'training_duration',
    'learning_rate': 'hyperparam_learning_rate',  # 注意：flag是--lr
    'weight_decay': 'l2_regularization',          # flag是--wd
    'seed': 'seed'
}

DEFAULT_VALUES = {
    'epochs': 200,
    'learning_rate': 0.1,
    'weight_decay': 0.0001,
    'seed': 1334
}
```

**Examples (MNIST系列)**:
```python
SPECIAL_MAPPING = {
    'epochs': 'training_duration',
    'learning_rate': 'hyperparam_learning_rate',
    'batch_size': 'hyperparam_batch_size',
    'seed': 'seed'
}

DEFAULT_VALUES = {
    'epochs': 10,
    'learning_rate': 0.01,
    'batch_size': 32,
    'seed': 1
}
```

---

## 四、并行模式特殊处理

### 4.1 并行模式JSON结构

**示例**: bug-localization_default_001_parallel
```json
{
  "experiment_id": "bug-localization-by-dnn-and-rvsm_default_001_parallel",
  "mode": "parallel",
  "foreground": {
    "repository": "bug-localization-by-dnn-and-rvsm",
    "model": "default",
    "hyperparameters": {},  // ❌ 空的（和非并行一样，未记录的用默认值）
    "energy_metrics": {...},
    "performance_metrics": {...}
  },
  "background": {
    "repository": "examples",
    "model": "mnist"
  }
}
```

### 4.2 并行模式提取逻辑

```python
def extract_parallel_experiment(experiment_json, models_config):
    """
    提取并行模式实验的数据

    Args:
        experiment_json: 并行模式的JSON数据
        models_config: models_config.json配置

    Returns:
        dict: 包含foreground完整数据的行
    """
    fg = experiment_json.get('foreground', {})
    repo = fg.get('repository')
    model = fg.get('model')

    # 1. 提取foreground的超参数（可能为空）
    recorded_params = fg.get('hyperparameters', {})

    # 2. 从model config回溯完整超参数
    supported_params = models_config['models'][repo]['supported_hyperparams']

    complete_params = {}
    for param_name, param_config in supported_params.items():
        if param_name in recorded_params:
            value = recorded_params[param_name]  # 使用记录值
        else:
            value = param_config['default']      # 使用默认值 ⭐

        # 应用字段映射
        unified_name = apply_field_mapping(repo, param_name)
        complete_params[unified_name] = value

    # 3. 提取能耗和性能指标
    row = {
        'experiment_id': experiment_json['experiment_id'],
        'timestamp': experiment_json['timestamp'],
        'repository': repo,
        'model': model,
        'mode': 'parallel',
        'is_parallel': 1,
        **complete_params,  # 完整超参数
        **extract_energy_metrics(fg.get('energy_metrics', {})),
        **extract_performance_metrics(fg.get('performance_metrics', {}))
    }

    return row
```

**关键改进**:
- ✅ 并行模式的空hyperparameters不再是问题
- ✅ 从model config回溯默认值，和非并行模式一致
- ✅ 所有并行实验都将有完整超参数

---

## 五、缺失值处理策略

### 5.1 优先级分层（和v1.0一致）

| 列类型 | 处理策略 | 原因 |
|--------|----------|------|
| **超参数列** | ~~中位数插补~~ **从默认值回溯** ⭐ | v2.0不需要插补，全部从model config提取 |
| **性能指标列** | 删除该行 | 目标变量，插补会严重偏倚因果估计 |
| **能耗指标列** | 删除该行 | 目标变量，不能插补 |
| **元信息列** | 必须完整 | experiment_id, timestamp等不能缺失 |

### 5.2 超参数处理（v2.0改进）

**v1.0方法** (已废弃):
```python
# 按任务组 × One-Hot分组插补
mnist_lr_median = df[(df['is_mnist']==1) & df['hyperparam_learning_rate'].notna()]['hyperparam_learning_rate'].median()
df.loc[(df['is_mnist']==1) & df['hyperparam_learning_rate'].isnull(), 'hyperparam_learning_rate'] = mnist_lr_median
```

**v2.0方法** ⭐⭐⭐:
```python
# 从model config直接提取，不需要插补！
# 提取时就已经完整，不会有缺失
for exp in experiments:
    complete_params = extract_complete_hyperparams(exp, models_config)
    # complete_params已经包含所有超参数（记录值 + 默认值）
```

**预期结果**:
- ✅ Bug定位的max_iter: 100%缺失 → **0%缺失**（从默认值10000）
- ✅ CIFAR-10的batch_size: 100%缺失 → **0%缺失**（从默认值...需要确认）
- ✅ VulBERTa的learning_rate: 64.8%缺失 → **0%缺失**（从默认值0.00003）

### 5.3 行删除规则（和v1.0一致）

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

## 六、数据质量目标（更新）

| 任务组 | 当前缺失率 | v2.0目标缺失率 | 当前完全无缺失行 | v2.0目标完全无缺失行 |
|--------|------------|----------------|------------------|---------------------|
| 图像分类 | 8.83% | **< 1%** ⭐ | 48.4% | **> 99%** ⭐ |
| Person_reID | 4.96% | **< 1%** ⭐ | 64.7% | **> 99%** ⭐ |
| VulBERTa | 28.87% | **< 2%** ⭐ | 0% | **> 85%** ⭐ |
| Bug定位 | 24.38% | **< 2%** ⭐ | 0% | **> 85%** ⭐ |

**v2.0改进点**:
- ✅ 超参数缺失率从32-100% → **接近0%**（全部从默认值回溯）
- ✅ 完全无缺失行比例大幅提升（只剩性能/能耗缺失）
- ✅ DiBS相关性矩阵计算成功率接近100%

**总体目标**:
- ✅ 所有超参数列填充率 > **99%**（v1.0目标: 90%）
- ✅ 所有任务组至少80个完全无缺失的行
- ✅ 相关性矩阵可计算（无nan）
- ✅ DiBS能发现因果边（预期3-8条/任务组）

---

## 七、实施计划（v2.0）

### 阶段1: 验证model config可访问性（30分钟）

**目标**: 确认models_config.json完整性

**脚本**: `scripts/validate_models_config.py`

```python
#!/usr/bin/env python3
"""验证models_config.json的完整性和有效性"""

import json
from pathlib import Path
from collections import defaultdict

def validate_models_config():
    """验证models_config.json"""

    config_file = Path('mutation/models_config.json')

    with open(config_file) as f:
        config = json.load(f)

    print("=" * 80)
    print("models_config.json 验证")
    print("=" * 80)

    # 统计
    stats = {
        'total_repos': 0,
        'total_models': 0,
        'total_hyperparams': 0,
        'missing_defaults': []
    }

    hyperparam_coverage = defaultdict(int)

    for repo, repo_config in config['models'].items():
        stats['total_repos'] += 1
        stats['total_models'] += len(repo_config.get('models', []))

        print(f"\n{repo}:")
        print(f"  模型: {repo_config.get('models', [])}")

        supported = repo_config.get('supported_hyperparams', {})
        print(f"  超参数数量: {len(supported)}")

        for param_name, param_config in supported.items():
            stats['total_hyperparams'] += 1
            hyperparam_coverage[param_name] += 1

            # 检查默认值
            if 'default' not in param_config:
                stats['missing_defaults'].append(f"{repo}.{param_name}")
                print(f"    ❌ {param_name}: 缺少default字段")
            else:
                default_val = param_config['default']
                param_type = param_config.get('type', 'unknown')
                print(f"    ✅ {param_name}: default={default_val} (type={param_type})")

    print("\n" + "=" * 80)
    print("超参数覆盖统计")
    print("=" * 80)
    for param, count in sorted(hyperparam_coverage.items(), key=lambda x: -x[1]):
        print(f"{param}: {count} 个仓库")

    print("\n" + "=" * 80)
    print("总体统计")
    print("=" * 80)
    print(f"仓库数: {stats['total_repos']}")
    print(f"模型总数: {stats['total_models']}")
    print(f"超参数总数: {stats['total_hyperparams']}")
    print(f"缺少默认值的参数: {len(stats['missing_defaults'])}")

    if stats['missing_defaults']:
        print("\n⚠️ 以下参数缺少默认值:")
        for param in stats['missing_defaults']:
            print(f"  - {param}")
    else:
        print("\n✅ 所有超参数都有默认值定义")

    return stats

if __name__ == '__main__':
    validate_models_config()
```

**运行**:
```bash
cd /home/green/energy_dl/nightly
python3 scripts/validate_models_config.py
```

**预期输出**:
- 6个仓库，11个模型
- 所有超参数都有default字段
- 超参数覆盖清单（如seed出现11次，epochs出现10次）

---

### 阶段2: 实现完整数据提取脚本（3-4小时）

**脚本**: `analysis/scripts/extract_from_json_with_defaults.py`

**核心功能**:

#### 模块1: 加载model config
```python
def load_models_config():
    """加载models_config.json"""
    config_path = Path('../mutation/models_config.json')
    with open(config_path) as f:
        return json.load(f)
```

#### 模块2: 完整超参数提取
```python
def extract_complete_hyperparams(exp_data, models_config):
    """
    从experiment.json + models_config联合提取完整超参数

    实现逻辑见第二节
    """
    # 实现代码...
```

#### 模块3: 字段映射
```python
def apply_field_mapping(repo, param_name):
    """
    应用字段映射规则

    实现逻辑见第三节
    """
    # 实现代码...
```

#### 模块4: 数据转换为DataFrame
```python
def experiments_to_dataframe(experiments, models_config):
    """
    将实验列表转换为DataFrame，包含完整超参数

    Args:
        experiments: 实验JSON列表
        models_config: models_config.json配置

    Returns:
        pd.DataFrame: 包含所有实验的DataFrame（超参数完整）
    """
    rows = []

    for exp in experiments:
        # 提取完整超参数
        complete_params = extract_complete_hyperparams(exp, models_config)

        # 构建行数据
        row = {
            'experiment_id': exp['experiment_id'],
            'timestamp': exp['timestamp'],
            'repository': exp.get('repository') or exp.get('foreground', {}).get('repository'),
            'model': exp.get('model') or exp.get('foreground', {}).get('model'),
            'mode': exp.get('mode', 'default'),
            'is_parallel': 1 if exp.get('mode') == 'parallel' else 0,
            **complete_params,  # ⭐ 完整超参数
            **extract_energy_metrics(exp),
            **extract_performance_metrics(exp)
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
```

#### 模块5: 缺失值处理（简化）
```python
def handle_missing_values(df):
    """
    处理缺失值（v2.0：主要删除行，超参数已完整）

    策略:
    1. 删除性能指标全缺失的行
    2. 删除能耗指标全缺失的行
    3. 超参数不需要插补（已从默认值回溯）
    """
    # 1. 删除性能指标全缺失的行
    perf_cols = [c for c in df.columns if c.startswith('perf_')]
    if perf_cols:
        df_clean = df[df[perf_cols].notna().any(axis=1)]
        print(f"  删除性能全缺失: {len(df) - len(df_clean)} 行")
    else:
        df_clean = df

    # 2. 删除能耗指标全缺失的行
    energy_cols = [c for c in df.columns if c.startswith('energy_')]
    if energy_cols:
        df_clean = df_clean[df_clean[energy_cols].notna().any(axis=1)]
        print(f"  删除能耗全缺失: {len(df) - len(df_clean)} 行")

    # 3. 验证超参数完整性
    hyperparam_cols = [c for c in df_clean.columns if c.startswith('hyperparam_') or c in ['training_duration', 'l2_regularization', 'seed']]
    missing_counts = df_clean[hyperparam_cols].isnull().sum()

    print("\n超参数缺失情况:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  ⚠️ {col}: {count} 行缺失 ({count/len(df_clean)*100:.2f}%)")
        else:
            print(f"  ✅ {col}: 完整")

    return df_clean
```

---

### 阶段3: 数据分层与保存（1小时）

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

### 阶段4: 数据质量验证（30分钟）

**脚本**: `analysis/scripts/validate_extracted_data_v2.py`

**验证项**:
1. ✅ 超参数缺失率 < 1%
2. ✅ 完全无缺失行比例 > 目标值（85-99%）
3. ✅ 相关性矩阵可计算（无nan）
4. ✅ 样本量充足（每组 > 80）
5. ✅ 超参数填充率 > 99%

**通过条件**: 所有验证项通过

---

### 阶段5: 重新运行DiBS分析（2小时）

**任务**: 使用新提取的数据重新运行因果分析

**脚本**: `analysis/scripts/experiments/run_energy_causal_analysis.sh`

**预期结果**:
- 图像分类: 发现 **3-6条因果边**
- Person_reID: 发现 **2-5条因果边**
- VulBERTa: 发现 **1-3条因果边**
- Bug定位: 发现 **1-3条因果边**

---

## 八、时间表与里程碑

| 阶段 | 任务 | 预计时间 | 负责人 | 状态 |
|------|------|----------|--------|------|
| **阶段1** | 验证model config完整性 | 0.5小时 | Claude | ⏳ 待开始 |
| **阶段2** | 实现数据提取脚本（含默认值回溯） | 3-4小时 | Claude | ⏳ 待开始 |
| **阶段3** | 数据分层与保存 | 1小时 | Claude | ⏳ 待开始 |
| **阶段4** | 数据质量验证 | 0.5小时 | Claude | ⏳ 待开始 |
| **阶段5** | 重新运行DiBS分析 | 2小时 | Claude | ⏳ 待开始 |
| **总计** | - | **7-8小时** | - | - |

**最快完成时间**: 1个工作日
**推荐完成时间**: 2个工作日（留出调试和验证时间）

---

## 九、风险与缓解措施

### 风险1: model config默认值缺失

**风险等级**: 🟡 中等

**描述**: 某些超参数可能没有定义默认值

**影响**: 无法回溯，仍有缺失

**缓解措施**:
1. **阶段1验证**: 检查所有超参数是否有default字段
2. **降级策略**: 如果缺少默认值，使用range的中位数
3. **最坏情况**: 使用v1.0的中位数插补方法

---

### 风险2: 数据质量仍不达标

**风险等级**: 🟢 低

**描述**: 即使重新提取，数据质量可能仍有问题

**影响**: DiBS仍无法学习因果边

**缓解措施**:
1. **诊断**: 使用`validate_extracted_data_v2.py`详细检查
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

## 十、成功标准

### 10.1 数据质量标准

| 指标 | 当前值 | v2.0目标值 | 验证方法 |
|------|--------|-----------|----------|
| **总体缺失率** | 8-28% | **< 2%** | `df.isnull().sum().sum()` |
| **超参数填充率** | 32-100%缺失 | **> 99%** ⭐ | 每列单独检查 |
| **完全无缺失行** | 0-64.7% | **> 85%** ⭐ | `df.dropna()` |
| **相关性可计算** | 失败(nan) | 成功 | `df.corr()` |
| **样本量** | 80-258 | > 80 | `len(df)` |

---

### 10.2 因果分析标准

| 任务组 | 当前因果边数 | 目标因果边数 | DiBS迭代次数 |
|--------|--------------|--------------|--------------| | 图像分类 | 0 | **3-6条** | 3000 |
| Person_reID | 0 | **2-5条** | 3000 |
| VulBERTa | 0 | **1-3条** | 3000 |
| Bug定位 | 0 | **1-3条** | 3000 |

**参考**: Adult数据集（10样本）发现6条边，能耗数据（80-258样本）应能发现更多

---

### 10.3 可交付成果

✅ **代码**:
1. `scripts/validate_models_config.py` - model config验证脚本
2. `analysis/scripts/extract_from_json_with_defaults.py` - 数据提取脚本（v2.0）
3. `analysis/scripts/validate_extracted_data_v2.py` - 数据验证脚本

✅ **数据**:
4. `analysis/data/energy_research/training/training_data_*.csv` (4个文件，高质量)

✅ **报告**:
5. `analysis/docs/reports/DATA_REEXTRACTION_EXECUTION_REPORT_V2.md` - 执行报告
6. `analysis/docs/reports/DATA_QUALITY_COMPARISON_V2.md` - v1.0 vs v2.0对比

✅ **因果分析结果**:
7. `analysis/results/energy_research/task_specific/*.npy` - DiBS因果图
8. `analysis/results/energy_research/task_specific/*.pkl` - 因果边和效应

---

## 十一、v1.0 vs v2.0 对比总结

| 维度 | v1.0 (已废弃) | v2.0 (推荐) | 改进 |
|------|--------------|------------|------|
| **超参数提取** | 直接读取experiment.json | JSON + model config联合提取 | ⭐⭐⭐ |
| **默认值处理** | 未考虑 | 从models_config.json回溯 | ⭐⭐⭐ |
| **缺失值插补** | 中位数插补 | 几乎不需要插补（已完整） | ⭐⭐⭐ |
| **预期超参数缺失率** | 5-10% | **< 1%** | ⭐⭐⭐ |
| **预期完全无缺失行** | 70-90% | **> 85%** | ⭐⭐ |
| **复杂度** | 中等 | 中等 | 相当 |
| **准确性** | 中等 | **高** | ⭐⭐⭐ |

**推荐**: **v2.0** - 更准确，更完整，更符合主项目设计 ⭐⭐⭐

---

## 十二、下一步行动

### 立即行动（今天）

1. **用户确认方案v2.0** ✅
   - 确认采用默认值回溯机制
   - 确认时间预算（7-8小时可接受）

2. **阶段1: model config验证**（30分钟）
   - 运行`validate_models_config.py`
   - 确认所有超参数有默认值
   - 汇总超参数清单

3. **阶段2: 构建提取脚本**（3-4小时）
   - 实现`extract_from_json_with_defaults.py`
   - 实现默认值回溯逻辑
   - 单元测试

### 明天行动

4. **阶段3-4: 数据生成与验证**（1.5小时）
   - 生成4个训练数据文件
   - 数据质量验证
   - 对比v1.0 vs v2.0

5. **阶段5: 重新运行DiBS**（2小时）
   - 执行因果分析
   - 生成对比报告

---

## 十三、附录

### 附录A: 完整模型超参数清单

| 模型 | 超参数1 | 超参数2 | 超参数3 | 超参数4 | 超参数5 |
|------|---------|---------|---------|---------|---------|
| **examples/mnist** | epochs (10) | learning_rate (0.01) | batch_size (32) | seed (1) | - |
| **VulBERTa/mlp** | epochs (10) | learning_rate (0.00003) | weight_decay (0.0) | seed (42) | - |
| **Bug定位** | max_iter (10000) | alpha (0.00001) | kfold (10) | seed (42) | - |
| **CIFAR-10** | epochs (200) | learning_rate (0.1) | weight_decay (0.0001) | seed (1334) | - |
| **Person_reID** | epochs (60) | learning_rate (0.05) | dropout (0.5) | seed (1334) | - |
| **MRT-OAST** | epochs (10) | learning_rate (0.0001) | dropout (0.2) | weight_decay (0.0) | seed (1334) |

**括号内为默认值**

### 附录B: 关键文件位置

**主项目**:
- `mutation/models_config.json` - 模型配置（包含默认值）⭐⭐⭐
- `results/run_*/*/experiment.json` - 实验结果JSON（部分记录）

**analysis模块**:
- `analysis/scripts/extract_from_json_with_defaults.py` - 数据提取脚本（待创建）
- `analysis/data/energy_research/training/training_data_*.csv` - 训练数据（待生成）
- `analysis/config_energy.py` - DiBS配置

---

**报告人**: Claude
**生成时间**: 2025-12-24
**版本**: v2.0
**状态**: ⏳ 等待用户确认方案v2.0
**优先级**: 🔴 P0 - 最高优先级

**关键改进**: 整合models_config.json默认值回溯机制，从根本上解决超参数缺失问题 ⭐⭐⭐
