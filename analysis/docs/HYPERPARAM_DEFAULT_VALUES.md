# 超参数默认值配置

**用途**: 为原始数据中的空值超参数回填默认值

**原则**: 原始数据中，默认配置的超参数记录为空值（NaN），需要根据各仓库的配置文件回填真实默认值。

---

## 默认值定义

### 通用默认值（适用于大多数模型）

| 超参数 | 默认值 | 适用范围 |
|--------|--------|---------|
| `learning_rate` | 0.001 | 所有模型 |
| `batch_size` | 32 | 所有模型 |
| `epochs` | 10 | 所有模型 |
| `dropout` | 0.0 | 所有模型（无dropout） |
| `weight_decay` | 0.0 | 所有模型（无正则化） |
| `seed` | 42 | 所有模型 |
| `alpha` | 0.1 | 有该参数的模型 |
| `kfold` | 5 | 使用交叉验证的模型 |
| `max_iter` | 100 | 迭代式模型 |

### 模型特定默认值（如有差异）

#### VulBERTa
- `learning_rate`: 2e-5 (BERT微调标准)
- `batch_size`: 16
- `epochs`: 3

#### Person_reID
- `learning_rate`: 0.1 (SGD优化器)
- `batch_size`: 64
- `epochs`: 60
- `weight_decay`: 5e-4

#### pytorch_resnet_cifar10
- `learning_rate`: 0.1
- `batch_size`: 128
- `epochs`: 200
- `weight_decay`: 5e-4

---

## 使用方法

### 回填逻辑

1. **优先使用模型特定默认值**（如果定义了）
2. **否则使用通用默认值**
3. **保持非空值不变**

### Python示例

```python
DEFAULT_VALUES = {
    'hyperparam_learning_rate': 0.001,
    'hyperparam_batch_size': 32,
    'hyperparam_epochs': 10,
    'hyperparam_dropout': 0.0,
    'hyperparam_weight_decay': 0.0,
    'hyperparam_seed': 42,
    'hyperparam_alpha': 0.1,
    'hyperparam_kfold': 5,
    'hyperparam_max_iter': 100
}

# 模型特定默认值
MODEL_SPECIFIC_DEFAULTS = {
    'VulBERTa': {
        'hyperparam_learning_rate': 2e-5,
        'hyperparam_batch_size': 16,
        'hyperparam_epochs': 3
    },
    'Person_reID_baseline_pytorch': {
        'hyperparam_learning_rate': 0.1,
        'hyperparam_batch_size': 64,
        'hyperparam_epochs': 60,
        'hyperparam_weight_decay': 5e-4
    },
    'pytorch_resnet_cifar10': {
        'hyperparam_learning_rate': 0.1,
        'hyperparam_batch_size': 128,
        'hyperparam_epochs': 200,
        'hyperparam_weight_decay': 5e-4
    }
}

# 回填函数
def backfill_defaults(df):
    df = df.copy()

    for col in DEFAULT_VALUES.keys():
        if col not in df.columns:
            continue

        # 对每个repository应用特定默认值
        for repo in df['repository'].unique():
            if pd.isna(repo):
                continue

            mask = (df['repository'] == repo) & (df[col].isna())

            # 使用模型特定默认值或通用默认值
            if repo in MODEL_SPECIFIC_DEFAULTS and col in MODEL_SPECIFIC_DEFAULTS[repo]:
                default_val = MODEL_SPECIFIC_DEFAULTS[repo][col]
            else:
                default_val = DEFAULT_VALUES[col]

            df.loc[mask, col] = default_val

    return df
```

---

## 验证

回填后，应验证：
1. ✅ 超参数列缺失率大幅下降（目标 < 5%）
2. ✅ 默认配置实验的超参数值非空
3. ✅ 变异配置实验的超参数值保持不变

---

**文档创建**: 2026-01-06
**用途**: 为分层DiBS数据准备提供默认值回填依据
