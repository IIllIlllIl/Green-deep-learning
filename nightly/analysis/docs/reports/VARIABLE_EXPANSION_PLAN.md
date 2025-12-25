# 因果分析变量扩展方案

**文档版本**: v3.0
**创建时间**: 2025-12-22
**最后更新**: 2025-12-22
**状态**: ✅ 全部方案已确认（最终版本）

---

## 📋 执行摘要

本文档记录了从原始15个变量（27.8%列使用率）扩展到更全面的因果分析变量集的决策过程。

### 核心决策

| 方案 | 状态 | 新增变量数 | 决策 |
|------|------|-----------|------|
| **方案1: 超参数统一** | ✅ 确认 | 2个统一变量 | 合并语义相同的超参数 |
| **方案2: 能耗中介变量** | ✅ 确认 | 5个中介变量 | 加入GPU利用率、温度、波动指标 |
| **方案3: 性能指标** | ✅ 确认 | **4个任务组 + One-Hot编码** | 合并MNIST+CIFAR-10，添加数据集/模型编码 |

### 变量数量对比

```
原始变量数: 15个 (27.8% 列使用率)
  - 超参数: 5个
  - 能耗: 2个
  - 性能: 2个（实际缺失）
  - 其他: 6个

扩展后变量数（每个任务组）:
  - 超参数: 6个（统一后：training_duration, l2_regularization, learning_rate, batch_size, dropout, seed）
  - 能耗总量: 2个（energy_cpu_total, energy_gpu_total）
  - 能耗中介: 5个（gpu_util_avg, gpu_temp_max, cpu_pkg_ratio, gpu_power_fluctuation, gpu_temp_fluctuation）
  - 性能指标: 1个（任务特定：test_accuracy, mAP, eval_loss, top1_accuracy）
  - One-Hot编码: 0-3个（数据集/模型区分变量）

总计: 14-17个变量/任务组（根据超参数填充率动态选择）
```

---

## ✅ 方案1: 超参数统一

### 1.1 问题背景

原始数据中存在**语义相同但命名不同**的超参数，导致数据稀疏：

| 原始列名 | 填充率 | 使用模型 | 语义 |
|---------|-------|---------|------|
| `hyperparam_epochs` | 高 | MNIST, ResNet, Person_reID, etc. | 训练轮数 |
| `hyperparam_max_iter` | 低 | bug-localization | 最大迭代次数 |
| `hyperparam_weight_decay` | 12.7% | VulBERTa, MRT-OAST, ResNet | L2正则化系数 (PyTorch) |
| `hyperparam_alpha` | 4.6% | bug-localization | L2正则化系数 (scikit-learn) |

**核心问题**：
- `max_iter` 和 `epochs` 功能相同：控制训练时长
- `alpha` 和 `weight_decay` **语义完全相同**：都是L2正则化系数
  - 源码证据：`main_batch.py:199` 明确标注 `"weight decay (L2 penalty)"`
  - 源码证据：`train_wrapper.py:96` 明确标注 `"L2 penalty parameter"`
  - 数据证据：676实验中，两者**完全互斥**（0行同时填充）

### 1.2 解决方案

#### 统一1: 训练时长

```python
# 变量名: hyperparam_training_duration
# 来源: epochs (优先) 或 max_iter

def unify_training_duration(df):
    """统一训练时长超参数"""
    df['hyperparam_training_duration'] = df['hyperparam_epochs'].fillna(
        df['hyperparam_max_iter']
    )
    return df
```

**数据分布**（预期）：
- 来源：~90% epochs, ~10% max_iter
- 互斥性：完全互斥（不同框架）
- 语义：控制训练迭代总次数

#### 统一2: L2正则化

```python
# 变量名: hyperparam_l2_regularization
# 来源: weight_decay (优先) 或 alpha

def unify_l2_regularization(df):
    """统一L2正则化超参数"""
    df['hyperparam_l2_regularization'] = df['hyperparam_weight_decay'].fillna(
        df['hyperparam_alpha']
    )

    # 可选：保留来源信息用于分层分析
    df['l2_reg_source'] = np.where(
        df['hyperparam_weight_decay'].notna(), 'weight_decay',
        np.where(df['hyperparam_alpha'].notna(), 'alpha', 'none')
    )

    return df
```

**数据验证**（已完成）：
```
总实验数: 676
  weight_decay 填充: 86行 (12.7%)
  alpha 填充:        31行 (4.6%)
  同时填充:          0行  (0.0%)  ← 完全互斥

值范围：
  weight_decay: [0.0, 9.5e-4]
  alpha:        [5e-6, 2e-5]
```

**统一后效果**：
- 填充率：117/676 (17.3%) ← 从12.7%提升
- 语义：L2正则化强度
- 因果解释："L2正则化提高0.0001 → 能耗下降X%"

### 1.3 排除的超参数

| 超参数 | 排除理由 |
|-------|---------|
| `hyperparam_kfold` | 实验设置，非训练超参数（交叉验证折数） |

### 1.4 包含的其他超参数

**seed（随机种子）**：
- 填充率：34.5% (233/676) ✅ **超过10%阈值**
- 值范围：1-9809
- **作用**：探索随机初始化对能耗和性能的影响
- **因果解释注意**：seed是离散变量，ATE解释需谨慎（seed变化影响随机初始化，进而影响训练动态和能耗）

---

## ✅ 方案2: 能耗中介变量

### 2.1 问题背景

当前分析只使用总量指标（`energy_cpu_avg`, `energy_gpu_avg`），缺失**因果中介变量**：

```
超参数 → ??? → 能耗
           ↑
        缺失的中介机制
```

**需要回答的因果问题**：
1. 超参数如何影响能耗？通过什么路径？
2. GPU利用率是否是中介变量？
3. 温度升高是否导致额外散热能耗？

### 2.2 解决方案

#### 新增变量清单

| 变量名 | 来源列 | 数据类型 | 填充率 | 因果意义 |
|-------|-------|---------|-------|---------|
| `gpu_util_avg` | `energy_gpu_util_avg_percent` | 百分比 | 79.4% | GPU计算密集度（主中介） |
| `gpu_temp_max` | `energy_gpu_temp_max_celsius` | 温度(°C) | 79.4% | 散热压力 |
| `cpu_pkg_ratio` | `energy_cpu_pkg_joules / energy_cpu_total_joules` | 比例 | 计算得出 | CPU计算vs内存能耗比 |
| `gpu_power_fluctuation` | `energy_gpu_max_watts - energy_gpu_min_watts` | 功率(W) | 79.4% | 负载波动性 |
| `gpu_temp_fluctuation` | `energy_gpu_temp_max - energy_gpu_temp_avg` | 温度差(°C) | 79.4% | **温度波动性** ✅ |

#### 重点：温度波动指标

**数据验证**（已完成）：

```
温度波动统计 (temp_max - temp_avg):
  有效样本: 537/676 (79.4%)
  平均温差: 3.84°C
  温差范围: [0.0, 11.1°C]

  温差 > 5°C:  191实验 (35.6%) ← 显著波动
  温差 > 10°C: 2实验   (0.4%)  ← 极端情况

按模型分布:
  VulBERTa/mlp:      5.28°C ← 复杂任务，高波动
  Person_reID/pcb:   5.89°C ← 大模型，高波动
  MNIST系列:         2-4°C  ← 简单任务，低波动
  bug-localization:  3.10°C ← 中等波动
```

**因果意义**：

1. **训练动态性指标**
   - 温差大 → 负载不均（训练过程剧烈波动）
   - 温差小 → 负载稳定（训练过程平稳）

2. **潜在因果路径**
   ```
   超参数 → GPU利用率波动 → 温度波动 → 散热能耗

   例如：
   - learning_rate大 → 参数更新剧烈 → 负载波动 → 温差大
   - batch_size大 → 每步计算稳定 → 负载平稳 → 温差小
   ```

3. **硬件健康影响**
   - 长期高温差 → 散热系统频繁调整 → 额外能耗
   - 温度循环 → 硬件疲劳 → 寿命影响

### 2.3 实现代码

```python
import pandas as pd
import numpy as np

def add_energy_mediators(df):
    """
    添加能耗中介变量

    Args:
        df: 包含原始能耗列的DataFrame

    Returns:
        df: 添加了中介变量的DataFrame
    """

    # 1. GPU利用率（直接复制）
    df['gpu_util_avg'] = df['energy_gpu_util_avg_percent']

    # 2. GPU最高温度（直接复制）
    df['gpu_temp_max'] = df['energy_gpu_temp_max_celsius']

    # 3. CPU package能耗比例
    df['cpu_pkg_ratio'] = df['energy_cpu_pkg_joules'] / (
        df['energy_cpu_total_joules'] + 1e-9  # 避免除零
    )

    # 4. GPU功率波动
    df['gpu_power_fluctuation'] = (
        df['energy_gpu_max_watts'] - df['energy_gpu_min_watts']
    )

    # 5. GPU温度波动 ✅
    df['gpu_temp_fluctuation'] = (
        df['energy_gpu_temp_max_celsius'] - df['energy_gpu_temp_avg_celsius']
    )

    return df
```

### 2.4 因果分析策略

#### 两阶段分析

**阶段1：主效应分析**
```
超参数 → 能耗总量 (energy_cpu_avg, energy_gpu_avg)
超参数 → 性能指标 (待定)
```

**阶段2：中介机制分析**
```
超参数 → 中介变量 (util, temp, fluctuation) → 能耗总量
```

**预期发现**：
- `batch_size` 通过 `gpu_util_avg` 影响 `energy_gpu_avg`
- `learning_rate` 通过 `gpu_temp_fluctuation` 影响散热能耗
- `l2_regularization` 可能不经过中介，直接影响能耗（计算量相同）

---

## ✅ 方案3: 分层因果分析（4个任务组 + One-Hot编码）

### 3.1 问题背景

当前数据中性能指标**高度异构**，不同任务使用完全不同的评价指标：

| 任务类型 | 主要性能指标 | 语义 | 典型值范围 |
|---------|-------------|------|-----------|
| **MNIST分类** | `perf_test_accuracy` | 分类正确率 | 0.06-1.00（波动大） |
| **CIFAR-10分类** | `perf_test_accuracy` | 分类正确率 | 0.91-0.93（稳定） |
| **Person_reID检索** | `perf_map` | 检索平均精度 | 0.60-0.80 |
| **VulBERTa漏洞检测** | `perf_eval_loss` | 评估损失 | 0.2-0.5（越低越好） |
| **Bug定位** | `perf_top1_accuracy` | 排序Top-1准确率 | 0.20-0.40 |

**核心发现**：
- ✅ **MNIST和CIFAR-10可合并**：都使用`perf_test_accuracy`，语义相同
- ⚠️ **但性能分布差异大**：MNIST波动30.7%，CIFAR-10仅0.46%
- ✅ **解决方案**：合并为"图像分类"组 + 添加One-Hot编码区分数据集

### 3.2 解决方案：4个任务组 + One-Hot编码

**核心思路**：
1. 合并MNIST和CIFAR-10为"图像分类"任务组（共用性能指标）
2. 添加One-Hot变量区分不同的数据集/模型（控制异质性）
3. 每个任务组独立进行DiBS因果分析

#### 任务分组定义（最终版本）

| 任务组 | 包含实验 | 有效样本数 | 性能指标 | One-Hot变量 | DiBS可行性 |
|-------|---------|----------|---------|------------|-----------|
| **图像分类** | examples (4模型) + pytorch_resnet_cifar10 | **185个** | perf_test_accuracy | `is_mnist`, `is_cifar10` (2个) | ✅ 充足 |
| **Person_reID检索** | Person_reID_baseline_pytorch (3模型) | 93个 | perf_map | `is_densenet121`, `is_hrnet18`, `is_pcb` (3个) | ✅ 充足 |
| **VulBERTa漏洞检测** | VulBERTa (mlp) | 52个 | perf_eval_loss | 无（单模型） | ✅ 可行 |
| **Bug定位** | bug-localization-by-dnn-and-rvsm (default) | 40个 | perf_top1_accuracy | 无（单模型） | ✅ 可行 |

**说明**：
- **有效样本数** = 同时包含能耗数据和性能数据的实验数
- **DiBS可行性标准**：最低10个，推荐20-50个，理想100+个
- **总计**：4个任务组，370个有效样本，所有组都满足DiBS最低要求
- **改进**：合并后图像分类组样本量从26个提升至185个（提升7倍）

#### 分层分析的优势

1. **保留语义完整性** ✅
   - 各任务组使用其原生性能指标
   - 因果解释清晰："learning_rate → mAP"（Person_reID），"batch_size → test_accuracy"（图像分类）
   - 无需强制归一化或语义丢失

2. **任务特定因果发现** ✅
   - 可发现任务特定的性能-能耗权衡模式
   - 例如：检索任务可能对dropout更敏感，分类任务对learning_rate更敏感
   - 支持针对性的优化建议

3. **样本量显著提升** ✅
   - **关键改进**：合并MNIST+CIFAR-10后，图像分类组从26个样本增至185个
   - 所有任务组样本量都满足DiBS要求
   - 图像分类组达到推荐标准上限（20-50个的上界）

4. **避免跨任务比较陷阱** ✅
   - 不需要回答"哪个任务的性能-能耗权衡最优？"（无意义问题）
   - 专注于各任务内部的因果机制

#### One-Hot编码详解 ⭐⭐⭐

**为什么需要One-Hot编码？**

合并任务组（如MNIST+CIFAR-10）后，需要**控制数据集/模型的异质性**，否则DiBS可能将数据集差异误判为因果关系。

**示例问题**（无One-Hot）：
```
DiBS可能学到：learning_rate → test_accuracy (ATE = 0.15)
但实际原因可能是：
  - MNIST的learning_rate更高（0.01），准确率波动大（82.3%±30.7%）
  - CIFAR-10的learning_rate更低（0.001），准确率稳定（91.8%±0.46%）
  → 混淆了"数据集差异"和"learning_rate因果效应"
```

**One-Hot编码的作用**（控制混淆）：

通过添加`is_mnist`和`is_cifar10`变量，DiBS可以学习到：
```
正确的因果图：
  is_mnist → learning_rate  （MNIST倾向用更高的学习率）
  is_mnist → test_accuracy   （MNIST的准确率基线不同）
  learning_rate → test_accuracy  （控制数据集后，学习率的真实因果效应）
```

**各任务组的One-Hot编码方案**：

##### 1. 图像分类任务组（2个One-Hot变量）

```python
def add_classification_onehot(df):
    """为图像分类任务组添加One-Hot编码"""
    df['is_mnist'] = (df['repository'] == 'examples').astype(int)
    df['is_cifar10'] = (df['repository'] == 'pytorch_resnet_cifar10').astype(int)
    return df
```

**变量含义**：
- `is_mnist = 1`：MNIST数据集（4个模型：mnist, mnist_ff, mnist_rnn, siamese）
- `is_cifar10 = 1`：CIFAR-10数据集（1个模型：resnet20）

**数据分布**：
- MNIST：159个样本（85.9%）
- CIFAR-10：26个样本（14.1%）

##### 2. Person_reID任务组（3个One-Hot变量）

```python
def add_person_reid_onehot(df):
    """为Person_reID任务组添加One-Hot编码"""
    df['is_densenet121'] = (df['model'] == 'densenet121').astype(int)
    df['is_hrnet18'] = (df['model'] == 'hrnet18').astype(int)
    df['is_pcb'] = (df['model'] == 'pcb').astype(int)
    return df
```

**变量含义**：
- `is_densenet121 = 1`：DenseNet121模型
- `is_hrnet18 = 1`：HRNet18模型
- `is_pcb = 1`：PCB模型

**数据分布**：
- densenet121：30个样本（32.3%）
- hrnet18：32个样本（34.4%）
- pcb：31个样本（33.3%）

##### 3. VulBERTa和Bug定位任务组（无One-Hot）

这两个任务组只有1个模型，无需One-Hot编码。

**One-Hot编码的注意事项** ⚠️：

1. **多重共线性**
   - One-Hot编码满足：`is_mnist + is_cifar10 = 1`（对图像分类组）
   - DiBS和DML可以处理多重共线性（使用正则化）
   - 如果出现数值问题，可删除一个变量（如删除`is_cifar10`）

2. **因果解释**
   - One-Hot变量的因果效应表示**数据集/模型的固有差异**
   - 示例：`ATE(is_mnist → test_accuracy) = -9.4%`表示MNIST比CIFAR-10的准确率低9.4%（基线差异）
   - **不可干预**：这些因果边仅用于控制混淆，不能用于优化建议

3. **变量数量权衡**
   - 图像分类：2个One-Hot（简洁，推荐）
   - Person_reID：3个One-Hot（可接受）
   - 如果模型数 > 5，考虑删除样本量少的模型（避免变量爆炸）

#### 分层分析的注意事项

1. **结果不可跨任务比较** ⚠️
   - 因果效应量纲不同："mAP提高0.1" ≠ "accuracy提高0.1"
   - 报告时需明确：每个任务组的结果是独立的

2. **超参数范围差异** ⚠️
   - 不同任务可能使用不同的超参数范围
   - 例如：MNIST的learning_rate可能比Person_reID更大
   - 解决：报告时注明各组的超参数分布

3. **能耗基线差异** ⚠️
   - MNIST（轻量级）vs Person_reID（大模型）的能耗基线差异大
   - 解决：同时报告绝对因果效应（ATE）和相对因果效应（ATE/baseline）

4. **计算成本** ⚠️
   - 需运行5次DiBS+DML分析
   - 预估总时间：约60分钟（可并行）

### 3.3 实现方案

#### 数据预处理（包含One-Hot编码）

为每个任务组生成独立的训练数据文件，并添加相应的One-Hot编码：

```python
import pandas as pd
import numpy as np

def prepare_stratified_data_with_onehot(df_raw):
    """
    为4个任务组准备数据（包含One-Hot编码）

    Args:
        df_raw: 原始数据 (raw_data.csv)

    Returns:
        task_groups: 任务组配置字典
    """

    # 定义任务组
    task_groups = {
        'image_classification': {
            'name': '图像分类',
            'filter': df_raw['repository'].isin(['examples', 'pytorch_resnet_cifar10']),
            'perf_col': 'perf_test_accuracy',
            'onehot_func': lambda df: pd.concat([
                df,
                pd.get_dummies(df['repository'], prefix='is', dtype=int)[['is_examples', 'is_pytorch_resnet_cifar10']]
                .rename(columns={'is_examples': 'is_mnist', 'is_pytorch_resnet_cifar10': 'is_cifar10'})
            ], axis=1)
        },
        'person_reid': {
            'name': 'Person_reID检索',
            'filter': df_raw['repository'] == 'Person_reID_baseline_pytorch',
            'perf_col': 'perf_map',
            'onehot_func': lambda df: pd.concat([
                df,
                pd.get_dummies(df['model'], prefix='is', dtype=int)[['is_densenet121', 'is_hrnet18', 'is_pcb']]
            ], axis=1)
        },
        'vulberta': {
            'name': 'VulBERTa漏洞检测',
            'filter': df_raw['repository'] == 'VulBERTa',
            'perf_col': 'perf_eval_loss',
            'onehot_func': lambda df: df  # 无One-Hot（单模型）
        },
        'bug_localization': {
            'name': 'Bug定位',
            'filter': df_raw['repository'] == 'bug-localization-by-dnn-and-rvsm',
            'perf_col': 'perf_top1_accuracy',
            'onehot_func': lambda df: df  # 无One-Hot（单模型）
        }
    }

    # 为每个任务组生成数据
    for task_name, config in task_groups.items():
        # 1. 筛选任务相关数据
        df_task = df_raw[config['filter']].copy()

        # 2. 只保留有能耗和性能数据的实验
        df_task = df_task[
            df_task['energy_cpu_total_joules'].notna() &
            df_task['energy_gpu_total_joules'].notna() &
            df_task[config['perf_col']].notna()
        ]

        # 3. 应用方案1：超参数统一
        df_task = unify_hyperparameters(df_task)

        # 4. 应用方案2：能耗中介变量
        df_task = add_energy_mediators(df_task)

        # 5. 应用方案3：One-Hot编码
        df_task = config['onehot_func'](df_task)

        # 6. 选择最终变量集（根据填充率动态调整）
        df_task = select_variables_for_task(df_task, task_name, config['perf_col'])

        # 7. 保存
        output_path = f"../data/training_data_{task_name}.csv"
        df_task.to_csv(output_path, index=False)
        print(f"任务组 {config['name']}: {len(df_task)} 个有效样本，{df_task.shape[1]} 个变量")

    return task_groups


def unify_hyperparameters(df):
    """应用方案1：超参数统一"""
    # 统一训练时长
    df['hyperparam_training_duration'] = df['hyperparam_epochs'].fillna(
        df['hyperparam_max_iter']
    )

    # 统一L2正则化
    df['hyperparam_l2_regularization'] = df['hyperparam_weight_decay'].fillna(
        df['hyperparam_alpha']
    )

    return df


def add_energy_mediators(df):
    """应用方案2：能耗中介变量"""
    # GPU利用率
    df['gpu_util_avg'] = df['energy_gpu_util_avg_percent']

    # GPU最高温度
    df['gpu_temp_max'] = df['energy_gpu_temp_max_celsius']

    # CPU package能耗比例
    df['cpu_pkg_ratio'] = df['energy_cpu_pkg_joules'] / (
        df['energy_cpu_total_joules'] + 1e-9
    )

    # GPU功率波动
    df['gpu_power_fluctuation'] = (
        df['energy_gpu_max_watts'] - df['energy_gpu_min_watts']
    )

    # GPU温度波动
    df['gpu_temp_fluctuation'] = (
        df['energy_gpu_temp_max_celsius'] - df['energy_gpu_temp_avg_celsius']
    )

    return df


def select_variables_for_task(df, task_name, perf_col):
    """
    根据填充率动态选择变量

    只保留填充率 > 10% 的超参数（避免DiBS学习稀疏变量）
    """

    # 基础变量（必选）
    base_vars = [
        'energy_cpu_total_joules',
        'energy_gpu_total_joules',
        perf_col
    ]

    # 能耗中介变量（必选，填充率100%）
    mediator_vars = [
        'gpu_util_avg',
        'gpu_temp_max',
        'cpu_pkg_ratio',
        'gpu_power_fluctuation',
        'gpu_temp_fluctuation'
    ]

    # 超参数（根据填充率筛选）
    hyperparam_candidates = [
        'hyperparam_learning_rate',
        'hyperparam_batch_size',
        'hyperparam_dropout',
        'hyperparam_seed',
        'hyperparam_training_duration',
        'hyperparam_l2_regularization'
    ]

    selected_hyperparams = []
    for hp in hyperparam_candidates:
        if hp in df.columns:
            fill_rate = df[hp].notna().sum() / len(df)
            if fill_rate > 0.10:  # 填充率阈值：10%
                selected_hyperparams.append(hp)
                print(f"  - 保留超参数 {hp}: 填充率 {fill_rate:.1%}")
            else:
                print(f"  - 排除超参数 {hp}: 填充率仅 {fill_rate:.1%}")

    # One-Hot变量（如果存在）
    onehot_vars = [col for col in df.columns if col.startswith('is_')]

    # 合并所有变量
    selected_vars = base_vars + mediator_vars + selected_hyperparams + onehot_vars

    # 只保留选中的变量
    df_selected = df[selected_vars].copy()

    return df_selected
```

**关键改进**：

1. **One-Hot编码集成**：每个任务组自动添加相应的One-Hot变量
2. **动态变量选择**：根据填充率（>10%）自动筛选超参数，避免稀疏变量
3. **模块化设计**：超参数统一、中介变量、One-Hot编码各自独立

#### DiBS分层分析

```python
def run_stratified_causal_analysis():
    """运行分层因果分析（4个任务组）"""

    task_groups = ['image_classification', 'person_reid', 'vulberta', 'bug_localization']

    results = {}

    for task_name in task_groups:
        print(f"\n{'='*80}")
        print(f"任务组: {task_name}")
        print(f"{'='*80}\n")

        # 1. 加载数据（已包含One-Hot编码）
        df = pd.read_csv(f"../data/training_data_{task_name}.csv")

        print(f"样本数: {len(df)}")
        print(f"变量数: {df.shape[1]}")
        print(f"变量列表: {df.columns.tolist()}")

        # 2. 运行DiBS因果图学习
        # DiBS会自动学习所有变量之间的因果关系（包括One-Hot变量）
        causal_graph = run_dibs(df, task_name)

        # 3. 运行DML因果推断
        # DML会估计每条因果边的平均因果效应（ATE）
        causal_effects = run_dml(df, causal_graph, task_name)

        # 4. 保存结果
        results[task_name] = {
            'graph': causal_graph,
            'effects': causal_effects,
            'sample_size': len(df),
            'num_vars': df.shape[1]
        }

        # 5. 生成任务特定报告
        generate_task_report(task_name, results[task_name])

    # 6. 生成综合报告（跨任务共性）
    generate_cross_task_summary(results)

    return results
```

### 3.4 报告结构

分层分析的报告将采用以下结构：

```markdown
# 分层因果分析报告（v3.0 - 4个任务组 + One-Hot编码）

## 1. 执行摘要
- 4个任务组的样本量和数据完整性
- One-Hot编码变量统计
- 关键发现概览

## 2. 跨任务共性发现
- 所有任务都发现的因果边（如 learning_rate → 能耗）
- 能耗中介机制的一致性（如 GPU利用率的作用）
- One-Hot变量的因果模式（如 is_mnist → learning_rate）

## 3. 任务特定分析

### 3.1 图像分类任务（MNIST + CIFAR-10）
- 样本量：185个有效实验
- 变量数：15个（包含2个One-Hot）
- 因果图：...
- 关键因果效应：
  - learning_rate → test_accuracy: ATE = X
  - is_mnist → test_accuracy: ATE = Y（数据集基线差异）
  - batch_size → energy_gpu: ATE = Z
- 性能-能耗权衡：...
- One-Hot变量的作用：控制MNIST和CIFAR-10的异质性

### 3.2 Person_reID检索任务
- 样本量：93个有效实验
- 变量数：16个（包含3个One-Hot）
- 因果图：...
- 关键因果效应：
  - dropout → mAP: ATE = X
  - is_densenet121 → mAP: ATE = Y（模型基线差异）
  - learning_rate → energy_gpu: ATE = Z
- 性能-能耗权衡：...
- One-Hot变量的作用：控制3个模型的架构差异

### 3.3 VulBERTa漏洞检测任务
- 样本量：52个有效实验
- 变量数：13个（无One-Hot）
- 因果图：...
- 关键因果效应：...

### 3.4 Bug定位任务
- 样本量：40个有效实验
- 变量数：13个（无One-Hot）
- 因果图：...
- 关键因果效应：...

## 4. 方法论讨论
- 4个任务组 vs 5个任务组的选择理由（合并MNIST+CIFAR-10）
- One-Hot编码在因果推断中的作用（控制混淆）
- 各任务组的统计功效和限制
- 超参数填充率差异的影响

## 5. 结论与建议
- 针对各任务的优化建议
- One-Hot变量的因果解释注意事项（基线差异 vs 可干预因果效应）
- 未来研究方向
```

---

## 📊 变量扩展前后对比

### v1.0 基线（Adult数据集分析）

```
总数: 15个变量

【超参数】(5个)
- learning_rate
- batch_size
- epochs
- dropout
- weight_decay

【能耗】(2个)
- energy_cpu_avg
- energy_gpu_avg

【性能】(2个)
- test_accuracy  ← 实际缺失
- test_loss      ← 实际缺失

【其他】(6个)
- repo, model, mode, method, etc.
```

### v3.0 最终版（能耗数据分层分析 + One-Hot编码）

```
每个任务组: 14-17个变量（根据超参数填充率动态选择）

【超参数】(2-6个，动态选择)
- learning_rate               （填充率 > 10% 时保留）
- batch_size                  （填充率 > 10% 时保留）
- dropout                     （填充率 > 10% 时保留）
- seed                        ✅ 新：随机种子
- hyperparam_training_duration  ✅ 新：epochs + max_iter统一
- hyperparam_l2_regularization   ✅ 新：weight_decay + alpha统一

【能耗总量】(2个)
- energy_cpu_total            （原有，从avg改为total）
- energy_gpu_total            （原有，从avg改为total）

【能耗中介】(5个) ✅ 新增
- gpu_util_avg                ✅ 新：GPU计算密集度
- gpu_temp_max                ✅ 新：散热压力
- cpu_pkg_ratio               ✅ 新：CPU计算/内存能耗比
- gpu_power_fluctuation       ✅ 新：负载波动性
- gpu_temp_fluctuation        ✅ 新：温度波动性

【性能】(1个) - 任务特定
- 图像分类: perf_test_accuracy
- Person_reID: perf_map
- VulBERTa: perf_eval_loss
- Bug定位: perf_top1_accuracy

【One-Hot编码】(0-3个) ✅ 新增
- 图像分类: is_mnist, is_cifar10 (2个)
- Person_reID: is_densenet121, is_hrnet18, is_pcb (3个)
- VulBERTa: 无（单模型）
- Bug定位: 无（单模型）
```

### 改进总结

| 维度 | v1.0 基线 | v3.0 最终版 | 改进 |
|------|---------|-----------|------|
| **任务组** | 1个（Adult） | 4个（图像分类、Person_reID、VulBERTa、Bug定位） | 扩展到真实能耗研究数据 |
| **样本量** | 10个配置 | 370个有效实验 | **37倍提升** ⭐⭐⭐ |
| **超参数** | 5个（存在重复） | 2-6个（统一后+seed，动态选择） | 消除框架差异，统一语义，避免稀疏变量 |
| **能耗变量** | 2个总量指标 | 2个总量 + 5个中介 | 新增因果中介机制分析能力 |
| **性能指标** | 0个（缺失） | 1个（任务特定） | 保留原生语义，支持分层分析 |
| **异质性控制** | 无 | 0-3个One-Hot变量 | ✅ 新增数据集/模型差异控制 |
| **列使用率** | 15/54 (27.8%) | 14-17个关键列 | 聚焦核心因果变量 |
| **变量总数** | 15个 | 14-17个/任务组 | 简洁且全面 |

**关键突破**：

1. **One-Hot编码** ⭐⭐⭐
   - **作用**：控制数据集/模型的异质性，避免DiBS将基线差异误判为因果关系
   - **示例**：`is_mnist → test_accuracy`表示MNIST的准确率基线（不可干预）
   - **优势**：使DiBS能在合并任务组中正确识别真实因果效应

2. **样本量提升**
   - 图像分类组：从26个（仅CIFAR-10）提升至185个（合并MNIST） - **7倍提升**
   - 总样本量：从10个（Adult）提升至370个（能耗数据） - **37倍提升**

3. **动态变量选择**
   - 根据填充率（>10%）自动筛选超参数
   - 避免DiBS学习稀疏变量（降低噪声）

4. **能耗中介机制**
   - 新增5个中介变量（GPU利用率、温度、波动指标）
   - 可回答"超参数如何影响能耗？"（中介路径）

---

## 🔧 实现计划

### 阶段1: 数据预处理脚本 ⏳

**文件**: `analysis/scripts/preprocess_stratified_data.py`

**功能**：
1. 应用方案1（超参数统一）
2. 应用方案2（能耗中介变量）
3. 应用方案3（按任务分组 + One-Hot编码，生成4个数据文件）

**输出**：
```
analysis/data/training_data_image_classification.csv  (185行, ~15列)
analysis/data/training_data_person_reid.csv           (93行, ~16列)
analysis/data/training_data_vulberta.csv              (52行, ~13列)
analysis/data/training_data_bug_localization.csv      (40行, ~13列)
```

**关键代码**（已在方案3.3中提供）

---

### 阶段2: 分层DiBS分析脚本 ⏳

**文件**: `analysis/scripts/run_stratified_causal_discovery.py`

**功能**：
1. 对4个任务组分别运行DiBS因果图学习
2. 对每个任务组运行DML因果推断
3. 生成任务特定报告
4. 生成跨任务共性总结

**预估时间**：
- 图像分类: ~30分钟（185样本，15变量）
- Person_reID: ~15分钟（93样本，16变量）
- VulBERTa: ~8分钟（52样本，13变量）
- Bug定位: ~6分钟（40样本，13变量）
- **总计**: ~60分钟（可GPU并行）

---

### 阶段3: 结果分析和报告生成 ⏳

**输出文件**：
```
analysis/docs/reports/STRATIFIED_CAUSAL_ANALYSIS_REPORT.md         # 综合报告
analysis/docs/reports/IMAGE_CLASSIFICATION_CAUSAL_ANALYSIS.md      # 任务特定报告
analysis/docs/reports/PERSON_REID_CAUSAL_ANALYSIS.md
analysis/docs/reports/VULBERTA_CAUSAL_ANALYSIS.md
analysis/docs/reports/BUG_LOCALIZATION_CAUSAL_ANALYSIS.md
```

**报告内容**（参见方案3.4）

---

### 阶段4: 方法验证和对比 ⏳

**对比实验**：
1. v1.0（Adult数据集，15变量）vs v3.0（能耗数据，4组13-16变量 + One-Hot）
2. 4个任务组 vs 5个任务组（评估合并MNIST+CIFAR-10的影响）
3. 有One-Hot vs 无One-Hot（评估异质性控制的作用）

**评估指标**：
- 因果边检测数量
- 统计显著因果边数量
- 因果效应的置信区间宽度
- One-Hot变量的因果模式（是否正确控制了基线差异）
- 计算时间和收敛性

---

## 📝 下一步行动

### ✅ 已完成

1. ✅ **数据可行性评估** - 5个任务组样本量充足
2. ✅ **方案1确认** - 超参数统一（training_duration, l2_regularization）
3. ✅ **方案2确认** - 能耗中介变量（5个新增变量）
4. ✅ **方案3确认** - 分层因果分析（按任务类型分组）
5. ✅ **文档化决策** - 本文档v2.0

### ✅ 已完成（数据预处理阶段0-5）⭐ **[2025-12-23更新]**

6. **阶段0: 数据验证**（stage0_data_validation.py）
   - ✅ 验证726行数据完整性
   - ✅ 生成验证报告
   - 输出：stage0_validated.csv (726行, 56列)

7. **阶段1: 超参数统一**（stage1_unify_hyperparameters.py）
   - ✅ 统一training_duration (epochs + max_iter)
   - ✅ 统一l2_regularization (weight_decay + alpha)
   - ✅ 新增seed变量
   - 输出：stage1_unified.csv (726行, 58列)

8. **阶段2: 能耗中介变量**（stage2_add_mediators.py）
   - ✅ 新增5个能耗中介变量（gpu_util_avg, gpu_temp_max, cpu_pkg_ratio, gpu_power_fluctuation, gpu_temp_fluctuation）
   - 输出：stage2_mediators.csv (726行, 63列)

9. **阶段3: 任务分组**（stage3_task_grouping.py）
   - ✅ 按repository分组为4个任务组
   - ✅ 排除MRT-OAST（性能指标不同）
   - 输出：4个任务组CSV (648行总计)

10. **阶段4: One-Hot编码**（stage4_onehot_encoding.py）
    - ✅ 图像分类：is_mnist, is_cifar10 (2个)
    - ✅ Person_reID：is_densenet121, is_hrnet18, is_pcb (3个)
    - ✅ VulBERTa和Bug定位：无需编码（单一模型）
    - 输出：4个编码后CSV

11. **阶段5: 变量选择**（stage5_variable_selection.py）
    - ✅ 动态选择超参数（填充率>10%）
    - ✅ 保留核心能耗和性能变量
    - 输出：4个最终分析文件 (14-20变量/任务组)

**数据质量报告**：
- ✅ [DATA_QUALITY_REPORT_DETAILED_20251223.md](../DATA_QUALITY_REPORT_DETAILED_20251223.md) - 完整数据质量分析

### ⏳ 待实施（归一化和DiBS分析）

12. **阶段6: 归一化**（stage6_normalization.py）⏳ **当前任务**
    - 优先级：**高**
    - 功能：StandardScaler标准化数值变量
    - 预估时间：30分钟

13. **阶段7: 最终验证**（stage7_final_validation.py）⏳ **当前任务**
    - 优先级：**高**
    - 功能：验证训练数据质量，生成DiBS就绪文件
    - 预估时间：30分钟

14. **运行分层DiBS分析**
    - 优先级：**高**
    - 预估时间：60分钟（运行时间）+ 1-2小时（脚本编写）
    - 依赖：步骤12-13完成

15. **生成分析报告**
    - 优先级：**中**
    - 预估时间：每个任务组1-2小时（分析+撰写）
    - 依赖：步骤14完成

16. **对比v1.0和v3.0**
    - 优先级：**中**
    - 预估时间：2-3小时
    - 依赖：步骤15完成

### 🔄 可选扩展

10. **统一能耗分析**（不含性能，537样本全集）
    - 目的：发现跨任务通用的能耗因果模式
    - 优先级：**低**
    - 时间：与分层分析并行

11. **敏感性分析**
    - 测试不同DiBS超参数的影响
    - 测试缺失数据插补方法
    - 优先级：**低**

---

## 📚 参考文档

- [COLUMN_USAGE_ANALYSIS.md](./COLUMN_USAGE_ANALYSIS.md) - 原始列使用率分析（引发本方案）
- [ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](./ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md) - DiBS基线分析（v1.0）
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - 数据迁移指南

---

## 📌 版本历史

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2025-12-22 | 初始版本：方案1-2确认，方案3待定 | Green + Claude |
| v2.0 | 2025-12-22 | **方案3确认**：采用分层因果分析（按任务类型分组）<br>- 完成5个任务组的样本量评估<br>- 确认所有任务组满足DiBS要求<br>- 更新实现计划和报告结构<br>- **状态变更**: ⚠️ 待定 → ✅ 全部方案已确认 | Green + Claude |
| v3.0 | 2025-12-22 | **最终版本**：优化为4个任务组 + One-Hot编码<br>- ✅ **合并MNIST+CIFAR-10**：图像分类组样本量从26→185（7倍提升）<br>- ✅ **新增One-Hot编码**：控制数据集/模型异质性（2-3个变量）<br>- ✅ **动态变量选择**：根据填充率（>10%）自动筛选超参数<br>- ✅ **完整实现代码**：包含One-Hot编码的预处理脚本<br>- ✅ **更新报告结构**：反映4组方案和One-Hot变量的因果解释<br>- **关键突破**：One-Hot编码避免DiBS混淆基线差异和因果效应<br>- **状态变更**: ✅ 全部方案已确认 → ✅ 全部方案已确认（最终版本） | Green + Claude |

---

**维护者**: Green
**文档状态**: ✅ 全部方案已确认（最终版本 v3.0）
**下次更新**: 完成预处理脚本后（添加实际代码示例和数据验证结果）
