# 能耗数据因果分析 - 数据处理流程方案

**版本**: v1.0
**日期**: 2025-12-22
**状态**: 📋 设计完成，待实施

---

## 📋 执行摘要

本文档描述了将主项目的能耗数据（`raw_data.csv`）转换为DiBS因果分析所需格式的**完整8阶段流程**。

### 核心目标

- 从676个实验中提取370个有效样本
- 分为4个任务组进行分层因果分析
- 每个任务组包含13-16个因果变量
- 生成标准化、无缺失值、可直接用于DiBS的训练数据

### 方案特点

✅ **分阶段设计**：8个独立阶段，每阶段可单独执行和验证
✅ **可追溯性**：每阶段都保存中间结果和统计报告
✅ **容错性**：失败时保存检查点，支持断点续传
✅ **验证完备**：每阶段都有明确的验证点和质量检查

---

## 🔄 整体流程图

```
原始数据 (raw_data.csv, 676行×87列)
    ↓
┌─────────────────────────────────────────────┐
│ 阶段0: 数据验证                             │
│ - 检查数据完整性                            │
│ - 统计任务组样本量                          │
│ 输出: stage0_validation_report.txt         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 阶段1: 超参数统一                           │
│ - training_duration = epochs + max_iter    │
│ - l2_regularization = weight_decay + alpha │
│ 输出: stage1_unified_data.csv (676×89)     │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 阶段2: 能耗中介变量                         │
│ - gpu_util_avg, gpu_temp_max               │
│ - cpu_pkg_ratio                            │
│ - gpu_power_fluctuation, gpu_temp_fluctuation│
│ 输出: stage2_with_mediators.csv (676×94)   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 阶段3: 任务分组                             │
│ - 4个任务组 (图像分类、Person_reID、       │
│   VulBERTa、Bug定位)                        │
│ 输出: stage3_*.csv (4个文件)               │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 阶段4: One-Hot编码                          │
│ - 图像分类: +2列 (is_mnist, is_cifar10)    │
│ - Person_reID: +3列 (模型One-Hot)          │
│ 输出: stage4_*.csv (4个文件)               │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 阶段5: 变量选择                             │
│ - 根据填充率>10%筛选超参数                 │
│ - 只保留13-16个核心变量                    │
│ 输出: stage5_*.csv (4个文件)               │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 阶段6: 归一化                               │
│ - StandardScaler标准化 (均值0, 方差1)      │
│ - 删除含NaN的行                            │
│ 输出: stage6_*.csv + scaler.pkl            │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 阶段7: 最终验证                             │
│ - 检查样本量≥10                            │
│ - 验证无NaN，均值~0，标准差~1              │
│ 输出: training_data_*.csv (4个) ✅         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 阶段8: DiBS + DML因果分析                   │
│ - DiBS学习因果图 (~60分钟)                 │
│ - DML估计因果效应                          │
│ 输出: 因果图 + 因果效应 + 报告             │
└─────────────────────────────────────────────┘
    ↓
最终成果
```

---

## 📊 各阶段详细说明

### 阶段0：数据验证 ✅

**目的**：确保raw_data.csv数据质量

**输入**：
- `../../data/raw_data.csv` (676行×87列)

**处理**：
1. 检查总行数、列数
2. 统计有效样本（能耗+性能同时存在）
3. 检查关键列存在性
4. 统计每个任务组的样本量

**输出**：
- `data/stage0_validation_report.txt`

**验证点**：
```
✓ 总行数: 676
✓ 有效样本数: 616 (91.1%)
✓ 任务组样本量:
  - examples: 159个
  - pytorch_resnet_cifar10: 26个
  - Person_reID_baseline_pytorch: 93个
  - VulBERTa: 52个
  - bug-localization: 40个
✓ 关键列完整性检查通过
```

**脚本**：`scripts/stage0_validate_raw_data.py`

---

### 阶段1：超参数统一 🔄

**目的**：合并语义相同的超参数

**输入**：
- `../../data/raw_data.csv`

**处理**：
```python
# 统一训练时长
df['hyperparam_training_duration'] = df['hyperparam_epochs'].fillna(
    df['hyperparam_max_iter']
)

# 统一L2正则化
df['hyperparam_l2_regularization'] = df['hyperparam_weight_decay'].fillna(
    df['hyperparam_alpha']
)
```

**输出**：
- `data/stage1_unified_data.csv` (676×89, +2列)
- `data/stage1_unification_stats.txt`

**验证点**：
```
✓ 新增列: hyperparam_training_duration, hyperparam_l2_regularization
✓ training_duration填充率: 96.1%
  - 来自epochs: 90.2%
  - 来自max_iter: 5.9%
✓ l2_regularization填充率: 17.3%
✓ 互斥性检验: 100%通过
```

**脚本**：`scripts/stage1_unify_hyperparameters.py`

---

### 阶段2：能耗中介变量 ⚡

**目的**：添加5个能耗中介变量

**输入**：
- `data/stage1_unified_data.csv`

**处理**：
```python
# 1. GPU利用率
df['gpu_util_avg'] = df['energy_gpu_util_avg_percent']

# 2. GPU最高温度
df['gpu_temp_max'] = df['energy_gpu_temp_max_celsius']

# 3. CPU Package能耗比例
df['cpu_pkg_ratio'] = df['energy_cpu_pkg_joules'] / (
    df['energy_cpu_total_joules'] + 1e-9
)

# 4. GPU功率波动
df['gpu_power_fluctuation'] = (
    df['energy_gpu_max_watts'] - df['energy_gpu_min_watts']
)

# 5. GPU温度波动
df['gpu_temp_fluctuation'] = (
    df['energy_gpu_temp_max_celsius'] - df['energy_gpu_temp_avg_celsius']
)
```

**输出**：
- `data/stage2_with_mediators.csv` (676×94, +5列)
- `data/stage2_mediator_stats.txt`

**验证点**：
```
✓ 新增5列中介变量
✓ 填充率统计: 79.4%-91.1%
✓ 数据范围检查:
  - gpu_util_avg: [0, 100] %
  - gpu_temp_max: [30, 85] °C
  - cpu_pkg_ratio: [0, 1]
  - gpu_temp_fluctuation: [0, 11.1] °C ✓ 合理
```

**脚本**：`scripts/stage2_add_mediators.py`

---

### 阶段3：任务分组 📂

**目的**：将数据分为4个任务组

**输入**：
- `data/stage2_with_mediators.csv`

**处理**：
```python
for task_name, config in TASK_GROUPS.items():
    # 筛选任务相关数据
    df_task = df[df['repository'].isin(config['repositories'])].copy()

    # 只保留有效样本（能耗+性能同时存在）
    df_task = df_task[
        df_task['energy_cpu_total_joules'].notna() &
        df_task['energy_gpu_total_joules'].notna() &
        df_task[config['perf_col']].notna()
    ]

    # 保存
    df_task.to_csv(f'data/stage3_{task_name}.csv', index=False)
```

**输出**：
- `data/stage3_image_classification.csv` (185×94)
- `data/stage3_person_reid.csv` (93×94)
- `data/stage3_vulberta.csv` (52×94)
- `data/stage3_bug_localization.csv` (40×94)
- `data/stage3_task_split_stats.txt`

**验证点**：
```
✓ 任务组分割完成:
  - image_classification: 185行 (50.0%)
  - person_reid: 93行 (25.1%)
  - vulberta: 52行 (14.1%)
  - bug_localization: 40行 (10.8%)
✓ 总有效样本: 370行
✓ 每个任务组都满足DiBS最低要求 (≥10)
```

**脚本**：`scripts/stage3_split_task_groups.py`

---

### 阶段4：One-Hot编码 🏷️

**目的**：为合并的任务组添加One-Hot变量

**输入**：
- `data/stage3_*.csv` (4个文件)

**处理**：
```python
# 图像分类：添加is_mnist, is_cifar10
df_image['is_mnist'] = (df_image['repository'] == 'examples').astype(int)
df_image['is_cifar10'] = (df_image['repository'] == 'pytorch_resnet_cifar10').astype(int)

# Person_reID：添加is_densenet121, is_hrnet18, is_pcb
df_reid['is_densenet121'] = (df_reid['model'] == 'densenet121').astype(int)
df_reid['is_hrnet18'] = (df_reid['model'] == 'hrnet18').astype(int)
df_reid['is_pcb'] = (df_reid['model'] == 'pcb').astype(int)

# VulBERTa和Bug定位：无需One-Hot
```

**输出**：
- `data/stage4_image_classification.csv` (185×96, +2列)
- `data/stage4_person_reid.csv` (93×97, +3列)
- `data/stage4_vulberta.csv` (52×94, 无变化)
- `data/stage4_bug_localization.csv` (40×94, 无变化)
- `data/stage4_onehot_stats.txt`

**验证点**：
```
✓ One-Hot编码添加完成:
  - image_classification: +2列
    - is_mnist: 159 (85.9%)
    - is_cifar10: 26 (14.1%)
    - 互斥性: 100% ✓
  - person_reid: +3列
    - densenet121: 30 (32.3%)
    - hrnet18: 32 (34.4%)
    - pcb: 31 (33.3%)
    - 互斥性: 100% ✓
```

**脚本**：`scripts/stage4_add_onehot.py`

---

### 阶段5：变量选择 🎯

**目的**：根据填充率动态选择超参数

**输入**：
- `data/stage4_*.csv` (4个文件)

**处理**：
```python
# 基础变量（必选）
base_vars = ['energy_cpu_total_joules', 'energy_gpu_total_joules', perf_col]

# 能耗中介（必选）
mediator_vars = ['gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
                 'gpu_power_fluctuation', 'gpu_temp_fluctuation']

# 超参数（填充率>10%筛选）
selected_hyperparams = [hp for hp in candidates
                       if df[hp].notna().sum() / len(df) > 0.10]

# One-Hot（如果存在）
onehot_vars = [col for col in df.columns if col.startswith('is_')]

# 合并
final_vars = base_vars + mediator_vars + selected_hyperparams + onehot_vars
```

**输出**：
- `data/stage5_image_classification.csv` (185×15)
- `data/stage5_person_reid.csv` (93×16)
- `data/stage5_vulberta.csv` (52×13)
- `data/stage5_bug_localization.csv` (40×13)
- `data/stage5_variable_selection_report.txt`

**验证点**：
```
✓ 变量选择完成:
  图像分类 (15变量):
    ✓ 超参数 (3个): learning_rate, batch_size, training_duration
    ✓ 能耗 (7个): 2总量 + 5中介
    ✓ 性能 (1个): perf_test_accuracy
    ✓ One-Hot (2个): is_mnist, is_cifar10
    ✗ 排除: dropout (7%), l2_regularization (9%), seed (填充率不足或任务特定)

  Person_reID (16变量):
    ✓ 超参数 (3个): learning_rate, dropout, training_duration
    ✓ One-Hot (3个): 3个模型
    ✗ 排除: batch_size (0%), l2_regularization (0%), seed (填充率不足或任务特定)
```

**脚本**：`scripts/stage5_select_variables.py`

---

### 阶段6：归一化 📐

**目的**：使用StandardScaler标准化数据

**输入**：
- `data/stage5_*.csv` (4个文件)

**处理**：
```python
from sklearn.preprocessing import StandardScaler

# 1. 处理缺失值（删除含NaN的行）
df_clean = df.dropna()

# 2. 标准化
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df_clean)

# 3. 保存数据和scaler
df_normalized.to_csv(f'data/stage6_{task_name}.csv', index=False)

import pickle
with open(f'data/stage6_{task_name}_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

**输出**：
- `data/stage6_*.csv` (4个文件，标准化后)
- `data/stage6_*_scaler.pkl` (4个scaler对象)
- `data/stage6_normalization_stats.txt`

**验证点**：
```
✓ 归一化完成:
  图像分类:
    ✓ 缺失值处理: 185行 → 185行 (100.0%)
    ✓ 均值: ~0.00 ✓
    ✓ 标准差: ~1.00 ✓

  Person_reID:
    ✓ 缺失值处理: 93行 → 85行 (91.4%)
    ⚠️ 损失8行（dropout列有缺失）

  VulBERTa:
    ✓ 100.0%保留

  Bug定位:
    ✓ 100.0%保留
```

**脚本**：`scripts/stage6_normalize.py`

---

### 阶段7：最终验证 ✔️

**目的**：验证训练数据质量

**输入**：
- `data/stage6_*.csv` (4个文件)

**处理**：
```python
# 1. 检查样本量≥10
assert len(df) >= 10

# 2. 检查列数
assert len(df.columns) == expected_cols

# 3. 检查无NaN
assert df.isna().sum().sum() == 0

# 4. 检查标准化
assert all(abs(df.mean()) < 1e-10)
assert all(abs(df.std() - 1) < 1e-10)

# 5. 复制为training_data
df.to_csv(f'data/training_data_{task_name}.csv', index=False)
```

**输出**：
- `data/training_data_image_classification.csv` (185×15) ✅
- `data/training_data_person_reid.csv` (85×16) ✅
- `data/training_data_vulberta.csv` (52×13) ✅
- `data/training_data_bug_localization.csv` (40×13) ✅
- `data/stage7_final_validation_report.txt`

**验证点**：
```
================================================================================
最终验证报告
================================================================================

✅ 所有任务组通过验证

【图像分类】
  ✓ 样本量: 185 (充足，推荐范围)
  ✓ 变量数: 15
  ✓ 无缺失值: 100%
  ✓ 标准化检查: 通过
  → data/training_data_image_classification.csv

【Person_reID检索】
  ✓ 样本量: 85 (充足)
  ✓ 变量数: 16
  ✓ 无缺失值: 100%
  ✓ 标准化检查: 通过

【VulBERTa漏洞检测】
  ✓ 样本量: 52 (充足)
  ✓ 变量数: 13
  ✓ 无缺失值: 100%

【Bug定位】
  ✓ 样本量: 40 (可用)
  ⚠️ 样本量较少，统计功效可能不足
  ✓ 变量数: 13
  ✓ 无缺失值: 100%

================================================================================
准备就绪，可以开始DiBS因果图学习
================================================================================
```

**脚本**：`scripts/stage7_final_validation.py`

---

### 阶段8：DiBS + DML因果分析 🔬

**目的**：学习因果图并估计因果效应

**输入**：
- `data/training_data_*.csv` (4个文件)

**处理**：
```python
# 1. DiBS因果图学习
from utils.causal_discovery import CausalGraphLearner

learner = CausalGraphLearner(
    n_vars=len(df.columns),
    n_steps=3000,
    alpha=0.1,
    random_seed=42
)

causal_graph = learner.fit(df, verbose=True)
learner.save_graph(f'results/{task_name}_causal_graph.npy')

# 2. DML因果推断
from utils.causal_inference import CausalInferenceEngine

engine = CausalInferenceEngine(verbose=True)
causal_effects = engine.analyze_all_edges(
    data=df,
    causal_graph=causal_graph,
    var_names=df.columns.tolist(),
    threshold=0.3
)

engine.save_results(f'results/{task_name}_causal_effects.csv')
```

**输出**：
- `results/image_classification_causal_graph.npy`
- `results/image_classification_causal_effects.csv`
- `results/image_classification_report.md`
- (Person_reID、VulBERTa、Bug定位各一套)
- `results/cross_task_summary.md` (综合报告)

**预估时间**：
- 图像分类: ~30分钟 (185样本, 15变量)
- Person_reID: ~15分钟 (85样本, 16变量)
- VulBERTa: ~8分钟 (52样本, 13变量)
- Bug定位: ~6分钟 (40样本, 13变量)
- **总计**: ~60分钟

**脚本**：`scripts/stage8_causal_analysis.py`

---

## 📁 脚本组织结构

```
analysis/
├── scripts/
│   ├── stage0_validate_raw_data.py
│   ├── stage1_unify_hyperparameters.py
│   ├── stage2_add_mediators.py
│   ├── stage3_split_task_groups.py
│   ├── stage4_add_onehot.py
│   ├── stage5_select_variables.py
│   ├── stage6_normalize.py
│   ├── stage7_final_validation.py
│   ├── stage8_causal_analysis.py
│   │
│   ├── run_all_stages.sh          # 🚀 一键运行所有阶段
│   └── run_stage.py                # 🎯 单独运行指定阶段
│
├── data/                            # 数据目录
│   ├── stage0_validation_report.txt
│   ├── stage1_unified_data.csv
│   ├── stage2_with_mediators.csv
│   ├── stage3_*.csv (4个)
│   ├── stage4_*.csv (4个)
│   ├── stage5_*.csv (4个)
│   ├── stage6_*.csv (4个)
│   ├── stage6_*_scaler.pkl (4个)
│   ├── training_data_*.csv (4个) ✅ 最终输入
│   └── *.txt (各阶段统计报告)
│
├── results/                         # 结果目录
│   ├── *_causal_graph.npy (4个)
│   ├── *_causal_effects.csv (4个)
│   ├── *_report.md (4个)
│   └── cross_task_summary.md
│
├── config_energy.py                 # 配置文件
└── docs/reports/
    └── DATA_PROCESSING_PIPELINE.md  # 本文档
```

---

## 🎮 执行方式

### 方式1：一键运行所有阶段

```bash
cd /home/green/energy_dl/nightly/analysis
conda activate fairness
bash scripts/run_all_stages.sh
```

### 方式2：逐阶段执行（推荐用于调试）

```bash
conda activate fairness

# 阶段0：数据验证
python scripts/stage0_validate_raw_data.py
# 检查: data/stage0_validation_report.txt

# 阶段1：超参数统一
python scripts/stage1_unify_hyperparameters.py
# 检查: data/stage1_unified_data.csv

# 阶段2：能耗中介变量
python scripts/stage2_add_mediators.py
# 检查: data/stage2_with_mediators.csv

# ... 依此类推
```

### 方式3：从指定阶段开始

```bash
# 从阶段5开始（假设0-4已完成）
python scripts/run_stage.py --start 5
```

---

## ⚠️ 关键注意事项

### 1. 数据质量检查

每个阶段都应该检查：
- ✅ 文件是否生成
- ✅ 行数/列数是否符合预期
- ✅ 统计报告是否有异常值或警告
- ✅ 数据质量（缺失值、重复值、异常值）

### 2. 中间结果保存

- 所有阶段的中间结果都保存在 `data/` 目录
- 失败时可从上一阶段的输出重新开始
- 避免重复运行耗时的早期阶段

### 3. 配置文件

- 所有参数都在 `config_energy.py` 中定义
- 修改配置后需重新运行相关阶段
- 保持配置文件和实际执行的一致性

### 4. 计算资源

- 阶段0-7：主要是数据处理，CPU即可，几分钟内完成
- 阶段8：DiBS + DML分析，推荐GPU，约60分钟
- 确保有足够的磁盘空间（约500MB）

---

## 📊 预期最终成果

### 训练数据 (4个文件)

| 任务组 | 文件名 | 样本量 | 变量数 | DiBS可行性 |
|--------|--------|--------|--------|-----------|
| 图像分类 | training_data_image_classification.csv | 185 | 15 | ✅ 充足 |
| Person_reID | training_data_person_reid.csv | 85 | 16 | ✅ 充足 |
| VulBERTa | training_data_vulberta.csv | 52 | 13 | ✅ 可行 |
| Bug定位 | training_data_bug_localization.csv | 40 | 13 | ⚠️ 较少 |

### 因果分析结果 (4组)

每个任务组包含：
1. **因果图** (`*_causal_graph.npy`) - DiBS学习的邻接矩阵
2. **因果效应** (`*_causal_effects.csv`) - DML估计的ATE、置信区间、p值
3. **分析报告** (`*_report.md`) - 关键发现和可视化

### 综合报告

- `cross_task_summary.md` - 跨任务共性发现和比较

---

## 🔗 相关文档

- [VARIABLE_EXPANSION_PLAN.md](VARIABLE_EXPANSION_PLAN.md) - 变量扩展方案详解 v3.0
- [DATA_PROCESSING_SUMMARY.md](DATA_PROCESSING_SUMMARY.md) - 数据处理方案总结
- [config_energy.py](../../config_energy.py) - 配置参数文件
- [ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md) - Adult实验参考

---

**维护者**: Green + Claude
**最后更新**: 2025-12-22
**下一步**: 实施阶段0-7的预处理脚本
