# 能耗因果分析 - 缺失值详细分析报告

**日期**: 2025-12-24
**分析对象**: 4个任务组训练数据文件
**目的**: 诊断缺失值问题，为修复提供依据

---

## 执行摘要

### 总体缺失率

| 任务组 | 总样本 | 总特征 | 总体缺失率 | 完全无缺失行 | 有缺失列数 | 状态 |
|--------|--------|--------|------------|--------------|------------|------|
| **图像分类** | 258 | 13 | **8.83%** | 125 (48.4%) | 10/13 | ⚠️ 中等 |
| **Person_reID** | 116 | 16 | **4.96%** | 75 (64.7%) | 3/16 | ⚠️ 可用 |
| **VulBERTa** | 142 | 10 | **28.87%** | **0 (0%)** | 10/10 | ❌ 严重 |
| **Bug定位** | 132 | 11 | **24.38%** | **0 (0%)** | 11/11 | ❌ 严重 |

**关键发现**:
- ❌ VulBERTa和Bug定位**没有任何一行完全无缺失**
- ⚠️ 图像分类和Person_reID有近一半/三分之一数据含缺失
- 🔴 **所有任务组的超参数列都有严重缺失**（25-100%）

---

## 1. 图像分类 (image_classification)

### 基本信息
- 文件: `data/energy_research/training/training_data_image_classification.csv`
- 样本数: 258行（MNIST: 219, CIFAR-10: 39）
- 特征数: 13列
- 总体缺失率: **8.83%** (296/3354单元格)
- 完全无缺失行: **125/258 (48.4%)**

### 缺失值详情

| 列名 | 缺失数量 | 缺失率 | 填充率 | 状态 |
|------|----------|--------|--------|------|
| **hyperparam_batch_size** | **122** | **47.29%** | 52.71% | ❌ |
| **hyperparam_learning_rate** | **84** | **32.56%** | 67.44% | ❌ |
| **training_duration** | **83** | **32.17%** | 67.83% | ❌ |
| gpu_util_avg | 1 | 0.39% | 99.61% | ⚠️ |
| gpu_temp_max | 1 | 0.39% | 99.61% | ⚠️ |
| cpu_pkg_ratio | 1 | 0.39% | 99.61% | ⚠️ |
| gpu_power_fluctuation | 1 | 0.39% | 99.61% | ⚠️ |
| gpu_temp_fluctuation | 1 | 0.39% | 99.61% | ⚠️ |
| energy_cpu_total_joules | 1 | 0.39% | 99.61% | ⚠️ |
| energy_gpu_total_joules | 1 | 0.39% | 99.61% | ⚠️ |
| perf_test_accuracy | 0 | 0.00% | 100.00% | ✅ |
| is_mnist | 0 | 0.00% | 100.00% | ✅ |
| is_cifar10 | 0 | 0.00% | 100.00% | ✅ |

### 缺失值模式

**前5种缺失组合**:
1. **无缺失值**: 125行 (48.4%) ✅
2. **3列缺失** (lr + bs + dur): 40行 (15.5%)
3. **2列缺失** (lr + bs): 34行 (13.2%)
4. **2列缺失** (bs + dur): 33行 (12.8%)
5. **1列缺失** (bs): 15行 (5.8%)

**关键发现**:
- 含有缺失值的行: **133/258 (51.6%)**
- 平均每行缺失: **1.15个单元格**
- 最大单行缺失: **7个单元格**

### 数据集差异分析

**CIFAR-10模型** (n=39):
| 超参数 | 缺失数量 | 缺失率 |
|--------|----------|--------|
| hyperparam_batch_size | 39 | **100.0%** ❌ |
| hyperparam_learning_rate | 18 | 46.2% |
| training_duration | 18 | 46.2% |

**MNIST模型** (n=219):
| 超参数 | 缺失数量 | 缺失率 |
|--------|----------|--------|
| hyperparam_batch_size | 83 | 37.9% |
| hyperparam_learning_rate | 66 | 30.1% |
| training_duration | 65 | 29.7% |

**重要发现**:
- 🔴 **CIFAR-10的batch_size全部缺失**！这说明CIFAR-10模型使用了不同的超参数命名
- CIFAR-10的lr和dur缺失率也高于MNIST
- 这是**系统性缺失**，不是随机缺失

---

## 2. Person_reID (person_reid)

### 基本信息
- 文件: `data/energy_research/training/training_data_person_reid.csv`
- 样本数: 116行（densenet121: 43, hrnet18: 37, pcb: 36）
- 特征数: 16列
- 总体缺失率: **4.96%** (92/1856单元格)
- 完全无缺失行: **75/116 (64.7%)**

### 缺失值详情

| 列名 | 缺失数量 | 缺失率 | 填充率 | 状态 |
|------|----------|--------|--------|------|
| **hyperparam_learning_rate** | **31** | **26.72%** | 73.28% | ❌ |
| **hyperparam_dropout** | **31** | **26.72%** | 73.28% | ❌ |
| **training_duration** | **30** | **25.86%** | 74.14% | ❌ |
| gpu_util_avg | 0 | 0.00% | 100.00% | ✅ |
| gpu_temp_max | 0 | 0.00% | 100.00% | ✅ |
| cpu_pkg_ratio | 0 | 0.00% | 100.00% | ✅ |
| gpu_power_fluctuation | 0 | 0.00% | 100.00% | ✅ |
| gpu_temp_fluctuation | 0 | 0.00% | 100.00% | ✅ |
| energy_cpu_total_joules | 0 | 0.00% | 100.00% | ✅ |
| energy_gpu_total_joules | 0 | 0.00% | 100.00% | ✅ |
| perf_map | 0 | 0.00% | 100.00% | ✅ |
| perf_rank1 | 0 | 0.00% | 100.00% | ✅ |
| perf_rank5 | 0 | 0.00% | 100.00% | ✅ |
| is_densenet121 | 0 | 0.00% | 100.00% | ✅ |
| is_hrnet18 | 0 | 0.00% | 100.00% | ✅ |
| is_pcb | 0 | 0.00% | 100.00% | ✅ |

### 缺失值模式

**前5种缺失组合**:
1. **无缺失值**: 75行 (64.7%) ✅
2. **2列缺失** (lr + dropout): 11行 (9.5%)
3. **2列缺失** (dropout + dur): 10行 (8.6%)
4. **2列缺失** (lr + dur): 10行 (8.6%)
5. **3列缺失** (lr + dropout + dur): 10行 (8.6%)

**关键发现**:
- 含有缺失值的行: **41/116 (35.3%)**
- 平均每行缺失: **0.79个单元格**
- 最大单行缺失: **3个单元格**

### 模型差异分析

**densenet121** (n=43):
| 超参数 | 缺失数量 | 缺失率 |
|--------|----------|--------|
| hyperparam_learning_rate | 18 | **41.9%** ❌ |
| hyperparam_dropout | 18 | **41.9%** ❌ |
| training_duration | 18 | **41.9%** ❌ |

**hrnet18** (n=37):
| 超参数 | 缺失数量 | 缺失率 |
|--------|----------|--------|
| hyperparam_learning_rate | 7 | 18.9% |
| hyperparam_dropout | 7 | 18.9% |
| training_duration | 6 | 16.2% |

**pcb** (n=36):
| 超参数 | 缺失数量 | 缺失率 |
|--------|----------|--------|
| hyperparam_learning_rate | 6 | 16.7% |
| hyperparam_dropout | 6 | 16.7% |
| training_duration | 6 | 16.7% |

**重要发现**:
- **densenet121的缺失率是其他模型的2.5倍**
- 这也是**系统性缺失**，暗示不同模型使用了不同的配置结构

---

## 3. VulBERTa (vulberta)

### 基本信息
- 文件: `data/energy_research/training/training_data_vulberta.csv`
- 样本数: 142行
- 特征数: 10列
- 总体缺失率: **28.87%** (410/1420单元格) ❌
- 完全无缺失行: **0/142 (0%)** ❌❌❌

### 缺失值详情

| 列名 | 缺失数量 | 缺失率 | 填充率 | 状态 |
|------|----------|--------|--------|------|
| **hyperparam_learning_rate** | **92** | **64.79%** | 35.21% | ❌ |
| **training_duration** | **90** | **63.38%** | 36.62% | ❌ |
| **perf_eval_loss** | **60** | **42.25%** | 57.75% | ❌ |
| gpu_util_avg | 24 | 16.90% | 83.10% | ⚠️ |
| gpu_temp_max | 24 | 16.90% | 83.10% | ⚠️ |
| cpu_pkg_ratio | 24 | 16.90% | 83.10% | ⚠️ |
| gpu_power_fluctuation | 24 | 16.90% | 83.10% | ⚠️ |
| gpu_temp_fluctuation | 24 | 16.90% | 83.10% | ⚠️ |
| energy_cpu_total_joules | 24 | 16.90% | 83.10% | ⚠️ |
| energy_gpu_total_joules | 24 | 16.90% | 83.10% | ⚠️ |

### 缺失值模式

**前7种缺失组合**:
1. **2列缺失** (lr + dur): 32行 (22.5%)
2. **1列缺失** (perf_eval_loss): 22行 (15.5%)
3. **1列缺失** (lr): 17行 (12.0%)
4. **3列缺失** (lr + dur + perf): 16行 (11.3%)
5. **1列缺失** (dur): 13行 (9.2%)
6. **2列缺失** (dur + perf): 10行 (7.0%)
7. **9列缺失**: 10行 (7.0%) ❌

**严重问题**:
- ❌ **所有142行都有缺失值**
- 含有缺失值的行: **142/142 (100%)**
- 平均每行缺失: **2.89个单元格**
- 最大单行缺失: **10个单元格**（全部列！）
- 有10行缺失9列（只剩1列有数据）

---

## 4. Bug定位 (bug_localization)

### 基本信息
- 文件: `data/energy_research/training/training_data_bug_localization.csv`
- 样本数: 132行
- 特征数: 11列
- 总体缺失率: **24.38%** (354/1452单元格) ❌
- 完全无缺失行: **0/132 (0%)** ❌❌❌

### 缺失值详情

| 列名 | 缺失数量 | 缺失率 | 填充率 | 状态 |
|------|----------|--------|--------|------|
| **hyperparam_learning_rate** | **132** | **100.00%** | 0.00% | ❌❌❌ |
| **training_duration** | **83** | **62.88%** | 37.12% | ❌ |
| **perf_top1_accuracy** | **52** | **39.39%** | 60.61% | ❌ |
| **perf_top5_accuracy** | **52** | **39.39%** | 60.61% | ❌ |
| gpu_util_avg | 5 | 3.79% | 96.21% | ⚠️ |
| gpu_temp_max | 5 | 3.79% | 96.21% | ⚠️ |
| cpu_pkg_ratio | 5 | 3.79% | 96.21% | ⚠️ |
| gpu_power_fluctuation | 5 | 3.79% | 96.21% | ⚠️ |
| gpu_temp_fluctuation | 5 | 3.79% | 96.21% | ⚠️ |
| energy_cpu_total_joules | 5 | 3.79% | 96.21% | ⚠️ |
| energy_gpu_total_joules | 5 | 3.79% | 96.21% | ⚠️ |

### 缺失值模式

**前6种缺失组合**:
1. **2列缺失** (lr + dur): 62行 (47.0%)
2. **3列缺失** (lr + top1 + top5): 30行 (22.7%)
3. **1列缺失** (lr): 18行 (13.6%)
4. **4列缺失** (lr + dur + top1 + top5): 17行 (12.9%)
5. **11列缺失** (全部): 4行 (3.0%) ❌❌❌
6. **10列缺失**: 1行 (0.8%)

**极其严重的问题**:
- ❌❌❌ **learning_rate列100%缺失**（完全无数据！）
- ❌ **所有132行都有缺失值**
- 含有缺失值的行: **132/132 (100%)**
- 平均每行缺失: **2.68个单元格**
- 最大单行缺失: **11个单元格**（全部列！）
- 有4行缺失全部11列（完全空行！）

---

## 缺失值根本原因分析

### 1. 超参数命名不一致 🔴🔴🔴

**问题**: 不同模型使用了不同的超参数名称，导致统一提取时大量缺失。

**证据**:
- CIFAR-10的`batch_size`100%缺失 → 可能使用了其他名称
- Bug定位的`learning_rate`100%缺失 → 完全不存在此字段
- VulBERTa的`learning_rate`64.8%缺失 → 部分实验使用其他名称

**影响**: 无法学习超参数的因果效应（这是研究的核心！）

### 2. 性能指标提取失败 ⚠️

**问题**: 某些实验的性能指标未被正确提取。

**证据**:
- VulBERTa的`eval_loss`42.3%缺失
- Bug定位的`top1/top5_accuracy`39.4%缺失

**影响**: 无法学习超参数 → 性能的因果路径

### 3. 能耗数据轻微缺失 ⚠️

**问题**: 少数实验的能耗监控失败。

**证据**:
- 图像分类: 1行缺失所有7个能耗指标
- VulBERTa: 24行缺失所有7个能耗指标（16.9%）
- Bug定位: 5行缺失所有7个能耗指标（3.8%）

**影响**: 样本量略微减少，但不严重

### 4. 数据提取脚本问题 🔴

**推断**: 从主项目的`data.csv`提取训练数据时，未正确处理：
1. **超参数字段映射**（未统一不同命名）
2. **缺失值标记**（NaN vs 空字符串 vs 未填充）
3. **数据验证**（未检测到100%缺失的列）

---

## 缺失值影响评估

### 对DiBS因果学习的影响

| 影响类型 | 严重程度 | 说明 |
|----------|----------|------|
| **相关性计算失败** | ❌❌❌ 极严重 | nan结果导致DiBS无法优化 |
| **样本量减少** | ⚠️ 中等 | 删除缺失行后仍有足够样本 |
| **因果路径阻断** | ❌❌❌ 极严重 | 超参数缺失 → 无法学习超参数的因果效应 |
| **偏差引入** | ⚠️ 中等 | 系统性缺失（非随机）可能引入选择偏差 |

### 当前结果的可信度

| 任务组 | 可信度 | 原因 |
|--------|--------|------|
| 图像分类 | ❌ 不可信 | 48.4%数据含缺失，相关性计算失败 |
| Person_reID | ⚠️ 部分可信 | 35.3%数据含缺失，但densenet121严重偏倚 |
| VulBERTa | ❌❌❌ 完全不可信 | 100%数据含缺失，28.87%总体缺失率 |
| Bug定位 | ❌❌❌ 完全不可信 | 100%数据含缺失，learning_rate完全缺失 |

**结论**: **当前所有因果分析结果完全不可信，不能用于任何科学结论。**

---

## 修复优先级和方案

### 🔴 紧急修复（P0 - 必须立即处理）

#### 1. 修复超参数字段映射

**目标**: 统一不同模型的超参数命名

**方法**:
```python
# 在数据提取脚本中添加字段映射
HYPERPARAM_MAPPING = {
    'bug-localization': {
        'lr': 'hyperparam_learning_rate',  # 如果源数据中是'lr'
        'max_iter': 'training_duration',   # 统一到training_duration
    },
    'CIFAR-10': {
        'epochs': 'training_duration',     # 统一命名
        # batch_size可能在其他字段中
    },
    ...
}
```

**预期**: learning_rate从100%缺失 → 0%缺失

#### 2. 检查源数据文件

**行动**: 返回主项目`data/data.csv`，检查：
```bash
# 检查Bug定位的learning_rate字段名
grep "bug-localization" ../data/data.csv | head -5

# 检查CIFAR-10的batch_size字段名
grep "cifar" ../data/data.csv | head -5
```

**目的**: 确认原始字段名，更新映射规则

### ⚠️ 高优先级（P1 - 今天完成）

#### 3. 实现缺失值插补

**针对不同列类型的策略**:

**超参数列** (learning_rate, batch_size, etc.):
- **方法**: 同模型同数据集的**中位数插补**
- **原因**: 超参数通常在一定范围内，中位数代表典型值
- **示例**:
  ```python
  # MNIST模型的learning_rate缺失 → 用MNIST其他实验的中位数
  mnist_lr_median = df[df['is_mnist']==1]['hyperparam_learning_rate'].median()
  df.loc[(df['is_mnist']==1) & df['hyperparam_learning_rate'].isnull(),
         'hyperparam_learning_rate'] = mnist_lr_median
  ```

**性能指标列** (test_accuracy, eval_loss, etc.):
- **方法**: **删除该行**（不插补）
- **原因**: 性能是因果分析的目标变量，插补会严重偏倚因果效应估计
- **影响**: VulBERTa减少60行 → 82行仍足够

**能耗指标列** (gpu_util_avg, energy_*, etc.):
- **方法**: **删除该行**（7个能耗列同时缺失）
- **原因**: 能耗也是目标变量
- **影响**: 图像分类减少1行，VulBERTa减少24行，Bug定位减少5行

#### 4. 数据质量验证

**验证清单**:
- ✅ 无100%缺失的列
- ✅ 相关性矩阵可计算（无nan）
- ✅ 每个任务组至少50个完全无缺失的行
- ✅ 超参数列填充率 > 80%

### 🟡 中优先级（P2 - 明天完成）

#### 5. 优化数据提取脚本

**改进点**:
1. 添加超参数字段映射表
2. 添加数据质量检查
3. 生成缺失值报告
4. 自动检测异常模式

#### 6. 文档化决策

**记录**:
- 哪些字段被映射
- 哪些行被删除（原因）
- 插补方法和参数
- 数据质量变化

---

## 推荐修复流程

### Step 1: 回源检查（30分钟）

```bash
cd /home/green/energy_dl/nightly

# 1. 检查Bug定位的learning_rate原始字段名
python3 -c "
import pandas as pd
df = pd.read_csv('results/data.csv')
bug_loc = df[df['repo'] == 'bug-localization-by-dnn-and-rvsm']
print('Bug定位列名:', list(bug_loc.columns))
print('前5行样例:')
print(bug_loc.head()[['repo', 'model']].to_string())
"

# 2. 检查CIFAR-10的batch_size原始字段名
python3 -c "
import pandas as pd
df = pd.read_csv('results/data.csv')
cifar = df[df['repo'] == 'pytorch_resnet_cifar10']
print('CIFAR-10列名:', list(cifar.columns))
print('前5行样例:')
print(cifar.head()[['repo', 'model']].to_string())
"
```

### Step 2: 创建字段映射表（1小时）

基于Step 1的发现，创建完整映射：

```python
# analysis/config_energy.py 或新文件
HYPERPARAM_FIELD_MAPPING = {
    'bug-localization-by-dnn-and-rvsm': {
        'source_fields': {
            'lr': 'hyperparam_learning_rate',  # 待确认
            'max_iter': 'training_duration',
        }
    },
    'pytorch_resnet_cifar10': {
        'source_fields': {
            'epochs': 'training_duration',
            # batch_size待确认
        }
    },
    ...
}
```

### Step 3: 重新生成训练数据（1小时）

修改数据提取脚本，应用映射和插补：

```bash
cd /home/green/energy_dl/nightly/analysis

# 备份当前数据
cp -r data/energy_research/training data/energy_research/training_backup_20251224

# 重新生成（修改后的脚本）
python scripts/preprocess_stratified_data.py \
    --apply-field-mapping \
    --impute-hyperparams \
    --remove-missing-targets \
    --output-quality-report

# 验证质量
python scripts/validate_training_data.py
```

### Step 4: 快速验证DiBS（30分钟）

仅运行图像分类任务组（样本量最大）：

```bash
# 只分析图像分类
python scripts/demos/demo_dibs_single_task.py \
    --task image_classification \
    --iterations 1000 \
    --alpha 0.05

# 预期: 至少发现1-3条因果边
```

### Step 5: 完整重新分析（2小时）

确认修复有效后，运行所有4个任务组：

```bash
nohup bash scripts/experiments/run_energy_causal_analysis.sh > logs/rerun_20251224.log 2>&1 &
```

---

## 附录：缺失值统计表

### 图像分类 - 按One-Hot分组

| 数据集 | 样本数 | batch_size缺失 | learning_rate缺失 | training_duration缺失 |
|--------|--------|----------------|-------------------|---------------------|
| MNIST | 219 | 83 (37.9%) | 66 (30.1%) | 65 (29.7%) |
| CIFAR-10 | 39 | **39 (100%)** ❌ | 18 (46.2%) | 18 (46.2%) |

### Person_reID - 按模型分组

| 模型 | 样本数 | learning_rate缺失 | dropout缺失 | training_duration缺失 |
|------|--------|-------------------|-------------|---------------------|
| densenet121 | 43 | **18 (41.9%)** ❌ | **18 (41.9%)** ❌ | **18 (41.9%)** ❌ |
| hrnet18 | 37 | 7 (18.9%) | 7 (18.9%) | 6 (16.2%) |
| pcb | 36 | 6 (16.7%) | 6 (16.7%) | 6 (16.7%) |

### VulBERTa - 缺失组合

| 缺失列数 | 行数 | 占比 | 主要组合 |
|----------|------|------|----------|
| 0列 | 0 | 0% | - |
| 1列 | 52 | 36.6% | lr/dur/perf单独缺失 |
| 2列 | 50 | 35.2% | lr+dur |
| 3列 | 16 | 11.3% | lr+dur+perf |
| 9列 | 10 | 7.0% | 缺失几乎全部 ❌ |

### Bug定位 - 缺失组合

| 缺失列数 | 行数 | 占比 | 主要组合 |
|----------|------|------|----------|
| 0列 | 0 | 0% | - |
| 1列 | 18 | 13.6% | lr（总是缺失） |
| 2列 | 62 | 47.0% | lr+dur |
| 3列 | 30 | 22.7% | lr+top1+top5 |
| 4列 | 17 | 12.9% | lr+dur+top1+top5 |
| 11列 | 4 | 3.0% | **全部缺失** ❌❌❌ |

---

## 总结和建议

### 关键发现

1. ❌❌❌ **超参数命名不统一是根本原因**
   - Bug定位的learning_rate **100%缺失**
   - CIFAR-10的batch_size **100%缺失**

2. ❌❌ **VulBERTa和Bug定位无任何完全无缺失的行**
   - 必须进行插补或删除

3. ⚠️ **系统性缺失模式明显**
   - densenet121的缺失率是其他模型的2.5倍
   - CIFAR-10的缺失率高于MNIST
   - 不是随机缺失，存在选择偏差

### 立即行动

1. 🔴 **回源检查字段名**（30分钟）
2. 🔴 **创建字段映射表**（1小时）
3. 🔴 **重新生成训练数据**（1小时）
4. ⚠️ **快速验证DiBS**（30分钟）
5. ⚠️ **完整重新分析**（2小时）

**预计修复时间**: 5-6小时

### 预期改进

修复后的数据质量目标:

| 任务组 | 当前缺失率 | 目标缺失率 | 当前完全无缺失行 | 目标完全无缺失行 |
|--------|------------|------------|------------------|------------------|
| 图像分类 | 8.83% | **< 3%** | 48.4% | **> 90%** |
| Person_reID | 4.96% | **< 2%** | 64.7% | **> 90%** |
| VulBERTa | 28.87% | **< 5%** | 0% | **> 70%** |
| Bug定位 | 24.38% | **< 5%** | 0% | **> 70%** |

**预期DiBS结果**: 每个任务组发现 **3-8条因果边**（基于Adult数据集的经验）

---

**报告人**: Claude
**生成时间**: 2025-12-24
**下一步**: 等待用户确认修复方案
