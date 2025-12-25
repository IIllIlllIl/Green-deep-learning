# 图像分类组l2_regularization修正报告

**日期**: 2025-12-24
**问题**: 图像分类组l2_regularization列有88.79%缺失，导致质量验证失败
**状态**: ✅ **已修正，4/4任务组全部通过验证**

---

## 一、问题描述

### 1.1 初始验证结果（修正前）

**阶段4.2初次运行结果**（2025-12-24）:
- **通过率**: 3/4 (75%)
- **失败任务组**: image_classification

**image_classification验证失败原因**:
| 检查项 | 结果 | 原因 |
|--------|------|------|
| 缺失率检查 | ❌ | l2_regularization: 88.79%缺失 |
| 完全行检查 | ❌ | 只有11.2%完全行（目标>90%） |
| 相关矩阵 | ❌ | 包含4个nan值 |
| 数值范围 | ✅ | 通过 |
| One-Hot编码 | ✅ | 通过 |

### 1.2 根本原因分析

**数据分布**:
- examples模型（103个样本）: **不支持**weight_decay参数
  - mnist, mnist_ff, mnist_rnn, siamese 全部不支持
- pytorch_resnet_cifar10模型（13个样本）: 支持weight_decay参数

**缺失计算**:
```
缺失率 = 103 / 116 = 88.79%
```

**为什么这是问题？**
1. DiBS会误认为"l2_regularization=空"是一个有意义的变异值
2. 88.79%缺失导致完全无缺失行只有11.2%（远低于90%目标）
3. 相关矩阵无法正常计算（包含nan值）

---

## 二、修正方案

### 2.1 解决方案选择

**方案**: 从image_classification任务组配置中删除`l2_regularization`列

**原因**:
1. examples模型（占89%）根本不支持该参数
2. 保留该列会误导DiBS因果分析
3. 删除后图像分类组从17列降至16列，所有检查将通过

### 2.2 实施步骤

#### 步骤1: 修改数据分层脚本配置

**文件**: `/home/green/energy_dl/nightly/analysis/scripts/preprocess_stratified_data.py`

**修改位置**: 第19-29行，TASK_GROUPS['image_classification']['hyperparams']

**修改前**:
```python
'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                'l2_regularization', 'seed'],
```

**修改后**:
```python
'hyperparams': ['training_duration', 'hyperparam_learning_rate', 'seed'],
```

#### 步骤2: 修改质量验证脚本配置

**文件**: `/home/green/energy_dl/nightly/analysis/scripts/verify_stratified_data_quality.py`

**修改位置**: 第18-28行，TASK_CONFIGS['image_classification']

**修改前**:
```python
'expected_cols': 17,
'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                'l2_regularization', 'seed'],
```

**修改后**:
```python
'expected_cols': 16,
'hyperparams': ['training_duration', 'hyperparam_learning_rate', 'seed'],
```

#### 步骤3: 重新生成所有分层数据

**命令**:
```bash
cd /home/green/energy_dl/nightly/analysis/scripts
source /home/green/miniconda3/etc/profile.d/conda.sh
conda activate fairness
python3 preprocess_stratified_data.py \
  --input data/energy_research/raw/energy_data_extracted_v2.csv \
  --output-dir data/energy_research/processed
```

**生成结果**:
- `training_data_image_classification.csv`: 116行 × **16列**（从17列减少）
- `training_data_person_reid.csv`: 69行 × 20列（未变）
- `training_data_vulberta.csv`: 96行 × 15列（未变）
- `training_data_bug_localization.csv`: 91行 × 16列（未变）

#### 步骤4: 重新运行质量验证

**命令**:
```bash
python3 verify_stratified_data_quality.py \
  --data-dir data/energy_research/processed \
  --output-report ../docs/reports/STRATIFIED_DATA_QUALITY_REPORT.md
```

---

## 三、修正后验证结果 ✅

### 3.1 整体结果

**通过率**: 4/4 (100%) ✅

| 任务组 | 缺失率 | 完全行 | 相关矩阵 | 数值范围 | One-Hot | 总体 |
|--------|--------|--------|----------|----------|---------|------|
| 图像分类 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Person_reID | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| VulBERTa | ✅ | ✅ | ✅ | ✅ | ⏭️ | ✅ |
| Bug定位 | ✅ | ✅ | ✅ | ✅ | ⏭️ | ✅ |

### 3.2 图像分类组详细验证结果

**基本信息**:
- 文件: `training_data_image_classification.csv`
- 行数: 116行
- 列数: 16列（修正前17列）
- 预期行数: 100-120 ✅
- 预期列数: 16 ✅

**1. 缺失率检查**: ✅ 全部通过
```
✅ training_duration: 0%缺失
✅ hyperparam_learning_rate: 0%缺失
✅ seed: 0%缺失
✅ energy_cpu_total_joules: 0%缺失
✅ energy_gpu_total_joules: 0%缺失
✅ gpu_power_avg_watts: 0%缺失
✅ perf_test_accuracy: 0%缺失
```

**2. 完全无缺失行检查**: ✅ 通过
```
完全无缺失行: 116/116 (100.0%)
目标: >90%
✅ 达到目标（从11.2%提升至100.0%）
```

**3. 相关矩阵检查**: ✅ 通过
```
✅ 相关矩阵可计算: (14, 14)
✅ 无nan值（修正前有4个nan值）
✅ 对角线全为1
✅ 范围在[-1, 1]内
```

**4. 数值范围检查**: ✅ 通过
```
✅ energy_cpu_total_joules: 范围合理 [0, 1000000000.0]J
✅ energy_gpu_total_joules: 范围合理 [0, 1000000000.0]J
✅ gpu_power_avg_watts: 范围合理 [0, 600]W
✅ gpu_util_avg: 范围合理 [0, 100]%
✅ gpu_temp_max: 范围合理 [20, 110]°C
```

**5. One-Hot编码检查**: ✅ 通过
```
One-Hot列: is_mnist, is_cifar10
✅ 互斥性验证通过（所有行和=1）
✅ is_mnist: 103 个样本
✅ is_cifar10: 13 个样本
```

---

## 四、影响评估

### 4.1 数据完整性

**修正前**:
- 总列数: 17列
- 完全无缺失行: 13/116 (11.2%)
- l2_regularization缺失: 103/116 (88.79%)

**修正后**:
- 总列数: 16列
- 完全无缺失行: 116/116 (100.0%) ✅
- 所有列缺失率: 0.00% ✅

**改进**:
- 完全无缺失行比例: 11.2% → 100.0% (**+88.8%**)
- 相关矩阵nan值: 4个 → 0个 (**完全修复**)

### 4.2 因果分析影响

**修正前的问题**:
1. DiBS会将"l2_regularization=缺失"误认为一个有意义的变异值
2. 88.79%缺失会导致DiBS学习虚假因果边
3. 相关矩阵nan值会导致DiBS优化失败

**修正后的优势**:
1. 所有超参数都是真实记录值（不含缺失）
2. 相关矩阵可正常计算，DiBS优化不会失败
3. 因果边学习基于完整数据，结果更可靠

### 4.3 其他任务组

**无影响**: 其他3个任务组（Person_reID、VulBERTa、Bug定位）的配置和数据未变化，继续保持100%通过。

---

## 五、经验教训

### 教训1: 不同模型支持不同超参数集合 ⭐⭐⭐

**问题**: 在任务组内混合不同模型时，某些超参数可能只被部分模型支持。

**解决方案**:
1. 按任务分层时，只保留该任务组内所有模型共同支持的超参数
2. 或者，为每个模型单独创建任务组（但会导致样本量过小）

**本例**:
- examples模型不支持weight_decay → 删除l2_regularization列
- 保留了training_duration、learning_rate、seed（所有模型都支持）

### 教训2: 缺失值会误导DiBS ⭐⭐

**原因**: DiBS是基于连续数值的因果图学习算法，无法正确处理缺失值。

**后果**:
1. DiBS会将"缺失"视为一个特殊数值（如0或nan）
2. 导致学习虚假因果边："l2_regularization=缺失 → 能耗低"
3. 实际上，这只是因为examples模型本身能耗较低，与l2_regularization无关

**最佳实践**: 确保所有超参数列的缺失率 < 10%，理想情况下0%。

### 教训3: 测试驱动开发的价值 ⭐⭐⭐

**本次修正流程**:
1. 发现问题：质量验证脚本检测到88.79%缺失
2. 诊断根因：分析数据分布，发现模型不支持该参数
3. 实施修正：修改配置，重新生成数据
4. 验证修正：重新运行验证脚本，确认4/4通过

**如果没有质量验证脚本**: 可能直接进入阶段5，DiBS会学习虚假因果边，浪费大量计算时间（60分钟+）。

---

## 六、修正文件清单

### 6.1 代码文件

| 文件 | 修改内容 | 行号 |
|------|---------|------|
| `preprocess_stratified_data.py` | 删除image_classification的l2_regularization | 第26行 |
| `verify_stratified_data_quality.py` | 更新expected_cols: 17→16, 删除l2_regularization | 第21-23行 |

### 6.2 数据文件

| 文件 | 变化 | 说明 |
|------|------|------|
| `training_data_image_classification.csv` | 17列→16列 | 删除l2_regularization列 |
| `training_data_person_reid.csv` | 无变化 | 20列保持不变 |
| `training_data_vulberta.csv` | 无变化 | 15列保持不变 |
| `training_data_bug_localization.csv` | 无变化 | 16列保持不变 |
| `stratified_data_stats.json` | 更新 | image_classification列数: 17→16 |

### 6.3 文档文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `STRATIFIED_DATA_QUALITY_REPORT.md` | ✅ 已更新 | 反映4/4通过结果 |
| `IMAGE_CLASSIFICATION_L2_REGULARIZATION_FIX_20251224.md` | ✅ 新建 | 本修正报告 |

---

## 七、阶段4最终状态

### 7.1 阶段4完成检查清单

- [x] 验证脚本编写完成（verify_stratified_data_quality.py）
- [x] 5维度验证实施（缺失率、完全行、相关矩阵、数值范围、One-Hot）
- [x] 质量报告生成（STRATIFIED_DATA_QUALITY_REPORT.md）
- [x] 问题识别（图像分类组l2_regularization缺失88.79%）
- [x] **问题修正**（删除l2_regularization列，重新生成数据）
- [x] **重新验证**（4/4任务组全部通过） ✅

### 7.2 数据质量最终状态

| 维度 | 状态 | 说明 |
|------|------|------|
| **任务组数量** | 4个 | image_classification, person_reid, vulberta, bug_localization |
| **总样本量** | 372行 | 116+69+96+91 |
| **超参数缺失率** | 0.00% | 所有超参数100%填充 ✅ |
| **能耗数据缺失率** | 0.00% | 11列能耗指标100%填充 ✅ |
| **性能指标缺失率** | 0.00% | 按任务分层后100%填充 ✅ |
| **完全无缺失行** | 100.0% | 所有任务组100%完全行 ✅ |
| **相关矩阵可计算性** | 100% | 4个任务组相关矩阵全部无nan值 ✅ |
| **One-Hot编码正确性** | 100% | 互斥性验证100%通过 ✅ |
| **质量验证通过率** | 4/4 (100%) | **全部通过** ✅ |

---

## 八、下一步建议

### ✅ 阶段4已100%完成

**阶段4目标**: 验证分层数据满足DiBS因果分析的前提条件
**达成状态**: ✅ **完全达成**（4/4任务组通过所有质量检查）

### 进入阶段5的准备就绪

**前提条件**:
- [x] 数据质量100%通过 ✅
- [x] 相关矩阵可计算（无nan值） ✅
- [x] 完全无缺失行比例 >90%（实际100%） ✅
- [x] One-Hot编码正确（控制异质性） ✅

**阶段5运行指令**（待用户确认后执行）:

#### 方式1: 串行运行（稳定，约60分钟）
```bash
screen -S dibs_analysis
cd /home/green/energy_dl/nightly/analysis/scripts
conda activate fairness

python3 run_stratified_dibs_analysis.py \
  --tasks image_classification person_reid vulberta bug_localization \
  --n-particles 20 \
  --n-steps 1000 \
  --output-dir ../results/energy_research/task_specific/

# 完成后按Ctrl+A+D退出screen
```

#### 方式2: 并行运行（快速，约20分钟）
```bash
screen -S dibs_analysis
cd /home/green/energy_dl/nightly/analysis/scripts
conda activate fairness

python3 run_stratified_dibs_analysis.py \
  --tasks image_classification person_reid vulberta bug_localization \
  --n-particles 20 \
  --n-steps 1000 \
  --parallel \
  --output-dir ../results/energy_research/task_specific/

# 完成后按Ctrl+A+D退出screen
```

#### 监控进度
```bash
# 重新连接screen
screen -r dibs_analysis

# 或查看日志
tail -f /home/green/energy_dl/nightly/analysis/logs/energy_research/stratified_dibs_*.log
```

---

## 九、总结

### 9.1 问题修正

✅ **成功修正图像分类组l2_regularization缺失问题**

**修正方法**: 从配置中删除该列（因examples模型不支持）
**修正结果**: 完全无缺失行从11.2%提升至100.0%
**修正影响**: 相关矩阵nan值从4个降至0个

### 9.2 阶段4完成

✅ **阶段4质量验证100%完成**

**验证通过率**: 4/4 (100%)
**所有检查**: 缺失率、完全行、相关矩阵、数值范围、One-Hot编码全部通过
**数据完整性**: 超参数、能耗、性能指标全部0.00%缺失

### 9.3 阶段5准备

✅ **数据已准备就绪，可以进入阶段5 DiBS因果分析**

**预期运行时间**: 串行60分钟，并行20分钟
**分析任务组**: 4个（image_classification, person_reid, vulberta, bug_localization）
**预期输出**: 每个任务组的因果图、因果边、因果效应估计

---

**报告人**: Claude
**生成时间**: 2025-12-24
**修正耗时**: 约10分钟
**下一阶段**: 阶段5 - DiBS因果分析验证（待用户确认）
**建议**: 数据质量已达到最优状态，建议立即进入阶段5 ✅
