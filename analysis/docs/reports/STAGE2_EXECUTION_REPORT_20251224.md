# 阶段2执行报告 - 数据重新提取完成

**日期**: 2025-12-24
**阶段**: 阶段2 - 实现数据提取脚本（从JSON+model config）
**状态**: ✅ **完成**
**耗时**: 约2.5小时

---

## 执行摘要

✅ **阶段2成功完成！** 已从experiment.json + models_config.json联合提取完整数据。

### 关键成果

1. ✅ **数据提取脚本**: `extract_from_json_with_defaults.py` (478行)
2. ✅ **测试验证脚本**: `test_extract_from_json.py` (5/5测试通过)
3. ✅ **安全性验证脚本**: `verify_extracted_data_safety.py`
4. ✅ **提取数据文件**: `energy_data_extracted_v2.csv` (418行 × 34列)
5. ✅ **数据来源**: 100%来自我们的实验结果
6. ✅ **超参数完整性**: 每个仓库相关超参数100%填充 ⭐⭐⭐

---

## 一、核心功能实现

### 1.1 默认值回溯机制 ⭐⭐⭐

**原理**: 结合experiment.json记录值 + models_config.json默认值

```python
对于每个实验的每个超参数:
  1. 先检查experiment.json中是否有记录值
     → 如果有，使用记录值（被变异的超参数）
  2. 如果没有记录，从models_config.json提取默认值
     → 使用config["models"][repo]["supported_hyperparams"][param]["default"]
  3. 应用字段映射统一命名
     → max_iter → training_duration
     → alpha → l2_regularization
```

**示例**:
- Bug定位实验只记录`{"seed": 917}`
- 从models_config回溯: `max_iter=10000`, `alpha=1e-05`, `kfold=10`
- 最终完整超参数: 4个字段全部填充 ✅

### 1.2 字段映射表

| 原始字段 | 统一字段 | 使用模型 |
|---------|---------|----------|
| **训练迭代次数** |||
| `epochs` | `training_duration` | examples, VulBERTa, Person_reID, CIFAR-10, MRT-OAST |
| `max_iter` | `training_duration` | Bug定位 |
| **L2正则化** |||
| `weight_decay` | `l2_regularization` | VulBERTa, CIFAR-10, MRT-OAST |
| `alpha` | `l2_regularization` | Bug定位 |
| **学习率** |||
| `learning_rate` | `hyperparam_learning_rate` | examples, VulBERTa, MRT-OAST, Person_reID |
| `lr` | `hyperparam_learning_rate` | CIFAR-10 |
| **其他** |||
| `batch_size` | `hyperparam_batch_size` | examples |
| `dropout`/`droprate` | `hyperparam_dropout` | MRT-OAST, Person_reID |
| `kfold` | `hyperparam_kfold` | Bug定位 |
| `seed` | `seed` | 所有模型 |

### 1.3 派生指标计算

新增3个中介变量（用于因果路径分析）:

1. **cpu_pkg_ratio**: CPU能耗占比 = CPU能耗 / (CPU + GPU总能耗)
2. **gpu_power_fluctuation**: GPU功率波动性 = max_watts - min_watts
3. **gpu_temp_fluctuation**: GPU温度波动性 = temp_max - temp_avg

---

## 二、测试验证结果

### 2.1 单元测试（5/5通过）✅

| 测试项 | 结果 | 说明 |
|-------|------|------|
| 字段映射 | ✅ 通过 | 8个映射规则全部正确 |
| 超参数提取（真实数据） | ✅ 通过 | Bug定位实验提取4个完整超参数 |
| 并行模式提取 | ✅ 通过 | 空hyperparameters成功回溯 |
| 能耗指标提取 | ✅ 通过 | 能耗字段正确提取 |
| Dry Run（前10个实验） | ✅ 通过 | 数据格式正确 |

### 2.2 完整数据提取统计

```
原始实验数: 558个
有效数据行: 418行 (删除140行性能/能耗全缺失)
总列数: 34列
并行模式: 231行
非并行模式: 187行
```

---

## 三、数据安全性验证 ✅

### 3.1 数据来源验证（100%通过）

| 验证项 | 结果 | 说明 |
|-------|------|------|
| **experiment_id格式** | ✅ 通过 | 418行全部有效 |
| **timestamp格式** | ✅ 通过 | 418行全部有效 |
| **repository来源** | ✅ 通过 | 6个仓库全部来自models_config.json |
| **model定义** | ✅ 通过 | 10个模型全部在models_config中定义 |
| **数据来源** | ✅ 通过 | **100%来自experiment.json + models_config.json** |
| **外部数据** | ✅ 通过 | **未引入任何外部数据** |
| **敏感信息** | ✅ 通过 | 列名不包含敏感词 |
| **重复行** | ✅ 通过 | 无重复行 |

### 3.2 数值范围合理性（100%通过）

| 指标 | 合理范围 | 验证结果 |
|------|---------|---------|
| CPU能耗 | 0 - 1GJ | ✅ 全部合理 |
| GPU能耗 | 0 - 1GJ | ✅ 全部合理 |
| GPU功率 | 0 - 600W | ✅ 全部合理 |
| GPU利用率 | 0 - 100% | ✅ 全部合理 |
| GPU温度 | 0 - 110°C | ✅ 全部合理 |

---

## 四、超参数完整性验证 ⭐⭐⭐

### 4.1 按仓库分组统计（关键发现）

**重要**: 全局看超参数缺失率41.29%是**误导性**的！按仓库分组后，每个仓库的相关超参数都是**100%填充**！

#### MRT-OAST (n=46)
```
支持的超参数: dropout, epochs, learning_rate, seed, weight_decay

hyperparam_dropout:        46/46 ✅ 100%完整
hyperparam_learning_rate:  46/46 ✅ 100%完整
l2_regularization:         46/46 ✅ 100%完整 (从weight_decay映射)
seed:                      46/46 ✅ 100%完整
training_duration:         46/46 ✅ 100%完整 (从epochs映射)
```

#### VulBERTa (n=96)
```
支持的超参数: epochs, learning_rate, seed, weight_decay

hyperparam_learning_rate:  96/96 ✅ 100%完整
l2_regularization:         96/96 ✅ 100%完整 (从weight_decay映射)
seed:                      96/96 ✅ 100%完整
training_duration:         96/96 ✅ 100%完整 (从epochs映射)
```

#### Bug定位 (n=91)
```
支持的超参数: alpha, kfold, max_iter, seed

hyperparam_kfold:          91/91 ✅ 100%完整
l2_regularization:         91/91 ✅ 100%完整 (从alpha映射) ⭐
seed:                      91/91 ✅ 100%完整
training_duration:         91/91 ✅ 100%完整 (从max_iter映射) ⭐

注意: Bug定位没有learning_rate超参数，使用alpha（L2正则化）
```

#### examples (n=103)
```
支持的超参数: batch_size, epochs, learning_rate, seed

hyperparam_batch_size:    103/103 ✅ 100%完整
hyperparam_learning_rate: 103/103 ✅ 100%完整
seed:                     103/103 ✅ 100%完整
training_duration:        103/103 ✅ 100%完整 (从epochs映射)
```

#### CIFAR-10 (n=13)
```
支持的超参数: epochs, learning_rate, seed, weight_decay

hyperparam_learning_rate:  13/13 ✅ 100%完整
l2_regularization:         13/13 ✅ 100%完整
seed:                      13/13 ✅ 100%完整
training_duration:         13/13 ✅ 100%完整
```

#### Person_reID (n=69)
```
支持的超参数: dropout, epochs, learning_rate, seed

hyperparam_dropout:        69/69 ✅ 100%完整
hyperparam_learning_rate:  69/69 ✅ 100%完整
seed:                      69/69 ✅ 100%完整
training_duration:         69/69 ✅ 100%完整
```

### 4.2 关键结论 ⭐⭐⭐

✅ **v2.0方案成功！超参数完整性达到100%！**

- **全局视角**: 超参数列有41.29%缺失（误导性）
  - 原因：不同模型支持不同超参数集合
  - 例如：batch_size只有examples使用，其他模型"全缺失"是正常的

- **分组视角**: 每个仓库的相关超参数**100%完整** ⭐⭐⭐
  - MRT-OAST的5个超参数: 100%填充
  - VulBERTa的4个超参数: 100%填充
  - Bug定位的4个超参数: 100%填充（包括默认值回溯）
  - examples的4个超参数: 100%填充

---

## 五、能耗和性能指标完整性

### 5.1 能耗指标（100%完整）✅

| 指标类型 | 列数 | 填充率 | 状态 |
|---------|------|--------|------|
| **基础能耗** | 3 | 100% | ✅ 完整 |
| - energy_cpu_total_joules | 1 | 100% | ✅ |
| - energy_gpu_total_joules | 1 | 100% | ✅ |
| - gpu_power_avg_watts | 1 | 100% | ✅ |
| **GPU利用率** | 1 | 100% | ✅ 完整 |
| **GPU温度** | 2 | 100% | ✅ 完整 |
| **派生指标** | 3 | 100% | ✅ 完整 |
| - cpu_pkg_ratio | 1 | 100% | ✅ |
| - gpu_power_fluctuation | 1 | 100% | ✅ |
| - gpu_temp_fluctuation | 1 | 100% | ✅ |

**总计**: 11个能耗指标列，**全部100%填充**！⭐⭐⭐

### 5.2 性能指标（82.32%缺失）⚠️

| 任务 | 性能指标 | 填充率 | 说明 |
|-----|---------|--------|------|
| 图像分类 | test_accuracy | 27.8% | 需要删除空行 |
| Person_reID | map, rank1, rank5 | 16.5% | 需要删除空行 |
| VulBERTa | eval_loss | 23.0% | 需要删除空行 |
| Bug定位 | top1_accuracy, top5_accuracy | 21.8% | 需要删除空行 |
| MRT-OAST | accuracy, precision, recall | 11.0% | 需要删除空行 |

**说明**: 性能指标缺失是因为不同任务有不同指标，这是**正常的**。
- 例如：图像分类实验没有mAP指标（Person_reID特有）
- 需要在阶段3按任务分组，删除性能全缺失的行

---

## 六、数据质量对比：v1.0 vs v2.0

| 指标 | v1.0（原始数据） | v2.0（重新提取） | 改进 |
|------|----------------|-----------------|------|
| **超参数缺失率（全局）** | 8-100% | 41.29% | ⚠️ 误导性 |
| **超参数缺失率（分组）** | 32-100% | **0%** | ✅ **100%改进** ⭐⭐⭐ |
| **能耗数据填充率** | 91.7% | **100%** | ✅ **+8.3%** |
| **派生指标填充率** | 0% | **100%** | ✅ **新增3个中介变量** |
| **数据来源可追溯** | 部分 | **100%** | ✅ **完全可追溯** |

**关键改进** ⭐⭐⭐:
1. **超参数100%填充**（按仓库分组）
2. **能耗数据100%填充**（删除无能耗行）
3. **新增派生指标100%填充**（中介变量）
4. **数据来源100%可追溯**（全部来自实验结果）

---

## 七、剩余问题和建议

### 7.1 需要在阶段3解决的问题

1. **性能指标缺失** (82.32%)
   - **原因**: 不同任务有不同性能指标
   - **解决**: 按任务分组，删除性能全缺失的行
   - **预期**: 每个任务组性能指标接近100%填充

2. **完全无缺失行为0%**
   - **原因**: 全局统计包含所有性能指标列
   - **解决**: 按任务分组后，只保留该任务的性能指标
   - **预期**: 每个任务组完全无缺失行 > 80%

3. **数据分层**
   - **任务**: 按4个任务组分层保存数据
   - **输出**: 4个CSV文件（image_classification, person_reid, vulberta, bug_localization）

### 7.2 阶段3建议

✅ **数据安全可靠，可以立即进入阶段3（数据分层与保存）**

**阶段3任务**:
1. 按任务分组（4个任务组）
2. 删除性能全缺失的行
3. 添加One-Hot编码（控制异质性）
4. 选择任务特定的变量
5. 保存为4个训练数据文件

**预期结果**:
- 图像分类: ~116行（examples 103 + CIFAR-10 13），完全无缺失行 > 90%
- Person_reID: ~69行，完全无缺失行 > 90%
- VulBERTa: ~96行，完全无缺失行 > 80%
- Bug定位: ~91行，完全无缺失行 > 80%

---

## 八、交付成果

### 8.1 代码文件

1. ✅ `extract_from_json_with_defaults.py` (478行)
   - 核心数据提取脚本
   - 实现默认值回溯机制
   - 实现字段映射
   - 计算派生指标

2. ✅ `test_extract_from_json.py` (257行)
   - 5个单元测试
   - 全部通过 ✅

3. ✅ `verify_extracted_data_safety.py` (268行)
   - 数据来源验证
   - 安全性检查
   - 质量分析
   - 缺失值详细报告

### 8.2 数据文件

1. ✅ `data/energy_research/raw/energy_data_extracted_v2.csv`
   - 418行 × 34列
   - 包含完整超参数（按仓库分组100%填充）
   - 包含完整能耗指标（100%填充）
   - 包含派生指标（100%填充）

2. ✅ `data/energy_research/raw/extracted_columns_info.json`
   - 列信息清单
   - 用于阶段3数据分层

### 8.3 文档

1. ✅ `analysis/docs/reports/DATA_REEXTRACTION_PROPOSAL_V2_20251224.md`
   - v2.0完整方案文档
   - 默认值回溯机制详解

2. ✅ `analysis/docs/reports/hyperparam_field_mapping.json`
   - 字段映射表

3. ✅ **本报告**: `STAGE2_EXECUTION_REPORT_20251224.md`
   - 阶段2完整执行报告

---

## 九、关键验证结论

### ✅ 数据来源验证

- **100%来自我们的实验结果**（experiment.json + models_config.json）
- **未引入任何外部数据**
- 所有repository和model都在models_config.json中定义
- 无敏感信息，无重复行

### ✅ 数据安全性

- 列名不包含敏感词
- 数值范围全部合理
- 无异常值或超出范围的数据

### ✅ 超参数完整性 ⭐⭐⭐

- **按仓库分组后，每个仓库的相关超参数100%填充**
- seed: 100%完整（所有模型）
- training_duration: 100%完整（所有模型）
- 其他超参数: 按模型100%完整

### ✅ 能耗数据完整性

- 11个能耗指标列全部100%填充
- 派生指标（中介变量）全部100%填充

### ⚠️ 需要在阶段3解决

- 性能指标需要按任务分组分析
- 完全无缺失行需要按任务分组统计

---

## 十、下一步行动

### ✅ 建议立即进入阶段3

**阶段3任务**: 数据分层与保存（预计1-1.5小时）

1. 创建 `preprocess_stratified_data.py`
2. 按4个任务组分层
3. 删除性能全缺失的行
4. 添加One-Hot编码
5. 保存为4个训练数据文件

**预期输出**:
- `training_data_image_classification.csv` (~116行)
- `training_data_person_reid.csv` (~69行)
- `training_data_vulberta.csv` (~96行)
- `training_data_bug_localization.csv` (~91行)

---

**报告人**: Claude
**生成时间**: 2025-12-24
**阶段状态**: ✅ 完成
**下一阶段**: 阶段3 - 数据分层与保存
