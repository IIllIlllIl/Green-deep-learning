# 从experiment.json重建CSV的完整流程

**创建日期**: 2025-12-11
**版本**: v1.0
**状态**: ✅ 已验证

---

## 概述

本文档说明如何从`experiment.json`文件重建完整的CSV数据文件，包括所有实验元数据、超参数、性能指标和能耗数据。

## 为什么需要重建

在项目发展过程中，CSV格式可能会发生变化，或者原始CSV文件可能损坏。由于每个实验都有完整的`experiment.json`文件，我们可以随时重建CSV数据。

---

## 完整流程（6步）

### 步骤概览

```
experiment.json文件
    ↓
步骤1: 分离新老实验
    ↓
步骤2: 扫描experiment.json字段
    ↓
步骤3: 设计新CSV表头
    ↓
步骤4: 从experiment.json重建新实验CSV
    ↓
步骤5: 添加变异分析列（增强版）
    ↓
步骤6: 验证数据完整性
```

---

## 步骤详解

### 步骤1: 分离新老实验

**目的**: 区分实验数据来源，避免混淆

**脚本**: `scripts/separate_old_new_experiments.py`

**执行**:
```bash
python3 scripts/separate_old_new_experiments.py
```

**输出**:
- `results/summary_old.csv` - 老实验数据（保持原样）
- 确定新实验范围（基于run目录时间戳）

**说明**:
- 老实验: 2025-12-01之前的实验，CSV格式可能不同
- 新实验: 2025-12-01及之后的实验，从experiment.json重建

---

### 步骤2: 扫描experiment.json字段

**目的**: 了解所有experiment.json文件中存在的字段

**脚本**: `scripts/step1_scan_experiment_json.py`

**执行**:
```bash
python3 scripts/step1_scan_experiment_json.py
```

**输出**:
- `results/experiment_json_fields_scan.json` - 字段扫描结果

**扫描内容**:
- 所有顶层字段
- 嵌套字段（hyperparameters, performance_metrics, energy_metrics）
- 并行实验特殊字段（foreground/background）

---

### 步骤3: 设计新CSV表头

**目的**: 基于扫描结果设计完整的CSV表头

**脚本**: `scripts/step2_design_csv_header.py`

**执行**:
```bash
python3 scripts/step2_design_csv_header.py
```

**输出**:
- `results/new_csv_header_design.json` - CSV表头设计

**表头结构** (80列):
```
基础信息 (7列):
  - experiment_id, timestamp, repository, model
  - training_success, duration_seconds, retries

超参数 (9列):
  - hyperparam_alpha, hyperparam_batch_size, hyperparam_dropout
  - hyperparam_epochs, hyperparam_kfold, hyperparam_learning_rate
  - hyperparam_max_iter, hyperparam_seed, hyperparam_weight_decay

性能指标 (9列):
  - perf_accuracy, perf_best_val_accuracy, perf_map
  - perf_precision, perf_rank1, perf_rank5
  - perf_recall, perf_test_accuracy, perf_test_loss

能耗指标 (11列):
  - energy_cpu_pkg_joules, energy_cpu_ram_joules, energy_cpu_total_joules
  - energy_gpu_avg_watts, energy_gpu_max_watts, energy_gpu_min_watts
  - energy_gpu_total_joules, energy_gpu_temp_avg_celsius, energy_gpu_temp_max_celsius
  - energy_gpu_util_avg_percent, energy_gpu_util_max_percent

元数据 (3列):
  - experiment_source, num_mutated_params, mutated_param

模式信息 (2列):
  - mode, error_message

前景实验详细信息 (36列): fg_*
  - 前景实验的所有超参数、性能、能耗数据

背景实验信息 (4列):
  - bg_repository, bg_model, bg_note, bg_log_directory
```

---

### 步骤4: 从experiment.json重建新实验CSV

**目的**: 读取所有experiment.json文件，构建完整CSV

**脚本**: `scripts/step3_rebuild_new_csv.py`

**执行**:
```bash
python3 scripts/step3_rebuild_new_csv.py
```

**输出**:
- `results/summary_new.csv` - 重建的CSV文件

**处理逻辑**:
1. 遍历所有`results/run_*/*/experiment.json`文件
2. 根据mode字段区分并行/非并行实验
3. 非并行实验: 直接提取顶层字段
4. 并行实验:
   - 从foreground中提取主数据
   - 填充fg_*前缀列
   - 记录background信息
5. 过滤无效实验（VulBERTa/cnn）

**关键代码逻辑**:
```python
if is_parallel:
    # 并行模式：数据在foreground中
    fg = data.get('foreground', {})
    bg = data.get('background', {})

    row['repository'] = fg.get('repository', '')
    row['model'] = fg.get('model', '')
    # ... 提取fg数据

    # 填充fg_*列
    row['fg_repository'] = fg.get('repository', '')
    # ... 填充所有fg_*列

    # 记录background
    row['bg_repository'] = bg.get('repository', '')
else:
    # 非并行模式：数据在顶层
    row['repository'] = data.get('repository', '')
    row['model'] = data.get('model', '')
    # ... 直接提取数据
```

---

### 步骤5: 添加变异分析列（增强版）⭐

**目的**: 分析超参数变异情况，增强CSV可分析性

**脚本**: `scripts/step5_enhance_mutation_analysis.py`

**执行**:
```bash
python3 scripts/step5_enhance_mutation_analysis.py
```

**功能**:
1. **更新num_mutated_params计数** - 正确计数seed参数
2. **填充mutated_param列** - 识别单参数变异的参数名
3. **填充默认值** - 从models_config.json读取默认值填充空列

**变异计数逻辑**:
```python
def count_mutated_params(row, models_config):
    defaults = get_default_hyperparams(models_config, repository, model)
    mutated_count = 0

    for param, default_value in defaults.items():
        actual_value = row.get(f'hyperparam_{param}', '').strip()

        # 空值 = 使用默认值
        if not actual_value:
            continue

        # default=None时（如seed），有值就算变异
        if default_value is None:
            mutated_count += 1
        # 比较实际值与默认值
        elif actual_value != default_value:
            mutated_count += 1

    return mutated_count
```

**mutated_param逻辑**:
```python
# 仅当变异1个参数时填充
if count == 1:
    row['mutated_param'] = mutated_params[0]  # 参数名称
else:
    row['mutated_param'] = ''  # 多参数或baseline时为空
```

**填充默认值逻辑**:
```python
# 对于空的hyperparam列，填充默认值
for param, default_value in defaults.items():
    col_name = f'hyperparam_{param}'
    if col_name in row and not row[col_name].strip():
        row[col_name] = format_value(default_value, param_type)
```

**输出**:
- `results/summary_new.csv` - 更新后的CSV（包含变异分析列）
- `results/summary_new.csv.backup_step5` - 备份

---

### 步骤6: 验证数据完整性

**目的**: 确保重建的CSV数据正确

**脚本**: `tests/test_step5_enhancement.py`

**执行**:
```bash
python3 tests/test_step5_enhancement.py
```

**验证项**:
1. ✓ num_mutated_params正确计数seed参数
2. ✓ mutated_param列正确识别单参数变异
3. ✓ hyperparam列正确填充默认值
4. ✓ 数据一致性检查

**测试输出示例**:
```
测试1: 验证num_mutated_params正确计数seed参数
找到 146 个设置了seed的实验
其中 51 个仅变异seed参数
✓ 测试通过

测试2: 验证mutated_param列正确识别单参数变异
单参数变异实验: 235
多参数变异实验: 26
Baseline实验: 4
✓ 测试通过

测试3: 验证hyperparam列填充默认值正确
检查了 1085 个超参数值
其中 724 个已填充
填充率: 66.7%
✓ 测试通过

测试4: 一致性检查
✓ 所有数据一致

测试结果: 4 通过, 0 失败
```

---

## 数据质量保证

### 过滤规则

1. **VulBERTa/cnn过滤**:
   - 原因: 训练代码未实现
   - 识别: `repository='VulBERTa' AND model='cnn'`
   - 操作: 在step3中自动跳过

2. **数据完整性检查**:
   - CSV格式: 80列标准格式
   - 必填字段: experiment_id, timestamp, repository, model
   - 能耗数据: CPU/GPU能耗指标应存在

### 备份策略

每个步骤都会创建备份:
- `summary_new.csv.backup_step5` - Step5执行前的备份
- 建议在执行前手动备份: `cp results/summary_new.csv results/summary_new.csv.manual_backup`

---

## 关键文件位置

### 脚本文件
- `scripts/separate_old_new_experiments.py` - 步骤1
- `scripts/step1_scan_experiment_json.py` - 步骤2
- `scripts/step2_design_csv_header.py` - 步骤3
- `scripts/step3_rebuild_new_csv.py` - 步骤4
- `scripts/step5_enhance_mutation_analysis.py` - 步骤5 ⭐

### 测试文件
- `tests/test_step5_enhancement.py` - 步骤6验证

### 数据文件
- `results/summary_old.csv` - 老实验数据
- `results/summary_new.csv` - 新实验数据（重建后）
- `results/experiment_json_fields_scan.json` - 字段扫描结果
- `results/new_csv_header_design.json` - CSV表头设计

### 配置文件
- `mutation/models_config.json` - 模型配置（默认值来源）

---

## 使用场景

### 场景1: 完整重建

当需要从头重建CSV时:
```bash
# 步骤1-6完整执行
python3 scripts/separate_old_new_experiments.py
python3 scripts/step1_scan_experiment_json.py
python3 scripts/step2_design_csv_header.py
python3 scripts/step3_rebuild_new_csv.py
python3 scripts/step5_enhance_mutation_analysis.py
python3 tests/test_step5_enhancement.py
```

### 场景2: 仅更新变异分析

当CSV已存在，只需更新变异分析列:
```bash
python3 scripts/step5_enhance_mutation_analysis.py
python3 tests/test_step5_enhancement.py
```

### 场景3: 添加新实验后重建

当有新实验数据时:
```bash
# 重建CSV（会自动包含所有实验）
python3 scripts/step3_rebuild_new_csv.py

# 更新变异分析
python3 scripts/step5_enhance_mutation_analysis.py

# 验证
python3 tests/test_step5_enhancement.py
```

---

## 常见问题

### Q1: 为什么seed被算作变异参数？

**A**: 在models_config.json中，某些模型的seed默认值为`null`（如Person_reID、pytorch_resnet）。当实验设置了seed值时，即使值为1、42等，也算作变异，因为改变了模型的随机性。

**示例**:
```json
{
  "seed": {
    "default": null,  // null表示默认不设置seed
    "range": [0, 9999]
  }
}
```

### Q2: mutated_param为空是正常的吗？

**A**: 是的。mutated_param仅在单参数变异时填充。多参数变异和baseline实验的mutated_param为空。

**规则**:
- `num_mutated_params == 1` → mutated_param填充参数名
- `num_mutated_params == 0` → mutated_param为空（baseline）
- `num_mutated_params > 1` → mutated_param为空（多参数变异）

### Q3: 如何添加新的CSV列？

**A**: 修改step2和step3脚本:
1. 在`step2_design_csv_header.py`中添加新列名
2. 在`step3_rebuild_new_csv.py`中添加提取逻辑
3. 重新运行step3和step5

### Q4: 如何验证重建的CSV与原始CSV一致？

**A**: 可以使用diff工具比较:
```bash
# 比较行数
wc -l results/summary_old.csv results/summary_new.csv

# 比较列数
head -1 results/summary_old.csv | tr ',' '\n' | wc -l
head -1 results/summary_new.csv | tr ',' '\n' | wc -l

# 详细比较（如果格式一致）
diff results/summary_old.csv results/summary_new.csv
```

---

## 技术细节

### 并行实验的数据结构

```json
{
  "experiment_id": "...",
  "mode": "parallel",
  "foreground": {
    "repository": "examples",
    "model": "mnist",
    "hyperparameters": {...},
    "performance_metrics": {...},
    "energy_metrics": {...}
  },
  "background": {
    "repository": "examples",
    "model": "siamese",
    "note": "Background training served as GPU load only"
  }
}
```

### 非并行实验的数据结构

```json
{
  "experiment_id": "...",
  "repository": "examples",
  "model": "mnist",
  "hyperparameters": {...},
  "performance_metrics": {...},
  "energy_metrics": {...}
}
```

---

## 版本历史

### v1.0 (2025-12-11)
- ✅ 完成6步流程实现
- ✅ 支持并行/非并行实验
- ✅ 实现变异分析增强
- ✅ 完整测试验证
- ✅ 文档化流程

---

## 参考文档

- [Step1 扫描脚本](../scripts/step1_scan_experiment_json.py)
- [Step2 设计脚本](../scripts/step2_design_csv_header.py)
- [Step3 重建脚本](../scripts/step3_rebuild_new_csv.py)
- [Step5 增强脚本](../scripts/step5_enhance_mutation_analysis.py)
- [测试脚本](../tests/test_step5_enhancement.py)
- [模型配置](../mutation/models_config.json)

---

**维护者**: Green
**最后更新**: 2025-12-11
**状态**: ✅ 已验证并投入使用
