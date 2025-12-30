# 工作总结 - 能耗数据因果分析准备 (2025-12-24)

**日期**: 2025-12-24
**任务**: 为能耗数据因果分析准备高质量训练数据
**状态**: 阶段2完成，已生成后续方案，等待用户确认执行

---

## 📋 已完成工作

### 1. 更新开发规范 (CLAUDE.md)

**修改内容**:
- ✅ 新增"脚本开发规范" ⭐⭐⭐
- ✅ 强制要求：每个脚本必须先编写测试
- ✅ 强制要求：必须先执行Dry Run验证
- ✅ 提供完整开发流程示例

**核心要求**:
```
1. 编写主脚本 → 2. 编写测试脚本 → 3. 执行Dry Run →
4. 检查结果 → 5. 全量执行 → 6. 验证最终结果
```

**文件位置**: `/home/green/energy_dl/nightly/CLAUDE.md` (行9-74)

---

### 2. 创建阶段3-5详细实施方案

**文档**: `analysis/docs/reports/STAGE3_5_DETAILED_PLAN_20251224.md`

**内容概要**:

#### 阶段3：数据分层与保存 (预计1.5小时)

**目标**: 将418行统一数据按4个任务组分层

**输出文件**:
```
analysis/data/energy_research/processed/
├── training_data_image_classification.csv  (~116行×17列)
├── training_data_person_reid.csv          (~69行×17列)
├── training_data_vulberta.csv             (~96行×14列)
└── training_data_bug_localization.csv     (~91行×14列)
```

**关键功能**:
1. **One-Hot编码**:
   - 图像分类: `is_mnist`, `is_cifar10`
   - Person_reID: `is_densenet121`, `is_hrnet18`, `is_pcb`
2. **任务相关列选择**: 只保留该任务的性能指标
3. **删除性能缺失行**: 确保每个任务组的性能指标100%填充

**实现计划**:
- 步骤1: 创建 `preprocess_stratified_data.py` (40分钟)
- 步骤2: 创建 `test_preprocess_stratified_data.py` (20分钟)
- 步骤3: Dry Run验证前20行 (10分钟)
- 步骤4: 全量执行 (5分钟)

**验证清单**:
- ✅ 4个CSV文件成功生成
- ✅ 列数符合预期（14-17列）
- ✅ One-Hot编码正确（互斥，和=1）
- ✅ 无性能全缺失行
- ✅ 相关矩阵可计算

#### 阶段4：数据质量验证 (预计45分钟)

**验证维度**:
1. **缺失率检查**: 超参数/能耗/性能列应0%缺失
2. **完全无缺失行**: 图像分类/Person_reID >90%，VulBERTa/Bug定位 >80%
3. **相关矩阵可计算**: 无nan值，对角线为1
4. **数值范围合理**: 能耗>0，温度20-110°C，功率0-600W
5. **One-Hot编码正确**: 互斥性100%

**实现计划**:
- 步骤1: 创建 `verify_stratified_data_quality.py` (30分钟)
- 步骤2: 运行验证 (5分钟)
- 步骤3: 生成质量报告 (10分钟)

**预期输出**: `STRATIFIED_DATA_QUALITY_REPORT.md` - 完整质量验证报告

#### 阶段5：DiBS因果分析验证 (预计2小时)

**目标**: 验证新数据能够成功发现因果边

**DiBS配置**:
```python
dibs_config = {
    'n_particles': 20,
    'n_steps': 1000,
    'optimizer': 'adam',
    'learning_rate': 0.005,
}
```

**运行模式**:
- 串行运行: 约60分钟（稳定）
- 并行运行: 约20分钟（快速）

**预期结果**:

| 任务组 | 样本数 | 变量数 | 预期因果边数 | 关键因果路径 |
|--------|--------|--------|------------|------------|
| 图像分类 | 116 | 17 | 3-8条 | learning_rate → energy, training_duration → accuracy |
| Person_reID | 69 | 17 | 3-6条 | dropout → mAP, learning_rate → energy |
| VulBERTa | 96 | 14 | 2-5条 | learning_rate → eval_loss, weight_decay → energy |
| Bug定位 | 91 | 14 | 2-5条 | kfold → top1_accuracy, max_iter → energy |

**对比v1.0改进**:

| 维度 | v1.0 (Adult) | v3.0 (能耗分层) | 改进 |
|------|-------------|----------------|------|
| 样本量 | 10个 | 69-116个/任务 | **7-11倍** |
| 因果边数 | **0条** | 预期3-8条/任务 | **质的飞跃** |
| 相关矩阵 | 包含nan（失败） | 完全可计算 | ✅ |
| 数据质量 | 32-100%缺失 | <2%缺失 | ✅ |

**实现计划**:
- 步骤1: 创建 `run_stratified_dibs_analysis.py` (40分钟)
- 步骤2: 创建 `test_stratified_dibs.py` (20分钟)
- 步骤3: Dry Run验证（少量步数） (15分钟)
- 步骤4: 全量执行4个任务组 (60分钟)
- 步骤5: 生成对比报告 (30分钟)

**最终输出**:
- `results/energy_research/task_specific/{task}/causal_graph.npy` - 因果图
- `results/energy_research/task_specific/{task}/causal_edges.pkl` - 因果边
- `DIBS_V1_VS_V3_COMPARISON_REPORT.md` - 对比报告

---

## 📊 整体进度跟踪

### 时间估算

| 阶段 | 预估时间 | 状态 |
|------|---------|------|
| 阶段1 | 30分钟 | ✅ 完成 (2025-12-24) |
| 阶段2 | 2.5小时 | ✅ 完成 (2025-12-24) |
| 阶段3 | 1.5小时 | ⏳ 待执行 |
| 阶段4 | 45分钟 | ⏳ 待执行 |
| 阶段5 | 2小时 | ⏳ 待执行 |
| **总计** | **约7小时** | **40%完成** |

### 待办清单

```markdown
- [x] 阶段1: 验证models_config.json完整性
- [x] 阶段2: 实现数据提取脚本（默认值回溯）
- [x] 更新CLAUDE.md开发规范
- [x] 创建阶段3-5详细方案
- [ ] 阶段3.1: preprocess_stratified_data.py
- [ ] 阶段3.2: test_preprocess_stratified_data.py
- [ ] 阶段3.3: Dry Run通过
- [ ] 阶段3.4: 生成4个分层文件
- [ ] 阶段4.1: verify_stratified_data_quality.py
- [ ] 阶段4.2: 生成质量报告
- [ ] 阶段4.3: 所有验证通过
- [ ] 阶段5.1: run_stratified_dibs_analysis.py
- [ ] 阶段5.2: test_stratified_dibs.py
- [ ] 阶段5.3: Dry Run通过
- [ ] 阶段5.4: 全量执行完成
- [ ] 阶段5.5: 生成对比报告
```

---

## 📝 关键成果文档

### 已创建文档

1. **阶段2执行报告** ⭐⭐⭐
   - 文件: `analysis/docs/reports/STAGE2_EXECUTION_REPORT_20251224.md`
   - 内容: 数据提取完整过程、测试结果、安全性验证

2. **阶段3-5详细方案** ⭐⭐⭐
   - 文件: `analysis/docs/reports/STAGE3_5_DETAILED_PLAN_20251224.md`
   - 内容: 完整实施计划、代码示例、风险应对

3. **开发规范更新** ⭐⭐⭐
   - 文件: `CLAUDE.md` (行9-74)
   - 内容: 测试驱动、Dry Run优先的开发流程

4. **工作总结**
   - 文件: `analysis/docs/reports/WORK_SUMMARY_20251224.md` (本文档)
   - 内容: 已完成工作汇总、进度跟踪

### 已创建脚本

#### 阶段2脚本 (已完成)

1. **extract_from_json_with_defaults.py** (478行)
   - 功能: 从experiment.json提取数据 + models_config.json默认值回溯
   - 核心逻辑:
     - `extract_complete_hyperparams()` - 超参数提取
     - `apply_field_mapping()` - 字段映射
     - `calculate_derived_energy_metrics()` - 派生指标计算
   - 测试: 5/5通过

2. **test_extract_from_json.py** (257行)
   - 测试用例:
     - `test_field_mapping()` - 字段映射测试
     - `test_hyperparam_extraction()` - 超参数提取测试
     - `test_parallel_mode()` - 并行模式测试
     - `test_energy_extraction()` - 能耗提取测试
     - `dry_run_test()` - Dry run测试

3. **verify_extracted_data_safety.py** (268行)
   - 验证维度:
     - 数据来源验证（100%来自实验）
     - 超参数完整性（按仓库分组100%）
     - 能耗数据完整性（100%）
     - 数值范围合理性
     - 无敏感信息、无重复行

#### 阶段2数据 (已生成)

1. **energy_data_extracted_v2.csv**
   - 行数: 418行 (从558个JSON中提取，删除140行能耗/性能缺失)
   - 列数: 34列
   - 数据来源: 100%来自experiment.json + models_config.json
   - 超参数: 100%填充（按仓库分组）
   - 能耗: 100%填充
   - 性能: 82.32%缺失（待阶段3按任务分组解决）

2. **extracted_columns_info.json**
   - 内容: 列信息清单（超参数、能耗、性能、派生指标）

---

## 🔍 关键技术细节

### 1. 默认值回溯机制 ⭐⭐⭐

**问题**: experiment.json只记录变异的超参数，未变异的为空

**解决方案**:
```python
for param_name, param_config in supported_params.items():
    if param_name in recorded_params:
        value = recorded_params[param_name]  # 使用记录值
    else:
        value = param_config.get('default')   # 使用默认值
```

**效果**: Bug定位实验从只记录`{seed: 917}`提取到完整4个超参数

### 2. 字段映射统一 ⭐⭐

**问题**: 不同模型使用不同超参数名

**解决方案**: 8个映射规则

| 原始字段 | 统一字段 | 使用模型 |
|---------|---------|----------|
| `epochs`, `max_iter` | `training_duration` | 所有模型 |
| `weight_decay`, `alpha` | `l2_regularization` | VulBERTa, Bug定位 |
| `learning_rate`, `lr` | `hyperparam_learning_rate` | 所有模型 |

**效果**: 实现跨模型超参数统一，支持分层因果分析

### 3. 派生指标计算 ⭐⭐

**目的**: 探索"超参数 → 中介变量 → 能耗"因果路径

**新增指标**:
- `cpu_pkg_ratio` = CPU能耗 / (CPU + GPU总能耗)
- `gpu_power_fluctuation` = max_watts - min_watts
- `gpu_temp_fluctuation` = temp_max - temp_avg

**效果**: 100%填充，为DiBS提供更多因果路径

### 4. One-Hot编码设计 (待阶段3实施)

**目的**: 避免DiBS将数据集/模型基线差异误判为因果关系

**编码规则**:
```python
# 图像分类
is_mnist = 1 if 'mnist' in model else 0
is_cifar10 = 1 if repo == 'pytorch_resnet_cifar10' else 0

# Person_reID
is_densenet121 = 1 if model == 'densenet121' else 0
is_hrnet18 = 1 if model == 'hrnet18' else 0
is_pcb = 1 if model == 'pcb' else 0
```

**验证**: 互斥性（每行和=1），覆盖所有样本

---

## 🎯 下一步行动

### 用户确认事项

**请用户确认是否执行阶段3-5**:

1. ✅ 阶段3：数据分层与保存（预计1.5小时）
   - 是否立即创建 `preprocess_stratified_data.py`？
   - 是否立即创建测试脚本？

2. ✅ 阶段4：数据质量验证（预计45分钟）
   - 是否立即创建验证脚本？

3. ✅ 阶段5：DiBS因果分析（预计2小时）
   - 是否立即创建分析脚本？
   - 串行运行（稳定，60分钟）还是并行运行（快速，20分钟）？

### 推荐执行顺序

**方案1: 一次性完成**
```
今天完成阶段3 → 阶段4 → 阶段5（约4.5小时）
优势: 快速看到因果分析结果
劣势: 时间较长
```

**方案2: 分阶段执行**
```
今天: 阶段3（1.5小时）
明天: 阶段4 + 阶段5（2.75小时）
优势: 每天时间可控
劣势: 需要两天
```

**方案3: 只执行关键路径**
```
今天: 阶段3（1.5小时）
等待用户确认质量后再执行阶段5
优势: 稳妥，可检查中间结果
劣势: 需要多次确认
```

---

## 📚 参考文档

### 核心文档

1. **STAGE2_EXECUTION_REPORT_20251224.md** - 阶段2完整报告
2. **STAGE3_5_DETAILED_PLAN_20251224.md** - 阶段3-5详细方案
3. **DATA_REEXTRACTION_PROPOSAL_V2_20251224.md** - v2.0完整方案
4. **VARIABLE_EXPANSION_PLAN.md** - v3.0变量扩展方案

### 数据文件

1. **energy_data_extracted_v2.csv** - 阶段2提取数据（418行×34列）
2. **extracted_columns_info.json** - 列信息清单

### 代码文件

1. **extract_from_json_with_defaults.py** - 数据提取脚本
2. **test_extract_from_json.py** - 测试脚本
3. **verify_extracted_data_safety.py** - 安全性验证

---

**总结人**: Claude
**生成时间**: 2025-12-24
**下一步**: 等待用户确认是否执行阶段3-5
