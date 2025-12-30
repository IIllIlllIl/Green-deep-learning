# 数据流程完整说明

**日期**: 2025-12-24
**目的**: 解释从 726 行原始数据到 536/594 行分层数据的完整流程

---

## 📊 数据流程总览

```
原始数据 (data.csv)
    726 行（包含11个模型的所有实验）
        ↓
    【第1步：任务组筛选】
    按5/6组任务筛选相关仓库
        ↓
    5组方案: 648 行 | 6组方案: 726 行 ✅
    ├─ image_classification_examples: 219 行
    ├─ image_classification_resnet: 39 行
    ├─ person_reid: 116 行
    ├─ vulberta: 142 行
    ├─ bug_localization: 132 行
    └─ mrt_oast (可选): 78 行

    排除数据（5组方案）:
    ❌ MRT-OAST: 78 行（不在5组任务中）

        ↓
    【第2步：性能指标清洗】
    删除性能指标全缺失的行
        ↓
    5组方案: 536 行 | 6组方案: 594 行 ✅
    ├─ image_classification_examples: 219 行 (0行删除)
    ├─ image_classification_resnet: 39 行 (0行删除)
    ├─ person_reid: 116 行 (0行删除)
    ├─ vulberta: 82 行 (60行删除) ❌
    ├─ bug_localization: 80 行 (52行删除) ❌
    └─ mrt_oast (可选): 58 行 (20行删除) ⚠️

    删除数据（5组方案）:
    ❌ 性能全缺失: 112 行

        ↓
    【第3步（待执行）：超参数填充】
    使用 models_config.json 填充默认值
        ↓
    预期结果:
    ✅ 超参数填充率: 70-90%
    ✅ DiBS分析就绪
```

---

## 🔍 详细数据统计

### 第1步：任务组筛选

| 方案 | 原始数据 | 筛选后 | 排除数据 | 排除原因 |
|------|---------|--------|---------|---------|
| **5组方案** | 726 行 | 648 行 | 78 行 | MRT-OAST 不在5组中 |
| **6组方案** | 726 行 | 726 行 | 0 行 | 包含所有11个模型 ✅ |

**5组任务覆盖的仓库**:
- `examples` (4个模型: mnist, mnist_ff, mnist_rnn, siamese)
- `pytorch_resnet_cifar10` (1个模型: resnet20)
- `Person_reID_baseline_pytorch` (3个模型: densenet121, hrnet18, pcb)
- `VulBERTa` (1个模型: mlp)
- `bug-localization-by-dnn-and-rvsm` (1个模型: default)

**被排除的仓库**（仅在5组方案中）:
- `MRT-OAST` (1个模型: default, 78行数据)

### 第2步：性能指标清洗

**删除规则**: 删除所有性能指标列都为空的行

| 任务组 | 筛选后行数 | 性能全缺失 | 保留行数 | 保留率 |
|-------|----------|-----------|---------|--------|
| image_classification_examples | 219 | 0 | 219 | 100.0% ✅ |
| image_classification_resnet | 39 | 0 | 39 | 100.0% ✅ |
| person_reid | 116 | 0 | 116 | 100.0% ✅ |
| vulberta | 142 | 60 | 82 | 57.7% ⚠️ |
| bug_localization | 132 | 52 | 80 | 60.6% ⚠️ |
| mrt_oast（6组） | 78 | 20 | 58 | 74.4% ⚠️ |
| **5组总计** | 648 | 112 | 536 | 82.7% |
| **6组总计** | 726 | 132 | 594 | 81.8% |

**性能缺失原因分析**:
- VulBERTa: 60行缺失 `perf_eval_loss`（部分实验训练失败或未记录）
- bug_localization: 52行缺失 `perf_top1_accuracy` 和 `perf_top5_accuracy`
- mrt_oast: 20行缺失 `perf_accuracy`, `perf_precision`, `perf_recall`

---

## 📈 数据保留率对比

| 方案 | 原始数据 | 最终有效数据 | 数据缺失 | 保留率 | 推荐度 |
|------|---------|------------|---------|--------|--------|
| **4组方案（旧）** | 726 | 372 | 354 | 51.2% | ⚠️ 保留率低 |
| **5组方案（当前）** | 726 | 536 | 190 | 73.8% | ✅ 显著改进 |
| **6组方案（建议）** | 726 | 594 | 132 | 81.8% | ⭐ 最优方案 |

**改进幅度**:
- 4组 → 5组: +164 行（+44.1%）
- 5组 → 6组: +58 行（+10.8%）
- 4组 → 6组: +222 行（+59.7%）✅

---

## 💡 方案对比与建议

### 方案1: 5组方案（当前）

**优点**:
- ✅ 数据保留率 73.8%（相比旧方案显著改进）
- ✅ 任务组定义清晰（图像分类、ReID、漏洞检测、Bug定位）
- ✅ 超参数冲突已解决（examples vs resnet 分离）

**缺点**:
- ❌ 排除 MRT-OAST（78行，10.7%数据）
- ⚠️ 未覆盖所有11个模型

**适用场景**: 如果只关注主要的4类任务

---

### 方案2: 6组方案（推荐）⭐

**优点**:
- ✅ 数据保留率 81.8%（最高）
- ✅ 覆盖所有11个模型
- ✅ 包含多目标优化任务（MRT-OAST）
- ✅ MRT-OAST 满足所有DiBS分析要求

**缺点**:
- ⚠️ MRT-OAST 性能指标 74.4% 填充率（略低于其他组）
- ⚠️ 需要额外计算资源（+1个任务组）

**适用场景**:
- 完整覆盖所有模型的因果分析
- 研究多目标优化任务的能耗特性

---

## 🚀 实施建议

### 推荐方案：采用6组方案 ⭐

**理由**:
1. **数据利用率最高**: 81.8% vs 73.8%（+8%）
2. **模型覆盖完整**: 11/11 模型 vs 10/11 模型
3. **任务多样性**: 包含多目标优化任务（MRT-OAST）
4. **DiBS可行性**: MRT-OAST 58行数据充足（远超20行要求）

**实施步骤**:
1. 修改 `preprocess_stratified_data.py`，添加 `mrt_oast` 任务组配置
2. 重新运行数据分层脚本
3. 验证6组数据质量
4. 进行DiBS因果分析（6个任务组并行）

**预期成果**:
- 6个分层CSV文件（总计 594 行）
- 覆盖所有11个模型
- 数据保留率达到 81.8%

---

## 📋 数据来源追踪

### 输入文件

| 文件 | 路径 | 行数 | 列数 | 说明 |
|------|------|------|------|------|
| data.csv | `../../results/data.csv` | 726 | 56 | 主项目精简数据 |
| energy_data_original.csv | `../data/energy_research/raw/` | 726 | 56 | analysis 模块本地副本 |
| stage2_mediators.csv | `../data/energy_research/processed.backup_*/` | 726 | 63 | 添加中介变量后的数据 |

**注**: 三个文件的数据行数一致（726行），列数差异因处理阶段不同

### 输出文件（5组方案）

| 任务组 | 文件名 | 行数 | 列数 |
|-------|--------|------|------|
| image_classification_examples | `training_data_image_classification_examples.csv` | 219 | 19 |
| image_classification_resnet | `training_data_image_classification_resnet.csv` | 39 | 15 |
| person_reid | `training_data_person_reid.csv` | 116 | 20 |
| vulberta | `training_data_vulberta.csv` | 82 | 15 |
| bug_localization | `training_data_bug_localization.csv` | 80 | 16 |

### 输出文件（6组方案，待生成）

在5组基础上新增：

| 任务组 | 文件名 | 预期行数 | 预期列数 |
|-------|--------|---------|---------|
| mrt_oast | `training_data_mrt_oast.csv` | 58 | 17 |

---

## ❓ 常见问题解答

### Q1: 为什么是 726 → 648 → 536，而不是 726 → 536？

**A**: 两步筛选过程：
1. **第1步（任务筛选）**: 726 → 648，排除不在5组中的 MRT-OAST（78行）
2. **第2步（质量清洗）**: 648 → 536，删除性能全缺失的行（112行）

### Q2: 为什么 VulBERTa 和 bug_localization 删除这么多行？

**A**: 这两个任务的性能指标缺失率较高：
- VulBERTa: 60/142 (42.3%) 行缺失 `perf_eval_loss`
- bug_localization: 52/132 (39.4%) 行缺失 `perf_top1_accuracy` 和 `perf_top5_accuracy`

可能原因：训练失败、日志提取失败、或实验未完成

### Q3: 6组方案会增加多少计算成本？

**A**:
- DiBS分析时间：每组约30-90分钟
- MRT-OAST（58行）预计：45-60分钟
- 总增加时间：约1小时
- 总计时间（6组）：4-7小时 vs 3-6小时（5组）

### Q4: MRT-OAST 的性能指标是什么？

**A**:
- `perf_accuracy`: 准确率（74.4%填充）
- `perf_precision`: 精确率（74.4%填充）
- `perf_recall`: 召回率（74.4%填充）

这是多目标优化任务的特有指标，与其他组的单一准确率不同。

---

## 📊 附录：完整数据统计

### 原始数据（726行）按仓库分布

| 仓库 | 模型数 | 实验数 | 占比 |
|------|--------|--------|------|
| examples | 4 | 219 | 30.2% |
| pytorch_resnet_cifar10 | 1 | 39 | 5.4% |
| Person_reID_baseline_pytorch | 3 | 116 | 16.0% |
| VulBERTa | 1 | 142 | 19.6% |
| bug-localization-by-dnn-and-rvsm | 1 | 132 | 18.2% |
| MRT-OAST | 1 | 78 | 10.7% |
| **总计** | **11** | **726** | **100%** |

### 有效数据（594行，6组方案）按任务分布

| 任务组 | 实验数 | 占比 |
|-------|--------|------|
| image_classification_examples | 219 | 36.9% |
| person_reid | 116 | 19.5% |
| vulberta | 82 | 13.8% |
| bug_localization | 80 | 13.5% |
| mrt_oast | 58 | 9.8% |
| image_classification_resnet | 39 | 6.6% |
| **总计** | **594** | **100%** |

---

**文档版本**: 1.0
**最后更新**: 2025-12-24
**状态**: 推荐采用6组方案 ⭐
