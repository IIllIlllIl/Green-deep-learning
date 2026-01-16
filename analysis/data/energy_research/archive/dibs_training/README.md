# DiBS训练数据集 - 文档索引

**目录**: `/home/green/energy_dl/nightly/analysis/data/energy_research/dibs_training/`

**生成日期**: 2026-01-15

**数据版本**: v1.0

---

## 快速开始 ⚡

1. **快速了解数据质量** → 阅读 [`QUICK_QUALITY_SUMMARY.md`](QUICK_QUALITY_SUMMARY.md)
2. **选择研究问题对应的数据** → 查看 [`USAGE_GUIDE.md`](USAGE_GUIDE.md)
3. **查看详细评估结果** → 参考 [`DATA_QUALITY_ASSESSMENT_20260115.md`](DATA_QUALITY_ASSESSMENT_20260115.md)

---

## 文件清单

### 数据文件（CSV）

| 文件名 | 样本数 | 特征数 | 超参数 | DiBS就绪 | 推荐用途 |
|--------|--------|--------|--------|----------|---------|
| [`group1_examples.csv`](group1_examples.csv) | 126 | 18 | 4 | ✅ | 问题1/2/3（最佳） |
| [`group2_vulberta.csv`](group2_vulberta.csv) | 52 | 16 | 0 | ⚠️ | 问题2（需清理） |
| [`group3_person_reid.csv`](group3_person_reid.csv) | 118 | 19 | 4 | ✅ | 问题1/2/3（最佳） |
| [`group4_bug_localization.csv`](group4_bug_localization.csv) | 40 | 17 | 0 | ✅ | 问题2 |
| [`group5_mrt_oast.csv`](group5_mrt_oast.csv) | 46 | 16 | 0 | ✅ | 问题2 |
| [`group6_resnet.csv`](group6_resnet.csv) | 41 | 18 | 4 | ✅ | 问题1/2 |

**说明**:
- **问题1**: 超参数对能耗的影响（需要超参数）
- **问题2**: 能耗和性能的权衡关系（所有组可用）
- **问题3**: 中间变量的中介效应（需要超参数，推荐高质量组）

### 文档文件

| 文件名 | 类型 | 用途 |
|--------|------|------|
| [`README.md`](README.md) | 索引 | 本文件，文档导航 |
| [`QUICK_QUALITY_SUMMARY.md`](QUICK_QUALITY_SUMMARY.md) | 总结 | **快速质量总结**（推荐优先阅读） ⭐ |
| [`USAGE_GUIDE.md`](USAGE_GUIDE.md) | 指南 | **使用指南**（代码示例） ⭐ |
| [`DATA_QUALITY_ASSESSMENT_20260115.md`](DATA_QUALITY_ASSESSMENT_20260115.md) | 报告 | 详细质量评估报告 |

### 元数据文件

| 文件名 | 类型 | 用途 |
|--------|------|------|
| [`generation_stats.json`](generation_stats.json) | JSON | 数据生成统计 |
| [`detailed_quality_analysis.json`](detailed_quality_analysis.json) | JSON | 详细质量分析结果 |
| [`analyze_dibs_data_quality.py`](analyze_dibs_data_quality.py) | Python | 质量评估脚本 |

---

## 数据集总览

### 总体统计

- **总组数**: 6组
- **总样本数**: 423个
- **DiBS就绪组**: 5/6 (83.3%)
- **包含超参数的组**: 3/6 (50%)
- **数据完整性**: 100%（零缺失值）

### 数据质量评级

| 评级 | 组数 | 组名 |
|------|------|------|
| 优秀 | 2 | examples, Person_reID |
| 良好 | 3 | pytorch_resnet, bug-localization, MRT-OAST |
| 需要清理 | 1 | VulBERTa |

### 特征覆盖

**所有组包含的特征**（一致性100%）:
- **能耗指标**: 11个（CPU能耗、GPU功耗、温度、利用率）
- **控制变量**: duration_seconds（所有组）, num_mutated_params（部分组）

**各组特有的特征**:
- **超参数**: 0-4个（因组而异）
- **性能指标**: 1-4个（因任务类型而异）

---

## 研究问题映射

### 问题1: 超参数对能耗的影响

**适用组**: 3个（305样本）

| 组名 | 样本数 | 超参数 | 推荐等级 |
|------|--------|--------|---------|
| examples | 126 | batch_size, epochs, learning_rate, seed | ⭐⭐⭐ 最佳 |
| Person_reID | 118 | dropout, epochs, learning_rate, seed | ⭐⭐⭐ 最佳 |
| pytorch_resnet | 41 | epochs, learning_rate, seed, weight_decay | ⭐⭐ 良好 |

**关键特征**:
- 输入: hyperparam_* (4个超参数)
- 输出: energy_* (11个能耗指标)
- 控制: duration_seconds

### 问题2: 能耗和性能的权衡关系

**适用组**: 5个（377样本）

| 组名 | 样本数 | 能耗指标 | 性能指标 | 推荐等级 |
|------|--------|---------|---------|---------|
| examples | 126 | 11 | 1 | ⭐⭐⭐ |
| Person_reID | 118 | 11 | 3 | ⭐⭐⭐ |
| pytorch_resnet | 41 | 11 | 2 | ⭐⭐ |
| MRT-OAST | 46 | 11 | 3 | ⭐⭐ |
| bug-localization | 40 | 11 | 4 | ⭐⭐ |

**关键特征**:
- 输入: energy_* (11个能耗指标)
- 输出: perf_* (1-4个性能指标)
- 控制: duration_seconds, 超参数（如果有）

### 问题3: 中间变量的中介效应

**适用组**: 2-3个（推荐高质量组）

| 组名 | 样本数 | 因果路径 | 推荐等级 |
|------|--------|---------|---------|
| examples | 126 | 4超参数 → 11能耗 → 1性能 | ⭐⭐⭐ 最佳 |
| Person_reID | 118 | 4超参数 → 11能耗 → 3性能 | ⭐⭐⭐ 最佳 |
| pytorch_resnet | 41 | 4超参数 → 11能耗 → 2性能 | ⭐⭐ 良好 |

**关键特征**:
- 输入: hyperparam_* (超参数)
- 中介: energy_* (能耗指标)
- 输出: perf_* (性能指标)

---

## 使用建议

### 立即可用（无需处理）

**推荐工作流**:

1. **加载数据**
   ```python
   import pandas as pd
   df = pd.read_csv('group1_examples.csv')  # 最佳组
   ```

2. **标准化**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
   ```

3. **输入DiBS**
   ```python
   # X = df_scaled.values
   # 运行DiBS因果图学习
   ```

### 需要简单清理

**VulBERTa组**（如果使用）:
```python
df = pd.read_csv('group2_vulberta.csv')
df_clean = df.drop(columns=['energy_gpu_util_max_percent'])  # 移除常数特征
```

### 小样本组稳定性

**对于样本量<50的组**（pytorch_resnet, bug-localization, MRT-OAST）:
- 使用k-fold交叉验证
- 或bootstrap评估不确定性
- 或优先使用大样本组

---

## 数据特征详解

### 超参数特征（hyperparam_*）

**examples组（4个）**:
- `hyperparam_batch_size`: 批量大小 [19, 10000]
- `hyperparam_epochs`: 训练轮数 [5, 15]
- `hyperparam_learning_rate`: 学习率 [0.0056, 0.0184]
- `hyperparam_seed`: 随机种子 [1, 9809]

**Person_reID组（4个）**:
- `hyperparam_dropout`: Dropout率 [0.30, 0.59]
- `hyperparam_epochs`: 训练轮数 [31, 90]
- `hyperparam_learning_rate`: 学习率 [0.025, 0.075]
- `hyperparam_seed`: 随机种子 [3, 9974]

**pytorch_resnet组（4个）**:
- `hyperparam_epochs`: 训练轮数 [108, 297]
- `hyperparam_learning_rate`: 学习率 [0.051, 0.135]
- `hyperparam_seed`: 随机种子 [409, 9992]
- `hyperparam_weight_decay`: 权重衰减 [0.000011, 0.00066]

### 能耗特征（energy_*）（所有组一致，11个）

**CPU能耗**（3个）:
- `energy_cpu_pkg_joules`: CPU包能耗（焦耳）
- `energy_cpu_ram_joules`: CPU内存能耗（焦耳）
- `energy_cpu_total_joules`: CPU总能耗（焦耳）

**GPU功耗**（4个）:
- `energy_gpu_avg_watts`: GPU平均功率（瓦特）
- `energy_gpu_max_watts`: GPU最大功率（瓦特）
- `energy_gpu_min_watts`: GPU最小功率（瓦特）
- `energy_gpu_total_joules`: GPU总能耗（焦耳）

**GPU温度**（2个）:
- `energy_gpu_temp_avg_celsius`: GPU平均温度（摄氏度）
- `energy_gpu_temp_max_celsius`: GPU最大温度（摄氏度）

**GPU利用率**（2个）:
- `energy_gpu_util_avg_percent`: GPU平均利用率（百分比）
- `energy_gpu_util_max_percent`: GPU最大利用率（百分比）

### 性能特征（perf_*）（各组不同）

**examples组（1个）**:
- `perf_test_accuracy`: 测试准确率

**Person_reID组（3个）**:
- `perf_map`: 平均精度均值（Mean Average Precision）
- `perf_rank1`: Rank-1准确率
- `perf_rank5`: Rank-5准确率

**pytorch_resnet组（2个）**:
- `perf_best_val_accuracy`: 最佳验证准确率
- `perf_test_accuracy`: 测试准确率

**VulBERTa组（3个）**:
- `perf_eval_loss`: 评估损失
- `perf_final_training_loss`: 最终训练损失
- `perf_eval_samples_per_second`: 评估样本速率

**bug-localization组（4个）**:
- `perf_top1_accuracy`: Top-1准确率
- `perf_top5_accuracy`: Top-5准确率
- `perf_top10_accuracy`: Top-10准确率
- `perf_top20_accuracy`: Top-20准确率

**MRT-OAST组（3个）**:
- `perf_accuracy`: 准确率
- `perf_precision`: 精确率
- `perf_recall`: 召回率

### 控制变量（2个）

- `duration_seconds`: 训练持续时间（秒）（所有组）
- `num_mutated_params`: 变异参数数量（部分组）

---

## 关键发现

### ✅ 优势

1. **数据完整性优秀**: 100%零缺失值
2. **样本量充分**: 5/6组满足DiBS要求（≥30）
3. **超参数多样性高**: 3组包含4个超参数，唯一值比率高达80-95%
4. **特征覆盖全面**: 16-19特征/组，包含超参数、能耗、性能
5. **任务类型多样**: 图像分类、行人重识别、缺陷定位等

### ⚠️ 限制

1. **3组缺少超参数**: VulBERTa, bug-localization, MRT-OAST
   - 只能分析能耗-性能关系（问题2）
   - 无法研究超参数影响（问题1/3）

2. **3组样本量<50**: pytorch_resnet(41), bug-localization(40), MRT-OAST(46)
   - 满足DiBS最低要求但略低于推荐值
   - 建议使用交叉验证或bootstrap

3. **1组需要清理**: VulBERTa
   - 存在1个常数特征需要移除

---

## 常见问题

### Q: 哪个组最适合开始DiBS分析？

**A**: **examples组（group1_examples.csv）**
- 样本量最大（126）
- 4个超参数，多样性高
- 数据质量最优，零缺失
- 可用于所有3个研究问题

### Q: 可以合并多个组的数据吗？

**A**: 可以，但需要注意：
- 不同组的性能指标不同，需要对齐或分组分析
- 超参数也可能不同，需要处理缺失值
- 推荐方法：分组分析或只使用公共特征（能耗指标）

### Q: 样本量40-46够用吗？

**A**: 满足DiBS最低要求（≥30），但略低于推荐值（50+）
- 建议使用k-fold交叉验证评估稳定性
- 或优先使用大样本组（examples: 126, Person_reID: 118）

### Q: VulBERTa组可以用吗？

**A**: 可以，但需要先移除常数特征：
```python
df = pd.read_csv('group2_vulberta.csv')
df_clean = df.drop(columns=['energy_gpu_util_max_percent'])
```
注意：该组无超参数，只能用于问题2（能耗-性能关系）

### Q: 如何处理异常值？

**A**: DiBS对异常值相对鲁棒，通常不需要特殊处理
- 但如果异常值率>20%，可考虑使用RobustScaler
- 或Winsorization截断极端值
- 或保持原样，让DiBS自动处理

---

## 后续工作

### 推荐下一步

1. **开始DiBS分析**: 使用examples组作为pilot study
2. **验证因果发现**: 在其他组上复现结果
3. **比较任务类型**: 分析不同任务的因果结构差异
4. **扩展分析**: 探索更复杂的因果关系和中介效应

### 可选扩展

1. **增加样本量**: 对于<50样本的组
2. **补充超参数**: 对于无超参数的组
3. **清理VulBERTa**: 移除常数特征后可用于问题2

---

## 相关链接

- **原始数据**: `/home/green/energy_dl/nightly/data/data.csv`
- **分析脚本**: `/home/green/energy_dl/nightly/analysis/scripts/`
- **DiBS实现**: `/home/green/energy_dl/nightly/analysis/utils/dibs_wrapper.py`

---

## 版本历史

- **v1.0** (2026-01-15): 初始版本
  - 生成6组DiBS训练数据
  - 完成质量评估
  - 创建文档

---

**生成时间**: 2026-01-15
**维护者**: Green
**联系方式**: green@example.com

---

## 快速参考卡片

```
┌─────────────────────────────────────────────────────────────┐
│ DiBS数据质量快速参考                                           │
├─────────────────────────────────────────────────────────────┤
│ 总组数:       6                                               │
│ 总样本数:     423                                             │
│ DiBS就绪:     5/6 (83.3%)                                    │
│ 数据完整性:   100% (零缺失)                                    │
├─────────────────────────────────────────────────────────────┤
│ 推荐组:                                                       │
│   • examples (126样本, 4超参数) ⭐⭐⭐ 最佳                    │
│   • Person_reID (118样本, 4超参数) ⭐⭐⭐ 最佳                 │
│   • pytorch_resnet (41样本, 4超参数) ⭐⭐ 良好                │
├─────────────────────────────────────────────────────────────┤
│ 研究问题:                                                     │
│   问题1 (超参数影响):   3组 (305样本)                         │
│   问题2 (能耗-性能):    5组 (377样本)                         │
│   问题3 (中介效应):     2组 (244样本)                         │
└─────────────────────────────────────────────────────────────┘
```

---

**开始使用**: 请阅读 [`USAGE_GUIDE.md`](USAGE_GUIDE.md) 获取代码示例
