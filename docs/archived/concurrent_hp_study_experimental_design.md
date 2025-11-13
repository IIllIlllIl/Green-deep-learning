# 并发训练超参数影响研究 - 实验设计方案
# Concurrent Training Hyperparameter Impact Study - Experimental Design

**研究日期**: 2025-11-11
**研究目标**: 探索并发训练场景下，超参数对能耗和性能的影响规律

---

## 🎯 研究问题

### 核心问题
**并发训练是否改变超参数-能耗-性能的关系？**

### 具体子问题
1. 在并发场景下，超参数的最优值是否与单独训练不同？
2. 并发训练中，一个模型的超参数变化如何影响另一个模型的能耗和性能？
3. 不同类型的模型组合（高+低显存，互补型GPU利用率等）对超参数敏感性有何影响？
4. 能耗优化的超参数配置与性能优化的配置在并发场景下是否一致？

---

## 🔬 实验设计

### 实验框架

对于每个选定的模型组合 `(ModelA, ModelB)`:

#### 第一阶段：Baseline测试
```
实验1: ModelA(默认) + ModelB(默认) [并发]
实验2: ModelA(默认) [单独]
实验3: ModelB(默认) [单独]
```
**目的**: 建立baseline，测量并发开销

#### 第二阶段：超参数变异测试
对ModelB的关键超参数（learning_rate, epochs）进行边界测试：

```
实验4: ModelA(默认) + ModelB(lr_min) [并发]
实验5: ModelB(lr_min) [单独]

实验6: ModelA(默认) + ModelB(lr_max) [并发]
实验7: ModelB(lr_max) [单独]

实验8: ModelA(默认) + ModelB(epochs_min) [并发]
实验9: ModelB(epochs_min) [单独]

实验10: ModelA(默认) + ModelB(epochs_max) [并发]
实验11: ModelB(epochs_max) [单独]
```

**每个组合总计**: 11次训练
**12个组合总计**: 132次训练

---

### 测量指标

#### 1. 能耗指标
- `gpu_energy_total_joules`: GPU总能耗
- `cpu_energy_total_joules`: CPU总能耗
- `total_energy_joules`: 总能耗
- `energy_per_epoch`: 单epoch能耗
- `energy_efficiency`: 能耗效率(能耗/性能)

#### 2. 性能指标
- 模型特定指标：accuracy, mAP, F1等
- `training_success`: 训练是否成功
- `convergence_speed`: 收敛速度

#### 3. 时间指标
- `duration_seconds`: 训练总时长
- `time_per_epoch`: 单epoch时长
- `concurrent_overhead`: 并发开销 = (并发时长 - 单独时长) / 单独时长

#### 4. 资源利用
- `gpu_util_avg_percent`: 平均GPU利用率
- `gpu_power_avg_watts`: 平均GPU功率
- `gpu_memory_peak`: GPU显存峰值

---

## 📊 模型组合选取方法分析

### 可行组合统计
- **总可行组合**: 120个 (显存≤9GB的模型对)
- **按类型分布**:
  - 低+低: 55个 (46%)
  - 低+中: 30个 (25%)
  - 低+高: 7个 (6%)
  - 中+中: 6个 (5%)
  - 中+高: 4个 (3%)
  - 高+中: 4个 (3%)
  - 高+低: 14个 (12%)

### 方法对比

| 方法 | 优点 | 缺点 | 推荐度 | 适用场景 |
|------|------|------|--------|---------|
| **随机抽样** | 无偏差，简单可重复 | 可能遗漏关键组合 | ⭐⭐⭐ | 初步探索 |
| **分层抽样** | 覆盖全面，代表性强 | 需要分层标准 | ⭐⭐⭐⭐ | 全面研究 |
| **代表性抽样** | 针对性强，可解释性好 | 主观性强 | ⭐⭐⭐⭐ | 深入理解 |
| **正交设计** | 统计效率高，可分析交互 | 设计复杂 | ⭐⭐⭐ | 多因素分析 |
| **实用性抽样** | 稳妥，易分析，成功率高 | 探索性不足 | ⭐⭐⭐⭐⭐ | 资源有限 |

---

## 💡 最终推荐方案

### 混合策略：实用性抽样 + 分层抽样

**组合方式**:
- 6个组合：实用性抽样（优先有数据、安全、互补）
- 6个组合：分层抽样（覆盖不同类型）

**优势**:
- ✅ 实验成功率高（有历史数据作参考）
- ✅ 覆盖不同类型组合
- ✅ 包含互补型和竞争型场景
- ✅ 结果具有泛化性

---

## 📋 推荐的12个模型组合

### 组1-6: 实用性抽样（优先安全+有数据）

#### ⭐ 组1：MNIST + DenseNet121（最佳互补）
```
Model A: examples_mnist (450MB, 12% GPU)
Model B: Person_reID_baseline_pytorch_densenet121 (3300MB, 72% GPU)
```
- **类型**: 低+高
- **显存**: 3750MB (安全)
- **GPU利用率**: 84% (互补⭐)
- **数据状态**: ✅✅ 双方有历史数据
- **特点**: 完美互补，最安全的高显存组合

#### ⭐ 组2：MNIST + MRT-OAST（完美互补）
```
Model A: examples_mnist (450MB, 12% GPU)
Model B: MRT-OAST_default (1950MB, 93% GPU)
```
- **类型**: 低+中
- **显存**: 2400MB (极安全)
- **GPU利用率**: 105% (互补⭐)
- **数据状态**: ✅✅ 双方有历史数据
- **特点**: GPU利用率最优配对

#### ⚠️ 组3：DenseNet121 + MRT-OAST（竞争型）
```
Model A: Person_reID_baseline_pytorch_densenet121 (3300MB, 72% GPU)
Model B: MRT-OAST_default (1950MB, 93% GPU)
```
- **类型**: 高+中
- **显存**: 5250MB (可接受)
- **GPU利用率**: 165% (竞争⚠️)
- **数据状态**: ✅✅ 双方有历史数据
- **特点**: GPU竞争激烈，用于研究竞争场景

#### 组4：MNIST + MNIST RNN（同族模型）
```
Model A: examples_mnist (450MB, 12% GPU)
Model B: examples_mnist_rnn (450MB, 50% GPU)
```
- **类型**: 低+低
- **显存**: 900MB (极安全)
- **GPU利用率**: 62%
- **特点**: 同一领域不同架构

#### 组5：MNIST + MNIST FF（同族模型）
```
Model A: examples_mnist (450MB, 12% GPU)
Model B: examples_mnist_ff (450MB, 50% GPU)
```
- **类型**: 低+低
- **显存**: 900MB (极安全)
- **GPU利用率**: 62%
- **特点**: 同一任务不同架构

#### 组6：MNIST + ResNet20（跨领域）
```
Model A: examples_mnist (450MB, 12% GPU)
Model B: pytorch_resnet_cifar10_resnet20 (720MB, 50% GPU)
```
- **类型**: 低+低
- **显存**: 1170MB (极安全)
- **GPU利用率**: 62%
- **特点**: 图像分类跨数据集

---

### 组7-12: 分层抽样（覆盖不同类型）

#### 组7：ResNet44 + ResNet56（同族扩展）
```
Model A: pytorch_resnet_cifar10_resnet44 (1080MB, 50% GPU)
Model B: pytorch_resnet_cifar10_resnet56 (1350MB, 50% GPU)
```
- **类型**: 低+低
- **显存**: 2430MB
- **特点**: 同架构不同深度

#### 组8：MNIST + VulBERTa CNN（跨领域）
```
Model A: examples_mnist (450MB, 12% GPU)
Model B: VulBERTa_cnn (1350MB, 50% GPU)
```
- **类型**: 低+低
- **显存**: 1800MB
- **特点**: 视觉任务 vs 代码安全

#### 组9：ResNet44 + HRNet18（中型组合）
```
Model A: pytorch_resnet_cifar10_resnet44 (1080MB, 50% GPU)
Model B: Person_reID_baseline_pytorch_hrnet18 (2250MB, 50% GPU)
```
- **类型**: 低+中
- **显存**: 3330MB
- **特点**: 两个中等GPU利用率模型

#### 组10：MNIST + Bug Localization（跨领域）
```
Model A: examples_mnist (450MB, 12% GPU)
Model B: bug-localization-by-dnn-and-rvsm_default (1800MB, 50% GPU)
```
- **类型**: 低+中
- **显存**: 2250MB
- **特点**: 视觉 vs 程序分析

#### 组11：MNIST + HRNet18（互补型）
```
Model A: examples_mnist (450MB, 12% GPU)
Model B: Person_reID_baseline_pytorch_hrnet18 (2250MB, 50% GPU)
```
- **类型**: 低+中
- **显存**: 2700MB
- **特点**: 小模型+中等模型

#### 组12：ResNet44 + DenseNet121（混合型）
```
Model A: pytorch_resnet_cifar10_resnet44 (1080MB, 50% GPU)
Model B: Person_reID_baseline_pytorch_densenet121 (3300MB, 72% GPU)
```
- **类型**: 低+高
- **显存**: 4380MB
- **GPU利用率**: 122%
- **特点**: 跨领域，GPU利用率较高

---

## 🔢 实验规模估算

### 训练次数
- **每个组合**: 11次训练 (1 baseline + 5组对比实验×2)
- **12个组合**: 132次训练

### 时间估算（粗略）

基于历史数据估算：

| 组合类型 | 代表组合 | 预估时长/次 | 11次总时长 |
|---------|---------|-----------|----------|
| 低+低 | MNIST + MNIST系列 | 5分钟 | 55分钟 |
| 低+中 | MNIST + MRT-OAST | 25分钟 | 4.6小时 |
| 低+高 | MNIST + DenseNet121 | 40分钟 | 7.3小时 |
| 中+中 | ResNet44 + HRNet18 | 30分钟 | 5.5小时 |
| 高+中 | DenseNet121 + MRT-OAST | 60分钟 | 11小时 |

**粗略总时长**: 40-50小时（串行）
**并发优化后**: 25-30小时（合理调度）

---

## 📈 数据分析计划

### 1. 并发开销分析
```
并发开销 = (并发训练时长 - 单独训练时长总和) / 单独训练时长总和

按组合类型分析:
- 互补型组合的并发开销
- 竞争型组合的并发开销
- 不同显存组合的并发开销
```

### 2. 超参数敏感性对比
```
敏感性定义: 性能变化% / 超参数变化%

对比:
- 单独训练时的lr敏感性 vs 并发训练时的lr敏感性
- 单独训练时的epochs敏感性 vs 并发训练时的epochs敏感性

假设检验:
H0: 并发训练不改变超参数敏感性
H1: 并发训练显著改变超参数敏感性
```

### 3. 能耗-性能权衡分析
```
能耗效率 = 性能提升 / 能耗增加

对比:
- 单独训练最优能耗效率点
- 并发训练最优能耗效率点

研究问题:
- 并发是否改变最优能耗配置?
- 不同组合类型的能耗优化策略
```

### 4. 模型间干扰分析
```
干扰度 = |ModelB(并发)性能 - ModelB(单独)性能| / ModelB(单独)性能

分析维度:
- 不同超参数配置下的干扰度
- 互补型 vs 竞争型的干扰差异
- 显存占用与干扰的关系
```

---

## 🎯 预期研究成果

### 学术贡献
1. **首次系统研究**并发训练场景下的超参数-能耗-性能关系
2. **量化分析**不同模型组合类型对超参数敏感性的影响
3. **提出指导原则**用于并发训练的超参数优化

### 实用价值
1. **优化建议**：哪些模型组合适合并发，哪些不适合
2. **配置策略**：并发场景下的推荐超参数配置
3. **能耗节省**：如何在并发场景下优化能耗

### 可能发现
- ✅ 互补型组合可能对超参数更鲁棒
- ✅ 竞争型组合可能需要更保守的超参数
- ✅ 最优能耗配置在并发场景可能更倾向于较小的学习率
- ❓ 并发可能改变最优超参数的值（待验证）

---

## 📝 实施建议

### 阶段1：Pilot实验（2-3个组合）
**目的**: 验证实验设计可行性

**选择组合**:
1. MNIST + DenseNet121 (最佳互补)
2. MNIST + MRT-OAST (完美互补)
3. DenseNet121 + MRT-OAST (竞争型)

**预计时间**: 20-25小时
**预期产出**: 实验流程优化，初步发现

### 阶段2：全量实验（所有12个组合）
**目的**: 获得完整数据

**执行方式**:
- 使用screen后台运行
- 合理安排并发，避免资源冲突
- 定期检查实验进度

**预计时间**: 25-30小时（并发优化后）

### 阶段3：数据分析与论文撰写
**分析工具**: Python (pandas, matplotlib, scipy)
**可视化**: 对比图、热力图、散点图
**统计检验**: t-test, ANOVA

---

## 🔚 附录

### A. 超参数边界值定义（调整后范围）

| 超参数 | 表达式 | 说明 |
|--------|-------|------|
| **epochs** | `[default×0.5, default×1.5]` | 半倍到1.5倍 |
| **learning_rate** | `[default×0.2, default×5.0]` | 0.2倍到5倍 |

### B. 实验配置文件格式

```json
{
  "experiment_name": "concurrent_hp_study_pair1",
  "concurrent_mode": true,
  "model_pairs": [
    {
      "model_a": {"repo": "examples", "model": "mnist", "hp": "default"},
      "model_b": {"repo": "Person_reID", "model": "densenet121", "hp": "default"}
    }
  ]
}
```

### C. 相关文件

- **分析脚本**: `scripts/analyze_model_pair_sampling.py`
- **选取结果**: `model_pair_sampling_analysis.json`
- **并发可行性**: `docs/concurrent_training_feasibility_report.md`
- **范围调整文档**: `docs/mutation_ranges_conservative_adjustment.md`

---

**文档生成**: 2025-11-11
**版本**: 1.0
**审核状态**: 待审核
