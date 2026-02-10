# DiBS数据清理与配置修改方案

**版本**: v1.2
**日期**: 2026-02-09
**状态**: ✅ 已验证，待实施
**作者**: Energy DL Project

---

## 执行摘要

本文档提出了DiBS因果分析的完整修改方案，包括：

**A部分：DiBS配置修改（匹配CTF论文）**
- 切换到`MarginalDiBS`变体
- 增加粒子数和步数
- 使用CTF验证的参数值

**B部分：数据清理**
- 删除每组中不适用（全0值）的超参数和模型列
- 修正超参数语义合并（`max_iter` → `epochs`，`alpha` → `l2_regularization`）

**最新进展**：
- ✅ 代码验证完成（bug-localization超参数含义确认）
- ✅ CTF源码验证完成（DiBS配置确认）
- ✅ 同行评审完成
- ⏳ 待实施代码修改

---

## 1. 问题分析

### 1.1 当前数据状态

在`6groups_dibs_ready/`中，存在大量全0值的列：

| 列名 | 类型 | Group1 | Group2 | Group3 | Group4 | Group5 | Group6 | 说明 |
|------|------|--------|--------|--------|--------|--------|--------|------|
| `hyperparam_batch_size` | 超参数 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | 仅examples使用 |
| `hyperparam_dropout` | 超参数 | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | Person_reID/MRT-OAST使用 |
| `hyperparam_alpha` | 超参数 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | Bug定位使用（实为L2） |
| `hyperparam_kfold` | 超参数 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | Bug定位使用 |
| `hyperparam_max_iter` | 超参数 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | Bug定位使用（语义=epochs） |
| `hyperparam_l2_regularization` | 超参数 | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | VulBERTa/MRT-OAST/ResNet使用 |
| `model_hrnet18` | 模型 | ✗ | ✗ | ✓* | ✗ | ✗ | ✗ | *部分0值 |
| `model_mnist_ff` | 模型 | ✓* | ✗ | ✗ | ✗ | ✗ | ✗ | *部分0值 |
| `model_mnist_rnn` | 模型 | ✓* | ✗ | ✗ | ✗ | ✗ | ✗ | *部分0值 |
| `model_pcb` | 模型 | ✗ | ✗ | ✓* | ✗ | ✗ | ✗ | *部分0值 |
| `model_siamese` | 模型 | ✓* | ✗ | ✗ | ✗ | ✗ | ✗ | *部分0值 |

注：✓=有变化，✗=全0，✓*=部分0值

### 1.2 数据来源追溯

```
data/data.csv → generate_6groups_final.py → 6groups_final/
                                          → create_global_standardized_data.py → 6groups_global_std/
                                                                          → preprocess_for_dibs_global_std.py → 6groups_dibs_ready/
```

**关键发现**：
- `6groups_global_std/`中保留了结构性NaN（正确的设计）
- `preprocess_for_dibs_global_std.py`将全NaN填充为0（符合DiBS要求）
- 这些0值表示"该变量不适用于此组"

### 1.3 超参数语义合并问题

根据`models_config.json`和`generate_6groups_final.py`：

| 原始超参数 | 语义 | 当前状态 | 应该合并为 | 验证状态 |
|-----------|------|---------|-----------|---------|
| `epochs` | 训练迭代次数 | 保留 | 保持 | - |
| `max_iter` | 训练迭代次数（bug-localization） | 保留 | `hyperparam_epochs` | ✅ 已验证 |
| `alpha` | L2正则化（bug-localization） | 保留 | `hyperparam_l2_regularization` | ✅ 已验证 |
| `weight_decay` | L2正则化 | 保留 | `hyperparam_l2_regularization` | 已实现 |
| `kfold` | K折交叉验证 | 保留 | 独立保留 | - |

#### 代码验证结果（2026-02-09）

**验证1：`max_iter`语义**

根据`repos/bug-localization-by-dnn-and-rvsm/train_wrapper.py`:
```python
parser.add_argument('--max_iter', type=int, default=10000,
                    help='Maximum iterations')
```

文档说明：`--max_iter N` - 最大迭代次数（默认: 10000）

**结论**：`max_iter`确实表示训练迭代次数，与其他组的`epochs`语义相同。✅ 可以合并。

**验证2：`alpha`语义**

根据`repos/bug-localization-by-dnn-and-rvsm/train.sh`:
```
--alpha ALPHA                 L2 penalty parameter (default: 1e-5)
```

根据`train_wrapper.py`:
```python
parser.add_argument('--alpha', type=float, default=1e-5,
                    help='L2 penalty parameter')
```

**结论**：`alpha`确实是L2正则化参数，与其他组的`l2_regularization`/`weight_decay`语义相同。✅ 可以合并。

**当前实现**：
- `generate_6groups_final.py`只合并了`alpha`和`weight_decay`
- `max_iter`未被合并到`epochs`
- 结果：Group4使用`max_iter`，其他组使用`epochs`，语义相同但列名不同

### 1.4 CTF源码验证结果（2026-02-09）

**验证方法**：查阅`CTF_original/src/discovery.py`源码

| 参数 | CTF值 | 源码位置 | 验证状态 |
|------|-------|---------|---------|
| variant | `MarginalDiBS` | L142 | ✅ 已确认 |
| n_particles | 50 | L148 | ✅ 已确认 |
| n_steps | 13000 | L148 | ✅ 已确认 |
| likelihood | `BGe` | L141-142 | ✅ 已确认 |
| edge_threshold | 0.99 | L193 | ✅ 已确认 |
| alpha_linear | 默认（~1.0） | 未显式设置 | ✅ 已确认 |
| beta_linear | 默认（~1.0） | 未显式设置 | ✅ 已确认 |
| interv_mask | 有（标记METHOD_COLUMNS） | L128-135 | ✅ 已确认 |

**源码证据**：

1. **DiBS变体**（L142）：
```python
dibs = MarginalDiBS(x=collected_data, interv_mask=interv_mask, inference_model=model)
```

2. **采样参数**（L148）：
```python
gs = dibs.sample(key=subk, n_particles=50, steps=13000, callback_every=1000, callback=dibs.visualize_callback())
```

3. **边阈值**（L193）：
```python
dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=0.99)
```

4. **似然模型**（L141-142）：
```python
model = BGe(graph_dist=model_graph)
dibs = MarginalDiBS(x=collected_data, interv_mask=interv_mask, inference_model=model)
```

---

## 2. DiBS配置修改方案

### 2.1 目标

将DiBS配置修改为匹配CTF论文的设置，以提高结果可比性和收敛质量。

### 2.2 配置对比

| 参数 | 当前值 | CTF值 | 目标值 | 变更说明 |
|------|-------|-------|--------|---------|
| variant | `JointDiBS` | `MarginalDiBS` | `MarginalDiBS` | 切换变体 |
| n_particles | 20 | 50 | 50 | 增加粒子数 |
| n_steps | 5000 | 13000 | 13000 | 增加训练步数 |
| alpha_linear | 0.05 | ~1.0（默认） | **1.0** | 修正：CTF使用默认值 |
| beta_linear | 0.1 | ~1.0（默认） | 1.0 | 恢复默认无环约束 |
| likelihood | `LinearGaussian` | `BGe` | `LinearGaussian` | 保持：结构性0值不适合BGe |
| edge_threshold | 0.3 | 0.99 | 0.3 | 保持：完成后手动尝试不同阈值 |

### 2.3 参数说明

**alpha_linear（修正）**：
- 之前错误设定为0.0
- 正确值：1.0（MarginalDiBS默认值）
- 作用：控制DAG惩罚的基础强度

**beta_linear**：
- 当前0.1是经过参数调优的最优值
- CTF使用默认值1.0
- 影响：1.0会加强无环约束，可能减少边数

**关于edge_threshold**：
- CTF使用0.99提取因果边（非常严格）
- 我们使用0.3（更宽松）
- 策略：保持0.3运行，完成后手动尝试不同阈值（0.5、0.7、0.99）

### 2.4 计算成本预估

| 配置 | 单组时间 | 6组总时间 |
|------|---------|----------|
| 当前（20×5000） | ~10分钟 | ~1小时 |
| CTF风格（50×13000） | ~65分钟 | ~6.5小时 |

---

## 3. 数据清理方案

### 3.1 目标

1. **删除不适用列**：每组只保留该组实际使用的超参数和模型列
2. **语义超参数合并**：`max_iter` → `epochs`，`alpha` → `l2_regularization`
3. **保持跨组可比性**：共享相同语义的超参数使用相同的列名

### 3.2 各组保留的列

#### Group1 (examples)
**保留的超参数**：
- `hyperparam_batch_size` (100%有效)
- `hyperparam_epochs` (100%有效)
- `hyperparam_learning_rate` (100%有效)
- `hyperparam_seed` (100%有效)

**保留的模型**：
- `model_mnist_ff`
- `model_mnist_rnn`
- `model_siamese`

**删除的全0列**：
- `hyperparam_dropout`, `hyperparam_alpha`, `hyperparam_kfold`, `hyperparam_max_iter`, `hyperparam_l2_regularization`
- `model_hrnet18`, `model_pcb`

#### Group2 (VulBERTa)
**保留的超参数**：
- `hyperparam_epochs`
- `hyperparam_learning_rate`
- `hyperparam_l2_regularization`
- `hyperparam_seed`

**删除的全0列**：
- 所有`model_*`列
- `batch_size`, `dropout`, `alpha`, `kfold`, `max_iter`

#### Group3 (Person_reID)
**保留的超参数**：
- `hyperparam_dropout`
- `hyperparam_epochs`
- `hyperparam_learning_rate`
- `hyperparam_seed`

**保留的模型**：
- `model_hrnet18`
- `model_pcb`

#### Group4 (bug-localization)
**保留的超参数**：
- `hyperparam_alpha` (L2正则化) → 重命名为`hyperparam_l2_regularization`
- `hyperparam_kfold`
- `hyperparam_max_iter` → 重命名为`hyperparam_epochs`
- `hyperparam_seed`

**关键变更**：
- `hyperparam_alpha` → `hyperparam_l2_regularization`
- `hyperparam_max_iter` → `hyperparam_epochs`

#### Group5 (MRT-OAST)
**保留的超参数**：
- `hyperparam_dropout`
- `hyperparam_epochs`
- `hyperparam_l2_regularization`
- `hyperparam_learning_rate`
- `hyperparam_seed`

#### Group6 (ResNet)
**保留的超参数**：
- `hyperparam_epochs`
- `hyperparam_learning_rate`
- `hyperparam_l2_regularization`
- `hyperparam_seed`

### 3.3 共有超参数（跨组可比）

清理后，以下超参数在多组中可比较：

| 超参数 | 可用组 | 说明 |
|--------|-------|------|
| `hyperparam_epochs` | 全部6组 | 统一训练迭代次数 |
| `hyperparam_learning_rate` | 全部6组 | 学习率 |
| `hyperparam_seed` | 全部6组 | 随机种子 |
| `hyperparam_l2_regularization` | Group2,4,5,6 | L2正则化 |
| `hyperparam_batch_size` | Group1 | 批量大小 |
| `hyperparam_dropout` | Group3,5 | Dropout率 |
| `hyperparam_kfold` | Group4 | K折交叉验证 |

---

## 4. 实施计划

### 4.1 阶段划分

```
阶段1：数据清理（优先，独立实施）
  ├─ 备份v1数据
  ├─ 修改预处理脚本（超参数合并 + 删除全0列）
  ├─ 生成v2数据
  └─ 单组验证

阶段2：DiBS配置修改
  ├─ 修改CausalGraphLearner支持MarginalDiBS
  ├─ 更新配置参数
  └─ 单组测试

阶段3：全面运行
  ├─ 运行所有6组
  ├─ 对比v1/v2数据结果
  └─ 更新下游分析
```

### 4.2 修改内容清单

**A. 数据清理**（`scripts/preprocess_for_dibs_global_std.py`）：
1. 删除全0列函数
2. Group4超参数重命名（`max_iter`→`epochs`, `alpha`→`l2_regularization`）
3. 删除原始超参数列（避免重复）

**B. DiBS配置**（`scripts/run_dibs_6groups_global_std.py`）：
1. 更新OPTIMAL_CONFIG：
   - 添加`variant: "MarginalDiBS"`
   - `n_particles: 50`
   - `n_steps: 13000`
   - `alpha_linear: 1.0`（修正）
   - `beta_linear: 1.0`

**C. DiBS学习器**（`utils/causal_discovery.py`）：
1. 支持`MarginalDiBS`变体
2. 添加`variant`参数

### 4.3 验证检查点

**数据清理验证**：
- [ ] 各组无全0列（除了真正不需要的列）
- [ ] Group4的`max_iter`正确重命名为`epochs`
- [ ] Group4的`alpha`正确重命名为`l2_regularization`
- [ ] 共有超参数列名一致

**DiBS配置验证**：
- [ ] MarginalDiBS成功初始化
- [ ] 单组测试运行成功
- [ ] 边数量在合理范围
- [ ] 运行时间可接受

---

## 5. 预期效果

### 5.1 数据质量提升

| 指标 | 清理前 | 清理后 |
|------|-------|-------|
| Group1列数 | 35 | ~23 |
| Group2列数 | 37 | ~24 |
| Group3列数 | 37 | ~25 |
| Group4列数 | 38 | ~24 |
| Group5列数 | 37 | ~26 |
| Group6列数 | 36 | ~24 |
| 全0列数 | 15列在所有组全0 | 0 |

### 5.2 DiBS计算影响

| 方面 | 当前配置 | CTF风格配置 |
|------|---------|-------------|
| 单组时间 | ~10分钟 | ~65分钟 |
| 6组总时间 | ~1小时 | ~6.5小时 |
| 变量数减少 | - | 减少30%（数据清理后） |
| 收敛质量 | 当前 | 预期提升 |

### 5.3 跨组可比性

- `epochs`在所有组中统一（包括Group4的`max_iter`）
- `l2_regularization`在Group2/4/5/6中统一
- 共有超参数可直接跨组对比

---

## 6. 风险与缓解

### 6.1 风险

1. **计算成本**：6.5小时运行时间
2. **历史结果不一致**：新数据生成的DiBS结果与之前不同
3. **下游脚本兼容性**：依赖旧列名的脚本可能失效
4. **beta_linear变化**：从0.1→1.0可能减少边数

### 6.2 缓解措施

1. **分阶段实施**：先数据清理，后DiBS配置修改
2. **保留旧数据**：将`6groups_dibs_ready/`备份为`v1`
3. **版本控制**：新数据命名为`v2`
4. **单组测试**：正式运行前用group1测试
5. **文档更新**：更新所有相关文档

---

## 7. 同行评审结论（2026-02-09）

### 7.1 评审概述

本方案经过严谨科研同行评审和CTF源码验证，评审范围包括：
1. CTF论文配置验证
2. DiBS配置参数正确性
3. 超参数语义合并的科学合理性
4. 实施风险和缓解措施

### 7.2 评审结果

| 评审项 | 结论 | 说明 |
|--------|------|------|
| **超参数语义合并** | ✅ 建议执行 | 代码验证通过，语义确认为等价 |
| **interv_mask方案** | ❌ 不建议执行 | 理论问题：观测性数据≠干预数据 |
| **整体可行性** | ✅ 可行 | 需做好备份和验证 |

### 7.3 关键建议

1. **✅ 执行超参数合并**：
   - `hyperparam_max_iter` → `hyperparam_epochs`
   - `hyperparam_alpha` → `hyperparam_l2_regularization`
   - 删除合并后的原始列（避免重复）

2. **❌ 不使用interv_mask**：
   - 超参数是实验设置变量，不是干预变量
   - 错误使用会导致DiBS似然计算错误
   - 如果需要表达先验知识，应使用edge_prior机制

3. **风险缓解**：
   - 保留原始数据作为备份
   - 分阶段实施，对比结果变化
   - 详细记录所有修改

### 7.4 关于interv_mask的详细说明

**为什么不能将超参数标记为干预？**

根据因果推断理论，"干预"（intervention）需要满足：
1. 外部强制改变变量的值
2. 切断该变量到其因变量的因果边
3. 变量值服从干预分布而非条件分布

**我们的数据特征**：
- 超参数虽然由实验者设定，但不是随机分配的
- 存在混杂因素（如模型类型、数据集大小）
- 缺乏随机化实验设计

**CTF论文中的interv_mask**：
- 标记的是**公平性方法**（FairMask, Fairway等）
- 这些方法被研究者**主动应用**到数据中以改变模型行为
- 这符合干预的定义

**结论**：我们的数据是**观测性研究**，不应使用interv_mask。

---

## 8. 后续步骤

1. **✅ 方案讨论完成**（2026-02-09）
2. **✅ CTF源码验证完成**（2026-02-09）
3. **⏳ 代码实现**：修改预处理脚本和DiBS配置
4. **⏳ 数据生成**：重新生成DiBS就绪数据
5. **⏳ 验证测试**：运行DiBS和ATE计算，验证结果
6. **⏳ 文档更新**：更新所有相关文档

---

## 变更历史

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2026-02-09 | 初始版本 | Energy DL Project |
| v1.1 | 2026-02-09 | 添加代码验证结果、同行评审结论 | Energy DL Project |
| v1.2 | 2026-02-09 | 添加CTF源码验证，修正alpha_linear错误值，添加DiBS配置修改 | Energy DL Project |

---

## 附录A：各组完整列清单

### Group1 (examples) - 保留列
```
能耗列 (11个):
energy_cpu_pkg_joules, energy_cpu_ram_joules, energy_cpu_total_joules,
energy_gpu_avg_watts, energy_gpu_max_watts, energy_gpu_min_watts,
energy_gpu_temp_avg_celsius, energy_gpu_temp_max_celsius,
energy_gpu_total_joules, energy_gpu_util_avg_percent, energy_gpu_util_max_percent

控制变量 (1个):
is_parallel

超参数 (4个):
hyperparam_batch_size, hyperparam_epochs, hyperparam_learning_rate, hyperparam_seed

模型 (3个):
model_mnist_ff, model_mnist_rnn, model_siamese

性能 (1个):
perf_test_accuracy

交互项 (2个):
hyperparam_batch_size_x_is_parallel, hyperparam_epochs_x_is_parallel,
hyperparam_learning_rate_x_is_parallel
```

### Group2 (VulBERTa) - 保留列
```
能耗列 (11个): [同上]
is_parallel

超参数 (4个):
hyperparam_epochs, hyperparam_learning_rate, hyperparam_l2_regularization, hyperparam_seed

性能 (3个):
perf_eval_loss, perf_final_training_loss, perf_eval_samples_per_second

交互项 (3个):
hyperparam_epochs_x_is_parallel, hyperparam_learning_rate_x_is_parallel,
hyperparam_l2_regularization_x_is_parallel
```

### Group3 (Person_reID) - 保留列
```
超参数 (4个):
hyperparam_dropout, hyperparam_epochs, hyperparam_learning_rate, hyperparam_seed

模型 (2个):
model_hrnet18, model_pcb

性能 (3个):
perf_map, perf_rank1, perf_rank5

交互项 (3个):
hyperparam_dropout_x_is_parallel, hyperparam_epochs_x_is_parallel,
hyperparam_learning_rate_x_is_parallel
```

### Group4 (bug-localization) - 保留列（带重命名）
```
超参数 (4个):
hyperparam_epochs (原max_iter), hyperparam_kfold,
hyperparam_l2_regularization (原alpha), hyperparam_seed

性能 (4个):
perf_top1_accuracy, perf_top5_accuracy, perf_top10_accuracy, perf_top20_accuracy

交互项 (3个):
hyperparam_epochs_x_is_parallel, hyperparam_kfold_x_is_parallel,
hyperparam_l2_regularization_x_is_parallel
```

### Group5 (MRT-OAST) - 保留列
```
超参数 (5个):
hyperparam_dropout, hyperparam_epochs, hyperparam_l2_regularization,
hyperparam_learning_rate, hyperparam_seed

性能 (3个):
perf_accuracy, perf_precision, perf_recall

交互项 (4个):
hyperparam_dropout_x_is_parallel, hyperparam_epochs_x_is_parallel,
hyperparam_l2_regularization_x_is_parallel, hyperparam_learning_rate_x_is_parallel
```

### Group6 (ResNet) - 保留列
```
超参数 (4个):
hyperparam_epochs, hyperparam_learning_rate, hyperparam_l2_regularization, hyperparam_seed

性能 (2个):
perf_best_val_accuracy, perf_test_accuracy

交互项 (3个):
hyperparam_epochs_x_is_parallel, hyperparam_learning_rate_x_is_parallel,
hyperparam_l2_regularization_x_is_parallel
```

---

## 变更历史

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2026-02-09 | 初始版本 | Energy DL Project |
