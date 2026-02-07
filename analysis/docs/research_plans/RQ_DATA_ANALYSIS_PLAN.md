# RQ数据分析方案

**版本**: v2.0
**日期**: 2026-02-07
**状态**: 数据验证完成

---

## 一、概述

本文档描述从已有因果分析结果生成RQ1-RQ3图表和表格的具体数据处理方案。

### 核心原则

| 问题 | RQa（主效应） | RQb（调节效应） |
|-----|-------------|---------------|
| **分析对象** | 原始超参数（`hyperparam_*`，排除交互项） | 交互项（`hyperparam_*_x_is_parallel`） |
| **数据源** | 全局标准化数据 | 全局标准化数据（含交互项） |
| **因果解释** | 超参数→结果的直接因果效应 | 并行模式如何调节超参数效应 |

### 变量分类（v2.0）

| 类别 | 变量 | 说明 |
|-----|------|------|
| **hyperparam** | `hyperparam_batch_size`, `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`, `hyperparam_dropout`, `hyperparam_l2_regularization`, `hyperparam_alpha`, `hyperparam_kfold`, `hyperparam_max_iter` | 处理变量 |
| **interaction** | `hyperparam_*_x_is_parallel` | 交互项调节变量 |
| **mediator** | `energy_gpu_temp_avg_celsius`, `energy_gpu_temp_max_celsius`, `energy_gpu_util_avg_percent`, `energy_gpu_util_max_percent` | 中间变量（仅物理状态变量） |
| **energy** | `energy_cpu_pkg_joules`, `energy_cpu_ram_joules`, `energy_cpu_total_joules`, `energy_gpu_total_joules`, `energy_gpu_avg_watts`, `energy_gpu_min_watts`, `energy_gpu_max_watts` | 能耗结果变量（含功率） |
| **performance** | `perf_test_accuracy`, `perf_map`, `perf_rank1`, `perf_rank5`, `perf_precision`, `perf_recall`, `perf_eval_samples_per_second`, `perf_top1_accuracy`, `perf_top5_accuracy`, `perf_top10_accuracy`, `perf_top20_accuracy` | 性能结果变量 |
| **control** | `model_mnist_ff`, `model_mnist_rnn`, `model_siamese`, `model_hrnet18`, `model_pcb` | 控制变量 |
| **mode** | `is_parallel` | 分层变量 |

**修订说明**: 功率变量(`energy_gpu_*_watts`)从mediator改为energy类别。

---

## 二、数据源清单与验证

### 2.1 数据源状态

| RQ | 数据文件 | 状态 | 说明 |
|----|---------|------|------|
| RQ1 | `results/energy_research/data/global_std_dibs_ate/*.csv` | ✅ 正确 | 全局标准化DiBS+ATE，6组共151条边 |
| RQ2 | 需从全局标准化因果图重新生成 | ⚠️ 需生成 | 现有间接路径不含交互项 |
| RQ3 | `results/energy_research/tradeoff_detection_global_std/` | ✅ 正确 | 来自global_std_dibs_ate，61条权衡 |

### 2.2 数据可用性统计

| 边类型 | 边数 | 覆盖组 |
|-------|------|-------|
| hyperparam → energy | 4 | group1, group3, group5 |
| hyperparam → performance | 3 | group1, group5 |
| interaction → energy | 7 | group1, group3 |
| interaction → performance | 6 | group1, group2, group4, group5 |

### 2.3 RQ2间接路径数据

**问题**: 现有`dibs_indirect_paths.csv`不包含交互项作为source（759条路径，均为原始hyperparam）

**解决方案**: 从全局标准化因果图重新生成间接路径
- 因果图位置: `results/energy_research/archived_data/global_std/group*/`
- 每组包含: 因果图矩阵 + 特征名（含8个交互项）
- 使用脚本: `scripts/convert_dibs_to_csv.py::generate_all_paths()`

---

## 三、RQ1：超参数对能耗的影响

### 3.1 RQ1a：主效应分析

#### 输入数据
- 文件: `results/energy_research/data/global_std_dibs_ate/group*_dibs_global_std_ate.csv` (6组)
- 字段: `source`, `target`, `strength`, `ate_global_std`, `ate_global_std_ci_lower`, `ate_global_std_ci_upper`, `ate_global_std_is_significant`

#### 过滤条件
```python
# 1. source为原始超参数（排除交互项）
source.startswith('hyperparam_') and '_x_is_parallel' not in source

# 2. target为能耗变量
target in ['energy_cpu_pkg_joules', 'energy_cpu_ram_joules',
           'energy_cpu_total_joules', 'energy_gpu_total_joules',
           'energy_gpu_avg_watts', 'energy_gpu_min_watts', 'energy_gpu_max_watts']

# 3. 边强度阈值
strength >= 0.3
```

#### 数据可用性
符合条件的边: **4条**
- group1: `learning_rate → gpu_min_watts` (ATE=-0.0021, 显著)
- group3: `epochs → gpu_min_watts` (ATE=-0.1267, 显著)
- group3: `epochs → gpu_total_joules` (ATE=+0.2579, 显著)
- group5: `epochs → gpu_max_watts` (ATE=0, 不显著)

#### 输出表格

**表1: 超参数对能耗的直接因果影响（RQ1a）**

| 超参数 | GPU_min_watts | GPU_max_watts | GPU_total | 其他 |
|-------|---------------|---------------|-----------|------|
| learning_rate | n=1, -0.0021 | — | — | — |
| epochs | n=1, -0.1267 | n=1, 0.0 | n=1, +0.2579 | — |

#### 图表

**图1: 超参数-能耗因果效应柱状图**
- 类型: 柱状图（因数据稀疏，改用柱状图替代热力图）
- X轴: 超参数-能耗组合
- Y轴: ATE
- 误差线: 95% CI
- 格式: PDF, 10×6 inch

---

### 3.2 RQ1b：调节效应分析

#### 过滤条件
```python
# 1. source为交互项
'_x_is_parallel' in source

# 2. target为能耗变量
target in energy_variables

# 3. 边强度阈值
strength >= 0.3
```

#### 数据可用性
符合条件的边: **7条**
- group1: `batch_size×parallel → gpu_avg_watts` (+0.0933)
- group1: `batch_size×parallel → gpu_min_watts` (-0.3399)
- group1: `learning_rate×parallel → gpu_avg_watts` (-0.0066)
- group1: `epochs×parallel → gpu_min_watts` (-0.1468)
- group3: `epochs×parallel → gpu_min_watts` (-0.0942)
- group3: `epochs×parallel → gpu_total_joules` (+0.4814)
- group3: `dropout×parallel → gpu_min_watts` (+0.3025)

#### 输出表格

**表2: 并行模式调节效应（RQ1b）**

| 交互项 | 能耗变量 | ATE | 95% CI | 调节方向 | 组 |
|-------|---------|-----|--------|---------|-----|
| batch_size × parallel | gpu_avg_watts | +0.093 | [...] | 并行增强 | group1 |
| batch_size × parallel | gpu_min_watts | -0.340 | [...] | 并行减弱 | group1 |
| epochs × parallel | gpu_min_watts | -0.147 | [...] | 并行减弱 | group1 |
| epochs × parallel | gpu_min_watts | -0.094 | [...] | 并行减弱 | group3 |
| epochs × parallel | gpu_total_joules | +0.481 | [...] | 并行增强 | group3 |
| dropout × parallel | gpu_min_watts | +0.303 | [...] | 并行增强 | group3 |
| learning_rate × parallel | gpu_avg_watts | -0.007 | [...] | 并行减弱 | group1 |

#### 图表

**图2: 交互项调节效应柱状图**
- 类型: 分组柱状图
- X轴: 交互项
- Y轴: ATE
- 分面: 按组
- 颜色: 调节方向（增强=红, 减弱=蓝）
- 格式: PDF, 12×6 inch

---

## 四、RQ2：中间变量解释

### 4.1 数据生成

**需要执行**: 从全局标准化因果图生成包含交互项的间接路径

```bash
# 使用现有脚本生成
conda activate causal-research
python scripts/convert_dibs_to_csv.py \
    --input-dir results/energy_research/archived_data/global_std \
    --output-dir results/energy_research/rq_analysis/indirect_paths \
    --min-strength 0.3
```

**生成逻辑**:
- 使用`generate_all_paths()`函数
- `is_key_variable()`已包含交互项判断
- 输出2-hop和3-hop间接路径

### 4.2 RQ2分析（主效应+调节效应统一）

#### 过滤条件

**主效应路径**:
```python
source.startswith('hyperparam_') and '_x_is_parallel' not in source
mediator1 in ['energy_gpu_temp_avg_celsius', 'energy_gpu_temp_max_celsius',
              'energy_gpu_util_avg_percent', 'energy_gpu_util_max_percent']
target in energy_variables
```

**调节效应路径**:
```python
'_x_is_parallel' in source
mediator1 in mediator_variables
target in energy_variables
```

#### 输出表格

**表3: 中介变量频率统计**

| 中间变量 | 主效应路径数 | 调节效应路径数 | 涉及超参数 | 涉及能耗 |
|---------|-------------|---------------|-----------|---------|
| gpu_temp_avg | ? | ? | ... | ... |
| gpu_temp_max | ? | ? | ... | ... |
| gpu_util_avg | ? | ? | ... | ... |
| gpu_util_max | ? | ? | ... | ... |

**表4: 主要因果路径**

| 路径 | 类型 | 组覆盖 | 路径强度 |
|-----|------|-------|---------|
| hyperparam → mediator → energy | 主效应 | ?/6 | min(s1,s2) |
| interaction → mediator → energy | 调节效应 | ?/6 | min(s1,s2) |

#### 图表

**图3: 因果路径Sankey图**
- 类型: Sankey图
- 分面: 主效应 vs 调节效应
- 格式: PDF, 12×8 inch

**图4: 6组中介模式热力图**
- 类型: 热力图
- X轴: 中介变量
- Y轴: 组
- 颜色: 使用频率
- 格式: PDF, 8×6 inch

---

## 五、RQ3：能耗与性能权衡

### 5.1 数据说明

RQ3不区分a/b，统一分析主效应和调节效应的权衡，重点关注场景差异。

#### 输入数据
- 文件: `results/energy_research/tradeoff_detection_global_std/tradeoff_detailed_global_std.csv`
- 来源: ✅ 已验证来自`global_std_dibs_ate`
- 记录数: 61条权衡，其中34条涉及交互项

#### 数据可用性

**能耗vs性能权衡**: 仅**1条**
- group1: `batch_size×parallel` → (gpu_min_watts vs perf_test_accuracy)

**原因分析**:
1. 原始超参数同时对能耗和性能有显著直接效应的情况不存在
   - group1: learning_rate→energy, epochs→perf（不同超参数）
   - group3: epochs→energy（无→perf边）
   - 其他组: 无hyperparam直接边

2. 交互项的能耗vs性能权衡也很少
   - 仅group1的batch_size×parallel同时指向energy和perf

**这是真实的数据发现**: 在当前数据集中，超参数调整导致的能耗-性能权衡不显著。

### 5.2 输出表格

**表5: 权衡关系汇总**

| 干预变量类型 | 总权衡数 | 能耗vs性能 | 能耗vs能耗 | 其他 |
|-------------|---------|-----------|-----------|------|
| 原始超参数 | 1 | 0 | 0 | 1 |
| 交互项 | 34 | 1 | 多 | 多 |
| 其他 | 26 | 0 | 多 | 多 |

**表6: 能耗vs性能权衡详情**

| 干预变量 | 能耗变量 | 性能变量 | ATE_能耗 | ATE_性能 | 权衡方向 | 组 |
|---------|---------|---------|---------|---------|---------|-----|
| batch_size×parallel | gpu_min_watts | test_accuracy | -0.340 | -0.164 | 同向减少 | group1 |

### 5.3 图表

**图5: 权衡散点图**
- 类型: 散点图
- X轴: ATE_metric1
- Y轴: ATE_metric2
- 颜色: intervention类型（超参数/交互项/其他）
- 格式: PDF, 8×6 inch

**图6: 6组权衡数量分布**
- 类型: 堆叠柱状图
- X轴: 组
- Y轴: 权衡数量
- 颜色: intervention类型
- 格式: PDF, 10×6 inch

---

## 六、实现规范

### 6.1 待执行任务

| 优先级 | 任务 | 说明 |
|-------|------|------|
| 1 | 生成间接路径数据 | 从全局标准化因果图生成，包含交互项 |
| 2 | RQ1分析+图表 | 数据已就绪 |
| 3 | RQ3分析+图表 | 数据已就绪 |
| 4 | RQ2分析+图表 | 依赖任务1 |

### 6.2 复用代码

| 功能 | 来源 | 函数 |
|-----|------|------|
| 变量分类 | `scripts/convert_dibs_to_csv.py` | `get_variable_category()` |
| 白名单规则 | `scripts/filter_causal_edges_by_whitelist.py` | `WHITELIST_RULES` |
| 路径提取 | `scripts/convert_dibs_to_csv.py` | `generate_all_paths()`, `is_key_variable()` |

### 6.3 输出规范

| 输出类型 | 格式 | 位置 |
|---------|------|------|
| 表格 | CSV + LaTeX | `results/energy_research/rq_analysis/tables/` |
| 图表 | PDF | `results/energy_research/rq_analysis/figures/` |
| 间接路径 | CSV | `results/energy_research/rq_analysis/indirect_paths/` |

### 6.4 图表样式

| 属性 | 规范 |
|-----|------|
| 字体 | Times New Roman, 12pt |
| DPI | 300 |
| 格式 | PDF |
| 配色 | 红蓝色盲友好配色 |

---

## 七、图表清单

| 图编号 | 类型 | RQ | 说明 | 尺寸 |
|-------|------|-----|------|------|
| 图1 | 柱状图 | RQ1a | 超参数主效应（4条边） | 10×6 |
| 图2 | 分组柱状图 | RQ1b | 交互项调节效应（7条边） | 12×6 |
| 图3 | Sankey图 | RQ2 | 因果路径流向 | 12×8 |
| 图4 | 热力图 | RQ2 | 6组中介模式 | 8×6 |
| 图5 | 散点图 | RQ3 | 权衡分布 | 8×6 |
| 图6 | 堆叠柱状图 | RQ3 | 6组权衡数量 | 10×6 |

---

## 八、表格清单

| 表编号 | RQ | 说明 |
|-------|-----|------|
| 表1 | RQ1a | 超参数对能耗的直接因果影响 |
| 表2 | RQ1b | 并行模式调节效应 |
| 表3 | RQ2 | 中介变量频率统计 |
| 表4 | RQ2 | 主要因果路径 |
| 表5 | RQ3 | 权衡关系汇总 |
| 表6 | RQ3 | 能耗vs性能权衡详情 |

---

## 九、修订历史

| 版本 | 日期 | 修订内容 |
|-----|------|---------|
| v1.0 | 2026-02-07 | 初始版本 |
| v2.0 | 2026-02-07 | 数据验证完成：1)确认RQ1/RQ3数据源正确 2)确认RQ2需重新生成间接路径 3)更新数据可用性统计 4)RQ3不分a/b统一分析 5)简化图表清单 |

---

**文档状态**: 数据验证完成，待实施
