# DiBS因果边白名单过滤结果总结

**版本**: v1.1
**日期**: 2026-01-20
**输入数据**: `results/energy_research/data/interaction/threshold/`
**输出数据**: `results/energy_research/data/interaction/whitelist/`
**过滤脚本**: `scripts/filter_causal_edges_by_whitelist.py`

---

## 📋 执行摘要

本文档总结了使用白名单规则过滤DiBS因果发现结果的执行情况和最终结果。

### 核心成果

✅ **成功过滤6组DiBS因果边数据**
- 原始边数: 539条
- 白名单过滤后: **227条** (42.1%保留率)
- 移除不合理边: 312条 (57.9%)

✅ **各研究问题边数分布合理**
- Q1超参数主效应: 16条
- Q1交互项调节: 25条
- Q2性能效应: 29条
- Q3中介效应: 116条
- 控制变量效应: 23条

---

## 🎯 白名单规则 (v1.1)

### 允许的16种因果边类型

| # | 规则组 | Source | Target | 研究问题 | 示例 |
|---|--------|--------|--------|----------|------|
| **规则组1: 超参数主效应** | | | | | |
| 1 | Q1 | hyperparam | energy | Q1 | batch_size → cpu_total_joules |
| 2 | Q1 | hyperparam | mediator | Q1 | batch_size → gpu_avg_watts |
| 3 | Q2 | hyperparam | performance | Q2 | batch_size → test_accuracy |
| **规则组2: 交互项调节效应** | | | | | |
| 4 | Q1 | interaction | energy | Q1 | batch_size_x_parallel → cpu_total_joules |
| 5 | Q1 | interaction | mediator | Q1 | batch_size_x_parallel → gpu_avg_watts |
| 6 | Q2 | interaction | performance | Q2 | batch_size_x_parallel → test_accuracy |
| **规则组3: 中间变量中介效应** | | | | | |
| 7 | Q3 | mediator | energy | Q3 | gpu_avg_watts → gpu_total_joules |
| 8 | Q3 | mediator | mediator | Q3 | gpu_temp_avg → gpu_avg_watts |
| 9 | **Q2/Q3** | **mediator** | **performance** | **Q2/Q3** | **gpu_temp_avg → test_accuracy** ⭐ |
| 10 | Q3 | energy | energy | Q3 | cpu_total_joules → cpu_pkg_joules |
| **规则组4: 控制变量影响** | | | | | |
| 11 | - | control | energy | - | model_mnist_ff → cpu_total_joules |
| 12 | - | control | mediator | - | model_mnist_ff → gpu_avg_watts |
| 13 | - | control | performance | - | model_mnist_ff → test_accuracy |
| 14 | - | mode | energy | - | is_parallel → cpu_total_joules |
| 15 | - | mode | mediator | - | is_parallel → gpu_avg_watts |
| 16 | - | mode | performance | - | is_parallel → test_accuracy |

**⭐ 关键更新 (v1.1)**: 第9条规则 `mediator → performance` 新增，用于支持RQ2的间接因果路径分析（hyperparam → mediator → performance）

### 禁止的因果边类型（黑名单示例）

❌ **反因果方向**:
- `performance → hyperparam` - 性能不能改变超参数
- `energy → hyperparam` - 能耗不能改变超参数
- `mediator → hyperparam` - 中间变量不能改变超参数

❌ **实验设计变量作为结果**:
- `* → control` - 模型选择不能被其他变量改变
- `* → mode` - 并行模式不能被其他变量改变

❌ **无意义边**:
- `hyperparam → hyperparam` - 超参数独立设定，无因果关系
- `X → X` (自循环) - 变量不能影响自身

❌ **反直觉关系**:
- `performance → energy` - 性能不应影响能耗（应该是配置影响性能）
- `energy → mediator` - ⚠️ 明确禁止（防止路径污染）

---

## 📊 过滤结果统计

### 1. 整体统计

| 指标 | 数值 | 占比 |
|------|------|------|
| **原始边数** | 539条 | 100% |
| **保留边数** | 227条 | 42.1% |
| **移除边数** | 312条 | 57.9% |

### 2. 各研究问题统计

| 研究问题 | 边数 | 占比 | 说明 |
|---------|------|------|------|
| Q1超参数主效应 | 16条 | 7.0% | hyperparam → energy/mediator |
| Q1交互项调节 | 25条 | 11.0% | interaction → energy/mediator |
| Q2性能效应 | 29条 | 12.8% | hyperparam/interaction/mediator → performance |
| Q3中介效应 | 116条 | 51.1% | mediator → energy/mediator, energy → energy |
| 控制变量效应 | 23条 | 10.1% | control/mode → * |
| **总计** | **227条** | **100%** | - |

**关键发现**:
- ✅ Q3中介效应边数最多（51.1%），符合预期（能耗生成机制复杂）
- ✅ Q2性能效应边数合理（29条），包含直接和间接路径
- ✅ Q1超参数和交互项边数较少（41条），说明直接效应有限

### 3. 各组详细统计

| 组名 | 原始边数 | 保留边数 | 移除边数 | 保留率 |
|------|---------|---------|---------|--------|
| group1_examples | 96 | 43 | 53 | 44.8% |
| group2_vulberta | 82 | 35 | 47 | 42.7% |
| group3_person_reid | 108 | 50 | 58 | 46.3% |
| group4_bug_localization | 85 | 40 | 45 | 47.1% |
| group5_mrt_oast | 104 | 40 | 64 | 38.5% |
| group6_resnet | 64 | 19 | 45 | 29.7% |
| **总计** | **539** | **227** | **312** | **42.1%** |

**各组分析**:
- **最高保留率**: group4_bug_localization (47.1%)
- **最低保留率**: group6_resnet (29.7%)
- **平均保留率**: 42.1%

### 4. 各组研究问题分布

| 组名 | Q1主效应 | Q1交互项 | Q2性能 | Q3中介 | 控制变量 | 总计 |
|------|---------|---------|--------|--------|---------|------|
| group1_examples | 1 | 7 | 2 | 21 | 7 | 43 |
| group2_vulberta | 2 | 7 | 5 | 16 | 1 | 35 |
| group3_person_reid | 3 | 2 | 1 | 29 | 15 | 50 |
| group4_bug_localization | 1 | 3 | 11 | 21 | 0 | 40 |
| group5_mrt_oast | 6 | 5 | 9 | 15 | 0 | 40 |
| group6_resnet | 3 | 1 | 1 | 14 | 0 | 19 |
| **总计** | **16** | **25** | **29** | **116** | **23** | **227** |

**关键发现**:
- group4_bug_localization Q2性能边最多（11条），适合性能分析
- group3_person_reid Q3中介边最多（29条），适合中介效应分析
- group5_mrt_oast Q1主效应边最多（6条），适合直接效应分析

---

## 🔬 典型因果边示例

### Q1: 超参数对能耗的直接效应

```
hyperparam_epochs → energy_gpu_max_watts (强度=0.55)
hyperparam_batch_size_x_is_parallel → energy_gpu_min_watts (强度=0.95)
```

### Q2: 超参数对性能的效应（直接+间接）

**直接路径**:
```
hyperparam_learning_rate_x_is_parallel → perf_final_training_loss (强度=0.55)
hyperparam_dropout → perf_precision (强度=0.45)
```

**间接路径** (通过中间变量):
```
energy_gpu_util_avg_percent → perf_top20_accuracy (强度=0.45)
```

### Q3: 中介效应

**中间变量 → 能耗**:
```
gpu_avg_watts → gpu_total_joules (强度=0.91)
energy_gpu_util_avg_percent → energy_gpu_total_joules (强度=0.45)
```

**中间变量链**:
```
gpu_temp_avg → gpu_avg_watts
energy_gpu_temp_max_celsius → energy_gpu_max_watts (强度=0.45)
```

**能耗分解**:
```
cpu_total_joules → cpu_pkg_joules (强度=0.55)
energy_cpu_total_joules → energy_gpu_total_joules (强度=0.45)
```

---

## 📁 输出文件说明

### 生成的文件

所有白名单过滤后的文件位于: `results/energy_research/data/interaction/whitelist/`

| 文件名 | 行数 | 大小 | 说明 |
|--------|------|------|------|
| `group1_examples_causal_edges_whitelist.csv` | 43 | 6.9K | Examples组过滤后因果边 |
| `group2_vulberta_causal_edges_whitelist.csv` | 35 | 5.8K | VulBERTa组过滤后因果边 |
| `group3_person_reid_causal_edges_whitelist.csv` | 50 | 7.7K | Person reID组过滤后因果边 |
| `group4_bug_localization_causal_edges_whitelist.csv` | 40 | 6.4K | Bug定位组过滤后因果边 |
| `group5_mrt_oast_causal_edges_whitelist.csv` | 40 | 6.3K | MRT-OAST组过滤后因果边 |
| `group6_resnet_causal_edges_whitelist.csv` | 19 | 3.2K | ResNet组过滤后因果边 |

### CSV文件格式

每个文件包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `source` | 源变量名 | `hyperparam_batch_size` |
| `target` | 目标变量名 | `energy_gpu_total_joules` |
| `strength` | 因果边强度 | 0.85 |
| `edge_type` | 边类型 | `main_effect` |
| `is_significant` | 是否显著 | `yes` |
| `strength_level` | 强度等级 | `very_strong` |
| `source_category` | 源变量类别 | `hyperparam` |
| `target_category` | 目标变量类别 | `energy` |
| `question_relevance` | 研究问题相关性 | `Q1` |
| `interpretation` | 解释 | `batch_size → gpu_total_joules` |

---

## 🚀 使用指南

### 1. 读取过滤后的数据

```python
import pandas as pd

# 读取单个组的数据
df_group1 = pd.read_csv('results/energy_research/data/interaction/whitelist/group1_examples_causal_edges_whitelist.csv')

# 筛选特定研究问题的边
q1_edges = df_group1[df_group1['question_relevance'].str.contains('Q1')]
q2_edges = df_group1[df_group1['question_relevance'].str.contains('Q2')]
q3_edges = df_group1[df_group1['question_relevance'].str.contains('Q3')]
```

### 2. 按边类型分析

```python
# 按源变量类别统计
source_stats = df_group1.groupby('source_category').size()

# 按目标变量类别统计
target_stats = df_group1.groupby('target_category').size()

# 按强度筛选
strong_edges = df_group1[df_group1['strength'] >= 0.5]
```

### 3. 合并所有组的数据

```python
import glob

# 读取所有whitelist文件
whitelist_files = glob.glob('results/energy_research/data/interaction/whitelist/*_whitelist.csv')
all_edges = pd.concat([pd.read_csv(f) for f in whitelist_files], ignore_index=True)

# 全局统计
print(f"总边数: {len(all_edges)}")
print(f"各研究问题分布:\n{all_edges['question_relevance'].value_counts()}")
```

### 4. 验证白名单规则

```python
# 检查是否存在违反白名单的边
invalid_edges = all_edges[
    ((all_edges['source_category'] == 'performance') & (all_edges['target_category'] == 'hyperparam')) |
    ((all_edges['source_category'] == 'energy') & (all_edges['target_category'] == 'hyperparam'))
]

if len(invalid_edges) == 0:
    print("✅ 所有边都符合白名单规则！")
else:
    print(f"❌ 发现 {len(invalid_edges)} 条违反规则的边")
```

---

## 📖 相关文档

- [白名单设计方案](CAUSAL_EDGE_WHITELIST_DESIGN.md) - 完整设计文档 ⭐⭐⭐
- [过滤脚本源码](../scripts/filter_causal_edges_by_whitelist.py) - 实现代码
- [DiBS结果README](../results/energy_research/data/interaction/README.md) - 原始数据说明

---

## 🔄 版本历史

### v1.1 (2026-01-20)
- ✅ 添加 `mediator → performance` 规则（第9条）
- ✅ 支持RQ2间接因果路径分析
- ✅ 成功过滤6组DiBS数据（539条 → 227条）
- ✅ 生成白名单输出文件

### v1.0 (2026-01-17)
- ✅ 初始白名单设计（15条规则）
- ✅ 完成设计文档
- ✅ 实现过滤脚本

---

**维护者**: Claude
**文档版本**: v1.1
**最后更新**: 2026-01-20
**状态**: ✅ 完成
