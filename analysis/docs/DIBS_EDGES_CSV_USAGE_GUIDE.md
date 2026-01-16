# DiBS因果边CSV文件使用指南

**创建日期**: 2026-01-16
**数据来源**: DiBS因果分析结果
**文件位置**: `results/energy_research/dibs_edges_csv/`

---

## 📁 文件列表

### 1. `dibs_direct_edges.csv` - 直接因果边 (18KB, 114条)

**包含的因果关系类型**:
- 超参数 → 能耗 (57条)
- 性能指标 → 能耗 (46条)
- 超参数 → 性能 (11条)

**列说明**:
| 列名 | 类型 | 说明 |
|------|------|------|
| `task_group` | str | 任务组ID (如 group1_examples) |
| `task_name` | str | 任务组名称 (如 examples（图像分类-小型）) |
| `edge_type` | str | 边类型 (hyperparam_to_energy, performance_to_energy, etc.) |
| `source` | str | 源变量名 |
| `target` | str | 目标变量名 |
| `strength` | float | 边强度 (0-1之间，阈值0.3) |
| `research_question` | str | 对应的研究问题 (Q1_direct, Q2_tradeoff, etc.) |
| `source_type` | str | 源变量类型 (hyperparam, performance, energy) |
| `target_type` | str | 目标变量类型 (energy, performance) |

**使用示例**:
```python
import pandas as pd

# 读取直接因果边
edges = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_direct_edges.csv')

# 筛选超参数→能耗的边
hp_to_energy = edges[edges['edge_type'] == 'hyperparam_to_energy']

# 查找最强的10条边
top_edges = edges.nlargest(10, 'strength')

# 查看特定任务组的边
group1_edges = edges[edges['task_group'] == 'group1_examples']

# 统计各类边的数量
edge_type_counts = edges['edge_type'].value_counts()
```

---

### 2. `dibs_indirect_paths.csv` - 间接因果路径 (234KB, 759条)

**包含的路径类型**:
- 超参数 → 中介 → 能耗 (266条)
- 性能 → 中介 → 能耗 (200条)
- 多步路径 (≥4节点) (278条)
- 超参数 → 中介 → 性能 (15条)

**列说明**:
| 列名 | 类型 | 说明 |
|------|------|------|
| `task_group` | str | 任务组ID |
| `task_name` | str | 任务组名称 |
| `path_type` | str | 路径类型 (hyperparam_mediator_energy, multi_step, etc.) |
| `source` | str | 起始变量名 |
| `mediator1` | str | 第一个中介变量 |
| `mediator2` | str/None | 第二个中介变量 (多步路径有值) |
| `target` | str | 目标变量名 |
| `strength_step1` | float | 第一步边强度 |
| `strength_step2` | float | 第二步边强度 |
| `strength_step3` | float/None | 第三步边强度 (多步路径有值) |
| `indirect_strength` | float | 间接效应强度 (各步强度的乘积) |
| `num_steps` | int | 路径步数 (2或3) |
| `research_question` | str | 对应的研究问题 |
| `path_description` | str | 路径描述 (如 "A → B → C") |
| `mediation_type` | str/None | 中介类型 (partial/full，仅问题3有) |
| `direct_strength` | float/None | 直接效应强度 (仅问题3有) |

**使用示例**:
```python
import pandas as pd

# 读取间接因果路径
paths = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_indirect_paths.csv')

# 筛选超参数→中介→能耗的路径
hp_mediated = paths[paths['path_type'] == 'hyperparam_mediator_energy']

# 查找最强的间接效应
top_paths = paths.nlargest(10, 'indirect_strength')

# 筛选多步路径
multi_step = paths[paths['num_steps'] == 3]

# 筛选完全中介路径
full_mediation = paths[paths['mediation_type'] == 'full']

# 统计不同中介变量的使用次数
mediator_counts = paths['mediator1'].value_counts()
```

---

### 3. `dibs_all_edges_summary.csv` - 汇总统计 (1.3KB, 6行)

**包含信息**:
- 每个任务组的样本数、特征数
- 变量分类统计 (超参数、性能、能耗、中介变量数量)
- 提取的边数统计 (直接边、间接路径)
- 研究问题证据统计 (Q1/Q2/Q3各项证据数量)
- DiBS运行时间和图统计

**列说明**:
| 列名 | 类型 | 说明 |
|------|------|------|
| `task_group` | str | 任务组ID |
| `task_name` | str | 任务组名称 |
| `n_samples` | int | 样本数 |
| `n_features` | int | 特征数 |
| `n_hyperparams` | int | 超参数数量 |
| `n_performance` | int | 性能指标数量 |
| `n_energy` | int | 能耗指标数量 |
| `n_mediators` | int | 中介变量数量 |
| `direct_edges_extracted` | int | 提取的直接边数 |
| `indirect_paths_extracted` | int | 提取的间接路径数 |
| `q1_direct_hp_to_energy` | int | 问题1直接边数 |
| `q1_mediated_hp_to_energy` | int | 问题1中介路径数 |
| `q2_perf_to_energy` | int | 问题2性能→能耗边数 |
| `q2_energy_to_perf` | int | 问题2能耗→性能边数 |
| `q2_common_hyperparams` | int | 问题2共同超参数数 |
| `q2_mediated_tradeoffs` | int | 问题2中介权衡路径数 |
| `q3_mediation_to_energy` | int | 问题3能耗中介路径数 |
| `q3_mediation_to_perf` | int | 问题3性能中介路径数 |
| `q3_multi_step_paths` | int | 问题3多步路径数 |
| `total_edges` | int | 总因果关系数 |
| `elapsed_time_minutes` | float | DiBS运行时间 (分钟) |
| `graph_max` | float | 因果图最大值 |
| `graph_mean` | float | 因果图平均值 |
| `strong_edges_gt_0.3` | int | 强边数 (>0.3) |
| `total_edges_gt_0.01` | int | 总边数 (>0.01) |

**使用示例**:
```python
import pandas as pd

# 读取汇总统计
summary = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_all_edges_summary.csv')

# 查看所有任务组的边数
print(summary[['task_name', 'direct_edges_extracted', 'indirect_paths_extracted', 'total_edges']])

# 计算总统计
total_direct = summary['direct_edges_extracted'].sum()
total_indirect = summary['indirect_paths_extracted'].sum()
print(f"总直接边: {total_direct}条")
print(f"总间接路径: {total_indirect}条")

# 查看运行效率
print(summary[['task_name', 'n_samples', 'elapsed_time_minutes']])
```

---

## 🎯 按研究问题使用

### 研究问题1: 超参数对能耗的影响

**直接效应**:
```python
# 从直接边中筛选
q1_direct = edges[edges['research_question'] == 'Q1_direct']

# 按超参数分组统计
hp_effects = q1_direct.groupby('source')['strength'].agg(['count', 'mean', 'max'])
```

**间接效应 (中介路径)**:
```python
# 从间接路径中筛选
q1_mediated = paths[paths['research_question'].str.contains('Q1')]

# 分析中介变量的作用
mediator_analysis = q1_mediated.groupby('mediator1').agg({
    'indirect_strength': ['count', 'mean', 'max']
})
```

---

### 研究问题2: 能耗-性能权衡关系

**直接权衡**:
```python
# 性能→能耗的边
perf_to_energy = edges[edges['edge_type'] == 'performance_to_energy']

# 能耗→性能的边 (应该很少或没有)
energy_to_perf = edges[edges['edge_type'] == 'energy_to_performance']
```

**共同超参数**:
```python
# 同时影响能耗和性能的超参数
common_hp_edges = edges[edges['research_question'] == 'Q2_common_hyperparam']

# 分析哪些超参数造成权衡
hp_analysis = common_hp_edges.groupby('source').agg({
    'target': 'count',
    'strength': 'mean'
})
```

**中介权衡路径**:
```python
# 性能→中介→能耗的路径
mediated_tradeoffs = paths[paths['path_type'] == 'performance_mediator_energy']

# 找到最强的权衡路径
top_tradeoffs = mediated_tradeoffs.nlargest(10, 'indirect_strength')
```

---

### 研究问题3: 中介效应路径

**中介效应分析**:
```python
# 超参数→中介→能耗
mediation_energy = paths[paths['research_question'] == 'Q3_mediation_energy']

# 区分完全中介和部分中介
full_med = mediation_energy[mediation_energy['mediation_type'] == 'full']
partial_med = mediation_energy[mediation_energy['mediation_type'] == 'partial']

# 计算中介比例
mediation_energy['mediation_ratio'] = (
    mediation_energy['indirect_strength'] /
    (mediation_energy['indirect_strength'] + mediation_energy['direct_strength'])
)
```

**多步路径**:
```python
# 筛选多步路径
multi_step = paths[paths['num_steps'] == 3]

# 分析复杂的因果链
complex_chains = multi_step.nlargest(20, 'indirect_strength')
```

---

## 📊 常见分析任务

### 1. 找到最重要的超参数

```python
import pandas as pd

edges = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_direct_edges.csv')
paths = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_indirect_paths.csv')

# 直接效应
direct_hp = edges[edges['source_type'] == 'hyperparam'].groupby('source').agg({
    'strength': ['count', 'mean', 'sum']
}).round(3)

# 间接效应
indirect_hp = paths[paths['source'].str.startswith('hyperparam_')].groupby('source').agg({
    'indirect_strength': ['count', 'mean', 'sum']
}).round(3)

# 合并分析
print("最重要的超参数 (直接效应):")
print(direct_hp.sort_values(('strength', 'sum'), ascending=False).head(10))
```

### 2. 分析中介变量的作用

```python
# 统计每个中介变量出现的频率
mediator_freq = paths['mediator1'].value_counts()

# 分析中介效应大小
mediator_strength = paths.groupby('mediator1').agg({
    'indirect_strength': ['count', 'mean', 'max', 'sum']
}).round(3)

print("最重要的中介变量:")
print(mediator_strength.sort_values(('indirect_strength', 'sum'), ascending=False).head(10))
```

### 3. 跨任务组对比

```python
# 按任务组统计边数
edges_by_group = edges.groupby(['task_group', 'edge_type']).size().unstack(fill_value=0)
paths_by_group = paths.groupby(['task_group', 'path_type']).size().unstack(fill_value=0)

# 可视化对比
import matplotlib.pyplot as plt
edges_by_group.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('不同任务组的因果边分布')
plt.xlabel('任务组')
plt.ylabel('边数')
plt.legend(title='边类型', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('edges_comparison.png')
```

### 4. 提取特定因果关系

```python
# 例如: learning_rate → 能耗
lr_to_energy = edges[
    (edges['source'] == 'hyperparam_learning_rate') &
    (edges['target'].str.contains('energy'))
]

# 或者使用路径
lr_paths = paths[paths['source'] == 'hyperparam_learning_rate']

print(f"learning_rate的直接能耗效应: {len(lr_to_energy)}条")
print(f"learning_rate的间接能耗效应: {len(lr_paths)}条")
```

---

## 🔧 脚本使用

### 重新生成CSV文件

```bash
# 使用默认阈值 (0.3)
python3 scripts/extract_dibs_edges_to_csv.py

# 使用自定义阈值
python3 scripts/extract_dibs_edges_to_csv.py --threshold 0.5

# 指定结果目录
python3 scripts/extract_dibs_edges_to_csv.py \
  --result-dir results/energy_research/dibs_6groups_final/20260116_004323 \
  --threshold 0.3 \
  --output-dir results/energy_research/custom_edges
```

### 脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--result-dir` | DiBS结果目录 | `results/energy_research/dibs_6groups_final` (自动使用最新) |
| `--threshold` | 边强度阈值 | 0.3 |
| `--output-dir` | 输出CSV文件目录 | `results/energy_research/dibs_edges_csv` |

---

## 📈 数据统计 (阈值=0.3)

### 总体统计
- **任务组数**: 6个
- **直接因果边总数**: 114条
- **间接因果路径总数**: 759条
- **总因果关系数**: 873条

### 按类型分布

**直接边**:
- 超参数 → 能耗: 57条 (50.0%)
- 性能 → 能耗: 46条 (40.4%)
- 超参数 → 性能: 11条 (9.6%)

**间接路径**:
- 多步路径 (≥4节点): 278条 (36.6%)
- 超参数 → 中介 → 能耗: 266条 (35.0%)
- 性能 → 中介 → 能耗: 200条 (26.4%)
- 超参数 → 中介 → 性能: 15条 (2.0%)

### 按任务组分布

| 任务组 | 直接边 | 间接路径 | 总计 |
|--------|--------|---------|------|
| group1_examples | 11 | 92 | 103 |
| group2_vulberta | 20 | 114 | 134 |
| group3_person_reid | 13 | 192 | 205 |
| group4_bug_localization | 27 | 142 | 169 |
| group5_mrt_oast | 21 | 102 | 123 |
| group6_resnet | 22 | 117 | 139 |

---

## ⚠️ 注意事项

### 1. 边强度阈值

- 当前使用阈值: **0.3**
- DiBS输出边强度范围: 0-1
- 阈值越高，边越可靠但数量越少
- 建议：
  - 探索性分析: threshold=0.1
  - 标准分析: threshold=0.3 (当前)
  - 保守分析: threshold=0.5

### 2. 因果方向

- DiBS学习的是有向因果图
- 边的方向: `source → target`
- 间接路径的方向: `source → mediator1 → mediator2 → target`
- ⚠️ 部分因果边可能是双向的，需要结合领域知识验证

### 3. 统计显著性

- DiBS给出的是边的**存在概率**（强度）
- 不等同于统计学上的p值
- 建议：
  1. 使用DiBS发现候选因果关系
  2. 使用回归分析/因果推断方法验证和量化效应大小
  3. 结合领域知识判断合理性

### 4. 多重比较

- 每个任务组有18-21个变量，可能的边数: 18×17 = 306 到 21×20 = 420
- 存在多重比较问题
- 建议：
  - 关注强度>0.5的非常强的边
  - 关注在多个任务组中一致出现的边
  - 使用Bonferroni或FDR校正

---

## 📚 相关文档

- DiBS分析脚本: `scripts/run_dibs_6groups_final.py`
- DiBS结果报告: `results/energy_research/dibs_6groups_final/20260116_004323/DIBS_6GROUPS_FINAL_REPORT.md`
- DiBS验证报告: `DIBS_VERIFICATION_REPORT_20260116.md`
- 参数调优报告: `docs/reports/DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md`

---

## 🆘 常见问题

### Q: 为什么有的边强度是1.0？

A: 强度1.0表示DiBS非常确信这条因果边的存在。这通常出现在：
- 变量之间存在明显的函数关系
- 数据中该因果关系非常强
- 可能存在数据泄漏（需要检查）

### Q: 如何判断因果边的可靠性？

A: 综合考虑：
1. 边强度 (>0.5更可靠)
2. 是否在多个任务组中出现
3. 是否符合领域知识
4. 使用回归分析验证

### Q: 间接效应强度如何计算？

A: 间接效应 = 路径各段边强度的乘积
- 例如: A→B (0.8) → C (0.9) 的间接效应 = 0.8 × 0.9 = 0.72

### Q: 完全中介和部分中介的区别？

A:
- **完全中介**: 直接效应≈0，所有效应通过中介变量
- **部分中介**: 直接效应>0，部分效应通过中介变量
- 计算: `mediation_ratio = indirect / (indirect + direct)`

---

**文档版本**: 1.0
**创建时间**: 2026-01-16
**作者**: Claude
**维护**: 请在更新CSV生成脚本后同步更新本文档
