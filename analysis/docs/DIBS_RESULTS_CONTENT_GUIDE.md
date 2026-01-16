# DiBS分析结果内容说明

**创建日期**: 2026-01-16
**相关文档**: [CSV使用指南](DIBS_EDGES_CSV_USAGE_GUIDE.md)

---

## 📦 DiBS分析结果包含的内容

### 1. 原始结果文件 (JSON)

**位置**: `results/energy_research/dibs_6groups_final/20260116_004323/`

**每个任务组3个文件**:

#### 1.1 因果图矩阵 (`.npy`)
```
group1_examples_causal_graph.npy
group2_vulberta_causal_graph.npy
...
```

**内容**:
- NumPy矩阵 (n_vars × n_vars)
- 元素 [i, j] 表示变量i → 变量j的因果边强度 (0-1)
- 主对角线为0（变量不能自己影响自己）

**读取方法**:
```python
import numpy as np
graph = np.load('group1_examples_causal_graph.npy')
print(f"图形状: {graph.shape}")  # 例如: (20, 20)
print(f"最大值: {graph.max()}")   # 最强的边
print(f"强边数(>0.3): {np.sum(graph > 0.3)}")
```

#### 1.2 特征名称 (`.json`)
```
group1_examples_feature_names.json
```

**内容**:
- 变量名称列表
- 对应因果图矩阵的行/列

**读取方法**:
```python
import json
with open('group1_examples_feature_names.json') as f:
    feature_names = json.load(f)

# 查找特定变量的索引
lr_idx = feature_names.index('hyperparam_learning_rate')
energy_idx = feature_names.index('energy_gpu_total_joules')

# 查询因果边强度
import numpy as np
graph = np.load('group1_examples_causal_graph.npy')
strength = graph[lr_idx, energy_idx]
print(f"learning_rate → energy: {strength:.4f}")
```

#### 1.3 完整分析结果 (`.json`)
```
group1_examples_result.json
```

**内容** (14个顶层键):
```json
{
  "task_id": "group1_examples",
  "task_name": "examples（图像分类-小型）",
  "success": true,
  "elapsed_time_minutes": 14.36,
  "n_samples": 304,
  "n_features": 20,
  "variable_classification": {
    "n_hyperparams": 4,
    "n_performance": 1,
    "n_energy": 4,
    "n_mediators": 7,
    "hyperparam_names": ["hyperparam_batch_size", ...],
    "performance_names": ["perf_test_accuracy"],
    "energy_names": ["energy_cpu_pkg_joules", ...],
    "mediator_names": ["energy_gpu_avg_watts", ...]
  },
  "graph_stats": {
    "min": 0.0,
    "max": 1.0,
    "mean": 0.29,
    "std": 0.35
  },
  "edges": {
    "threshold_0.01": 230,
    "threshold_0.1": 195,
    "threshold_0.3": 135,
    "threshold_0.5": 80
  },
  "question1_evidence": {
    "direct_hyperparam_to_energy": [...],
    "mediated_hyperparam_to_energy": [...]
  },
  "question2_evidence": {
    "direct_edges_perf_to_energy": [...],
    "direct_edges_energy_to_perf": [...],
    "common_hyperparams": [...],
    "mediated_tradeoffs": [...]
  },
  "question3_evidence": {
    "mediation_paths_to_energy": [...],
    "mediation_paths_to_performance": [...],
    "multi_step_paths": [...]
  },
  "config": {
    "alpha_linear": 0.05,
    "beta_linear": 0.1,
    "n_particles": 20,
    "tau": 1.0,
    "n_steps": 5000,
    ...
  },
  "feature_names": [...]
}
```

**读取方法**:
```python
import json
with open('group1_examples_result.json') as f:
    result = json.load(f)

# 查看基本信息
print(f"任务组: {result['task_name']}")
print(f"样本数: {result['n_samples']}")
print(f"强边数: {result['edges']['threshold_0.3']}")

# 查看问题1的直接效应
q1_direct = result['question1_evidence']['direct_hyperparam_to_energy']
print(f"超参数→能耗直接边: {len(q1_direct)}条")

# 提取第一条边的信息
if len(q1_direct) > 0:
    edge = q1_direct[0]
    print(f"  {edge['hyperparam']} → {edge['energy_var']}: {edge['strength']:.4f}")
```

---

### 2. 总结报告 (Markdown)

**文件**: `DIBS_6GROUPS_FINAL_REPORT.md`

**内容**:
- 总体统计（成功率、耗时）
- 任务组详细结果表格
- 研究问题1/2/3的证据汇总
- Top 10最强边列表
- 下一步建议

---

### 3. CSV文件 (便于分析) ⭐ 推荐使用

**位置**: `results/energy_research/dibs_edges_csv/`

#### 3.1 直接因果边 (`dibs_direct_edges.csv`)
- **114条边**
- 包含: 超参数→能耗、性能→能耗、超参数→性能
- **最常用** - 用于回答"哪些因素直接影响能耗？"

#### 3.2 间接因果路径 (`dibs_indirect_paths.csv`)
- **759条路径**
- 包含: 中介路径、多步路径
- **用于中介效应分析** - 回答"影响如何传递？"

#### 3.3 汇总统计 (`dibs_all_edges_summary.csv`)
- **6行** (每任务组1行)
- 包含: 样本数、特征数、边数统计、运行时间
- **用于跨组对比**

**详细使用说明**: 参见 [DIBS_EDGES_CSV_USAGE_GUIDE.md](DIBS_EDGES_CSV_USAGE_GUIDE.md)

---

## 🎯 按研究问题查找内容

### 研究问题1: 超参数对能耗的影响

**方法1: 使用CSV文件** (推荐)
```python
import pandas as pd
edges = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_direct_edges.csv')

# 直接效应
q1_direct = edges[edges['research_question'] == 'Q1_direct']
print(f"找到{len(q1_direct)}条超参数→能耗的直接边")

# 查看具体边
print(q1_direct[['source', 'target', 'strength', 'task_name']])
```

**方法2: 使用JSON文件**
```python
import json
with open('group1_examples_result.json') as f:
    result = json.load(f)

# 直接效应
direct_edges = result['question1_evidence']['direct_hyperparam_to_energy']

# 间接效应（中介路径）
mediated_paths = result['question1_evidence']['mediated_hyperparam_to_energy']

print(f"直接边: {len(direct_edges)}条")
print(f"中介路径: {len(mediated_paths)}条")
```

---

### 研究问题2: 能耗-性能权衡关系

**方法1: 使用CSV文件** (推荐)
```python
import pandas as pd
edges = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_direct_edges.csv')

# 性能→能耗的权衡
perf_to_energy = edges[edges['edge_type'] == 'performance_to_energy']
print(f"找到{len(perf_to_energy)}条性能→能耗的边")

# 共同影响能耗和性能的超参数
common_hp = edges[edges['research_question'] == 'Q2_common_hyperparam']
print(f"找到{len(common_hp)}个共同超参数")
```

**方法2: 使用JSON文件**
```python
q2_evidence = result['question2_evidence']

# 性能→能耗
perf_to_energy = q2_evidence['direct_edges_perf_to_energy']

# 能耗→性能（通常很少或没有）
energy_to_perf = q2_evidence['direct_edges_energy_to_perf']

# 共同超参数（同时影响能耗和性能）
common_hyperparams = q2_evidence['common_hyperparams']

# 中介权衡路径
mediated_tradeoffs = q2_evidence['mediated_tradeoffs']
```

---

### 研究问题3: 中介效应路径

**方法1: 使用CSV文件** (推荐)
```python
import pandas as pd
paths = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_indirect_paths.csv')

# 超参数→中介→能耗
mediation_energy = paths[paths['research_question'] == 'Q3_mediation_energy']

# 区分完全中介和部分中介
full_mediation = mediation_energy[mediation_energy['mediation_type'] == 'full']
partial_mediation = mediation_energy[mediation_energy['mediation_type'] == 'partial']

print(f"完全中介: {len(full_mediation)}条")
print(f"部分中介: {len(partial_mediation)}条")

# 多步路径
multi_step = paths[paths['num_steps'] == 3]
print(f"多步路径: {len(multi_step)}条")
```

**方法2: 使用JSON文件**
```python
q3_evidence = result['question3_evidence']

# 超参数→中介→能耗
mediation_to_energy = q3_evidence['mediation_paths_to_energy']

# 超参数→中介→性能
mediation_to_perf = q3_evidence['mediation_paths_to_performance']

# 多步路径（≥4节点）
multi_step_paths = q3_evidence['multi_step_paths']

# 查看第一条中介路径
if len(mediation_to_energy) > 0:
    path = mediation_to_energy[0]
    print(f"路径: {path['hyperparam']} → {path['mediator']} → {path['outcome']}")
    print(f"间接效应: {path['indirect_strength']:.4f}")
    print(f"直接效应: {path['direct_strength']:.4f}")
    print(f"中介类型: {path['mediation_type']}")
```

---

## 🔍 高级用法：直接查询因果图矩阵

**适用场景**: 需要查询特定变量对之间的因果关系

```python
import numpy as np
import json

# 1. 加载因果图和特征名称
graph = np.load('group1_examples_causal_graph.npy')
with open('group1_examples_feature_names.json') as f:
    features = json.load(f)

# 2. 创建特征名到索引的映射
feat_to_idx = {name: idx for idx, name in enumerate(features)}

# 3. 查询特定因果关系
def get_causal_strength(source, target):
    """查询source → target的因果边强度"""
    src_idx = feat_to_idx[source]
    tgt_idx = feat_to_idx[target]
    return graph[src_idx, tgt_idx]

# 示例: learning_rate → energy_gpu_total_joules
strength = get_causal_strength('hyperparam_learning_rate', 'energy_gpu_total_joules')
print(f"因果边强度: {strength:.4f}")

# 4. 找到某个变量的所有因果效应
def get_all_effects(source, threshold=0.3):
    """找到source影响的所有变量"""
    src_idx = feat_to_idx[source]
    effects = []
    for tgt_idx, name in enumerate(features):
        strength = graph[src_idx, tgt_idx]
        if strength > threshold:
            effects.append((name, strength))
    return sorted(effects, key=lambda x: x[1], reverse=True)

# 示例: learning_rate影响哪些变量？
effects = get_all_effects('hyperparam_learning_rate')
print(f"learning_rate的因果效应 ({len(effects)}个):")
for target, strength in effects:
    print(f"  → {target}: {strength:.4f}")

# 5. 可视化因果图（可选）
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(graph, xticklabels=features, yticklabels=features,
            cmap='YlOrRd', vmin=0, vmax=1, cbar_kws={'label': '因果边强度'})
plt.title('因果图热力图')
plt.xlabel('目标变量')
plt.ylabel('源变量')
plt.tight_layout()
plt.savefig('causal_graph_heatmap.png', dpi=300)
```

---

## 📊 推荐的分析流程

### 步骤1: 从CSV快速探索 ⭐ 最简单

```python
import pandas as pd

# 读取CSV
edges = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_direct_edges.csv')
paths = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_indirect_paths.csv')
summary = pd.read_csv('results/energy_research/dibs_edges_csv/dibs_all_edges_summary.csv')

# 快速浏览
print("=== 总体统计 ===")
print(f"直接边: {len(edges)}条")
print(f"间接路径: {len(paths)}条")

print("\n=== 最强的10条边 ===")
print(edges.nlargest(10, 'strength')[['source', 'target', 'strength', 'task_name']])

print("\n=== 各类边的数量 ===")
print(edges['edge_type'].value_counts())
```

### 步骤2: 针对研究问题深入分析

```python
# 问题1: 哪些超参数显著影响能耗？
q1_edges = edges[edges['edge_type'] == 'hyperparam_to_energy']
hp_effects = q1_edges.groupby('source').agg({
    'strength': ['count', 'mean', 'max']
})
print("超参数对能耗的影响:")
print(hp_effects.sort_values(('strength', 'mean'), ascending=False))
```

### 步骤3: 使用JSON获取详细信息（如需要）

```python
# 只在需要完整上下文时才读取JSON
import json
with open('group1_examples_result.json') as f:
    result = json.load(f)

# 获取变量分类
var_class = result['variable_classification']
print(f"超参数: {var_class['hyperparam_names']}")
print(f"性能指标: {var_class['performance_names']}")
print(f"能耗指标: {var_class['energy_names']}")
print(f"中介变量: {var_class['mediator_names']}")
```

### 步骤4: 查询因果图矩阵（如需要精确值）

```python
# 只在需要查询特定变量对时才读取矩阵
import numpy as np
graph = np.load('group1_examples_causal_graph.npy')

# 查询特定因果边
lr_idx = features.index('hyperparam_learning_rate')
energy_idx = features.index('energy_gpu_total_joules')
strength = graph[lr_idx, energy_idx]
print(f"learning_rate → energy的精确强度: {strength}")
```

---

## 💡 最佳实践

### ✅ 推荐做法

1. **优先使用CSV文件** - 最方便，适合大部分分析
2. **使用pandas进行数据处理** - 灵活强大
3. **结合汇总统计了解全局** - 再深入细节
4. **交叉验证多个任务组** - 提高可靠性
5. **结合领域知识判断合理性** - DiBS不能保证100%正确

### ⚠️ 避免的做法

1. **不要直接信任强度=1.0的边** - 可能存在数据泄漏
2. **不要忽略多重比较问题** - 需要校正
3. **不要只看单个任务组** - 可能是偶然
4. **不要把边强度当作效应大小** - 需要回归分析量化
5. **不要忽略因果方向** - A→B ≠ B→A

---

## 📚 相关文档

- **CSV使用指南**: [DIBS_EDGES_CSV_USAGE_GUIDE.md](DIBS_EDGES_CSV_USAGE_GUIDE.md) ⭐ 重点
- **DiBS验证报告**: `DIBS_VERIFICATION_REPORT_20260116.md`
- **DiBS分析报告**: `results/energy_research/dibs_6groups_final/20260116_004323/DIBS_6GROUPS_FINAL_REPORT.md`
- **参数调优报告**: `docs/reports/DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md`

---

**文档版本**: 1.0
**创建时间**: 2026-01-16
**作者**: Claude
