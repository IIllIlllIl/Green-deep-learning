# DiBS结果展示优化方案

**创建日期**: 2026-01-17
**问题**: 当前DiBS结果可读性差，难以快速理解完整因果关系
**目标**: 设计人类可读、信息完整、易于分析的结果格式

---

## 📋 目录

- [当前问题分析](#当前问题分析)
- [方案设计](#方案设计)
- [详细格式说明](#详细格式说明)
- [实施建议](#实施建议)
- [使用示例](#使用示例)

---

## 当前问题分析

### 现有文件格式

每个任务组生成3个文件：

```
group1_examples_causal_graph.npy       (2.2 KB)  - 因果图矩阵（二进制）
group1_examples_feature_names.json     (648 B)   - 变量名列表
group1_examples_result.json            (3.7 KB)  - 分析摘要
```

### 存在的问题

#### 问题1: **分离存储导致可读性差** ⭐⭐⭐⭐⭐

**现状**:
- 因果图矩阵存储在 `.npy` 文件（NumPy二进制格式）
- 变量名存储在单独的 `_feature_names.json`
- 用户需要手动组合才能理解因果关系

**问题**:
```python
# 当前阅读流程（繁琐）
causal_graph = np.load('causal_graph.npy')  # 23x23矩阵
feature_names = json.load('feature_names.json')  # 23个名称

# 要知道 "batch_size → energy_gpu" 的强度:
i = feature_names.index('hyperparam_batch_size')  # 找索引
j = feature_names.index('energy_gpu_total_joules')
strength = causal_graph[i, j]  # 手动索引

# 😫 太麻烦了！
```

#### 问题2: **result.json信息不完整** ⭐⭐⭐⭐

**现状**:
- `result.json` 只包含摘要信息（问题1/2/3的证据）
- **不包含完整的因果图边列表**
- 用户无法从JSON中直接看到所有因果关系

**缺失信息**:
```json
// result.json 有的:
"question1_evidence": {
  "moderation_effects": [...]  // 只有调节效应的边
}

// result.json 没有的:
"all_edges": [...]  // ❌ 缺失！无法看到所有强边
```

#### 问题3: **npy格式不通用** ⭐⭐⭐

**现状**:
- `.npy` 需要Python + NumPy才能读取
- 其他工具（R, MATLAB, Excel）无法直接打开
- 不利于跨平台分析

#### 问题4: **缺少边的语义信息** ⭐⭐⭐⭐

**现状**:
- 只知道边的强度（0.35）
- 不知道边的类型（主效应？调节效应？中介？）
- 不知道边的研究意义

**示例**:
```
当前: batch_size → energy_gpu (0.35)
需要: batch_size → energy_gpu (0.35, 主效应, 问题1证据)
```

---

## 方案设计

### 核心思路

✅ **统一存储**: 将因果图、变量名、边列表、分析结果整合到单一文件
✅ **人类可读**: 使用自解释的JSON/CSV格式
✅ **信息完整**: 包含所有边和语义标注
✅ **多层次**: 支持快速概览和深度分析

### 推荐方案（三文件组合）

每个任务组生成3个互补文件：

```
group1_examples_causal_edges.csv         ⭐⭐⭐⭐⭐ 核心（边列表）
group1_examples_analysis_summary.json   ⭐⭐⭐⭐  补充（分析摘要）
group1_examples_causal_graph.npy        ⭐⭐     备用（原始矩阵）
```

#### 文件1: **causal_edges.csv** （核心，人类可读）⭐⭐⭐⭐⭐

**用途**: 列出所有显著因果边，一目了然

**格式**:
```csv
source,target,strength,edge_type,question_relevance,interpretation
hyperparam_batch_size,energy_cpu_total_joules,0.35,main_effect,Q1,batch_size直接影响CPU能耗
hyperparam_batch_size_x_is_parallel,energy_cpu_total_joules,0.35,moderation,Q1,并行模式调节batch_size效应
hyperparam_epochs,energy_gpu_total_joules,0.40,main_effect,Q1,epochs直接影响GPU能耗
energy_gpu_avg_watts,perf_test_accuracy,0.25,mediator,Q3,GPU功率通过某种机制影响性能
...
```

**优点**:
- ✅ 可直接用Excel/Google Sheets打开
- ✅ 可快速筛选（如只看强度>0.5的边）
- ✅ 包含语义信息（边类型、研究意义）
- ✅ 可排序、可搜索

#### 文件2: **analysis_summary.json** （补充，结构化）⭐⭐⭐⭐

**用途**: 保留原有的分析摘要，增加完整边列表

**新增字段**:
```json
{
  "task_id": "group1_examples",
  "n_samples": 304,
  "n_features": 23,

  // ⭐ 新增：完整边列表（强度>0.01）
  "all_edges": [
    {
      "source": "hyperparam_batch_size",
      "target": "energy_cpu_total_joules",
      "strength": 0.35,
      "edge_type": "main_effect",
      "question_relevance": ["Q1"]
    },
    ...
  ],

  // ⭐ 新增：按强度分层的边
  "edges_by_strength": {
    "very_strong": [  // >0.5
      {"source": "...", "target": "...", "strength": 0.55}
    ],
    "strong": [       // 0.3-0.5
      {"source": "...", "target": "...", "strength": 0.40}
    ],
    "moderate": [     // 0.1-0.3
      ...
    ]
  },

  // ⭐ 新增：按变量类型分类的边
  "edges_by_type": {
    "hyperparam_to_energy": [...],      // 超参数 → 能耗
    "interaction_to_energy": [...],     // 交互项 → 能耗 (调节效应)
    "hyperparam_to_performance": [...], // 超参数 → 性能
    "energy_to_performance": [...],     // 能耗 → 性能
    "mediator_edges": [...]             // 中介变量相关
  },

  // 保留原有字段
  "question1_evidence": {...},
  "question2_evidence": {...},
  "question3_evidence": {...},
  "variable_classification": {...},
  "graph_stats": {...}
}
```

#### 文件3: **causal_graph.npy** （备用，原始数据）⭐⭐

**用途**: 保留原始矩阵用于高级分析

**保留原因**:
- 某些算法需要完整矩阵（如路径搜索）
- 精确数值计算
- 与其他工具集成

---

## 详细格式说明

### CSV格式规范 (causal_edges.csv)

#### 必需列

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `source` | string | 源变量名（因） | `hyperparam_batch_size` |
| `target` | string | 目标变量名（果） | `energy_gpu_total_joules` |
| `strength` | float | 边强度（0-1） | `0.35` |
| `edge_type` | string | 边类型 | `main_effect`, `moderation`, `mediator` |
| `question_relevance` | string | 相关研究问题 | `Q1`, `Q2`, `Q3`, `Q1,Q3` |
| `interpretation` | string | 人类可读解释 | `并行模式调节batch_size对能耗的效应` |

#### 可选列（推荐）

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `source_category` | string | 源变量类别 | `hyperparam`, `interaction`, `energy` |
| `target_category` | string | 目标变量类别 | `energy`, `performance`, `mediator` |
| `strength_level` | string | 强度等级 | `very_strong`, `strong`, `moderate`, `weak` |
| `is_direct` | boolean | 是否直接效应 | `true`, `false` |
| `statistical_significance` | float | 统计显著性（如有） | `0.001` |

#### 边类型分类

| edge_type | 含义 | 示例 |
|-----------|------|------|
| `main_effect` | 主效应（超参数直接影响能耗/性能） | `batch_size → energy_gpu` |
| `moderation` | 调节效应（交互项→能耗） | `batch_size_x_parallel → energy_gpu` |
| `mediator` | 中介效应（通过中间变量） | `batch_size → gpu_watts → energy_gpu` |
| `control_effect` | 控制变量效应 | `model_mnist_ff → energy_cpu` |
| `reverse` | 反向因果（能耗→性能，罕见） | `energy_gpu → accuracy` |
| `confounding` | 混淆关系 | 待定 |

#### 示例CSV内容

```csv
source,target,strength,edge_type,question_relevance,source_category,target_category,strength_level,interpretation
hyperparam_batch_size,energy_cpu_total_joules,0.00,main_effect,Q1,hyperparam,energy,weak,batch_size主效应很弱（几乎为0）
hyperparam_batch_size_x_is_parallel,energy_cpu_total_joules,0.35,moderation,Q1,interaction,energy,strong,并行模式显著增强batch_size的能耗效应（纯调节）
hyperparam_batch_size_x_is_parallel,energy_gpu_total_joules,0.30,moderation,Q1,interaction,energy,strong,并行模式调节batch_size对GPU能耗的影响
hyperparam_epochs,energy_gpu_total_joules,0.40,main_effect,Q1,hyperparam,energy,strong,epochs直接影响GPU能耗（主效应）
hyperparam_epochs_x_is_parallel,energy_gpu_total_joules,0.40,moderation,Q1,interaction,energy,strong,并行模式进一步放大epochs的能耗效应
energy_gpu_avg_watts,perf_test_accuracy,0.25,mediator,Q2,mediator,performance,moderate,GPU功率可能通过某种机制影响性能
hyperparam_batch_size,energy_gpu_max_watts,0.30,main_effect,Q1,hyperparam,mediator,strong,batch_size影响GPU峰值功率
energy_gpu_max_watts,energy_gpu_util_max_percent,0.40,mediator,Q3,mediator,mediator,strong,GPU功率影响利用率
energy_gpu_util_max_percent,perf_test_accuracy,0.35,mediator,Q3,mediator,performance,strong,GPU利用率影响性能
```

---

### JSON格式增强 (analysis_summary.json)

#### 新增字段详细说明

**1. all_edges（完整边列表）**

```json
"all_edges": [
  {
    "source": "hyperparam_batch_size_x_is_parallel",
    "target": "energy_cpu_total_joules",
    "strength": 0.35,
    "edge_type": "moderation",
    "question_relevance": ["Q1"],
    "source_index": 20,  // 在矩阵中的索引
    "target_index": 2,
    "interpretation": "并行模式调节batch_size对CPU能耗的效应"
  },
  ...
]
```

**2. edges_by_strength（按强度分层）**

```json
"edges_by_strength": {
  "very_strong": {  // >0.5
    "count": 12,
    "edges": [
      {"source": "...", "target": "...", "strength": 0.55},
      ...
    ]
  },
  "strong": {  // 0.3-0.5
    "count": 53,
    "edges": [...]
  },
  "moderate": {  // 0.1-0.3
    "count": 187,
    "edges": [...]
  },
  "weak": {  // 0.01-0.1
    "count": 119,
    "edges": [...]
  }
}
```

**3. edges_by_type（按因果类型分类）**

```json
"edges_by_type": {
  "hyperparam_to_energy": {
    "description": "超参数直接影响能耗（主效应）",
    "count": 2,
    "edges": [
      {
        "source": "hyperparam_epochs",
        "target": "energy_gpu_total_joules",
        "strength": 0.40,
        "edge_type": "main_effect"
      }
    ]
  },
  "interaction_to_energy": {
    "description": "交互项影响能耗（调节效应）⭐",
    "count": 5,
    "edges": [...]
  },
  "hyperparam_to_performance": {
    "description": "超参数直接影响性能",
    "count": 0,
    "edges": []
  },
  "energy_to_performance": {
    "description": "能耗影响性能（权衡关系）",
    "count": 0,
    "edges": []
  },
  "mediator_paths": {
    "description": "中介变量路径",
    "count": 8,
    "edges": [...]
  }
}
```

**4. variable_summary（变量级汇总）**

```json
"variable_summary": {
  "hyperparam_batch_size": {
    "category": "hyperparam",
    "outgoing_edges": 3,  // 作为源的边数
    "incoming_edges": 0,  // 作为目标的边数
    "strongest_outgoing": {
      "target": "energy_gpu_max_watts",
      "strength": 0.30
    },
    "affects_energy": true,
    "affects_performance": false
  },
  "hyperparam_batch_size_x_is_parallel": {
    "category": "interaction",
    "base_hyperparam": "hyperparam_batch_size",
    "outgoing_edges": 2,
    "incoming_edges": 0,
    "strongest_outgoing": {
      "target": "energy_cpu_total_joules",
      "strength": 0.35
    },
    "moderation_targets": ["energy_cpu_total_joules", "energy_gpu_total_joules"]
  },
  ...
}
```

**5. causal_paths（多步因果路径）**

```json
"causal_paths": {
  "two_step": [
    {
      "path": "hyperparam_batch_size → energy_gpu_max_watts → energy_gpu_total_joules",
      "strength_step1": 0.30,
      "strength_step2": 0.40,
      "path_strength": 0.12,
      "interpretation": "batch_size通过GPU峰值功率影响总能耗"
    }
  ],
  "three_step": [
    {
      "path": "hyperparam_batch_size → energy_gpu_max_watts → energy_gpu_util_max_percent → perf_test_accuracy",
      "strength_step1": 0.30,
      "strength_step2": 0.40,
      "strength_step3": 0.35,
      "path_strength": 0.042,
      "interpretation": "batch_size通过GPU功率和利用率影响性能"
    }
  ]
}
```

---

## 实施建议

### 优先级

| 优先级 | 任务 | 工作量 | 价值 |
|--------|------|--------|------|
| ⭐⭐⭐⭐⭐ | 实现CSV边列表生成 | 2-3小时 | 极高 |
| ⭐⭐⭐⭐ | 增强JSON摘要（all_edges等） | 2小时 | 高 |
| ⭐⭐⭐ | 添加边类型自动分类 | 1-2小时 | 中 |
| ⭐⭐ | 变量级汇总统计 | 1小时 | 中 |
| ⭐ | 因果路径自动发现 | 3-4小时 | 低 |

### 实施步骤

#### 阶段1: CSV边列表生成（核心）⭐⭐⭐⭐⭐

**输入**:
- `causal_graph.npy` (23x23矩阵)
- `feature_names.json` (23个变量名)

**输出**:
- `causal_edges.csv`

**伪代码**:
```python
import numpy as np
import pandas as pd
import json

# 读取数据
causal_graph = np.load('causal_graph.npy')
feature_names = json.load(open('feature_names.json'))

# 提取边（强度>0.01）
edges = []
for i, source in enumerate(feature_names):
    for j, target in enumerate(feature_names):
        strength = causal_graph[i, j]
        if strength > 0.01:
            edge_type = classify_edge_type(source, target)
            question_relevance = get_question_relevance(source, target, edge_type)
            interpretation = generate_interpretation(source, target, strength, edge_type)

            edges.append({
                'source': source,
                'target': target,
                'strength': strength,
                'edge_type': edge_type,
                'question_relevance': question_relevance,
                'source_category': get_category(source),
                'target_category': get_category(target),
                'strength_level': get_strength_level(strength),
                'interpretation': interpretation
            })

# 转换为DataFrame并保存
df = pd.DataFrame(edges)
df = df.sort_values('strength', ascending=False)  # 按强度降序
df.to_csv('causal_edges.csv', index=False)
```

**关键函数**:
```python
def classify_edge_type(source, target):
    """分类边类型"""
    if '_x_is_parallel' in source and 'energy' in target:
        return 'moderation'
    elif 'hyperparam' in source and 'energy' in target:
        return 'main_effect'
    elif 'energy_gpu' in source and source != target and 'energy' in target:
        return 'mediator'
    elif 'hyperparam' in source and 'perf' in target:
        return 'main_effect'
    elif 'model_' in source:
        return 'control_effect'
    else:
        return 'other'

def get_category(var_name):
    """获取变量类别"""
    if 'hyperparam_' in var_name and '_x_is_parallel' not in var_name:
        return 'hyperparam'
    elif '_x_is_parallel' in var_name:
        return 'interaction'
    elif 'energy_cpu' in var_name or 'energy_gpu_total' in var_name:
        return 'energy'
    elif 'energy_gpu' in var_name:
        return 'mediator'
    elif 'perf_' in var_name:
        return 'performance'
    elif 'model_' in var_name:
        return 'control'
    else:
        return 'other'

def get_strength_level(strength):
    """获取强度等级"""
    if strength > 0.5:
        return 'very_strong'
    elif strength > 0.3:
        return 'strong'
    elif strength > 0.1:
        return 'moderate'
    else:
        return 'weak'

def get_question_relevance(source, target, edge_type):
    """判断与研究问题的相关性"""
    relevance = []

    # Q1: 超参数对能耗的影响
    if ('hyperparam' in source or '_x_is_parallel' in source) and \
       ('energy_cpu' in target or 'energy_gpu_total' in target):
        relevance.append('Q1')

    # Q2: 能耗-性能权衡
    if ('energy' in source and 'perf' in target) or \
       ('perf' in source and 'energy' in target):
        relevance.append('Q2')

    # Q3: 中介效应
    if edge_type == 'mediator':
        relevance.append('Q3')

    return ','.join(relevance) if relevance else 'other'

def generate_interpretation(source, target, strength, edge_type):
    """生成人类可读解释"""
    source_clean = source.replace('hyperparam_', '').replace('energy_', '').replace('_x_is_parallel', '')
    target_clean = target.replace('energy_', '').replace('perf_', '')

    if edge_type == 'moderation':
        base_param = source.replace('_x_is_parallel', '').replace('hyperparam_', '')
        return f"并行模式调节{base_param}对{target_clean}的效应"
    elif edge_type == 'main_effect':
        return f"{source_clean}直接影响{target_clean}"
    elif edge_type == 'mediator':
        return f"{source_clean}通过某种机制影响{target_clean}"
    else:
        return f"{source_clean} → {target_clean}"
```

#### 阶段2: 增强JSON摘要 ⭐⭐⭐⭐

**在现有result.json基础上添加**:
```python
# 读取现有结果
with open('result.json') as f:
    result = json.load(f)

# 添加all_edges
result['all_edges'] = edges  # 从阶段1获取

# 添加edges_by_strength
result['edges_by_strength'] = {
    'very_strong': [e for e in edges if e['strength'] > 0.5],
    'strong': [e for e in edges if 0.3 < e['strength'] <= 0.5],
    'moderate': [e for e in edges if 0.1 < e['strength'] <= 0.3],
    'weak': [e for e in edges if 0.01 < e['strength'] <= 0.1]
}

# 添加edges_by_type
result['edges_by_type'] = group_edges_by_type(edges)

# 保存
with open('analysis_summary.json', 'w') as f:
    json.dump(result, f, indent=2)
```

#### 阶段3: 后处理工具 ⭐⭐⭐

**快速查询脚本**:
```bash
# 查看所有调节效应
cat causal_edges.csv | grep "moderation" | column -t -s,

# 查看强度>0.4的边
cat causal_edges.csv | awk -F, '$3 > 0.4' | column -t -s,

# 查看batch_size相关的边
cat causal_edges.csv | grep "batch_size"
```

---

## 使用示例

### 场景1: 快速浏览所有调节效应

**使用CSV**:
```bash
# Excel/Google Sheets中打开causal_edges.csv
# 筛选 edge_type = "moderation"
# 按 strength 降序排序

# 命令行快速查看
cat causal_edges.csv | grep "moderation" | sort -t, -k3 -rn
```

**输出**:
```
source,target,strength,edge_type,question_relevance,interpretation
hyperparam_epochs_x_is_parallel,energy_gpu_total_joules,0.40,moderation,Q1,并行模式调节epochs对gpu_total_joules的效应
hyperparam_batch_size_x_is_parallel,energy_cpu_total_joules,0.35,moderation,Q1,并行模式调节batch_size对cpu_total_joules的效应
hyperparam_batch_size_x_is_parallel,energy_gpu_total_joules,0.30,moderation,Q1,并行模式调节batch_size对gpu_total_joules的效应
```

### 场景2: 分析某个超参数的完整因果链

**使用JSON**:
```python
import json

with open('analysis_summary.json') as f:
    result = json.load(f)

# 查找batch_size相关的所有边
batch_size_edges = [
    e for e in result['all_edges']
    if 'batch_size' in e['source']
]

# 按目标分类
for edge in batch_size_edges:
    print(f"{edge['source']} → {edge['target']}: {edge['strength']:.2f} ({edge['edge_type']})")
```

**输出**:
```
hyperparam_batch_size → energy_gpu_max_watts: 0.30 (main_effect)
hyperparam_batch_size_x_is_parallel → energy_cpu_total_joules: 0.35 (moderation)
hyperparam_batch_size_x_is_parallel → energy_gpu_total_joules: 0.30 (moderation)
```

### 场景3: 验证研究问题1的证据

**使用JSON**:
```python
# 提取Q1相关的边
q1_edges = result['edges_by_type']['hyperparam_to_energy']
q1_moderation = result['edges_by_type']['interaction_to_energy']

print(f"主效应: {len(q1_edges)}个")
for edge in q1_edges:
    print(f"  - {edge['source']} → {edge['target']}: {edge['strength']:.2f}")

print(f"\n调节效应: {len(q1_moderation)}个")
for edge in q1_moderation:
    print(f"  - {edge['source']} → {edge['target']}: {edge['strength']:.2f}")
```

### 场景4: 导出到其他工具

**R语言**:
```r
# 读取CSV
library(readr)
edges <- read_csv("causal_edges.csv")

# 筛选强边
strong_edges <- edges %>% filter(strength > 0.3)

# 绘制网络图
library(igraph)
g <- graph_from_data_frame(strong_edges[, c("source", "target", "strength")])
plot(g)
```

**MATLAB**:
```matlab
% 读取CSV
edges = readtable('causal_edges.csv');

% 筛选调节效应
moderation_edges = edges(strcmp(edges.edge_type, 'moderation'), :);

% 显示
disp(moderation_edges);
```

---

## 方案对比

| 维度 | 当前方案 | 新方案 | 改进 |
|------|---------|--------|------|
| **可读性** | ⭐⭐ (需编程) | ⭐⭐⭐⭐⭐ (直接打开CSV) | +3 |
| **信息完整性** | ⭐⭐⭐ (分散) | ⭐⭐⭐⭐⭐ (统一) | +2 |
| **跨平台** | ⭐⭐ (仅Python) | ⭐⭐⭐⭐⭐ (Excel/R/MATLAB) | +3 |
| **语义丰富** | ⭐ (仅强度) | ⭐⭐⭐⭐⭐ (类型+解释) | +4 |
| **查询效率** | ⭐⭐ (手动索引) | ⭐⭐⭐⭐⭐ (筛选/排序) | +3 |
| **文件大小** | ⭐⭐⭐⭐⭐ (2.2KB) | ⭐⭐⭐⭐ (约10-20KB) | -1 |

**总体评价**: 新方案在可用性、可读性、语义丰富度上显著优于当前方案，文件大小增加可忽略。

---

## 潜在扩展

### 扩展1: 交互式可视化HTML

**生成自包含HTML文件**:
```html
<!-- causal_graph_interactive.html -->
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
  // 读取edges数据
  const edges = [
    {source: "batch_size", target: "energy_gpu", strength: 0.35, ...},
    ...
  ];

  // D3.js绘制交互式因果图
  // 点击节点 → 高亮相关边
  // 拖拽调整布局
  // 鼠标悬停 → 显示interpretation
</script>
```

### 扩展2: Markdown报告生成

**自动生成可读报告**:
```markdown
# Group1: examples组因果分析报告

## 调节效应发现

发现2个调节效应:

1. **batch_size × is_parallel → CPU能耗** (强度0.35)
   - 解释: 并行模式调节batch_size对CPU能耗的效应
   - 含义: batch_size在非并行时几乎无影响，并行时影响显著

2. **batch_size × is_parallel → GPU能耗** (强度0.30)
   - 解释: 同上

## 主效应发现
...
```

### 扩展3: 差异分析

**对比不同组的因果图**:
```python
# 比较examples和Person_reID的调节效应
common_moderation = set(group1_moderation) & set(group3_moderation)
unique_to_group1 = set(group1_moderation) - set(group3_moderation)
unique_to_group3 = set(group3_moderation) - set(group1_moderation)
```

---

## 总结

### 核心优势

✅ **人类可读**: CSV可直接用Excel打开，无需编程
✅ **信息完整**: 包含所有边和语义标注，不再需要手动组合
✅ **跨平台**: 支持Python/R/MATLAB/Excel等工具
✅ **多层次**: 支持快速概览（CSV）和深度分析（JSON）
✅ **可扩展**: 易于添加新字段或生成新格式

### 实施优先级

⭐⭐⭐⭐⭐ **立即实施**: CSV边列表生成（2-3小时）
⭐⭐⭐⭐ **短期实施**: JSON摘要增强（2小时）
⭐⭐⭐ **中期实施**: 边类型自动分类（1-2小时）

### 预期收益

- **时间节省**: 从5分钟手动索引 → 10秒直接查看
- **错误减少**: 避免手动索引错误
- **分析效率**: 支持快速筛选、排序、分组
- **协作友好**: 非编程人员也能理解结果

---

**文档版本**: v1.0
**创建时间**: 2026-01-17
**状态**: 待用户审核
**下一步**: 用户确认后开始实施
