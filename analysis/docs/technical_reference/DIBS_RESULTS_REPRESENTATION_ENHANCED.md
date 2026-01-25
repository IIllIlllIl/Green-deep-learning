# DiBS结果展示增强方案 v2.0

**创建日期**: 2026-01-17
**版本**: v2.0 (增强版)
**改进点**:
- ✅ 间接边可读性优化
- ✅ 展示所有边（不做强度筛选）

---

## 📋 目录

- [用户需求](#用户需求)
- [核心设计改进](#核心设计改进)
- [文件格式详细说明](#文件格式详细说明)
- [间接边展示方案对比](#间接边展示方案对比)
- [实施方案](#实施方案)

---

## 用户需求

### 需求1: 间接边的可读性 ⭐⭐⭐⭐⭐

**问题**:
- `causal_edges.csv` 只包含直接边（A → B）
- 间接边（A → M → B）需要用户手动查找中介变量M
- 例如：`batch_size → gpu_watts → energy_gpu` 需要查看2条边才能理解

**用户期望**:
- 自动识别间接路径
- 以可读方式展示多步因果链
- 区分直接效应和间接效应

### 需求2: 不做筛选，全部展示 ⭐⭐⭐⭐⭐

**问题**:
- 原方案只保存强度>0.01的边（筛选掉了约300条弱边）
- 用户可能需要检查所有边（包括弱边）

**用户期望**:
- 保存所有边（包括强度=0.00的边）
- 让用户自己决定是否需要筛选

---

## 核心设计改进

### 方案总览（4文件组合）

每个任务组生成4个文件：

```
group1_examples_causal_edges_all.csv          ⭐⭐⭐⭐⭐ 所有直接边（无筛选）
group1_examples_causal_paths.csv             ⭐⭐⭐⭐⭐ 间接路径（2步、3步）
group1_examples_analysis_summary.json        ⭐⭐⭐⭐  分析摘要（增强版）
group1_examples_causal_graph.npy             ⭐⭐     原始矩阵（备用）
```

### 改进点1: 所有边展示（无筛选）

**文件**: `causal_edges_all.csv`

**包含**:
- **所有** 23×23 = 529 条可能的边
- 包括强度=0.00的边（表示无因果关系）
- 用户可自行筛选（Excel筛选器）

**优势**:
- ✅ 完整信息，无遗漏
- ✅ 用户灵活筛选（如只看>0.3的边）
- ✅ 可验证"为什么某条边不存在"

### 改进点2: 间接路径专用文件

**文件**: `causal_paths.csv`

**包含**:
- 所有2步路径（A → M → B）
- 所有3步路径（A → M1 → M2 → B）
- 路径强度、路径类型、研究意义

**优势**:
- ✅ 直观展示间接效应
- ✅ 无需手动拼接路径
- ✅ 自动计算路径总强度

---

## 文件格式详细说明

### 文件1: causal_edges_all.csv（所有直接边）⭐⭐⭐⭐⭐

#### 格式设计

**列定义**:
```csv
source,target,strength,edge_type,question_relevance,source_category,target_category,strength_level,is_significant,interpretation
```

**完整示例**（包含弱边和零边）:
```csv
source,target,strength,edge_type,question_relevance,source_category,target_category,strength_level,is_significant,interpretation
hyperparam_batch_size,energy_cpu_total_joules,0.00,main_effect,Q1,hyperparam,energy,zero,no,batch_size对CPU能耗无直接影响
hyperparam_batch_size_x_is_parallel,energy_cpu_total_joules,0.35,moderation,Q1,interaction,energy,strong,yes,并行模式调节batch_size对CPU能耗的效应（纯调节）
hyperparam_batch_size,energy_gpu_max_watts,0.30,main_effect,Q1,hyperparam,mediator,strong,yes,batch_size影响GPU峰值功率
hyperparam_epochs,energy_gpu_total_joules,0.40,main_effect,Q1,hyperparam,energy,strong,yes,epochs直接影响GPU能耗
is_parallel,energy_cpu_total_joules,0.15,main_effect,Q1,mode,energy,moderate,yes,并行模式直接增加CPU能耗基线
model_mnist_ff,energy_cpu_pkg_joules,0.05,control_effect,other,control,energy,weak,no,模型控制变量的微弱影响
hyperparam_seed,perf_test_accuracy,0.00,irrelevant,other,hyperparam,performance,zero,no,随机种子不影响性能（预期）
```

#### 关键字段说明

**is_significant** (新增):
- `yes`: 强度>0.1（显著）
- `no`: 强度≤0.1（不显著或无关）
- 用途：快速筛选显著边

**strength_level** (修订):
- `very_strong`: >0.5
- `strong`: 0.3-0.5
- `moderate`: 0.1-0.3
- `weak`: 0.01-0.1
- `very_weak`: 0.001-0.01
- `zero`: <0.001（实际无关）

**edge_type** (扩展):
- `main_effect`: 主效应
- `moderation`: 调节效应 ⭐
- `mediator`: 中介效应
- `control_effect`: 控制变量
- `mode_effect`: is_parallel的直接效应
- `irrelevant`: 无关边（强度≈0）

#### 数据量

- **总行数**: 529行（23变量 × 23变量）
- **有意义的边** (>0.01): 约371行
- **显著边** (>0.1): 约252行
- **强边** (>0.3): 约65行

**文件大小估计**: 约80-100 KB（可接受）

---

### 文件2: causal_paths.csv（间接路径）⭐⭐⭐⭐⭐

#### 方案A: 单文件展平设计 ⭐⭐⭐⭐⭐（推荐）

**格式**:
```csv
path_id,path_length,source,target,path,path_strength,path_type,question_relevance,interpretation
```

**示例**:
```csv
path_id,path_length,source,target,path,path_strength,step1_strength,step2_strength,step3_strength,path_type,question_relevance,interpretation
P001,2,hyperparam_batch_size,energy_gpu_total_joules,batch_size → gpu_max_watts → gpu_total,0.12,0.30,0.40,,mediation_to_energy,Q1-Q3,batch_size通过GPU峰值功率间接影响总能耗
P002,3,hyperparam_batch_size,perf_test_accuracy,batch_size → gpu_max_watts → gpu_util_max → accuracy,0.042,0.30,0.40,0.35,mediation_to_performance,Q3,batch_size通过GPU功率和利用率影响性能
P003,2,hyperparam_epochs,energy_cpu_total_joules,epochs → gpu_temp_max → cpu_total,0.09,0.30,0.30,,mediation_to_energy,Q1-Q3,epochs通过GPU温度间接影响CPU能耗
P004,2,hyperparam_batch_size_x_is_parallel,perf_test_accuracy,batch_size_x_parallel → gpu_util_max → accuracy,0.105,0.30,0.35,,moderation_mediated,Q1-Q3,并行调节效应通过GPU利用率影响性能
```

**关键字段**:
- `path_id`: 路径唯一标识（P001, P002...）
- `path_length`: 路径步数（2=两步，3=三步）
- `path`: 人类可读路径（简化变量名）
- `path_strength`: 路径总强度（各步强度相乘）
- `step1_strength`, `step2_strength`, `step3_strength`: 各步强度
- `path_type`: 路径类型
  - `mediation_to_energy`: 超参数→中介→能耗
  - `mediation_to_performance`: 超参数→中介→性能
  - `moderation_mediated`: 调节效应通过中介
  - `energy_perf_mediated`: 能耗和性能的中介路径

**数据量**:
- **2步路径**: 约50-100条（显著路径）
- **3步路径**: 约20-50条（显著路径）
- **总行数**: 约100-200行

**文件大小估计**: 约20-30 KB

#### 方案B: 分层展示设计 ⭐⭐⭐⭐

**格式**（更直观，但占用空间稍大）:
```csv
path_id,path_length,source,target,step1_source,step1_target,step1_strength,step2_source,step2_target,step2_strength,step3_source,step3_target,step3_strength,path_strength,path_type,interpretation
```

**示例**:
```csv
path_id,path_length,source,target,step1_source,step1_target,step1_strength,step2_source,step2_target,step2_strength,step3_source,step3_target,step3_strength,path_strength,path_type,interpretation
P001,2,hyperparam_batch_size,energy_gpu_total_joules,hyperparam_batch_size,energy_gpu_max_watts,0.30,energy_gpu_max_watts,energy_gpu_total_joules,0.40,,,0.12,mediation_to_energy,batch_size→GPU峰值功率→总能耗
P002,3,hyperparam_batch_size,perf_test_accuracy,hyperparam_batch_size,energy_gpu_max_watts,0.30,energy_gpu_max_watts,energy_gpu_util_max_percent,0.40,energy_gpu_util_max_percent,perf_test_accuracy,0.35,0.042,mediation_to_performance,batch_size→功率→利用率→性能
```

**优势**:
- ✅ 每一步都有明确的源和目标
- ✅ 易于筛选（如找出包含某个中介变量的所有路径）

**劣势**:
- ⚠️ 列数较多（对于3步路径需要13列）

#### 方案C: JSON嵌套设计 ⭐⭐⭐

**格式** (在`analysis_summary.json`中):
```json
"causal_paths": {
  "two_step": [
    {
      "path_id": "P001",
      "source": "hyperparam_batch_size",
      "target": "energy_gpu_total_joules",
      "steps": [
        {"from": "hyperparam_batch_size", "to": "energy_gpu_max_watts", "strength": 0.30},
        {"from": "energy_gpu_max_watts", "to": "energy_gpu_total_joules", "strength": 0.40}
      ],
      "path_strength": 0.12,
      "path_type": "mediation_to_energy",
      "interpretation": "batch_size通过GPU峰值功率间接影响总能耗"
    }
  ],
  "three_step": [...]
}
```

**优势**:
- ✅ 结构清晰，易于程序化访问
- ✅ 适合深度分析

**劣势**:
- ⚠️ 不能用Excel直接查看

---

### 推荐方案组合 ⭐⭐⭐⭐⭐

**CSV使用方案A**（单文件展平，人类可读）:
- `causal_paths.csv`: 使用`path`列展示完整路径（如"A → B → C"）
- 优点：Excel友好，直观
- 同时保留各步强度（step1_strength, step2_strength等）

**JSON使用方案C**（嵌套结构，程序化访问）:
- 在`analysis_summary.json`中包含详细路径信息
- 优点：适合编程分析

**两者互补**:
- 快速查看 → CSV
- 深度分析 → JSON

---

### 文件3: analysis_summary.json（增强版）⭐⭐⭐⭐

#### 新增字段

```json
{
  "task_id": "group1_examples",
  "n_samples": 304,
  "n_features": 23,

  // ========== 直接边信息 ==========

  // ⭐ 所有边（包括零边）
  "all_edges": [
    {
      "source": "hyperparam_batch_size",
      "target": "energy_cpu_total_joules",
      "strength": 0.00,
      "edge_type": "main_effect",
      "is_significant": false
    },
    ...
  ],

  // ⭐ 按强度分层（新增"零边"层）
  "edges_by_strength": {
    "very_strong": {
      "count": 12,
      "threshold": ">0.5",
      "edges": [...]
    },
    "strong": {
      "count": 53,
      "threshold": "0.3-0.5",
      "edges": [...]
    },
    "moderate": {
      "count": 187,
      "threshold": "0.1-0.3",
      "edges": [...]
    },
    "weak": {
      "count": 119,
      "threshold": "0.01-0.1",
      "edges": [...]
    },
    "very_weak": {
      "count": 30,
      "threshold": "0.001-0.01",
      "edges": [...]
    },
    "zero": {
      "count": 128,
      "threshold": "<0.001",
      "edges": []  // 不保存详细信息，仅统计
    }
  },

  // 按类型分类
  "edges_by_type": {
    "hyperparam_to_energy": [...],
    "interaction_to_energy": [...],  // 调节效应⭐
    "mediator_edges": [...],
    "irrelevant": [...]  // 新增：无关边
  },

  // ========== 间接路径信息 ==========

  // ⭐ 两步路径
  "causal_paths_2step": [
    {
      "path_id": "P001",
      "source": "hyperparam_batch_size",
      "target": "energy_gpu_total_joules",
      "steps": [
        {"from": "hyperparam_batch_size", "to": "energy_gpu_max_watts", "strength": 0.30},
        {"from": "energy_gpu_max_watts", "to": "energy_gpu_total_joules", "strength": 0.40}
      ],
      "path_strength": 0.12,
      "path_type": "mediation_to_energy",
      "interpretation": "batch_size通过GPU峰值功率间接影响总能耗"
    },
    ...
  ],

  // ⭐ 三步路径
  "causal_paths_3step": [
    {
      "path_id": "P101",
      "source": "hyperparam_batch_size",
      "target": "perf_test_accuracy",
      "steps": [
        {"from": "hyperparam_batch_size", "to": "energy_gpu_max_watts", "strength": 0.30},
        {"from": "energy_gpu_max_watts", "to": "energy_gpu_util_max_percent", "strength": 0.40},
        {"from": "energy_gpu_util_max_percent", "to": "perf_test_accuracy", "strength": 0.35}
      ],
      "path_strength": 0.042,
      "path_type": "mediation_to_performance",
      "interpretation": "batch_size通过GPU功率和利用率影响性能"
    },
    ...
  ],

  // ⭐ 路径统计
  "path_statistics": {
    "total_2step_paths": 87,
    "significant_2step_paths": 52,  // path_strength > 0.05
    "total_3step_paths": 43,
    "significant_3step_paths": 18,
    "max_path_strength_2step": 0.15,
    "max_path_strength_3step": 0.063
  },

  // ⭐ 变量的路径汇总
  "variable_path_summary": {
    "hyperparam_batch_size": {
      "outgoing_2step_paths": 12,  // batch_size作为起点的2步路径数
      "outgoing_3step_paths": 8,
      "strongest_mediation_to_energy": {
        "path": "batch_size → gpu_max_watts → gpu_total",
        "strength": 0.12
      },
      "strongest_mediation_to_performance": {
        "path": "batch_size → gpu_max_watts → gpu_util_max → accuracy",
        "strength": 0.042
      }
    },
    ...
  },

  // 保留原有字段
  "question1_evidence": {...},
  "question2_evidence": {...},
  "question3_evidence": {...}
}
```

---

## 间接边展示方案对比

### 方案对比表

| 方案 | 格式 | 可读性 | Excel友好 | 编程友好 | 文件大小 | 推荐度 |
|------|------|--------|----------|---------|---------|--------|
| **A. 单文件展平** | CSV（path列） | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 约30KB | ⭐⭐⭐⭐⭐ |
| **B. 分层展示** | CSV（多列） | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 约40KB | ⭐⭐⭐⭐ |
| **C. JSON嵌套** | JSON | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | 约20KB | ⭐⭐⭐⭐ |
| **组合方案** | CSV(A) + JSON(C) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 约50KB | ⭐⭐⭐⭐⭐ |

### 最终推荐：组合方案 ⭐⭐⭐⭐⭐

**CSV文件**（方案A，人类可读）:
```csv
path_id,path_length,source,target,path,path_strength,step1_strength,step2_strength,step3_strength,path_type,interpretation
P001,2,batch_size,gpu_total,batch_size → gpu_max_watts → gpu_total,0.12,0.30,0.40,,mediation_to_energy,batch_size通过GPU峰值功率影响总能耗
```

**JSON文件**（方案C，程序化访问）:
```json
"causal_paths_2step": [
  {
    "path_id": "P001",
    "steps": [
      {"from": "hyperparam_batch_size", "to": "energy_gpu_max_watts", "strength": 0.30},
      {"from": "energy_gpu_max_watts", "to": "energy_gpu_total_joules", "strength": 0.40}
    ],
    ...
  }
]
```

**优势**:
- ✅ 快速查看：打开CSV，直接看到`path`列
- ✅ 深度分析：读取JSON，程序化处理路径
- ✅ 互补性强

---

## 实施方案

### 阶段1: 生成所有直接边（2小时）

**输入**:
- `causal_graph.npy` (23×23矩阵)
- `feature_names.json`

**输出**:
- `causal_edges_all.csv` (529行)

**关键函数**:
```python
def generate_all_edges_csv(causal_graph, feature_names):
    """生成所有边的CSV（无筛选）"""
    edges = []
    n = len(feature_names)

    for i in range(n):
        for j in range(n):
            source = feature_names[i]
            target = feature_names[j]
            strength = causal_graph[i, j]

            # 不做筛选，所有边都保存
            edge_type = classify_edge_type(source, target)
            is_significant = 'yes' if strength > 0.1 else 'no'
            strength_level = get_strength_level(strength)

            edges.append({
                'source': source,
                'target': target,
                'strength': strength,
                'edge_type': edge_type,
                'is_significant': is_significant,
                'strength_level': strength_level,
                'source_category': get_category(source),
                'target_category': get_category(target),
                'question_relevance': get_question_relevance(source, target, edge_type),
                'interpretation': generate_interpretation(source, target, strength, edge_type)
            })

    # 转换为DataFrame
    df = pd.DataFrame(edges)

    # 按强度降序排序
    df = df.sort_values('strength', ascending=False)

    return df
```

### 阶段2: 自动发现间接路径（3-4小时）

**算法**: 图遍历（BFS/DFS）

**伪代码**:
```python
def find_causal_paths(causal_graph, feature_names, min_strength=0.05, max_length=3):
    """
    自动发现间接因果路径

    参数:
        min_strength: 路径最小强度阈值（默认0.05）
        max_length: 最大路径长度（默认3步）
    """
    paths_2step = []
    paths_3step = []

    # 转换为邻接表（仅保留显著边）
    graph = build_adjacency_list(causal_graph, feature_names, threshold=0.1)

    # ========== 2步路径搜索 ==========
    for source_idx, source in enumerate(feature_names):
        # 跳过非关键变量（如控制变量）
        if not is_key_variable(source):
            continue

        # 第一步邻居
        for mediator_idx in graph[source_idx]:
            mediator = feature_names[mediator_idx]
            step1_strength = causal_graph[source_idx, mediator_idx]

            # 第二步邻居
            for target_idx in graph[mediator_idx]:
                target = feature_names[target_idx]
                step2_strength = causal_graph[mediator_idx, target_idx]

                # 计算路径强度
                path_strength = step1_strength * step2_strength

                # 过滤弱路径
                if path_strength < min_strength:
                    continue

                # 过滤无意义路径（如 A → B → A）
                if source == target:
                    continue

                # 检查路径类型
                path_type = classify_path_type(source, mediator, target)

                paths_2step.append({
                    'path_id': f'P2_{len(paths_2step)+1:03d}',
                    'path_length': 2,
                    'source': source,
                    'target': target,
                    'mediator': mediator,
                    'path': format_path([source, mediator, target]),
                    'path_strength': path_strength,
                    'step1_strength': step1_strength,
                    'step2_strength': step2_strength,
                    'path_type': path_type,
                    'question_relevance': get_path_question_relevance(source, target, path_type),
                    'interpretation': generate_path_interpretation(source, mediator, target, path_type)
                })

    # ========== 3步路径搜索（类似）==========
    for source_idx, source in enumerate(feature_names):
        if not is_key_variable(source):
            continue

        for m1_idx in graph[source_idx]:
            m1 = feature_names[m1_idx]
            step1_strength = causal_graph[source_idx, m1_idx]

            for m2_idx in graph[m1_idx]:
                m2 = feature_names[m2_idx]
                step2_strength = causal_graph[m1_idx, m2_idx]

                for target_idx in graph[m2_idx]:
                    target = feature_names[target_idx]
                    step3_strength = causal_graph[m2_idx, target_idx]

                    path_strength = step1_strength * step2_strength * step3_strength

                    if path_strength < min_strength or source == target:
                        continue

                    path_type = classify_path_type_3step(source, m1, m2, target)

                    paths_3step.append({
                        'path_id': f'P3_{len(paths_3step)+1:03d}',
                        'path_length': 3,
                        'source': source,
                        'target': target,
                        'mediator1': m1,
                        'mediator2': m2,
                        'path': format_path([source, m1, m2, target]),
                        'path_strength': path_strength,
                        'step1_strength': step1_strength,
                        'step2_strength': step2_strength,
                        'step3_strength': step3_strength,
                        'path_type': path_type,
                        'question_relevance': get_path_question_relevance(source, target, path_type),
                        'interpretation': generate_path_interpretation_3step(source, m1, m2, target, path_type)
                    })

    return paths_2step, paths_3step

def format_path(nodes):
    """格式化路径为可读字符串"""
    # 简化变量名
    simplified = [simplify_var_name(n) for n in nodes]
    return ' → '.join(simplified)

def simplify_var_name(var_name):
    """简化变量名（用于CSV显示）"""
    var_name = var_name.replace('hyperparam_', '')
    var_name = var_name.replace('energy_', '')
    var_name = var_name.replace('_x_is_parallel', '_x_parallel')
    var_name = var_name.replace('perf_', '')
    return var_name

def classify_path_type(source, mediator, target):
    """分类2步路径类型"""
    if 'hyperparam' in source and 'energy_gpu' in mediator and 'energy' in target:
        return 'mediation_to_energy'
    elif 'hyperparam' in source and 'energy' in mediator and 'perf' in target:
        return 'mediation_to_performance'
    elif '_x_is_parallel' in source and 'energy' in mediator:
        return 'moderation_mediated'
    elif 'energy' in source and 'energy' in mediator and 'perf' in target:
        return 'energy_perf_mediated'
    else:
        return 'other_mediation'

def is_key_variable(var_name):
    """判断是否为关键变量（用于路径搜索）"""
    # 关键起点：超参数、交互项
    if 'hyperparam' in var_name or '_x_is_parallel' in var_name:
        return True
    # 排除控制变量、种子等
    if 'model_' in var_name or 'seed' in var_name:
        return False
    return False
```

**关键设计**:
- 只搜索关键变量作为起点（超参数、交互项）
- 过滤弱路径（路径强度<0.05）
- 避免循环路径（A → B → A）
- 自动分类路径类型

### 阶段3: 生成CSV和JSON（1小时）

**CSV生成**:
```python
# 2步路径CSV
df_2step = pd.DataFrame(paths_2step)
df_2step.to_csv('causal_paths.csv', index=False)

# 或分开保存
df_2step.to_csv('causal_paths_2step.csv', index=False)
df_3step = pd.DataFrame(paths_3step)
df_3step.to_csv('causal_paths_3step.csv', index=False)
```

**JSON增强**:
```python
# 在现有result.json基础上添加
result['causal_paths_2step'] = paths_2step
result['causal_paths_3step'] = paths_3step

# 添加路径统计
result['path_statistics'] = {
    'total_2step_paths': len(paths_2step),
    'significant_2step_paths': sum(1 for p in paths_2step if p['path_strength'] > 0.05),
    'total_3step_paths': len(paths_3step),
    'significant_3step_paths': sum(1 for p in paths_3step if p['path_strength'] > 0.05),
    'max_path_strength_2step': max([p['path_strength'] for p in paths_2step]) if paths_2step else 0,
    'max_path_strength_3step': max([p['path_strength'] for p in paths_3step]) if paths_3step else 0
}

with open('analysis_summary.json', 'w') as f:
    json.dump(result, f, indent=2)
```

---

## 使用示例

### 场景1: 查看所有调节效应（包括弱的）

```bash
# Excel
打开 causal_edges_all.csv
筛选: edge_type = "moderation"
排序: strength 降序

# 命令行
cat causal_edges_all.csv | grep "moderation" | sort -t, -k3 -rn
```

### 场景2: 查看batch_size的间接效应

```bash
# Excel
打开 causal_paths.csv
筛选: source 包含 "batch_size"
查看 path 列

# 命令行
cat causal_paths.csv | grep "batch_size" | column -t -s,
```

**输出示例**:
```
path_id  path_length  source           target              path                                      path_strength  interpretation
P001     2            batch_size       gpu_total           batch_size → gpu_max_watts → gpu_total    0.12           batch_size通过GPU峰值功率影响总能耗
P002     3            batch_size       accuracy            batch_size → gpu_max → gpu_util → acc    0.042          batch_size通过功率和利用率影响性能
P003     2            batch_size       cpu_total           batch_size → gpu_temp_avg → cpu_total    0.06           batch_size通过GPU温度影响CPU能耗
```

### 场景3: 比较直接效应和间接效应

```python
import pandas as pd

# 读取直接边
edges = pd.read_csv('causal_edges_all.csv')
direct = edges[(edges['source'] == 'hyperparam_batch_size') &
               (edges['target'] == 'energy_gpu_total_joules')]

# 读取间接路径
paths = pd.read_csv('causal_paths.csv')
indirect = paths[(paths['source'] == 'hyperparam_batch_size') &
                 (paths['target'] == 'energy_gpu_total_joules')]

print(f"直接效应: {direct['strength'].values[0]:.3f}")
print(f"间接效应数量: {len(indirect)}")
print(f"最强间接效应: {indirect['path_strength'].max():.3f}")
print(f"间接路径: {indirect['path'].tolist()}")
```

### 场景4: 验证中介效应假设

**假设**: `epochs` 通过 `gpu_temp_max` 影响 `energy_gpu_total`

```bash
# 检查直接边
cat causal_edges_all.csv | grep "hyperparam_epochs,energy_gpu_total_joules"
# 输出: epochs → gpu_total, strength=0.40 (主效应)

# 检查间接路径
cat causal_paths.csv | grep "epochs" | grep "gpu_temp_max" | grep "gpu_total"
# 输出: epochs → gpu_temp_max → gpu_total, path_strength=0.09 (中介效应)
```

**结论**:
- 直接效应: 0.40（主导）
- 间接效应: 0.09（辅助）
- epochs对能耗的影响主要是直接的，部分通过温度中介

---

## 文件大小估计

| 文件 | 行数 | 列数 | 文件大小 |
|------|------|------|---------|
| `causal_edges_all.csv` | 529 | 10 | 约80-100 KB |
| `causal_paths.csv` | 100-200 | 11 | 约30-40 KB |
| `analysis_summary.json` | - | - | 约50-80 KB |
| `causal_graph.npy` | - | - | 约2 KB |
| **总计** | - | - | **约160-220 KB** |

**评价**: 文件大小完全可接受（每个任务组<300 KB）

---

## 总结

### 核心改进

✅ **需求1满足**: 间接边可读性
- 专用文件 `causal_paths.csv` 展示所有2步和3步路径
- `path` 列直观显示完整因果链（如"A → B → C"）
- 自动计算路径强度
- 无需手动拼接

✅ **需求2满足**: 不做筛选
- `causal_edges_all.csv` 包含所有529条边
- 包括强度=0.00的边
- 用户可自行筛选（Excel筛选器）

### 最终文件结构

```
group1_examples_causal_edges_all.csv     ⭐⭐⭐⭐⭐ 所有直接边（529行，无筛选）
group1_examples_causal_paths.csv         ⭐⭐⭐⭐⭐ 间接路径（100-200行，2步+3步）
group1_examples_analysis_summary.json    ⭐⭐⭐⭐  增强摘要（包含路径详情）
group1_examples_causal_graph.npy         ⭐⭐     原始矩阵（备用）
```

### 实施优先级

⭐⭐⭐⭐⭐ **立即实施**（5-6小时）:
1. 所有直接边CSV生成（2小时）
2. 间接路径自动发现算法（3-4小时）
3. JSON增强和验证（1小时）

---

**下一步**: 等待用户确认方案，然后开始实施！
