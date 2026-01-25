# DiBS场景差异建模策略

**创建日期**: 2026-01-11
**文档版本**: v1.0
**状态**: 📋 方案设计

---

## 📋 问题背景

**研究目标**: 在DiBS因果图学习中，强调训练场景（并行/非并行）的差异性

**数据现状**:
- 总实验数: 970个
- 能耗+性能完整: 717个 (73.9%)
- 6个任务组 × 2个场景 = 12个子组

**核心挑战**:
- 12分组样本量不足（平均59.8个/组）
- 6分组无法直接区分场景差异
- 需要在样本量和场景区分度之间平衡

---

## 🎯 方案总览

| 方案 | 核心思路 | 数据分组 | 样本量/组 | 场景建模方式 | 复杂度 |
|------|---------|---------|-----------|-------------|--------|
| **方案1: 12分组** | 完全分离 | 12组 | 59.8 | 独立因果图 | ⭐ |
| **方案2: 6分组+协变量** | 场景作为特征 | 6组 | 119.5 | 协变量 | ⭐ |
| **方案3: 6分组+分层DiBS** | 场景分层分析 | 6×2=12 | 59.8 | 对比因果图 | ⭐⭐ |
| **方案4: 场景交互特征** | 特征工程 | 6组 | 119.5 | 交互项 | ⭐⭐ |
| **方案5: 联合因果发现** | 多域学习 | 6组 | 119.5 | 共享+特定边 | ⭐⭐⭐ |
| **方案6: 差分因果分析** | 场景对比 | 6组 | 119.5 | 差分图 | ⭐⭐⭐ |

---

## 📊 方案详解

### 方案1: 12分组独立DiBS ⭐ 基础方案

#### 核心思路
- 将数据完全分为12个独立组（6任务组 × 2场景）
- 对每组独立运行DiBS
- 人工对比12个因果图，识别场景差异

#### 数据分组
```
group1a_examples_nonparallel    (126个)
group1a_examples_parallel       (119个)
group1b_resnet_nonparallel      (31个)  ⚠️ 样本不足
group1b_resnet_parallel         (30个)  ⚠️ 样本不足
...（12个子组）
```

#### 优势
- ✅ **概念简单**: 每组独立分析，易于理解
- ✅ **场景完全分离**: 不存在场景混淆
- ✅ **灵活性高**: 可以针对每组调整DiBS参数

#### 劣势
- ❌ **样本量不足**: 平均59.8个/组，低于推荐水平（100个）
- ❌ **统计功效低**: 4个组样本量<50，DiBS结果不稳定
- ❌ **对比困难**: 需要人工对比12个因果图，缺乏系统性
- ❌ **需要更多数据**: 建议补充约483个实验

#### 适用场景
- 数据充足时（>1200个实验）
- 对场景差异高度关注
- 可接受较低的统计功效

---

### 方案2: 6分组 + 场景作为协变量 ⭐ 推荐方案

#### 核心思路
- 保持6个任务组（不区分场景）
- 将 `is_parallel` 作为DiBS的输入变量之一
- DiBS学习包含场景变量的因果图

#### 数据结构
```python
# 每个任务组的特征向量
features = [
    'hyperparam_learning_rate',
    'hyperparam_batch_size',
    'hyperparam_epochs',
    'is_parallel',              # ⭐ 场景作为二值协变量
    ...
]

# 目标变量
targets = [
    'energy_gpu_total_joules',
    'perf_accuracy',
    ...
]
```

#### DiBS输入示例
```python
# 以group1a_examples为例
X = np.array([
    # 实验1: [lr, bs, epochs, is_parallel, ...]
    [0.001, 32, 10, 0, ...],  # 非并行
    [0.001, 64, 10, 1, ...],  # 并行
    ...
])

Y = np.array([
    # 实验1: [energy, accuracy, ...]
    [5000, 0.98, ...],
    [6000, 0.98, ...],
    ...
])
```

#### 因果图解读
DiBS会学习到：
```
is_parallel → energy_gpu_total_joules  (场景对能耗的直接影响)
is_parallel → learning_rate → energy  (场景调制学习率的影响)
```

#### 优势
- ✅ **样本量充足**: 每组119.5个，达到推荐水平
- ✅ **统计功效高**: DiBS结果稳定可靠
- ✅ **可识别场景效应**: 场景变量的因果边代表场景影响
- ✅ **立即可用**: 当前数据无需补充
- ✅ **简单实现**: 只需添加is_parallel列

#### 劣势
- ⚠️ **场景效应假设**: 假设场景对所有任务的影响是相似的（可能不成立）
- ⚠️ **因果结构共享**: 假设并行和非并行的因果结构大致相同
- ⚠️ **交互效应隐含**: 场景与参数的交互效应不直接可见

#### 适用场景 ⭐⭐⭐
- **当前数据状况**（样本量有限）
- 希望快速获得结果
- 对场景效应的基本了解
- **强烈推荐作为第一步分析**

---

### 方案3: 6分组 + 分层DiBS ⭐⭐ 进阶方案

#### 核心思路
- 对每个任务组，分别对并行和非并行数据运行DiBS
- 得到12个因果图（6组 × 2场景）
- 系统化对比同一任务组的两个因果图，识别场景差异

#### 分析流程
```python
# 步骤1: 对每个任务组分层运行DiBS
for task_group in ['group1a_examples', 'group1b_resnet', ...]:
    # 非并行数据
    data_nonparallel = filter(data, group=task_group, is_parallel=0)
    graph_nonparallel = run_dibs(data_nonparallel)

    # 并行数据
    data_parallel = filter(data, group=task_group, is_parallel=1)
    graph_parallel = run_dibs(data_parallel)

    # 步骤2: 对比因果图差异
    diff = compare_graphs(graph_nonparallel, graph_parallel)

    # 步骤3: 识别场景特定的因果边
    scenario_specific_edges = identify_diff_edges(diff)
```

#### 因果图对比指标
```python
def compare_graphs(G_nonparallel, G_parallel):
    """
    对比两个因果图的差异

    返回:
        - shared_edges: 两个场景共享的因果边
        - nonparallel_only: 仅在非并行中存在的边
        - parallel_only: 仅在并行中存在的边
        - strength_diff: 边强度的差异
    """
    shared_edges = set(G_nonparallel.edges) & set(G_parallel.edges)
    nonparallel_only = set(G_nonparallel.edges) - set(G_parallel.edges)
    parallel_only = set(G_parallel.edges) - set(G_nonparallel.edges)

    strength_diff = {}
    for edge in shared_edges:
        strength_diff[edge] = abs(
            G_nonparallel[edge]['weight'] - G_parallel[edge]['weight']
        )

    return {
        'shared': shared_edges,
        'nonparallel_only': nonparallel_only,
        'parallel_only': parallel_only,
        'strength_diff': strength_diff
    }
```

#### 样本量分配
| 任务组 | 非并行样本 | 并行样本 | 总计 |
|-------|-----------|---------|------|
| group1a_examples | 126 ✅ | 119 ✅ | 245 |
| group1b_resnet | 31 ❌ | 30 ❌ | 61 |
| group2_person_reid | 93 ⚠️ | 90 ⚠️ | 183 |
| group3_vulberta | 25 ❌ | 47 ❌ | 72 |
| group4_bug_localization | 25 ❌ | 65 ⚠️ | 90 |
| group5_mrt_oast | 36 ❌ | 30 ❌ | 66 |

#### 优势
- ✅ **场景因果结构清晰**: 明确区分场景特定的因果关系
- ✅ **系统化对比**: 可量化场景差异（边的数量、强度）
- ✅ **灵活性**: 每个场景可独立调参
- ✅ **解释性强**: "在并行模式下，learning_rate对能耗的影响更强"

#### 劣势
- ❌ **样本量减半**: 每个子场景样本量不足（5个组<100）
- ❌ **结果不稳定**: 小样本DiBS可能出现虚假边
- ❌ **对比复杂性**: 需要设计边对比的统计显著性检验
- ❌ **需要补充数据**: 建议每组至少100个样本（需补充数据）

#### 适用场景
- 数据充足后（每个子场景>100个样本）
- 对场景差异的深入探索
- 有足够的分析时间和资源

---

### 方案4: 场景交互特征工程 ⭐⭐ 创新方案

#### 核心思路
- 创建场景交互特征（超参数 × 场景指示变量）
- 扩展特征空间，让DiBS学习场景调制的因果关系
- 保持6分组，样本量充足

#### 特征构造
```python
def create_scenario_interaction_features(df):
    """
    创建场景交互特征

    示例:
        learning_rate = 0.001
        is_parallel = 1
        →
        learning_rate_parallel = 0.001 * 1 = 0.001
        learning_rate_nonparallel = 0.001 * 0 = 0.0
    """
    hyperparams = ['learning_rate', 'batch_size', 'epochs', 'weight_decay']

    for param in hyperparams:
        # 并行场景下的参数值
        df[f'{param}_parallel'] = df[param] * df['is_parallel']

        # 非并行场景下的参数值
        df[f'{param}_nonparallel'] = df[param] * (1 - df['is_parallel'])

    return df
```

#### DiBS输入示例
```python
# 原始特征
X_original = [
    'learning_rate',
    'batch_size',
    'epochs',
    'is_parallel'
]

# 扩展后的特征
X_expanded = [
    'learning_rate_parallel',      # ⭐ 新特征
    'learning_rate_nonparallel',   # ⭐ 新特征
    'batch_size_parallel',
    'batch_size_nonparallel',
    'epochs_parallel',
    'epochs_nonparallel',
    'is_parallel'                  # 保留原始场景变量
]
```

#### 因果图解读
DiBS可能学习到：
```
learning_rate_parallel → energy_gpu_total_joules  (并行场景的学习率影响)
learning_rate_nonparallel → energy_gpu_total_joules  (非并行场景的学习率影响)

# 对比边的强度，识别场景调制效应
if weight(learning_rate_parallel → energy) > weight(learning_rate_nonparallel → energy):
    → "并行模式下，学习率对能耗的影响更强"
```

#### 优势
- ✅ **样本量充足**: 保持6分组（119.5个/组）
- ✅ **场景调制显式**: 交互特征直接捕捉场景差异
- ✅ **统计显著性**: 可比较两个场景特征的因果边强度
- ✅ **立即可用**: 当前数据无需补充
- ✅ **灵活扩展**: 可以选择性创建关键参数的交互特征

#### 劣势
- ⚠️ **特征冗余**: 特征数量翻倍（可能影响DiBS收敛）
- ⚠️ **多重共线性**: 交互特征与原始特征高度相关
- ⚠️ **解释复杂**: 因果图变得更复杂
- ⚠️ **需要特征选择**: 可能需要先用回归分析筛选关键交互特征

#### 适用场景
- 样本量有限但希望探索场景差异
- 对特定超参数的场景调制感兴趣
- 有一定的因果推断分析经验

---

### 方案5: 联合因果发现（Multi-Domain Causal Discovery） ⭐⭐⭐ 高级方案

#### 核心思路
- 使用多域因果发现算法（如CD-NOD, JCI等）
- 同时学习并行和非并行场景的因果图
- 自动识别：共享因果边 + 场景特定因果边

#### 方法原理
```
传统DiBS:
    单一因果图 G

多域DiBS:
    共享因果图 G_shared（两个场景都存在的边）
    场景特定图 G_nonparallel（仅非并行）
    场景特定图 G_parallel（仅并行）

最终因果图:
    G_nonparallel = G_shared ∪ G_nonparallel_specific
    G_parallel = G_shared ∪ G_parallel_specific
```

#### 算法选择
| 算法 | 优势 | 实现 |
|------|------|------|
| **CD-NOD** | 明确区分共享/特定边 | Python包: causal-learn |
| **JCI (Joint Causal Inference)** | 利用多域信息提升统计功效 | R包: JCI |
| **Multi-Domain DiBS** | DiBS的多域扩展 | 需自行实现（基于DiBS源码） |

#### 实现示例（CD-NOD）
```python
from causallearn.search.ConstraintBased.CDNOD import cdnod

# 准备数据
data_domains = {
    'nonparallel': df[df['is_parallel'] == 0][feature_cols],
    'parallel': df[df['is_parallel'] == 1][feature_cols]
}

# 运行CD-NOD
result = cdnod(
    data_list=[data_domains['nonparallel'], data_domains['parallel']],
    c_indx=None,  # 所有变量都是可能因果的
    alpha=0.05    # 显著性水平
)

# 提取结果
G_shared = result['shared_graph']        # 共享因果边
G_nonparallel = result['domain_graphs'][0]  # 非并行特定
G_parallel = result['domain_graphs'][1]     # 并行特定
```

#### 优势
- ✅ **样本量利用最优**: 共享边利用全部数据（119.5个/组）
- ✅ **统计功效高**: 多域信息互补
- ✅ **自动识别差异**: 算法自动区分共享/特定边
- ✅ **理论保证**: 基于统计因果推断理论

#### 劣势
- ❌ **算法复杂**: 需要专门的多域因果发现算法
- ❌ **实现难度**: DiBS没有官方多域版本，需自行实现或使用替代算法
- ❌ **调参复杂**: 需要调整共享/特定边的先验权重
- ❌ **解释困难**: 需要深入理解多域因果推断

#### 适用场景
- 有因果推断研究经验
- 愿意探索前沿方法
- 对研究质量要求高
- **推荐作为后续深入研究的方向**

---

### 方案6: 差分因果分析（Differential Causal Analysis） ⭐⭐⭐ 研究方案

#### 核心思路
- 不直接学习两个场景的因果图
- 学习"场景差异"的因果图
- 输入变量：场景之间的差值（Δ能耗、Δ性能等）

#### 数据构造
```python
def create_differential_data(df):
    """
    创建差分数据（配对实验）

    要求: 同一超参数配置在两个场景都有实验
    """
    # 找到配对实验
    paired_experiments = df.groupby(['model', 'hyperparam_config']).filter(
        lambda x: (x['is_parallel'] == 0).any() and (x['is_parallel'] == 1).any()
    )

    # 计算差值
    diff_data = []
    for (model, config), group in paired_experiments.groupby(['model', 'hyperparam_config']):
        nonparallel = group[group['is_parallel'] == 0].iloc[0]
        parallel = group[group['is_parallel'] == 1].iloc[0]

        diff_row = {
            'model': model,
            'Δenergy': parallel['energy'] - nonparallel['energy'],
            'Δaccuracy': parallel['accuracy'] - nonparallel['accuracy'],
            'Δduration': parallel['duration'] - nonparallel['duration'],
            # 超参数保持不变（作为协变量）
            'learning_rate': nonparallel['learning_rate'],
            'batch_size': nonparallel['batch_size'],
            ...
        }
        diff_data.append(diff_row)

    return pd.DataFrame(diff_data)
```

#### DiBS输入
```python
# 特征: 超参数（配对实验的配置）
X = [
    'learning_rate',
    'batch_size',
    'epochs',
    ...
]

# 目标: 场景差异（并行 - 非并行）
Y = [
    'Δenergy_gpu',      # 并行相对非并行的能耗增量
    'Δaccuracy',        # 性能差异
    'Δduration'         # 时长差异
]
```

#### 因果图解读
DiBS学习到：
```
learning_rate → Δenergy_gpu  (学习率对场景能耗差异的影响)
batch_size → Δenergy_gpu

# 解读:
if learning_rate → Δenergy_gpu 的边存在且为正:
    → "学习率越高，并行相对非并行的能耗增量越大"
    → "即：并行模式下，学习率对能耗的影响更敏感"
```

#### 优势
- ✅ **直接建模差异**: 因果图直接表示场景效应
- ✅ **控制混淆**: 配对设计天然控制实验间差异
- ✅ **解释直观**: "哪些参数影响场景差异"
- ✅ **样本量**: 取决于配对实验数量

#### 劣势
- ❌ **需要配对实验**: 当前数据可能没有足够的配对实验
- ❌ **样本量损失**: 只能使用配对的实验（可能<50%）
- ❌ **信息损失**: 丢失了单个场景的绝对因果关系
- ❌ **适用性有限**: 仅适合研究场景差异，不适合研究绝对效应

#### 适用场景
- 专门研究场景差异的论文
- 有大量配对实验
- 对差异因果建模感兴趣

---

## 🎯 方案推荐与实施路径

### 阶段1: 立即可用（当前数据，无需补充）⭐⭐⭐

#### 推荐方案: **方案2（6分组+协变量）+ 方案4（场景交互特征）**

**实施步骤**:
```python
# 步骤1: 方案2 - 场景作为协变量
data = load_data()
for task_group in task_groups:
    X = data[task_group][['learning_rate', 'batch_size', 'epochs', 'is_parallel']]
    Y = data[task_group][['energy_gpu', 'accuracy']]
    graph = run_dibs(X, Y)
    analyze_scenario_edges(graph, 'is_parallel')

# 步骤2: 方案4 - 场景交互特征（关键参数）
# 先用回归分析筛选哪些参数的场景交互显著
interaction_params = ['learning_rate', 'batch_size']  # 假设这两个显著
data_expanded = create_interactions(data, interaction_params)
for task_group in task_groups:
    X = data_expanded[task_group]  # 包含交互特征
    Y = data_expanded[task_group][['energy_gpu', 'accuracy']]
    graph = run_dibs(X, Y)
    compare_scenario_effects(graph)
```

**预期成果**:
- ✅ 识别场景对能耗/性能的直接影响
- ✅ 识别哪些参数受场景调制
- ✅ 量化场景调制的强度

**时间估计**: 1-2周（包括数据准备、DiBS运行、结果分析）

---

### 阶段2: 数据补充后（建议补充到1200个实验）⭐⭐

#### 推荐方案: **方案3（6分组+分层DiBS）**

**实施步骤**:
```python
# 补充数据后，每个子场景至少100个样本
for task_group in task_groups:
    # 非并行DiBS
    data_nonparallel = filter(data, group=task_group, is_parallel=0)
    graph_nonparallel = run_dibs(data_nonparallel)

    # 并行DiBS
    data_parallel = filter(data, group=task_group, is_parallel=1)
    graph_parallel = run_dibs(data_parallel)

    # 系统化对比
    diff = compare_graphs_statistical(graph_nonparallel, graph_parallel)
    visualize_graph_diff(graph_nonparallel, graph_parallel, diff)
```

**预期成果**:
- ✅ 明确区分场景特定的因果结构
- ✅ 量化场景对因果边的影响
- ✅ 高质量的科研结果

**时间估计**: 2-3周（补充实验）+ 1-2周（DiBS分析）

---

### 阶段3: 深入研究（可选，前沿方法）⭐⭐⭐

#### 推荐方案: **方案5（联合因果发现）**

**实施步骤**:
```python
# 使用CD-NOD或自行实现Multi-Domain DiBS
from causallearn.search.ConstraintBased.CDNOD import cdnod

result = cdnod(
    data_list=[data_nonparallel, data_parallel],
    alpha=0.05
)

# 分析共享vs特定边
analyze_shared_edges(result['shared_graph'])
analyze_specific_edges(result['domain_graphs'])
```

**预期成果**:
- ✅ 顶级会议/期刊论文
- ✅ 方法论创新（Multi-Domain DiBS for Energy Analysis）
- ✅ 更深入的因果机制理解

**时间估计**: 1-2个月（算法实现和调试）

---

## 📊 方案对比总结

| 维度 | 方案1 | 方案2 | 方案3 | 方案4 | 方案5 | 方案6 |
|------|-------|-------|-------|-------|-------|-------|
| **样本量充足性** | ❌ | ✅ | ❌ | ✅ | ✅ | ⚠️ |
| **当前可用** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **场景区分度** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **实现难度** | ⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **结果解释性** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **科研价值** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🎯 最终建议

### 短期（当前数据）：**方案2 + 方案4**
- 立即可用，样本量充足
- 可以初步识别场景效应
- 为后续分析奠定基础

### 中期（补充数据后）：**方案3**
- 明确区分场景因果结构
- 高质量的科研结果
- 适合论文发表

### 长期（研究方法）：**方案5**
- 前沿方法探索
- 方法论创新
- 顶级期刊论文

---

## 📚 参考文献

1. **CD-NOD**: Huang, B., et al. (2020). "Causal Discovery from Heterogeneous/Nonstationary Data." JMLR.
2. **Multi-Domain Causal Discovery**: Mooij, J., et al. (2020). "Joint Causal Inference from Multiple Contexts." JMLR.
3. **DiBS**: Lorch, L., et al. (2021). "DiBS: Differentiable Bayesian Structure Learning." NeurIPS.
4. **Scenario Heterogeneity**: Pearl, J., & Bareinboim, E. (2014). "External Validity." Statistical Science.

---

**维护者**: Green + Claude
**文档版本**: v1.0
**最后更新**: 2026-01-11
**状态**: 📋 方案设计完成，待用户选择
