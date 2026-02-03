# RQ1/RQ2分析方案：基于DiBS强度网络（不使用ATE）

**版本**: v1.0 | **创建**: 2026-02-03 | **依据**: 用户决策 + 数据质量分析

> **核心决策**: 放弃ATE分析，使用DiBS强度(≥0.3)+显著性作为主要分析指标

---

## 🎯 分析策略调整

### 放弃ATE的原因
1. **系统性数据问题**: 60%的边缺少ATE值（特别是能耗→性能边）
2. **技术限制**: ATE计算方法无法处理性能指标作为结果变量
3. **决策依据**: 用户明确要求不使用ATE和边强度进行RQ1/RQ2分析

### 替代方案：DiBS强度网络分析
- **主要指标**: DiBS后验概率强度（strength）
- **阈值选择**: ≥0.3（与参考论文一致）
- **筛选条件**: 强度≥0.3 + 统计显著（is_significant=True）
- **分析焦点**: 因果路径而非效应量大小

---

## 📊 数据基础验证

### 1. DiBS强度分布（原始数据）
**文件**: `analysis/results/energy_research/data/interaction/raw/*_causal_edges_all.csv`

**group1强度值示例**:
```
0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, ...
```

### 2. ATE过滤逻辑确认
**脚本**: `analysis/scripts/compute_ate_dibs_global_std.py:123`
```python
def compute_ate_for_group(..., threshold: float = 0.3):
    # 第176行：统计>0.3的边
    n_edges = np.sum(causal_graph > threshold)
    # 第261行：实际过滤
    if weight > threshold:  # 严格大于0.3
```

**结论**: ATE文件中边强度≥0.35是**脚本过滤**的结果，不是DiBS原始输出特性

---

## 🔧 RQ1分析方案：超参数到能耗的因果路径

### 分析目标
识别超参数（含交互项）到能耗指标的所有直接和间接因果路径

### 输入数据
1. **全局标准化ATE文件**（仅用强度列）:
   - `analysis/results/energy_research/data/global_std_dibs_ate/*.csv`
   - 列：source, target, strength, is_significant, ...
2. **原始DiBS边文件**（作为补充）:
   - `analysis/results/energy_research/data/interaction/raw/*_causal_edges_all.csv`

### 筛选条件
```python
# 核心筛选条件
strength_threshold = 0.3
is_significant = True  # 对应原始强度>0.1

# 变量类型过滤
source_patterns = ['hyperparam_', '_x_is_parallel']  # 超参数和交互项
target_patterns = ['energy_']  # 能耗指标
```

### 分析步骤

#### 步骤1：直接边提取
```python
# 提取所有强度≥0.3且显著的超参数→能耗边
direct_edges = edges[
    (edges['strength'] >= 0.3) &
    (edges['is_significant'] == True) &
    (edges['source'].str.contains('hyperparam_|_x_is_parallel')) &
    (edges['target'].str.startswith('energy_'))
]
```

#### 步骤2：间接路径搜索（2跳）
搜索模式：超参数 → 能耗中介 → 目标能耗

**算法**:
1. 找到所有超参数→能耗中介的边（强度≥0.3）
2. 找到所有能耗中介→目标能耗的边（强度≥0.3）
3. 组合形成2跳路径
4. 计算路径强度：`path_strength = strength1 × strength2`

#### 步骤3：间接路径搜索（3跳）
搜索模式：超参数 → 中介1 → 中介2 → 目标能耗

#### 步骤4：交互项效应分析
区分三种效应：
1. **基础超参数效应**: `hyperparam_*` → `energy_*`
2. **交互项直接效应**: `hyperparam_*_x_is_parallel` → `energy_*`
3. **交互项调节路径**: `hyperparam_*_x_is_parallel` → 中介 → `energy_*`

### 输出结果

#### 表格输出
1. **RQ1_direct_edges.csv** (直接边)
   - source, target, strength, edge_type, interpretation
   - 按strength降序排列

2. **RQ1_indirect_paths_2step.csv** (2跳路径)
   - path_id, source, target, mediator, path_strength, step1_strength, step2_strength
   - 按path_strength降序排列

3. **RQ1_indirect_paths_3step.csv** (3跳路径)
   - path_id, source, target, mediator1, mediator2, path_strength, step1_strength, step2_strength, step3_strength

4. **RQ1_interaction_effects.csv** (交互项效应)
   - effect_type, source, target, strength, interpretation
   - 分类：direct_interaction, mediated_interaction, moderation_path

#### 可视化输出
1. **RQ1_direct_network.png**: 直接因果网络图
2. **RQ1_path_strength_distribution.png**: 路径强度分布图
3. **RQ1_interaction_vs_base_comparison.png**: 交互项vs基础效应对比

---

## 🔬 RQ2分析方案：能耗网络中介机制

### 分析目标
通过能耗变量间的因果网络，解释超参数效应的传递机制

### 核心概念：能耗中介网络
```
超参数 → [能耗中介网络] → 总能耗指标
           ↑        ↑
      GPU利用率  GPU温度
           ↓        ↓
      GPU功耗  → 总能耗
```

### 分析步骤

#### 步骤1：能耗网络构建
提取所有energy_→energy_的因果边（强度≥0.3）

```python
energy_network_edges = edges[
    (edges['strength'] >= 0.3) &
    (edges['is_significant'] == True) &
    (edges['source'].str.startswith('energy_')) &
    (edges['target'].str.startswith('energy_')) &
    (edges['source'] != edges['target'])
]
```

#### 步骤2：网络指标计算
1. **节点中心性**:
   - 度中心性（degree centrality）
   - 中介中心性（betweenness centrality）
   - 接近中心性（closeness centrality）

2. **网络属性**:
   - 连通分量（connected components）
   - 聚类系数（clustering coefficient）
   - 平均路径长度（average path length）

#### 步骤3：关键中介识别
基于中介中心性，识别高中介性节点：
1. **计算每个能耗节点的中介中心性**
2. **排序选取Top 5关键中介**
3. **分析这些节点在网络中的位置和作用**

#### 步骤4：典型传递路径分析
对每个关键中介，分析典型传递路径：
1. **上游路径**: 超参数 → 关键中介
2. **下游路径**: 关键中介 → 总能耗指标
3. **完整路径**: 超参数 → 关键中介 → 总能耗

#### 步骤5：机制解释构建
基于网络结构，构建机制解释：
1. **温度调节路径**: 超参数 → GPU温度 → GPU功耗 → 总能耗
2. **利用率传递路径**: 超参数 → GPU利用率 → GPU功耗 → 总能耗
3. **跨组件传递**: 超参数 → CPU功耗 → GPU功耗 → 总能耗

### 输出结果

#### 表格输出
1. **RQ2_energy_network_edges.csv** (能耗网络边)
   - source, target, strength, edge_type, interpretation

2. **RQ2_node_centrality.csv** (节点中心性)
   - node, degree_centrality, betweenness_centrality, closeness_centrality, mediator_rank

3. **RQ2_key_mediators.csv** (关键中介)
   - mediator, betweenness_centrality, upstream_edges_count, downstream_edges_count, example_path

4. **RQ2_mediation_mechanisms.csv** (中介机制)
   - mechanism_name, key_mediator, typical_path, path_strength, interpretation

#### 可视化输出
1. **RQ2_energy_network_graph.png**: 能耗因果网络图（节点大小=中介中心性）
2. **RQ2_centrality_distribution.png**: 节点中心性分布图
3. **RQ2_mediation_pathways.png**: 关键中介路径示意图

---

## ⚙️ 技术实现细节

### 依赖库
```python
import pandas as pd
import networkx as nx  # 网络分析
import matplotlib.pyplot as plt  # 可视化
import numpy as np
```

### 核心函数设计

#### 1. 数据加载与清洗
```python
def load_and_filter_edges(threshold=0.3):
    """加载并过滤因果边"""
    # 从ATE文件或原始DiBS文件加载
    # 应用强度阈值和显著性筛选
    # 返回过滤后的DataFrame
```

#### 2. 间接路径搜索
```python
def find_indirect_paths(edges_df, max_hops=3):
    """搜索间接因果路径"""
    # 构建因果图
    # 搜索2跳和3跳路径
    # 计算路径强度
    # 返回路径DataFrame
```

#### 3. 网络分析
```python
def analyze_energy_network(energy_edges):
    """分析能耗网络"""
    # 构建网络图
    # 计算网络指标
    # 识别关键中介
    # 返回分析结果
```

#### 4. 可视化生成
```python
def visualize_causal_network(edges_df, output_path):
    """可视化因果网络"""
    # 使用networkx和matplotlib
    # 节点颜色=变量类型
    # 边宽度=强度
    # 保存为PNG
```

### 文件结构
```
analysis/scripts/rq_analysis_no_ate/
├── __init__.py
├── data_loader.py          # 数据加载
├── path_finder.py          # 路径搜索
├── network_analyzer.py     # 网络分析
├── visualizer.py           # 可视化
├── config.py               # 配置参数
└── main.py                 # 主程序
```

---

## 📅 实施计划

### Phase 1: 框架搭建（1天）
1. 创建脚本目录结构
2. 实现数据加载模块
3. 编写基本测试

### Phase 2: RQ1实现（2天）
1. 直接边提取和分类
2. 间接路径搜索算法
3. 交互项效应分析
4. 结果输出和可视化

### Phase 3: RQ2实现（2天）
1. 能耗网络构建
2. 网络指标计算
3. 关键中介识别
4. 机制解释构建

### Phase 4: 整合验证（1天）
1. 整合两个RQ的分析
2. 验证结果一致性
3. 生成综合报告

**总计**: 6天

---

## ⚠️ 注意事项

### 数据限制处理
1. **强度阈值主观性**: 进行敏感性分析（0.3 vs 0.35 vs 0.4）
2. **样本量差异**: 主要分析group1(305样本)和group3(207样本)，小样本组结果谨慎解释
3. **缺失ATE信息**: 明确在论文中说明放弃ATE分析的原因

### 方法学论证
1. **阈值选择依据**: 引用相关论文使用0.3阈值
2. **网络分析合理性**: 论证能耗变量间存在合理的因果机制
3. **路径强度解释**: 明确路径强度是概率乘积，不是效应量

### 结果解释
1. **避免过度解读**: 路径强度≠实际效应大小
2. **机制推测明确标注**: 区分已验证路径和推测路径
3. **交互项效应谨慎解释**: 区分统计交互和实际调节机制

---

## 📞 快速开始

### 环境准备
```bash
conda activate causal-research
cd /home/green/energy_dl/nightly/analysis
```

### 运行示例
```bash
# 运行RQ1分析
python scripts/rq_analysis_no_ate/main.py --rq 1 --threshold 0.3

# 运行RQ2分析
python scripts/rq_analysis_no_ate/main.py --rq 2 --threshold 0.3

# 运行完整分析
python scripts/rq_analysis_no_ate/main.py --all --threshold 0.3
```

### 输出检查
```bash
# 查看RQ1结果
ls -la results/energy_research/rq_analysis_no_ate/rq1/

# 查看RQ2结果
ls -la results/energy_research/rq_analysis_no_ate/rq2/
```

---

**维护者**: Green | **版本**: v1.0 | **创建**: 2026-02-03

**关联文档**:
- `RQ_FEASIBILITY_AND_PROCESS_REFINEMENT.md`: 可行性评估
- `RQ_RESEARCH_METHODOLOGY.md`: 原始研究方案
- `DIBS_RESULTS_REPRESENTATION_FINAL.md`: DiBS结果格式说明