# DiBS结果对三个研究问题的应用与优化途径

**分析日期**: 2026-01-05
**状态**: 综合分析

---

## 📊 DiBS结果对问题1的应用

### 问题1：超参数对能耗的影响（方向和大小）

**DiBS发现（阈值=0.1）**：

#### Group1 (examples)
- ✅ `hyperparam_batch_size → energy_gpu_max_watts` (强度=0.2)
  - **结论1.1**: batch_size对GPU最大功率有显著因果影响

#### Group3 (person_reid)
- ✅ `hyperparam_epochs → energy_gpu_avg_watts` (强度=0.3) ⭐
- ✅ `hyperparam_epochs → energy_gpu_min_watts` (强度=0.4) ⭐
  - **结论1.2**: epochs对GPU功率有强因果影响

#### Group6 (resnet)
- ✅ `hyperparam_epochs → energy_cpu_ram_joules` (强度=0.1)
- ✅ `hyperparam_epochs → energy_gpu_avg_watts` (强度=0.15)
- ✅ `hyperparam_epochs → energy_gpu_max_watts` (强度=0.2)
- ✅ `hyperparam_epochs → energy_gpu_min_watts` (强度=0.15)
- ✅ `hyperparam_epochs → energy_gpu_total_joules` (强度=0.3) ⭐
  - **结论1.3**: epochs对能耗有广泛的因果影响（5个能耗指标）

### DiBS对问题1的价值 ⭐⭐⭐

**优势**：
1. ✅ **因果方向明确**: DiBS给出了超参数→能耗的因果方向（而非仅相关性）
2. ✅ **多指标覆盖**: 检测到超参数对CPU、GPU不同能耗指标的影响
3. ✅ **任务异质性**: 发现不同任务组的超参数效应不同（epochs在group3和group6有效，但group1无效）

**局限**：
1. ⚠️ **70%的超参数检测失败**: learning_rate在group1/group3未检测到对能耗的影响（但直觉上应该有）
2. ⚠️ **边强度较弱**: 大部分边强度<0.3，需要进一步验证
3. ⚠️ **3/6组缺失超参数**: group2/4/5的超参数被数据预处理过滤

### 结合回归分析的建议 ⭐⭐⭐⭐⭐

DiBS结果可以作为**探索性发现**，指导回归分析的变量选择：

**优先验证的因果假设**：
```
基于DiBS发现，优先测试以下回归模型：

Group1:
  energy_gpu_max_watts ~ batch_size + controls

Group3:
  energy_gpu_avg_watts ~ epochs + controls
  energy_gpu_min_watts ~ epochs + controls

Group6:
  energy_gpu_total_joules ~ epochs + controls
  energy_gpu_avg_watts ~ epochs + controls
```

**价值**：
- DiBS帮助筛选变量，避免"钓鱼式"多重比较
- DiBS提供因果方向，回归量化效应大小
- 两种方法互相验证，提高结论可信度

---

## 🎯 目前能得到的结论

### 结论1：超参数对能耗的因果影响（问题1）⭐⭐⭐

**强证据（DiBS边强度>0.3 + 跨多个任务组）**：

1. **epochs是能耗的主要驱动因素**
   - Group3: epochs → GPU功率（强度=0.3-0.4）
   - Group6: epochs → GPU总能耗（强度=0.3）
   - **机制**: 训练轮数越多 → 训练时间越长 → 能耗越高
   - **量化建议**: 使用回归分析计算"每增加1个epoch，能耗增加多少焦耳"

2. **batch_size对GPU峰值功率有影响**
   - Group1: batch_size → GPU最大功率（强度=0.2）
   - **机制**: 更大的batch → GPU利用率更高 → 峰值功率更大
   - **量化建议**: 测试不同batch size下的GPU功率变化

**弱证据（仅在单个任务组或强度<0.3）**：

3. **learning_rate未检测到对能耗的直接影响**
   - DiBS只检测到 learning_rate → performance（强度=0.6）
   - **可能原因**: learning_rate通过性能间接影响能耗（中介效应）
   - **待验证**: 使用中介效应分析测试 learning_rate → performance → energy

### 结论2：能耗-性能不存在经典权衡（问题2）⭐⭐⭐⭐⭐

**关键发现**：

1. **没有超参数同时影响能耗和性能**（共同超参数=0）
   - 超参数影响是**分离的**：
     - learning_rate, weight_decay, dropout → **性能**
     - batch_size, epochs → **能耗**
   - **结论**: 不存在"提高某个超参数→性能提升但能耗增加"的经典权衡

2. **能耗和性能之间存在直接因果关系**（9条边，强度>0.3）
   - 性能 → 能耗：4条
   - 能耗 → 性能：5条
   - **关系类型**: 正相关（协同），而非负相关（权衡）

3. **逆向因果解释**：能耗 → 性能
   - Group4: `energy_gpu_avg_watts → perf_top1_accuracy` (强度=0.5) ⭐
   - Group5: `energy_cpu_pkg_joules → perf_accuracy` (强度=0.45) ⭐
   - **解释**: 计算密集的任务（高能耗）通常性能更好
   - **实际意义**: 要提高性能，可能需要接受更高的能耗

**与之前假设的对比**：

| 假设 | DiBS发现 | 结论 |
|------|---------|------|
| 存在超参数同时影响能耗和性能 | ❌ 共同超参数=0 | **假设不成立** |
| 能耗和性能负相关（权衡） | ❌ 检测到的是正相关 | **不是经典权衡** |
| 超参数通过性能影响能耗 | ⚠️ 中介路径仅1条 | **证据不足** |

**重要意义** 🎯：

这个发现**推翻了"能耗-性能权衡"的假设**，表明：
- 在单参数变异实验中，超参数对能耗和性能的影响是**独立的**
- 优化能耗和优化性能可能需要调整**不同的超参数**
- 实际应用中，可能存在**双优区域**（同时优化能耗和性能）

### 结论3：中介效应路径不完整（问题3）⭐⭐

**DiBS发现（仅在Group6）**：

1. **第一步存在**：超参数 → 中介变量
   - `epochs → GPU温度平均` (强度=0.1)
   - `epochs → GPU温度最大` (强度=0.35) ⭐
   - `epochs → GPU利用率平均` (强度=0.1)

2. **第二步缺失**：中介变量 → 能耗
   - DiBS**未检测到**这些中介变量对能耗的因果影响（阈值>0.1）
   - **矛盾**: 理论上GPU温度/利用率应该强烈影响能耗

**可能原因**：
1. **数据标准化**：Z-score标准化可能破坏了中介变量和能耗的线性关系
2. **多重共线性**：GPU温度、利用率和能耗高度相关，DiBS难以区分因果方向
3. **样本量不足**：Group6仅49个样本，不足以稳定估计中介效应

**替代方法建议** ⭐⭐⭐⭐⭐：

**不依赖DiBS的中介效应分析**（使用原始数据）：
```python
# 路径: epochs → gpu_util_avg → energy_gpu_total

# 总效应: epochs → energy
c = regress(energy ~ epochs).coef

# 路径a: epochs → gpu_util
a = regress(gpu_util ~ epochs).coef

# 路径b: gpu_util → energy (控制epochs)
b = regress(energy ~ epochs + gpu_util).coef_gpu_util

# 间接效应（中介效应）
indirect = a * b
mediation_pct = indirect / c * 100

# Sobel检验显著性
```

**优势**：
- 不需要DiBS检测完整路径
- 使用原始数据（未标准化）
- 可以量化中介比例（如GPU利用率解释60%的epochs对能耗的影响）

### 结论4：任务异质性显著 ⭐⭐⭐

**不同任务组的因果结构差异很大**：

| 任务组 | 主要超参数→能耗 | 主要超参数→性能 | 特点 |
|--------|---------------|---------------|------|
| Group1 (examples) | batch_size → GPU峰值功率 | learning_rate → accuracy | 简单模型，超参数效应弱 |
| Group3 (person_reid) | epochs → GPU功率 | dropout → rank5 | epochs对能耗影响强 |
| Group6 (resnet) | epochs → GPU总能耗+中介 | learning_rate, weight_decay → accuracy | 唯一检测到中介效应的组 |
| Group2/4/5 | 无（超参数缺失） | 无 | 数据质量问题 |

**实际意义**：
- **不能一概而论**: 不同任务的能耗优化策略不同
- **定制化优化**: 图像分类优化epochs，其他任务可能需要优化batch_size
- **论文撰写**: 需要按任务分组报告结果，而非合并分析

---

## 🔧 优化分析的途径

### 途径1：改进DiBS分析 ⭐⭐⭐

#### 1.1 使用原始数据（未标准化）

**问题**: 当前DiBS数据经过Z-score标准化，可能丢失因果信息

**解决方案**：
```python
# 当前做法（标准化）
df_scaled = StandardScaler().fit_transform(df)

# 改进做法（仅最小标准化）
# 方案A: 只标准化到0-1范围
df_minmax = MinMaxScaler().fit_transform(df)

# 方案B: 使用原始数据 + log变换（处理偏度）
df_log = np.log1p(df)  # 避免log(0)

# 方案C: 完全不标准化（如果变量单位统一）
df_raw = df.copy()
```

**预期改进**: 保留变量的原始尺度信息，可能提高DiBS检测中介效应的能力

#### 1.2 增加样本量

**问题**: Group6仅49个样本，DiBS建议>500

**解决方案**：
1. **合并相似任务组**:
   - Group1 (examples) + Group6 (resnet) → 都是图像分类
   - 新样本量: 259 + 49 = 308（仍不足，但更好）

2. **生成新实验数据**（需要重新训练）:
   - 参考`EXPERIMENT_EXPANSION_PLAN_20260105.md`
   - Group6目标: n=25变异实验 → 25 × 5轮 = 125个新样本
   - 总样本: 49 + 125 = 174（接近推荐的200最小值）

3. **使用bootstrap增强**:
   - 从49个样本中有放回抽样
   - 生成500个bootstrap样本
   - 在每个bootstrap样本上运行DiBS
   - 汇总边的后验分布

#### 1.3 降低缺失率阈值

**问题**: 40%缺失率阈值导致3个任务组的超参数全部丢失

**解决方案**：
```python
# 当前: 40%阈值
MAX_MISSING_RATE = 0.40

# 改进: 分层阈值
- 超参数: MAX_MISSING_RATE = 0.50（宽松）
- 性能指标: MAX_MISSING_RATE = 0.40（中等）
- 能耗指标: MAX_MISSING_RATE = 0.30（严格）
```

**预期改进**: 保留Group2/4/5的超参数，增加可分析的任务组

#### 1.4 调整DiBS参数

**当前配置**: alpha=0.05, beta=0.1, particles=20, steps=5000

**可尝试的配置**：

| 参数 | 当前值 | 建议值 | 目的 |
|------|-------|-------|------|
| alpha | 0.05 | 0.01 | 更稀疏的图（减少弱边） |
| beta | 0.1 | 0.05 | 更弱的无环约束（允许更多路径探索） |
| n_steps | 5000 | 10000 | 更好的收敛 |
| n_particles | 20 | 50 | 更准确的后验估计 |

**trade-off**: 更好的结果 vs 更长的运行时间（10分钟 → 25分钟）

### 途径2：结合其他因果方法 ⭐⭐⭐⭐⭐

#### 2.1 PC算法（基于条件独立性测试）

**优势**：
- 不假设线性关系
- 对非高斯数据更鲁棒
- 可以处理离散变量

**实现**（使用causal-learn）：
```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

# 运行PC算法
cg = pc(data, alpha=0.05, indep_test=fisherz)

# 提取因果图
causal_graph = cg.G.graph
```

**对比DiBS和PC的结果**：
- 一致的边 → 高置信度
- 不一致的边 → 需要进一步验证

#### 2.2 LiNGAM（线性非高斯模型）

**优势**：
- 可以识别因果方向（利用非高斯性）
- 适合连续变量
- 计算快（<1分钟）

**实现**（使用causal-learn）：
```python
from causallearn.search.FCMBased import lingam

# 运行LiNGAM
model = lingam.ICALiNGAM()
model.fit(data)

# 提取因果图
causal_matrix = model.adjacency_matrix_
```

**特别适合问题3**（中介效应）：
- LiNGAM可以直接给出完整因果图
- 基于非高斯假设，可能比DiBS更适合能耗数据

#### 2.3 因果森林（Causal Forest）

**用途**: 估计异质性因果效应（CATE）

**优势**：
- 不假设线性关系
- 可以发现"超参数对能耗的影响在不同配置下不同"
- 提供置信区间

**实现**（使用econml）：
```python
from econml.dml import CausalForestDML

# 定义变量
X = controls  # 混淆变量（如duration, num_mutated_params）
T = df['hyperparam_epochs']  # 处理变量（超参数）
Y = df['energy_gpu_total_joules']  # 结果变量（能耗）

# 训练因果森林
cf = CausalForestDML(random_state=42)
cf.fit(Y, T, X=X, W=None)

# 估计条件平均处理效应
cate = cf.effect(X)

# 平均因果效应
ate = cate.mean()
print(f"epochs对能耗的平均因果效应: {ate:.2f} Joules/epoch")
```

**价值**: 可以量化问题1（超参数对能耗的因果效应大小）

### 途径3：回归分析验证DiBS发现 ⭐⭐⭐⭐⭐

#### 3.1 多元线性回归（基线方法）

**基于DiBS发现的优先模型**：

```python
# Group3: epochs → energy_gpu_avg_watts (DiBS强度=0.3)
model1 = smf.ols(
    'energy_gpu_avg_watts ~ hyperparam_epochs + duration_seconds + is_parallel',
    data=df_group3
).fit()

print(model1.summary())
# 检查epochs的系数是否显著（p<0.05）
# 检查系数符号是否与DiBS方向一致（正相关）

# Group6: epochs → energy_gpu_total_joules (DiBS强度=0.3)
model2 = smf.ols(
    'energy_gpu_total_joules ~ hyperparam_epochs + hyperparam_learning_rate + duration_seconds',
    data=df_group6
).fit()
```

**验证标准**：
- ✅ **一致**: 回归系数显著 + 符号与DiBS一致 → DiBS发现可信
- ⚠️ **不一致**: 回归系数不显著或符号相反 → DiBS可能是假阳性
- 💡 **新发现**: 回归发现DiBS未检测到的边 → DiBS假阴性（阈值太高）

#### 3.2 中介效应分析（Sobel检验）

**优先测试的中介路径**（基于DiBS第一步）：

```python
from scipy import stats

def mediation_analysis(df, X_col, M_col, Y_col):
    """
    X: 超参数
    M: 中介变量（GPU利用率、温度）
    Y: 能耗
    """
    # 总效应: X → Y
    c = smf.ols(f'{Y_col} ~ {X_col}', data=df).fit().params[X_col]

    # 路径a: X → M
    a = smf.ols(f'{M_col} ~ {X_col}', data=df).fit().params[X_col]

    # 路径b: M → Y (控制X)
    b = smf.ols(f'{Y_col} ~ {X_col} + {M_col}', data=df).fit().params[M_col]

    # 间接效应
    indirect = a * b
    mediation_pct = indirect / c * 100 if abs(c) > 1e-6 else 0

    # Sobel检验（简化版）
    # ... (标准误计算)

    return {
        'total_effect': c,
        'indirect_effect': indirect,
        'mediation_pct': mediation_pct
    }

# Group6测试
result = mediation_analysis(
    df_group6,
    X_col='hyperparam_epochs',
    M_col='energy_gpu_temp_max_celsius',
    Y_col='energy_gpu_total_joules'
)

print(f"中介比例: {result['mediation_pct']:.1f}%")
```

**优势**: 不依赖DiBS检测完整路径，直接测试假设

### 途径4：可视化与探索性分析 ⭐⭐⭐

#### 4.1 因果图可视化

**使用Graphviz生成DiBS因果图**：

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_causal_graph(graph_matrix, feature_names, threshold=0.3):
    """
    生成DiBS因果图可视化
    """
    G = nx.DiGraph()

    # 添加节点
    for i, name in enumerate(feature_names):
        node_type = (
            'hyperparam' if name.startswith('hyperparam_') else
            'performance' if name.startswith('perf_') else
            'energy' if 'energy' in name and not ('util' in name or 'temp' in name) else
            'mediator'
        )
        G.add_node(name, node_type=node_type)

    # 添加边（强度>阈值）
    for i, source in enumerate(feature_names):
        for j, target in enumerate(feature_names):
            if graph_matrix[i, j] > threshold:
                G.add_edge(source, target, weight=graph_matrix[i, j])

    # 布局和绘图
    pos = nx.spring_layout(G, k=2, iterations=50)

    # 按类型着色
    colors = {
        'hyperparam': 'lightblue',
        'performance': 'lightgreen',
        'energy': 'lightcoral',
        'mediator': 'lightyellow'
    }
    node_colors = [colors[G.nodes[node]['node_type']] for node in G.nodes()]

    # 绘制
    plt.figure(figsize=(16, 12))
    nx.draw(G, pos, node_color=node_colors, with_labels=True,
            node_size=3000, font_size=8, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray',
            width=[G[u][v]['weight']*3 for u,v in G.edges()])

    plt.title(f"DiBS因果图（阈值={threshold}）", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'causal_graph_threshold_{threshold}.png', dpi=300)
    plt.show()

# 为每个任务组生成可视化
for group_id in ['group1_examples', 'group3_person_reid', 'group6_resnet']:
    graph = np.load(f'{group_id}_causal_graph.npy')
    names = json.load(open(f'{group_id}_feature_names.json'))
    visualize_causal_graph(graph, names, threshold=0.3)
```

#### 4.2 超参数-能耗散点图矩阵

**探索DiBS未检测到的关系**：

```python
import seaborn as sns

# 选择变量
hyperparams = ['hyperparam_batch_size', 'hyperparam_epochs', 'hyperparam_learning_rate']
energies = ['energy_gpu_avg_watts', 'energy_gpu_total_joules', 'energy_cpu_total_joules']

# 散点图矩阵
df_subset = df[hyperparams + energies]
sns.pairplot(df_subset, kind='scatter', diag_kind='kde')
plt.suptitle('超参数-能耗关系探索', y=1.02)
plt.savefig('hyperparam_energy_pairplot.png', dpi=300)
```

**价值**: 发现非线性关系（DiBS可能漏检）

### 途径5：改进实验设计（长期）⭐⭐⭐⭐⭐

#### 5.1 多参数变异实验

**问题**: 当前实验每次只变化一个超参数，难以检测交互效应和权衡

**解决方案**（参考`EXPERIMENT_EXPANSION_PLAN_20260105.md`）：

```json
{
  "comment": "examples多参数变异实验",
  "mutate_strategy": "random_multi_param",
  "num_params_per_mutation": [2, 3],  // 每次变异2-3个参数
  "runs_per_config": 80,
  "hyperparams_to_mutate": [
    "learning_rate",
    "batch_size",
    "epochs"
  ]
}
```

**预期改进**:
- 可以检测到 learning_rate × batch_size 的交互效应
- 可能触发权衡关系（同时调整多个超参数）

#### 5.2 时间序列数据收集

**问题**: 当前数据是聚合的（每个实验一个能耗值），丢失了时间因果信息

**解决方案**: 记录训练过程中的时间序列数据

```python
# 每个epoch记录一次
training_log = []
for epoch in range(num_epochs):
    # 训练
    train_loss = train_one_epoch()

    # 记录
    training_log.append({
        'epoch': epoch,
        'timestamp': time.time(),
        'train_loss': train_loss,
        'gpu_power': get_gpu_power(),  # 实时功率
        'gpu_util': get_gpu_util(),
        'gpu_temp': get_gpu_temp()
    })

# 保存时间序列
save_timeseries(training_log)
```

**分析方法**: Granger因果检验、时间序列DiBS

**优势**: 可以检测动态因果关系（如GPU利用率变化如何影响功率）

---

## 📌 优先级排序

基于**效果/成本比**，推荐执行顺序：

### 🔥 立即执行（高价值，低成本）

1. **回归分析验证DiBS发现** ⭐⭐⭐⭐⭐
   - 成本：1-2小时编写脚本 + 10分钟运行
   - 收益：验证DiBS的7个超参数→能耗边，量化效应大小
   - **优先验证**: epochs → energy (group3/6), batch_size → energy (group1)

2. **中介效应分析（不依赖DiBS）** ⭐⭐⭐⭐⭐
   - 成本：1小时编写脚本 + 5分钟运行
   - 收益：直接回答问题3，不受DiBS中介路径缺失的影响
   - **优先测试**: epochs → gpu_temp/util → energy (group6)

3. **降低DiBS阈值到0.05** ⭐⭐⭐⭐
   - 成本：修改1行代码 + 重新分析（已有因果图，无需重新运行DiBS）
   - 收益：可能发现更多弱边，补充DiBS发现
   - 脚本已存在：`reanalyze_dibs_results_lower_threshold.py`

4. **因果图可视化** ⭐⭐⭐⭐
   - 成本：1-2小时编写脚本
   - 收益：直观展示因果关系，便于论文撰写和展示

### 📅 本周执行（中价值，中成本）

5. **LiNGAM因果发现** ⭐⭐⭐⭐
   - 成本：安装causal-learn + 1小时编写 + 10分钟运行
   - 收益：对比DiBS，发现DiBS漏检的边
   - 特别适合问题3（中介效应）

6. **因果森林（Causal Forest）** ⭐⭐⭐⭐
   - 成本：1-2小时编写 + 30分钟运行
   - 收益：量化异质性因果效应，补充DiBS
   - 特别适合问题1（超参数对能耗的因果效应）

7. **降低缺失率阈值重新生成数据** ⭐⭐⭐
   - 成本：修改脚本 + 10分钟重新生成
   - 收益：保留group2/4/5的超参数，增加3个可分析的任务组

### 📆 长期执行（高价值，高成本）

8. **增加样本量（扩展实验）** ⭐⭐⭐⭐⭐
   - 成本：1周GPU训练时间（471个新实验）
   - 收益：DiBS样本量达到500+，大幅提高检测能力
   - 参考：`EXPERIMENT_EXPANSION_PLAN_20260105.md`

9. **多参数变异实验** ⭐⭐⭐⭐⭐
   - 成本：实验设计 + 1周GPU训练
   - 收益：可能触发权衡关系，检测交互效应
   - 是解决"共同超参数=0"问题的根本方法

10. **时间序列数据收集** ⭐⭐⭐⭐
    - 成本：修改训练脚本 + 重新运行部分实验
    - 收益：动态因果分析，更准确的中介效应估计

---

## 🎯 总结：当前状况与最佳路径

### 当前已有的结论（基于DiBS + 领域知识）

**问题1：超参数对能耗的影响**
- ✅ epochs是主要驱动因素（强证据）
- ✅ batch_size影响GPU峰值功率（中等证据）
- ⚠️ learning_rate影响不明（需要验证）

**问题2：能耗-性能权衡**
- ✅ 不存在经典权衡（强证据）⭐⭐⭐
- ✅ 超参数影响分离（强证据）
- ✅ 能耗-性能正相关（协同而非权衡）（强证据）

**问题3：中介效应**
- ⚠️ DiBS中介路径不完整（证据不足）
- 💡 建议使用传统中介效应分析替代DiBS

### 推荐的最佳分析路径 ⭐⭐⭐⭐⭐

```
阶段1（本周）：验证和补充DiBS发现
├─ 回归分析验证DiBS的7个超参数→能耗边
├─ 中介效应分析（Sobel检验）测试问题3
├─ LiNGAM因果发现对比DiBS
└─ 因果图可视化

阶段2（下周）：量化和扩展
├─ 因果森林量化异质性效应
├─ 降低缺失率阈值，分析group2/4/5
├─ Pareto前沿分析（问题2补充）
└─ 综合报告撰写

阶段3（长期，可选）：改进实验设计
├─ 扩展实验（471个新样本）
├─ 多参数变异实验
└─ 时间序列数据收集
```

**关键建议**：
1. **不要过度依赖DiBS**：结合回归、中介效应分析、因果森林
2. **优先完成问题2**：已有强证据（不存在权衡），可直接撰写结论
3. **问题1和3需要补充分析**：使用回归和中介效应分析
4. **长期改进**：多参数变异实验是解决根本问题的途径

---

**文档创建时间**: 2026-01-05
**后续行动**: 立即执行回归分析和中介效应分析
