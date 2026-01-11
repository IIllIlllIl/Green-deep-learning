# 并行/非并行模式影响分析方案

**创建日期**: 2026-01-06
**研究问题**: 并行/非并行训练模式对能耗和超参数-能耗因果关系的影响

---

## 📊 当前状态分析

### 数据分布

| 模式 | 样本数 | 占比 |
|------|--------|------|
| 并行模式 (parallel) | 433 | 51.8% |
| 非并行模式 (default) | 403 | 48.2% |
| **总计** | 836 | 100% |

### 当前DiBS分析的局限

**问题**: 当前的DiBS分析混合了两种模式的数据，**没有**将`is_parallel`或`mode`作为输入变量。

**潜在影响**:
1. **混淆效应**: 如果并行模式本身影响能耗，当前分析可能错误归因
2. **调节效应遗漏**: 不同模式下，超参数对能耗的因果效应可能不同（交互效应）
3. **因果结构差异**: 并行和非并行可能有完全不同的因果图

---

## 🎯 分析目标

我们希望回答以下问题：

1. **主效应**: 并行/非并行模式本身是否影响能耗？（直接效应）
2. **调节效应**: 模式是否改变超参数对能耗的影响大小？（交互效应）
3. **因果结构差异**: 两种模式下的因果图是否不同？

---

## 🔬 方法论设计

### 方法对比

| 方法 | 优势 | 劣势 | 推荐场景 |
|------|------|------|----------|
| **方法1: DiBS包含模式变量** | 简单，一次分析 | DiBS对离散变量处理较弱 | 快速探索主效应 |
| **方法2: 分层DiBS** | 发现因果结构差异 | 样本量减半 | 发现调节效应 |
| **方法3: 回归交互分析** | 标准方法，易解释 | 需要预先知道因果边 | 量化已知边的交互 |
| **方法4: 组合分析** | 最全面，证据收敛 | 工作量大 | **推荐** ⭐⭐⭐ |

### 推荐方案: 组合分析（4步）

---

## 📋 详细分析步骤

### 步骤1: 探索性分析 - 模式的主效应

**目的**: 快速了解模式对能耗的直接影响

**方法**: 描述性统计 + t检验

```python
# 比较并行vs非并行的能耗差异
parallel_df = df[df['mode'] == 'parallel']
non_parallel_df = df[df['mode'] != 'parallel']

# 对每个能耗指标进行t检验
energy_metrics = [
    'energy_gpu_total_joules',
    'energy_gpu_avg_watts',
    'energy_gpu_max_watts'
]

for metric in energy_metrics:
    t_stat, p_value = stats.ttest_ind(
        parallel_df[metric].dropna(),
        non_parallel_df[metric].dropna()
    )
    print(f"{metric}: t={t_stat:.3f}, p={p_value:.4f}")
```

**预期输出**:
- 模式对各能耗指标的平均差异
- 统计显著性（是否需要进一步分析）

**决策点**:
- ✅ 如果p<0.05 → 模式有显著影响，继续后续分析
- ❌ 如果p>0.05 → 模式影响不显著，可以简化分析

---

### 步骤2: 分层DiBS分析 - 发现因果结构差异

**目的**: 发现并行和非并行模式下的因果图差异

**方法**: 分别对两种模式运行DiBS

#### 2.1 数据准备

```python
# 为并行模式准备DiBS数据
parallel_groups = prepare_dibs_data(
    df[df['mode'] == 'parallel'],
    output_dir='data/energy_research/dibs_training_parallel/'
)

# 为非并行模式准备DiBS数据
non_parallel_groups = prepare_dibs_data(
    df[df['mode'] != 'parallel'],
    output_dir='data/energy_research/dibs_training_non_parallel/'
)
```

**样本量检查**:
```
group1_examples:
  - 并行: ~130样本
  - 非并行: ~129样本

group3_person_reid:
  - 并行: ~73样本
  - 非并行: ~73样本

group6_resnet:
  - 并行: ~25样本 ⚠️
  - 非并行: ~24样本 ⚠️
```

**注意**: group6样本量较小，DiBS结果可能不稳定。

#### 2.2 运行DiBS

```bash
# 并行模式
python scripts/run_dibs_parallel_mode.py

# 非并行模式
python scripts/run_dibs_non_parallel_mode.py
```

#### 2.3 比较因果图

**定义因果图差异度量**:

1. **边差异（Edge Difference）**:
   ```python
   # 在并行模式中存在但非并行中不存在的边
   parallel_only_edges = set(parallel_edges) - set(non_parallel_edges)

   # 在非并行模式中存在但并行中不存在的边
   non_parallel_only_edges = set(non_parallel_edges) - set(parallel_edges)

   # 共同的边
   common_edges = set(parallel_edges) & set(non_parallel_edges)
   ```

2. **边强度差异（Strength Difference）**:
   ```python
   # 对共同的边，比较强度差异
   for edge in common_edges:
       strength_diff = parallel_strength[edge] - non_parallel_strength[edge]
       if abs(strength_diff) > 0.1:  # 显著差异阈值
           print(f"{edge}: Δ强度 = {strength_diff:.3f}")
   ```

3. **关键边对比**（针对3个研究问题）:

   **问题1: 超参数→能耗**
   - epochs → energy: 强度是否不同？
   - batch_size → energy: 强度是否不同？

   **问题2: 共同超参数**
   - 并行模式: 有几个共同超参数？
   - 非并行模式: 有几个共同超参数？

   **问题3: 中介路径**
   - 并行模式: 哪些中介路径存在？
   - 非并行模式: 哪些中介路径存在？

**预期输出**:
- 两个因果图的可视化对比
- 差异边列表和强度差异表格
- 针对3个研究问题的模式差异总结

---

### 步骤3: 回归交互分析 - 量化调节效应

**目的**: 量化模式对已知因果边的调节作用

**方法**: 交互效应回归

#### 3.1 验证DiBS发现的边（加入交互项）

对于DiBS发现的每条边（如epochs → energy），运行以下回归：

```python
# 模型1: 无交互项（基线）
formula_1 = "energy ~ hyperparam + is_parallel + duration"
model_1 = smf.ols(formula_1, data=df).fit()

# 模型2: 有交互项
formula_2 = "energy ~ hyperparam * is_parallel + duration"
model_2 = smf.ols(formula_2, data=df).fit()

# 比较模型拟合度
print(f"R² 提升: {model_2.rsquared - model_1.rsquared:.4f}")

# 交互项系数
interaction_coef = model_2.params['hyperparam:is_parallel']
interaction_p = model_2.pvalues['hyperparam:is_parallel']

print(f"交互效应: β={interaction_coef:.4f}, p={interaction_p:.4f}")
```

**解释交互效应**:
- `interaction_coef > 0`: 并行模式**增强**超参数对能耗的影响
- `interaction_coef < 0`: 并行模式**减弱**超参数对能耗的影响
- `interaction_p < 0.05`: 交互效应显著

#### 3.2 关键边的交互分析

**优先分析的边**（基于当前DiBS + 回归结果）:

1. **epochs → energy_gpu_total_joules** (group6)
   - 基础效应: β=0.141 (R²=0.997)
   - 交互效应: β_interaction = ?

2. **epochs → energy_gpu_avg_watts** (group3, group6)
   - 基础效应: β=0.263 (group3), β=0.436 (group6)
   - 交互效应: β_interaction = ?

3. **batch_size → energy_gpu_max_watts** (group1)
   - 基础效应: β=0.129 (R²=0.089)
   - 交互效应: β_interaction = ?

#### 3.3 可视化交互效应

```python
# 为每条显著的边绘制交互图
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 例子: epochs → energy (group6)
for i, (mode, label) in enumerate([('parallel', '并行'), ('', '非并行')]):
    mode_df = df[df['mode'] == mode] if mode else df[df['mode'] != 'parallel']

    ax = axes[i]
    sns.regplot(x='hyperparam_epochs', y='energy_gpu_total_joules',
                data=mode_df, ax=ax)
    ax.set_title(f'{label}模式 (n={len(mode_df)})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('GPU Total Energy (J)')

# 第三个图: 两条回归线对比
ax = axes[2]
for mode, label, color in [('parallel', '并行', 'red'), ('', '非并行', 'blue')]:
    mode_df = df[df['mode'] == mode] if mode else df[df['mode'] != 'parallel']
    sns.regplot(x='hyperparam_epochs', y='energy_gpu_total_joules',
                data=mode_df, ax=ax, label=label, color=color)
ax.set_title('模式对比')
ax.legend()

plt.tight_layout()
plt.savefig('results/mode_interaction_epochs_energy.png', dpi=300)
```

**预期输出**:
- 交互效应系数表格
- 可视化对比图（每条关键边）
- 显著交互效应的边列表

---

### 步骤4: 中介效应的模式差异

**目的**: 检验模式是否改变中介路径

**方法**: 分层中介分析

#### 4.1 分别对两种模式进行中介分析

```python
# 对于已发现的显著中介路径
# 例如: epochs → GPU利用率 → 总能耗

# 并行模式
result_parallel = sobel_test_mediation(
    df[df['mode'] == 'parallel'],
    X='hyperparam_epochs',
    M='energy_gpu_util_avg_percent',
    Y='energy_gpu_total_joules'
)

# 非并行模式
result_non_parallel = sobel_test_mediation(
    df[df['mode'] != 'parallel'],
    X='hyperparam_epochs',
    M='energy_gpu_util_avg_percent',
    Y='energy_gpu_total_joules'
)

# 比较中介效应
print(f"并行模式 - 间接效应: {result_parallel['indirect_effect']:.4f}")
print(f"非并行模式 - 间接效应: {result_non_parallel['indirect_effect']:.4f}")
```

#### 4.2 中介效应差异的显著性检验

使用**Bootstrap方法**检验两组中介效应的差异:

```python
from scipy import stats

# Bootstrap重采样
n_bootstrap = 1000
indirect_effects_parallel = []
indirect_effects_non_parallel = []

for _ in range(n_bootstrap):
    # 重采样并计算中介效应
    sample_p = df[df['mode'] == 'parallel'].sample(frac=1, replace=True)
    sample_np = df[df['mode'] != 'parallel'].sample(frac=1, replace=True)

    result_p = sobel_test_mediation(sample_p, X, M, Y)
    result_np = sobel_test_mediation(sample_np, X, M, Y)

    indirect_effects_parallel.append(result_p['indirect_effect'])
    indirect_effects_non_parallel.append(result_np['indirect_effect'])

# 计算差异的置信区间
diff = np.array(indirect_effects_parallel) - np.array(indirect_effects_non_parallel)
ci_lower, ci_upper = np.percentile(diff, [2.5, 97.5])

print(f"中介效应差异的95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
if 0 < ci_lower or 0 > ci_upper:
    print("✅ 差异显著")
else:
    print("❌ 差异不显著")
```

**预期输出**:
- 两种模式下的中介效应对比表
- 中介效应差异的显著性检验结果
- 中介路径的模式依赖性结论

---

## 📊 整合分析报告结构

### 报告大纲

```markdown
# 并行/非并行模式对深度学习能耗的影响分析

## 1. 主效应分析
- 模式对能耗的平均影响
- t检验结果
- 效应量（Cohen's d）

## 2. 因果结构差异
- 并行模式因果图
- 非并行模式因果图
- 差异分析（边差异、强度差异）

## 3. 针对3个研究问题的模式影响

### 问题1: 超参数→能耗（模式调节）
- epochs效应: 并行 vs 非并行
- batch_size效应: 并行 vs 非并行
- 交互效应量化

### 问题2: 能耗-性能权衡（模式依赖性）
- 并行模式: 共同超参数数量
- 非并行模式: 共同超参数数量
- 权衡关系是否因模式而异

### 问题3: 中介机制（模式差异）
- 并行模式: 哪些中介路径显著
- 非并行模式: 哪些中介路径显著
- 中介效应的模式依赖性

## 4. 综合结论
- 模式的主要影响
- 模式对因果关系的调节作用
- 实践建议（是否需要区分模式优化）
```

---

## 🎯 预期研究发现类型

### 情景1: 模式无显著影响

**发现**:
- 主效应: 并行vs非并行能耗无显著差异（p>0.05）
- 交互效应: 所有超参数的交互项均不显著
- 因果图: 两种模式下因果结构高度相似

**结论**:
✅ 当前的混合分析结果是可靠的
✅ 模式不是重要的混淆变量
✅ 优化建议适用于两种模式

### 情景2: 模式有主效应但无交互

**发现**:
- 主效应: 并行模式能耗显著高于非并行（或相反）
- 交互效应: 超参数对能耗的影响斜率相同
- 因果图: 两种模式下因果结构相似，但能耗基线不同

**结论**:
✅ 需要在分析中控制模式变量
✅ 超参数优化策略对两种模式通用
❌ 当前的DiBS分析遗漏了模式的主效应

**建议**: 重新运行DiBS，将`is_parallel`作为输入变量

### 情景3: 模式有显著交互效应

**发现**:
- 交互效应: epochs对能耗的影响在并行模式下更强（或更弱）
- 因果图: 两种模式下因果结构明显不同
  - 并行: epochs → energy (强度=0.5)
  - 非并行: epochs → energy (强度=0.1)

**结论**:
❌ 当前的混合分析结果可能误导
✅ 需要分别为两种模式提供优化建议
✅ 模式是重要的调节变量

**建议**:
- 论文中需要分层报告结果
- 优化工具需要考虑训练模式

### 情景4: 模式改变因果结构

**发现**:
- 并行模式: epochs → energy （强因果）
- 非并行模式: batch_size → energy （强因果）
- 完全不同的因果驱动因素

**结论**:
🔥 重大发现！模式根本改变了能耗的因果机制
✅ 需要针对不同模式设计完全不同的优化策略
❌ 当前的混合分析完全不可用

**建议**:
- 发表独立的研究论文
- 为AI社区提供模式相关的最佳实践

---

## 🛠️ 实现清单

### 脚本开发

- [ ] `scripts/analyze_mode_main_effect.py` - 步骤1: t检验主效应
- [ ] `scripts/prepare_dibs_data_by_mode.py` - 步骤2.1: 按模式准备数据
- [ ] `scripts/run_dibs_parallel_mode.py` - 步骤2.2: 并行模式DiBS
- [ ] `scripts/run_dibs_non_parallel_mode.py` - 步骤2.2: 非并行模式DiBS
- [ ] `scripts/compare_causal_graphs.py` - 步骤2.3: 因果图比较
- [ ] `scripts/regression_interaction_analysis.py` - 步骤3: 交互效应回归
- [ ] `scripts/mediation_by_mode.py` - 步骤4: 分层中介分析
- [ ] `scripts/generate_mode_analysis_report.py` - 整合报告

### 数据准备

- [ ] 生成并行模式DiBS训练数据（6个组）
- [ ] 生成非并行模式DiBS训练数据（6个组）
- [ ] 验证样本量充足性（每组>30）

### 分析执行

- [ ] 运行步骤1-4的所有分析
- [ ] 生成所有可视化图表
- [ ] 编写综合分析报告

### 预计工作量

- **数据准备**: 30分钟
- **DiBS分析**: 2小时（并行+非并行各40分钟）
- **回归交互分析**: 1小时
- **中介分析**: 1小时
- **报告编写**: 1.5小时
- **总计**: 约6小时

---

## 💡 关键决策点

### 决策1: 是否需要完整的模式分析？

**判断依据**: 步骤1的t检验结果

- ✅ 如果p<0.05 → 进行完整分析（步骤2-4）
- ❌ 如果p>0.05 → 简化分析（只做步骤3的交互检验）

### 决策2: 样本量不足怎么办？

**问题**: group6在分层后可能只有25个样本/模式

**解决方案**:
1. **合并策略**: 将group6与其他小样本组合并
2. **排除策略**: 只分析样本量充足的组（group1, group3）
3. **Bootstrap**: 使用Bootstrap方法增加稳定性

### 决策3: 如果两种模式因果图完全不同？

**处理方式**:
- 单独为两种模式编写分析报告
- 在论文中明确说明发现
- 为实践者提供模式相关的建议

---

## 📚 参考文献

关于调节效应和交互分析的经典文献：

1. **Hayes (2018)**: Introduction to Mediation, Moderation, and Conditional Process Analysis
   - 标准的调节效应分析方法

2. **Baron & Kenny (1986)**: The moderator-mediator variable distinction
   - 调节vs中介的经典区分

3. **Preacher et al. (2007)**: Addressing Moderated Mediation Hypotheses
   - 调节中介的组合分析

---

## ✅ 下一步行动

1. **立即执行**: 步骤1（t检验主效应）- 快速判断是否需要深入分析
2. **如果显著**: 准备分层数据并运行完整分析流程
3. **如果不显著**: 只运行步骤3的交互检验，验证当前结论的稳健性

**预计开始时间**: 需要用户确认是否立即开始

---

**文档创建**: 2026-01-06
**状态**: 待执行
**优先级**: 高 ⭐⭐⭐
