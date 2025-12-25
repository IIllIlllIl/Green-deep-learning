# 应用因果推断方法到新研究问题 - 使用指南

**目标受众**: 希望使用本项目的因果推断方法研究其他问题的研究者

---

## 🎯 本指南的目的

本项目已完成ASE 2023论文核心方法的实现和验证。现在您可以将这些**因果推断工具**应用到您自己的研究问题中。

本指南将帮助您：
1. 理解核心方法的工作原理
2. 准备适合的数据
3. 应用因果推断分析
4. 解释和使用结果

---

## 📚 核心方法概述

### 方法流程

```
您的数据
    ↓
步骤1: 因果图学习 (DiBS)
    → 发现变量间的因果关系
    ↓
步骤2: 因果推断 (DML)
    → 量化因果效应的大小
    ↓
步骤3: 效应分析
    → 找出哪些干预能改进目标
```

### 三个核心工具

1. **CausalGraphLearner** - 学习因果图
   - 输入: 多变量数据
   - 输出: 因果关系图（有向无环图）
   - 用途: 发现"谁影响谁"

2. **CausalInferenceEngine** - 估计因果效应
   - 输入: 数据 + 因果图
   - 输出: 每条因果边的效应大小（ATE）
   - 用途: 量化"影响有多大"

3. **TradeoffDetector** - 检测权衡关系
   - 输入: 因果效应 + 优化方向
   - 输出: 冲突的改进目标
   - 用途: 发现"改进A会恶化B"

---

## 🔧 准备您的数据

### 数据格式要求

您的数据应该是**表格形式**（DataFrame或CSV），其中：
- 每行是一个观测/样本
- 每列是一个变量
- 变量可以是**连续**或**离散**的

**示例数据结构**:

```python
import pandas as pd

# 示例：研究学习方法对学生表现的影响
data = pd.DataFrame({
    'study_hours': [2, 4, 6, 3, 5, ...],      # 学习时间
    'method': [1, 2, 1, 3, 2, ...],            # 学习方法 (1=传统, 2=混合, 3=在线)
    'score': [70, 85, 90, 75, 88, ...],        # 考试分数
    'attendance': [80, 90, 95, 85, 92, ...],   # 出勤率
    'motivation': [6, 8, 9, 7, 8, ...],        # 动机 (1-10)
    'homework_done': [5, 9, 10, 6, 9, ...],    # 完成作业数
})
```

### 数据准备清单

- [ ] 数据至少有**50-100个样本**（越多越好）
- [ ] 数据至少有**5-10个变量**
- [ ] 数据没有太多缺失值（<10%）
- [ ] 变量之间可能存在因果关系
- [ ] 您有明确的**干预变量**（想改变的）和**结果变量**（想优化的）

---

## 🚀 完整使用流程

### 步骤1: 学习因果图

**目标**: 发现数据中的因果关系结构

```python
from utils.causal_discovery import CausalGraphLearner
import pandas as pd

# 1. 加载您的数据
data = pd.read_csv('your_data.csv')

# 2. 创建因果图学习器
learner = CausalGraphLearner(
    n_vars=len(data.columns),  # 变量数量
    n_steps=2000,              # 迭代次数（越多越准确，但越慢）
    alpha=0.1,                 # 稀疏性参数（越小图越稀疏）
    random_seed=42             # 保证可复现
)

# 3. 学习因果图
print("正在学习因果图，这可能需要几分钟...")
causal_graph = learner.fit(data, verbose=True)

# 4. 获取因果边
edges = learner.get_edges(threshold=0.3)  # 阈值：边的最小强度

print(f"\n检测到 {len(edges)} 条因果边:")
for source, target, weight in edges[:10]:  # 显示前10条
    print(f"  {data.columns[source]} → {data.columns[target]}: {weight:.3f}")

# 5. 保存结果
learner.save_graph('results/my_causal_graph.npy')

# 6. 可视化（可选）
learner.visualize_causal_graph(
    var_names=data.columns,
    output_path='results/causal_graph.png',
    threshold=0.3
)
```

**输出解读**:
- **因果边**: `X → Y` 表示X是Y的原因
- **权重**: 表示因果关系的强度（0-1）
- **稀疏图**: 边越少，因果关系越清晰

### 步骤2: 估计因果效应

**目标**: 量化每个干预的效果大小

```python
from utils.causal_inference import CausalInferenceEngine

# 1. 创建因果推断引擎
engine = CausalInferenceEngine(verbose=True)

# 2. 对整个因果图进行分析
print("正在估计因果效应...")
causal_effects = engine.analyze_all_edges(
    data=data,
    causal_graph=causal_graph,
    var_names=data.columns.tolist(),
    threshold=0.3  # 只分析强度>0.3的边
)

# 3. 查看结果
print(f"\n检测到 {len(causal_effects)} 个因果效应:")
for edge, result in list(causal_effects.items())[:10]:
    print(f"\n  {edge}:")
    print(f"    ATE = {result['ate']:.4f}")
    print(f"    95% CI = [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"    显著? {'是' if result['is_significant'] else '否'}")

# 4. 只看显著的效应
significant = engine.get_significant_effects()
print(f"\n显著的因果效应 (共{len(significant)}个):")
for edge, result in significant.items():
    print(f"  {edge}: ATE={result['ate']:.4f} ***")

# 5. 保存结果
engine.save_results('results/causal_effects.csv')
```

**输出解读**:
- **ATE (平均处理效应)**: 干预X增加1单位时，Y平均改变多少
  - ATE > 0: 正向影响
  - ATE < 0: 负向影响
  - ATE ≈ 0: 几乎无影响
- **95% 置信区间**: 估计的不确定性范围
- **显著性**: 如果置信区间不包含0，则效应显著

### 步骤3: 定义优化方向

**目标**: 告诉系统哪些指标"越大越好"，哪些"越小越好"

```python
from utils.metrics import define_sign_functions

# 1. 使用默认的sign函数（适用于公平性研究）
sign_functions = define_sign_functions()

# 2. 或者自定义您的sign函数
# Sign函数格式: lambda current_value, change -> '+' 或 '-'
# '+' 表示改进, '-' 表示恶化

custom_sign_functions = {}

# 示例1: 越大越好的指标（如考试分数）
custom_sign_functions['score'] = lambda cur, change: '+' if change > 0 else '-'

# 示例2: 越小越好的指标（如错误率）
custom_sign_functions['error_rate'] = lambda cur, change: '+' if change < 0 else '-'

# 示例3: 接近某个目标值的指标（如体温，理想值37°C）
def temperature_sign(current, change):
    target = 37.0
    new_value = current + change
    return '+' if abs(new_value - target) < abs(current - target) else '-'

custom_sign_functions['temperature'] = temperature_sign

# 示例4: 范围约束的指标（如PH值，理想范围6.5-7.5）
def ph_sign(current, change):
    new_value = current + change
    ideal_range = (6.5, 7.5)

    # 计算到理想范围的距离
    def distance_to_range(value):
        if ideal_range[0] <= value <= ideal_range[1]:
            return 0
        elif value < ideal_range[0]:
            return ideal_range[0] - value
        else:
            return value - ideal_range[1]

    return '+' if distance_to_range(new_value) < distance_to_range(current) else '-'

custom_sign_functions['ph_value'] = ph_sign

# 使用您的自定义函数
sign_functions.update(custom_sign_functions)
```

### 步骤4: 检测权衡关系

**目标**: 找出哪些改进目标是冲突的

```python
from utils.tradeoff_detection import TradeoffDetector

# 1. 创建权衡检测器
detector = TradeoffDetector(sign_functions, verbose=True)

# 2. 检测权衡
tradeoffs = detector.detect_tradeoffs(
    causal_effects=causal_effects,
    require_significance=True  # 只考虑显著的因果效应
)

# 3. 查看结果
if tradeoffs:
    print(f"\n检测到 {len(tradeoffs)} 个权衡关系:")

    for i, tradeoff in enumerate(tradeoffs, 1):
        print(f"\n权衡 {i}:")
        print(f"  干预变量: {tradeoff['intervention']}")
        print(f"  冲突指标:")
        print(f"    - {tradeoff['target1']}: {tradeoff['sign1']} (ATE={tradeoff['ate1']:.4f})")
        print(f"    - {tradeoff['target2']}: {tradeoff['sign2']} (ATE={tradeoff['ate2']:.4f})")
        print(f"  含义: 改进{tradeoff['target1']}会恶化{tradeoff['target2']}")

    # 4. 生成摘要表
    summary = detector.summarize_tradeoffs(tradeoffs)
    print("\n权衡摘要表:")
    print(summary.to_string(index=False))

    # 5. 保存结果
    summary.to_csv('results/tradeoffs.csv', index=False)

    # 6. 可视化（如果matplotlib可用）
    try:
        detector.visualize_tradeoffs(tradeoffs, 'results/tradeoffs.png')
        print("\n权衡可视化已保存到: results/tradeoffs.png")
    except Exception as e:
        print(f"\n可视化失败: {e}")
else:
    print("\n✓ 未检测到权衡关系（可能都是双赢或双输的情况）")
```

**输出解读**:
- **权衡关系**: 改进指标A会导致指标B恶化
- **干预变量**: 可以操作的变量（如学习时间、方法等）
- **冲突指标**: 相互矛盾的优化目标
- **实践意义**: 需要在冲突目标之间做出选择

---

## 📊 完整示例：学生学习效果研究

### 研究问题

**目标**: 研究不同学习策略对学生表现的影响

**变量**:
- 干预: 学习时间、学习方法
- 结果: 考试分数、学习效率、压力水平

### 完整代码

```python
import pandas as pd
import numpy as np
from utils.causal_discovery import CausalGraphLearner
from utils.causal_inference import CausalInferenceEngine
from utils.tradeoff_detection import TradeoffDetector

# ============================================================================
# 步骤1: 准备数据
# ============================================================================
print("=" * 70)
print("学生学习效果因果分析")
print("=" * 70)

# 生成模拟数据（实际使用时替换为您的真实数据）
np.random.seed(42)
n_students = 200

data = pd.DataFrame({
    'study_hours': np.random.uniform(1, 8, n_students),
    'method_intensity': np.random.uniform(0, 1, n_students),  # 0=传统, 1=创新
    'prev_score': np.random.uniform(60, 90, n_students),
    'motivation': np.random.uniform(1, 10, n_students),
})

# 生成因果关系的数据
data['homework_done'] = (
    0.5 * data['study_hours'] +
    0.3 * data['motivation'] +
    np.random.normal(0, 1, n_students)
)

data['exam_score'] = (
    data['prev_score'] +
    5 * data['study_hours'] +
    10 * data['method_intensity'] +
    2 * data['homework_done'] +
    np.random.normal(0, 5, n_students)
)

data['stress_level'] = (
    3 * data['study_hours'] +
    -5 * data['motivation'] +
    np.random.normal(0, 2, n_students)
)

data['efficiency'] = (
    data['exam_score'] / (data['study_hours'] + 1) +
    np.random.normal(0, 2, n_students)
)

print(f"\n数据准备完成: {len(data)} 个学生, {len(data.columns)} 个变量")
print("\n变量列表:")
for col in data.columns:
    print(f"  - {col}")

# ============================================================================
# 步骤2: 学习因果图
# ============================================================================
print("\n" + "=" * 70)
print("步骤1: 学习因果图")
print("=" * 70)

learner = CausalGraphLearner(
    n_vars=len(data.columns),
    n_steps=1000,  # 演示用，实际可以增加到2000-5000
    alpha=0.1,
    random_seed=42
)

print("\n正在学习因果图（这可能需要1-2分钟）...")
causal_graph = learner.fit(data, verbose=True)

edges = learner.get_edges(threshold=0.3)
print(f"\n✓ 检测到 {len(edges)} 条因果边 (阈值=0.3)")

print("\n关键因果关系:")
for source, target, weight in edges[:10]:
    print(f"  {data.columns[source]} → {data.columns[target]}: {weight:.3f}")

learner.save_graph('results/student_causal_graph.npy')

# ============================================================================
# 步骤3: 估计因果效应
# ============================================================================
print("\n" + "=" * 70)
print("步骤2: 估计因果效应")
print("=" * 70)

engine = CausalInferenceEngine(verbose=True)

print("\n正在估计因果效应（这可能需要1-2分钟）...")
causal_effects = engine.analyze_all_edges(
    data=data,
    causal_graph=causal_graph,
    var_names=data.columns.tolist(),
    threshold=0.3
)

print(f"\n✓ 分析了 {len(causal_effects)} 个因果效应")

significant = engine.get_significant_effects()
if significant:
    print(f"\n显著的因果效应 (共{len(significant)}个):")
    for edge, result in list(significant.items())[:10]:
        print(f"  {edge}:")
        print(f"    ATE = {result['ate']:.4f}")
        print(f"    95% CI = [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

engine.save_results('results/student_causal_effects.csv')

# ============================================================================
# 步骤4: 定义优化方向
# ============================================================================
print("\n" + "=" * 70)
print("步骤3: 定义优化方向")
print("=" * 70)

sign_functions = {}

# 越高越好
sign_functions['exam_score'] = lambda cur, change: '+' if change > 0 else '-'
sign_functions['efficiency'] = lambda cur, change: '+' if change > 0 else '-'
sign_functions['homework_done'] = lambda cur, change: '+' if change > 0 else '-'
sign_functions['motivation'] = lambda cur, change: '+' if change > 0 else '-'

# 越低越好
sign_functions['stress_level'] = lambda cur, change: '+' if change < 0 else '-'

# 其他变量（中性）
sign_functions['study_hours'] = lambda cur, change: '+' if change > 0 else '-'
sign_functions['method_intensity'] = lambda cur, change: '+' if change > 0 else '-'
sign_functions['prev_score'] = lambda cur, change: '0'  # 不可改变

print("\n定义的优化方向:")
print("  越高越好: exam_score, efficiency, homework_done, motivation")
print("  越低越好: stress_level")

# ============================================================================
# 步骤5: 检测权衡
# ============================================================================
print("\n" + "=" * 70)
print("步骤4: 检测权衡关系")
print("=" * 70)

detector = TradeoffDetector(sign_functions, verbose=True)

tradeoffs = detector.detect_tradeoffs(
    causal_effects=causal_effects,
    require_significance=True
)

if tradeoffs:
    print(f"\n检测到 {len(tradeoffs)} 个权衡关系:")

    for i, tradeoff in enumerate(tradeoffs, 1):
        print(f"\n权衡 {i}:")
        print(f"  如果增加: {tradeoff['intervention']}")
        print(f"    → {tradeoff['target1']}: {tradeoff['sign1']} ({tradeoff['ate1']:+.4f})")
        print(f"    → {tradeoff['target2']}: {tradeoff['sign2']} ({tradeoff['ate2']:+.4f})")

    summary = detector.summarize_tradeoffs(tradeoffs)
    print("\n权衡摘要:")
    print(summary.to_string(index=False))

    summary.to_csv('results/student_tradeoffs.csv', index=False)

    try:
        detector.visualize_tradeoffs(tradeoffs, 'results/student_tradeoffs.png')
    except:
        pass
else:
    print("\n✓ 未检测到权衡关系")

# ============================================================================
# 步骤6: 总结
# ============================================================================
print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)

print("\n生成的文件:")
print("  - results/student_causal_graph.npy")
print("  - results/student_causal_effects.csv")
if tradeoffs:
    print("  - results/student_tradeoffs.csv")

print("\n关键发现:")
if significant:
    print(f"  1. 发现 {len(significant)} 个显著因果效应")
if tradeoffs:
    print(f"  2. 发现 {len(tradeoffs)} 个权衡关系")
    print(f"     → 改进某些目标会导致其他目标恶化")
else:
    print(f"  2. 未发现权衡关系")
    print(f"     → 可能存在双赢策略")
```

---

## 🎓 进阶使用技巧

### 技巧1: 调整DiBS参数

```python
# 更准确但更慢
learner = CausalGraphLearner(
    n_vars=20,
    n_steps=5000,    # 增加迭代次数
    alpha=0.9,       # 更高的alpha得到更密集的图
    random_seed=42
)

# 更快但可能不够准确
learner = CausalGraphLearner(
    n_vars=20,
    n_steps=500,     # 减少迭代次数
    alpha=0.05,      # 更低的alpha得到更稀疏的图
    random_seed=42
)
```

### 技巧2: 处理大规模数据

```python
# 如果变量太多，选择关键变量
key_variables = [
    'intervention_var1',
    'intervention_var2',
    'outcome_var1',
    'outcome_var2',
    'important_confounder'
]

data_subset = data[key_variables]

# 然后在子集上运行分析
learner = CausalGraphLearner(n_vars=len(key_variables), ...)
```

### 技巧3: 估计单个因果效应

```python
# 如果只关心特定的因果关系
engine = CausalInferenceEngine()

# 单独估计 X → Y
ate, ci = engine.estimate_ate(
    data=data,
    treatment='study_hours',
    outcome='exam_score',
    confounders=['prev_score', 'motivation']
)

print(f"study_hours → exam_score: ATE={ate:.4f}, CI={ci}")
```

### 技巧4: 保存和加载结果

```python
# 保存因果图
learner.save_graph('results/graph.npy')

# 稍后加载
learner2 = CausalGraphLearner(n_vars=20)
learner2.load_graph('results/graph.npy')

# 保存因果效应
engine.save_results('results/effects.csv')

# 加载
import pandas as pd
effects_df = pd.read_csv('results/effects.csv')
```

---

## ❓ 常见问题

### Q1: 我的数据样本量很小（<50），可以用吗？

**答**: DiBS需要至少50-100个样本才能可靠。如果样本太少：
- 考虑使用简化的相关性分析
- 或收集更多数据
- 或使用先验知识构建因果图

### Q2: 学习出的因果图太复杂/太稀疏怎么办？

**答**: 调整alpha参数：
- 图太密集 → 降低alpha (0.1 → 0.05)
- 图太稀疏 → 提高alpha (0.1 → 0.5)
- 也可以调整threshold (边的最小强度)

### Q3: DML估计失败怎么办？

**答**: 可能原因：
- EconML未安装 → 会自动降级到简化方法
- 数据有问题 → 检查缺失值和异常值
- 混淆因素太少 → 手动指定更多混淆因素

### Q4: 没有检测到权衡关系，正常吗？

**答**: 完全正常！可能原因：
- 真的没有权衡（存在双赢策略）
- 因果效应不显著（增加样本量）
- Sign函数定义不当（重新检查）

### Q5: 运行太慢怎么办？

**答**: 加速方法：
- 减少DiBS迭代次数 (5000 → 1000)
- 减少变量数量（选择关键变量）
- 提高threshold（只分析强因果关系）

---

## 📚 参考资源

### 本项目文档

1. `PROJECT_STATUS_SUMMARY.md` - 项目完成状态
2. `TEST_VALIDATION_REPORT.md` - 测试验证报告
3. `STAGE1_2_COMPLETE_REPORT.md` - 技术实现细节
4. `PAPER_COMPARISON_REPORT.md` - 与论文对比

### 外部资源

1. **DiBS**: https://github.com/larslorch/dibs
2. **EconML**: https://econml.azurewebsites.net/
3. **因果推断教程**: "Causal Inference: The Mixtape" (免费在线书)

---

## 💡 最后的建议

1. **从小规模开始**: 先用少量变量（5-10个）验证流程
2. **可视化很重要**: 查看因果图能帮助理解结果
3. **解释要谨慎**: 因果关系不等于相关性，但也需要领域知识验证
4. **迭代改进**: 根据结果调整变量选择和参数设置

---

**祝您研究顺利！** 🎉

如有问题，请参考项目文档或查看代码注释。
