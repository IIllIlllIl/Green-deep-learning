# 能耗数据因果分析处理方案

**创建日期**: 2025-12-22
**数据源**: `data/energy_research/raw/energy_data_original.csv` (data.csv的副本, 54列)
**状态**: 📋 方案阶段 - 等待决策

---

## 📊 当前数据概况

### 基本信息

- **文件**: `energy_data_original.csv`
- **行数**: 677行（676个实验 + 1行表头）
- **列数**: 54列
- **来源**: `/home/green/energy_dl/nightly/data/data.csv`
- **复制时间**: 2025-12-22

### 数据完整性

| 指标 | 数量 | 比例 |
|------|------|------|
| 总实验数 | 676 | 100% |
| 训练成功 | 676 | 100.0% |
| 能耗数据完整 | 616 | 91.1% |
| 性能数据完整 | 616 | 91.1% |
| **有效样本**（能耗+性能） | **616** | **91.1%** |

---

## 🎯 研究框架

### 核心研究问题

基于您提出的三个关键点：

1. **敏感属性**: `is_parallel` - 是否为并行训练（类比Adult中的性别）
2. **干预方法**: 变异的超参数（如learning_rate变异）
3. **模型种类**: 11个不同模型 - 需要决定如何处理

### 研究目标

1. **超参数 → 能耗** 的因果影响
2. **超参数 → 性能** 的因果影响
3. **能耗 vs 性能** 的权衡关系
4. **并行模式** 的调节作用

---

## 🔍 模型种类处理 - 三个方案对比

### 当前数据中的模型分布

```python
# 11个模型（从CLAUDE.md获取）
models = {
    'fast': [
        'examples/mnist',
        'examples/mnist_ff',
        'examples/mnist_rnn',
        'examples/siamese'
    ],  # 约150样本
    'medium': [
        'pytorch_resnet_cifar10/resnet20',
        'MRT-OAST/default',
        'VulBERTa/mlp',
        'bug-localization-by-dnn-and-rvsm/default'
    ],  # 约250样本
    'slow': [
        'Person_reID_baseline_pytorch/densenet121',
        'Person_reID_baseline_pytorch/hrnet18',
        'Person_reID_baseline_pytorch/pcb'
    ]  # 约270样本
}
```

### 方案A: 作为协变量（One-Hot编码）⭐ **推荐**

**处理方式**:
```python
# 1. 创建模型ID
df['model_id'] = df['repository'] + '/' + df['model']

# 2. One-Hot编码
df_encoded = pd.get_dummies(df, columns=['model_id'], prefix='model')
# 生成: model_examples/mnist, model_VulBERTa/mlp, ... (11列)

# 3. 加入因果分析
inputs = ['learning_rate', 'batch_size', 'seed', ...] + model_cols  # 6+11=17个输入
outputs = ['energy_gpu_avg_watts', 'perf_test_accuracy', ...]  # 6个输出
# 总变量数: 17 + 6 = 23个
```

**优势**:
- ✅ 充分利用全部616个有效样本（统计功效最高）
- ✅ DiBS自动学习"模型类型 → 能耗/性能"的因果关系
- ✅ DML控制模型的混淆效应
- ✅ 只需运行1次DiBS（耗时最短）

**劣势**:
- ⚠️ 增加11个变量维度（总变量数23个，仍在合理范围）
- ⚠️ 不同模型的因果机制被平均化（无法看到模型特定模式）

**适用场景**:
- 研究"平均因果效应"（所有模型的整体影响）
- **快速验证DiBS+DML在能耗数据上的可行性**
- 发现主要的因果关系（如learning_rate → energy）

**DiBS参数建议**:
```python
DIBS_N_STEPS = 5000    # 样本多(616)，迭代数增加
DIBS_ALPHA = 0.05      # 弱稀疏性（允许更多边）
EDGE_THRESHOLD = 0.3   # 标准阈值
预期耗时: 30-60分钟（GPU）
```

---

### 方案B: 分层分析（每个模型独立）

**处理方式**:
```python
# 按模型分组
for model_id in df['model_id'].unique():  # 11个模型
    df_model = df[df['model_id'] == model_id]

    if len(df_model) >= 20:  # 样本量足够
        # 独立运行DiBS
        learner = CausalGraphLearner(...)
        graph = learner.fit(df_model)

        # 保存模型特定因果图
        learner.save_graph(f'results/{model_id}_causal_graph.npy')
```

**优势**:
- ✅ 每个模型有独立的因果图（更精细）
- ✅ 可以发现"模型特定"的因果机制
  - 例如: CNN vs Transformer的能耗模式不同
- ✅ 可对比不同模型的因果结构差异

**劣势**:
- ❌ 样本量减少（616÷11 ≈ 56个/模型）
- ❌ 统计功效降低（部分边可能不显著）
- ❌ 需要运行11次DiBS（耗时长：11 × 30分钟 ≈ 5.5小时）

**适用场景**:
- 深入研究每个模型的因果机制
- 对比不同模型的能耗因果模式
- **第二阶段深入分析**（在方案A验证成功后）

**DiBS参数建议**:
```python
DIBS_N_STEPS = 3000    # 样本少(~56)，迭代数适中
DIBS_ALPHA = 0.1       # 标准稀疏性
EDGE_THRESHOLD = 0.3
预期耗时: 11 × 30分钟 ≈ 5.5小时（GPU）
```

---

### 方案C: 模型类型分组（折中）

**处理方式**:
```python
# 按训练速度分为3组
groups = {
    'fast_models': 4个模型,    # ~150样本
    'medium_models': 4个模型,  # ~250样本
    'slow_models': 3个模型     # ~270样本
}

# 每组独立分析
for group_name, models in groups.items():
    df_group = df[df['model_id'].isin(models)]
    # 运行DiBS...
```

**优势**:
- ✅ 平衡样本量（150-270/组）和精细度
- ✅ 可以对比"快速 vs 中速 vs 慢速模型"的能耗模式
- ✅ 每组样本量充足（>100）
- ✅ 只需运行3次DiBS（耗时适中）

**劣势**:
- ⚠️ 组内模型差异被忽略
- ⚠️ 仍需运行3次（耗时 ≈ 1.5小时）

**适用场景**:
- 对比不同训练速度模型的能耗特征
- 平衡精细度和效率

**DiBS参数建议**:
```python
DIBS_N_STEPS = 5000    # 样本充足
DIBS_ALPHA = 0.05
预期耗时: 3 × 30分钟 ≈ 1.5小时（GPU）
```

---

## 📊 方案对比总结

| 维度 | 方案A（协变量） | 方案B（分层） | 方案C（分组） |
|------|---------------|--------------|--------------|
| **样本量** | 616 | ~56/模型 | 150-270/组 |
| **DiBS次数** | 1次 | 11次 | 3次 |
| **预期耗时** | 0.5-1小时 | 5-6小时 | 1-2小时 |
| **变量数** | 23个 | 11个 | 16个 |
| **精细度** | 平均效应 | 模型特定 | 组特定 |
| **统计功效** | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| **推荐阶段** | **第一阶段** | 第二阶段 | 可选 |

---

## 🔬 变量设计（方案A）

### 输入变量（Causes）

#### 1. 超参数（连续型）- 6个

```python
hyperparams = [
    'learning_rate',      # 学习率
    'batch_size',         # 批量大小
    'epochs',             # 训练轮数
    'dropout',            # Dropout比率
    'weight_decay',       # 权重衰减
    'seed'                # 随机种子 ⭐ 用户要求添加
]
```

#### 2. 并行模式（二值）- 1个

```python
mode = [
    'is_parallel'         # 0=非并行, 1=并行
]
```

**作用**: 类似Adult中的性别，作为敏感属性/分组变量

#### 3. 模型类型（One-Hot编码）- 11个

```python
models = [
    'model_examples/mnist',
    'model_examples/mnist_ff',
    'model_examples/mnist_rnn',
    'model_examples/siamese',
    'model_pytorch_resnet_cifar10/resnet20',
    'model_MRT-OAST/default',
    'model_VulBERTa/mlp',
    'model_bug-localization-by-dnn-and-rvsm/default',
    'model_Person_reID_baseline_pytorch/densenet121',
    'model_Person_reID_baseline_pytorch/hrnet18',
    'model_Person_reID_baseline_pytorch/pcb'
]
```

**总输入**: 6 + 1 + 11 = **18个变量**

---

### 输出变量（Effects）

#### 1. 能耗指标 - 3个

```python
energy = [
    'energy_gpu_avg_watts',       # GPU平均功率（主要关注）⭐
    'energy_gpu_total_joules',    # GPU总能耗
    'energy_cpu_total_joules'     # CPU总能耗
]
```

#### 2. 性能指标 - 2个

```python
performance = [
    'perf_test_accuracy',         # 测试准确率（主要关注）⭐
    'perf_test_loss'              # 测试损失
]
```

#### 3. 训练时长 - 1个

```python
time = [
    'duration_seconds'            # 训练时长（中介变量）
]
```

**总输出**: 3 + 2 + 1 = **6个变量**

**总变量数**: 18 + 6 = **24个变量**

---

## 🔄 数据处理流程（方案A）

### 步骤1: 数据加载与清洗

```python
import pandas as pd
import numpy as np

# 1. 加载数据
df = pd.read_csv('data/energy_research/raw/energy_data_original.csv')
print(f"原始数据: {len(df)} 行")

# 2. 筛选有效数据（训练成功 + 能耗完整 + 性能完整）
mask = (
    (df['training_success'] == True) &
    (df['energy_gpu_avg_watts'].notna()) &
    (df['perf_test_accuracy'].notna())
)
df_valid = df[mask].copy()
print(f"有效数据: {len(df_valid)} 行 ({len(df_valid)/len(df)*100:.1f}%)")
# 预期: 616 行 (91.1%)
```

### 步骤2: 创建复合字段

```python
# 1. 模型ID
df_valid['model_id'] = df_valid['repository'] + '/' + df_valid['model']
print(f"模型种类: {df_valid['model_id'].nunique()} 个")
# 预期: 11个

# 2. 并行模式（如果不是Boolean，则转换）
if df_valid['is_parallel'].dtype == 'object':
    df_valid['is_parallel'] = df_valid['is_parallel'].map({
        'True': 1, 'False': 0, True: 1, False: 0
    })
```

### 步骤3: 处理缺失超参数

```python
# 某些模型可能没有某些超参数，用默认值填充
defaults = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
    'dropout': 0.0,
    'weight_decay': 0.0,
    'seed': 42
}

for param, default_val in defaults.items():
    n_missing = df_valid[param].isna().sum()
    if n_missing > 0:
        print(f"{param}: {n_missing} 缺失 → 填充为 {default_val}")
        df_valid[param].fillna(default_val, inplace=True)
```

### 步骤4: One-Hot编码模型

```python
# One-Hot编码
df_encoded = pd.get_dummies(df_valid, columns=['model_id'], prefix='model')

# 识别模型列
model_cols = [col for col in df_encoded.columns if col.startswith('model_')]
print(f"模型列数: {len(model_cols)}")
# 预期: 11列
```

### 步骤5: 标准化连续特征

```python
from sklearn.preprocessing import StandardScaler

# 识别需要标准化的列
continuous_cols = [
    'learning_rate', 'batch_size', 'epochs', 'dropout', 'weight_decay', 'seed',
    'energy_gpu_avg_watts', 'energy_gpu_total_joules', 'energy_cpu_total_joules',
    'perf_test_accuracy', 'perf_test_loss', 'duration_seconds'
]

# 标准化
scaler = StandardScaler()
df_encoded[continuous_cols] = scaler.fit_transform(df_encoded[continuous_cols])

# 验证
print(f"均值: {df_encoded[continuous_cols].mean().mean():.6f}")  # 应接近0
print(f"标准差: {df_encoded[continuous_cols].std().mean():.6f}")  # 应接近1
```

### 步骤6: 准备因果分析数据

```python
# 定义变量
input_vars = [
    'learning_rate', 'batch_size', 'epochs', 'dropout', 'weight_decay', 'seed',
    'is_parallel'
] + model_cols

output_vars = [
    'energy_gpu_avg_watts', 'energy_gpu_total_joules', 'energy_cpu_total_joules',
    'perf_test_accuracy', 'perf_test_loss', 'duration_seconds'
]

# 提取因果分析数据
all_vars = input_vars + output_vars
causal_data = df_encoded[all_vars].copy()

print(f"因果分析数据: {len(causal_data)} 样本 × {len(all_vars)} 变量")
# 预期: 616 样本 × 24 变量
```

### 步骤7: 保存处理后的数据

```python
# 保存到processed目录
causal_data.to_csv('data/energy_research/processed/energy_causal_data.csv', index=False)

# 保存变量信息（JSON）
import json
var_info = {
    'input_vars': input_vars,
    'output_vars': output_vars,
    'n_samples': len(causal_data),
    'n_vars': len(all_vars)
}
with open('data/energy_research/processed/variable_info.json', 'w') as f:
    json.dump(var_info, f, indent=2)
```

---

## 🚀 DiBS + DML 执行方案

### DiBS因果图学习

```python
from utils.causal_discovery import CausalGraphLearner

# 创建学习器（保守参数）
learner = CausalGraphLearner(
    n_vars=24,           # 24个变量
    n_steps=5000,        # 样本多(616)，迭代多
    alpha=0.05,          # 弱稀疏性（允许更多边）
    random_seed=42
)

# 学习因果图
causal_graph = learner.fit(causal_data, verbose=True)

# 提取因果边（阈值=0.3）
edges = learner.get_edges(threshold=0.3)
print(f"检测到 {len(edges)} 条因果边")

# 保存
learner.save_graph('results/energy_research/energy_causal_graph.npy')
```

**预期结果**:
- 因果边数: 15-40条
- 运行时间: 30-60分钟（GPU）
- 关键边示例:
  - `learning_rate → energy_gpu_avg_watts`
  - `batch_size → energy_gpu_avg_watts`
  - `is_parallel → energy_gpu_avg_watts`（并行模式影响）
  - `model_* → energy_gpu_avg_watts`（模型影响）

---

### DML因果推断

```python
from utils.causal_inference import CausalInferenceEngine

# 创建推断引擎
engine = CausalInferenceEngine(verbose=True)

# 分析所有因果边
causal_effects = engine.analyze_all_edges(
    data=causal_data,
    causal_graph=causal_graph,
    var_names=all_vars,
    threshold=0.3
)

# 保存结果
engine.save_results('results/energy_research/energy_causal_effects.csv')

# 筛选显著效应
significant = engine.get_significant_effects()
print(f"统计显著: {len(significant)} / {len(causal_effects)}")
```

**预期结果**:
- ATE示例:
  - `learning_rate → energy_gpu_avg_watts`: ATE = +X watts（学习率提高 → 能耗增加）
  - `batch_size → energy_gpu_avg_watts`: ATE = +Y watts
  - `is_parallel → energy_gpu_avg_watts`: ATE = +Z watts（并行 → 能耗增加）

---

### 权衡检测

```python
from utils.tradeoff_detection import TradeoffDetector

detector = TradeoffDetector()

# 检测"能耗 vs 性能"权衡
tradeoffs = detector.detect_tradeoffs(
    causal_graph=causal_graph,
    causal_effects=causal_effects,
    outcome1='energy_gpu_avg_watts',
    outcome2='perf_test_accuracy'
)

# 示例输出:
# {
#   'common_cause': 'learning_rate',
#   'outcome1_effect': +15.3 (能耗增加),
#   'outcome2_effect': -0.05 (准确率降低),
#   'direction': 'opposite',    # 权衡！
#   'strength': 'moderate'
# }
```

---

## ✅ 推荐执行计划

### 阶段1: 快速验证（方案A）⭐ **优先**

**目标**: 验证DiBS+DML在能耗数据上的可行性

**步骤**:
1. 执行数据处理流程（步骤1-7）
2. 运行DiBS学习因果图
3. 运行DML估计因果效应
4. 生成初步分析报告

**预期耗时**: 1-2小时
**预期输出**:
- `energy_causal_data.csv` - 处理后的因果分析数据
- `energy_causal_graph.npy` - 因果图
- `energy_causal_effects.csv` - 因果效应

**成功标准**:
- DiBS成功完成（不超时）
- 检测到 >10条因果边
- >50%的边统计显著
- 发现至少1个"能耗 vs 性能"权衡

---

### 阶段2: 深入分析（方案B或C）

**前提**: 阶段1成功完成，且发现有趣结果

**选择**:
- **方案B**: 如果想对比不同模型的因果机制
- **方案C**: 如果想对比快速/中速/慢速模型的能耗模式

**预期耗时**: 2-6小时

---

## 📝 需要您决定的问题

### 1. 模型种类处理方式 ⭐ **核心决策**

**问题**: 11个不同的模型应该如何处理？

**选项**:
- [ ] **方案A**: 作为协变量（One-Hot编码）- 推荐首选
- [ ] **方案B**: 分层分析（每个模型独立）
- [ ] **方案C**: 模型类型分组（快速/中速/慢速）

**建议**: 先选择方案A快速验证，如果成功再考虑B或C深入分析

---

### 2. 变量选择

**问题**: 是否同意以下变量设计？

**输入变量（18个）**:
- [ ] 6个超参数: learning_rate, batch_size, epochs, dropout, weight_decay, seed ⭐
- [ ] 1个并行模式: is_parallel
- [ ] 11个模型: model_* (One-Hot编码)

**输出变量（6个）**:
- [ ] 3个能耗指标: energy_gpu_avg_watts, energy_gpu_total_joules, energy_cpu_total_joules
- [ ] 2个性能指标: perf_test_accuracy, perf_test_loss
- [ ] 1个时长: duration_seconds

**修改建议**: 如果您想添加/删除变量，请说明

---

### 3. DiBS参数

**问题**: 是否同意以下DiBS参数？

- [ ] 迭代次数: 5000步（样本量616，较多）
- [ ] 稀疏性: alpha=0.05（弱稀疏，允许更多边）
- [ ] 边阈值: threshold=0.3（标准）

**修改建议**: 如果想调整参数，请说明

---

### 4. 执行优先级

**问题**: 是否同意以下执行计划？

- [ ] **阶段1**: 先执行方案A（协变量方法），验证可行性
- [ ] **阶段2**: 如果阶段1成功，再考虑方案B或C深入分析

---

## 📋 下一步行动

**等待您的决策**:

1. **选择模型处理方案** (A/B/C)
2. **确认变量设计** (或提出修改)
3. **确认DiBS参数** (或提出调整)
4. **批准执行计划**

**决策后**:
- 我将创建数据转换脚本（`convert_energy_data.py`）
- 我将创建实验运行脚本（`demo_energy_analysis.py`）
- 您可以运行脚本开始因果分析

---

**文档状态**: 📋 方案阶段 - 等待决策
**创建时间**: 2025-12-22
**维护者**: Analysis模块
