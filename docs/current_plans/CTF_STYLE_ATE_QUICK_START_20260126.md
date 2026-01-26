# CTF风格ATE计算快速使用指南

**版本**: 1.0
**最后更新**: 2026-01-26
**状态**: 可用于生产环境

---

## 快速开始

### 安装依赖

```bash
# 激活causal-research环境（已安装EconML）
conda activate causal-research
```

### 基本使用

```python
from analysis.utils.causal_inference import CausalInferenceEngine
import pandas as pd
import numpy as np

# 准备数据
data = pd.read_csv('data/data.csv')
var_names = ['learning_rate', 'batch_size', 'energy', 'accuracy']

# 准备因果图
causal_graph = np.load('causal_graph.npy')

# 创建引擎
ci = CausalInferenceEngine(verbose=True)

# 分析所有边
results = ci.analyze_all_edges_ctf_style(
    data=data,
    causal_graph=causal_graph,
    var_names=var_names,
    threshold=0.3
)
```

---

## 功能特性

### 1. MVP版本（迭代1）

**核心功能**：CTF风格混淆因素识别

```python
# 无需ref_df和T0/T1
results = ci.analyze_all_edges_ctf_style(
    data=data,
    causal_graph=causal_graph,
    var_names=var_names,
    threshold=0.3
)
```

**适用场景**：
- 快速验证CTF逻辑
- 不需要baseline对比
- 简单的因果分析

### 2. V1.0版本（迭代2）

**扩展功能**：支持ref_df和T0/T1

#### 使用ref_df

```python
# 构建参考数据集
ref_df = ci.build_reference_df(
    data,
    strategy="non_parallel"  # 推荐
)

# 使用ref_df分析
results = ci.analyze_all_edges_ctf_style(
    data=data,
    causal_graph=causal_graph,
    var_names=var_names,
    ref_df=ref_df
)
```

#### 使用T0/T1

```python
# 使用T0/T1策略
results = ci.analyze_all_edges_ctf_style(
    data=data,
    causal_graph=causal_graph,
    var_names=var_names,
    t_strategy="quantile"  # 25/75分位数
)
```

#### 完整V1.0

```python
# 同时使用ref_df和T0/T1
ref_df = ci.build_reference_df(data, strategy="non_parallel")

results = ci.analyze_all_edges_ctf_style(
    data=data,
    causal_graph=causal_graph,
    var_names=var_names,
    ref_df=ref_df,
    t_strategy="quantile"
)
```

---

## 参数说明

### ref_df策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `non_parallel` | 非并行模式作为baseline | 能耗分析，有is_parallel列 |
| `mean` | 全局均值 | 简单baseline |
| `group_mean` | 分组均值 | 多模型分析 |

### T0/T1策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `quantile` | 25/75分位数 | 通用分析（推荐） |
| `min_max` | 最小值/最大值 | 探索极端效应 |
| `mean_std` | 均值±标准差 | 正态分布数据 |

---

## 结果解读

### 结果格式

```python
{
    'edge_name': {
        'ate': float,              # 平均处理效应
        'ci_lower': float,         # 95%置信区间下界
        'ci_upper': float,         # 95%置信区间上界
        'is_significant': bool,    # 是否统计显著（p<0.05）
        'weight': float,           # 因果图边权重
        'confounders': List[str]   # 混淆因素列表
    }
}
```

### 判断显著性

```python
for edge, result in results.items():
    if result['is_significant']:
        print(f"{edge}: ATE={result['ate']:.4f} ***")
    else:
        print(f"{edge}: ATE={result['ate']:.4f} (不显著)")
```

### 显著性判断规则

- **显著**：置信区间不包含0
  - `ci_lower > 0`：正向因果效应
  - `ci_upper < 0`：负向因果效应
- **不显著**：置信区间包含0

---

## 常见问题

### Q1: 如何选择ref_df策略？

**能耗分析**：
```python
# 使用非并行模式作为baseline
ref_df = ci.build_reference_df(data, strategy="non_parallel")
```

**通用分析**：
```python
# 使用全局均值
ref_df = ci.build_reference_df(data, strategy="mean")
```

**多模型分析**：
```python
# 按模型分组
ref_df = ci.build_reference_df(
    data,
    strategy="group_mean",
    groupby_cols=['model_name']
)
```

### Q2: 如何选择T0/T1策略？

**通用情况**（推荐）：
```python
# 使用25/75分位数
t_strategy="quantile"
```

**正态数据**：
```python
# 使用均值±标准差
t_strategy="mean_std"
```

**探索性分析**：
```python
# 使用最小最大值
t_strategy="min_max"
```

### Q3: 如何处理错误？

**ref_df缺少列**：
```python
# 错误：ref_df中缺少混淆因素列
# 解决：确保ref_df包含所有需要的列
ref_df = data[confounders + [treatment]].copy()
```

**T1 <= T0**：
```python
# 错误：T1必须大于T0
# 解决：检查数据变异性
print(data[treatment].describe())
```

### Q4: 性能优化建议

**大数据集**：
```python
# 使用均值策略（更快）
ref_df = ci.build_reference_df(data, strategy="mean")

# 或使用分组均值
ref_df = ci.build_reference_df(
    data,
    strategy="group_mean",
    groupby_cols=['category']
)
```

**大量边**：
```python
# 批量分析，verbose=False
ci = CausalInferenceEngine(verbose=False)
results = ci.analyze_all_edges_ctf_style(...)
```

---

## 完整示例

### 示例1：能耗分析

```python
from analysis.utils.causal_inference import CausalInferenceEngine
import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv('data/data.csv')

# 选择数值列
numeric_cols = data.select_dtypes(include=[np.number]).columns[:20]
data_subset = data[numeric_cols].dropna()

# 准备因果图（假设已有）
causal_graph = np.load('causal_graph.npy')
var_names = list(data_subset.columns)

# 创建引擎
ci = CausalInferenceEngine(verbose=True)

# 构建ref_df（非并行模式）
ref_df = ci.build_reference_df(
    data_subset,
    strategy="non_parallel"
)

# 分析
results = ci.analyze_all_edges_ctf_style(
    data=data_subset,
    causal_graph=causal_graph,
    var_names=var_names,
    threshold=0.3,
    ref_df=ref_df,
    t_strategy="quantile"
)

# 输出显著结果
print("\n显著的因果效应:")
for edge, result in results.items():
    if result['is_significant']:
        print(f"  {edge}: ATE={result['ate']:.4f}, CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
```

### 示例2：单边分析

```python
# 分析单条边
ate, ci = ci.estimate_ate(
    data=data_subset,
    treatment='learning_rate',
    outcome='energy',
    confounders=['batch_size', 'epochs'],
    ref_df=ref_df,
    t_strategy="quantile"
)

print(f"ATE: {ate:.4f}")
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"显著: {'是' if not (ci[0] <= 0 <= ci[1]) else '否'}")
```

---

## 性能基准

| 数据量 | 边数 | MVP耗时 | V1.0耗时 |
|--------|------|---------|----------|
| 500行  | 10条 | ~1s     | ~2s      |
| 1000行 | 20条 | ~2s     | ~4s      |
| 5000行 | 50条 | ~10s    | ~20s     |

**注**：性能取决于数据规模和计算资源。

---

## API参考

### CausalInferenceEngine

#### 方法

**`analyze_all_edges_ctf_style()`**
```python
def analyze_all_edges_ctf_style(
    data: pd.DataFrame,
    causal_graph: np.ndarray,
    var_names: List[str],
    threshold: float = 0.3,
    ref_df: pd.DataFrame = None,
    t_strategy: str = None
) -> Dict[str, Dict]
```

**`build_reference_df()`**
```python
def build_reference_df(
    data: pd.DataFrame,
    strategy: str = "non_parallel",
    groupby_cols: Optional[List[str]] = None
) -> pd.DataFrame
```

**`compute_T0_T1()`**
```python
def compute_T0_T1(
    data: pd.DataFrame,
    treatment: str,
    strategy: str = "quantile"
) -> Tuple[float, float]
```

**`estimate_ate()`**
```python
def estimate_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    controls: Optional[List[str]] = None,
    ref_df: Optional[pd.DataFrame] = None,
    T0: Optional[float] = None,
    T1: Optional[float] = None,
    t_strategy: Optional[str] = None
) -> Tuple[float, Tuple[float, float]]
```

---

## 相关文档

- 实施报告：`ATE_INTEGRATION_COMPLETION_REPORT_20260126.md`
- 实施方案：`ATE_INTEGRATION_IMPLEMENTATION_PLAN_20260125.md`
- 可行性报告：`ATE_INTEGRATION_FEASIBILITY_REPORT_20260125.md`

---

## 支持

如有问题，请：
1. 查看完整实施报告
2. 查看测试用例：`tests/test_ctf_style_ate.py`
3. 查看代码文档：`utils/causal_inference.py`

---

**文档版本**: 1.0
**最后更新**: 2026-01-26
