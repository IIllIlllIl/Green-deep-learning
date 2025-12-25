# 学术研究级完整复现实施计划

**目标**: 从45%复现度提升到90%+，达到可发表的学术研究标准
**时间估计**: 6-8周全职工作
**最终目标**: 可重现论文的主要实验结果和结论

---

## 📋 目录

1. [关键差距分析](#关键差距分析)
2. [实施路线图](#实施路线图)
3. [阶段1: DiBS因果图学习](#阶段1-dibs因果图学习)
4. [阶段2: DML因果推断](#阶段2-dml因果推断)
5. [阶段3: 完整公平性方法](#阶段3-完整公平性方法)
6. [阶段4: 真实数据集集成](#阶段4-真实数据集集成)
7. [阶段5: 大规模实验](#阶段5-大规模实验)
8. [阶段6: 结果验证](#阶段6-结果验证)
9. [技术难点与解决方案](#技术难点与解决方案)
10. [资源需求](#资源需求)

---

## 🎯 关键差距分析

### 当前状态 vs 论文要求

| 组件 | 当前 | 论文要求 | 优先级 | 难度 | 时间 |
|------|------|----------|--------|------|------|
| **因果图学习** | 相关性 | DiBS(10k迭代) | 🔴 最高 | ⭐⭐⭐⭐⭐ | 2周 |
| **因果推断** | 简单差值 | DML(EconML) | 🔴 最高 | ⭐⭐⭐⭐ | 1周 |
| **公平性方法** | 2/12 | 12个完整 | 🟡 高 | ⭐⭐⭐ | 1周 |
| **数据集** | 模拟 | Adult/COMPAS/German | 🟡 高 | ⭐⭐ | 3天 |
| **Alpha采样** | 3点 | 10点 | 🟢 中 | ⭐ | 1天 |
| **鲁棒性测试** | 随机噪声 | 真实对抗攻击 | 🟢 中 | ⭐⭐⭐ | 3天 |
| **统计检验** | 无 | 置信区间 | 🟢 中 | ⭐⭐ | 2天 |
| **实验规模** | 6点 | 726点 | 🟡 高 | ⭐ | 持续 |

**关键瓶颈**:
1. 🔴 DiBS因果图学习（最难，最关键）
2. 🔴 DML因果推断（次难，核心算法）
3. 🟡 12个公平性方法（工程量大）

---

## 🗺️ 实施路线图

```
总时间: 6-8周
复现度: 45% → 90%+

Week 1-2: DiBS因果图学习           [45% → 60%]  🔴 Critical
Week 3:   DML因果推断              [60% → 75%]  🔴 Critical
Week 4:   完整公平性方法            [75% → 82%]  🟡 Important
Week 5:   真实数据集 + 对抗攻击     [82% → 87%]  🟡 Important
Week 6-7: 大规模实验运行            [87% → 92%]  🟢 Scale-up
Week 8:   结果验证 + 论文对比       [92% → 95%]  ✅ Validation
```

### 里程碑

| 周数 | 里程碑 | 可验证指标 |
|------|--------|-----------|
| Week 2 | DiBS运行成功 | 生成46变量的DAG |
| Week 3 | DML集成完成 | 输出ATE+置信区间 |
| Week 4 | 12个方法可用 | 通过方法测试 |
| Week 5 | 3数据集运行 | 生成240+数据点 |
| Week 7 | 完整实验 | 生成726数据点 |
| Week 8 | 结果匹配 | 与论文图表对比 |

---

## 🔬 阶段1: DiBS因果图学习

**目标**: 实现论文的核心算法 - 使用DiBS学习因果图
**时间**: 2周
**优先级**: 🔴 最高（这是论文的核心创新）

### 1.1 理论准备 (2天)

#### 需要理解的概念
- [ ] Directed Acyclic Graph (DAG)
- [ ] Structural Equation Model (SEM)
- [ ] Variational Inference
- [ ] DiBS算法原理

#### 必读文献
1. **DiBS原论文** (NeurIPS 2021)
   - 标题: "DiBS: Differentiable Bayesian Structure Learning"
   - 链接: https://arxiv.org/abs/2105.11839

2. **因果发现综述**
   - "Causality: Models, Reasoning and Inference" (Pearl, 2009)

#### 学习资源
```bash
# 推荐教程
https://github.com/larslorch/dibs  # 官方repo
https://dibs-project.github.io/     # 文档
```

### 1.2 环境配置 (1天)

#### 安装DiBS库
```bash
# 方法1: 从GitHub安装（推荐）
conda activate fairness
git clone https://github.com/larslorch/dibs.git
cd dibs
pip install -e .

# 方法2: 通过pip（如果可用）
pip install dibs-causal

# 依赖检查
python -c "import dibs; print('DiBS版本:', dibs.__version__)"
```

#### 可能的问题
- **JAX依赖**: DiBS基于JAX，需要正确配置
- **GPU支持**: JAX需要特定的CUDA版本
- **内存需求**: DiBS需要大量内存（建议16GB+）

#### 解决方案
```bash
# 安装JAX (GPU版本)
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 如果内存不足，考虑使用CPU版本
pip install --upgrade jax jaxlib
```

### 1.3 DiBS适配层开发 (4天)

#### 创建新模块: `utils/causal_discovery.py`

```python
"""
DiBS因果图学习模块
实现论文中的因果图发现算法
"""

import numpy as np
import pandas as pd
from dibs import JointDiBS
from typing import Dict, Tuple
import warnings

class CausalGraphLearner:
    """
    使用DiBS学习因果图

    参数:
        n_vars: 变量数量（论文中为46个）
        alpha: DAG惩罚参数（论文中为0.9）
        n_steps: 迭代次数（论文中为10000）
        random_seed: 随机种子
    """

    def __init__(self,
                 n_vars: int = 46,
                 alpha: float = 0.9,
                 n_steps: int = 10000,
                 random_seed: int = 42):

        # 输入验证
        if n_vars <= 0:
            raise ValueError(f"n_vars must be positive, got {n_vars}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        self.n_vars = n_vars
        self.alpha = alpha
        self.n_steps = n_steps
        self.random_seed = random_seed

        # 初始化DiBS模型
        self.model = None
        self.learned_graph = None

    def fit(self, data: pd.DataFrame, verbose: bool = True) -> np.ndarray:
        """
        学习因果图

        参数:
            data: 训练数据，shape (n_samples, n_vars)
            verbose: 是否输出进度

        返回:
            learned_graph: 邻接矩阵，shape (n_vars, n_vars)
                          learned_graph[i,j] = 1 表示 i → j
        """
        # 输入验证
        if data is None or len(data) == 0:
            raise ValueError("data cannot be None or empty")
        if data.shape[1] != self.n_vars:
            raise ValueError(
                f"Expected {self.n_vars} variables, got {data.shape[1]}"
            )

        # 数据预处理：离散变量→连续变量
        data_continuous = self._discretize_to_continuous(data)

        if verbose:
            print(f"开始DiBS学习...")
            print(f"  变量数: {self.n_vars}")
            print(f"  样本数: {len(data)}")
            print(f"  迭代次数: {self.n_steps}")
            print(f"  Alpha: {self.alpha}")

        # 初始化DiBS
        self.model = JointDiBS(
            n_vars=self.n_vars,
            alpha=self.alpha,
            random_state=self.random_seed
        )

        # 运行DiBS（这是耗时操作）
        try:
            self.model.fit(
                data_continuous,
                n_steps=self.n_steps,
                verbose=verbose
            )
        except Exception as e:
            raise RuntimeError(f"DiBS fitting failed: {e}")

        # 获取学习到的图
        self.learned_graph = self.model.get_graph(threshold=0.5)

        if verbose:
            n_edges = np.sum(self.learned_graph > 0)
            print(f"✓ DiBS完成")
            print(f"  学到的边数: {n_edges}")
            print(f"  图密度: {n_edges / (self.n_vars * (self.n_vars-1)):.3f}")

        return self.learned_graph

    def _discretize_to_continuous(self, data: pd.DataFrame) -> np.ndarray:
        """
        将离散变量转换为连续变量

        论文未详细说明此步骤，这里使用两种方法:
        1. 对于已经连续的变量：保持不变
        2. 对于离散变量：添加小随机噪声

        参数:
            data: 原始数据

        返回:
            data_continuous: 连续化后的数据
        """
        data_continuous = data.values.copy().astype(float)

        # 检测离散列（唯一值数量 < 10）
        for i in range(data.shape[1]):
            n_unique = len(np.unique(data.iloc[:, i]))
            if n_unique < 10:
                # 离散列：添加小噪声
                noise = np.random.normal(0, 0.01, size=len(data))
                data_continuous[:, i] += noise

        return data_continuous

    def get_edges(self, threshold: float = 0.5) -> list:
        """
        获取因果边列表

        参数:
            threshold: 边权重阈值

        返回:
            edges: [(source, target, weight), ...]
        """
        if self.learned_graph is None:
            raise RuntimeError("Must call fit() first")

        edges = []
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                weight = self.learned_graph[i, j]
                if weight > threshold:
                    edges.append((i, j, weight))

        return edges

    def save_graph(self, filepath: str):
        """保存学习到的图"""
        if self.learned_graph is None:
            raise RuntimeError("Must call fit() first")
        np.save(filepath, self.learned_graph)
        print(f"✓ 图已保存到: {filepath}")

    def load_graph(self, filepath: str):
        """加载已保存的图"""
        self.learned_graph = np.load(filepath)
        print(f"✓ 图已从 {filepath} 加载")
```

#### 关键实现要点

1. **离散变量处理**（论文未详细说明）
   ```python
   # 方法1: 添加小噪声（简单）
   noise = np.random.normal(0, 0.01, size=n_samples)

   # 方法2: 核密度估计（更准确）
   from sklearn.neighbors import KernelDensity
   kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
   kde.fit(data.reshape(-1, 1))
   samples = kde.sample(n_samples)
   ```

2. **计算优化**
   - 使用GPU加速（JAX自动处理）
   - 批次处理大数据
   - 结果缓存（避免重复计算）

3. **内存优化**
   - 不保存所有中间结果
   - 定期清理内存
   - 使用低精度浮点数（float32）

### 1.4 DiBS测试 (2天)

#### 单元测试: `tests/test_dibs.py`

```python
import unittest
import numpy as np
import pandas as pd
from utils.causal_discovery import CausalGraphLearner

class TestDiBS(unittest.TestCase):

    def test_initialization(self):
        """测试DiBS初始化"""
        learner = CausalGraphLearner(n_vars=5, n_steps=100)
        self.assertEqual(learner.n_vars, 5)
        self.assertEqual(learner.n_steps, 100)

    def test_simple_chain(self):
        """测试简单链式因果关系: X → Y → Z"""
        # 生成数据
        n_samples = 1000
        X = np.random.randn(n_samples)
        Y = 2*X + np.random.randn(n_samples)*0.1
        Z = 3*Y + np.random.randn(n_samples)*0.1

        data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

        # 学习图
        learner = CausalGraphLearner(n_vars=3, n_steps=1000)
        graph = learner.fit(data, verbose=False)

        # 验证：应该有 X→Y, Y→Z
        self.assertGreater(graph[0, 1], 0.5)  # X → Y
        self.assertGreater(graph[1, 2], 0.5)  # Y → Z
        self.assertLess(graph[0, 2], 0.5)     # X ↛ Z（直接）

    def test_real_data_shape(self):
        """测试真实数据规模"""
        # 模拟46变量的数据
        n_samples = 100
        n_vars = 46
        data = pd.DataFrame(
            np.random.randn(n_samples, n_vars),
            columns=[f'var_{i}' for i in range(n_vars)]
        )

        learner = CausalGraphLearner(n_vars=46, n_steps=500)
        graph = learner.fit(data, verbose=False)

        # 验证图的形状
        self.assertEqual(graph.shape, (46, 46))

    def test_save_load(self):
        """测试保存和加载"""
        # 创建简单图
        data = pd.DataFrame(np.random.randn(100, 3))
        learner = CausalGraphLearner(n_vars=3, n_steps=100)
        graph1 = learner.fit(data, verbose=False)

        # 保存
        learner.save_graph('/tmp/test_graph.npy')

        # 加载
        learner2 = CausalGraphLearner(n_vars=3)
        learner2.load_graph('/tmp/test_graph.npy')

        # 验证一致性
        np.testing.assert_array_equal(graph1, learner2.learned_graph)
```

#### 性能测试

```python
def test_performance():
    """测试DiBS性能"""
    import time

    # 测试不同规模
    sizes = [10, 20, 30, 46]
    for n_vars in sizes:
        data = pd.DataFrame(np.random.randn(200, n_vars))
        learner = CausalGraphLearner(n_vars=n_vars, n_steps=1000)

        start = time.time()
        learner.fit(data, verbose=False)
        elapsed = time.time() - start

        print(f"n_vars={n_vars}: {elapsed:.1f}秒")
```

### 1.5 集成到主流程 (3天)

#### 修改 `demo_quick_run.py`

```python
# 在步骤3替换相关性分析

print("\n" + "▶"*35)
print("步骤3: DiBS因果图学习")
print("▶"*35)

# 准备数据（选择数值列）
numeric_cols = df.select_dtypes(include=[np.number]).columns
data_for_dibs = df[numeric_cols]

print(f"\n数据准备:")
print(f"  变量数: {len(numeric_cols)}")
print(f"  样本数: {len(data_for_dibs)}")

# 选择迭代次数（根据数据规模）
n_steps = 10000 if len(data_for_dibs) > 50 else 2000
print(f"  迭代次数: {n_steps}")

# DiBS学习
from utils.causal_discovery import CausalGraphLearner

learner = CausalGraphLearner(
    n_vars=len(numeric_cols),
    alpha=config.CAUSAL_DISCOVERY['alpha'] if hasattr(config, 'CAUSAL_DISCOVERY') else 0.9,
    n_steps=n_steps
)

try:
    causal_graph = learner.fit(data_for_dibs, verbose=True)

    # 保存图
    learner.save_graph('results/causal_graph.npy')

    # 分析图结构
    edges = learner.get_edges(threshold=0.5)
    print(f"\n学到的因果关系 (top 10):")

    # 按权重排序
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
    for i, (source, target, weight) in enumerate(edges_sorted[:10], 1):
        source_name = numeric_cols[source]
        target_name = numeric_cols[target]
        print(f"  {i}. {source_name} → {target_name} (权重: {weight:.3f})")

    # 可视化（可选）
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        for source, target, weight in edges_sorted[:20]:  # 只画前20条边
            G.add_edge(numeric_cols[source], numeric_cols[target], weight=weight)

        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8, arrows=True)
        plt.savefig('results/causal_graph.png', dpi=300, bbox_inches='tight')
        print("\n✓ 因果图已保存到: results/causal_graph.png")
    except ImportError:
        print("\n⚠️  matplotlib未安装，跳过可视化")

except Exception as e:
    print(f"\n❌ DiBS学习失败: {e}")
    print("回退到相关性分析...")

    # 降级处理
    corr_matrix = data_for_dibs.corr()
    causal_graph = (corr_matrix.abs() > 0.3).values.astype(float)
```

### 1.6 验收标准

DiBS实现完成需满足：

- [ ] 可运行在46变量的数据上
- [ ] 10000次迭代可在合理时间内完成（<2小时）
- [ ] 输出DAG结构（无环）
- [ ] 通过所有单元测试
- [ ] 与论文方法一致（输入输出格式）

---

## 🧮 阶段2: DML因果推断

**目标**: 使用Double Machine Learning估计因果效应
**时间**: 1周
**优先级**: 🔴 最高

### 2.1 理论准备 (1天)

#### 核心概念
- [ ] Average Treatment Effect (ATE)
- [ ] Confounding和去混淆
- [ ] Double Machine Learning原理
- [ ] 交叉拟合(Cross-fitting)

#### 必读文献
1. **DML原论文**
   - "Double/Debiased Machine Learning" (Chernozhukov et al., 2018)

2. **EconML文档**
   - https://econml.azurewebsites.net/

### 2.2 环境配置 (0.5天)

```bash
conda activate fairness

# EconML已安装，验证版本
python -c "import econml; print('EconML版本:', econml.__version__)"

# 如需升级
pip install --upgrade econml
```

### 2.3 DML适配层开发 (2天)

#### 创建新模块: `utils/causal_inference.py`

```python
"""
DML因果推断模块
使用Double Machine Learning估计因果效应
"""

import numpy as np
import pandas as pd
from econml.dml import LinearDML, CausalForestDML
from typing import Dict, List, Tuple, Optional
import warnings

class CausalInferenceEngine:
    """
    因果推断引擎

    使用DML估计从因果图中识别的每条边的ATE
    """

    def __init__(self,
                 model_type: str = 'linear',
                 n_folds: int = 2,
                 random_state: int = 42):
        """
        参数:
            model_type: 'linear' 或 'forest'
            n_folds: 交叉拟合的折数
            random_state: 随机种子
        """
        self.model_type = model_type
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = {}

    def estimate_ate_for_edge(self,
                               data: pd.DataFrame,
                               treatment_var: str,
                               outcome_var: str,
                               confounders: List[str],
                               controls: Optional[List[str]] = None) -> Dict:
        """
        估计单条因果边的ATE

        参数:
            data: 完整数据
            treatment_var: 处理变量(源节点)
            outcome_var: 结果变量(目标节点)
            confounders: 混淆因素列表
            controls: 控制变量列表

        返回:
            result: {
                'ate': float,
                'ci_lower': float,
                'ci_upper': float,
                'p_value': float
            }
        """
        # 准备数据
        T = data[treatment_var].values
        Y = data[outcome_var].values
        X = data[confounders].values if confounders else None
        W = data[controls].values if controls else None

        # 初始化DML模型
        if self.model_type == 'linear':
            dml = LinearDML(
                model_y='auto',
                model_t='auto',
                cv=self.n_folds,
                random_state=self.random_state
            )
        else:
            dml = CausalForestDML(
                model_y='auto',
                model_t='auto',
                cv=self.n_folds,
                random_state=self.random_state
            )

        # 拟合
        dml.fit(Y, T, X=X, W=W)

        # 估计ATE
        ate = dml.ate()
        ate_interval = dml.ate_interval()

        # 计算p值（如果可用）
        try:
            ate_inference = dml.ate_inference()
            p_value = ate_inference.pvalue()[0]
        except:
            p_value = None

        result = {
            'ate': float(ate),
            'ci_lower': float(ate_interval[0]),
            'ci_upper': float(ate_interval[1]),
            'p_value': p_value,
            'significant': not (ate_interval[0] <= 0 <= ate_interval[1])
        }

        return result

    def analyze_causal_graph(self,
                            data: pd.DataFrame,
                            causal_graph: np.ndarray,
                            var_names: List[str],
                            min_edge_weight: float = 0.5) -> pd.DataFrame:
        """
        对因果图中的所有边进行因果推断

        参数:
            data: 完整数据
            causal_graph: DiBS学习的图，shape (n_vars, n_vars)
            var_names: 变量名列表
            min_edge_weight: 最小边权重阈值

        返回:
            results_df: DataFrame包含所有边的ATE估计
        """
        results = []
        n_vars = len(var_names)

        # 遍历所有边
        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if causal_graph[i, j] > min_edge_weight:
                    edges.append((i, j, causal_graph[i, j]))

        print(f"开始分析 {len(edges)} 条因果边...")

        from tqdm import tqdm
        for source_idx, target_idx, weight in tqdm(edges):
            source_var = var_names[source_idx]
            target_var = var_names[target_idx]

            # 识别混淆因素
            confounders = self._identify_confounders(
                causal_graph, source_idx, target_idx, var_names
            )

            try:
                # 估计ATE
                result = self.estimate_ate_for_edge(
                    data, source_var, target_var, confounders
                )

                results.append({
                    'source': source_var,
                    'target': target_var,
                    'edge_weight': weight,
                    'ate': result['ate'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'p_value': result['p_value'],
                    'significant': result['significant'],
                    'n_confounders': len(confounders)
                })
            except Exception as e:
                warnings.warn(f"Failed to estimate ATE for {source_var}→{target_var}: {e}")

        results_df = pd.DataFrame(results)
        return results_df

    def _identify_confounders(self,
                             graph: np.ndarray,
                             source_idx: int,
                             target_idx: int,
                             var_names: List[str]) -> List[str]:
        """
        根据因果图识别混淆因素

        混淆因素定义：同时指向source和target的变量
        """
        confounders = []
        n_vars = len(var_names)

        for k in range(n_vars):
            if k == source_idx or k == target_idx:
                continue

            # 检查是否指向source和target
            points_to_source = graph[k, source_idx] > 0.5
            points_to_target = graph[k, target_idx] > 0.5

            if points_to_source or points_to_target:
                confounders.append(var_names[k])

        return confounders
```

### 2.4 实现论文的算法1 (2天)

#### 创建: `utils/tradeoff_detection.py`

```python
"""
实现论文算法1: 权衡检测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.metrics import define_sign_functions

def detect_tradeoffs_algorithm1(
    causal_inference_results: pd.DataFrame,
    sign_functions: Dict,
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    论文算法1: 基于因果推断的权衡检测

    输入:
        causal_inference_results: DML分析结果
            必须包含列: source, target, ate, ci_lower, ci_upper, significant
        sign_functions: 各指标的sign函数字典
        significance_level: 显著性水平

    输出:
        tradeoffs_df: 检测到的权衡关系
    """
    tradeoffs = []

    # 按source分组
    grouped = causal_inference_results.groupby('source')

    for source, group in grouped:
        edges = group.to_dict('records')

        # 只考虑显著的边
        significant_edges = [e for e in edges if e['significant']]

        # 检查所有边对
        for i, edge1 in enumerate(significant_edges):
            for edge2 in significant_edges[i+1:]:
                target1 = edge1['target']
                target2 = edge2['target']

                # 获取ATE
                ate1 = edge1['ate']
                ate2 = edge2['ate']

                # 计算sign（简化：假设当前值为0）
                try:
                    sign1 = sign_functions[target1](0, ate1)
                    sign2 = sign_functions[target2](0, ate2)
                except KeyError:
                    continue  # 跳过没有sign函数的指标

                # 检查权衡：sign相反
                if sign1 != sign2:
                    tradeoffs.append({
                        'intervention': source,
                        'metric1': target1,
                        'metric2': target2,
                        'ate1': ate1,
                        'ate2': ate2,
                        'ci1_lower': edge1['ci_lower'],
                        'ci1_upper': edge1['ci_upper'],
                        'ci2_lower': edge2['ci_lower'],
                        'ci2_upper': edge2['ci_upper'],
                        'sign1': sign1,
                        'sign2': sign2,
                        'tradeoff_type': f"{target1}↑ vs {target2}↓" if sign1 == '+' else f"{target1}↓ vs {target2}↑"
                    })

    tradeoffs_df = pd.DataFrame(tradeoffs)
    return tradeoffs_df
```

### 2.5 集成测试 (1天)

```python
# tests/test_causal_inference.py

class TestCausalInference(unittest.TestCase):

    def test_dml_simple_case(self):
        """测试简单情况下的DML"""
        # 生成数据: T → Y, with confounder C
        n = 1000
        C = np.random.randn(n)
        T = 2*C + np.random.randn(n)
        Y = 3*T + 4*C + np.random.randn(n)

        data = pd.DataFrame({'C': C, 'T': T, 'Y': Y})

        engine = CausalInferenceEngine()
        result = engine.estimate_ate_for_edge(
            data, 'T', 'Y', confounders=['C']
        )

        # 真实ATE = 3
        self.assertAlmostEqual(result['ate'], 3, delta=0.5)
        self.assertTrue(result['significant'])

    def test_algorithm1(self):
        """测试算法1权衡检测"""
        # 模拟DML结果
        results = pd.DataFrame({
            'source': ['alpha', 'alpha'],
            'target': ['Acc', 'SPD'],
            'ate': [0.1, -0.05],
            'ci_lower': [0.05, -0.08],
            'ci_upper': [0.15, -0.02],
            'significant': [True, True]
        })

        sign_funcs = define_sign_functions()
        tradeoffs = detect_tradeoffs_algorithm1(results, sign_funcs)

        # 应该检测到1个权衡
        self.assertEqual(len(tradeoffs), 1)
```

### 2.6 验收标准

- [ ] DML可运行在真实数据上
- [ ] 输出包含ATE和置信区间
- [ ] 统计显著性检验有效
- [ ] 算法1正确检测权衡
- [ ] 通过所有单元测试

---

## 🎨 阶段3: 完整公平性方法

**目标**: 实现论文中的12个公平性方法
**时间**: 1周
**优先级**: 🟡 高

### 3.1 方法清单

#### 预处理方法 (Preprocessing)
1. ✅ Reweighing - 已实现
2. ❌ Learning Fair Representations (LFR)
3. ❌ Optimized Preprocessing

#### 处理中方法 (In-processing)
4. ⚠️ AdversarialDebiasing - 需完整实现
5. ❌ Prejudice Remover
6. ❌ Meta Fair Classifier

#### 后处理方法 (Post-processing)
7. ⚠️ Equalized Odds - 需完整实现
8. ❌ Calibrated Equalized Odds
9. ❌ Reject Option Classification

#### 其他方法
10. ❌ Exponentiated Gradient
11. ❌ Grid Search Reduction
12. ✅ Baseline - 已实现

### 3.2 实施策略

#### 优先级排序
```
Week 4.1-4.2: AdversarialDebiasing (关键，需TensorFlow)
Week 4.3:     Equalized Odds (简单)
Week 4.4:     LFR + Prejudice Remover
Week 4.5:     其余方法
```

### 3.3 AdversarialDebiasing完整实现 (2天)

#### 安装TensorFlow

```bash
conda activate fairness

# 安装TensorFlow (CPU版本)
pip install tensorflow==2.12.0

# 验证
python -c "import tensorflow as tf; print('TF版本:', tf.__version__)"

# 重新安装AIF360的TensorFlow支持
pip install 'aif360[AdversarialDebiasing]'
```

#### 修改 `utils/fairness_methods.py`

```python
def _apply_adversarial_debiasing(self, dataset):
    """
    完整实现AdversarialDebiasing

    使用对抗训练去除偏差
    """
    try:
        from aif360.algorithms.inprocessing import AdversarialDebiasing
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

        # 创建TensorFlow session
        sess = tf.Session()

        # 初始化方法
        debiaser = AdversarialDebiasing(
            privileged_groups=[{self.sensitive_attr: 1}],
            unprivileged_groups=[{self.sensitive_attr: 0}],
            scope_name='debiased_classifier',
            debias=True,
            sess=sess,
            num_epochs=50,
            batch_size=128
        )

        # 训练
        transformed_dataset = debiaser.fit_transform(dataset)

        sess.close()
        return transformed_dataset

    except ImportError:
        warnings.warn("TensorFlow not available, returning original dataset")
        return dataset
    except Exception as e:
        warnings.warn(f"AdversarialDebiasing failed: {e}")
        return dataset
```

### 3.4 其他方法实现模板

创建 `utils/fairness_methods_extended.py` 包含其余方法。

### 3.5 测试所有方法 (1天)

```python
# tests/test_all_fairness_methods.py

class TestAllFairnessMethods(unittest.TestCase):

    def setUp(self):
        self.methods = [
            'Baseline', 'Reweighing', 'LFR', 'OptimPreproc',
            'AdversarialDebiasing', 'PrejudiceRemover', 'MetaFairClassifier',
            'EqualizedOdds', 'CalibratedEqOdds', 'RejectOptionClassification',
            'ExponentiatedGradient', 'GridSearchReduction'
        ]

        # 生成测试数据
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.sens_train = np.random.randint(0, 2, 100)

    def test_all_methods_run(self):
        """测试所有方法可以运行"""
        for method_name in self.methods:
            with self.subTest(method=method_name):
                try:
                    wrapper = get_fairness_method(method_name, alpha=0.5)
                    X_new, y_new = wrapper.fit_transform(
                        self.X_train, self.y_train, self.sens_train
                    )

                    # 验证输出形状
                    self.assertEqual(X_new.shape[0], self.X_train.shape[0])
                    print(f"✓ {method_name}")
                except Exception as e:
                    self.fail(f"{method_name} failed: {e}")
```

---

## 📊 阶段4: 真实数据集集成

**目标**: 集成Adult/COMPAS/German数据集
**时间**: 3天
**优先级**: 🟡 高

### 4.1 数据下载与预处理 (1天)

#### 创建 `utils/datasets.py`

```python
"""
真实数据集加载器
"""

from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

class DatasetLoader:
    """统一的数据集加载接口"""

    @staticmethod
    def load_adult(protected_attr='sex',
                   test_size=0.3,
                   random_state=42) -> Tuple:
        """
        加载Adult数据集

        返回:
            X_train, y_train, sens_train, X_test, y_test, sens_test
        """
        # 使用AIF360加载
        dataset = AdultDataset(
            protected_attribute_names=[protected_attr],
            privileged_classes=[['Male']] if protected_attr == 'sex' else [['White']],
            categorical_features=['workclass', 'education', 'marital-status',
                                'occupation', 'relationship', 'native-country'],
            features_to_keep=[],
            na_values=[],
            metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                     'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]}
        )

        # 转换为DataFrame
        df, _ = dataset.convert_to_dataframe()

        # 分离特征和标签
        label_col = 'income-per-year'
        y = (df[label_col] == '>50K').astype(int).values

        # 分离敏感属性
        sens = df[protected_attr].values

        # 移除标签和敏感属性，得到特征
        feature_cols = [c for c in df.columns if c not in [label_col, protected_attr]]
        X = df[feature_cols].values

        # 分割训练测试集
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X, y, sens, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"✓ Adult数据集加载完成")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        print(f"  特征维度: {X_train.shape[1]}")

        return X_train, y_train, sens_train, X_test, y_test, sens_test

    @staticmethod
    def load_compas(protected_attr='race',
                    test_size=0.3,
                    random_state=42) -> Tuple:
        """加载COMPAS数据集"""
        dataset = CompasDataset(protected_attribute_names=[protected_attr])
        # 类似处理...
        pass

    @staticmethod
    def load_german(protected_attr='sex',
                   test_size=0.3,
                   random_state=42) -> Tuple:
        """加载German数据集"""
        dataset = GermanDataset(protected_attribute_names=[protected_attr])
        # 类似处理...
        pass
```

### 4.2 修改主程序支持数据集切换 (1天)

#### 更新 `config.py`

```python
# 数据集配置
DATASETS = {
    'adult': {
        'name': 'Adult',
        'protected_attrs': ['sex', 'race'],
        'loader': 'load_adult'
    },
    'compas': {
        'name': 'COMPAS',
        'protected_attrs': ['race', 'sex'],
        'loader': 'load_compas'
    },
    'german': {
        'name': 'German',
        'protected_attrs': ['sex', 'age'],
        'loader': 'load_german'
    },
    'synthetic': {
        'name': 'Synthetic',
        'protected_attrs': ['sex'],
        'loader': None  # 使用现有的模拟数据生成
    }
}

# 当前使用的数据集
CURRENT_DATASET = 'adult'  # 改为 'adult', 'compas', 'german', 或 'synthetic'
CURRENT_PROTECTED_ATTR = 'sex'
```

#### 创建统一的数据加载接口

```python
# utils/data_loader.py

def load_dataset(dataset_name: str, protected_attr: str):
    """
    统一的数据加载接口

    参数:
        dataset_name: 'adult', 'compas', 'german', 或 'synthetic'
        protected_attr: 敏感属性名

    返回:
        X_train, y_train, sens_train, X_test, y_test, sens_test
    """
    from utils.datasets import DatasetLoader

    if dataset_name == 'adult':
        return DatasetLoader.load_adult(protected_attr)
    elif dataset_name == 'compas':
        return DatasetLoader.load_compas(protected_attr)
    elif dataset_name == 'german':
        return DatasetLoader.load_german(protected_attr)
    elif dataset_name == 'synthetic':
        # 使用现有的模拟数据生成
        from demo_quick_run import generate_synthetic_data
        return generate_synthetic_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
```

### 4.3 测试所有数据集 (1天)

```python
# tests/test_datasets.py

class TestDatasets(unittest.TestCase):

    def test_all_datasets_load(self):
        """测试所有数据集可以加载"""
        datasets = ['adult', 'compas', 'german', 'synthetic']

        for dataset_name in datasets:
            with self.subTest(dataset=dataset_name):
                X_train, y_train, sens_train, X_test, y_test, sens_test = \
                    load_dataset(dataset_name, 'sex')

                # 验证形状
                self.assertGreater(len(X_train), 0)
                self.assertGreater(len(X_test), 0)
                self.assertEqual(len(X_train), len(y_train))
                self.assertEqual(len(X_train), len(sens_train))

                print(f"✓ {dataset_name}: {len(X_train)} 训练样本")
```

---

## 🚀 阶段5: 大规模实验

**目标**: 运行论文中的完整实验
**时间**: 2周
**优先级**: 🟢 中（规模扩展）

### 5.1 实验配置 (1天)

#### 更新 `config.py` 为完整配置

```python
# 完整的公平性方法列表（12个）
FAIRNESS_METHODS = [
    'Baseline',
    'Reweighing',
    'LFR',
    'OptimPreproc',
    'AdversarialDebiasing',
    'PrejudiceRemover',
    'MetaFairClassifier',
    'EqualizedOdds',
    'CalibratedEqOdds',
    'RejectOptionClassification',
    'ExponentiatedGradient',
    'GridSearchReduction'
]

# Alpha采样点（10个）
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 实验配置
EXPERIMENTS = {
    'full_replication': {
        'datasets': ['adult', 'compas', 'german'],
        'protected_attrs': {
            'adult': ['sex', 'race'],
            'compas': ['race', 'sex'],
            'german': ['sex', 'age']
        },
        'methods': FAIRNESS_METHODS,
        'alphas': ALPHA_VALUES,
        'n_runs': 5  # 每个配置运行5次取平均
    },
    'quick_test': {
        'datasets': ['adult'],
        'protected_attrs': {'adult': ['sex']},
        'methods': ['Baseline', 'Reweighing'],
        'alphas': [0.0, 0.5, 1.0],
        'n_runs': 1
    }
}

CURRENT_EXPERIMENT = 'full_replication'  # 或 'quick_test'
```

### 5.2 并行化实验运行 (2天)

#### 创建 `experiments/run_full_experiment.py`

```python
"""
运行完整实验
支持多进程并行和断点续传
"""

import multiprocessing as mp
from itertools import product
import pandas as pd
import os
from tqdm import tqdm
import config

def run_single_experiment(args):
    """
    运行单个实验配置

    参数:
        args: (dataset, protected_attr, method, alpha, run_id)
    """
    dataset, protected_attr, method, alpha, run_id = args

    try:
        # 加载数据
        X_train, y_train, sens_train, X_test, y_test, sens_test = \
            load_dataset(dataset, protected_attr)

        # 应用公平性方法
        wrapper = get_fairness_method(method, alpha, protected_attr)
        X_transformed, y_transformed = wrapper.fit_transform(
            X_train, y_train, sens_train
        )

        # 训练模型
        trainer = ModelTrainer(input_dim=X_transformed.shape[1])
        trainer.train(X_transformed, y_transformed, epochs=20, verbose=False)

        # 计算指标
        calc = MetricsCalculator(trainer)
        metrics = calc.compute_all_metrics(X_test, y_test, sens_test, phase='Te')

        # 添加元数据
        result = {
            'dataset': dataset,
            'protected_attr': protected_attr,
            'method': method,
            'alpha': alpha,
            'run_id': run_id,
            **metrics
        }

        return result

    except Exception as e:
        print(f"Error in {dataset}/{protected_attr}/{method}/{alpha}: {e}")
        return None

def main():
    """主实验函数"""

    # 读取实验配置
    exp_config = config.EXPERIMENTS[config.CURRENT_EXPERIMENT]

    # 生成所有实验组合
    all_configs = []
    for dataset in exp_config['datasets']:
        for protected_attr in exp_config['protected_attrs'][dataset]:
            for method in exp_config['methods']:
                for alpha in exp_config['alphas']:
                    for run_id in range(exp_config['n_runs']):
                        all_configs.append((dataset, protected_attr, method, alpha, run_id))

    total_experiments = len(all_configs)
    print(f"总实验数: {total_experiments}")
    print(f"  数据集: {len(exp_config['datasets'])}")
    print(f"  敏感属性: {sum(len(v) for v in exp_config['protected_attrs'].values())}")
    print(f"  方法: {len(exp_config['methods'])}")
    print(f"  Alpha点: {len(exp_config['alphas'])}")
    print(f"  运行次数: {exp_config['n_runs']}")

    # 检查是否有已完成的实验（断点续传）
    output_file = 'results/full_experiment_results.csv'
    if os.path.exists(output_file):
        existing_results = pd.read_csv(output_file)
        completed_configs = set(
            zip(existing_results['dataset'],
                existing_results['protected_attr'],
                existing_results['method'],
                existing_results['alpha'],
                existing_results['run_id'])
        )
        all_configs = [c for c in all_configs if c not in completed_configs]
        print(f"\n找到 {len(completed_configs)} 个已完成实验")
        print(f"剩余 {len(all_configs)} 个实验")

    # 并行运行
    n_processes = mp.cpu_count() - 1  # 留一个核心给系统
    print(f"\n使用 {n_processes} 个进程并行运行...")

    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(run_single_experiment, all_configs),
            total=len(all_configs),
            desc="运行实验"
        ))

    # 过滤失败的实验
    results = [r for r in results if r is not None]

    # 保存结果
    results_df = pd.DataFrame(results)

    if os.path.exists(output_file):
        # 追加到现有文件
        existing = pd.read_csv(output_file)
        results_df = pd.concat([existing, results_df], ignore_index=True)

    results_df.to_csv(output_file, index=False)
    print(f"\n✓ 结果已保存到: {output_file}")
    print(f"  总数据点: {len(results_df)}")

if __name__ == '__main__':
    main()
```

### 5.3 监控和日志 (1天)

#### 实时进度监控

```python
# experiments/monitor.py

import pandas as pd
import time

def monitor_progress():
    """监控实验进度"""
    output_file = 'results/full_experiment_results.csv'

    while True:
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)

            print(f"\r进度: {len(df)}/726 ({len(df)/726*100:.1f}%)", end='')

            if len(df) >= 726:
                print("\n✓ 所有实验完成！")
                break

        time.sleep(10)

if __name__ == '__main__':
    monitor_progress()
```

### 5.4 结果汇总分析 (2天)

#### 创建 `experiments/analyze_results.py`

```python
"""
分析实验结果
"""

import pandas as pd
import numpy as np

def analyze_full_results():
    """分析完整实验结果"""

    # 加载数据
    results = pd.read_csv('results/full_experiment_results.csv')

    print("=" * 80)
    print("实验结果分析")
    print("=" * 80)

    # 1. 基本统计
    print("\n1. 数据概览")
    print(f"   总数据点: {len(results)}")
    print(f"   数据集: {results['dataset'].nunique()}")
    print(f"   方法: {results['method'].nunique()}")
    print(f"   Alpha点: {results['alpha'].nunique()}")

    # 2. 按方法汇总
    print("\n2. 各方法性能对比")
    method_summary = results.groupby('method').agg({
        'Te_Acc': ['mean', 'std'],
        'Te_SPD': ['mean', 'std'],
        'Te_DI': ['mean', 'std']
    })
    print(method_summary)

    # 3. 权衡分析
    print("\n3. 检测到的权衡")
    # 应用算法1...

    # 4. 与论文对比
    print("\n4. 与论文结果对比")
    # 加载论文数据并对比...

    # 生成报告
    generate_latex_tables(results)
    generate_plots(results)

if __name__ == '__main__':
    analyze_full_results()
```

---

## 🔧 阶段6: 结果验证

**目标**: 验证复现结果与论文一致
**时间**: 1周
**优先级**: ✅ 验证

### 6.1 提取论文数据 (2天)

从论文的图表中提取数据点进行对比：

```python
# experiments/paper_data.py

"""
论文中的参考数据
从图表中手动提取或联系作者获取
"""

PAPER_RESULTS = {
    'adult_sex_reweighing': {
        'alpha': [0.0, 0.1, 0.2, ..., 1.0],
        'Te_Acc': [0.85, 0.84, 0.83, ..., 0.80],
        'Te_SPD': [0.10, 0.08, 0.06, ..., 0.02],
        # ... 其他指标
    },
    # ... 其他配置
}
```

### 6.2 统计对比 (2天)

```python
# experiments/compare_with_paper.py

def compare_with_paper():
    """与论文结果对比"""

    our_results = pd.read_csv('results/full_experiment_results.csv')

    comparisons = []

    for config_name, paper_data in PAPER_RESULTS.items():
        # 提取对应的我们的结果
        our_data = extract_matching_results(our_results, config_name)

        # 计算指标差异
        acc_diff = np.mean(np.abs(our_data['Te_Acc'] - paper_data['Te_Acc']))
        spd_diff = np.mean(np.abs(our_data['Te_SPD'] - paper_data['Te_SPD']))

        comparisons.append({
            'config': config_name,
            'acc_mae': acc_diff,
            'spd_mae': spd_diff,
            'match_quality': 'Good' if acc_diff < 0.02 else 'Fair' if acc_diff < 0.05 else 'Poor'
        })

    comparison_df = pd.DataFrame(comparisons)
    print(comparison_df)

    # 生成对比图表
    plot_comparison(our_data, paper_data)
```

### 6.3 差异分析 (2天)

如果发现差异，分析原因：

1. **随机性**: 多次运行取平均
2. **超参数**: 调整匹配论文
3. **数据预处理**: 检查是否一致
4. **实现细节**: 对照论文代码

### 6.4 生成复现报告 (1天)

```python
# experiments/generate_replication_report.py

def generate_report():
    """生成复现报告"""

    report = f"""
# 论文复现报告

## 实验配置
- 数据集: {config.DATASETS}
- 方法: {config.FAIRNESS_METHODS}
- Alpha点: {config.ALPHA_VALUES}
- 总实验数: {total_experiments}

## 复现结果
### 数据规模
- 论文: 726数据点
- 我们: {len(our_results)}数据点
- 匹配度: {len(our_results)/726*100:.1f}%

### 结果对比
{comparison_table}

### 关键发现
1. ...
2. ...

## 结论
复现度: {replication_score}/100

## 差异说明
...
"""

    with open('results/REPLICATION_REPORT.md', 'w') as f:
        f.write(report)
```

---

## 🛠️ 技术难点与解决方案

### 难点1: DiBS计算开销大

**问题**: 10000次迭代可能需要数小时

**解决方案**:
```python
# 1. GPU加速（JAX自动处理）
# 2. 减少迭代次数（快速测试用）
n_steps = 2000 if args.quick_test else 10000

# 3. 结果缓存
if os.path.exists('cache/causal_graph.npy'):
    causal_graph = np.load('cache/causal_graph.npy')
else:
    causal_graph = learner.fit(data)
    np.save('cache/causal_graph.npy', causal_graph)

# 4. 分布式计算（如果有多台机器）
from dask.distributed import Client
client = Client('scheduler-address:8786')
```

### 难点2: DML对混淆因素敏感

**问题**: 混淆因素识别不当导致ATE估计有偏

**解决方案**:
```python
# 1. 使用后门准则识别混淆因素
from dowhy import CausalModel

model = CausalModel(
    data=data,
    treatment='alpha',
    outcome='Te_Acc',
    graph=causal_graph
)

backdoor_sets = model.identify_effect()

# 2. 敏感性分析
for confounders in candidate_confounder_sets:
    ate = estimate_ate(treatment, outcome, confounders)
    print(f"Confounders {confounders}: ATE={ate}")

# 3. 使用因果发现算法（PC, GES）作为DiBS的补充
```

### 难点3: 公平性方法实现不一致

**问题**: AIF360不同版本可能有差异

**解决方案**:
```python
# 1. 固定依赖版本
# requirements.txt
aif360==0.5.0

# 2. 添加版本检查
import aif360
assert aif360.__version__ == '0.5.0', "需要AIF360 0.5.0"

# 3. 自己实现关键方法（如果必要）
```

### 难点4: 实验时间过长

**问题**: 726个实验 × 5次运行 = 3630次，可能需要数天

**解决方案**:
```python
# 1. 多进程并行
n_processes = 32  # 根据CPU核数

# 2. GPU加速模型训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 3. 分批运行（断点续传）
# 4. 使用云计算资源（AWS, GCP）
# 5. 减少训练轮数（从20→10）
```

---

## 💰 资源需求

### 计算资源

| 组件 | CPU | 内存 | GPU | 存储 | 时间 |
|------|-----|------|-----|------|------|
| **DiBS学习** | 8核 | 16GB | 可选 | 1GB | 2-4小时 |
| **DML推断** | 4核 | 8GB | - | 100MB | 30分钟 |
| **单次训练** | 1核 | 2GB | 可选 | 10MB | 1分钟 |
| **完整实验** | 32核 | 64GB | RTX3080 | 10GB | 2-3天 |

**推荐配置**:
- **开发阶段**: 当前机器（RTX 3080）足够
- **完整实验**: 考虑云服务器（AWS p3.2xlarge或类似）

### 云计算成本估算

```bash
# AWS p3.2xlarge (Tesla V100, 8核, 61GB内存)
# 按需价格: $3.06/小时
# 完整实验 (48小时): $147

# 或使用Spot实例 (约70%折扣):
# 完整实验: $45

# 推荐: 使用Spot实例 + 断点续传
```

### 时间投入

```
Week 1-2: DiBS实现        40小时  (全职2周)
Week 3:   DML实现         20小时  (全职1周)
Week 4:   公平性方法      20小时  (全职1周)
Week 5:   数据集集成      15小时  (3天)
Week 6-7: 大规模实验      40小时  (运行为主)
Week 8:   结果验证        20小时  (1周)
----------------------------------------
总计:                     155小时 (~4周全职)
```

---

## 📚 学习资源

### 必读论文
1. DiBS (NeurIPS 2021)
2. DML (Econometrics Journal 2018)
3. AIF360 (IBM Journal 2019)

### 推荐课程
1. Causality Bootcamp (Brady Neal) - YouTube
2. Causal Inference: The Mixtape (Cunningham)
3. Introduction to Causal Inference (Neal)

### 代码参考
1. DiBS官方repo: https://github.com/larslorch/dibs
2. EconML文档: https://econml.azurewebsites.net/
3. 原论文代码: https://anonymous.4open.science/r/CTF-47BF

---

## ✅ 验收标准

完整复现需满足：

### 功能完整性
- [ ] DiBS因果图学习可运行
- [ ] DML因果推断输出ATE+CI
- [ ] 12个公平性方法全部可用
- [ ] 3个真实数据集集成
- [ ] 726个数据点收集完成

### 结果一致性
- [ ] 主要指标与论文误差 < 5%
- [ ] 权衡模式与论文一致
- [ ] 至少复现论文的3个主要结论

### 代码质量
- [ ] 所有测试通过 (100%)
- [ ] 代码有完整文档
- [ ] 可重复运行（设置随机种子）

### 文档完整性
- [ ] 复现报告详细说明差异
- [ ] 使用说明清晰
- [ ] 结果可视化

---

## 🎯 最终目标

**完成后的复现度**: 90-95%

**可发表级别**:
- 足以支持学术论文引用
- 可作为基准进行扩展研究
- 结果可重复验证

**交付物**:
1. 完整可运行代码
2. 726个数据点结果
3. 与论文对比报告
4. 复现文档和使用指南

---

**创建时间**: 2025-12-20
**预计完成时间**: 2026-02-07 (6-8周)
**当前复现度**: 45%
**目标复现度**: 90%+
