# ATE集成到因果分析白名单：完整技术方案

**版本**: v1.0
**状态**: 🟡 有条件通过（需解决P0风险）
**评审日期**: 2026-01-25
**预计工期**: 15-22天（3-4周）

---

## 📋 执行摘要

### 项目目标

将CTF论文（Causality-Aided Trade-off Analysis for Machine Learning Fairness）的ATE（Average Treatment Effect）计算方法集成到深度学习能耗研究的因果分析流程中，扩展白名单数据格式以支持RQ2的trade-off分析。

### 方案状态

- **技术可行性**: 4/5 ⭐⭐⭐⭐
- **方案完整性**: 4/5 ⭐⭐⭐⭐
- **风险可控性**: 3/5 ⭐⭐⭐
- **综合评分**: **3.75/5** - 🟡 有条件通过

### 关键决策

✅ **建议实施**，但必须：
1. 先完成Phase 0准备工作（确认CTF逻辑）
2. 解决所有P0风险
3. 采用分阶段实施策略
4. 建立完整测试体系

---

## 📚 文档导航

### 阶段性文档

| 阶段 | 文档 | 内容 |
|------|------|------|
| 阶段1 | `STAGE1_SOURCE_CODE_VERIFICATION_20260125.md` | CTF源码完整性验证 |
| 阶段2 | `STAGE2_CODE_COMPARISON_20260125.md` | 代码差异对比分析 |
| 阶段3 | `STAGE3_ATE_INTEGRATION_PLAN_20260125.md` | ATE集成方案设计 |
| 阶段4 | `STAGE4_PEER_REVIEW_REPORT_20260125.md` | 同行评审风险评估 |
| **总览** | **`ATE_INTEGRATION_COMPLETE_PLAN_20260125.md`** | **本文档 - 完整方案** |

### 参考文档

- `CTF_SOURCE_CODE_COMPARISON_20260125.md` - 详细代码对比
- `CAUSAL_EDGE_WHITELIST_DESIGN.md` - 白名单设计规范
- `CLAUDE_FULL_REFERENCE.md` - 项目完整参考

---

## 🔍 背景与动机

### 研究问题

**RQ1**: To what extent do hyperparameters affect GPU energy consumption?
**RQ2**: What is the trade-off relationship between energy consumption and performance?

### 为什么需要ATE？

1. **量化因果效应** - ATE提供标准化的因果效应度量
2. **支持trade-off分析** - RQ2需要ATE来检测能耗-性能权衡
3. **对齐CTF方法** - 提升研究可信度和可复现性
4. **扩展白名单** - 当前白名单缺少因果推断信息

### CTF论文简介

**论文**: Causality-Aided Trade-off Analysis for Machine Learning Fairness
**仓库**: https://anonymous.4open.science/r/CTF-47BF
**核心贡献**: 
- 使用因果推断分析fairness trade-offs
- Algorithm 1: 基于因果图的trade-off检测
- 集成DiBS因果发现 + DML因果推断

---

## 📊 当前状况分析

### CTF源码完整性

✅ **已验证** - `CTF_original/` 仓库完整

**核心文件**:
- `src/inf.py` (337行) - ATE计算和trade-off检测
- `src/collect.py` (1418行) - 数据收集
- `src/fairness/in_p.py` (466行) - Fairness方法

**关键函数**:
```python
# CTF_original/src/inf.py:78
def compute_ate(parent, child, data_df, ref_df, dg, T0, T1):
    """使用LinearDML + RandomForest计算ATE"""
```

### 我们的代码现状

**现有实现**:
- `utils/causal_inference.py` - CausalInferenceEngine类（部分实现）
- `utils/tradeoff_detection.py` - TradeoffDetector类（缺少原因寻找）
- `results/energy_research/data/interaction/whitelist/*.csv` - 白名单数据（无ATE列）

**功能完整性**: 约60%

### 五大关键差异

| 功能 | CTF | 我们 | 影响 |
|------|-----|------|------|
| ATE计算 | ref_df + T0/T1 | 原数据 | ⚠️ 关键 |
| 混淆因素 | 自动识别 | 手动传入 | ⚠️ 关键 |
| 原因寻找 | 深度分析 | 不支持 | ⚠️ 关键 |
| 模型选择 | RandomForest | 'auto' | ⚠️ 中等 |
| DoWhy | 集成 | 无 | ⚠️ 中等 |

---

## 🎯 技术方案

### 方案1: ATE计算函数扩展

#### 设计原则

1. **向后兼容** - 默认保持原有行为
2. **CTF对齐** - 可选开启CTF兼容模式
3. **统一接口** - 避免代码重复
4. **结构化返回** - 易于扩展和验证

#### 接口设计

```python
def estimate_ate(self,
                 data: pd.DataFrame,
                 treatment: str,
                 outcome: str,
                 confounders: Optional[List[str]] = None,
                 controls: Optional[List[str]] = None,
                 ref_df: Optional[pd.DataFrame] = None,
                 T0: Optional[float] = None,
                 T1: Optional[float] = None,
                 mode: str = 'auto',
                 verbose: bool = False) -> Dict[str, Any]:
    """
    估计平均处理效应(ATE)
    
    参数:
        data: 实验数据
        treatment: 处理变量（干预节点）
        outcome: 结果变量（目标节点）
        confounders: 混淆因素列表（可选，mode='ctf'时自动识别）
        controls: 控制变量列表（可选）
        ref_df: 参考数据集（CTF模式）
        T0: 对照值（CTF模式）
        T1: 处理值（CTF模式）
        mode: 计算模式
            - 'auto': 自动选择模型（默认）
            - 'ctf': CTF兼容模式（RandomForest + ref_df）
            - 'hybrid': 混合模式
        verbose: 是否输出详细信息
    
    返回:
        {
            'ate': float,              # 平均处理效应
            'ci_lower': float,         # 置信区间下界
            'ci_upper': float,         # 置信区间上界
            'is_significant': bool,    # 是否统计显著
            'T0': float,               # 对照值
            'T1': float,               # 处理值
            'ref_mean': float,         # 参考均值
            'method': str,             # 计算方法
            'confounders': List[str],  # 使用的混淆因素
            'n_samples': int           # 样本数
        }
    """
```

#### 实现结构

```python
class CausalInferenceEngine:
    """因果推断引擎"""
    
    def estimate_ate(self, ...):
        """统一ATE计算接口"""
        # 1. 准备数据
        X, T, Y, W = self._prepare_data(...)
        
        # 2. 识别混淆因素（如果需要）
        if confounders is None and mode == 'ctf':
            confounders, controls = self._identify_confounders_from_graph(...)
        
        # 3. 构建模型
        if mode == 'ctf':
            model = self._build_ctf_model()
        else:
            model = self._build_auto_model()
        
        # 4. 拟合模型
        model.fit(Y, T, X=X, W=W)
        
        # 5. 计算ATE
        if T0 is not None and T1 is not None:
            ate = model.ate(X=X_eval, T0=T0, T1=T1)
        else:
            ate = model.ate(X=X_eval)
        
        # 6. 计算置信区间
        ci = self._compute_confidence_interval(model, X_eval)
        
        return self._format_result(ate, ci, ...)
    
    def _prepare_data(self, ...):
        """公共数据准备逻辑"""
        # 提取为私有方法，避免重复
        
    def _identify_confounders_from_graph(self, treatment, outcome, causal_graph):
        """从因果图识别混淆因素"""
        # 实现CTF的逻辑
        
    def _build_ctf_model(self):
        """构建CTF兼容模型"""
        from sklearn.ensemble import RandomForestRegressor
        return LinearDML(
            model_y=RandomForestRegressor(),
            model_t=RandomForestRegressor(),
            random_state=0
        )
    
    def _build_auto_model(self):
        """构建自动模型"""
        return LinearDML(
            model_y='auto',
            model_t='auto',
            random_state=42
        )
```

### 方案2: 白名单格式扩展

#### 新增列定义

| 列名 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| ate | float | NaN | 平均处理效应 |
| ate_ci_lower | float | NaN | 95%置信区间下界 |
| ate_ci_upper | float | NaN | 95%置信区间上界 |
| ate_is_significant | bool | False | 是否统计显著 |
| T0 | float | NaN | 对照值 |
| T1 | float | NaN | 处理值 |
| ref_mean | float | NaN | 参考均值 |
| ate_method | str | 'N/A' | 计算方法标识 |

#### CSV格式示例

```csv
# whitelist_format_version: 2.0
source,target,strength,edge_type,is_significant,strength_level,source_category,target_category,question_relevance,interpretation,ate,ate_ci_lower,ate_ci_upper,ate_is_significant,T0,T1,ref_mean,ate_method
hyperparam_batch_size,energy_gpu_min_watts,0.95,moderation,yes,very_strong,hyperparam,energy,other,并行模式调节batch_size对gpu_min_watts的效应,0.123,0.089,0.157,true,0.0,1.0,0.5,DML_CTF
```

#### 迁移脚本

**文件**: `tools/data_management/add_ate_to_whitelist.py`

```python
def add_ate_to_whitelist(whitelist_path: str,
                        data: pd.DataFrame,
                        causal_graph: nx.DiGraph,
                        mode: str = 'ctf') -> pd.DataFrame:
    """
    为白名单CSV添加ATE列
    
    参数:
        whitelist_path: 白名单CSV路径
        data: 原始数据
        causal_graph: 因果图
        mode: ATE计算模式
    
    返回:
        添加了ATE列的DataFrame
    """
    # 1. 读取白名单
    df = pd.read_csv(whitelist_path)
    
    # 2. 初始化ATE列
    ate_columns = ['ate', 'ate_ci_lower', 'ate_ci_upper', 
                   'ate_is_significant', 'T0', 'T1', 'ref_mean', 'ate_method']
    for col in ate_columns:
        if col not in df.columns:
            df[col] = np.nan if col != 'ate_method' else 'N/A'
            if col == 'ate_is_significant':
                df[col] = False
    
    # 3. 创建ATE计算引擎
    engine = CausalInferenceEngine(verbose=True)
    
    # 4. 为每条边计算ATE
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="计算ATE"):
        source = row['source']
        target = row['target']
        
        try:
            # 构建ref_df（CTF模式）
            if mode == 'ctf':
                ref_df = build_reference_df(data, [source])
                T0 = data[source].min()
                T1 = data[source].max()
            else:
                ref_df, T0, T1 = None, None, None
            
            # 计算ATE
            result = engine.estimate_ate(
                data=data,
                treatment=source,
                outcome=target,
                confounders=None,  # 自动识别
                ref_df=ref_df,
                T0=T0,
                T1=T1,
                mode=mode
            )
            
            # 填充结果
            df.loc[idx, 'ate'] = result['ate']
            df.loc[idx, 'ate_ci_lower'] = result['ci_lower']
            df.loc[idx, 'ate_ci_upper'] = result['ci_upper']
            df.loc[idx, 'ate_is_significant'] = result['is_significant']
            df.loc[idx, 'T0'] = result['T0']
            df.loc[idx, 'T1'] = result['T1']
            df.loc[idx, 'ref_mean'] = result['ref_mean']
            df.loc[idx, 'ate_method'] = result['method']
            
        except Exception as e:
            warnings.warn(f"计算ATE失败 ({source}->{target}): {e}")
            continue
    
    # 5. 添加格式版本
    df.attrs['whitelist_format_version'] = '2.0'
    
    return df
```

### 方案3: 原因寻找算法

#### 算法流程

```python
class TradeoffDetector:
    
    def find_causes(self,
                    metric_A: str,
                    metric_B: str,
                    intervention: str,
                    causal_graph: nx.DiGraph,
                    data_df: pd.DataFrame,
                    ref_df: pd.DataFrame,
                    rules: Dict[str, str],
                    ate_engine: CausalInferenceEngine) -> List[str]:
        """
        寻找trade-off的根本原因
        
        实现CTF论文的原因寻找算法：
        1. 找common ancestors
        2. 分析路径依赖
        3. 对每个潜在原因计算ATE
        4. 判断是否也产生trade-off
        """
        # 步骤1: 找common ancestors
        ancestors_A = set(nx.ancestors(causal_graph, metric_A))
        ancestors_B = set(nx.ancestors(causal_graph, metric_B))
        common_ancestors = ancestors_A & ancestors_B
        common_ancestors.discard(intervention)
        
        if not common_ancestors:
            return []
        
        # 步骤2: 分析路径依赖，过滤冗余原因
        explored_step = set()
        potential_causes = self._filter_potential_causes(
            common_ancestors, causal_graph, 
            metric_A, metric_B, explored_step
        )
        
        # 步骤3: 对每个潜在原因计算ATE
        causes = []
        for pc in potential_causes:
            if pc not in data_df.columns:
                continue
            
            # 计算ATE
            T0 = ref_df[pc].mean()
            T1 = data_df[data_df[intervention] == 1][pc].mean()
            
            ate_A = ate_engine.estimate_ate(
                data=data_df, treatment=pc, outcome=metric_A,
                ref_df=ref_df, T0=T0, T1=T1, mode='ctf'
            )['ate']
            
            ate_B = ate_engine.estimate_ate(
                data=data_df, treatment=pc, outcome=metric_B,
                ref_df=ref_df, T0=T0, T1=T1, mode='ctf'
            )['ate']
            
            # 判断方向
            cf_direction_A = '+' if ate_A > 0 else '-'
            cf_direction_B = '+' if ate_B > 0 else '-'
            
            cf_improve_A = (cf_direction_A == rules.get(metric_A, '+'))
            cf_improve_B = (cf_direction_B == rules.get(metric_B, '+'))
            
            # 如果也产生冲突，则是根本原因
            if cf_improve_A != cf_improve_B:
                causes.append(pc)
        
        return causes
    
    def _filter_potential_causes(self, common_ancestors, causal_graph, 
                                 metric_A, metric_B, explored_step):
        """过滤潜在原因，移除冗余"""
        potential_causes = set(common_ancestors)
        ca_last_step = {}
        
        for ca in common_ancestors:
            toX_paths = list(nx.all_simple_paths(
                causal_graph, ca, metric_A, cutoff=5
            ))
            toY_paths = list(nx.all_simple_paths(
                causal_graph, ca, metric_B, cutoff=5
            ))
            
            toX_last_step = set([x[-2] for x in toX_paths if len(x) > 1])
            toY_last_step = set([y[-2] for y in toY_paths if len(y) > 1])
            
            ca_last_step[ca] = (toX_last_step, toY_last_step)
        
        # 按拓扑排序（从最远的原因开始）
        sorted_nodes = list(nx.topological_sort(causal_graph))
        ca_last_step = dict(sorted(
            ca_last_step.items(),
            key=lambda x: sorted_nodes.index(x[0]),
            reverse=True
        ))
        
        # 过滤已探索的原因
        for ca, (toX_last_step, toY_last_step) in ca_last_step.items():
            if toX_last_step.issubset(explored_step) and \
               toY_last_step.issubset(explored_step):
                potential_causes.remove(ca)
            else:
                explored_step.update(toX_last_step)
                explored_step.update(toY_last_step)
        
        return potential_causes
```

---

## ⚠️ 风险与缓解

### 风险矩阵

| 风险 | 概率 | 影响 | 优先级 | 缓解措施 |
|------|------|------|--------|---------|
| ref_df构建错误 | 中 | 高 | P0 | 确认CTF逻辑 |
| T0/T1选择不当 | 中 | 高 | P0 | 多种策略 |
| 混淆因素遗漏 | 中 | 高 | P0 | 验证函数 |
| 性能问题 | 高 | 中 | P1 | 缓存+并行 |
| 代码重复 | 高 | 中 | P1 | 重构 |
| 兼容性问题 | 低 | 中 | P2 | 版本控制 |

### P0风险缓解方案

#### 1. ref_df构建逻辑

```python
def build_reference_df(data: pd.DataFrame,
                      groupby_columns: List[str],
                      agg_method: str = 'mean') -> pd.DataFrame:
    """
    构建参考数据集（需确认CTF逻辑）
    
    ⚠️ 重要：需要先阅读CTF的load_data.py确认正确方式
    """
    if agg_method == 'mean':
        ref_df = data.groupby(groupby_columns).mean().reset_index()
    elif agg_method == 'median':
        ref_df = data.groupby(groupby_columns).median().reset_index()
    else:
        raise ValueError(f"Unknown agg_method: {agg_method}")
    
    return ref_df
```

**行动项**:
- [ ] 阅读CTF_original/src/load_data.py
- [ ] 确认ref_df的确切构建方式
- [ ] 实现并测试

#### 2. T0/T1选择策略

```python
def select_treatment_levels(data: pd.DataFrame,
                           treatment: str,
                           strategy: str = 'minmax') -> Tuple[float, float]:
    """
    选择T0和T1的值
    
    策略:
    - 'minmax': data[treatment].min(), data[treatment].max()
    - 'quantile': quantile(0.25), quantile(0.75)
    - 'mean_std': mean - std, mean + std
    """
    if strategy == 'minmax':
        T0 = data[treatment].min()
        T1 = data[treatment].max()
    elif strategy == 'quantile':
        T0 = data[treatment].quantile(0.25)
        T1 = data[treatment].quantile(0.75)
    elif strategy == 'mean_std':
        mean = data[treatment].mean()
        std = data[treatment].std()
        T0 = mean - std
        T1 = mean + std
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return T0, T1
```

#### 3. 混淆因素验证

```python
def identify_confounders_from_graph(treatment: str,
                                   outcome: str,
                                   causal_graph: nx.DiGraph) -> Tuple[List[str], List[str]]:
    """
    从因果图识别混淆因素
    
    返回:
        (confounders, controls)
        - confounders: 同时影响treatment和outcome的变量
        - controls: 只影响treatment的变量
    """
    treatment_parents = set(causal_graph.predecessors(treatment))
    outcome_parents = set(causal_graph.predecessors(outcome))
    
    # 混淆因素：同时指向两者
    confounders = treatment_parents & outcome_parents
    
    # 控制变量：只指向treatment
    controls = treatment_parents - confounders
    
    return list(confounders), list(controls)

def validate_confounders(data, treatment, outcome, confounders):
    """验证混淆因素的有效性"""
    diagnostics = {
        'missing_values': data[confounders].isnull().sum(),
        'variance': data[confounders].var(),
        'correlation_treatment': data[confounders].corrwith(data[treatment]),
        'correlation_outcome': data[confounders].corrwith(data[outcome])
    }
    return diagnostics
```

---

## 📋 实施计划

### Phase 0: 准备工作（2-3天）⚠️ 必须先完成

**目标**: 确认CTF逻辑，建立测试框架

**任务**:
1. [ ] 阅读CTF源码确认关键逻辑
   - `CTF_original/src/inf.py` - ref_df和T0/T1的使用
   - `CTF_original/src/load_data.py` - 数据加载和ref_df构建
   
2. [ ] 实现混淆因素识别函数
   - `identify_confounders_from_graph()`
   - `validate_confounders()`
   - 单元测试
   
3. [ ] 创建性能基准测试框架
   - `tests/benchmark/test_ate_performance.py`
   - 定义性能指标

**产出**:
- CTF逻辑理解文档
- 混淆因素识别模块
- 性能基准测试框架

---

### Phase 1: P0功能实现（5-7天）

**目标**: 实现核心ATE计算功能

**任务**:
1. [ ] 重构CausalInferenceEngine
   - 统一estimate_ate接口
   - 实现mode='ctf'逻辑
   - 提取公共方法
   
2. [ ] 实现辅助函数
   - `build_reference_df()`
   - `select_treatment_levels()`
   - `identify_confounders_from_graph()`
   
3. [ ] 错误处理和验证
   - 数据验证
   - 模型验证
   - 结果验证
   
4. [ ] 单元测试
   - `tests/test_ate_calculation.py`
   - 覆盖率 > 80%

**产出**:
- 重构后的causal_inference.py
- 完整的单元测试
- 测试报告

---

### Phase 2: P1功能实现（5-7天）

**目标**: 实现原因寻找和白名单扩展

**任务**:
1. [ ] 实现原因寻找算法
   - `TradeoffDetector.find_causes()`
   - 路径分析逻辑
   - 单元测试
   
2. [ ] 白名单格式扩展
   - 创建`add_ate_to_whitelist.py`
   - 批量处理脚本
   - 格式验证
   
3. [ ] 性能优化
   - 实现缓存机制
   - 并行计算支持
   - 进度条和日志
   
4. [ ] 集成测试
   - 端到端测试
   - 与CTF对比验证

**产出**:
- 原因寻找算法
- 白名单扩展工具
- 性能优化代码
- 集成测试报告

---

### Phase 3: 验证和文档（3-5天）

**目标**: 完善文档和验证结果

**任务**:
1. [ ] 与CTF结果对比
   - 使用相同数据
   - 计算相关系数
   - 分析差异原因
   
2. [ ] 编写完整文档
   - API文档
   - 使用指南
   - 技术报告
   
3. [ ] 代码审查和重构
   - 代码审查
   - 重构优化
   - 文档补充

**产出**:
- 对比验证报告
- 完整文档
- 优化后的代码

---

## 🧪 测试策略

### 单元测试

```python
# tests/test_ate_calculation.py

def test_ate_basic():
    """基本ATE计算测试"""
    data = generate_synthetic_data()
    engine = CausalInferenceEngine()
    result = engine.estimate_ate(
        data=data, treatment='X', outcome='Y',
        confounders=['Z'], mode='ctf'
    )
    assert 'ate' in result
    assert isinstance(result['ate'], float)

def test_confounder_identification():
    """混淆因素识别测试"""
    graph = create_test_causal_graph()
    confounders, controls = identify_confounders_from_graph(
        'X', 'Y', graph
    )
    assert 'Z' in confounders

def test_ref_df_building():
    """ref_df构建测试"""
    data = load_test_data()
    ref_df = build_reference_df(data, ['group_col'])
    assert 'group_col' in ref_df.columns
```

### 集成测试

```python
# tests/test_integration.py

def test_ctf_alignment():
    """与CTF对齐测试"""
    # 使用CTF相同的数据和参数
    ctf_ate = compute_with_ctf(...)
    our_ate = engine.estimate_ate(..., mode='ctf')['ate']
    
    # 允许1%的相对误差
    assert np.isclose(ctf_ate, our_ate, rtol=1e-2)

def test_end_to_end():
    """端到端测试"""
    # 加载数据
    data = load_energy_data()
    graph = load_causal_graph()
    
    # 计算ATE
    engine = CausalInferenceEngine()
    result = engine.estimate_ate(
        data=data, treatment='batch_size',
        outcome='gpu_energy', causal_graph=graph, mode='ctf'
    )
    
    # 验证结果
    assert result['is_significant'] == True
    assert result['ate'] > 0
```

### 性能测试

```python
# tests/benchmark/test_performance.py

def test_ate_performance():
    """性能基准测试"""
    data = load_large_dataset(n_samples=10000)
    edges = get_test_edges(n=100)
    
    engine = CausalInferenceEngine()
    times = []
    
    for source, target in edges:
        start = time.time()
        engine.estimate_ate(
            data=data, treatment=source,
            outcome=target, mode='ctf'
        )
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    assert avg_time < 1.0, f"ATE计算过慢: {avg_time:.2f}s"
```

---

## 📦 交付物清单

### 代码文件

| 文件 | 说明 | 状态 |
|------|------|------|
| `utils/causal_inference.py` | ATE计算引擎 | 待重构 |
| `utils/tradeoff_detection.py` | Trade-off检测（含原因寻找） | 待扩展 |
| `tools/data_management/add_ate_to_whitelist.py` | 白名单扩展工具 | 新建 |
| `utils/ref_df_builder.py` | ref_df构建工具 | 新建 |
| `utils/confounder_identifier.py` | 混淆因素识别 | 新建 |

### 测试文件

| 文件 | 说明 |
|------|------|
| `tests/test_ate_calculation.py` | ATE计算单元测试 |
| `tests/test_confounder_identification.py` | 混淆因素测试 |
| `tests/test_tradeoff_detection.py` | Trade-off检测测试 |
| `tests/test_integration.py` | 集成测试 |
| `tests/benchmark/test_performance.py` | 性能测试 |

### 文档文件

| 文件 | 说明 |
|------|------|
| `docs/ATE_INTEGRATION_API.md` | API文档 |
| `docs/ATE_INTEGRATION_GUIDE.md` | 使用指南 |
| `docs/CTF_ALIGNMENT_REPORT.md` | 与CTF对比报告 |
| `docs/ATE_INTEGRATION_COMPLETE_PLAN_20260125.md` | 本文档 |

### 数据文件

| 文件 | 说明 |
|------|------|
| `results/energy_research/data/interaction/whitelist/*_with_ate.csv` | 扩展后的白名单 |

---

## 📈 成功标准

### 技术指标

- [ ] 与CTF的ATE相关系数 > 0.95
- [ ] 单条边ATE计算时间 < 1s
- [ ] 单元测试覆盖率 > 80%
- [ ] 所有P0风险已缓解
- [ ] 与CTF结果误差 < 5%

### 功能完整性

- [ ] 支持CTF兼容模式
- [ ] 自动识别混淆因素
- [ ] 实现原因寻找算法
- [ ] 白名单格式扩展
- [ ] 性能优化（缓存/并行）

### 文档和质量

- [ ] API文档完整
- [ ] 使用指南清晰
- [ ] 代码审查通过
- [ ] 可复现性验证

---

## 🔄 后续工作

### 短期（Phase完成后）

1. **应用到实际数据**
   - 为所有6个任务组计算ATE
   - 生成扩展白名单
   - 分析trade-off模式

2. **RQ2分析**
   - 使用扩展白名单进行trade-off分析
   - 识别能耗-性能权衡
   - 生成研究报告

### 中期（1-2个月）

1. **性能优化**
   - 实现分布式计算
   - 优化内存使用
   - 添加增量更新

2. **功能增强**
   - 支持更多因果推断方法
   - 可视化ATE结果
   - 交互式探索工具

### 长期（3-6个月）

1. **方法改进**
   - 研究更好的ATE估计方法
   - 探索异质性处理效应（HTE）
   - 因果机制学习

2. **工具开源**
   - 整理代码库
   - 编写完整文档
   - 发布为开源工具

---

## 📞 联系和支持

### 问题反馈

如有问题或建议，请：
1. 查阅相关文档（见文档导航）
2. 检查测试用例
3. 联系项目负责人

### 文档维护

- **维护者**: Green
- **版本**: v1.0
- **最后更新**: 2026-01-25
- **下次审查**: Phase 1完成后

---

## 附录

### A. 参考资源

**论文**:
- CTF: Causality-Aided Trade-off Analysis for ML Fairness
- Chernozhukov et al. (2018): Double/Debiased Machine Learning

**工具**:
- EconML: https://github.com/py-why/EconML
- DoWhy: https://github.com/py-why/dowhy
- NetworkX: https://networkx.org/

**项目文档**:
- CLAUDE_FULL_REFERENCE.md
- CAUSAL_EDGE_WHITELIST_DESIGN.md
- QUESTIONS_2_3_DIBS_ANALYSIS_PLAN.md

### B. 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| ATE | Average Treatment Effect | 平均处理效应 |
| DML | Double Machine Learning | 双重机器学习 |
| ref_df | Reference DataFrame | 参考数据集 |
| T0/T1 | Treatment levels | 对照/处理值 |
| Trade-off | 权衡 | 两个目标之间的冲突 |
| Confounder | 混淆因素 | 同时影响干预和结果的变量 |

### C. 变更日志

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-01-25 | 初始版本，完整方案 |

---

**文档结束**
