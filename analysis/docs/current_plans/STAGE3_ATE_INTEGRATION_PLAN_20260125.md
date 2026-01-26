# 阶段3：ATE集成到白名单方案设计

**执行日期**: 2026-01-25
**状态**: 🚝 进行中

---

## 📋 方案目标

将CTF论文的ATE计算功能集成到我们的因果分析流程中，扩展白名单数据格式以支持RQ2的trade-off分析。

---

## 1️⃣ ATE计算函数修改方案

### 1.1 当前实现 (utils/causal_inference.py)

```python
def estimate_ate(self, data, treatment, outcome, confounders, controls=None):
    # 使用'auto'模型
    dml = LinearDML(model_y='auto', model_t='auto', random_state=42)
    dml.fit(Y, T, X=X, W=W)
    
    # 在原数据上计算
    ate = dml.ate(X=X)
    return ate, (ci_lower, ci_upper)
```

**问题**:
- ❌ 不支持T0/T1显式指定
- ❌ 不支持ref_df（参考数据集）
- ❌ 使用'auto'模型（与CTF不一致）
- ❌ 混淆因素需手动传入

### 1.2 修改方案 - 添加CTF兼容模式

#### 方案A: 扩展现有函数（推荐）

```python
def estimate_ate(self,
                 data: pd.DataFrame,
                 treatment: str,
                 outcome: str,
                 confounders: List[str],
                 controls: Optional[List[str]] = None,
                 ref_df: Optional[pd.DataFrame] = None,
                 T0: Optional[float] = None,
                 T1: Optional[float] = None,
                 use_ctf_compatibility: bool = False) -> Dict:
    """
    估计平均处理效应(ATE)
    
    参数:
        data: 实验数据
        treatment: 处理变量
        outcome: 结果变量
        confounders: 混淆因素列表
        controls: 控制变量列表
        ref_df: 参考数据集（CTF模式）
        T0: 对照值（CTF模式）
        T1: 处理值（CTF模式）
        use_ctf_compatibility: 是否使用CTF兼容模式
    
    返回:
        {
            'ate': float,
            'ci_lower': float,
            'ci_upper': float,
            'is_significant': bool,
            'T0': float,
            'T1': float,
            'ref_mean': float,
            'method': str
        }
    """
    if use_ctf_compatibility:
        # CTF兼容模式：使用RandomForest
        from sklearn.ensemble import RandomForestRegressor
        
        dml = LinearDML(
            model_y=RandomForestRegressor(),
            model_t=RandomForestRegressor(),
            random_state=0
        )
        
        # 在ref_df上评估（如果提供）
        X_eval = ref_df[confounders].values if ref_df is not None else X
        T0_eval = T0 if T0 is not None else None
        T1_eval = T1 if T1 is not None else None
        
        dml.fit(Y, T, X=X)
        
        if T0 is not None and T1 is not None:
            ate = dml.ate(X=X_eval, T0=T0_eval, T1=T1_eval)
        else:
            ate = dml.ate(X=X_eval)
    else:
        # 默认模式：保持向后兼容
        dml = LinearDML(model_y='auto', model_t='auto', random_state=42)
        dml.fit(Y, T, X=X, W=W)
        ate = dml.ate(X=X)
    
    # 计算置信区间
    # ...
    
    return {
        'ate': float(ate),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'is_significant': is_significant,
        'T0': T0,
        'T1': T1,
        'ref_mean': ref_df[treatment].mean() if ref_df is not None else None,
        'method': 'DML_CTF' if use_ctf_compatibility else 'DML_auto'
    }
```

**优点**:
- ✅ 向后兼容（默认保持原有行为）
- ✅ 支持CTF模式（可选开启）
- ✅ 返回结构化字典（易于扩展）
- ✅ 保留置信区间功能

**缺点**:
- ⚠️ 函数签名较长
- ⚠️ 需要文档说明不同模式

#### 方案B: 创建独立CTF函数

```python
def compute_ate_ctf(parent: str,
                    child: str,
                    data_df: pd.DataFrame,
                    ref_df: pd.DataFrame,
                    dg: nx.DiGraph,
                    T0: float,
                    T1: float) -> float:
    """
    CTF论文原版ATE计算函数
    
    完全复现CTF的compute_ate逻辑
    """
    # 识别混淆因素
    parent_parents = list(dg.predecessors(parent))
    child_parents = list(dg.predecessors(child))
    X_cols = list(set(parent_parents + child_parents))
    
    # 移除特定变量
    if "EGR" in X_cols:
        X_cols.remove("EGR")
    if parent in X_cols:
        X_cols.remove(parent)
    
    # 提取数据
    X = data_df[X_cols]
    T = data_df[parent]
    Y = data_df[child]
    
    # 使用LinearDML + RandomForest
    est = LinearDML(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor(),
        random_state=0
    )
    est.fit(Y, T, X=X)
    
    # 在ref_df上计算ATE
    ate = est.ate(X=ref_df[X_cols], T0=T0, T1=T1)
    return float(ate)
```

**优点**:
- ✅ 完全对齐CTF实现
- ✅ 简单直接，无副作用
- ✅ 易于验证正确性

**缺点**:
- ❌ 代码重复（DML逻辑）
- ❌ 维护成本高

### 1.3 推荐方案

**采用方案A（扩展现有函数）+ 方案B（独立函数）组合**:

1. **保留方案A**: 作为通用ATE计算接口
2. **添加方案B**: 作为CTF专用函数，用于验证和对比
3. **统一返回格式**: 两者都返回结构化字典

---

## 2️⃣ 白名单数据格式扩展方案

### 2.1 当前格式

```csv
source,target,strength,edge_type,is_significant,strength_level,source_category,target_category,question_relevance,interpretation
hyperparam_batch_size,energy_gpu_min_watts,0.95,moderation,yes,very_strong,hyperparam,energy,other,并行模式调节batch_size对gpu_min_watts的效应
```

**问题**:
- ❌ 不包含ATE信息
- ❌ 无法用于trade-off分析
- ❌ 缺少统计推断细节

### 2.2 扩展格式（添加ATE列）

#### 新增列定义

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| ate | float | 平均处理效应 | 0.123 |
| ci_lower | float | 95%置信区间下界 | 0.089 |
| ci_upper | float | 95%置信区间上界 | 0.157 |
| is_significant | bool | 是否统计显著 | True |
| T0 | float | 对照值 | 0.0 |
| T1 | float | 处理值 | 1.0 |
| ref_mean | float | 参考均值 | 0.5 |
| ate_method | str | 计算方法 | DML_CTF |

#### 扩展后的CSV示例

```csv
source,target,strength,edge_type,is_significant,strength_level,source_category,target_category,question_relevance,interpretation,ate,ci_lower,ci_upper,ate_ci_lower,ate_ci_upper,T0,T1,ref_mean,ate_method
hyperparam_batch_size,energy_gpu_min_watts,0.95,moderation,yes,very_strong,hyperparam,energy,other,并行模式调节batch_size对gpu_min_watts的效应,0.123,0.089,0.157,0.089,0.157,0.0,1.0,0.5,DML_CTF
```

### 2.3 迁移策略

#### 步骤1: 创建扩展脚本

**文件**: `tools/data_management/add_ate_to_whitelist.py`

```python
def add_ate_columns(whitelist_path: str,
                    data: pd.DataFrame,
                    causal_graph: nx.DiGraph,
                    use_ctf_mode: bool = True) -> pd.DataFrame:
    """
    为白名单CSV添加ATE列
    
    参数:
        whitelist_path: 白名单CSV路径
        data: 原始数据
        causal_graph: 因果图
        use_ctf_mode: 是否使用CTF兼容模式
    
    返回:
        添加了ATE列的DataFrame
    """
    # 读取白名单
    df = pd.read_csv(whitelist_path)
    
    # 初始化ATE列
    df['ate'] = np.nan
    df['ate_ci_lower'] = np.nan
    df['ate_ci_upper'] = np.nan
    df['ate_is_significant'] = False
    df['T0'] = np.nan
    df['T1'] = np.nan
    df['ref_mean'] = np.nan
    df['ate_method'] = 'N/A'
    
    # 创建ATE计算引擎
    engine = CausalInferenceEngine()
    
    # 为每条边计算ATE
    for idx, row in df.iterrows():
        source = row['source']
        target = row['target']
        
        try:
            if use_ctf_mode:
                # 构建ref_df
                ref_df = data.groupby([source]).mean().reset_index()
                T0 = data[source].min()
                T1 = data[source].max()
                
                # 计算ATE（CTF模式）
                result = engine.estimate_ate(
                    data=data,
                    treatment=source,
                    outcome=target,
                    confounders=[],  # 自动识别
                    ref_df=ref_df,
                    T0=T0,
                    T1=T1,
                    use_ctf_compatibility=True
                )
            else:
                # 默认模式
                result = engine.estimate_ate(
                    data=data,
                    treatment=source,
                    outcome=target,
                    confounders=[]
                )
            
            # 填充ATE列
            df.loc[idx, 'ate'] = result['ate']
            df.loc[idx, 'ate_ci_lower'] = result['ci_lower']
            df.loc[idx, 'ate_ci_upper'] = result['ci_upper']
            df.loc[idx, 'ate_is_significant'] = result['is_significant']
            df.loc[idx, 'T0'] = result['T0']
            df.loc[idx, 'T1'] = result['T1']
            df.loc[idx, 'ref_mean'] = result['ref_mean']
            df.loc[idx, 'ate_method'] = result['method']
            
        except Exception as e:
            warnings.warn(f"计算ATE失��� ({source}->{target}): {e}")
            continue
    
    return df
```

#### 步骤2: 批量处理

```python
# 处理所有白名单文件
whitelist_files = glob.glob("results/energy_research/data/interaction/whitelist/*.csv")

for file in whitelist_files:
    df_extended = add_ate_columns(
        whitelist_path=file,
        data=data,
        causal_graph=causal_graph,
        use_ctf_mode=True
    )
    
    # 保存扩展版本
    output_path = file.replace(".csv", "_with_ate.csv")
    df_extended.to_csv(output_path, index=False)
```

### 2.4 向后兼容性

**确保兼容**:
- ✅ 新列添加到末尾（不影响现有列顺序）
- ✅ 新列默认为NaN（不影响旧脚本）
- ✅ 旧文件仍可正常读取

---

## 3️⃣ 原因寻找算法实现方案

### 3.1 CTF原始算法 (inf.py:280-330)

**核心逻辑**:
```python
# 1. 找到两个指标的common ancestors
metric_A_ancestors = set(nx.ancestors(dg, metric_A))
metric_B_ancestors = set(nx.ancestors(dg, metric_B))
common_ancestors = metric_A_ancestors & metric_B_ancestors

# 2. 过滤潜在原因
potential_causes = []
for ca in common_ancestors:
    # 检查是否通过fair_method影响两个指标
    toX_paths = nx.all_simple_paths(dg, source=ca, target=metric_A)
    toY_paths = nx.all_simple_paths(dg, source=ca, target=metric_B)
    
    # 检查路径的最后一步
    toX_last_step = set([x[-2] for x in toX_paths])
    toY_last_step = set([x[-2] for x in toY_paths])
    
    # 如果已经探索过，则移除
    if toX_last_step.issubset(explored_step) and toY_last_step.issubset(explored_step):
        potential_causes.remove(ca)
    else:
        explored_step.update(toX_last_step)
        explored_step.update(toY_last_step)

# 3. 对每个潜在原因计算ATE
for pc in potential_causes:
    ate_A = compute_ate(pc, metric_A, data_df, ref_df, dg, T0, T1)
    ate_B = compute_ate(pc, metric_B, data_df, ref_df, dg, T0, T1)
    
    cf_direction_A = '+' if ate_A > 0 else '-'
    cf_direction_B = '+' if ate_B > 0 else '-'
    
    cf_improve_A = (cf_direction_A == rules[metric_A])
    cf_improve_B = (cf_direction_B == rules[metric_B])
    
    if cf_improve_A != cf_improve_B:
        causes.append(pc)
```

### 3.2 我们的实现方案

**文件**: `utils/tradeoff_detection.py`

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
        
        实现CTF论文的原因寻找算法
        
        参数:
            metric_A, metric_B: 冲突的两个指标
            intervention: 干预变量
            causal_graph: 因果图
            data_df: 实验数据
            ref_df: 参考数据
            rules: 期望改善方向字典
            ate_engine: ATE计算引擎
        
        返回:
            causes: 原因列表
        """
        # 步骤1: 找common ancestors
        ancestors_A = set(nx.ancestors(causal_graph, metric_A))
        ancestors_B = set(nx.ancestors(causal_graph, metric_B))
        common_ancestors = ancestors_A & ancestors_B
        
        if self.verbose:
            print(f"\n寻找原因: {metric_A} vs {metric_B}")
            print(f"  Common ancestors: {len(common_ancestors)}")
        
        # 移除intervention本身
        common_ancestors.discard(intervention)
        
        if not common_ancestors:
            return []
        
        # 步骤2: 分析路径依赖
        explored_step = set()
        potential_causes = set(common_ancestors)
        
        ca_last_step = {}
        for ca in common_ancestors:
            toX_paths = list(nx.all_simple_paths(causal_graph, ca, metric_A, cutoff=5))
            toY_paths = list(nx.all_simple_paths(causal_graph, ca, metric_B, cutoff=5))
            
            toX_last_step = set([x[-2] for x in toX_paths if len(x) > 1])
            toY_last_step = set([y[-2] for y in toY_paths if len(y) > 1])
            
            ca_last_step[ca] = (toX_last_step, toY_last_step)
        
        # 按拓扑排序（从最远的原因开始）
        sorted_nodes = list(nx.topological_sort(causal_graph))
        ca_last_step = dict(
            sorted(ca_last_step.items(), key=lambda x: sorted_nodes.index(x[0]), reverse=True)
        )
        
        # 过滤已探索的原因
        for ca, (toX_last_step, toY_last_step) in ca_last_step.items():
            if toX_last_step.issubset(explored_step) and toY_last_step.issubset(explored_step):
                potential_causes.remove(ca)
            else:
                explored_step.update(toX_last_step)
                explored_step.update(toY_last_step)
        
        if self.verbose:
            print(f"  Potential causes: {len(potential_causes)}")
        
        # 步骤3: 计算每个潜在原因的ATE
        causes = []
        for pc in potential_causes:
            if pc in data_df.columns:
                # 计算ATE
                T0 = ref_df[pc].mean()
                T1 = data_df[data_df[intervention] == 1][pc].mean()
                
                ate_A = ate_engine.estimate_ate(
                    data=data_df,
                    treatment=pc,
                    outcome=metric_A,
                    ref_df=ref_df,
                    T0=T0,
                    T1=T1,
                    use_ctf_compatibility=True
                )['ate']
                
                ate_B = ate_engine.estimate_ate(
                    data=data_df,
                    treatment=pc,
                    outcome=metric_B,
                    ref_df=ref_df,
                    T0=T0,
                    T1=T1,
                    use_ctf_compatibility=True
                )['ate']
                
                # 判断方向
                cf_direction_A = '+' if ate_A > 0 else '-'
                cf_direction_B = '+' if ate_B > 0 else '-'
                
                # 检查是否改善
                cf_improve_A = (cf_direction_A == rules.get(metric_A, '+'))
                cf_improve_B = (cf_direction_B == rules.get(metric_B, '+'))
                
                # 如果也产生冲突，则是根本原因
                if cf_improve_A != cf_improve_B:
                    causes.append(pc)
                    if self.verbose:
                        print(f"  ✓ 找到原因: {pc}")
                        print(f"    {metric_A}: ATE={ate_A:.4f} ({cf_direction_A})")
                        print(f"    {metric_B}: ATE={ate_B:.4f} ({cf_direction_B})")
        
        return causes
```

### 3.3 集成到TradeoffDetector

```python
def detect_tradeoffs_with_causes(self,
                                 causal_effects: Dict,
                                 causal_graph: nx.DiGraph,
                                 data_df: pd.DataFrame,
                                 ref_df: pd.DataFrame,
                                 rules: Dict[str, str],
                                 ate_engine: CausalInferenceEngine) -> List[Dict]:
    """
    检测trade-off并寻找原因（完整版）
    
    返回:
        [
            {
                'intervention': str,
                'metric1': str,
                'metric2': str,
                'ate1': float,
                'ate2': float,
                'sign1': str,
                'sign2': str,
                'causes': List[str],  # ⭐ 新增
                'is_significant': bool
            },
            ...
        ]
    """
    tradeoffs = []
    
    # 检测trade-off
    for source, targets in edges_by_source.items():
        for (target1, result1), (target2, result2) in combinations:
            tradeoff = self._check_tradeoff_pair(...)
            
            if tradeoff:
                # ⭐ 寻找原因
                causes = self.find_causes(
                    metric_A=target1,
                    metric_B=target2,
                    intervention=source,
                    causal_graph=causal_graph,
                    data_df=data_df,
                    ref_df=ref_df,
                    rules=rules,
                    ate_engine=ate_engine
                )
                
                tradeoff['causes'] = causes
                tradeoffs.append(tradeoff)
    
    return tradeoffs
```

---

## 4️⃣ 与CTF源码的对齐策略

### 4.1 对齐优先级

| 功能 | 对齐程度 | 优先级 | 策略 |
|------|---------|--------|------|
| ATE计算逻辑 | 60% → 95% | P0 | 扩展函数，添加CTF模式 |
| 混淆因素识别 | 0% → 100% | P0 | 实现自动识别 |
| Trade-off检测 | 80% → 90% | P1 | 添加原因寻找 |
| 模型选择 | 0% → 100% | P1 | 添加RandomForest选项 |
| DoWhy集成 | 0% → 50% | P2 | 可选验证 |

### 4.2 验证策略

#### 单元测试

```python
# tests/test_ctf_alignment.py

def test_ate_calculation():
    """验证ATE计算与CTF一致"""
    # 使用CTF相同的数据和参数
    ctf_ate = compute_ate_ctf(...)
    our_ate = engine.estimate_ate(..., use_ctf_compatibility=True)
    
    assert np.isclose(ctf_ate, our_ate['ate'], rtol=1e-3)

def test_cause_finding():
    """验证原因寻找算法"""
    # 使用已知的trade-off案例
    causes = detector.find_causes(...)
    
    assert len(causes) > 0
    assert expected_cause in causes
```

#### 对比实验

```python
# scripts/compare_with_ctf.py

def run_comparison():
    """在相同数据上对比CTF和我们的实现"""
    results = {
        'CTF': ctf_results,
        'Ours': our_results
    }
    
    # 计算相关系数
    correlation = results['CTF']['ate'].corr(results['Ours']['ate'])
    
    print(f"ATE相关系数: {correlation:.3f}")
```

### 4.3 分阶段实施

**Phase 1 (本周)**: P0功能
- ✅ 扩展ATE计算函数
- ✅ 实现混淆因素自动识别
- ✅ 添加T0/T1/ref_df支持

**Phase 2 (下周)**: P1功能
- ✅ 实现原因寻找算法
- ✅ 添加RandomForest模型选项
- ✅ 白名单格式扩展

**Phase 3 (可选)**: P2功能
- ⏳ DoWhy集成验证
- ⏳ 性能优化
- ⏳ 文档完善

---

## 📋 阶段3总结

### 核心修改

1. **ATE计算函数扩展** - 添加CTF兼容模式
2. **白名单格式扩展** - 添加8个ATE列
3. **原因寻找算法** - 实现CTF的完整逻辑
4. **验证策略** - 单元测试 + 对比实验

### 预期成果

- ✅ 与CTF源码95%对齐
- ✅ 白名单支持完整ATE信息
- ✅ 支持RQ2 trade-off分析
- ✅ 保持向后兼容性

### 风险缓解

- ⚠️ 性能影响: 使用lazy evaluation
- ⚠️ 数据缺失: 提供默认值和警告
- ⚠️ 兼容性问题: 保留旧接口

---

## 🎯 下一步

**阶段4**: 评估方案风险
