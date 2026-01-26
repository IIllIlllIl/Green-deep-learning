# CTF源码 vs 复现代码对比分析

## 1. ATE计算实现对比

### CTF原始实现 (CTF_original/src/inf.py:78-97)

```python
def compute_ate(parent, child, data_df, ref_df, dg, T0, T1):
    """
    参数:
        parent: 处理变量
        child: 结果变量
        data_df: 实验数据
        ref_df: 参考数据（基准）
        dg: 因果图 (networkx.DiGraph)
        T0, T1: 对照/处理的值
    
    返回: ATE值 (float)
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
    
    # 使用LinearDML
    est = LinearDML(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor(),
        random_state=0
    )
    est.fit(Y, T, X=X)
    
    # 在参考数据上计算ATE
    ate = est.ate(X=ref_df[X_cols], T0=T0, T1=T1)
    return ate
```

### 我们的实现 (utils/causal_inference.py)

```python
class CausalInferenceEngine:
    def estimate_ate(self, data, treatment, outcome, confounders, controls=None):
        """
        参数:
            data: DataFrame
            treatment: 处理变量名
            outcome: 结果变量名
            confounders: 混淆因素列表
            controls: 控制变量列表（可选）
        
        返回: (ate, (ci_lower, ci_upper))
        """
        from econml.dml import LinearDML
        
        T = data[treatment].values
        Y = data[outcome].values
        X = data[confounders].values
        W = data[controls].values if controls else None
        
        dml = LinearDML(model_y='auto', model_t='auto', random_state=42)
        dml.fit(Y, T, X=X, W=W)
        
        ate = dml.ate(X=X)
        # ... 计算置信区间
        return ate, (ci_lower, ci_upper)
```

### 关键差异

| 特性 | CTF原始 | 我们的实现 | 差异影响 |
|------|---------|-----------|---------|
| **输入参数** | ref_df, dg, T0, T1 | confounders列表 | ⚠️ 需要适配 |
| **混淆因素识��** | 自动从因果图提取 | 手动传入 | ⚠️ 需要自动化 |
| **模型选择** | RandomForestRegressor | 'auto' | ⚠️ 可能影响结果 |
| **ATE计算** | 在ref_df上评估 | 在原数据上评估 | ⚠️ **关键差异** |
| **返回值** | float (ATE) | tuple (ATE, CI) | ✅ 更丰富 |
| **T0/T1支持** | ✅ 显式指定 | ❌ 缺失 | ❌ **功能缺失** |

---

## 2. Trade-off检测算法对比

### CTF原始实现 (CTF_original/src/inf.py:150-330)

**算法流程**:
1. 定义规则 (rules) - 指标期望改善方向
2. 使用DoWhy CausalModel计算因果效应
3. 检测冲突: improve_A != improve_B
4. 寻找原因:
   - 找common ancestors
   - 对每个潜在原因计算ATE
   - 判断cf_improve方向

**核心代码结构**:
```python
# 步骤1: 定义规则
rules = {
    "Test_sens_DP": "+",  # 正向改善
    "Test_TI": "-",       # 负向改善
    ...
}

# 步骤2: 计算fair_effect
for mf in METHOD_COLUMNS:
    for mt in metrics:
        if mt in nx.descendants(dg, mf):
            model = CausalModel(data=data_df, treatment=mf, outcome=mt, graph=temp_graph)
            causal_estimate = model.estimate_effect(...)
            fair_effect[(mf, mt)] = causal_estimate.value
        else:
            fair_effect[(mf, mt)] = 0

# 步骤3: 检测trade-off
for metric_A in fair_m:
    for metric_B in robust_m:
        for fair_method in METHOD_COLUMNS:
            if fair_effect[(fair_method, metric_A)] == 0 or \
               fair_effect[(fair_method, metric_B)] == 0:
                continue
            
            direction_A = '+' if fair_effect[(fair_method, metric_A)] > 0 else '-'
            direction_B = '+' if fair_effect[(fair_method, metric_B)] > 0 else '-'
            improve_A = (rules[metric_A] == direction_A)
            improve_B = (rules[metric_B] == direction_B)
            
            if improve_A != improve_B:  # 检测到冲突
                # 步骤4: 寻找原因
                causes = find_causes(...)
                record = {'trade-off': f"{metric_A}--{metric_B}", ...}
```

### 我们的实现 (utils/tradeoff_detection.py)

```python
class TradeoffDetector:
    def detect_tradeoffs(self, causal_effects, require_significance=True):
        """
        参数:
            causal_effects: 预计算的因果效应字典
                {'A->B': {'ate': float, 'ci_lower': float, ...}}
        
        返回: 权衡列表
        """
        # 将边按源节点分组
        edges_by_source = {}
        for edge, result in causal_effects.items():
            source, target = edge.split('->')
            edges_by_source[source].append((target, result))
        
        # 检查所有目标节点对
        for source, targets in edges_by_source.items():
            for (target1, result1), (target2, result2) in combinations:
                tradeoff = self._check_tradeoff_pair(...)
                if tradeoff:
                    tradeoffs.append(tradeoff)
```

### 关键差异

| 特性 | CTF原始 | 我们的实现 | 差异影响 |
|------|---------|-----------|---------|
| **因果效应计算** | 内置DoWhy | 外部输入 | ⚠️ 需要集成 |
| **Trade-off定义** | improve_A != improve_B | sign1 != sign2 | ✅ 逻辑一致 |
| **原因寻找** | 深度分析common ancestors | 不支持 | ❌ **功能缺失** |
| **结果输出** | CSV文件 + 原因列表 | Python对象 | ⚠️ 格式不同 |
| **Sign函数** | 基于rules字典 | 传入的sign_functions | ⚠️ 接口不同 |

---

## 3. 关键功能缺失清单

### 3.1 缺失的核心功能

| 功能 | 重要性 | 影响 |
|------|--------|------|
| ❌ **ref_df支持** | ⭐⭐⭐ | 无法在参考数据上评估ATE |
| ❌ **T0/T1显式指定** | ⭐⭐⭐ | 无法指定干预水平 |
| ❌ **原因寻找算法** | ⭐⭐⭐ | 无法识别trade-off的根本原因 |
| ❌ **DoWhy集成** | ⭐⭐ | 缺少因果模型验证 |
| ❌ **Common ancestors分析** | ⭐⭐ | 无法追踪因果路径 |

### 3.2 实现差异

| 特性 | CTF | 复现代码 | 建议 |
|------|-----|---------|------|
| 混淆因素识别 | 自动从图提取 | 手动传入 | ⚠️ 需自动化 |
| 模型选择 | RandomForest | 'auto' | ⚠️ 建议统一 |
| 结果保存 | CSV | Python对象 | ✅ 可保留 |

---

## 4. 白名单数据格式分析

### 当前白名单格式
根据CAUSAL_EDGE_WHITELIST_DESIGN.md，白名单包含：
- task_name
- edge_name
- is_whitelisted
- confidence_score
- evidence_summary
- ...

### 需要添加的ATE列
为了支持RQ2的trade-off分析，需要添加：
- ate: 平均处理效应
- ci_lower: 置信区间下界
- ci_upper: 置信区间上界
- is_significant: 是否统计显著
- T0: 对照值
- T1: 处理值
- ref_mean: 参考均值

---

## 5. 修改优先级

### P0 (必须修改)
1. ✅ 添加T0/T1支持到ATE计算
2. ✅ 添加ref_df支持
3. ✅ 实现原因寻找算法
4. ✅ 扩展白名单数据格式

### P1 (强烈建议)
5. ⚠️ 统一模型选择（RandomForest）
6. ⚠️ 集成DoWhy进行验证
7. ⚠️ 自动化混淆因素识别

### P2 (可选)
8. 💡 添加CSV输出功能
9. 💡 完善错误处理
