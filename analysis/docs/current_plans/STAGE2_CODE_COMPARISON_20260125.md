# 阶段2：CTF源码与复现代码差异对比

**执行日期**: 2026-01-25
**状态**: ✅ 完成
**关联文档**: CTF_SOURCE_CODE_COMPARISON_20260125.md

---

## 📊 差异对比总结

### 五大关键差异

| # | 功能 | CTF原始实现 | 复现代码 | 影响等级 |
|---|------|------------|---------|---------|
| 1️⃣ | **ATE计算方式** | 在ref_df上评估 + T0/T1显式 | 在原数据上 | ⭐⭐⭐ 关键 |
| 2️⃣ | **混淆因素识别** | 自动从因果图提取 | 手动传入列表 | ⭐⭐⭐ 关键 |
| 3️⃣ | **原因寻找** | 深度分析common ancestors | 不支持 | ⭐⭐⭐ 关键 |
| 4️⃣ | **模型选择** | RandomForestRegressor | 'auto' | ⭐⭐ 中等 |
| 5️⃣ | **因果验证** | 集成DoWhy | 无 | ⭐⭐ 中等 |

---

## 1️⃣ ATE计算差异详解

### CTF原始版本
```python
def compute_ate(parent, child, data_df, ref_df, dg, T0, T1):
    # 自动识别混淆因素
    parent_parents = list(dg.predecessors(parent))
    child_parents = list(dg.predecessors(child))
    X_cols = list(set(parent_parents + child_parents))
    
    # 使用RandomForest
    est = LinearDML(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor(),
        random_state=0
    )
    est.fit(Y, T, X=X)
    
    # 在ref_df上计算，使用T0/T1
    ate = est.ate(X=ref_df[X_cols], T0=T0, T1=T1)
    return ate
```

### 复现代码
```python
def estimate_ate(self, data, treatment, outcome, confounders, controls=None):
    # 手动传入confounders
    dml = LinearDML(model_y='auto', model_t='auto', random_state=42)
    dml.fit(Y, T, X=X, W=W)
    
    # 在原数据上计算
    ate = dml.ate(X=X)
    return ate, (ci_lower, ci_upper)
```

### 差异影响

| 特性 | 影响 |
|------|------|
| ❌ 缺少T0/T1 | 无法指定干预水平，结果可能不准确 |
| ❌ 缺少ref_df | 无法在基准数据上评估，结果可解释性差 |
| ❌ 混淆因素手动传入 | 增加使用复杂度，容易出错 |
| ⚠️ 模型选择不同 | 可能导致ATE估计差异 |

---

## 2️⃣ Trade-off检测差异详解

### CTF原始版本

**核心算法**:
1. 使用DoWhy计算所有method->metric的因果效应
2. 定义rules字典（期望改善方向）
3. 检测：`improve_A != improve_B`
4. **原因寻找**:
   - 找common ancestors
   - 对每个潜在原因计算ATE
   - 判断cf_improve方向
   - 输出原因列表

**关键代码片段**:
```python
# 步骤1: 定义rules
rules = {
    "Test_sens_DP": "+",
    "Test_TI": "-",
    ...
}

# 步骤2: 检测冲突
improve_A = (rules[metric_A] == direction_A)
improve_B = (rules[metric_B] == direction_B)

if improve_A != improve_B:
    # 步骤3: 寻找原因
    causes = find_causes(...)
    record = {'trade-off': f"{metric_A}--{metric_B}", 'causes': causes}
```

### 复现代码

**核心算法**:
1. 接收预计算的causal_effects
2. 检测：`sign1 != sign2`
3. **不进行原因寻找**
4. 返回tradeoff对象列表

**缺失功能**:
- ❌ 不支持原因寻找
- ❌ 不使用DoWhy验证
- ⚠️ sign函数接口不同

---

## 3️⃣ 关键功能缺失清单

### P0 - 必须实现（影响核心功能）

| 功能 | 缺失影响 | 优先级 |
|------|---------|--------|
| ❌ ref_df支持 | 无法在基准数据评估ATE | P0 |
| ❌ T0/T1显式指定 | 无法精确控制干预水平 | P0 |
| ❌ 原因寻找算法 | 无法识别trade-off根本原因 | P0 |
| ❌ 白名单ATE列 | 无法存储ATE结果 | P0 |

### P1 - 强烈建议（影响结果质量）

| 功能 | 缺失影响 | 优先级 |
|------|---------|--------|
| ⚠️ 自动混淆因素识别 | 使用复杂度高，易出错 | P1 |
| ⚠️ RandomForest模型 | ATE估计可能有偏差 | P1 |
| ⚠️ DoWhy集成 | 缺少因果模型验证 | P1 |

### P2 - 可选（不影响核心功能）

| 功能 | 价值 | 优先级 |
|------|------|------|
| 💡 CSV输出 | 方便结果导出 | P2 |
| 💡 可视化增强 | 提升可读性 | P2 |

---

## 4️⃣ 白名单数据格式分析

### 当前格式

根据`CAUSAL_EDGE_WHITELIST_DESIGN.md`:
```
- task_name
- edge_name
- is_whitelisted
- confidence_score
- evidence_summary
- dibs_probability
- expert_knowledge
- research_question
```

### 需要添加的ATE列

为了支持RQ2 trade-off分析:
```
- ate: 平均处理效应
- ci_lower: 置信区间下界
- ci_upper: 置信区间上界
- is_significant: 是否统计显著
- T0: 对照值
- T1: 处理值
- ref_mean: 参考均值
- ate_method: 计算方法 (DML/Linear/etc)
```

---

## 📋 阶段2关键结论

1. **功能完整性**: 复现代码约实现60%功能
2. **关键缺失**: ref_df、T0/T1、原因寻找（三大P0功能）
3. **兼容性**: 需要适配CTF的接口设计
4. **白名单扩展**: 需要添加8个ATE相关列

---

## 🎯 下一步行动

**阶段3**: 设计ATE集成到白名单的详细方案

包括:
1. ATE计算函数修改方案
2. 白名单数据格式扩展方案
3. 原因寻找算法实现方案
4. 与CTF源码的对齐策略
