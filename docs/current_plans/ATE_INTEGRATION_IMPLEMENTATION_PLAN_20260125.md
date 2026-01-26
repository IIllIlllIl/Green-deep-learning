# ATE集成实施方案

**日期**: 2026-01-25
**制定人**: Claude (Sonnet 4.5)
**版本**: v1.0
**状态**: ⏳ 待实施
**基于报告**: ATE_INTEGRATION_FEASIBILITY_REPORT_20260125.md

---

## 📋 执行摘要

### 总体评分: 3.5/5 🟡
**结论**: **建议实施，采用渐进式混合方案**

### 核心决策

根据可行性评估报告的核心发现：
- ✅ **技术路径清晰** - Phase 0-3的技术方案基本合理
- ⚠️ **领域适配挑战** - CTF的fairness逻辑不能直接应用到能耗研究
- ❌ **ref_df策略缺失** - 这是最关键的问题，需要创新设计
- ⚠️ **时间估算偏乐观** - 15-22天可能需要30-40天
- ✅ **P0风险识别准确** - 但有2个新风险需要添加

### 实施策略

**采用渐进式混合方案**，核心特点：
1. **快速验证优先** - 5-7天完成MVP，立即验证可行性
2. **迭代改进** - 3个迭代阶段，每个都有可交付成果
3. **风险可控** - 设置明确的停止标准，准备降级方案
4. **灵活适配** - 保留CTF核心逻辑，简化领域不适配的部分

### 关键修改

基于评估报告的建议，对原方案进行以下关键修改：
- ❌ **不再强制使用ref_df** - 改为可选参数，默认不使用
- ✅ **保留predecessors()逻辑** - CTF的核心必须一致
- ⚠️ **T0/T1采用25/75分位** - 替代min/max策略
- ⏳ **原因寻找降级为P2** - 从必须功能变为可选功能
- ⏱️ **时间调整为10-22天** - 取决于迭代3是否执行

---

## 🎯 方案选择

### 推荐方案: 渐进式混合方案 ⭐⭐⭐⭐⭐

#### 核心理念
**"快速验证，迭代改进，风险可控"**

混合方案的核心是：
1. **保留CTF的核心逻辑** - 特别是混淆因素识别使用predecessors()
2. **简化不适配的部分** - 不强制使用ref_df和T0/T1
3. **渐进式实施** - 从MVP开始，逐步完善

#### 为什么选择混合方案

| 维度 | 混合方案 | 完全复现CTF | 最小可行方案 |
|------|----------|-------------|-------------|
| **实施时间** | 10-22天 | 40-50天 | 5-7天 |
| **风险等级** | 🟡 中等 | 🔴 高 | 🟢 低 |
| **功能完整性** | 🟡 中等 | 🟢 高 | 🔴 低 |
| **领域适配性** | 🟢 高 | 🔴 低 | 🟢 高 |
| **快速验证** | ✅ 第7天 | ❌ 第40天 | ✅ 第7天 |

**选择理由**：
1. **平衡性最好** - 在风险、时间、功能之间取得最佳平衡
2. **快速反馈** - 7天内就能验证核心可行性
3. **风险可控** - 每个迭代都有明确的停止标准
4. **灵活性高** - 可以根据实际情况调整后续迭代

#### 与评估报告的对应关系

基于评估报告的核心发现，混合方案解决了以下关键问题：

| 评估报告结论 | 混合方案解决方案 |
|-------------|----------------|
| ❌ ref_df策略不明确 | ❌ 不使用ref_df（简化版） |
| ⚠️ T0/T1策略缺失 | ✅ 采用25/75分位策略 |
| ⚠️ 领域适配挑战 | ✅ 保留核心逻辑，简化不适配部分 |
| ⚠️ 时间估算偏乐观 | ✅ 调整为10-22天 |
| ❌ 原因寻找功能复杂 | ⚠️ 降级为P2可选功能 |

---

## 📊 3个迭代概览

### 迭代1: MVP (最小可行产品) - 5-7天

#### 🎯 目标
**验证核心可行性** - 确认CTF的核心逻辑在能耗数据上可行

#### 📋 范围
```python
# 核心实现
def compute_ate_mvp(parent, child, data_df, causal_graph):
    # 1. ✅ 使用predecessors()识别混淆因素（CTF核心）
    X_cols = _get_confounders_from_graph(parent, child, causal_graph)

    # 2. ❌ 不使用ref_df（简化）
    # 3. ❌ 不使用T0/T1（简化）

    # 4. 保持现有DML逻辑
    est = LinearDML(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor()
    )
    est.fit(Y, T, X=data_df[X_cols])
    ate = est.ate(X=data_df[X_cols])

    return ate
```

#### ✅ 包含功能
1. **重构混淆因素识别** - 使用NetworkX的predecessors()方法
2. **简化ATE计算** - 不使用ref_df和T0/T1参数
3. **基础接口** - 保持与现有代码的兼容性
4. **Group1验证** - 在最大的数据组上快速验证

#### ❌ 不包含功能
1. 原因寻找功能（CTF的复杂逻辑）
2. ref_df构建策略
3. T0/T1策略选择
4. 全面的测试和文档

#### 📈 成功标准
- **技术成功**: ATE计算无错误，能正常运行
- **统计成功**: 至少50%的边ATE统计显著（p<0.05）
- **对比成功**: 与现有方法结果可对比，趋势一致
- **速度成功**: 在7天内完成所有开发

#### 🚨 停止标准
如果出现以下情况，建议停止项目：
- ATE计算出现严重错误（如NaN、无穷大）
- 显著率低于10%（说明方法可能无效）
- 混淆因素识别包含target变量（逻辑错误）
- 第7天仍未完成基本验证

### 迭代2: V1.0 (增强版) - +5天

#### 🎯 目标
**完善ATE计算** - 添加ref_df和T0/T1支持，提升准确性

#### 📋 范围
```python
# 增强实现
def compute_ate_v1(parent, child, data_df, causal_graph, ref_df=None, T0=None, T1=None):
    # 1. ✅ 使用predecessors()识别混淆因素
    X_cols = _get_confounders_from_graph(parent, child, causal_graph)

    # 2. ✅ 可选ref_df参数
    if ref_df is None:
        ref_df = data_df

    # 3. ✅ T0/T1策略（25/75分位）
    if T0 is None:
        T0 = data_df[parent].quantile(0.25)
    if T1 is None:
        T1 = data_df[parent].quantile(0.75)

    # 4. 增强错误检查
    if abs(T1 - T0) < 0.1 * data_df[parent].std():
        warnings.warn(f"T1-T0差值过小: {T1-T0}")

    # 5. 完整ATE计算
    est = LinearDML(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor()
    )
    est.fit(Y, T, X=data_df[X_cols])
    ate = est.ate(X=ref_df[X_cols], T0=T0, T1=T1)

    return ate
```

#### ✅ 新增功能
1. **ref_df可选支持** - 支持数据分组基准
2. **T0/T1策略** - 实现25/75分位策略
3. **错误检查** - 添加数据质量检查
4. **扩展验证** - 覆盖所有6组数据

#### ✅ 改进点
- 增强的数据验证
- 更灵活的配置选项
- 更好的错误处理
- 完整的6组数据支持

#### 📈 成功标准
- **功能成功**: ref_df和T0/T1版本正常工作
- **质量成功**: 显著率比MVP提升（目标>60%）
- **覆盖成功**: 6组数据全部通过验证
- **兼容成功**: 与MVP版本结果一致

#### 🚨 停止标准
如果出现以下情况，建议跳过迭代3：
- ref_df版本比MVP更差（显著率下降）
- T0/T1策略不稳定（结果变化大）
- 6组数据中失败超过2组
- 时间或预算紧张

### 迭代3: V2.0 (完整版) - +10天 (可选)

#### 🎯 目标
**实现完整功能** - 添加原因寻找，支持trade-off检测

#### 📋 范围
```python
# 完整实现包含：
# 1. ✅ CTF的原因寻找逻辑（280-330行）
# 2. ✅ trade-off检测扩展
# 3. ✅ 白名单ATE集成工具
# 4. ✅ 完整的技术文档
```

#### ✅ 新增功能
1. **原因寻找功能** - 实现CTF的复杂逻辑
2. **Trade-off检测** - 自动识别性能-能耗权衡
3. **白名单工具** - ATE结果集成到现有流程
4. **完整文档** - 技术文档和使用指南

#### ⚠️ 风险提示
- **复杂度高**: 原因寻找功能有280-330行复杂代码
- **调试困难**: 可能出现难以追踪的逻辑错误
- **时间风险**: 可能需要10-15天而非10天
- **收益不确定**: 原因寻找可能发现很少的trade-off

#### 📈 成功标准
- **功能成功**: 原因寻找功能可用
- **发现成功**: 检测到≥5个有意义的trade-off
- **质量成功**: 所有功能稳定运行
- **文档成功**: 完整的技术文档

#### 🚨 降级方案
如果迭代3过于复杂，可以：
1. **保留V1.0** - 作为最终版本
2. **简化原因寻找** - 只做简单的ATE sign检测
3. **手动分析** - 用V1.0的结果进行人工分析

---

## 📅 实施时间线

### 总体时间安排

| 阶段 | 时间 | 累计 | 里程碑 |
|------|------|------|--------|
| **准备阶段** | 1-2天 | 1-2天 | 环境准备，代码理解 |
| **迭代1** | 5-7天 | 6-9天 | MVP完成并验证 |
| **决策点1** | 1天 | 7-10天 | 决定是否继续 |
| **迭代2** | 5天 | 12-15天 | V1.0完成 |
| **决策点2** | 1天 | 13-16天 | 决定是否迭代3 |
| **迭代3** | 10天 | 23-26天 | V2.0完成（可选） |
| **总时长** | | **10-26天** | |

### 关键里程碑

1. **第7天**: MVP原型验证
   - 输出：可工作的ATE计算
   - 决策：是否继续实施

2. **第14天**: V1.0版本发布
   - 输出：增强的ATE计算
   - 决策：是否实现完整功能

3. **第24天**: V2.0版本（可选）
   - 输出：完整的trade-off检测
   - 最终交付

### 缓冲时间考虑

- **乐观情况**: 10-15天（跳过迭代3）
- **中等情况**: 15-20天（简化迭代3）
- **悲观情况**: 20-26天（完整迭代3）

---

## 🛡️ 风险管理

### 主要风险及缓解

| 风险 | 等级 | 缓解方案 |
|------|------|----------|
| **技术风险**: CTF逻辑不适用于能耗 | 🟡 中 | MVP快速验证，不行则调整 |
| **时间风险**: 实际时间比计划长 | 🟡 中 | 设置缓冲时间，准备降级方案 |
| **质量风险**: ATE结果不可靠 | 🟡 中 | 加强验证，使用多种检查方法 |
| **范围风险**: 迭代3过于复杂 | 🔴 高 | 明确边界，准备简化方案 |

### 风险触发器

**立即停止**:
- MVP验证失败（显著率<10%）
- 出现无法修复的技术错误
- 预算或时间严重超支

**考虑降级**:
- 迭代2质量不佳
- 迭代3复杂度超出预期
- 业务优先级变化

### 应急预案

1. **技术降级**: 回到现有DML方法
2. **功能降级**: 只保留ATE计算，放弃原因寻找
3. **时间降级**: 缩短迭代3，只做核心功能
4. **目标降级**: 聚焦问题1，放弃问题2

---

## 📋 后续任务

### 立即开始的任务

1. **环境准备** (1-2天)
   - 激活causal-research环境
   - 准备CTF源码参考
   - 设置开发环境

2. **Phase 0: 理解** (1-2天)
   - 深入理解CTF的compute_ate逻辑
   - 分析现有causal_inference.py代码
   - 识别需要修改的具体位置

3. **Phase 0.5: 原型设计** (1天)
   - 设计MVP的具体实现方案
   - 制定测试计划
   - 准备验证数据

### 依赖关系

```
环境准备 → Phase 0理解 → Phase 0.5设计 → 迭代1开发
    ↓
测试验证 ← 迭代2开发 ← 决策点1
    ↓
迭代3开发 ← 决策点2 ← 迭代2验证
```

### 资源需求

1. **人力资源**: 1人全职开发
2. **计算资源**: GPU环境（现有）
3. **数据资源**: 现有的6组数据
4. **时间资源**: 10-26天（视迭代3而定）

---

## 📝 附录

### 参考文件

1. **评估报告**: ATE_INTEGRATION_FEASIBILITY_REPORT_20260125.md
2. **CTF源码**: `/home/green/energy_dl/nightly/analysis/CTF_original/src/inf.py`
3. **当前实现**:
   - `/home/green/energy_dl/nightly/analysis/utils/causal_inference.py`
   - `/home/green/energy_dl/nightly/analysis/utils/tradeoff_detection.py`

### 关键修改列表

| 原方案 | 修改后 | 理由 |
|--------|--------|------|
| 强制使用ref_df | 可选参数（默认不使用） | 能耗研究没有明确baseline |
| T0/T1使用min/max | 使用25/75分位 | 更鲁棒，避免极端值 |
| 15-22天 | 10-26天 | 更实际的时间估算 |
| 必须实现原因寻找 | 降级为可选功能 | 降低复杂度和风险 |
| 完全复现CTF | 混合方案 | 更好的领域适配 |

### 决策检查清单

在关键决策点，检查以下问题：

**迭代1决策点**:
- [ ] ATE计算是否成功运行？
- [ ] 显著率是否≥50%？
- [ ] 与现有方法结果是否一致？
- [ ] 是否有严重的性能问题？

**迭代2决策点**:
- [ ] V1.0是否比MVP有明显改进？
- [ ] 6组数据是否全部成功？
- [ ] ref_df和T0/T1是否正常工作？
- [ ] 是否有资源继续迭代3？

---

## 📝 第2部分：详细实施步骤

### 迭代1详细计划 (MVP: 5-7天)

#### Day 1-2: 环境准备和代码理解

**任务1.1: 环境准备**
```bash
# 1. 激活causal-research环境
conda activate causal-research

# 2. 验证依赖
python -c "import networkx; print('NetworkX:', networkx.__version__)"
python -c "from econml import dml; print('econml OK')"

# 3. 准备参考文件
ln -s ~/energy_dl/nightly/analysis/CTF_original ~/temp_ctf_ref
```

**任务1.2: 代码理解**
- 阅读CTF的`compute_ate`函数（78-100行）
- 理解`predecessors()`的作用
- 分析当前`causal_inference.py`的`_identify_confounders`方法
- 列出需要修改的所有函数和位置

**输出**:
- 代码理解笔记（1-2页）
- 修改清单（Excel或Markdown表格）

#### Day 3-4: 核心代码重构

**任务1.3: 重构混淆因素识别**

在`causal_inference.py`中添加新函数：

```python
def _get_confounders_from_graph(self,
                                 parent: str,
                                 child: str,
                                 causal_graph: nx.DiGraph) -> List[str]:
    """
    使用CTF方法识别混淆因素：使用因果图的predecessors

    参数:
        parent: 父节点（treatment变量）
        child: 子节点（outcome变量）
        causal_graph: NetworkX有向图

    返回:
        confounders: 混淆因素列表

    注意:
        - CTF使用dg.predecessors()获取所有父节点
        - 必须移除parent自身（CTF第86-87行）
    """
    if not isinstance(causal_graph, nx.DiGraph):
        raise ValueError("causal_graph必须是NetworkX.DiGraph类型")

    # 1. 获取parent的所有父节点
    parent_parents = list(causal_graph.predecessors(parent))

    # 2. 获取child的所有父节点
    child_parents = list(causal_graph.predecessors(child))

    # 3. 合并并去重
    X_cols = list(set(parent_parents + child_parents))

    # 4. 移除parent自身（CTF关键逻辑）
    if parent in X_cols:
        X_cols.remove(parent)

    # 5. 可选：移除已知的控制变量
    # 例如：if "EGR" in X_cols: X_cols.remove("EGR")

    return X_cols
```

**任务1.4: 重构ATE计算**

修改现有的`_compute_ate_dml`方法，添加`use_ctf_style`参数：

```python
def _compute_ate_dml(self,
                     data: pd.DataFrame,
                     treatment: str,
                     outcome: str,
                     confounders: List[str] = None,
                     causal_graph: nx.DiGraph = None,
                     use_ctf_style: bool = True) -> Tuple[float, Tuple[float, float]]:
    """
    使用双重机器学习计算ATE

    参数:
        data: 数据框
        treatment: 处理变量
        outcome: 结果变量
        confounders: 混淆因素列表（可选）
        causal_graph: 因果图（可选，用于CTF风格）
        use_ctf_style: 是否使用CTF风格的混淆因素识别

    返回:
        (ate, (ci_lower, ci_upper))
    """
    # 1. 识别混淆因素
    if use_ctf_style and causal_graph is not None:
        # 使用CTF方法
        confounders = self._get_confounders_from_graph(
            treatment, outcome, causal_graph
        )
    elif confounders is None:
        # 旧方法：向后兼容
        confounders = self._identify_confounders(treatment, outcome, causal_graph)

    # 验证混淆因素不包含outcome
    if outcome in confounders:
        raise ValueError(f"混淆因素不应包含outcome变量: {outcome}")

    # 2. 准备数据
    X = data[confounders]
    T = data[treatment]
    Y = data[outcome]

    # 3. 拟合DML模型
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, random_state=42),
        random_state=42
    )

    try:
        est.fit(Y, T, X=X)
    except Exception as e:
        print(f"DML拟合失败: {e}")
        return 0.0, (0.0, 0.0)

    # 4. 计算ATE（MVP版本：不使用ref_df和T0/T1）
    ate = est.ate(X=X)

    # 5. 计算置信区间
    # 注意：econml的ate()方法不直接返回CI，需要使用effect()方法
    effects = est.effect(X=X)
    ci_lower = np.percentile(effects, 2.5)
    ci_upper = np.percentile(effects, 97.5)

    return float(ate), (float(ci_lower), float(ci_upper))
```

**任务1.5: 更新analyze_all_edges方法**

修改`analyze_all_edges`以支持CTF风格：

```python
def analyze_all_edges(self,
                     data: pd.DataFrame,
                     causal_graph: nx.DiGraph,
                     var_names: List[str],
                     threshold: float = 0.3,
                     use_ctf_style: bool = True) -> Dict[str, Dict]:
    """
    对因果图中的所有边进行因果推断

    新增参数:
        use_ctf_style: 是否使用CTF风格的混淆因素识别
    """
    # ... 现有代码 ...

    for idx, (source_idx, target_idx, weight) in enumerate(edges, 1):
        source = var_names[source_idx]
        target = var_names[target_idx]

        # 使用CTF风格
        ate, (ci_lower, ci_upper) = self._compute_ate_dml(
            data=data,
            treatment=source,
            outcome=target,
            confounders=None,  # 自动识别
            causal_graph=causal_graph,
            use_ctf_style=use_ctf_style
        )

        # ... 存储结果 ...
```

#### Day 5-6: 测试和验证

**任务1.6: 创建测试脚本**

创建`tests/test_ctf_style_ate.py`:

```python
import pandas as pd
import numpy as np
import networkx as nx
from sys import path
path.append('~/energy_dl/nightly/analysis')
from utils.causal_inference import CausalInference

def test_mvp_basic():
    """测试MVP基本功能"""
    # 1. 加载测试数据（Group1）
    data_path = "~/energy_dl/nightly/analysis/results/energy_research/data/interaction/group1_data.csv"
    df = pd.read_csv(data_path)

    # 2. 加载因果图
    graph_path = "~/energy_dl/nightly/analysis/results/energy_research/causal_graphs/group1_graph.gml"
    dg = nx.read_gml(graph_path)

    # 3. 初始化推断器
    ci = CausalInference(verbose=True)

    # 4. 测试单条边
    ate, ci_vals = ci._compute_ate_dml(
        data=df,
        treatment='learning_rate',
        outcome='energy_gpu_min_watts',
        causal_graph=dg,
        use_ctf_style=True
    )

    print(f"ATE: {ate:.4f}, 95% CI: [{ci_vals[0]:.4f}, {ci_vals[1]:.4f}]")

    # 5. 验证混淆因素
    confounders = ci._get_confounders_from_graph(
        'learning_rate', 'energy_gpu_min_watts', dg
    )
    print(f"混淆因素: {confounders}")

    assert 'energy_gpu_min_watts' not in confounders, "错误：混淆因素包含outcome"

if __name__ == "__main__":
    test_mvp_basic()
```

**任务1.7: Group1完整验证**

```python
def test_group1_all_edges():
    """测试Group1的所有边"""
    # 加载白名单边
    whitelist_path = "~/energy_dl/nightly/analysis/results/energy_research/data/interaction/whitelist/group1_*.csv"
    # ... 读取白名单 ...

    # 对所有边计算ATE
    results = ci.analyze_all_edges(
        data=df,
        causal_graph=dg,
        var_names=list(df.columns),
        use_ctf_style=True
    )

    # 统计显著率
    significant = sum(1 for r in results.values() if r['is_significant'])
    significance_rate = significant / len(results) * 100

    print(f"\n显著率: {significance_rate:.1f}% ({significant}/{len(results)})")

    # 成功标准：≥50%
    assert significance_rate >= 50, f"显著率过低: {significance_rate:.1f}%"
```

#### Day 7: 决策和文档

**任务1.8: MVP评估报告**

生成MVP验证报告，包含：
- ATE计算结果汇总
- 显著率统计
- 与现有方法对比
- 性能指标（运行时间）
- 问题列表和解决方案
- 是否继续迭代2的决策建议

---

### 迭代2详细计划 (V1.0: +5天)

#### Day 8-9: 添加ref_df和T0/T1支持

**任务2.1: 实现ref_df构建策略**

```python
def build_reference_df(self,
                       data: pd.DataFrame,
                       strategy: str = "non_parallel") -> pd.DataFrame:
    """
    构建参考数据集（ref_df）

    参数:
        data: 原始数据
        strategy: 构建策略
            - "non_parallel": 使用非并行模式作为baseline
            - "mean": 全局均值
            - "group_mean": 按模型分组均值

    返回:
        ref_df: 参考数据集

    注意:
        - 对于能耗研究，推荐使用"non_parallel"策略
        - CTF使用normal_df.groupby().mean()
    """
    if strategy == "non_parallel":
        # 使用非并行模式
        if 'is_parallel' in data.columns:
            ref_df = data[data['is_parallel'] == 0].copy()
        else:
            print("警告：is_parallel列不存在，使用全部数据")
            ref_df = data.copy()

    elif strategy == "mean":
        # 全局均值（单行）
        ref_df = data.mean().to_frame().T

    elif strategy == "group_mean":
        # 按模型分组
        if 'model_name' in data.columns:
            ref_df = data.groupby('model_name').mean().reset_index()
        else:
            ref_df = data.copy()

    else:
        ref_df = data.copy()

    return ref_df
```

**任务2.2: 实现T0/T1策略**

```python
def compute_T0_T1(self,
                  data: pd.DataFrame,
                  treatment: str,
                  strategy: str = "quantile") -> Tuple[float, float]:
    """
    计算T0和T1值

    参数:
        data: 数据集
        treatment: 处理变量名
        strategy: 策略
            - "quantile": 25/75分位（推荐）
            - "min_max": 最小/最大值
            - "mean_std": 均值±标准差

    返回:
        (T0, T1): 基准和处理值
    """
    if strategy == "quantile":
        T0 = data[treatment].quantile(0.25)
        T1 = data[treatment].quantile(0.75)

    elif strategy == "min_max":
        T0 = data[treatment].min()
        T1 = data[treatment].max()

    elif strategy == "mean_std":
        mean = data[treatment].mean()
        std = data[treatment].std()
        T0 = mean - std
        T1 = mean + std

    else:
        raise ValueError(f"未知策略: {strategy}")

    # 验证T1 > T0
    if T1 <= T0:
        raise ValueError(f"T1 ({T1}) 必须大于 T0 ({T0})")

    # 检查差值是否过小
    diff = T1 - T0
    threshold = 0.1 * data[treatment].std()
    if diff < threshold:
        warnings.warn(f"T1-T0差值过小: {diff:.4f} < {threshold:.4f}")

    return T0, T1
```

**任务2.3: 重构compute_ate以支持ref_df和T0/T1**

```python
def _compute_ate_dml_v1(self,
                       data: pd.DataFrame,
                       treatment: str,
                       outcome: str,
                       causal_graph: nx.DiGraph,
                       ref_df: pd.DataFrame = None,
                       T0: float = None,
                       T1: float = None,
                       t_strategy: str = "quantile") -> Tuple[float, Tuple[float, float]]:
    """
    增强版ATE计算（V1.0）

    新增参数:
        ref_df: 参考数据集（可选）
        T0: 基准值（可选）
        T1: 处理值（可选）
        t_strategy: T0/T1计算策略
    """
    # 1. 识别混淆因素
    confounders = self._get_confounders_from_graph(treatment, outcome, causal_graph)

    # 2. 准备数据
    X = data[confounders]
    T = data[treatment]
    Y = data[outcome]

    # 3. 准备ref_df（如果未提供）
    if ref_df is None:
        ref_df = data  # 默认使用原始数据

    # 4. 计算T0/T1（如果未提供）
    if T0 is None or T1 is None:
        T0, T1 = self.compute_T0_T1(data, treatment, strategy=t_strategy)

    # 5. 拟合DML模型
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, random_state=42),
        random_state=42
    )
    est.fit(Y, T, X=X)

    # 6. 计算ATE（使用ref_df和T0/T1）
    ate = est.ate(X=ref_df[confounders], T0=T0, T1=T1)

    # 7. 计算置信区间
    effects = est.effect(X=ref_df[confounders], T0=T0, T1=T1)
    ci_lower = np.percentile(effects, 2.5)
    ci_upper = np.percentile(effects, 97.5)

    return float(ate), (float(ci_lower), float(ci_upper))
```

#### Day 10-11: 扩展验证

**任务2.4: 6组数据完整测试**

创建`tests/test_v1_all_groups.py`:

```python
def test_all_6_groups():
    """测试所有6组数据"""
    groups = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6']
    results_summary = {}

    for group in groups:
        print(f"\n{'='*60}")
        print(f"测试 {group}")
        print(f"{'='*60}")

        # 加载数据
        data_path = f"~/energy_dl/nightly/analysis/results/energy_research/data/interaction/{group}_data.csv"
        df = pd.read_csv(data_path)

        # 构建ref_df
        ref_df = ci.build_reference_df(df, strategy="non_parallel")

        # 测试所有边
        results = ci.analyze_all_edges_v1(
            data=df,
            causal_graph=dg,
            var_names=list(df.columns),
            ref_df=ref_df,
            t_strategy="quantile"
        )

        # 统计
        significant = sum(1 for r in results.values() if r['is_significant'])
        significance_rate = significant / len(results) * 100
        results_summary[group] = {
            'total': len(results),
            'significant': significant,
            'rate': significance_rate
        }

        print(f"显著率: {significance_rate:.1f}%")

    # 打印汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    for group, stats in results_summary.items():
        print(f"{group}: {stats['significant']}/{stats['total']} ({stats['rate']:.1f}%)")
```

#### Day 12: 对比和决策

**任务2.5: MVP vs V1.0对比**

创建对比报告：
- 显著率对比
- ATE值对比
- 运行时间对比
- 改进评估

**任务2.6: 迭代3决策**

基于对比结果决定是否继续迭代3。

---

### 迭代3详细计划 (V2.0: +10天，可选)

#### Day 13-17: 原因寻找功能

**任务3.1: 研究CTF的原因寻找逻辑**

仔细阅读CTF的280-330行，理解：
- 共同祖先分析
- 路径查找逻辑
- Trade-off判断条件

**任务3.2: 实现简化版原因寻找**

```python
def find_tradeoff_causes(self,
                        parent: str,
                        child: str,
                        data: pd.DataFrame,
                        causal_graph: nx.DiGraph,
                        ate_threshold: float = 0.1) -> Dict:
    """
    寻找trade-off的根本原因

    参数:
        parent: trade-off中的父节点
        child: trade-off中的子节点
        data: 数据集
        causal_graph: 因果图
        ate_threshold: ATE阈值

    返回:
        {
            'has_tradeoff': bool,
            'causes': List[Dict],
            'explanation': str
        }
    """
    results = {
        'has_tradeoff': False,
        'causes': [],
        'explanation': ''
    }

    # 1. 计算ATE
    ate, (ci_low, ci_high) = self._compute_ate_dml_v1(
        data, parent, child, causal_graph
    )

    # 2. 判断是否存在trade-off（简化版：只看ATE符号）
    # 完整版需要考虑多个路径

    # 3. 寻找共同祖先
    common_ancestors = self._find_common_ancestors(parent, child, causal_graph)

    # 4. 对每个共同祖先计算ATE
    for ancestor in common_ancestors:
        ate_anc, _ = self._compute_ate_dml_v1(
            data, ancestor, parent, causal_graph
        )
        ate_anc_child, _ = self._compute_ate_dml_v1(
            data, ancestor, child, causal_graph
        )

        # 判断是否是原因
        if abs(ate_anc) > ate_threshold and abs(ate_anc_child) > ate_threshold:
            results['causes'].append({
                'ancestor': ancestor,
                'ate_to_parent': ate_anc,
                'ate_to_child': ate_anc_child
            })

    results['has_tradeoff'] = len(results['causes']) > 0

    return results
```

#### Day 18-20: 白名单ATE集成

**任务3.3: 创建白名单ATE工具**

创建`analysis/scripts/add_ate_to_whitelist.py`:

```python
#!/usr/bin/env python3
"""
为所有白名单边计算ATE并添加到白名单CSV中
"""
import pandas as pd
import sys
sys.path.append('~/energy_dl/nightly/analysis')
from utils.causal_inference import CausalInference

def main():
    # 1. 加载白名单
    whitelist_path = "analysis/results/energy_research/data/interaction/whitelist/"
    # ... 读取所有白名单CSV ...

    # 2. 初始化推断器
    ci = CausalInference(verbose=True)

    # 3. 为每条边计算ATE
    for edge in whitelist_edges:
        ate, ci_vals = ci._compute_ate_dml_v1(
            data=df,
            treatment=edge['source'],
            outcome=edge['target'],
            causal_graph=dg,
            ref_df=ref_df,
            t_strategy="quantile"
        )

        edge['ate'] = ate
        edge['ate_ci_lower'] = ci_vals[0]
        edge['ate_ci_upper'] = ci_vals[1]
        edge['ate_significant'] = abs(ate) > 0.1

    # 4. 保存更新后的白名单
    output_path = whitelist_path + "with_ate/"
    # ... 保存 ...

if __name__ == "__main__":
    main()
```

#### Day 21-22: 文档和收尾

**任务3.4: 编写技术文档**

生成完整的技术文档：
- 实施总结
- 代码架构
- 使用指南
- 案例研究

---

## 🛠️ 第3部分：技术实现细节

### 代码修改清单

| 文件 | 函数 | 修改类型 | 说明 |
|------|------|----------|------|
| `causal_inference.py` | `_get_confounders_from_graph` | 新增 | CTF风格的混淆因素识别 |
| `causal_inference.py` | `_compute_ate_dml` | 修改 | 添加use_ctf_style参数 |
| `causal_inference.py` | `build_reference_df` | 新增 | ref_df构建策略 |
| `causal_inference.py` | `compute_T0_T1` | 新增 | T0/T1计算策略 |
| `causal_inference.py` | `_compute_ate_dml_v1` | 新增 | 增强版ATE计算 |
| `causal_inference.py` | `find_tradeoff_causes` | 新增 | 原因寻找功能（可选） |
| `tradeoff_detection.py` | `detect_tradeoffs` | 修改 | 集成CTF风格ATE |
| `tests/test_ctf_style_ate.py` | `test_mvp_basic` | 新增 | MVP测试 |
| `tests/test_v1_all_groups.py` | `test_all_6_groups` | 新增 | V1.0测试 |
| `scripts/add_ate_to_whitelist.py` | `main` | 新增 | 白名单ATE工具 |

### 关键代码对比

#### CTF vs 当前实现

| 功能 | CTF实现 | 当前实现 | 修改后 |
|------|---------|----------|--------|
| 混淆因素识别 | `dg.predecessors()` | 图遍历 | `predecessors()` ✅ |
| ATE计算 | `est.ate(X=ref_df, T0, T1)` | `est.ate(X=X)` | 支持两者 ✅ |
| ref_df | `normal_df.groupby()` | ❌ 无 | 可选支持 ✅ |
| T0/T1 | 分组均值 | ❌ 无 | 25/75分位 ✅ |
| 原因寻找 | 共同祖先分析 | 简单sign检测 | 简化版 ✅ |

### 测试验证计划

#### 单元测试

1. **test_confounders.py**: 测试混淆因素识别
   - 验证predecessors()正确性
   - 验证parent被移除
   - 验证outcome不在混淆因素中

2. **test_ref_df.py**: 测试ref_df构建
   - 测试非并行模式策略
   - 测试均值策略
   - 验证数据完整性

3. **test_T0_T1.py**: 测试T0/T1计算
   - 测试分位策略
   - 测试min/max策略
   - 验证T1 > T0

#### 集成测试

1. **test_group1.py**: Group1完整测试
2. **test_all_groups.py**: 6组数据测试
3. **test_whitelist.py**: 白名单ATE测试

#### 数据验证清单

- [ ] ATE值不为NaN
- [ ] ATE值不为无穷大
- [ ] 置信区间合理（下限<上限）
- [ ] 显著率在合理范围（30-80%）
- [ ] 混淆因素不包含outcome
- [ ] T1 > T0
- [ ] ref_df不为空

---

## 📚 参考文件清单

### CTF源码
- **主文件**: `/home/green/energy_dl/nightly/analysis/CTF_original/src/inf.py`
  - `compute_ate`: 第78-100行
  - 原因寻找: 第280-330行
  - ref_df构建: 第122-126行

### 当前代码
- **主模块**: `/home/green/energy_dl/nightly/analysis/utils/causal_inference.py`
  - 当前行数: ~400行
  - 主要函数: `_compute_ate_dml`, `analyze_all_edges`, `_identify_confounders`

- **Trade-off检测**: `/home/green/energy_dl/nightly/analysis/utils/tradeoff_detection.py`
  - 当前行数: ~400行
  - 主要函数: `detect_tradeoffs`, `find_conflicting_effects`

### 数据文件
- **白名单**: `/home/green/energy_dl/nightly/analysis/results/energy_research/data/interaction/whitelist/`
  - `group1_examples_causal_edges_whitelist.csv`
  - `group2_*_whitelist.csv`
  - ... 共227条边

- **6组数据**: `/home/green/energy_dl/nightly/analysis/results/energy_research/data/interaction/`
  - `group1_data.csv`
  - `group2_data.csv`
  - ... `group6_data.csv`

### 文档参考
- **评估报告**: `/home/green/energy_dl/nightly/docs/current_plans/ATE_INTEGRATION_FEASIBILITY_REPORT_20260125.md`
- **完整参考**: `/home/green/energy_dl/nightly/docs/CLAUDE_FULL_REFERENCE.md`

---

**文档制定**: 2026-01-25
**最后更新**: 2026-01-25
**版本**: v1.0
**状态**: ⏳ 等待实施批准