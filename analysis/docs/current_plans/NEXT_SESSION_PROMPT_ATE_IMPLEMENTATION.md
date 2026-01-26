# 下一个对话的启动Prompt：ATE集成代码实现

**用途**: 在下一个对话中快速启动代码实现工作
**创建日期**: 2026-01-25
**相关文档**: ATE_INTEGRATION_COMPLETE_PLAN_20260125.md

---

## 🚀 任务概述

我们需要实现将CTF论文的ATE（Average Treatment Effect）计算方法集成到我们的因果分析流程中，以支持RQ2的trade-off分析。

**核心目标**:
1. 重构`CausalInferenceEngine.estimate_ate()`函数，添加CTF兼容模式
2. 实现ref_df和T0/T1支持
3. 实现混淆因素自动识别
4. 创建白名单扩展工具
5. 实现原因寻找算法

**预计工期**: 15-22天（分3个Phase）

---

## 📂 关键文件路径

### 需要修改的文件

```
utils/causal_inference.py              # 重构ATE计算引擎（P0）
utils/tradeoff_detection.py            # 扩展Trade-off检测（P1）
tools/data_management/add_ate_to_whitelist.py  # 新建（P1）
```

### 参考文件

```
CTF_original/src/inf.py                # CTF原版ATE计算（78-97行）
CTF_original/src/inf.py                # CTF原版原因寻找（280-330行）
results/energy_research/data/interaction/whitelist/*.csv  # 白名单数据
```

### 测试文件

```
tests/test_ate_calculation.py          # 新建
tests/test_confounder_identification.py # 新建
tests/test_tradeoff_detection.py       # 扩展
tests/benchmark/test_performance.py    # 新建
```

---

## 🎯 实施优先级

### Phase 0: 准备工作（必须先完成，2-3天）

**目标**: 确认CTF逻辑，避免走弯路

**任务清单**:
1. [ ] 阅读并理解`CTF_original/src/inf.py`的关键函数
   - `compute_ate()` (78-97行) - ATE计算逻辑
   - `read_data()` (40-77行) - 数据加载和ref_df构建
   - Trade-off检测主逻辑 (150-330行)
   
2. [ ] 确认关键细节
   - ref_df是如何构建的？
   - T0和T1是如何选择的？
   - 混淆因素是如何识别的？
   
3. [ ] 创建测试框架
   - 性能基准测试模板
   - 合成数据生成器

**重要提示**: ⚠️ 不要跳过这个阶段！错误的理解会导致后续大量返工。

---

### Phase 1: P0核心功能（5-7天）

#### 1.1 重构`CausalInferenceEngine.estimate_ate()`

**当前签名**:
```python
def estimate_ate(self, data, treatment, outcome, confounders, controls=None):
    # 返回 (ate, (ci_lower, ci_upper))
```

**目标签名**:
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
    返回结构化字典:
    {
        'ate': float,
        'ci_lower': float,
        'ci_upper': float,
        'is_significant': bool,
        'T0': float,
        'T1': float,
        'ref_mean': float,
        'method': str,
        'confounders': List[str],
        'n_samples': int
    }
    """
```

**关键修改点**:
1. 添加`ref_df`, `T0`, `T1`, `mode`参数
2. 返回结构化字典而非tuple
3. 实现mode='ctf'的逻辑（使用RandomForest）
4. 支持自动混淆因素识别

#### 1.2 实现辅助函数

**新建文件**: `utils/ref_df_builder.py`
```python
def build_reference_df(data: pd.DataFrame,
                      groupby_columns: List[str],
                      agg_method: str = 'mean') -> pd.DataFrame:
    """
    构建参考数据集
    
    ⚠️ 需要先确认CTF的ref_df构建逻辑
    """
```

**新建文件**: `utils/confounder_identifier.py`
```python
def identify_confounders_from_graph(treatment: str,
                                   outcome: str,
                                   causal_graph: nx.DiGraph) -> Tuple[List[str], List[str]]:
    """
    从因果图识别混淆因素
    
    返回:
        confounders: 同时影响treatment和outcome
        controls: 只影响treatment
    """
```

**新建文件**: `utils/treatment_level_selector.py`
```python
def select_treatment_levels(data: pd.DataFrame,
                           treatment: str,
                           strategy: str = 'minmax') -> Tuple[float, float]:
    """
    选择T0和T1
    
    策略: 'minmax' | 'quantile' | 'mean_std'
    """
```

#### 1.3 错误处理和验证

添加数据验证逻辑：
- 检查数据完整性
- 验证因果图结构
- 处理缺失值
- 降级策略（DML失败时使用简化方法）

---

### Phase 2: P1扩展功能（5-7天）

#### 2.1 实现原因寻找算法

**修改**: `utils/tradeoff_detection.py`

**添加方法**:
```python
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
    
    实现CTF的逻辑：
    1. 找common ancestors
    2. 分析路径依赖
    3. 对每个潜在原因计算ATE
    4. 判断是否也产生trade-off
    """
```

#### 2.2 白名单扩展工具

**新建**: `tools/data_management/add_ate_to_whitelist.py`

```python
def add_ate_to_whitelist(whitelist_path: str,
                        data: pd.DataFrame,
                        causal_graph: nx.DiGraph,
                        mode: str = 'ctf') -> pd.DataFrame:
    """
    为白名单添加ATE列
    
    新增8列:
    - ate, ate_ci_lower, ate_ci_upper, ate_is_significant
    - T0, T1, ref_mean, ate_method
    """
```

**批量处理脚本**:
```python
def process_all_whitelists(data_path: str,
                          data: pd.DataFrame,
                          causal_graph: nx.DiGraph):
    """
    批量处理所有白名单文件
    """
```

#### 2.3 性能优化

- 实现ATE结果缓存
- 添加进度条（tqdm）
- 支持并行计算（joblib）

---

### Phase 3: 测试和验证（3-5天）

#### 3.1 单元测试

**新建**: `tests/test_ate_calculation.py`

```python
def test_ate_basic():
    """基本ATE计算"""
    
def test_ate_ctf_mode():
    """CTF模式测试"""
    
def test_confounder_identification():
    """混淆因素识别测试"""
    
def test_ref_df_building():
    """ref_df构建测试"""
```

#### 3.2 集成测试

**新建**: `tests/test_integration.py`

```python
def test_ctf_alignment():
    """与CTF对齐验证"""
    
def test_end_to_end():
    """端到端测试"""
```

#### 3.3 性能测试

**新建**: `tests/benchmark/test_performance.py`

```python
def test_ate_performance():
    """性能基准测试"""
    # 单条边 < 1s
```

---

## ⚠️ 关键风险和注意事项

### P0风险（必须处理）

1. **ref_df构建方式不明确**
   - 行动: 先阅读CTF的load_data.py
   - 验证: 使用CTF相同的数据测试
   
2. **T0/T1选择策略不确定**
   - 行动: 实现多种策略供选择
   - 默认: 使用min/max，但提供其他选项
   
3. **混淆因素识别可能错误**
   - 行动: 实现验证函数
   - 测试: 与CTF结果对比

### 实施建议

1. **分步验证**: 每完成一个功能立即测试
2. **保持简单**: 先实现基本功能，再优化
3. **记录决策**: 在代码注释中记录设计决策
4. **版本控制**: 每个Phase完成后打tag

---

## 📊 验收标准

### Phase 0验收

- [ ] CTF源码关键逻辑已理解
- [ ] ref_df构建方式已确认
- [ ] 测试框架已建立

### Phase 1验收

- [ ] estimate_ate()支持mode='ctf'
- [ ] 自动混淆因素识别正常工作
- [ ] 单元测试覆盖率 > 80%
- [ ] 通过基本功能测试

### Phase 2验收

- [ ] 原因寻找算法实现
- [ ] 白名单扩展工具可用
- [ ] 批量处理脚本完成

### Phase 3验收

- [ ] 与CTF结果相关系数 > 0.95
- [ ] 单条边ATE计算 < 1s
- [ ] 所有测试通过
- [ ] 文档完整

---

## 🔗 相关文档

### 必读文档

1. `ATE_INTEGRATION_COMPLETE_PLAN_20260125.md` - 完整技术方案
2. `STAGE4_PEER_REVIEW_REPORT_20260125.md` - 风险评估
3. `CTF_SOURCE_CODE_COMPARISON_20260125.md` - 代码对比

### 参考文档

4. `CAUSAL_EDGE_WHITELIST_DESIGN.md` - 白名单设计
5. `CLAUDE_FULL_REFERENCE.md` - 项目参考
6. `docs/technical_reference/DATA_USAGE_GUIDE.md` - 数据使用指南

---

## 🚀 快速开始

### 第一步：理解CTF逻辑

```bash
# 阅读CTF关键代码
cat CTF_original/src/inf.py | less

# 重点查看：
# - compute_ate函数（78-97行）
# - read_data函数（40-77行）
# - Trade-off检测（280-330行）
```

### 第二步：设置开发环境

```bash
# 激活causal-research环境
conda activate causal-research

# 验证依赖
python -c "import econml; print(econml.__version__)"
python -c "import networkx; print(networkx.__version__)"
```

### 第三步：创建测试框架

```bash
# 创建测试目录
mkdir -p tests/benchmark

# 创建合成数据生成器
# tests/fixtures.py
```

### 第四步：开始实现

按照Phase 0 → Phase 1 → Phase 2 → Phase 3的顺序实施。

---

## 💡 代码模板

### CausalInferenceEngine重构模板

见`ATE_INTEGRATION_COMPLETE_PLAN_20260125.md`的"方案1: ATE计算函数扩展"章节。

### 辅助函数模板

见`ATE_INTEGRATION_COMPLETE_PLAN_20260125.md`的"方案2/3"章节。

---

## 📞 获取帮助

如遇到问题：
1. 查阅相关文档
2. 查看CTF原代码
3. 检查测试用例

---

**文档结束**

**创建时间**: 2026-01-25
**预计完成**: 2026-02-15（3周后）
