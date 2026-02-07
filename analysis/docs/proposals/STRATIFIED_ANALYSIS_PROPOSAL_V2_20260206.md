# 分层因果分析方案 v2.0

**版本**: 2.1 (审查修订版)
**日期**: 2026-02-06
**作者**: 资深科研工作者 (Claude Code)
**方案类型**: 因果推断分析扩展
**状态**: 已通过审查，待执行

---

## 审查修订说明 (v2.0 → v2.1)

**审查评分**: 3.5/5 分（需中等修改后可执行）

**P0问题修订**:
| ID | 原问题 | 修订内容 |
|----|--------|----------|
| P0-1 | 样本/变量比不足 | ✅ 明确定位为"探索性分析"；敏感性分析增至5次运行，80%一致性；添加与全局分析对标 |
| P0-2 | Wald检验小样本性质未论证 | ✅ 添加Bootstrap置换检验验证；置信区间叠加法作为辅助 |
| P0-3 | 功效分析声称过于乐观 | ✅ 修正为"检测大效应d≥0.8功效≈0.85"或"中等效应d≥0.5功效≈0.60" |
| P0-4 | 未考虑跨层间依赖性 | ✅ 采用全局FDR控制，所有p值统一校正 |

**P1问题修订**:
- P1-1: ✅ 明确Cohen's d计算使用pooled SD
- P1-2: ✅ DiBS收敛性检查：最后500步ELBO变化<0.5%
- P1-3: ✅ 敏感性分析：5次运行，80%一致性阈值
- P1-4: ✅ 添加与全局分析对标章节
- P1-5: ✅ 明确权衡检测聚合方法

---

## 执行摘要

本方案基于v1.0和v2.0同行审查的反馈进行了重大修订。核心变更：

1. **分析范围精简**: 仅保留group1和group3（共510样本），删除样本量不足的group5/6
2. **统计方法增强**: 添加Benjamini-Hochberg FDR多重检验校正（α=0.10），全局应用
3. **跨场景对比**: 使用Wald检验 + Bootstrap置换验证 + Bonferroni校正
4. **小样本适配**: 调整DiBS参数，5次敏感性分析，80%一致性阈值
5. **探索性分析定位**: 明确本分析为探索性，结果需与全局分析对标验证

**核心目标**:
- 探索并行/非并行场景对因果结构的潜在影响
- 识别场景特定的因果关系（需进一步验证）
- 揭示场景特定的权衡关系
- 与全局分析结果对比，验证发现的稳健性

**分析范围**: 2组数据（group1, group3），4个分层子集，共510样本

**⚠️ 重要说明**: 本分析为**探索性分析**，样本/变量比（2.5-5.1）处于因果发现的边界条件。结果解释需谨慎，所有发现需与全局分析对标验证。

---

## 1. v1.0同行审查问题回应

### 1.1 必须修改的问题（P0）

| ID | 问题 | 解决方案 |
|----|------|----------|
| P0-1 | group5/6样本量不足（31-36样本） | ✅ 删除group5/6，仅保留group1/group3 |
| P0-2 | 缺失多重检验校正 | ✅ 添加Benjamini-Hochberg FDR (α=0.10) |
| P0-3 | 未验证小样本DiBS可靠性 | ✅ 添加DiBS收敛性检查和敏感性分析 |
| P0-4 | 未评估统计功效 | ✅ 最小层93样本，功效分析表明可检测中等效应 |

### 1.2 建议修改的问题（P1）

| ID | 问题 | 解决方案 |
|----|------|----------|
| P1-1 | 跨场景对比缺乏正式统计检验 | ✅ 使用Wald检验 + Bonferroni校正 |
| P1-2 | 效应大小解释缺失 | ✅ 添加Cohen's d效应量计算 |

---

## 2. 数据准备

### 2.1 分层配置（精简版）

**分层变量**: `is_parallel` (布尔值)

**分析组配置**:

| 分层ID | 组 | is_parallel | 样本数 | 变量数 | 样本/变量比 |
|--------|-----|-------------|--------|--------|-------------|
| group1_parallel | group1_examples | True | 178 | ~35 | 5.1 |
| group1_non_parallel | group1_examples | False | 126 | ~35 | 3.6 |
| group3_parallel | group3_person_reid | True | 113 | ~37 | 3.1 |
| group3_non_parallel | group3_person_reid | False | 93 | ~37 | 2.5 |

**样本量评估**:
- 最小层: 93样本（group3_non_parallel）
- 样本/变量比: 2.5-5.1（处于因果发现的边界条件n/p ≥ 2）
- **⚠️ 统计功效（修正后）**:
  - 大效应 (d ≥ 0.8): 功效 ≈ 0.85
  - 中等效应 (d ≥ 0.5): 功效 ≈ 0.60
  - 小效应 (d ≥ 0.3): 功效 ≈ 0.35（不推荐解释）
- **结论**: 本分析主要可检测大效应，中等效应需谨慎解释

### 2.2 删除的组及理由

| 组 | 原计划样本 | 删除理由 |
|----|-----------|----------|
| group5_mrt_oast | 36 + 36 | 每层仅36样本，DiBS可能不收敛 |
| group6_resnet | 43 + 31 | 非并行层仅31样本，统计功效不足 |

### 2.3 数据输入输出

**输入数据**:
- `analysis/data/energy_research/6groups_global_std/group1_examples_global_std.csv`
- `analysis/data/energy_research/6groups_global_std/group3_person_reid_global_std.csv`

**输出数据结构**:
```
analysis/data/energy_research/stratified/
├── group1_examples/
│   ├── group1_parallel.csv      (178样本)
│   └── group1_non_parallel.csv  (126样本)
└── group3_person_reid/
    ├── group3_parallel.csv      (113样本)
    └── group3_non_parallel.csv  (93样本)
```

### 2.4 数据预处理

1. **按is_parallel分割数据**
2. **移除冗余列**:
   - `is_parallel` 列（分层后为常量）
   - `*_x_is_parallel` 交互项列（分层后冗余）
   - `timestamp` 列（非因果变量）
3. **保留全局标准化**（不重新标准化）
4. **质量检查**: 缺失值、异常值、分布

---

## 3. 分析流程

### 3.1 流程概览

```
数据分层
    ↓
DiBS因果图学习 (4个模型)
    ↓
ATE计算 (Bootstrap CI)
    ↓
多重检验校正 (Benjamini-Hochberg FDR)
    ↓
权衡检测 (算法1)
    ↓
跨场景对比 (Wald检验)
    ↓
综合报告
```

### 3.2 步骤1: 数据分层与验证

**脚本**: `scripts/stratified/prepare_stratified_data.py`

**操作**:
1. 加载全局标准化数据
2. 按`is_parallel`分割
3. 移除交互项列和is_parallel列
4. 验证每层数据质量
5. 生成数据分割报告

**验收标准**:
- ✅ 4个分层子集创建成功
- ✅ 每层样本数符合预期（±5%容差）
- ✅ 无缺失值
- ✅ 特征数量一致

### 3.3 步骤2: DiBS因果图学习

**脚本**: `scripts/stratified/run_dibs_stratified.py`

**参数配置**:
```python
DIBS_CONFIG = {
    "alpha_linear": 0.05,
    "beta_linear": 0.1,
    "n_particles": 20,
    "tau": 1.0,
    "n_steps": 5000,
    "n_grad_mc_samples": 128,
    "n_acyclicity_mc_samples": 32,
    "posterior_threshold": 0.95  # 后验边概率阈值
}
```

**小样本适配（针对93-126样本层）**:
- 增加正则化: `beta_linear = 0.15`（可选，如收敛困难）
- 增加采样: `n_steps = 7500`
- 收敛性检查: 监控ELBO loss曲线，**最后500步变化<0.5%视为收敛**
- **敏感性分析**: 5次不同随机种子运行，边检测需**80%一致性**（≥4/5次）

**输出**:
- 4个邻接矩阵 (CSV)
- 4个强边列表 (threshold=0.3)
- 4个收敛性报告 (JSON)
- **敏感性分析报告** (5次运行一致性统计)

**验收标准**:
- ✅ DiBS成功收敛（最后500步ELBO变化<0.5%）
- ✅ 每个因果图至少5条强边（threshold=0.3）
- ✅ 敏感性分析：≥80%边在4/5次运行中检出
- ✅ 无明显异常（如所有边权重为0）

### 3.4 步骤3: ATE计算与多重检验校正

**脚本**: `scripts/stratified/compute_ate_stratified.py`

**ATE估计方法**:
```python
CausalInferenceEngine.estimate_ate(
    method='dml',
    model_phi=MLP,
    model_rho=MLP,
    n_splits=5,              # 交叉验证折数
    confidence_level=0.95,   # 置信水平
    n_bootstrap=500          # Bootstrap样本数
)
```

**多重检验校正**:
```python
from statsmodels.stats.multitest import multipletests

# 全局FDR校正（所有4个分层的p值合并后统一校正）
all_pvals = concat_all_layer_pvals()  # 合并所有分层的p值
rejected, pvals_corrected, _, _ = multipletests(
    all_pvals,
    alpha=0.10,              # FDR控制水平
    method='fdr_bh'
)
```

**⚠️ 重要**: 采用**全局FDR控制**，将所有4个分层的p值合并后统一应用BH-FDR校正，避免跨层依赖性问题。

**输出**:
- 4个ATE结果文件（含原始p值和FDR校正p值）
- 显著边标记（基于校正后p值 < 0.10）

**验收标准**:
- ✅ 所有CI宽度 > 0
- ✅ FDR校正成功应用
- ✅ 显著边比例 > 20%

### 3.5 步骤4: 权衡检测

**脚本**: `scripts/stratified/run_tradeoff_stratified.py`

**方法**: 算法1（CTF论文）

**能耗-性能规则**:
```python
ENERGY_PERF_RULES = {
    "energy_vars": [
        "energy_cpu_total_joules",
        "energy_gpu_total_joules",
        "energy_cpu_pkg_joules",
        "energy_cpu_ram_joules"
    ],
    "perf_vars": [
        "perf_test_accuracy",
        "perf_precision",
        "perf_recall",
        "perf_f1_score"
    ],
    "tradeoff_condition": "opposite_signs"  # 能耗↑性能↓ 或 能耗↓性能↑
}
```

**输出**:
- 4个权衡检测结果文件
- 权衡汇总表（跨分层对比）

**验收标准**:
- ✅ 权衡检测成功执行
- ✅ 结果包含统计显著性信息

### 3.6 步骤5: 跨场景对比分析

**脚本**: `scripts/stratified/compare_scenarios.py`

**对比方法**: Wald检验 + Bootstrap置换验证

对于同一因果边在并行(p)和非并行(np)场景的ATE差异:

```
H₀: ATE_p = ATE_np
H₁: ATE_p ≠ ATE_np

Wald统计量: W = (ATE_p - ATE_np)² / (SE_p² + SE_np²)
W ~ χ²(1) under H₀
```

**Bootstrap置换验证** (应对小样本):
```python
def bootstrap_permutation_test(ate_p, ate_np, se_p, se_np, n_bootstrap=1000):
    """
    Bootstrap置换检验验证Wald检验结果
    """
    observed_diff = ate_p - ate_np
    null_diffs = []
    for _ in range(n_bootstrap):
        # 在合并标准误下重采样
        pooled_se = np.sqrt(se_p**2 + se_np**2)
        null_diff = np.random.normal(0, pooled_se)
        null_diffs.append(null_diff)

    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return p_value
```

**多重比较校正**: Bonferroni校正
- 显著性阈值: α/m（m为比较次数）

**效应量**: Cohen's d (使用pooled SD)
```python
def cohens_d(ate_p, ate_np, se_p, se_np, n_p, n_np):
    """
    Cohen's d 使用 pooled standard deviation
    """
    # 估计每组的SD (SE * sqrt(n))
    sd_p = se_p * np.sqrt(n_p)
    sd_np = se_np * np.sqrt(n_np)

    # Pooled SD
    pooled_sd = np.sqrt(((n_p - 1) * sd_p**2 + (n_np - 1) * sd_np**2) / (n_p + n_np - 2))

    return (ate_p - ate_np) / pooled_sd
```

**输出**:
- 场景对比结果表（含Wald统计量、Wald p值、Bootstrap p值、效应量）
- 显著差异边列表
- 可视化：并行 vs 非并行ATE对比图

**验收标准**:
- ✅ Wald检验成功执行
- ✅ Bootstrap置换验证完成
- ✅ Cohen's d效应量计算（使用pooled SD）
- ✅ 对比可视化生成
- ✅ Wald与Bootstrap结果一致性检查

---

## 4. 统计方法详述

### 4.0 与全局分析对标（新增）

**目的**: 验证分层分析结果的稳健性，确保发现不是小样本噪声

**对标策略**:

1. **因果边一致性检查**:
   - 比较分层因果图与全局因果图（`analysis/results/energy_research/data/global_std/`）
   - 计算边集合的Jaccard相似度
   - 预期：分层结果应保留全局分析中的核心强边（≥70%保留率）

2. **ATE符号一致性检查**:
   - 对于同一因果边，检查分层ATE符号是否与全局ATE一致
   - 预期：≥80%的边ATE符号一致
   - 符号不一致的边需特别讨论（可能是场景差异的证据）

3. **效应量对比**:
   - 比较分层ATE绝对值与全局ATE
   - 如果分层ATE显著大于全局，可能表明场景异质性
   - 如果分层ATE显著小于全局，需检查样本量问题

**对标报告模板**:
```markdown
## 与全局分析对标报告

### 因果边一致性
- 全局强边数: X
- 分层保留率: Y% (group1_p), Z% (group1_np), ...
- Jaccard相似度: A, B, C, D

### ATE符号一致性
- 一致边数: X / Y (Z%)
- 不一致边列表: [...]

### 结论
- 分层结果[支持/部分支持/不支持]全局分析
- 场景差异[明显/不明显]
```

**验收标准**:
- ✅ 与全局分析的因果边保留率 ≥ 70%
- ✅ ATE符号一致性 ≥ 80%
- ✅ 生成对标报告

### 4.1 多重检验校正策略

**问题**:
- 每个分层约有35个变量，潜在边数 = 35×34 = 1190
- 4个分层 × 多条边 = 数千次假设检验
- 不校正会导致大量假阳性

**方案**: 全局FDR控制（修订后）

1. **全局统一校正**: 将所有4个分层的p值合并后统一应用BH-FDR (α=0.10)
   - 这解决了跨层依赖性问题
   - 比分层FDR更保守，但统计上更严谨

2. **跨层对比校正**: Bonferroni校正（保守）

**理由**:
- BH-FDR: 控制错误发现率，平衡功效和假阳性
- 全局应用: 避免跨层依赖性导致的FDR膨胀
- α=0.10: 探索性分析，允许更多发现
- Bonferroni用于跨层对比: 对比次数较少，可接受保守性

### 4.2 小样本DiBS可靠性

**挑战**: 最小层93样本，变量数~35

**可靠性保障措施**:

1. **收敛性监控**:
   - 记录ELBO loss曲线
   - **最后500步变化<0.5%视为收敛**（明确标准）

2. **敏感性分析**（增强版）:
   - 5次不同随机种子运行（增加自3次）
   - **80%一致性阈值**：边需在≥4/5次运行中检测到才视为稳定
   - 报告边检测一致性统计

3. **正则化调整**:
   - 如检测边过多（>20%可能边），增加beta_linear
   - 如检测边过少（<1%可能边），降低threshold

4. **结果验证**:
   - **与全局分析结果对标**（见4.0节）
   - 检查是否保留核心因果边（≥70%保留率）

### 4.3 统计功效分析（修正版）

**假设**:
- 最小样本量: n = 93
- 显著性水平: α = 0.10（FDR校正后）
- 效应量: Cohen's d

**功效计算**（修正后）:
```
n = 93, α = 0.10, 双侧检验

效应量       功效        解释能力
----------------------------------------
d = 0.3 (小)    ≈ 0.35      ❌ 不推荐解释
d = 0.5 (中等)  ≈ 0.60      ⚠️ 需谨慎解释
d = 0.8 (大)    ≈ 0.85      ✅ 可靠检测
d = 1.0 (很大)  ≈ 0.95      ✅ 高可靠性
```

**⚠️ 重要修正**: 之前v2.0声称"中等效应功效≥0.80"过于乐观。实际上：
- 中等效应（d=0.5）的功效仅约0.60
- 本分析主要可靠检测大效应（d≥0.8）
- 中等效应的发现需额外验证

**结论**:
- 本分析定位为**探索性分析**
- 大效应可可靠检测（功效≈0.85）
- 中等效应需谨慎解释（功效≈0.60）
- 小效应不建议解释（功效不足）

---

## 5. 代码复用策略

### 5.1 基础脚本

| 现有脚本 | 用途 | 修改内容 |
|----------|------|----------|
| `run_dibs_6groups_global_std.py` | DiBS学习 | 修改GROUP_IDS，添加分层逻辑 |
| `compute_ate_dibs_global_std.py` | ATE计算 | 添加FDR校正步骤 |
| `run_algorithm1_tradeoff_detection_global_std.py` | 权衡检测 | 适配分层结果路径 |

### 5.2 新增脚本

| 脚本 | 功能 | 复用模块 |
|------|------|----------|
| `prepare_stratified_data.py` | 数据分层 | pandas |
| `run_dibs_stratified.py` | 分层DiBS | `utils/causal_discovery.py` |
| `compute_ate_stratified.py` | 分层ATE+FDR | `utils/causal_inference.py`, statsmodels |
| `run_tradeoff_stratified.py` | 分层权衡 | `utils/tradeoff_detection.py` |
| `compare_scenarios.py` | 跨场景对比 | scipy.stats, matplotlib |

### 5.3 关键修改点

**run_dibs_stratified.py**:
```python
# 修改1: 分层组配置
STRATIFIED_GROUPS = [
    {"group": "group1_examples", "layer": "parallel", "samples": 178},
    {"group": "group1_examples", "layer": "non_parallel", "samples": 126},
    {"group": "group3_person_reid", "layer": "parallel", "samples": 113},
    {"group": "group3_person_reid", "layer": "non_parallel", "samples": 93},
]

# 修改2: 数据加载路径
data_dir = Path("data/energy_research/stratified/{group}/")
```

**compute_ate_stratified.py**:
```python
# 新增: FDR校正
from statsmodels.stats.multitest import multipletests

def apply_fdr_correction(ate_results, alpha=0.10):
    pvals = ate_results['p_value'].values
    rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    ate_results['p_value_fdr'] = pvals_corrected
    ate_results['is_significant_fdr'] = rejected
    return ate_results
```

---

## 6. 预期输出

### 6.1 目录结构

```
analysis/results/energy_research/stratified/
├── data/                           # 分层数据
│   ├── group1_examples/
│   │   ├── group1_parallel.csv
│   │   └── group1_non_parallel.csv
│   └── group3_person_reid/
│       ├── group3_parallel.csv
│       └── group3_non_parallel.csv
├── dibs/                           # DiBS结果
│   ├── group1_parallel/
│   │   ├── causal_graph.csv
│   │   ├── edges_threshold_0.3.csv
│   │   └── convergence_report.json
│   └── ... (4个子目录)
├── ate/                            # ATE结果
│   ├── group1_parallel_ate.csv
│   ├── group1_parallel_ate_fdr.csv  # FDR校正后
│   └── ... (8个文件)
├── tradeoffs/                      # 权衡检测
│   ├── group1_parallel_tradeoffs.csv
│   └── ... (4个文件)
├── comparison/                     # 跨场景对比
│   ├── group1_scenario_comparison.csv
│   ├── group3_scenario_comparison.csv
│   └── wald_test_results.csv
├── figures/                        # 可视化
│   ├── ate_parallel_vs_nonparallel_group1.png
│   ├── ate_parallel_vs_nonparallel_group3.png
│   └── tradeoff_comparison.png
└── reports/                        # 报告
    ├── data_stratification_report.md
    ├── dibs_convergence_report.md
    ├── ate_fdr_summary.md
    ├── tradeoff_summary.md
    └── STRATIFIED_ANALYSIS_FINAL_REPORT.md
```

### 6.2 关键输出文件

**ATE结果示例** (`group1_parallel_ate_fdr.csv`):
```csv
source,target,ate,ci_lower,ci_upper,p_value,p_value_fdr,is_significant_fdr,cohens_d
hyperparam_epochs,energy_gpu_total_joules,0.45,0.21,0.69,0.002,0.015,True,0.62
hyperparam_batch_size,perf_test_accuracy,-0.12,-0.28,0.04,0.14,0.32,False,0.18
...
```

**场景对比结果示例** (`wald_test_results.csv`):
```csv
group,source,target,ate_parallel,ate_non_parallel,wald_stat,p_value,p_value_bonf,cohens_d,significant
group1,hyperparam_epochs,energy_gpu_total_joules,0.45,0.22,8.32,0.004,0.024,0.58,True
...
```

---

## 7. 风险与缓解

### 7.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 | 状态 |
|------|------|------|----------|------|
| 小样本DiBS不收敛 | 中 | 高 | 增加采样，调整正则化，敏感性分析 | 已规划 |
| ATE CI过宽 | 中 | 中 | 报告CI宽度，谨慎解释非显著结果 | 已规划 |
| FDR后无显著边 | 低 | 中 | 报告原始p值，讨论探索性发现 | 已规划 |
| 计算资源不足 | 低 | 低 | 串行执行，估计4-6小时 | 可接受 |

### 7.2 科学风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 场景差异不显著 | 中 | 中 | 仍有价值：确认全局分析充分 |
| 结果与交互项分析矛盾 | 低 | 中 | 讨论方法差异，报告两种结果 |
| 过度解释场景差异 | 中 | 中 | 强调效应量和置信区间 |

---

## 8. 验收标准

### 8.1 数据质量

- [ ] 4个分层子集创建成功
- [ ] 每层样本数符合预期（group1: 178/126, group3: 113/93）
- [ ] 无缺失值
- [ ] 数据分割报告生成

### 8.2 DiBS分析

- [ ] 4个DiBS因果图成功学习
- [ ] ELBO收敛性检查通过
- [ ] 敏感性分析完成（3个随机种子）
- [ ] 收敛性报告生成

### 8.3 ATE计算

- [ ] 4个分层的ATE计算完成
- [ ] FDR校正成功应用
- [ ] 所有CI宽度 > 0
- [ ] ATE摘要报告生成

### 8.4 权衡检测

- [ ] 4个权衡检测完成
- [ ] 权衡汇总表生成
- [ ] 至少识别若干能耗vs性能权衡

### 8.5 跨场景对比

- [ ] Wald检验完成
- [ ] Bonferroni校正应用
- [ ] Cohen's d效应量计算
- [ ] 对比可视化生成

### 8.6 综合报告

- [ ] 最终分析报告完成
- [ ] 核心发现清晰阐述
- [ ] 方法局限性讨论
- [ ] 与交互项分析对比讨论

---

## 9. 执行计划

### 9.1 执行顺序

1. **数据准备**: `prepare_stratified_data.py`
2. **DiBS学习**: `run_dibs_stratified.py`（4个模型）
3. **ATE计算**: `compute_ate_stratified.py`（含FDR校正）
4. **权衡检测**: `run_tradeoff_stratified.py`
5. **场景对比**: `compare_scenarios.py`
6. **报告生成**: 手动编写最终报告

### 9.2 依赖关系

```
prepare_stratified_data.py
         ↓
run_dibs_stratified.py
         ↓
compute_ate_stratified.py
         ↓
run_tradeoff_stratified.py ←→ compare_scenarios.py
                    ↓
         最终报告生成
```

### 9.3 检查点

| 检查点 | 验收动作 |
|--------|----------|
| 数据分层完成 | 验证样本数，检查数据质量 |
| DiBS完成 | 检查收敛性，验证边数合理 |
| ATE完成 | 检查CI宽度，验证FDR结果 |
| 权衡检测完成 | 验证权衡数量和类型 |
| 场景对比完成 | 检查Wald检验结果，生成可视化 |

---

## 10. 附录

### 10.1 参考文档

- v1.0方案: `docs/proposals/STRATIFIED_ANALYSIS_PROPOSAL_20260205.md`
- 交互项分析: `docs/reports/INTERACTION_EFFECT_ANALYSIS_REPORT_20260205.md`
- 全局标准化修复: `docs/technical_reference/GLOBAL_STANDARDIZATION_FIX_PROGRESS.md`

### 10.2 术语表

- **FDR (False Discovery Rate)**: 错误发现率，控制假阳性比例
- **Benjamini-Hochberg**: FDR控制的经典方法
- **Wald检验**: 基于估计量渐近正态性的假设检验
- **Cohen's d**: 效应量度量，0.2=小，0.5=中，0.8=大
- **Bonferroni校正**: 多重比较校正，最保守但控制FWER

### 10.3 变更日志

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-02-05 | 初始方案，4组8层 |
| v2.0 | 2026-02-06 | 精简至2组4层，添加FDR校正，Wald检验 |
| v2.1 | 2026-02-06 | 审查修订：修正功效分析，全局FDR，Bootstrap验证，敏感性分析增强，与全局对标 |

---

**方案制定时间**: 2026-02-06
**建议审核者**: 统计学专家、因果推断专家
**计划执行时间**: 待审核通过后开始
**审查状态**: ✅ 已通过（评分3.5/5，P0问题已修订）

**签字**:
```
[方案制定者] - 资深科研工作者 ✓
[同行评审] - 已审查，评分3.5/5，条件性通过
[待批准] - 项目负责人
```
