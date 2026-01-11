# 研究问题2和3的DiBS分析方案

**创建日期**: 2026-01-05
**文档版本**: v1.0
**状态**: ✅ 方案设计

---

## 📋 执行摘要

本文档记录了针对研究问题2（能耗-性能权衡）和问题3（中间变量中介效应）的DiBS因果分析方案。在DiBS参数调优成功后（2026-01-05），我们现在可以使用最优配置在新生成的6组数据上执行完整的因果发现分析。

### 核心研究问题

**问题2**: 能耗和性能之间的权衡关系 🔄

- **研究目标**:
  - 检测能耗和性能之间是否存在因果关系
  - 识别哪些超参数同时影响能耗和性能
  - 量化权衡强度和方向

**问题3**: 中间变量的中介效应 🔬

- **研究目标**:
  - 识别超参数通过哪些中介变量影响能耗
  - 发现完整的因果路径：超参数 → 中间变量 → 能耗/性能
  - 量化直接效应 vs 间接效应

---

## 🎯 DiBS分析框架

### 1. 为什么使用DiBS？

DiBS（Differentiable Bayesian Structure Learning）是一个基于贝叶斯的因果图学习算法，相比回归分析和中介效应分析，DiBS有以下优势：

**优势1: 自动发现因果结构**
- 回归分析需要预先指定因果方向
- DiBS可以自动发现所有变量间的因果关系
- 输出完整的因果图（DAG）

**优势2: 同时回答问题2和3**
- 问题2（权衡关系）：通过因果图中的边发现能耗↔性能的因果关系
- 问题3（中介效应）：通过因果图中的路径发现中介变量

**优势3: 发现未预期的因果关系**
- 可能发现回归分析中未考虑的因果路径
- 例如：性能 → GPU利用率 → 能耗（逆向因果）

### 2. DiBS成功配置

根据2026-01-05的参数调优报告，最优配置为：

```python
OPTIMAL_CONFIG = {
    "alpha_linear": 0.05,        # DiBS默认值，平衡稀疏性
    "beta_linear": 0.1,          # 低无环约束，允许更多边探索
    "n_particles": 20,           # 最佳性价比
    "tau": 1.0,                  # Gumbel-softmax温度
    "n_steps": 5000,             # 足够收敛
    "n_grad_mc_samples": 128,    # MC梯度样本数
    "n_acyclicity_mc_samples": 32  # 无环性MC样本数
}

# 预期结果（基于examples组测试）:
# - 强边(>0.3): 20-25条
# - 总边(>0.01): 120-130条
# - 耗时: 约10-15分钟/组
```

---

## 📊 数据准备

### 3. 六组数据概览

新生成的6组数据（40%阈值）全部包含性能指标：

| 组ID | 任务类型 | 样本数 | 特征数 | 性能指标数 | 能耗指标数 | 超参数数 |
|------|---------|-------|-------|-----------|-----------|---------|
| **group1** | examples | 259 | 18 | 1 (test_accuracy) | 10 | 4 |
| **group2** | vulberta | 152 | 16 | 3 (eval_loss等) | 10 | ~2 |
| **group3** | person_reid | 146 | 20 | 3 (map, rank1, rank5) | 10 | 3 |
| **group4** | bug_localization | 142 | 17 | 4 (top1/5/10/20) | 10 | ~2 |
| **group5** | mrt_oast | 88 | 16 | 3 (accuracy等) | 10 | ~2 |
| **group6** | resnet | 49 | 18 | 2 (val_acc, test_acc) | 10 | 3 |

**关键变量分类**：

1. **性能指标（Y_perf）**:
   - group1: `perf_test_accuracy`
   - group2: `perf_eval_loss`, `perf_final_training_loss`, `perf_eval_samples_per_second`
   - group3: `perf_map`, `perf_rank1`, `perf_rank5`
   - group4: `perf_top1/5/10/20_accuracy`
   - group5: `perf_accuracy`, `perf_precision`, `perf_recall`
   - group6: `perf_best_val_accuracy`, `perf_test_accuracy`

2. **能耗指标（Y_energy）**:
   - `energy_cpu_pkg_joules` - CPU Package能耗
   - `energy_cpu_ram_joules` - CPU RAM能耗
   - `energy_cpu_total_joules` - CPU总能耗
   - `energy_gpu_avg_watts` - GPU平均功率
   - `energy_gpu_max_watts` - GPU最大功率
   - `energy_gpu_min_watts` - GPU最小功率
   - `energy_gpu_total_joules` - GPU总能耗 ⭐ **主要指标**
   - `energy_gpu_temp_avg_celsius` - GPU平均温度
   - `energy_gpu_temp_max_celsius` - GPU最高温度
   - `energy_gpu_util_avg_percent` - GPU平均利用率 ⭐ **中介变量候选**

3. **超参数（X）**:
   - group1: `hyperparam_batch_size`, `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`
   - group3: `hyperparam_dropout`, `hyperparam_epochs`, `hyperparam_learning_rate`
   - group6: `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_weight_decay`

4. **其他控制变量**:
   - `duration_seconds` - 训练时长
   - `retries` - 重试次数（部分组）
   - `num_mutated_params` - 变异参数数量

---

## 🔬 分析方法

### 4. DiBS分析流程

#### 步骤1: 对每组执行DiBS因果发现（6个任务组）

```python
# 伪代码
for group in ["group1_examples", "group2_vulberta", ..., "group6_resnet"]:
    # 1. 加载数据
    data = load_csv(f"dibs_training/{group}.csv")

    # 2. 标准化（DiBS要求）
    data_scaled = standardize(data)

    # 3. 执行DiBS
    graph_samples = run_dibs(
        data=data_scaled,
        config=OPTIMAL_CONFIG
    )

    # 4. 提取因果图（后验均值）
    causal_graph = graph_samples.mean(axis=0)

    # 5. 筛选强边（>0.3）
    strong_edges = [(i, j) for i, j in edges if causal_graph[i,j] > 0.3]

    # 6. 保存结果
    save_graph(causal_graph, f"results/{group}_causal_graph.npy")
    save_edges(strong_edges, f"results/{group}_strong_edges.json")
```

**预计总耗时**: 6组 × 10-15分钟 = **1-1.5小时**

#### 步骤2: 提取研究问题2的证据（能耗-性能权衡）

对于每个任务组，分析因果图中的关键关系：

**2a. 直接因果关系**
```
性能 → 能耗  或  能耗 → 性能
```
- 检测 `perf_*` 和 `energy_*` 之间是否有直接的因果边
- 如果有，记录方向和强度
- 示例：`perf_test_accuracy → energy_gpu_total_joules` (强度=0.45)

**2b. 共同超参数因果**
```
超参数 → 性能
超参数 → 能耗
```
- 识别同时影响性能和能耗的超参数
- 分析效应方向是否相反（权衡）或相同（协同）
- 示例：`learning_rate → perf_test_accuracy (-)` 且 `learning_rate → energy_gpu_total_joules (+)`
  → 存在权衡（学习率提高 → 性能降低 + 能耗增加）

**2c. 间接权衡关系**
```
性能 → 中间变量 → 能耗
```
- 分析性能是否通过中间变量（如GPU利用率）影响能耗
- 示例：`perf_test_accuracy → gpu_util_avg → energy_gpu_total_joules`

#### 步骤3: 提取研究问题3的证据（中介效应）

对于每个任务组，分析因果图中的中介路径：

**3a. 三节点中介路径**
```
超参数 → 中间变量 → 能耗/性能
```
- 识别所有三节点路径
- 中介变量候选：
  - `energy_gpu_util_avg_percent` - GPU利用率 ⭐ **主要候选**
  - `energy_gpu_temp_avg_celsius` - GPU平均温度
  - `energy_gpu_temp_max_celsius` - GPU最高温度
  - `duration_seconds` - 训练时长
- 示例路径：
  - `learning_rate → gpu_util_avg → energy_gpu_total_joules`
  - `batch_size → gpu_temp_max → energy_gpu_avg_watts`

**3b. 多节点中介路径（复杂路径）**
```
超参数 → 中间变量1 → 中间变量2 → 能耗/性能
```
- 识别更长的因果链
- 示例：`epochs → duration_seconds → gpu_temp_max → energy_gpu_total_joules`

**3c. 中介效应量化**

对于每条中介路径，可以结合DiBS图和回归分析量化中介比例：

```python
# 路径: X → M → Y
# 使用DiBS确认路径存在后，用回归量化效应

# 总效应: X → Y
c = regress(Y ~ X).coef

# 路径a: X → M
a = regress(M ~ X).coef

# 路径b: M → Y (控制X)
b = regress(Y ~ X + M).coef_M

# 间接效应
indirect = a * b

# 中介比例
mediation_pct = indirect / c * 100
```

---

## 📈 预期输出

### 5. 针对问题2（能耗-性能权衡）

**输出1: 权衡关系汇总表**

| 任务组 | 直接因果边 | 共同超参数 | 权衡类型 | 强度 |
|-------|-----------|-----------|---------|------|
| examples | perf → gpu_util | learning_rate | 存在权衡 | 0.45 |
| vulberta | 无直接边 | - | 通过中介 | - |
| ... | ... | ... | ... | ... |

**输出2: 因果图可视化（每组一个）**
- 显示性能、能耗、超参数之间的因果边
- 用颜色区分：超参数（蓝色）、性能（绿色）、能耗（红色）、中介（黄色）

**输出3: 权衡发现总结**
```
发现1: learning_rate同时影响性能和能耗（6/6组一致）
  - 方向: learning_rate ↑ → perf ↓, energy ↑
  - 结论: 存在强权衡关系

发现2: batch_size影响不一致
  - group1: batch_size ↑ → perf ↑, energy ↑ (协同)
  - group3: batch_size ↑ → perf ↓, energy ↑ (权衡)
  - 结论: 任务相关的权衡

发现3: 性能和能耗无直接因果（5/6组）
  - 大多数组中，性能和能耗通过超参数或中介变量间接关联
  - 仅group1有弱的直接边: perf → gpu_util
```

### 6. 针对问题3（中介效应）

**输出1: 中介路径汇总表**

| 路径ID | 超参数 (X) | 中介变量 (M) | 结果 (Y) | 强度 | 显著性 |
|-------|-----------|-------------|---------|------|--------|
| P1 | learning_rate | gpu_util_avg | energy_gpu | 0.65 → 0.78 | ✅ |
| P2 | batch_size | gpu_temp_max | energy_gpu | 0.42 → 0.55 | ✅ |
| P3 | epochs | duration_seconds | energy_cpu | 0.88 → 0.45 | ✅ |
| ... | ... | ... | ... | ... | ... |

**输出2: 中介效应分解（每组一个）**

以group1为例：
```
路径: learning_rate → gpu_util_avg → energy_gpu_total_joules

DiBS因果图:
  learning_rate → gpu_util_avg: 边强度 = 0.65 ✅
  gpu_util_avg → energy_gpu_total_joules: 边强度 = 0.78 ✅

回归量化:
  总效应 (c)  = 42.35 W
  直接效应 (c') = 15.23 W (36.0%)
  间接效应 (a×b) = 27.12 W (64.0%)

结论: gpu_util_avg显著中介了learning_rate对能耗的影响（64.0%）
```

**输出3: 因果路径网络图**
- 显示所有检测到的中介路径
- 节点大小 = 中介重要性
- 边粗细 = 因果强度

**输出4: 核心中介变量排名**

| 中介变量 | 参与路径数 | 平均中介比例 | 重要性评分 |
|---------|-----------|-------------|----------|
| gpu_util_avg | 18条 | 68.3% | ⭐⭐⭐⭐⭐ |
| gpu_temp_max | 12条 | 42.1% | ⭐⭐⭐⭐ |
| duration_seconds | 8条 | 35.7% | ⭐⭐⭐ |
| gpu_temp_avg | 5条 | 28.4% | ⭐⭐ |

---

## 🚀 执行计划

### 7. 任务分解与时间表

#### 任务1: 准备DiBS分析脚本（30分钟）
- 创建 `scripts/run_dibs_for_questions_2_3.py`
- 功能：
  - 遍历6个任务组
  - 执行DiBS因果发现
  - 保存因果图矩阵和强边列表
  - 生成日志和进度报告

#### 任务2: 执行DiBS分析（1-1.5小时）
```bash
# 在causal-research环境中执行
conda activate causal-research
python scripts/run_dibs_for_questions_2_3.py --config optimal

# 输出目录: results/energy_research/questions_2_3_dibs/
# - group1_examples_graph.npy
# - group1_examples_edges.json
# - ... (每组2个文件)
# - dibs_run_log.txt
```

#### 任务3: 提取问题2证据（30分钟）
- 创建 `scripts/extract_tradeoff_evidence_from_dibs.py`
- 功能：
  - 读取6组因果图
  - 识别性能-能耗的因果边
  - 识别共同超参数
  - 生成权衡关系汇总表

#### 任务4: 提取问题3证据（30分钟）
- 创建 `scripts/extract_mediation_paths_from_dibs.py`
- 功能：
  - 读取6组因果图
  - 识别所有三节点中介路径
  - 分类中介变量（GPU利用率、温度等）
  - 生成中介路径汇总表

#### 任务5: 可视化与报告（1小时）
- 创建因果图可视化（每组一个）
- 创建因果路径网络图
- 生成完整分析报告
  - `docs/reports/QUESTIONS_2_3_DIBS_ANALYSIS_REPORT.md`

**总计预估时间**: 3.5-4小时

---

## 🎯 成功标准

### 8. 分析质量标准

**问题2（能耗-性能权衡）**:
- ✅ 至少在4/6组中检测到超参数对性能和能耗的共同影响
- ✅ 识别至少2个存在权衡的超参数（如learning_rate）
- ✅ 生成清晰的因果图可视化

**问题3（中介效应）**:
- ✅ 至少识别15-20条中介路径（跨所有6组）
- ✅ gpu_util_avg被识别为主要中介变量
- ✅ 中介比例在合理范围（20%-80%）

**技术质量**:
- ✅ DiBS收敛（最大边强度 > 0.3）
- ✅ 因果图无环（或环数极少）
- ✅ 与领域知识一致（如learning_rate ↑ → gpu_util ↑ → energy ↑）

---

## 📝 关键决策记录

### 决策1: 使用DiBS而非回归分析 ✅

**决定**: 优先使用DiBS进行因果发现，再结合回归分析量化效应

**理由**:
1. DiBS已经调优成功（2026-01-05）
2. DiBS可以自动发现因果结构，避免遗漏关键路径
3. 回归分析作为补充，用于量化中介效应

**决策时间**: 2026-01-05
**决策者**: Green + Claude

---

### 决策2: 对6个任务组分别分析 ✅

**决定**: 对6个任务组独立执行DiBS，而非合并分析

**理由**:
1. 不同任务组的超参数集合不同（group1有4个，group5只有2个）
2. 性能指标不同（test_accuracy vs eval_loss vs map）
3. 分组分析可以发现任务特定的因果关系

**权衡**: 样本量减少（最小仅49个），但DiBS已验证在259样本上成功

**决策时间**: 2026-01-05
**决策者**: Green + Claude

---

### 决策3: 使用最优配置（alpha=0.05, beta=0.1） ✅

**决定**: 所有6组统一使用最优配置

**理由**:
1. 已在group1（259样本）上验证成功
2. 最小组（group6, 49样本）样本量仍然足够
3. 统一配置便于对比分析

**备选方案**: 如果某组失败（0边），尝试更小的alpha（0.01）或更低的beta（0.05）

**决策时间**: 2026-01-05
**决策者**: Green + Claude

---

## 🔍 风险与应对

### 风险1: 小样本组（group6, 49样本）可能失败

**应对**:
- 优先执行大样本组（group1-4），积累经验
- 如果group6失败，尝试：
  - 降低alpha至0.01（更保守）
  - 增加粒子数至50（更多探索）
  - 与group1合并分析（都是图像分类）

### 风险2: 某些组可能0边

**应对**:
- 记录失败原因（样本量？变量数？数据质量？）
- 尝试备选配置（exploratory config）
- 至少分析成功的组，总结共性发现

### 风险3: 因果图中的环（非DAG）

**应对**:
- beta=0.1可能产生少量环
- 使用 `nx.is_directed_acyclic_graph()` 检测环
- 如果有环，应用阈值筛选（仅保留>0.4的强边）移除弱边
- 或者手动移除逻辑上不合理的边

### 风险4: 计算时间过长

**应对**:
- 如果单组超过30分钟，考虑减少n_steps至3000
- 或者减少粒子数至10（虽然可能降低边强度）
- 分批执行（先执行3组，再执行3组）

---

## 📚 参考文档

- [DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md](reports/DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md) - DiBS成功报告 ⭐⭐⭐
- [DATA_UPDATE_40PERCENT_THRESHOLD_20260105.md](reports/DATA_UPDATE_40PERCENT_THRESHOLD_20260105.md) - 6组数据生成报告
- [RESEARCH_QUESTIONS_METHOD_RECOMMENDATIONS_20251228.md](reports/RESEARCH_QUESTIONS_METHOD_RECOMMENDATIONS_20251228.md) - 旧方法推荐（已过时，DiBS已成功）
- [QUESTION1_REGRESSION_ANALYSIS_PLAN.md](QUESTION1_REGRESSION_ANALYSIS_PLAN.md) - 问题1方案（回归分析）

---

## 📌 版本历史

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-01-05 | 初始版本：设计问题2和3的DiBS分析方案，基于成功调优的DiBS配置 | Green + Claude |

---

**文档状态**: ✅ 方案设计完成
**维护者**: Green + Claude
**下次更新**: 执行DiBS分析后更新实际结果

**执行状态**: ⏳ 准备执行
**预计完成时间**: 2026-01-05（今日内）
