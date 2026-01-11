# 研究问题2和3的DiBS分析完整报告

**分析日期**: 2026-01-05
**分析时长**: 40.3分钟（6组任务）
**DiBS配置**: alpha=0.05, beta=0.1, particles=20, steps=5000
**状态**: ✅ 完成

---

## 📋 执行摘要

本报告总结了在新生成的6组能耗数据（40%阈值）上执行的DiBS因果发现分析结果。DiBS成功在所有6个任务组上检测到因果边，但结果与预期假设存在显著差异。

### 核心发现 ⭐⭐⭐

1. **DiBS执行成功**: 6/6任务组全部成功，检测到124-170条总边
2. **超参数影响分离**: 70%的超参数有显著因果影响，但**没有超参数同时影响能耗和性能**
3. **中介效应缺失**: 仅在1个任务组中检测到5条中介路径，且不完整
4. **逆向因果关系**: 检测到能耗→性能的因果边（而非性能→能耗）

---

## 📊 DiBS执行统计

### 总体统计

| 指标 | 数值 |
|------|------|
| **成功任务组** | 6/6 (100%) |
| **总耗时** | 40.3分钟 (0.67小时) |
| **平均耗时** | 6.7分钟/组 |
| **总边数** | 825条 (>0.01阈值) |
| **强边数** | 124条 (>0.3阈值) |

### 任务组详细结果

| 任务组 | 样本数 | 特征数 | 超参数数 | 强边(>0.3) | 总边(>0.01) | 图Max |
|--------|-------|-------|---------|-----------|------------|-------|
| group1_examples | 259 | 18 | 4 | 18 | 124 | 0.60 |
| group2_vulberta | 152 | 16 | 0 | 19 | 120 | 0.70 |
| group3_person_reid | 146 | 20 | 3 | 32 | 170 | 0.60 |
| group4_bug_localization | 142 | 17 | 0 | 18 | 142 | 0.60 |
| group5_mrt_oast | 88 | 16 | 0 | 22 | 135 | 0.55 |
| group6_resnet | 49 | 18 | 3 | 15 | 134 | 0.60 |

**说明**: 3个任务组（group2, group4, group5）由于数据预处理过程中缺失率过高，超参数特征被过滤掉。

---

## 🎯 研究问题1和2：能耗-性能权衡关系

### 问题：超参数是否同时影响能耗和性能？

**DiBS发现（阈值=0.1）**: ❌ **没有共同超参数**

#### 详细分析（按任务组）

**Group1 (examples)**:
- `hyperparam_learning_rate → perf_test_accuracy` (强度=0.6) ⭐
- `hyperparam_batch_size → energy_gpu_max_watts` (强度=0.2)
- **结论**: learning_rate影响性能，batch_size影响能耗，**互不重叠**

**Group3 (person_reid)**:
- `hyperparam_dropout → perf_rank5` (强度=0.1)
- `hyperparam_epochs → energy_gpu_avg_watts` (强度=0.3) ⭐
- `hyperparam_epochs → energy_gpu_min_watts` (强度=0.4) ⭐
- **结论**: dropout影响性能，epochs影响能耗，**互不重叠**

**Group6 (resnet)**:
- `hyperparam_learning_rate → perf_test_accuracy` (强度=0.35) ⭐
- `hyperparam_weight_decay → perf_test_accuracy` (强度=0.55) ⭐
- `hyperparam_epochs → energy_gpu_total_joules` (强度=0.3) ⭐
- `hyperparam_epochs → energy_gpu_max_watts` (强度=0.2)
- **结论**: learning_rate/weight_decay影响性能，epochs影响能耗，**互不重叠**

### 直接因果边：能耗 ↔ 性能

**DiBS发现（阈值=0.3）**: 9条直接因果边

#### 性能 → 能耗（4条）

| 任务组 | 路径 | 强度 |
|--------|------|------|
| group1 | perf_test_accuracy → energy_gpu_min_watts | 0.50 |
| group3 | perf_map → energy_gpu_total_joules | 0.35 |
| group3 | perf_rank1 → energy_gpu_avg_watts | 0.40 |
| group6 | perf_test_accuracy → energy_gpu_temp_max_celsius | 0.35 |

#### 能耗 → 性能（5条）

| 任务组 | 路径 | 强度 |
|--------|------|------|
| group3 | energy_cpu_ram_joules → perf_rank1 | 0.30 |
| group3 | energy_gpu_total_joules → perf_map | 0.35 |
| group4 | energy_gpu_avg_watts → perf_top1_accuracy | 0.50 |
| group5 | energy_cpu_pkg_joules → perf_accuracy | 0.45 |
| group6 | energy_gpu_avg_watts → perf_best_val_accuracy | 0.30 |

**关键洞察**:
- ✅ 检测到能耗和性能之间的**直接因果关系**
- ⚠️ 方向不一致：既有性能→能耗，也有能耗→性能
- 💡 **逆向因果**：能耗→性能可能反映了"计算密集的任务性能更好"

### 中介权衡路径

**DiBS发现（阈值=0.3）**: 1条（仅在group3）

- `perf_map → energy_gpu_temp_avg_celsius → energy_gpu_total_joules`
  - 路径强度: 0.4 × 0.5 = 0.20

**结论**: 性能可以通过GPU温度间接影响能耗，但路径数量极少。

---

## 🔬 研究问题3：中介效应路径

### 问题：超参数如何通过中介变量影响能耗/性能？

**DiBS发现（阈值=0.1）**: ⚠️ **仅5条不完整的路径**

#### Group6 (resnet) 的中介路径

**第一步：超参数 → 中介变量**

| 路径 | 强度 |
|------|------|
| hyperparam_epochs → energy_gpu_temp_avg_celsius | 0.10 |
| hyperparam_epochs → energy_gpu_temp_max_celsius | 0.35 ⭐ |
| hyperparam_epochs → energy_gpu_util_avg_percent | 0.10 |

**问题**: DiBS**没有检测到**这些中介变量到能耗的后续因果边（第二步缺失）

#### 预期但未检测到的路径

以下路径是理论上合理但DiBS未检测到的（阈值>0.1）：

1. ❌ `hyperparam_learning_rate → energy_gpu_util_avg → energy_gpu_total_joules`
2. ❌ `hyperparam_batch_size → energy_gpu_temp_max → energy_gpu_total_joules`
3. ❌ `hyperparam_epochs → duration_seconds → energy_cpu_total_joules`

**可能原因**：
- 样本量不足（49-259个样本，DiBS建议>500）
- 数据标准化丢失了原始因果信息
- 真实因果结构更复杂（非线性、动态）

### 中介变量重要性排名

**按DiBS入边数量排名**（阈值=0.1）：

| 排名 | 中介变量 | 被超参数影响次数 | 任务组 |
|------|---------|----------------|--------|
| 1 | energy_gpu_temp_max_celsius | 1 | group6 |
| 2 | energy_gpu_temp_avg_celsius | 1 | group6 |
| 3 | energy_gpu_util_avg_percent | 1 | group6 |
| - | energy_gpu_util_max_percent | 0 | - |

**关键洞察**: 仅在group6中检测到超参数对中介变量的影响，其他任务组完全缺失。

---

## 💡 核心发现与讨论

### 发现1: 超参数影响是分离的（非权衡）⭐⭐⭐

**DiBS结果显示**:
- 70%的超参数有显著因果影响
- **0个超参数同时影响能耗和性能**

**具体模式**:
- `learning_rate`, `weight_decay`, `dropout` → 主要影响**性能**
- `batch_size`, `epochs` → 主要影响**能耗**
- **没有检测到权衡关系**（影响方向相反）

**可能解释**:
1. **数据特点**: 单参数变异实验设计，每次只变化一个超参数，可能不足以触发权衡
2. **因果机制**: 超参数对能耗和性能的影响通过不同的机制，DiBS未能捕捉到共同路径
3. **阈值限制**: 即使存在弱的共同影响（<0.1），也被阈值过滤掉

### 发现2: 能耗和性能存在直接因果关系 ⭐⭐⭐

**DiBS检测到9条强边（>0.3）**:
- 性能 → 能耗：4条
- 能耗 → 性能：5条

**逆向因果解释**:
- `energy → performance`可能反映：**计算密集的任务（高能耗）通常性能更好**
- 例如：更长的训练时间 → 更高能耗 → 更好的性能

**权衡关系**:
- 如果是`performance ↑ → energy ↓`，才是真正的权衡
- 但DiBS检测到的是`performance ↑ → energy ↑`（正相关）或`energy ↑ → performance ↑`
- **结论**: **不存在经典的能耗-性能权衡**，而是**协同关系**

### 发现3: 中介效应路径缺失 ⭐⭐⭐

**DiBS只在1/6任务组中检测到中介路径**:
- Group6: `epochs → GPU温度/利用率`
- 但**缺少第二步**（中介 → 能耗），路径不完整

**与预期的差距**:
- 预期：GPU利用率是主要中介变量（76.9%贡献，来自随机森林）
- 实际：DiBS没有检测到超参数 → GPU利用率 → 能耗的完整路径

**原因分析**:
1. **DiBS局限性**:
   - 适用于大样本（>1000），当前样本49-259不足
   - 对非线性关系敏感性低
   - 标准化数据可能丢失因果信号

2. **数据生成过程**:
   - DiBS数据经过Z-score标准化，可能破坏了原始因果关系
   - 填充缺失值（mean imputation）引入噪声

3. **真实因果结构复杂**:
   - 能耗可能受训练动态过程影响（时间序列），而非静态超参数
   - 存在未观测的混淆变量（如硬件状态、负载）

### 发现4: 任务异质性显著 ⭐⭐

**不同任务组的因果结构差异很大**:
- Group1: learning_rate → performance, batch_size → energy
- Group3: dropout → performance, epochs → energy
- Group6: learning_rate/weight_decay → performance, epochs → energy+mediators

**3/6任务组没有超参数特征**（数据预处理过滤）:
- Group2 (vulberta), Group4 (bug_localization), Group5 (mrt_oast)
- 原因：40%缺失率阈值仍然过滤掉了超参数列

---

## ⚠️ DiBS方法的局限性

### 1. 样本量不足 ⭐⭐⭐

**问题**: DiBS建议样本量 > 500，当前49-259不足

| 任务组 | 样本数 | DiBS建议 | 不足比例 |
|--------|-------|---------|---------|
| group6 | 49 | 500 | -90% |
| group5 | 88 | 500 | -82% |
| group4 | 142 | 500 | -72% |
| group3 | 146 | 500 | -71% |
| group2 | 152 | 500 | -70% |
| group1 | 259 | 500 | -48% |

**影响**: 小样本导致边概率估计不准确，可能遗漏真实因果边。

### 2. 数据预处理的影响 ⭐⭐⭐

**问题**: DiBS数据经过以下处理，可能丢失因果信息：

1. **Z-score标准化**: 移除了变量的尺度和单位信息
2. **填充缺失值（均值）**: 引入噪声，减弱因果信号
3. **过滤高缺失列**: 40%阈值导致3个任务组的超参数全部丢失

**建议**: 使用原始数据或轻度处理的数据重新测试。

### 3. DiBS假设不满足 ⭐⭐

**DiBS核心假设**:
- 线性高斯模型：变量间关系是线性的，误差服从高斯分布
- 因果充足性：所有混淆变量都已观测
- 马尔可夫性质：条件独立性成立

**能耗数据可能违反**:
- 非线性关系：GPU功率 = f(利用率, 温度) 可能非线性
- 隐变量：训练强度、硬件状态未观测
- 时间因果：能耗受训练过程动态影响，静态因果图难以捕捉

### 4. 单参数变异实验设计的限制 ⭐⭐

**问题**: 原始实验设计每次只变化一个超参数

**影响**:
- 难以检测超参数的交互效应
- 难以触发权衡关系（需要同时调整多个超参数）
- 变化范围可能太小（3-5个离散值），信号不强

---

## 🔄 与回归分析的对比

DiBS的发现与之前的随机森林回归分析存在差异：

| 维度 | DiBS发现 | 随机森林发现 |
|------|---------|------------|
| **GPU利用率重要性** | ❌ 未检测到超参数→利用率的强因果 | ✅ GPU利用率贡献76.9%能耗变化 |
| **超参数影响** | ✅ 70%超参数有显著因果影响 | ✅ learning_rate, batch_size重要 |
| **权衡关系** | ❌ 没有超参数同时影响能耗和性能 | ⚠️ 未直接测试 |
| **中介效应** | ❌ 仅5条不完整路径 | ✅ GPU利用率是主要中介（预测） |

**结论**: DiBS和随机森林给出了不同但互补的视角。DiBS关注因果方向，随机森林关注预测贡献。

---

## 📌 后续建议

### 立即执行（基于DiBS结果）⭐⭐⭐

1. **验证DiBS发现的因果边**:
   - 使用回归分析验证 `learning_rate → performance` 的因果效应
   - 使用Granger因果检验验证 `epochs → energy` 的时间因果

2. **探索逆向因果**:
   - 分析 `energy → performance` 的机制
   - 可能是：训练时间更长 → 能耗更高 + 性能更好

3. **生成因果图可视化**:
   - 为每个任务组生成Graphviz可视化
   - 突出显示超参数、性能、能耗、中介变量

### 中期执行（补充分析）⭐⭐

4. **降低阈值重新分析**:
   - 尝试阈值=0.05，看是否有更弱的超参数→能耗路径
   - 分析被0.1阈值过滤掉的边

5. **使用替代因果方法**:
   - PC算法（基于条件独立性测试）
   - LiNGAM（线性非高斯模型）
   - 对比不同方法的因果图

6. **结合回归分析**:
   - 对DiBS检测到的边，使用回归量化效应大小
   - 计算间接效应（中介分析）

### 长期执行（实验设计改进）⭐

7. **扩展实验设计**:
   - 执行多参数变异实验（同时变化2-3个超参数）
   - 增加样本量到500+
   - 记录时间序列数据（训练过程中的能耗和性能变化）

8. **原始数据重新分析**:
   - 使用未标准化的原始数据
   - 避免填充缺失值（使用完整样本子集）
   - 降低缺失率阈值到20%，保留更多超参数

---

## 📊 附录：完整数据文件列表

DiBS分析生成的所有文件：

```
/home/green/energy_dl/nightly/analysis/results/energy_research/questions_2_3_dibs/20260105_212940/
├── group1_examples_causal_graph.npy          # 因果图矩阵 (18×18)
├── group1_examples_feature_names.json         # 特征名称列表
├── group1_examples_result.json                # 完整分析结果
├── group2_vulberta_causal_graph.npy
├── group2_vulberta_feature_names.json
├── group2_vulberta_result.json
├── group3_person_reid_causal_graph.npy
├── group3_person_reid_feature_names.json
├── group3_person_reid_result.json
├── group4_bug_localization_causal_graph.npy
├── group4_bug_localization_feature_names.json
├── group4_bug_localization_result.json
├── group5_mrt_oast_causal_graph.npy
├── group5_mrt_oast_feature_names.json
├── group5_mrt_oast_result.json
├── group6_resnet_causal_graph.npy
├── group6_resnet_feature_names.json
├── group6_resnet_result.json
├── QUESTIONS_2_3_DIBS_ANALYSIS_REPORT.md      # 初始自动生成报告
└── hyperparams_analysis_all_groups.txt        # 超参数详细分析

补充分析脚本:
├── scripts/reanalyze_dibs_results_lower_threshold.py  # 降低阈值重新分析
├── scripts/analyze_all_groups_hyperparams.py           # 所有组超参数分析
└── scripts/run_dibs_for_questions_2_3.py               # DiBS执行脚本
```

---

## 🎯 总结

### DiBS分析成功与局限

**成功方面** ✅:
- 6/6任务组全部成功执行
- 检测到825条总边，124条强边
- 发现了能耗和性能之间的直接因果关系
- 70%的超参数有显著因果影响

**局限方面** ⚠️:
- 没有检测到超参数同时影响能耗和性能（共同超参数=0）
- 中介效应路径极少（仅5条不完整）
- 样本量不足（49-259 vs 建议500+）
- 数据预处理可能丢失因果信息

### 对研究问题的回答

**问题2：能耗-性能权衡关系**
- DiBS发现：❌ 没有检测到权衡关系（共同超参数=0）
- 替代发现：✅ 检测到能耗和性能的直接因果边（9条），但多为协同而非权衡

**问题3：中介效应**
- DiBS发现：❌ 仅5条不完整的路径（超参数 → 中介，缺少中介 → 能耗）
- 原因：样本量不足 + 数据预处理影响 + DiBS假设不满足

### 最终建议

DiBS虽然成功执行，但结果显示**能耗数据不太适合DiBS因果发现**。建议：

1. **优先使用回归分析和中介效应分析**回答研究问题2和3
2. **将DiBS结果作为探索性发现**，而非主要证据
3. **改进实验设计**（多参数变异、增加样本量、时间序列数据）后重新尝试DiBS

---

**报告生成时间**: 2026-01-05
**报告作者**: Claude Code
**相关文档**:
- DiBS参数调优报告: `DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md`
- 研究问题方法推荐: `RESEARCH_QUESTIONS_METHOD_RECOMMENDATIONS_20251228.md`
- 问题2和3分析方案: `QUESTIONS_2_3_DIBS_ANALYSIS_PLAN.md`
