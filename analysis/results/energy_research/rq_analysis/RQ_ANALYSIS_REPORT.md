# RQ数据分析总结报告

**生成日期**: 2026-02-07
**分析版本**: v1.0
**数据来源**: 全局标准化DiBS+ATE结果

---

## 一、分析概览

本报告总结RQ1-RQ3的数据分析结果，基于全局标准化因果分析数据。

### 1.1 研究问题

| RQ | 研究问题 | 子问题 |
|----|---------|--------|
| **RQ1** | 超参数对能耗的影响 | a) 主效应 b) 调节效应 |
| **RQ2** | 中间变量解释因果机制 | 中介路径分析 |
| **RQ3** | 能耗与性能权衡 | 权衡检测与分布 |

### 1.2 分析环境

- **Python环境**: causal-research (conda)
- **关键依赖**: pandas, numpy, matplotlib
- **执行日期**: 2026-02-07

---

## 二、数据源

### 2.1 RQ1数据源

| 数据文件 | 位置 | 说明 |
|---------|------|------|
| DiBS+ATE结果 | `results/energy_research/data/global_std_dibs_ate/group*_dibs_global_std_ate.csv` | 6组，共151条边 |

**字段说明**:
- `source`: 源变量
- `target`: 目标变量
- `strength`: 边强度 (DiBS)
- `ate_global_std`: 平均处理效应 (全局标准化)
- `ate_global_std_ci_lower/upper`: 95%置信区间
- `ate_global_std_is_significant`: 显著性标记

### 2.2 RQ2数据源

| 数据文件 | 位置 | 说明 |
|---------|------|------|
| 因果图矩阵 | `results/energy_research/archived_data/global_std/group*/` | 6组因果图 |
| 边文件 | `*_dibs_edges_threshold_0.3.csv` | 强边 (≥0.3) |
| 特征名 | `*_feature_names.json` | 35个变量（含8个交互项） |

**数据验证**:
- 因果图创建日期: 2026-02-03 18:55
- 全局标准化数据创建日期: 2026-02-03 14:40
- 时间顺序正确，数据来源确认

### 2.3 RQ3数据源

| 数据文件 | 位置 | 说明 |
|---------|------|------|
| 权衡检测结果 | `results/energy_research/tradeoff_detection_global_std/tradeoff_detailed_global_std.csv` | 61条权衡 |

**字段说明**:
- `group_id`: 组别
- `intervention`: 干预变量
- `metric1/metric2`: 权衡的两个指标
- `ate1/ate2`: 对应ATE
- `sign1/sign2`: ATE符号
- `is_significant`: 显著性

---

## 三、分析脚本

| 脚本 | 位置 | 功能 |
|-----|------|------|
| `rq1_analysis.py` | `scripts/rq1_analysis.py` | RQ1主效应和调节效应分析 |
| `rq2_analysis.py` | `scripts/rq2_analysis.py` | RQ2中介路径分析 |
| `rq3_analysis.py` | `scripts/rq3_analysis.py` | RQ3权衡检测分析 |

**执行命令**:
```bash
conda activate causal-research
python3 scripts/rq1_analysis.py
python3 scripts/rq2_analysis.py
python3 scripts/rq3_analysis.py
```

---

## 四、输出文件清单

### 4.1 表格 (`tables/`)

| 文件名 | RQ | 说明 | 记录数 |
|-------|-----|------|--------|
| `table1_rq1a_main_effects.csv` | RQ1a | 超参数对能耗的主效应 | 4条边 |
| `table2_rq1b_moderation_effects.csv` | RQ1b | 并行化调节效应 | 7条边 |
| `table3_rq2_mediator_frequency.csv` | RQ2 | 中介变量频率统计 | 4个中介 |
| `table4_rq2_main_paths.csv` | RQ2 | 主要因果路径 | 7条路径 |
| `table5_rq3_tradeoff_summary.csv` | RQ3 | 权衡关系汇总 | 4类汇总 |
| `table6_rq3_energy_perf_tradeoffs.csv` | RQ3 | 能耗vs性能权衡详情 | 4条权衡 |

### 4.2 图表 (`figures/`)

| 文件名 | RQ | 类型 | 尺寸 |
|-------|-----|------|------|
| `figure1_rq1a_main_effects_bar.pdf` | RQ1a | 柱状图 | 10×6 inch |
| `figure1_rq1a_main_effects_forest.pdf` | RQ1a | 森林图 | 动态 |
| `figure2_rq1b_moderation_effects_bar.pdf` | RQ1b | 柱状图 | 12×6 inch |
| `figure2_rq1b_moderation_effects_forest.pdf` | RQ1b | 森林图 | 动态 |
| `figure3_rq2_causal_paths.pdf` | RQ2 | 流程图 | 12×8 inch |
| `figure4_rq2_mediator_heatmap.pdf` | RQ2 | 热力图 | 8×6 inch |
| `figure5_rq3_tradeoff_scatter.pdf` | RQ3 | 散点图 | 8×6 inch |
| `figure6_rq3_group_tradeoff_bar.pdf` | RQ3 | 堆叠柱状图 | 10×6 inch |

**图表规格**:
- 字体: Times New Roman 12pt
- 分辨率: 300 DPI
- 格式: PDF + PNG
- 配色: 红蓝色盲友好

---

## 五、主要发现

### 5.1 RQ1: 超参数对能耗的影响

#### RQ1a: 主效应 (4条边，3条显著)

| 组 | 超参数 | 能耗变量 | ATE | 方向 |
|----|--------|---------|-----|------|
| Person ReID | epochs | gpu_total_joules | +0.258 | ↑增加能耗 |
| Person ReID | epochs | gpu_min_watts | -0.127 | ↓减少能耗 |
| Examples | learning_rate | gpu_min_watts | -0.002 | ↓减少能耗 |

**发现**: epochs对GPU能耗有双向影响（增加总能耗但降低最小功率）

#### RQ1b: 调节效应 (7条边，100%显著)

| 组 | 交互项 | 能耗变量 | ATE | 方向 |
|----|--------|---------|-----|------|
| Person ReID | epochs×P | gpu_total_joules | +0.481 | ↑ |
| Examples | batch_size×P | gpu_min_watts | -0.340 | ↓ |
| Person ReID | dropout×P | gpu_min_watts | +0.303 | ↑ |

**发现**: 并行化显著调节超参数对能耗的影响，调节效应(7条)多于主效应(4条)

### 5.2 RQ2: 中间变量解释

#### 路径统计 (7条路径)

| 路径类型 | 数量 | 说明 |
|---------|------|------|
| 主效应路径 | 2 | hyperparam → mediator → energy |
| 调节效应路径 | 5 | interaction → mediator → energy |

#### 中介变量使用频率

| 中介变量 | 路径数 | 组覆盖 |
|---------|--------|--------|
| gpu_temp_avg_celsius | 2 | 1/6 |
| gpu_util_avg_percent | 2 | 1/6 |
| gpu_util_max_percent | 2 | 2/6 |
| gpu_temp_max_celsius | 1 | 1/6 |

**发现**:
- 温度和利用率变量均参与中介
- 仅Examples和VulBERTa组有间接路径
- 调节效应路径多于主效应路径

### 5.3 RQ3: 能耗与性能权衡

#### 权衡汇总 (61条)

| 干预类型 | 总数 | 能耗vs性能 | 能耗vs能耗 |
|---------|------|-----------|-----------|
| Hyperparameter | 1 | 0 | 1 |
| Interaction | 34 | 1 | 2 |
| Other | 26 | 3 | 0 |
| **Total** | **61** | **4** | **3** |

#### 能耗vs性能权衡详情 (4条)

| 组 | 干预变量 | 能耗 | 性能 | 方向 |
|----|---------|------|------|------|
| Examples | batch_size×P | gpu_min_watts | test_accuracy | 同向↓ |
| VulBERTa | energy_cpu_ram | cpu_pkg_joules | training_loss | Energy↑ Perf↓ |
| VulBERTa | energy_cpu_ram | cpu_total_joules | training_loss | Energy↑ Perf↓ |
| VulBERTa | energy_cpu_ram | gpu_total_joules | training_loss | Energy↑ Perf↓ |

**发现**:
- 超参数直接导致的能耗-性能权衡极少（仅1条交互项相关）
- 大多数权衡发生在能耗变量之间或非超参数变量
- Examples组权衡最多（30条），交互项主导

---

## 六、需求符合性验证

### 6.1 数据过滤条件

| RQ | 条件 | 验证 |
|----|------|------|
| RQ1a | source以hyperparam_开头且不含_x_is_parallel，target为energy变量，strength≥0.3 | ✅ |
| RQ1b | source包含_x_is_parallel，target为energy变量，strength≥0.3 | ✅ |
| RQ2 | hyperparam/interaction → mediator → energy，两段边strength≥0.3 | ✅ |
| RQ3 | 使用tradeoff_detailed_global_std.csv | ✅ |

### 6.2 预期边数对比

| 数据项 | 需求预期 | 实际结果 | 符合 |
|-------|---------|---------|------|
| RQ1a边数 | 4条 | 4条 | ✅ |
| RQ1b边数 | 7条 | 7条 | ✅ |
| RQ2路径数 | 需生成 | 7条 | ✅ |
| RQ3权衡数 | 61条 | 61条 | ✅ |
| RQ3能耗vs性能 | 1条 | 4条 | ⚠️ 超出预期 |

**说明**: RQ3能耗vs性能权衡实际发现4条，超出需求文档预期的1条，原因是VulBERTa组的能耗变量作为intervention也产生了能耗vs性能权衡。

---

## 七、变量分类参考

### 7.1 能耗变量 (energy)

```
energy_cpu_pkg_joules, energy_cpu_ram_joules, energy_cpu_total_joules,
energy_gpu_total_joules, energy_gpu_avg_watts, energy_gpu_min_watts, energy_gpu_max_watts
```

### 7.2 中介变量 (mediator)

```
energy_gpu_temp_avg_celsius, energy_gpu_temp_max_celsius,
energy_gpu_util_avg_percent, energy_gpu_util_max_percent
```

### 7.3 性能变量 (performance)

```
perf_test_accuracy, perf_map, perf_rank1, perf_rank5, perf_precision, perf_recall,
perf_eval_samples_per_second, perf_top1/5/10/20_accuracy, perf_final_training_loss, perf_eval_loss
```

### 7.4 超参数 (hyperparam)

```
hyperparam_batch_size, hyperparam_epochs, hyperparam_learning_rate, hyperparam_seed,
hyperparam_dropout, hyperparam_l2_regularization, hyperparam_alpha, hyperparam_kfold, hyperparam_max_iter
```

### 7.5 交互项 (interaction)

```
hyperparam_*_x_is_parallel (8个)
```

---

## 八、目录结构

```
analysis/results/energy_research/rq_analysis/
├── tables/
│   ├── table1_rq1a_main_effects.csv
│   ├── table2_rq1b_moderation_effects.csv
│   ├── table3_rq2_mediator_frequency.csv
│   ├── table4_rq2_main_paths.csv
│   ├── table5_rq3_tradeoff_summary.csv
│   └── table6_rq3_energy_perf_tradeoffs.csv
├── figures/
│   ├── figure1_rq1a_main_effects_bar.pdf/png
│   ├── figure1_rq1a_main_effects_forest.pdf/png
│   ├── figure2_rq1b_moderation_effects_bar.pdf/png
│   ├── figure2_rq1b_moderation_effects_forest.pdf/png
│   ├── figure3_rq2_causal_paths.pdf/png
│   ├── figure4_rq2_mediator_heatmap.pdf/png
│   ├── figure5_rq3_tradeoff_scatter.pdf/png
│   └── figure6_rq3_group_tradeoff_bar.pdf/png
└── RQ_ANALYSIS_REPORT.md (本报告)
```

---

## 九、参考文档

| 文档 | 位置 | 说明 |
|-----|------|------|
| 数据分析方案 | `docs/research_plans/RQ_DATA_ANALYSIS_PLAN.md` | v2.0，执行规范 |
| 研究方案 | `docs/research_plans/RQ_ANALYSIS_PLAN_v2.md` | 整体研究框架 |
| 全局标准化修复报告 | `results/energy_research/reports/GLOBAL_STD_FIX_ACCEPTANCE_REPORT_20260201.md` | 数据验收 |

---

**报告生成**: Claude Code
**审核状态**: 已完成
**下一步**: 论文写作
