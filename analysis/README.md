# Analysis - 因果推断分析模��

**项目**: ASE 2023论文复现 + 能耗数据因果分析扩展
**最后更新**: 2026-01-23
**维护者**: Green

---

## 📋 项目概述

本模块是独立的因果推断分析模块，用于研究**深度学习训练超参数对能耗和性能的因果影响**。

**核心技术**:
- **DiBS** (Differentiable Bayesian Structure Learning) - 因果图学习
- **DML** (Double Machine Learning) - 因果效应估计
- **回归分析** - 因果效应量化
- **中介效应分析** - 间接路径识别

**研究目标**: 回答3个核心研究问题
1. 🔬 超参数对能耗的影响（方向和大小）
2. ⚖️ 能耗和性能之间的权衡关系
3. 🔍 中间变量的中介效应

---

## 什么是因果推断分析?

### 为什么需要因果推断?

**传统回归分析**可以识别**相关性**，但无法确定**因果性**：

**相关性示例**:
```python
import pandas as pd
df = pd.read_csv('data/data.csv')
# 发现: learning_rate与能耗相关 (r = -0.65)
df.plot.scatter(x='learning_rate', y='gpu_total_joules')
```
❌ **问题**: 是learning_rate导致能耗降低?还是低能耗配置恰好用了低learning_rate?

**因果推断示例**:
```
learning_rate → gpu_utilization → gpu_total_joules
    ↓                ↓                ↓
  (直接效应)      (中介变量)       (最终能耗)
```
✅ **优势**:
- 识别真实的因果关系（learning_rate → 能耗）
- 量化中介效应（通过GPU利用率）
- 控制混淆变量（模型类型、数据集大小）

### 核心技术对比

| 技术 | 用途 | 输入 | 输出 | 优势 |
|------|------|------|------|------|
| **回归分析** | 预测/相关性 | 数据(X,Y) | 系数(β) | 简单、快速 |
| **DiBS** | 因果图发现 | 数据(X) | 有向无环图 | 自动学习因果结构 |
| **DML** | 因果效应估计 | 图+数据 | ATE(平均处理效应) | 处理混淆变量 |

### 典型分析流程

```
步骤1: DiBS因果发现
  ↓
输入: 6组超参数+能耗数据
输出: 因果图 (如: learning_rate → gpu_total_joules)
  ↓
步骤2: 白名单过滤 (删除虚假因果)
  ↓
输出: 合理的因果边 (227条)
  ↓
步骤3: 回归验证 (确认因果效应)
  ↓
输出: 因果效应大小 (如: learning_rate ↑1 → 能耗 ↓15J)
```

### 需要什么背景知识?

**必需**:
- Python数据分析基础
- 统计学基础（相关性、回归、p值）
- 机器学习概念（超参数、训练）

**推荐**（但非必需）:
- 因果推断入门
- 图论（有向无环图DAG）
- 贝叶斯方法

**学习资源**:
- [因果推断入门](https://github.com/AMLab-Amsterdam/CE_tutorial)
- [DiBS论文](https://openreview.net/forum?id=NkYlTAl3Z1)
- [DML教程](https://arxiv.org/abs/1608.00060)

---

## 🎉 重大突破

### DiBS参数调优成功 (2026-01-05) ⭐⭐⭐⭐⭐

**核心成果**:
- ✅ **11个实验全部成功**（100%成功率）
- ✅ **检测到23条强边**（>0.3阈值）
- ✅ **找到根本问题**: alpha值范围错误（0.1-0.9 vs 正确的0.001-0.05）
- ✅ **最优配置**: alpha=0.05, beta=0.1, particles=20

**影响**: DiBS现在可用于能耗数据因果发现！

详细报告: [docs/technical_reference/DIBS_PARAMETER_TUNING_ANALYSIS.md](docs/technical_reference/DIBS_PARAMETER_TUNING_ANALYSIS.md)

---

## 数据流程

### 从主项目到因果分析

```
主项目数据 (data/raw_data.csv, 836行)
    ↓
[generate_6groups_final.py] 语义合并+分组
    ↓
6组数据 (analysis/data/energy_research/6groups_final/)
  ├── group1_examples.csv (304样本, 4超参数)
  ├── group2_vulberta.csv (72样本, 4超参数)
  ├── group3_person_reid.csv (206样本, 4超参数)
  ├── group4_bug_localization.csv (90样本, 4超参数)
  ├── group5_mrt_oast.csv (72样本, 5超参数)
  └── group6_resnet.csv (74样本, 4超参数)
    ↓
[run_dibs_6groups_final.py] DiBS因果发现
    ↓
DiBS结果 (analysis/results/energy_research/questions_2_3_dibs/)
  ├── group1_*.npy (因果图矩阵)
  ├── group1_*.json (特征名称)
  └── group1_dibs_results.json (完整结果)
    ↓
[extract_dibs_edges_to_csv.py] 提取因果边
    ↓
因果边CSV (analysis/results/energy_research/data/interaction/threshold/)
  ├── group1_examples_causal_edges.csv (23条边)
  ├── group2_vulberta_causal_edges.csv (35条边)
  └── ... (共6个文件, 539条原始边)
    ↓
[filter_causal_edges_by_whitelist.py] 白名单过滤
    ↓
过滤后边 (analysis/results/energy_research/data/interaction/whitelist/)
  ├── group1_examples_causal_edges_whitelist.csv (43条边)
  ├── group2_vulberta_causal_edges_whitelist.csv (35条边)
  └── ... (共6个文件, 227条边, 42.1%保留率)
    ↓
[validate_dibs_with_regression.py] 回归验证
    ↓
最终结果 (因果效应大小 + 统计显著性)
```

### 核心分析流程

```
┌──────────────────────────────────────────────────┐
│          主项目 (mutation.py)                    │
│  实验训练 → 能耗监控 → raw_data.csv (836行)     │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│      数据准备 (generate_6groups_final.py)        │
│  语义合并 → 模型变量 → 6组CSV                     │
└──────────────────┬───────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
  ▼────▼─────┐          ▼─────▼─────┐
  │ DiBS分析  │          │ 交互项分析  │
  │(标准版)   │          │ (含交互项)   │
  └────┬──────┘          └─────┬──────┘
       │                       │
       └───────────┬───────────┘
                   ▼
       ┌───────────────────────┐
       │  因果图 (539条原始边)   │
       └───────┬───────────────┘
               ▼
       ┌───────────────────────┐
       │  白名单过滤 (16条规则)  │
       └───────┬───────────────┘
               ▼
       ┌───────────────────────┐
       │  合理因果边 (227条)     │
       └───────┬───────────────┘
               ▼
       ┌───────────────────────┐
       │  回归验证 (因果效应)     │
       └───────┬───────────────┘
               ▼
       ┌───────────────────────┐
       │  最终结论              │
       │ - Q1: 超参数→能耗       │
       │ - Q2: 能耗↔性能权衡     │
       │ - Q3: 中介效应          │
       └───────────────────────┘
```

---

## 📁 项目结构

```
analysis/
├── README.md                           # 本文件
├── environment.yaml                    # concausal-research环境配置
├── requirements.txt                    # Python依赖
│
├── docs/                               # 📚 文档目录（已重组）
│   ├── INDEX.md                        # ⭐ 文档总索引
│   ├── README.md                       # ⭐ 文档组织说明
│   ├── essential_guides/               # ⚠️ 必读指南（4个）
│   ├── current_plans/                  # 📋 当前方案（4个）
│   ├── technical_reference/            # 📚 技术参考（25个）
│   └── guides/                         # 使用指南（6个）
│
├── scripts/                            # 🔧 核心脚本（20个）
│   ├── 数据准备脚本 (3个)
│   │   ├── generate_6groups_final.py          # ⭐⭐⭐ 生成6分组数据
│   │   ├── extract_from_json_with_defaults.py # 从JSON提取（回填默认值）
│   │   └── backfill_hyperparameters_from_models_config.py # 回填超参数
│   │
│   ├── DiBS因果发现脚本 (4个)
│   │   ├── run_dibs_6groups_final.py         # ⭐⭐⭐ 6组DiBS分析（标准版）
│   │   ├── run_dibs_6groups_interaction.py   # ⭐⭐⭐ 6组DiBS分析（交互项版）
│   │   ├── run_dibs_for_questions_2_3.py     # 问题2/3专用分析
│   │   └── dibs_parameter_sweep.py           # 参数扫描测试
│   │
│   ├── 因果边处理脚本 (3个) ⭐⭐⭐
│   │   ├── extract_dibs_edges_to_csv.py      # ⭐⭐⭐ 提取因果边到CSV
│   │   ├── filter_causal_edges_by_whitelist.py # ⭐⭐⭐ 白名单过滤
│   │   └── visualize_dibs_causal_graphs.py    # ⭐⭐ 因果图可视化
│   │
│   ├── 验证分析脚本 (3个)
│   │   ├── validate_dibs_with_regression.py  # ⭐⭐ 回归验证DiBS发现
│   │   ├── mediation_analysis_question3.py   # ⭐⭐ 中介效应分析
│   │   └── convert_dibs_to_csv.py            # DiBS结果转CSV
│   │
│   ├── 工具脚本 (7个)
│   │   ├── check_dibs_progress.py            # 检查DiBS进度
│   │   ├── check_dibs_interaction_config.py  # 检查交互项配置
│   │   ├── verify_5groups_data.py            # 验证5组数据
│   │   ├── test_preprocess_stratified_data.py # 测试数据预处理
│   │   ├── config.py / config_energy.py      # 配置文件
│   │   └── run_6groups_dibs_parallel.sh      # 并行运行脚本
│   │
│   └── demos/                             # 演示脚本（5个，用于学习）
│       ├── demo_quick_run.py
│       ├── demo_adult_dataset.py
│       └── ...
│
├── utils/                               # 核心模块
│   ├── causal_discovery.py               # DiBS因果图学习
│   ├── causal_inference.py               # DML因果推断
│   └── ...
│
├── tests/                               # 测试套件
│   ├── test_energy_causal_analysis.py   # 能耗因果分析测试
│   ├── test_whitelist_implementation.py # 白名单实现测试
│   └── ...
│
├── data/                                # 数据目录
│   ├── paper_replication/               # 论文复现数据（Adult等）
│   └── energy_research/                 # 能耗研究数据
│       ├── dibs_training/               # 6组DiBS训练数据
│       └── ...
│
└── results/                             # 结果目录
    ├── paper_replication/               # 论文复现结果
    └── energy_research/                 # 能耗研究结果
        ├── questions_2_3_dibs/          # DiBS分析结果（JSON+NPY）
        ├── data/                        # 处理后的数据
        │   └── interaction/whitelist/   # ⭐ 白名单过滤后的因果边CSV
        └── causal_graph_visualizations/ # 因果图可视化
```

---

## 🔥 核心脚本使用指南

### 1. 数据准备流程

```bash
# 步骤1: 生成6分组数据（超参数语义合并 + 模型变量）
python3 analysis/scripts/generate_6groups_final.py

# 输出: analysis/data/energy_research/6groups_final/*.csv
# - group1_examples.csv (304样本)
# - group2_vulberta.csv (72样本)
# - group3_person_reid.csv (206样本)
# - group4_bug_localization.csv (90样本)
# - group5_mrt_oast.csv (72样本)
# - group6_resnet.csv (74样本)
```

### 2. DiBS因果发现流程

```bash
# 步骤1: 标准版DiBS分析（6组，无交互项）
conda activate causal-research  # ⚠️ 必须使用此环境！
python3 analysis/scripts/run_dibs_6groups_final.py

# 步骤2: 交互项版DiBS分析（6组，含交互项）
python3 analysis/scripts/run_dibs_6groups_interaction.py

# 输出: analysis/results/energy_research/questions_2_3_dibs/
# - {group_id}_causal_graph.npy  # 因果图矩阵
# - {group_id}_feature_names.json  # 特征名称
# - {group_id}_dibs_results.json  # 完整结果
```

**最优配置**（run_dibs_6groups_final.py）:
```python
OPTIMAL_CONFIG = {
    "alpha_linear": 0.05,      # DiBS默认值
    "beta_linear": 0.1,        # 低无环约束，允许更多边探索
    "n_particles": 20,         # 粒子数
    "tau": 1.0,                # Gumbel-softmax温度
    "n_steps": 5000,           # 迭代步数
}
```

### 3. 因果边处理流程 ⭐⭐⭐

```bash
# 步骤1: 提取因果边到CSV
python3 analysis/scripts/extract_dibs_edges_to_csv.py

# 输出: analysis/results/energy_research/data/interaction/threshold/
# - group1_examples_causal_edges.csv
# - ... (共6个文件)

# 步骤2: 白名单过滤因果边
python3 analysis/scripts/filter_causal_edges_by_whitelist.py

# 输出: analysis/results/energy_research/data/interaction/whitelist/
# - group1_examples_causal_edges_whitelist.csv (43条边)
# - group2_vulberta_causal_edges_whitelist.csv (35条边)
# - group3_person_reid_causal_edges_whitelist.csv (50条边)
# - group4_bug_localization_causal_edges_whitelist.csv (40条边)
# - group5_mrt_oast_causal_edges_whitelist.csv (40条边)
# - group6_resnet_causal_edges_whitelist.csv (19条边)
# 总计: 227条边 (539条原始边 → 42.1%保留率)

# 步骤3: 可视化因果图
python3 analysis/scripts/visualize_dibs_causal_graphs.py

# 输出: analysis/results/energy_research/causal_graph_visualizations/
# - {group_id}_causal_graph.png
```

### 4. 验证分析流程

```bash
# 回归分析验证DiBS发现
python3 analysis/scripts/validate_dibs_with_regression.py

# 中介效应分析（问题3）
python3 analysis/scripts/mediation_analysis_question3.py
```

---

## 🎯 因果边白名单规则 v1.1

**文档**: [docs/technical_reference/CAUSAL_EDGE_WHITELIST_SUMMARY.md](docs/technical_reference/CAUSAL_EDGE_WHITELIST_SUMMARY.md)

### 允许的16种因果边类型

| # | 规则组 | Source | Target | 研究问题 | 示例 |
|---|--------|--------|--------|----------|------|
| **规则组1: 超参数主效应** | | | | | |
| 1 | Q1 | hyperparam | energy | Q1 | batch_size → cpu_total_joules |
| 2 | Q1 | hyperparam | mediator | Q1 | batch_size → gpu_avg_watts |
| 3 | Q2 | hyperparam | performance | Q2 | batch_size → test_accuracy |
| **规则组2: 交互项调节效应** | | | | | |
| 4 | Q1 | interaction | energy | Q1 | batch_size_x_parallel → cpu_total_joules |
| 5 | Q1 | interaction | mediator | Q1 | batch_size_x_parallel → gpu_avg_watts |
| 6 | Q2 | interaction | performance | Q2 | batch_size_x_parallel → test_accuracy |
| **规则组3: 中间变量中介效应** | | | | | |
| 7 | Q3 | mediator | energy | Q3 | gpu_avg_watts → gpu_total_joules |
| 8 | Q3 | mediator | mediator | Q3 | gpu_temp_avg → gpu_avg_watts |
| 9 | **Q2/Q3** | **mediator** | **performance** | **Q2/Q3** | **gpu_temp_avg → test_accuracy** ⭐ |
| 10 | Q3 | energy | energy | Q3 | cpu_total_joules → cpu_pkg_joules |
| **规则组4: 控制变量影响** | | | | | |
| 11 | - | control | energy | - | model_mnist_ff → cpu_total_joules |
| 12 | - | control | mediator | - | model_mnist_ff → gpu_avg_watts |
| 13 | - | control | performance | - | model_mnist_ff → test_accuracy |
| 14 | - | mode | energy | - | is_parallel → cpu_total_joules |
| 15 | - | mode | mediator | - | is_parallel → gpu_avg_watts |
| 16 | - | mode | performance | - | is_parallel → test_accuracy |

**⭐ v1.1新增**: 第9条规则 `mediator → performance` 支持RQ2间接因果路径分析

### 禁止的因果边（黑名单）

❌ **反因果方向**:
- `performance → hyperparam` - 性能不能改变超参数
- `energy → hyperparam` - 能耗不能改变超参数
- `mediator → hyperparam` - 中间变量不能改变超参数

❌ **实验设计变量作为结果**:
- `* → control` - 模型选择不能被其他变量改变
- `* → mode` - 并行模式不能被其他变量改变

❌ **无意义边**:
- `hyperparam → hyperparam` - 超参数独立设定
- `X → X` (自循环) - 变量不能影响自身

❌ **反直觉关系**:
- `performance → energy` - 性能不应影响能耗
- `energy → mediator` - ⚠️ 明确禁止（防止路径污染）

---

## 🔬 三个核心研究问题

### 问题1: 超参数对能耗的影响（方向和大小）🔬 **[当前进行中]**

**研究目标**:
- 识别哪些超参数显著影响GPU/CPU能耗
- 量化每个超参数变化1单位时，能耗变化多少焦耳
- 区分不同任务类型的超参数效应差异

**分析方法**:
- 任务组分层回归（方案A'优化版，6组）
- 随机森林特征重要性
- 因果森林（heterogeneous treatment effects）

**数据分组**:
- group1_examples (304样本): batch_size, epochs, learning_rate, seed
- group2_vulberta (72样本): epochs, learning_rate, weight_decay, seed
- group3_person_reid (206样本): dropout, epochs, learning_rate, seed
- group4_bug_localization (90样本): alpha, kfold, max_iter, seed
- group5_mrt_oast (72样本): dropout, epochs, learning_rate, weight_decay, seed
- group6_resnet (74样本): epochs, learning_rate, weight_decay, seed

**详细方案**: [docs/current_plans/QUESTION1_REGRESSION_ANALYSIS_PLAN.md](docs/current_plans/QUESTION1_REGRESSION_ANALYSIS_PLAN.md)

---

### 问题2: 能耗和性能之间的权衡关系 ⏳

**研究目标**:
- 检测是否存在"能耗 vs 性能"的Pareto权衡
- 识别同时影响能耗和性能的超参数
- 量化权衡强度

**分析方法**:
- Pareto前沿分析
- 多目标回归分析
- 权衡检测算法（论文Algorithm 1）

**当前进度**: ⏳ 待问题1完成后执行

---

### 问题3: 中间变量的中介效应 ⏳

**研究目标**:
- 识别超参数通过哪些中介变量影响能耗
- 量化直接效应 vs 间接效应
- 示例：learning_rate → GPU利用率 → GPU能耗

**中介变量候选**:
- `gpu_util_avg` - GPU利用率（主中介变量）
- `gpu_temp_max` - 最高温度（散热压力）
- `cpu_pkg_ratio` - CPU计算能耗比
- `duration_seconds` - 训练时长

**当前进度**: ⏳ 待问题1完成后执行

**DiBS结果**: [docs/technical_reference/DIBS_VERIFICATION_REPORT_20260116.md](docs/technical_reference/DIBS_VERIFICATION_REPORT_20260116.md)

---

## 📊 DiBS分析结果摘要

### 6任务组DiBS分析完成 (2026-01-05)

**分析结果**:
- ✅ **6/6组DiBS分析成功**
- ✅ **825条总边检测成功**（包含交互项）
- ✅ **白名单过滤后227条边**（42.1%保留率）

**各研究问题边数分布**:
| 研究问题 | 边数 | 占比 | 说明 |
|---------|------|------|------|
| Q1超参数主效应 | 16条 | 7.0% | hyperparam → energy/mediator |
| Q1交互项调节 | 25条 | 11.0% | interaction → energy/mediator |
| Q2性能效应 | 29条 | 12.8% | hyperparam/interaction/mediator → performance |
| Q3中介效应 | 116条 | 51.1% | mediator → energy/mediator, energy → energy |
| 控制变量效应 | 23条 | 10.1% | control/mode → * |
| **总计** | **227条** | **100%** | - |

**关键发现**:
- ✅ Q3中介效应边数最多（51.1%），符合预期（能耗生成机制复杂）
- ✅ Q2性能效应边数合理（29条），包含直接和间接路径
- ✅ Q1超参数和交互项边数较少（41条），说明直接效应有限

**详细报告**: [docs/technical_reference/DIBS_VERIFICATION_REPORT_20260116.md](docs/technical_reference/DIBS_VERIFICATION_REPORT_20260116.md)

---

## ⚙️ 环境配置

### conda环境

**重要**: DiBS分析需要专用conda环境！

```bash
# 激活causal-research环境（已安装DiBS）
conda activate causal-research

# 或使用完整路径
/home/green/miniconda3/envs/causal-research/bin/python script.py
```

⚠️ **注意**: base环境没有安装DiBS，会导致分析失败！

### 安装依赖

```bash
# 创建新环境（如果需要）
conda create -n causal-research python=3.9
conda activate causal-research

# 安装依赖
pip install -r analysis/requirements.txt

# 或使用environment.yaml
conda env update -f analysis/environment.yaml
```

---

## 📚 文档导航

### 核心文档

| 文档 | 用途 | 优先级 |
|------|------|--------|
| [docs/INDEX.md](docs/INDEX.md) | 文档总索引 | ⭐⭐⭐⭐⭐ |
| [docs/technical_reference/CAUSAL_EDGE_WHITELIST_SUMMARY.md](docs/technical_reference/CAUSAL_EDGE_WHITELIST_SUMMARY.md) | 白名单v1.1总结 | ⭐⭐⭐⭐⭐ |
| [docs/current_plans/QUESTION1_REGRESSION_ANALYSIS_PLAN.md](docs/current_plans/QUESTION1_REGRESSION_ANALYSIS_PLAN.md) | 问题1方案 | ⭐⭐⭐⭐⭐ |
| [docs/current_plans/QUESTIONS_2_3_DIBS_ANALYSIS_PLAN.md](docs/current_plans/QUESTIONS_2_3_DIBS_ANALYSIS_PLAN.md) | 问题2/3方案 | ⭐⭐⭐⭐ |

### 关键报告

| 报告 | 内容 | 日期 |
|------|------|------|
| [DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md](docs/technical_reference/DIBS_PARAMETER_TUNING_ANALYSIS.md) | DiBS参数调优成功 | 2026-01-05 |
| [QUESTIONS_2_3_DIBS_COMPLETE_REPORT_20260105.md](docs/technical_reference/DIBS_VERIFICATION_REPORT_20260116.md) | 问题2/3 DiBS分析 | 2026-01-05 |
| [DIBS_EDGES_CSV_QUALITY_VERIFICATION.md](docs/technical_reference/DIBS_EDGES_CSV_QUALITY_VERIFICATION.md) | 因果边CSV质量验证 | 2026-01-16 |

### 历史参考

历史报告已归档或移除。当前可用的技术文档请查看 [docs/technical_reference/](docs/technical_reference/)。

---

## 🔧 工具脚本使用频率预测

### 高频使用 ⭐⭐⭐ (因果分析阶段每日)

| 脚本 | 原因 |
|------|------|
| run_dibs_6groups_final.py | 核心DiBS分析，6个任务组 |
| extract_dibs_edges_to_csv.py | 提取因果边，每次DiBS分析后必用 |
| filter_causal_edges_by_whitelist.py | 过滤因果边，提取后必用 |
| validate_dibs_with_regression.py | 验证DiBS发现，核心分析 |
| check_dibs_progress.py | 检查分析进度 |

### 中频使用 ⭐⭐ (每周)

| 脚本 | 原因 |
|------|------|
| visualize_dibs_causal_graphs.py | 生成因果图可视化 |
| mediation_analysis_question3.py | 问题3中介分析 |
| run_dibs_6groups_interaction.py | 交互项版DiBS分析 |
| dibs_parameter_sweep.py | 参数调优（调试时） |
| generate_6groups_final.py | 数据更新后重新生成 |

---

## 🚀 快速开始

### 新用户

1. **环境设置**: 激活causal-research环境
2. **数据准备**: 运行generate_6groups_final.py
3. **DiBS分析**: 运行run_dibs_6groups_final.py
4. **边提取**: 运行extract_dibs_edges_to_csv.py
5. **白名单过滤**: 运行filter_causal_edges_by_whitelist.py

### 理解代码

1. **整体流程**: 阅读 [docs/INDEX.md](docs/INDEX.md)
2. **白名单规则**: 阅读 [docs/technical_reference/CAUSAL_EDGE_WHITELIST_SUMMARY.md](docs/technical_reference/CAUSAL_EDGE_WHITELIST_SUMMARY.md)
3. **问题方案**: 阅读 [docs/current_plans/QUESTION1_REGRESSION_ANALYSIS_PLAN.md](docs/current_plans/QUESTION1_REGRESSION_ANALYSIS_PLAN.md)

### 应用到新数据集

参考主项目文档:
- [../../docs/DATA_MASTER_GUIDE.md](../../docs/DATA_MASTER_GUIDE.md) - 数据使用主指南
- [../../CLAUDE.md](../../CLAUDE.md) - 项目快速指南

---

## 📊 项目里程碑

### 2026-01-23: README更新
- ✅ 更新脚本目录结构（精简至20个核心脚本）
- ✅ 同步DiBS和白名单最新状态
- ✅ 更新研究问题进度

### 2026-01-20: 白名单v1.1完成
- ✅ 添加 `mediator → performance` 规则
- ✅ 成功过滤6组DiBS数据（539条 → 227条）
- ✅ 支持RQ2间接因果路径分析

### 2026-01-16: DiBS边CSV质量验证
- ✅ 数据完整性验证
- ✅ 脚本正确性验证
- ✅ 文档准确性验证

### 2026-01-05: DiBS参数调优成功 ⭐⭐⭐⭐⭐
- ✅ 11个实验全部成功
- ✅ 检测到23条强边
- ✅ 找到最优配置

---

## 📞 获取帮助

### 文档导航

1. **快速开始**: 本文档（README.md）
2. **文档索引**: docs/INDEX.md
3. **白名单总结**: docs/technical_reference/CAUSAL_EDGE_WHITELIST_SUMMARY.md
4. **问题方案**: docs/current_plans/QUESTION1_REGRESSION_ANALYSIS_PLAN.md

### 常见问题

如果遇到问题，请依次检查：
1. 本文档的"环境配置"章节
2. docs/INDEX.md中的"常见问题"
3. 相关的专题文档

---

**维护者**: Green
**最后更新**: 2026-01-23
**版本**: v2.0 (精简优化版)
