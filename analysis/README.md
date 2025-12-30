# ASE 2023论文因果推断方法复现

## 项目简介

这是论文 *"Causality-Aided Trade-off Analysis for Machine Learning Fairness" (ASE 2023)* 的**因果推断方法复现**。

**项目状态**:
- ✅ **Adult数据集完整因果分析成功** (2025-12-21)
- ✅ **能耗数据方法对比测试完成** (2025-12-26) ⭐⭐⭐
- ✅ **能耗数据研究问题方法推荐完成** (2025-12-28) ⭐⭐⭐

**综合复现度**: **90%** (4.5/5) - 首次完成端到端流程

**能耗数据结论**:
- DiBS因果发现不适用（0边，5大失败原因已分析）
- 推荐使用：回归分析（R²=0.999）、Pareto分析、中介效应分析
- 替代因果方法：中介效应分析和因果森林成功率95%

## 📚 重要文档

### 📖 文档总索引
- **docs/INDEX.md** - 所有文档的总索引和导航 ⭐

### 核心技术文档
- **docs/CODE_WORKFLOW_EXPLAINED.md** - 代码流程、算法原理和性能分析
- **docs/MIGRATION_GUIDE.md** - 应用到新数据集的完整指南

### 快速开始
- **docs/guides/ENVIRONMENT_SETUP.md** - 环境配置
- **docs/guides/REPLICATION_QUICK_START.md** - 快速复现

### 最新实验报告
- **docs/reports/RESEARCH_QUESTIONS_METHOD_RECOMMENDATIONS_20251228.md** - **能耗数据3个研究问题的方法推荐**（超参数影响、权衡关系、中介效应）⭐⭐⭐
- **docs/reports/CAUSAL_METHODS_COMPARISON_20251228.md** - **因果分析方法对比**（DiBS vs 9种替代方法）⭐⭐⭐
- **docs/reports/METHOD_COMPARISON_REPORT_20251226.md** - **能耗数据方法对比完整报告**（5方法测试，DiBS vs 推荐方法，R²=0.999）⭐⭐⭐
- **docs/reports/DIBS_FINAL_FAILURE_REPORT_20251226.md** - DiBS最终失败报告（5大失败原因系统性分析）⭐⭐⭐
- **docs/reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md** - Adult数据集完整分析报告（61.4分钟，6条因果边，4条显著）⭐

## 🚀 快速开始

### 1. 查看完整文档

```bash
# 查看文档总索引
cat docs/INDEX.md

# 或查看核心技术文档
cat docs/CODE_WORKFLOW_EXPLAINED.md
```

### 2. 环境配置

详细配置说明见 [docs/guides/ENVIRONMENT_SETUP.md](docs/guides/ENVIRONMENT_SETUP.md)

```bash
# 激活conda环境
conda activate fairness

# 或创建新环境
conda create -n fairness python=3.9
conda activate fairness

# 安装依赖
pip install -r requirements.txt
```

### 3. 运行Adult数据集分析

```bash
# 运行完整的Adult数据集因果分析（约60分钟，GPU加速）
bash scripts/experiments/run_adult_analysis.sh

# 或者在后台运行
nohup bash scripts/experiments/run_adult_analysis.sh > adult_analysis.log 2>&1 &

# 监控进度
bash scripts/utils/monitor_progress.sh
```

**预期输出**:
- 10个配置训练完成
- DiBS学习因果图（6条边）
- DML估计因果效应（4条显著）
- 结果保存到 `data/` 和 `results/` 目录

### 4. 应用到新数据集

完整指南见 [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)

## 项目结构

```
analysis/
├── README.md                    # 本文件
├── REORGANIZATION_PLAN.md       # 项目重组方案
├── config.py                    # 配置参数
├── requirements.txt             # 依赖清单
│
├── docs/                        # 📚 文档目录
│   ├── INDEX.md                 # ⭐ 文档总索引（必读）
│   ├── CODE_WORKFLOW_EXPLAINED.md    # 代码流程详解
│   ├── MIGRATION_GUIDE.md            # 迁移指南
│   ├── guides/                  # 使用指南
│   │   ├── ENVIRONMENT_SETUP.md
│   │   ├── REPLICATION_QUICK_START.md
│   │   ├── USAGE_GUIDE_FOR_NEW_RESEARCH.md
│   │   └── IMPROVEMENT_GUIDE.md
│   └── reports/                 # 实验报告
│       ├── ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md  ⭐ 最新
│       ├── ADULT_DATASET_VALIDATION_REPORT.md
│       ├── LARGE_SCALE_EXPERIMENT_REPORT.md
│       └── archives/            # 归档报告（13个）
│
├── scripts/                     # 🔧 脚本目录
│   ├── demos/                   # 演示脚本
│   │   ├── demo_quick_run.py
│   │   ├── demo_large_scale.py
│   │   ├── demo_adult_dataset.py
│   │   └── demo_adult_full_analysis.py
│   ├── experiments/             # 实验脚本
│   │   └── run_adult_analysis.sh
│   ├── utils/                   # 工具脚本
│   │   ├── monitor_progress.sh
│   │   └── activate_env.sh
│   └── testing/                 # 测试脚本
│       ├── test_dibs_quick.py
│       └── run_tests.py
│
├── logs/                        # 📊 日志目录
│   ├── experiments/             # 实验日志
│   ├── demos/                   # 演示日志
│   └── status/                  # 状态文件
│
├── utils/                       # 核心模块
│   ├── causal_discovery.py     # DiBS因果图学习
│   ├── causal_inference.py     # DML因果推断
│   ├── tradeoff_detection.py   # 权衡检测
│   ├── model.py                # 神经网络模型
│   ├── metrics.py              # 指标计算
│   ├── fairness_methods.py     # 公平性方法
│   └── aif360_utils.py         # AIF360工具
│
├── tests/                      # 测试套件
│   ├── test_units.py           # 单元测试
│   └── test_integration.py     # 集成测试
│
├── data/                       # 数据目录（运行后生成）
│   └── adult_training_data.csv # Adult数据集训练数据
│
└── results/                    # 结果目录（运行后生成）
    ├── adult_causal_graph.npy  # 因果图
    ├── adult_causal_edges.pkl  # 因果边
    └── adult_data_checkpoint.pkl # 数据检查点
```

## 🎯 主要成就

### 能耗数据方法对比测试 (2025-12-26) ⭐⭐⭐

✅ **系统性测试5种分析方法，找到最适合能耗数据的方法**

**测试结果对比**:

| 方法 | 成功率 | 耗时 | 核心指标 | 推荐等级 |
|------|--------|------|---------|---------|
| **相关性分析** | ✅ 100% | 0.01秒 | r=0.931 (GPU功率↔温度) | ⭐⭐⭐⭐⭐ |
| **回归分析** | ✅ 100% | 0.42秒 | **R²=0.999** (随机森林) | ⭐⭐⭐⭐⭐ |
| **偏相关分析** | ✅ 100% | 0.09秒 | r=0.925 (控制后CPU↔GPU能耗) | ⭐⭐⭐ |
| **互信息分析** | ✅ 100% | 0.06秒 | MI=1.951 (GPU利用率) | ⭐⭐⭐⭐ |
| **DiBS** | ❌ 失败 | 14.3分钟 | 0条边 | ❌ |

**核心发现**:
- **GPU利用率** 驱动76.9%的能耗变化（绝对主导）
- **GPU温度** 贡献16.9%
- 可99.9%准确预测GPU功率（R²=0.999）
- DiBS完全失败原因：能耗数据缺乏明确因果链

**结论**: 能耗数据适合**预测建模**（相关性+回归），不适合**因果推断**（DiBS）

详细报告: [METHOD_COMPARISON_REPORT_20251226.md](docs/reports/METHOD_COMPARISON_REPORT_20251226.md)

---

### Adult数据集因果分析 (2025-12-21)

✅ **首次完成Adult数据集端到端因果分析**
- **运行时间**: 61.4分钟（GPU加速）
- **配置数**: 10个（2方法 × 5 alpha值）
- **因果边检测**: 6条高置信度因果边
- **统计显著**: 4条边的因果效应统计显著
- **复现质量**: 90% (4.5/5)

### 关键发现

1. **过拟合证据**: Tr_F1 → Te_Acc, ATE = -0.052
   - 训练F1提高1单位，测试准确率降低5.2%
   - 验证了训练性能与泛化性能的权衡

2. **DiBS性能突破**: 从超时(>1小时) → 成功(1.6分钟)
   - 速度提升 >97%
   - 成功学习因果图结构

3. **DML因果推断**: 4/6边统计显著
   - 提供置信区间的可靠因果效应估计
   - 验证了论文的方法论

详细报告: [ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](docs/reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md)

## 技术栈

### 核心技术
- **DiBS** (Differentiable Bayesian Structure Learning): 因果图学习
- **DML** (Double Machine Learning): 因果效应估计
- **PyTorch**: 神经网络训练
- **JAX**: DiBS优化
- **EconML**: 因果推断库
- **AIF360**: 公平性方法

### 硬件要求
- **推荐**: GPU (NVIDIA RTX 3080或更高)
- **最低**: CPU (8GB+ RAM)
- **存储**: 2GB

## 引用

如果使用本代码，请引用原论文：

```bibtex
@inproceedings{ji2023causality,
  title={Causality-Aided Trade-off Analysis for Machine Learning Fairness},
  author={Ji, Zhenlan and Ma, Pingchuan and Wang, Shuai and Li, Yanhui},
  booktitle={2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  year={2023}
}
```

## 许可

本精简版代码仅用于学术研究和教育目的。

## 联系

如有问题，请参考：
- 原论文代码: https://anonymous.4open.science/r/CTF-47BF
- 补充材料: https://sites.google.com/view/causal-tradeoff-fairness/home
