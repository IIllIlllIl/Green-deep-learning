# DiBS失败尝试归档报告

**归档日期**: 2025-12-30
**执行者**: Claude
**归档原因**: DiBS因果图学习失败，转向回归分析方法

---

## 📊 归档统计

| 维度 | 数量 |
|------|------|
| **归档文件总数** | 148个 |
| **归档CSV文件** | 54个 |
| **归档脚本** | 16个 |
| **归档文档** | 6个 |
| **归档目录大小** | 4.9 MB |

---

## 📁 归档内容清单

### 1. 脚本文件（16个）

**数据处理流程脚本**:
- `stage0_data_validation.py` - 数据验证
- `stage1_hyperparam_unification.py` - 超参数统一
- `stage2_energy_mediators.py` - 能耗中介变量
- `stage3_task_grouping.py` - 任务分组
- `stage4_onehot_encoding.py` - One-Hot编码
- `stage5_variable_selection.py` - 变量选择
- `stage6_normalization.py` - 数据归一化
- `stage7_final_validation.py` - 最终验证

**DiBS演示脚本**:
- `demo_single_task_dibs.py` (v1)
- `demo_single_task_dibs_v2.py`
- `demo_single_task_dibs_v3.py`

**测试与检查脚本**:
- `test_dibs_quick.py` - DiBS快速测试
- `stage3_safety_check.py` - Stage3安全检查
- `stage4_safety_check.py` - Stage4安全检查
- `stage5_safety_check.py` - Stage5安全检查
- `analyze_all_stages_quality.py` - 全阶段质量分析

### 2. 数据文件（54个CSV）

**Stage0-7中间数据**:
- `stage0_validated.csv` (726行) - 验证后的原始数据
- `stage1_unified.csv` (726行) - 超参数统一后
- `stage2_mediators.csv` (726行) - 添加中介变量后
- `stage3_*.csv` (4个任务组) - 任务分组后
- `stage4_*.csv` (4个任务组) - One-Hot编码后
- `stage5_*.csv` (4个任务组) - 变量选择后
- `stage6_*.csv` (4个任务组) - 归一化后

**DiBS训练数据**（6组）:
- `training_data_image_classification_examples.csv` (219行)
- `training_data_image_classification_resnet.csv` (39行)
- `training_data_person_reid.csv` (116行)
- `training_data_vulberta.csv` (82行)
- `training_data_bug_localization.csv` (80行)
- `training_data_mrt_oast.csv` (58行)

**备份数据**:
- `processed.backup_4groups_20251224/` - 4组方案备份

### 3. 实验结果（7个目录）

- `6groups/` - 6组DiBS实验结果（v1）
- `6groups_v2/` - 6组DiBS实验结果（v2）
- `6groups_v3/` - 6组DiBS实验结果（v3）
- `method_comparison/` - 方法对比实验
- `task_specific/` - 任务特定分析
- `processed_original/` - 原始processed目录完整备份
- `training/` - 训练数据备份

### 4. 文档（6个）

**流程文档**:
- `CODE_WORKFLOW_EXPLAINED.md` - DiBS代码流程详解（61分钟完整过程）
- `DATA_PREPROCESSING_DECISIONS.md` - 数据预处理决策
- `DATA_QUALITY_REPORT_DETAILED_20251223.md` - 数据质量详细报告

**Stage报告**:
- `STAGE6_7_DATA_QUALITY_REPORT.md` - Stage6-7数据质量报告
- `STAGE8_SCREEN_RUNNING_GUIDE.md` - Stage8后台运行指南
- `STAGE8_TEST_VALIDATION_REPORT.md` - Stage8测试验证报告

---

## ✅ 保留的活跃内容

### 数据（保留）

**原始数据** ✅:
- `data/energy_research/raw/energy_data_original.csv` (727行 = 726数据 + 1 header)
  - 56列
  - 从主项目 `results/data.csv` 复制而来

**工作目录** ✅:
- `data/energy_research/processed/` - 空目录（准备新方案数据）
- `data/energy_research/experiments/` - 实验元数据（保留）

### 脚本（保留）

**核心工具** ✅（可能以后使用）:
- `utils/causal_discovery.py` - DiBS因果图学习工具
- `utils/causal_inference.py` - DML因果推断工具
- `utils/model.py` - 神经网络模型
- `utils/metrics.py` - 指标计算
- `utils/fairness_methods.py` - 公平性方法

**其他演示脚本** ✅:
- `scripts/demos/` 目录下的其他脚本（非DiBS相关）

### 文档（保留）

**新方案文档** ✅:
- `docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md` - **问题1回归分析方案** ⭐⭐⭐
- `docs/ENERGY_DATA_PROCESSING_PROPOSAL.md` - 能耗数据处理方案（历史参考）

**失败分析报告** ✅（重要历史记录）:
- `docs/reports/DIBS_FINAL_FAILURE_REPORT_20251226.md` - DiBS失败总结
- `docs/reports/6GROUPS_DIBS_ZERO_EDGES_DIAGNOSIS_20251226.md` - 0因果边诊断
- `docs/reports/CAUSAL_METHODS_COMPARISON_20251228.md` - 因果方法对比

**数据分析文档** ✅:
- `docs/reports/DATA_COMPARISON_OLD_VS_NEW_20251229.md` - 新旧数据对比
- `docs/reports/VARIABLE_EXPANSION_PLAN.md` - 变量扩展计划 v3.0
- `docs/COLUMN_USAGE_ANALYSIS.md` - 列使用率分析
- `docs/DATA_FILES_COMPARISON.md` - data.csv vs raw_data.csv对比

**其他文档** ✅:
- `docs/INDEX.md` - 文档总索引
- `docs/MIGRATION_GUIDE.md` - 数据迁移指南
- `docs/DATA_ISOLATION_MIGRATION_REPORT.md` - 数据隔离迁移报告
- 其他通用文档...

---

## 🎯 归档后的工作目录状态

### 目录结构（清理后）

```
analysis/
├── archived_dibs_attempts/      # 归档的DiBS尝试 ⭐ 新增
│   ├── scripts/                 # 16个脚本
│   ├── data/                    # 54个CSV + 备份
│   ├── results/                 # 7个实验结果目录
│   ├── docs/                    # 6个流程文档
│   └── README.md                # 归档说明
│
├── data/
│   ├── paper_replication/       # 论文复现数据（保留）
│   └── energy_research/
│       ├── raw/                 # 原始数据 ✅ 保留
│       │   └── energy_data_original.csv (727行)
│       ├── processed/           # 空目录 ✅ 准备新数据
│       └── experiments/         # 实验元数据 ✅ 保留
│
├── results/
│   ├── paper_replication/       # 论文复现结果（保留）
│   └── energy_research/         # 空目录 ✅ 准备新结果
│
├── scripts/
│   ├── demos/                   # 演示脚本（部分保留）
│   ├── experiments/             # 实验脚本（保留）
│   └── utils/                   # 工具脚本（保留）
│
├── utils/                       # 核心工具 ✅ 全部保留
│   ├── causal_discovery.py
│   ├── causal_inference.py
│   ├── model.py
│   ├── metrics.py
│   └── fairness_methods.py
│
├── docs/                        # 文档目录
│   ├── QUESTION1_REGRESSION_ANALYSIS_PLAN.md ⭐ 新方案
│   ├── reports/                 # 报告（部分归档，部分保留）
│   └── guides/                  # 指南（全部保留）
│
└── tests/                       # 测试（保留）
```

### 清理效果

| 维度 | 清理前 | 清理后 | 减少 |
|------|--------|--------|------|
| **scripts/下的脚本** | ~25个 | ~9个 | -16个 ✅ |
| **data/processed/下的文件** | 54个 | 0个 | -54个 ✅ |
| **results/energy_research/下的目录** | 7个 | 0个 | -7个 ✅ |
| **docs/下的DiBS流程文档** | 6个 | 0个 | -6个 ✅ |

---

## ⚠️ 重要警告

### ❌ 不要使用归档内容

1. **不要运行归档的脚本**:
   - 这些脚本是为DiBS设计的
   - 与新的回归分析方案不兼容
   - 可能产生错误或误导性结果

2. **不要使用归档的数据**:
   - 经过了DiBS特定的预处理
   - 包括归一化、One-Hot编码等
   - 不适合回归分析

3. **不要删除归档目录**:
   - 保留作为历史记录
   - 可供参考DiBS失败原因
   - 避免重复犯错

### ✅ 如何使用归档

**仅用于参考**:
- 了解DiBS失败的原因
- 学习数据预处理流程
- 避免重复尝试DiBS

**如需恢复**:
```bash
# 不推荐！仅在极特殊情况下使用
cp -r archived_dibs_attempts/scripts/* scripts/
cp -r archived_dibs_attempts/data/* data/energy_research/
```

---

## 📝 下一步行动

### 立即任务（问题1回归分析）

1. **创建新脚本** ⏳:
   - `scripts/backfill_hyperparameters.py` - 默认值回溯
   - `scripts/validate_backfilled_data.py` - 数据质量验证
   - `scripts/generate_regression_groups.py` - 生成6组数据
   - `scripts/run_group_regression.py` - 运行回归分析

2. **生成新数据** ⏳:
   - `data/energy_research/processed/group*.csv` (6个任务组)
   - 预期总行数: 633行（87.1%保留率）

3. **执行分析** ⏳:
   - 多元线性回归
   - 随机森林回归
   - 因果森林（可选）

**详细方案**: 查看 `docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md`

---

## 📚 相关文档

- [QUESTION1_REGRESSION_ANALYSIS_PLAN.md](../docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md) - 问题1回归分析方案 ⭐⭐⭐
- [DIBS_FINAL_FAILURE_REPORT_20251226.md](../docs/reports/DIBS_FINAL_FAILURE_REPORT_20251226.md) - DiBS失败原因总结
- [DATA_COMPARISON_OLD_VS_NEW_20251229.md](../docs/reports/DATA_COMPARISON_OLD_VS_NEW_20251229.md) - 新旧数据对比

---

## ✅ 归档确认

- [x] 16个DiBS脚本已归档
- [x] 54个CSV数据文件已归档
- [x] 7个实验结果目录已归档
- [x] 6个流程文档已归档
- [x] 原始数据完好保留（727行）
- [x] 核心工具保留（utils/）
- [x] 新方案文档创建（QUESTION1_REGRESSION_ANALYSIS_PLAN.md）
- [x] 工作目录清理完成

**归档状态**: ✅ 完成
**验证时间**: 2025-12-30 16:50
**归档目录**: `/home/green/energy_dl/nightly/analysis/archived_dibs_attempts/`

---

**维护者**: Green + Claude
**报告版本**: v1.0
**最后更新**: 2025-12-30
