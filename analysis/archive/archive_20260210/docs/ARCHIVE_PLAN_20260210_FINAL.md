# analysis目录归档方案（最终版）

**日期**: 2026-02-10
**范围**: 仅限 `analysis/` 目录下的数据、结果和脚本
**归档位置**: `analysis/archive/archived_20260210/`
**状态**: 待用户审核

---

## 执行摘要

- **保留**: 8个脚本 + 3个数据目录 + 5个结果目录
- **归档**: 37个脚本 + 4个数据目录 + 5个结果目录
- **风险**: 低（使用mv命令，可恢复）

---

## 1. 保留清单 ✅ KEEP

### 1.1 数据文件（3个目录）

| 目录 | 用途 | 创建日期 |
|------|------|---------|
| `data/energy_research/6groups_global_std/` | **全局标准化数据**（删除0列+超参数合并） | 2026-01-30 |
| `data/energy_research/6groups_dibs_ready/` | DiBS预处理数据 | 2026-01-30 |
| `data/energy_research/6groups_dibs_ready_v1_backup/` | 预处理备份 | 2026-02-10 |

### 1.2 分析结果（5个目录）

| 目录 | 用途 | 更新日期 |
|------|------|---------|
| `results/energy_research/data/global_std/` | **DiBS因果图**（6组） | 2026-02-10 |
| `results/energy_research/data/global_std_dibs_ate/` | **ATE计算**（6组） | 2026-02-03 |
| `results/energy_research/tradeoff_detection_global_std/` | **权衡检测**（61个） | 2026-02-10 |
| `results/energy_research/rq_analysis/` | 研究问题分析+可视化 | 2026-02-07 |
| `results/energy_research/archive/` | 已有归档（不动） | - |

### 1.3 分析脚本（8个文件）

| 脚本 | 用途 | 更新日期 |
|------|------|---------|
| `run_dibs_6groups_global_std.py` | DiBS训练（13000步） | 2026-01-30 |
| `validate_dibs_results.py` | DiBS结果验证 | 2026-01-06 |
| `compute_ate_dibs_global_std.py` | ATE计算（DML） | 2026-02-03 |
| `compute_ate_global_std.py` | ATE计算（备用） | 2026-01-30 |
| `run_algorithm1_tradeoff_detection_global_std.py` | 权衡检测 | 2026-02-03 |
| `preprocess_for_dibs_global_std.py` | DiBS数据预处理 | 2026-01-30 |
| `sensitivity_analysis_global_std.py` | 敏感性分析 | 2026-01-31 |
| `visualize_dibs_causal_graphs.py` | 可视化 | 2026-01-06 |

---

## 2. 归档清单 📦 ARCHIVE

### 2.1 数据文件（4个目录）

```
data/energy_research/
├── 6groups_final/              → 归档（旧版）
├── 6groups_interaction/        → 归档（交互项版本）
├── stratified/                 → 归档（实验性）
└── archive/                    → 归档（旧归档）
```

### 2.2 分析结果（5个目录）

```
results/energy_research/
├── archived_data/                        → 归档（旧版）
├── interaction_tradeoff_verification/    → 归档（交互项）
├── tradeoff_detection_interaction_based/ → 归档（交互项）
├── stratified/                           → 归档（实验性）
└── reports/                              → 归档（旧版报告）
```

### 2.3 分析脚本（37个文件）

#### DiBS相关（6个）
```
scripts/
├── run_dibs_6groups_final.py         → 归档（旧版）
├── run_dibs_6groups_interaction.py   → 归档（交互项）
├── run_dibs_for_questions_2_3.py     → 归档（旧版）
├── run_dibs_on_new_6groups.py        → 归档（旧版）
├── check_dibs_interaction_config.py  → 归档（工具）
└── check_dibs_progress.py            → 归档（工具）
```

#### ATE相关（7个）
```
scripts/
├── compute_ate_for_whitelist.py      → 归档（旧版）
├── compute_ate_whitelist.py          → 归档（旧版）
├── analyze_ate_data_quality.py       → 归档（分析）
├── check_ate_data_quality.py         → 归档（检查）
├── check_ate_quality.py              → 归档（检查）
└── validate_dibs_with_regression.py  → 归档（验证）
```

#### 权衡相关（4个）
```
scripts/
├── run_algorithm1_tradeoff_detection.py          → 归档（旧版）
├── analyze_tradeoff_results.py                   → 归档（分析）
├── diagnose_zero_tradeoff_groups.py              → 归档（诊断）
└── verify_interaction_tradeoffs.py               → 归档（验证）
```

#### 数据处理（7个）
```
scripts/
├── create_global_standardized_data.py              → 归档（工具）
├── generate_6groups_final.py                       → 归档（旧版）
├── backfill_hyperparameters_from_models_config.py  → 归档（工具）
├── filter_causal_edges_by_whitelist.py             → 归档（工具）
├── validate_dibs_readiness.py                      → 归档（验证）
├── test_preprocess_stratified_data.py             → 归档（测试）
└── verify_5groups_data.py                          → 归档（验证）
```

#### 其他分析（13个）
```
scripts/
├── rq1_analysis.py                        → 归档（RQ分析）
├── rq2_analysis.py                        → 归档（RQ分析）
├── rq3_analysis.py                        → 归档（RQ分析）
├── mediation_analysis_question3.py        → 归档（中介分析）
├── sensitivity_analysis_global_std.py     → 保留（已在保留列表）
├── compare_dibs_results.py                → 归档（比较）
├── compare_standardization_methods.py     → 归档（比较）
├── convert_dibs_to_csv.py                 → 归档（转换）
├── extract_dibs_edges_to_csv.py           → 归档（提取）
├── extract_from_json_with_defaults.py     → 归档（工具）
├── diagnose_group2_data.py                → 归档（诊断）
├── diagnose_missing_patterns.py           → 归档（诊断）
├── dibs_parameter_sweep.py                → 归档（实验）
├── config.py                              → 归档（配置）
└── config_energy.py                       → 归档（配置）
```

---

## 3. 归档目录结构

```
archive/archived_20260210/
├── README.md                    # 归档说明
├── data/
│   ├── 6groups_final/
│   ├── 6groups_interaction/
│   ├── stratified/
│   └── archive/
├── results/
│   ├── archived_data/
│   ├── interaction_tradeoff_verification/
│   ├── tradeoff_detection_interaction_based/
│   ├── stratified/
│   └── reports/
└── scripts/
    ├── dibs/                    # 6个DiBS脚本
    ├── ate/                     # 7个ATE脚本
    ├── tradeoff/                # 4版权衡脚本
    ├── data_processing/         # 7个数据处理脚本
    └── other/                   # 13个其他脚本
```

---

## 4. 执行命令

### 4.1 创建归档目录
```bash
mkdir -p archive/archived_20260210/{data,results,scripts/{dibs,ate,tradeoff,data_processing,other}}
```

### 4.2 归档数据文件
```bash
# 数据文件
mv data/energy_research/6groups_final archive/archived_20260210/data/
mv data/energy_research/6groups_interaction archive/archived_20260210/data/
mv data/energy_research/stratified archive/archived_20260210/data/
mv data/energy_research/archive archive/archived_20260210/data/
```

### 4.3 归档结果文件
```bash
# 结果文件
mv results/energy_research/archived_data archive/archived_20260210/results/
mv results/energy_research/interaction_tradeoff_verification archive/archived_20260210/results/
mv results/energy_research/tradeoff_detection_interaction_based archive/archived_20260210/results/
mv results/energy_research/stratified archive/archived_20260210/results/
mv results/energy_research/reports archive/archived_20260210/results/
```

### 4.4 归档脚本文件
```bash
# DiBS脚本
mv scripts/run_dibs_6groups_final.py archive/archived_20260210/scripts/dibs/
mv scripts/run_dibs_6groups_interaction.py archive/archived_20260210/scripts/dibs/
mv scripts/run_dibs_for_questions_2_3.py archive/archived_20260210/scripts/dibs/
mv scripts/run_dibs_on_new_6groups.py archive/archived_20260210/scripts/dibs/
mv scripts/check_dibs_interaction_config.py archive/archived_20260210/scripts/dibs/
mv scripts/check_dibs_progress.py archive/archived_20260210/scripts/dibs/

# ATE脚本
mv scripts/compute_ate_for_whitelist.py archive/archived_20260210/scripts/ate/
mv scripts/compute_ate_whitelist.py archive/archived_20260210/scripts/ate/
mv scripts/analyze_ate_data_quality.py archive/archived_20260210/scripts/ate/
mv scripts/check_ate_data_quality.py archive/archived_20260210/scripts/ate/
mv scripts/check_ate_quality.py archive/archived_20260210/scripts/ate/
mv scripts/validate_dibs_with_regression.py archive/archived_20260210/scripts/ate/

# 权衡脚本
mv scripts/run_algorithm1_tradeoff_detection.py archive/archived_20260210/scripts/tradeoff/
mv scripts/analyze_tradeoff_results.py archive/archived_20260210/scripts/tradeoff/
mv scripts/diagnose_zero_tradeoff_groups.py archive/archived_20260210/scripts/tradeoff/
mv scripts/verify_interaction_tradeoffs.py archive/archived_20260210/scripts/tradeoff/

# 数据处理脚本
mv scripts/create_global_standardized_data.py archive/archived_20260210/scripts/data_processing/
mv scripts/generate_6groups_final.py archive/archived_20260210/scripts/data_processing/
mv scripts/backfill_hyperparameters_from_models_config.py archive/archived_20260210/scripts/data_processing/
mv scripts/filter_causal_edges_by_whitelist.py archive/archived_20260210/scripts/data_processing/
mv scripts/validate_dibs_readiness.py archive/archived_20260210/scripts/data_processing/
mv scripts/test_preprocess_stratified_data.py archive/archived_20260210/scripts/data_processing/
mv scripts/verify_5groups_data.py archive/archived_20260210/scripts/data_processing/

# 其他脚本
mv scripts/rq1_analysis.py archive/archived_20260210/scripts/other/
mv scripts/rq2_analysis.py archive/archived_20260210/scripts/other/
mv scripts/rq3_analysis.py archive/archived_20260210/scripts/other/
mv scripts/mediation_analysis_question3.py archive/archived_20260210/scripts/other/
mv scripts/compare_dibs_results.py archive/archived_20260210/scripts/other/
mv scripts/compare_standardization_methods.py archive/archived_20260210/scripts/other/
mv scripts/convert_dibs_to_csv.py archive/archived_20260210/scripts/other/
mv scripts/extract_dibs_edges_to_csv.py archive/archived_20260210/scripts/other/
mv scripts/extract_from_json_with_defaults.py archive/archived_20260210/scripts/other/
mv scripts/diagnose_group2_data.py archive/archived_20260210/scripts/other/
mv scripts/diagnose_missing_patterns.py archive/archived_20260210/scripts/other/
mv scripts/dibs_parameter_sweep.py archive/archived_20260210/scripts/other/
mv scripts/config.py archive/archived_20260210/scripts/other/
mv scripts/config_energy.py archive/archived_20260210/scripts/other/
```

---

## 5. 归档后验证

```bash
# 验证保留文件
ls -la data/energy_research/
ls -la results/energy_research/
ls -la scripts/*.py | wc -l  # 应该是8个

# 验证归档文件
tree archive/archived_20260210/
```

---

## 6. 风险控制

### 6.1 安全措施
- ✅ 使用 `mv` 命令（可恢复）
- ✅ 保留目录结构
- ✅ 分类清晰（data/results/scripts）
- ✅ 生成归档说明文档

### 6.2 回滚示例
```bash
# 如果需要恢复某个目录
mv archive/archived_20260210/data/6groups_final data/energy_research/

# 如果需要恢复某个脚本
mv archive/archived_20260210/scripts/dibs/run_dibs_6groups_final.py scripts/
```

---

## 7. 审核确认清单

请审核以下内容：

- [ ] **保留文件正确**：3个数据目录 + 5个结果目录 + 8个脚本
- [ ] **归档文件正确**：4个数据目录 + 5个结果目录 + 37个脚本
- [ ] **归档结构合理**：按类型分类（dibs/ate/tradeoff/other）
- [ ] **安全措施充分**：mv命令 + 可恢复

---

**方案制定**: Claude Code
**日期**: 2026-02-10
**状态**: ⏳ 待审核 - 请确认后执行
