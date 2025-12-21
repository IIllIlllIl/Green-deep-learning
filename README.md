# Mutation-Based Training Energy Profiler

自动化深度学习模型训练的超参数变异与能耗性能分析框架

**当前版本**: v4.7.11 (2025-12-19)
**状态**: ✅ Production Ready

---

## 项目概述

研究深度学习训练超参数对能耗和性能的影响。通过自动化变异超参数、监控能耗、收集性能指标,支持大规模实验研究。

### 核心功能

- ✅ **超参数变异** - 自动生成超参数变体（log-uniform/uniform分布）
- ✅ **能耗监控** - CPU (perf) + GPU (nvidia-smi),CPU误差<2%
- ✅ **并行训练** - 支持前台监控+后台负载的并行训练模式
- ✅ **完整元数据** - 并行实验记录完整前景+背景模型信息
- ✅ **去重机制** - 自动跳过重复实验，支持历史数据去重，区分并行/非并行模式
- ✅ **数据完整性** - CSV安全追加模式，防止数据覆盖丢失
- ✅ **离线训练** - 支持完全离线运行，避免网络依赖
- ✅ **快速验证** - 1-epoch配置，15-20分钟验证全部模型
- ✅ **结果组织** - 分层目录结构 + CSV汇总 + JSON详细数据
- ✅ **批量实验** - 配置文件支持复杂实验设计
- ✅ **分阶段执行** - 大规模实验分割成可管理的阶段

---

## 快速开始

### 1. 列出可用模型

```bash
python3 mutation.py --list
```

### 2. 运行单个实验

```bash
# 基本用法
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs,learning_rate -n 3
```

### 3. 运行批量实验（推荐）

```bash
# 快速验证（15-20分钟，1 epoch）
HF_HUB_OFFLINE=1 python3 mutation.py -ec settings/11_models_quick_validation_1epoch.json

# 变异实验验证（1次变异，快速测试）
sudo -E python3 mutation.py -ec settings/mutation_validation_1x.json -g performance

# 完整变异实验（3次变异，完整数据）
sudo -E python3 mutation.py -ec settings/mutation_all_models_3x_dynamic.json -g performance

# 完整基线实验（9+小时）
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py -ec settings/11_models_sequential_and_parallel_training.json -g performance
```

### 4. 分阶段实验（大规模实验推荐）

```bash
# 阶段1-4: 已完成 ✓
# sudo -E python3 mutation.py -ec settings/stage1_nonparallel_completion.json
# sudo -E python3 mutation.py -ec settings/stage2_optimized_nonparallel_and_fast_parallel.json
# sudo -E python3 mutation.py -ec settings/stage3_4_merged_optimized_parallel.json

# 阶段7-8: 已完成 ✓ (2025-12-06~07)
# sudo -E python3 mutation.py -ec settings/stage7_nonparallel_fast_models.json
# sudo -E python3 mutation.py -ec settings/stage8_nonparallel_medium_slow_models.json

# 阶段最终: 所有剩余实验 (37.8h, 78实验) **[一次性完成]**
sudo -E python3 mutation.py -ec settings/stage_final_all_remaining.json

# 注: Stage11-13已合并为单一配置，Stage9-10已归档
```

---

## 支持的模型 (11个有效模型)

| 仓库 | 模型 | 超参数 | Epochs范围 | 实验数 |
|------|------|--------|-----------|--------|
| **examples** | mnist, mnist_rnn, mnist_ff, siamese | epochs, lr, batch_size, seed | [5, 15] | 179 |
| **Person_reID_baseline_pytorch** | densenet121, hrnet18, pcb | epochs, lr, dropout, seed | [30, 90] | 116 |
| **VulBERTa** | mlp | epochs, lr, weight_decay, seed | [5, 20] | 45 |
| **pytorch_resnet_cifar10** | resnet20 | epochs, lr, weight_decay, seed | [100, 300] | 39 |
| **MRT-OAST** | default | epochs, lr, dropout, weight_decay, seed | [5, 15] | 57 |
| **bug-localization** | default | max_iter, alpha, kfold, seed | - | 40 |
| **总计** | **11个模型** | **9个超参数** | - | **476** |

**重要说明**:
- ✅ 所有11个模型已100%完成实验目标 (90/90参数-模式组合)
- ❌ VulBERTa/cnn已移除 (训练代码未实现)
- ⏱️ **总运行时间**: 238.51小时 (9.94天) | 平均30分钟/实验
- 📊 详细模型定义: [docs/11_MODELS_FINAL_DEFINITION.md](docs/11_MODELS_FINAL_DEFINITION.md)
- 📈 超参数变异范围: [docs/MUTATION_RANGES_QUICK_REFERENCE.md](docs/MUTATION_RANGES_QUICK_REFERENCE.md)
- 📈 运行时间统计: [docs/results_reports/RUNTIME_STATISTICS_20251211.md](docs/results_reports/RUNTIME_STATISTICS_20251211.md)

---

## 结果输出

每次运行创建独立session,包含CSV汇总和详细JSON数据：

```
results/run_YYYYMMDD_HHMMSS/
├── summary.csv                    # 所有实验汇总
└── {repo}_{model}_{id}_parallel/  # 并行实验（或不带_parallel为顺序实验）
    ├── experiment.json            # 完整数据（超参数+性能+能耗）
    ├── training.log               # 训练日志
    ├── energy/                    # 能耗原始数据
    └── background_logs/           # 后台训练日志（仅并行实验）
```

**数据文件** (`results/` 目录):
- **raw_data.csv**: 合并后的原始数据（87列，676行） - **原始数据源** ⭐⭐⭐
  - 包含所有实验的原始数据（211老实验 + 465新实验）
  - 100%训练成功，91.1%能耗完整，91.1%性能数据完整
  - 验证报告: [scripts/validate_raw_data.py](scripts/validate_raw_data.py)
- **data.csv**: 统一性能指标数据（54列，676行） - **主分析文件** ⭐⭐⭐
  - 基于raw_data.csv生成，统一了并行/非并行字段
  - **列结构优化** (v4.7.11): 54列精简设计（从87列原始数据）
    - 删除2个空列: perf_accuracy, perf_eval_loss
    - 14个有效性能指标列（从16个减少）
    - 0个空列剩余
  - **性能指标合并** (v4.7.10): 统一命名提升填充率
    - 合并: `accuracy` → `test_accuracy`, `eval_loss` → `test_loss`
    - 填充率提升: test_accuracy +6.8%, test_loss +10.7%
    - 无数据丢失，100%实验目标达成维持
  - **数据质量评估** (v4.7.11): ⭐⭐⭐ 优秀 ✅
    - 数据一致性: 2704/2704检查点 (100.0%)
    - 数据完整性: 676行100%保留
    - 无冲突，无缺失，设计合理
  - 质量报告: [数据质量评估报告](docs/results_reports/DATA_CSV_QUALITY_ASSESSMENT_REPORT.md) ⭐⭐⭐
  - 合并报告: [性能指标合并完成报告](docs/results_reports/PERFORMANCE_METRICS_MERGE_COMPLETION_REPORT.md) ⭐⭐
  - 列优化: [备份清理与列优化报告](docs/results_reports/BACKUP_CLEANUP_AND_COLUMN_OPTIMIZATION_REPORT.md) ⭐⭐
  - 生成脚本: [scripts/create_unified_data_csv.py](scripts/create_unified_data_csv.py)
- **summary_old.csv**: 老实验数据（93列，211行） - 源数据，供参考
- **summary_new.csv**: 新实验数据（80列，265行） - 源数据，供参考
- **summary_archive/**: 过时的summary文件归档目录
  - 包含13个过时文件（summary_all.csv, summary_all_enhanced.csv等）
  - 归档说明: [results/archived/summary_archive/README_ARCHIVE.md](results/archived/summary_archive/README_ARCHIVE.md)
- **backup_archive_20251219/**: Phase 4-7备份归档目录
  - 保留9个关键备份（phase/fix/dedup等 + 最新2个时间戳备份）
  - 已清理11个过时中间备份（2025-12-19清理，节省2.8MB）
  - 归档说明: [results/backup_archive_20251219/README_ARCHIVE.md](results/backup_archive_20251219/README_ARCHIVE.md)

**数据格式**:
- **80列格式**: 优化的数据记录（背景训练使用默认值，监控全局能耗）
- **93列格式**: 历史数据格式（包含部分背景超参数，仅老实验使用）
- JSON详细数据: 包含完整的超参数、性能指标、能耗数据
- **详细说明**: [docs/OUTPUT_STRUCTURE_QUICKREF.md](docs/OUTPUT_STRUCTURE_QUICKREF.md)
- **格式对比**: [docs/results_reports/SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md](docs/results_reports/SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md)
- **设计决定**: [docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md)

---

## 常用命令

```bash
# 列出所有模型
python3 mutation.py --list

# 单次训练
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs -n 3

# 快速验证（15-20分钟）
HF_HUB_OFFLINE=1 python3 mutation.py -ec settings/11_models_quick_validation_1epoch.json

# 变异实验（动态生成超参数）
sudo -E python3 mutation.py -ec settings/mutation_validation_1x.json -g performance

# 完整实验（9+小时）
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py -ec settings/11_models_sequential_and_parallel_training.json -g performance

# 查看结果
cat results/run_*/summary.csv | column -t -s,
```

**完整参数说明**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

---

## 系统要求

- **Python**: 3.6+ (仅标准库)
- **能耗监控**: `perf` (CPU), `nvidia-smi` (GPU)
- **权限**: 需要sudo以获取准确的CPU能量数据

```bash
# 安装perf
sudo apt-get install linux-tools-common linux-tools-generic
sudo sysctl -w kernel.perf_event_paranoid=-1
```

---

## 📚 文档

### 核心文档
| 文档 | 说明 |
|-----|------|
| [项目进度完整总结](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md) | **项目总体状况与完成情况** ⭐⭐⭐ |
| [CLAUDE.md](CLAUDE.md) | **Claude助手指南（简洁版）** ⭐⭐⭐ |
| [超参数变异范围](docs/MUTATION_RANGES_QUICK_REFERENCE.md) | 变异范围速查 ⭐⭐⭐ |
| [快速参考](docs/QUICK_REFERENCE.md) | 命令速查表 ⭐⭐ |
| [实验配置指南](docs/SETTINGS_CONFIGURATION_GUIDE.md) | Settings JSON编写 ⭐⭐ |
| [JSON配置书写规范](docs/JSON_CONFIG_WRITING_STANDARDS.md) | **JSON配置标准** ⭐⭐⭐ |

### 模型与功能
| 文档 | 说明 |
|-----|------|
| [11个模型概览](docs/11_MODELS_OVERVIEW.md) | 模型详细信息 ⭐⭐ |
| [11个模型最终定义](docs/11_MODELS_FINAL_DEFINITION.md) | 模型完整定义 ⭐⭐ |
| [并行训练使用](docs/PARALLEL_TRAINING_USAGE.md) | 并行训练配置 ⭐⭐ |
| [输出结构](docs/OUTPUT_STRUCTURE_QUICKREF.md) | 结果目录结构 ⭐ |
| [功能总览](docs/FEATURES_OVERVIEW.md) | 所有功能说明 ⭐⭐ |
| [脚本快速参考](docs/SCRIPTS_QUICKREF.md) | 核心脚本工具 ⭐⭐ |

### 数据分析报告
| 文档 | 说明 |
|-----|------|
| [数据格式设计决定](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md) | **数据格式设计理念** ⭐⭐⭐ |
| [运行时间统计](docs/results_reports/RUNTIME_STATISTICS_20251211.md) | 476个实验运行时间分析 ⭐⭐ |
| [完整文档索引](docs/README.md) | 所有文档列表 |

---

## Settings配置文件

### 可用配置（settings/目录）

#### 基础配置
| 配置文件 | 说明 | 预计时间 |
|---------|------|---------|
| `11_models_quick_validation_1epoch.json` | 快速验证（1 epoch） | 15-20分钟 |
| `mutation_validation_1x.json` | 变异验证（1次） | 按模型而定 |
| `mutation_all_models_3x_dynamic.json` | 完整变异（3次） | 较长时间 |
| `11_models_sequential_and_parallel_training.json` | 完整基线 | 9+小时 |
| `person_reid_dropout_boundary_test.json` | Dropout边界测试 | ~6.5小时 |

#### 分阶段实验配置（推荐用于大规模实验）⭐

**最终分阶段配置** (v4.7.2 - 最终合并):
| 配置文件 | 说明 | 预计时间 | 实验数 | 状态 |
|---------|------|---------|--------|------|
| `stage1_nonparallel_completion.json` | 阶段1: 非并行补全 | 9h | 12 | ✅ 已完成 |
| `stage2_optimized_nonparallel_and_fast_parallel.json` | 阶段2: 非并行补充 + 快速模型并行 | 7.3h | 25 | ✅ 已完成 |
| `stage3_4_merged_optimized_parallel.json` | 阶段3-4: 中速模型 + VulBERTa + densenet121 | 12.2h | 25 | ✅ 已完成 |
| `stage7_nonparallel_fast_models.json` | 阶段7: 非并行快速模型 | 0.7h | 7 | ✅ 已完成 |
| `stage8_nonparallel_medium_slow_models.json` | 阶段8: 非并行中慢速模型 | ~13h | 12 | ✅ 已完成 |
| `stage_final_all_remaining.json` | **最终阶段**: 所有剩余实验 **(pcb+hrnet18+快速模型+MRT-OAST+VulBERTa/cnn)** | 37.8h | 78 | **⏳ 一次性完成** |
| **总计** | **6个阶段** | **79.96h** | **159** | **81/159完成 (51%)** |
| ~~stage9/10/11/12/13~~ | ~~独立Stage配置~~ | ~~71.6h~~ | ~~104~~ | 🗑️ 已归档或合并 |

**v4.7.1修复与优化** (2025-12-07):
- 🔴 **严重Bug修复**: Stage7-13配置文件存在多参数混合变异问题
  - 问题: 配置使用`mutate_params=[多个参数]`导致混合变异而非单参数独立运行
  - 影响: 预期370个实验，实际只会运行108个（缺失262个，70.8%）
  - 修复: 自动拆分为单参数配置项（19项 → 62项）
  - 工具: `scripts/fix_stage_configs.py`
- ✅ **Stage7-8执行完成**:
  - Stage7: 实际0.7小时（预期38.3h），去重率96.5%，7个新实验
  - Stage8: 实际~13小时（预期35.1h），12个新实验
  - 总实验数: 400→419（新增19个实验）
- ✅ **Stage13配置合并**:
  - 合并: Stage13 + Stage14 + VulBERTa/cnn完整覆盖
  - 发现: VulBERTa/cnn模型在所有阶段中完全遗漏（0个实验）
  - 新增: 8个VulBERTa/cnn配置项（非并行4个+并行4个）
  - 配置: `stage13_merged_final_supplement.json` (18项, 90实验, 12.5h)
- ✅ **Stage9-10冗余移除**:
  - Stage9: hrnet18非并行全部达标（5-6个唯一值），已归档
  - Stage10: pcb非并行全部达标（5-6个唯一值），已归档
  - 节省时间: 48.7小时，节省实验: 40个
- 📊 **最终实验计划**:
  - 剩余3个必需阶段: Stage11, Stage12, Stage13
  - 预计时间: 64.2小时（去重后可能<20小时）
  - 预计实验: 130个
  - 完成后: 100%覆盖（90/90参数-模式组合）
- 📚 **文档完善**:
  - [JSON配置最佳实践](docs/JSON_CONFIG_BEST_PRACTICES.md) - 防止配置错误 ⭐⭐⭐
  - [Stage13-14合并报告](docs/results_reports/STAGE13_14_MERGE_AND_COMPLETION_REPORT.md) ⭐⭐
  - [Stage9-13优化报告](docs/results_reports/STAGE9_13_OPTIMIZATION_REPORT.md) - 冗余分析 ⭐⭐

**v3.0优化重点** (2025-12-05):
- ✅ **模式区分**: 修复去重机制，正确区分并行/非并行模式
- ✅ **精确规划**: 基于实际完成度（28/90组合），重新设计Stage7-13
- ✅ **效率提升**: Stage5-6被更优的Stage11-12替代（节省44.3小时）
- ✅ **全面覆盖**: 完成后达到100%（90/90组合，450个唯一值）

**详细执行计划**: [docs/settings_reports/STAGE7_13_EXECUTION_PLAN.md](docs/settings_reports/STAGE7_13_EXECUTION_PLAN.md)
**需求分析报告**: [docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md](docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md)

**归档配置** (`settings/archived/` 和备份文件):
- **Stage5-6**: 已归档（被Stage11-12替代）
- **Stage9-10**: 已归档到`redundant_stages_20251207/`（参数全部达标，节省48.7小时）
- **旧版配置**: v1.0-v2.0配置已归档
- **Stage13/14原始**: 已备份（`.bak_20251207`）：
  - `stage13_parallel_fast_models_supplement.json.bak_20251207`
  - `stage14_stage7_8_supplement.json.bak_20251207`

### 配置格式

**顺序训练**:
```json
{
  "repo": "examples",
  "model": "mnist",
  "mode": "mutation",
  "mutate": ["epochs", "learning_rate"]
}
```

**并行训练**:
```json
{
  "mode": "parallel",
  "foreground": {
    "repo": "examples",
    "model": "mnist",
    "mode": "mutation",
    "mutate": ["epochs"]
  },
  "background": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "densenet121",
    "hyperparameters": {"epochs": 60, "learning_rate": 0.05}
  }
}
```

**详细指南**: [docs/SETTINGS_CONFIGURATION_GUIDE.md](docs/SETTINGS_CONFIGURATION_GUIDE.md)

---

## 版本信息

**v4.7.11** (2025-12-19) - 🎯 **列结构最终优化** ✅
- ✅ **删除2个空列**: perf_accuracy, perf_eval_loss（v4.7.10合并后变空）
  - 列数: 56 → 54 (精简3.6%)
  - 性能指标: 16列 → 14个有效列
  - 空列: 2个 → 0个 ✅
- ✅ **备份清理**: 删除2个冗余备份文件，节省0.56MB
  - 清理前: 35个备份文件，6.96MB
  - 清理后: 34个备份文件，6.66MB
  - 保留: 3个最新关键备份
- ✅ **数据质量评估完成**: ⭐⭐⭐ 优秀
  - 数据一致性: 2704/2704检查点 (100.0%)
  - 数据完整性: 676行100%保留，无数据丢失
  - 与raw_data.csv对比: 无冲突，无缺失
  - 统一视图设计: 正确合并并行/非并行格式
- 📚 **文档完善**:
  - 质量评估报告: [数据质量评估报告](docs/results_reports/DATA_CSV_QUALITY_ASSESSMENT_REPORT.md) ⭐⭐⭐
  - 列优化报告: [备份清理与列优化报告](docs/results_reports/BACKUP_CLEANUP_AND_COLUMN_OPTIMIZATION_REPORT.md) ⭐⭐
  - 列分析报告: [data.csv列结构分析](docs/results_reports/DATA_CSV_COLUMN_ANALYSIS_AND_MERGE_RECOMMENDATIONS.md) ⭐
- 🔧 **执行脚本**:
  - `scripts/cleanup_backups.py` - 备份分析与清理
  - `scripts/remove_empty_performance_columns.py` - 删除空列
  - `scripts/evaluate_data_quality.py` - 数据质量评估
- 💾 **备份文件**:
  - `results/data.csv.backup_before_column_removal_20251219_182227` - 删列前备份
  - `results/data.csv.backup_before_merge_20251219_180149` - 合并前备份

**v4.7.10** (2025-12-19) - 🎯 **性能指标保守合并完成** ✅
- ✅ **性能指标统一命名**: 保守合并方案成功执行
  - 合并1: `accuracy` → `test_accuracy` (MRT-OAST的46个实验)
  - 合并2: `eval_loss` → `test_loss` (VulBERTa/mlp的72个实验)
  - 影响: 118个实验（17.5%），100%正确迁移
- ✅ **数据质量验证通过**:
  - 数据完整性: 676行无变化，无数据丢失 ✅
  - 合并正确性: 46+72=118个实验 100%正确 ✅
  - 实验目标: 90/90组合（100%）维持不变 ✅
- ✅ **填充率显著提升**:
  - `test_accuracy`: 38.2% → 45.0% (+6.8%) ✅
  - `test_loss`: 18.0% → 28.7% (+10.7%) ✅
- ✅ **有效列数优化**: 16列 → 14个有效列
  - 保留: test_accuracy, test_loss（统一命名）
  - 清空: accuracy, eval_loss（源列保留但已清空）
- 📊 **合并效果评估**:
  - 语义统一: 相同含义指标使用统一命名 ✅
  - 数据集中: 主要指标填充率提升 ✅
  - 降低复杂度: 有效指标从16减少到14 ✅
  - 向后兼容: CSV结构保持稳定 ✅
- 📚 **文档完善**:
  - 可行性分析: [性能指标合并可行性分析](docs/results_reports/PERFORMANCE_METRICS_MERGE_FEASIBILITY_ANALYSIS.md) ⭐⭐⭐
  - 完成报告: [性能指标合并完成报告](docs/results_reports/PERFORMANCE_METRICS_MERGE_COMPLETION_REPORT.md) ⭐⭐
  - 分类分析: [性能指标分类与含义分析](docs/results_reports/PERFORMANCE_METRICS_CLASSIFICATION_AND_ANALYSIS.md) ⭐
  - 数据精度: [数据精度分析报告](docs/results_reports/DATA_PRECISION_ANALYSIS_REPORT.md) ⭐
- 🔧 **执行脚本**:
  - `scripts/merge_performance_metrics.py` - 合并脚本
  - `scripts/validate_merged_metrics.py` - 验证脚本
- 💾 **备份文件**:
  - `results/data.csv.backup_before_merge_20251219_180149` - 合并前备份

**v4.7.6** (2025-12-15) - 🎯 **Phase 5并行模式补充完成** ✅
- ✅ **Phase 5并行模式优先补充执行成功**: 2个模型完全达标，3个模型接近达标
  - 执行日期: 2025-12-14 16:09 - 2025-12-15 17:05
  - 实验数: 72个 (VulBERTa/mlp:12, bug-localization:8, MRT-OAST:12, mnist:20, mnist_ff:20)
  - 执行时长: ~24.9小时 (时间预测误差<0.1% ⭐)
  - 数据完整性: 100% (训练成功、能耗、性能指标)
  - 详细报告: [Phase 5执行报告](docs/results_reports/PHASE5_PARALLEL_SUPPLEMENT_EXECUTION_REPORT.md) ⭐⭐⭐
- ✅ **并行模式达标情况**:
  - examples/mnist: 所有参数 ≥6个唯一值 ✅ **完全达标**
  - examples/mnist_ff: 所有参数 ≥8个唯一值 ✅ **完全达标**
  - VulBERTa/mlp: 3/4参数达标，learning_rate缺2个 ⚠️
  - bug-localization: 3/4参数达标，alpha缺2个 ⚠️
  - MRT-OAST: 3/5参数达标，dropout和learning_rate各缺2个 ⚠️
- ✅ **非并行模式完成情况**: 11/11模型全部达标 ✅
- ✅ **数据追加完成**: raw_data.csv更新（512→584行）
  - 新增: 72行Phase 5实验数据
  - 验证: 数据完整性验证通过
  - 备份: `raw_data.csv.backup_20251215_171809`
- 📊 **下一步**: Phase 6精细补充（8个实验，~4-5小时）完成3个接近达标的模型

**v4.7.5** (2025-12-14) - 🎯 **Phase 4验证完成** ✅
- ✅ **Phase 4验证配置执行成功**: 补充3个关键模型的实验数据
  - 执行日期: 2025-12-13 20:35 - 2025-12-14 15:02
  - 实验数: 33个 (VulBERTa/mlp:14, bug-localization:11, MRT-OAST:8)
  - 执行时长: ~18.5小时
  - 数据完整性: 100% (训练成功、能耗、性能指标)
  - 详细报告: [Phase 4执行报告](docs/results_reports/PHASE4_VALIDATION_EXECUTION_REPORT.md) ⭐⭐⭐
- ✅ **并行模式数据提取修复**: `append_session_to_raw_data.py`完全支持并行实验
  - 问题: 并行实验的experiment.json结构不同（repository/model在foreground中）
  - 修复: 区分并行/非并行模式，从foreground提取数据
  - 验证: 修复前跳过10个并行实验，修复后成功提取所有33个实验
  - 映射: 正确映射energy_metrics和performance_metrics字段名
- ✅ **非并行模式3个模型达标**:
  - VulBERTa/mlp: 所有参数 ≥12个唯一值 ✅
  - bug-localization: 所有参数 ≥10个唯一值 ✅
  - MRT-OAST: 所有参数 ≥9个唯一值 ✅
- ✅ **数据追加完成**: raw_data.csv更新（479→512行）
  - 新增: 33行Phase 4实验数据
  - 验证: 数据完整性验证通过
  - 备份: `raw_data.csv.backup_20251214_153258`
- 📊 **实验目标距离更新**:
  - 非并行模式: 3个模型达标 + 8个模型待补充
  - 并行模式: 11个模型全部待补充
  - 下一步: Phase 5大规模并行补充（~300实验，~150小时）

**v4.7.4-dev** (2025-12-12) - 🚧 **数据提取问题修复中** ⏳
- 🔧 **终端输出捕获功能开发**: Phase 1已完成 ✅
  - 功能: 添加`capture_stdout`参数到`mutation/command_runner.py`
  - 保存: 训练过程的stdout/stderr到`terminal_output.txt`
  - 测试: 4/4自动化测试通过
  - 配置: 创建8个调试实验配置（4个问题模型 × 2种模式）
  - 文档: [终端输出捕获指南](docs/TERMINAL_OUTPUT_CAPTURE_GUIDE.md)
- ⏳ **数据提取问题诊断**: Phase 2待开始
  - 目标: 运行测试实验，分析151个缺失性能数据的实验
  - 问题模型: examples/mnist_ff (46), VulBERTa/mlp (45), bug-localization (40), MRT-OAST (20)
  - 下一步: 运行`settings/test_data_extraction_debug.json`（预计4-6小时）
  - 进度追踪: [任务进度记录](docs/TASK_PROGRESS_DATA_EXTRACTION_FIX.md)
- 📊 **预期收益**:
  - 有效实验: 327/458 (71.4%) → 458/458 (100%) [+40%]
  - 数据完整模型: 7/11 (63.6%) → 11/11 (100%) [+57%]
  - 无需重新训练，节省50-100小时计算时间

**v4.7.3** (2025-12-12)
- 🔴 **实验目标重新澄清**: 识别出重大完成度差距 ⭐⭐⭐
  - **目标**: 每个超参数在两种模式下需要1个默认值 + 5个唯一单参数变异
  - **当前**: 0%完全达标（0/90组合），73.3%部分完成（66/90），26.7%完全缺失（24/90）
  - **需补齐**: 330个实验（90个默认值 + 240个变异）
  - **主要问题**: 性能数据缺失26.9%（128个实验，主要集中在3个模型）、缺少所有默认值实验、26个多参数变异实验
  - **详细报告**: [EXPERIMENT_GOAL_CLARIFICATION_AND_COMPLETION_REPORT.md](docs/results_reports/EXPERIMENT_GOAL_CLARIFICATION_AND_COMPLETION_REPORT.md)
  - **分析脚本**: `scripts/analyze_experiment_completion.py` - 已更新性能数据验证逻辑（检查所有性能指标）
- ✅ **summary_old.csv 93列重建完成**: 从experiment.json直接重建，确保数据完整性
  - 问题: 原80列格式缺少13个字段（6个bg_hyperparam + 7个bg_energy）
  - 解决: 编写重建脚本，直接从211个experiment.json文件提取完整数据
  - 结果: 93列格式，211行数据，100%数据完整性验证通过
  - 验证: 训练成功率100%，CPU/GPU能耗100%完整，随机抽样100%通过
  - 备份: `summary_old.csv.backup_80col`, `summary_old.csv.backup_before_93col_replacement`
  - 脚本: `scripts/rebuild_summary_old_93col.py`, `scripts/validate_93col_rebuild.py`
- ✅ **数据格式设计决定分析**: 80列vs93列完整对比 ⭐⭐⭐
  - 分析: 老实验的13个"多出"列对后续分析完全无价值
  - 验证: 背景超参数100%为默认值，背景能耗100%为空
  - 结论: 新实验的80列格式是设计改进，不是缺陷
  - 报告: [docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md)
- ✅ **数据合并与归档**: 生成主数据文件raw_data.csv ⭐⭐⭐
  - 合并: summary_old.csv (211行) + summary_new.csv (265行) → raw_data.csv (476行，80列)
  - 验证: 100%训练成功，100%能耗完整，66.2%性能指标完整
  - 归档: 13个过时文件移至summary_archive/目录
  - 清理: 8个过时备份文件
  - 脚本: `scripts/merge_csv_to_raw_data.py`, `scripts/validate_raw_data.py`, `scripts/archive_summary_files.py`
- ✅ **去重迁移到raw_data.csv**: summary_all.csv → raw_data.csv完全迁移 ⭐⭐⭐
  - 原因: 停止维护summary_all.csv，使用raw_data.csv作为单一数据源
  - 更新: 9个配置文件的historical_csvs字段从summary_all.csv改为raw_data.csv
  - 修改: mutation.py的-S参数改为`--enable-summary-append`（默认不追加到summary_all.csv）
  - 验证: 创建5个功能测试，全部通过（数据提取、去重、配置执行）
  - 统计: 从raw_data.csv提取371个变异（78%提取率），341个唯一组合（92%唯一率）
  - 工具: `scripts/update_historical_csv_refs.py`, `tests/test_dedup_raw_data.py`
  - 报告: [docs/results_reports/V4_7_3_DEDUPLICATION_MIGRATION_REPORT.md](docs/results_reports/V4_7_3_DEDUPLICATION_MIGRATION_REPORT.md)
- ✅ **项目整理与归档**: 文档和脚本大规模整理 ⭐⭐
  - 归档: 22个已完成任务的脚本 → `scripts/archived/completed_tasks_20251212/`
  - 归档: 5个临时分析报告 → `docs/archived/temporary_analysis_20251212/`
  - 保留: 10个核心脚本（核心工具3+配置工具3+分析工具3+下载工具1）
  - 文档: `docs/SCRIPTS_QUICKREF.md` - 核心脚本快速参考
- 📚 **文档完善**:
  - README.md和CLAUDE.md添加"数据结构说明"章节
  - 详细说明并行/非并行模式的数据存储差异
  - 明确背景能耗0%填充率为设计决定（背景训练仅作GPU负载，不监控能耗）
  - 新增4个分析报告文档（数据格式、去重迁移等）和1个脚本快速参考文档

**v4.7.2** (2025-12-08)
- 🔴 **并行模式runs_per_config Bug修复**: 修复v4.7.0遗留问题
  - 问题: Stage11只运行4个实验而非20个（缺失80%）
  - 根因: v4.7.0修复仅覆盖mutation/default模式，未修复parallel模式
  - 修复: 修改并行模式读取逻辑，支持外层experiment定义runs_per_config
  - 优先级: 外层exp > foreground > 全局（三级fallback）
  - 测试: 创建`test_parallel_runs_per_config_fix.py`，全部通过
  - 详细报告: [Stage11 Bug修复报告](docs/results_reports/STAGE11_BUG_FIX_REPORT.md)
- ✅ **配置最终合并** (单一Stage方案):
  - **最终整合**: 将Stage11+12+13合并为单一配置文件
  - **配置文件**: `settings/stage_final_all_remaining.json`
  - **实验数**: 78个（pcb:12 + hrnet18:8 + 快速模型:43 + MRT-OAST:7 + VulBERTa/cnn:8）
  - **预计时间**: 37.8小时（一次性完成所有剩余实验）
  - **资源节省**: 78个实验（原126），37.8小时（原58.5h），仅需执行1个阶段
  - 详细报告: [最终配置整合报告](docs/results_reports/FINAL_CONFIG_INTEGRATION_REPORT.md)
- 📊 **数据审计修正**:
  - hrnet18并行: 实际已有3个唯一值（非1个），节省8个实验
  - pcb并行: 实际已有2个唯一值，节省8个实验
  - 修正报告: [Stage11实际状态修正](docs/results_reports/STAGE11_ACTUAL_STATE_CORRECTION.md)
- 🧹 **项目清理**:
  - 归档10个过时文档（减少35.7%）
  - 归档4个过时配置（Stage11/12/13独立配置）
  - 归档3个根目录文档（Stage11相关执行指南）
  - 删除9个错误备份文件（100%清理）
- ✅ **向后兼容性**: 完全兼容，支持三种配置方式

**v4.7.1** (2025-12-07)
- 🔴 **严重配置Bug修复**: Stage7-13配置文件多参数混合变异问题
  - 问题: 配置使用`mutate_params=[多个参数]`导致生成混合变异实验，而非每个参数独立运行
  - 根因: 对`runs_per_config`和`mutate_params`语义理解错误
  - 影响: 预期370个实验，实际只会运行108个（缺失262个，70.8%）
  - 修复: 使用`scripts/fix_stage_configs.py`自动拆分为单参数配置（19项 → 62项）
  - 详细分析: [Stage7-13配置Bug分析](docs/results_reports/STAGE7_13_CONFIG_BUG_ANALYSIS.md)
- ✅ **Stage7-8执行情况分析**:
  - Stage7: 虽然配置错误但去重率96.5%，大部分参数已达标（历史实验覆盖）
  - Stage8: 所有参数已超标（≥10个唯一值），无需补充
  - Stage14: 新增补充配置，仅需7个实验（2.5小时）补充MRT-OAST/default epochs
  - 执行报告: [Stage7-8修复执行报告](docs/results_reports/STAGE7_8_FIX_EXECUTION_REPORT.md)
- 📚 **配置最佳实践文档** (新增):
  - 文档: [JSON配置最佳实践](docs/JSON_CONFIG_BEST_PRACTICES.md) ⭐⭐⭐
  - 内容: 核心概念、常见错误、正确示例、验证清单、故障排查
  - 重点: "单参数原则" - 每个配置项只变异一个参数
  - 示例: 正确vs错误配置对比，参考Stage2作为最佳模板

**v4.7.0** (2025-12-06)
- ✅ **Per-experiment runs_per_config bug修复**: 修复配置文件读取逻辑
  - 问题: `runner.py`第881行全局`runs_per_config`默认为1，导致仅使用每个实验配置的runs_per_config
  - 影响: Stage7等使用per-experiment值的配置只运行1个实验而非指定的7个
  - 修复: 在mutation、parallel、default三种模式中添加per-experiment值读取逻辑
  - 修改文件: `mutation/runner.py` (lines 1001-1126, 三个实验模式)
  - 测试: 创建5个测试用例全部通过 (`tests/test_runs_per_config_fix.py`)
  - 向后兼容: 保留全局fallback机制，旧配置仍可正常运行

**v4.6.0** (2025-12-05)
- ✅ **Stage3-4执行完成**: mnist_ff剩余 + 中速模型 + VulBERTa + densenet121
  - 实际运行: 12.2小时 (预期57.1小时)
  - 实验数量: 25个实验 (预期57个)
  - 去重效果: 跳过32个重复实验 (56.1%跳过率)
  - 总实验数: 381个 (356→381)
- ✅ **去重模式区分修复**: 修复去重机制，区分并行/非并行模式
  - 问题: 原去重机制未区分模式，导致并行实验被错误跳过
  - 修复: 在去重key中包含mode信息，确保不同模式独立去重
  - 修改文件: `hyperparams.py`, `dedup.py`, `runner.py` (共约30行代码)
  - 测试验证: 创建2个测试套件（10个测试用例）全部通过
  - 向后兼容: mode参数可选，旧代码仍可正常运行
  - 详细报告: [docs/results_reports/DEDUP_MODE_FIX_REPORT.md](docs/results_reports/DEDUP_MODE_FIX_REPORT.md)
- ✅ **实验进度分析**: 当前完成度评估
  - 总实验数: 381个 (非并行261 + 并行120)
  - 参数达标率: 100% (不区分模式) / 80% (区分模式)
  - 非并行模式: 97.8% (44/45参数达标)
  - 并行模式: 62.2% (28/45参数达标)
  - 缺失: 18个参数-模式组合，需补充50个唯一值
  - 详细分析: [docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md](docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md)

**v4.5.0** (2025-12-04)
- ✅ **Stage2执行完成**: 非并行补充 + 快速模型并行实验完成
  - 实际运行: 7.3小时 (预期20-24小时)
  - 实验数量: 25个实验 (预期44个)
  - 核心目标达成: 所有29个目标参数达到5个唯一值
  - 去重效果: 跳过40个重复实验 (61.5%跳过率)
- ✅ **参数精确优化** (runs_per_config v2.0): 每个参数使用精确的runs_per_config值
  - 优化原理: runs_per_config = (5 - current_unique_count) + 1
  - 资源利用率: 从26.9%提升到>90%
  - 减少无效尝试: 从390次降低到105-145次（节省63%）
  - 节省GPU时间: 约160小时
- ✅ **配置文件合并**: 将Stage3和Stage4合并为单个配置文件
  - 合并后配置: `stage3_4_merged_optimized_parallel.json`
  - 实验项: 25个实验项，57个预期实验
  - 时间预估: 57.1小时 (基于Stage2 38.5%完成率重新预估)
- ✅ **完整文档**: 优化报告、快速参考、执行准备清单
- ✅ **配置归档**: 旧版统一runs_per_config配置移至archive/目录

**v4.4.0** (2025-12-02)
- ✅ **CSV追加bug修复**: 修复了`_append_to_summary_all()`方法导致的数据覆盖问题
  - 问题: 调用`aggregate_csvs.py`使用'w'模式���盖整个文件
  - 修复: 直接使用CSV模块的'a'模式安全追加数据
  - 测试: 创建了11个单元测试和4个手动测试验证修复
- ✅ **去重机制**: 支持基于历史CSV文件的实验去重
  - 配置: `use_deduplication: true` + `historical_csvs: ["results/summary_all.csv"]`
  - 效果: 自动跳过已存在的超参数组合，生成新的随机值
- ✅ **分阶段实验计划**: 大规模实验分割成7个可管理的阶段
  - 总计256小时实验分割为20-46小时/阶段
  - 便于监控、暂停和验证中间结果
  - 所有阶段配置文件已准备完毕
- ✅ **实验完成度分析**: 当前319个实验（214非并行 + 105并行）
  - 非并行完成度: 73.3% (33/45参数达到5个唯一值)
  - 并行完成度: 0% (0/45参数达到5个唯一值)
  - 分阶段计划将完成剩余269个实验达到100%

**v4.3.0** (2025-11-19)
- ✅ 11个模型完整支持（基线+变异）
- ✅ 动态变异系统（log-uniform/uniform分布）
- ✅ 并行训练（前景+背景GPU同时利用）
- ✅ 离线训练（HF_HUB_OFFLINE=1）
- ✅ 高精度能耗监控（CPU误差<2%）
- ✅ Dropout范围优化：统一采用d-0.2到d+0.1策略
  - MRT-OAST: [0.0, 0.3] (d=0.2)
  - Person_reID: [0.3, 0.6] (d=0.5)
  - 基于边界测试验证，dropout=0.6性能变化<1%，dropout=0.7导致严重劣化(-3%~-8%)

**完整版本历史**: [docs/FEATURES_OVERVIEW.md](docs/FEATURES_OVERVIEW.md)

---

**维护者**: Green | **状态**: ✅ Production Ready | **更新**: 2025-12-08
