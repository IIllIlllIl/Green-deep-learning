# Claude 助手指南 - Mutation-Based Training Energy Profiler

**项目版本**: v4.7.5 (2025-12-14)
**最后更新**: 2025-12-14
**状态**: ✅ Production Ready

---

## 🎯 项目概述

### 核心目标
研究深度学习训练超参数对能耗和性能的影响。通过自动化变异超参数、监控能耗、收集性能指标，支持大规模实验研究。

### 关键特性
- ✅ **超参数变异** - 自动生成超参数变体（log-uniform/uniform分布）
- ✅ **能耗监控** - CPU (perf) + GPU (nvidia-smi), CPU误差<2%
- ✅ **并行训练** - 支持前台监控+后台负载的并行训练模式
- ✅ **去重机制** - 自动跳过重复实验，支持历史数据去重，**区分并行/非并行模式**
- ✅ **数据完整性** - CSV安全追加模式，防止数据覆盖丢失
- ✅ **分阶段执行** - 大规模实验分割成可管理的阶段
- ✅ **参数精确优化** - 每个参数使用精确的runs_per_config值

### 当前状态 ⚠️

**⚠️ 重要更新 (2025-12-14)**: Phase 4验证完成，raw_data.csv更新至512行

- **实验总数**: 512个（追加Phase 4后，原479个）
- **有效模型**: 11个（VulBERTa/cnn已移除）
- **数据质量状况**:
  - ✅ 训练成功: 502/512 (98.0%)
  - ✅ 能耗数据: 475/512 (92.8%)
  - ⚠️ 性能数据: 376/512 (73.4%)
- **Phase 4补充**:
  - 新增实验: 33个 (VulBERTa/mlp:14, bug-localization:11, MRT-OAST:8)
  - 数据完整性: 100% (训练、能耗、性能全部完整)
  - 执行时长: ~18.5小时
  - 详细报告: [PHASE4_VALIDATION_EXECUTION_REPORT.md](docs/results_reports/PHASE4_VALIDATION_EXECUTION_REPORT.md)
- **非并行模式3个模型达标**:
  - VulBERTa/mlp: 所有参数 ≥12个唯一值 ✅
  - bug-localization: 所有参数 ≥10个唯一值 ✅
  - MRT-OAST: 所有参数 ≥9个唯一值 ✅
- **数据追加**:
  - 追加前: 479行
  - 追加后: 512行
  - 新增: 33行Phase 4实验数据
  - 备份: `raw_data.csv.backup_20251214_153258`

### 实验目标（重新澄清）

**每个超参数在两种模式（并行/非并行）下都需要**:
1. **1个默认值实验** - 建立基线
2. **5个唯一单参数变异实验** - 研究单参数影响
3. **完整数据**: 能耗 + 任意性能指标（accuracy、mAP、test_accuracy等）

**总目标**: 45参数 × 2模式 × 6实验 = **540个有效实验**

**当前缺口**: 需补齐**330个实验**
- 默认值: 90个
- 单参数变异: 240个
- 性能数据缺失: 128个（主要集中在3个模型：mnist_ff, VulBERTa/mlp, bug-localization）

**详细分析**: [实验目标澄清报告](docs/results_reports/EXPERIMENT_GOAL_CLARIFICATION_AND_COMPLETION_REPORT.md)

### 最新进展

**2025-12-14** - v4.7.5 Phase 4验证完成 ⭐⭐⭐
- ✅ **Phase 4验证配置执行成功**: 补充3个关键模型的实验数据
  - **执行日期**: 2025-12-13 20:35 - 2025-12-14 15:02
  - **实验数**: 33个 (VulBERTa/mlp:14, bug-localization:11, MRT-OAST:8)
  - **执行时长**: ~18.5小时
  - **数据完整性**: 100% (训练成功、能耗、性能指标)
  - **详细报告**: [PHASE4_VALIDATION_EXECUTION_REPORT.md](docs/results_reports/PHASE4_VALIDATION_EXECUTION_REPORT.md)
- ✅ **并行模式数据提取修复**: `append_session_to_raw_data.py`完全支持并行实验
  - **问题**: 并行实验的experiment.json结构不同（repository/model在foreground中）
  - **修复**: 区分并行/非并行模式，从foreground提取数据
  - **验证**: 修复前跳过10个并行实验，修复后成功提取所有33个实验
  - **映射**: 正确映射energy_metrics和performance_metrics字段名
  - **修改位置**: `scripts/append_session_to_raw_data.py` (行 119-227, 236-242)
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

**2025-12-12** - v4.7.3更新完成 ⭐⭐⭐
- ✅ **数据格式设计决定分析**: 80列vs93列完整对比
  - 分析: 老实验的13个"多出"列对后续分析完全无价值
  - 验证: 背景超参数100%为默认值，背景能耗100%为空
  - 结论: 新实验的80列格式是设计改进，不是缺陷
  - 详细报告: [DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md)
- ✅ **数据合并与归档**: 生成主数据文件raw_data.csv
  - 合并: summary_old.csv (211行) + summary_new.csv (265行) → raw_data.csv (476行，80列)
  - 验证: 100%训练成功，100%能耗完整，66.2%性能指标完整
  - 归档: 13个过时文件移至summary_archive/目录
  - 清理: 8个过时备份文件
  - 脚本: `merge_csv_to_raw_data.py`, `validate_raw_data.py`, `archive_summary_files.py`
- ✅ **去重迁移到raw_data.csv**: summary_all.csv → raw_data.csv完全迁移 ⭐⭐⭐
  - **原因**: 停止维护summary_all.csv，使用raw_data.csv作为单一数据源
  - **更新**: 9个配置文件的historical_csvs字段从summary_all.csv改为raw_data.csv
  - **修改**: mutation.py的-S参数改为`--enable-summary-append`（默认不追加到summary_all.csv）
  - **验证**: 创建5个功能测试，全部通过（配置文件、数据提取、去重、配置执行）
  - **统计**: 从raw_data.csv提取371个变异（78%提取率），341个唯一组合（92%唯一率）
  - **依赖检查**: mutation/目录仅runner.py有25个引用（全部在_append_to_summary_all方法中，默认禁用）
  - **工具**:
    - `scripts/update_historical_csv_refs.py` - 批量更新配置文件
    - `tests/test_dedup_raw_data.py` - 功能测试套件（5个测试全部通过）
  - **报告**: [V4_7_3_DEDUPLICATION_MIGRATION_REPORT.md](docs/results_reports/V4_7_3_DEDUPLICATION_MIGRATION_REPORT.md) - 完整迁移报告
  - **影响**: 零破坏性变更，完全向后兼容（可选-S标志恢复旧行为）
- ✅ **项目整理与归档**: 文档和脚本大规模整理 ⭐⭐
  - **归档脚本** (22个):
    - 数据重建 (7个): rebuild_summary_old_from_json_93col.py等
    - 数据修复 (7个): step1_fix_experiment_source.py等
    - 配置修复 (1个): fix_stage_configs.py
    - 数据分离 (2个): separate_old_new_experiments.py等
    - 临时分析 (3个): analyze_summary_all_columns.py等
    - 验证脚本 (1个): validate_93col_rebuild.py
    - 已废弃 (1个): aggregate_csvs.py
  - **归档文档** (5个): CSV_FIX_COMPREHENSIVE_SUMMARY.md, DEDUP_MODE_FIX_REPORT.md等
  - **保留脚本** (10个):
    - 核心工具 (3个): merge_csv_to_raw_data.py, validate_raw_data.py, archive_summary_files.py
    - 配置工具 (3个): generate/validate/verify配置
    - 分析工具 (3个): analyze_baseline.py, analyze_experiments.py, analyze_archive_plan.py
    - 下载工具 (1个): download_pretrained_models.py
- 📚 **文档完善**:
  - README.md和CLAUDE.md更新数据文件说明，添加去重迁移说明
  - 新增 [SCRIPTS_QUICKREF.md](docs/SCRIPTS_QUICKREF.md) - 脚本快速参考
  - 新增4个分析报告（数据格式分析2+去重迁移1+脚本文档1）

**2025-12-12** - summary_old.csv 93列重建完成 ⭐⭐⭐
- ✅ **数据重建**: 从experiment.json直接重建，确保数据完整性
  - 原问题: 80列格式缺少13个字段（6个bg_hyperparam + 7个bg_energy）
  - 解决方案: 编写专用重建脚本，直接从211个JSON文件提取
  - 重建结果: 93列格式，211行数据，100%完整性验证通过
- ✅ **数据验证**:
  - 训练成功率: 211/211 (100.0%)
  - CPU能耗完整: 211/211 (100.0%)
  - GPU能耗完整: 211/211 (100.0%)
  - 随机抽样: 10/10 (100%) 通过
  - **关键发现**: 背景能耗0%填充率为正确行为（设计决定：背景训练仅作GPU负载，不监控能耗）
- 📚 **数据结构文档化**:
  - 详细记录并行/非并行模式的数据存储差异
  - 明确说明93列结构和字段含义
  - 添加到README.md和CLAUDE.md
- 🔧 **工具和脚本**:
  - `scripts/rebuild_summary_old_93col.py` - 重建脚本
  - `scripts/validate_93col_rebuild.py` - 验证脚本
- 💾 **备份管理**:
  - `summary_old.csv.backup_80col` - 原始80列版本
  - `summary_old.csv.backup_before_93col_replacement` - 替换前备份

**2025-12-11** - VulBERTa/cnn清理与100%完成确认 ⭐⭐⭐
- 🔍 **问题发现**: VulBERTa/cnn训练代码未实现
  - 根因: `train_vulberta.py` 中 `train_cnn()` 函数仅打印消息,无实际训练
  - 影响: 42个实验记录全部无效(平均3.92秒,性能指标全空)
- 🧹 **数据清理**: 从summary_all.csv移除42条无效记录
  - 清理前: 518条记录, 12个模型
  - 清理后: 476条记录, 11个有效模型
  - 备份: `summary_all.csv.backup_20251211_144013`
- ✅ **100%完成确认**: 清理后重新验证,所有实验目标已达成
  - 90/90 参数-模式组合全部达标 (100.0%)
  - 11个有效模型全部100%完成
  - 476个有效实验数据完整
- 📚 **文档更新**:
  - [11个模型最终定义](docs/11_MODELS_FINAL_DEFINITION.md) - 正式确认11个有效模型
  - [VulBERTa/cnn清理报告](docs/results_reports/VULBERTA_CNN_CLEANUP_REPORT_20251211.md) - 详细清理过程
  - 更新README.md和CLAUDE.md中的模型列表

**2025-12-08** - 配置最终合并完成 ⭐⭐⭐
  - ✅ **配置最终合并完成** ⭐⭐⭐
    - **最终整合**: 将Stage11+12+13合并为单一配置文件
    - **配置文件**: `settings/stage_final_all_remaining.json`
    - **实验数**: 78个（pcb:12 + hrnet18:8 + 快速模型:43 + MRT-OAST:7 + VulBERTa/cnn:8）
    - **预计时间**: 37.8小时（一次性完成所有剩余实验）
    - **资源节省**: 78个实验（原126），37.8小时（原58.5h），仅需执行1个阶段
    - **详细**: [最终配置整合报告](docs/results_reports/FINAL_CONFIG_INTEGRATION_REPORT.md)

---

## 📁 文件结构规范

### 目录结构标准
```
energy_dl/nightly/
├── docs/                    # 所有文档文件
│   ├── archived/           # 过时文档归档
│   ├── environment/        # 环境配置文档
│   ├── results_reports/    # 实验结果报告
│   └── settings_reports/   # 配置相关报告
├── scripts/                # 辅助脚本（Python/Bash）
├── tests/                  # 测试文件
├── settings/               # 运行配置文件
│   └── archived/          # 过时配置归档
├── results/               # 实验结果数据
│   └── run_YYYYMMDD_HHMMSS/  # 每次运行的session
├── mutation/              # 核心代码
├── config/                # 模型配置
└── environment/           # 环境设置（空目录，文档已移至docs）
```

### 文件放置规则

#### 1. 文档文件 (`docs/`)
- **所有** Markdown文档必须放在`docs/`目录下
- 按类型组织子目录：
  - `docs/results_reports/` - 实验结果分析报告
  - `docs/environment/` - 环境配置说明
  - `docs/settings_reports/` - 配置相关报告
  - `docs/archived/` - 过时文档（按日期组织）

#### 2. 脚本文件 (`scripts/`)
- 数据处理脚本、分析脚本、工具脚本
- 命名规范：`{用途}_{描述}.py` 或 `.sh`
- 示例：`analyze_unique_values.py`, `verify_csv_format.py`

#### 3. 测试文件 (`tests/`)
- 单元测试、集成测试、验证测试
- 命名规范：`test_{模块}_{功能}.py`
- 示例：`test_runner_append.py`, `verify_csv_append_fix.py`

#### 4. 配置文件 (`settings/`)
- JSON实验配置文件
- 命名规范：`{stage}_{描述}.json`
- 示例：`stage2_optimized_nonparallel_and_fast_parallel.json`

#### 5. 归档文件 (`*/archived/`)
- 每个目录可包含`archived/`子目录
- 过时文件按日期或版本组织
- 保留`README_ARCHIVE.md`说明归档原因

#### 6. 代码文件 (`mutation/`)
- 核心业务逻辑
- 模块化组织：`runner.py`, `session.py`, `energy_monitor.py`等

#### 7. 数据文件 (`results/`)
- 实验结果CSV、JSON、日志文件
- 每次运行创建独立session目录：`run_YYYYMMDD_HHMMSS/`
- 全局汇总文件：`summary_all.csv`

---

## 📋 数据格式设计决定 (新实验 vs 老实验)

### summary_new.csv (80列) vs summary_old.csv (93列)

**设计理念**: 新实验优化了数据记录策略，仅记录有分析价值的数据

#### 格式对比

| 数据类型 | 老实验 (93列) | 新实验 (80列) | 设计决定 |
|---------|--------------|--------------|---------|
| 前景超参数 | ✅ 完整 | ✅ 完整 | 核心数据 |
| 前景性能 | ✅ 完整 | ✅ 完整 | 核心数据 |
| 前景能耗 | ✅ 完整 | ✅ 完整 | 核心数据 |
| 背景超参数 | ⚠️ 部分有数据 | ❌ 无数据 | **设计改进**: 背景使用默认值，无需记录 |
| 背景能耗 | ❌ 无数据 | ❌ 无数据 | **设计决定**: 监控全局能耗，无需拆分 |

#### 13个"缺失"列的设计理由

**背景超参数字段 (6列) - 无需记录**:
- `bg_hyperparam_batch_size`
- `bg_hyperparam_dropout`
- `bg_hyperparam_epochs`
- `bg_hyperparam_learning_rate`
- `bg_hyperparam_seed`
- `bg_hyperparam_weight_decay`

**理由**:
1. 背景训练仅作为GPU负载，使用默认超参数配置
2. 老实验验证：105个并行实验中，100%背景超参数为默认值（无变异）
3. 记录默认值没有实际分析意义
4. 减少数据冗余，提高可维护性

**背景能耗字段 (7列) - 监控全局能耗**:
- `bg_energy_cpu_pkg_joules`
- `bg_energy_cpu_ram_joules`
- `bg_energy_cpu_total_joules`
- `bg_energy_gpu_avg_watts`
- `bg_energy_gpu_max_watts`
- `bg_energy_gpu_min_watts`
- `bg_energy_gpu_total_joules`

**理由**:
1. 能耗监控粒度：系统级监控（CPU + GPU全局）
2. 并行实验：前景+背景同时运行，前景能耗数据**已包含总能耗**
3. 无法单独拆分：无法也无需测量背景训练的能耗
4. 研究重点：关注前景模型训练，背景仅作GPU负载
5. 老实验验证：105个并行实验中，100%背景能耗字段为空

#### 数据完整性分析结果

**老实验深度分析** (105个并行实验):

| 背景模型 | 使用次数 | 超参数状态 |
|---------|---------|-----------|
| Person_reID/pcb | 30 | epochs=60, lr=0.05, dropout=0.5, seed=1334 (固定) |
| VulBERTa/mlp | 23 | epochs=10, lr=3e-05, seed=42, weight_decay=0.0 (固定) |
| examples/mnist | 13 | epochs=10, lr=0.01, batch_size=32, seed=1 (固定) |
| examples/mnist_rnn | 26 | epochs=10, lr=0.01, batch_size=32, seed=1 (固定) |
| examples/mnist_ff | 13 | epochs=10, lr=0.01, batch_size=32, seed=1或1334 (仅1个变化) |

**关键发现**:
- ✅ **100%使用默认值**（仅1个实验的seed有微小变化）
- ✅ **背景能耗100%为空**
- ✅ **13个"缺失"列对后续分析无价值**

#### 最终建议

**推荐方案**: 保持80列格式 ⭐⭐⭐

**优点**:
- ✅ 符合精简数据的设计理念
- ✅ 不引入无意义的空列
- ✅ 80列已包含所有核心数据
- ✅ 反映了项目的设计演进

**不推荐扩展为93列的原因**:
- ❌ 违背设计初衷（精简数据）
- ❌ 引入13个永久为空的列
- ❌ 可能误导使用者（空列看起来像数据缺失）

**详细分析报告**:
- [格式对比分析](docs/results_reports/SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md) - 80列vs93列完整对比 ⭐⭐⭐
- [老实验背景超参数分析](docs/results_reports/OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md) - 证明100%使用默认值 ⭐⭐

---

## 🔧 开发工作流程

### 1. 添加新功能
1. 在`mutation/`相应模块中实现功能
2. 在`tests/`中添加测试用例
3. 在`docs/`中更新相关文档
4. 验证通过后提交

### 2. 修复Bug
1. 在`tests/`中创建复现测试
2. 在`mutation/`中修复代码
3. 运行所有测试确保无回归
4. 在`docs/results_reports/`中记录修复报告

### 3. 更新配置
1. 在`settings/`中创建新配置文件
2. 在`docs/settings_reports/`中说明配置变更
3. 测试配置有效性
4. 归档旧配置文件到`settings/archived/`

### 4. 生成报告
1. 分析脚本放在`scripts/`
2. 报告文档放在`docs/results_reports/`
3. 使用标准命名：`{主题}_{描述}.md`
4. 包含时间戳和版本信息

---

## 📊 关键文件位置

### 核心代码
- `mutation/runner.py` - 主运行器（已修复CSV追加bug）
- `mutation/session.py` - Session管理
- `mutation/energy_monitor.py` - 能耗监控
- `config/models_config.json` - 模型配置

### 重要数据文件
- `results/raw_data.csv` - **主数据文件** (512行，80列) ⭐⭐⭐
  - 合并所有实验数据（211老实验 + 268新实验 + 33个Phase 4实验）
  - 98.0%训练成功，92.8%能耗完整，73.4%性能数据完整
  - 验证脚本: `scripts/validate_raw_data.py`
  - Phase 4报告: [PHASE4_VALIDATION_EXECUTION_REPORT.md](docs/results_reports/PHASE4_VALIDATION_EXECUTION_REPORT.md)
- `results/summary_old.csv` - 历史实验数据（211行，93列）
- `results/summary_new.csv` - 新实验数据（265行，80列）
- `results/summary_archive/` - 过时文件归档目录
  - 13个过时summary文件
  - 8个过时备份文件
  - 归档说明: `summary_archive/README_ARCHIVE.md`
- `results/summary_all.csv.backup` - 数据备份（已归档）
- `settings/EXECUTION_READY.md` - 执行准备清单（已归档）

#### summary_old.csv数据结构 (93列)

**非并行模式实验** (106个, 50.2%):
- **数据存储**: 顶层字段（`training_success`, `energy_cpu_total_joules`, `energy_gpu_avg_watts`等）
- **超参数**: `hyperparam_*` 字段（9个）
- **性能指标**: `perf_*` 字段（9个）
- **能耗数据**: `energy_*` 字段（11个）
- **前景/背景字段**: 全部为空（`fg_*`, `bg_*`字段不适用）

**并行模式实验** (105个, 49.8%):
- **前景模型数据** (被监控的主实验):
  - 训练状态: `fg_training_success`
  - 超参数: `fg_hyperparam_*` 字段（9个）
  - 性能指标: `fg_perf_*` 字段（9个）
  - 能耗数据: `fg_energy_*` 字段（11个，CPU+GPU完整监控）
- **背景模型数据** (GPU负载辅助训练):
  - 基本信息: `bg_repository`, `bg_model`, `bg_note`, `bg_log_directory`
  - 超参数: `bg_hyperparam_*` 字段（6个，部分填充，取决于模型）
  - **能耗数据**: `bg_energy_*` 字段（7个）**全部为空** ⚠️
    - **设计决定**: 背景训练仅作为GPU负载使用，不监控其能耗
    - **验证**: bg_note字段确认 "Background training served as GPU load only (not monitored)"
- **顶层字段**: 全部为空（数据在`fg_*`字段中）

**列结构总览** (93列):
1. 基础字段: 7列（experiment_id, timestamp, repository, model, training_success, duration_seconds, retries）
2. 顶层超参数: 9列（hyperparam_alpha, hyperparam_batch_size等）
3. 顶层性能指标: 9列（perf_accuracy, perf_precision等）
4. 顶层能耗: 11列（energy_cpu_pkg_joules, energy_gpu_avg_watts等）
5. 实验元数据: 5列（experiment_source, num_mutated_params, mutated_param, mode, error_message）
6. 前景字段: 42列（fg_repository, fg_model, fg_duration_seconds, fg_training_success, fg_hyperparam_*, fg_perf_*, fg_energy_*）
7. 背景字段: 10列（bg_repository, bg_model, bg_note, bg_log_directory, bg_hyperparam_*, bg_energy_*）

**数据完整性验证** (2025-12-12):
- ✅ 所有211个实验数据成功从experiment.json提取
- ✅ 训练成功率: 211/211 (100.0%)
- ✅ CPU能耗完整: 211/211 (100.0%)
- ✅ GPU能耗完整: 211/211 (100.0%)
- ✅ 随机抽样验证: 10/10 (100%) 通过
- ✅ 背景能耗0%填充率为正确行为（设计决定）

**重建历史** (2025-12-12):
- 原始格式: 80列（缺少6个bg_hyperparam + 7个bg_energy字段）
- 重建过程: 直接从211个experiment.json文件提取，确保无数据丢失
- 备份文件: `summary_old.csv.backup_80col`, `summary_old.csv.backup_before_93col_replacement`
- 验证脚本: `scripts/validate_93col_rebuild.py`

### 配置文件（已完成）
- `settings/stage2_optimized_nonparallel_and_fast_parallel.json` - Stage2 (已完成 ✓)
- `settings/stage3_4_merged_optimized_parallel.json` - Stage3-4合并 (已完成 ✓)
- `settings/stage7_nonparallel_fast_models.json` - Stage7 (已完成 ✓, 7实验)
- `settings/stage8_nonparallel_medium_slow_models.json` - Stage8 (已完成 ✓, 12实验)

### 配置文件（最终执行）⭐⭐⭐
- `settings/stage_final_all_remaining.json` - **最终阶段**: 所有剩余实验 **[一次性完成]**
  - 版本: v4.7.2-final
  - 合并内容: pcb并行(12) + hrnet18并行(8) + 快速模型(43) + MRT-OAST(7) + VulBERTa/cnn(8)
  - 实验数: 78个
  - 预计时间: ~37.8小时
  - 目标: 100%覆盖（90/90参数-模式组合，159/159实验完成）
  - 执行命令: `sudo -E python3 mutation.py -ec settings/stage_final_all_remaining.json`
  - 详细报告: [最终配置整合报告](docs/results_reports/FINAL_CONFIG_INTEGRATION_REPORT.md)

### 归档配置
- `settings/archived/stage5_optimized_hrnet18_parallel.json` - 被Stage11替代
- `settings/archived/stage6_optimized_pcb_parallel.json` - 被Stage12替代
- `settings/archived/redundant_stages_20251207/` - Stage9-10归档（非并行已达标）
- `settings/archived/20251208_final_merge/` - Stage11/12/13独立配置（已合并为最终配置）
  - `stage11_supplement_parallel_hrnet18.json`
  - `stage12_parallel_pcb.json`
  - `stage13_final_merged.json`
  - `stage13_merged_final_supplement.json`

### 测试文件
- `tests/verify_csv_append_fix.py` - CSV修复验证测试
- `tests/test_dedup_mode_distinction.py` - 去重模式区分功能测试
- `tests/test_integration_after_mode_fix.py` - 修复后集成测试
- `tests/test_runs_per_config_fix.py` - Per-experiment runs_per_config修复测试 (v4.7.0)
- `tests/test_parallel_runs_per_config_fix.py` - 并行模式runs_per_config修复测试 (v4.7.2) ⭐
- `tests/unit/test_append_to_summary_all.py` - 单元测试

### 文档索引

#### 功能与配置指南
- `docs/FEATURES_OVERVIEW.md` - 功能特性总览
- `docs/QUICK_REFERENCE.md` - 快速参考
- `docs/SETTINGS_CONFIGURATION_GUIDE.md` - 配置指南
- `docs/JSON_CONFIG_WRITING_STANDARDS.md` - JSON配置书写规范 ⭐⭐⭐ **[v4.7.3新增]**
- `docs/JSON_CONFIG_BEST_PRACTICES.md` - JSON配置最佳实践 ⭐⭐⭐ **[v4.7.1新增]**
- `docs/SCRIPTS_DOCUMENTATION.md` - Scripts目录完整文档 **[v4.7.2新增]**

#### 数据格式与结构 ⭐⭐⭐
- `docs/results_reports/SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md` - 80列vs93列完整对比 ⭐⭐⭐ **[v4.7.3新增]**
- `docs/results_reports/OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md` - 老实验背景超参数深度分析 ⭐⭐ **[v4.7.3新增]**
- `docs/OUTPUT_STRUCTURE_QUICKREF.md` - 输出结构快速参考

#### 实验执行报告
- `docs/results_reports/STAGE3_4_EXECUTION_REPORT.md` - Stage3-4执行报告
- `docs/results_reports/STAGE7_EXECUTION_REPORT.md` - Stage7执行报告
- `docs/results_reports/STAGE7_8_FIX_EXECUTION_REPORT.md` - Stage7-8修复执行报告 **[v4.7.1]**
- `docs/results_reports/PHASE4_VALIDATION_EXECUTION_REPORT.md` - Phase 4验证执行报告 ⭐⭐⭐ **[v4.7.5新增]**

#### Bug修复与优化
- `docs/results_reports/CSV_FIX_COMPREHENSIVE_SUMMARY.md` - CSV修复综合报告
- `docs/results_reports/DEDUP_MODE_FIX_REPORT.md` - 去重模式修复报告
- `docs/results_reports/STAGE7_CONFIG_FIX_REPORT.md` - Stage7-13配置修复报告
- `docs/results_reports/STAGE7_13_CONFIG_BUG_ANALYSIS.md` - Stage7-13配置Bug详细分析 **[v4.7.1]**
- `docs/results_reports/STAGE11_BUG_FIX_REPORT.md` - Stage11 Bug修复报告 ⭐⭐⭐ **[v4.7.2]**

#### 实验规划与分析
- `docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md` - 实验需求分析
- `docs/results_reports/MISSING_COLUMNS_DETAILED_ANALYSIS.md` - 缺失列详细分析
- `docs/results_reports/STAGE11_ACTUAL_STATE_CORRECTION.md` - Stage11实际状态修正 ⭐⭐ **[v4.7.2新增]**
- `docs/results_reports/FINAL_CONFIG_INTEGRATION_REPORT.md` - 最终配置整合报告 ⭐⭐⭐ **[v4.7.2新增]**
- `docs/results_reports/DEDUPLICATION_RANDOM_MUTATION_ANALYSIS.md` - 去重与随机变异分析 ⭐ **[v4.7.2]**
- `docs/settings_reports/STAGE7_13_EXECUTION_PLAN.md` - Stage7-13执行计划
- `docs/results_reports/STAGE7_13_DESIGN_SUMMARY.md` - Stage7-13设计总结

#### 项目总结
- `docs/results_reports/PROJECT_CLEANUP_SUMMARY_20251208.md` - 项目清理总结 **[v4.7.2]**
- `docs/results_reports/DAILY_SUMMARY_20251205.md` - 2025-12-05每日总结

---

## 🚀 快速开始命令

### 查看项目状态
```bash
# 检查CSV格式
python3 -c "import csv; f=open('results/summary_all.csv'); r=csv.reader(f); h=next(r); rows=list(r); print(f'✓ {len(rows)} experiments, ✓ {len(h)} columns')"

# 查看最新版本
grep '当前版本' README.md
```

### 运行测试
```bash
# 运行CSV修复验证测试
python3 tests/verify_csv_append_fix.py

# 运行所有单元测试
python3 -m pytest tests/unit/
```

### 执行实验
```bash
# Stage2优化配置（已完成 ✓）
# sudo -E python3 mutation.py -ec settings/stage2_optimized_nonparallel_and_fast_parallel.json

# 查看运行状态
ls -lht results/run_* | head -3
```

### 数据分析
```bash
# 分析唯一值数量
python3 scripts/analyze_unique_values.py

# 检查实验完成度
python3 scripts/check_completion_status.py
```

---

## ⚠️ 注意事项

### 1. 实验ID唯一性 ⭐⭐⭐ **[关键经验]**
**实验ID不唯一，必须使用复合键**

**问题**: 不同批次的实验会产生相同的experiment_id（如 `VulBERTa_mlp_001_parallel`）

**正确做法**: 使用 **experiment_id + timestamp** 作为唯一标识符

**示例**:
```python
# ❌ 错误 - 仅使用experiment_id查询
phase5_rows = [row for row in reader if 'mlp' in row['experiment_id']]
# 结果: 混合了不同批次的实验（Phase 4 + Phase 5）

# ✅ 正确 - 使用复合键
composite_key = f"{row['experiment_id']}|{row['timestamp']}"
if composite_key in existing_keys:
    ...

# ✅ 或使用时间范围过滤
phase5_start = datetime.fromisoformat('2025-12-14T17:48:00')
phase5_end = datetime.fromisoformat('2025-12-15T17:06:00')
if phase5_start <= timestamp <= phase5_end:
    phase5_rows.append(row)
```

**影响**:
- 数据查询错误（混合不同批次）
- 去重机制失效（误判为重复）
- 统计分析不准确（计数错误）

**已更新**:
- ✅ `scripts/append_session_to_raw_data.py` 已使用复合键去重（行236-256）
- ✅ `docs/results_reports/PHASE5_PERFORMANCE_METRICS_MISSING_ISSUE.md` 记录此教训

**相关报告**: [Phase 5性能指标缺失问题分析](docs/results_reports/PHASE5_PERFORMANCE_METRICS_MISSING_ISSUE.md) - 详细说明此问题 ⭐⭐⭐

### 2. CSV数据完整性
- `summary_all.csv`必须保持37列格式
- 追加数据时使用`extrasaction='ignore'`自动对齐列
- 每次修改前备份：`cp summary_all.csv summary_all.csv.backup`

### 3. 配置版本控制
- 创建新配置时保留旧版本在`settings/archived/`
- 在配置文件中添加`comment`字段说明变更原因
- 更新`docs/settings_reports/`中的相关文档

### 3. JSON配置书写规范 ⭐⭐⭐
**核心原则**: 每个实验配置只能变异一个超参数

**必须遵循**:
- ✅ 使用`"mutate": ["参数名"]`数组格式（**单参数变异**）
- ✅ 使用`"repo"`而非`"repository"`
- ✅ 使用`"mode"`而非`"mutation_type"`
- ❌ **禁止**使用`"mutate_params"`对象格式（会导致多参数同时变异）

**配置类型**:
1. **默认值实验**: `"mode": "default"` - 建立基线
2. **单参数变异**: `"mode": "mutation"`, `"mutate": ["参数名"]` - 研究单参数影响
3. **并行模式**: 使用`foreground/background`嵌套结构

**示例**:
```json
{
  "comment": "VulBERTa/mlp - learning_rate变异",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "mutate": ["learning_rate"],
  "runs_per_config": 5
}
```

**详细文档**:
- `docs/JSON_CONFIG_WRITING_STANDARDS.md` - 完整配置书写规范 ⭐⭐⭐
- `docs/JSON_CONFIG_BEST_PRACTICES.md` - 配置最佳实践
- 参考`settings/stage2_optimized_nonparallel_and_fast_parallel.json` - 已验证的标准配置

### 4. 测试要求
- 所有代码变更必须通过现有测试
- 新功能必须添加测试用例
- CSV格式变更必须运行`verify_csv_append_fix.py`

### 5. 文档同步
- 代码变更时更新相关文档
- 使用标准Markdown格式
- 在文档头部包含版本和日期信息

---

## 🔄 版本历史摘要

### v4.7.3 (2025-12-12) - 当前版本 ⭐
- ✅ **summary_old.csv 93列重建完成**: 从experiment.json直接重建，确保数据完整性
  - **问题**: 原80列格式缺少13个字段（6个bg_hyperparam + 7个bg_energy）
  - **根因**: 历史数据生成时未提取背景模型的超参数和能耗字段
  - **解决**: 编写重建脚本，直接从211个experiment.json文件提取完整数据
  - **结果**: 93列格式，211行数据，100%数据完整性验证通过
  - **验证**:
    - 训练成功率: 211/211 (100.0%)
    - CPU/GPU能耗: 211/211 (100.0%)
    - 随机抽样: 10/10 (100%) 通过
    - 背景能耗0%为设计决定（背景训练仅作GPU负载，不监控能耗）
  - **备份**: `summary_old.csv.backup_80col`, `summary_old.csv.backup_before_93col_replacement`
  - **重建脚本**: `scripts/rebuild_summary_old_93col.py`
  - **验证脚本**: `scripts/validate_93col_rebuild.py`
- ✅ **数据格式设计决定分析**: 80列vs93列完整对比 ⭐⭐⭐
  - **分析**: 老实验的13个"多出"列对后续分析完全无价值
  - **验证**:
    - 背景超参数: 100%为默认值（无变异）
    - 背景能耗: 100%为空（设计决定）
  - **结论**: 新实验的80列格式是设计改进，不是缺陷
  - **报告**:
    - [SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md](docs/results_reports/SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md) - 格式对比
    - [OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md](docs/results_reports/OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md) - 深度分析
    - [DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md) - 总结报告
- ✅ **数据合并与归档**: 生成主数据文件raw_data.csv ⭐⭐⭐
  - **合并**: summary_old.csv (211行) + summary_new.csv (265行) → raw_data.csv (476行，80列)
  - **验证**:
    - 100%训练成功
    - 100%能耗完整（CPU + GPU）
    - 66.2%性能指标完整
  - **归档**: 13个过时文件移至summary_archive/目录
  - **清理**: 8个过时备份文件
  - **脚本**:
    - `scripts/merge_csv_to_raw_data.py` - 合并脚本
    - `scripts/validate_raw_data.py` - 验证脚本
    - `scripts/archive_summary_files.py` - 归档脚本
- ✅ **项目整理与归档**: 文档和脚本大规模整理 ⭐⭐
  - **归档脚本** (22个): 已完成任务的脚本归档至`scripts/archived/completed_tasks_20251212/`
  - **归档文档** (5个): 临时分析报告归档至`docs/archived/temporary_analysis_20251212/`
  - **保留脚本** (10个): 核心工具3 + 配置工具3 + 分析工具3 + 下载工具1
  - **参数修改**: mutation.py的-S参数改为`--enable-summary-append`（默认不追加到summary_all.csv）
- 📚 **文档更新**:
  - README.md: 更新"结果输出"章节，添加raw_data.csv说明，更新v4.7.3版本信息
  - CLAUDE.md: 更新"重要数据文件"章节，添加数据格式设计决定说明，更新最新进展
  - 新增: [SCRIPTS_QUICKREF.md](docs/SCRIPTS_QUICKREF.md) - 脚本快速参考（10个核心脚本说明）
  - 新增3个分析报告文档
  - 明确说明背景能耗0%填充率为正确行为（项目设计决定）

### v4.7.2 (2025-12-08) ⭐
- 🔴 **并行模式runs_per_config Bug修复**: 修复v4.7.0遗留问题
  - **问题**: Stage11只运行4个实验而非20个（缺失80%）
  - **根因**: v4.7.0修复仅覆盖mutation/default模式，未修复parallel模式
  - **原因**: 并行模式的配置嵌套更复杂（foreground/background结构）
  - **修复**: 修改并行模式的两处（mutation和default分支）
    - 修改位置: `runner.py:1010-1011, 1030-1031`
    - 新逻辑: `exp.get("runs_per_config", foreground_config.get("runs_per_config", runs_per_config))`
    - 优先级: 外层exp > foreground > 全局（三级fallback）
  - **测试**: 创建`test_parallel_runs_per_config_fix.py`，全部通过
  - **详细报告**: [Stage11 Bug修复报告](docs/results_reports/STAGE11_BUG_FIX_REPORT.md)
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

### v4.7.1 (2025-12-07)
- ✅ **Stage7-13配置Bug修复**: 多参数混合变异问题
  - 自动修复工具: `scripts/fix_stage_configs.py`
  - 19个配置项 → 62个配置项（单参数原则）
- ✅ **Stage7-8执行完成**:
  - Stage7: 7个实验，0.74小时，96.5%去重率
  - Stage8: 12个实验，~13小时
- ✅ **JSON配置最佳实践文档**: 新增配置规范指南

### v4.7.0 (2025-12-06)
- ✅ **Per-experiment runs_per_config bug修复**: 修复配置文件读取逻辑
  - 问题: `runner.py`第881行全局`runs_per_config`默认为1，导致每个实验配置无法使用独立的runs_per_config值
  - 影响: Stage7等使用per-experiment值的配置只运行1个实验而非指定的7个
  - 修复: 在mutation、parallel、default三种模式中添加per-experiment值读取逻辑
  - 修改文件: `mutation/runner.py` (lines 1001-1126, 三个实验模式)
  - 测试: 创建5个测试用例全部通过 (`tests/test_runs_per_config_fix.py`)
  - 向后兼容: 保留全局fallback机制，旧配置仍可正常运行
- 📊 **关键要点**:
  - 每个实验配置现在可以独立设置`runs_per_config`值
  - 如果未设置，自动fallback到全局`runs_per_config`（默认为1）
  - 所有三种实验模式（mutation、parallel、default）均已修复
  - 向后兼容性保证：旧配置文件无需修改即可正常运行

### v4.6.1 (2025-12-06)
- ✅ **Stage7执行完成**: 非并行模式快速和中速模型完成
  - 实际运行: 0.74小时 (预期38.3小时)
  - 实验数量: 7个 (预期199个)
  - 去重效果: 跳过192个重复实验 (96.5%跳过率，节省37.6小时)
  - 总实验数: 388个 (381→388)
  - 完成时间: 2025-12-06 22:02
  - 成功率: 100% (7/7)
- ✅ **数据质量完美**:
  - CSV格式: 37列标准格式 ✓
  - 能耗数据: 所有实验CPU/GPU数据完整 ✓
  - 训练成功: 100%成功率，0次重试 ✓
- ✅ **关键发现**:
  - Stage7目标模型在非并行模式下已全部达到5+个唯一值
  - 验证了分阶段执行计划的有效性
  - 证明了历史数据去重机制的高效性
- 📊 **下一步建议**: 快速验证Stage8，根据去重率决定后续策略

### v4.6.0 (2025-12-05)
- ✅ **Stage3-4执行完成**: mnist_ff剩余 + 中速模型 + VulBERTa + densenet121
  - 实际运行: 12.2小时 (预期57.1小时)
  - 实验数量: 25个 (预期57个)
  - 去重效果: 跳过32个重复实验 (56.1%跳过率)
  - 总实验数: 381个 (356→381)
  - 完成时间: 2025-12-05 04:02
- ✅ **去重模式区分修复**: 修复去重机制，区分并行/非并行模式
  - 问题: 原去重机制未区分模式，导致并行实验被错误跳过
  - 修复: 在去重key中包含mode信息 (`hyperparams.py`, `dedup.py`, `runner.py`)
  - 测试: 创建2个测试套件（10个测试用例）全部通过
  - 向后兼容: mode参数可选，旧代码仍可正常运行
- ✅ **实验进度重新分析**:
  - 基于summary_all.csv准确数据
  - 参数-模式组合: 28/90 (31.1%)
  - 非并行: 0/45 (0%), 并行: 28/45 (62.2%)
  - 缺失: 62个组合，274个唯一值
- ✅ **Stage7-13配置设计完成**: 7个配置文件设计并验证
  - 预计新增: 370个实验，178.8小时
  - Stage5-6已归档（被更优的Stage11-12替代，节省44.3小时）
- ✅ **Stage7-13配置格式修复**: 修复配置文件格式错误（2025-12-05 18:45）
  - 问题: 使用了错误的键名`"repository"`导致KeyError
  - 修复: 将所有配置改为标准`"repo"`键名（7个文件，18个实验定义）
  - 重构: 并行模式配置重写为标准`foreground/background`格式
  - 验证: 所有配置JSON格式通过验证，Stage7测试运行成功
  - 影响: 解除370个实验（178.8小时）的执行阻塞

### v4.5.0 (2025-12-04)
- ✅ **Stage2执行完成**：非并行补充 + 快速模型并行实验成功完成
  - 实际运行: 7.3小时 (预期20-24小时)
  - 实验数量: 25个 (预期44个)
  - 核心目标: 所有29个目标参数达到5个唯一值
  - 去重效果: 跳过40个重复实验 (61.5%跳过率)
- ✅ **CSV列不匹配修复**：修复_append_to_summary_all()使用错误的fieldnames问题
- ✅ **参数精确优化**：每个参数使用精确的runs_per_config值，资源利用率>90%
- ✅ **配置文件合并**：将Stage3和Stage4合并为单个配置文件，重新预估时间至57.1小时

### v4.4.0 (2025-12-02)
- ✅ **CSV追加bug修复**：修复数据覆盖问题，使用安全追加模式
- ✅ **去重机制**：支持基于历史CSV文件的实验去重
- ✅ **分阶段实验计划**：大规模实验分割成可管理的阶段

### v4.3.0 (2025-11-19)
- ✅ **11个模型完整支持**：基线+变异
- ✅ **动态变异系统**：log-uniform/uniform分布
- ✅ **并行训练**：前景+背景GPU同时利用
- ✅ **高精度能耗监控**：CPU误差<2%

### 完整版本历史
查看`docs/FEATURES_OVERVIEW.md`获取完整版本历史。

---

## 📞 问题排查

### 常见问题

#### 1. CSV格式错误
```
症状: GitHub报告"row X should actually have Y columns"
原因: _append_to_summary_all()列不匹配
解决: 运行验证测试，检查runner.py第167-200行
```

#### 2. 实验重复运行
```
症状: 相同超参数重复实验
原因: 去重机制未启用或配置错误
解决: 检查配置文件中的use_deduplication和historical_csvs设置
```

#### 3. 能耗数据缺失
```
症状: energy_*列为空
原因: perf权限问题或nvidia-smi不可用
解决: 使用sudo运行，检查GPU驱动
```

#### 4. 配置执行失败
```
症状: JSON配置文件解析错误或KeyError
原因: 格式错误、路径不正确、或使用了错误的键名
解决:
  - 使用python -m json.tool验证JSON格式
  - 确保使用"repo"而非"repository"
  - 并行模式使用foreground/background结构
  - 参考Stage2配置作为模板
```

#### 5. KeyError: 'repo'
```
症状: 运行配置文件时报KeyError: 'repo'错误
原因: 配置文件使用了"repository"键而非"repo"
解决: 将配置中所有"repository"改为"repo"
参考: docs/results_reports/STAGE7_CONFIG_FIX_REPORT.md
```

### 调试命令
```bash
# 检查CSV格式
python3 tests/verify_csv_append_fix.py

# 验证JSON配置
python3 -m json.tool settings/stage2_optimized_*.json

# 检查去重效果
python3 scripts/check_deduplication.py
```

---

## 📈 项目路线图

### 近期目标（1-2周）⭐
1. ✅ **Stage7-8执行完成** - 19个实验，13.74小时
2. ✅ **配置最终合并完成** - Stage11+12+13整合为单一配置
   - 配置文件: `settings/stage_final_all_remaining.json`
   - 实验数: 78个（pcb:12 + hrnet18:8 + 快速模型:43 + MRT-OAST:7 + VulBERTa/cnn:8）
   - 预计时间: 37.8小时（一次性完成）
   - 资源节省: 38.1%实验数，35.4%时间
3. 🔄 **执行最终阶段** - 完成所有剩余实验 **[下一步]**
   - 执行命令: `sudo -E python3 mutation.py -ec settings/stage_final_all_remaining.json`
   - 预计时间: ~37.8小时
   - 预期: 78个新实验，100%覆盖（90/90参数-模式组合，159/159实验完成）

### 中期改进（1个月）
1. 实现标准37列模板，避免列不匹配问题
2. 增强性能指标提取逻辑（特别是mnist_ff的test指标）
3. 添加CSV格式自动验证和修复工具

### 长期规划（3个月）
1. 扩展到更多深度学习模型
2. 实现更精细的能耗分析功能
3. 开发Web可视化界面

---

## ✅ 质量保证清单

### 每次提交前检查
- [ ] 所有测试通过：`python3 -m pytest tests/`
- [ ] CSV格式正确：`python3 tests/verify_csv_append_fix.py`
- [ ] 文档已更新：相关Markdown文件
- [ ] 配置已归档：过时文件移至`*/archived/`
- [ ] 版本号已更新：README.md和FEATURES_OVERVIEW.md

### 每次发布前检查
- [ ] 完整功能测试：所有11个模型运行正常
- [ ] 数据完整性验证：summary_all.csv格式正确
- [ ] 性能指标提取：所有模型性能数据完整
- [ ] 能耗监控：CPU/GPU数据准确
- [ ] 文档一致性：所有文档反映最新状态

---

**维护者**: Green
**Claude助手指南版本**: 1.5
**最后更新**: 2025-12-14
**状态**: ✅ 有效 - v4.7.5完成Phase 4验证，3个模型非并行模式达标

> 提示：将此文件保存在项目根目录，作为Claude助手的主要参考。当项目结构或规范变更时，及时更新此文档。
