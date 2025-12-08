# Claude 助手指南 - Mutation-Based Training Energy Profiler

**项目版本**: v4.7.2 (2025-12-08)
**最后更新**: 2025-12-08
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

### 当前状态
- **实验总数**: 404个（results/summary_all.csv）
- **完成度**: Stage1-4, Stage7-8完成（81/159实验，51%）
- **剩余任务**: 最终阶段（78个实验，37.8小时）**[一次性完成所有剩余实验]**
- **参数-模式组合**: 需补充17个组合达到100%覆盖
- **配置状态**: 已完成最终整合和优化
- **最新进展** (2025-12-08):
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
- `results/summary_all.csv` - 所有实验汇总（388行，37列）
- `results/summary_all.csv.backup` - 数据备份
- `settings/EXECUTION_READY.md` - 执行准备清单（已归档）

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
- `docs/FEATURES_OVERVIEW.md` - 功能特性总览
- `docs/QUICK_REFERENCE.md` - 快速参考
- `docs/SETTINGS_CONFIGURATION_GUIDE.md` - 配置指南
- `docs/JSON_CONFIG_BEST_PRACTICES.md` - JSON配置最佳实践 ⭐⭐⭐ **[v4.7.1新增]**
- `docs/results_reports/CSV_FIX_COMPREHENSIVE_SUMMARY.md` - CSV修复综合报告
- `docs/results_reports/MISSING_COLUMNS_DETAILED_ANALYSIS.md` - 缺失列详细分析
- `docs/results_reports/DEDUP_MODE_FIX_REPORT.md` - 去重模式修复报告
- `docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md` - 实验需求分析
- `docs/results_reports/STAGE3_4_EXECUTION_REPORT.md` - Stage3-4执行报告
- `docs/results_reports/STAGE7_CONFIG_FIX_REPORT.md` - Stage7-13配置修复报告
- `docs/results_reports/STAGE7_EXECUTION_REPORT.md` - Stage7执行报告
- `docs/results_reports/STAGE7_13_CONFIG_BUG_ANALYSIS.md` - Stage7-13配置Bug详细分析 **[v4.7.1]**
- `docs/results_reports/STAGE7_8_FIX_EXECUTION_REPORT.md` - Stage7-8修复执行报告 **[v4.7.1]**
- `docs/results_reports/STAGE11_BUG_FIX_REPORT.md` - Stage11 Bug修复报告 ⭐⭐⭐ **[v4.7.2]**
- `docs/results_reports/STAGE11_ACTUAL_STATE_CORRECTION.md` - Stage11实际状态修正 ⭐⭐ **[v4.7.2新增]**
- `docs/results_reports/FINAL_CONFIG_INTEGRATION_REPORT.md` - 最终配置整合报告 ⭐⭐⭐ **[v4.7.2新增]**
- `docs/results_reports/DEDUPLICATION_RANDOM_MUTATION_ANALYSIS.md` - 去重与随机变异分析 ⭐ **[v4.7.2]**
- `docs/results_reports/PROJECT_CLEANUP_SUMMARY_20251208.md` - 项目清理总结 **[v4.7.2]**
- `docs/SCRIPTS_DOCUMENTATION.md` - Scripts目录完整文档 **[v4.7.2新增]**
- `docs/results_reports/DAILY_SUMMARY_20251205.md` - 2025-12-05每日总结
- `docs/settings_reports/STAGE7_13_EXECUTION_PLAN.md` - Stage7-13执行计划
- `docs/results_reports/STAGE7_13_DESIGN_SUMMARY.md` - Stage7-13设计总结

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

### 1. CSV数据完整性
- `summary_all.csv`必须保持37列格式
- 追加数据时使用`extrasaction='ignore'`自动对齐列
- 每次修改前备份：`cp summary_all.csv summary_all.csv.backup`

### 2. 配置版本控制
- 创建新配置时保留旧版本在`settings/archived/`
- 在配置文件中添加`comment`字段说明变更原因
- 更新`docs/settings_reports/`中的相关文档
- **配置格式规范**：
  - 使用`"repo"`而非`"repository"`（与runner.py一致）
  - 非并行模式：直接在实验定义中指定`repo`、`model`、`mutate_params`
  - 并行模式：使用`foreground/background`嵌套结构
  - 参考已验证的配置文件（如Stage2）作为模板

### 3. 测试要求
- 所有代码变更必须通过现有测试
- 新功能必须添加测试用例
- CSV格式变更必须运行`verify_csv_append_fix.py`

### 4. 文档同步
- 代码变更时更新相关文档
- 使用标准Markdown格式
- 在文档头部包含版本和日期信息

---

## 🔄 版本历史摘要

### v4.7.2 (2025-12-08) - 当前版本 ⭐
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
**Claude助手指南版本**: 1.3
**最后更新**: 2025-12-08
**状态**: ✅ 有效 - v4.7.2修复了并行模式runs_per_config bug

> 提示：将此文件保存在项目根目录，作为Claude助手的主要参考。当项目结构或规范变更时，及时更新此文档。
