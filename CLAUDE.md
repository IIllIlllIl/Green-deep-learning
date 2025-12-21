# Claude 助手指南 - Mutation-Based Training Energy Profiler

**项目版本**: v4.7.9 (2025-12-20)
**最后更新**: 2025-12-20
**状态**: 🎉 Complete - All 11 Models 100% Achieved!

---

## 📋 任务执行要求

**重要**: 每次执行任务前必须遵循以下流程：

1. **理解与规划**：首先确认从 `CLAUDE.md` 中获取的当前进度和规范，并列出本次任务的具体步骤。
2. **开发与检查**：若需修改、生成或删除代码，请先编写/更新测试验证，再运行全量测试确保兼容性。
3. **维护与归档**：任务完成后，更新 `README.md`、`CLAUDE.md` 中的进度，并归档旧文件。
4. **使用中文回答**：所有回复和文档必须使用中文。

---

## 🎯 项目概述

### 核心目标
研究深度学习训练超参数对能耗和性能的影响。通过自动化变异超参数、监控能耗、收集性能指标，支持大规模实验研究。

### 当前状态 🎉

**🎉 重大成就 (2025-12-19)**: Phase 7完成，项目目标100%达成！

| 指标 | 当前值 | 状态 |
|------|--------|------|
| **实验总数** | 676个 | ✅ 完成 |
| **有效模型** | 11个 | ✅ 100%达标 |
| **训练成功率** | 676/676 (100.0%) | ✅ 完美 |
| **能耗数据** | 616/676 (91.1%) | ✅ 优秀 |
| **性能数据** | 616/676 (91.1%) | ✅ 优秀 |
| **参数-模式组合** | 90/90 (100%) | ✅ 完成 |

**详细进度**: 查看 [项目进度完整总结](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md) ⭐⭐⭐

### 实验目标

每个超参数在两种模式（并行/非并行）下都需要：
1. **1个默认值实验** - 建立基线
2. **5个唯一单参数变异实验** - 研究单参数影响
3. **完整数据**: 能耗 + 性能指标

**总目标**: 45参数 × 2模式 × 6实验 = **540个有效实验**
**实际达成**: 616个有效实验，90/90参数-模式组合全部达标 (100%) 🎊

---

## 📁 项目结构

### 目录结构
```
energy_dl/nightly/
├── docs/                    # 所有文档文件
│   ├── results_reports/    # 实验结果报告 ⭐
│   ├── environment/        # 环境配置文档
│   ├── settings_reports/   # 配置相关报告
│   └── archived/           # 过时文档归档
├── scripts/                # 辅助脚本（Python/Bash）
├── tests/                  # 测试文件
├── settings/               # 运行配置文件
│   └── archived/          # 过时配置归档
├── results/               # 实验结果数据
├── mutation/              # 核心代码
└── config/                # 模型配置
```

### 核心文件位置

**代码文件**:
- `mutation/runner.py` - 主运行器
- `mutation/session.py` - Session管理
- `mutation/energy_monitor.py` - 能耗监控
- `config/models_config.json` - 模型配置

**数据文件**:
- `results/raw_data.csv` - **主数据文件** (676行，87列) ⭐⭐⭐
  - 验证脚本: `scripts/validate_raw_data.py`
  - 追加脚本: `scripts/append_session_to_raw_data.py`
  - 备份文件: `results/raw_data.csv.backup_20251219_154656`

**配置文件**:
- 已完成: `settings/stage2_optimized_*.json`, `stage3_4_merged_*.json`
- 归档: `settings/archived/` - 过时配置文件

---

## 📚 核心文档索引

### 快速参考指南
- [项目进度完整总结](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md) - **项目总体状况** ⭐⭐⭐
- [FEATURES_OVERVIEW.md](docs/FEATURES_OVERVIEW.md) - 功能特性总览
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - 快速参考
- [SCRIPTS_QUICKREF.md](docs/SCRIPTS_QUICKREF.md) - 脚本快速参考

### 配置与规范
- [JSON_CONFIG_WRITING_STANDARDS.md](docs/JSON_CONFIG_WRITING_STANDARDS.md) - **JSON配置书写规范** ⭐⭐⭐
- [JSON_CONFIG_BEST_PRACTICES.md](docs/JSON_CONFIG_BEST_PRACTICES.md) - 配置最佳实践
- [SETTINGS_CONFIGURATION_GUIDE.md](docs/SETTINGS_CONFIGURATION_GUIDE.md) - 配置指南
- [11_MODELS_FINAL_DEFINITION.md](docs/11_MODELS_FINAL_DEFINITION.md) - 11个模型最终定义

### 执行报告
- [PHASE7_EXECUTION_REPORT.md](docs/results_reports/PHASE7_EXECUTION_REPORT.md) - **Phase 7最终补齐** ⭐⭐⭐
- [PHASE6_EXECUTION_REPORT.md](docs/results_reports/PHASE6_EXECUTION_REPORT.md) - Phase 6 VulBERTa补齐
- [PHASE5_PERFORMANCE_METRICS_MISSING_ISSUE.md](docs/results_reports/PHASE5_PERFORMANCE_METRICS_MISSING_ISSUE.md) - Phase 5执行
- [PHASE4_VALIDATION_EXECUTION_REPORT.md](docs/results_reports/PHASE4_VALIDATION_EXECUTION_REPORT.md) - Phase 4验证
- [STAGE3_4_EXECUTION_REPORT.md](docs/results_reports/STAGE3_4_EXECUTION_REPORT.md) - Stage3-4合并执行
- [STAGE7_8_FIX_EXECUTION_REPORT.md](docs/results_reports/STAGE7_8_FIX_EXECUTION_REPORT.md) - Stage7-8修复执行

### Bug修复报告
- [STAGE11_BUG_FIX_REPORT.md](docs/results_reports/STAGE11_BUG_FIX_REPORT.md) - 并行runs_per_config修复 ⭐⭐⭐
- [CSV_FIX_COMPREHENSIVE_SUMMARY.md](docs/results_reports/CSV_FIX_COMPREHENSIVE_SUMMARY.md) - CSV追加bug综合报告
- [DEDUP_MODE_FIX_REPORT.md](docs/results_reports/DEDUP_MODE_FIX_REPORT.md) - 去重模式区分修复
- [STAGE7_CONFIG_FIX_REPORT.md](docs/results_reports/STAGE7_CONFIG_FIX_REPORT.md) - Stage7-13配置修复
- [VULBERTA_CNN_CLEANUP_REPORT_20251211.md](docs/results_reports/VULBERTA_CNN_CLEANUP_REPORT_20251211.md) - VulBERTa/cnn清理

### 数据格式与设计
- [DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md) - **数据格式设计决定** ⭐⭐⭐
- [SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md](docs/results_reports/SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md) - 80列vs93列对比
- [OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md](docs/results_reports/OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md) - 背景超参数分析
- [OUTPUT_STRUCTURE_QUICKREF.md](docs/OUTPUT_STRUCTURE_QUICKREF.md) - 输出结构参考

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

## ⚠️ 关键注意事项

### 1. 实验ID唯一性 ⭐⭐⭐ **[关键经验]**

**问题**: 不同批次的实验会产生相同的experiment_id

**正确做法**: 使用 **experiment_id + timestamp** 作为唯一标识符

```python
# ✅ 正确 - 使用复合键
composite_key = f"{row['experiment_id']}|{row['timestamp']}"
if composite_key in existing_keys:
    ...
```

**详细**: [Phase 5性能指标缺失问题分析](docs/results_reports/PHASE5_PERFORMANCE_METRICS_MISSING_ISSUE.md) ⭐⭐⭐

### 2. JSON配置书写规范 ⭐⭐⭐

**核心原则**: 每个实验配置只能变异一个超参数

**必须遵循**:
- ✅ 使用`"mutate": ["参数名"]`数组格式（**单参数变异**）
- ✅ 使用`"repo"`而非`"repository"`
- ✅ 使用`"mode"`而非`"mutation_type"`
- ❌ **禁止**使用`"mutate_params"`对象格式

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

**详细文档**: [JSON_CONFIG_WRITING_STANDARDS.md](docs/JSON_CONFIG_WRITING_STANDARDS.md) ⭐⭐⭐

### 3. CSV数据完整性
- `raw_data.csv`为主数据文件，使用复合键去重
- 追加数据时使用`append_session_to_raw_data.py`
- 每次修改前备份

### 4. 测试要求
- 所有代码变更必须通过现有测试
- 新功能必须添加测试用例
- CSV格式变更必须运行`verify_csv_append_fix.py`

### 5. 文档同步
- 代码变更时更新相关文档
- 使用标准Markdown格式
- 在文档头部包含版本和日期信息

---

## 🚀 快速开始命令

### 查看项目状态
```bash
# 验证数据完整性
python3 scripts/validate_raw_data.py

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

### 数据分析
```bash
# 分析唯一值数量
python3 scripts/analyze_unique_values.py

# 计算实验缺口
python3 scripts/calculate_experiment_gap.py
```

### 调试命令
```bash
# 检查CSV格式
python3 tests/verify_csv_append_fix.py

# 验证JSON配置
python3 -m json.tool settings/stage2_optimized_*.json
```

---

## 📊 最新进展

### v4.7.9 (2025-12-19) - 当前版本 🎉 **[项目完成]**

**Phase 7执行完成**: 最终补齐所有剩余实验（52实验，26.79小时）
- **实验数**: 52个 (VulBERTa/mlp非并行:20 + bug-localization非并行:20 + bug-localization并行:12)
- **执行时间**: 2025-12-17 21:13 - 2025-12-19 01:14
- **数据完整性**: 100% (训练成功、能耗、性能指标)
- **详细报告**: [PHASE7_EXECUTION_REPORT.md](docs/results_reports/PHASE7_EXECUTION_REPORT.md)

**关键Bug修复**: 非并行模式数据提取Bug ⭐⭐⭐
- **问题**: `append_session_to_raw_data.py` 使用错误字段名 `energy_consumption`
- **影响**: 所有非并行实验的能耗和性能数据为空
- **修复**: 统一使用 `energy_metrics` 和 `performance_metrics`，添加字段映射
- **修改文件**: `scripts/append_session_to_raw_data.py:215-268`
- **结果**: 重新合并后100%数据完整

**项目目标100%达成** 🎊:
- 11个模型在两种模式（非并行+并行）下全部达标
- 90/90 参数-模式组合 (100%)
- 有效实验: 616个
- 剩余缺口: 0个实验 ✅

**完整版本历史**: 查看 [项目进度完整总结](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md)

---

## 📞 常见问题

### 1. CSV格式错误
```
症状: GitHub报告"row X should actually have Y columns"
原因: _append_to_summary_all()列不匹配
解决: 运行验证测试，检查runner.py第167-200行
```

### 2. 实验重复运行
```
症状: 相同超参数重复实验
原因: 去重机制未启用或配置错误
解决: 检查配置文件中的use_deduplication和historical_csvs设置
```

### 3. 能耗数据缺失
```
症状: energy_*列为空
原因: perf权限问题或nvidia-smi不可用
解决: 使用sudo运行，检查GPU驱动
```

### 4. 配置执行失败
```
症状: JSON配置文件解析错误或KeyError
解决:
  - 使用python -m json.tool验证JSON格式
  - 确保使用"repo"而非"repository"
  - 并行模式使用foreground/background结构
  - 参考Stage2配置作为模板
```

### 5. KeyError: 'repo'
```
症状: 运行配置文件时报KeyError: 'repo'错误
原因: 配置文件使用了"repository"键而非"repo"
解决: 将配置中所有"repository"改为"repo"
参考: docs/results_reports/STAGE7_CONFIG_FIX_REPORT.md
```

---

## ✅ 质量保证清单

### 每次提交前检查
- [ ] 所有测试通过：`python3 -m pytest tests/`
- [ ] CSV格式正确：`python3 tests/verify_csv_append_fix.py`
- [ ] 文档已更新：相关Markdown文件
- [ ] 配置已归档：过时文件移至`*/archived/`
- [ ] 版本号已更新：README.md和相关文档

### 每次发布前检查
- [ ] 完整功能测试：所有11个模型运行正常
- [ ] 数据完整性验证：raw_data.csv格式正确
- [ ] 性能指标提取：所有模型性能数据完整
- [ ] 能耗监控：CPU/GPU数据准确
- [ ] 文档一致性：所有文档反映最新状态

---

## 📈 数据文件说明

### raw_data.csv (主数据文件) ⭐⭐⭐

**文件信息**:
- 总行数: 676行
- 列数: 87列
- 数据来源: 合并所有实验数据（历史211 + 新265 + Phase4-7共200）

**数据完整性**:
- ✅ 训练成功: 676/676 (100.0%)
- ✅ 能耗数据: 616/676 (91.1%)
- ✅ 性能数据: 616/676 (91.1%)

**相关脚本**:
- `scripts/validate_raw_data.py` - 验证数据完整性
- `scripts/append_session_to_raw_data.py` - 追加session数据
- `scripts/merge_csv_to_raw_data.py` - 合并数据文件

**数据格式设计**: 查看 [DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md) ⭐⭐⭐

---

## 🗂️ 11个有效模型

### 快速模型（训练时间 < 5分钟）
1. **examples/mnist** - 基础MNIST CNN (~2分钟)
2. **examples/mnist_ff** - MNIST前馈网络 (~1.5分钟)
3. **examples/mnist_rnn** - MNIST RNN (~3分钟)
4. **examples/siamese** - Siamese网络 (~4分钟)

### 中速模型（训练时间 5-30分钟）
5. **pytorch_resnet_cifar10/resnet20** - ResNet20 CIFAR10 (~15分钟)
6. **MRT-OAST/default** - 多目标优化 (~20分钟)
7. **VulBERTa/mlp** - 漏洞检测MLP (~25分钟)
8. **bug-localization-by-dnn-and-rvsm/default** - Bug定位 (~18分钟)

### 慢速模型（训练时间 > 30分钟）
9. **Person_reID/densenet121** - 行人重识别 (~45分钟)
10. **Person_reID/hrnet18** - 行人重识别HRNet (~50分钟)
11. **Person_reID/pcb** - 行人重识别PCB (~60分钟)

**详细定义**: [11_MODELS_FINAL_DEFINITION.md](docs/11_MODELS_FINAL_DEFINITION.md)

---

## 🛠️ 关键脚本工具

### 数据处理
- `scripts/merge_csv_to_raw_data.py` - 合并所有实验数据
- `scripts/append_session_to_raw_data.py` - 追加session数据
- `scripts/validate_raw_data.py` - 验证数据完整性

### 分析工具
- `scripts/calculate_experiment_gap.py` - 计算实验缺口
- `scripts/analyze_experiments.py` - 实验数据分析
- `scripts/analyze_unique_values.py` - 唯一值分析

### 配置工具
- `scripts/generate_config.py` - 生成实验配置
- `scripts/validate_config.py` - 验证配置文件
- `scripts/fix_stage_configs.py` - 修复配置文件

### 测试脚本
- `tests/verify_csv_append_fix.py` - CSV追加修复验证
- `tests/test_dedup_mode_distinction.py` - 去重模式区分测试
- `tests/test_runs_per_config_fix.py` - runs_per_config修复测试
- `tests/test_parallel_runs_per_config_fix.py` - 并行模式修复测试

**完整文档**: [SCRIPTS_QUICKREF.md](docs/SCRIPTS_QUICKREF.md)

---

**维护者**: Green
**Claude助手指南版本**: 2.0
**最后更新**: 2025-12-20
**项目状态**: 🎉 完成 - v4.7.9 项目目标100%达成，11个模型全部完全达标！

> **提示**：
> - 本文件为Claude助手的主要参考指南，保持简洁易读
> - 详细内容查看对应的文档链接
> - 每次任务前请先阅读"任务执行要求"章节
> - 当项目结构或规范变更时，及时更新此文档
