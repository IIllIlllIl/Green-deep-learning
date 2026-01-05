# Claude 助手快速指南 - Energy DL Project

**版本**: v5.2.0
**最后更新**: 2026-01-05
**状态**: ⏳ 进行中 - 能耗数据因果分析阶段

> **提示**: 本文档为精简版快速指南。完整详细文档请查看 [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md)

---

## 📋 快速开始

### 项目概述

本项目研究**深度学习训练超参数对能耗和性能的影响**，通过自动化变异超参数、监控能耗、收集性能指标来支持大规模实验研究。

**核心成果**:
- ✅ **836个实验** (795个有效能耗数据，**95.1%完整性**) 🎉
- ✅ 11个模型在并行/非并行模式下全覆盖
- ✅ 90/90参数-模式组合100%达标
- ✅ **数据完整性修复完成** (69.7% → 95.1%) - 2026-01-04
- ⏳ 能耗数据因果分析进行中

### 当前主要任务 (2026-01-04)

**任务**: 能耗数据回归分析 - 问题1: 超参数对能耗的影响

**进度**:
1. ✅ 新旧数据对比分析完成
2. ✅ 分析方案设计完成（方案A': 6组分层分析）
3. ✅ **数据完整性修复完成** (253个实验数据恢复)
4. ⏳ **当前**: 实现默认值回溯脚本
5. ⏳ 待执行: 6组回归分析
6. ⏳ 待执行: 生成分析报告

**详细信息**: 查看 [docs/CLAUDE_FULL_REFERENCE.md § 当前任务](docs/CLAUDE_FULL_REFERENCE.md#当前任务能耗数据因果与回归分析-2025-12-29-)

---

## 📁 项目结构快览

```
energy_dl/nightly/
├── CLAUDE.md              # 本文件 - 快速指南 ⭐
├── README.md              # 项目总览
├── mutation.py            # 训练入口脚本
│
├── data/                  # ⭐ 核心数据（重组后上浮）
│   ├── raw_data.csv       # 主数据文件（87列，836行，95.1%完整性）⭐⭐⭐
│   ├── data.csv           # 精简数据文件（56列）⭐⭐
│   ├── recoverable_energy_data.json  # 可恢复能耗数据
│   └── backups/           # 数据备份
│
├── tools/                 # ⭐ 数据处理工具（重组后分类）
│   ├── data_management/   # 数据管理工具（15个活跃脚本）
│   ├── config_management/ # 配置管理工具（4个脚本）
│   └── legacy/            # 历史脚本归档
│
├── docs/                  # 项目文档
│   ├── CLAUDE_FULL_REFERENCE.md    # 完整参考文档 ⭐⭐⭐
│   ├── RESTRUCTURE_PLAN_20260105.md  # 重组方案文档
│   ├── reports/           # 实验结果报告
│   ├── environment/       # 环境配置
│   └── archived/          # 过时文档归档
│
├── analysis/              # 因果推断分析模块 🔬（保持独立）
│   ├── docs/
│   │   ├── INDEX.md       # 分析模块文档总索引 ⭐
│   │   ├── QUESTION1_REGRESSION_ANALYSIS_PLAN.md  # 当前任务方案 ⭐⭐⭐
│   │   └── reports/       # 实验报告
│   ├── scripts/           # 分析脚本
│   ├── utils/             # 核心模块
│   └── data/              # 分析数据
│
├── archives/              # ⭐ 历史数据归档（重组后集中）
│   ├── runs/              # 17个历史运行结果
│   └── data_snapshots/    # 历史数据快照
│
├── repos/                 # 训练模型仓库（11个模型）
│   ├── examples/          # 4个示例模型（mnist等）
│   ├── VulBERTa/          # 漏洞检测模型
│   ├── Person_reID_baseline_pytorch/  # 行人重识别模型
│   ├── pytorch_resnet_cifar10/        # ResNet模型
│   ├── MRT-OAST/          # 缺陷定位模型
│   └── bug-localization-by-dnn-and-rvsm/  # Bug定位模型
│
├── mutation/              # 训练核心代码
├── tests/                 # 测试文件
├── settings/              # 实验配置
└── environment/           # 环境配置
```

**✅ 2026-01-05 重组完成**: 数据文件上浮，脚本分类整理，历史数据归档

---

## 📚 核心文档索引

### 必读文档 ⭐⭐⭐

| 文档 | 用途 | 优先级 |
|------|------|--------|
| [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md) | **完整项目参考** - 所有详细信息 | ⭐⭐⭐ |
| [analysis/docs/INDEX.md](analysis/docs/INDEX.md) | **分析模块文档总索引** | ⭐⭐⭐ |
| [analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md](analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md) | **当前任务方案** - 回归分析计划 | ⭐⭐⭐ |
| [docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md) | 项目进度总结 | ⭐⭐ |

### 快速参考

| 文档 | 用途 |
|------|------|
| [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | 快速参考手册 |
| [docs/SCRIPTS_QUICKREF.md](docs/SCRIPTS_QUICKREF.md) | 脚本快速参考 |
| [docs/FEATURES_OVERVIEW.md](docs/FEATURES_OVERVIEW.md) | 功能特性总览 |

### 配置与规范

| 文档 | 用途 |
|------|------|
| [docs/JSON_CONFIG_WRITING_STANDARDS.md](docs/JSON_CONFIG_WRITING_STANDARDS.md) | **JSON配置书写规范** ⭐⭐⭐ |
| [docs/JSON_CONFIG_BEST_PRACTICES.md](docs/JSON_CONFIG_BEST_PRACTICES.md) | 配置最佳实践 |
| [docs/11_MODELS_FINAL_DEFINITION.md](docs/11_MODELS_FINAL_DEFINITION.md) | 11个模型定义 |

### 数据分析相关

| 文档 | 用途 |
|------|------|
| [docs/results_reports/DATA_REPAIR_REPORT_20260104.md](docs/results_reports/DATA_REPAIR_REPORT_20260104.md) | **数据完整性修复报告** (2026-01-04) ⭐⭐⭐ |
| [analysis/docs/reports/DATA_COMPARISON_OLD_VS_NEW_20251229.md](analysis/docs/reports/DATA_COMPARISON_OLD_VS_NEW_20251229.md) | 新旧数据对比分析 |
| [docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md](docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md) | 数据格式设计决定 |

---

## ⚠️ 关键开发规范

### 1. 任务执行流程 ⭐⭐⭐

**每次执行任务前必须遵循**:

1. **理解与规划**: 确认当前进度和规范，列出任务步骤
2. **开发与检查**:
   - ✅ 先编写测试脚本
   - ✅ 先执行Dry Run（10-20行数据）
   - ✅ 验证通过后再全量执行
   - ✅ 运行全量测试
3. **维护与归档**: 更新文档，归档旧文件
4. **使用中文回答**: 所有回复和文档必须使用中文

### 2. 脚本开发规范 ⭐⭐⭐

```bash
# ❌ 错误做法：直接全量执行
python3 script.py --input data.csv --output result.csv

# ✅ 正确做法：测试 → Dry Run → 全量执行
# 步骤1: 编写并运行测试
python3 test_script.py

# 步骤2: Dry Run（前10行）
python3 script.py --input data.csv --output result.csv --dry-run --limit 10

# 步骤3: 检查Dry Run结果
head -20 result.csv
python3 verify_result.py --file result.csv

# 步骤4: 全量执行
python3 script.py --input data.csv --output result.csv
```

### 3. JSON配置规范 ⭐⭐⭐

**核心原则**: 每个实验配置只能变异一个超参数

```json
{
  "comment": "VulBERTa/mlp - learning_rate变异",
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "mutate": ["learning_rate"],  // 单参数变异
  "runs_per_config": 5
}
```

**必须遵循**:
- ✅ 使用 `"repo"` 而非 `"repository"`
- ✅ 使用 `"mode"` 而非 `"mutation_type"`
- ✅ 使用 `"mutate": ["参数名"]` 数组格式
- ❌ 禁止使用 `"mutate_params"` 对象格式

**详细规范**: [docs/JSON_CONFIG_WRITING_STANDARDS.md](docs/JSON_CONFIG_WRITING_STANDARDS.md)

---

## 🚨 常见错误与解决方案

### 1. 实验ID唯一性问题 ⭐⭐⭐

**问题**: 不同批次的实验会产生相同的experiment_id

**正确做法**: 使用 **experiment_id + timestamp** 作为唯一标识符

```python
# ✅ 正确 - 使用复合键
composite_key = f"{row['experiment_id']}|{row['timestamp']}"
if composite_key in existing_keys:
    ...
```

### 2. CSV数据完整性

- `raw_data.csv` 为主数据文件（位置：`data/raw_data.csv`），使用复合键去重
- 追加数据时使用 `tools/data_management/append_session_to_raw_data.py`
- 每次修改前务必备份

### 3. 能耗数据缺失

**症状**: energy_* 列为空
**原因**: perf权限问题或nvidia-smi不可用
**解决**: 使用sudo运行，检查GPU驱动

### 4. 配置执行失败

**症状**: JSON配置文件解析错误或KeyError
**解决**:
- 使用 `python -m json.tool` 验证JSON格式
- 确保使用 `"repo"` 而非 `"repository"`
- 并行模式使用 `foreground/background` 结构
- 参考 `settings/stage2_optimized_*.json` 作为模板

---

## 🔬 分析模块 (analysis/)

### 概述

`analysis/` 是独立的因果推断分析模块，用于研究超参数对能耗和性能的因果影响。

**核心技术**: DiBS（因果图学习）+ DML（因果推断） + 回归分析

**当前任务**: 使用回归分析回答3个核心研究问题
1. ⏳ 超参数对能耗的影响（方向和大小）
2. ⏳ 能耗和性能之间的权衡关系
3. ⏳ 中间变量的中介效应

### 快速参考

**主要文档**:
- [analysis/README.md](analysis/README.md) - 模块总体介绍
- [analysis/docs/INDEX.md](analysis/docs/INDEX.md) - 文档总索引
- [analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md](analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md) - 问题1方案

**数据位置**:
- 论文复现: `analysis/data/paper_replication/`
- 能耗研究: `analysis/data/energy_research/`

---

## 🛠️ 常用命令

### 查看项目状态

```bash
# 验证数据完整性
python3 tools/data_management/validate_raw_data.py

# 查看最新版本
grep '当前版本' README.md
```

### 运行测试

```bash
# CSV修复验证测试
python3 tests/verify_csv_append_fix.py

# 所有单元测试
python3 -m pytest tests/unit/
```

### 数据分析

```bash
# 分析实验状况
python3 tools/data_management/analyze_experiment_status.py

# 分析缺失能耗数据
python3 tools/data_management/analyze_missing_energy_data.py

# 验证数据可恢复性
python3 tools/data_management/verify_recoverable_data.py

# 修复缺失能耗数据
python3 tools/data_management/repair_missing_energy_data.py
```

---

## 📊 数据文件说明

### raw_data.csv (主数据文件) ⭐⭐⭐

- **位置**: `data/raw_data.csv` ✅ (2026-01-05 重组后)
- **旧位置**: ~~`results/raw_data.csv`~~ (已废弃)
- **规模**: 836行（含header），87列
- **完整性**: **795/836 (95.1%)** 有效能耗数据 🎉
- **最后更新**: 2026-01-04 (数据修复完成)
- **备份**: `data/backups/raw_data.csv.backup_*`
- **验证**: `tools/data_management/validate_raw_data.py`

### data.csv (精简数据文件) ⭐⭐

- **位置**: `data/data.csv` ✅ (2026-01-05 重组后)
- **旧位置**: ~~`results/data.csv`~~ (已废弃)
- **规模**: 待更新（原726行，56列）
- **特性**: 统一并行/非并行字段，添加 is_parallel 列
- **生成**: `tools/data_management/create_unified_data_csv.py`
- **状态**: ⚠️ 需要重新生成以反映最新数据

### 数据修复记录

**修复日期**: 2026-01-04
**修复实验数**: 253个
**完整性提升**: 69.7% → 95.1% (+25.4%)

**相关文件**:
- 修复报告: `docs/results_reports/DATA_REPAIR_REPORT_20260104.md`
- 修复日志: `data/backups/data_repair_log_*`
- 数据来源: `data/recoverable_energy_data.json`

---

## 📞 获取帮助

### 文档导航

1. **快速开始**: 本文档（CLAUDE.md）
2. **详细参考**: [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md)
3. **分析模块**: [analysis/docs/INDEX.md](analysis/docs/INDEX.md)
4. **项目进度**: [docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md)

### 常见问题

如果遇到问题，请依次检查：
1. 本文档的"常见错误与解决方案"章节
2. [docs/CLAUDE_FULL_REFERENCE.md § 常见问题](docs/CLAUDE_FULL_REFERENCE.md#📞-常见问题)
3. 相关的专题文档（如JSON配置规范、数据格式设计等）

---

**维护者**: Green
**文档版本**: 5.2.0
**最后更新**: 2026-01-05
**重要更新**: 项目文件结构重组完成 (数据上浮 + 脚本分类 + 历史归档)

> **记住**: 本文档只是快速指南！详细信息和完整上下文请始终参考 [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md)
