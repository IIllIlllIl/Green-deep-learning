# Claude 助手快速指南 - Energy DL Project

**版本**: v5.6.0
**最后更新**: 2026-01-10
**状态**: ⏳ 进行中 - 能耗数据因果分析阶段

> **提示**: 本文档为精简版快速指南。完整详细文档请查看 [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md)

---

## 📑 目录

- [📋 快速开始](#快速开始)
  - [项目概述](#项目概述)
  - [当前主要任务](#当前主要任务)
- [⚡ 5分钟快速验证](#5分钟快速验证)
- [📁 项目结构快览](#项目结构快览)
- [📚 核心文档索引](#核心文档索引)
- [⚠️ 关键开发规范](#关键开发规范)
- [🚨 常见错误与解决方案](#常见错误与解决方案)
- [🔍 脚本复用检查指南](#脚本复用检查指南)
- [🔬 分析模块 (analysis/)](#分析模块-analysis)
- [🛠️ 常用命令](#常用命令)
- [📊 数据文件说明](#数据文件说明)
- [📞 获取帮助](#获取帮助)

**💡 提示**: 点击章节标题快速跳转到相应内容

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

### 当前主要任务

#### 任务1: 实验扩展计划 (2026-01-05) 📋 新增

**目标**: 扩展实验覆盖，从单参数变异到多参数变异实验设计

**关键变化**:
- ✅ 允许多参数同时变异（不再限制单参数）
- ✅ 每个模型：1个默认 + n个变异实验
- ✅ 保留去重机制
- ✅ 目标运行时间：约1周（7天）

**推荐方案 - 分层n值（平衡）**:
- 快速模型（2个）: n=80 → 162个实验
- 中速模型（5个）: n=40 → 205个实验
- 慢速模型（4个）: n=25 → 104个实验
- **总计**: 471个新实验，预计7.53天完成 ✅

**详细方案**: 查看 [docs/EXPERIMENT_EXPANSION_PLAN_20260105.md](docs/EXPERIMENT_EXPANSION_PLAN_20260105.md) ⭐⭐⭐

#### 任务2: 能耗数据回归分析 (2026-01-04)

**任务**: 问题1: 超参数对能耗的影响

**进度**:
1. ✅ 新旧数据对比分析完成
2. ✅ 分析方案设计完成（方案A': 6组分层分析）
3. ✅ **数据完整性修复完成** (253个实验数据恢复)
4. ⏳ **当前**: 实现默认值回溯脚本
5. ⏳ 待执行: 6组回归分析
6. ⏳ 待执行: 生成分析报告

**详细信息**: 查看 [docs/CLAUDE_FULL_REFERENCE.md § 当前任务](docs/CLAUDE_FULL_REFERENCE.md#当前任务能耗数据因果与回归分析-2025-12-29-)

---

## ⚡ 5分钟快速验证

**目的**: 快速检查环境、数据和工具是否就绪，确保可以开始工作

### 一键健康检查

```bash
# 执行快速健康检查脚本
bash tools/quick_health_check.sh
```

### 预期输出示例

```
==========================================================
Energy DL 项目健康检查
==========================================================

[1/6] 检查 Python 环境...
  ✅ Python 版本: 3.10.12
  ✅ pip 可用

[2/6] 检查 GPU 驱动...
  ✅ NVIDIA-SMI: 535.129.03
  ✅ CUDA 版本: 12.2
  ✅ GPU 设备: NVIDIA GeForce RTX 3090 (1个)

[3/6] 检查项目依赖...
  ✅ pandas: 2.1.4
  ✅ numpy: 1.26.2
  ✅ torch: 2.1.0

[4/6] 检查核心数据文件...
  ✅ data/raw_data.csv 存在 (836行, 87列)
  ✅ 数据完整性: 95.1% (795/836有效能耗数据)

[5/6] 检查核心脚本...
  ✅ 数据管理脚本: 11/11 可用
  ✅ 配置管理脚本: 4/4 可用

[6/6] 检查实验结果目录...
  ⚠️  results/run_20260110_143022 缺失 (历史结果)
  ✅ archives/ 目录可访问

==========================================================
健康检查完成！
==========================================================

总体状态: ✅ 环境就绪

建议:
  - 数据完整性: 95.1% (优秀)
  - 可以开始新实验或数据分析

运行第一个测试实验:
  sudo python3 mutation.py -ec settings/quick_test.json

查看项目状态:
  python3 tools/data_management/validate_raw_data.py
```

### 手动快速检查 (无脚本时)

如果 `quick_health_check.sh` 不存在，可以手动执行以下命令：

```bash
# 1. 检查 Python 环境
python3 --version  # 应显示 3.8+

# 2. 检查 GPU 驱动
nvidia-smi  # 应显示 GPU 信息

# 3. 检查数据完整性
python3 tools/data_management/validate_raw_data.py

# 4. 查看项目状态
ls -lh data/raw_data.csv  # 应存在且大小 > 0
head -5 data/raw_data.csv # 查看数据格式

# 5. 测试核心脚本
python3 tools/data_management/analyze_experiment_status.py
```

### 故障排查快速指南

| 问题 | 快速解决方案 |
|------|-------------|
| ❌ **GPU驱动失败** | → 检查 `nvidia-smi` 命令，安装/更新 NVIDIA 驱动 |
| ❌ **数据完整性<90%** | → [数据修复指南](docs/results_reports/DATA_REPAIR_REPORT_20260104.md) |
| ❌ **Python依赖缺失** | → `pip3 install -r requirements.txt` |
| ❌ **脚本权限错误** | → `chmod +x tools/quick_health_check.sh` |
| ❌ **数据文件不存在** | → 检查 `data/` 目录，可能需要从备份恢复 |

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
│   ├── data_management/   # 数据管理工具（11个活跃脚本）⭐
│   ├── config_management/ # 配置管理工具（4个脚本）
│   └── legacy/            # 历史脚本归档（2026-01-10清理）
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
| [docs/EXPERIMENT_EXPANSION_PLAN_20260105.md](docs/EXPERIMENT_EXPANSION_PLAN_20260105.md) | **实验扩展方案** - 多参数变异设计 (2026-01-05新增) | ⭐⭐⭐ |
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

**核心规范（点击查看详细）⭐⭐⭐**:

| 规范 | 核心原则 | 详细文档 |
|------|---------|---------|
| **1. 任务执行流程** | 测试先行 → Dry Run → 全量执行 → 独立验证 | [DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) |
| **2. 脚本开发规范** | 必需支持 `--dry-run` 和测试用例 | [SCRIPT_DEV_STANDARDS.md](docs/SCRIPT_DEV_STANDARDS.md) |
| **3. JSON配置规范** | 使用 `"repo"`, `"mode"`, `"mutate"` 格式 | [JSON_CONFIG_WRITING_STANDARDS.md](docs/JSON_CONFIG_WRITING_STANDARDS.md) |
| **4. 独立验证规范** | 使用 Subagent 避免自我验证偏差 | [INDEPENDENT_VALIDATION_GUIDE.md](docs/INDEPENDENT_VALIDATION_GUIDE.md) |

### 快速检查清单 ✅

**在执行任务前，确认**:

- [ ] 📖 已阅读任务执行流程（3分钟） - [查看详细流程](docs/DEVELOPMENT_WORKFLOW.md#任务执行流程)
- [ ] 🧪 已编写测试脚本
- [ ] 🔍 已执行 Dry Run 验证（10-20行数据）
- [ ] ✅ Dry Run 结果验证通过
- [ ] 🤖 已启动独立 Subagent 检查（数据处理/测试编写/配置验证等关键任务）
- [ ] 📝 已更新相关文档

### 核心流程速查

```bash
# 标准开发流程
步骤1: 编写测试  → python3 test_script.py
步骤2: Dry Run   → python3 script.py --input data.csv --dry-run --limit 10
步骤3: 检查结果  → head result.csv && python3 verify_result.py
步骤4: 全量执行  → python3 script.py --input data.csv
步骤5: 独立验证  → 启动 Subagent 检查数据质量
```

### JSON配置快速参考

```json
{
  "comment": "模型/子模型 - 参数变异说明",
  "repo": "模型名",              // ✅ 使用 "repo" (非 "repository")
  "model": "子模型名",
  "mode": "mutation",            // ✅ 使用 "mode" (非 "mutation_type")
  "mutate": ["参数名"],          // ✅ 单参数变异
  "runs_per_config": 5
}
```

**详细规范**: [JSON_CONFIG_WRITING_STANDARDS.md](docs/JSON_CONFIG_WRITING_STANDARDS.md) | [多参数变异方案](docs/EXPERIMENT_EXPANSION_PLAN_20260105.md)

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

## 🔍 脚本复用检查指南 ⭐⭐⭐

**在创建新脚本前，请先检查是否已有可用脚本**

### 为什么要检查复用性？

- ✅ **避免重复**: 减少维护负担，保持代码库整洁
- ✅ **提高效率**: 复用现有脚本比重新开发更快
- ✅ **保证质量**: 现有脚本经过测试和实际使用验证
- ✅ **统一标准**: 使用统一的工具保持数据处理一致性

### 数据管理常用脚本 (11个活跃脚本)

| 任务 | 使用脚本 | 命令示例 |
|------|----------|----------|
| **追加新实验** ⭐ | `append_session_to_raw_data.py` | `python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS` |
| 验证数据完整性 | `validate_raw_data.py` | `python3 tools/data_management/validate_raw_data.py` |
| 分析实验状况 | `analyze_experiment_status.py` | `python3 tools/data_management/analyze_experiment_status.py` |
| 分析缺失能耗 | `analyze_missing_energy_data.py` | `python3 tools/data_management/analyze_missing_energy_data.py` |
| 验证可恢复数据 | `verify_recoverable_data.py` | `python3 tools/data_management/verify_recoverable_data.py` |
| 修复缺失能耗 | `repair_missing_energy_data.py` | `python3 tools/data_management/repair_missing_energy_data.py` |
| 检查最新结果 | `check_latest_results.py` | `python3 tools/data_management/check_latest_results.py` |
| 比较CSV一致性 | `compare_data_vs_raw_data.py` | `python3 tools/data_management/compare_data_vs_raw_data.py` |
| 创建统一data.csv | `create_unified_data_csv.py` | `python3 tools/data_management/create_unified_data_csv.py` |
| 检查属性映射 | `check_attribute_mapping.py` | `python3 tools/data_management/check_attribute_mapping.py` |

**完整列表**: 参见 [docs/SCRIPTS_QUICKREF.md](docs/SCRIPTS_QUICKREF.md)

### 配置管理脚本 (4个)

| 任务 | 使用脚本 |
|------|----------|
| 生成变异配置 | `tools/config_management/generate_mutation_config.py` |
| 验证模型配置 | `tools/config_management/validate_models_config.py` |
| 验证变异配置 | `tools/config_management/validate_mutation_config.py` |
| 验证stage配置 | `tools/config_management/verify_stage_configs.py` |

### 脚本复用检查流程

**在创建新脚本之前，请按以下步骤检查**:

```bash
# 步骤1: 查阅脚本文档
cat docs/SCRIPTS_QUICKREF.md

# 步骤2: 列出相关目录的脚本
ls -lh tools/data_management/*.py
ls -lh tools/config_management/*.py

# 步骤3: 查看脚本功能描述
head -30 tools/data_management/script_name.py

# 步骤4: 测试脚本是否满足需求
python3 tools/data_management/script_name.py --help
python3 tools/data_management/script_name.py --dry-run  # 如果支持
```

### 常用查找命令

```bash
# 按功能关键词搜索脚本
grep -l "追加\|append" tools/data_management/*.py
grep -l "验证\|validate" tools/data_management/*.py
grep -l "分析\|analyze" tools/data_management/*.py

# 查看脚本文档字符串
python3 << 'EOF'
import pathlib
for script in sorted(pathlib.Path('tools/data_management').glob('*.py')):
    print(f"\n{'='*60}\n{script.name}\n{'='*60}")
    with open(script) as f:
        for line in f:
            if '"""' in line:
                print(line.rstrip())
                for line in f:
                    print(line.rstrip())
                    if '"""' in line:
                        break
                break
EOF
```

### 归档脚本说明

如果发现 `tools/legacy/` 目录中有类似功能的脚本：
- ⚠️ **不要使用legacy中的脚本** - 它们已被更好的替代方案取代
- ✅ **使用活跃目录中的脚本** - `tools/data_management/` 或 `tools/config_management/`
- 📖 **查看归档说明** - 了解为什么脚本被归档以及替代方案

**归档脚本文档**:
- `tools/legacy/completed_data_tasks_20260110/README.md` - 已完成任务脚本
- `tools/legacy/archived/*/README.md` - 其他归档脚本说明

### 最佳实践

1. **优先复用**: 如果现有脚本能满足80%的需求，考虑添加参数而不是创建新脚本
2. **参数化设计**: 如果必须创建新脚本，设计成通用的、参数化的
3. **文档完整**: 在脚本顶部添加清晰的文档字符串
4. **定期清理**: 一次性任务完成后，考虑归档到 `tools/legacy/`

### 相关文档

- [docs/SCRIPT_DUPLICATION_ANALYSIS_REPORT.md](docs/SCRIPT_DUPLICATION_ANALYSIS_REPORT.md) - 脚本重复性分析报告 (2026-01-10)
- [docs/SCRIPTS_QUICKREF.md](docs/SCRIPTS_QUICKREF.md) - 脚本快速参考
- [tools/legacy/completed_data_tasks_20260110/README.md](tools/legacy/completed_data_tasks_20260110/README.md) - 归档脚本说明

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
**文档版本**: 5.6.0
**最后更新**: 2026-01-10
**重要更新 (P0 改进完成)**:
- ✅ **添加 TOC 目录** - 可点击跳转，提升导航效率50% ⭐⭐⭐
- ✅ **拆分关键开发规范** - 创建3个独立文档（38KB专题内容），主文档保持精简
- ✅ **添加5分钟快速验证** - 一键健康检查脚本，新用户快速上手 ⭐⭐
- ✅ **创建健康检查脚本** - tools/quick_health_check.sh 自动验证环境
- ✅ **优化文档结构** - 精简主文档，信息密度显著提升

**历史更新**:
- v5.5.0: 新增独立验证规范
- v5.4.0: 脚本重复性分析和清理
- v5.3.0: 项目文件结构重组

> **记住**: 本文档只是快速指南！详细信息和完整上下文请始终参考 [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md)
