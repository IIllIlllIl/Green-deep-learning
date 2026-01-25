# Claude 助手快速指南 - Energy DL Project

**版本**: v5.8.0
**最后更新**: 2026-01-25
**状态**: ⏳ 进行中 - 能耗数据因果分析阶段

> **提示**: 本文档为5分钟快速指南。完整详细文档请查看 [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md)

---

## 📋 5分钟快速开始

### 项目概述

本项目研究**深度学习训练超参数对能耗和性能的影响**，通过自动化变异超参数、监控能耗、收集性能指标来支持大规模实验研究。

**核心成果**:
- ✅ **836个实验** (795个有效能耗数据，**95.1%完整性**)
- ✅ 11个模型在并行/非并行模式下全覆盖
- ✅ 90/90参数-模式组合100%达标
- ✅ **数据完整性修复完成** (69.7% → 95.1%)
- ⏳ 能耗数据因果分析进行中

### 当前主要任务

**任务1**: 实验扩展计划 (2026-01-05) - 多参数变异实验设计
**任务2**: 能耗数据回归分析 (2026-01-04) - 超参数对能耗的影���

详细方案: [docs/EXPERIMENT_EXPANSION_PLAN_20260105.md](docs/EXPERIMENT_EXPANSION_PLAN_20260105.md)

---

## ⚡ 5分钟快速验证

### 一键健康检查

```bash
# 执行快速健康检查脚本
bash tools/quick_health_check.sh
```

### 手动快速检查

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

## 🗂️ 文件创建决策规则 ⭐⭐⭐

**重要**: 当你需要创建新文件时，请按以下规则决策：

| 文件类型 | 放置目录 | 命名规范 | 示例 |
|---------|---------|---------|------|
| 📄 **项目文档** | `docs/` | 小写_下划线.md | `EXPERIMENT_PLAN.md` |
| 🔧 **数据处理脚本** | `tools/data_management/` | 动词_名词.py | `append_session_to_raw_data.py` |
| ⚙️ **配置管理脚本** | `tools/config_management/` | 动词_名词.py | `validate_config.py` |
| 🧪 **测试文件** | `tests/unit/` 或 `tests/integration/` | `test_名词.py` | `test_data_validation.py` |
| 📊 **分析脚本** | `analysis/scripts/` | 动词_名词.py | `run_causal_inference.py` |
| 🗃️ **过时文档** | `docs/archived/` 或 `tools/legacy/` | 原名_归档日期.md | `OLD_PLAN_20260101.md` |
| 📋 **实验报告** | `docs/reports/` | 主题_日期.md | `DATA_REPAIR_REPORT_20260104.md` |

**决策原则**:
- ✅ **优先编辑现有文件**，而非创建新文件
- ✅ **创建前先搜索**是否已存在类似工具（`ls tools/`、`ls docs/`）
- ⚠️ **不确定时询问用户**："应该创建新文件还是编辑现有文件？"

**核心目录用途**:
- `data/` - 数据文件（raw_data.csv、data.csv、backups/）
- `tools/` - 可执行脚本（data_management/、config_management/）
- `docs/` - 项目文档（非代码类文档）
- `tests/` - 测试脚本
- `analysis/` - 因果推断分析模块

## 📁 项目结构快览

```
energy_dl/nightly/
├── CLAUDE.md              # 本文件 - 快速指南 ⭐
├── README.md              # 项目总览
├── mutation.py            # 训练入口脚本
│
├── data/                  # ⭐ 核心数据
│   ├── raw_data.csv       # 主数据文件（87列，836行，95.1%完整性）⭐⭐⭐
│   ├── data.csv           # 精简数据文件（56列）⭐⭐
│   └── backups/           # 数据备份
│
├── tools/                 # ⭐ 数据处理工具
│   ├── data_management/   # 数据管理工具（11个活跃脚本）⭐
│   ├── config_management/ # 配置管理工具（4个脚本）
│   └── legacy/            # 历史脚本归档
│
├── docs/                  # 项目文档
│   ├── CLAUDE_FULL_REFERENCE.md    # 完整参考文档 ⭐⭐⭐
│   ├── reports/           # 实验结果报告
│   └── archived/          # 过时文档归档
│
├── analysis/              # 因果推断分析模块 🔬
│   ├── docs/              # 分析模块文档总索引 ⭐
│   ├── scripts/           # 分析脚本
│   └── utils/             # 核心模块
│
├── archives/              # ⭐ 历史数据归档
├── repos/                 # 训练模型仓库（11个模型）
├── mutation/              # 训练核心代码
├── tests/                 # 测试文件
├── settings/              # 实验配置
└── environment/           # 环境配置
```

---

## 🛠️ 常用命令速查

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

## 🚨 紧急问题快速解决

### 1. 实验ID唯一性问题 ⭐⭐⭐

**问题**: 不同批次的实验会产生相同的experiment_id

**正确做法**: 使用 **experiment_id + timestamp** 作为唯一标识符

```python
# ✅ 正确 - 使用复合键
composite_key = f"{row['experiment_id']}|{row['timestamp']}"
if composite_key in existing_keys:
    ...
```

### 2. 数据文件选择错误

**❌ 错误**: 使用 raw_data.csv 而不理解 fg_ 前缀
**✅ 正确**: 使用 data.csv（已自动统一并行/非并行字段）

```python
# ✅ 推荐：使用 data.csv（简单易用）
import pandas as pd
df = pd.read_csv('data/data.csv')
# 直接使用字段，无需考虑 fg_ 前缀
learning_rate = df['learning_rate']
```

### 3. CSV数据完整性

- `raw_data.csv` 为主数据文件，使用复合键去重
- 追加数据时使用 `tools/data_management/append_session_to_raw_data.py`
- 每次修改前务必备份

### 4. 能耗数据缺失

**症状**: energy_* 列为空
**原因**: perf权限问题或nvidia-smi不可用
**解决**: 使用sudo运行，检查GPU驱动

### 5. 配置执行失败

**症状**: JSON配置文件解析错误或KeyError
**解决**:
- 使用 `python -m json.tool` 验证JSON格式
- 确保使用 `"repo"` 而非 `"repository"`
- 并行模式使用 `foreground/background` 结构
- 参考 `settings/stage2_optimized_*.json` 作为模板

---

## 📚 核心文档索引

**完整参考**: [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md) ⭐⭐⭐

### 三大必读文档

| 文档 | 用途 |
|------|------|
| 📘 [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md) | **完整项目参考** - 所有详细信息和规范 |
| 📗 [docs/DATA_USAGE_GUIDE.md](docs/DATA_USAGE_GUIDE.md) | **数据使用指南** - 处理数据前必读 |
| 📙 [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) | **开发工作流** - 任务执行规范（测试→Dry Run→执行） |

**更多文档**: 完整文档索引见 [docs/CLAUDE_FULL_REFERENCE.md § 文档导航](docs/CLAUDE_FULL_REFERENCE.md#📚-文档导航)

---

## 📊 数据文件说明

### 数据可用性概览

**最新分析结果**:
- **总记录数**: 970条（含header，实际969条数据）
- **✅ 完全可用记录**: **577条 (59.5%)** - 训练成功、有能耗数据、有性能指标
- **能耗数据**: **828/969 (85.4%)** 有能耗数据

**详细分析**: [docs/DATA_USABILITY_SUMMARY_20260113.md](docs/DATA_USABILITY_SUMMARY_20260113.md)

### raw_data.csv (主数据文件) ⭐⭐⭐

- **位置**: `data/raw_data.csv`
- **规模**: 970行（含header），87列
- **数据可用性**: 577/969 (59.5%) 完全可用记录
- **能耗数据**: 828/969 (85.4%) 有能耗数据
- **验证**: `tools/data_management/validate_raw_data.py`

### data.csv (精简数据文件) ⭐⭐

- **位置**: `data/data.csv`
- **规模**: 970行（含header），56列
- **特性**: 统一并行/非并行字段，添加 is_parallel 列
- **生成**: `tools/data_management/create_unified_data_csv.py`

### 数据使用建议

根据研究目的选择合适的数据集：

| 使用场景 | 推荐数据 | 记录数 | 数据质量 |
|---------|---------|--------|---------|
| **高质量分析（推荐）** | 8个100%可用模型 | 487条 | ⭐⭐⭐ 优秀 |
| **平衡分析** | +MRT-OAST可用部分 | 552条 | ⭐⭐ 良好 |
| **最大化分析** | 所有可用记录 | 577条 | ⭐⚠️ 混合 |
| **能耗专项分析** | 有能耗数据的记录 | 828条 | ⭐ 仅能耗可用 |

---

## 🔬 分析模块 (analysis/)

### 概述

`analysis/` 是独立的因果推断分析模块，用于研究超参数对能耗和性能的因果影响。

**核心技术**: DiBS（因果图学习）+ DML（因果推断） + 回归分析

**环境配置 ⭐ 重要**:
```bash
# 激活causal-research环境（已安装DiBS）
conda activate causal-research
```

⚠️ **注意**: base环境没有安装DiBS，会导致分析失败！

**主要文档**:
- [analysis/README.md](analysis/README.md) - 模块总体介绍
- [analysis/docs/INDEX.md](analysis/docs/INDEX.md) - 文档总索引
- [analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md](analysis/docs/QUESTION1_REGRESSION_ANALYSIS_PLAN.md) - 问题1方案

---

## 📞 获取帮助

### 文档导航

1. **快速开始**: 本文档（CLAUDE.md）
2. **详细参考**: [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md) ⭐⭐⭐
3. **分析模块**: [analysis/docs/INDEX.md](analysis/docs/INDEX.md)
4. **项目进度**: [docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md](docs/results_reports/PROJECT_PROGRESS_COMPLETE_SUMMARY.md)

### 常见问题

如果遇到问题，请依次检查：
1. 本文档的"紧急问题快速解决"章节
2. [docs/CLAUDE_FULL_REFERENCE.md § 常见问题](docs/CLAUDE_FULL_REFERENCE.md#📞-常见问题)
3. 相关的专题文档（如JSON配置规范、数据格式设计等）

---

## 📝 版本管理规则

- **当前版本**: v5.9.0
- **最后更新**: 2026-01-25
- **更新规则**:
  1. **版本同步**: CLAUDE.md 和 CLAUDE_FULL_REFERENCE.md 保持版本号同步
  2. **更新流程**: 重大更新必须同时更新两个文件的版本号
  3. **记录分工**:
     - CLAUDE.md: 只记录重大里程碑（如"数据完整性修复完成"）
     - CLAUDE_FULL_REFERENCE.md: 记录详细更新历史
  4. **版本号格式**: v{主版本}.{次版本}.{修订号}（如 v5.9.0）

**维护者**: Green
**文档版本**: 5.9.0
**最后更新**: 2026-01-25
**重要更新 (v5.9.0)**:
- ✅ **新增** - 文件创建决策规则表（🗂️ 章节）
- ✅ **新增** - 版本管理规则说明
- ✅ **优化** - 精简文档导航，保留核心链接

> **记住**: 本文档只是快速指南！详细信息和完整上下文请始终参考 [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md)
