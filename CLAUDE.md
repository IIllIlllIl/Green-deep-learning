# Claude 助手快速指南 - Energy DL Project

**版本**: v6.1.0 | **状态**: ✅ 因果分析完成+归档整理 | **更新**: 2026-02-10

> **5分钟快速指南** · 完整参考见 [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md) (1089行)

---

## 📊 项目状态

**当前阶段**: ✅ 因果分析阶段完成 + 文件归档整理

**核心成果**:
- 📦 **836个实验** (95.1%完整性)
- 🔬 **6组DiBS因果图** (全局标准化，13000步训练，强边比例2.0%-17.8%)
- ⚖️ **61个显著权衡** (100%统计显著性，7个能耗vs性能)
- 📊 **完整工作流文档** (DiBS训练→ATE计算→权衡检测)

**最新进展**: DiBS后续工作流完成 + 文件归档整理
- [DiBS工作流文档](../docs/technical_reference/DIBS_END_TO_END_WORKFLOW_20260210.md)
- [归档操作日志](analysis/archive/archive_log_20260210.md)
- [归档执行总结](analysis/ARCHIVE_EXECUTION_SUMMARY_20260210.md)

---

## ⚡ 快速开始

### 环境验证

```bash
# 1. Python 环境 (3.8+)
python3 --version

# 2. GPU 驱动
nvidia-smi

# 3. 激活因果推断环境
conda activate causal-research
```

⚠️ **注意**: 因果推断分析必须使用 `causal-research` 环境（base环境无DiBS库）

### 快速健康检查

```bash
# 验证数据完整性
python3 tools/data_management/validate_raw_data.py

# 查看分析进度
cat analysis/docs/technical_reference/GLOBAL_STANDARDIZATION_FIX_PROGRESS.md
```

---

## 🔬 分析模块

`analysis/` 是独立的因果推断分析模块，用于研究超参数对能耗和性能的因果影响。

**技术栈**: DiBS（因果图学习）+ DML（因果推断）+ 权衡检测

**最新成果** (v2.0):
- 全局标准化数据: 818样本，50列，35列统一标准化
- DiBS因果图: 6组完整（49×49邻接矩阵，13000步训练）
- ATE计算: 6组成功，使用EconML的DML方法
- 权衡检测: 61个显著权衡，7个能耗vs性能

**文档索引**:
- [模块总览](analysis/README.md)
- [文档索引](analysis/docs/INDEX.md)
- [DiBS工作流](../docs/technical_reference/DIBS_END_TO_END_WORKFLOW_20260210.md)

**关键文件**:
- 数据: `analysis/data/energy_research/6groups_global_std/`
- 结果: `analysis/results/energy_research/tradeoff_detection_global_std/`
- 图表: `analysis/results/energy_research/tradeoff_detection_global_std/figures/`
- 归档: `analysis/archive/archive_20260210/`

---

## 🗂️ 文件规则

### 文件摆放索引

| 文件类型 | 位置 | 命名规范 | 详细说明 |
|---------|------|---------|---------|
| 📄 项目文档 | `docs/` | 小写_下划线.md | [项目文档管理](#文档导航) |
| 🔧 数据处理脚本 | `tools/data_management/` | 动词_名词.py | |
| ⚙️ 配置管理脚本 | `tools/config_management/` | 动词_名词.py | |
| 📊 分析脚本 | `analysis/scripts/` | 动词_名词.py | |
| 🧪 测试文件 | `analysis/tests/unit/` | test_名词.py | |
| 📋 实验报告 | `docs/reports/` | 主题_日期.md | |
| 🗃️ 归档文档 | `docs/archived/` | 原名_归档日期.md | |

**核心目录**:
- `data/` - 数据文件（raw_data.csv, data.csv, backups/）
- `tools/` - 数据处理工具（data_management/, config_management/）
- `analysis/` - 因果推断分析模块
- `docs/` - 项目文档

**决策原则**:
- ✅ 优先编辑现有文件，而非创建新文件
- ✅ 创建前先搜索是否已存在类似工具
- ⚠️ 不确定时询问用户

---

## 📚 文档导航

### 核心文档

| 文档 | 路径 | 行数 | 用途 |
|------|------|------|------|
| 完整参考 | [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md) | 1089 | 所有详细信息和规范 |
| 数据使用指南 | [docs/DATA_USAGE_GUIDE.md](docs/DATA_USAGE_GUIDE.md) | 430 | 数据处理必读 |
| 开发工作流 | [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) | - | 任务执行规范 |
| 文档总索引 | [docs/INDEX.md](docs/INDEX.md) | 2600+ | 完整文档索引 |

### 分析文档

| 文档 | 路径 | 用途 |
|------|------|------|
| 模块总览 | [analysis/README.md](analysis/README.md) | 分析模块介绍 |
| 文档索引 | [analysis/docs/INDEX.md](analysis/docs/INDEX.md) | 分析文档总索引 |
| 进度跟踪 | [analysis/docs/technical_reference/GLOBAL_STANDARDIZATION_FIX_PROGRESS.md](analysis/docs/technical_reference/GLOBAL_STANDARDIZATION_FIX_PROGRESS.md) | 最新进度和验收 |

### 命令速查

| 文档 | 路径 | 用途 |
|------|------|------|
| 命令参考 | [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md#常用命令速查) | 常用命令查询 |

### 数据文件

| 文档 | 路径 | 用途 |
|------|------|------|
| 数据指南 | [docs/DATA_USAGE_GUIDE.md](docs/DATA_USAGE_GUIDE.md) | 数据使用必读 |

---

## 🚨 常见问题 (Top 5)

**Q1: GPU驱动失败？**
→ 检查 `nvidia-smi`，安装/更新NVIDIA驱动
→ 详细: [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md#常见问题)

**Q2: 因果推断分析失败？**
→ 确认使用 `causal-research` 环境（base环境无DiBS库）
→ 验证: `conda activate causal-research && python -c "import dibs; print('OK')"`

**Q3: 能耗数据缺失？**
→ 已修复至95.1%完整性
→ 验证: `python3 tools/data_management/validate_raw_data.py`

**Q4: 文件应该放哪里？**
→ 参考"文件规则"章节
→ 详细: [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md)

**Q5: 如何开始分析？**
→ 1. 激活环境: `conda activate causal-research`
→ 2. 查看进度: `cat analysis/docs/.../GLOBAL_STANDARDIZATION_FIX_PROGRESS.md`
→ 3. 阅读报告: `docs/reports/`

更多问题: [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md#常见问题)

---

## 📊 数据文件说明

### 主数据文件

| 文件 | 位置 | 规模 | 完整性 |
|------|------|------|--------|
| raw_data.csv | `data/raw_data.csv` | 970行×87列 | 85.4%能耗数据 |
| data.csv | `data/data.csv` | 970行×56列 | 精简版，统一字段 |

**详细说明**: [docs/DATA_USAGE_GUIDE.md](docs/DATA_USAGE_GUIDE.md)

### 分析数据文件

| 数据 | 位置 | 说明 |
|------|------|------|
| 全局标准化数据 | `analysis/data/energy_research/6groups_global_std/` | 818样本×50列 |
| DiBS因果图 | `analysis/results/energy_research/data/global_std/` | 6组因果图结果 |
| 权衡检测结果 | `analysis/results/energy_research/tradeoff_detection_global_std/` | 61个权衡+图表 |
| 归档文件 | `analysis/archive/archive_20260210/` | 已归档的旧版本 |

---

## 🗃️ 归档文件说明

**最近归档**: 2026-02-10（黑名单策略）

已归档的旧版本文件（保留30天）：
- 数据: 6groups_final, 6groups_interaction, 6groups_dibs_ready_v1_backup
- 结果: archived_data, interaction_tradeoff_verification, tradeoff_detection_interaction_based
- 脚本: run_algorithm1_tradeoff_detection.py（旧版）

**归档文档**:
- [归档日志](analysis/archive/archive_log_20260210.md) - 完整操作记录+回滚方案
- [执行总结](analysis/ARCHIVE_EXECUTION_SUMMARY_20260210.md) - 归档统计

**回滚方法**: 参见归档日志中的manifest.txt和恢复脚本

---

## 📞 获取帮助

1. **快速问题**: 本文档"常见问题"章节
2. **详细信息**: [docs/CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md) (1089行完整参考)
3. **分析模块**: [analysis/docs/INDEX.md](analysis/docs/INDEX.md) (784行分析文档索引)
4. **完整索引**: [docs/INDEX.md](docs/INDEX.md) (2600+行总索引)

---

**维护者**: Green | **版本**: v6.1.0 | **更新**: 2026-02-10

**版本历史**:
- v6.1.0 (2026-02-10): DiBS工作流完成，文件归档整理
- v6.0.0 (2026-02-01): 重构为精简版，因果分析完成
- v5.9.0 (2026-01-25): 添加文件创建规则
- v5.8.0 (2026-01-25): 优化文档导航
