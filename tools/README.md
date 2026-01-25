# Tools 目录说明

本目录包含项目的核心工具脚本，按功能分类组织。

**最后更新**: 2026-01-23
**状态**: ✅ 已精简优化，保留核心脚本

---

## 📁 目录结构

```
tools/
├── data_management/     # 数据管理工具（8个活跃脚本）⭐
├── config_management/   # 配置管理工具（4个脚本）
└── README.md           # 本文件
```

---

## 🔧 数据管理工具 (data_management/)

**活跃脚本**: 8个

### 数据追加与验证 (核心高频脚本) ⭐⭐⭐

| 脚本 | 功能 | 使用频率 |
|------|------|----------|
| `append_session_to_raw_data.py` ⭐⭐⭐ | 从session目录追加实验数据到raw_data.csv | 每批新实验后必用 |
| `validate_raw_data.py` ⭐⭐⭐ | 验证raw_data.csv数据完整性 | 每批新实验后必用 |
| `check_latest_results.py` ⭐⭐ | 检查最新实验是否已加入数据文件 | 实验后检查 |

### 数据转换与验证

| 脚本 | 功能 | 使用频率 |
|------|------|----------|
| `create_unified_data_csv.py` ⭐⭐ | 创建统一并行数据版本的data.csv | 每批实验后 |
| `compare_data_vs_raw_data.py` ⭐⭐ | 比较data.csv和raw_data.csv一致性 | 数据变动后 |
| `validate_merged_metrics.py` ⭐ | 验证合并后的性能指标数据质量 | 生成data.csv后 |

### 数据分析

| 脚本 | 功能 | 使用频率 |
|------|------|----------|
| `analyze_experiment_status.py` ⭐⭐⭐ | 分析实验状况统计（模型、参数覆盖等） | 日常检查 |
| `independent_quality_assessment.py` ⭐⭐ | 独立数据质量评估 | 论文/报告前 |

### 使用示例

```bash
# === 高频命令（每批实验后） ===
# 1. 追加新实验数据
python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS

# 2. 验证数据完整性
python3 tools/data_management/validate_raw_data.py

# 3. 分析实验状况
python3 tools/data_management/analyze_experiment_status.py

# === 数据转换 ===
# 生成统一格式数据
python3 tools/data_management/create_unified_data_csv.py

# === 数据验证 ===
# 比较两个数据文件
python3 tools/data_management/compare_data_vs_raw_data.py
```

---

## ⚙️ 配置管理工具 (config_management/)

**活跃脚本**: 4个

| 脚本 | 功能 | 使用频率 |
|------|------|----------|
| `generate_mutation_config.py` ⭐⭐ | 生成变异配置文件（11个模型，每个超参数变异3次） | 新实验批次 |
| `validate_models_config.py` ⭐ | 验证models_config.json完整性和有效性 | 添加新模型后 |
| `validate_mutation_config.py` ⭐⭐ | 验证变异配置JSON格式和正确性 | 生成配置后 |
| `verify_stage_configs.py` ⭐ | 检查stage配置文件的runs_per_config定义 | 创建新Stage后 |

### 使用示例

```bash
# === 配置生成与验证 ===
# 生成变异配置
python3 tools/config_management/generate_mutation_config.py

# 验证变异配置
python3 tools/config_management/validate_mutation_config.py settings/stage2_*.json

# 验证模型配置
python3 tools/config_management/validate_models_config.py

# 检查stage配置
python3 tools/config_management/verify_stage_configs.py
```

---

## 📋 脚本使用频率预测

基于项目当前状态（2026-01-23）：

### 高频使用 ⭐⭐⭐ (每周多次)

| 脚本 | 原因 |
|------|------|
| `append_session_to_raw_data.py` | 471个新实验计划，需要频繁追加 |
| `validate_raw_data.py` | 追加数据后必须验证完整性 |
| `analyze_experiment_status.py` | 日常检查实验覆盖情况 |
| `validate_mutation_config.py` | 生成新实验配置后验证 |

### 中频使用 ⭐⭐ (每月1-2次)

| 脚本 | 原因 |
|------|------|
| `create_unified_data_csv.py` | 每批实验后生成分析用数据 |
| `compare_data_vs_raw_data.py` | 确保两文件一致性 |
| `generate_mutation_config.py` | 新实验批次生成配置 |
| `check_latest_results.py` | 实验后快速检查 |

### 低频使用 ⭐ (季度或更少)

| 脚本 | 原因 |
|------|------|
| `validate_merged_metrics.py` | 生成data.csv后验证 |
| `validate_models_config.py` | 11个模型已固定，很少变更 |
| `verify_stage_configs.py` | Stage配置已稳定 |
| `independent_quality_assessment.py` | 论文/报告前独立验证 |

---

## 🔍 如何查找脚本

### 按功能查找

```bash
# 列出所有数据管理脚本
ls -lh tools/data_management/*.py

# 列出所有配置管理脚本
ls -lh tools/config_management/*.py

# 按关键词搜索
grep -l "追加\|append" tools/data_management/*.py
grep -l "验证\|validate" tools/data_management/*.py
grep -l "分析\|analyze" tools/data_management/*.py
```

### 查看脚本文档

```bash
# 查看脚本顶部文档
head -30 tools/data_management/script_name.py

# 查看脚本帮助（如果支持）
python3 tools/data_management/script_name.py --help
```

---

## 📋 脚本开发最佳实践

### 创建新脚本前的检查清单

- [ ] 查阅本README确认无类似脚本
- [ ] 搜索现有脚本目录
- [ ] 测试现有脚本是否能满足需求（80%即可考虑复用）
- [ ] 确认确实需要新脚本后再开发

### 新脚本开发规范

1. **添加完整文档字符串** - 说明功能、参数、使用示例
2. **设计为通用工具** - 使用命令行参数而非硬编码值
3. **支持Dry Run模式** - 对数据修改操作添加`--dry-run`参数
4. **包含测试示例** - 在文档中添加示例用法
5. **一次性任务归档** - 任务完成后移到其他目录或删除

---

## 📊 数据流向

```
实验运行 (mutation.py)
    ↓
results/run_YYYYMMDD_HHMMSS/
    ↓
append_session_to_raw_data.py ⭐ 核心输入脚本
    ↓
data/raw_data.csv (87列，主数据文件)
    ↓
┌───────────────────┴───────────────────┐
↓                                       ↓
各类分析/验证脚本                  create_unified_data_csv.py
(analyze_*, validate_*)                     ↓
                                    data/data.csv (56列，统一格式)
```

---

## 📚 相关文档

- [CLAUDE.md § 脚本复用检查指南](../CLAUDE.md#-脚本复用检查指南-) - 使用指南
- [docs/SCRIPTS_QUICKREF.md](../docs/SCRIPTS_QUICKREF.md) - 脚本快速参考
- [docs/JSON_CONFIG_WRITING_STANDARDS.md](../docs/JSON_CONFIG_WRITING_STANDARDS.md) - JSON配置规范
- [docs/DATA_MASTER_GUIDE.md](../docs/DATA_MASTER_GUIDE.md) - 数据使用主指南

---

## 📊 统计信息

**最后统计**: 2026-01-23

- **活跃脚本总数**: 12个
  - 数据管理: 8个（核心高频）
  - 配置管理: 4个
- **项目结构**: 精简优化，保留核心功能
- **主要用途**: 实验数据管理、配置生成、数据验证

---

## ⚡ 快速命令参考

```bash
# === 核心工作流 ===
# 1. 生成实验配置
python3 tools/config_management/generate_mutation_config.py

# 2. 验证配置
python3 tools/config_management/validate_mutation_config.py settings/new_config.json

# 3. 运行实验（使用mutation.py）
sudo python3 mutation.py -ec settings/new_config.json

# 4. 追加实验数据
python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS

# 5. 验证数据完整性
python3 tools/data_management/validate_raw_data.py

# 6. 查看实验状况
python3 tools/data_management/analyze_experiment_status.py
```

---

**维护者**: Green
**最后更新**: 2026-01-23
**版本**: v6.0 (精简优化版)
