# 脚本重复性分析报告

**生成日期**: 2026-01-10
**分析范围**: tools/ 和 analysis/scripts/ 目录下的所有Python脚本

---

## 执行摘要

经过详细分析，在 `tools/data_management/` 目录中发现 **4个一次性任务脚本**，它们的功能已被通用脚本取代或仅用于历史任务，建议归档到 `tools/legacy/`。其余脚本功能互补，无实质性重复。

### 关键发现

- ✅ **通用工具脚本**: 11个（应保留在active目录）
- ⚠️ **一次性任务脚本**: 4个（建议归档到legacy）
- ✅ **配置管理脚本**: 4个（功能独立，无重复）
- 📊 **分析脚本**: 35+个（需要进一步审查，但不在本次整合范围）

---

## 详细分析

### 1. tools/data_management/ (15个脚本)

#### ✅ 通用工具脚本（应保留，11个）

这些脚本功能互补，构成完整的数据管理工作流：

| 脚本名 | 大小 | 主要功能 | 保留原因 |
|--------|------|----------|----------|
| **append_session_to_raw_data.py** | 21K | 通用数据追加工具⭐ | 主要的数据追加工具，功能完整 |
| analyze_experiment_status.py | 8.9K | 实验状况统计 | 核心分析工具 |
| analyze_missing_energy_data.py | 12K | 缺失能耗数据分析 | 数据质量检查 |
| verify_recoverable_data.py | 14K | 验证数据可恢复性 | 数据修复前置工具 |
| repair_missing_energy_data.py | 8.3K | 修复缺失能耗数据 | 数据修复工具 |
| validate_raw_data.py | 7.8K | 验证数据完整性 | 数据质量验证 |
| check_latest_results.py | 9.2K | 检查最新结果 | 实验同步检查 |
| compare_data_vs_raw_data.py | 17K | 比较CSV一致性 | 数据一致性验证 |
| create_unified_data_csv.py | 11K | 创建统一data.csv | 数据格式统一 |
| check_attribute_mapping.py | 8.1K | 检查属性映射 | 模式验证工具 |
| validate_merged_metrics.py | 8.4K | 验证合并指标质量 | 数据合并验证 |

**数据管理工作流**:
```
实验运行 → append_session_to_raw_data.py (追加数据)
         ↓
      validate_raw_data.py (验证完整性)
         ↓
      analyze_missing_energy_data.py (分析缺失)
         ↓
      verify_recoverable_data.py (检查可恢复性)
         ↓
      repair_missing_energy_data.py (修复数据)
         ↓
      create_unified_data_csv.py (生成统一格式)
```

#### ⚠️ 一次性任务脚本（建议归档，4个）

这些脚本用于特定的历史任务，功能已被通用脚本取代：

| 脚本名 | 大小 | 功能 | 归档原因 |
|--------|------|------|----------|
| **add_new_experiments_to_raw_data.py** | 6.0K | 从特定会话提取4个实验 | 特定任务，已被 `append_session_to_raw_data.py` 取代 |
| **merge_csv_to_raw_data.py** | 8.7K | 合并 summary_old/new → raw_data | 一次性合并任务，已完成 |
| **update_raw_data_with_reextracted.py** | 11K | 重新提取性能指标 | 一次性修复任务，可能已完成 |
| **merge_performance_metrics.py** | 4.3K | 合并性能指标列 | 一次性任务，与下一个配套 |

**具体说明**:

1. **add_new_experiments_to_raw_data.py**
   - 用途：从 `run_20251212_224937` 提取4个Phase 2诊断实验
   - 问题：硬编码特定session路径，不通用
   - 替代：使用 `append_session_to_raw_data.py` + 命令行参数

2. **merge_csv_to_raw_data.py**
   - 用途：合并 summary_old.csv (93列) 和 summary_new.csv (80列)
   - 问题：特定文件名，一次性任务
   - 状态：可能已完成历史数据合并

3. **update_raw_data_with_reextracted.py**
   - 用途：使用更新后的正则表达式重新提取性能指标
   - 问题：一次性修复任务
   - 状态：需确认是否已完成

4. **merge_performance_metrics.py**
   - 用途：将特定模型的指标重命名（MRT-OAST accuracy→test_accuracy）
   - 问题：一次性列合并任务
   - 状态：应已完成

---

### 2. tools/config_management/ (4个脚本)

✅ **无重复，功能独立**

| 脚本名 | 主要功能 |
|--------|----------|
| generate_mutation_config.py | 生成变异配置 |
| validate_models_config.py | 验证models_config.json完整性 |
| validate_mutation_config.py | 验证变异配置JSON格式 |
| verify_stage_configs.py | 检查stage配置文件 |

这4个脚本各司其职，分别负责配置的生成、验证和检查，无功能重复。

---

### 3. analysis/scripts/ (35+个脚本)

📊 **分析脚本较多，建议单独审查**

此目录包含大量因果分析、数据处理和探索性分析脚本。初步观察：

- 有些脚本可能是探索性分析（如参数扫描、方法比较）
- 有些脚本用于数据预处理和质量检查
- 有些脚本用于特定研究问题的分析

**建议**：
- 分析脚本通常保留历史记录有价值（可复现研究过程）
- 可以分类整理到 `analysis/scripts/` 的子目录：
  - `exploratory/` - 探索性分析
  - `preprocessing/` - 数据预处理
  - `final/` - 最终分析脚本

---

## 整合方案

### 方案A：归档到 tools/legacy/ （推荐）⭐

将4个一次性任务脚本移动到 `tools/legacy/completed_data_tasks/`：

```bash
mkdir -p tools/legacy/completed_data_tasks

mv tools/data_management/add_new_experiments_to_raw_data.py \
   tools/data_management/merge_csv_to_raw_data.py \
   tools/data_management/update_raw_data_with_reextracted.py \
   tools/data_management/merge_performance_metrics.py \
   tools/legacy/completed_data_tasks/
```

**优点**:
- 保留历史记录，可追溯
- 清理主工作目录
- 明确区分活跃脚本和历史脚本

**注意事项**:
- 在归档前，确认这些脚本的任务确实已完成
- 在 `tools/legacy/completed_data_tasks/README.md` 中记录归档原因和时间

### 方案B：合并功能（不推荐）

虽然理论上可以将 `add_new_experiments_to_raw_data.py` 的功能合并到通用脚本，但：
- 通用脚本 `append_session_to_raw_data.py` 已经实现了所有需要的功能
- 特定脚本仅用于历史任务，合并无实际价值
- 保留历史脚本更有利于问题追溯

**结论**: 归档优于合并

---

## 文档更新建议

### 1. 更新 CLAUDE.md 或 SCRIPTS_QUICKREF.md

添加"脚本复用检查指南"章节：

```markdown
## 🔍 脚本复用检查指南

**在创建新脚本前，请先检查是否已有可用脚本**

### 数据管理常用脚本

| 任务 | 使用脚本 | 命令示例 |
|------|----------|----------|
| 追加新实验到raw_data.csv | `append_session_to_raw_data.py` | `python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS` |
| 验证数据完整性 | `validate_raw_data.py` | `python3 tools/data_management/validate_raw_data.py` |
| 分析实验状况 | `analyze_experiment_status.py` | `python3 tools/data_management/analyze_experiment_status.py` |
| 分析缺失能耗 | `analyze_missing_energy_data.py` | `python3 tools/data_management/analyze_missing_energy_data.py` |
| 修复缺失能耗 | `repair_missing_energy_data.py` | `python3 tools/data_management/repair_missing_energy_data.py` |

### 检查流程

1. **查阅文档**: 先查看 `docs/SCRIPTS_QUICKREF.md`
2. **搜索功能**: 使用 `grep` 或 `ls` 查找相关脚本
3. **测试现有脚本**: 使用 `--dry-run` 或 `--help` 参数测试
4. **确认无法复用**: 仅在确认现有脚本无法满足需求时创建新脚本

### 命令速查

```bash
# 查找脚本
ls tools/data_management/*.py
ls tools/config_management/*.py

# 查看脚本功能（从文档字符串）
head -30 tools/data_management/script_name.py | grep -A 10 '"""'

# 查看脚本帮助
python3 tools/data_management/script_name.py --help
```
```

### 2. 创建脚本索引文档

建议在 `tools/README.md` 中维护活跃脚本的完整索引。

---

## 执行计划

### 阶段1: 验证和归档（立即执行）

1. ✅ 确认4个脚本的任务已完成
2. ✅ 创建归档目录
3. ✅ 移动脚本到legacy
4. ✅ 创建归档说明文档

### 阶段2: 文档更新（随后执行）

1. ✅ 更新 CLAUDE.md 添加脚本复用指南
2. ✅ 更新 SCRIPTS_QUICKREF.md
3. ✅ 创建 tools/README.md 索引

### 阶段3: 分析脚本审查（可选）

1. 审查 analysis/scripts/ 目录
2. 分类整理到子目录
3. 更新 analysis/docs/INDEX.md

---

## 附录：完整脚本清单

### tools/data_management/ (活跃脚本 - 11个)

1. append_session_to_raw_data.py ⭐ - 通用数据追加工具
2. analyze_experiment_status.py - 实验状况统计
3. analyze_missing_energy_data.py - 缺失能耗分析
4. verify_recoverable_data.py - 验证数据可恢复性
5. repair_missing_energy_data.py - 修复缺失能耗
6. validate_raw_data.py - 验证数据完整性
7. check_latest_results.py - 检查最新结果
8. compare_data_vs_raw_data.py - 比较CSV一致性
9. create_unified_data_csv.py - 创建统一data.csv
10. check_attribute_mapping.py - 检查属性映射
11. validate_merged_metrics.py - 验证合并指标

### tools/data_management/ (归档脚本 - 4个)

1. add_new_experiments_to_raw_data.py - 特定会话实验追加
2. merge_csv_to_raw_data.py - CSV文件合并
3. update_raw_data_with_reextracted.py - 重新提取性能指标
4. merge_performance_metrics.py - 性能指标列合并

### tools/config_management/ (全部活跃 - 4个)

1. generate_mutation_config.py - 生成变异配置
2. validate_models_config.py - 验证模型配置
3. validate_mutation_config.py - 验证变异配置
4. verify_stage_configs.py - 验证stage配置

---

**报告完成**
**建议**: 立即执行归档操作，并更新文档以防止未来创建重复脚本。
