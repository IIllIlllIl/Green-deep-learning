# 已完成的数据任务脚本归档

**归档日期**: 2026-01-10
**归档原因**: 脚本重复性分析和清理

本目录包含已完成的一次性数据处理任务脚本。这些脚本的功能已被通用工具取代，或仅用于历史特定任务，不再需要在日常工作流中使用。

---

## 归档脚本清单

### 1. add_new_experiments_to_raw_data.py (6.0K)

**用途**: 从特定会话 `run_20251212_224937` 中提取4个Phase 2诊断实验并追加到raw_data.csv

**归档原因**:
- 硬编码特定session路径，不通用
- 功能已被通用脚本 `tools/data_management/append_session_to_raw_data.py` 完全取代
- 仅用于一次性任务，已完成

**替代方案**:
```bash
# 使用通用脚本追加任意session的数据
python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS
```

---

### 2. merge_csv_to_raw_data.py (8.7K)

**用途**: 合并 `summary_old.csv` (93列) 和 `summary_new.csv` (80列) 为统一的 `raw_data.csv` (80列格式)

**归档原因**:
- 特定文件名，专门用于历史数据格式统一任务
- 一次性任务，已完成
- 不适用于日常数据追加工作流

**历史背景**:
项目早期存在两种CSV格式（93列和80列），此脚本用于统一数据格式。现在所有新数据已使用80列标准格式。

---

### 3. update_raw_data_with_reextracted.py (11K)

**用途**: 使用更新后的正则表达式从 `terminal_output.txt` 重新提取性能指标，并更新 `raw_data.csv`

**归档原因**:
- 一次性数据修复任务
- 用于修复早期提取模式不完善导致的数据缺失
- 任务已完成，数据已修复

**历史背景**:
早期实验的性能指标提取正则表达式不够完善，导致部分数据未正确提取。此脚本用于批量重新提取和修复。

**注意**: 如果未来需要类似功能，应使用 `append_session_to_raw_data.py` 的重提取功能。

---

### 4. merge_performance_metrics.py (4.3K)

**用途**: 合并和重命名性能指标列
- MRT-OAST: `accuracy` → `test_accuracy`
- VulBERTa/mlp: `eval_loss` → `test_loss`

**归档原因**:
- 一次性列合并和重命名任务
- 用于统一不同模型的性能指标命名
- 任务已完成

**历史背景**:
不同模型使用不同的性能指标名称。此脚本用于统一命名规范，便于后续分析。

**配套脚本**: `validate_merged_metrics.py` (仍在活跃目录，用于验证合并质量)

---

## 使用建议

### ⚠️ 不建议直接使用这些脚本

这些脚本保留仅用于：
1. **历史参考**: 了解历史数据处理过程
2. **问题追溯**: 调查历史数据问题时参考
3. **审计合规**: 保持完整的代码变更历史

### ✅ 推荐使用的替代方案

对于日常数据管理任务，请使用以下活跃脚本：

| 任务 | 推荐脚本 |
|------|----------|
| 追加新实验数据 | `tools/data_management/append_session_to_raw_data.py` ⭐ |
| 验证数据完整性 | `tools/data_management/validate_raw_data.py` |
| 分析实验状况 | `tools/data_management/analyze_experiment_status.py` |
| 修复缺失能耗 | `tools/data_management/repair_missing_energy_data.py` |

完整脚本列表参见: `docs/SCRIPTS_QUICKREF.md`

---

## 相关文档

- [脚本重复性分析报告](../../../docs/SCRIPT_DUPLICATION_ANALYSIS_REPORT.md) - 详细分析过程
- [脚本快速参考](../../../docs/SCRIPTS_QUICKREF.md) - 活跃脚本使用指南
- [数据管理工具文档](../../../docs/CLAUDE_FULL_REFERENCE.md) - 完整参考手册

---

## 技术细节

### 归档决策依据

根据 2026-01-10 的脚本重复性分析：

1. **功能重复**: 通用脚本已实现相同或更好的功能
2. **特定任务**: 仅用于历史的一次性任务
3. **维护成本**: 保持活跃状态会增加维护负担
4. **混淆风险**: 多个相似脚本容易导致错误使用

### 归档标准

符合以下任一条件的脚本会被归档：
- ✅ 一次性任务已完成
- ✅ 功能已被通用脚本取代
- ✅ 硬编码特定参数，不适用于通用场景
- ✅ 超过3个月未使用

---

**维护者**: Green
**归档日期**: 2026-01-10
**归档批次**: completed_data_tasks_20260110
