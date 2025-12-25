# v4.7.7 更新说明

**更新日期**: 2025-12-15
**版本**: v4.7.7
**主要内容**: 数据验证修复、实验差距分析、Phase 6配置、脚本整理

---

## 更新摘要

### 1. 数据验证与修复 ✅
- **共享ID数据验证**: 37个experiment_id（77行），31行修复完成
- **calculate_experiment_gap.py修复**: 正确处理并行模式数据
- **实验差距重新计算**: 90个实验缺口，56.98小时

### 2. Phase 6配置设计 ✅
- **配置文件**: `settings/phase6_vulberta_mlp_completion.json`
- **实验数**: 40个（VulBERTa/mlp完全补齐）
- **运行时间**: 41.72小时（36-42h目标范围内）
- **目标**: VulBERTa/mlp模型100%达标

### 3. 脚本修复与整理 ✅
- **修复**: `update_raw_data_with_reextracted.py` - 使用experiment_id + timestamp
- **归档**: 8个Phase 5任务脚本 → `scripts/archived/completed_phase5_tasks_20251215/`
- **发现**: 重复功能脚本（`add_new_experiments_to_raw_data.py` vs `append_session_to_raw_data.py`）

### 4. 文档更新 ✅
- 新增: `V4_7_7_PROJECT_UPDATE_SUMMARY.md` - 完整更新总结
- 新增: `PHASE6_CONFIGURATION_REPORT.md` - Phase 6配置报告
- 更新: `calculate_experiment_gap.py` - Bug修复说明

---

## 关键修复

### Bug 1: calculate_experiment_gap.py - 并行模式数据判断
**问题**: 脚本未检查fg_*字段，导致并行实验被低估
**影响**: 错误显示133个缺口（实际90个）
**修复**: 根据mode字段动态选择字段前缀

### Bug 2: calculate_experiment_gap.py - 能耗数据验证 ⭐⭐
**问题**: 遗漏能耗数据完整性检查，只验证训练成功+性能数据
**影响**: 高估有效实验数（~430 vs 实际371），低估实验缺口
**修复**: 添加能耗数据验证（has_energy检查）
**关键发现**: 37%实验有性能数据但缺能耗数据
**符合研究目标**: CLAUDE.md明确要求"完整数据: 能耗 + 任意性能指标"

### Bug 3: update_raw_data_with_reextracted.py - 索引唯一性
**问题**: 仅使用experiment_id查找，不同批次experiment_id重复
**影响**: 可能匹配到错误的实验目录
**修复**: 使用experiment_id + timestamp复合键

---

## Phase 6 配置

**配置文件**: `settings/phase6_vulberta_mlp_completion.json`

| 项目 | 值 |
|------|---|
| 实验数 | 40个 |
| 预计时间 | 41.72小时 |
| 目标模型 | VulBERTa/mlp |
| 完成后达标 | 19/90 (+2) |

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/phase6_vulberta_mlp_completion.json
```

---

## 项目整理成果

### 归档脚本（8个）
**Phase 5任务脚本**:
- analyze_shared_performance_issue.py
- check_shared_performance_metrics.py
- analyze_phase5_completion.py
- estimate_phase5_time.py

**一次性任务脚本**:
- expand_raw_data_columns.py
- restore_and_reappend_phase5.py
- recalculate_num_mutated_params.py
- reextract_performance_metrics.py

**归档位置**: `scripts/archived/completed_phase5_tasks_20251215/`

### 重复功能标记
- `add_new_experiments_to_raw_data.py` (189行) vs
- `append_session_to_raw_data.py` (476行) ← **保留**

**建议**: 整合前者功能到后者

---

## 数据质量状态

| 指标 | 值 | 百分比 |
|------|---|--------|
| 总行数 | 584 | - |
| 训练成功 | 502/584 | 86.0% |
| 能耗数据 | 488/584 | 83.6% |
| 性能数据 | 376/584 | 64.4% |
| 复合键唯一性 | 584/584 | 100% ✅ |

---

## 实验进度

| 模式 | 达标模型 | 总模型 | 达标率 |
|------|---------|--------|--------|
| 非并行 | 9 | 11 | 81.8% |
| 并行 | 8 | 11 | 72.7% |
| **总计** | **17** | **90** | **18.9%** |

**剩余缺口**:
- VulBERTa/mlp: 40个实验（41.72h）← Phase 6目标
- bug-localization: 40个实验（10.90h）
- MRT-OAST: 10个实验（4.36h）

---

**详细报告**: `docs/results_reports/V4_7_7_PROJECT_UPDATE_SUMMARY.md`
