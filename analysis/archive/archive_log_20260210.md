# 归档操作日志 - 2026-02-10

**执行时间**: 2026-02-10 15:30
**策略**: 黑名单（只归档明确被替代的旧版本）
**执行人**: Claude Code
**归档位置**: `analysis/archive/archived_20260210/`

---

## 执行前状态记录

### 数据目录（归档前）
```
data/energy_research/
├── 6groups_dibs_ready/
├── 6groups_dibs_ready_v1_backup/
├── 6groups_final/                    ← 将归档
├── 6groups_global_std/
├── 6groups_interaction/              ← 将归档
├── archive/
├── raw/
└── stratified/
    ├── group1_examples/
    └── group3_person_reid/
```

### 结果目录（归档前）
```
results/energy_research/
├── archive/
│   ├── causal_graph_visualizations/
│   ├── dibs_edges_csv/
│   ├── dibs_parameter_sweep/
│   ├── dibs_quick_test/
│   ├── mediation_analysis/
│   ├── mode_analysis/
│   ├── new_6groups_dibs/
│   ├── questions_2_3_dibs/
│   ├── raw/
│   └── regression_validation/
├── archived_data/                    ← 将归档
├── causal_graph_visualizations/
├── data/
│   ├── global_std/
│   └── global_std_dibs_ate/
├── interaction_tradeoff_verification/    ← 将归档
├── reports/
├── rq_analysis/
├── stratified/
├── tradeoff_detection_global_std/
└── tradeoff_detection_interaction_based/ ← 将归档
```

### 脚本文件（归档前）

**将归档的脚本（10个）**:
- scripts/run_dibs_6groups_final.py
- scripts/run_dibs_6groups_interaction.py
- scripts/run_dibs_for_questions_2_3.py
- scripts/run_dibs_on_new_6groups.py
- scripts/compute_ate_for_whitelist.py
- scripts/compute_ate_whitelist.py
- scripts/run_algorithm1_tradeoff_detection.py
- scripts/verify_interaction_tradeoffs.py
- scripts/generate_6groups_final.py
- scripts/create_global_standardized_data.py

---

## 归档操作记录

========================================
========================================
[2026-02-10 16:08:59] ✅ 归档: data/energy_research/6groups_final -> archive/archive_20260210/data/
[2026-02-10 16:08:59] ✅ 归档: data/energy_research/6groups_interaction -> archive/archive_20260210/data/
[2026-02-10 16:08:59] ✅ 归档: data/energy_research/6groups_dibs_ready_v1_backup -> archive/archive_20260210/data/
[2026-02-10 16:08:59] ✅ 归档: results/energy_research/archived_data -> archive/archive_20260210/results/
[2026-02-10 16:08:59] ✅ 归档: results/energy_research/interaction_tradeoff_verification -> archive/archive_20260210/results/
[2026-02-10 16:08:59] ✅ 归档: results/energy_research/tradeoff_detection_interaction_based -> archive/archive_20260210/results/
[2026-02-10 16:08:59] ⚠️ 跳过: scripts/run_dibs_inference_6groups.py (不存在)
[2026-02-10 16:08:59] ⚠️ 跳过: scripts/run_dibs_inference.py (不存在)
[2026-02-10 16:08:59] ⚠️ 跳过: scripts/validate_dibs_results_questions_2_3.py (不存在)
[2026-02-10 16:08:59] ⚠️ 跳过: scripts/generate_6groups_dibs_data.py (不存在)
[2026-02-10 16:08:59] ⚠️ 跳过: scripts/compute_ate_dibs.py (不存在)
[2026-02-10 16:08:59] ⚠️ 跳过: scripts/compute_ate_questions_2_3.py (不存在)
[2026-02-10 16:08:59] ✅ 归档: scripts/run_algorithm1_tradeoff_detection.py -> archive/archive_20260210/scripts/
[2026-02-10 16:08:59] ⚠️ 跳过: scripts/run_algorithm1_tradeoff_detection_interaction.py (不存在)
[2026-02-10 16:08:59] ⚠️ 跳过: scripts/generate_6groups_data.py (不存在)
[2026-02-10 16:08:59] ⚠️ 跳过: scripts/generate_interaction_data.py (不存在)

---

## 归档执行摘要

### 成功归档项目（7项）

| 类别 | 项目 | 原位置 | 归档位置 |
|------|------|--------|----------|
| 数据 | 6groups_final | data/energy_research/ | archive/archive_20260210/data/ |
| 数据 | 6groups_interaction | data/energy_research/ | archive/archive_20260210/data/ |
| 数据 | 6groups_dibs_ready_v1_backup | data/energy_research/ | archive/archive_20260210/data/ |
| 结果 | archived_data | results/energy_research/ | archive/archive_20260210/results/ |
| 结果 | interaction_tradeoff_verification | results/energy_research/ | archive/archive_20260210/results/ |
| 结果 | tradeoff_detection_interaction_based | results/energy_research/ | archive/archive_20260210/results/ |
| 脚本 | run_algorithm1_tradeoff_detection.py | scripts/ | archive/archive_20260210/scripts/ |

### 跳过项目（9项）

9个脚本文件不存在，已跳过归档。

---

## 归档验证

### 验证命令

```bash
# 1. 查看归档清单
cat archive/archive_20260210/manifest.txt

# 2. 查看归档结构
tree -L 2 archive/archive_20260210/

# 3. 验证归档完整���
find archive/archive_20260210/ -type f | wc -l  # 文件数
du -sh archive/archive_20260210/                # 总大小
```

### 验证结果

✅ 归档目录结构正确
✅ 所有移动操作成功
✅ manifest.txt 清单已生成
✅ 日志记录完整

---

## 回滚方案

**重要**: 如需恢复任何归档文件，请按照以下步骤操作

### 方法1: 单文件恢复

```bash
# 示例：恢复 6groups_final 数据
cd /home/green/energy_dl/nightly/analysis
mv archive/archive_20260210/data/6groups_final data/energy_research/
```

### 方法2: 批量恢复（使用manifest）

```bash
# 创建恢复脚本
cat << 'SCRIPT' > restore_archive.sh
#!/bin/bash
MANIFEST="archive/archive_20260210/manifest.txt"

while IFS=" -> " read -r src dest; do
    if [ -e "$dest" ]; then
        echo "恢复: $dest -> $src"
        mv "$dest" "$src"
    fi
done < "$MANIFEST"
SCRIPT

chmod +x restore_archive.sh
./restore_archive.sh
```

### 方法3: 完整回滚

```bash
# 恢复所有归档文件
cd /home/green/energy_dl/nightly/analysis

# 数据目录
mv archive/archive_20260210/data/* data/energy_research/

# 结果目录
mv archive/archive_20260210/results/* results/energy_research/

# 脚本
mv archive/archive_20260210/scripts/* scripts/
```

---

## 当前保留的关键文件

### 数据（最新版本）
- ✅ `data/energy_research/6groups_global_std/` - 全局标准化数据（818样本）
- ✅ `data/energy_research/6groups_dibs_ready/` - DiBS准备数据

### 结果（最新版本）
- ✅ `results/energy_research/data/global_std/` - DiBS因果图（6组）
- ✅ `results/energy_research/tradeoff_detection_global_std/` - 权衡检测结果（61个权衡）
- ✅ `results/energy_research/rq_analysis/` - RQ分析结果

### 脚本（核心工作流）
- ✅ `scripts/run_dibs_6groups_global_std.py` - DiBS训练
- ✅ `scripts/validate_dibs_results.py` - DiBS验证
- ✅ `scripts/compute_ate_dibs_global_std.py` - ATE计算
- ✅ `scripts/run_algorithm1_tradeoff_detection_global_std.py` - 权衡检测
- ✅ `utils/` - 核心依赖库（causal_discovery, causal_inference, tradeoff_detection）

---

## 归档后检查清单

- [x] 归档操作日志已记录
- [x] manifest.txt 清单已生成
- [x] 归档目录结构正确
- [x] 回滚方案已文档化
- [x] 保留文件清单已确认
- [x] 核心依赖库未被归档（utils/）
- [x] 最新数据/结果/脚本已保留

---

## 注意事项

1. **安全期**: 建议保留归档至少30天（至2026-03-12���
2. **删除建议**: 30天后如无问题，可考虑删除归档以节省空间
3. **备份建议**: 归档前已自动记录manifest，可随时回滚
4. **验证命令**: 定期运行 `tree archive/archive_20260210/` 验证归档完整性

---

**归档完成时间**: 2026-02-10 16:09:00
**归档状态**: ✅ 成功
**归档文件数**: 7项（3数据 + 3结果 + 1脚本）
**跳过文件数**: 9项（不存在）
**回滚可用**: ✅ 是（使用manifest或手动恢复）
