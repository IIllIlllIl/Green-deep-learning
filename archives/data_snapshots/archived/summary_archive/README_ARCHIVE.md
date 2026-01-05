# Summary Files Archive

**归档日期**: 2025-12-12 20:38:35

## 归档原因

随着v4.7.3版本的发布，我们完成了以下工作：

1. ✅ 合并 summary_old.csv (93列) 和 summary_new.csv (80列) 为 raw_data.csv (80列)
2. ✅ 验证 raw_data.csv 数据完整性和安全性 (476行, 100%完整)
3. ✅ 归档所有过时的summary文件和备份

## 归档文件清单

### Summary文件 (5个)

- **summary_all.csv**: 历史汇总文件（已被raw_data.csv替代）
- **summary_all_enhanced.csv**: 增强版汇总文件（已废弃）
- **summary_all_reorganized.csv**: 重组版汇总文件（已废弃）
- **summary_new_old_separation.csv**: 临时分离文件（已废弃）
- **summary_old_93col.csv**: 93列格式文件（已转换为80列）

### 备份文件 (8个)

- **summary_all.csv.backup_20251211_144013**: 旧版汇总备份
- **summary_all.csv.backup_before_reorganization_20251211_153625**: 重组前备份
- **summary_all_enhanced.csv.backup_before_add_3cols**: 增强版备份
- **summary_new.csv.backup_step5**: 新数据临时备份
- **summary_old_80col.csv.backup_step5**: 80列转换备份
- **summary_old.csv.backup_20251212_163203**: 旧版备份1
- **summary_old.csv.backup_20251212_174304**: 旧版备份2
- **summary_old.csv.backup_20251212_194255**: 旧版备份3

## 保留的文件

以下文件保留在 `results/` 目录中：

- **raw_data.csv**: 合并后的原始数据（80列） - 主数据文件
- **summary_old.csv**: 老实验数据（93列） - 源数据
- **summary_new.csv**: 新实验数据（80列） - 源数据
- **summary_old.csv.backup_80col**: 80列原始备份 - 重要备份
- **summary_old.csv.backup_before_93col_replacement**: 93列替换前备份 - 重要备份

## 数据访问

如需访问原始数据，请使用：

- **主数据文件**: `results/raw_data.csv` (476行, 80列, 100%完整)
- **老实验数据**: `results/summary_old.csv` (211行, 93列)
- **新实验数据**: `results/summary_new.csv` (265行, 80列)

## 归档文件使用

归档文件仅供历史参考，不推荐用于分析。如需恢复归档文件，请联系项目维护者。

---

**归档人**: Claude (AI助手)
**项目版本**: v4.7.3
