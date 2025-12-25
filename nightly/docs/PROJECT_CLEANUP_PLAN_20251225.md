# 项目清理和归档计划

**创建时间**: 2025-12-25
**版本**: v1.0
**状态**: 待执行

---

## 📋 执行摘要

基于全面的项目文件检查，识别出以下可归档/清理的文件：

| 类别 | 文件数 | 操作 | 节省空间（估计） |
|------|--------|------|-----------------|
| 过时数据文件 | 2 | 归档 | ~240KB |
| 旧备份文件 | 5 | 清理 | ~1.4MB |
| Analysis中间文件 | 19 | 清理 | ~估计1MB |
| Analysis备份目录 | 1 | 清理 | ~估计2MB |
| 错位数据文件 | 2 | 移动 | 0 (移动) |
| **总计** | **29** | **-** | **~4.6MB** |

---

## 🎯 清理目标

1. **提高项目结构清晰度** - 移除过时和重复文件
2. **减少存储空间占用** - 清理不必要的备份
3. **改善文件组织** - 确保数据文件在正确位置
4. **保持可追溯性** - 归档而非删除重要历史文件

---

## 📁 详细清理计划

### 1. Results目录 - 过时数据文件归档

#### 1.1 summary_new.csv 和 summary_old.csv

**文件信息**:
- `results/summary_new.csv` (132KB, 266行) - 新实验数据（80列）
- `results/summary_old.csv` (105KB, 212行) - 老实验数据（93列）

**状态**: 已被合并到 `raw_data.csv`

**引用情况**:
- 仅在归档脚本 `scripts/archive_summary_files.py` 中引用
- 不再被主流程使用

**建议操作**: ✅ **归档**

```bash
# 创建归档目录
mkdir -p results/archived/merged_20251212

# 归档文件
mv results/summary_new.csv results/archived/merged_20251212/
mv results/summary_old.csv results/archived/merged_20251212/

# 创建README
cat > results/archived/merged_20251212/README.md << 'EOF'
# 已合并数据文件归档

**归档时间**: 2025-12-25
**原因**: 已合并到 raw_data.csv

## 文件说明

- `summary_new.csv` (266行, 80列) - 新实验数据（2025-11-26后）
- `summary_old.csv` (212行, 93列) - 老实验数据（2025-11-26前）

## 合并情况

这两个文件已在2025-12-12合并为 `raw_data.csv`（80列标准格式）。
详见: `docs/results_reports/DATA_FORMAT_DESIGN_DECISION_SUMMARY.md`

## 保留原因

保留作为历史参考，用于追溯数据来源。
EOF
```

#### 1.2 旧备份文件清理

**建议清理的备份**:

| 文件 | 大小 | 日期 | 清理原因 |
|------|------|------|----------|
| `raw_data.backup_20251221_215643.csv` | 301KB | 12-21 | 已有更新备份 (12-23) |
| `raw_data.csv.backup_before_clean` | 301KB | 12-19 | 清理前备份，已完成清理 |
| `data.csv.backup_before_column_removal_20251219_182227` | 277KB | 12-19 | 列移除前备份，已确认无问题 |
| `data.csv.backup_before_merge_20251219_180149` | 277KB | 12-19 | 合并前备份，已确认无问题 |

**保留的备份**:
- ✅ `raw_data.csv.backup_20251223_195253` (302KB) - 最新有效备份
- ✅ `data.csv.backup_20251223_202113` (276KB) - 最新有效备份

**建议操作**: ✅ **清理旧备份**

```bash
# 清理旧备份（保留最新的即可）
rm results/raw_data.backup_20251221_215643.csv
rm results/raw_data.csv.backup_before_clean
rm results/data.csv.backup_before_column_removal_20251219_182227
rm results/data.csv.backup_before_merge_20251219_180149
```

---

### 2. Analysis目录 - 中间文件清理

#### 2.1 错位的数据文件移动

**问题**: 数据文件出现在脚本目录下

**文件**:
- `analysis/scripts/data/energy_research/raw/energy_data_extracted_v2.csv`
- `analysis/scripts/data/energy_research/raw/extracted_columns_info.json`

**建议操作**: ✅ **移动到正确位置**

```bash
# 检查是否已存在于正确位置
ls -lh data/energy_research/raw/energy_data_extracted_v2.csv

# 如果不存在，移动过去
if [ ! -f data/energy_research/raw/energy_data_extracted_v2.csv ]; then
    mv analysis/scripts/data/energy_research/raw/energy_data_extracted_v2.csv \
       data/energy_research/raw/
fi

# 移动json文件
if [ ! -f data/energy_research/raw/extracted_columns_info.json ]; then
    mv analysis/scripts/data/energy_research/raw/extracted_columns_info.json \
       data/energy_research/raw/
fi

# 清理空目录
rm -rf analysis/scripts/data/
```

#### 2.2 Processed中间文件清理

**问题**: `data/energy_research/processed/` 包含19个stage中间文件

**文件列表**:
```
stage0_validated.csv
stage1_unified.csv
stage2_mediators.csv
stage3_bug_localization.csv
stage3_image_classification.csv
stage3_person_reid.csv
stage3_vulberta.csv
stage4_*.csv (4个)
stage5_*.csv (4个)
stage6_*.csv (4个)
```

**当前状态**:
- 最终输出在 `data/energy_research/training/` 目录
- stage文件为调试和验证中间步骤

**建议操作**: ✅ **清理中间文件**（保留最终训练数据）

```bash
# 备份当前processed目录（如果需要）
cd analysis
cp -r data/energy_research/processed data/energy_research/processed.backup_cleanup_$(date +%Y%m%d)

# 清理stage中间文件
rm data/energy_research/processed/stage*.csv

# 仅保留最终训练数据文件
ls data/energy_research/processed/training_data_*.csv

# 或者全部清理processed目录（因为training目录有最终数据）
# rm -rf data/energy_research/processed/*
```

#### 2.3 旧备份目录清理

**目录**: `data/energy_research/processed.backup_4groups_20251224/`

**内容**: 25个CSV文件（包括19个stage文件）

**创建时间**: 2025-12-24

**建议操作**: ⚠️ **评估后清理**

```bash
# 检查备份目录大小
du -sh data/energy_research/processed.backup_4groups_20251224/

# 评估：如果当前training数据正常工作，可以删除
# 如果需要保留作为历史记录，可以压缩
cd analysis
tar -czf data/energy_research/processed.backup_4groups_20251224.tar.gz \
        data/energy_research/processed.backup_4groups_20251224/
rm -rf data/energy_research/processed.backup_4groups_20251224/
```

---

### 3. 文档归档检查

#### 3.1 已归档文档（无需操作）

以下文档已在 `docs/archived/` 目录下：
- ✅ `STAGE_CONFIG_FIX_REPORT.md`
- ✅ `WORK_SUMMARY_DROPOUT_TEST_20251119.md`
- ✅ `COMPLETE_FIX_SUMMARY.md`
- ✅ `MUTATION_MECHANISMS_DETAILED.md`
- ✅ `DEFAULT_BASELINE_REPORT_20251118.md`
- 等等...

#### 3.2 包含"OLD"的文档

以下文档名包含"OLD"但仍在活跃使用：
- ✅ `docs/results_reports/SUMMARY_OLD_REBUILD_80COL_REPORT_20251212.md` - 记录80列重建过程
- ✅ `docs/results_reports/OLD_EXPERIMENT_BG_HYPERPARAM_ANALYSIS.md` - 背景超参数分析
- ✅ `docs/results_reports/SUMMARY_NEW_VS_OLD_COLUMN_ANALYSIS.md` - 新旧列对比

**建议**: ✅ **保留**（这些是分析报告，不是过时文档）

---

### 4. 脚本和测试文件评估

#### 4.1 脚本文件

**总数**: 35个Python脚本在 `scripts/`

**评估结果**: 大多数脚本仍在使用或作为工具保留

**建议归档的脚本**:
```
scripts/archived/completed_tasks_20251212/ (已归档)
├── convert_summary_old_to_80col.py
├── step4_add_mutation_count.py
└── ...
```

**建议**: ✅ **当前脚本保持原样**（均有用途）

#### 4.2 测试文件

**总数**: 17个测试文件在 `tests/`

**可能过时的测试**:
- `tests/test_old_csv_rebuild.py` - 测试旧CSV重建（已完成）
- `tests/validate_80col_format.py` - 验证80列格式（已完成）
- `tests/validate_rebuilt_summary_old.py` - 验证重建的summary_old（已完成）

**建议操作**: ✅ **归档完成的测试**

```bash
# 创建测试归档目录
mkdir -p tests/archived/completed_20251212

# 归档已完成的测试
mv tests/test_old_csv_rebuild.py tests/archived/completed_20251212/
mv tests/validate_80col_format.py tests/archived/completed_20251212/
mv tests/validate_rebuilt_summary_old.py tests/archived/completed_20251212/

# 创建README
cat > tests/archived/completed_20251212/README.md << 'EOF'
# 已完成测试归档

**归档时间**: 2025-12-25
**原因**: 测试目标已完成且验证通过

## 归档的测试

1. `test_old_csv_rebuild.py` - 测试summary_old重建为80列
2. `validate_80col_format.py` - 验证80列格式一致性
3. `validate_rebuilt_summary_old.py` - 验证重建后的summary_old

## 归档原因

这些测试是针对2025-12-12的数据格式统一任务。任务已完成并验证通过，
数据已合并到raw_data.csv，因此这些测试不再需要运行。

保留作为历史参考。
EOF
```

---

### 5. 其他建议

#### 5.1 Python缓存清理

```bash
# 清理__pycache__目录
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
```

#### 5.2 日志文件检查

```bash
# 检查是否有大型日志文件
find . -name "*.log" -size +10M 2>/dev/null

# 检查analysis日志
ls -lh analysis/logs/experiments/*.log 2>/dev/null | tail -10
```

---

## ✅ 执行清单

### Phase 1: 安全归档（可逆操作）

- [ ] 归档 `summary_new.csv` 和 `summary_old.csv`
- [ ] 移动错位的数据文件到正确位置
- [ ] 归档已完成的测试文件

### Phase 2: 清理备份（不可逆，需确认）

- [ ] 确认最新备份文件完整性
- [ ] 清理4个旧备份文件
- [ ] 清理analysis备份目录（压缩后删除）

### Phase 3: 清理中间文件（不可逆，需确认）

- [ ] 确认training目录数据完整
- [ ] 清理processed目录的19个stage文件
- [ ] 清理Python缓存文件

### Phase 4: 文档更新

- [ ] 更新CLAUDE.md中的文件结构
- [ ] 生成清理执行报告
- [ ] 更新相关文档的文件路径引用

---

## 🔒 安全措施

### 执行前检查

```bash
# 1. 确认主数据文件完整
wc -l results/raw_data.csv results/data.csv
head -3 results/raw_data.csv
tail -3 results/raw_data.csv

# 2. 确认最新备份可用
wc -l results/raw_data.csv.backup_20251223_195253
diff <(head -1 results/raw_data.csv) <(head -1 results/raw_data.csv.backup_20251223_195253)

# 3. 确认training数据完整
ls -lh analysis/data/energy_research/training/
wc -l analysis/data/energy_research/training/*.csv
```

### 创建完整备份

```bash
# 执行清理前创建完整备份
cd /home/green/energy_dl/nightly
tar -czf ../nightly_backup_before_cleanup_20251225.tar.gz \
    --exclude='results/run_*' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    .
```

---

## 📊 预期效果

### 清理前

```
results/
├── *.csv (7个文件, ~2.2MB)
├── archived/ (已有归档)

analysis/data/
├── energy_research/
│   ├── processed/ (19个stage文件 + 6个training文件)
│   ├── processed.backup_4groups_20251224/ (25个文件)
│   └── ...

tests/
├── *.py (17个文件)
```

### 清理后

```
results/
├── raw_data.csv (321KB) ✅ 主数据文件
├── data.csv (296KB) ✅ 精简数据文件
├── raw_data.csv.backup_20251223_195253 (302KB) ✅ 最新备份
├── data.csv.backup_20251223_202113 (276KB) ✅ 最新备份
├── archived/
│   └── merged_20251212/ (summary_new.csv, summary_old.csv)

analysis/data/
├── energy_research/
│   ├── raw/ (包括移动过来的extracted文件)
│   ├── training/ (4个最终训练数据文件) ✅
│   └── processed.backup_4groups_20251224.tar.gz (压缩备份)

tests/
├── *.py (14个活跃测试)
├── archived/
│   └── completed_20251212/ (3个已完成测试)
```

---

## 📝 执行后验证

```bash
# 1. 验证主数据文件
python3 scripts/validate_raw_data.py

# 2. 验证training数据可用
cd analysis
python3 -c "
import pandas as pd
for task in ['bug_localization', 'image_classification', 'person_reid', 'vulberta']:
    df = pd.read_csv(f'data/energy_research/training/training_data_{task}.csv')
    print(f'{task}: {len(df)} rows')
"

# 3. 验证脚本依然可用
python3 scripts/validate_models_config.py
python3 scripts/calculate_experiment_gap.py

# 4. 运行关键测试
python3 -m pytest tests/verify_csv_append_fix.py -v
```

---

## 🎯 总结

### 清理收益

1. **空间节省**: ~4.6MB
2. **文件减少**: 29个文件
3. **结构清晰**: 归档过时文件，保留活跃文件
4. **可维护性**: 更容易找到当前使用的文件

### 风险控制

1. ✅ **归档而非删除** - 重要历史文件可追溯
2. ✅ **完整备份** - 执行前创建tar.gz备份
3. ✅ **分阶段执行** - 可逆操作优先
4. ✅ **验证机制** - 每阶段验证数据完整性

---

**下一步**: 等待用户确认后执行清理计划
