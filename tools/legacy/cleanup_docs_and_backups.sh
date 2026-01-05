#!/bin/bash
# 文档和备份清理脚本
# 日期: 2025-12-08
# 目的: 归档过时文档，删除不必要的备份文件

set -e

echo "=================================="
echo "文档和备份清理脚本 v4.7.2"
echo "=================================="

# 创建归档目录
ARCHIVE_BASE="docs/archived"
ARCHIVE_DIR="${ARCHIVE_BASE}/2025-12-08_pre_v4.7.2"

echo ""
echo "步骤1: 创建归档目录"
mkdir -p "$ARCHIVE_DIR"
echo "✅ 创建: $ARCHIVE_DIR"

# 归档过时的去重相关文档（这些已被v4.6.0的去重模式区分修复取代）
echo ""
echo "步骤2: 归档过时的去重文档"
DEDUP_DOCS=(
    "docs/ARCHIVE_DEDUPLICATION_V2.md"
    "docs/DEDUPLICATION_FINAL_SUMMARY.md"
    "docs/DEDUPLICATION_UPDATE_V2.md"
    "docs/INTER_ROUND_DEDUPLICATION.md"
)

for doc in "${DEDUP_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" "$ARCHIVE_DIR/"
        echo "  📦 归档: $(basename $doc)"
    fi
done

# 归档过时的组织文档
echo ""
echo "步骤3: 归档过时的文件组织文档"
ORG_DOCS=(
    "docs/FILE_ORGANIZATION_UPDATE.md"
    "docs/QUICK_FILE_INDEX.md"
)

for doc in "${ORG_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" "$ARCHIVE_DIR/"
        echo "  📦 归档: $(basename $doc)"
    fi
done

# 归档过时的脚本整合文档
echo ""
echo "步骤4: 归档脚本整合文档"
SCRIPT_DOCS=(
    "docs/SCRIPTS_CONSOLIDATION_PLAN.md"
    "docs/SCRIPTS_CONSOLIDATION_REPORT.md"
)

for doc in "${SCRIPT_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" "$ARCHIVE_DIR/"
        echo "  📦 归档: $(basename $doc)"
    fi
done

# 归档其他过时文档
echo ""
echo "步骤5: 归档其他过时文档"
OTHER_DOCS=(
    "docs/MISSING_EXPERIMENTS_CHECKLIST.md"
    "docs/SUMMARY_APPEND_CONTROL_UPDATE.md"
)

for doc in "${OTHER_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" "$ARCHIVE_DIR/"
        echo "  📦 归档: $(basename $doc)"
    fi
done

# 创建归档README
echo ""
echo "步骤6: 创建归档说明"
cat > "${ARCHIVE_DIR}/README_ARCHIVE.md" << 'EOF'
# 归档文档说明 (2025-12-08)

**归档日期**: 2025-12-08
**归档原因**: v4.7.2发布前清理，这些文档已过时或被更新版本取代

## 归档内容

### 去重机制文档（被v4.6.0+更新取代）
- `ARCHIVE_DEDUPLICATION_V2.md` - v2.0去重更新归档（2025-11-26）
- `DEDUPLICATION_FINAL_SUMMARY.md` - 去重机制完成总结（2025-11-26）
- `DEDUPLICATION_UPDATE_V2.md` - 去重v2.0更新（2025-11-26）
- `INTER_ROUND_DEDUPLICATION.md` - 轮间去重文档（2025-11-26）

**取代文档**:
- `docs/results_reports/DEDUP_MODE_FIX_REPORT.md` (v4.6.0)
- `docs/DEDUPLICATION_USER_GUIDE.md` (仍有效)

### 文件组织文档（已整合到CLAUDE.md）
- `FILE_ORGANIZATION_UPDATE.md` - 文件组织规范更新（2025-11-26）
- `QUICK_FILE_INDEX.md` - 快速文件索引

**取代文档**: `CLAUDE.md` 的"文件结构规范"章节

### 脚本整合文档（已完成）
- `SCRIPTS_CONSOLIDATION_PLAN.md` - 脚本整合计划
- `SCRIPTS_CONSOLIDATION_REPORT.md` - 脚本整合报告

**状态**: 整合已完成，文档仅作历史记录

### 其他过时文档
- `MISSING_EXPERIMENTS_CHECKLIST.md` - 缺失实验检查清单（被Stage规划取代）
- `SUMMARY_APPEND_CONTROL_UPDATE.md` - CSV追加控制更新（被v4.4.0修复取代）

## 当前有效文档

参考以下文档获取最新信息：

### 核心指南
- `CLAUDE.md` - Claude助手完整指南（v1.3）
- `README.md` - 项目主README（v4.7.2）
- `docs/FEATURES_OVERVIEW.md` - 功能总览

### 配置与使用
- `docs/JSON_CONFIG_BEST_PRACTICES.md` - JSON配置最佳实践 ⭐⭐⭐
- `docs/SETTINGS_CONFIGURATION_GUIDE.md` - 配置指南
- `docs/QUICK_REFERENCE.md` - 快速参考

### 实验报告
- `docs/results_reports/STAGE11_BUG_FIX_REPORT.md` - Stage11 Bug修复 (v4.7.2) ⭐
- `docs/results_reports/STAGE_CONFIG_VERIFICATION_REPORT.md` - 配置验证 (v4.7.2)
- `docs/results_reports/DEDUP_MODE_FIX_REPORT.md` - 去重模式修复 (v4.6.0)

---

**维护**: 这些文档已归档，仅作历史参考，不应用于当前开发。
EOF

echo "  📝 创建: README_ARCHIVE.md"

# 删除不必要的备份文件（这些是修复前的配置，包含错误）
echo ""
echo "步骤7: 删除settings备份文件"
echo "  ⚠️  这些备份包含多参数混合变异错误，已被修复版本取代"

BACKUP_FILES=(
    "settings/stage7_nonparallel_fast_models.json.bak"
    "settings/stage8_nonparallel_medium_slow_models.json.bak"
    "settings/stage9_nonparallel_hrnet18.json.bak"
    "settings/stage10_nonparallel_pcb.json.bak"
    "settings/stage11_parallel_hrnet18.json.bak"
    "settings/stage12_parallel_pcb.json.bak"
    "settings/stage13_parallel_fast_models_supplement.json.bak"
    "settings/stage13_parallel_fast_models_supplement.json.bak_20251207"
    "settings/stage14_stage7_8_supplement.json.bak_20251207"
)

for backup in "${BACKUP_FILES[@]}"; do
    if [ -f "$backup" ]; then
        rm "$backup"
        echo "  🗑️  删除: $(basename $backup)"
    fi
done

# 保留一个全量备份（2025-12-01的完整配置）
echo ""
echo "步骤8: 保留的备份文件"
KEEP_BACKUPS=(
    "settings/archived/mutation_2x_supplement_full_backup_20251201.json"
)

for keep in "${KEEP_BACKUPS[@]}"; do
    if [ -f "$keep" ]; then
        echo "  ✅ 保留: $(basename $keep) - 完整历史备份"
    fi
done

# 统计
echo ""
echo "=================================="
echo "清理完成统计"
echo "=================================="
ARCHIVED_COUNT=$(ls -1 "$ARCHIVE_DIR" 2>/dev/null | grep -v "README" | wc -l)
echo "  📦 归档文档: ${ARCHIVED_COUNT} 个"
echo "  🗑️  删除备份: ${#BACKUP_FILES[@]} 个"
echo "  ✅ 保留备份: ${#KEEP_BACKUPS[@]} 个"
echo ""
echo "归档位置: $ARCHIVE_DIR"
echo "=================================="
