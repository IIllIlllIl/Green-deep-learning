#!/bin/bash
# 文件归档脚本
# 日期: 2026-02-10
# 用途: 归档旧版本数据和结果文件

set -e  # 遇到错误即退出

BASE_DIR="/home/green/energy_dl/nightly/analysis"
ARCHIVE_DIR="$BASE_DIR/archive/archived_20260210"

echo "=========================================="
echo "文件归档操作"
echo "=========================================="
echo "归档目录: $ARCHIVE_DIR"
echo "归档项目: {len(archive_items)}个目录"
echo ""
echo "⚠️  警告: 此操作将移动以下目录到归档文件夹"
echo "请仔细检查列表！"
echo ""
read -p "确认继续？(输入 YES 继续): " confirm

if [ "$confirm" != "YES" ]; then
    echo "已取消"
    exit 1
fi

# 创建归档目录
mkdir -p "$ARCHIVE_DIR"/{data,results,scripts}

echo ""
echo "开始归档..."

# 归档: data/energy_research/6groups_final (24天)
if [ -e "$BASE_DIR/data/energy_research/6groups_final" ]; then
    echo "  归档: data/energy_research/6groups_final"
    mv "$BASE_DIR/data/energy_research/6groups_final" "$ARCHIVE_DIR/data/"
else
    echo "  ⚠️  跳过（不存在）: data/energy_research/6groups_final"
fi

# 归档: data/energy_research/6groups_final/archive (24天)
if [ -e "$BASE_DIR/data/energy_research/6groups_final/archive" ]; then
    echo "  归档: data/energy_research/6groups_final/archive"
    mv "$BASE_DIR/data/energy_research/6groups_final/archive" "$ARCHIVE_DIR/data/"
else
    echo "  ⚠️  跳过（不存在）: data/energy_research/6groups_final/archive"
fi

# 归档: data/energy_research/6groups_interaction (24天)
if [ -e "$BASE_DIR/data/energy_research/6groups_interaction" ]; then
    echo "  归档: data/energy_research/6groups_interaction"
    mv "$BASE_DIR/data/energy_research/6groups_interaction" "$ARCHIVE_DIR/data/"
else
    echo "  ⚠️  跳过（不存在）: data/energy_research/6groups_interaction"
fi

# 归档: data/energy_research/archive (24天)
if [ -e "$BASE_DIR/data/energy_research/archive" ]; then
    echo "  归档: data/energy_research/archive"
    mv "$BASE_DIR/data/energy_research/archive" "$ARCHIVE_DIR/data/"
else
    echo "  ⚠️  跳过（不存在）: data/energy_research/archive"
fi

# 归档: data/energy_research/raw (25天)
if [ -e "$BASE_DIR/data/energy_research/raw" ]; then
    echo "  归档: data/energy_research/raw"
    mv "$BASE_DIR/data/energy_research/raw" "$ARCHIVE_DIR/data/"
else
    echo "  ⚠️  跳过（不存在）: data/energy_research/raw"
fi

# 归档: data/energy_research/stratified (3天)
if [ -e "$BASE_DIR/data/energy_research/stratified" ]; then
    echo "  归档: data/energy_research/stratified"
    mv "$BASE_DIR/data/energy_research/stratified" "$ARCHIVE_DIR/data/"
else
    echo "  ⚠️  跳过（不存在）: data/energy_research/stratified"
fi

# 归档: data/energy_research/stratified/group1_examples (3天)
if [ -e "$BASE_DIR/data/energy_research/stratified/group1_examples" ]; then
    echo "  归档: data/energy_research/stratified/group1_examples"
    mv "$BASE_DIR/data/energy_research/stratified/group1_examples" "$ARCHIVE_DIR/data/"
else
    echo "  ⚠️  跳过（不存在）: data/energy_research/stratified/group1_examples"
fi

# 归档: data/energy_research/stratified/group3_person_reid (3天)
if [ -e "$BASE_DIR/data/energy_research/stratified/group3_person_reid" ]; then
    echo "  归档: data/energy_research/stratified/group3_person_reid"
    mv "$BASE_DIR/data/energy_research/stratified/group3_person_reid" "$ARCHIVE_DIR/data/"
else
    echo "  ⚠️  跳过（不存在）: data/energy_research/stratified/group3_person_reid"
fi

# 归档: results/energy_research/interaction_tradeoff_verification (7天)
if [ -e "$BASE_DIR/results/energy_research/interaction_tradeoff_verification" ]; then
    echo "  归档: results/energy_research/interaction_tradeoff_verification"
    mv "$BASE_DIR/results/energy_research/interaction_tradeoff_verification" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/interaction_tradeoff_verification"
fi

# 归档: results/energy_research/reports (6天)
if [ -e "$BASE_DIR/results/energy_research/reports" ]; then
    echo "  归档: results/energy_research/reports"
    mv "$BASE_DIR/results/energy_research/reports" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/reports"
fi

# 归档: results/energy_research/stratified (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified" ]; then
    echo "  归档: results/energy_research/stratified"
    mv "$BASE_DIR/results/energy_research/stratified" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified"
fi

# 归档: results/energy_research/stratified/ate (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/ate" ]; then
    echo "  归档: results/energy_research/stratified/ate"
    mv "$BASE_DIR/results/energy_research/stratified/ate" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/ate"
fi

# 归档: results/energy_research/stratified/ate/group1_non_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/ate/group1_non_parallel" ]; then
    echo "  归档: results/energy_research/stratified/ate/group1_non_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/ate/group1_non_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/ate/group1_non_parallel"
fi

# 归档: results/energy_research/stratified/ate/group1_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/ate/group1_parallel" ]; then
    echo "  归档: results/energy_research/stratified/ate/group1_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/ate/group1_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/ate/group1_parallel"
fi

# 归档: results/energy_research/stratified/ate/group3_non_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/ate/group3_non_parallel" ]; then
    echo "  归档: results/energy_research/stratified/ate/group3_non_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/ate/group3_non_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/ate/group3_non_parallel"
fi

# 归档: results/energy_research/stratified/ate/group3_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/ate/group3_parallel" ]; then
    echo "  归档: results/energy_research/stratified/ate/group3_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/ate/group3_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/ate/group3_parallel"
fi

# 归档: results/energy_research/stratified/benchmark (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/benchmark" ]; then
    echo "  归档: results/energy_research/stratified/benchmark"
    mv "$BASE_DIR/results/energy_research/stratified/benchmark" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/benchmark"
fi

# 归档: results/energy_research/stratified/dibs (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/dibs" ]; then
    echo "  归档: results/energy_research/stratified/dibs"
    mv "$BASE_DIR/results/energy_research/stratified/dibs" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/dibs"
fi

# 归档: results/energy_research/stratified/dibs/group1_non_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/dibs/group1_non_parallel" ]; then
    echo "  归档: results/energy_research/stratified/dibs/group1_non_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/dibs/group1_non_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/dibs/group1_non_parallel"
fi

# 归档: results/energy_research/stratified/dibs/group1_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/dibs/group1_parallel" ]; then
    echo "  归档: results/energy_research/stratified/dibs/group1_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/dibs/group1_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/dibs/group1_parallel"
fi

# 归档: results/energy_research/stratified/dibs/group3_non_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/dibs/group3_non_parallel" ]; then
    echo "  归档: results/energy_research/stratified/dibs/group3_non_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/dibs/group3_non_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/dibs/group3_non_parallel"
fi

# 归档: results/energy_research/stratified/dibs/group3_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/dibs/group3_parallel" ]; then
    echo "  归档: results/energy_research/stratified/dibs/group3_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/dibs/group3_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/dibs/group3_parallel"
fi

# 归档: results/energy_research/stratified/tradeoff (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/tradeoff" ]; then
    echo "  归档: results/energy_research/stratified/tradeoff"
    mv "$BASE_DIR/results/energy_research/stratified/tradeoff" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/tradeoff"
fi

# 归档: results/energy_research/stratified/tradeoff/group1_non_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/tradeoff/group1_non_parallel" ]; then
    echo "  归档: results/energy_research/stratified/tradeoff/group1_non_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/tradeoff/group1_non_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/tradeoff/group1_non_parallel"
fi

# 归档: results/energy_research/stratified/tradeoff/group1_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/tradeoff/group1_parallel" ]; then
    echo "  归档: results/energy_research/stratified/tradeoff/group1_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/tradeoff/group1_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/tradeoff/group1_parallel"
fi

# 归档: results/energy_research/stratified/tradeoff/group3_non_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/tradeoff/group3_non_parallel" ]; then
    echo "  归档: results/energy_research/stratified/tradeoff/group3_non_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/tradeoff/group3_non_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/tradeoff/group3_non_parallel"
fi

# 归档: results/energy_research/stratified/tradeoff/group3_parallel (3天)
if [ -e "$BASE_DIR/results/energy_research/stratified/tradeoff/group3_parallel" ]; then
    echo "  归档: results/energy_research/stratified/tradeoff/group3_parallel"
    mv "$BASE_DIR/results/energy_research/stratified/tradeoff/group3_parallel" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/stratified/tradeoff/group3_parallel"
fi

# 归档: results/energy_research/tradeoff_detection_interaction_based (8天)
if [ -e "$BASE_DIR/results/energy_research/tradeoff_detection_interaction_based" ]; then
    echo "  归档: results/energy_research/tradeoff_detection_interaction_based"
    mv "$BASE_DIR/results/energy_research/tradeoff_detection_interaction_based" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/tradeoff_detection_interaction_based"
fi

# 归档: results/energy_research/tradeoff_detection_interaction_based/figures (8天)
if [ -e "$BASE_DIR/results/energy_research/tradeoff_detection_interaction_based/figures" ]; then
    echo "  归档: results/energy_research/tradeoff_detection_interaction_based/figures"
    mv "$BASE_DIR/results/energy_research/tradeoff_detection_interaction_based/figures" "$ARCHIVE_DIR/results/"
else
    echo "  ⚠️  跳过（不存在）: results/energy_research/tradeoff_detection_interaction_based/figures"
fi

echo ""
echo "=========================================="
echo "✅ 归档完成"
echo "=========================================="
echo "归档位置: $ARCHIVE_DIR"
echo ""
echo "归档内容:"
echo "  data/: $(ls $ARCHIVE_DIR/data/ 2>/dev/null | wc -l) 个目录"
echo "  results/: $(ls $ARCHIVE_DIR/results/ 2>/dev/null | wc -l) 个目录"
