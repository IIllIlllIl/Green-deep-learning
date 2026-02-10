#!/bin/bash
# 归档操作脚本 - 2026-02-10
# 策略：黑名单（仅归档明确被替代的文件）

ARCHIVE_DIR="archive/archive_20260210"
LOG_FILE="archive/archive_log_20260210.md"

# 创建归档目录
mkdir -p "$ARCHIVE_DIR"/{data,results,scripts}

# 记录函数
log_operation() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 归档函数（带日志）
archive_item() {
    local src="$1"
    local dest="$2"
    
    if [ -e "$src" ]; then
        # 记录原始位置
        echo "$src -> $dest" >> "$ARCHIVE_DIR/manifest.txt"
        # 执行移动
        mv "$src" "$dest"
        log_operation "✅ 归档: $src -> $dest"
        echo "✅ 归档: $src"
    else
        log_operation "⚠️ 跳过: $src (不存在)"
        echo "⚠️ 跳过: $src (不存在)"
    fi
}

echo "开始归档操作..."
echo "========================================" >> "$LOG_FILE"

# 1. 数据目录归档
echo "=== 归档数据目录 ==="
archive_item "data/energy_research/6groups_final" "$ARCHIVE_DIR/data/"
archive_item "data/energy_research/6groups_interaction" "$ARCHIVE_DIR/data/"
archive_item "data/energy_research/6groups_dibs_ready_v1_backup" "$ARCHIVE_DIR/data/"

# 2. 结果目录归档
echo "=== 归档结果目录 ==="
archive_item "results/energy_research/archived_data" "$ARCHIVE_DIR/results/"
archive_item "results/energy_research/interaction_tradeoff_verification" "$ARCHIVE_DIR/results/"
archive_item "results/energy_research/tradeoff_detection_interaction_based" "$ARCHIVE_DIR/results/"

# 3. 脚本归档
echo "=== 归档脚本 ==="
# DiBS旧版
archive_item "scripts/run_dibs_inference_6groups.py" "$ARCHIVE_DIR/scripts/"
archive_item "scripts/run_dibs_inference.py" "$ARCHIVE_DIR/scripts/"
archive_item "scripts/validate_dibs_results_questions_2_3.py" "$ARCHIVE_DIR/scripts/"
archive_item "scripts/generate_6groups_dibs_data.py" "$ARCHIVE_DIR/scripts/"

# ATE旧版
archive_item "scripts/compute_ate_dibs.py" "$ARCHIVE_DIR/scripts/"
archive_item "scripts/compute_ate_questions_2_3.py" "$ARCHIVE_DIR/scripts/"

# 权衡检测旧版
archive_item "scripts/run_algorithm1_tradeoff_detection.py" "$ARCHIVE_DIR/scripts/"
archive_item "scripts/run_algorithm1_tradeoff_detection_interaction.py" "$ARCHIVE_DIR/scripts/"

# 数据生成旧版
archive_item "scripts/generate_6groups_data.py" "$ARCHIVE_DIR/scripts/"
archive_item "scripts/generate_interaction_data.py" "$ARCHIVE_DIR/scripts/"

echo "========================================"
echo "归档完成！"
echo "日志文件: $LOG_FILE"
echo "归档清单: $ARCHIVE_DIR/manifest.txt"
