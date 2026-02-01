#!/bin/bash
# 串行运行6组DiBS因果发现（全局标准化数据）
# 使用方法: bash scripts/run_6groups_dibs_serial_global_std.sh

set -e  # 出错时退出

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh

# 组列表（按顺序）
GROUPS=(
    "group1_examples"
    "group2_vulberta"
    "group3_person_reid"
    "group4_bug_localization"
    "group5_mrt_oast"
    "group6_resnet"
)

# 输出目录
OUTPUT_DIR="results/energy_research/data/global_std"
LOG_DIR="logs"

# 创建目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "DiBS串行运行（全局标准化数据）"
echo "开始时间: $(date)"
echo "================================================================"

# 遍历所有组
for GROUP in "${GROUPS[@]}"; do
    LOG_FILE="$LOG_DIR/dibs_${GROUP}_global_std.log"

    echo ""
    echo "========================================================================"
    echo "运行组: $GROUP"
    echo "日志文件: $LOG_FILE"
    echo "开始时间: $(date)"
    echo "========================================================================"

    # 运行DiBS
    set +e  # 允许错误继续检查
    conda run -n causal-research python scripts/run_dibs_6groups_global_std.py \
        --group "$GROUP" \
        --n-steps 5000 \
        --verbose 2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ 组 $GROUP 完成 (退出码: $EXIT_CODE)"
        echo "完成时间: $(date)"

        # 检查是否生成摘要文件
        SUMMARY_FILE="$OUTPUT_DIR/$GROUP/${GROUP}_dibs_summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            echo "摘要文件: $SUMMARY_FILE"
            # 显示强边比例
            STRONG_PERCENT=$(grep -o '"strong_edge_percentage": [0-9.]*' "$SUMMARY_FILE" | cut -d' ' -f2)
            echo "强边比例: ${STRONG_PERCENT}%"
        fi
    else
        echo "❌ 组 $GROUP 失败 (退出码: $EXIT_CODE)"
        echo "请检查日志文件: $LOG_FILE"
        echo "⚠️ 后续组将停止执行"
        exit 1
    fi

    # 每组完成后等待5秒，让系统清理
    sleep 5
done

echo ""
echo "================================================================"
echo "所有组完成!"
echo "结束时间: $(date)"
echo "结果目录: $OUTPUT_DIR"
echo "================================================================"