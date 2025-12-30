#!/bin/bash
# 监控Adult数据集完整分析的进度

LOG_FILE=$(ls -t adult_full_analysis_*.log 2>/dev/null | head -1)
STATUS_FILE="adult_analysis_status.txt"

echo "========================================"
echo "Adult数据集完整因果分析 - 进度监控"
echo "========================================"
echo ""

# 检查进程状态
PID=$(ps aux | grep "[d]emo_adult_full_analysis.py" | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✓ 进程状态: 运行中 (PID: $PID)"
    ps aux | grep "[d]emo_adult_full_analysis.py" | awk '{printf "  CPU: %s%% | 内存: %s%% | 运行时间: %s\n", $3, $4, $10}'
else
    echo "✗ 进程状态: 未运行"
fi

echo ""

# 检查GPU使用
GPU_INFO=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep "$PID")
if [ -n "$GPU_INFO" ]; then
    echo "✓ GPU使用: $(echo $GPU_INFO | awk -F, '{print $2}')"
else
    echo "○ GPU使用: 未使用"
fi

echo ""

# 检查状态文件
if [ -f "$STATUS_FILE" ]; then
    STATUS=$(cat $STATUS_FILE)
    echo "状态标记: $STATUS"
fi

echo ""

# 检查生成的文件
echo "生成的文件:"
if [ -f "results/adult_data_checkpoint.pkl" ]; then
    SIZE=$(ls -lh results/adult_data_checkpoint.pkl | awk '{print $5}')
    echo "  ✓ 数据检查点: $SIZE"
fi

if [ -f "data/adult_training_data.csv" ]; then
    ROWS=$(wc -l < data/adult_training_data.csv)
    SIZE=$(ls -lh data/adult_training_data.csv | awk '{print $5}')
    echo "  ✓ 训练数据: $ROWS 行, $SIZE"
else
    echo "  ○ 训练数据: 收集中..."
fi

if [ -f "results/adult_causal_graph.npy" ]; then
    SIZE=$(ls -lh results/adult_causal_graph.npy | awk '{print $5}')
    echo "  ✓ 因果图: $SIZE"
else
    echo "  ○ 因果图: 待生成"
fi

if [ -f "results/adult_causal_effects.csv" ]; then
    ROWS=$(wc -l < results/adult_causal_effects.csv)
    SIZE=$(ls -lh results/adult_causal_effects.csv | awk '{print $5}')
    echo "  ✓ 因果效应: $ROWS 行, $SIZE"
else
    echo "  ○ 因果效应: 待生成"
fi

echo ""

# 显示最新日志
if [ -f "$LOG_FILE" ]; then
    LINES=$(wc -l < "$LOG_FILE")
    echo "日志文件: $LOG_FILE ($LINES 行)"
    echo ""
    echo "最新日志 (最后20行):"
    echo "----------------------------------------"
    tail -20 "$LOG_FILE"
else
    echo "未找到日志文件"
fi

echo ""
echo "========================================"
echo "实时监控命令:"
echo "  watch -n 10 bash monitor_progress.sh"
echo "  tail -f $LOG_FILE"
echo "========================================"
