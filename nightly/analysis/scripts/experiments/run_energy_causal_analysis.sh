#!/bin/bash
# 能耗研究因果分析 - Screen后台运行脚本
# 基于Adult数据集成功的运行模式

set -e

echo "========================================================================"
echo "  能耗研究因果分析 - 4任务组分层DiBS + DML"
echo "========================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 进入analysis目录
cd "$(dirname "$0")/../.."
echo "工作目录: $(pwd)"

# 创建日志目录
mkdir -p logs/energy_research/experiments
mkdir -p results/energy_research/task_specific

# 日志文件（带时间戳）
LOG_FILE="logs/energy_research/experiments/energy_causal_analysis_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="logs/energy_research/experiments/analysis_status.txt"

echo "日志文件: $LOG_FILE"
echo "状态文件: $STATUS_FILE"
echo ""

# 初始化状态
echo "RUNNING" > $STATUS_FILE
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $STATUS_FILE

# 激活conda环境
echo "激活conda环境: fairness"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairness

# 执行分析（保留完整输出）
echo "========================================================================"
echo "  开始执行DiBS + DML分析..."
echo "  预计时间: 60-120分钟"
echo "  后台监控: tail -f $LOG_FILE"
echo "========================================================================"
echo ""

python3 scripts/demos/demo_energy_task_specific.py 2>&1 | tee -a $LOG_FILE
EXIT_CODE=${PIPESTATUS[0]}

# 更新状态
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS" > $STATUS_FILE
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $STATUS_FILE
    echo "退出码: $EXIT_CODE" >> $STATUS_FILE

    echo ""
    echo "========================================================================"
    echo "  ✅ 分析成功完成！"
    echo "========================================================================"
    echo "查看日志: cat $LOG_FILE"
    echo "查看摘要: cat results/energy_research/task_specific/analysis_summary.txt"
    echo ""
else
    echo "FAILED:$EXIT_CODE" > $STATUS_FILE
    echo "失败时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $STATUS_FILE
    echo "退出码: $EXIT_CODE" >> $STATUS_FILE

    echo ""
    echo "========================================================================"
    echo "  ❌ 分析失败 (退出码: $EXIT_CODE)"
    echo "========================================================================"
    echo "查看日志: cat $LOG_FILE"
    echo "查看最后50行: tail -50 $LOG_FILE"
    echo ""
    exit $EXIT_CODE
fi

# 显示结果文件
echo "生成的文件:"
find results/energy_research/task_specific -type f -name "*.npy" -o -name "*.pkl" -o -name "*.csv" -o -name "*.txt" | sort

echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
