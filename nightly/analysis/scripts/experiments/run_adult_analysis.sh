#!/bin/bash
# Adult数据集完整因果分析 - 后台运行版本
# 预计运行时间: 2-3小时

# 设置变量
LOG_FILE="adult_full_analysis_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="adult_analysis_status.txt"
RESULT_DIR="results"

# 记录开始时间
echo "========================================" | tee -a $LOG_FILE
echo "Adult数据集完整因果分析" | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "RUNNING" > $STATUS_FILE

# 激活conda环境并运行Python脚本
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairness

# 运行完整分析
echo "正在运行完整因果分析..." | tee -a $LOG_FILE
echo "日志文件: $LOG_FILE" | tee -a $LOG_FILE
echo "状态文件: $STATUS_FILE" | tee -a $LOG_FILE

python demo_adult_full_analysis.py 2>&1 | tee -a $LOG_FILE

# 检查退出状态
EXIT_CODE=$?
echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "分析完成" | tee -a $LOG_FILE
echo "结束时间: $(date)" | tee -a $LOG_FILE
echo "退出代码: $EXIT_CODE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS" > $STATUS_FILE
    echo "✓ 分析成功完成！" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    echo "生成的文件:" | tee -a $LOG_FILE
    ls -lh $RESULT_DIR/adult* 2>/dev/null | tee -a $LOG_FILE
else
    echo "FAILED:$EXIT_CODE" > $STATUS_FILE
    echo "✗ 分析失败，退出代码: $EXIT_CODE" | tee -a $LOG_FILE
fi

# 显示使用说明
echo "" | tee -a $LOG_FILE
echo "查看结果:" | tee -a $LOG_FILE
echo "  tail -f $LOG_FILE          # 实时查看日志" | tee -a $LOG_FILE
echo "  cat $STATUS_FILE           # 查看状态" | tee -a $LOG_FILE
echo "  ls -lh $RESULT_DIR/adult*  # 查看生成文件" | tee -a $LOG_FILE
