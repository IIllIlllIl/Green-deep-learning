#!/bin/bash
# 改进的DiBS运行脚本，确保输出被正确捕获

LOG_FILE="logs/dibs/dibs_full_run_improved_$(date +%Y%m%d_%H%M%S).log"
echo "启动DiBS分析，日志文件: $LOG_FILE"
echo "开始时间: $(date)"

# 创建日志目录
mkdir -p logs/dibs

# 激活conda环境并运行DiBS，使用无缓冲模式
source /home/green/miniconda3/etc/profile.d/conda.sh
conda activate causal-research

# 运行DiBS，使用stdbuf确保无缓冲输出，同时使用tee输出到文件和终端
stdbuf -o0 -e0 python -u scripts/run_dibs_6groups_global_std.py --n-steps 5000 --verbose 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "结束时间: $(date)"
echo "退出代码: $EXIT_CODE"
exit $EXIT_CODE
