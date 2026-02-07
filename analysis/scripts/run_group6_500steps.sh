#!/bin/bash
# 运行group6的DiBS分析，500步

LOG_FILE="logs/dibs/dibs_group6_500steps_$(date +%Y%m%d_%H%M%S).log"
echo "启动group6 DiBS分析（500步），日志文件: $LOG_FILE"
echo "开始时间: $(date)"

# 创建日志目录
mkdir -p logs/dibs

# 激活conda环境并运行DiBS，使用无缓冲模式
source /home/green/miniconda3/etc/profile.d/conda.sh
conda activate causal-research

# 运行DiBS仅针对group6，500步
stdbuf -o0 -e0 python -u scripts/run_dibs_6groups_global_std.py --group group6_resnet --n-steps 500 --verbose 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "结束时间: $(date)"
echo "退出代码: $EXIT_CODE"
exit $EXIT_CODE
