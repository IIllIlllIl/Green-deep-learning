#!/bin/bash
# 快速训练脚本
# 用于一键启动DNN+rVSM模型训练

echo "======================================"
echo "Bug Localization DNN+rVSM 训练脚本"
echo "======================================"
echo ""

# 激活conda环境
echo "正在激活conda环境 'dnn_rvsm'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dnn_rvsm

if [ $? -ne 0 ]; then
    echo "错误: 无法激活conda环境 'dnn_rvsm'"
    echo "请先运行环境验证脚本: ./scripts/verify_environment.sh"
    exit 1
fi

echo "环境已激活"
echo ""

# 进入源代码目录
cd /home/green/energy_dl/test/bug-localization-by-dnn-and-rvsm/src

# 创建输出目录
mkdir -p ../output

# 获取开始时间
START_TIME=$(date +%s)
echo "训练开始时间: $(date)"
echo "======================================"
echo ""

# 运行训练并保存输出
echo "开始训练模型..."
echo "注意: 10折交叉验证将并行执行，预计耗时10-20分钟"
echo ""

python main.py 2>&1 | tee ../output/training_log_$(date +%Y%m%d_%H%M%S).txt

# 获取结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "======================================"
echo "训练完成！"
echo "训练时间: ${MINUTES}分${SECONDS}秒"
echo "结果已保存到: output/training_log_*.txt"
echo "======================================"
