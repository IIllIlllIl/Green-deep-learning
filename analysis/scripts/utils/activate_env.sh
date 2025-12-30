#!/bin/bash
# Fairness Trade-off Analysis 项目环境激活脚本
# 使用方法: source activate_env.sh

echo "========================================"
echo "激活 fairness 环境..."
echo "========================================"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairness

# 验证环境
echo ""
echo "当前Python版本:"
python --version

echo ""
echo "环境路径:"
which python

echo ""
echo "✅ 环境激活成功！"
echo ""
echo "可用命令："
echo "  - python demo_quick_run.py    # 运行快速演示"
echo "  - python run_tests.py         # 运行测试"
echo ""
