#!/bin/bash
#
# DiBS步数验证测试运行脚本
#
# 用法:
#   ./run_dibs_step_test.sh [STEPS] [PARTICLES]
#
# 示例:
#   ./run_dibs_step_test.sh           # 默认1000步，20粒子
#   ./run_dibs_step_test.sh 500       # 500步，20粒子
#   ./run_dibs_step_test.sh 1000 50   # 1000步，50粒子
#

set -e  # 遇到错误立即退出

# 参数
STEPS=${1:-1000}
PARTICLES=${2:-20}
CALLBACK_EVERY=${3:-10}

echo "========================================"
echo "DiBS步数验证测试"
echo "========================================"
echo ""
echo "测试参数:"
echo "  训练步数: $STEPS"
echo "  粒子数: $PARTICLES"
echo "  Callback间隔: $CALLBACK_EVERY"
echo "  测试组: group5_mrt_oast (60样本，最小)"
echo ""

# 激活conda环境
echo "激活conda环境: causal-research"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate causal-research

# 切换到分析目录
cd /home/green/energy_dl/nightly/analysis

# 运行测试
echo ""
echo "开始测试..."
echo ""

python3 tests/test_dibs_step_verification.py \
    --steps $STEPS \
    --particles $PARTICLES \
    --callback-every $CALLBACK_EVERY

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试成功完成！"
else
    echo "❌ 测试失败（退出码: $EXIT_CODE）"
fi

exit $EXIT_CODE
