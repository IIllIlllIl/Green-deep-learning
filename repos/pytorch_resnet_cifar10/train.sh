#!/bin/bash

#===================================================================================
# train.sh - PyTorch ResNet CIFAR-10 训练脚本
# 用于在screen会话中自动执行模型训练
#===================================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Conda环境Python路径（不依赖手动激活）
CONDA_ENV_PATH="/home/green/miniconda3/envs/pytorch_resnet_cifar10"
PYTHON="${CONDA_ENV_PATH}/bin/python"

# 检查Python是否存在
if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}错误: 未找到conda环境Python: $PYTHON${NC}"
    echo "请先运行以下命令创建环境："
    echo "  conda create -n pytorch_resnet_cifar10 python=3.10 -y"
    echo "  conda activate pytorch_resnet_cifar10"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

#===================================================================================
# 默认参数（与原始仓库一致）
#===================================================================================
MODEL_NAME="resnet20"
EPOCHS=200
BATCH_SIZE=128
LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.0001
WORKERS=4
PRINT_FREQ=50
SAVE_EVERY=10
SEED=""  # 空字符串表示不设置seed（保持原始随机行为）

#===================================================================================
# 参数解析
#===================================================================================
usage() {
    cat << EOF
用法: $0 [选项]

选项:
    -n, --name MODEL_NAME       模型名称 (默认: resnet20)
                                可选: resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
    -e, --epochs EPOCHS         训练轮数 (默认: 200)
    -b, --batch-size SIZE       批次大小 (默认: 128)
    --lr LEARNING_RATE          初始学习率 (默认: 0.1)
    --momentum MOMENTUM         SGD动量 (默认: 0.9)
    --wd WEIGHT_DECAY           权重衰减 (默认: 0.0001)
    -j, --workers NUM           数据加载线程数 (默认: 4)
    --print-freq NUM            打印频率 (默认: 50)
    --save-every NUM            保存频率(epochs) (默认: 10)
    --seed SEED                 随机种子 (默认: 不设置，保持原始随机行为)
    --half                      使用半精度训练
    -h, --help                  显示此帮助信息

示例:
    # 使用默认参数训练ResNet20
    $0

    # 训练ResNet56，减少epochs
    $0 -n resnet56 -e 100

    # 训练ResNet1202，使用小批次和半精度
    $0 -n resnet1202 -b 32 --half

    # 使用固定种子训练（可重复）
    $0 --seed 42

EOF
}

# 半精度标志
USE_HALF=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --momentum)
            MOMENTUM="$2"
            shift 2
            ;;
        --wd)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        -j|--workers)
            WORKERS="$2"
            shift 2
            ;;
        --print-freq)
            PRINT_FREQ="$2"
            shift 2
            ;;
        --save-every)
            SAVE_EVERY="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --half)
            USE_HALF="--half"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            usage
            exit 1
            ;;
    esac
done

#===================================================================================
# 验证模型名称
#===================================================================================
VALID_MODELS=("resnet20" "resnet32" "resnet44" "resnet56" "resnet110" "resnet1202")
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL_NAME} " ]]; then
    echo -e "${RED}错误: 无效的模型名称: $MODEL_NAME${NC}"
    echo "有效的模型名称: ${VALID_MODELS[@]}"
    exit 1
fi

# ResNet1202特殊处理建议
if [ "$MODEL_NAME" == "resnet1202" ] && [ "$BATCH_SIZE" -gt 64 ] && [ -z "$USE_HALF" ]; then
    echo -e "${YELLOW}警告: ResNet1202可能需要大量显存${NC}"
    echo -e "${YELLOW}建议使用: -b 32 --half${NC}"
    echo -n "继续当前配置? (y/n): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

#===================================================================================
# 设置保存目录
#===================================================================================
SAVE_DIR="save_${MODEL_NAME}"
mkdir -p "$SAVE_DIR"

# 清理权限问题文件
# 检查目录中是否有当前用户无法写入的文件
PERMISSION_ISSUES=false
for file in "${SAVE_DIR}"/*.th; do
    [ -f "$file" ] || continue  # Skip if no .th files exist
    if [ ! -w "$file" ]; then
        echo -e "${YELLOW}警告: 发现无写权限的文件: $file (所有者: $(stat -c '%U' "$file" 2>/dev/null || echo 'unknown'))${NC}"
        # 尝试重命名（备份）
        backup_name="${file}.bak.$(date +%s)"
        if mv "$file" "$backup_name" 2>/dev/null; then
            echo -e "${GREEN}已将文件重命名为: $backup_name${NC}"
        else
            echo -e "${RED}无法移动文件: $file${NC}"
            echo -e "${RED}这可能导致训练失败。请手动删除该文件或使用sudo权限清理。${NC}"
            PERMISSION_ISSUES=true
        fi
    fi
done

if [ "$PERMISSION_ISSUES" = true ]; then
    echo -e "${YELLOW}建议: 运行 'rm -f ${SAVE_DIR}/*.th' 清理旧文件${NC}"
    echo -n "是否继续训练? (y/n): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

#===================================================================================
# 训练前准备
#===================================================================================
echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}PyTorch ResNet CIFAR-10 训练脚本${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo ""
echo -e "${GREEN}训练配置:${NC}"
echo "  模型: $MODEL_NAME"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  初始学习率: $LR"
echo "  动量: $MOMENTUM"
echo "  权重衰减: $WEIGHT_DECAY"
echo "  数据加载线程: $WORKERS"
echo "  随机种子: $([ -n "$SEED" ] && echo "$SEED" || echo '未设置（原始随机行为）')"
echo "  ��精度训练: $([ -n "$USE_HALF" ] && echo '是' || echo '否')"
echo "  保存目录: $SAVE_DIR"
echo ""
echo -e "${GREEN}环境信息:${NC}"
echo "  Python: $PYTHON"
$PYTHON --version
echo ""

# 记录开始时间
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

echo -e "${GREEN}训练开始时间: $START_TIME_STR${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo ""

#===================================================================================
# 执行训练
#===================================================================================
TRAINING_LOG="${SAVE_DIR}/training_output.log"
ERROR_LOG="${SAVE_DIR}/error.log"

# 构建训练命令
TRAIN_CMD="$PYTHON -u trainer.py \
    --arch=$MODEL_NAME \
    --epochs=$EPOCHS \
    --batch-size=$BATCH_SIZE \
    --lr=$LR \
    --momentum=$MOMENTUM \
    --weight-decay=$WEIGHT_DECAY \
    --workers=$WORKERS \
    --print-freq=$PRINT_FREQ \
    --save-every=$SAVE_EVERY \
    --save-dir=$SAVE_DIR \
    $([ -n "$SEED" ] && echo "--seed=$SEED") \
    $USE_HALF"

echo -e "${YELLOW}执行命令:${NC}"
echo "$TRAIN_CMD"
echo ""

# 执行训练并捕获输出
set +e  # 允许命令失败
$TRAIN_CMD 2>&1 | tee "$TRAINING_LOG"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e

# 记录结束时间
END_TIME=$(date +%s)
END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED_TIME=$((END_TIME - START_TIME))

#===================================================================================
# 训练后评估（如果训练成功）
#===================================================================================
EVAL_ACCURACY="N/A"
EVAL_ERROR="N/A"

if [ $TRAIN_EXIT_CODE -eq 0 ] && [ -f "${SAVE_DIR}/model.th" ]; then
    echo ""
    echo -e "${BLUE}=========================================================================${NC}"
    echo -e "${GREEN}开始模型评估...${NC}"
    echo -e "${BLUE}=========================================================================${NC}"
    echo ""

    EVAL_LOG="${SAVE_DIR}/evaluation.log"
    EVAL_CMD="$PYTHON trainer.py \
        --arch=$MODEL_NAME \
        --resume=${SAVE_DIR}/model.th \
        --evaluate \
        $USE_HALF"

    set +e
    $EVAL_CMD 2>&1 | tee "$EVAL_LOG"
    EVAL_EXIT_CODE=$?
    set -e

    # 从评估日志中提取准确率
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        EVAL_ACCURACY=$(grep "Prec@1" "$EVAL_LOG" | tail -1 | grep -oP 'Prec@1 \K[0-9.]+')
        if [ -n "$EVAL_ACCURACY" ]; then
            EVAL_ERROR=$(echo "100 - $EVAL_ACCURACY" | bc -l)
            EVAL_ERROR=$(printf "%.2f" "$EVAL_ERROR")
            EVAL_ACCURACY=$(printf "%.2f" "$EVAL_ACCURACY")
        fi
    fi
fi

#===================================================================================
# 提取训练信息
#===================================================================================
# 检查是否有错误
TRAINING_ERRORS=""
if [ -f "$TRAINING_LOG" ]; then
    # 提取可能的错误信息
    TRAINING_ERRORS=$(grep -i "error\|exception\|traceback\|cuda out of memory" "$TRAINING_LOG" | head -5 || true)
fi

# 提取最终训练精度
FINAL_TRAIN_ACC="N/A"
if [ -f "$TRAINING_LOG" ]; then
    FINAL_TRAIN_ACC=$(grep "Prec@1" "$TRAINING_LOG" | grep -v "Test:" | tail -1 | grep -oP 'Prec@1 \K[0-9.]+' || echo "N/A")
fi

# 提取最佳验证精度
BEST_VAL_ACC="N/A"
if [ -f "$TRAINING_LOG" ]; then
    BEST_VAL_ACC=$(grep "\* Prec@1" "$TRAINING_LOG" | tail -1 | grep -oP 'Prec@1 \K[0-9.]+' || echo "N/A")
fi

#===================================================================================
# 生成训练报告
#===================================================================================
echo ""
echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}                          训练报告                                        ${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo ""
echo -e "${GREEN}模型信息:${NC}"
echo "  模型名称: $MODEL_NAME"
echo "  保存目录: $SAVE_DIR"
echo ""
echo -e "${GREEN}训练配置:${NC}"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  初始学习率: $LR"
echo "  半精度训练: $([ -n "$USE_HALF" ] && echo '是' || echo '否')"
echo ""
echo -e "${GREEN}时间统计:${NC}"
echo "  开始时间: $START_TIME_STR"
echo "  结束时间: $END_TIME_STR"
echo "  总耗时: $((ELAPSED_TIME / 3600))小时 $((ELAPSED_TIME % 3600 / 60))分钟 $((ELAPSED_TIME % 60))秒"
echo ""
echo -e "${GREEN}性能指标:${NC}"
if [ "$EVAL_ACCURACY" != "N/A" ]; then
    echo "  测试准确率: ${EVAL_ACCURACY}%"
    echo "  测试错误率: ${EVAL_ERROR}%"
else
    echo "  测试准确率: 未评估"
fi

if [ "$BEST_VAL_ACC" != "N/A" ]; then
    echo "  最佳验证准确率: ${BEST_VAL_ACC}%"
fi

if [ "$FINAL_TRAIN_ACC" != "N/A" ]; then
    echo "  最终训练准确率: ${FINAL_TRAIN_ACC}%"
fi
echo ""

echo -e "${GREEN}训练状态:${NC}"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "  ${GREEN}✓ 训练成功完成${NC}"
else
    echo -e "  ${RED}✗ 训练失败 (退出码: $TRAIN_EXIT_CODE)${NC}"
fi
echo ""

if [ -n "$TRAINING_ERRORS" ]; then
    echo -e "${RED}训练过程中的错误:${NC}"
    echo "$TRAINING_ERRORS"
    echo ""
    echo "完整日志: $TRAINING_LOG"
    echo ""
fi

echo -e "${GREEN}输出文件:${NC}"
echo "  训练日志: $TRAINING_LOG"
if [ -f "$EVAL_LOG" ]; then
    echo "  评估日志: $EVAL_LOG"
fi
if [ -f "${SAVE_DIR}/model.th" ]; then
    echo "  最佳模型: ${SAVE_DIR}/model.th"
fi
if [ -f "${SAVE_DIR}/checkpoint.th" ]; then
    echo "  检查点: ${SAVE_DIR}/checkpoint.th"
fi
echo ""

echo -e "${BLUE}=========================================================================${NC}"

# 返回训练退出码
exit $TRAIN_EXIT_CODE
