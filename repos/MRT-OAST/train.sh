#!/bin/bash

################################################################################
# MRT-OAST 训练与评估脚本
#
# 功能：
# 1. 训练模型并自动评估
# 2. 支持命令行参数配置所有超参数
# 3. 自动记录训练日志
# 4. 训练完成后报告性能指标
#
# 使用示例：
#   ./train.sh --dataset OJClone --epochs 10 --batch-size 64 --lr 0.0001
#   ./train.sh --dataset GCJ --epochs 5 --quick
#
################################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 显示使用说明
show_usage() {
    cat << EOF
用法: $0 [选项]

数据集选项:
    --dataset DATASET       数据集名称 (OJClone|GCJ|BCB，默认: OJClone)
    --ast-type TYPE         AST类型 (AST|OAST，默认: OAST)

训练参数:
    --epochs N              训练轮数 (默认: 10)
    --batch-size N          批次大小 (默认: 64)
    --lr RATE               学习率 (默认: 0.0001)
    --dropout RATE          Dropout率 (默认: 0.2)
    --weight-decay DECAY    权重衰减/L2正则化 (默认: 0.0)
    --seed N                随机种子 (默认: 1334)
    --valid-step N          验证步数 (默认: 1750, 0表示每epoch验证)

模型参数:
    --max-len N             最大序列长度 (默认: 256)
    --layers N              Transformer层数 (默认: 2)
    --d-model N             模型维度 (默认: 128)
    --d-ff N                前馈网络维度 (默认: 512)
    --heads N               注意力头数 (默认: 8)
    --output-dim N          输出维度 (默认: 512)

评估参数:
    --threshold FLOAT       测试阈值 (默认: 0.9)
    --valid-threshold FLOAT 验证阈值 (默认: 0.8)

其他选项:
    --quick                 快速模式（减少训练量用于测试）
    --no-cuda               不使用GPU
    --tag TAG               模型标签（用于保存路径）
    --save-dir DIR          模型保存目录
    --log-dir DIR           日志保存目录 (默认: logs/)
    -h, --help              显示此帮助信息

示例:
    # 使用默认参数训练OJClone数据集
    $0

    # 训练OJClone数据集，5个epoch，快速模式
    $0 --dataset OJClone --epochs 5 --quick

    # 自定义所有参数
    $0 --dataset BCB --epochs 20 --batch-size 32 --lr 0.0005 --layers 4

    # 小模型快速测试
    $0 --dataset OJClone --epochs 2 --batch-size 16 --max-len 128 --quick

EOF
}

# 默认参数
DATASET="OJClone"
AST_TYPE="OAST"
EPOCHS=10
BATCH_SIZE=64
LEARNING_RATE=0.0001
DROPOUT=0.2
WEIGHT_DECAY=0.0
SEED=1334
VALID_STEP=1750
MAX_LEN=256
LAYERS=2
D_MODEL=128
D_FF=512
HEADS=8
OUTPUT_DIM=512
THRESHOLD=0.9
VALID_THRESHOLD=0.8
USE_CUDA="--cuda"
QUICK_MODE=0
TAG=""
SAVE_DIR=""
LOG_DIR="logs"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --ast-type)
            AST_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --valid-step)
            VALID_STEP="$2"
            shift 2
            ;;
        --max-len)
            MAX_LEN="$2"
            shift 2
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --d-model)
            D_MODEL="$2"
            shift 2
            ;;
        --d-ff)
            D_FF="$2"
            shift 2
            ;;
        --heads)
            HEADS="$2"
            shift 2
            ;;
        --output-dim)
            OUTPUT_DIM="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --valid-threshold)
            VALID_THRESHOLD="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=1
            shift
            ;;
        --no-cuda)
            USE_CUDA=""
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 快速模式调整
if [ $QUICK_MODE -eq 1 ]; then
    print_warning "快速模式已启用（减少训练量用于测试）"
    if [ $EPOCHS -gt 5 ]; then
        EPOCHS=3
    fi
    if [ "$DATASET" = "BCB" ]; then
        DATASET="OJClone"
        print_warning "快速模式：将BCB数据集改为OJClone"
    fi
fi

# 验证数据集参数
case $DATASET in
    OJClone|GCJ|BCB)
        ;;
    *)
        print_error "无效的数据集: $DATASET (支持: OJClone, GCJ, BCB)"
        exit 1
        ;;
esac

# 设置数据集相关路径
DATA_FILE="origindata/${DATASET}_with_AST+OAST.csv"
case $DATASET in
    OJClone)
        TRAIN_PAIR="origindata/OJClone_train.csv"
        VALID_PAIR="origindata/OJClone_valid.csv"
        TEST_PAIR="origindata/OJClone_test.csv"
        ;;
    GCJ)
        TRAIN_PAIR="origindata/GCJ_train11.csv"
        VALID_PAIR="origindata/GCJ_valid.csv"
        TEST_PAIR="origindata/GCJ_test.csv"
        ;;
    BCB)
        TRAIN_PAIR="origindata/BCB_train.csv"
        VALID_PAIR="origindata/BCB_valid.csv"
        TEST_PAIR="origindata/BCB_test.csv"
        ;;
esac

DICTIONARY="origindata/${DATASET}_${AST_TYPE}_dictionary.txt"

# 检查文件是否存在
for file in "$DATA_FILE" "$TRAIN_PAIR" "$VALID_PAIR" "$TEST_PAIR" "$DICTIONARY"; do
    if [ ! -f "$file" ]; then
        print_error "文件不存在: $file"
        exit 1
    fi
done

# 设置标签和保存路径
if [ -z "$TAG" ]; then
    TAG="${DATASET}_${AST_TYPE}"
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="model/BTransfrom_MRT_len${MAX_LEN}_batch${BATCH_SIZE}_${TAG}"
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${DATASET}_${TIMESTAMP}.log"
METRICS_FILE="${LOG_DIR}/metrics_${DATASET}_${TIMESTAMP}.txt"

# 打印配置信息
print_info "====== 训练配置 ======"
echo "数据集: $DATASET"
echo "AST类型: $AST_TYPE"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "最大序列长度: $MAX_LEN"
echo "Transformer层数: $LAYERS"
echo "模型维度: $D_MODEL"
echo "前馈维度: $D_FF"
echo "注意力头数: $HEADS"
echo "Dropout: $DROPOUT"
echo "权重衰减: $WEIGHT_DECAY"
echo "随机种子: $SEED"
echo "GPU: $([ -n "$USE_CUDA" ] && echo "启用" || echo "禁用")"
echo "模型保存路径: $SAVE_DIR"
echo "日志文件: $LOG_FILE"
echo "======================"

# 检查conda环境（在sudo下跳过，因为conda命令不可用）
# if ! conda info --envs | grep -q "mrt-oast"; then
#     print_error "conda环境 'mrt-oast' 不存在，请先运行 'conda env create -f environment.yml'"
#     exit 1
# fi

# 激活conda环境并训练
print_info "开始训练..."
echo ""

# 构建训练命令
TRAIN_CMD="python main_batch.py \
    --data $DATA_FILE \
    --train_pair $TRAIN_PAIR \
    --valid_pair $VALID_PAIR \
    --test_pair $TEST_PAIR \
    --dictionary $DICTIONARY \
    --ast_type $AST_TYPE \
    --tag $TAG \
    --save $SAVE_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --dropout $DROPOUT \
    --weight_decay $WEIGHT_DECAY \
    --seed $SEED \
    --valid_step $VALID_STEP \
    --sen_max_len $MAX_LEN \
    --transformer_nlayers $LAYERS \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --h $HEADS \
    --output_dim $OUTPUT_DIM \
    --valid_threshold $VALID_THRESHOLD \
    $USE_CUDA"

# 记录命令到日志
echo "训练命令: $TRAIN_CMD" > "$LOG_FILE"
echo "开始时间: $(date)" >> "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 执行训练
source /home/green/miniconda3/bin/activate mrt-oast

# 训练并记录日志
if eval "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"; then
    print_success "训练完成！"
else
    print_error "训练失败！"
    exit 1
fi

echo "" >> "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"
echo "结束时间: $(date)" >> "$LOG_FILE"

# 评估模型
print_info "开始评估模型性能..."
echo ""

EVAL_CMD="python main_batch.py \
    --data $DATA_FILE \
    --train_pair $TRAIN_PAIR \
    --valid_pair $VALID_PAIR \
    --test_pair $TEST_PAIR \
    --dictionary $DICTIONARY \
    --ast_type $AST_TYPE \
    --save $SAVE_DIR \
    --sen_max_len $MAX_LEN \
    --threshold $THRESHOLD \
    --is_test \
    --quick_test \
    $USE_CUDA"

# 执行评估并捕获输出
EVAL_OUTPUT=$(eval "$EVAL_CMD" 2>&1 | tee -a "$LOG_FILE")

# 提取性能指标
print_info "正在提取性能指标..."

# 从输出中提取指标（匹配多种输出格式）
# 匹配格式: "Accuracy: 4676/5000 = 0.9352" 或 "Accuracy on test is 4505/4992 = 0.902444"
ACC=$(echo "$EVAL_OUTPUT" | grep -oP "Accuracy.*?=\s*\K[0-9.]+" | tail -1)
# 匹配格式: "Precision:0.9810" 或 "Precision: 0.982983"
PRECISION=$(echo "$EVAL_OUTPUT" | grep -oP "Precision:\s*\K[0-9.]+" | tail -1)
# 匹配格式: "Recall:0.8879" 或 "Recall   : 0.812578"
RECALL=$(echo "$EVAL_OUTPUT" | grep -oP "Recall\s*:\s*\K[0-9.]+" | tail -1)
# 匹配格式: "F1 score:0.9322" 或 "F1 score : 0.889694"
F1=$(echo "$EVAL_OUTPUT" | grep -oP "F1\s+score\s*:\s*\K[0-9.]+" | tail -1)

# 生成性能报告
print_success "====== 性能报告 ======"
{
    echo "======================================"
    echo "MRT-OAST 模型性能报告"
    echo "======================================"
    echo ""
    echo "训练配置:"
    echo "  数据集: $DATASET"
    echo "  AST类型: $AST_TYPE"
    echo "  训练轮数: $EPOCHS"
    echo "  批次大小: $BATCH_SIZE"
    echo "  学习率: $LEARNING_RATE"
    echo "  序列长度: $MAX_LEN"
    echo "  模型层数: $LAYERS"
    echo ""
    echo "性能指标:"
    [ -n "$ACC" ] && echo "  准确率 (Accuracy): $ACC" || echo "  准确率 (Accuracy): N/A"
    [ -n "$PRECISION" ] && echo "  精确率 (Precision): $PRECISION" || echo "  精确率 (Precision): N/A"
    [ -n "$RECALL" ] && echo "  召回率 (Recall): $RECALL" || echo "  召回率 (Recall): N/A"
    [ -n "$F1" ] && echo "  F1分数 (F1-Score): $F1" || echo "  F1分数 (F1-Score): N/A"
    echo ""
    echo "模型位置: $SAVE_DIR/model.pt"
    echo "日志文件: $LOG_FILE"
    echo "完成时间: $(date)"
    echo "======================================"
} | tee "$METRICS_FILE"

print_success "性能报告已保存到: $METRICS_FILE"
print_success "训练日志已保存到: $LOG_FILE"

# 如果有TensorBoard日志
if [ -d "$SAVE_DIR" ] && [ -n "$(find "$SAVE_DIR" -name "log_*" -type d)" ]; then
    print_info "TensorBoard日志位置: $SAVE_DIR/log_*"
    print_info "查看训练过程: tensorboard --logdir $SAVE_DIR"
fi

print_success "全部完成！"
