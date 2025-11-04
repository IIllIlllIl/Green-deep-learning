#!/bin/bash

################################################################################
# MRT-OAST 模型评估脚本
#
# 功能：评估已训练的模型并生成详细的性能报告
#
# 使用示例：
#   ./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_GCJ_OAST
#   ./evaluate_model.sh model/BTransfrom_MRT_len256_batch64_OJClone_OAST --threshold 0.85
#
################################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 显示使用说明
show_usage() {
    cat << EOF
用法: $0 MODEL_PATH [选项]

参数:
    MODEL_PATH              模型目录路径 (必需)

选项:
    --dataset DATASET       数据集名称 (OJClone|GCJ|BCB，自动检测)
    --ast-type TYPE         AST类型 (AST|OAST，默认: OAST)
    --max-len N             最大序列长度 (默认: 256)
    --threshold FLOAT       测试阈值 (默认: 0.9)
    --no-cuda               不使用GPU
    --detailed              使用详细评估模式（更慢但更详细）
    --output FILE           输出报告文件路径
    -h, --help              显示此帮助信息

示例:
    # 评估GCJ模型
    $0 model/BTransfrom_MRT_len256_batch64_GCJ_OAST

    # 使用自定义阈值
    $0 model/BTransfrom_MRT_len256_batch64_OJClone_OAST --threshold 0.85

    # 详细评估模式
    $0 model/BTransfrom_MRT_len256_batch64_BCB_OAST --detailed

EOF
}

# 检查是否请求帮助
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# 检查参数
if [ $# -lt 1 ]; then
    print_error "错误：缺少模型路径参数"
    show_usage
    exit 1
fi

# 默认参数
MODEL_PATH="$1"
shift

DATASET=""
AST_TYPE="OAST"
MAX_LEN=256
THRESHOLD=0.9
USE_CUDA="--cuda"
DETAILED_MODE=""
OUTPUT_FILE=""

# 解析选项
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
        --max-len)
            MAX_LEN="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --no-cuda)
            USE_CUDA=""
            shift
            ;;
        --detailed)
            DETAILED_MODE="--no-quick"
            shift
            ;;
        --output)
            OUTPUT_FILE="$2"
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

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    print_error "模型目录不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH/model.pt" ]; then
    print_error "模型文件不存在: $MODEL_PATH/model.pt"
    exit 1
fi

# 从路径自动检测数据集
if [ -z "$DATASET" ]; then
    if [[ "$MODEL_PATH" == *"OJClone"* ]]; then
        DATASET="OJClone"
    elif [[ "$MODEL_PATH" == *"GCJ"* ]]; then
        DATASET="GCJ"
    elif [[ "$MODEL_PATH" == *"BCB"* ]]; then
        DATASET="BCB"
    else
        print_error "无法从路径自动检测数据集，请使用 --dataset 参数指定"
        exit 1
    fi
fi

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

# 检查文件
for file in "$DATA_FILE" "$TEST_PAIR" "$DICTIONARY"; do
    if [ ! -f "$file" ]; then
        print_error "文件不存在: $file"
        exit 1
    fi
done

# 设置输出文件
if [ -z "$OUTPUT_FILE" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_FILE="logs/evaluation_${DATASET}_${TIMESTAMP}.txt"
    mkdir -p logs
fi

# 打印评估信息
print_info "====== 模型评估 ======"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET"
echo "AST类型: $AST_TYPE"
echo "最大序列长度: $MAX_LEN"
echo "阈值: $THRESHOLD"
echo "GPU: $([ -n "$USE_CUDA" ] && echo "启用" || echo "禁用")"
echo "评估模式: $([ -n "$DETAILED_MODE" ] && echo "详细" || echo "快速")"
echo "输出文件: $OUTPUT_FILE"
echo "======================"
echo ""

# 激活conda环境
print_info "激活conda环境..."
eval "$(conda shell.bash hook)"
conda activate mrt-oast

# 构建评估命令
if [ -n "$DETAILED_MODE" ]; then
    # 详细评估模式
    EVAL_CMD="python main_batch.py \
        --data $DATA_FILE \
        --train_pair $TRAIN_PAIR \
        --valid_pair $VALID_PAIR \
        --test_pair $TEST_PAIR \
        --dictionary $DICTIONARY \
        --ast_type $AST_TYPE \
        --save $MODEL_PATH \
        --sen_max_len $MAX_LEN \
        --threshold $THRESHOLD \
        --is_test \
        $USE_CUDA"
else
    # 快速评估模式
    EVAL_CMD="python main_batch.py \
        --data $DATA_FILE \
        --train_pair $TRAIN_PAIR \
        --valid_pair $VALID_PAIR \
        --test_pair $TEST_PAIR \
        --dictionary $DICTIONARY \
        --ast_type $AST_TYPE \
        --save $MODEL_PATH \
        --sen_max_len $MAX_LEN \
        --threshold $THRESHOLD \
        --is_test \
        --quick_test \
        $USE_CUDA"
fi

# 执行评估
print_info "开始评估..."
EVAL_OUTPUT=$(eval "$EVAL_CMD" 2>&1 | tee /dev/tty)

# 提取性能指标
print_info "提取性能指标..."

# 匹配格式: "Accuracy: 4676/5000 = 0.9352" 或 "Accuracy on test is 4505/4992 = 0.902444"
ACC=$(echo "$EVAL_OUTPUT" | grep -oP "Accuracy.*?=\s*\K[0-9.]+" | tail -1)
# 匹配格式: "Precision:0.9810" 或 "Precision: 0.982983"
PRECISION=$(echo "$EVAL_OUTPUT" | grep -oP "Precision:\s*\K[0-9.]+" | tail -1)
# 匹配格式: "Recall:0.8879" 或 "Recall   : 0.812578"
RECALL=$(echo "$EVAL_OUTPUT" | grep -oP "Recall\s*:\s*\K[0-9.]+" | tail -1)
# 匹配格式: "F1 score:0.9322" 或 "F1 score : 0.889694"
F1=$(echo "$EVAL_OUTPUT" | grep -oP "F1\s+score\s*:\s*\K[0-9.]+" | tail -1)

# 生成评估报告
{
    echo "======================================"
    echo "MRT-OAST 模型评估报告"
    echo "======================================"
    echo ""
    echo "评估时间: $(date)"
    echo ""
    echo "模型信息:"
    echo "  模型路径: $MODEL_PATH"
    echo "  数据集: $DATASET"
    echo "  AST类型: $AST_TYPE"
    echo ""
    echo "评估配置:"
    echo "  最大序列长度: $MAX_LEN"
    echo "  阈值: $THRESHOLD"
    echo "  评估模式: $([ -n "$DETAILED_MODE" ] && echo "详细" || echo "快速")"
    echo ""
    echo "性能指标:"
    [ -n "$ACC" ] && echo "  准确率 (Accuracy): $ACC" || echo "  准确率 (Accuracy): N/A"
    [ -n "$PRECISION" ] && echo "  精确率 (Precision): $PRECISION" || echo "  精确率 (Precision): N/A"
    [ -n "$RECALL" ] && echo "  召回率 (Recall): $RECALL" || echo "  召回率 (Recall): N/A"
    [ -n "$F1" ] && echo "  F1分数 (F1-Score): $F1" || echo "  F1分数 (F1-Score): N/A"
    echo ""
    echo "======================================"
} | tee "$OUTPUT_FILE"

echo ""
print_success "评估完成！"
print_success "评估报告已保存到: $OUTPUT_FILE"

# 显示性能总结
if [ -n "$F1" ]; then
    print_success "F1分数: $F1"
fi
