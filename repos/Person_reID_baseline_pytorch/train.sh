#!/bin/bash
# Person ReID Baseline - Unified Training Script
# Usage: ./train.sh -n model_name [options] 2>&1 | tee training.log

# 设置错误处理
set -e
trap 'error_handler $? $LINENO' ERR

# 错误处理函数
error_handler() {
    local exit_code=$1
    local line_no=$2
    echo "ERROR: Script failed at line $line_no with exit code $exit_code"
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    print_report "FAILED"
    exit $exit_code
}

# 记录开始时间
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

# 默认参数（与仓库原始默认值一致）
MODEL_NAME=""
DATA_DIR="./Market/Market-1501-v15.09.15/pytorch"
GPU_IDS="0"
BATCHSIZE=32  # 默认32，但对于显存密集型模型会自动调整
LR=0.05
TOTAL_EPOCH=60
TRAIN_ALL="--train_all"
WARM_EPOCH=0
STRIDE=2
ERASING_P=0
DROPRATE=0.5
LINEAR_NUM=512

# 模型特定参数
USE_DENSE=""
USE_SWIN=""
USE_SWINV2=""
USE_HR=""
USE_EFFICIENT=""
USE_CONVNEXT=""
PCB=""
IBM=""
USAM=""
FP16=""
BF16=""
CIRCLE=""
ARCFACE=""
COSFACE=""
TRIPLET=""
LIFTED=""
CONTRAST=""
INSTANCE=""
SPHERE=""
DG=""
COLOR_JITTER=""
COSINE=""

# 训练报错记录
TRAIN_ERROR=""
TEST_ERROR=""
EVAL_ERROR=""

# Conda环境Python路径
CONDA_ENV="reid_baseline"
PYTHON_PATH="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"

# 检查Python路径
if [ ! -f "$PYTHON_PATH" ]; then
    echo "ERROR: Python not found at $PYTHON_PATH"
    echo "Please check conda environment: $CONDA_ENV"
    exit 1
fi

# 打印使用说明
print_usage() {
    cat << EOF
Usage: ./train.sh -n MODEL_NAME [OPTIONS]

Required:
  -n MODEL_NAME          Model configuration name (see available models below)

Optional Training Parameters:
  --data_dir PATH        Data directory (default: $DATA_DIR)
  --gpu_ids IDS          GPU IDs (default: $GPU_IDS)
  --batchsize SIZE       Batch size (default: auto-adjusted per model)
  --lr RATE              Learning rate (default: $LR)
  --total_epoch EPOCH    Total epochs (default: $TOTAL_EPOCH)
  --warm_epoch EPOCH     Warm-up epochs (default: $WARM_EPOCH)
  --stride STRIDE        Stride for ResNet (default: $STRIDE)
  --erasing_p PROB       Random erasing probability (default: $ERASING_P)
  --droprate RATE        Dropout rate (default: $DROPRATE)
  --linear_num NUM       Linear feature dimension (default: $LINEAR_NUM)
  --no_train_all         Do not use all training data (use train/val split)

Optional Flags:
  --fp16                 Use float16 precision
  --bf16                 Use bfloat16 precision
  --circle               Use circle loss
  --triplet              Use triplet loss
  --contrast             Use contrastive loss
  --cosine               Use cosine learning rate
  --dg                   Use DG-Market dataset

Available Model Configurations:
  resnet50                     - ResNet-50 baseline (Rank@1≈88.84%)
  resnet50_fp16                - ResNet-50 with fp16 (Rank@1≈88.03%)
  resnet50_bf16                - ResNet-50 with bf16 (faster, 2GB VRAM)
  resnet50_ibn                 - ResNet-50-ibn (Rank@1≈89.13%)
  resnet50_tricks              - ResNet-50 with all tricks (Rank@1≈92.13%)
  resnet50_circle              - ResNet-50 with circle loss (Rank@1≈92.46%)
  densenet121                  - DenseNet-121 (Rank@1≈90.17%)
  densenet121_circle           - DenseNet-121 with circle loss (Rank@1≈91.00%)
  pcb                          - PCB model (Rank@1≈92.64%)
  hrnet18                      - HRNet-18 (Rank@1≈90.83%)
  swin                         - Swin Transformer (Rank@1≈92.75%, needs 11GB)
  efficientnet                 - EfficientNet-b4 (Rank@1≈85.78%)
  convnext                     - ConvNeXt (Rank@1≈88.98%)
  quick_test                   - Quick test config (20-30min, Rank@1≈85%)

Example:
  ./train.sh -n resnet50
  ./train.sh -n resnet50_tricks --batchsize 16 --total_epoch 40
  ./train.sh -n quick_test --total_epoch 20

EOF
    exit 1
}

# 配置模型参数
configure_model() {
    case "$MODEL_NAME" in
        "resnet50")
            # ResNet-50 baseline
            ACTUAL_NAME="ft_ResNet50"
            BATCHSIZE=32
            LR=0.05
            ;;
        "resnet50_fp16")
            # ResNet-50 with fp16
            ACTUAL_NAME="ft_ResNet50_fp16"
            FP16="--fp16"
            BATCHSIZE=32
            ;;
        "resnet50_bf16")
            # ResNet-50 with bf16 (low memory)
            ACTUAL_NAME="ft_ResNet50_bf16"
            BF16="--bf16"
            BATCHSIZE=32
            ;;
        "resnet50_ibn")
            # ResNet-50-ibn
            ACTUAL_NAME="ft_ResNet50_ibn"
            IBM="--ibn"
            BATCHSIZE=32
            ;;
        "resnet50_tricks")
            # ResNet-50 with all tricks
            ACTUAL_NAME="ft_ResNet50_all_tricks"
            BATCHSIZE=8
            LR=0.02
            WARM_EPOCH=5
            STRIDE=1
            ERASING_P=0.5
            CIRCLE="--circle"
            ;;
        "resnet50_circle")
            # ResNet-50 with circle loss
            ACTUAL_NAME="ft_ResNet50_circle"
            WARM_EPOCH=5
            STRIDE=1
            ERASING_P=0.5
            BATCHSIZE=32
            LR=0.08
            CIRCLE="--circle"
            TOTAL_EPOCH=100
            ;;
        "densenet121")
            # DenseNet-121
            ACTUAL_NAME="ft_DenseNet121"
            USE_DENSE="--use_dense"
            BATCHSIZE=24
            ;;
        "densenet121_circle")
            # DenseNet-121 with circle loss
            ACTUAL_NAME="ft_DenseNet121_circle"
            USE_DENSE="--use_dense"
            CIRCLE="--circle"
            WARM_EPOCH=5
            BATCHSIZE=32
            ;;
        "pcb")
            # PCB model (降低batchsize适配10GB显存)
            ACTUAL_NAME="PCB"
            PCB="--PCB"
            BATCHSIZE=32
            LR=0.02
            ;;
        "hrnet18")
            # HRNet-18
            ACTUAL_NAME="ft_HRNet18"
            USE_HR="--use_hr"
            BATCHSIZE=24
            ;;
        "swin")
            # Swin Transformer (需要更多显存)
            ACTUAL_NAME="ft_Swin"
            USE_SWIN="--use_swin"
            BATCHSIZE=16
            ERASING_P=0.5
            CIRCLE="--circle"
            WARM_EPOCH=5
            ;;
        "efficientnet")
            # EfficientNet-b4
            ACTUAL_NAME="ft_EfficientNet"
            USE_EFFICIENT="--use_efficient"
            BATCHSIZE=24
            ;;
        "convnext")
            # ConvNeXt
            ACTUAL_NAME="ft_ConvNeXt"
            USE_CONVNEXT="--use_convnext"
            BATCHSIZE=32
            ;;
        "quick_test")
            # Quick test configuration (20-30 min)
            ACTUAL_NAME="quick_test"
            BATCHSIZE=16
            TOTAL_EPOCH=20
            BF16="--bf16"
            LR=0.035
            ;;
        *)
            echo "ERROR: Unknown model name: $MODEL_NAME"
            echo ""
            print_usage
            ;;
    esac
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            MODEL_NAME="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --batchsize)
            BATCHSIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --total_epoch)
            TOTAL_EPOCH="$2"
            shift 2
            ;;
        --warm_epoch)
            WARM_EPOCH="$2"
            shift 2
            ;;
        --stride)
            STRIDE="$2"
            shift 2
            ;;
        --erasing_p)
            ERASING_P="$2"
            shift 2
            ;;
        --droprate)
            DROPRATE="$2"
            shift 2
            ;;
        --linear_num)
            LINEAR_NUM="$2"
            shift 2
            ;;
        --no_train_all)
            TRAIN_ALL=""
            shift
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        --bf16)
            BF16="--bf16"
            shift
            ;;
        --circle)
            CIRCLE="--circle"
            shift
            ;;
        --triplet)
            TRIPLET="--triplet"
            shift
            ;;
        --contrast)
            CONTRAST="--contrast"
            shift
            ;;
        --cosine)
            COSINE="--cosine"
            shift
            ;;
        --dg)
            DG="--DG"
            shift
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            print_usage
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: Model name (-n) is required"
    echo ""
    print_usage
fi

# 配置模型
configure_model

# 打印配置信息
print_config() {
    echo "========================================="
    echo "Training Configuration"
    echo "========================================="
    echo "Model: $MODEL_NAME ($ACTUAL_NAME)"
    echo "Start Time: $START_TIME_STR"
    echo "Python: $PYTHON_PATH"
    echo ""
    echo "Training Parameters:"
    echo "  Data Directory: $DATA_DIR"
    echo "  GPU IDs: $GPU_IDS"
    echo "  Batch Size: $BATCHSIZE"
    echo "  Learning Rate: $LR"
    echo "  Total Epochs: $TOTAL_EPOCH"
    echo "  Warm Epochs: $WARM_EPOCH"
    echo "  Stride: $STRIDE"
    echo "  Erasing Prob: $ERASING_P"
    echo "  Dropout Rate: $DROPRATE"
    echo ""
    echo "Model Options:"
    [ -n "$USE_DENSE" ] && echo "  - DenseNet-121"
    [ -n "$USE_SWIN" ] && echo "  - Swin Transformer"
    [ -n "$USE_HR" ] && echo "  - HRNet"
    [ -n "$USE_EFFICIENT" ] && echo "  - EfficientNet"
    [ -n "$PCB" ] && echo "  - PCB"
    [ -n "$IBM" ] && echo "  - IBN"
    [ -n "$FP16" ] && echo "  - FP16 Precision"
    [ -n "$BF16" ] && echo "  - BF16 Precision"
    [ -n "$CIRCLE" ] && echo "  - Circle Loss"
    [ -n "$TRIPLET" ] && echo "  - Triplet Loss"
    [ -n "$CONTRAST" ] && echo "  - Contrastive Loss"
    [ -n "$TRAIN_ALL" ] && echo "  - Train on all data"
    echo "========================================="
    echo ""
}

# 构建训练命令
build_train_command() {
    local cmd="$PYTHON_PATH train.py"
    cmd="$cmd --gpu_ids $GPU_IDS"
    cmd="$cmd --name $ACTUAL_NAME"
    cmd="$cmd --data_dir $DATA_DIR"
    cmd="$cmd --batchsize $BATCHSIZE"
    cmd="$cmd --lr $LR"
    cmd="$cmd --total_epoch $TOTAL_EPOCH"

    [ $WARM_EPOCH -gt 0 ] && cmd="$cmd --warm_epoch $WARM_EPOCH"
    [ $STRIDE -ne 2 ] && cmd="$cmd --stride $STRIDE"
    [ $(echo "$ERASING_P > 0" | bc) -eq 1 ] && cmd="$cmd --erasing_p $ERASING_P"
    [ $(echo "$DROPRATE != 0.5" | bc) -eq 1 ] && cmd="$cmd --droprate $DROPRATE"
    [ $LINEAR_NUM -ne 512 ] && cmd="$cmd --linear_num $LINEAR_NUM"

    [ -n "$TRAIN_ALL" ] && cmd="$cmd $TRAIN_ALL"
    [ -n "$USE_DENSE" ] && cmd="$cmd $USE_DENSE"
    [ -n "$USE_SWIN" ] && cmd="$cmd $USE_SWIN"
    [ -n "$USE_SWINV2" ] && cmd="$cmd $USE_SWINV2"
    [ -n "$USE_HR" ] && cmd="$cmd $USE_HR"
    [ -n "$USE_EFFICIENT" ] && cmd="$cmd $USE_EFFICIENT"
    [ -n "$USE_CONVNEXT" ] && cmd="$cmd $USE_CONVNEXT"
    [ -n "$PCB" ] && cmd="$cmd $PCB"
    [ -n "$IBM" ] && cmd="$cmd $IBM"
    [ -n "$USAM" ] && cmd="$cmd $USAM"
    [ -n "$FP16" ] && cmd="$cmd $FP16"
    [ -n "$BF16" ] && cmd="$cmd $BF16"
    [ -n "$CIRCLE" ] && cmd="$cmd $CIRCLE"
    [ -n "$ARCFACE" ] && cmd="$cmd $ARCFACE"
    [ -n "$COSFACE" ] && cmd="$cmd $COSFACE"
    [ -n "$TRIPLET" ] && cmd="$cmd $TRIPLET"
    [ -n "$LIFTED" ] && cmd="$cmd $LIFTED"
    [ -n "$CONTRAST" ] && cmd="$cmd $CONTRAST"
    [ -n "$INSTANCE" ] && cmd="$cmd $INSTANCE"
    [ -n "$SPHERE" ] && cmd="$cmd $SPHERE"
    [ -n "$DG" ] && cmd="$cmd $DG"
    [ -n "$COSINE" ] && cmd="$cmd $COSINE"

    echo "$cmd"
}

# 构建测试命令
build_test_command() {
    local cmd="$PYTHON_PATH test.py"
    cmd="$cmd --gpu_ids $GPU_IDS"
    cmd="$cmd --name $ACTUAL_NAME"
    cmd="$cmd --test_dir $DATA_DIR"
    cmd="$cmd --batchsize $BATCHSIZE"
    cmd="$cmd --which_epoch last"

    [ -n "$PCB" ] && cmd="$cmd $PCB"

    echo "$cmd"
}

# 提取评估结果
extract_results() {
    local result_file="$1"
    if [ -f "$result_file" ]; then
        # 提取Rank@1, Rank@5, Rank@10, mAP
        RANK1=$(grep -oP "Rank@1:\s*\K[\d.]+" "$result_file" | tail -1)
        RANK5=$(grep -oP "Rank@5:\s*\K[\d.]+" "$result_file" | tail -1)
        RANK10=$(grep -oP "Rank@10:\s*\K[\d.]+" "$result_file" | tail -1)
        MAP=$(grep -oP "mAP:\s*\K[\d.]+" "$result_file" | tail -1)
    fi
}

# 打印训练报告
print_report() {
    local status=$1
    local end_time_str=$(date '+%Y-%m-%d %H:%M:%S')
    local duration_sec=$((END_TIME - START_TIME))
    local duration_min=$((duration_sec / 60))
    local duration_hour=$((duration_min / 60))
    local duration_min_rem=$((duration_min % 60))

    echo ""
    echo "========================================="
    echo "Training Report"
    echo "========================================="
    echo "Model: $MODEL_NAME ($ACTUAL_NAME)"
    echo "Status: $status"
    echo ""
    echo "Timeline:"
    echo "  Start Time:    $START_TIME_STR"
    echo "  End Time:      $end_time_str"
    echo "  Total Duration: ${duration_hour}h ${duration_min_rem}m (${duration_sec}s)"
    echo ""

    if [ "$status" == "SUCCESS" ]; then
        echo "Model Performance:"
        if [ -n "$RANK1" ]; then
            echo "  Rank@1:  ${RANK1}%"
            echo "  Rank@5:  ${RANK5}%"
            echo "  Rank@10: ${RANK10}%"
            echo "  mAP:     ${MAP}%"
        else
            echo "  Performance metrics not available"
        fi
        echo ""
    fi

    if [ -n "$TRAIN_ERROR" ] || [ -n "$TEST_ERROR" ] || [ -n "$EVAL_ERROR" ]; then
        echo "Errors Encountered:"
        [ -n "$TRAIN_ERROR" ] && echo "  Training Error: $TRAIN_ERROR"
        [ -n "$TEST_ERROR" ] && echo "  Testing Error: $TEST_ERROR"
        [ -n "$EVAL_ERROR" ] && echo "  Evaluation Error: $EVAL_ERROR"
        echo ""
    fi

    echo "Model saved at: ./model/$ACTUAL_NAME/"
    echo "Training curve: ./model/$ACTUAL_NAME/train.jpg"
    if [ -f "result.txt" ]; then
        echo "Evaluation results: ./result.txt"
    fi
    echo "========================================="
    echo ""
}

# 主程序
main() {
    print_config

    # 构建训练命令
    TRAIN_CMD=$(build_train_command)
    echo "Training Command:"
    echo "$TRAIN_CMD"
    echo ""

    # 训练
    echo "========================================="
    echo "Starting Training..."
    echo "========================================="
    if ! eval "$TRAIN_CMD"; then
        TRAIN_ERROR="Training command failed with exit code $?"
        END_TIME=$(date +%s)
        print_report "FAILED"
        exit 1
    fi

    echo ""
    echo "========================================="
    echo "Training Completed!"
    echo "========================================="
    echo ""

    # 测试
    TEST_CMD=$(build_test_command)
    echo "Testing Command:"
    echo "$TEST_CMD"
    echo ""

    echo "========================================="
    echo "Starting Testing..."
    echo "========================================="
    if ! eval "$TEST_CMD"; then
        TEST_ERROR="Testing command failed with exit code $?"
        END_TIME=$(date +%s)
        print_report "PARTIAL"
        exit 1
    fi

    echo ""
    echo "========================================="
    echo "Testing Completed!"
    echo "========================================="
    echo ""

    # 评估
    echo "========================================="
    echo "Starting Evaluation..."
    echo "========================================="
    if ! $PYTHON_PATH evaluate_gpu.py > eval_output.txt 2>&1; then
        EVAL_ERROR="Evaluation command failed with exit code $?"
        cat eval_output.txt
    else
        cat eval_output.txt
        # 提取结果
        extract_results "eval_output.txt"
    fi

    # 结束时间
    END_TIME=$(date +%s)

    # 打印报告
    print_report "SUCCESS"
}

# 运行主程序
main
