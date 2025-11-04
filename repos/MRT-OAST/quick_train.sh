#!/bin/bash

################################################################################
# MRT-OAST 快速训练脚本
#
# 功能：预设的快速训练配置，适合快速测试和验证
#
# 使用示例：
#   ./quick_train.sh                  # 使用默认配置（OJClone, 3 epochs）
#   ./quick_train.sh GCJ              # 训练GCJ数据集
#   ./quick_train.sh BCB 5            # 训练BCB数据集，5个epoch
#
################################################################################

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# 默认参数
DATASET="${1:-OJClone}"
EPOCHS="${2:-3}"
BATCH_SIZE=32
MAX_LEN=128
LEARNING_RATE=0.0001

print_warning "=== 快速训练模式 ==="
print_info "数据集: $DATASET"
print_info "训练轮数: $EPOCHS"
print_info "批次大小: $BATCH_SIZE (减小以加快训练)"
print_info "最大序列长度: $MAX_LEN (减小以加快训练)"
echo ""

# 调用主训练脚本
./train_and_evaluate.sh \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-len "$MAX_LEN" \
    --lr "$LEARNING_RATE" \
    --quick

print_success "快速训练完成！"
