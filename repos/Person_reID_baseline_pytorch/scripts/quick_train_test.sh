#!/bin/bash
# Person ReID Baseline - 快速验证训练脚本
# 预计训练时长: 20-30分钟
# 预计性能: Rank@1 ≈ 85-87%, mAP ≈ 65-68%

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reid_baseline

# 设置工作目录
cd "$(dirname "$0")/.." || exit

# 设置数据路径
DATA_DIR="../Market/Market-1501-v15.09.15/pytorch"
MODEL_NAME="ft_ResNet50_quick_test"

echo "========================================="
echo "开始快速验证训练"
echo "模型名称: $MODEL_NAME"
echo "数据路径: $DATA_DIR"
echo "预计训练时长: 20-30分钟"
echo "========================================="

# 训练模型 (快速验证配置)
python train.py \
    --gpu_ids 0 \
    --name "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --train_all \
    --batchsize 16 \
    --total_epoch 20 \
    --bf16 \
    --lr 0.035

echo ""
echo "========================================="
echo "训练完成！开始测试..."
echo "========================================="

# 测试模型
python test.py \
    --gpu_ids 0 \
    --name "$MODEL_NAME" \
    --test_dir "$DATA_DIR" \
    --batchsize 32 \
    --which_epoch last

echo ""
echo "========================================="
echo "评估模型性能..."
echo "========================================="

# 评估结果
python evaluate_gpu.py

echo ""
echo "========================================="
echo "完成！"
echo "结果保存在: ./model/$MODEL_NAME/"
echo "查看训练曲线: ./model/$MODEL_NAME/train.jpg"
echo "========================================="
