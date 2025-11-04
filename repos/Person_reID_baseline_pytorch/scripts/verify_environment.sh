#!/bin/bash
# 环境和数据集验证脚本

echo "========================================="
echo "Person ReID Baseline 环境验证"
echo "========================================="
echo ""

# 检查conda环境
echo "1. 检查conda环境..."
if conda env list | grep -q "reid_baseline"; then
    echo "   ✓ Conda环境 'reid_baseline' 存在"
else
    echo "   ✗ Conda环境 'reid_baseline' 不存在"
    echo "   请运行: conda create -n reid_baseline python=3.10 -y"
    exit 1
fi

# 激活环境并检查Python包
echo ""
echo "2. 检查Python包..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reid_baseline

# 检查PyTorch
if python -c "import torch; print('PyTorch版本:', torch.__version__)" 2>/dev/null; then
    echo "   ✓ PyTorch已安装"
    python -c "import torch; print('   - CUDA可用:', torch.cuda.is_available())"
    python -c "import torch; print('   - CUDA版本:', torch.version.cuda)"
else
    echo "   ✗ PyTorch未安装"
    exit 1
fi

# 检查其他必要的包
packages=("torchvision" "timm" "scipy" "tqdm" "matplotlib" "yaml")
for pkg in "${packages[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        echo "   ✓ $pkg 已安装"
    else
        echo "   ✗ $pkg 未安装"
    fi
done

# 检查GPU
echo ""
echo "3. 检查GPU..."
if nvidia-smi &>/dev/null; then
    echo "   ✓ NVIDIA驱动已安装"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "   - GPU: $line"
    done
else
    echo "   ✗ NVIDIA驱动未安装或GPU不可用"
fi

# 检查数据集
echo ""
echo "4. 检查数据集..."
MARKET_DIR="../Market/Market-1501-v15.09.15"
if [ -d "$MARKET_DIR" ]; then
    echo "   ✓ Market-1501数据集存在: $MARKET_DIR"

    if [ -d "$MARKET_DIR/pytorch" ]; then
        echo "   ✓ 数据集已准备 (pytorch格式)"

        # 统计数据
        train_classes=$(ls "$MARKET_DIR/pytorch/train_all/" 2>/dev/null | wc -l)
        train_images=$(find "$MARKET_DIR/pytorch/train_all/" -name "*.jpg" 2>/dev/null | wc -l)
        query_images=$(find "$MARKET_DIR/pytorch/query/" -name "*.jpg" 2>/dev/null | wc -l)
        gallery_images=$(find "$MARKET_DIR/pytorch/gallery/" -name "*.jpg" 2>/dev/null | wc -l)

        echo "   - 训练类别数: $train_classes"
        echo "   - 训练图片数: $train_images"
        echo "   - 查询图片数: $query_images"
        echo "   - Gallery图片数: $gallery_images"
    else
        echo "   ✗ 数据集未准备，请运行: python prepare.py"
        exit 1
    fi
else
    echo "   ✗ Market-1501数据集不存在"
    echo "   请下载数据集到: $MARKET_DIR"
    exit 1
fi

# 检查训练脚本
echo ""
echo "5. 检查训练脚本..."
if [ -f "train.py" ]; then
    echo "   ✓ train.py 存在"
else
    echo "   ✗ train.py 不存在"
fi

if [ -f "test.py" ]; then
    echo "   ✓ test.py 存在"
else
    echo "   ✗ test.py 不存在"
fi

# 快速测试
echo ""
echo "6. 运行快速测试..."
echo "   测试数据加载..."
python -c "
import sys
sys.path.append('.')
from torchvision import datasets, transforms
import os

data_dir = '$MARKET_DIR/pytorch'
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'train_all'), transform)
    print('   ✓ 数据加载成功')
    print(f'   - 数据集大小: {len(dataset)}')
    print(f'   - 类别数: {len(dataset.classes)}')
except Exception as e:
    print(f'   ✗ 数据加载失败: {e}')
    sys.exit(1)
" || exit 1

echo ""
echo "========================================="
echo "环境验证完成！"
echo "========================================="
echo ""
echo "可用的训练脚本："
echo "  1. scripts/quick_train_test.sh          # 快速验证 (20-30分钟)"
echo "  2. scripts/baseline_train_test.sh       # 标准baseline (1-1.5小时)"
echo "  3. scripts/high_performance_train_test.sh # 高性能 (2-3小时)"
echo ""
echo "示例使用："
echo "  bash scripts/quick_train_test.sh"
echo ""
