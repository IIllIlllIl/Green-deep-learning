#!/bin/bash
################################################################################
# PyTorch升级脚本 - 支持RTX 3080 (sm_86)
#
# 此脚本将升级PyTorch从1.7.0+cu101到1.8.0+cu111
# 以支持RTX 3080 GPU训练
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "================================================================================"
echo "PyTorch升级脚本 - 支持RTX 3080 (CUDA sm_86)"
echo "================================================================================"
echo ""

# 检查conda
if ! command -v conda &> /dev/null; then
    print_error "Conda未找到，请先安装Miniconda或Anaconda"
    exit 1
fi

# 激活conda
print_info "初始化conda..."
source /home/green/miniconda3/etc/profile.d/conda.sh

# 检查vulberta环境是否存在
if ! conda env list | grep -q "^vulberta "; then
    print_error "vulberta环境不存在"
    exit 1
fi

# 显示当前状态
print_info "当前环境状态："
conda activate vulberta
python -c "
import torch
import sys
print('Python版本:', sys.version.split()[0])
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print('GPU:', torch.cuda.get_device_name(0))
    except:
        print('GPU: 检测到但无法使用 (CUDA版本不兼容)')
" 2>&1 || print_warning "当前环境检测失败（预期，因为CUDA不兼容）"

echo ""
print_warning "即将执行以下操作："
echo "  1. 卸载PyTorch 1.7.0+cu101"
echo "  2. 安装PyTorch 1.8.0+cu111（支持RTX 3080）"
echo "  3. 验证安装"
echo ""

# 询问是否继续
read -p "是否继续？[y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "用户取消操作"
    exit 0
fi

echo ""
print_info "开始升级..."
echo ""

# 卸载旧版本
print_info "步骤 1/3: 卸载旧版PyTorch..."
pip uninstall torch torchvision torchaudio -y

print_success "旧版PyTorch已卸载"
echo ""

# 安装新版本
print_info "步骤 2/3: 安装PyTorch 1.8.0+cu111..."
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

print_success "PyTorch 1.8.0+cu111 安装完成"
echo ""

# 验证安装
print_info "步骤 3/3: 验证安装..."
echo ""

VALIDATION_OUTPUT=$(python -c "
import torch
import sys

print('='*80)
print('验证结果')
print('='*80)
print('Python版本:', sys.version.split()[0])
print('PyTorch版本:', torch.__version__)
print('CUDA版本:', torch.version.cuda)
print('CUDA可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU名称:', torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print(f'GPU Compute Capability: {cap[0]}.{cap[1]}')

    # 测试GPU运算
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x)
        print('GPU计算测试: PASSED ✓')
        print('='*80)
        sys.exit(0)
    except Exception as e:
        print('GPU计算测试: FAILED ✗')
        print('错误:', str(e))
        print('='*80)
        sys.exit(1)
else:
    print('GPU计算测试: SKIPPED (CUDA不可用)')
    print('='*80)
    sys.exit(1)
" 2>&1)

echo "$VALIDATION_OUTPUT"

# 检查验证结果
if echo "$VALIDATION_OUTPUT" | grep -q "PASSED ✓"; then
    echo ""
    print_success "✓ 升级成功！RTX 3080 GPU已可用"
    echo ""
    echo "================================================================================"
    echo "后续操作"
    echo "================================================================================"
    echo "现在可以使用GPU进行训练："
    echo ""
    echo "  # 测试训练（1个epoch）"
    echo "  ./train.sh -n mlp -d devign --epochs 1 --batch_size 2 2>&1 | tee test.log"
    echo ""
    echo "  # 完整训练（默认参数）"
    echo "  ./train.sh -n mlp -d devign 2>&1 | tee training.log"
    echo ""
    echo "  # 使用FP16混合精度训练"
    echo "  ./train.sh -n mlp -d devign --fp16 2>&1 | tee training.log"
    echo ""
    echo "================================================================================"
    exit 0
else
    echo ""
    print_error "✗ GPU验证失败"
    echo ""
    echo "可能的原因："
    echo "  1. PyTorch安装不完整"
    echo "  2. CUDA驱动问题"
    echo ""
    echo "临时解决方案："
    echo "  使用CPU训练: ./train.sh -n mlp -d devign --cpu 2>&1 | tee training.log"
    echo ""
    exit 1
fi
