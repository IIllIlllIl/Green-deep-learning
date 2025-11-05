#!/bin/bash
################################################################################
# Environment Verification Script
#
# Checks which environments are installed and their status
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Environment Status Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Required environments
REQUIRED_ENVS=(
    "mutation_runner:mutation_runner.yml:主程序运行环境"
    "pytorch_resnet_cifar10:pytorch_resnet_cifar10.yml:ResNet CIFAR-10"
    "vulberta:vulberta.yml:VulBERTa漏洞检测"
    "reid_baseline:reid_baseline.yml:Person Re-ID"
    "mrt-oast:mrt-oast.yml:MRT-OAST代码克隆"
    "dnn_rvsm:dnn_rvsm.yml:Bug定位"
    "pytorch_examples:pytorch_examples.yml:PyTorch示例"
)

TOTAL=0
INSTALLED=0
MISSING=0

echo -e "${BLUE}环境名称${NC}                 ${BLUE}状态${NC}    ${BLUE}说明${NC}"
echo "------------------------------------------------------------"

for env_info in "${REQUIRED_ENVS[@]}"; do
    IFS=':' read -r env_name env_file description <<< "$env_info"
    ((TOTAL++))

    if conda env list | grep -q "^$env_name "; then
        echo -e "${env_name//_/ }$(printf '%*s' $((25-${#env_name})) '')${GREEN}✓ 已安装${NC}  $description"
        ((INSTALLED++))
    else
        echo -e "${env_name//_/ }$(printf '%*s' $((25-${#env_name})) '')${RED}✗ 缺失${NC}    $description"
        ((MISSING++))
    fi
done

echo ""
echo -e "总计: ${BLUE}$TOTAL${NC}  |  已安装: ${GREEN}$INSTALLED${NC}  |  缺失: ${RED}$MISSING${NC}"
echo ""

if [ $MISSING -gt 0 ]; then
    echo -e "${YELLOW}[提示]${NC} 缺失的环境可以通过以下命令创建："
    echo "  cd environment"
    echo "  ./setup_environments.sh --all"
    echo ""
    echo "或单独创建："
    echo "  conda env create -f environment/<环境文件>.yml"
fi

echo ""
echo -e "${BLUE}详细信息:${NC}"
echo "  查看所有环境: conda env list"
echo "  激活环境: conda activate <环境名>"
echo "  查看环境包: conda list -n <环境名>"
echo "  删除环境: conda env remove -n <环境名>"
