#!/bin/bash
# 环境验证脚本
# 用于验证conda环境和数据集的完整性

echo "======================================"
echo "Bug Localization DNN+rVSM 环境验证"
echo "======================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查计数器
SUCCESS=0
FAIL=0
WARN=0

# 1. 检查conda环境
echo "1. 检查Conda环境..."
if conda env list | grep -q "dnn_rvsm"; then
    echo -e "${GREEN}✓${NC} Conda环境 'dnn_rvsm' 已创建"
    ((SUCCESS++))
else
    echo -e "${RED}✗${NC} Conda环境 'dnn_rvsm' 未找到"
    ((FAIL++))
fi
echo ""

# 2. 激活环境并检查Python版本
echo "2. 检查Python版本..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dnn_rvsm
PYTHON_VERSION=$(python --version 2>&1)
if [[ $PYTHON_VERSION == *"3.7"* ]]; then
    echo -e "${GREEN}✓${NC} Python版本: $PYTHON_VERSION"
    ((SUCCESS++))
else
    echo -e "${YELLOW}⚠${NC} Python版本: $PYTHON_VERSION (期望: 3.7.x)"
    ((WARN++))
fi
echo ""

# 3. 检查Python包
echo "3. 检查Python依赖包..."
# 注意: scikit-learn导入时使用sklearn
PACKAGES=("joblib" "nltk" "numpy" "sklearn" "scipy")
DISPLAY_NAMES=("joblib" "nltk" "numpy" "scikit-learn" "scipy")
for i in "${!PACKAGES[@]}"; do
    pkg="${PACKAGES[$i]}"
    display="${DISPLAY_NAMES[$i]}"
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
        echo -e "${GREEN}✓${NC} $display ($VERSION)"
        ((SUCCESS++))
    else
        echo -e "${RED}✗${NC} $display 未安装"
        ((FAIL++))
    fi
done
echo ""

# 4. 检查NLTK数据
echo "4. 检查NLTK数据..."
if [ -d ~/nltk_data/tokenizers/punkt ]; then
    echo -e "${GREEN}✓${NC} NLTK punkt tokenizer 已安装"
    ((SUCCESS++))
else
    echo -e "${RED}✗${NC} NLTK punkt tokenizer 未找到"
    ((FAIL++))
fi

if [ -d ~/nltk_data/corpora/stopwords ]; then
    echo -e "${GREEN}✓${NC} NLTK stopwords 已安装"
    ((SUCCESS++))
else
    echo -e "${YELLOW}⚠${NC} NLTK stopwords 未找到 (不影响训练，因为使用预提取特征)"
    ((WARN++))
fi
echo ""

# 5. 检查数据集
echo "5. 检查数据集文件..."
DATA_DIR="/home/green/energy_dl/test/bug-localization-by-dnn-and-rvsm/data"

if [ -f "$DATA_DIR/Eclipse_Platform_UI.txt" ]; then
    SIZE=$(du -h "$DATA_DIR/Eclipse_Platform_UI.txt" | cut -f1)
    echo -e "${GREEN}✓${NC} Eclipse_Platform_UI.txt 存在 (大小: $SIZE)"
    ((SUCCESS++))
else
    echo -e "${RED}✗${NC} Eclipse_Platform_UI.txt 未找到"
    ((FAIL++))
fi

if [ -f "$DATA_DIR/features.csv" ]; then
    SIZE=$(du -h "$DATA_DIR/features.csv" | cut -f1)
    LINES=$(wc -l < "$DATA_DIR/features.csv")
    echo -e "${GREEN}✓${NC} features.csv 存在 (大小: $SIZE, 行数: $LINES)"
    ((SUCCESS++))
else
    echo -e "${RED}✗${NC} features.csv 未找到"
    ((FAIL++))
fi

if [ -d "$DATA_DIR/eclipse.platform.ui" ]; then
    echo -e "${GREEN}✓${NC} Eclipse源码仓库已克隆"
    ((SUCCESS++))
else
    echo -e "${YELLOW}⚠${NC} Eclipse源码仓库未克隆 (仅重新提取特征时需要)"
    ((WARN++))
fi
echo ""

# 6. 检查源代码文件
echo "6. 检查源代码文件..."
SRC_DIR="/home/green/energy_dl/test/bug-localization-by-dnn-and-rvsm/src"
SRC_FILES=("main.py" "dnn_model.py" "rvsm_model.py" "util.py" "feature_extraction.py")
for file in "${SRC_FILES[@]}"; do
    if [ -f "$SRC_DIR/$file" ]; then
        echo -e "${GREEN}✓${NC} $file 存在"
        ((SUCCESS++))
    else
        echo -e "${RED}✗${NC} $file 未找到"
        ((FAIL++))
    fi
done
echo ""

# 7. 测试导入
echo "7. 测试Python导入..."
if python << 'EOF'
import sys
sys.path.append('/home/green/energy_dl/test/bug-localization-by-dnn-and-rvsm/src')
try:
    from util import csv2dict, tsv2dict
    from sklearn.neural_network import MLPRegressor
    print("导入成功")
    exit(0)
except Exception as e:
    print(f"导入失败: {e}")
    exit(1)
EOF
then
    echo -e "${GREEN}✓${NC} 核心模块导入成功"
    ((SUCCESS++))
else
    echo -e "${RED}✗${NC} 核心模块导入失败"
    ((FAIL++))
fi
echo ""

# 8. 检查硬件资源
echo "8. 检查硬件资源..."
CPU_COUNT=$(nproc)
MEMORY=$(free -h | awk '/^Mem:/ {print $2}')
echo -e "${GREEN}✓${NC} CPU核心数: $CPU_COUNT"
echo -e "${GREEN}✓${NC} 可用内存: $MEMORY"
((SUCCESS+=2))

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} GPU: $GPU_INFO"
    ((SUCCESS++))
else
    echo -e "${YELLOW}⚠${NC} 未检测到NVIDIA GPU (不影响训练，CPU即可)"
    ((WARN++))
fi
echo ""

# 总结
echo "======================================"
echo "验证总结"
echo "======================================"
echo -e "${GREEN}成功: $SUCCESS${NC}"
echo -e "${YELLOW}警告: $WARN${NC}"
echo -e "${RED}失败: $FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ 环境验证通过，可以开始训练！${NC}"
    echo ""
    echo "运行训练命令:"
    echo "  cd /home/green/energy_dl/test/bug-localization-by-dnn-and-rvsm/src"
    echo "  conda activate dnn_rvsm"
    echo "  python main.py"
    exit 0
else
    echo -e "${RED}✗ 环境验证失败，请检查上述错误${NC}"
    exit 1
fi
