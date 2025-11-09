#!/bin/bash
#
# 参数缩写功能验证脚本
# 测试 mutation.py 的所有参数缩写是否正常工作
#

echo "=========================================="
echo "mutation.py 参数缩写功能验证"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试计数
PASSED=0
FAILED=0

# 测试函数
test_abbreviation() {
    local test_name=$1
    local command=$2
    local expected_pattern=$3

    echo -n "测试: $test_name ... "

    output=$(eval $command 2>&1)

    if echo "$output" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}✓ 通过${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ 失败${NC}"
        echo "  期望输出包含: $expected_pattern"
        echo "  实际输出: $output"
        ((FAILED++))
        return 1
    fi
}

echo "1. 测试 --list 缩写 (-l)"
test_abbreviation \
    "--list 缩写" \
    "python3 mutation.py -l" \
    "Available Repositories and Models"

echo ""
echo "2. 测试 --help 缩写 (-h)"
test_abbreviation \
    "--help 缩写" \
    "python3 mutation.py -h" \
    "Mutation-based Training Energy Profiler"

echo ""
echo "3. 验证所有缩写在帮助中显示"
echo -n "测试: 所有缩写在帮助中显示 ... "

help_output=$(python3 mutation.py -h 2>&1)
all_found=true

# 检查所有缩写
abbreviations=("-ec" "-r" "-m" "-mt" "-n" "-g" "-mr" "-l" "-c" "-s")
for abbr in "${abbreviations[@]}"; do
    if ! echo "$help_output" | grep -q -- "$abbr"; then
        echo -e "${RED}✗ 失败${NC}"
        echo "  缺失缩写: $abbr"
        all_found=false
        ((FAILED++))
        break
    fi
done

if [ "$all_found" = true ]; then
    echo -e "${GREEN}✓ 通过${NC}"
    ((PASSED++))
fi

echo ""
echo "4. 测试参数解析（不实际运行训练）"
test_abbreviation \
    "缺少必需参数时的错误提示" \
    "python3 mutation.py -r pytorch_resnet_cifar10 2>&1" \
    "usage:"

echo ""
echo "5. 测试混用完整参数和缩写"
test_abbreviation \
    "混用参数格式" \
    "python3 mutation.py -r pytorch_resnet_cifar10 --model resnet20 2>&1" \
    "usage:"

echo ""
echo "=========================================="
echo "测试总结"
echo "=========================================="
echo -e "通过: ${GREEN}$PASSED${NC}"
echo -e "失败: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ 所有测试通过！参数缩写功能正常工作。${NC}"
    exit 0
else
    echo -e "${RED}❌ 有测试失败，请检查上述错误信息。${NC}"
    exit 1
fi
