#!/bin/bash
# 文件结构整理脚本
# 创建日期: 2026-02-01
# 状态: 待确认后执行

set -e  # 遇到错误立即退出

echo "=========================================="
echo "文件结构整理脚本"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 当前目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo -e "${YELLOW}当前工作目录: $(pwd)${NC}"
echo ""

# 步骤1: 创建必要目录
echo -e "${GREEN}步骤1: 创建必要目录${NC}"
echo "--------------------------------------"
mkdir -p docs/reports
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tools/legacy
echo "✓ 目录创建完成"
echo ""

# 步骤2: 移动测试文件
echo -e "${GREEN}步骤2: 移动测试文件${NC}"
echo "--------------------------------------"

# 单元测试
if [ -f "test_tradeoff_logic.py" ]; then
    echo "移动: test_tradeoff_logic.py → tests/unit/"
    git mv test_tradeoff_logic.py tests/unit/ 2>/dev/null || mv test_tradeoff_logic.py tests/unit/
fi

if [ -f "test_ctf_alignment.py" ]; then
    echo "移动: test_ctf_alignment.py → tests/unit/"
    git mv test_ctf_alignment.py tests/unit/ 2>/dev/null || mv test_ctf_alignment.py tests/unit/
fi

# 集成测试
if [ -f "test_tradeoff_simple.py" ]; then
    echo "移动: test_tradeoff_simple.py → tests/integration/"
    git mv test_tradeoff_simple.py tests/integration/ 2>/dev/null || mv test_tradeoff_simple.py tests/integration/
fi

if [ -f "test_tradeoff_optimization.py" ]; then
    echo "移动: test_tradeoff_optimization.py → tests/integration/"
    git mv test_tradeoff_optimization.py tests/integration/ 2>/dev/null || mv test_tradeoff_optimization.py tests/integration/
fi

echo "✓ 测试文件移动完成"
echo ""

# 步骤3: 移动脚本文件
echo -e "${GREEN}步骤3: 移动脚本文件到归档${NC}"
echo "--------------------------------------"

if [ -f "check_correlation.py" ]; then
    echo "移动: check_correlation.py → tools/legacy/"
    git mv check_correlation.py tools/legacy/ 2>/dev/null || mv check_correlation.py tools/legacy/
fi

if [ -f "diagnose_ate_issue.py" ]; then
    echo "移动: diagnose_ate_issue.py → tools/legacy/"
    git mv diagnose_ate_issue.py tools/legacy/ 2>/dev/null || mv diagnose_ate_issue.py tools/legacy/
fi

if [ -f "check_tradeoff_sources.py" ]; then
    echo "移动: check_tradeoff_sources.py → tools/legacy/"
    git mv check_tradeoff_sources.py tools/legacy/ 2>/dev/null || mv check_tradeoff_sources.py tools/legacy/
fi

echo "✓ 脚本文件归档完成"
echo ""

# 步骤4: 移动文档文件
echo -e "${GREEN}步骤4: 移动文档文件${NC}"
echo "--------------------------------------"

# 报告文档
if [ -f "TRADEOFF_OPTIMIZATION_ACCEPTANCE_REPORT.md" ]; then
    echo "移动: TRADEOFF_OPTIMIZATION_ACCEPTANCE_REPORT.md → docs/reports/"
    git mv TRADEOFF_OPTIMIZATION_ACCEPTANCE_REPORT.md docs/reports/ 2>/dev/null || mv TRADEOFF_OPTIMIZATION_ACCEPTANCE_REPORT.md docs/reports/
fi

if [ -f "TRADEOFF_OPTIMIZATION_FINAL_SUMMARY.md" ]; then
    echo "移动: TRADEOFF_OPTIMIZATION_FINAL_SUMMARY.md → docs/reports/"
    git mv TRADEOFF_OPTIMIZATION_FINAL_SUMMARY.md docs/reports/ 2>/dev/null || mv TRADEOFF_OPTIMIZATION_FINAL_SUMMARY.md docs/reports/
fi

if [ -f "DIBS_GLOBAL_STD_VS_INTERACTION_COMPARISON_REPORT.md" ]; then
    echo "移动: DIBS_GLOBAL_STD_VS_INTERACTION_COMPARISON_REPORT.md → docs/reports/"
    git mv DIBS_GLOBAL_STD_VS_INTERACTION_COMPARISON_REPORT.md docs/reports/ 2>/dev/null || mv DIBS_GLOBAL_STD_VS_INTERACTION_COMPARISON_REPORT.md docs/reports/
fi

if [ -f "DIBS_METHODS_COMPARISON_SUMMARY.md" ]; then
    echo "移动: DIBS_METHODS_COMPARISON_SUMMARY.md → docs/reports/"
    git mv DIBS_METHODS_COMPARISON_SUMMARY.md docs/reports/ 2>/dev/null || mv DIBS_METHODS_COMPARISON_SUMMARY.md docs/reports/
fi

# 临时文档
if [ -f "NEXT_CONVERSATION_PROMPT.md" ]; then
    echo "移动: NEXT_CONVERSATION_PROMPT.md → docs/archived/"
    git mv NEXT_CONVERSATION_PROMPT.md docs/archived/ 2>/dev/null || mv NEXT_CONVERSATION_PROMPT.md docs/archived/
fi

echo "✓ 文档文件移动完成"
echo ""

# 步骤5: 验证整理结果
echo -e "${GREEN}步骤5: 验证整理结果${NC}"
echo "--------------------------------------"

echo ""
echo "根目录文件（应只有README.md）:"
ls -1 *.py *.md 2>/dev/null | grep -v "^README.md$" || echo "  (无其他文件)"

echo ""
echo "tests/unit/ 内容:"
ls -la tests/unit/*.py 2>/dev/null | tail -n +2 | awk '{print "  " $NF}'

echo ""
echo "tests/integration/ 内容:"
ls -la tests/integration/*.py 2>/dev/null | tail -n +2 | awk '{print "  " $NF}'

echo ""
echo "tools/legacy/ 内容:"
ls -la tools/legacy/*.py 2>/dev/null | tail -n +2 | awk '{print "  " $NF}'

echo ""
echo "docs/reports/ 内容:"
ls -la docs/reports/*.md 2>/dev/null | tail -n +2 | awk '{print "  " $NF}'

echo ""
echo "docs/archived/ 新增内容:"
ls -la docs/archived/NEXT_CONVERSATION_PROMPT.md 2>/dev/null | tail -n +2 | awk '{print "  " $NF}' || echo "  (未找到)"

echo ""
echo -e "${GREEN}=========================================="
echo "文件结构整理完成！"
echo "==========================================${NC}"
echo ""
echo -e "${YELLOW}注意事项：${NC}"
echo "1. 检查是否有其他文件引用了这些移动的文件"
echo "2. 运行测试验证功能正常"
echo "3. 提交Git变更: git commit -m '重构: 整理文件结构，移动错位文件到正确位置'"
echo ""
