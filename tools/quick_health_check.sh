#!/usr/bin/env bash
#
# Energy DL 项目快速健康检查脚本
#
# 用途: 快速验证环境、数据和工具是否就绪
# 作者: Green
# 创建日期: 2026-01-10
# 使用方法: bash tools/quick_health_check.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 计数器
total_checks=0
passed_checks=0
warnings=0

# 打印函数
print_header() {
    echo ""
    echo "=========================================================="
    echo "$1"
    echo "=========================================================="
    echo ""
}

print_success() {
    echo -e "  ${GREEN}✅${NC} $1"
    ((passed_checks++))
}

print_error() {
    echo -e "  ${RED}❌${NC} $1"
}

print_warning() {
    echo -e "  ${YELLOW}⚠️ ${NC} $1"
    ((warnings++))
}

print_info() {
    echo -e "  ${BLUE}ℹ️ ${NC} $1"
}

# 检查函数
check_python() {
    print_header "[1/6] 检查 Python 环境..."
    ((total_checks++))

    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python 版本: $python_version"

        if command -v pip3 &> /dev/null; then
            print_success "pip 可用"
        else
            print_error "pip3 未安装"
            return 1
        fi
    else
        print_error "Python3 未安装"
        return 1
    fi
}

check_gpu() {
    print_header "[2/6] 检查 GPU 驱动..."
    ((total_checks++))

    if command -v nvidia-smi &> /dev/null; then
        nvidia_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

        print_success "NVIDIA-SMI: $nvidia_version"
        print_success "CUDA 版本: $cuda_version"
        print_success "GPU 设备: $gpu_name (${gpu_count}个)"
    else
        print_warning "nvidia-smi 不可用 (可能无 GPU 或驱动未安装)"
        print_info "某些实验需要 GPU，请确认是否需要"
    fi
}

check_dependencies() {
    print_header "[3/6] 检查项目依赖..."
    ((total_checks++))

    # 检查关键 Python 包
    dependencies=("pandas" "numpy" "torch")
    all_installed=true

    for dep in "${dependencies[@]}"; do
        if python3 -c "import $dep; print($dep.__version__)" &> /dev/null; then
            version=$(python3 -c "import $dep; print($dep.__version__)" 2>&1)
            print_success "$dep: $version"
        else
            print_error "$dep 未安装"
            all_installed=false
        fi
    done

    if [ "$all_installed" = false ]; then
        print_info "运行 'pip3 install -r requirements.txt' 安装依赖"
        return 1
    fi
}

check_data_files() {
    print_header "[4/6] 检查核心数据文件..."
    ((total_checks++))

    # 检查 raw_data.csv
    if [ -f "data/raw_data.csv" ]; then
        rows=$(wc -l < data/raw_data.csv)
        cols=$(head -1 data/raw_data.csv | tr ',' '\n' | wc -l)
        print_success "data/raw_data.csv 存在 (${rows}行, ${cols}列)"

        # 检查数据完整性（如果验证脚本存在）
        if [ -f "tools/data_management/validate_raw_data.py" ]; then
            # 运行验证脚本并捕获输出
            validation_output=$(python3 tools/data_management/validate_raw_data.py 2>&1 | grep "完整性" || echo "")
            if [ -n "$validation_output" ]; then
                # 提取完整性百分比
                completeness=$(echo "$validation_output" | grep -oP '\d+\.\d+%' | head -1)
                if [ -n "$completeness" ]; then
                    print_success "数据完整性: $completeness"
                fi
            fi
        fi
    else
        print_error "data/raw_data.csv 不存在"
        print_info "数据文件可能在 results/ 或 archives/ 目录"
        return 1
    fi

    # 检查 data.csv
    if [ -f "data/data.csv" ]; then
        print_info "data/data.csv 存在（精简数据文件）"
    else
        print_warning "data/data.csv 不存在（可选）"
    fi
}

check_scripts() {
    print_header "[5/6] 检查核心脚本..."
    ((total_checks++))

    # 检查数据管理脚本
    if [ -d "tools/data_management" ]; then
        data_scripts_count=$(find tools/data_management -name "*.py" -type f | wc -l)
        print_success "数据管理脚本: ${data_scripts_count} 个可用"
    else
        print_error "tools/data_management 目录不存在"
        return 1
    fi

    # 检查配置管理脚本
    if [ -d "tools/config_management" ]; then
        config_scripts_count=$(find tools/config_management -name "*.py" -type f | wc -l)
        print_success "配置管理脚本: ${config_scripts_count} 个可用"
    else
        print_warning "tools/config_management 目录不存在"
    fi
}

check_directories() {
    print_header "[6/6] 检查实验结果目录..."
    ((total_checks++))

    # 检查 results 目录
    if [ -d "results" ]; then
        recent_runs=$(find results -maxdepth 1 -type d -name "run_*" 2>/dev/null | wc -l)
        if [ $recent_runs -gt 0 ]; then
            print_info "results/ 目录包含 $recent_runs 个运行结果"
        else
            print_warning "results/ 目录为空（无实验结果）"
        fi
    else
        print_warning "results/ 目录不存在"
    fi

    # 检查 archives 目录
    if [ -d "archives" ]; then
        print_success "archives/ 目录可访问"
    else
        print_warning "archives/ 目录不存在"
    fi
}

# 主函数
main() {
    print_header "Energy DL 项目健康检查"

    # 执行所有检查
    check_python
    check_gpu
    check_dependencies
    check_data_files
    check_scripts
    check_directories

    # 总结
    print_header "健康检查完成！"

    echo ""
    if [ $passed_checks -eq $total_checks ] && [ $warnings -eq 0 ]; then
        echo -e "${GREEN}总体状态: ✅ 环境完美就绪${NC}"
        echo ""
        echo "建议:"
        echo "  - 所有检查通过，可以开始新实验或数据分析"
    elif [ $passed_checks -ge 4 ]; then
        echo -e "${YELLOW}总体状态: ⚠️  环境基本就绪（有 $warnings 个警告）${NC}"
        echo ""
        echo "建议:"
        echo "  - 核心功能可用，但建议修复警告项"
    else
        echo -e "${RED}总体状态: ❌ 环境存在问题${NC}"
        echo ""
        echo "建议:"
        echo "  - 请修复上述错误后再继续"
        return 1
    fi

    echo ""
    echo "运行第一个测试实验:"
    echo "  sudo python3 mutation.py -ec settings/quick_test.json"
    echo ""
    echo "查看项目状态:"
    echo "  python3 tools/data_management/validate_raw_data.py"
    echo ""
}

# 运行主函数
main
