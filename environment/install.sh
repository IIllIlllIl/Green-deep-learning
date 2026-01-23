#!/bin/bash
################################################################################
# Energy DL 项目环境安装脚本
#
# 用途: 批量创建/更新项目所需的所有 conda 和 venv 环境
#
# 使用方法:
#   ./install.sh                    # 安装所有环境
#   ./install.sh --training         # 仅安装训练环境
#   ./install.sh --analysis         # 仅安装分析环境
#   ./install.sh --env dnn_rvsm     # 安装单个环境
#   ./install.sh --check            # 检查环境状态
#   ./install.sh --help             # 显示帮助
#
################################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Conda 环境列表
CONDA_ENVS=(
    "dnn_rvsm:bug-localization-by-dnn-and-rvsm"
    "mrt-oast:MRT-OAST"
    "pytorch_resnet_cifar10:pytorch_resnet_cifar10"
    "reid_baseline:Person_reID_baseline_pytorch"
    "vulberta:VulBERTa"
)

# 分析环境列表
ANALYSIS_ENVS=(
    "causal-research:分析模块(DiBS/DML)"
)

# Venv 环境列表 (venv_path:name:description)
VENV_ENVS=(
    "repos/examples/venv:pytorch_examples:PyTorch示例模型"
)

# 显示帮助
show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    (无参数)                安装所有环境
    --training              仅安装训练环境
    --analysis              仅安装分析环境
    --env <name>            安装单个环境 (如: --env dnn_rvsm)
    --check                 检查环境状态
    --skip-existing         跳过已存在的环境
    --force                 重新创建已存在的环境
    -h, --help              显示此帮助

示例:
    $0                      # 安装所有环境
    $0 --training           # 仅安装训练环境
    $0 --env causal-research # 仅安装分析环境
    $0 --check              # 检查哪些环境已安装

环境列表:
训练环境:
EOF
    for entry in "${CONDA_ENVS[@]}"; do
        name="${entry%%:*}"
        desc="${entry##*:}"
        echo "  - $name ($desc)"
    done
    cat << EOF

分析环境:
EOF
    for entry in "${ANALYSIS_ENVS[@]}"; do
        name="${entry%%:*}"
        desc="${entry##*:}"
        echo "  - $name ($desc)"
    done
    cat << EOF

Venv 环境:
EOF
    for entry in "${VENV_ENVS[@]}"; do
        path="${entry%%:*}"
        rest="${entry#*:}"
        name="${rest%%:*}"
        desc="${rest##*:}"
        echo "  - $name ($desc) -> $PROJECT_ROOT/$path"
    done
}

# 检查 conda 是否可用
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "conda 未找到。请先安装 Miniconda 或 Anaconda。"
        exit 1
    fi
}

# 检查环境是否存在
conda_env_exists() {
    conda env list | grep -q "^$1 "
}

venv_exists() {
    [ -d "$1" ] && [ -f "$1/bin/python" ]
}

# 安装单个 conda 环境
install_conda_env() {
    local env_name=$1
    local env_file="$SCRIPT_DIR/conda/${env_name}.yml"

    if [ ! -f "$env_file" ]; then
        print_error "配置文件不存在: $env_file"
        return 1
    fi

    print_header "安装 conda 环境: $env_name"

    if conda_env_exists "$env_name"; then
        if [ "$SKIP_EXISTING" = "true" ]; then
            print_info "环境 $env_name 已存在，跳过"
            return 0
        fi
        if [ "$FORCE" = "true" ]; then
            print_warning "删除旧环境: $env_name"
            conda env remove -y -n "$env_name" || true
        else
            print_warning "环境 $env_name 已存在。使用 --force 重新创建"
            return 0
        fi
    fi

    print_info "创建环境: $env_name"
    conda env create -f "$env_file" || {
        print_error "创建环境失败: $env_name"
        return 1
    }

    print_success "环境 $env_name 安装完成"
}

# 安装单个 venv 环境
install_venv_env() {
    local venv_path=$1
    local env_name=$2
    local req_file="$SCRIPT_DIR/venv/${env_name}/requirements.txt"

    # 转换为绝对路径
    if [[ ! "$venv_path" = /* ]]; then
        venv_path="$PROJECT_ROOT/$venv_path"
    fi

    print_header "安装 venv 环境: $env_name"

    if venv_exists "$venv_path"; then
        if [ "$SKIP_EXISTING" = "true" ]; then
            print_info "venv 已存在，跳过: $venv_path"
            return 0
        fi
        if [ "$FORCE" = "true" ]; then
            print_warning "删除旧 venv: $venv_path"
            rm -rf "$venv_path"
        else
            print_warning "venv 已存在: $venv_path。使用 --force 重新创建"
            return 0
        fi
    fi

    print_info "创建 venv: $venv_path"
    python3 -m venv "$venv_path" || {
        print_error "创建 venv 失败: $venv_path"
        return 1
    }

    # 激活并安装依赖
    source "$venv_path/bin/activate"

    if [ -f "$req_file" ]; then
        print_info "安装依赖: $req_file"
        pip install -r "$req_file" || {
            print_error "安装依赖失败"
            deactivate
            return 1
        }
    fi

    deactivate
    print_success "venv $env_name 安装完成"
}

# 检查环境状态
check_envs() {
    print_header "环境状态检查"

    echo ""
    echo -e "${CYAN}Conda 环境:${NC}"
    for entry in "${CONDA_ENVS[@]}" "${ANALYSIS_ENVS[@]}"; do
        name="${entry%%:*}"
        desc="${entry##*:}"
        if conda_env_exists "$name"; then
            echo -e "  ${GREEN}✓${NC} $name ($desc)"
            # 显示 Python 版本
            version=$(conda run -n "$name" python --version 2>&1 || echo "?")
            echo -e "     └─ $version"
        else
            echo -e "  ${RED}✗${NC} $name ($desc) - 未安装"
        fi
    done

    echo ""
    echo -e "${CYAN}Venv 环境:${NC}"
    for entry in "${VENV_ENVS[@]}"; do
        path="${entry%%:*}"
        rest="${entry#*:}"
        name="${rest%%:*}"
        desc="${rest##*:}"
        if [[ ! "$path" = /* ]]; then
            full_path="$PROJECT_ROOT/$path"
        else
            full_path="$path"
        fi
        if venv_exists "$full_path"; then
            echo -e "  ${GREEN}✓${NC} $name ($desc)"
            version=$("$full_path/bin/python" --version 2>&1 || echo "?")
            echo -e "     └─ $version"
        else
            echo -e "  ${RED}✗${NC} $name ($desc) - 未安装"
        fi
    done

    echo ""
}

# 安装所有环境
install_all() {
    install_training
    install_analysis
    install_venvs
}

# 安装训练环境
install_training() {
    print_header "安装训练环境"
    for entry in "${CONDA_ENVS[@]}"; do
        env_name="${entry%%:*}"
        install_conda_env "$env_name"
    done
    install_venvs
}

# 安装分析环境
install_analysis() {
    print_header "安装分析环境"
    for entry in "${ANALYSIS_ENVS[@]}"; do
        env_name="${entry%%:*}"
        install_conda_env "$env_name"
    done
}

# 安装 venv 环境
install_venvs() {
    print_header "安装 Venv 环境"
    for entry in "${VENV_ENVS[@]}"; do
        path="${entry%%:*}"
        rest="${entry#*:}"
        name="${rest%%:*}"
        install_venv_env "$path" "$name"
    done
}

# 解析命令行参数
SKIP_EXISTING=false
FORCE=false
INSTALL_MODE="all"
SINGLE_ENV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --training)
            INSTALL_MODE="training"
            shift
            ;;
        --analysis)
            INSTALL_MODE="analysis"
            shift
            ;;
        --env)
            INSTALL_MODE="single"
            SINGLE_ENV="$2"
            shift 2
            ;;
        --check)
            check_conda
            check_envs
            exit 0
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主逻辑
main() {
    print_header "Energy DL 环境安装"
    echo "项目根目录: $PROJECT_ROOT"
    echo ""

    check_conda

    case "$INSTALL_MODE" in
        all)
            install_all
            ;;
        training)
            install_training
            ;;
        analysis)
            install_analysis
            ;;
        single)
            if [ -z "$SINGLE_ENV" ]; then
                print_error "--env 需要指定环境名称"
                exit 1
            fi

            # 检查是否是 conda 环境
            is_conda=false
            for entry in "${CONDA_ENVS[@]}" "${ANALYSIS_ENVS[@]}"; do
                env_name="${entry%%:*}"
                if [ "$env_name" = "$SINGLE_ENV" ]; then
                    is_conda=true
                    break
                fi
            done

            if [ "$is_conda" = true ]; then
                install_conda_env "$SINGLE_ENV"
            else
                # 检查是否是 venv
                for entry in "${VENV_ENVS[@]}"; do
                    path="${entry%%:*}"
                    rest="${entry#*:}"
                    env_name="${rest%%:*}"
                    if [ "$env_name" = "$SINGLE_ENV" ]; then
                        install_venv_env "$path" "$env_name"
                        exit 0
                    fi
                done
                print_error "未知环境: $SINGLE_ENV"
                exit 1
            fi
            ;;
    esac

    echo ""
    print_header "安装完成"
    print_info "使用以下命令检查环境状态:"
    echo "  $0 --check"
}

main
