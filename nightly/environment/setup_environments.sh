#!/bin/bash
################################################################################
# Environment Setup Script
#
# This script helps you create all conda environments for the mutation framework
# Usage: ./setup_environments.sh [options]
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CREATED=0
SKIPPED=0
FAILED=0

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

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

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_header "Conda Environment Setup"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "conda not found. Please install Miniconda or Anaconda first."
    echo "Download: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_success "conda found: $(conda --version)"
echo ""

# Parse arguments
CREATE_ALL=false
FORCE=false
USE_MAMBA=false
SELECTED_ENVS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            CREATE_ALL=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --mamba)
            USE_MAMBA=true
            shift
            ;;
        --env)
            SELECTED_ENVS+=("$2")
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [options]

Options:
    --all           Create all environments
    --force         Remove and recreate existing environments
    --mamba         Use mamba instead of conda (faster)
    --env NAME      Create specific environment (can be used multiple times)
    -h, --help      Show this help message

Examples:
    # Create all environments
    $0 --all

    # Create only mutation_runner and pytorch_resnet_cifar10
    $0 --env mutation_runner --env pytorch_resnet_cifar10

    # Force recreate all environments using mamba
    $0 --all --force --mamba
EOF
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if mamba is requested but not available
if $USE_MAMBA; then
    if ! command -v mamba &> /dev/null; then
        print_warning "mamba not found, falling back to conda"
        print_info "To install mamba: conda install mamba -n base -c conda-forge"
        USE_MAMBA=false
    else
        print_success "Using mamba for faster environment creation"
    fi
fi

# Set conda/mamba command
if $USE_MAMBA; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi

# List of environments
ENVIRONMENTS=(
    "mutation_runner"
    "pytorch_resnet_cifar10"
    "vulberta"
    "reid_baseline"
    "mrt-oast"
    "dnn_rvsm"
    "pytorch_examples"
)

# If no specific environments selected, ask user
if ! $CREATE_ALL && [ ${#SELECTED_ENVS[@]} -eq 0 ]; then
    print_info "No environments specified. Available environments:"
    echo ""
    for i in "${!ENVIRONMENTS[@]}"; do
        env="${ENVIRONMENTS[$i]}"
        if conda env list | grep -q "^$env "; then
            echo "  $((i+1)). $env (already exists)"
        else
            echo "  $((i+1)). $env"
        fi
    done
    echo ""
    read -p "Enter environment numbers to create (comma-separated), or 'all': " selection

    if [ "$selection" == "all" ]; then
        CREATE_ALL=true
    else
        IFS=',' read -ra NUMS <<< "$selection"
        for num in "${NUMS[@]}"; do
            num=$(echo "$num" | xargs)  # trim whitespace
            if [ "$num" -ge 1 ] && [ "$num" -le "${#ENVIRONMENTS[@]}" ]; then
                SELECTED_ENVS+=("${ENVIRONMENTS[$((num-1))]}")
            fi
        done
    fi
fi

# Determine which environments to create
if $CREATE_ALL; then
    TO_CREATE=("${ENVIRONMENTS[@]}")
else
    TO_CREATE=("${SELECTED_ENVS[@]}")
fi

if [ ${#TO_CREATE[@]} -eq 0 ]; then
    print_warning "No environments selected"
    exit 0
fi

echo ""
print_header "Environments to Create"
for env in "${TO_CREATE[@]}"; do
    echo "  - $env"
done
echo ""

if ! $FORCE; then
    read -p "Proceed with environment creation? (y/n): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
fi

echo ""
print_header "Creating Environments"

# Create each environment
for env in "${TO_CREATE[@]}"; do
    env_file="$SCRIPT_DIR/${env}.yml"

    if [ ! -f "$env_file" ]; then
        print_error "Environment file not found: $env_file"
        ((FAILED++))
        continue
    fi

    echo ""
    print_info "Processing: $env"

    # Check if environment already exists
    if conda env list | grep -q "^$env "; then
        if $FORCE; then
            print_warning "Removing existing environment: $env"
            conda env remove -n "$env" -y
        else
            print_warning "Environment already exists: $env"
            print_info "Use --force to recreate, or update with:"
            echo "  conda env update -n $env -f $env_file --prune"
            ((SKIPPED++))
            continue
        fi
    fi

    # Create environment
    print_info "Creating environment from: $env_file"
    if $CONDA_CMD env create -f "$env_file"; then
        print_success "Created: $env"
        ((CREATED++))
    else
        print_error "Failed to create: $env"
        ((FAILED++))
    fi
done

# Summary
echo ""
print_header "Summary"
echo -e "Created:  ${GREEN}$CREATED${NC}"
echo -e "Skipped:  ${YELLOW}$SKIPPED${NC}"
echo -e "Failed:   ${RED}$FAILED${NC}"
echo ""

if [ $CREATED -gt 0 ]; then
    print_success "Environment setup completed!"
    echo ""
    print_info "To activate an environment:"
    echo "  conda activate <environment_name>"
    echo ""
    print_info "To verify an environment:"
    echo "  conda activate <environment_name>"
    echo "  python --version"
    echo "  conda list"
fi

if [ $FAILED -gt 0 ]; then
    echo ""
    print_warning "Some environments failed to create. Common issues:"
    echo "  - Package version conflicts: Try editing the .yml file to use version ranges"
    echo "  - CUDA mismatch: Install PyTorch for your CUDA version after creating the env"
    echo "  - Network issues: Check your internet connection or use a mirror"
    echo ""
    print_info "For help, see: environment/README.md"
fi

exit 0
