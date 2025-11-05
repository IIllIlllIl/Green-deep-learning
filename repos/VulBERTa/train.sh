#!/bin/bash
################################################################################
# VulBERTa Training Script
#
# This script trains VulBERTa models (MLP or CNN) on vulnerability detection datasets
#
# Usage: ./train.sh -n model_name -d dataset [options]
#
# Required arguments:
#   -n, --model_name      Model architecture: mlp or cnn
#   -d, --dataset         Dataset: devign, draper, reveal, mvd, vuldeepecker, d2a
#
# Optional arguments:
#   --batch_size          Batch size (default: 2 for MLP, 128 for CNN)
#   --epochs              Number of epochs (default: 10 for MLP, 20 for CNN)
#   --learning_rate       Learning rate (default: 3e-05 for MLP, 0.0005 for CNN)
#   --weight_decay        Weight decay (default: 0.0)
#   --seed                Random seed (default: 42 for MLP, 1234 for CNN)
#   --fp16                Use mixed precision training
#   -h, --help            Show this help message
#
# Example:
#   ./train.sh -n mlp -d devign
#   ./train.sh -n mlp -d devign --batch_size 2 --epochs 5
################################################################################

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to show usage
show_help() {
    head -n 25 "$0" | grep "^#" | sed 's/^# \?//'
    exit 0
}

# Parse command line arguments
# Set default values if no arguments provided
if [[ $# -eq 0 ]]; then
    set -- -n mlp -d d2a --batch_size 2
fi

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
fi

# Find Python from vulberta conda environment
CONDA_ENV_NAME="vulberta"
CONDA_BASE="/home/green/miniconda3"
PYTHON_PATH="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    print_error "Python not found at $PYTHON_PATH"
    print_info "Trying to find Python in conda environments..."

    # Try to find any available Python
    for env_path in "${CONDA_BASE}/envs/"*/bin/python; do
        if [ -f "$env_path" ]; then
            print_warning "Found Python at: $env_path"
            PYTHON_PATH="$env_path"
            break
        fi
    done

    if [ ! -f "$PYTHON_PATH" ]; then
        print_error "No suitable Python found. Please check your conda installation."
        exit 1
    fi
fi

print_info "Using Python: $PYTHON_PATH"
print_info "Python version: $($PYTHON_PATH --version)"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    print_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    print_warning "nvidia-smi not found. Running on CPU."
fi

# Record start time
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

print_info "Training started at: $START_TIME_STR"
print_info "Working directory: $SCRIPT_DIR"

# Run the Python training script
print_info "Executing training script..."
echo "=================================================================================="

# Set environment variables to fix memory alignment issues
export MALLOC_CHECK_=0
export MALLOC_ARENA_MAX=1

# Execute with all arguments passed through
"$PYTHON_PATH" train_vulberta.py "$@"
EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((END_TIME - START_TIME))

# Convert duration to human readable format
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=================================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    print_success "Training completed successfully!"
else
    print_error "Training failed with exit code $EXIT_CODE"
fi

print_info "End time: $END_TIME_STR"
print_info "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=================================================================================="

exit $EXIT_CODE
