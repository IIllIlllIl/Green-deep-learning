#!/bin/bash

# Bug Localization Model Training Script
# This script trains DNN or RVSM models for bug localization
# Usage: ./train.sh -n model_name [options]

set -e  # Exit on error

# Conda environment configuration
CONDA_ENV="dnn_rvsm"
CONDA_BASE="/home/green/miniconda3"
PYTHON_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"

# Check if Python exists in conda environment
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python not found in conda environment '${CONDA_ENV}'"
    echo "Expected path: ${PYTHON_PATH}"
    echo "Please ensure the conda environment is set up correctly."
    exit 1
fi

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default parameters (matching repository defaults)
MODEL_NAME="dnn"
KFOLD=10
HIDDEN_SIZES="300"  # Original repository default (can use 200 for better GPU performance)
ALPHA=1e-5
MAX_ITER=10000
N_ITER_NO_CHANGE=30
SOLVER="sgd"
N_JOBS=-2
SEED=""  # 空字符串表示不设置seed（保持原始随机行为）

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --kfold)
            KFOLD="$2"
            shift 2
            ;;
        --hidden_sizes)
            HIDDEN_SIZES="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --max_iter)
            MAX_ITER="$2"
            shift 2
            ;;
        --n_iter_no_change)
            N_ITER_NO_CHANGE="$2"
            shift 2
            ;;
        --solver)
            SOLVER="$2"
            shift 2
            ;;
        --n_jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-n MODEL_NAME] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  -n, --model_name MODEL        Model to train: dnn or rvsm (default: dnn)"
            echo ""
            echo "Optional arguments (DNN only):"
            echo "  --kfold N                     Number of folds (default: 10)"
            echo "  --hidden_sizes SIZES          Hidden layer sizes (default: 200)"
            echo "                                Example: --hidden_sizes \"300 200\" for two layers"
            echo "  --alpha ALPHA                 L2 penalty parameter (default: 1e-5)"
            echo "  --max_iter N                  Maximum iterations (default: 10000)"
            echo "  --n_iter_no_change N          Early stopping patience (default: 30)"
            echo "  --solver SOLVER               Optimizer: sgd, adam, lbfgs (default: sgd)"
            echo "  --n_jobs N                    Parallel jobs (default: -2, all cores but one)"
            echo "  --seed SEED                   Random seed for reproducibility (default: not set)"
            echo ""
            echo "Examples:"
            echo "  $0 -n dnn                                    # Train DNN with default params"
            echo "  $0 -n rvsm                                   # Train RVSM baseline"
            echo "  $0 -n dnn --hidden_sizes \"200\" --kfold 5   # Custom DNN config"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate model name
if [ "$MODEL_NAME" != "dnn" ] && [ "$MODEL_NAME" != "rvsm" ]; then
    echo "Error: Invalid model name '$MODEL_NAME'. Must be 'dnn' or 'rvsm'"
    exit 1
fi

# Build Python command
PYTHON_CMD="$PYTHON_PATH train_wrapper.py -n $MODEL_NAME"

if [ "$MODEL_NAME" = "dnn" ]; then
    PYTHON_CMD="$PYTHON_CMD --kfold $KFOLD"
    PYTHON_CMD="$PYTHON_CMD --hidden_sizes $HIDDEN_SIZES"
    PYTHON_CMD="$PYTHON_CMD --alpha $ALPHA"
    PYTHON_CMD="$PYTHON_CMD --max_iter $MAX_ITER"
    PYTHON_CMD="$PYTHON_CMD --n_iter_no_change $N_ITER_NO_CHANGE"
    PYTHON_CMD="$PYTHON_CMD --solver $SOLVER"
    PYTHON_CMD="$PYTHON_CMD --n_jobs $N_JOBS"
    [ -n "$SEED" ] && PYTHON_CMD="$PYTHON_CMD --seed $SEED"
fi

# Display configuration
echo "========================================"
echo "Bug Localization Training Configuration"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Python: $PYTHON_PATH"
echo "Working directory: $SCRIPT_DIR"
if [ "$MODEL_NAME" = "dnn" ]; then
    echo "K-fold: $KFOLD"
    echo "Hidden sizes: $HIDDEN_SIZES"
    echo "Alpha: $ALPHA"
    echo "Max iterations: $MAX_ITER"
    echo "Early stopping: $N_ITER_NO_CHANGE"
    echo "Solver: $SOLVER"
    echo "Parallel jobs: $N_JOBS"
    echo "Random seed: $([ -n "$SEED" ] && echo "$SEED" || echo 'Not set (original non-deterministic behavior)')"
fi
echo "========================================"
echo ""

# Execute training
$PYTHON_CMD

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
else
    echo ""
    echo "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
