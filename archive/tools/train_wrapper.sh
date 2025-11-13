#!/bin/bash
################################################################################
# Training Wrapper Script
#
# This script wraps the execution of repository-specific training scripts
# It handles:
# 1. Changing to the correct directory
# 2. Executing the training script with proper arguments
# 3. Capturing output to a log file
# 4. Returning the training script's exit code
#
# Usage:
#   ./train_wrapper.sh <repo_path> <train_script> <log_file> [train_args...]
#
# Example:
#   ./train_wrapper.sh repos/pytorch_resnet_cifar10 ./train.sh training.log -n resnet20 --lr 0.1
################################################################################

set -e

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <repo_path> <train_script> <log_file> [train_args...]" >&2
    exit 1
fi

REPO_PATH="$1"
TRAIN_SCRIPT="$2"
LOG_FILE="$3"
shift 3
TRAIN_ARGS="$@"

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_FULL_PATH="$PROJECT_ROOT/$REPO_PATH"
LOG_FULL_PATH="$PROJECT_ROOT/$LOG_FILE"

echo "[Train Wrapper] Repository: $REPO_PATH"
echo "[Train Wrapper] Training script: $TRAIN_SCRIPT"
echo "[Train Wrapper] Log file: $LOG_FILE"
echo "[Train Wrapper] Arguments: $TRAIN_ARGS"

# Check if repository exists
if [ ! -d "$REPO_FULL_PATH" ]; then
    echo "[Train Wrapper] ERROR: Repository not found: $REPO_FULL_PATH" >&2
    exit 1
fi

# Check if training script exists
if [ ! -f "$REPO_FULL_PATH/$TRAIN_SCRIPT" ]; then
    echo "[Train Wrapper] ERROR: Training script not found: $REPO_FULL_PATH/$TRAIN_SCRIPT" >&2
    exit 1
fi

# Change to repository directory
cd "$REPO_FULL_PATH"
echo "[Train Wrapper] Changed to directory: $(pwd)"

# Make training script executable if needed
if [ ! -x "$TRAIN_SCRIPT" ]; then
    chmod +x "$TRAIN_SCRIPT"
fi

# Execute training script with arguments and capture output
echo "[Train Wrapper] Starting training..."
echo "[Train Wrapper] Command: $TRAIN_SCRIPT $TRAIN_ARGS"
echo "========================================"

# Execute and capture exit code
set +e
if [ -n "$TRAIN_ARGS" ]; then
    $TRAIN_SCRIPT $TRAIN_ARGS 2>&1 | tee "$LOG_FULL_PATH"
    EXIT_CODE=${PIPESTATUS[0]}
else
    $TRAIN_SCRIPT 2>&1 | tee "$LOG_FULL_PATH"
    EXIT_CODE=${PIPESTATUS[0]}
fi
set -e

echo "========================================"
echo "[Train Wrapper] Training finished with exit code: $EXIT_CODE"

# Return to original directory
cd "$PROJECT_ROOT"

exit $EXIT_CODE
