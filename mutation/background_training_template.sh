#!/bin/bash
################################################################################
# Background Training Template Script
#
# Purpose: Reusable script for running background training loops in parallel
#          experiments. This script accepts parameters at runtime instead of
#          embedding them in generated scripts.
#
# Usage:
#   ./background_training_template.sh <repo_path> <train_script> <train_args> \
#                                     <log_dir> [restart_delay]
#
# Arguments:
#   $1 - REPO_PATH: Absolute path to the repository directory
#   $2 - TRAIN_SCRIPT: Name of the training script (e.g., train.py)
#   $3 - TRAIN_ARGS: Training arguments as a single string
#   $4 - LOG_DIR: Directory to store run logs
#   $5 - RESTART_DELAY (optional): Seconds to wait between runs (default: 2)
#
# Example:
#   ./background_training_template.sh \
#       /home/user/VulBERTa \
#       train.py \
#       "--epochs 10 --lr 0.001" \
#       /home/user/results/background_logs_xxx \
#       2
#
# Created: 2025-11-12
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Validate required arguments
if [ $# -lt 4 ]; then
    echo "Error: Insufficient arguments" >&2
    echo "Usage: $0 <repo_path> <train_script> <train_args> <log_dir> [restart_delay]" >&2
    echo "" >&2
    echo "Arguments:" >&2
    echo "  repo_path      - Absolute path to repository directory" >&2
    echo "  train_script   - Training script name (e.g., train.py)" >&2
    echo "  train_args     - Training arguments (quoted string)" >&2
    echo "  log_dir        - Directory for run logs" >&2
    echo "  restart_delay  - Optional: seconds between runs (default: 2)" >&2
    exit 1
fi

# Parse arguments
REPO_PATH="$1"
TRAIN_SCRIPT="$2"
TRAIN_ARGS="$3"
LOG_DIR="$4"
RESTART_DELAY="${5:-2}"

# Validate repository path
if [ ! -d "$REPO_PATH" ]; then
    echo "Error: Repository path does not exist: $REPO_PATH" >&2
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Change to repository directory
cd "$REPO_PATH" || {
    echo "Error: Failed to change to repository directory: $REPO_PATH" >&2
    exit 1
}

# Verify training script exists
if [ ! -f "$TRAIN_SCRIPT" ] && [ ! -x "$TRAIN_SCRIPT" ]; then
    echo "Warning: Training script may not exist or is not executable: $TRAIN_SCRIPT" >&2
fi

# Print startup information
echo "================================================================================"
echo "Background Training Loop Started"
echo "================================================================================"
echo "Timestamp:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repository:     $REPO_PATH"
echo "Train Script:   $TRAIN_SCRIPT"
echo "Arguments:      $TRAIN_ARGS"
echo "Log Directory:  $LOG_DIR"
echo "Restart Delay:  ${RESTART_DELAY}s"
echo "PID:            $$"
echo "PGID:           $(ps -o pgid= -p $$)"
echo "================================================================================"
echo ""

# Initialize run counter
run_count=0

# Infinite training loop
while true; do
    run_count=$((run_count + 1))
    run_log="$LOG_DIR/run_$run_count.log"

    echo "[Background] Run #$run_count starting at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Log file: $run_log"

    # Run training and capture output to separate log file
    # Note: We use eval to properly handle quoted arguments in TRAIN_ARGS
    # Prepend ./ if script doesn't contain a path separator
    if [[ "$TRAIN_SCRIPT" != */* ]]; then
        eval "./$TRAIN_SCRIPT $TRAIN_ARGS" > "$run_log" 2>&1
    else
        eval "$TRAIN_SCRIPT $TRAIN_ARGS" > "$run_log" 2>&1
    fi

    exit_code=$?
    echo "[Background] Run #$run_count finished at $(date '+%Y-%m-%d %H:%M:%S') with exit code $exit_code"

    # Log exit code for debugging
    echo "Exit Code: $exit_code" >> "$run_log"

    # Brief sleep to avoid excessive restarts
    if [ "$exit_code" -ne 0 ]; then
        echo "[Background] Warning: Training failed with exit code $exit_code"
        echo "[Background] Waiting ${RESTART_DELAY}s before retry..."
    fi

    sleep "$RESTART_DELAY"
done
