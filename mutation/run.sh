#!/bin/bash
################################################################################
# Training Wrapper Script with Integrated Energy Monitoring
#
# This script wraps the execution of repository-specific training scripts
# It handles:
# 1. Changing to the correct directory
# 2. Executing the training script with proper arguments
# 3. Integrated energy monitoring (CPU via perf, GPU via nvidia-smi)
# 4. Capturing output to a log file
# 5. Returning the training script's exit code
#
# Usage:
#   ./train_wrapper.sh <repo_path> <train_script> <log_file> <energy_dir> [train_args...]
#
# Example:
#   ./train_wrapper.sh repos/pytorch_resnet_cifar10 ./train.sh training.log results/energy_exp1 -n resnet20 --lr 0.1
################################################################################

set -e

# Check arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <repo_path> <train_script> <log_file> <energy_dir> [train_args...]" >&2
    exit 1
fi

REPO_PATH="$1"
TRAIN_SCRIPT="$2"
LOG_FILE="$3"
ENERGY_DIR="$4"
shift 4
TRAIN_ARGS="$@"

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_FULL_PATH="$PROJECT_ROOT/$REPO_PATH"
LOG_FULL_PATH="$PROJECT_ROOT/$LOG_FILE"
ENERGY_FULL_PATH="$PROJECT_ROOT/$ENERGY_DIR"

# Create energy directory
mkdir -p "$ENERGY_FULL_PATH"

# Define energy output files
CPU_ENERGY_RAW="$ENERGY_FULL_PATH/cpu_energy_raw.txt"
CPU_ENERGY_SUMMARY="$ENERGY_FULL_PATH/cpu_energy.txt"
GPU_POWER_CSV="$ENERGY_FULL_PATH/gpu_power.csv"
GPU_TEMPERATURE_CSV="$ENERGY_FULL_PATH/gpu_temperature.csv"
GPU_UTILIZATION_CSV="$ENERGY_FULL_PATH/gpu_utilization.csv"

echo "[Train Wrapper] Repository: $REPO_PATH"
echo "[Train Wrapper] Training script: $TRAIN_SCRIPT"
echo "[Train Wrapper] Log file: $LOG_FILE"
echo "[Train Wrapper] Energy directory: $ENERGY_DIR"
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

# Function to cleanup monitoring processes
cleanup_monitoring() {
    if [ -n "$GPU_MONITOR_PID" ] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        echo "[Train Wrapper] Stopping GPU monitoring..."
        kill -TERM "$GPU_MONITOR_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}

trap cleanup_monitoring EXIT INT TERM

# Start GPU monitoring (if nvidia-smi is available)
GPU_MONITOR_PID=""
if command -v nvidia-smi &> /dev/null; then
    echo "[Train Wrapper] Starting GPU monitoring..."

    # Write CSV headers
    echo "timestamp,power_draw_w" > "$GPU_POWER_CSV"
    echo "timestamp,gpu_temp_c,memory_temp_c" > "$GPU_TEMPERATURE_CSV"
    echo "timestamp,gpu_util_percent,memory_util_percent" > "$GPU_UTILIZATION_CSV"

    # Start background monitoring
    (
        while true; do
            TIMESTAMP=$(date +%s)

            # Query all metrics at once for consistency
            METRICS=$(nvidia-smi --query-gpu=power.draw,temperature.gpu,temperature.memory,utilization.gpu,utilization.memory \
                      --format=csv,noheader,nounits 2>/dev/null)

            if [ -n "$METRICS" ]; then
                # Parse metrics (format: power, gpu_temp, mem_temp, gpu_util, mem_util)
                POWER=$(echo "$METRICS" | cut -d',' -f1 | tr -d ' ')
                GPU_TEMP=$(echo "$METRICS" | cut -d',' -f2 | tr -d ' ')
                MEM_TEMP=$(echo "$METRICS" | cut -d',' -f3 | tr -d ' ')
                GPU_UTIL=$(echo "$METRICS" | cut -d',' -f4 | tr -d ' ')
                MEM_UTIL=$(echo "$METRICS" | cut -d',' -f5 | tr -d ' ')

                echo "$TIMESTAMP,$POWER" >> "$GPU_POWER_CSV"
                echo "$TIMESTAMP,$GPU_TEMP,$MEM_TEMP" >> "$GPU_TEMPERATURE_CSV"
                echo "$TIMESTAMP,$GPU_UTIL,$MEM_UTIL" >> "$GPU_UTILIZATION_CSV"
            fi

            sleep 1
        done
    ) &
    GPU_MONITOR_PID=$!
    echo "[Train Wrapper] GPU monitoring PID: $GPU_MONITOR_PID"
else
    echo "[Train Wrapper] WARNING: nvidia-smi not found, skipping GPU monitoring"
fi

# Execute training script with arguments and capture output
# Wrap with perf for CPU energy monitoring (direct wrapping approach from run.sh)
echo "[Train Wrapper] Starting training with integrated energy monitoring..."
echo "[Train Wrapper] Command: $TRAIN_SCRIPT $TRAIN_ARGS"
echo "========================================"

# Execute and capture exit code
set +e
if [ -n "$TRAIN_ARGS" ]; then
    perf stat -e power/energy-pkg/,power/energy-ram/ -o "$CPU_ENERGY_RAW" \
        $TRAIN_SCRIPT $TRAIN_ARGS 2>&1 | tee "$LOG_FULL_PATH"
    EXIT_CODE=${PIPESTATUS[0]}
else
    perf stat -e power/energy-pkg/,power/energy-ram/ -o "$CPU_ENERGY_RAW" \
        $TRAIN_SCRIPT 2>&1 | tee "$LOG_FULL_PATH"
    EXIT_CODE=${PIPESTATUS[0]}
fi
set -e

echo "========================================"
echo "[Train Wrapper] Training finished with exit code: $EXIT_CODE"

# Stop GPU monitoring
cleanup_monitoring

# Process CPU energy data
if [ -f "$CPU_ENERGY_RAW" ]; then
    echo "[Train Wrapper] Processing CPU energy data..."

    # Extract energy values from perf output
    # perf stat outputs the total energy consumed during command execution
    PKG_ENERGY=$(grep "power/energy-pkg/" "$CPU_ENERGY_RAW" | awk '{print $1}' | tr -d ',')
    RAM_ENERGY=$(grep "power/energy-ram/" "$CPU_ENERGY_RAW" | awk '{print $1}' | tr -d ',')

    # Convert to Joules (perf reports in Joules)
    # Write summary
    {
        echo "CPU Energy Consumption Summary"
        echo "=============================="
        echo "Package Energy (Joules): ${PKG_ENERGY:-0}"
        echo "RAM Energy (Joules): ${RAM_ENERGY:-0}"
        echo "Total CPU Energy (Joules): $(echo "${PKG_ENERGY:-0} + ${RAM_ENERGY:-0}" | bc -l)"
        echo ""
        echo "Note: Measured using perf stat with direct command wrapping"
        echo "This method provides more accurate results than interval sampling"
    } > "$CPU_ENERGY_SUMMARY"

    echo "[Train Wrapper] CPU energy saved to: $CPU_ENERGY_SUMMARY"
fi

# Process GPU data
if [ -f "$GPU_POWER_CSV" ] && [ $(wc -l < "$GPU_POWER_CSV") -gt 1 ]; then
    echo "[Train Wrapper] GPU monitoring data saved to: $ENERGY_FULL_PATH"
fi

# Return to original directory
cd "$PROJECT_ROOT"

echo "[Train Wrapper] Energy monitoring completed"
exit $EXIT_CODE
