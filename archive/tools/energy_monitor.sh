#!/bin/bash
################################################################################
# Energy Monitoring Script
#
# This script monitors energy consumption during training using:
# 1. perf stat for CPU energy (package + RAM)
# 2. nvidia-smi for GPU power draw
#
# Usage:
#   ./energy_monitor.sh <output_dir> <pid_to_monitor>
#
# The script will:
# - Monitor the specified PID
# - Output CPU energy to <output_dir>/cpu_energy.txt
# - Output GPU power to <output_dir>/gpu_power.csv
# - Stop automatically when the monitored PID exits
################################################################################

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_dir> <pid_to_monitor>" >&2
    exit 1
fi

OUTPUT_DIR="$1"
MONITOR_PID="$2"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

CPU_ENERGY_FILE="$OUTPUT_DIR/cpu_energy.txt"
GPU_POWER_FILE="$OUTPUT_DIR/gpu_power.csv"
PERF_RAW_FILE="$OUTPUT_DIR/perf_raw.txt"

echo "[Energy Monitor] Starting energy monitoring for PID: $MONITOR_PID"
echo "[Energy Monitor] Output directory: $OUTPUT_DIR"

# Function to cleanup on exit
cleanup() {
    echo "[Energy Monitor] Cleaning up monitoring processes..."
    if [ -n "$PERF_PID" ] && kill -0 "$PERF_PID" 2>/dev/null; then
        kill "$PERF_PID" 2>/dev/null || true
    fi
    if [ -n "$NVIDIA_PID" ] && kill -0 "$NVIDIA_PID" 2>/dev/null; then
        kill "$NVIDIA_PID" 2>/dev/null || true
    fi
    echo "[Energy Monitor] Monitoring stopped"
}

trap cleanup EXIT INT TERM

# Check if target PID exists
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo "[Energy Monitor] ERROR: PID $MONITOR_PID does not exist" >&2
    exit 1
fi

# Start CPU energy monitoring with perf
# We'll monitor the process and all its children
echo "[Energy Monitor] Starting CPU energy monitoring (perf)..."
(
    # Monitor until the target process exits
    while kill -0 "$MONITOR_PID" 2>/dev/null; do
        sleep 1
    done
) | perf stat -e power/energy-pkg/,power/energy-ram/ -o "$PERF_RAW_FILE" -I 1000 cat > /dev/null 2>&1 &
PERF_PID=$!

# Start GPU power monitoring with nvidia-smi (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "[Energy Monitor] Starting GPU power monitoring (nvidia-smi)..."

    # Write CSV header
    echo "timestamp,power_draw_w" > "$GPU_POWER_FILE"

    # Monitor GPU power
    (
        while kill -0 "$MONITOR_PID" 2>/dev/null; do
            TIMESTAMP=$(date +%s)
            POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
            if [ -n "$POWER" ]; then
                echo "$TIMESTAMP,$POWER" >> "$GPU_POWER_FILE"
            fi
            sleep 1
        done
    ) &
    NVIDIA_PID=$!
else
    echo "[Energy Monitor] WARNING: nvidia-smi not found, skipping GPU monitoring"
    NVIDIA_PID=""
fi

# Wait for the monitored process to exit
echo "[Energy Monitor] Monitoring energy consumption..."
while kill -0 "$MONITOR_PID" 2>/dev/null; do
    sleep 2
done

# Give monitoring processes time to flush their outputs
sleep 2

# Stop monitoring processes
cleanup

# Process perf output to extract total energy
if [ -f "$PERF_RAW_FILE" ]; then
    echo "[Energy Monitor] Processing CPU energy data..."

    # Extract energy values from perf output
    # perf stat -I outputs interval measurements, we need to sum them
    PKG_ENERGY=$(grep "power/energy-pkg/" "$PERF_RAW_FILE" | awk '{sum+=$2} END {print sum}')
    RAM_ENERGY=$(grep "power/energy-ram/" "$PERF_RAW_FILE" | awk '{sum+=$2} END {print sum}')

    # Write summary
    {
        echo "CPU Energy Consumption Summary"
        echo "=============================="
        echo "Package Energy (Joules): ${PKG_ENERGY:-0}"
        echo "RAM Energy (Joules): ${RAM_ENERGY:-0}"
        echo "Total CPU Energy (Joules): $(echo "${PKG_ENERGY:-0} + ${RAM_ENERGY:-0}" | bc -l)"
    } > "$CPU_ENERGY_FILE"

    echo "[Energy Monitor] CPU energy data saved to: $CPU_ENERGY_FILE"
fi

# Process GPU power data if available
if [ -f "$GPU_POWER_FILE" ] && [ $(wc -l < "$GPU_POWER_FILE") -gt 1 ]; then
    echo "[Energy Monitor] GPU power data saved to: $GPU_POWER_FILE"
fi

echo "[Energy Monitor] Energy monitoring completed"
exit 0
