#!/bin/bash
################################################################################
# Sequential Experiments Runner
# Purpose: Run multiple experiment configurations sequentially
# Usage: ./run_sequential_experiments.sh
################################################################################

set -e  # Exit on error (remove this if you want to continue even on failure)

# Change to project directory
cd "$(dirname "$0")/.."

# Log file
LOG_FILE="sequential_experiments_$(date +%Y%m%d_%H%M%S).log"

# Function to run experiment with logging
run_experiment() {
    local config_file=$1
    local experiment_name=$(basename "$config_file" .json)

    echo "================================================================================"
    echo "Starting experiment: $experiment_name"
    echo "Config: $config_file"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"

    # Run the experiment
    sudo -E python3 mutation.py -ec "$config_file" 2>&1 | tee -a "$LOG_FILE"

    local exit_code=${PIPESTATUS[0]}

    echo ""
    echo "================================================================================"
    echo "Completed: $experiment_name"
    echo "Exit code: $exit_code"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    echo ""

    return $exit_code
}

# Main execution
echo "Sequential Experiments Runner - Started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file: $LOG_FILE"
echo ""

# Experiment 1: Person ReID Dropout Boundary Test (~6.5 hours)
run_experiment "settings/person_reid_dropout_boundary_test.json"

# Optional: Add a break between experiments
echo "Waiting 60 seconds before next experiment..."
sleep 60

# Experiment 2: Mutation Validation 1x
run_experiment "settings/mutation_validation_1x.json"

echo ""
echo "================================================================================"
echo "All experiments completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
