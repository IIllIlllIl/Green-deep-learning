#!/bin/bash

# HRNet18 Validation Test Script
# This script runs hrnet18 in both sequential and parallel modes to validate the fixes

echo "======================================================================================================"
echo "HRNet18 Validation Test"
echo "======================================================================================================"
echo ""
echo "This will run 2 experiments:"
echo "  1. Sequential hrnet18 (5 epochs) - to verify offline model loading"
echo "  2. Parallel hrnet18 (5 epochs) + mnist_ff background - to verify parallel fix"
echo ""
echo "Estimated time: 30-40 minutes"
echo "======================================================================================================"
echo ""

# Set offline mode
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "✓ Offline mode enabled (HF_HUB_OFFLINE=1)"
echo ""

# Navigate to project directory
cd /home/green/energy_dl/nightly

# Run experiments
echo "Starting experiments..."
echo ""

sudo -E python3 mutation.py settings/test_hrnet18_validation.json 2>&1 | tee /tmp/hrnet18_validation_$(date +%Y%m%d_%H%M%S).log

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================================================"
    echo "✅ Experiments completed successfully!"
    echo "======================================================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Check results in the latest run_* directory"
    echo "  2. Verify hrnet18 training logs have no SSL errors"
    echo "  3. Verify parallel directory structure is correct"
    echo ""
else
    echo ""
    echo "======================================================================================================"
    echo "❌ Experiments failed!"
    echo "======================================================================================================"
    echo ""
    echo "Check the log file for errors"
fi
