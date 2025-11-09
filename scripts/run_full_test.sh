#!/bin/bash
# Full Test Run Script
# Runs complete training test for all repositories
# Usage: sudo ./scripts/run_full_test.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to script directory
cd "$(dirname "$0")/.."

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Full Test Run Starting${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Configuration: settings/full_test_run.json"
echo "Models to train:"
echo "  1. examples/mnist_cnn (10 epochs) - Quick validation"
echo "  2. MRT-OAST/default (10 epochs)"
echo "  3. bug-localization-by-dnn-and-rvsm/default (10 epochs)"
echo "  4. VulBERTa/mlp (10 epochs)"
echo "  5. Person_reID_baseline_pytorch/densenet121 (60 epochs)"
echo "  6. pytorch_resnet_cifar10/resnet20 (200 epochs) - Longest"
echo ""
echo -e "${YELLOW}Estimated total time: Several hours${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run with sudo${NC}"
    echo "Usage: sudo ./scripts/run_full_test.sh"
    exit 1
fi

# Check if config file exists
if [ ! -f "settings/full_test_run.json" ]; then
    echo -e "${RED}Error: Configuration file not found${NC}"
    echo "Expected: settings/full_test_run.json"
    exit 1
fi

# Display start time
echo -e "${GREEN}Start time: $(date)${NC}"
echo ""

# Run the training
echo -e "${YELLOW}Starting training...${NC}"
python3 mutation.py --experiment-config settings/full_test_run.json

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Full Test Run Completed!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}End time: $(date)${NC}"
    echo ""
    echo "Results saved in: results/"
    echo ""
    echo "To view results:"
    echo "  ls -lht results/*.json | head -10"
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}Full Test Run Failed${NC}"
    echo -e "${RED}================================${NC}"
    echo "Check the error messages above"
    exit 1
fi
