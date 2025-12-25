#!/bin/bash
################################################################################
# Environment Files Validation Script
#
# Validates that all .yml files are properly formatted
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Environment Files Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

TOTAL=0
VALID=0
INVALID=0

# Check each .yml file
for yml_file in "$SCRIPT_DIR"/*.yml; do
    if [ -f "$yml_file" ]; then
        ((TOTAL++))
        filename=$(basename "$yml_file")

        echo -n "Checking $filename ... "

        # Validate YAML syntax using Python
        if python3 -c "import yaml; yaml.safe_load(open('$yml_file'))" 2>/dev/null; then
            # Check if name field exists
            env_name=$(python3 -c "import yaml; print(yaml.safe_load(open('$yml_file')).get('name', ''))" 2>/dev/null)

            if [ -n "$env_name" ]; then
                echo -e "${GREEN}✓ Valid${NC} (name: $env_name)"
                ((VALID++))
            else
                echo -e "${YELLOW}⚠ Warning${NC} (missing 'name' field)"
                ((VALID++))
            fi
        else
            echo -e "${RED}✗ Invalid YAML${NC}"
            ((INVALID++))
        fi
    fi
done

echo ""
echo -e "Total: ${BLUE}$TOTAL${NC}  |  Valid: ${GREEN}$VALID${NC}  |  Invalid: ${RED}$INVALID${NC}"
echo ""

if [ $INVALID -eq 0 ]; then
    echo -e "${GREEN}✓ All environment files are valid!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some environment files have errors${NC}"
    exit 1
fi
