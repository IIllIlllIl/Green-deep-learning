#!/bin/bash
################################################################################
# Test Runner for Mutation Framework
#
# This script tests all functionality of the mutation-based training system
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Get the directory where this script is located
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TEST_DIR/.." && pwd)"

print_header "Mutation Framework Testing Suite"
echo "Project Root: $PROJECT_ROOT"
echo "Test Directory: $TEST_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Clean up previous test results
print_info "Cleaning up previous test results..."
rm -rf results/test_* results/energy_* test/results 2>/dev/null || true

################################################################################
# Test 1: Check if all required files exist
################################################################################
print_header "Test 1: File Existence Check"

files_to_check=(
    "mutation.py"
    "scripts/run.sh"
    "config/models_config.json"
    "test/test_config.json"
    "test/repos/test_repo/train.sh"
    "test/governor.sh"
    "test/validate_energy_monitoring.sh"
)

for file in "${files_to_check[@]}"; do
    print_test "Checking: $file"
    if [ -f "$file" ]; then
        print_pass "$file exists"
    else
        print_fail "$file not found"
    fi
done

################################################################################
# Test 2: Check script executability
################################################################################
print_header "Test 2: Script Executability Check"

scripts_to_check=(
    "mutation.py"
    "scripts/run.sh"
    "test/repos/test_repo/train.sh"
    "test/governor.sh"
    "test/validate_energy_monitoring.sh"
)

for script in "${scripts_to_check[@]}"; do
    print_test "Checking executability: $script"
    if [ -x "$script" ]; then
        print_pass "$script is executable"
    else
        print_fail "$script is not executable"
    fi
done

################################################################################
# Test 3: Configuration File Validation
################################################################################
print_header "Test 3: Configuration File Validation"

print_test "Validating test_config.json"
if python3 -c "import json; json.load(open('test/test_config.json'))" 2>/dev/null; then
    print_pass "test_config.json is valid JSON"
else
    print_fail "test_config.json is invalid JSON"
fi

print_test "Validating models_config.json"
if python3 -c "import json; json.load(open('config/models_config.json'))" 2>/dev/null; then
    print_pass "models_config.json is valid JSON"
else
    print_fail "models_config.json is invalid JSON"
fi

################################################################################
# Test 4: Mock Training Script Test
################################################################################
print_header "Test 4: Mock Training Script Test"

print_test "Running mock training script directly"
if cd test/repos/test_repo && ./train.sh -n model_a --epochs 3 --lr 0.001 > /tmp/test_train_output.txt 2>&1; then
    cd "$PROJECT_ROOT"
    if grep -q "Training SUCCESS" /tmp/test_train_output.txt; then
        print_pass "Mock training script executed successfully"
    else
        print_fail "Mock training script did not report success"
        cat /tmp/test_train_output.txt
    fi
else
    cd "$PROJECT_ROOT"
    print_fail "Mock training script failed"
    cat /tmp/test_train_output.txt
fi

################################################################################
# Test 5: Run Script Test (with integrated energy monitoring)
################################################################################
print_header "Test 5: Run Script Test"

print_test "Testing run.sh with integrated energy monitoring"
mkdir -p results/test_energy
if ./scripts/run.sh test/repos/test_repo ./train.sh results/test_run.log results/test_energy -n model_a --epochs 2 > /tmp/test_run_output.txt 2>&1; then
    if [ -f "results/test_run.log" ]; then
        print_pass "Run script executed successfully"
        # Check if energy files were created
        if [ -f "results/test_energy/cpu_energy.txt" ]; then
            print_pass "Energy monitoring created CPU energy file"
        fi
    else
        print_fail "Run script did not create log file"
    fi
else
    print_fail "Run script failed"
    cat /tmp/test_run_output.txt
fi

################################################################################
# Test 6: Mock Governor Script Test
################################################################################
print_header "Test 6: Mock Governor Script Test"

print_test "Testing mock governor script"
if ./test/governor.sh performance > /tmp/test_governor.txt 2>&1; then
    if grep -q "performance" /tmp/test_governor.txt; then
        print_pass "Mock governor script works"
    else
        print_fail "Mock governor script output unexpected"
    fi
else
    print_fail "Mock governor script failed"
fi

################################################################################
# Test 7: Mutation Runner --list Test
################################################################################
print_header "Test 7: Mutation Runner --list Test"

print_test "Testing mutation.py --list with test config"
if python3 mutation.py --list --config test/test_config.json > /tmp/test_list.txt 2>&1; then
    if grep -q "test_repo" /tmp/test_list.txt && grep -q "model_a" /tmp/test_list.txt; then
        print_pass "Mutation runner --list works correctly"
    else
        print_fail "Mutation runner --list output unexpected"
        cat /tmp/test_list.txt
    fi
else
    print_fail "Mutation runner --list failed"
    cat /tmp/test_list.txt
fi

################################################################################
# Test 8: Mutation Runner Full Integration Test
################################################################################
print_header "Test 8: Full Integration Test (Mutation Runner)"

print_test "Running mutation.py with test config"
print_info "This will take approximately 15-20 seconds..."

# Copy governor.sh to project root for this test
cp test/governor.sh governor.sh.backup 2>/dev/null || true
cp test/governor.sh governor.sh

if python3 mutation.py \
    --repo test_repo \
    --model model_a \
    --mutate epochs,learning_rate,seed \
    --runs 1 \
    --config test/test_config.json \
    --max-retries 1 > /tmp/test_integration.txt 2>&1; then

    # Check if results were created
    RESULT_COUNT=$(find results -name "*.json" -type f | wc -l)
    if [ "$RESULT_COUNT" -gt 0 ]; then
        print_pass "Integration test completed successfully ($RESULT_COUNT result files created)"

        # Validate result JSON
        LATEST_RESULT=$(find results -name "*.json" -type f | sort | tail -1)
        if python3 -c "import json; data=json.load(open('$LATEST_RESULT')); assert 'experiment_id' in data; assert 'hyperparameters' in data" 2>/dev/null; then
            print_pass "Result JSON is valid and contains expected fields"
        else
            print_fail "Result JSON is invalid or missing fields"
        fi
    else
        print_fail "Integration test did not create result files"
        cat /tmp/test_integration.txt
    fi
else
    print_fail "Integration test failed"
    cat /tmp/test_integration.txt
fi

# Restore governor
rm -f governor.sh
mv governor.sh.backup governor.sh 2>/dev/null || true

################################################################################
# Test 9: Hyperparameter Mutation Test
################################################################################
print_header "Test 9: Hyperparameter Mutation Test"

print_test "Testing hyperparameter mutation logic"

cat > /tmp/test_mutation.py << 'EOF'
import sys
sys.path.insert(0, '.')
from mutation_runner import MutationRunner

runner = MutationRunner(config_path='test/test_config.json')

# Test mutation generation
mutations = runner.generate_mutations('test_repo', 'model_a', ['epochs', 'learning_rate', 'seed'], num_mutations=3)

assert len(mutations) == 3, "Should generate 3 mutations"
assert all('epochs' in m for m in mutations), "All mutations should have epochs"
assert all('learning_rate' in m for m in mutations), "All mutations should have learning_rate"
assert all('seed' in m for m in mutations), "All mutations should have seed"

# Check if values are within range
for m in mutations:
    assert 5 <= m['epochs'] <= 20, f"epochs {m['epochs']} out of range"
    assert 0.0001 <= m['learning_rate'] <= 0.01, f"learning_rate {m['learning_rate']} out of range"
    assert 0 <= m['seed'] <= 9999, f"seed {m['seed']} out of range"

print("All mutation tests passed!")
EOF

if python3 /tmp/test_mutation.py > /tmp/test_mutation_output.txt 2>&1; then
    print_pass "Hyperparameter mutation logic works correctly"
else
    print_fail "Hyperparameter mutation logic failed"
    cat /tmp/test_mutation_output.txt
fi

################################################################################
# Test Summary
################################################################################
print_header "Test Summary"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
PASS_RATE=$(echo "scale=1; $TESTS_PASSED * 100 / $TOTAL_TESTS" | bc)

echo ""
echo -e "Total Tests: ${CYAN}$TOTAL_TESTS${NC}"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo -e "Pass Rate: ${YELLOW}$PASS_RATE%${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✨ All tests passed! The mutation framework is working correctly.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please review the output above.${NC}"
    exit 1
fi
