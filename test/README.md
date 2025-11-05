# Mutation Framework Testing Suite

This directory contains a complete testing environment for the mutation-based training energy profiler.

## Test Structure

```
test/
├── run_tests.sh              # Main test runner script
├── test_config.json          # Test configuration file
├── governor.sh               # Mock governor script (no root required)
├── repos/
│   └── test_repo/
│       └── train.sh          # Mock training script
└── README.md                 # This file
```

## What Gets Tested

The test suite validates the following functionality:

1. **File Existence** - Checks that all required files are present
2. **Script Executability** - Verifies scripts have execute permissions
3. **Configuration Validation** - Validates JSON configuration files
4. **Mock Training Script** - Tests the mock training script directly
5. **Train Wrapper** - Tests the training wrapper script
6. **Mock Governor** - Tests the governor script (mock, no root needed)
7. **Energy Monitor** - Tests energy monitoring functionality
8. **Mutation Runner --list** - Tests listing available models
9. **Full Integration Test** - Runs a complete mutation experiment
10. **Hyperparameter Mutation** - Tests mutation logic

## Running Tests

### Quick Test (All Tests)

```bash
cd /home/green/energy_dl/nightly
./test/run_tests.sh
```

### Individual Component Tests

#### Test Mock Training Script Only
```bash
cd test/repos/test_repo
./train.sh -n model_a --epochs 5 --lr 0.001
```

#### Test Mutation Runner with Test Config
```bash
python3 mutation_runner.py --list --config test/test_config.json
```

#### Run Single Mutation Test
```bash
# Copy mock governor to project root (for testing without sudo)
cp test/governor.sh governor.sh

# Run mutation test
python3 mutation_runner.py \
    --repo test_repo \
    --model model_a \
    --mutate epochs,learning_rate,seed \
    --runs 1 \
    --config test/test_config.json

# Clean up
rm governor.sh
```

## Test Configuration

The test environment uses `test/test_config.json` which defines:

- **Repository**: `test_repo`
- **Models**: `model_a`, `model_b`
- **Supported Hyperparameters**:
  - `epochs` (int, range: 5-20)
  - `learning_rate` (float, range: 0.0001-0.01)
  - `seed` (int, range: 0-9999)
  - `dropout` (float, range: 0.0-0.8)
  - `weight_decay` (float, range: 0.0-0.01)

## Mock Components

### Mock Training Script (`test/repos/test_repo/train.sh`)

Simulates a real training process:
- Accepts all standard hyperparameter flags
- Runs for the specified number of epochs (1 second per epoch)
- Generates realistic training output
- Reports test accuracy and loss
- Always succeeds (for testing success path)

### Mock Governor Script (`test/governor.sh`)

Simulates the CPU governor script:
- Accepts same arguments as real governor.sh
- Does NOT require root privileges
- Does NOT actually change CPU settings
- Useful for testing without system modifications

## Expected Test Output

When all tests pass, you should see:

```
========================================
Test Summary
========================================

Total Tests: 20+
Passed: 20+
Failed: 0
Pass Rate: 100.0%

✨ All tests passed! The mutation framework is working correctly.
```

## Test Results Location

Test outputs are stored in:
- `results/` - JSON result files and energy monitoring data
- `/tmp/test_*.txt` - Temporary test output files

## Troubleshooting

### Permission Errors

Make sure all scripts are executable:
```bash
chmod +x test/run_tests.sh
chmod +x test/repos/test_repo/train.sh
chmod +x test/governor.sh
chmod +x mutation_runner.py
chmod +x scripts/*.sh
```

### Missing Dependencies

The test suite requires:
- Python 3.6+
- bash
- bc (for calculations in mock scripts)
- Standard Unix utilities (grep, sed, etc.)

### Test Failures

If tests fail:
1. Check the error messages in the test output
2. Review temporary output files in `/tmp/test_*.txt`
3. Ensure you're in the project root directory
4. Verify all files were created correctly

## Extending Tests

To add new tests, edit `test/run_tests.sh` and add a new test section following the existing pattern:

```bash
print_header "Test N: Your Test Name"
print_test "Testing something"
if [ your test condition ]; then
    print_pass "Test passed"
else
    print_fail "Test failed"
fi
```

## Real System Testing

After tests pass, you can test with real repositories:

```bash
# List real models
python3 mutation_runner.py --list

# Run mutation on real model (example)
python3 mutation_runner.py \
    --repo pytorch_resnet_cifar10 \
    --model resnet20 \
    --mutate epochs,learning_rate \
    --runs 1
```

## Notes

- The test suite is designed to run without sudo privileges
- Tests complete in approximately 30-60 seconds
- Energy monitoring tests may not collect real data without proper hardware support
- Mock scripts simulate realistic behavior but don't perform actual training
