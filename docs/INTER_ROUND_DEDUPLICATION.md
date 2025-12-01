# Inter-Round Hyperparameter Deduplication Mechanism

**Author**: Mutation-Based Training Energy Profiler Team
**Created**: 2025-11-26
**Version**: 1.0

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Solution Design](#solution-design)
4. [Implementation Details](#implementation-details)
5. [Usage Guide](#usage-guide)
6. [Testing](#testing)
7. [Integration Example](#integration-example)

## Overview

This document describes the inter-round hyperparameter deduplication mechanism implemented to prevent duplicate hyperparameter combinations across multiple experiment rounds.

### Key Features

- ✅ **CSV-based History Loading**: Extract hyperparameter combinations from historical CSV files
- ✅ **Model-Specific Filtering**: Filter by repository/model for targeted deduplication
- ✅ **Backward Compatible**: Existing code works without changes; deduplication is optional
- ✅ **Precision Handling**: Normalizes floating-point values to avoid false duplicates
- ✅ **Performance Efficient**: Uses hash-based set for O(1) lookuptime

## Problem Statement

### Before: Within-Round Deduplication Only

The original system prevented duplicates **within a single experiment round**:

```
Round 1 (default):
  - epochs=10, lr=0.01 ✓ (default, excluded)
  - epochs=8, lr=0.005 ✓ (generated)
  - epochs=12, lr=0.02 ✓ (generated)

Round 2 (mutation_1x):
  - epochs=10, lr=0.01 ✓ (default, excluded)
  - epochs=8, lr=0.005 ✗ (DUPLICATE from Round 1, but not detected!)
  - epochs=15, lr=0.03 ✓ (generated)
```

**Problem**: The `seen_mutations` set was reset at the start of each round, so Round 2 could regenerate the same combinations as Round 1.

### Analysis Results

From `results/summary_all.csv` (211 experiments across 3 rounds):

**4 Inter-round Duplicate Groups Detected**:

1. **MRT-OAST/default** - epochs=8.0
   - mutation_1x: 2 experiments
   - mutation_2x_safe: 1 experiment
   - **Total duplicates**: 3

2. **VulBERTa/mlp** - epochs=12.0
   - mutation_1x: 1 experiment
   - mutation_2x_safe: 1 experiment
   - **Total duplicates**: 2

3. **examples/mnist_rnn** - epochs=12.0
   - mutation_1x: 1 experiment
   - mutation_2x_safe: 2 experiments
   - **Total duplicates**: 3

4. **examples/siamese** - epochs=14.0
   - mutation_1x: 1 experiment
   - mutation_2x_safe: 1 experiment
   - **Total duplicates**: 2

**Total Inter-round Duplicates**: 10 experiments out of 211 (4.7%)

## Solution Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   New Round Starting                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Load Historical CSV Files                         │
│  ├─ results/defualt/summary.csv                            │
│  ├─ results/mutation_1x/summary.csv                        │
│  └─ results/mutation_2x_20251122_175401/summary_safe.csv   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Extract Hyperparameter Combinations               │
│  - Parse CSV columns (hyperparam_*)                        │
│  - Filter by repository/model (optional)                   │
│  - Extract only non-None values                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Build Deduplication Set                           │
│  - Normalize float values to 6 decimal places              │
│  - Convert to sorted tuples for hashing                    │
│  - Store in set for O(1) lookup                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Generate New Mutations                            │
│  - Pass historical set to generate_mutations()             │
│  - Check each candidate against historical + defaults      │
│  - Retry if duplicate detected                             │
│  - Return only unique combinations                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **CSV as Source of Truth**: Historical data stored in CSV files, avoiding need for separate database

2. **Normalized Keys**: Uses `_normalize_mutation_key()` to create hashable tuple representations:
   ```python
   # Example mutation
   {"epochs": 10, "learning_rate": 0.01}

   # Normalized key
   (("epochs", "10"), ("learning_rate", "0.010000"))
   ```

3. **Optional Filtering**: Can filter by repository/model for model-specific deduplication:
   ```python
   # Load only examples/mnist history
   mutations, stats = load_historical_mutations(
       csv_files,
       filter_by_repo="examples",
       filter_by_model="mnist"
   )
   ```

4. **Backward Compatibility**: Existing code works without changes; pass `existing_mutations=None` to disable inter-round deduplication

## Implementation Details

### New Module: `mutation/dedup.py`

Core functions for CSV-based deduplication:

#### 1. `extract_mutations_from_csv()`

Extracts hyperparameter combinations from a single CSV file.

**Parameters**:
- `csv_path`: Path to CSV file
- `filter_by_repo`: Optional repository filter
- `filter_by_model`: Optional model filter
- `logger`: Optional logger instance

**Returns**: `(mutations_list, statistics_dict)`

**Example**:
```python
from pathlib import Path
from mutation.dedup import extract_mutations_from_csv

csv_path = Path("results/defualt/summary.csv")
mutations, stats = extract_mutations_from_csv(csv_path)

print(f"Extracted {stats['extracted']} mutations")
# Output: Extracted 22 mutations
```

#### 2. `load_historical_mutations()`

Loads hyperparameter combinations from multiple CSV files.

**Parameters**:
- `csv_paths`: List of CSV file paths
- `filter_by_repo`: Optional repository filter
- `filter_by_model`: Optional model filter
- `logger`: Optional logger instance

**Returns**: `(all_mutations, combined_statistics)`

**Example**:
```python
from mutation.dedup import load_historical_mutations

csv_files = [
    Path("results/defualt/summary.csv"),
    Path("results/mutation_1x/summary.csv"),
    Path("results/mutation_2x_20251122_175401/summary_safe.csv"),
]

mutations, stats = load_historical_mutations(csv_files)
print(f"Loaded {stats['total_extracted']} total mutations from {stats['successful_files']} files")
# Output: Loaded 231 total mutations from 3 files
```

#### 3. `build_dedup_set()`

Builds a set of normalized mutation keys for deduplication.

**Parameters**:
- `mutations`: List of mutation dictionaries
- `logger`: Optional logger instance

**Returns**: `Set[tuple]` of normalized mutation keys

**Example**:
```python
from mutation.dedup import build_dedup_set

mutations = [
    {"epochs": 10, "learning_rate": 0.001},
    {"epochs": 20, "learning_rate": 0.01},
    {"epochs": 10, "learning_rate": 0.001},  # Duplicate
]

dedup_set = build_dedup_set(mutations)
print(f"Unique mutations: {len(dedup_set)}")
# Output: Unique mutations: 2
```

#### 4. `print_dedup_statistics()`

Prints human-readable statistics about loaded historical data.

**Example**:
```python
from mutation.dedup import load_historical_mutations, build_dedup_set, print_dedup_statistics

mutations, stats = load_historical_mutations(csv_files)
dedup_set = build_dedup_set(mutations)
print_dedup_statistics(stats, dedup_set)
```

**Output**:
```
================================================================================
Historical Hyperparameter Loading Statistics
================================================================================
CSV Files Processed: 3/3
Total Rows: 231
Filtered Rows: 0
Extracted Mutations: 231
Unique Mutations: 189

Breakdown by Model:
  MRT-OAST/default: 32
  Person_reID_baseline_pytorch/densenet121: 26
  Person_reID_baseline_pytorch/hrnet18: 11
  Person_reID_baseline_pytorch/pcb: 10
  VulBERTa/mlp: 26
  bug-localization-by-dnn-and-rvsm/default: 20
  examples/mnist: 38
  examples/mnist_ff: 20
  examples/mnist_rnn: 24
  examples/siamese: 24
  pytorch_resnet_cifar10/resnet20: 26
================================================================================
```

### Modified: `mutation/hyperparams.py`

#### Updated `generate_mutations()` Signature

Added optional `existing_mutations` parameter:

```python
def generate_mutations(
    supported_params: Dict,
    mutate_params: List[str],
    num_mutations: int = 1,
    random_seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    existing_mutations: Optional[set] = None  # NEW PARAMETER
) -> List[Dict[str, Any]]:
```

**Behavior**:
- If `existing_mutations=None` (default): Works exactly as before (within-round deduplication only)
- If `existing_mutations=set(...)`: Initializes `seen_mutations` with historical data

**Implementation**:
```python
# Initialize seen_mutations set (may include historical mutations)
if existing_mutations is not None:
    seen_mutations = existing_mutations.copy()  # Make a copy to avoid modifying the original
    print(f"   Loaded {len(existing_mutations)} historical mutations for deduplication")
else:
    seen_mutations = set()  # Track unique mutations using normalized keys
```

## Usage Guide

### Basic Usage (No Inter-Round Deduplication)

Works exactly as before - no changes needed:

```python
from mutation.hyperparams import generate_mutations

mutations = generate_mutations(
    supported_params=config["models"]["examples"]["supported_hyperparams"],
    mutate_params=["epochs", "learning_rate"],
    num_mutations=10
)
```

### Usage with Inter-Round Deduplication

**Step 1**: Load historical data from CSV files

```python
from pathlib import Path
from mutation.dedup import load_historical_mutations, build_dedup_set

# Define CSV files to load
csv_files = [
    Path("results/defualt/summary.csv"),
    Path("results/mutation_1x/summary.csv"),
    Path("results/mutation_2x_20251122_175401/summary_safe.csv"),
]

# Load all historical mutations
mutations, stats = load_historical_mutations(csv_files)
print(f"Loaded {stats['total_extracted']} mutations from {stats['successful_files']} files")
```

**Step 2**: Build deduplication set

```python
# Build set of normalized keys
dedup_set = build_dedup_set(mutations)
print(f"Unique historical mutations: {len(dedup_set)}")
```

**Step 3**: Generate new mutations with deduplication

```python
from mutation.hyperparams import generate_mutations

new_mutations = generate_mutations(
    supported_params=config["models"]["examples"]["supported_hyperparams"],
    mutate_params=["epochs", "learning_rate"],
    num_mutations=10,
    existing_mutations=dedup_set  # Pass historical mutations
)
```

### Model-Specific Deduplication

Filter historical data by repository/model for targeted deduplication:

```python
# Load only examples/mnist history
mutations, stats = load_historical_mutations(
    csv_files,
    filter_by_repo="examples",
    filter_by_model="mnist"
)

dedup_set = build_dedup_set(mutations)

# Generate new mutations for examples/mnist (won't duplicate previous examples/mnist experiments)
new_mutations = generate_mutations(
    supported_params=config["models"]["examples"]["supported_hyperparams"],
    mutate_params=["epochs", "learning_rate"],
    num_mutations=10,
    existing_mutations=dedup_set
)
```

## Testing

### Test Script: `tests/unit/test_dedup_mechanism.py`

Comprehensive test suite with 6 test cases:

1. **Extract single CSV**: Test CSV parsing and hyperparameter extraction
2. **Extract multiple CSVs**: Test loading from multiple files
3. **Filter by model**: Test repository/model filtering
4. **Build dedup set**: Test deduplication set building
5. **Generate with dedup**: Test mutation generation with historical data
6. **Integration test**: End-to-end test with real CSV files

### Running Tests

```bash
python3 tests/unit/test_dedup_mechanism.py
```

**Expected Output**:
```
================================================================================
Inter-Round Deduplication Test Suite
================================================================================
Project root: /home/green/energy_dl/nightly
CSV files to test: 3
  ✓ results/defualt/summary.csv
  ✓ results/mutation_1x/summary.csv
  ✓ results/mutation_2x_20251122_175401/summary_safe.csv
================================================================================

Test 1: Extract mutations from single CSV
--------------------------------------------------------------------------------
  ✓ Extracted mutations: 22 > 0
  ✓ Total rows counted: 22 > 0
  ✓ Extracted count matches: 22 == 22
  ✓ Mutation has hyperparameters: 5 > 0

✓ Extract single CSV PASSED

[... more test output ...]

================================================================================
Test Summary
================================================================================
Total tests: 6
Passed: 6
Failed: 0

✓ All tests passed!
================================================================================
```

## Integration Example

### Complete Workflow for Round 4

Here's how to use the inter-round deduplication mechanism when starting a new experiment round:

```python
#!/usr/bin/env python3
"""
Example: Running Round 4 with Inter-Round Deduplication
"""
from pathlib import Path
from mutation.runner import MutationRunner
from mutation.dedup import load_historical_mutations, build_dedup_set, print_dedup_statistics

# Step 1: Initialize runner
runner = MutationRunner(
    config_path="configs/mutation_config.json",
    results_dir="results/mutation_4x_$(date +%Y%m%d_%H%M%S)"
)

# Step 2: Load historical data from previous rounds
print("=" * 80)
print("Loading Historical Hyperparameter Data")
print("=" * 80)

csv_files = [
    Path("results/defualt/summary.csv"),
    Path("results/mutation_1x/summary.csv"),
    Path("results/mutation_2x_20251122_175401/summary_safe.csv"),
    # Add Round 3 CSV when available
]

mutations, stats = load_historical_mutations(csv_files)
dedup_set = build_dedup_set(mutations)
print_dedup_statistics(stats, dedup_set)

# Step 3: Run experiments with deduplication
# Note: You'll need to modify MutationRunner to pass existing_mutations
# to generate_mutations() calls. This is the next integration step.

# For now, use the dedup_set in custom experiment scripts
from mutation.hyperparams import generate_mutations

config = runner.config
repo_config = config["models"]["examples"]

new_mutations = generate_mutations(
    supported_params=repo_config["supported_hyperparams"],
    mutate_params=["epochs", "learning_rate"],
    num_mutations=20,
    existing_mutations=dedup_set  # ← Inter-round deduplication!
)

print(f"\n✓ Generated {len(new_mutations)} unique mutations for Round 4")
```

## Summary

### What We Achieved

✅ **CSV-based History Loading**: Load hyperparameter combinations from existing CSV files
✅ **Deduplication Set Building**: Convert historical data into efficient lookup structure
✅ **Backward Compatible**: Existing code works without changes
✅ **Model-Specific Filtering**: Filter by repository/model for targeted deduplication
✅ **Comprehensive Testing**: 6 test cases, all passing
✅ **Well-Documented**: Complete usage guide and integration examples

### Next Steps

1. **Integration with MutationRunner**: Modify `mutation/runner.py` to automatically load historical data and pass to `generate_mutations()`

2. **Configuration Option**: Add `--use-deduplication` flag to enable/disable inter-round deduplication

3. **Performance Optimization**: Cache deduplication sets to avoid reloading CSV files on each run

4. **Monitoring**: Add metrics to track how many generation attempts were rejected due to duplicates

## References

- **Main Implementation**: `mutation/dedup.py` (273 lines)
- **Modified Module**: `mutation/hyperparams.py` (lines 172-230)
- **Test Script**: `tests/unit/test_dedup_mechanism.py` (374 lines)
- **Duplicate Analysis**: See earlier analysis showing 4 inter-round duplicate groups

---

**Maintainer**: Mutation-Based Training Energy Profiler Team
**Version**: 1.0
**Last Updated**: 2025-11-26
