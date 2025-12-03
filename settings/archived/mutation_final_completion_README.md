# Experiment Completion Plan

## Overview
This document outlines the remaining experiments needed to achieve 5 unique values per parameter in both parallel and non-parallel modes for all 11 models.

## Current Status (2025-12-02)
- **Total experiments completed**: 319
  - Non-parallel: 214 experiments
  - Parallel: 105 experiments
- **Non-parallel completion**: 33/45 parameters (73.3%)
- **Parallel completion**: 0/45 parameters (0%)
- **Overall completion (both modes)**: 0/45 parameters (0%)

## Configuration File
**File**: `settings/mutation_final_completion.json`

### Key Settings
- `runs_per_config`: 1 (each experiment runs once to generate 1 unique value)
- `use_deduplication`: true (prevents duplicate experiments)
- `historical_csvs`: ["results/summary_all.csv"] (uses existing data for deduplication)

### Experiment Count by Model
The configuration contains experiments for all 11 models:

| Model | Non-parallel Experiments | Parallel Experiments | Total |
|-------|------------------------|---------------------|-------|
| MRT-OAST | 1 | 5 | 6 |
| bug-localization | 0 | 4 | 4 |
| pytorch_resnet_cifar10 | 0 | 4 | 4 |
| VulBERTa | 0 | 4 | 4 |
| Person_reID_densenet121 | 0 | 4 | 4 |
| Person_reID_hrnet18 | 3 | 4 | 7 |
| Person_reID_pcb | 4 | 4 | 8 |
| examples/mnist | 0 | 4 | 4 |
| examples/mnist_rnn | 0 | 4 | 4 |
| examples/siamese | 0 | 4 | 4 |
| examples/mnist_ff | 4 | 4 | 8 |
| **TOTAL** | **12** | **45** | **57** |

## Detailed Breakdown

### Models Needing Only Parallel Work (7 models)
These models are complete in non-parallel mode and only need parallel experiments:
1. **bug-localization-by-dnn-and-rvsm**: 4 parameters × 5 values = 20 parallel experiments
2. **pytorch_resnet_cifar10/resnet20**: 4 parameters × 5 values = 20 parallel experiments
3. **VulBERTa/mlp**: 4 parameters × 5 values = 20 parallel experiments
4. **Person_reID_baseline_pytorch/densenet121**: 4 parameters × 5 values = 20 parallel experiments
5. **examples/mnist**: 4 parameters × 5 values = 20 parallel experiments
6. **examples/mnist_rnn**: 4 parameters × 5 values = 20 parallel experiments
7. **examples/siamese**: 4 parameters × 5 values = 20 parallel experiments

### Models Needing Both Non-Parallel and Parallel Work (4 models)

#### 1. MRT-OAST/default
- **Non-parallel**: 1 experiment (dropout needs 1 more value)
- **Parallel**: 25 experiments (5 parameters × 5 values each)

#### 2. Person_reID_baseline_pytorch/hrnet18
- **Non-parallel**: 3 experiments (learning_rate, dropout, seed each need 1 value)
- **Parallel**: 20 experiments (4 parameters × 5 values each)

#### 3. Person_reID_baseline_pytorch/pcb
- **Non-parallel**: 20 experiments (4 parameters × 5 values each)
- **Parallel**: 20 experiments (4 parameters × 5 values each)

#### 4. examples/mnist_ff
- **Non-parallel**: 20 experiments (4 parameters × 5 values each)
- **Parallel**: 20 experiments (4 parameters × 5 values each)

## Execution Strategy

### Option 1: Single Run (Recommended)
Run all experiments in one command with deduplication enabled:
```bash
sudo -E python3 mutation.py -ec settings/mutation_final_completion.json
```

**Pros**:
- Simplest approach
- Deduplication prevents duplicates automatically
- Single results directory

**Cons**:
- Very long runtime (estimated 200-300+ hours)
- Cannot easily pause/resume
- All results in one session directory

### Option 2: Staged Execution
Split the configuration into multiple files and run in stages:

**Stage 1**: Non-parallel experiments (faster, ~20-40 hours)
**Stage 2**: Parallel experiments (slower, ~200-300 hours)

This allows:
- Monitoring progress between stages
- Pausing between stages
- Validating intermediate results

### Option 3: Model-by-Model
Create separate configs for each model and run sequentially.

**Pros**:
- Maximum control
- Can prioritize critical models
- Easy to track per-model progress

**Cons**:
- Requires creating 11 separate config files
- More manual management

## Estimated Runtime

Based on previous experiments:

| Model Type | Avg Time/Experiment | Non-parallel Total | Parallel Total |
|-----------|-------------------|-------------------|---------------|
| Fast (mnist, mnist_rnn, siamese, mnist_ff) | ~5-10 min | ~2-4 hours | ~10-20 hours |
| Medium (bug-localization, pytorch_resnet) | ~20-30 min | ~5-10 hours | ~40-60 hours |
| Slow (VulBERTa, Person_reID) | ~60-90 min | ~20-30 hours | ~100-150 hours |

**Total estimated runtime**:
- Non-parallel: ~30-50 hours
- Parallel: ~200-300 hours
- **Grand total**: ~230-350 hours (9-15 days continuous)

## Validation After Completion

After running the experiments, verify completion with:

```bash
python3 scripts/analyze_completion.py results/summary_all.csv
```

This will show:
- Unique values per parameter in each mode
- Models with incomplete data
- Any parameters still needing experiments

## Notes

1. **Deduplication**: The configuration uses deduplication to prevent running duplicate experiments. This is critical since summary_all.csv already contains 319 experiments.

2. **Parallel Mode Background Model**: All parallel experiments use lightweight models (mnist or siamese) as background GPU load to ensure consistent experiment conditions.

3. **Data Integrity**: After the recent CSV append bug fix, all new experiments will properly append to summary_all.csv without data loss.

4. **GPU Memory**: The configuration includes GPU cleanup between experiments to prevent memory accumulation issues.

5. **Failure Handling**: Each experiment has `max_retries: 2`, allowing automatic retry on transient failures.

## Next Steps

1. **Review the configuration**: Check `settings/mutation_final_completion.json` to ensure it matches your requirements

2. **Choose execution strategy**: Decide whether to run all at once (Option 1) or in stages (Option 2 or 3)

3. **Monitor progress**: The system will print progress updates showing:
   - Current experiment number
   - Success/failure status
   - Deduplication statistics

4. **Verify completion**: After running, check summary_all.csv to confirm all parameters have 5 unique values in both modes

5. **Analyze results**: Use the completion analysis script to generate final statistics
