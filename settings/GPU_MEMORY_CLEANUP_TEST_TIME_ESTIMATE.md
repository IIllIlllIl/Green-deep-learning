# GPU Memory Cleanup Test - Runtime Estimation

**Generated**: 2025-11-26
**Configuration**: `settings/gpu_memory_cleanup_test.json`
**Purpose**: Estimate total runtime for GPU memory cleanup testing

## Configuration Summary

- **Experiment Name**: gpu_memory_cleanup_test
- **Mode**: default
- **Runs per Config**: 2
- **Max Retries**: 1
- **Governor**: performance
- **Total Configs**: 5
- **Total Experiment Runs**: 10 (5 configs × 2 runs)

## Experiments Overview

| # | Model | Epochs | Runs | Notes |
|---|-------|--------|------|-------|
| 1 | VulBERTa/mlp | 5 | 2 | Baseline small model |
| 2 | examples/mnist_ff | 5 | 2 | Previously failed model (batch_size=10000) |
| 3 | VulBERTa/mlp | 7 | 2 | Verify GPU recovery |
| 4 | Person_reID_baseline_pytorch/pcb | 3 | 2 | Large model with small epochs |
| 5 | VulBERTa/mlp | 5 | 2 | Final cleanup verification |

## Historical Performance Data

Based on analysis of `results/summary_all.csv` (211 experiments):

### VulBERTa/mlp
- **Historical Experiments**: 26
- **Average Runtime**: 1486.3 seconds (24.8 minutes)
- **Range**: 86.5 - 5565.1 seconds (1.4 - 92.8 minutes)
- **Epochs Range**: 5 - 20
- **Time per Epoch**: 177.6 seconds

### Person_reID_baseline_pytorch/pcb
- **Historical Experiments**: 10
- **Average Runtime**: 4168.5 seconds (69.5 minutes)
- **Range**: 3889.5 - 4374.1 seconds (64.8 - 72.9 minutes)
- **Epochs Range**: 57 - 65
- **Time per Epoch**: 69.7 seconds

### examples/mnist_ff
- **Historical Experiments**: 0 (⚠️ ALL FAILED previously)
- **Estimation Method**: Based on examples/mnist × 1.5 coefficient
- **examples/mnist Time per Epoch**: 21.3 seconds
- **Estimated mnist_ff Time per Epoch**: 31.95 seconds

## Detailed Time Estimation

### Experiment 1: VulBERTa/mlp (epochs=5)
- **Single Run**: 888 seconds (14.8 minutes)
- **2 Runs**: 1,776 seconds (29.6 minutes)
- **Cumulative**: 29.6 minutes (0.49 hours)

### Experiment 2: examples/mnist_ff (epochs=5, batch_size=10000)
- **Single Run**: 160 seconds (2.7 minutes)
- **2 Runs**: 320 seconds (5.3 minutes)
- **Cumulative**: 34.9 minutes (0.58 hours)
- ⚠️ **Note**: No historical data, estimated from examples/mnist × 1.5

### Experiment 3: VulBERTa/mlp (epochs=7)
- **Single Run**: 1,243 seconds (20.7 minutes)
- **2 Runs**: 2,486 seconds (41.4 minutes)
- **Cumulative**: 76.3 minutes (1.27 hours)

### Experiment 4: Person_reID_baseline_pytorch/pcb (epochs=3)
- **Single Run**: 209 seconds (3.5 minutes)
- **2 Runs**: 418 seconds (7.0 minutes)
- **Cumulative**: 83.3 minutes (1.39 hours)

### Experiment 5: VulBERTa/mlp (epochs=5)
- **Single Run**: 888 seconds (14.8 minutes)
- **2 Runs**: 1,776 seconds (29.6 minutes)
- **Cumulative**: 112.9 minutes (1.88 hours)

## Time Distribution Visualization

```
1. VulBERTa/mlp (5 epochs)     ███████████████████████████████████     29.6 min
   Cumulative: 29.6 min (0.49 hours)

2. mnist_ff (5 epochs)          ██████                                  5.3 min
   Cumulative: 34.9 min (0.58 hours)

3. VulBERTa/mlp (7 epochs)     ██████████████████████████████████████  41.4 min
   Cumulative: 76.3 min (1.27 hours)

4. pcb (3 epochs)              ████████                                 7.0 min
   Cumulative: 83.3 min (1.39 hours)

5. VulBERTa/mlp (5 epochs)     ███████████████████████████████████     29.6 min
   Cumulative: 112.9 min (1.88 hours)
```

## Timeline

```
Time    Experiment                           Duration
────────────────────────────────────────────────────────
0:00    VulBERTa/mlp (5 epochs)              29.6 min
0:30    mnist_ff (5 epochs)                   5.3 min
0:35    VulBERTa/mlp (7 epochs)              41.4 min
1:16    pcb (3 epochs)                        7.0 min
1:23    VulBERTa/mlp (5 epochs)              29.6 min
────────────────────────────────────────────────────────
1:53    End (ideal case)
```

## Total Time Estimates

### Scenario 1: Ideal Case (No Failures/Retries)
- **Total Runtime**: 6,776 seconds
- **Total Runtime**: 112.9 minutes
- **Total Runtime**: **1.88 hours**

### Scenario 2: With Retry Buffer (+10%)
- **Total Runtime**: 7,454 seconds
- **Total Runtime**: 124.2 minutes
- **Total Runtime**: **2.07 hours**
- **Assumption**: ~10% of experiments may need retry

### Scenario 3: With System Overhead (Recommended)
- **Total Runtime**: 7,604 seconds
- **Total Runtime**: 126.7 minutes
- **Total Runtime**: **2.11 hours** ✅
- **Includes**:
  - Retry buffer: +10%
  - System overhead: ~30 seconds per experiment (setup/cleanup)

## Risk Factors

### High Risk
- ⚠️ **mnist_ff**: No historical success data
  - May be faster or slower than estimated
  - Could fail with batch_size=10000 (though less likely than 50000)
  - If fails, retry will add time

### Medium Risk
- ⚠️ **GPU OOM During pcb**: epochs=3 is very short
  - Historical data shows pcb typically runs 57-65 epochs
  - Short runs may behave differently
  - PCB is memory-intensive

### Low Risk
- ✓ **VulBERTa/mlp**: Well-tested model with consistent performance
- ✓ **Small epoch counts**: Reduces overall risk exposure

## Recommendations

### 1. Timing
- ✅ **Plan for 2.5 hours** to be safe
- ✅ Run during off-peak hours (evening/weekend)
- ✅ Avoid running when other GPU jobs are scheduled

### 2. Monitoring
- 🔍 Watch GPU memory usage during mnist_ff
- 🔍 Monitor if cleanup mechanism triggers between experiments
- 🔍 Check if pcb with epochs=3 completes successfully

### 3. Contingency Planning
- 📋 If mnist_ff fails repeatedly (>2 retries):
  - Consider reducing batch_size further (10000 → 5000)
  - Or skip mnist_ff and document the issue
- 📋 If pcb fails:
  - May need to increase epochs (3 → 5)
  - Or reduce batch size

### 4. Success Criteria
- ✅ All 10 experiments complete successfully
- ✅ No GPU OOM errors
- ✅ GPU memory properly cleaned between experiments
- ✅ Total runtime within 2.5 hours

## Comparison with Previous Runs

### Previous Batch Experiment Runtimes

From `results/summary_all.csv`:

| Experiment Round | Total Experiments | Date Range | Estimated Duration |
|------------------|-------------------|------------|-------------------|
| default | 20 | 2025-11-18 to 11-19 | ~14 hours |
| mutation_1x | 74 | 2025-11-20 to 11-22 | ~52 hours |
| mutation_2x_safe | 117 | 2025-11-22 to 11-25 | ~81 hours |

**This test (10 experiments)**: ~2.1 hours

**Efficiency**: This test is much shorter because:
- Small number of experiments (10 vs 20-117)
- Short epoch counts (3-7 vs default 10-200)
- Focused model selection (3 models vs 11 models)

## Appendix: Calculation Details

### VulBERTa/mlp (5 epochs)
```
Time per epoch: 177.6 seconds (from 26 historical experiments)
Single run: 5 × 177.6 = 888 seconds = 14.8 minutes
2 runs: 2 × 888 = 1,776 seconds = 29.6 minutes
```

### examples/mnist_ff (5 epochs)
```
Base time per epoch (mnist): 21.3 seconds
Estimated time per epoch (mnist_ff): 21.3 × 1.5 = 31.95 seconds
Single run: 5 × 31.95 = 160 seconds = 2.7 minutes
2 runs: 2 × 160 = 320 seconds = 5.3 minutes
```

### VulBERTa/mlp (7 epochs)
```
Time per epoch: 177.6 seconds
Single run: 7 × 177.6 = 1,243 seconds = 20.7 minutes
2 runs: 2 × 1,243 = 2,486 seconds = 41.4 minutes
```

### Person_reID_baseline_pytorch/pcb (3 epochs)
```
Time per epoch: 69.7 seconds (from 10 historical experiments)
Single run: 3 × 69.7 = 209 seconds = 3.5 minutes
2 runs: 2 × 209 = 418 seconds = 7.0 minutes
```

---

**Generated by**: Mutation-Based Training Energy Profiler Time Estimator
**Based on**: results/summary_all.csv (211 historical experiments)
**Confidence Level**: Medium-High (except mnist_ff which has no historical data)
