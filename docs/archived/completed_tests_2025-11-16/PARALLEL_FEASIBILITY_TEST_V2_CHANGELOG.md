# Parallel Feasibility Test v2 - Changelog

**Date**: 2025-11-15
**File**: `settings/parallel_feasibility_test_v2.json`
**Change**: Reduced from 12 to 11 experiments

---

## Summary

Removed experiment 7 (VulBERTa_mlp + VulBERTa_cnn) from the parallel feasibility test configuration to reduce the total number of experiments from 12 to 11 while maintaining good model and GPU memory coverage.

---

## Rationale

**Why experiment 7 was deleted:**

1. **❌ VulBERTa-CNN training not implemented**
   - `train_vulberta.py` lines 333-340 show `train_cnn()` immediately returns `None, None`
   - Background process exited after ~1 second and restarted 103 times
   - No actual GPU load generated during foreground training
   - Test was invalid

2. **✅ Minimal impact on coverage**
   - Experiment 6 also targets 3000MB (same memory tier)
   - VulBERTa_mlp appears in experiments 2, 9, and 12
   - No unique model combinations lost

3. **✅ Zero modification cost**
   - Only deletion required, no changes to other experiments
   - All remaining 11 experiments are verified successful
   - No risk of introducing new issues

---

## Changes Made

### Deleted Experiment

**Original Experiment 7:**
```json
{
  "mode": "parallel",
  "foreground": {
    "repo": "VulBERTa",
    "model": "mlp",
    "mode": "default",
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 3e-05,
      "seed": 1334
    }
  },
  "background": {
    "repo": "VulBERTa",
    "model": "cnn",  // ❌ Not implemented
    "hyperparameters": {
      "epochs": 1,
      "learning_rate": 3e-05,
      "seed": 1334
    }
  },
  "note": "Group 7: 中显存 (3000MB) - VulBERTa_mlp + VulBERTa_cnn"
}
```

### Updated Description

```json
"description": "Parallel training feasibility test - 11 model combinations with 1 epoch each to verify GPU memory compatibility and training stability"
```

---

## Resulting Configuration

### Stratified Sampling Distribution

| 层级 | 实验数 | 占比 | 序号 | 显存范围 (MB) |
|------|--------|------|------|---------------|
| **超低显存** | 1个 | 9.1% | 1 | 1300 |
| **低显存** | 4个 | 36.4% | 2,3,4,5 | 2000-2700 |
| **中显存** | 5个 | 45.5% | 6,8,9,10,11 | 3000-4000 |
| **高显存** | 1个 | 9.1% | 12 | 5000 |

**Comparison with original design:**
- Original: 1:4:6:1 (超低:低:中:高)
- v2: 1:4:5:1 (超低:低:中:高)
- Change: -1 from 中显存 tier

### Complete Experiment List (11 experiments)

| 序号 | 层级 | 总显存 | 前景模型 | 背景模型 |
|------|------|--------|----------|----------|
| 1 | 超低显存 | 1300MB | mnist_ff | resnet20 |
| 2 | 低显存 | 2000MB | mnist | VulBERTa_mlp |
| 3 | 低显存 | 2000MB | mnist | siamese |
| 4 | 低显存 | 2500MB | mnist_rnn | pcb |
| 5 | 低显存 | 2700MB | mnist_rnn | MRT-OAST |
| 6 | 中显存 | 3000MB | mnist_rnn | hrnet18 |
| 8 | 中显存 | 3500MB | siamese | pcb |
| 9 | 中显存 | 3500MB | pcb | VulBERTa_mlp |
| 10 | 中显存 | 4000MB | mnist_ff | densenet121 |
| 11 | 中显存 | 4000MB | pcb | bug-localization |
| 12 | 高显存 | 5000MB | densenet121 | VulBERTa_mlp |

---

## Coverage Analysis

### Model Coverage (11 unique models)

**Foreground models (7):**
- examples_mnist_ff (2次)
- examples_mnist (2次)
- examples_mnist_rnn (3次)
- examples_siamese (1次)
- Person_reID_pcb (2次)
- Person_reID_densenet121 (1次)

**Background models (8):**
- pytorch_resnet_cifar10/resnet20 (1次)
- VulBERTa/mlp (3次)
- examples/siamese (1次)
- Person_reID_baseline_pytorch/pcb (2次)
- MRT-OAST/default (1次)
- Person_reID_baseline_pytorch/hrnet18 (1次)
- Person_reID_baseline_pytorch/densenet121 (1次)
- bug-localization-by-dnn-and-rvsm/default (1次)

**Total unique models: 11** (down from 12 in original)

### GPU Memory Coverage

**Memory points tested: 8**
- 1300MB, 2000MB, 2500MB, 2700MB, 3000MB, 3500MB, 4000MB, 5000MB

**Range:** 1300MB - 5000MB (3700MB span)

**Deleted memory point:** None (3000MB still covered by experiment 6)

---

## Impact Assessment

### ✅ Advantages

1. **Maintains stratified sampling principle**
   - All 4 tiers still represented
   - Proportions remain balanced (1:4:5:1)

2. **Preserves all successful experiments**
   - 11/11 remaining experiments are verified successful
   - No loss of valid training data

3. **Removes invalid test case**
   - Experiment 7 generated no GPU load
   - Background training failed (VulBERTa-CNN not implemented)

4. **Minimal modification**
   - Single deletion, no edits to other experiments
   - Easy to revert if needed

5. **Maintains good coverage**
   - 11 unique models tested
   - 8 memory utilization points
   - Full range from 1300MB to 5000MB

### ⚠️ Trade-offs

1. **Reduced 中显存 sampling**
   - 中显存 experiments: 6 → 5
   - Still well-represented (45.5% of total)

2. **One fewer experiment overall**
   - Total runtime reduced by ~1/12 (expected: 1.16h → 1.06h)

---

## Validation Status

All 11 experiments in v2 have been **previously validated** in the original parallel_feasibility_test run (run_20251115_165340):

| Exp# | Status | Foreground | Background | Coverage |
|------|--------|------------|------------|----------|
| 1 | ✅ | Success | Success | Valid |
| 2 | ✅ | Success | Success | Valid |
| 3 | ✅ | Success | Success | Valid |
| 4 | ✅ | Success | Success | Valid |
| 5 | ✅ | Success | Success | Valid |
| 6 | ✅ | Success | Success | Valid |
| 8 | ✅ | Success | Success | Valid |
| 9 | ✅ | Success | Success | Valid |
| 10 | ✅ | Success | Success | Valid |
| 11 | ✅ | Success | Success | Valid |
| 12 | ✅ | Success | Success | Valid |

**Success rate: 100% (11/11)**

---

## Next Steps

### Ready to execute

```bash
python mutation.py -ec settings/parallel_feasibility_test_v2.json
```

**Expected runtime:** ~1.06 hours (reduced from 1.16 hours)

**Expected results:**
- All 11 experiments should complete successfully
- GPU memory utilization: 1300MB - 5000MB
- All foreground and background processes should function correctly

---

## Related Documents

- `docs/PARALLEL_TRAINING_VALIDATION_REPORT.md` - Analysis of original 12-experiment run
- `docs/VULBERTA_CNN_REPLACEMENT_RECOMMENDATION.md` - Alternative replacement options considered
- `/tmp/reduce_to_11.py` - Analysis script for reduction strategy
- `settings/parallel_feasibility_test.json` - Original 12-experiment configuration

---

**Status**: ✅ Configuration ready
**Risk level**: Low (all experiments pre-validated)
**Recommendation**: Ready for execution
