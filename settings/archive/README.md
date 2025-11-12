# Settings Archive

This directory contains archived configuration files from previous experiments.

## Archive History

### 2025-11-11: boundary_test_lr_dropout_focused

**Original file**: `boundary_test_lr_dropout_focused.json`
**Archived as**: `boundary_test_lr_dropout_focused_2025-11-11.json`
**Reason**: Failed boundary test revealed that LR=5×default causes training crash on DenseNet121

**Test Results Summary**:
- **Success rate**: 13/13 (100% completed, but 1 with performance crash)
- **Execution time**: 2025-11-11 19:37 - 2025-11-12 01:16

**Key Findings**:
1. ✅ All trainings completed successfully (no execution failures)
2. ❌ **DenseNet121 with LR=5×default (0.25) crashed** - Rank@1: 90.29% → 0%, mAP: 75.15% → 0.19%
3. ⚠️ MRT-OAST with LR=0.2×default showed 9% precision drop
4. ⚠️ MRT-OAST with Dropout=0.5 showed 12% recall drop
5. ✅ ResNet20 was robust across all tested ranges

**Configuration Details**:
```json
Learning Rate Range: [0.2×default, 5×default]
Dropout Range: [0.0, 0.5]

Tested Models:
- Person_reID_baseline_pytorch/densenet121 (5 experiments)
- MRT-OAST/default (5 experiments)
- pytorch_resnet_cifar10/resnet20 (3 experiments)
```

**Outcome**: Range too aggressive, especially LR upper bound. Led to creation of `boundary_test_conservative.json` with adjusted ranges: LR=[0.25×, 4×], Dropout=[0.0, 0.4]

---

## How to Use Archived Files

Archived configurations can still be run for reference:

```bash
python mutation.py -ec settings/archive/boundary_test_lr_dropout_focused_2025-11-11.json
```

However, **it is not recommended** to use these configurations in production as they may contain ranges that cause performance issues.
