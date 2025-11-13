# Scripts Archive

This directory contains legacy scripts that have been superseded by the refactored mutation runner.

## Archive Structure

```
archive/
├── testing/          # Development test scripts
│   ├── test_abbreviations.sh
│   └── run_full_test.sh
└── tools/            # Superseded tool scripts
    ├── run.sh                    # Moved to mutation/run.sh
    ├── train_wrapper.sh          # Functionality integrated into mutation/run.sh
    └── energy_monitor.sh         # Functionality integrated into mutation/run.sh
```

## Superseded Scripts

### tools/run.sh
**Status**: Superseded by `mutation/run.sh`
**Date Archived**: 2025-11-13
**Reason**: Moved to mutation package during refactoring
**Replacement**: Use `mutation/run.sh` instead

### tools/train_wrapper.sh
**Status**: Functionality integrated
**Date Archived**: 2025-11-13
**Reason**: All functionality absorbed into `mutation/run.sh`
**Replacement**: Use `mutation/run.sh` for training execution

### tools/energy_monitor.sh
**Status**: Functionality integrated
**Date Archived**: 2025-11-13
**Reason**: Energy monitoring now integrated into `mutation/run.sh` (lines 91-156)
**Replacement**: Energy monitoring is automatic with mutation runner

## Test Scripts

### testing/test_abbreviations.sh
**Status**: Development test
**Date Archived**: 2025-11-13
**Reason**: Development-time testing tool no longer actively used
**Replacement**: Use `test_refactoring.py` and `tests/` directory for testing

### testing/run_full_test.sh
**Status**: Development test
**Date Archived**: 2025-11-13
**Reason**: Development-time integration test
**Replacement**: Use `test_refactoring.py` for comprehensive functional testing

## Notes

These scripts are preserved for historical reference but are not maintained or supported. For current functionality, use the refactored mutation runner and its test suite.

**Active Scripts Location**: `/scripts/` (contains only `background_training_template.sh`)
**Research Scripts Location**: `/research/sampling_analysis/`
