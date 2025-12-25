# Tests Archived - 2025-12-05

## Reason for Archival

The following test files were created during the deduplication mode fix (v4.6.0) and have been archived because their functionality has been fully integrated into the existing `tests/unit/test_dedup_mechanism.py` test suite.

## Archived Files

### 1. test_dedup_mode_distinction.py
**Created**: 2025-12-05
**Purpose**: Test deduplication mode distinction functionality
**Tests**: 5 test cases
- Mode distinction in _normalize_mutation_key
- Mode detection from experiment_id
- Mode distinction in build_dedup_set
- Backward compatibility
- Real-world scenario

**Integration Status**: ✅ All functionality integrated into test_dedup_mechanism.py as "Test 7: Mode distinction"

### 2. test_integration_after_mode_fix.py
**Created**: 2025-12-05
**Purpose**: Verify existing functionality after mode fix
**Tests**: 4 test cases
- Backward compatibility
- Mode parameter support
- Deduplication across modes
- Uniqueness checking

**Integration Status**: ✅ All functionality covered by updated test_dedup_mechanism.py tests

## Integrated Test Suite

The updated `tests/unit/test_dedup_mechanism.py` now includes 7 comprehensive tests that cover:
1. Extract single CSV
2. Extract multiple CSVs
3. Filter by model
4. Build dedup set (with mode information)
5. Generate with dedup (mode-aware)
6. Integration test (mode-aware)
7. Mode distinction (NEW - covers archived test functionality)

## Test Results

All 7 tests in the integrated suite pass successfully (100% pass rate).

## Archival Date

2025-12-05

## Recovery Instructions

If needed, these archived tests can be restored from the `tests/archived/` directory. However, the integrated test suite in `tests/unit/test_dedup_mechanism.py` provides equivalent or better coverage.
