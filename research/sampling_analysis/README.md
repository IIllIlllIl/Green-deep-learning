# Sampling Analysis Research Scripts

This directory contains research and analysis scripts used during experimental design for the mutation-based training energy profiler.

## Purpose

These scripts were created to evaluate and compare different sampling strategies for hyperparameter mutation and model pair selection. They are preserved for reference but are not part of the production mutation runner.

## Scripts

### Analysis Scripts

1. **analyze_boundary_results.py** (260 lines)
   - Analyzes boundary test experiment results
   - Purpose: Evaluate edge cases in hyperparameter ranges

2. **analyze_boundary_test.py** (7.4K)
   - Boundary condition testing analysis
   - Purpose: Test behavior at parameter range limits

3. **analyze_concurrent_training_feasibility.py** (12.6K)
   - Feasibility study for concurrent training experiments
   - Purpose: Evaluate parallel training capabilities

4. **analyze_model_pair_sampling.py** (18.5K)
   - Model pair sampling strategy analysis
   - Purpose: Compare different strategies for selecting model pairs

5. **analyze_performance_metrics.py** (5.1K)
   - Performance metrics extraction and analysis
   - Purpose: Validate metric parsing correctness

6. **analyze_stratified_sampling_strategies.py** (21.3K)
   - Comprehensive stratified sampling strategy comparison
   - Purpose: Evaluate different stratification approaches

### Implementation Scripts

7. **memory_stratified_sampling.py** (10.9K)
   - Implementation of memory-aware stratified sampling
   - Purpose: Sampling strategy that accounts for GPU memory constraints

8. **explain_memory_stratified_sampling.py** (19.0K)
   - Documentation and explanation of memory stratified sampling
   - Purpose: Detailed explanation with examples

## Key Findings

These research scripts informed the final hyperparameter mutation strategy implemented in `mutation/hyperparams.py`:

- **Log-uniform distribution** for exponentially-sensitive parameters (learning rate, weight decay)
- **Uniform distribution** for linearly-sensitive parameters (epochs, batch size)
- **Uniqueness guarantees** for generated mutations
- **Zero-probability** for optional parameters (weight decay)

## Status

**Date Archived**: 2025-11-13
**Status**: Research artifacts - not maintained
**Production Code**: See `mutation/hyperparams.py` for implemented strategies

## Usage

These scripts are provided for reference only. They may have dependencies on old experiment data or configurations that no longer exist.

If you need to run them, they were originally executed as:
```bash
python3 scripts/analyze_*.py
python3 scripts/memory_stratified_sampling.py
```

## References

- Production implementation: `mutation/hyperparams.py`
- Test suite: `tests/test_hyperparams.py`
- Documentation: `docs/REFACTORING_COMPLETE.md`
