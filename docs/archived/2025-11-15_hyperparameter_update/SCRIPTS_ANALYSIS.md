# Scripts Directory Analysis

## Overview

Analysis of `/home/green/energy_dl/nightly/scripts/` directory to categorize scripts as **ACTIVE** (currently used in production code) or **LEGACY** (historical/analysis scripts not part of core workflow).

**Analysis Date**: 2025-11-13
**Refactoring Context**: After mutation.py modularization, run.sh moved to mutation/ folder

---

## Summary

| Category | Count | Notes |
|----------|-------|-------|
| **ACTIVE** | 1 | Production scripts used by mutation runner |
| **LEGACY - Testing** | 5 | Historical test/development scripts |
| **LEGACY - Analysis** | 8 | Research/analysis scripts for experimental design |
| **Total** | 14 | |

---

## ACTIVE Scripts (Currently Used) ✅

### 1. `background_training_template.sh` (4.2K)
**Status**: ✅ **ACTIVE - CRITICAL**

**Usage**:
- Called by `mutation/command_runner.py:267`
- Required for parallel experiments (foreground + background training)

**Code reference**:
```python
# mutation/command_runner.py:267
template_script_path = self.project_root / "scripts" / "background_training_template.sh"
```

**Purpose**:
- Manages background training loops for GPU load generation
- Used in `run_parallel_experiment()` workflow
- Essential for testing models under parallel GPU utilization

**Last Modified**: 2024-11-12 17:37

**Recommendation**: ✅ **KEEP - In active use**

---

## LEGACY Scripts (Not Currently Used) 🗄️

### Testing/Development Scripts (5 files)

#### 2. `run.sh` (6.8K)
**Status**: 🗄️ **LEGACY - SUPERSEDED**

**Reason**:
- **MOVED TO** `mutation/run.sh` during refactoring
- `mutation/command_runner.py:92` now uses `Path(__file__).parent / "run.sh"`
- This version is obsolete

**Last Modified**: 2024-11-11 19:36

**Recommendation**: 🗑️ **DELETE - Superseded by mutation/run.sh**

---

#### 3. `energy_monitor.sh` (4.2K)
**Status**: 🗄️ **LEGACY - Standalone tool**

**Usage**: NOT referenced in current codebase

**Purpose**:
- Standalone energy monitoring script
- Functionality now integrated into `mutation/run.sh` (lines 91-156)
- CPU energy via perf, GPU via nvidia-smi

**Search Results**:
```bash
$ grep -r "energy_monitor.sh" --include="*.py" .
# No matches (not used in Python code)
```

**Last Modified**: 2024-11-11 19:36

**Recommendation**: 🗑️ **ARCHIVE - Functionality integrated into run.sh**

**Notes**: Keep if useful as standalone monitoring tool, otherwise archive.

---

#### 4. `train_wrapper.sh` (2.5K)
**Status**: 🗄️ **LEGACY - Superseded**

**Reason**:
- Functionality absorbed into `mutation/run.sh`
- Not referenced in codebase

**Last Modified**: 2024-11-11 19:36

**Recommendation**: 🗑️ **DELETE - Superseded by mutation/run.sh**

---

#### 5. `test_abbreviations.sh` (2.6K)
**Status**: 🗄️ **LEGACY - Development test**

**Purpose**: Development-time testing of command-line abbreviation handling

**Usage**: NOT referenced in current codebase

**Last Modified**: 2024-11-11 19:36

**Recommendation**: 🗑️ **ARCHIVE - Development test**

---

#### 6. `run_full_test.sh` (2.2K)
**Status**: 🗄️ **LEGACY - Development test**

**Purpose**:
```bash
# Usage: sudo ./scripts/run_full_test.sh
# Full-stack integration test
```

**Usage**: NOT referenced in current codebase

**Last Modified**: 2024-11-11 19:36

**Recommendation**: 🗑️ **ARCHIVE - Development test**
**Alternative**: Consider adapting into new test suite if needed

---

### Analysis/Research Scripts (8 files)

All Python analysis scripts in `scripts/` are **research/design tools** not part of the core mutation runner workflow.

#### 7. `analyze_boundary_results.py` (9.4K)
**Purpose**: Analyze boundary test experiment results
**Status**: 🗄️ **LEGACY - Analysis tool**
**Last Modified**: 2024-11-11 15:30

#### 8. `analyze_boundary_test.py` (7.4K)
**Purpose**: Boundary condition testing analysis
**Status**: 🗄️ **LEGACY - Analysis tool**
**Last Modified**: 2024-11-11 19:36

#### 9. `analyze_concurrent_training_feasibility.py` (12.6K)
**Purpose**: Feasibility study for concurrent training
**Status**: 🗄️ **LEGACY - Research tool**
**Last Modified**: 2024-11-11 16:04

#### 10. `analyze_model_pair_sampling.py` (18.5K)
**Purpose**: Model pair sampling strategy analysis
**Status**: 🗄️ **LEGACY - Research tool**
**Last Modified**: 2024-11-11 16:59

#### 11. `analyze_performance_metrics.py` (5.1K)
**Purpose**: Performance metrics analysis
**Status**: 🗄️ **LEGACY - Analysis tool**
**Last Modified**: 2024-11-11 19:36

#### 12. `analyze_stratified_sampling_strategies.py` (21.3K)
**Purpose**: Stratified sampling strategy comparison
**Status**: 🗄️ **LEGACY - Research tool**
**Last Modified**: 2024-11-11 17:07

#### 13. `explain_memory_stratified_sampling.py` (19.0K)
**Purpose**: Documentation/explanation of memory stratified sampling
**Status**: 🗄️ **LEGACY - Research documentation**
**Last Modified**: 2024-11-11 17:28

#### 14. `memory_stratified_sampling.py` (10.9K)
**Purpose**: Implementation of memory-aware stratified sampling
**Status**: 🗄️ **LEGACY - Research tool**
**Last Modified**: 2024-11-11 17:32

**Analysis Scripts - Collective Recommendation**:
- 🗄️ **ARCHIVE** - Move to `research/` or `analysis/` subdirectory
- These are valuable for understanding experimental design decisions
- Not part of production mutation runner workflow
- Keep for reference but separate from core codebase

---

## Recommendations

### Immediate Actions

1. **Keep Active Script**:
   - ✅ `background_training_template.sh` - Currently used by mutation/command_runner.py

2. **Delete Superseded Scripts**:
   ```bash
   rm scripts/run.sh                    # Moved to mutation/run.sh
   rm scripts/train_wrapper.sh          # Functionality absorbed into mutation/run.sh
   ```

3. **Archive Test Scripts**:
   ```bash
   mkdir -p archive/testing
   mv scripts/test_abbreviations.sh archive/testing/
   mv scripts/run_full_test.sh archive/testing/
   ```

4. **Archive Standalone Tools** (Optional):
   ```bash
   mkdir -p archive/tools
   mv scripts/energy_monitor.sh archive/tools/
   ```

5. **Reorganize Analysis Scripts**:
   ```bash
   mkdir -p research/sampling_analysis
   mv scripts/analyze_*.py research/sampling_analysis/
   mv scripts/memory_stratified_sampling.py research/sampling_analysis/
   mv scripts/explain_memory_stratified_sampling.py research/sampling_analysis/
   ```

### After Cleanup

Expected `scripts/` directory structure:
```
scripts/
└── background_training_template.sh    # ONLY active script
```

All other scripts archived to:
```
archive/
├── testing/
│   ├── test_abbreviations.sh
│   └── run_full_test.sh
└── tools/
    ├── energy_monitor.sh (optional)
    ├── run.sh (superseded)
    └── train_wrapper.sh (superseded)

research/
└── sampling_analysis/
    ├── analyze_boundary_results.py
    ├── analyze_boundary_test.py
    ├── analyze_concurrent_training_feasibility.py
    ├── analyze_model_pair_sampling.py
    ├── analyze_performance_metrics.py
    ├── analyze_stratified_sampling_strategies.py
    ├── explain_memory_stratified_sampling.py
    └── memory_stratified_sampling.py
```

---

## Verification Commands

### Check for any missed references:
```bash
# Search for script references in codebase
for script in scripts/*.{sh,py}; do
    basename_only=$(basename "$script")
    echo "=== Checking $basename_only ==="
    grep -r "$basename_only" --include="*.py" --exclude-dir=scripts --exclude-dir=__pycache__ .
done
```

### Result:
- Only `background_training_template.sh` has active references
- All other scripts have NO references in production code

---

## Impact on Current Workflow

### ✅ No Breaking Changes
- Active script (`background_training_template.sh`) remains in place
- All production workflows unaffected
- Mutation runner continues to work as before

### 🗄️ Benefits of Cleanup
- Clearer project structure
- Easier to identify production vs research code
- Reduced confusion for new developers
- Preserved historical analysis work in organized archive

---

## Historical Context

These scripts were created during:
1. **Phase 1**: Initial mutation runner development (run.sh, train_wrapper.sh)
2. **Phase 2**: Energy monitoring improvements (energy_monitor.sh)
3. **Phase 3**: Parallel training experiments (background_training_template.sh)
4. **Phase 4**: Experimental design research (all analyze_*.py scripts)
5. **Phase 5**: Refactoring (run.sh moved to mutation/)

The analysis scripts represent valuable research artifacts but are not part of the core mutation runner production workflow.

---

**Analysis Complete**: 1 active script, 13 legacy scripts identified
**Recommendation**: Archive legacy scripts to keep codebase clean
**Risk**: None - only one script actively used by production code
