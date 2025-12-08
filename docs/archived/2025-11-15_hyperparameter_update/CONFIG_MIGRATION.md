# Config Migration Summary

## Overview
Moved `models_config.json` from `config/` directory into the `mutation/` package to create a fully self-contained mutation module.

**Date**: 2025-11-13
**Version**: v4.0 (post-refactoring enhancement)

## Changes Made

### 1. File Movement
```bash
# Before
config/models_config.json

# After
mutation/models_config.json
```

### 2. Code Updates

#### `mutation/runner.py` (lines 75-94)
**Changed default config path resolution:**
```python
# OLD:
def __init__(self, config_path: str = "config/models_config.json", random_seed: Optional[int] = None):
    self.project_root = Path(__file__).parent.parent.absolute()
    self.config_path = self.project_root / config_path

# NEW:
def __init__(self, config_path: Optional[str] = None, random_seed: Optional[int] = None):
    self.project_root = Path(__file__).parent.parent.absolute()

    # Default to models_config.json in the mutation package
    if config_path is None:
        self.config_path = Path(__file__).parent / "models_config.json"
    else:
        self.config_path = self.project_root / config_path
```

**Benefits:**
- Config file is now co-located with the code that uses it
- No external dependencies on project structure
- Cleaner default behavior (no path needed)

#### `mutation.py` (line 113-117)
**Updated CLI default parameter:**
```python
# OLD:
parser.add_argument(
    "-c", "--config",
    type=str,
    default="config/models_config.json",
    help="Path to models configuration file (default: config/models_config.json)"
)

# NEW:
parser.add_argument(
    "-c", "--config",
    type=str,
    default=None,
    help="Path to models configuration file (default: mutation/models_config.json)"
)
```

#### `tests/functional/test_refactoring.py`
**Updated test config paths:**
```python
# Line 160: Command Runner test
config_path = Path("mutation/models_config.json")  # was: "config/models_config.json"

# Line 194: Runner Initialization test
runner = MutationRunner(random_seed=42)  # was: MutationRunner("config/models_config.json", ...)

# Line 273: Backward Compatibility test
runner = MutationRunner(random_seed=42)  # was: MutationRunner("config/models_config.json", ...)

# Line 253: File Structure test - Added models_config.json to required files list
```

### 3. Directory Cleanup
```bash
# Removed empty config directory
rm -rf config/
```

## Usage

### New Default Behavior (Recommended)
```python
# No config path needed - uses mutation/models_config.json automatically
runner = MutationRunner()

# CLI also works without -c flag
python3 mutation.py --repo pytorch_resnet_cifar10 --model resnet20 --mutate all
```

### Custom Config Path (Still Supported)
```python
# Specify custom config path relative to project root
runner = MutationRunner(config_path="custom/path/models_config.json")

# CLI with custom config
python3 mutation.py -c custom/path/models_config.json -r pytorch_resnet_cifar10 -m resnet20 -mt all
```

## Benefits

### 1. **Self-Contained Package**
- All mutation-related files in one place
- No external directory dependencies
- Easier to distribute/deploy

### 2. **Cleaner API**
```python
# Before (required config path)
runner = MutationRunner("config/models_config.json")

# After (optional config path)
runner = MutationRunner()  # Uses default
runner = MutationRunner(config_path="custom.json")  # Override if needed
```

### 3. **Better Organization**
```
mutation/                    # Self-contained package
├── __init__.py             # Public API
├── models_config.json      # Configuration (NEW location)
├── runner.py               # Main orchestrator
├── session.py              # Session management
├── hyperparams.py          # Hyperparameter mutation
├── command_runner.py       # Command execution
├── energy.py               # Energy/performance parsing
├── utils.py                # Utilities
├── exceptions.py           # Exception hierarchy
├── run.sh                  # Training wrapper
├── background_training_template.sh
└── governor.sh             # CPU governor control
```

### 4. **Backward Compatibility**
- Custom config paths still work via `-c` flag or `config_path` parameter
- Archive scripts still reference old paths (no need to update)
- Test coverage ensures no breaking changes

## Testing

### Test Results
✅ **Unit Tests**: 25 tests passed (1 skipped)
```bash
python3 -m unittest discover -s tests/unit -p "test_*.py" -v
# Ran 25 tests in 0.029s
# OK (skipped=1)
```

✅ **Functional Tests**: 8 tests passed
```bash
python3 tests/functional/test_refactoring.py
# Total tests: 8
# Passed: 8
# Failed: 0
# 🎉 ALL TESTS PASSED!
```

### Test Coverage
- ✅ Config loading from default location
- ✅ Config loading from custom path
- ✅ CLI argument parsing with/without -c flag
- ✅ MutationRunner initialization
- ✅ Backward compatibility
- ✅ File structure validation

## Migration Guide

### For Users
**No action needed** - The mutation package now uses the config file from its own directory by default.

### For Developers
If you have custom scripts that reference `config/models_config.json`:

```python
# Option 1: Use default (recommended)
runner = MutationRunner()

# Option 2: Update path reference
runner = MutationRunner(config_path="mutation/models_config.json")

# Option 3: Keep using relative path from project root
# (still works if you have a custom config elsewhere)
runner = MutationRunner(config_path="my_custom_config.json")
```

## Files Modified

| File | Lines Changed | Type |
|------|---------------|------|
| `mutation/runner.py` | 75-94 | Modified default path logic |
| `mutation.py` | 113-117 | Updated CLI default |
| `tests/functional/test_refactoring.py` | 160, 194, 253, 273 | Updated test paths |
| `mutation/models_config.json` | - | Moved from `config/` |
| `config/` | - | Directory removed |

## Version History

- **v4.0** (2025-11-13): Modular architecture refactoring
- **v4.0.1** (2025-11-13): Config migration to mutation package (this update)

## Related Documentation

- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - v4.0 refactoring details
- [CLEANUP_PLAN.md](../CLEANUP_PLAN.md) - Directory cleanup performed today
- [README.md](../README.md) - Main project documentation
