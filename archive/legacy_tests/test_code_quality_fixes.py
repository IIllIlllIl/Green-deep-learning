#!/usr/bin/env python3
"""
Test code quality fixes (constants, helper methods, timeout changes)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation import MutationRunner

def test_constants_defined():
    """Test that all constants are properly defined"""
    print("\n" + "=" * 80)
    print("TEST 1: Constants Defined")
    print("=" * 80)

    runner = MutationRunner()

    # Check TIMESTAMP_FORMAT
    assert hasattr(runner, 'TIMESTAMP_FORMAT'), "TIMESTAMP_FORMAT not defined"
    assert runner.TIMESTAMP_FORMAT == "%Y%m%d_%H%M%S", f"TIMESTAMP_FORMAT incorrect: {runner.TIMESTAMP_FORMAT}"
    print(f"✅ PASS: TIMESTAMP_FORMAT = {runner.TIMESTAMP_FORMAT}")

    # Check FLOAT_PRECISION
    assert hasattr(runner, 'FLOAT_PRECISION'), "FLOAT_PRECISION not defined"
    assert runner.FLOAT_PRECISION == 6, f"FLOAT_PRECISION incorrect: {runner.FLOAT_PRECISION}"
    print(f"✅ PASS: FLOAT_PRECISION = {runner.FLOAT_PRECISION}")

    # Check EMPTY_STATS_DICT
    assert hasattr(runner, 'EMPTY_STATS_DICT'), "EMPTY_STATS_DICT not defined"
    expected_dict = {"avg": None, "max": None, "min": None, "sum": None}
    assert runner.EMPTY_STATS_DICT == expected_dict, f"EMPTY_STATS_DICT incorrect: {runner.EMPTY_STATS_DICT}"
    print(f"✅ PASS: EMPTY_STATS_DICT = {runner.EMPTY_STATS_DICT}")

    # Check DEFAULT_TRAINING_TIMEOUT_SECONDS
    assert hasattr(runner, 'DEFAULT_TRAINING_TIMEOUT_SECONDS'), "DEFAULT_TRAINING_TIMEOUT_SECONDS not defined"
    assert runner.DEFAULT_TRAINING_TIMEOUT_SECONDS is None, f"DEFAULT_TRAINING_TIMEOUT_SECONDS should be None, got: {runner.DEFAULT_TRAINING_TIMEOUT_SECONDS}"
    print(f"✅ PASS: DEFAULT_TRAINING_TIMEOUT_SECONDS = {runner.DEFAULT_TRAINING_TIMEOUT_SECONDS}")

    # Check FAST_TRAINING_TIMEOUT_SECONDS
    assert hasattr(runner, 'FAST_TRAINING_TIMEOUT_SECONDS'), "FAST_TRAINING_TIMEOUT_SECONDS not defined"
    assert runner.FAST_TRAINING_TIMEOUT_SECONDS == 3600, f"FAST_TRAINING_TIMEOUT_SECONDS incorrect: {runner.FAST_TRAINING_TIMEOUT_SECONDS}"
    print(f"✅ PASS: FAST_TRAINING_TIMEOUT_SECONDS = {runner.FAST_TRAINING_TIMEOUT_SECONDS}")

    return True

def test_helper_methods_exist():
    """Test that helper methods are defined"""
    print("\n" + "=" * 80)
    print("TEST 2: Helper Methods Exist")
    print("=" * 80)

    runner = MutationRunner()

    # Check _format_hyperparam_value
    assert hasattr(runner, '_format_hyperparam_value'), "_format_hyperparam_value not defined"
    print("✅ PASS: _format_hyperparam_value method exists")

    # Check _build_hyperparam_args
    assert hasattr(runner, '_build_hyperparam_args'), "_build_hyperparam_args not defined"
    print("✅ PASS: _build_hyperparam_args method exists")

    # Check _cleanup_all_background_processes
    assert hasattr(runner, '_cleanup_all_background_processes'), "_cleanup_all_background_processes not defined"
    print("✅ PASS: _cleanup_all_background_processes method exists")

    # Check __del__
    assert hasattr(runner, '__del__'), "__del__ destructor not defined"
    print("✅ PASS: __del__ destructor exists")

    return True

def test_format_hyperparam_value():
    """Test _format_hyperparam_value helper method"""
    print("\n" + "=" * 80)
    print("TEST 3: _format_hyperparam_value Functionality")
    print("=" * 80)

    runner = MutationRunner()

    # Test integer formatting
    result = runner._format_hyperparam_value(100.7, "int")
    assert result == "100", f"Int formatting failed: expected '100', got '{result}'"
    print(f"✅ PASS: Integer formatting - {100.7} → '{result}'")

    # Test float formatting
    result = runner._format_hyperparam_value(0.001234567, "float")
    assert result == "0.001235", f"Float formatting failed: expected '0.001235', got '{result}'"
    print(f"✅ PASS: Float formatting - {0.001234567} → '{result}'")

    # Test float precision
    result = runner._format_hyperparam_value(3.14159265359, "float")
    assert result == "3.141593", f"Float precision failed: expected '3.141593', got '{result}'"
    print(f"✅ PASS: Float precision - {3.14159265359} → '{result}'")

    # Test other type (string passthrough)
    result = runner._format_hyperparam_value("test", "string")
    assert result == "test", f"String passthrough failed: expected 'test', got '{result}'"
    print(f"✅ PASS: String passthrough - 'test' → '{result}'")

    return True

def test_build_hyperparam_args():
    """Test _build_hyperparam_args helper method"""
    print("\n" + "=" * 80)
    print("TEST 4: _build_hyperparam_args Functionality")
    print("=" * 80)

    runner = MutationRunner()

    # Test with sample parameters
    supported_params = {
        "epochs": {"flag": "--epochs", "type": "int"},
        "learning_rate": {"flag": "--lr", "type": "float"},
        "dropout": {"flag": "--dropout", "type": "float"}
    }

    hyperparams = {
        "epochs": 100,
        "learning_rate": 0.001234567,
        "dropout": 0.5
    }

    # Test as_list=True (default)
    result = runner._build_hyperparam_args(supported_params, hyperparams, as_list=True)
    assert isinstance(result, list), "Result should be a list"
    assert "--epochs" in result, "Missing --epochs flag"
    assert "100" in result, "Missing epochs value"
    assert "--lr" in result, "Missing --lr flag"
    assert "0.001235" in result, "Missing learning_rate value (formatted)"
    assert "--dropout" in result, "Missing --dropout flag"
    assert "0.500000" in result, "Missing dropout value (formatted)"
    print(f"✅ PASS: as_list=True - {result}")

    # Test as_list=False (string)
    result = runner._build_hyperparam_args(supported_params, hyperparams, as_list=False)
    assert isinstance(result, str), "Result should be a string"
    assert "--epochs 100" in result, "Missing '--epochs 100' in string"
    assert "--lr 0.001235" in result, "Missing '--lr 0.001235' in string"
    assert "--dropout 0.500000" in result, "Missing '--dropout 0.500000' in string"
    print(f"✅ PASS: as_list=False - '{result}'")

    # Test with unsupported parameter (should be ignored)
    hyperparams_extra = {
        "epochs": 50,
        "unsupported_param": 123  # Should be ignored
    }
    result = runner._build_hyperparam_args(supported_params, hyperparams_extra, as_list=True)
    assert "unsupported_param" not in str(result), "Unsupported parameter should be ignored"
    assert "--epochs" in result, "Missing --epochs flag"
    assert "50" in result, "Missing epochs value"
    print(f"✅ PASS: Unsupported parameters ignored - {result}")

    return True

def test_process_tracking():
    """Test that process tracking list is initialized"""
    print("\n" + "=" * 80)
    print("TEST 5: Process Tracking Initialization")
    print("=" * 80)

    runner = MutationRunner()

    # Check that _active_background_processes list exists
    assert hasattr(runner, '_active_background_processes'), "_active_background_processes not defined"
    assert isinstance(runner._active_background_processes, list), "_active_background_processes should be a list"
    assert len(runner._active_background_processes) == 0, "_active_background_processes should be empty initially"
    print(f"✅ PASS: Process tracking list initialized - {runner._active_background_processes}")

    return True

def test_integration_with_config():
    """Test that helper methods work with actual config"""
    print("\n" + "=" * 80)
    print("TEST 6: Integration with Config")
    print("=" * 80)

    runner = MutationRunner()

    # Check that we can access config
    assert "models" in runner.config, "Config missing 'models' key"
    print(f"✅ PASS: Config loaded - {len(runner.config['models'])} repositories")

    # Try with a real repository (if pytorch_resnet_cifar10 exists)
    if "pytorch_resnet_cifar10" in runner.config["models"]:
        repo_config = runner.config["models"]["pytorch_resnet_cifar10"]
        supported_params = repo_config["supported_hyperparams"]

        # Build args for real parameters
        hyperparams = {"epochs": 10, "learning_rate": 0.001}
        result = runner._build_hyperparam_args(supported_params, hyperparams, as_list=False)

        assert "--epochs" in result or "-e" in result, "Missing epochs flag in real config"
        print(f"✅ PASS: Real config integration - '{result}'")
    else:
        print("⚠️  SKIP: pytorch_resnet_cifar10 not in config")

    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CODE QUALITY FIXES - FUNCTIONAL TESTS")
    print("=" * 80)

    tests = [
        test_constants_defined,
        test_helper_methods_exist,
        test_format_hyperparam_value,
        test_build_hyperparam_args,
        test_process_tracking,
        test_integration_with_config
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"❌ FAIL: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {e}")

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✅ All tests passed!")
        print("=" * 80)
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed!")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
